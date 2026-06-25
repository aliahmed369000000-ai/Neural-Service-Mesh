"""
Few-Shot Prompt Engine — محرك الأمثلة الديناميكية
===================================================
يبني Prompts ذكية بتضمين أمثلة مشابهة مُسترجعة من الذاكرة الحدثية.

المبدأ:
  بدلاً من الإجابة المباشرة، نبحث في EpisodicMemory عن k أسئلة مشابهة
  للسؤال الحالي ونبني سياقاً بالشكل:
    "مثال 1: السؤال → الإجابة
     مثال 2: السؤال → الإجابة
     السؤال الحالي: ..."

  هذا يُحسّن دقة الإجابات خاصةً للأسئلة النادرة أو المتخصصة.

التكامل:
  - يستخدم ai/episodic_memory.py للبحث الدلالي
  - يستخدم ai/experience_store.py كبديل احتياطي
  - يُضمن في nsm_chat.py عبر FewShotChatWrapper

الاستخدام:
    from ai.prompt_engine import FewShotPromptEngine, FewShotChatWrapper
    engine  = FewShotPromptEngine(k=3)
    prompt  = engine.build(query="ما هي أركان الإسلام؟")
    # أو كـ Wrapper حول NSMChat الموجود:
    wrapper = FewShotChatWrapper(nsm_chat_instance)
    reply   = wrapper.chat("ما هي أركان الإسلام؟")
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_NOW = lambda: datetime.now(timezone.utc).isoformat()

# ── مسارات الذاكرة الافتراضية ────────────────────────────────────────────────
_EPISODIC_DB  = Path("memory/episodic.db")
_EXPERIENCE_DB = Path("memory/experience.db")
_EMBED_PATH   = Path("nsm_embedding.npz")


# ════════════════════════════════════════════════════════════════════════════
# RetrievedExample — مثال مُسترجع من الذاكرة
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievedExample:
    """مثال واحد مُسترجع ليُدمج في الـ Prompt."""
    question: str
    answer: str
    similarity: float
    source: str       # "episodic" | "experience" | "fallback"
    topic: str = ""

    def to_prompt_line(self, index: int) -> str:
        q = self.question.strip()[:120]
        a = self.answer.strip()[:200]
        return f"مثال {index}: س: {q}\n         ج: {a}"


# ════════════════════════════════════════════════════════════════════════════
# EmbeddingIndex — فهرس متجهات للبحث السريع
# ════════════════════════════════════════════════════════════════════════════

class EmbeddingIndex:
    """
    فهرس بسيط للبحث بالتشابه الدلالي (Cosine Similarity).
    يعمل بدون مكتبات خارجية — NumPy فقط.
    """

    def __init__(self, embed_path: Path = _EMBED_PATH):
        self._E: Optional[np.ndarray] = None
        self._load_embedding(embed_path)

    def _load_embedding(self, path: Path) -> None:
        try:
            data    = np.load(str(path))
            self._E = data["E"].astype(np.float64)
            logger.debug(f"[PromptEngine] Loaded embedding E{self._E.shape}")
        except Exception as exc:
            logger.warning(f"[PromptEngine] Could not load embedding: {exc}")
            self._E = None

    def encode(self, text: str) -> np.ndarray:
        """يحوّل النص إلى متجه (128,) باستخدام مصفوفة E."""
        raw = _text_to_vec(text, dim=784)
        if self._E is not None:
            z = self._E.T @ raw       # (128,)
        else:
            # fallback: أول 128 عنصر من المتجه الخام
            z = raw[:128] if len(raw) >= 128 else np.pad(raw, (0, 128 - len(raw)))
        norm = np.linalg.norm(z) + 1e-8
        return z / norm               # L2 normalised

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b)           # كلاهما L2 normalised مسبقاً


# ════════════════════════════════════════════════════════════════════════════
# FewShotPromptEngine — المحرك الرئيسي
# ════════════════════════════════════════════════════════════════════════════

class FewShotPromptEngine:
    """
    يبني Few-shot prompts ديناميكية عبر استرجاع أمثلة مشابهة.

    مصادر الاسترجاع (بالأولوية):
      1. ai/episodic_memory.py (EpisodicStore — SQLite)
      2. ai/experience_store.py (EpisodeStore — neural_episodes)
      3. أمثلة ثابتة (Fallback built-in)
    """

    def __init__(
        self,
        k: int = 3,
        min_similarity: float = 0.30,
        max_answer_len: int = 200,
        episodic_db: Path | str = _EPISODIC_DB,
        experience_db: Path | str = _EXPERIENCE_DB,
        embed_path: Path | str = _EMBED_PATH,
        system_prompt: str = "",
    ):
        self.k               = k
        self.min_similarity  = min_similarity
        self.max_answer_len  = max_answer_len
        self.episodic_db     = Path(episodic_db)
        self.experience_db   = Path(experience_db)
        self.system_prompt   = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._index          = EmbeddingIndex(Path(embed_path))
        self._cache: Dict[str, List[RetrievedExample]] = {}   # cache بسيط

    # ── واجهة عامة ──────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: Optional[int] = None) -> List[RetrievedExample]:
        """
        يسترجع أفضل k أمثلة مشابهة من الذاكرة.

        Args:
            query: السؤال الحالي
            k:     عدد الأمثلة (يتجاوز self.k إن مُحدَّد)

        Returns:
            قائمة مُرتَّبة تنازلياً حسب التشابه
        """
        n = k or self.k
        cache_key = _hash(query + str(n))
        if cache_key in self._cache:
            return self._cache[cache_key]

        q_vec = self._index.encode(query)
        candidates: List[RetrievedExample] = []

        # المصدر 1: EpisodicMemory
        candidates.extend(self._retrieve_from_episodic(q_vec, limit=n * 3))

        # المصدر 2: ExperienceStore (احتياطي)
        if len(candidates) < n:
            candidates.extend(self._retrieve_from_experience(q_vec, limit=n * 3))

        # إزالة التكرار وترتيب بالتشابه
        seen_q: set = set()
        unique: List[RetrievedExample] = []
        for ex in sorted(candidates, key=lambda x: x.similarity, reverse=True):
            q_norm = ex.question.strip().lower()[:50]
            if q_norm not in seen_q and ex.similarity >= self.min_similarity:
                seen_q.add(q_norm)
                unique.append(ex)
            if len(unique) >= n:
                break

        # Fallback إذا لم نجد شيئاً
        if not unique:
            unique = self._fallback_examples(query, n)

        self._cache[cache_key] = unique
        if len(self._cache) > 500:
            # مسح نصف الـ cache عند الامتلاء
            keys = list(self._cache.keys())[:250]
            for key in keys:
                del self._cache[key]

        return unique

    def build(
        self,
        query: str,
        include_system: bool = True,
        k: Optional[int] = None,
    ) -> str:
        """
        يبني نصّ الـ Prompt الكامل.

        البنية:
            [System Prompt]
            ─────────────
            أمثلة مشابهة من الذاكرة:
            مثال 1: س: ... ج: ...
            مثال 2: س: ... ج: ...
            ─────────────
            السؤال الحالي: <query>
            الإجابة:

        Args:
            query:          السؤال الحالي
            include_system: تضمين رسالة النظام
            k:              عدد الأمثلة

        Returns:
            str — نص الـ prompt الجاهز للاستخدام
        """
        examples = self.retrieve(query, k=k)
        parts: List[str] = []

        if include_system and self.system_prompt:
            parts.append(self.system_prompt)
            parts.append("─" * 50)

        if examples:
            parts.append("📚 أمثلة مشابهة من الذاكرة:")
            for i, ex in enumerate(examples, 1):
                parts.append(ex.to_prompt_line(i))
                if ex.topic:
                    parts.append(f"         [موضوع: {ex.topic}]")
            parts.append("─" * 50)
        else:
            parts.append("(لا توجد أمثلة مشابهة في الذاكرة)")
            parts.append("─" * 50)

        parts.append(f"❓ السؤال الحالي: {query.strip()}")
        parts.append("✅ الإجابة:")

        return "\n".join(parts)

    def build_compact(self, query: str, k: Optional[int] = None) -> str:
        """
        نسخة مضغوطة من الـ Prompt للاستخدام كسياق مُضاف.
        مناسب للدمج مع إجابة نظام آخر.
        """
        examples = self.retrieve(query, k=k)
        if not examples:
            return query

        ctx_parts = []
        for ex in examples:
            ctx_parts.append(f"[مرجع: {ex.question.strip()[:60]} → {ex.answer.strip()[:80]}]")

        return " ".join(ctx_parts) + f" || {query}"

    def get_stats(self) -> Dict[str, Any]:
        """إحصاءات المحرك."""
        return {
            "k": self.k,
            "min_similarity": self.min_similarity,
            "cache_size": len(self._cache),
            "episodic_db_exists": self.episodic_db.exists(),
            "experience_db_exists": self.experience_db.exists(),
            "embedding_loaded": self._index._E is not None,
        }

    # ── استرجاع من EpisodicMemory ────────────────────────────────────────────

    def _retrieve_from_episodic(
        self, q_vec: np.ndarray, limit: int = 20
    ) -> List[RetrievedExample]:
        """يبحث في جدول episodic_memory في SQLite."""
        if not self.episodic_db.exists():
            return []
        results: List[RetrievedExample] = []
        try:
            conn = sqlite3.connect(str(self.episodic_db))
            conn.row_factory = sqlite3.Row

            # نحاول جداول مختلفة
            for table in ("episodes", "episodic_events", "memories"):
                try:
                    rows = conn.execute(
                        f"SELECT * FROM {table} ORDER BY timestamp DESC LIMIT ?",
                        (limit * 5,)
                    ).fetchall()
                    for row in rows:
                        ex = self._row_to_example(row, q_vec, source="episodic")
                        if ex:
                            results.append(ex)
                    if results:
                        break
                except sqlite3.OperationalError:
                    continue
            conn.close()
        except Exception as exc:
            logger.debug(f"[PromptEngine] episodic retrieval error: {exc}")
        return results

    # ── استرجاع من ExperienceStore ───────────────────────────────────────────

    def _retrieve_from_experience(
        self, q_vec: np.ndarray, limit: int = 20
    ) -> List[RetrievedExample]:
        """يبحث في جدول neural_episodes في SQLite."""
        if not self.experience_db.exists():
            return []
        results: List[RetrievedExample] = []
        try:
            conn = sqlite3.connect(str(self.experience_db))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """SELECT question, answer, matched_concepts, confidence
                   FROM neural_episodes
                   ORDER BY json_extract(quality, '$.overall_quality') DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()
            conn.close()

            for row in rows:
                q_text = row["question"] or ""
                a_text = row["answer"]   or ""
                if not q_text or not a_text:
                    continue

                ex_vec = self._index.encode(q_text)
                sim    = self._index.cosine(q_vec, ex_vec)

                try:
                    mc    = json.loads(row["matched_concepts"] or "[]")
                    topic = mc[0]["concept"] if mc else ""
                except Exception:
                    topic = ""

                results.append(RetrievedExample(
                    question=q_text,
                    answer=a_text[:self.max_answer_len],
                    similarity=sim,
                    source="experience",
                    topic=topic,
                ))
        except Exception as exc:
            logger.debug(f"[PromptEngine] experience retrieval error: {exc}")
        return results

    def _row_to_example(
        self, row: sqlite3.Row, q_vec: np.ndarray, source: str
    ) -> Optional[RetrievedExample]:
        """يحوّل صف SQLite إلى RetrievedExample."""
        try:
            col_names = row.keys()
            q_text, a_text, topic = "", "", ""

            for q_col in ("question", "query", "input", "user_input", "text"):
                if q_col in col_names and row[q_col]:
                    q_text = str(row[q_col])
                    break

            for a_col in ("answer", "response", "output", "reply", "content"):
                if a_col in col_names and row[a_col]:
                    a_text = str(row[a_col])
                    break

            for t_col in ("topic", "category", "label", "type"):
                if t_col in col_names and row[t_col]:
                    topic = str(row[t_col])
                    break

            if not q_text or not a_text:
                return None

            ex_vec = self._index.encode(q_text)
            sim    = self._index.cosine(q_vec, ex_vec)

            return RetrievedExample(
                question=q_text,
                answer=a_text[:self.max_answer_len],
                similarity=sim,
                source=source,
                topic=topic,
            )
        except Exception:
            return None

    # ── أمثلة احتياطية ─────────────────────────────────────────────────────

    def _fallback_examples(self, query: str, k: int) -> List[RetrievedExample]:
        """أمثلة ثابتة تُستخدم عند فراغ الذاكرة."""
        fallbacks = _FALLBACK_EXAMPLES
        q_vec = self._index.encode(query)
        scored = []
        for fb in fallbacks:
            ex_vec = self._index.encode(fb["q"])
            sim    = self._index.cosine(q_vec, ex_vec)
            scored.append((sim, fb))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievedExample(
                question=fb["q"],
                answer=fb["a"],
                similarity=sim,
                source="fallback",
                topic=fb.get("t", ""),
            )
            for sim, fb in scored[:k]
        ]


# ════════════════════════════════════════════════════════════════════════════
# FewShotChatWrapper — يُدمج المحرك مع NSMChat الموجود
# ════════════════════════════════════════════════════════════════════════════

class FewShotChatWrapper:
    """
    Wrapper حول NSMChat يُضيف Few-shot context تلقائياً.

    مثال:
        from nsm_chat import NSMChat
        from ai.prompt_engine import FewShotChatWrapper

        base_chat = NSMChat()
        chat      = FewShotChatWrapper(base_chat, k=3)
        reply     = chat.chat("ما هو الإيمان؟")
    """

    def __init__(
        self,
        base_chat: Any,
        engine: Optional[FewShotPromptEngine] = None,
        k: int = 3,
        mode: str = "compact",
    ):
        """
        Args:
            base_chat: كائن NSMChat الأصلي
            engine:    FewShotPromptEngine (يُنشأ تلقائياً إن لم يُعطَ)
            k:         عدد الأمثلة
            mode:      "compact" (يُضيف السياق للسؤال) |
                       "full"    (يبني prompt كامل)
        """
        self._chat   = base_chat
        self._engine = engine or FewShotPromptEngine(k=k)
        self._mode   = mode
        self._k      = k
        self._history: List[Dict[str, str]] = []

    def chat(self, query: str) -> str:
        """
        يعالج سؤالاً مع تضمين أمثلة مشابهة من الذاكرة.

        Args:
            query: سؤال المستخدم

        Returns:
            str — الإجابة المُعزَّزة بالأمثلة
        """
        try:
            if self._mode == "compact":
                enriched_query = self._engine.build_compact(query, k=self._k)
            else:
                enriched_query = self._engine.build(query, k=self._k)

            # استدعاء NSMChat الأصلي بالسؤال المُعزَّز
            if hasattr(self._chat, "chat"):
                reply = self._chat.chat(enriched_query)
            elif callable(self._chat):
                reply = self._chat(enriched_query)
            else:
                reply = str(enriched_query)

            self._history.append({"query": query, "reply": reply})
            return reply

        except Exception as exc:
            logger.error(f"[FewShotChatWrapper] Error: {exc}")
            # fallback: استدعاء NSMChat بدون تعزيز
            if hasattr(self._chat, "chat"):
                return self._chat.chat(query)
            return ""

    def retrieve_examples(self, query: str) -> List[RetrievedExample]:
        """عرض الأمثلة المُسترجعة لسؤال معين (للمراقبة)."""
        return self._engine.retrieve(query, k=self._k)

    def build_prompt(self, query: str) -> str:
        """بناء الـ Prompt الكامل (للمراقبة والتصحيح)."""
        return self._engine.build(query, k=self._k)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "mode": self._mode,
            "k": self._k,
            "total_chats": len(self._history),
            "engine_stats": self._engine.get_stats(),
        }


# ════════════════════════════════════════════════════════════════════════════
# أدوات مساعدة
# ════════════════════════════════════════════════════════════════════════════

def _fnv1a(s: str) -> int:
    h = 0x811c9dc5
    for ch in s:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h

def _text_to_vec(text: str, dim: int = 784) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float64)
    t   = re.sub(r'\s+', ' ', text.strip().lower())
    for n in (1, 2, 3):
        for i in range(len(t) - n + 1):
            vec[_fnv1a(t[i:i+n]) % dim] += 1.0
    total = vec.sum()
    if total > 0:
        vec = np.log1p(vec * 10.0 / total) / math.log1p(10.0)
    return vec

def _hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:16]


# ════════════════════════════════════════════════════════════════════════════
# ثوابت
# ════════════════════════════════════════════════════════════════════════════

_DEFAULT_SYSTEM_PROMPT = (
    "أنت NSM — النظام المعرفي العربي المتخصص في الذكاء الاصطناعي والمعرفة الإسلامية.\n"
    "أجب بدقة وإيجاز، واستند إلى الأمثلة المعطاة عند الحاجة."
)

_FALLBACK_EXAMPLES = [
    {"q": "ما هي أركان الإسلام؟",
     "a": "أركان الإسلام خمسة: الشهادتان، والصلاة، والزكاة، والصوم، والحج.",
     "t": "إسلام"},
    {"q": "ما هو الذكاء الاصطناعي؟",
     "a": "الذكاء الاصطناعي علم برمجة الحواسيب لتحاكي القدرات الإدراكية البشرية.",
     "t": "ذكاء اصطناعي"},
    {"q": "كم عدد سور القرآن الكريم؟",
     "a": "يحتوي القرآن الكريم على 114 سورة.",
     "t": "قرآن"},
    {"q": "ما هو التعلم الآلي؟",
     "a": "التعلم الآلي فرع من الذكاء الاصطناعي يُمكّن الحواسيب من التعلم من البيانات.",
     "t": "ذكاء اصطناعي"},
    {"q": "ما هي الشبكة العصبية؟",
     "a": "الشبكة العصبية نموذج رياضي مستوحى من الدماغ البشري يتعلم من الأمثلة.",
     "t": "ذكاء اصطناعي"},
]

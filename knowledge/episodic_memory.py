"""
episodic_memory.py
===================
نظام الذاكرة التجريبية (Episodic Memory) لتفاعلات الأسئلة والأجوبة.

يحوّل المشروع من قاعدة معرفة ثابتة إلى نظام يتعلّم من تاريخه:
  - يحفظ كل تفاعل سؤال/جواب كـ "حلقة" (episode) في قاعدة بيانات دائمة
  - يدعم البحث عن أسئلة سابقة مشابهة وإعادة استخدامها
  - يستخرج أزواج مفاهيم متكررة ويولّد "قواعد دلالية" تُحفظ في CKG

لا يضيف طبقات عصبية ولا مصادر خارجية — يعمل فوق قاعدة memory/episodic.db
الموجودة مسبقاً (يضيف جداول جديدة فقط، دون لمس جدول `episodes` الحالي
الخاص بنظام التدريب العصبي).
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# qa_engine موجود في نفس المجلد (knowledge/)
try:
    from qa_engine import question_similarity, normalize_arabic
except ImportError:  # pragma: no cover - fallback عند الاستيراد من مسار مختلف
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from qa_engine import question_similarity, normalize_arabic


_NOW = lambda: datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════════════════
# تهيئة قاعدة البيانات (جداول جديدة فقط — لا تلمس جدول episodes الحالي)
# ═══════════════════════════════════════════════════════════════════════════
def init_db(db_path: Path) -> None:
    """ينشئ جداول الذاكرة التجريبية لو لم تكن موجودة."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # جدول حلقات الأسئلة والأجوبة
    cur.execute("""
        CREATE TABLE IF NOT EXISTS qa_episodes (
            id               TEXT PRIMARY KEY,
            question         TEXT NOT NULL,
            question_norm    TEXT NOT NULL,
            concepts         TEXT NOT NULL,   -- JSON list of {name, cluster, frequency, match}
            related_concepts TEXT NOT NULL,   -- JSON list of {concept, weight, relation_type}
            verses           TEXT NOT NULL,   -- JSON list of {surah, ayah, text, concept}
            answer_summary   TEXT NOT NULL,
            confidence       REAL NOT NULL,
            timestamp        TEXT NOT NULL
        )
    """)

    # جدول القواعد الدلالية المستخرجة من التوحيد (consolidation)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS semantic_rules (
            id           TEXT PRIMARY KEY,
            concept_a    TEXT NOT NULL,
            concept_b    TEXT NOT NULL,
            co_count     INTEGER NOT NULL,
            rule_text    TEXT NOT NULL,
            confidence   REAL NOT NULL,
            created_at   TEXT NOT NULL,
            UNIQUE(concept_a, concept_b)
        )
    """)

    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════════════════
# 1) تخزين حلقة سؤال/جواب
# ═══════════════════════════════════════════════════════════════════════════
def store_episode(db_path: Path, question: str, qa_result: Dict[str, Any]) -> str:
    """
    يحفظ تفاعل سؤال/جواب كحلقة جديدة في قاعدة البيانات الدائمة.
    يعيد معرّف الحلقة (episode id).
    """
    init_db(db_path)

    ts = _NOW()
    episode_id = f"qa_{int(datetime.now(timezone.utc).timestamp() * 1000)}"

    concepts = [
        {
            "name":      c["name"],
            "cluster":   c["cluster"],
            "frequency": c["frequency"],
            "match":     c["match"],
        }
        for c in qa_result.get("primary_concepts", [])
    ]

    related = [
        {
            "concept":       r["concept"],
            "weight":        r["weight"],
            "relation_type": r["relation_type"],
        }
        for r in qa_result.get("related_concepts", [])
    ]

    verses = [
        {
            "surah":   v["surah"],
            "ayah":    v["ayah"],
            "text":    v["text"],
            "concept": v["concept"],
        }
        for v in qa_result.get("verses", [])
    ]

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        INSERT INTO qa_episodes
            (id, question, question_norm, concepts, related_concepts,
             verses, answer_summary, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            episode_id,
            question,
            normalize_arabic(question),
            json.dumps(concepts, ensure_ascii=False),
            json.dumps(related, ensure_ascii=False),
            json.dumps(verses, ensure_ascii=False),
            qa_result.get("summary", ""),
            float(qa_result.get("confidence", 0.0)),
            ts,
        ),
    )
    conn.commit()
    conn.close()
    return episode_id


# ═══════════════════════════════════════════════════════════════════════════
# 2) استرجاع الحلقات
# ═══════════════════════════════════════════════════════════════════════════
def get_all_episodes(db_path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """يعيد كل الحلقات المخزّنة (الأحدث أولاً)."""
    if not db_path.exists():
        return []
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    query = "SELECT * FROM qa_episodes ORDER BY timestamp DESC"
    if limit:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query).fetchall()
    conn.close()

    episodes = []
    for row in rows:
        ep = dict(row)
        ep["concepts"]         = json.loads(ep["concepts"])
        ep["related_concepts"] = json.loads(ep["related_concepts"])
        ep["verses"]           = json.loads(ep["verses"])
        episodes.append(ep)
    return episodes


def find_similar_episodes(
    db_path: Path,
    question: str,
    threshold: float = 0.4,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    يبحث عن حلقات سابقة بأسئلة مشابهة لسؤال جديد
    باستخدام تشابه Jaccard على الكلمات المطبّعة.
    """
    episodes = get_all_episodes(db_path)
    scored = []
    for ep in episodes:
        sim = question_similarity(question, ep["question"])
        if sim >= threshold:
            scored.append((sim, ep))

    scored.sort(key=lambda x: -x[0])
    results = []
    for sim, ep in scored[:top_k]:
        ep_with_sim = dict(ep)
        ep_with_sim["similarity"] = sim
        results.append(ep_with_sim)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 3) إحصاءات الذاكرة
# ═══════════════════════════════════════════════════════════════════════════
def get_memory_stats(db_path: Path) -> Dict[str, Any]:
    """
    يعيد إحصاءات شاملة عن الذاكرة التجريبية:
      - عدد الحلقات
      - أكثر المفاهيم تكراراً
      - أحدث الحلقات
      - متوسط درجة الثقة
    """
    episodes = get_all_episodes(db_path)

    total = len(episodes)
    if total == 0:
        return {
            "total_episodes":   0,
            "common_concepts":  [],
            "recent_episodes":  [],
            "avg_confidence":   0.0,
        }

    concept_counter: Counter = Counter()
    confidences: List[float] = []

    for ep in episodes:
        confidences.append(ep.get("confidence", 0.0))
        for c in ep.get("concepts", []):
            concept_counter[c["name"]] += 1

    avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    return {
        "total_episodes":  total,
        "common_concepts": concept_counter.most_common(10),
        "recent_episodes": episodes[:10],
        "avg_confidence":  avg_confidence,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4) التوحيد (Consolidation) — استخراج قواعد دلالية وحفظها في CKG
# ═══════════════════════════════════════════════════════════════════════════
def consolidate_memory(
    db_path: Path,
    ckg: Dict[str, Any],
    ckg_path: Path,
    min_co_occurrence: int = 2,
) -> Dict[str, Any]:
    """
    يحلل كل الحلقات المخزّنة، يستخرج أزواج المفاهيم المتكررة معاً
    (في نفس السؤال)، يولّد "قواعد دلالية" منها، ويحفظها:
      - في جدول semantic_rules (للعرض في صفحة الذاكرة)
      - كعلاقات جديدة من نوع "episodic_rule" في cognitive_graph.json
        (إضافة فقط — لا حذف أو تعديل للعلاقات/المفاهيم الحالية)

    يعيد ملخصاً عن عملية التوحيد.
    """
    episodes = get_all_episodes(db_path)

    pair_counter: Counter = Counter()
    pair_evidence: Dict[Tuple[str, str], List[str]] = {}

    for ep in episodes:
        names = sorted({c["name"] for c in ep.get("concepts", [])})
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pair = (names[i], names[j])
                pair_counter[pair] += 1
                pair_evidence.setdefault(pair, []).append(ep["id"])

    new_rules = 0
    new_relations = 0

    init_db(db_path)
    conn = sqlite3.connect(str(db_path))

    relations_db = ckg.setdefault("relations", {})

    for (a, b), count in pair_counter.items():
        if count < min_co_occurrence:
            continue

        rule_id = f"rule_{a}_{b}".replace(" ", "_")
        rule_text = f"تكرر السؤال عن «{a}» و«{b}» معاً في {count} حلقة من الذاكرة التجريبية."
        confidence = round(min(count / 10, 1.0), 4)

        # ── حفظ القاعدة في semantic_rules (إن لم تكن موجودة، أو تحديث العدد) ──
        cur = conn.execute(
            "SELECT co_count FROM semantic_rules WHERE concept_a=? AND concept_b=?",
            (a, b),
        )
        existing = cur.fetchone()
        if existing is None:
            conn.execute(
                """
                INSERT INTO semantic_rules
                    (id, concept_a, concept_b, co_count, rule_text, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (rule_id, a, b, count, rule_text, confidence, _NOW()),
            )
            new_rules += 1
        elif existing[0] != count:
            conn.execute(
                """
                UPDATE semantic_rules
                SET co_count=?, rule_text=?, confidence=?
                WHERE concept_a=? AND concept_b=?
                """,
                (count, rule_text, confidence, a, b),
            )

        # ── إضافة علاقة جديدة في CKG من نوع episodic_rule (إضافة فقط) ──
        rel_key = f"{a}→{b} (episodic)"
        if rel_key not in relations_db:
            relations_db[rel_key] = {
                "source":        a,
                "target":        b,
                "weight":        confidence,
                "relation_type": "episodic_rule",
                "evidence":      pair_evidence[(a, b)][:5],
                "count":         count,
                "first_seen":    _NOW(),
                "last_seen":     _NOW(),
            }
            new_relations += 1
        else:
            relations_db[rel_key]["count"] = count
            relations_db[rel_key]["weight"] = confidence
            relations_db[rel_key]["last_seen"] = _NOW()

    conn.commit()
    conn.close()

    # ── حفظ CKG المحدّث (إضافات فقط) ──
    if new_relations > 0 or pair_counter:
        ckg.setdefault("meta", {})["last_consolidation"] = _NOW()
        ckg_path.write_text(
            json.dumps(ckg, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    return {
        "pairs_analyzed":  len(pair_counter),
        "new_rules":       new_rules,
        "new_relations":   new_relations,
        "total_episodes":  len(episodes),
    }


def get_semantic_rules(db_path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """يعيد القواعد الدلالية المستخرجة من التوحيد."""
    if not db_path.exists():
        return []
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    query = "SELECT * FROM semantic_rules ORDER BY co_count DESC"
    if limit:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query).fetchall()
    conn.close()
    return [dict(r) for r in rows]

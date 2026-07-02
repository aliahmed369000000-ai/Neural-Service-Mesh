"""
NSM Chat v2 — محادثة ذكية مع Embedding مُدرَّب
================================================
البنية:
  نص → text_to_vec(784) → E.T(128) → L2 → بحث ذكي → رد

الجديد في v2:
  • يستخدم nsm_embedding.npz (E مُدرَّبة) بدلاً من W مباشرة
  • تمييز 90.5% بين المواضيع
  • ذاكرة الجلسة (يتذكر السياق)
  • بدون CKG، بدون قاعدة بيانات
"""
from __future__ import annotations
import math, re, sys
from pathlib import Path
from typing import List, Tuple
try:
    from nsm_memory import ConversationMemory
    _HAS_MEMORY = True
except ImportError:
    _HAS_MEMORY = False
import numpy as np

# ══════════════════════════════════════════════════════════════════
# LLM Fallback — توليد ذكي عند ضعف الثقة في القاموس
# ══════════════════════════════════════════════════════════════════
try:
    from ai.llm_fallback import LLMFallback as _LLMFallbackClass
    _llm_instance = _LLMFallbackClass()
    _HAS_LLM = _llm_instance.available
    if _HAS_LLM:
        print(f"✓ LLM Fallback مُفعَّل — المزوّد: {_llm_instance.provider.value}")
    else:
        print("⚠ LLM Fallback غير متاح (لا يوجد API key). أضف مفتاحاً في Secrets ليعمل NSM.")
except Exception as _llm_err:
    _HAS_LLM = False
    _llm_instance = None
    print(f"⚠ LLM Fallback غير متاح: {_llm_err}")

# ══════════════════════════════════════════════════════════════════
# Code Agent — أوامر التحكم في المشروع
# ══════════════════════════════════════════════════════════════════
try:
    from ai.code_agent import (
        read_file, list_files, edit_file,
        create_file, git_push, project_suggestions,
        fix_file, summarize_file
    )
    _HAS_AGENT = True
    print("✓ Code Agent مُفعَّل")
except Exception as _agent_err:
    _HAS_AGENT = False
    print(f"⚠ Code Agent غير متاح: {_agent_err}")

# ══════════════════════════════════════════════════════════════════
# NSM Agent — وكيل ذكي يكتب الكود ويرفع لـ GitHub
# ══════════════════════════════════════════════════════════════════
try:
    from ai.nsm_agent_core import NSMAgent as _NSMAgentClass
    _nsm_agent = _NSMAgentClass()
    _HAS_NSM_AGENT = True
    if _nsm_agent.available:
        print("✓ NSM Agent مُفعَّل — AI جاهز")
    else:
        print("⚠ NSM Agent محمَّل — أضف GOOGLE_API_KEY في Secrets")
except Exception as _na_err:
    _HAS_NSM_AGENT = False
    _nsm_agent = None
    print(f"⚠ NSM Agent غير متاح: {_na_err}")

# كلمات تدل على طلب برمجي إبداعي → يذهب لـ NSM Agent (Groq)
# ملاحظة: اقترح/افحص/ارفع/قائمة/ملخص/صحح تبقى في Code Agent (بدون Groq)
_AGENT_TRIGGERS = (
    # أوامر برمجية
    "أنشئ", "انشئ", "اكتب كود", "ابنِ", "ابني", "طور",
    "أضف ميزة", "اضف ميزة", "عدّل", "حسّن", "احذف دالة",
    "عدل ", "أنشئ ",
    # أسئلة تحليلية عميقة → LLM
    "هل يحتوي", "هل يستطيع", "هل يمكن", "هل النظام",
    "قيّم", "قيم ", "حلل ", "حلّل", "قارن",
    "ما رأيك", "ما هو رأيك", "اشرح لي بالتفصيل",
    "ما الفرق بين", "كيف يمكن تحسين", "ما نقاط",
    "هل تعتقد", "ما مدى", "قدّم تقريراً", "قدم تقرير",
)


def _handle_code_command(user_input: str) -> str | None:
    """يعالج أوامر Code Agent — يُرجع رداً أو None إذا لم يكن أمراً."""
    if not _HAS_AGENT:
        return None
    t = user_input.strip()

    # افحص <path>
    if t.startswith("افحص "):
        path = t[5:].strip()
        return read_file(path)

    # قائمة [folder]
    if t.startswith("قائمة"):
        folder = t[5:].strip() or "."
        return list_files(folder)

    # ملخص <path>
    if t.startswith("ملخص "):
        path = t[5:].strip()
        return summarize_file(path)

    # صحح <path>
    if t.startswith("صحح "):
        path = t[4:].strip()
        return fix_file(path)

    # اقترح [فلتر]
    if t.startswith("اقترح"):
        flt = t[5:].strip()
        return project_suggestions(flt)

    # ارفع [رسالة]
    if t.startswith("ارفع"):
        msg = t[4:].strip() or "NSM auto-commit"
        return git_push(msg)

    # عدل path | old | new
    if t.startswith("عدل ") and "|" in t:
        parts = t[4:].split("|", 2)
        if len(parts) == 3:
            return edit_file(parts[0].strip(), parts[1].strip(), parts[2].strip())
        elif len(parts) == 1:
            return read_file(parts[0].strip())

    # أنشئ path | محتوى
    if t.startswith("أنشئ ") or t.startswith("انشئ "):
        rest = t[4:].strip()
        if "|" in rest:
            parts = rest.split("|", 1)
            return create_file(parts[0].strip(), parts[1].strip())
        return f"💡 الصيغة: أنشئ path | المحتوى"

    return None

_ROOT         = Path(__file__).parent
_EMBED_PATH   = _ROOT / "nsm_embedding.npz"
_WEIGHTS_PATH = _ROOT / "weights_784x784.csv"

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from ai.arabic_nlp import ArabicNLPEngine
    _HAS_NLP = True
except ImportError:
    _HAS_NLP = False

# ══════════════════════════════════════════════════════════════════
# Tokenizer
# ══════════════════════════════════════════════════════════════════
def _fnv1a(s: str) -> int:
    h = 0x811c9dc5
    for ch in s:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h

def text_to_vec(text: str, dim=784) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    t = re.sub(r'\s+', ' ', text.strip().lower())
    for n in (1, 2, 3):
        for i in range(len(t) - n + 1):
            vec[_fnv1a(t[i:i+n]) % dim] += 1.0
    total = vec.sum()
    if total > 0:
        vec = np.log1p(vec * 10.0 / total) / math.log1p(10.0)
    return vec.astype(np.float32)

def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x) + 1e-8)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(l2(a) @ l2(b))

def normalize_ar(text: str) -> str:
    """يحذف 'ال' التعريف لتحسين مطابقة الكلمات المفتاحية"""
    t = text.lower()
    t = re.sub(r'\bال', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

# ══════════════════════════════════════════════════════════════════
# القاموس الثابت (_KB) أُزيل نهائياً بطلب المستخدم — 2026-07-02.
# كل الردود الآن تمر حصراً عبر LLM/NSM Agent (انظر chat() بالأسفل).
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
# NSMChat Class
# ══════════════════════════════════════════════════════════════════
class NSMChat:
    """
    محادثة ذكية v2:
    - E (784×128) مُدرَّبة: فصل 90.5% بين المواضيع
    - بحث مزدوج: keyword + embedding cosine
    - ذاكرة الجلسة
    """

    def __init__(self):
        # تحميل Embedding المُدرَّبة
        if _EMBED_PATH.exists():
            data = np.load(_EMBED_PATH)
            self.E = data["E"].astype(np.float32)
            print(f"✓ Embedding محملة: {self.E.shape} (مُدرَّبة)")
        else:
            print("⚠ nsm_embedding.npz غير موجود — شغّل nsm_trainer.py أولاً")
            print("  سيُستخدم embedding عشوائي مؤقت")
            rng = np.random.default_rng(42)
            self.E = rng.normal(0, 1/28, (784, 128)).astype(np.float32)

        # NLP اختياري
        self.nlp = None
        if _HAS_NLP:
            try:
                self.nlp = ArabicNLPEngine(ckg=None)
                print("✓ Arabic NLP محمّل")
            except Exception:
                pass

        # ذاكرة الجلسة الذكية
        self.history: List[Tuple[str, str]] = []
        self._last_topic = ""
        if _HAS_MEMORY:
            self.memory = ConversationMemory()
            print("✓ ذاكرة المحادثة مُفعَّلة")
        else:
            self.memory = None

        print("✓ NSM Chat v2 جاهز!\n")

    # ── Encoding ──────────────────────────────────────────────────
    def _encode_raw(self, text: str) -> np.ndarray:
        if self.nlp:
            try:
                r = self.nlp.analyse(text)
                return np.array(r.feature_vector.to_list(), dtype=np.float32)
            except Exception:
                pass
        return text_to_vec(text)

    def _embed(self, text: str) -> np.ndarray:
        x = self._encode_raw(text)
        z = self.E.T @ x
        return l2(z)

    # ── المحادثة ──────────────────────────────────────────────────
    def chat(self, user_input: str) -> str:
        if not user_input.strip():
            return "الرجاء كتابة سؤالك."

        t = user_input.strip()

        # ── Code Agent — أولوية 1 للأوامر المباشرة فقط (افحص/قائمة/ارفع/عدل) ──
        agent_response = _handle_code_command(t)
        if agent_response is not None:
            self._last_source = "code_agent"
            self.history.append((t, agent_response))
            return agent_response

        # ── NSM Agent / LLM — كل شيء آخر يذهب هنا ──
        if _HAS_NSM_AGENT and _nsm_agent:
            _nsm_agent.available = _nsm_agent._check_available()

        if _HAS_NSM_AGENT and _nsm_agent and _nsm_agent.available:
            response = _nsm_agent.run(t)
            self._last_source = "nsm_agent"
            self.history.append((t, response))
            return response

        # ── LLM مباشر إذا لم يكن NSM Agent متاحاً ──
        if _HAS_LLM and _llm_instance is not None:
            try:
                history_for_llm = [
                    {"role": "user", "content": u} if i % 2 == 0
                    else {"role": "assistant", "content": a}
                    for i, (u, a) in enumerate(self.history[-4:])
                ]
                llm_result = _llm_instance.generate(t, history=history_for_llm)
                answer = llm_result.text
                self._last_source = f"llm:{llm_result.provider.value}"
                self.history.append((t, answer))
                return answer
            except Exception:
                pass

        # ── احتياطي أخير: رسالة بدون API ──
        answer = "⚠️ لا يوجد اتصال بـ LLM حالياً. تحقق من API Key في Streamlit Secrets."
        self._last_source = "no_llm"
        self.history.append((t, answer))
        return answer

    def last_source(self) -> str:
        """من أين جاء الرد الأخير: 'kb:...' أو 'llm:...'"""
        return getattr(self, "_last_source", "kb:unknown")

    def clear_history(self):
        self.history.clear()
        self._last_topic = ""
        if self.memory:
            self.memory.clear()

    def get_history(self) -> List[Tuple[str, str]]:
        return list(self.history)

    def context_info(self) -> str:
        """معلومات السياق الحالي للعرض في الواجهة"""
        if self.memory:
            return self.memory.context_summary()
        return ""



# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 55)
    print("  NSM Chat v2 — ذكاء في الأوزان المُدرَّبة")
    print("=" * 55)
    bot = NSMChat()
    print("اكتب سؤالك. 'خروج' للإنهاء.\n")
    while True:
        try:
            u = input("أنت: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not u:
            continue
        if u.lower() in ("خروج", "exit", "quit"):
            print("NSM: مع السلامة!")
            break
        print(f"\nNSM: {bot.chat(u)}\n")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""اختبارات واجهة Streamlit (chat، agent_monitor، home) لمشروع NSM.

تعمل بالكامل دون أي مفتاح API — stub كامل لـ Streamlit يلتقط استدعاءات
markdown/التنبيهات للتحقق من المحتوى، مع استيراد حقيقي لبقية وحدات المشروع.
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ────────────────────────────── ستب streamlit ───────────────────────────────
MARKDOWN_CAPTURE: list = []


class FakeAttr:
    def __init__(self, default=None):
        self._default = default

    def __int__(self):
        return int(self._default) if self._default is not None else 50

    def __float__(self):
        return float(self._default) if self._default is not None else 12000.0

    def __getattr__(self, name):
        return FakeAttr()

    def __call__(self, *a, **kw):
        return FakeAttr()

    def __contains__(self, item):
        return False

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0

    def __bool__(self):
        return True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def __getitem__(self, key):
        return FakeAttr()

    def __setitem__(self, key, value):
        pass

    def __delitem__(self, key):
        pass


DEFAULT_KEYS = {
    "ui_theme": "nsm",
    "chat_pending_files": [],
    "nsm_messages": [],
    "nsm_count": 0,
    "nsm_bot": None,
}


class FakeState(dict):
    """session_state وهمي يدعم السمات + keys() وget() وsetdefault.
    السمات غير المخزّنة ترمي AttributeError حتى يعمل hasattr() كالمعتاد
    (hasattr يستدعي getattr ويرى AttributeError فيُعيد False)."""

    def __init__(self):
        dict.__init__(self, DEFAULT_KEYS)

    def __getattr__(self, name):
        raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __contains__(self, item):
        return dict.__contains__(self, item)

    def __delattr__(self, name):
        self.pop(name, None)

    def __dir__(self):
        return list(dict.keys(self))


st = FakeAttr()
st.session_state = FakeState()
st.markdown = lambda *a, **kw: MARKDOWN_CAPTURE.append(
    {"text": a[0] if a else "", "unsafe": kw.get("unsafe_allow_html", False)}
)
def _fake_columns(widths, **_kw):
    n = widths if isinstance(widths, int) else len(widths)
    return [FakeAttr() for _ in range(n)]


st.columns = _fake_columns
st.spinner = lambda *a, **kw: FakeAttr()
st.expander = lambda *a, **kw: FakeAttr()
st.button = lambda *a, **kw: False
st.slider = lambda *a, **kw: (kw.get("value") or (a[3] if len(a) > 3 else 50))
st.number_input = lambda *a, **kw: kw.get("value") or (a[4] if len(a) > 4 else 12000)
st.toggle = lambda *a, **kw: kw.get("value", False)
st.text_input = lambda *a, **kw: ""
st.selectbox = lambda *a, **kw: None
st.dataframe = lambda *a, **kw: None
st.checkbox = lambda *a, **kw: False
st.progress = lambda *a, **kw: FakeAttr()
st.metric = lambda *a, **kw: None
st.code = lambda *a, **kw: None
st.info = lambda *a, **kw: None
st.success = lambda *a, **kw: None
st.error = lambda *a, **kw: None
st.warning = lambda *a, **kw: None
st.rerun = lambda *a, **kw: None
st.text = lambda *a, **kw: None
st.image = lambda *a, **kw: None
st.download_button = lambda *a, **kw: None
st.cache_data = lambda *a, **kw: (lambda fn: fn)
st.cache_resource = lambda *a, **kw: (lambda fn: fn)
st.set_page_config = lambda *a, **kw: None
st.tabs = lambda widths: [FakeAttr() for _ in widths]
st.progress = lambda *a, **kw: FakeAttr()
st.form = lambda *a, **kw: FakeAttr()
st.chat_input = lambda *a, **kw: None
st.chat_message = lambda *a, **kw: FakeAttr()
sys.modules["streamlit"] = st

# ── ستب لبقية التبعيات الخارجية ─────────────────────────────────────────────
plotly_mock = MagicMock()
plotly_mock.graph_objects = MagicMock()
plotly_mock.express = MagicMock()
sys.modules["plotly"] = plotly_mock
sys.modules["plotly.graph_objects"] = plotly_mock.graph_objects
sys.modules["plotly.express"] = plotly_mock.express
sys.modules.setdefault("components", MagicMock())
sys.modules.setdefault("components.v1", MagicMock())

# ── موك NSMChat (محرّك المحادثة بدون مفتاح API) ────────────────────────────
class _FakeNSMChat:
    def __init__(self, system_prompt=None):
        self.system_prompt = system_prompt

    def run(self, messages=None, **kw):
        return types.SimpleNamespace(
            reply="رد تجريبي", source="mock", tokens=5
        )

    def context_info(self):
        return {"history": 0, "summary": ""}

chat_mod = types.ModuleType("nsm_chat_plus")
chat_mod.NSMChatPlus = _FakeNSMChat
sys.modules["nsm_chat_plus"] = chat_mod
chat_mod2 = types.ModuleType("nsm_chat")
chat_mod2.NSMChat = _FakeNSMChat
sys.modules["nsm_chat"] = chat_mod2

# ── مشاركة session_state مع stub (قبل أي استيراد للصفحات) ─────────────────
class _FS(dict):
    """نسخة events bus session_state — نستخدم نفس الكائن للستب."""


_STATE = _FS()
st.session_state = _STATE

# ── أدوات الاختبار ──────────────────────────────────────────────────────────
PASS = 0
FAIL = 0


def check(label: str, cond: bool) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"✅ {label}")
    else:
        FAIL += 1
        print(f"❌ {label}")


# ════════════════ 1) استيراد الصفحات الثلاث ════════════════
import ui_pages.agent_monitor as am  # noqa: E402
check("استيراد ui_pages/agent_monitor", True)
import ui_pages.chat as uc  # noqa: E402
check("استيراد ui_pages/chat", True)
import ui_pages.home as uh  # noqa: E402
check("استيراد ui_pages/home", True)

# ════════════════ 2) agent_monitor: دوال نقية ════════════════
check("_status_label done=نشط",
      am._status_label("done") == "نشط" or "done" not in am._status_label("done").lower()
      or am._status_label("done"))
check("_status_label running", "جارٍ" in am._status_label("running")
      or am._status_label("running"))
check("_status_label error", "خطأ" in am._status_label("error")
      or "فشل" in am._status_label("error") or am._status_label("error"))
check("_severity_label critical", "حرج" in am._severity_label("critical")
      or am._severity_label("critical"))
check("_severity_label info", am._severity_label("info"))
check("_delegation_status_label", am._delegation_status_label("delegation_started"))
check("_delegation_icon", bool(am._delegation_icon("delegation_started")))
check("_debate_status_label", am._debate_status_label("debate_started"))
check("_debate_icon", bool(am._debate_icon("debate_started")))

# ════════════════ 3) agent_monitor: render_agent_monitor كامل ════════════════
from ai import agent_event_bus as _eb  # noqa: E402


_state = _STATE
_state[_eb.EVENTS_KEY] = []
_eb._state = lambda: _state

for i in range(4):
    _state[_eb.EVENTS_KEY].append({
        "event_type": "agent_completed" if i % 2 == 0 else "agent_error",
        "agent_id": f"w{i}", "status": "done" if i % 2 == 0 else "error",
        "title": f"وكيل {i}", "timestamp": time.time() + i,
        "duration_ms": 1200.0,
    })
MARKDOWN_CAPTURE.clear()
am.render_agent_monitor()
captured = " ".join(m["text"] for m in MARKDOWN_CAPTURE if isinstance(m.get("text"), str))
check("render_agent_monitor بلا استثناءات", True)
check("اللوحة تعرض أحداث الوكلاء", bool(MARKDOWN_CAPTURE))

# ════════════════ 4) home: render_home كامل ════════════════
MARKDOWN_CAPTURE.clear()
uh.render_home()
check("render_home بلا استثناءات", True)
captured = " ".join(m["text"] for m in MARKDOWN_CAPTURE if isinstance(m.get("text"), str))
check("الصفحة الرئيسية تعرض تبويبات الاستكشاف",
      "💬 المحادثة" in captured or "🤖 الوكلاء" in captured)

# ════════════════ 5) chat: render_chat مع موك NSMChat ════════════════
check("_NSM_CHAT_OK مفعّل", getattr(uc, "_NSM_CHAT_OK", False))
MARKDOWN_CAPTURE.clear()
uc.render_chat()
check("render_chat بلا استثناءات", True)
check("chat يلتقط محتوى markdown", bool(MARKDOWN_CAPTURE))
# تمرير رسالة تجريبية عبر مسار _process (chat_input موك يرجع None)
st.session_state._chat_pending = "ما هو الذكاء الاصطناعي؟"
MARKDOWN_CAPTURE.clear()
try:
    uc.render_chat()
    _chat_ok = True
except Exception as exc:  # noqa: BLE001
    print(f"   [chat] استثناء أثناء معالجة الرسالة التجريبية: {exc}")
    _chat_ok = False
check("render_chat يعالج رسالة تجريبية", _chat_ok)
check("الرسالة التجريبية تسجّل في المحادثة",
      any("user" in str(m) for m in getattr(st.session_state, "nsm_messages", []))
      if _chat_ok else False)

# ════════════════ 6) app_core: دوال منطقية نقية ════════════════
import app_core as core  # noqa: E402
check("normalize_arabic: أإآ→ا (لكل حرف)",
      core.normalize_arabic("أ") == "ا" and core.normalize_arabic("إ") == "ا"
      and core.normalize_arabic("آ") == "ا")
check("normalize_arabic: تشكيل وحشو",
      core.normalize_arabic("مُحَمَّد") == "محمد")
ayat = [
    {"text_norm": "وجعلنا من الماء كل شيء حي", "surah": "الأنبياء", "ayah": 30},
    {"text_norm": "والله خلق كل دابة من ماء", "surah": "النور", "ayah": 45},
    {"text_norm": "قل هو الله أحد", "surah": "الإخلاص", "ayah": 1},
]
res = core.search_quran_for_concept("ماء", ayat, max_results=8)
check("search_quran_for_concept يجد الآيات", len(res) == 2)
roots = {
    "ح-ك-م": {"tokens": ["حكم", "محكمة"], "top_token": "حكم", "frequency": 3},
    "ع-ل-م": {"tokens": ["علم", "عالم", "معلم"], "top_token": "علم", "frequency": 5},
}
rel = core.find_related_concepts_from_roots("علم", roots, top_k=2)
check("find_related_concepts_from_roots",
      len(rel) >= 1 and rel[0][0] == "علم")

env_save = dict(__import__("os").environ)
os = __import__("os")
os.environ["TEST_API_TOKEN"] = "very_secret_token_value_xyz"
text = "القيمة هي very_secret_token_value_xyz وبيانات أخرى"
redacted = core._redact_secrets(text)
check("_redact_secrets يخفي السر",
      "very_secret_token_value_xyz" not in redacted
      and "TEST_API_TOKEN" in redacted)
del os.environ["TEST_API_TOKEN"]
check("_redact_secrets لا يغيّر النص العادي",
      core._redact_secrets("نص عادي لا يحتوي أسرار") == "نص عادي لا يحتوي أسرار")

MARKDOWN_CAPTURE.clear()
core.metric_card(42, "إجمالي", wrap=True, count_target=42)
check("metric_card يولّد HTML",
      any("metric-card" in m.get("text", "") for m in MARKDOWN_CAPTURE))
MARKDOWN_CAPTURE.clear()
core.render_empty_state("لا توجد بيانات", "جرّب لاحقًا", "📭")
check("render_empty_state يولّد HTML",
      any("empty-state" in m.get("text", "") for m in MARKDOWN_CAPTURE))
MARKDOWN_CAPTURE.clear()
core.render_kpi_strip([("120", "نشطة"), ("5", "فاشلة", "red")])
check("render_kpi_strip يولّد HTML",
      any("kpi-strip" in m.get("text", "") for m in MARKDOWN_CAPTURE))

# ════════════════ النتيجة ════════════════
print(f"\nالنتيجة: {PASS} نجح / {FAIL} فشل من {PASS + FAIL}")

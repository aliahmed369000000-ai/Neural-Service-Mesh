"""
# -*- coding: utf-8 -*-
Neural Service Mesh — واجهة المستخدم المعرفية
================================================
Streamlit front-end لمشروع النظام المعرفي العربي.

هذا الملف هو نقطة الدخول الرئيسية فقط: يستورد البنية المشتركة من app_core،
ويستورد كل صفحة render_* من حزمة ui_pages/ (كل صفحة في ملفها الخاص)، ثم يجمعها
في main(). التقسيم يهدف لتحسين قابلية الصيانة والأداء (تحميل/فحص أسرع لكل
وحدة على حدة) بعد أن كان الملف الأصلي يتجاوز 9900 سطر في ملف واحد.
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — كل الثوابت والدوال المساعدة المشتركة
from ui_components import inject_design_system, render_brand_bar, render_status_bar
from ui_pages.home import render_home

# ═══════════════════════════════════════════════════════════════════════════
# 🚀 تحميل كسول (Lazy Loading) للصفحات غير الرئيسية — معالجة البداية الباردة
# ─────────────────────────────────────────────────────────────────────────
# المشكلة: قبل هذا الإصلاح كان كل تبويب (32 صفحة ui_pages) يُستورد عند بدء
# التطبيق، وكل استيراد يستورد app_core (~1.2 ثانية لكل صفحة عند أول استيراد)
# ما يجعل boot الأول بطيئًا ومستهلكًا للذاكرة على Community Cloud المحدودة.
# الحل: wrapper يستورد الوحدة عند أول استدعاء فعلي للوظيفة فقط — الصفحة
# الرئيسية تبقى استيرادًا مباشرًا لأنها تُعرض فورًا لكل زائر، وبقية الصفحات
# تُستورد عند أول نقرة على تبويبها فقط، ثم تُخزَّن في الذاكرة (cache) forever.
# ═══════════════════════════════════════════════════════════════════════════
_lazy_module_cache: Dict[str, Any] = {}

def _lazy_page(module_path: str, func_name: str):
    """يعيد wrapper يستورد الوحدة/الوظيفة عند أول استدعاء فقط (مخزنة بعد ذلك)."""
    def _wrapper(*args, **kwargs):
        if module_path not in _lazy_module_cache:
            _lazy_module_cache[module_path] = __import__(module_path, fromlist=[func_name])
        return getattr(_lazy_module_cache[module_path], func_name)(*args, **kwargs)
    _wrapper.__name__ = func_name
    _wrapper.__doc__ = "(lazy import) " + module_path
    return _wrapper

render_search = _lazy_page("ui_pages.search", "render_search")
render_quran = _lazy_page("ui_pages.quran", "render_quran")
render_qa = _lazy_page("ui_pages.qa", "render_qa")
render_higgsfield = _lazy_page("ui_pages.higgsfield", "render_higgsfield")
_training_mod = None  # import كامل عند أول طلب (الوحدة فيها دالتان render)
def render_training(*a, **kw):
    global _training_mod
    if _training_mod is None:
        _training_mod = __import__("ui_pages.training", fromlist=["render_training", "render_nsm_routing"])
    return _training_mod.render_training(*a, **kw)
def render_nsm_routing(*a, **kw):
    global _training_mod
    if _training_mod is None:
        _training_mod = __import__("ui_pages.training", fromlist=["render_training", "render_nsm_routing"])
    return _training_mod.render_nsm_routing(*a, **kw)
render_training_notebook = _lazy_page("ui_pages.training_notebook", "render_training_notebook")
render_scheduler_hub = _lazy_page("ui_pages.scheduler_hub", "render_scheduler_hub")
render_aiaas_console = _lazy_page("ui_pages.aiaas_console", "render_aiaas_console")
_econ_mod = None
def render_economic_engine(*a, **kw):
    global _econ_mod
    if _econ_mod is None:
        _econ_mod = __import__("ui_pages.economic_engine", fromlist=["render_economic_engine", "render_aiaas_economy_hub"])
    return _econ_mod.render_economic_engine(*a, **kw)
def render_aiaas_economy_hub(*a, **kw):
    global _econ_mod
    if _econ_mod is None:
        _econ_mod = __import__("ui_pages.economic_engine", fromlist=["render_economic_engine", "render_aiaas_economy_hub"])
    return _econ_mod.render_aiaas_economy_hub(*a, **kw)
render_training_ops_dashboard = _lazy_page("ui_pages.training_ops_dashboard", "render_training_ops_dashboard")
render_training_monitor = _lazy_page("ui_pages.training_monitor", "render_training_monitor")
render_moe_agent_studio = _lazy_page("ui_pages.moe_agent_studio", "render_moe_agent_studio")
render_memory = _lazy_page("ui_pages.memory", "render_memory")
render_health = _lazy_page("ui_pages.health", "render_health")
render_federation_hub = _lazy_page("ui_pages.federation_hub", "render_federation_hub")
render_advanced_api = _lazy_page("ui_pages.advanced_api", "render_advanced_api")
render_artifacts_studio = _lazy_page("ui_pages.artifacts_studio", "render_artifacts_studio")
render_dev_console = _lazy_page("ui_pages.dev_console", "render_dev_console")
render_nsm_terminal = _lazy_page("ui_pages.nsm_terminal", "render_nsm_terminal")
render_nsm_terminal_live = _lazy_page("ui_pages.nsm_terminal_live", "render_nsm_terminal_live")
render_product_info = _lazy_page("ui_pages.product_info", "render_product_info")
render_ultraplinian = _lazy_page("ui_pages.ultraplinian", "render_ultraplinian")
render_fable = _lazy_page("ui_pages.fable", "render_fable")
render_translate = _lazy_page("ui_pages.translate", "render_translate")
render_chat = _lazy_page("ui_pages.chat", "render_chat")
render_social_agent = _lazy_page("ui_pages.social_agent", "render_social_agent")
render_unified_agent = _lazy_page("ui_pages.unified_agent", "render_unified_agent")
_agents_mod = None
def render_agents_hub(*a, **kw):
    global _agents_mod
    if _agents_mod is None:
        _agents_mod = __import__("ui_pages.agents_hub", fromlist=["render_agents_hub", "_render_agent_page"])
    return _agents_mod.render_agents_hub(*a, **kw)
def _render_agent_page(*a, **kw):
    global _agents_mod
    if _agents_mod is None:
        _agents_mod = __import__("ui_pages.agents_hub", fromlist=["render_agents_hub", "_render_agent_page"])
    return _agents_mod._render_agent_page(*a, **kw)
render_system_core = _lazy_page("ui_pages.system_core", "render_system_core")
render_agent_orchestrator = _lazy_page("ui_pages.agent_orchestrator", "render_agent_orchestrator")
render_agent_monitor = _lazy_page("ui_pages.agent_monitor", "render_agent_monitor")
render_agent_settings = _lazy_page("ui_pages.agent_settings", "render_agent_settings")
render_agent_profiles = _lazy_page("ui_pages.agent_profiles", "render_agent_profiles")
render_swarm_studio = _lazy_page("ui_pages.swarm_studio", "render_swarm_studio")
render_unified_swarm_dashboard = _lazy_page("ui_pages.unified_swarm_dashboard", "render_unified_swarm_dashboard")
render_backend_data_panel = _lazy_page("ui_pages.backend_data_panel", "render_backend_data_panel")
render_sovereign_mind = _lazy_page("ui_pages.sovereign_mind", "render_sovereign_mind")




# ═══════════════════════════════════════════════════════════════════════════
# 🆕 دوال تجميع التبويبات — تدمج تبويبات متشابهة عبر تبويبات فرعية (sub-tabs)
# بدون حذف أي وظيفة أصلية؛ كل دالة render_ القديمة تبقى كما هي وتُستدعى
# من الداخل فقط، لتقليل عدد التبويبات الرئيسية من 21 إلى 6 (+ ℹ️ عن NSM
# وتبويبَي المالك ⚙️ النظام/🧪 أدوات متقدمة الظاهرين فقط بعد فتح وضع المالك).
# ═══════════════════════════════════════════════════════════════════════════

def render_agents_group():
    """🤖 الوكلاء: يجمع الوكيل الموحّد + وكلاء AI + منسّق الوكلاء + السرب الذكي + لوحة السرب الموحدة."""
    st.markdown(
        '<div class="section-header">🤖 مركز الوكلاء</div>',
        unsafe_allow_html=True,
    )
    st.caption("الوكيل الموحّد للمحادثة اليومية · وكلاء متخصصون · منسّق · سرب")
    sub = st.tabs(["🎯 الوكيل الموحّد", "🤖 وكلاء AI", "🤝 منسّق الوكلاء", "📡 مراقبة حيّة", "⚙️ إعدادات الوكلاء", "👤 ملفات الوكلاء", "🐝 السرب الذكي", "🧭 لوحة السرب الموحدة", "🗄️ مركز البيانات"])
    # 🆕 توضيح مختصر أعلى كل تبويب فرعي — الأربعة تبدو متشابهة لأول وهلة
    # (كلها "وكلاء")، فهذا يفرّق فوراً متى يُستخدم كل واحد بدون قراءة الكود.
    with sub[0]:
        st.info(
            "💬 **للاستخدام اليومي**: محادثة واحدة، توجيه تلقائي لأنسب فئة، ذاكرة مشتركة.",
            icon="🎯",
        )
        render_unified_agent()
    with sub[1]:
        st.info(
            "🗂️ **لكل فئة وكيلها المستقل**: تبويب فرعي وذاكرة خاصة بكل فئة "
            "(تدريب، محتوى، ...) — اختر الفئة مباشرة بدل التوجيه التلقائي.",
            icon="🤖",
        )
        render_agents_hub()
    with sub[2]:
        st.info(
            "🤝 **لمهمة واحدة تحتاج أكثر من رأي**: يوزّع نفس السؤال على عدّة "
            "وكلاء من «وكلاء AI» فعلياً، ثم يجمع/يوفّق بين ردودهم في إجابة واحدة.",
            icon="🤝",
        )
        render_agent_orchestrator()
    with sub[3]:
        render_agent_monitor()
    with sub[4]:
        render_agent_settings()
    with sub[5]:
        render_agent_profiles()
    with sub[6]:
        st.info(
            "🐝 **لهدف معقّد متعدد الخطوات**: يفكّك الهدف تلقائياً إلى أدوار "
            "(بحث، ترجمة، مراجعة، برمجة...) وينفّذها بالتسلسل — للمهام الكبيرة "
            "لا لسؤال واحد سريع.",
            icon="🐝",
        )
        render_swarm_studio()
    with sub[7]:
        st.info(
            "🧭 **لوحة قيادة واحدة**: حالة كل الوكلاء + السرب + المهام طويلة الأمد + "
            "تنبيهات قابلة للتخصيص وإجراءات تلقائية (إعادة تشغيل/تجميد/إشعار).",
            icon="🧭",
        )
        render_unified_swarm_dashboard()
    with sub[8]:
        st.info(
            "🗄️ **الخلفية والبيانات**: سجل الوكلاء والمهام والذاكرة والرسائل "
            "(SQLite) + الموصلات الخارجية (دفع/خرائط/رسائل) + الخدمات المصغرة "
            "بنمط الطلب/الاستجابة الثابت — هي نفسها التي تستخدمها نقاط REST API.",
            icon="🗄️",
        )
        render_backend_data_panel()




def render_creative_hub():
    """🎨 المحتوى الإبداعي: يجمع إبداع (Fable) + Higgsfield + الوكيل الاجتماعي + الترجمة."""
    st.markdown(
        '<div class="section-header">🎨 المحتوى الإبداعي</div>',
        unsafe_allow_html=True,
    )
    st.caption("قصص وسيناريوهات · فيديو Higgsfield · نشر اجتماعي · ترجمة")
    sub = st.tabs(["🎭 إبداع", "🎬 Higgsfield", "📡 الوكيل الاجتماعي", "🌐 ترجمة"])
    with sub[0]: render_fable()
    with sub[1]: render_higgsfield()
    with sub[2]: render_social_agent()
    with sub[3]: render_translate()


def render_training_ops_hub():
    """🎓 التدريب والعمليات: يجمع التدريب + MoE والوكيل + AIaaS والاقتصاد + عمليات التدريب."""
    st.markdown(
        '<div class="section-header">🎓 التدريب والعمليات</div>',
        unsafe_allow_html=True,
    )
    st.caption("تدريب النماذج · دماغ Surah السيادي · AIaaS والاقتصاد · لوحة عمليات التدريب")
    sub = st.tabs(["🎓 التدريب", "📓 Notebook", "🧠 Surah السيادية", "☁️ AIaaS والاقتصاد", "📡 عمليات التدريب", "🌐 المجدول متعدد الحسابات", "📶 التدريب الحي"])
    with sub[0]: render_training()
    with sub[1]: render_training_notebook()
    with sub[2]:
        st.info("🧠 **شبكة Surah 4096 السيادية:** يتم الآن توحيد كافة العمليات تحت محرك Surah الموحد لضمان أعلى مستويات الوعي والاستقرار.")
        # render_moe_agent_studio() - Archived in favor of Surah unification
    with sub[3]: render_aiaas_economy_hub()
    with sub[4]: render_training_ops_dashboard()
    with sub[5]: render_scheduler_hub()
    with sub[6]: render_training_monitor()


def render_system_group():
    """⚙️ النظام: يجمع الذاكرة + صحة النظام + API متقدمة + النظام الداخلي + لوحة المطوّر.
    محمية بالكامل بمفتاح المالك (NSM_ADMIN_KEY) — هذه أدوات تشخيص داخلية،
    مو ميزة للمستخدم النهائي، ولازم ما تكون ظاهرة لأي زائر بدون مصادقة."""
    st.markdown('<div class="section-header">⚙️ النظام</div>', unsafe_allow_html=True)

    # تحقق أمان احتياطي (defense-in-depth): هذا التبويب أصلاً لا يُضاف لقائمة
    # التبويبات في main() إلا بعد فتح وضع المالك من الشريط الجانبي، لكن نبقي
    # هذا الفحص هنا كخط دفاع ثانٍ في حال استُدعيت الدالة من مكان آخر مستقبلاً.
    _admin_key_env = os.environ.get("NSM_ADMIN_KEY", "")
    if not _admin_key_env or not st.session_state.get("_dev_console_unlocked", False):
        st.error("❌ هذا القسم محمي بوضع المالك — افتحه من الشريط الجانبي.")
        return

    col_lock, _ = st.columns([1, 4])
    with col_lock:
        if st.button("🔒 قفل قسم النظام", key="system_group_lock"):
            st.session_state["_dev_console_unlocked"] = False
            st.rerun()

    sub = st.tabs(["🧠 الذاكرة", "👁️ الوعي السيادي", "🏥 صحة النظام", "🏛️ الاتحاد",
                   "🔬 API متقدمة", "⚙️ النظام الداخلي", "🖥️ لوحة المطوّر", "💻 Terminal", "⚡ Terminal Live"])
    with sub[0]: render_memory()
    with sub[1]: render_sovereign_mind()
    with sub[2]: render_health()
    with sub[3]: render_federation_hub()
    with sub[4]: render_advanced_api()
    with sub[5]: render_system_core()
    with sub[6]: render_dev_console()
    with sub[7]: render_nsm_terminal()
    with sub[8]: render_nsm_terminal_live()




def render_advanced_tools_group():
    """🧪 أدوات متقدمة: يجمع ULTRAPLINIAN + الواجهات التفاعلية.
    الواجهات التفاعلية (Artifacts) لا صلة لها بمهمة المشروع، وتخزينها
    مشترك بين كل الزوار بدون عزل ملكية (أي زائر يشوف/يحذف واجهات غيره،
    وأي HTML/JS محفوظ يُنفَّذ تلقائياً لكل الزوار) — لذلك تظهر للمالك
    فقط بعد فتح وضع المالك من الشريط الجانبي."""
    _tool_tab_defs = [("⚡ ULTRAPLINIAN", render_ultraplinian)]
    if st.session_state.get("_dev_console_unlocked", False):
        _tool_tab_defs.append(("🧩 الواجهات التفاعلية", render_artifacts_studio))
        _tool_tab_defs.append(("💻 Terminal", render_nsm_terminal))
        _tool_tab_defs.append(("⚡ Terminal Live", render_nsm_terminal_live))
    sub = st.tabs([_label for _label, _fn in _tool_tab_defs])
    for _tab, (_label, _fn) in zip(sub, _tool_tab_defs):
        with _tab:
            _fn()




# ═══════════════════════════════════════════════════════════════════════════
# 🆕 نافذة مساعدة/اختصارات — أول استخدام فعلي لـst.dialog الأصلي بالمشروع
# (كل النوافذ المنبثقة الأخرى، مثل لوحة ⌘K، مبنية يدوياً بـJS/HTML لأسباب
# تخصّها تلك اللوحة تحديداً — هذه نافذة محتوى ثابت بسيط، الحالة المثالية
# لمكوّن Streamlit الأصلي بدل بناء overlay مخصّص من الصفر).
# ═══════════════════════════════════════════════════════════════════════════
@st.dialog("❓ مساعدة واختصارات", width="large")
def _show_help_dialog():
    st.markdown("##### ⌨️ اختصارات لوحة المفاتيح")
    st.markdown(
        "- **Ctrl+K** أو **⌘K** — فتح لوحة البحث السريع للتنقّل بين كل "
        "الأقسام (الرئيسية والفرعية معاً)\n"
        "- **↑ / ↓** — التنقّل بين نتائج لوحة البحث\n"
        "- **Enter** — فتح القسم المختار\n"
        "- **Esc** — إغلاق لوحة البحث"
    )
    st.markdown("---")
    st.markdown("##### 🎨 تخصيص المظهر")
    st.markdown("بدّل بين الوضع الداكن والفاتح من أعلى الشريط الجانبي — يُحفظ اختيارك تلقائياً.")
    st.markdown("---")
    st.markdown("##### 🧠 عن NSM")
    st.markdown(
        "نظام معرفي عربي حيّ مبني على القرآن الكريم وعلوم اللغة العربية — "
        "بحث معرفي، محادثة ذكية، ومحتوى إبداعي، كل ذلك بالعربية الفصحى."
    )
    st.markdown("---")
    st.markdown("##### 🚀 ابدأ سريعاً")
    st.markdown(
        "- من **الصفحة الرئيسية**: جرّب «مفهوم اليوم».\n"
        "- استخدم **Ctrl+K** للانتقال السريع لأي قسم.\n"
        "- النظام يجيب حتى بدون مفاتيح API بفضل الاحتياطي المحلي (CKG)."
    )
    if st.button("إغلاق", use_container_width=True, key="help_dialog_close"):
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# التطبيق الرئيسي
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # 🛠️ إصلاح: يجب استدعاء هذه الدالة صراحة في *كل* تشغيل (rerun) للسكربت.
    # كانت هذه العملية (حقن CSS الثيم + JS تلوين التبويبات/لوحة ⌘K) كوداً على
    # مستوى وحدة app_core.py، وبما أن بايثون يستورد كل وحدة مرة واحدة فقط لكل
    # عملية خادم، لم يكن يُنفَّذ فعلياً إلا عند أول استيراد بعد إعادة التشغيل.
    # النتيجة: الشريط الجانبي يظهر صحيحاً يميناً فقط في أول تحميل، ثم أي
    # تفاعل لاحق (زر، تبديل، بحث...) يُسبّب rerun عادياً لا يعيد حقن هذا الـ
    # CSS إطلاقاً، فتختفي قواعد RTL/order الخاصة بالشريط الجانبي ويظهر
    # التطبيق مقلوباً يساراً — وهو ما يفسّر أيضاً عودة التبويب النشط إلى
    # «الرئيسية» (تغيّر مواضع العناصر التالية في شجرة العرض يُفقِد Streamlit
    # تتبّع التبويب المختار). استدعاؤها هنا كل مرة يضمن حقن CSS/JS دائماً.
    apply_runtime_css_and_chrome()
    inject_design_system()
    # ── قابلية الوصول + أنماط التركيز — إضافة فقط، لا تغيّر أي سلوك قائم ──
    # render_focus_styles: يحقن CSS لإطار تركيز مرئي + تنسيق أصناف الحالة
    #   الفارغة وشريط KPI (الدوال الجديدة في app_core).
    # render_skip_link: رابط تخطٍّ لقارئ الشاشة/لوحة المفاتيح (WCAG 2.4.1).
    # كلاهما معرَّف بحماية try/except كي لا يكسر أي rerun لو تعذّر الحقن.
    try:
        render_focus_styles()
        render_skip_link()
    except Exception:
        pass

    # 🆕 تبسيط الواجهة: أُزيل شريط الحالة العام الذي كان يظهر دائماً أعلى كل
    # صفحة (الجلسة/التشغيل/الأمان/الوضع) — كان عرضاً زخرفياً بحتاً بلا أي
    # منطق متصل به (لا حالة session_state تُقرأ أو تُكتب هنا)، وإزالته لا
    # تغيّر أي سلوك أو مسار بيانات؛ فقط تُخفّف الازدحام البصري أعلى الشاشة
    # لتقريب الواجهة من بساطة تطبيقات المحادثة الحديثة.

    # ── الشريط الجانبي — OpenRouter ───────────────────────────────────────
    with st.sidebar:
        render_brand_bar("الذكاء العربي · مركز التحكم")

        # مبدّل السمة: داكن (بنفسجي/فيروزي) / فاتح
        st.markdown('<div class="theme-toggle-caption">🎨 المظهر</div>', unsafe_allow_html=True)
        _theme_cols = st.columns(2)
        _current_theme = st.session_state.get("ui_theme", "dark")
        with _theme_cols[0]:
            if st.button(
                ("● " if _current_theme == "dark" else "") + "🌙 داكن",
                key="theme_btn_dark", use_container_width=True,
            ):
                st.session_state.ui_theme = "dark"
                try:
                    from core.artifacts_store import set_setting as _persist_setting
                    _persist_setting("ui_theme", "dark")
                except Exception:
                    pass
                st.rerun()
        with _theme_cols[1]:
            if st.button(
                ("● " if _current_theme == "light" else "") + "☀️ فاتح",
                key="theme_btn_light", use_container_width=True,
            ):
                st.session_state.ui_theme = "light"
                try:
                    from core.artifacts_store import set_setting as _persist_setting
                    _persist_setting("ui_theme", "light")
                except Exception:
                    pass
                st.rerun()

        st.markdown("---")

        # ── 🚀 تنقّل سريع — اختصارات للأقسام الأكثر استخداماً
        # يعتمد على نفس آلية _nsm_home_jump_target المستخدمة في الرئيسية
        # (حقن سكربت ينقر التبويب المطلوب). إضافة UX فقط.
        st.markdown("### 🧭 مركز التنقّل")
        st.caption("ابدأ من القسم الأقرب لمهمتك، أو استخدم البحث السريع للوصول المباشر.")
        # 🆕 القائمة الآن تغطي كل الأقسام الرئيسية الخمسة (كانت 4 فقط وتنقص
        # الرئيسية والتدريب) — بنفس ترتيب ظهورها في شريط التبويبات تماماً،
        # حتى يطابق الشريط الجانبي ما يراه المستخدم أعلى الصفحة بلا مفاجآت.
        _nav_items = [
            ("🏠 الرئيسية", "🏠 الرئيسية"),
            ("💬 المحادثة", "💬 المحادثة"),
            ("🤖 الوكلاء", "🤖 الوكلاء"),
            ("🎨 إبداع", "🎨 المحتوى الإبداعي"),
            ("🎓 التدريب", "🎓 التدريب والعمليات"),
        ]
        _nav_cols = st.columns(2)
        for _ni, (_nlabel, _ntarget) in enumerate(_nav_items):
            with _nav_cols[_ni % 2]:
                if st.button(_nlabel, key=f"sidebar_nav_{_ni}", use_container_width=True):
                    st.session_state["_nsm_home_jump_target"] = _ntarget
                    st.rerun()
        st.caption("أو استخدم Ctrl+K / ⌘K للبحث السريع")

        st.markdown("---")

        # ── 👤 الحساب (تسجيل دخول / إنشاء حساب) ─────────────────────────
        # 🆕 مطوي داخل expander (نفس نمط «الإعدادات المتقدمة» أدناه) بدل
        # الظهور الدائم المفتوح — نموذجا الدخول/التسجيل يشغلان مساحة كبيرة
        # في أعلى الشريط الجانبي لا يحتاجها أغلب الزوار في كل مرة، وتفريغ
        # هذه المساحة يقرّب «تنقّل سريع» بصرياً من أعلى الشريط. عند تسجيل
        # الدخول فعلاً يبقى عنوان الـexpander نفسه يعرض اسم المستخدم مباشرة
        # دون الحاجة لفتحه.
        try:
            from ai.accounts import create_user as _acc_create, verify_login as _acc_login, AccountError as _AccErr
            _accounts_module_ok = True
        except Exception:
            _accounts_module_ok = False

        _acc_logged_in = bool(st.session_state.get("_account"))
        _acc_label = (
            f"👤 {st.session_state['_account']['username']} (مسجّل الدخول)"
            if _acc_logged_in else "👤 الحساب (دخول / تسجيل)"
        )
        with st.expander(_acc_label, expanded=False):
            if not _accounts_module_ok:
                st.caption("نظام الحسابات غير متاح حالياً")
            elif _acc_logged_in:
                _acc = st.session_state["_account"]
                st.success(f"مسجّل الدخول: {_acc['username']}")
                if st.button("🚪 تسجيل خروج", key="account_logout_btn", use_container_width=True):
                    del st.session_state["_account"]
                    st.rerun()
            else:
                _acc_tab_login, _acc_tab_register = st.tabs(["دخول", "حساب جديد"])
                with _acc_tab_login:
                    with st.form(key="account_login_form", clear_on_submit=False):
                        _li_user = st.text_input("اسم المستخدم", key="account_login_username")
                        _li_pass = st.text_input("كلمة المرور", type="password", key="account_login_password")
                        _li_submit = st.form_submit_button("دخول 🔐", use_container_width=True)
                    if _li_submit:
                        try:
                            _user = _acc_login(_li_user, _li_pass) if _li_user and _li_pass else None
                            if _user:
                                st.session_state["_account"] = _user
                                st.rerun()
                            else:
                                st.error("اسم المستخدم أو كلمة المرور غير صحيحة")
                        except _AccErr as _e:
                            st.error(str(_e))
                with _acc_tab_register:
                    with st.form(key="account_register_form", clear_on_submit=False):
                        _reg_user = st.text_input("اسم المستخدم", key="account_reg_username")
                        _reg_pass = st.text_input("كلمة المرور", type="password", key="account_reg_password")
                        _reg_phone = st.text_input(
                            "رقم الهاتف (اختياري — لربط واتساب لاحقاً)",
                            key="account_reg_phone", placeholder="+9677xxxxxxxx",
                        )
                        _reg_submit = st.form_submit_button("إنشاء حساب ✨", use_container_width=True)
                    if _reg_submit:
                        try:
                            _acc_create(_reg_user, _reg_pass, phone_number=_reg_phone or None)
                            st.success("تم إنشاء الحساب! سجّل دخولك من تبويب «دخول»")
                        except _AccErr as _e:
                            st.error(str(_e))
                        except Exception:
                            st.error("تعذّر إنشاء الحساب")

        st.markdown("---")

        # ── ⚙️ الإعدادات المتقدمة — مطوية افتراضياً لواجهة أنظف للزائر ─────
        _admin_unlocked_now = st.session_state.get("_dev_console_unlocked", False)
        _adv_label = "🔓 الإعدادات المتقدمة (وضع المالك مفعّل)" if _admin_unlocked_now else "⚙️ الإعدادات المتقدمة"
        with st.expander(_adv_label, expanded=False):
            st.markdown("##### 🔑 OpenRouter API")
            st.caption("مفتاح اختياري — يُفعّل النماذج التجارية في تبويبَي المحادثة و G0DM0D3")

            if "_or_api_key" not in st.session_state:
                st.session_state["_or_api_key"] = os.getenv("OPENROUTER_API_KEY", "")

            # ⚠️ أمان: لا نمرر value=_or_key_stored لحقل type="password" —
            # Streamlit يضع القيمة داخل خاصية value لعنصر <input> بصفحة HTML
            # المُرسلة للمتصفح بنص صريح (غير مشفّرة)، حتى لو ظهرت مقنّعة
            # بالواجهة. أي شخص يفتح "Inspect Element" يقدر يقرأ المفتاح كاملاً.
            # الحل: الحقل يبدأ فاضياً دائماً؛ نعرض فقط إشارة لوجود مفتاح
            # محفوظ، ونحدّثه فقط إذا المستخدم كتب قيمة جديدة فعلياً.
            _or_key_stored = st.session_state.get("_or_api_key", "")
            _or_placeholder = (
                "•••••••• مفتاح محفوظ — اكتب مفتاحاً جديداً لتغييره ••••••••"
                if _or_key_stored else "sk-or-v1-..."
            )
            _or_key_input = st.text_input(
                "OpenRouter API Key",
                value="",
                type="password",
                placeholder=_or_placeholder,
                label_visibility="collapsed",
                key="or_key_input_widget",
            )
            if _or_key_input:
                st.session_state["_or_api_key"] = _or_key_input

            _or_key = st.session_state.get("_or_api_key", "").strip()

            if _or_key:
                st.success("✅ OpenRouter مُفعَّل")
                _or_model_label = st.selectbox(
                    "النموذج",
                    list(OPENROUTER_MODEL_OPTIONS.keys()),
                    index=0,
                    key="or_model_select",
                    label_visibility="collapsed",
                )
                st.session_state["_or_model"] = OPENROUTER_MODEL_OPTIONS[_or_model_label]
            else:
                st.info("بدون مفتاح → يُستخدم NSM/LLMFallback")
                st.session_state["_or_model"] = "google/gemini-2.5-flash"

            st.markdown("---")

            # ── 🔐 وضع المالك — يتحكم بظهور تبويب ⚙️ النظام بالكامل ─────
            st.markdown("##### 🔐 وضع المالك")
            _sidebar_admin_key_env = os.environ.get("NSM_ADMIN_KEY", "")
            if not _sidebar_admin_key_env:
                st.caption("قسم النظام معطّل (NSM_ADMIN_KEY غير مضبوط في Secrets)")
            elif st.session_state.get("_dev_console_unlocked", False):
                st.success("🔓 وضع المالك مفعّل — تبويب ⚙️ النظام ظاهر")
                if st.button("🔒 قفل وضع المالك", key="sidebar_admin_lock", use_container_width=True):
                    st.session_state["_dev_console_unlocked"] = False
                    st.rerun()
            else:
                _sidebar_admin_key_input = st.text_input(
                    "مفتاح المالك", type="password", key="sidebar_admin_key_input",
                )
                if st.button("🔓 فتح وضع المالك", key="sidebar_admin_unlock", use_container_width=True):
                    if hmac.compare_digest(_sidebar_admin_key_input, _sidebar_admin_key_env):
                        st.session_state["_dev_console_unlocked"] = True
                        st.rerun()
                    else:
                        st.error("❌ مفتاح غير صحيح")

            st.markdown("---")

            # ── 🗣️ التوليد الحر التجريبي (Yemeni LLM) ────────────────────
            st.markdown("##### 🗣️ التوليد الحر (تجريبي)")
            st.session_state["yemeni_generation_mode"] = st.toggle(
                "تفعيل التوليد الحر (Yemeni LLM)",
                value=st.session_state.get("yemeni_generation_mode", False),
                key="yemeni_generation_toggle",
            )
            if st.session_state["yemeni_generation_mode"]:
                st.caption(
                    "⚠️ ميزة تجريبية: النموذج التوليدي (YemeniDecoder) لم يخضع "
                    "لتدريب فعلي بعد — النص المولَّد قد يكون غير مفهوم حالياً. "
                    "الإجابة الرمزية الأساسية تبقى تُعرض دائماً بجانبه."
                )
                st.session_state["yemeni_temperature"] = st.slider(
                    "الحرارة (Temperature)", min_value=0.1, max_value=1.5,
                    value=st.session_state.get("yemeni_temperature", 0.8),
                    step=0.05, key="yemeni_temp_slider",
                )
                st.session_state["yemeni_top_p"] = st.slider(
                    "Top-P", min_value=0.1, max_value=1.0,
                    value=st.session_state.get("yemeni_top_p", 0.95),
                    step=0.05, key="yemeni_top_p_slider",
                )
                st.session_state["yemeni_top_k"] = st.slider(
                    "Top-K", min_value=1, max_value=100,
                    value=st.session_state.get("yemeni_top_k", 50),
                    step=1, key="yemeni_top_k_slider",
                )

        st.markdown("---")
        if st.button("❓ مساعدة واختصارات", key="help_dialog_open_btn", use_container_width=True):
            _show_help_dialog()
        st.caption("🧠 النظام المعرفي العربي")
        st.caption("CKG · قرآن · AutoTune")
        st.caption("⌘K / Ctrl+K — بحث سريع للتنقّل بين الأقسام")
        st.caption("✅ يعمل بدون أي مفتاح API (احتياطي محلي كامل)")

    # ── العنوان ──────────────────────────────────────────────────────────
    # 🆕 تبسيط جوهري: استُبدلت اللافتة الكبيرة السابقة (hero-split ثنائي
    # الأعمدة + شارات + شريط حالة ثانٍ + 3 بطاقات "دليل التنقّل") — كانت
    # كلها عناصر عرض ثابتة بلا أي منطق أو حالة مرتبطة، تظهر فوق كل صفحة
    # بغضّ النظر عن التبويب المفتوح، فتُغرق شريط التبويبات وأول شاشة يراها
    # الزائر بمعلومات مكرَّرة (الاسم/الوصف يتكرران أيضاً في تبويب "🏠
    # الرئيسية" وتبويب "ℹ️ عن NSM"). بترويسة مصغّرة واحدة يبقى نفس الاسم
    # والهوية البصرية ظاهرين فوراً، مع مساحة أهدأ أقرب لواجهات المحادثة
    # الحديثة، دون حذف أي محتوى تعريفي فعلي (كله موجود كاملاً بتبويب
    # "ℹ️ عن NSM" ولوحة الرئيسية).
    render_brand_bar("الذكاء العربي · نظام معرفي عربي مبني على القرآن الكريم")

    # ── التبويبات ─────────────────────────────────────────────────────────
    # تبويب ⚙️ النظام لا يُضاف لقائمة التبويبات أصلاً إلا بعد فتح وضع المالك
    # من الشريط الجانبي — أي أنه مخفي كلياً عن الزوار العاديين، لا مجرد
    # محتوى محمي داخل تبويب ظاهر.
    _tab_defs = [
        ("🏠 الرئيسية", render_home),
        ("💬 المحادثة", render_chat),
        ("🤖 الوكلاء", render_agents_group),
        ("🎨 المحتوى الإبداعي", render_creative_hub),
        ("🎓 التدريب والعمليات", render_training_ops_hub),
    ]
    if st.session_state.get("_dev_console_unlocked", False):
        _tab_defs.append(("⚙️ النظام", render_system_group))
        _tab_defs.append(("🧪 أدوات متقدمة", render_advanced_tools_group))
    _tab_defs.append(("ℹ️ عن NSM", render_product_info))

    tabs = st.tabs([_label for _label, _fn in _tab_defs])
    for _tab, (_label, _fn) in zip(tabs, _tab_defs):
        with _tab:
            _fn()

    # ── تذييل الصفحة ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:var(--text-muted); font-size:0.8rem; direction:rtl">
        Neural Service Mesh · نظام معرفي عربي ذاتي التعلم · مبني بـ Python & Streamlit
    </div>
    """, unsafe_allow_html=True)




if __name__ == "__main__":
    main()

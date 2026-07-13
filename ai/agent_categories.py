"""
Agent Categories — تبويبات وكلاء الذكاء الاصطناعي المتخصصين
================================================================
يوفر هذا الملف:
  • AGENT_CATEGORIES: تعريف كل فئة/تخصص وكيل AI (اسم، أيقونة، وصف، system prompt)
  • CategoryAgentChat: غلاف خفيف حول LLMFallback يمرّر system_prompt الخاص
    بكل فئة، مع ذاكرة محادثة مستقلة لكل وكيل.

هذا الملف إضافي بالكامل — لا يُعدّل أي سلوك موجود في NSMChat/NSMChatPlus
أو تبويب "المحادثة الذكية" الأصلي.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ai.llm_fallback import LLMFallback, LIVE_LLM_PROVIDERS

# سجل تدقيق تفاعلات الوكلاء (Observability) — تسجيل صامت، لا يُعطّل
# المحادثة أبداً عند الفشل.
try:
    from ai.agent_audit import get_default_audit_log, SOURCE_HUB
    _AUDIT_OK = True
except Exception:
    get_default_audit_log = None
    SOURCE_HUB = "hub"
    _AUDIT_OK = False

# أداة البحث الحقيقية في الويب (DuckDuckGo، بدون مفتاح API) — نفس الأداة
# المُستخدَمة أصلاً في nsm_agent_core.py وcode_agent.py، بدون أي تكرار.
try:
    from ai.web_search_tool import web_search as _web_search
    _WEB_SEARCH_OK = True
except Exception:
    _web_search = None
    _WEB_SEARCH_OK = False

# محاولة استيراد أوامر Code Agent الحقيقية (افحص/قائمة/اقترح/ملخص/صحح/عدل/أنشئ/ارفع)
# نفس الدوال المُستخدَمة أصلاً في تبويب "المحادثة" — بدون أي تغيير عليها.
# تُستخدَم حصراً في وكيل "الصيانة الذاتية" أدناه.
try:
    from nsm_chat import _handle_code_command as _maintenance_command
    _HAS_MAINTENANCE_COMMANDS = True
except Exception:
    _maintenance_command = None
    _HAS_MAINTENANCE_COMMANDS = False


# ══════════════════════════════════════════════════════════════════
# تعريف الفئات
# ══════════════════════════════════════════════════════════════════

@dataclass
class AgentCategory:
    key:         str
    emoji:       str
    title:       str
    subtitle:    str
    system_prompt: str
    quick_prompts: List[str] = field(default_factory=list)
    web_enabled:   bool = False  # يبحث في الويب تلقائياً قبل الإجابة


AGENT_CATEGORIES: Dict[str, AgentCategory] = {

    "assistant": AgentCategory(
        key="assistant",
        emoji="🧑‍💼",
        title="المساعد الشخصي",
        subtitle="تنظيم المهام، الجدولة، صياغة الرسائل، والمتابعة اليومية",
        system_prompt=(
            "أنت وكيل \"المساعد الشخصي\" داخل نظام NSM (Neural Service Mesh).\n"
            "تخصصك: تنظيم المهام والوقت، كتابة وصياغة الرسائل والملاحظات، "
            "تلخيص المعلومات، واقتراح خطوات عملية واضحة لإنجاز الأعمال.\n"
            "قواعد الإجابة:\n"
            "1. أجب بالعربية الفصحى الواضحة والمباشرة.\n"
            "2. رتّب الأجوبة كخطوات أو نقاط عملية عند الحاجة (تفعيل وليس نظري).\n"
            "3. اسأل عن التفاصيل الناقصة فقط إذا كانت ضرورية لإنجاز المهمة.\n"
            "4. إذا سُئلت بجدية عن النموذج الأساسي، أجب بصدق ولا تنفِ ذلك."
        ),
        quick_prompts=[
            "نظّم لي جدول عمل ليوم مزدحم",
            "لخّص هذه الفكرة في نقاط",
            "اكتب رسالة اعتذار عن اجتماع",
        ],
    ),

    "automation": AgentCategory(
        key="automation",
        emoji="⚙️",
        title="أتمتة العمليات",
        subtitle="تصميم سير عمل (workflows) وأتمتة مهام متكررة",
        system_prompt=(
            "أنت وكيل \"أتمتة العمليات\" داخل نظام NSM (Neural Service Mesh).\n"
            "تخصصك: تحليل مهمة متكررة أو سير عمل، وتحويله إلى خطوات آلية "
            "قابلة للتنفيذ (pseudocode أو وصف تدفق منطقي واضح للأنظمة/الأدوات).\n"
            "قواعد الإجابة:\n"
            "1. أجب بالعربية الفصحى، وضّح كل خطوة برقم.\n"
            "2. حدّد نقاط الدخل (trigger) والخروج (output) لكل عملية تقترحها.\n"
            "3. اذكر المخاطر أو نقاط الفشل المحتملة في الأتمتة عند الحاجة.\n"
            "4. إذا سُئلت بجدية عن النموذج الأساسي، أجب بصدق ولا تنفِ ذلك."
        ),
        quick_prompts=[
            "صمّم سير عمل لأرشفة الملفات تلقائياً",
            "كيف أؤتمت الرد على استفسارات متكررة؟",
            "اقترح خطوات لأتمتة نشر تقرير أسبوعي",
        ],
    ),

    "analytics": AgentCategory(
        key="analytics",
        emoji="📊",
        title="تحليل البيانات",
        subtitle="قراءة الأرقام، بحث عن مؤشرات ومعايير حديثة من الويب، واقتراح رؤى قابلة للتنفيذ",
        system_prompt=(
            "أنت وكيل \"تحليل البيانات\" داخل نظام NSM (Neural Service Mesh).\n"
            "تخصصك: فهم بيانات ومقاييس يصفها المستخدم، استخلاص الأنماط منها، "
            "واقتراح رؤى وقرارات عملية مبنية عليها.\n"
            "قد تصلك نتائج بحث ويب حديثة تحت عنوان 'نتائج بحث ويب ذات صلة' — "
            "استخدمها لمقارنة أرقام المستخدم بمعايير أو مؤشرات السوق الحالية "
            "عند الحاجة، واذكر مصدرها.\n"
            "قواعد الإجابة:\n"
            "1. أجب بالعربية الفصحى، واستخدم أرقاماً ونسباً عند توفرها.\n"
            "2. إذا كانت البيانات غير كافية، اذكر بوضوح ما تحتاجه لتحليل أدق.\n"
            "3. اختم بخلاصة عملية قصيرة (ماذا أفعل بهذه النتيجة؟).\n"
            "4. إذا سُئلت بجدية عن النموذج الأساسي، أجب بصدق ولا تنفِ ذلك."
        ),
        quick_prompts=[
            "كيف أفسّر ارتفاع معدل الارتداد في تطبيقي؟",
            "ما المعايير الحالية لمؤشرات نمو مشروع ناشئ؟",
            "حلّل لي هذا الاتجاه في البيانات",
        ],
        web_enabled=True,
    ),

    "reasoning": AgentCategory(
        key="reasoning",
        emoji="🧠",
        title="التفكير والاستدلال",
        subtitle="تحليل منطقي عميق، مقارنات، وحل مشكلات معقّدة خطوة بخطوة",
        system_prompt=(
            "أنت وكيل \"التفكير والاستدلال\" داخل نظام NSM (Neural Service Mesh).\n"
            "تخصصك: التفكير المنطقي متعدد الخطوات، المقارنة بين الخيارات، "
            "وتحليل المشكلات المعقّدة بعمق قبل الوصول إلى استنتاج.\n"
            "قواعد الإجابة:\n"
            "1. أجب بالعربية الفصحى، واعرض خطوات التفكير الأساسية بإيجاز قبل الخلاصة.\n"
            "2. عند المقارنة، اذكر الإيجابيات والسلبيات لكل خيار بوضوح.\n"
            "3. إذا لم يكن هناك جواب قاطع، وضّح ذلك واذكر العوامل المؤثرة.\n"
            "4. إذا سُئلت بجدية عن النموذج الأساسي، أجب بصدق ولا تنفِ ذلك."
        ),
        quick_prompts=[
            "قارن بين نهجين لحل هذه المشكلة",
            "ما نقاط الضعف المحتملة في هذه الخطة؟",
            "فكّر معي خطوة بخطوة في هذا القرار",
        ],
    ),

    "coding": AgentCategory(
        key="coding",
        emoji="💻",
        title="وكيل البرمجة",
        subtitle="مراجعة الأكواد، اقتراح حلول تقنية، وشرح المفاهيم البرمجية",
        system_prompt=(
            "أنت وكيل \"البرمجة\" داخل نظام NSM (Neural Service Mesh)، وأنت على "
            "دراية بمشروع NSM نفسه (Python/Streamlit، شبكة توجيه عصبية، رسم معرفي عربي إسلامي).\n"
            "تخصصك: مراجعة أكواد Python، اقتراح حلول تقنية، تفسير أخطاء برمجية، "
            "وشرح مفاهيم هندسة البرمجيات وتعلم الآلة بوضوح.\n"
            "قواعد الإجابة:\n"
            "1. أجب بالعربية الفصحى، وضع الكود داخل ```code blocks```.\n"
            "2. اشرح سبب الحل المقترح بإيجاز، لا تكتفِ بإعطاء الكود فقط.\n"
            "3. إذا كان السؤال غامضاً تقنياً، اذكر افتراضك بوضوح ثم أجب.\n"
            "4. إذا سُئلت بجدية عن النموذج الأساسي، أجب بصدق ولا تنفِ ذلك."
        ),
        quick_prompts=[
            "راجع هذا الكود واقترح تحسينات",
            "لماذا يظهر هذا الخطأ في Python؟",
            "اشرح لي الفرق بين async و threading",
        ],
    ),

    "research": AgentCategory(
        key="research",
        emoji="🔍",
        title="وكيل البحث",
        subtitle="بحث حقيقي في الويب (DuckDuckGo)، تلخيص مواضيع، وإجابات بمصادر فعلية",
        system_prompt=(
            "أنت وكيل \"البحث\" داخل نظام NSM (Neural Service Mesh).\n"
            "تخصصك: تجميع المعلومات حول موضوع معيّن، تلخيصها بشكل منظم.\n"
            "قبل كل رد، تحصل تلقائياً على نتائج بحث ويب حقيقية حديثة (ستظهر لك "
            "تحت عنوان 'نتائج بحث ويب ذات صلة') — استخدمها كمصدر أساسي لإجابتك، "
            "واذكر الروابط المهمة منها إن وُجدت.\n"
            "قواعد الإجابة:\n"
            "1. أجب بالعربية الفصحى، ونظّم الإجابة في نقاط عند تعدد الجوانب.\n"
            "2. إذا لم تُقدَّم لك نتائج بحث أو كانت غير كافية، قل ذلك بصراحة "
            "بدل اختلاق معلومة.\n"
            "3. اقترح زوايا بحث إضافية إذا كان الموضوع واسعاً.\n"
            "4. إذا سُئلت بجدية عن النموذج الأساسي، أجب بصدق ولا تنفِ ذلك."
        ),
        quick_prompts=[
            "ما آخر الأخبار عن هذا الموضوع؟",
            "ما الفرق بين هذين المفهومين؟",
            "ابحث لي عن أفضل الممارسات الحالية في هذا المجال",
        ],
        web_enabled=True,
    ),

    "maintenance": AgentCategory(
        key="maintenance",
        emoji="🛡️",
        title="الصيانة الذاتية",
        subtitle="يفحص ملفات المشروع، يكتشف الأخطاء المحتملة، ويقترح تصحيحات حقيقية",
        system_prompt=(
            "أنت وكيل \"الصيانة الذاتية\" داخل نظام NSM (Neural Service Mesh).\n"
            "تخصصك: مراقبة صحة مشروع NSM نفسه — اكتشاف الملفات الكبيرة، الدوال "
            "بدون معالجة أخطاء (try/except)، الملفات غير المستخدَمة، والوحدات "
            "المكررة، ثم اقتراح خطوات تصحيح واضحة وآمنة (لا تُطبَّق تلقائياً).\n"
            "قواعد الإجابة:\n"
            "1. أجب بالعربية الفصحى، ورتّب النتائج كنقاط أو جدول عند الحاجة.\n"
            "2. إذا سُئلت عن أمر تشخيص حقيقي (افحص/قائمة/اقترح/ملخص/صحح)، "
            "فهذه الأوامر تُنفَّذ فعلياً على ملفات المشروع — لا تخترع نتائج.\n"
            "3. لا تقترح حذف أو تعديل أي ملف مباشرة دون أن يطلب المستخدم ذلك صراحة "
            "عبر أمر 'عدل' بالصيغة الدقيقة.\n"
            "4. إذا سُئلت بجدية عن النموذج الأساسي، أجب بصدق ولا تنفِ ذلك."
        ),
        quick_prompts=[
            "اقترح أخطاء",
            "اقترح كبير",
            "اقترح مكررة",
            "قائمة",
        ],
    ),
}


CATEGORY_ORDER: List[str] = [
    "assistant", "automation", "analytics", "reasoning", "coding", "research",
    "maintenance",
]


# ══════════════════════════════════════════════════════════════════
# CategoryAgentChat — محادثة معزولة لكل فئة وكيل
# ══════════════════════════════════════════════════════════════════

class CategoryAgentChat:
    """
    غلاف خفيف حول LLMFallback: نفس مزوّد LLM المُكتشَف تلقائياً في المشروع
    (Anthropic/Cloudflare/Gemini/Groq/...)، لكن مع system_prompt مخصّص
    لكل فئة وذاكرة محادثة مستقلة بذاتها لكل وكيل.
    """

    def __init__(self, category_key: str):
        if category_key not in AGENT_CATEGORIES:
            raise ValueError(f"فئة غير معروفة: {category_key}")
        self.category: AgentCategory = AGENT_CATEGORIES[category_key]
        self.fallback = LLMFallback()
        self.history: List[Tuple[str, str]] = []
        self._last_provider = ""

    def chat(
        self,
        user_input: str,
        force_web: "bool | None" = None,
        source: str = SOURCE_HUB,
    ) -> str:
        """
        Parameters
        ----------
        source : str
            من أين استُدعي هذا الوكيل — للتدقيق فقط، لا يؤثر على السلوك.
            "hub" (افتراضي): من تبويب الوكيل المباشر داخل "🤖 وكلاء AI".
            "orchestrator": من تبويب "🤝 منسّق الوكلاء".
        """
        if not user_input.strip():
            return "الرجاء كتابة سؤالك."

        # ── وكيل الصيانة الذاتية فقط: أوامر تشخيص حقيقية على ملفات المشروع ──
        if (
            self.category.key == "maintenance"
            and _HAS_MAINTENANCE_COMMANDS
            and _maintenance_command is not None
        ):
            cmd_response = _maintenance_command(user_input.strip())
            if cmd_response is not None:
                self._last_provider = "🛠️ Code Agent"
                self.history.append((user_input, cmd_response))
                self._log_audit(user_input, cmd_response, source, "🛠️ Code Agent", False)
                return cmd_response

        # ── بحث ويب حقيقي (DuckDuckGo) قبل توليد الرد ──
        # يُفعَّل تلقائياً للفئات ذات web_enabled=True، أو يدوياً عبر force_web
        # من الواجهة (تفعيل/تعطيل لكل سؤال بغض النظر عن الفئة).
        use_web = getattr(self.category, "web_enabled", False) if force_web is None else force_web
        sp = self.category.system_prompt
        searched = False
        if use_web and _WEB_SEARCH_OK:
            try:
                web_results = _web_search(user_input.strip(), max_results=5)
            except Exception as _web_err:
                web_results = f"❌ تعذّر البحث: {_web_err}"
            if web_results and not web_results.startswith("❌"):
                sp = sp + "\n\nنتائج بحث ويب ذات صلة بسؤال المستخدم:\n" + web_results
                searched = True

        result = self.fallback.generate(
            query=user_input,
            history=self.history[-4:],
            system_prompt=sp,
        )
        provider_label = (
            result.model if result.provider in LIVE_LLM_PROVIDERS else "رسم معرفي"
        )
        self._last_provider = f"🌐 {provider_label}" if searched else provider_label
        self.history.append((user_input, result.text))
        self._log_audit(user_input, result.text, source, provider_label, searched)
        return result.text

    def _log_audit(
        self, question: str, response: str, source: str,
        provider: str, web_used: bool,
    ) -> None:
        """يسجّل التفاعل في سجل التدقيق (Observability) — لا يرفع أي
        استثناء أبداً؛ فشل التدقيق لا يجب أن يكسر المحادثة."""
        if not _AUDIT_OK or get_default_audit_log is None:
            return
        try:
            get_default_audit_log().log_event(
                category_key=self.category.key,
                category_title=self.category.title,
                source=source,
                question=question,
                response=response,
                provider=provider,
                web_used=web_used,
            )
        except Exception:
            pass

    def last_provider_badge(self) -> str:
        if not self._last_provider:
            return ""
        if self._last_provider.startswith("🛠️"):
            return self._last_provider
        return f"🤖 {self._last_provider}"

    def clear_history(self):
        self.history.clear()

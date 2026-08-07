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

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ai.llm_fallback import LLMFallback, LIVE_LLM_PROVIDERS

# توجيه تلقائي لكل رسالة (UnifiedAgentChat) — نفس منطق التوجيه المستخدَم
# أصلاً في تبويب "🤝 منسّق الوكلاء"، بدون أي استيراد دائري (godmode.py
# لا يستورد من هذا الملف إطلاقاً).
try:
    from ai.godmode import route_query_verbose
    _ROUTING_OK = True
except Exception:
    route_query_verbose = None
    _ROUTING_OK = False


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

# خط أنابيب "صناعة المحتوى" الحقيقي (ترند → مقال SEO → نشر/جدولة) —
# يُستخدَم حصراً في وكيل "صناعة المحتوى" أدناه، بدون أي تأثير على بقية الفئات.
try:
    from ai.content_agent import run_content_pipeline
    _CONTENT_OK = True
except Exception:
    run_content_pipeline = None
    _CONTENT_OK = False

# وكيل تدريب النماذج — أدوات حقيقية لإدارة دورة حياة التدريب (CKG / ArabicTransformer)
try:
    from ai.model_training_agent import handle_training_command
    _TRAINING_AGENT_OK = True
except Exception:
    handle_training_command = None
    _TRAINING_AGENT_OK = False

# 🆕 Chain-of-Thought + Few-shot Prompting (ai/chain_of_thought.py +
# ai/prompt_engine.py) — كانتا موجودتين بالمشروع منذ فترة وموثَّقتين
# كـ"أولوية #2" في تقرير تحليل سابق، لكن غير مربوطتين بأي مكان إطلاقاً
# (orphan modules). تُبنى الأمثلة المشابهة (few-shot) والمفاهيم المرتبطة
# من CKG وتُدمَج في نص السؤال قبل إرساله لـ LLMFallback؛ أي فشل هنا
# (قاعدة بيانات فارغة، CKG غير محمَّل...) يُبتلَع بأمان ويُستخدَم السؤال
# الخام كما كان يحدث سابقاً.
try:
    from ai.chain_of_thought import ChainOfThoughtBuilder
    from knowledge.cognitive_graph import get_ckg
    _COT_OK = True
except Exception:
    ChainOfThoughtBuilder = None
    get_ckg = None
    _COT_OK = False

# 🆕 تقييم جودة استكشافي (heuristic) لكل رد — نفس ai/response_quality.py
# المُستخدَم أصلاً في تبويب "🤖 وكلاء AI" لعرض شارة الجودة، لكنه هنا يُستخدَم
# فعلياً داخل حلقة المحادثة نفسها لإعادة توليد رد ضعيف الجودة تلقائياً
# (مرة واحدة فقط)، بدل الاكتفاء بعرض الجودة الضعيفة للمستخدم بدون فعل شيء
# حيالها. أي فشل في التقييم نفسه يُبتلَع بأمان ولا يمنع إرجاع الرد الأصلي.
try:
    from ai.response_quality import score_response as _score_quality
    _QUALITY_OK = True
except Exception:
    _score_quality = None
    _QUALITY_OK = False

# حد الجودة الذي تحته تُعتبر الإجابة "ضعيفة" وتستحق إعادة توليد واحدة —
# نفس عتبة تصنيف "ضعيف" في response_quality.py (overall < 0.40).
_LOW_QUALITY_THRESHOLD = 0.40


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

    "content": AgentCategory(
        key="content",
        emoji="📝",
        title="صناعة المحتوى",
        subtitle="يكتشف المواضيع الرائجة فعلياً، يكتب مقالات متوافقة مع SEO، ويجدول نشرها",
        system_prompt=(
            "أنت وكيل \"صناعة المحتوى\" داخل نظام NSM (Neural Service Mesh).\n"
            "تخصصك: اقتراح أفكار مقالات، شرح أساسيات SEO، ومساعدة المستخدم على "
            "صياغة عناوين ووصف تعريفي (meta description) جذابين.\n"
            "إذا طلب المستخدم بصيغة صريحة كتابة مقال أو البحث عن ترند "
            "('اكتب مقال عن...'، 'ابحث عن ترند')، فهذا الأمر يُنفَّذ فعلياً "
            "عبر خط أنابيب حقيقي (بحث ويب + توليد + جدولة نشر) — النتيجة "
            "التي تصلك هي مخرجات حقيقية وليست شيئاً عليك اختلاقه.\n"
            "قواعد الإجابة:\n"
            "1. أجب بالعربية الفصحى، وضّح نصائح SEO بأمثلة عملية قصيرة.\n"
            "2. لا تختلق إحصائيات بحث أو أرقام ترند — إن لم تُقدَّم لك بيانات "
            "حقيقية، قل ذلك صراحة واقترح استخدام أمر 'ابحث عن ترند'.\n"
            "3. ذكّر المستخدم أن النشر التلقائي يتطلب ذكر اسم منصة صراحة "
            "(مثل: تويتر، تيليجرام) وإلا يتوقف الخط عند كتابة المقال فقط.\n"
            "4. إذا سُئلت بجدية عن النموذج الأساسي، أجب بصدق ولا تنفِ ذلك."
        ),
        quick_prompts=[
            "ابحث عن ترند واكتب مقال",
            "اكتب مقال عن الذكاء الاصطناعي",
            "ما أساسيات كتابة عنوان متوافق مع SEO؟",
        ],
    ),

    "model_trainer": AgentCategory(
        key="model_trainer",
        emoji="🧬",
        title="تدريب النماذج",
        subtitle="وكيل عام: دورة حياة أي نموذج — بيانات، خوارزميات، تدريب، تقييم، وحفظ (NSM + sklearn + PyTorch)",
        system_prompt=(
            "أنت وكيل \"تدريب النماذج\" داخل نظام NSM (Neural Service Mesh).\n"
            "تخصصك: أتمتة دورة حياة تدريب نماذج الذكاء الاصطناعي بشكل عام — "
            "وليس مقصوراً على CKG فقط. تغطي:\n"
            "- نماذج NSM الداخلية (ArabicTransformer/CKG، NeuralCore، KnowledgeTrainer، سكربتات train_*.py)\n"
            "- نماذج scikit-learn (تصنيف/انحدار) عند توفر المكتبة\n"
            "- شبكات PyTorch البسيطة (MLP) عند توفر torch\n"
            "- تخطيط واختيار خوارزميات لأي مهمة يصفها المستخدم\n"
            "أوامر أدوات حقيقية (نفّذها المستخدم حرفياً تقريباً):\n"
            "- جرد البيئة / inventory\n"
            "- خطة دورة الحياة [وصف المهمة]\n"
            "- اقترح خوارزمية لـ …\n"
            "- درّب تصنيف تجريبي / درّب انحدار تجريبي\n"
            "- درّب شبكة torch [عدد الحقب]\n"
            "- حالة ckg / إعدادات ckg / شغّل تدريب ckg [تجريبي]\n"
            "- شغّل train_yemeni أو pilot أو batch\n"
            "- نماذج محفوظة\n"
            "- فحص ckg / تقرير ckg\n"
            "- تفعيل تعلم معزز / دورة RL\n"
            "- ضبط معلمات تدريب / سرب اجتماعي / سيادة النظام\n"
            "قواعد:\n"
            "1. أجب بالعربية الفصحى العملية.\n"
            "2. لا تختلق مقاييس تدريب — إن لم تُنفَّذ أداة، وجّه للأمر المناسب.\n"
            "3. لا تعد بإعادة تدريب CKG من الصفر أو رفع أوزان ضخمة إلى git.\n"
            "4. نبّه عند نقص الرام أو المكتبات.\n"
            "5. إذا سُئلت بجدية عن النموذج الأساسي، أجب بصدق ولا تنفِ ذلك."
        ),
        quick_prompts=[
            "فحص ckg",
            "حالة التعلم المعزز",
            "ضبط معلمات تدريب",
            "ماذا بعد",
            "حالة gpu",
            "سرب اجتماعي",
            "سيادة النظام",
            "جسر حساسات",
            "سجل النماذج",
        ],
        web_enabled=False,
    ),

}

CATEGORY_ORDER: List[str] = [
    "assistant", "automation", "analytics", "reasoning", "coding", "research",
    "maintenance", "content", "model_trainer",
]


# ══════════════════════════════════════════════════════════════════
# أوامر وكيل "صناعة المحتوى" الحقيقية (ترند → مقال SEO → نشر/جدولة)
# ══════════════════════════════════════════════════════════════════

_CONTENT_TOPIC_RE = re.compile(r"مقال(?:اً|ا)?\s*(?:عن|حول)\s*(.+)$")

_PLATFORM_KEYWORDS: Dict[str, List[str]] = {
    "telegram":  ["تيليجرام", "telegram"],
    "discord":   ["ديسكورد", "discord"],
    "instagram": ["انستقرام", "انستغرام", "instagram"],
    "facebook":  ["فيسبوك", "facebook"],
    "youtube":   ["يوتيوب", "youtube"],
    "tiktok":    ["تيك توك", "تيكتوك", "tiktok"],
    "reddit":    ["ريديت", "reddit"],
    "threads":   ["ثريدز", "threads"],
}


def _detect_platforms(text: str) -> List[str]:
    low = text.lower()
    return [pid for pid, kws in _PLATFORM_KEYWORDS.items() if any(kw in low for kw in kws)]


def _format_pipeline_result(result) -> str:
    art = result.article
    lines: List[str] = []
    lines.append(f"📝 **{art.title}**" if art else f"📝 موضوع: {result.topic}")
    if art:
        lines += [
            "",
            f"_{art.meta_description}_" if art.meta_description else "",
            "",
            f"الكلمات المفتاحية: {', '.join(art.keywords) if art.keywords else '—'}",
            f"عدد الكلمات: {art.word_count} | تقييم SEO: {art.seo_score}/100",
        ]
        if not art.structured:
            lines.append("⚠️ لم يلتزم المزوّد النشط بصيغة JSON — راجع المقال قبل أي نشر (جودة أقل موثوقية).")
        if art.seo_issues:
            lines.append("ملاحظات SEO: " + " | ".join(art.seo_issues))
        lines += ["", "--- نص المقال (Markdown) ---", art.to_markdown(), ""]

    lines += ["--- التشويقة المقترحة للنشر ---", result.teaser, ""]

    if result.platforms:
        if result.publish_mode == "scheduled":
            lines.append(
                f"📅 تمت جدولة النشر على: {', '.join(result.platforms)} "
                f"(معرّف الجدولة #{result.schedule_id})"
            )
        elif result.publish_mode == "published":
            lines.append("🚀 نتيجة النشر الفوري:")
            for pid, res in result.publish_result.items():
                lines.append(f"  - {pid}: {res}")
        else:
            lines.append("⏭️ لم يُنشر — راجع الأخطاء أدناه.")
    else:
        lines.append(
            "ℹ️ لم تُذكر منصة نشر صراحة، فتوقف الخط عند كتابة المقال فقط "
            "(وضع مراجعة). اذكر اسم منصة (مثل: تويتر) لجدولة النشر تلقائياً."
        )

    if result.errors:
        lines += ["", "❌ أخطاء: " + " | ".join(result.errors)]

    return "\n".join(l for l in lines if l is not None)


def _handle_content_command(user_input: str) -> Optional[str]:
    """يتعرّف على أوامر صناعة المحتوى الحقيقية (اكتب/ابحث عن ترند/انشر
    مقال) وينفّذها فعلياً عبر run_content_pipeline. يعيد None لو النص ليس
    أمر محتوى معروفاً، فتتابع المحادثة بمسارها العادي (LLM حر)."""
    if not _CONTENT_OK or run_content_pipeline is None:
        return None
    text = user_input.strip()
    if "مقال" not in text and "ترند" not in text and "رائج" not in text:
        return None

    m = _CONTENT_TOPIC_RE.search(text)
    topic = m.group(1).strip(" .؟!") if m else None
    platforms = _detect_platforms(text)

    try:
        result = run_content_pipeline(topic=topic or None, platforms=platforms)
    except Exception as e:
        return f"❌ تعذّر تنفيذ خط أنابيب صناعة المحتوى: {e}"
    return _format_pipeline_result(result)


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

        # 🆕 CKG (إن توفّر) يُمرَّر لكل من LLMFallback (يُحسّن مسار الرجوع
        # الاحتياطي ckg_synthesis عند غياب كل مزوّدي LLM) وChainOfThoughtBuilder
        # (مفاهيم مرتبطة ضمن التفكير التسلسلي). فشل تحميله لا يمنع عمل الوكيل.
        _ckg = None
        if _COT_OK:
            try:
                _ckg = get_ckg()
            except Exception:
                _ckg = None

        self.fallback = LLMFallback(ckg=_ckg)
        self.cot = None
        if _COT_OK:
            try:
                self.cot = ChainOfThoughtBuilder(ckg=_ckg)
            except Exception:
                self.cot = None

        self.history: List[Tuple[str, str]] = []
        self._last_provider = ""
        # 🆕 آخر تقييم جودة (نسبة% وتصنيف)، وهل أُعيد توليد الرد الأخير
        # بسبب جودة ضعيفة — تُقرَأ من الواجهة (Unified/Hub/Orchestrator)
        # لعرض شارة جودة موحّدة بدل إعادة حساب score_response في كل مكان.
        self._last_quality_percent: Optional[int] = None
        self._last_quality_label: str = ""
        self._last_regenerated: bool = False

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

        # ── جسر المشروع: أوامر الوكيل المدمجة (تدريب/CKG/سرب/RL/…) ──
        # يعمل لكل الفئات حتى لا تُفقد الأوامر عند التوجيه الخاطئ
        try:
            from ai.agent_project_bridge import dispatch_with_meta
            _br, _badge = dispatch_with_meta(user_input.strip())
            if _br is not None:
                self._last_provider = _badge or "🧬 Project Agent"
                self.history.append((user_input, _br))
                self._log_audit(user_input, _br, source, self._last_provider, True)
                return _br
        except Exception:
            pass

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

        # ── وكيل صناعة المحتوى فقط: خط أنابيب حقيقي (ترند → مقال SEO → نشر) ──
        if self.category.key == "content" and _CONTENT_OK:
            cmd_response = _handle_content_command(user_input.strip())
            if cmd_response is not None:
                self._last_provider = "📝 Content Pipeline"
                self.history.append((user_input, cmd_response))
                self._log_audit(user_input, cmd_response, source, "📝 Content Pipeline", True)
                return cmd_response

        # ── وكيل تدريب النماذج فقط: أوامر دورة حياة التدريب الحقيقية ──
        if (
            self.category.key == "model_trainer"
            and _TRAINING_AGENT_OK
            and handle_training_command is not None
        ):
            cmd_response = handle_training_command(user_input.strip())
            if cmd_response is not None:
                self._last_provider = "🧬 Model Training Agent"
                self.history.append((user_input, cmd_response))
                self._log_audit(user_input, cmd_response, source, "🧬 Model Training Agent", True)
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

        # 🆕 تعزيز السؤال بأمثلة مشابهة (few-shot) ومفاهيم مرتبطة من CKG عبر
        # ChainOfThoughtBuilder (خطوة "تفكير" شفافة قبل الإرسال للنموذج).
        # أي فشل (ذاكرة فارغة، CKG غير محمَّل...) يُبتلَع ويُستخدَم السؤال
        # الخام كما كان يحدث قبل هذا الربط.
        llm_query = user_input
        if self.cot is not None:
            try:
                llm_query = self.cot.build_llm_query(user_input, history=self.history[-4:])
            except Exception:
                llm_query = user_input

        result = self.fallback.generate(
            query=llm_query,
            history=self.history[-4:],
            system_prompt=sp,
        )

        # 🆕 إعادة توليد تلقائية مرة واحدة إذا كانت الإجابة ضعيفة الجودة
        # (قصيرة جداً/فارغة/رفض عام/مؤشر خطأ صريح...) وكانت من مزوّد LLM
        # حي فعلاً — إعادة توليد رد "رسم معرفي" (CKG synthesis) الاحتياطي
        # لا فائدة منها لأنه نفس المسار الحتمي في كل مرة. أي فشل في تقييم
        # الجودة نفسه أو في محاولة الإعادة يُبتلَع بأمان ويُستخدَم الرد
        # الأصلي كما كان يحدث قبل هذه الميزة.
        self._last_quality_percent = None
        self._last_quality_label = ""
        self._last_regenerated = False
        if _QUALITY_OK and result.provider in LIVE_LLM_PROVIDERS:
            try:
                _q = _score_quality(result.text, query=user_input)
                if _q.overall < _LOW_QUALITY_THRESHOLD:
                    retry_sp = (
                        sp + "\n\nملاحظة: إجابتك السابقة على سؤال مشابه كانت غير "
                        "كافية أو غير واضحة. قدّم إجابة أوضح وأكثر تفصيلاً والتزم "
                        "بقواعد الفئة أعلاه."
                    )
                    retry_result = self.fallback.generate(
                        query=llm_query, history=self.history[-4:], system_prompt=retry_sp,
                    )
                    _q_retry = _score_quality(retry_result.text, query=user_input)
                    if _q_retry.overall > _q.overall:
                        result = retry_result
                        _q = _q_retry
                        self._last_regenerated = True
                self._last_quality_percent = _q.as_percent()
                self._last_quality_label = _q.label
            except Exception:
                pass  # التقييم/الإعادة إضافيان وغير حرجَين — لا يُسقطان الرد الأصلي

        provider_label = (
            result.model if result.provider in LIVE_LLM_PROVIDERS else "رسم معرفي"
        )
        self._last_provider = f"🌐 {provider_label}" if searched else provider_label
        self.history.append((user_input, result.text))
        self._log_audit(user_input, result.text, source, provider_label, searched)
        return result.text

    def last_quality_badge(self) -> str:
        """🆕 شارة جودة جاهزة للعرض (نسبة% + تصنيف)، وإشارة إعادة توليد إن
        حدثت — لاستخدام موحّد عبر Unified Agent / Orchestrator بدل إعادة
        استيراد وحساب response_quality في كل واجهة على حدة. تعيد نصاً فارغاً
        إذا لم يُحسَب تقييم لأي سبب (وحدة غير متاحة، مزوّد CKG احتياطي...)."""
        if self._last_quality_percent is None:
            return ""
        badge = f"🔎 {self._last_quality_percent}٪ {self._last_quality_label}"
        if self._last_regenerated:
            badge += " · 🔁 أُعيد التوليد"
        return badge

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


class UnifiedAgentChat:
    """🎯 الوكيل الموحّد = مدير المشروع الشخصي (Master Orchestrator)

    الواجهة الواحدة التي تجمع كل شيء:
    - يفكر ويقرر متى يبحث في الويب، متى يفوّض لوكلاء متخصصين، متى يستخدم أدوات المشروع.
    - الوكلاء المتخصصون (AGENT_CATEGORIES) يعملون **تحته**؛ هو يعطيهم المهام ويجمع نتائجهم.
    - القرار والمسؤولية النهائية عن الجواب تبقى عنده دائماً.

    السلوك:
      1) أوامر المشروع التنفيذية → عبر agent_project_bridge / NSM Agent أولاً.
      2) المهام المعقّدة / متعددة الأبعاد → يختار 2–3 وكلاء، يشغّلهم، يولّف إجابة واحدة.
      3) المهام البسيطة → يوجّه لأنسب متخصص واحد مع الحفاظ على الذاكرة المشتركة.
    """

    # كلمات تشير إلى مهمة مركّبة تستحق تفويضاً متعدد الوكلاء
    _COMPLEX_HINTS = (
        "حلل", "حلّل", "راجع", "قارن", "خطة", "استراتيجية", "من جميع الجوانب",
        "شامل", "متكامل", "تقرير", "قيّم", "اقترح خطة", "صمّم", "ابنِ",
        "ابحث ثم", "حلل ثم", "من ناحية", "وزوايا", "متعدد", "وكلاء",
        "analyze", "compare", "plan", "strategy", "comprehensive", "report",
    )

    def __init__(self) -> None:
        self._bots: Dict[str, CategoryAgentChat] = {}
        self.shared_history: List[Tuple[str, str]] = []
        self.turns_meta: List[dict] = []

    def _get_bot(self, key: str) -> CategoryAgentChat:
        if key not in self._bots:
            self._bots[key] = CategoryAgentChat(key)
        return self._bots[key]

    def _is_complex(self, text: str) -> bool:
        """هل الطلب يبدو مركّباً بما يكفي لتفويض عدة وكلاء؟"""
        t = (text or "").strip()
        if len(t) < 40:
            return False
        low = t.lower()
        hits = sum(1 for h in self._COMPLEX_HINTS if h in low)
        # أسئلة طويلة أو تحتوي أكثر من مؤشر تعقيد
        return hits >= 1 or (len(t) > 120 and ("و" in t or "ثم" in t or "و" in low))

    def _synthesize(self, task: str, agent_replies: Dict[str, str]) -> str:
        """يولّف ردود الوكلاء الفرعية في إجابة واحدة نهائية تحت مسؤولية المدير."""
        if not agent_replies:
            return "لم أستطع جمع ردود من الوكلاء الفرعيين."
        if len(agent_replies) == 1:
            return next(iter(agent_replies.values()))

        parts = []
        for key, reply in agent_replies.items():
            cat = AGENT_CATEGORIES.get(key)
            title = cat.title if cat else key
            emoji = cat.emoji if cat else "•"
            parts.append(f"[{emoji} {title}]\n{reply.strip()}")

        combined = "\n\n───\n\n".join(parts)
        synth_prompt = (
            "أنت المدير الأعلى (Master Orchestrator) في نظام NSM.\n"
            "الوكلاء المتخصصون أرسلوا لك التقارير التالية حول مهمة المستخدم.\n"
            "مهمتك: وَلِّف إجابة واحدة نهائية واضحة بالعربية، احتفظ بالأفضل من كل تقرير، "
            "احذف التكرار، وكن أنت المسؤول النهائي عن الجواب (لا تذكر أسماء الوكلاء إلا إذا لزم).\n"
            "ابدأ مباشرة بالجواب المفيد، بدون مقدمات طويلة.\n\n"
            f"مهمة المستخدم:\n{task.strip()}\n\n"
            f"تقارير الوكلاء:\n{combined}"
        )
        try:
            _llm = LLMFallback()
            if getattr(_llm, "available", False):
                result = _llm.generate(
                    synth_prompt,
                    system_prompt=(
                        "أنت مدير مشروع شخصي ذكي. تجمع نتائج فريقك وتقدّم جواباً واحداً "
                        "واضحاً ومسؤولاً بالعربية. لا تختلق معلومات غير موجودة في التقارير."
                    ),
                )
                text = (result.text or "").strip()
                if text:
                    return text
        except Exception:
            pass

        # fallback بسيط إذا فشل التوليف عبر LLM
        return (
            "📋 **ملخص من فريق الوكلاء** (توليف تلقائي):\n\n"
            + "\n\n".join(
                f"**{AGENT_CATEGORIES[k].emoji} {AGENT_CATEGORIES[k].title}**\n{v}"
                for k, v in agent_replies.items()
            )
        )

    def chat(self, user_input: str, force_web: "bool | None" = None) -> "Tuple[str, dict]":
        """الواجهة الرئيسية: أقرر، أفوّض، أجمع، وأتحمل المسؤولية النهائية."""
        if not user_input.strip():
            return "الرجاء كتابة سؤالك أو هدفك.", {}

        # 1) أوامر المشروع التنفيذية الحقيقية لها أولوية مطلقة
        try:
            from ai.agent_project_bridge import dispatch_with_meta
            _br, _badge = dispatch_with_meta(user_input.strip())
            if _br is not None:
                meta = {
                    "category_key": "project_bridge",
                    "category_title": "وكيل المشروع التنفيذي",
                    "category_emoji": "🧬",
                    "route_method": "project_bridge",
                    "provider_badge": _badge or "🧬 Project Agent",
                    "quality_badge": "",
                    "delegated_agents": [],
                }
                self.shared_history.append((user_input, _br))
                self.turns_meta.append(meta)
                return _br, meta
        except Exception:
            pass

        # 2) تحديد الوكلاء المناسبين
        max_agents = 3 if self._is_complex(user_input) else 1
        if _ROUTING_OK and route_query_verbose is not None:
            selected, route_method, _scores = route_query_verbose(
                user_input, AGENT_CATEGORIES, max_agents=max_agents
            )
        else:
            selected, route_method = [], "default"

        if not selected:
            selected = ["assistant"] if "assistant" in AGENT_CATEGORIES else [next(iter(AGENT_CATEGORIES))]
            route_method = "default"

        # 3) حالة وكيل واحد فقط → نفس السلوك السابق (سريع ودقيق)
        if len(selected) == 1:
            key = selected[0]
            bot = self._get_bot(key)
            bot.history = list(self.shared_history[-4:])
            response = bot.chat(user_input, force_web=force_web, source="unified")
            cat = AGENT_CATEGORIES[key]
            meta = {
                "category_key": key,
                "category_title": cat.title,
                "category_emoji": cat.emoji,
                "route_method": route_method,
                "provider_badge": bot.last_provider_badge(),
                "quality_badge": bot.last_quality_badge(),
                "delegated_agents": [key],
            }
            self.shared_history.append((user_input, response))
            self.turns_meta.append(meta)
            return response, meta

        # 4) تفويض متعدد: أشغّل الوكلاء وأولّف النتيجة بنفسي
        agent_replies: Dict[str, str] = {}
        failed: List[str] = []
        badges: List[str] = []

        for key in selected:
            try:
                bot = self._get_bot(key)
                # سياق مشترك مختصر
                bot.history = list(self.shared_history[-3:])
                resp = bot.chat(user_input, force_web=force_web, source="unified_multi")
                agent_replies[key] = resp
                try:
                    badges.append(bot.last_provider_badge() or "")
                except Exception:
                    pass
            except Exception as e:
                failed.append(key)
                agent_replies[key] = f"⚠️ تعذّر الحصول على رد من هذا الوكيل: {e}"

        final = self._synthesize(user_input, {k: v for k, v in agent_replies.items() if k not in failed})

        # وصف الشفافية للمستخدم (بدون إضعاف دور المدير)
        team_label = " + ".join(
            f"{AGENT_CATEGORIES[k].emoji}{AGENT_CATEGORIES[k].title}"
            for k in selected if k in AGENT_CATEGORIES
        )
        meta = {
            "category_key": "master_orchestrator",
            "category_title": "المدير الموحّد",
            "category_emoji": "🎯",
            "route_method": f"multi:{route_method}",
            "provider_badge": f"🎯 مدير المشروع · فريق: {team_label}",
            "quality_badge": "",
            "delegated_agents": selected,
        }
        self.shared_history.append((user_input, final))
        self.turns_meta.append(meta)
        return final, meta

    def clear_history(self) -> None:
        self.shared_history.clear()
        self.turns_meta.clear()
        for bot in self._bots.values():
            bot.clear_history()


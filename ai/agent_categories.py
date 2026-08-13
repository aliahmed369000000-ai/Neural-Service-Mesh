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

# 🆕 ذاكرة دائمة/دلالية لكل وكيل — نفس محرك ConversationMemory المُستخدَم
# أصلاً في تبويب "المحادثة الذكية" (nsm_memory.py)، بنفس سلسلة التراجع:
# Qdrant Cloud (إن توفرت المفاتيح) ← TF-IDF محلي ← بحث كلمات مفتاحية.
# كل فئة وكيل تحصل على session_id ثابت خاص بها في نفس قاعدة SQLite
# المشتركة (memory/nsm_context.db)، فتصبح ذاكرتها دائمة عبر إعادة التشغيل
# وقابلة للاسترجاع الدلالي عبر جلسات سابقة، بدل الاعتماد فقط على قائمة
# self.history بالذاكرة المؤقتة التي كانت تُفقد عند انتهاء الجلسة.
try:
    from nsm_memory import ConversationMemory
    _AGENT_MEMORY_OK = True
except Exception:
    ConversationMemory = None
    _AGENT_MEMORY_OK = False


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
        title="مدير تدريب النماذج",
        subtitle="مدير متخصص يعمل تحت الوكيل الموحّد: لوحة تحكم، خطوة تالية ذكية، CKG/sklearn/torch، تدريب مستمر",
        system_prompt=(
            "أنت **مدير تدريب النماذج** داخل نظام NSM (Neural Service Mesh).\n"
            "تعمل تحت الوكيل الموحّد (Master Orchestrator): هو يفوّض لك مهام التدريب، "
            "وأنت تنفّذ بأدوات حقيقية وتُرجع تقريراً واضحاً يتحمل هو المسؤولية النهائية عنه.\n\n"
            "تخصصك — دورة حياة أي نموذج تقريباً:\n"
            "- نماذج NSM الداخلية (ArabicTransformer/CKG، NeuralCore، KnowledgeTrainer، train_*.py)\n"
            "- scikit-learn (تصنيف/انحدار) وPyTorch (MLP/CNN/نص) عند التوفر\n"
            "- لوحة تحكم موحّدة + اقتراح الخطوة التالية حسب الموارد والحالة\n"
            "- تدريب ذاتي مستمر عند ضعف جودة الإجابات\n\n"
            "أوامر أدوات حقيقية (فضّل تنفيذها على الشرح النظري):\n"
            "- لوحة التحكم / dashboard / نظرة عامة\n"
            "- ماذا بعد / الخطوة التالية / ابدأ تدريب ذكي\n"
            "- جرد البيئة · خطة دورة الحياة · اقترح خوارزمية\n"
            "- درّب تصنيف/انحدار تجريبي · درّب شبكة torch · درّب من csv\n"
            "- حالة ckg · خسارة · شغّل تدريب ckg [تجريبي]\n"
            "- نماذج محفوظة · قائمة csv · حالة gpu · تدريب مستمر\n"
            "- Hierarchical MoE: ملخص moe · قائمة خبراء · تقرير موازنة · أضف خبير … · ابنِ moe\n"
            "- فحص ckg · ضبط معلمات · سرب اجتماعي · سيادة النظام\n\n"
            "قواعد:\n"
            "1. أجب بالعربية الفصحى العملية والمختصرة عند الإمكان.\n"
            "2. لا تختلق مقاييس تدريب — إمّا أداة حقيقية أو توجيه صريح للأمر المناسب.\n"
            "3. لا تعد بإعادة تدريب CKG من الصفر أو رفع أوزان ضخمة إلى git.\n"
            "4. نبّه فوراً عند نقص الرام أو غياب المكتبات، واقترح dry-run أو Colab.\n"
            "5. عند سؤال عام مثل «ماذا أفعل الآن؟» ابدأ بلوحة التحكم أو الخطوة التالية.\n"
            "6. إذا سُئلت بجدية عن النموذج الأساسي، أجب بصدق ولا تنفِ ذلك."
        ),
        quick_prompts=[
            "لوحة التحكم",
            "ماذا بعد",
            "حالة ckg",
            "شغّل تدريب ckg تجريبي",
            "درّب شبكة torch",
            "درّب تصنيف تجريبي",
            "ملخص moe",
            "قائمة خبراء",
            "تدريب مستمر",
            "جرد",
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


# عبارات الاستعلام عن حالة مهمة صناعة محتوى شُغِّلت بالخلفية سابقاً.
_CONTENT_STATUS_KEYWORDS = ("جاهز", "حالة المهمة", "حالة المقال",
                            "انتهى المقال", "نتيجة المهمة", "نتيجة المقال")


def _format_job_status(text: str) -> str:
    """يبني رد حالة مهمة صناعة محتوى خلفية: قيد التشغيل / فشلت / جاهزة.
    لو ذُكر معرّف مهمة صراحة (#3) يُستخدَم، وإلا فأحدث مهمة."""
    from ai.content_job_manager import get_content_job_manager
    mgr = get_content_job_manager()

    id_match = re.search(r"#(\d+)", text)
    job = mgr.get(int(id_match.group(1))) if id_match else None
    if job is None and id_match is None:
        jobs = mgr.list_jobs()
        job = jobs[0] if jobs else None

    if job is None:
        return "ℹ️ لا توجد أي مهمة صناعة محتوى قيد التشغيل أو منتهية بعد. اطلب مقالاً جديداً أولاً."
    if job.status == "running":
        return f"⏳ المهمة #{job.job_id} لسه شغّالة بالخلفية — جرّب تسأل «جاهز؟» بعد شوي."
    if job.status == "failed":
        return f"❌ فشلت المهمة #{job.job_id}: {job.error}"
    return f"✅ المهمة #{job.job_id} خلصت.\n\n" + _format_pipeline_result(job.result)


def _handle_content_command(user_input: str) -> Optional[str]:
    """يتعرّف على أوامر صناعة المحتوى الحقيقية (اكتب/ابحث عن ترند/انشر
    مقال) وعلى استعلامات الحالة (جاهز؟)، وينفّذها فعلياً عبر
    run_content_pipeline — لكن في خيط خلفية (ai/content_job_manager.py)
    بدل تجميد واجهة Streamlit حتى انتهاء الخط (LLM + بحث ويب قد يأخذان
    عشرات الثواني). يعيد None لو النص ليس أمر محتوى معروفاً، فتتابع
    المحادثة بمسارها العادي (LLM حر)."""
    if not _CONTENT_OK or run_content_pipeline is None:
        return None
    text = user_input.strip()

    if any(k in text for k in _CONTENT_STATUS_KEYWORDS):
        return _format_job_status(text)

    if "مقال" not in text and "ترند" not in text and "رائج" not in text:
        return None

    m = _CONTENT_TOPIC_RE.search(text)
    topic = m.group(1).strip(" .؟!") if m else None
    platforms = _detect_platforms(text)

    try:
        from ai.content_job_manager import get_content_job_manager
        job_id = get_content_job_manager().start(topic=topic or None, platforms=platforms)
    except Exception as e:
        return f"❌ تعذّر بدء خط أنابيب صناعة المحتوى: {e}"

    plat_txt = f" ونشره/جدولته على {', '.join(platforms)}" if platforms else ""
    return (
        f"🚀 بدأ تنفيذ خط أنابيب صناعة المحتوى بالخلفية{plat_txt} "
        f"(معرّف المهمة #{job_id}) — تقدر تكمّل استخدام الواجهة عادي بدون انتظار.\n"
        f"اسأل «جاهز؟» لاحقاً لمتابعة النتيجة."
    )


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

        # 🆕 ذاكرة دائمة/دلالية خاصة بهذا الوكيل — session_id ثابت لكل فئة
        # (agent_<key>) يعزلها عن باقي الفئات وعن جلسة "المحادثة الذكية"
        # الرئيسية، مع مشاركة نفس قاعدة SQLite ونفس مرآة Qdrant الاختيارية.
        # فشل التهيئة (نادر) يُبتلَع بأمان — الوكيل يستمر بالعمل بذاكرة
        # الجلسة المؤقتة self.history فقط كما كان قبل هذه الميزة.
        self.memory: Optional["ConversationMemory"] = None
        if _AGENT_MEMORY_OK:
            try:
                self.memory = ConversationMemory(session_id=f"agent_{category_key}")
            except Exception:
                self.memory = None
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

        # 🆕 حقن سياق الذاكرة الدائمة/الدلالية لهذا الوكيل (حقائق محفوظة +
        # محادثات سابقة ذات صلة من جلسات قديمة، عبر Qdrant إن توفرت وإلا
        # TF-IDF محلي). أي فشل يُبتلَع بأمان ولا يمنع الرد الأصلي.
        if self.memory is not None:
            try:
                mem_ctx = self.memory.build_memory_context(user_input)
                if mem_ctx:
                    sp = sp + "\n\nسياق من ذاكرة هذا الوكيل:\n" + mem_ctx
            except Exception:
                pass

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
        # 🆕 حفظ الدور في الذاكرة الدائمة/الدلالية لهذا الوكيل (SQLite +
        # مرآة Qdrant اختيارية) — best-effort، لا يوقف المحادثة عند الفشل.
        if self.memory is not None:
            try:
                self.memory.add(user_input, result.text)
            except Exception:
                pass
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

    def _synthesize(self, task: str, agent_replies: Dict[str, str], adaptive_events=None, failure_warnings="") -> str:
        """يولّف ردود الوكلاء الفرعية في إجابة واحدة نهائية تحت مسؤولية المدير."""
        from ai.agent_event_bus import emit_event as _sy_emit, get_events as _sy_get_events
        _get_events_local = _sy_get_events
        if not agent_replies:
            return "لم أستطع جمع ردود من الوكلاء الفرعيين."
        if len(agent_replies) == 1:
            return next(iter(agent_replies.values()))

        # 🆕 السرب المتعلم: ترتيب التقارير في البرومبت حسب أداء كل وكيل التاريخي
        _ad_events = adaptive_events
        _ordered = list(agent_replies)
        if _ad_events is None:
            try:
                _ad_events = _get_events_local(250)
            except Exception:
                _ad_events = []
        if _ad_events:
            try:
                from ai.adaptive_swarm import rank_agents, weighted_synth_prompt
                _ad_titles = {
                    k: (AGENT_CATEGORIES[k].title if k in AGENT_CATEGORIES else k)
                    for k in agent_replies
                }
                _ord = rank_agents(list(agent_replies), _ad_events)
                if _ord != list(agent_replies):
                    _ordered = _ord
                    from ai.adaptive_swarm import announce_adaptive_reweight
                    _ad_profiles = None
                    try:
                        from ai.adaptive_swarm import agent_profiles
                        _ad_profiles = agent_profiles(_ad_events)
                    except Exception:
                        pass
                    announce_adaptive_reweight(_sy_emit, _ord, _ad_profiles, parent_task_id="synthesis")
                # weighted_synth_prompt يعيد نصًا كاملًا (ردود مرتبة بترجيح)
                combined = weighted_synth_prompt(task, agent_replies, _ad_events, _ad_titles)
            except Exception:
                combined = None
        else:
            combined = None
        if combined is None:
            parts = []
            for key, reply in agent_replies.items():
                cat = AGENT_CATEGORIES.get(key)
                title = cat.title if cat else key
                emoji = cat.emoji if cat else "•"
                parts.append(f"[{emoji} {title}]\n{reply.strip()}")
            combined = "\n\n───\n\n".join(parts)

        # ── الذاكرة الجماعية: دروس مستفادة من مهام سابقة ذات صلة ──────
        _cm_lessons = ""
        try:
            from ai.agent_event_bus import emit_event as _cm_emit
            from ai.collective_memory import get_collective_memory
            _recalled = get_collective_memory().recall(task, top_k=3)
            if _recalled:
                _cm_lessons = "\n\n".join(
                    f"• {lesson['lesson']}"
                    for lesson in _recalled if lesson.get("lesson")
                )
                try:
                    _cm_emit(
                        "lesson_recalled",
                        agent_id="master_orchestrator",
                        title="المدير الموحّد",
                        status="done",
                        detail=f"استرجعت {len(_recalled)} دروسًا جماعية",
                        metadata={"count": len(_recalled)},
                    )
                except Exception:
                    pass
        except Exception:
            pass

        synth_prompt = (
            "أنت المدير الأعلى (Master Orchestrator) في نظام NSM.\n"
            "الوكلاء المتخصصون أرسلوا لك التقارير التالية حول مهمة المستخدم.\n"
            "مهمتك: وَلِّف إجابة واحدة نهائية واضحة بالعربية، احتفظ بالأفضل من كل تقرير، "
            "احذف التكرار، وكن أنت المسؤول النهائي عن الجواب (لا تذكر أسماء الوكلاء إلا إذا لزم).\n"
            "ابدأ مباشرة بالجواب المفيد، بدون مقدمات طويلة.\n\n"
            f"مهمة المستخدم:\n{task.strip()}\n\n"
            f"تقارير الوكلاء:\n{combined}"
        )
        if _cm_lessons:
            synth_prompt += (
                "\n\n📚 دروس مستفادة من ذاكرة النظام الجماعية (استخدمها إن كانت ذات صلة، "
                "ولا تُبرزها إذا لم تكن مهمة):\n"
                + _cm_lessons
            )
        if failure_warnings:
            synth_prompt += (
                "\n\n⚠️ تحذيرات من أخطاء وكلاء سابقين في هذا المجال "
                "(انتبه لها أثناء التوليف):\n"
                + failure_warnings
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
        from ai.agent_event_bus import emit_event, get_events
        emit_event(
            "task_started",
            agent_id="master_orchestrator",
            title="المدير الموحّد",
            status="running",
            detail="استلام مهمة جديدة",
        )
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

        # 🆕 السرب المتعلم: ترتيب الوكلاء حسب أدائهم التاريخي عبر ناقل الأحداث
        _adaptive_events = get_events(250)
        try:
            from ai.adaptive_swarm import (
                adaptive_max_agents,
                announce_adaptive_exclusion,
                announce_adaptive_ranking,
                excluded_agents,
                rank_agents,
            )
            _adaptive_max = adaptive_max_agents(max_agents, _adaptive_events)
            selected = list(selected)[:_adaptive_max]
            _ordered = rank_agents(selected, _adaptive_events)
            if _ordered != list(selected):
                selected = _ordered
                _profiles = agent_profiles(_adaptive_events)
                announce_adaptive_ranking(emit_event, selected, _profiles, parent_task_id=f"multi:{route_method}")
            _excluded = excluded_agents(selected, _adaptive_events)
            if _excluded:
                from ai.adaptive_swarm import exclude_agents as _excl
                _kept = _excl(selected, _adaptive_events)
                if len(_kept) < len(selected):
                    announce_adaptive_exclusion(emit_event, _excluded, _kept, parent_task_id=f"multi:{route_method}")
                    selected = _kept
            max_agents = _adaptive_max
        except Exception as _ad_err:
            # الوضع المتعلم اختياري: فشل التحليل لا يعطّل اختيار الوكلاء
            import logging as _ad_log
            _ad_log.getLogger("adaptive_swarm").warning("adaptive_swarm: تعطّل التحليل — يعمل النظام كالعادة: %s", _ad_err)
        emit_event(
            "route_selected",
            agent_id="master_orchestrator",
            title="المدير الموحّد",
            status="running",
            detail=f"التوجيه: {route_method} · الوكلاء: {', '.join(selected)} · سقف تكيّفي: {max_agents}",
            metadata={"route_method": route_method, "selected": list(selected), "adaptive_max": max_agents},
        )

        # 3) حالة وكيل واحد فقط → نفس السلوك السابق (سريع ودقيق)
        if len(selected) == 1:
            key = selected[0]
            bot = self._get_bot(key)
            cat = AGENT_CATEGORIES[key]
            bot.history = list(self.shared_history[-4:])
            from ai.agent_reflection import ReflectionContext, reflecting_call
            _ref_ctx_single = ReflectionContext()
            try:
                def _single_call() -> str:
                    emit_event("agent_started", agent_id=key, title=cat.title, status="running", detail="تنفيذ المهمة")
                    return bot.chat(user_input, force_web=force_web, source="unified")
                def _single_retry(_att: int, info: dict) -> None:
                    _strategy_note = {
                        "retry_with_backoff": "إعادة المحاولة بعد انتظار قصير",
                        "switch_provider_hint": "محاولة عبر مسار مزوّد بديل",
                        "simplify_prompt": "تبسيط الطلب وإعادة المحاولة",
                    }.get(info.get("strategy", ""), "إعادة المحاولة")
                    emit_event("agent_started", agent_id=key, title=cat.title, status="running", detail=f"إعادة محاولة: {_strategy_note}")
                response = reflecting_call(key, cat.title, _single_call, _ref_ctx_single, on_retry=_single_retry)
                emit_event("agent_done", agent_id=key, title=cat.title, status="done", detail="اكتمل الرد")
            except Exception as exc:
                emit_event("agent_error", agent_id=key, title=cat.title, status="error", detail=str(exc)[:180])
                emit_event("task_error", agent_id="master_orchestrator", title="المدير الموحّد", status="error", detail="فشل الوكيل الوحيد")
                raise
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
            emit_event(
                "task_done",
                agent_id="master_orchestrator",
                title="المدير الموحّد",
                status="done",
                detail="اكتملت المهمة عبر وكيل متخصص واحد",
            )
            return response, meta

        # 4) تفويض متعدد: أشغّل الوكلاء وأولّف النتيجة بنفسي
        agent_replies: Dict[str, str] = {}
        failed: List[str] = []
        badges: List[str] = []

        for key in selected:
            cat = AGENT_CATEGORIES.get(key)
            emit_event(
                "agent_started",
                agent_id=key,
                title=cat.title if cat else key,
                status="running",
                detail="بدأ الوكيل المتخصص العمل",
            )
            from ai.agent_reflection import ReflectionContext, reflecting_call
            _ref_ctx_team = ReflectionContext()
            _agent_title = cat.title if cat else key
            try:
                bot = self._get_bot(key)
                # سياق مشترك مختصر
                bot.history = list(self.shared_history[-3:])
                def _team_call() -> str:
                    emit_event("agent_started", agent_id=key, title=_agent_title, status="running", detail="بدأ الوكيل المتخصص العمل")
                    return bot.chat(user_input, force_web=force_web, source="unified_multi")
                def _team_retry(_att: int, info: dict) -> None:
                    _strategy_note = {
                        "retry_with_backoff": "إعادة المحاولة بعد انتظار قصير",
                        "switch_provider_hint": "محاولة عبر مسار مزوّد بديل",
                        "simplify_prompt": "تبسيط الطلب وإعادة المحاولة",
                    }.get(info.get("strategy", ""), "إعادة المحاولة")
                    emit_event("agent_started", agent_id=key, title=_agent_title, status="running", detail=f"إعادة محاولة: {_strategy_note}")
                resp = reflecting_call(key, _agent_title, _team_call, _ref_ctx_team, on_retry=_team_retry)
                agent_replies[key] = resp
                emit_event(
                    "agent_done",
                    agent_id=key,
                    title=_agent_title,
                    status="done",
                    detail="اكتمل رد الوكيل",
                )
                try:
                    badges.append(bot.last_provider_badge() or "")
                except Exception:
                    pass
            except Exception as e:
                failed.append(key)
                agent_replies[key] = f"⚠️ تعذّر الحصول على رد من هذا الوكيل: {e}"
                emit_event(
                    "agent_error",
                    agent_id=key,
                    title=_agent_title,
                    status="error",
                    detail=str(e)[:180],
                )

        # 🆕 مرحلة التعاون المتقدمة: تنفيذ طلبات التفويض بين الوكلاء
        from ai.agent_delegation import (
            DelegationTracker,
            announce_delegation_rejected,
            announce_delegation_requested,
            announce_delegation_resolved,
            announce_delegation_started,
            parse_delegation_requests,
            strip_delegation_tags,
        )
        _delegation = DelegationTracker()
        _category_titles = {k: c.title for k, c in AGENT_CATEGORIES.items()}
        _running_delegates = {k: (AGENT_CATEGORIES[k].title if k in AGENT_CATEGORIES else k)
                              for k in selected}
        for key in selected:
            _raw = agent_replies.get(key, "") or ""
            _cat = AGENT_CATEGORIES.get(key)
            _reqs = parse_delegation_requests(_raw)
            for _dkey, _subtask in _reqs:
                _dcat = AGENT_CATEGORIES.get(_dkey)
                announce_delegation_requested(
                    delegator_key=key, delegator_title=(_cat.title if _cat else key),
                    delegate_key=_dkey, delegate_title=(_dcat.title if _dcat else _dkey),
                    subtask=_subtask, parent_task_id=f"multi:{route_method}",
                )
                _reject = _delegation.is_allowed(
                    key, _dkey, (_dcat.title if _dcat else _dkey),
                    _category_titles, _running_delegates,
                )
                if _reject:
                    announce_delegation_rejected(
                        key, (_cat.title if _cat else key), _dkey, _reject,
                    )
                    _delegation.register_request(
                        key, (_cat.title if _cat else key),
                        _dkey, (_dcat.title if _dcat else _dkey), _subtask,
                    )
                    _delegation.mark_result(_dkey, "rejected", _reject)
                    continue
                announce_delegation_started(
                    _dkey, (_dcat.title if _dcat else _dkey), key, _subtask,
                )
                _delegation.register_request(
                    key, (_cat.title if _cat else key),
                    _dkey, (_dcat.title if _dcat else _dkey), _subtask,
                )
                try:
                    from ai.agent_reflection import ReflectionContext, reflecting_call
                    _del_ctx = ReflectionContext()
                    _del_title = _dcat.title if _dcat else _dkey
                    def _del_call(_dsub: str = _subtask) -> str:
                        _del_bot = self._get_bot(_dkey)
                        _del_bot.history = []
                        return _del_bot.chat(
                            f"مهمة فرعية مفوّضة إليك من زميلك:\n{_dsub.strip()}",
                            force_web=force_web, source="delegation",
                        )
                    _del_resp = reflecting_call(_dkey, _del_title, _del_call, _del_ctx)
                    agent_replies[key] = (
                        strip_delegation_tags(_raw or "")
                        + (f"\n\n[نتيجة التفويض من {_del_title}]\n{_del_resp}" if _del_resp else "")
                    )
                    announce_delegation_resolved(
                        _dkey, _del_title, key, "done",
                        f"اكتملت المهمة الفرعية المفوّضة (≈{len(_del_resp or '')} حرف)",
                    )
                    _delegation.mark_result(_dkey, "resolved")
                except Exception as _del_err:
                    announce_delegation_resolved(
                        _dkey, (_dcat.title if _dcat else _dkey), key, "fail",
                        f"فشلت المهمة الفرعية المفوّضة: {_del_err}",
                    )
                    _delegation.mark_result(_dkey, "rejected", str(_del_err)[:150])

        # 🆕 تعلّم الأخطاء الجماعي: تسجيل دروس فشل هذه المهمة للذاكرة الجماعية
        _failure_warnings_text = ""
        try:
            from ai.failure_learning import sync_and_warn
            _failure_warnings_text = sync_and_warn(
                get_events(250), user_input, top_k=3)
        except Exception as _fl_err:
            import logging as _fl_log
            _fl_log.getLogger("failure_learning").warning(
                "failure_learning: تعطّل — يعمل النظام كالعادة: %s", _fl_err)

        emit_event("synthesis_started", agent_id="master_orchestrator", title="المدير الموحّد", status="running", detail="توليف ردود الفريق")
        final = self._synthesize(
            user_input,
            {k: strip_delegation_tags(v or "") for k, v in agent_replies.items() if k not in failed},
            adaptive_events=locals().get("_adaptive_events"),
            failure_warnings=_failure_warnings_text,
        )
        emit_event("synthesis_done", agent_id="master_orchestrator", title="المدير الموحّد", status="done", detail="اكتملت الإجابة الموحّدة")

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
        emit_event(
            "task_done",
            agent_id="master_orchestrator",
            title="المدير الموحّد",
            status="done",
            detail=f"اكتملت المهمة مع {len(selected) - len(failed)} وكيل فعال",
        )
        return final, meta

    def clear_history(self) -> None:
        self.shared_history.clear()
        self.turns_meta.clear()
        for bot in self._bots.values():
            bot.clear_history()


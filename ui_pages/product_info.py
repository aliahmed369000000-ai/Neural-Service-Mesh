"""
ui_pages/product_info.py
تم تفكيكه تلقائياً من streamlit_app.py الأصلي (تقسيم الكود لتحسين القابلية للصيانة والأداء).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — إعادة تصدير كل الاستيرادات والدوال المساعدة المشتركة



# ══════════════════════════════════════════════════════════════════════════
# تبويب ℹ️ عن NSM — معلومات المنتج
# ══════════════════════════════════════════════════════════════════════════
def render_product_info():
    st.markdown('<div class="section-header">ℹ️ عن Neural Service Mesh (NSM)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card" style="direction:rtl;line-height:2;font-size:1.02rem">
    <p style="margin:0"><strong>Neural Service Mesh (NSM)</strong> — النظام المعرفي العربي — هو منصة ذكاء اصطناعي
    عربية متخصصة تجمع بين محرك معرفي ذاتي التعلّم (Cognitive Knowledge Graph) ونماذج لغوية كبيرة،
    لتقديم تجربة بحث ومحادثة ومعرفة عربية أصيلة، مع تخصص خاص بالمعرفة الإسلامية والقرآن الكريم.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="font-size:1.05rem">🧭 ماذا يقدّم NSM؟</div>', unsafe_allow_html=True)
    features = [
        ("🌐", "بحث ويب حقيقي", "بحث فعلي في الإنترنت عبر DuckDuckGo بدون الحاجة لمفتاح API."),
        ("🖼️", "بحث عن الصور", "بحث عن صور حقيقية عبر Unsplash مع الوصف واسم المصوّر."),
        ("💬", "محادثة ذكية بذاكرة", "محادثة تتذكر السياق عبر الجلسات باستخدام ذاكرة SQLite طويلة الأمد."),
        ("📖", "معرفة قرآنية", "فهرسة وتحليل لغوي للقرآن الكريم — جذور، مفاهيم، علاقات دلالية."),
        ("🤖", "وكلاء AI", "وكلاء متخصصون لتحليل المشروع، البرمجة، والمهام المعرفية."),
        ("🧩", "واجهات تفاعلية", "إنشاء وعرض محتوى HTML/SVG تفاعلي واستدعاء أي API مباشرة."),
        ("🧠", "ذاكرة متقدمة", "ذاكرة دلالية (CKG) + ذاكرة حقائق + سجل محادثات قابل للاستعراض والبحث."),
        ("🖥️", "لوحة مطوّر", "تنفيذ أوامر Bash/Python محمي بمفتاح خاص بالمالك فقط."),
        ("☁️", "AIaaS — تدريب كخدمة", "مستأجرون معزولون، خطط وحصص استخدام، وفوترة تقديرية لتدريب نماذج مخصّصة."),
        ("🎓", "تدريب نماذج حقيقي", "محرك تدريب Transformer عربي بأكثر من طريقة توكنة (BPE، WordPiece، SentencePiece وغيرها) مع دعم GPU ولوحة متابعة حيّة."),
        ("🐝", "مصنع ومسرح الوكلاء", "توليد وكلاء متخصصين وتنسيق سرب منهم (SwarmCoordinator) لتنفيذ مهام معقّدة بالتوازي."),
        ("🔌", "سيرفر MCP", "أدوات NSM (سؤال الوكيل، بحث CKG، تقرير جاهزية المشروع) متاحة لأي عميل MCP خارجي مثل Claude Desktop."),
    ]
    _pi_cards_html = "".join(f"""
            <div class="feature-card" style="cursor:default;">
                <div class="feature-icon">{_icon}</div>
                <div class="feature-title">{_title}</div>
                <div class="feature-desc">{_desc}</div>
            </div>""" for _icon, _title, _desc in features)
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));'
        f'gap:1rem;direction:rtl;">{_pi_cards_html}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown('<div class="section-header" style="font-size:1.05rem">🔗 روابط</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card" style="direction:rtl">
        <p style="margin:0 0 0.4rem 0">📦 المستودع:
        <a href="https://github.com/aliahmed369000000-ai/Neural-Service-Mesh" target="_blank">
        Neural-Service-Mesh على GitHub</a></p>
        <p style="margin:0;color:var(--text-muted)">🛠️ بُني بـ Python · Streamlit · SQLite ·
        نماذج لغوية عبر OpenRouter/Anthropic</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

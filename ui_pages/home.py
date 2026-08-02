"""
ui_pages/home.py
تم تفكيكه تلقائياً من streamlit_app.py الأصلي (تقسيم الكود لتحسين القابلية للصيانة والأداء).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — إعادة تصدير كل الاستيرادات والدوال المساعدة المشتركة



def render_home():
    """الصفحة الرئيسية — نظرة سريعة واستكشاف أقسام NSM."""

    # فهرس التبويبات الرئيسية بترتيبها الفعلي في main() (streamlit_app.py):
    # 0=الرئيسية 1=المعرفة 2=المحادثة 3=الوكلاء 4=إبداع. مُعرَّف هنا أول
    # الدالة لأنه يُستخدم في أكثر من موضع أدناه (بطاقات الاستكشاف + التنقّل
    # الآلي بعد البحث السريع/متابعة المحادثة) — نسخة واحدة بدل تكرارها.
    _tab_index_map = {"📚 المعرفة": 1, "💬 المحادثة": 2, "🤖 الوكلاء": 3, "🎭 إبداع": 4}

    # ── 🧭 تنقّل آلي معلَّق: بعد أي زر Python حقيقي (بحث سريع / متابعة
    # محادثة أدناه) يطلب الانتقال لتبويب معيّن، نخزّن الهدف بـsession_state
    # ثم st.rerun()، وهنا — أول ما تُعاد الصفحة — نحقن سكربت مرة واحدة
    # يحاكي نفس نقرة التبويب المستخدمة أصلاً ببطاقات الاستكشاف (أسفل)،
    # بدل تكرار منطق البحث عن عناصر التبويب من جديد في كل مكان.
    _jump_target = st.session_state.pop("_nsm_home_jump_target", None)
    if _jump_target:
        st.components.v1.html("""
        <script>
        (function() {
            const doc = window.parent.document;
            const label = """ + json.dumps(_jump_target) + """;
            function findTabElements() {
                const strategies = [
                    '.stTabs [role="tablist"] [role="tab"]',
                    '[role="tablist"] [role="tab"]',
                    '.stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]',
                    '[data-baseweb="tab-list"] [data-baseweb="tab"]',
                    '[data-testid="stTab"]'
                ];
                for (const sel of strategies) {
                    const found = doc.querySelectorAll(sel);
                    if (found && found.length) return Array.from(found);
                }
                return [];
            }
            function fireFullClick(el) {
                const opts = { bubbles: true, cancelable: true, view: doc.defaultView || window };
                try { el.dispatchEvent(new PointerEvent('pointerdown', opts)); } catch (e) {}
                el.dispatchEvent(new MouseEvent('mousedown', opts));
                el.dispatchEvent(new MouseEvent('mouseup', opts));
                el.dispatchEvent(new MouseEvent('click', opts));
                el.click();
            }
            function attempt(tries) {
                const tabs = findTabElements();
                const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
                let target = tabs.find(t => norm(t.textContent) === norm(label));
                if (!target) target = tabs.find(t => norm(t.textContent).includes(norm(label)));
                if (target) { fireFullClick(target); return; }
                if (tries > 0) setTimeout(function() { attempt(tries - 1); }, 150);
            }
            attempt(12);
        })();
        </script>
        """, height=0)

    # ── ↩️ أكمل من حيث توقفت — يظهر فقط لمن لديه محادثة سابقة بهذه
    # الجلسة (nsm_messages تُعبَّأ من ui_pages/chat.py)؛ لا شيء يظهر لزائر
    # جديد بلا سجل، تفادياً لعنصر واجهة فارغ أو مضلِّل ──
    _prev_msgs = st.session_state.get("nsm_messages") or []
    _last_user_msg = next((m[1] for m in reversed(_prev_msgs) if m and m[0] == "user"), "")
    if _last_user_msg:
        _preview = _last_user_msg.strip().replace("\n", " ")
        if len(_preview) > 110:
            _preview = _preview[:110].rstrip() + "…"
        st.markdown('<div class="section-header">↩️ أكمل من حيث توقفت</div>', unsafe_allow_html=True)
        _cont_cols = st.columns([5, 1])
        with _cont_cols[0]:
            st.markdown(f"""
            <div class="concept-card" style="margin-bottom:0;">
                <div style="color:var(--text-muted);font-size:0.82rem;margin-bottom:0.3rem;">
                    آخر سؤال في المحادثة ({len(_prev_msgs)} رسالة)
                </div>
                <div style="font-size:0.98rem;">💬 {_preview}</div>
            </div>
            """, unsafe_allow_html=True)
        with _cont_cols[1]:
            if st.button("متابعة ←", key="home_continue_chat", use_container_width=True):
                st.session_state["_nsm_home_jump_target"] = "💬 المحادثة"
                st.rerun()
        st.markdown("")

    # ══════════════════════════════════════════════════════════════════
    # 📊 الشبكة المعرفية بالأرقام — أعداد حقيقية من CKG المحمَّل فعلياً
    # (لا بيانات وهمية). تُخفى الفقرة كاملة تلقائياً إن تعذّر تحميل
    # الشبكة (مثلاً أول تشغيل قبل أي تدريب) بدل إظهار أصفار مضلِّلة.
    # ══════════════════════════════════════════════════════════════════
    _ckg_stats = load_ckg_stats()
    if _ckg_stats:
        st.markdown('<div class="section-header">📊 الشبكة المعرفية بالأرقام</div>',
                    unsafe_allow_html=True)
        _stat_defs = [
            ("concepts",  "مفهوم معرفي مترابط",      True),
            ("relations", "علاقة مستنتجة بين المفاهيم", False),
            ("roots",     "جذر عربي مكتشَف",           False),
            ("clusters",  "عنقود موضوعي",              False),
        ]
        _bento_html = "".join(
            f'''<div class="metric-card{' bento-featured' if _featured else ''}">
                <div class="metric-value{'' if _featured else ' metric-value--wrap'}"
                     data-count-target="{_ckg_stats.get(_key, 0)}">0</div>
                <div class="metric-label">{_label}</div>
            </div>'''
            for _key, _label, _featured in _stat_defs
        )
        st.markdown(f'<div class="bento-grid">{_bento_html}</div>', unsafe_allow_html=True)
        st.markdown("")

    # ── 🎬 كيف يعمل NSM؟ — دليل تفاعلي بخط أنابيب متحرك يشرح رحلة السؤال ──
    st.markdown('<div class="section-header">🎬 كيف يعمل NSM؟ <span class="live-dot"></span></div>',
                unsafe_allow_html=True)

    _pipeline_steps = [
        ("📝", "إدخال عربي", "تكتب سؤالك", "تكتب سؤالك أو مفهومك بالعربية الفصحى — بدون أي قوالب أو صياغة خاصة، تماماً كما تتحدث."),
        ("🌱", "تحليل الجذر", "استخراج الجذر اللغوي", "يحلّل النظام الجذر الثلاثي/الرباعي والبنية الصرفية للكلمة من قاعدة تضم آلاف الجذور العربية المكتشفة."),
        ("🕸️", "ربط CKG", "شبكة المفاهيم المعرفية", "يربط المفهوم بشبكة المعرفة الحية (CKG) — آلاف المفاهيم وعشرات آلاف العلاقات المستنتجة بينها."),
        ("📖", "مطابقة قرآنية", "بحث آية بآية", "يبحث آلياً عن الآيات القرآنية ذات الصلة الدلالية بالمفهوم، مربوطة بنفس شبكة الجذور والمعاني."),
        ("💬", "رد ذكي", "إجابة مدعومة بالسياق", "يولّد رداً نهائياً مدعوماً بالمصادر والسياق المستخرج من كل الخطوات السابقة، بالعربية الفصحى."),
    ]

    _nodes_html = "".join(
        f'''<div class="pipeline-node{' active' if i == 0 else ''}" data-step="{i}"
                data-title="{title}" data-text="{text}" data-icon="{icon}">
            <div class="pipeline-node-icon">{icon}</div>
            <div class="pipeline-node-title">{label}</div>
        </div>'''
        for i, (icon, label, title, text) in enumerate(_pipeline_steps)
    )
    _dots_html = "".join(
        f'<span class="{"active" if i == 0 else ""}" data-dot="{i}"></span>'
        for i in range(len(_pipeline_steps))
    )
    _icon0, _label0, _title0, _text0 = _pipeline_steps[0]

    st.markdown(f"""
    <div class="nsm-pipeline-wrap">
        <div class="nsm-pipeline" id="nsm-pipeline" tabindex="0"
             role="group" aria-label="خطوات عمل NSM — استخدم الأسهم للتنقّل">
            <div class="nsm-pipeline-track"><div class="nsm-pipeline-track-fill" id="nsm-pipeline-fill"></div></div>
            {_nodes_html}
        </div>
        <div class="pipeline-detail">
            <div class="pipeline-detail-inner" id="nsm-pipeline-detail">
                <div class="pipeline-detail-title"><span id="nsm-pd-icon">{_icon0}</span>
                    <span id="nsm-pd-title">{_title0}</span>
                    <span class="pipeline-step-counter" id="nsm-pd-counter">1 / {len(_pipeline_steps)}</span>
                </div>
                <div class="pipeline-detail-text" id="nsm-pd-text">{_text0}</div>
            </div>
        </div>
        <div class="pipeline-progress-hint">{_dots_html}</div>
    </div>
    """, unsafe_allow_html=True)

    st.components.v1.html("""
    <script>
    (function() {
        const doc = window.parent.document;
        const pipeline = doc.getElementById('nsm-pipeline');
        if (!pipeline || pipeline.dataset.nsmBound) return;
        pipeline.dataset.nsmBound = "1";

        const nodes = Array.from(pipeline.querySelectorAll('.pipeline-node'));
        const fill  = doc.getElementById('nsm-pipeline-fill');
        const dInner= doc.getElementById('nsm-pipeline-detail');
        const dIcon = doc.getElementById('nsm-pd-icon');
        const dTitle= doc.getElementById('nsm-pd-title');
        const dText = doc.getElementById('nsm-pd-text');
        const dCounter = doc.getElementById('nsm-pd-counter');
        const dots  = Array.from(doc.querySelectorAll('.pipeline-progress-hint span'));
        const total = nodes.length;
        let current = 0;
        let timer = null;
        let paused = false;

        function setActive(idx, fromClick) {
            current = ((idx % total) + total) % total;
            nodes.forEach((n, i) => n.classList.toggle('active', i === current));
            dots.forEach((d, i) => d.classList.toggle('active', i === current));
            if (fill) fill.style.width = (current / (total - 1) * 100) + '%';
            if (dInner) {
                dInner.style.opacity = '0';
                setTimeout(function() {
                    const n = nodes[current];
                    dIcon.textContent    = n.getAttribute('data-icon');
                    dTitle.textContent   = n.getAttribute('data-title');
                    dText.textContent    = n.getAttribute('data-text');
                    if (dCounter) dCounter.textContent = (current + 1) + ' / ' + total;
                    dInner.style.opacity = '1';
                }, 180);
            }
            if (fromClick) restart();
        }

        function tick() { if (!paused) setActive(current + 1, false); }
        function restart() {
            if (timer) clearInterval(timer);
            timer = setInterval(tick, 3400);
        }

        nodes.forEach((n, i) => {
            n.addEventListener('click', function() { setActive(i, true); });
        });

        // إيقاف مؤقت أثناء التحويم/اللمس حتى لا يفوّت القارئ الوصف
        pipeline.addEventListener('mouseenter', function() { paused = true; });
        pipeline.addEventListener('mouseleave', function() { paused = false; });
        pipeline.addEventListener('touchstart', function() { paused = true; }, { passive: true });

        // تنقّل بالأسهم (يمين/يسار) عند تركيز العنصر — يدعم اتجاه RTL
        pipeline.addEventListener('keydown', function(e) {
            if (e.key === 'ArrowLeft') { setActive(current + 1, true); e.preventDefault(); }
            else if (e.key === 'ArrowRight') { setActive(current - 1, true); e.preventDefault(); }
        });

        setActive(0, false);
        restart();
    })();
    </script>
    """, height=0)

    st.markdown("---")

    # ── 🔎 جرّب البحث الآن — بحث معرفي فوري من الصفحة الرئيسية، بلا حاجة
    # للانتقال لتبويب البحث أولاً. يستخدم نفس محرك search_knowledge
    # المستخدم فعلياً بتبويب "🔍 البحث المعرفي" (ui_pages/search.py) —
    # لا منطق بحث موازٍ أو مكرر، فقط عرض مصغَّر للنتيجة هنا ──
    st.markdown('<div class="section-header">🔎 جرّب البحث الآن</div>', unsafe_allow_html=True)
    with st.form("home_quick_search_form", clear_on_submit=False):
        _qs_cols = st.columns([5, 1])
        with _qs_cols[0]:
            _qs_query = st.text_input(
                "", placeholder="اكتب مفهوماً... مثل: الصبر، الجاذبية، التوبة، العلم",
                key="home_quick_search_input", label_visibility="collapsed",
            )
        with _qs_cols[1]:
            _qs_submitted = st.form_submit_button("🔎 ابحث", use_container_width=True)

    if _qs_submitted and _qs_query.strip():
        with st.spinner("🔍 جارٍ البحث..."):
            _qs_result = search_knowledge(_qs_query.strip())
        st.session_state["_nsm_home_last_quick_search"] = (_qs_query.strip(), _qs_result)

    _qs_cached = st.session_state.get("_nsm_home_last_quick_search")
    if _qs_cached:
        _qs_q, _qs_result = _qs_cached
        if not _qs_result["found"]:
            st.warning(f"لم يُعثر على معلومات كافية عن «{_qs_q}» حتى الآن. يتعلم النظام بشكل مستمر!")
        else:
            _related = []
            if _qs_result["ckg_related"]:
                _related = _qs_result["ckg_related"]
            elif _qs_result["root_matches"]:
                _related = [m[0] for m in _qs_result["root_matches"] if m[0] != _qs_q]
            _tags_html = "".join(f'<span class="related-tag">{c}</span>' for c in _related[:3])

            _ayah_html = ""
            if _qs_result["quran_matches"]:
                _a = _qs_result["quran_matches"][0]
                _ayah_html = f"""<div class="quran-verse" style="margin-top:0.6rem;">
                    {_a.get('text', '')}
                    <div class="verse-ref">سورة {_a.get('surah', '')}، الآية {_a.get('ayah', '')}</div>
                </div>"""

            st.markdown(f"""
            <div class="concept-card">
                <div class="concept-name">💡 {_qs_result['query']}</div>
                <div style="color:var(--text-muted);font-size:0.85rem;margin-top:0.2rem;">
                    درجة الثقة: {_qs_result['confidence']:.0%}
                </div>
                {f'<div style="margin-top:0.5rem;">{_tags_html}</div>' if _tags_html else ''}
                {_ayah_html}
            </div>
            """, unsafe_allow_html=True)

            if st.button("📖 عرض النتيجة الكاملة في تبويب البحث", key="home_quick_search_open_full"):
                st.session_state["search_query"] = _qs_q
                st.session_state["_nsm_home_jump_target"] = "📚 المعرفة"
                st.toast("تم فتح البحث الكامل", icon="🔍")
                st.rerun()

    st.markdown("---")
    st.markdown('<div class="section-header">🚀 استكشف NSM</div>', unsafe_allow_html=True)

    _features = [
        ("🔍", "البحث المعرفي", "ابحث عن أي مفهوم (الصبر، الجاذبية، الرحمة، العدل...) وشاهد الآيات المرتبطة والجذور والعلاقات المعرفية.", "📚 المعرفة"),
        ("💬", "محادثة ذكية", "تحدّث مع النظام بالعربية الفصحى، مدعوماً بشبكة المفاهيم المعرفية.", "💬 المحادثة"),
        ("📖", "القرآن الكريم", "بحث آية بآية، مرتبط تلقائياً بشبكة المفاهيم والجذور العربية.", "📚 المعرفة"),
        ("🤖", "الوكلاء الأذكياء", "وكلاء مستقلون للتنفيذ والتنسيق ضمن سرب ذكي متكامل.", "🤖 الوكلاء"),
        ("🎭", "المحتوى الإبداعي", "توليد نصوص ومحتوى إبداعي عربي بأسلوب متعدد الأنماط.", "🎭 إبداع"),
    ]
    _cards_html = "".join(f"""
            <div class="feature-card" data-tab-target="{_target_tab}" tabindex="0" role="button">
                <div class="feature-icon">{_icon}</div>
                <div class="feature-title">{_title}</div>
                <div class="feature-desc">{_desc}</div>
                <div class="feature-nav-hint">← انتقل إلى هذا القسم</div>
            </div>""" for _icon, _title, _desc, _target_tab in _features)
    st.markdown(f'<div class="feature-scroll">{_cards_html}</div>', unsafe_allow_html=True)

    # ── سكربت: عدّادات متحركة للمقاييس + نقر بطاقات الاستكشاف للتنقّل ──
    # تنبيه: كان هذا مُحقناً سابقاً عبر st.markdown، وهو أسلوب لا يُنفَّذ
    # فيه <script> أبداً (عنصر <script> المُدرَج عبر innerHTML لا يعمل،
    # سلوك موثّق بالمتصفحات وليس مجرد "أحياناً" — لهذا كان النقر على
    # البطاقات بلا أي أثر). الحل المضمون: st.components.v1.html الذي
    # يُنشئ iframe حقيقياً يُنفَّذ فيه JS، ومنه نصل للصفحة الأم عبر
    # window.parent.document (نفس الحل المطبَّق أعلاه لتلوين التبويبات).
    #
    # ملاحظة إضافية: فهرس كل تبويب هدف بترتيب قائمة _tab_defs الفعلية
    # (بدالة main، أسفل الملف) — يُستخدم كخط دفاع ثانٍ بعد المطابقة
    # النصية، لأن بنية DOM الداخلية لِـ st.tabs قد تختلف بين إصدارات
    # Streamlit (data-baseweb مقابل role="tab" ...إلخ)، فالاعتماد على
    # نص + فهرس معاً أكثر مقاومة لتغيّر الإصدار من نص فقط.
    # الترتيب الحالي: 0=الرئيسية 1=المعرفة 2=المحادثة 3=الوكلاء 4=إبداع
    # (_tab_index_map مُعرَّف أول الدالة الآن — يُستخدم هنا وبقسم البحث
    # السريع/متابعة المحادثة أعلاه بلا تكرار)
    # أداء: هذا السكربت (iframe جديد + مسح DOM) كان يُعاد حقنه بالكامل من
    # جديد عند *كل* rerun لتبويب الرئيسية — أي عند كل تفاعل بأي مكان
    # بالتطبيق، لأن render_home() يُنفَّذ دوماً بغض النظر عن التبويب
    # النشط فعلياً (st.tabs تُخفي المحتوى بصرياً فقط ولا توقف تنفيذه).
    # هذا آمن للتخطي بعد أول حقن لأن الـMutationObserver بداخله يبقى حياً
    # ويعيد ربط العدّادات وبطاقات التنقل ذاتياً عند أي إعادة رسم لاحقة
    # لها (كما يوثّق تعليق الكود بالأسفل)، تماماً كما هو مُثبَت ومُطبَّق
    # مسبقاً على سكربت تلوين التبويبات (أعلاه في هذا الملف).
    if not st.session_state.get("_nsm_home_cards_js_injected"):
        st.session_state["_nsm_home_cards_js_injected"] = True
        st.components.v1.html("""
    <script>
    (function() {
        const doc = window.parent.document;
        const TAB_INDEX = """ + str(_tab_index_map).replace("'", '"') + """;

        function findTabElements() {
            // عدّة استراتيجيات بترتيب الأولوية — أول واحدة تُعيد نتائج تُستخدم
            const strategies = [
                '.stTabs [role="tablist"] [role="tab"]',
                '[role="tablist"] [role="tab"]',
                '.stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]',
                '[data-baseweb="tab-list"] [data-baseweb="tab"]',
                '[data-testid="stTab"]'
            ];
            for (const sel of strategies) {
                const found = doc.querySelectorAll(sel);
                if (found && found.length) return Array.from(found);
            }
            return [];
        }

        function fireFullClick(el) {
            const opts = { bubbles: true, cancelable: true, view: doc.defaultView || window };
            try { el.dispatchEvent(new PointerEvent('pointerdown', opts)); } catch (e) {}
            el.dispatchEvent(new MouseEvent('mousedown', opts));
            el.dispatchEvent(new MouseEvent('mouseup', opts));
            el.dispatchEvent(new MouseEvent('click', opts));
            el.click();
        }

        function goToTab(label) {
            const tabs = findTabElements();
            if (!tabs.length) return false;
            const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
            // 1) مطابقة نصية دقيقة
            let target = tabs.find(t => norm(t.textContent) === norm(label));
            // 2) مطابقة نصية جزئية (احتياط لو أضيف نص إضافي مخفي بالعنصر)
            if (!target) target = tabs.find(t => norm(t.textContent).includes(norm(label)));
            // 3) مطابقة بالفهرس الرقمي كخط دفاع أخير
            if (!target && TAB_INDEX.hasOwnProperty(label) && tabs[TAB_INDEX[label]]) {
                target = tabs[TAB_INDEX[label]];
            }
            if (!target) return false;
            fireFullClick(target);
            return true;
        }

        function bindAll() {
            // 1) عدّاد متحرك من 0 حتى القيمة الفعلية لكل بطاقة مقياس
            const counters = doc.querySelectorAll('.metric-value[data-count-target]');
            counters.forEach(function(el) {
                if (el.dataset.nsmAnimated) return;
                el.dataset.nsmAnimated = "1";
                const target = parseInt(el.getAttribute('data-count-target'), 10) || 0;
                const duration = 900;
                const start = performance.now();
                function tick(now) {
                    const p = Math.min(1, (now - start) / duration);
                    const eased = 1 - Math.pow(1 - p, 3);
                    el.textContent = Math.round(eased * target).toLocaleString('en-US');
                    if (p < 1) requestAnimationFrame(tick);
                    else el.textContent = target.toLocaleString('en-US');
                }
                requestAnimationFrame(tick);
            });

            // 2) نقر بطاقة الاستكشاف ← تفعيل تبويب Streamlit المطابق بالاسم
            const cards = doc.querySelectorAll('.feature-card[data-tab-target]');
            cards.forEach(function(card) {
                if (card.dataset.nsmBound) return;
                card.dataset.nsmBound = "1";
                card.addEventListener('click', function() {
                    const label = card.getAttribute('data-tab-target');
                    if (!goToTab(label)) {
                        // إعادة محاولة واحدة بعد لحظة قصيرة احتياطاً لتأخر رسم التبويبات
                        setTimeout(function() { goToTab(label); }, 200);
                    }
                });
                // إتاحة: تفعيل بالضغط على Enter/مسافة أيضاً (tabindex="0" role="button")
                card.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); card.click(); }
                });
            });
        }

        bindAll();
        // البطاقات تُعاد رسمتها بكل rerun من Streamlit، فنراقب DOM
        // الصفحة الأم ونعيد الربط تلقائياً بدل الاكتفاء بمرة واحدة فقط.
        new MutationObserver(bindAll).observe(doc.body, { childList: true, subtree: true });
    })();
    </script>
    """, height=0)

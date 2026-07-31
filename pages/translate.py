"""
pages/translate.py
تم تفكيكه تلقائياً من streamlit_app.py الأصلي (تقسيم الكود لتحسين القابلية للصيانة والأداء).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — إعادة تصدير كل الاستيرادات والدوال المساعدة المشتركة

_TRANSLATE_LANGS = {
    "🌐 اكتشاف تلقائي": "auto",
    "🇸🇦 العربية": "العربية",
    "🇬🇧 الإنجليزية": "الإنجليزية",
    "🇫🇷 الفرنسية": "الفرنسية",
    "🇪🇸 الإسبانية": "الإسبانية",
    "🇩🇪 الألمانية": "الألمانية",
    "🇹🇷 التركية": "التركية",
    "🇮🇷 الفارسية": "الفارسية",
    "🇵🇰 الأردية": "الأردية",
    "🇮🇩 الإندونيسية": "الإندونيسية",
    "🇲🇾 الملايوية": "الملايوية",
    "🇮🇳 الهندية": "الهندية",
    "🇷🇺 الروسية": "الروسية",
    "🇨🇳 الصينية": "الصينية",
    "🇧🇩 البنغالية": "البنغالية",
}


def render_translate():
    """تبويب الترجمة الفورية بين العربية ولغات أخرى شائعة لدى مستخدمي NSM،
    عبر نفس سلسلة LLMFallback المستخدمة بباقي النظام (بدون مفتاح API إضافي)."""

    st.markdown('<div class="section-header">🌐 ترجمة فورية</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="tab-intro">ترجمة نص باستخدام نفس نماذج NSM اللغوية '
        '(Anthropic ← Cloudflare ← Gemini ← OpenRouter ← Groq) — بدون حاجة '
        'لأي مفتاح Google Translate أو DeepL.</p>',
        unsafe_allow_html=True,
    )

    # يجب تطبيق أي "إعادة استخدام" من التاريخ *قبل* إنشاء ودجت text_area
    # مباشرة — تعيين session_state[key] بعد إنشاء الودجت بنفس الجولة يرفع
    # StreamlitAPIException.
    if "_tr_pending_reuse" in st.session_state:
        st.session_state["tr_source_text"] = st.session_state.pop("_tr_pending_reuse")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        src_label = st.selectbox("من لغة:", list(_TRANSLATE_LANGS.keys()), index=0, key="tr_src_lang")
    with c2:
        tgt_label = st.selectbox("إلى لغة:", list(_TRANSLATE_LANGS.keys()), index=2, key="tr_tgt_lang")

    source_text = st.text_area(
        "النص المراد ترجمته:",
        height=150,
        placeholder="اكتب أو الصق النص هنا...",
        key="tr_source_text",
    )

    translate_clicked = st.button(
        "🌐 ترجم الآن", type="primary", key="tr_translate_btn", use_container_width=True,
        disabled=not bool(source_text and source_text.strip()),
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if translate_clicked and not source_text.strip():
        st.warning("أدخل نصاً للترجمة أولاً.")
    elif translate_clicked and source_text.strip():
        src = _TRANSLATE_LANGS[src_label]
        tgt = _TRANSLATE_LANGS[tgt_label]

        if src == tgt and src != "auto":
            st.toast("⚠️ لغة المصدر ولغة الهدف متطابقتان", icon="⚠️")
        else:
            src_instruction = "اكتشف لغة النص تلقائياً ثم" if src == "auto" else f"ترجم من {src} إلى"
            system_prompt = (
                f"أنت مترجم محترف. {src_instruction} {tgt}. "
                "أعد فقط النص المترجم دون أي شرح أو مقدمات أو علامات اقتباس إضافية، "
                "مع الحفاظ على المعنى والأسلوب الأصلي بدقة."
            )
            _tr_skeleton_ph = st.empty()
            with _tr_skeleton_ph.container():
                _skeleton(lines=3)
            try:
                from ai.llm_fallback import LLMFallback
                _tr_llm = LLMFallback(max_tokens=1200, temperature=0.2)
                result = _tr_llm.generate(source_text.strip(), history=[], system_prompt=system_prompt)
                st.session_state.tr_result = result
                _tr_skeleton_ph.empty()
                st.toast("✅ تمت الترجمة بنجاح", icon="✅")
                if result and (result.text or "").strip() and not getattr(result, "error", None):
                    try:
                        from ai.translation_history import get_history
                        get_history().save(
                            src_lang=src_label, tgt_lang=tgt_label,
                            source_text=source_text.strip(), translated_text=result.text,
                            provider=getattr(result.provider, "value", str(result.provider)),
                        )
                    except Exception:
                        pass
            except Exception as e:  # noqa: BLE001
                _tr_skeleton_ph.empty()
                st.toast(f"⚠️ فشلت الترجمة: {e}", icon="⚠️")
                st.session_state.tr_result = None

    result = st.session_state.get("tr_result")
    if result is not None:
        st.markdown("#### 📄 الترجمة")
        st.markdown(f"""
        <div class="root-item" style="text-align:right; direction:rtl; line-height:1.9">
            {result.text}
        </div>
        """, unsafe_allow_html=True)
        provider_label = getattr(result.provider, "value", str(result.provider))
        st.caption(f"المزوّد: {provider_label}" + (f" · ⚠️ {result.error}" if getattr(result, "error", None) else ""))
        _copy_col, _dl_col = st.columns([1, 2])
        with _copy_col:
            _copy_button(result.text, key="tr_result")
        with _dl_col:
            st.download_button(
                "⬇️ تحميل الترجمة (txt)",
                data=result.text,
                file_name="translation.txt",
                mime="text/plain",
                key="tr_download_btn",
            )

    st.markdown("")
    st.markdown('<div class="section-header">🕘 آخر الترجمات</div>', unsafe_allow_html=True)
    try:
        from ai.translation_history import get_history
        _tr_history = get_history().list_recent(limit=15)
    except Exception as e:  # noqa: BLE001
        _tr_history = []
        st.caption(f"⚠️ تعذّر تحميل السجل: {e}")

    if not _tr_history:
        st.caption("📭 لا توجد ترجمات محفوظة بعد — أول ترجمة ناجحة ستظهر هنا تلقائياً.")
    else:
        for _tr_row in _tr_history:
            _tr_id = _tr_row["id"]
            _tr_excerpt = (_tr_row["source_text"] or "")[:60]
            _tr_header = f"{_tr_row['src_lang']} ← {_tr_row['tgt_lang']} — {_tr_excerpt}…"
            with st.expander(_tr_header):
                st.markdown(f"**النص الأصلي:**\n\n{_tr_row['source_text']}")
                st.markdown(f"**الترجمة:**\n\n{_tr_row['translated_text']}")
                _tr_reuse_col, _tr_del_col = st.columns(2)
                with _tr_reuse_col:
                    if st.button("↩️ استخدم هذا النص مجدداً", key=f"tr_reuse_{_tr_id}", use_container_width=True):
                        st.session_state["_tr_pending_reuse"] = _tr_row["source_text"]
                        st.rerun()
                with _tr_del_col:
                    if st.button("🗑️ حذف", key=f"tr_delete_{_tr_id}", use_container_width=True):
                        try:
                            from ai.translation_history import get_history as _gh2
                            _gh2().delete(_tr_id)
                        except Exception:
                            pass
                        st.rerun()

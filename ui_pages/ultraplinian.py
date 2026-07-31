"""
ui_pages/ultraplinian.py
تم تفكيكه تلقائياً من streamlit_app.py الأصلي (تقسيم الكود لتحسين القابلية للصيانة والأداء).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — إعادة تصدير كل الاستيرادات والدوال المساعدة المشتركة



# ══════════════════════════════════════════════════════════════════════════
# تبويب ⚡ ULTRAPLINIAN — سباق النماذج المتوازي عبر OpenRouter
# ══════════════════════════════════════════════════════════════════════════
def render_ultraplinian():
    st.markdown("### ⚡ ULTRAPLINIAN — سباق النماذج المتوازي")

    _or_key = st.session_state.get("_or_api_key", "").strip()
    _providers = available_providers()
    _has_direct = any(_providers.values())

    if not _ULTRAPLINIAN_OK:
        st.warning("⚠️ تعذّر تحميل وحدة ai/ultraplinian.py.")
        return
    if not _or_key and not _has_direct:
        st.info(
            "🔑 لا يوجد أي مزوّد جاهز — أضِف OpenRouter API Key في الشريط "
            "الجانبي، أو GROQ_API_KEY / GOOGLE_API_KEY / (CF_API_TOKEN + "
            "CF_ACCOUNT_ID) في Streamlit Secrets لتفعيل السباق مجاناً بدون "
            "OpenRouter."
        )
        return

    _direct_names = {"groq": "Groq", "gemini": "Gemini", "cloudflare": "Cloudflare"}
    _active = [v for k, v in _direct_names.items() if _providers.get(k)]
    if _active:
        st.caption("✅ مزوّدون مباشرون مفعّلون (مجاناً بدون OpenRouter): " + "، ".join(_active))
    elif not _or_key:
        st.caption("ℹ️ لا يوجد مزوّد مباشر مفعّل — سيُستخدم OpenRouter فقط لكل النماذج.")

    st.caption(
        f"يرسل نفس السؤال إلى عدة نماذج في آنٍ واحد (حتى {total_model_count()} نموذجاً "
        "عبر 5 مستويات)، يُقيّم كل رد بنقاط مركّبة (جودة النص + تصويت Borda + "
        "تشابه دلالي)، ويعرض الفائز."
    )
    st.markdown("---")

    if "ultraplinian_tier" not in st.session_state:
        st.session_state["ultraplinian_tier"] = "fast"
    if "ultraplinian_max_models" not in st.session_state:
        st.session_state["ultraplinian_max_models"] = DEFAULT_MAX_MODELS
    if "ultraplinian_results" not in st.session_state:
        st.session_state["ultraplinian_results"] = None
    if "ultraplinian_query" not in st.session_state:
        st.session_state["ultraplinian_query"] = ""

    c1, c2 = st.columns(2)
    with c1:
        tier_labels = {
            "fast": f"⚡ FAST ({TIER_CUMULATIVE.get('fast', 10)} نموذج تراكمياً)",
            "standard": f"🎯 STANDARD ({TIER_CUMULATIVE.get('standard', 20)} نموذج تراكمياً)",
            "smart": f"🧠 SMART ({TIER_CUMULATIVE.get('smart', 31)} نموذج تراكمياً)",
            "power": f"⚔️ POWER ({TIER_CUMULATIVE.get('power', 41)} نموذج تراكمياً)",
            "ultra": f"🔱 ULTRA ({TIER_CUMULATIVE.get('ultra', 51)} نموذج تراكمياً)",
        }
        sel_tier = st.selectbox(
            "المستوى", list(tier_labels.keys()),
            index=list(tier_labels.keys()).index(st.session_state["ultraplinian_tier"]),
            format_func=lambda k: tier_labels[k])
        st.session_state["ultraplinian_tier"] = sel_tier
    with c2:
        st.session_state["ultraplinian_max_models"] = st.slider(
            "عدد النماذج في السباق", min_value=2, max_value=10,
            value=min(st.session_state["ultraplinian_max_models"], 10),
            help="عدد أكبر = تكلفة API أعلى ووقت أطول. يُنصح بـ 3-6 للاستخدام العادي.")

    include_lower = st.checkbox(
        "تضمين المستويات الأدنى أيضاً (كما في النسخة الأصلية)", value=False)

    race_query = st.text_area(
        "السؤال للسباق", value=st.session_state["ultraplinian_query"],
        placeholder="اكتب سؤالاً لإرساله لجميع النماذج المختارة في آنٍ واحد...",
        height=100)

    run_col, clear_col = st.columns([3, 1])
    with run_col:
        launch = st.button("🏁 ابدأ السباق", type="primary", use_container_width=True,
                            disabled=not race_query.strip())
    with clear_col:
        if st.button("🗑 مسح النتائج", use_container_width=True):
            st.session_state["ultraplinian_results"] = None
            st.rerun()

    if launch and race_query.strip():
        st.session_state["ultraplinian_query"] = race_query.strip()
        models = get_tier_models(
            sel_tier, st.session_state["ultraplinian_max_models"], include_lower)

        sys_prompt = NSM_PERSONA_PROMPT if _ORCHESTRATOR_OK else NSM_SYSTEM_PROMPT

        progress_box = st.empty()
        progress_bar = st.progress(0.0)

        def _on_progress(model_name, done, total):
            progress_box.caption(f"✓ اكتمل: {model_name.split('/')[-1]} ({done}/{total})")
            progress_bar.progress(done / total)

        with st.spinner(f"⚡ يتسابق {len(models)} نموذجاً..."):
            results = run_race(
                user_query=race_query.strip(),
                system_prompt=sys_prompt,
                api_key=_or_key,
                models=models,
                on_progress=_on_progress,
            )
        progress_box.empty()
        progress_bar.empty()
        st.session_state["ultraplinian_results"] = results
        st.rerun()

    results = st.session_state["ultraplinian_results"]
    if results:
        st.markdown("---")
        successes = [r for r in results if not r.error]
        failures = [r for r in results if r.error]

        if successes:
            winner = successes[0]
            st.markdown(
                f"""<div style="border:2px solid var(--gold);border-radius:10px;padding:16px;
                background:var(--gold-soft);margin-bottom:16px;">
                🏆 <b style="color:var(--gold);font-size:1.1rem;"> {winner.model.split('/')[-1]}</b>
                <span style="color:var(--text-muted);font-size:.75rem;"> — نقاط مركّبة: {winner.compound_score}</span>
                </div>""",
                unsafe_allow_html=True,
            )
            st.markdown(winner.content)
            st.markdown("---")
            st.markdown("**📊 جميع النتائج (مرتبة تنازلياً)**")
            for r in successes:
                label = f"{'🏆 ' if r.is_winner else f'#{r.rank} '}{r.model.split('/')[-1]}"
                with st.expander(
                    f"{label} — مركّبة: {r.compound_score} | "
                    f"خام: {r.raw_score} | Borda: {r.borda_score} | تشابه: {r.cluster_score} | "
                    f"{r.duration_ms:.0f}ms"
                ):
                    st.markdown(r.content[:3000] + ("…" if len(r.content) > 3000 else ""))

        if failures:
            with st.expander(f"⚠ {len(failures)} نموذج فشل"):
                for r in failures:
                    st.caption(f"**{r.model}**")
                    st.caption(friendly_error(r.error))

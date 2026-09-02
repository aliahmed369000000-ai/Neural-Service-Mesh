"""
ui_pages/health.py
تم تفكيكه تلقائياً من streamlit_app.py الأصلي (تقسيم الكود لتحسين القابلية للصيانة والأداء).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — إعادة تصدير كل الاستيرادات والدوال المساعدة المشتركة



def render_health():
    """تبويب صحة النظام."""
    st.markdown('<div class="section-header">🏥 صحة النظام</div>', unsafe_allow_html=True)

    # نبض موحّد للمشروع
    try:
        from ai.system_hub import system_snapshot, format_system_report
        snap = system_snapshot()
        st.metric("الصحة الكلية (System Hub)", f"{snap['score']:.0%}")
        with st.expander("🌐 تقرير النظام الموحّد", expanded=True):
            st.markdown(format_system_report(snap))
    except Exception as _e:
        st.caption(f"System Hub غير متاح: {_e}")

    checks = []

    # ── 0. وضع النشر المغلق (offline)
    try:
        from ai.offline_mode import offline_status, disabled_online_features
        _os = offline_status()
        if _os.get("offline_mode"):
            _ok = bool(_os.get("ollama_reachable"))
            checks.append((
                "✅" if _ok else "⚠️",
                "وضع النشر المغلق",
                _os.get("message") or "نشط",
                _ok,
            ))
            with st.expander("🔒 تفاصيل الوضع المغلق"):
                st.caption(_os.get("message", ""))
                st.json({k: v for k, v in _os.items() if k != "message"})
                st.caption("ميزات معطّلة: " + "، ".join(disabled_online_features()))
        else:
            checks.append(("ℹ️", "وضع النشر المغلق", "غير مفعّل (وضع متصل)", True))
    except Exception as _off_err:
        checks.append(("⚠️", "وضع النشر المغلق", f"تعذّر الفحص: {_off_err}", False))

    # ── 1. الأوزان محفوظة؟
    weights_path = CHECKPOINTS_DIR / "neural_weights.npy"
    if weights_path.exists():
        size_kb = weights_path.stat().st_size / 1024
        checks.append(("✅", "الأوزان العصبية", f"محفوظة ({size_kb:.1f} KB)", True))
    else:
        checks.append(("❌", "الأوزان العصبية", "ملف الأوزان غير موجود", False))

    # ── 2. CKG محفوظ؟
    ckg_path = KNOWLEDGE_DIR / "cognitive_graph.json"
    if ckg_path.exists() and ckg_path.stat().st_size > 10:
        ckg = load_ckg()
        n_concepts = len(ckg.get("concepts", {}))
        checks.append(("✅", "قاعدة المعرفة CKG", f"موجودة ({n_concepts} مفهوم)", True))
    else:
        checks.append(("⚠️", "قاعدة المعرفة CKG", "فارغة أو غير موجودة", False))

    # ── 3. قاعدة البيانات
    db_path = MEMORY_DIR / "episodic.db"
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            conn.close()
            checks.append(("✅", "قاعدة الذاكرة (SQLite)", f"متصلة ({count} سجل)", True))
        except Exception as e:
            checks.append(("❌", "قاعدة الذاكرة (SQLite)", f"خطأ: {e}", False))
    else:
        checks.append(("❌", "قاعدة الذاكرة (SQLite)", "غير موجودة", False))

    # ── 4. القرآن الكريم
    chunks = list(KNOWLEDGE_DIR.glob("quran_chunk_*.json"))
    if len(chunks) >= 60:
        checks.append(("✅", "بيانات القرآن الكريم", f"{len(chunks)} chunk محمّل (6,236 آية)", True))
    else:
        checks.append(("⚠️", "بيانات القرآن الكريم", f"وُجد {len(chunks)} chunk فقط", False))

    # ── 5. الجذور العربية
    roots = load_arabic_roots()
    if len(roots) > 100:
        checks.append(("✅", "فهرس الجذور العربية", f"{len(roots)} جذر مكتشف", True))
    else:
        checks.append(("⚠️", "فهرس الجذور العربية", f"{len(roots)} جذر فقط", False))

    # ── 6. نقطة حفظ حديثة
    checkpoint_files = sorted(CHECKPOINTS_DIR.glob("brain_checkpoint_*.json"), reverse=True)
    if checkpoint_files:
        latest = load_latest_checkpoint()
        saved_at = latest.get("saved_at", "")
        checks.append(("✅", "نقطة الحفظ الأخيرة (Checkpoint)", saved_at[:19] if saved_at else "موجودة", True))
    else:
        checks.append(("❌", "نقطة الحفظ الأخيرة (Checkpoint)", "لا توجد نقطة حفظ", False))

    # ── 7. التدريب
    training = load_training_summary()
    if training.get("train_steps", 0) > 0:
        checks.append(("✅", "حالة التدريب", f"{training['train_steps']:,} خطوة مكتملة", True))
    else:
        checks.append(("⚠️", "حالة التدريب", "لم يكتمل تدريب بعد", False))

    # ── 8. مزوّد LLM الحالي ─────────────────────────────────────────────
    try:
        from ai.llm_fallback import LLMFallback
        _fb = LLMFallback()
        fb_info = _fb.info()
        _prov   = fb_info.get("provider", "غير محدد")
        _model  = fb_info.get("model", "غير محدد")
        _live   = fb_info.get("live_llm", "❌")
        checks.append(("✅" if "✅" in _live else "⚠️", f"مزوّد LLM — {_prov}", _model, "✅" in _live))
    except Exception as _e:
        checks.append(("⚠️", "مزوّد LLM", str(_e)[:60], False))

    # عرض النتائج
    all_ok = sum(1 for c in checks if c[3])
    total  = len(checks)

    if all_ok == total:
        st.success(f"✅ النظام يعمل بكفاءة كاملة ({all_ok}/{total})")
    elif all_ok >= total * 0.7:
        st.warning(f"⚠️ النظام يعمل جزئياً ({all_ok}/{total})")
    else:
        st.error(f"❌ بعض مكونات النظام تحتاج انتباهاً ({all_ok}/{total})")

    st.markdown("")
    for icon, name, detail, ok in checks:
        # 🩹 إصلاح تباين: كانت الخلفية/الحد بلون فاتح ثابت (#f0fdf4/#fef2f2)
        # بينما النص أبيض تقريباً في الوضع الداكن (--text) → نص شبه غير
        # مقروء فوق خلفية فاتحة. الآن نستخدم متغيرات السمة emerald/rose
        # الموجودة أصلاً، فتتكيّف الخلفية والنص معاً مع الوضع الداكن/الفاتح.
        _status_color = "var(--emerald)" if ok else "var(--rose)"
        _status_bg = "var(--emerald-soft)" if ok else "var(--rose-soft)"
        st.markdown(f"""
        <div style="padding: 0.6rem 1rem; margin: 0.3rem 0; background: {_status_bg};
                    border-radius: 8px; border: 1px solid {_status_color}; color: var(--text);">
            <span style="font-size:1.2rem">{icon}</span>
            &nbsp;<strong>{name}</strong>
            &nbsp;&nbsp;<small style="color:var(--text-muted)">{detail}</small>
        </div>
        """, unsafe_allow_html=True)

    # ── فحص تشخيصي: كل وحدة اختيارية يستوردها app_core.py ضمن try/except
    #    صامت — يعيد المحاولة بشكل مستقل ويعرض سبب الفشل الحقيقي بدل
    #    اختفائه بصمت (هذا بالضبط ما أخفى عطل NSM Router Bridge سابقاً).
    st.markdown("")
    st.markdown('<div class="section-header">🔬 فحص تشخيصي للوحدات الاختيارية</div>', unsafe_allow_html=True)
    try:
        from ai.module_health import run_module_health_checks
        _mh_results = run_module_health_checks()
        _mh_ok = sum(1 for r in _mh_results if r.ok)
        _mh_total = len(_mh_results)

        if _mh_ok == _mh_total:
            st.success(f"✅ كل الوحدات الاختيارية تعمل ({_mh_ok}/{_mh_total})")
        else:
            _mh_failed = [r for r in _mh_results if not r.ok]
            st.error(f"❌ {len(_mh_failed)} وحدة معطّلة صامتاً — لولا هذا الفحص لما ظهر السبب")
            for r in _mh_failed:
                st.markdown(f"""
                <div style="padding: 0.6rem 1rem; margin: 0.3rem 0; background: var(--rose-soft);
                            border-radius: 8px; border: 1px solid var(--rose); color: var(--text);">
                    <span style="font-size:1.2rem">❌</span>
                    &nbsp;<strong>{r.label}</strong>
                    &nbsp;<code style="font-size:0.75rem">{r.module}</code><br/>
                    <small style="color:var(--text-muted)">{r.reason}</small>
                </div>
                """, unsafe_allow_html=True)

        with st.expander(f"📋 التفاصيل الكاملة ({_mh_total} وحدة)"):
            for r in _mh_results:
                icon = "✅" if r.ok else "❌"
                extra = f" — {r.reason}" if not r.ok else (f" — {r.note}" if r.note else "")
                st.caption(f"{icon} **{r.label}** (`{r.module}`){extra}")
    except Exception as _mh_err:
        st.caption(f"الفحص التشخيصي غير متاح: {_mh_err}")

    # ── نماذج Anthropic المتاحة (من That.md) ────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🤖 نماذج Anthropic المتاحة</div>', unsafe_allow_html=True)
    try:
        from ai.llm_fallback import ANTHROPIC_MODELS
        model_rows = {
            "sonnet":  ("claude-sonnet-4-6",          "⚡ Sonnet 4",  "الافتراضي — توازن مثالي بين الجودة والسرعة"),
            "opus":    ("claude-opus-4-8",             "💎 Opus 4",    "المهام المعقدة — الأعلى جودةً"),
            "haiku":   ("claude-haiku-4-5-20251001",   "🚀 Haiku 4",   "الردود الفورية — الأخف والأسرع"),
            "stable":  ("claude-sonnet-4-20250514",    "🔒 Sonnet Stable", "الإصدار المستقر للإنتاج"),
        }
        cols = st.columns(len(model_rows))
        for col, (key, (model_id, label, desc)) in zip(cols, model_rows.items()):
            with col:
                is_active = ANTHROPIC_MODELS.get(key) == model_id
                border_color = "var(--gold)" if is_active else "var(--text-muted)"
                st.markdown(f"""
                <div style="background:var(--surface2);border:2px solid {border_color};border-radius:10px;
                            padding:0.8rem;text-align:center;direction:ltr;color:var(--text)">
                    <div style="font-size:1.3rem">{label}</div>
                    <code style="font-size:0.72rem;color:var(--gold)">{model_id}</code>
                    <div style="font-size:0.78rem;color:var(--text-muted);margin-top:0.4rem;direction:rtl">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        st.caption("المصدر: Claude.ai System Prompt (That.md) — محدَّث 2026")
    except Exception as _me:
        st.info(f"تعذّر تحميل قائمة النماذج: {_me}")

    # ── جاهزية المشروع (تحليل ثابت للكود، لا يحتاج بيانات تشغيل حيّة) ────
    st.markdown("---")
    st.markdown('<div class="section-header">🩺 جاهزية المشروع (تحليل الكود)</div>', unsafe_allow_html=True)
    try:
        from ai.validator import Phase6Validator
        _v = Phase6Validator(mesh=None)
        _report = _v.generate()
        _dc = _report.get("dead_code", {})
        _pc = _report.get("phase_coverage", {})
        _p7 = _report.get("phase7_readiness", {})

        _h1, _h2, _h3 = st.columns(3)
        _h1.metric("🧬 تغطية المراحل", f"{_pc.get('overall_coverage_pct', 0):.0f}%")
        _h2.metric("💀 كود ميت", f"{_dc.get('dead_pct', 0):.1f}%",
                    delta=None if _dc.get("dead_count", 0) == 0 else f"{_dc.get('dead_count')} ملف",
                    delta_color="inverse")
        _h3.metric("🎯 جاهزية Phase 7", f"{_p7.get('score', 0):.0f}/{_p7.get('max_score', 100)}")

        if _dc.get("dead_files"):
            with st.expander(f"📄 الملفات غير المربوطة ({_dc['dead_count']})"):
                for _f in _dc["dead_files"]:
                    st.code(_f, language=None)

        _recs = _p7.get("recommendations") or []
        if _recs:
            with st.expander(f"💡 توصيات لرفع الجاهزية ({len(_recs)})"):
                for _r in _recs:
                    st.markdown(f"- {_r}")
    except Exception as _ve:
        st.info(f"تعذّر تحميل تقرير الجاهزية: {_ve}")

    # 🆕 تبسيط: أصبحت ميزة "رفع إلى GitHub" (نادرة الاستخدام يومياً)
    # خلف عنصر قابل للطي بدل الظهور الدائم المفتوح — نفس النموذج ونفس
    # منطق git add/commit/push بالضبط بلا أي تعديل.
    with st.expander("🚀 رفع إلى GitHub", expanded=False):
        _gh_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "").strip()
        if not _gh_token:
            st.warning("🔑 أضف **GITHUB_PERSONAL_ACCESS_TOKEN** في Secrets لتفعيل هذه الميزة.")
        else:
            col_gh1, col_gh2 = st.columns([3, 1])
            with col_gh1:
                commit_msg = st.text_input(
                    "رسالة الـ Commit",
                    value="NSM update — رفع من الواجهة",
                    key="gh_commit_msg",
                    label_visibility="visible",
                )
            with col_gh2:
                st.markdown("<br>", unsafe_allow_html=True)
                push_btn = st.button("⬆️ Push", key="gh_push_btn", use_container_width=True, type="primary")

            if push_btn:
                if not commit_msg.strip():
                    st.warning("أدخل رسالة commit أولاً.")
                else:
                    import subprocess as _sp
                    with st.spinner("⟳ جارٍ الرفع إلى GitHub..."):
                        try:
                            # git add
                            r_add = _sp.run(
                                ["git", "add", "-A"],
                                cwd=str(BASE), capture_output=True, text=True, timeout=15
                            )
                            if r_add.returncode != 0:
                                st.error(f"❌ فشل git add:\n{r_add.stderr[:400] or r_add.stdout[:400]}")
                                raise RuntimeError("git add failed")
                            # git commit
                            r_commit = _sp.run(
                                ["git", "-c", "user.email=nsm@replit.com",
                                 "-c", "user.name=NSM Agent",
                                 "commit", "-m", commit_msg.strip()],
                                cwd=str(BASE), capture_output=True, text=True, timeout=15,
                                env={**os.environ,
                                     "GIT_AUTHOR_NAME": "NSM Agent",
                                     "GIT_AUTHOR_EMAIL": "nsm@replit.com",
                                     "GIT_COMMITTER_NAME": "NSM Agent",
                                     "GIT_COMMITTER_EMAIL": "nsm@replit.com"},
                            )
                            # إذا لا يوجد تغيير جديد، نكمل الـ push للـ commit الحالي
                            nothing_to_commit = (
                                r_commit.returncode != 0 and
                                "nothing to commit" in (r_commit.stdout + r_commit.stderr)
                            )
                            if r_commit.returncode != 0 and not nothing_to_commit:
                                st.error(f"❌ فشل Commit:\n{r_commit.stderr[:400] or r_commit.stdout[:400]}")
                            else:
                                # git push
                                _remote = (
                                    f"https://aliahmed369000000-ai:{_gh_token}"
                                    "@github.com/aliahmed369000000-ai/Neural-Service-Mesh.git"
                                )
                                r_push = _sp.run(
                                    ["git", "push", _remote, "main"],
                                    cwd=str(BASE), capture_output=True, text=True, timeout=30
                                )
                                if r_push.returncode == 0:
                                    st.success("✅ تم الرفع إلى GitHub بنجاح!")
                                    # عرض معلومات الـ commit الأخير
                                    r_log = _sp.run(
                                        ["git", "log", "--oneline", "-1"],
                                        cwd=str(BASE), capture_output=True, text=True
                                    )
                                    st.code(r_log.stdout.strip(), language="text")
                                else:
                                    st.error(f"❌ فشل Push:\n{r_push.stderr[:400] or r_push.stdout[:400]}")
                        except Exception as _gh_err:
                            st.error(f"❌ خطأ غير متوقع: {_gh_err}")

            # عرض آخر commit
            # 🐛 إصلاح خطأ حقيقي (اكتُشف بالفحص الشامل بعد إصلاح مشابه):
            # كان هنا st.expander("📋 آخر 3 commits") أصبح متداخلاً داخل
            # st.expander("🚀 رفع إلى GitHub") الخارجي بعد لفّ القسم بالكامل
            # بطيّة — يظهر فقط عند ضبط GITHUB_PERSONAL_ACCESS_TOKEN، ولذلك
            # لم يظهر بالاختبار الأول (لا يوجد التوكن ببيئة الاختبار). استُبدل
            # بعرض مباشر (بلا طيّة ثانية) بما أن القسم كله أصلاً خلف الطيّة
            # الخارجية.
            try:
                import subprocess as _sp2
                _log = _sp2.run(
                    ["git", "log", "--oneline", "-3"],
                    cwd=str(BASE), capture_output=True, text=True, timeout=5
                )
                if _log.stdout.strip():
                    st.caption("📋 آخر 3 commits")
                    st.code(_log.stdout.strip(), language="text")
            except Exception:
                pass

    # أزرار الإجراءات
    st.markdown("---")
    st.markdown('<div class="section-header">⚙️ إجراءات</div>', unsafe_allow_html=True)
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("🔄 تحديث الإحصاءات", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col_r2:
        st.markdown("""
        <div style="background:color-mix(in srgb, #38bdf8 12%, var(--surface2)); border:1px solid color-mix(in srgb, #38bdf8 40%, var(--border)); border-radius:8px; padding:0.6rem 1rem; font-size:0.85rem; direction:rtl; color:var(--text)">
            لتشغيل دورة تدريب، افتح Google Colab وشغّل <code>train_simulate.py</code>
        </div>
        """, unsafe_allow_html=True)

    # ── رقابة/تدقيق تفاعلات الوكلاء (Observability) ──
    # سجل مستقل تماماً عن CKG (القرآن) — يتتبّع فقط استدعاءات وكلاء AI
    # (ai/agent_categories.py) من "hub" أو "orchestrator" لأغراض التشخيص.
    st.markdown("---")
    with st.expander("🔎 رقابة وكلاء AI (Observability)", expanded=False):
        try:
            from ai.agent_audit import get_default_audit_log
            _audit = get_default_audit_log()
            _summary = _audit.summary()
        except Exception as _audit_err:
            _audit = None
            _summary = None
            st.caption(f"⚠️ تعذّر تحميل سجل تدقيق الوكلاء: {_audit_err}")

        if _summary:
            if _summary["total_events"] == 0:
                st.caption("لا توجد تفاعلات مسجَّلة بعد — استخدم تبويب \"🤖 وكلاء AI\" أو \"🤝 منسّق الوكلاء\" أولاً.")
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("إجمالي التفاعلات", _summary["total_events"])
                m2.metric("عبر hub", _summary["by_source"].get("hub", 0))
                m3.metric("عبر orchestrator", _summary["by_source"].get("orchestrator", 0))

                web_pct = (
                    (_summary["web_used_count"] / _summary["total_events"]) * 100
                    if _summary["total_events"] else 0
                )
                st.caption(f"🌐 استخدم بحث ويب حقيقي في {_summary['web_used_count']} تفاعل ({web_pct:.0f}%)")

                if _summary["by_category"]:
                    st.markdown(
                        "**حسب الوكيل:** " + "، ".join(
                            f"{k}: {v}" for k, v in _summary["by_category"].items()
                        )
                    )

                # 🆕 ملاحظة: كان هذا أصلاً st.expander متداخل داخل expander آخر —
                # Streamlit لا يدعم تعشيش expander داخل expander (يرفع
                # StreamlitAPIException وقت التشغيل). بما أن القسم كله أصلاً
                # أصبح خلف طيّة "🔎 رقابة وكلاء AI" الخارجية، استُبدل بعنوان
                # فرعي بسيط بدل طيّة ثانية — نفس المحتوى والبيانات بالضبط.
                st.markdown("**📋 آخر التفاعلات المسجَّلة**")
                recent = _audit.get_recent(15)
                for entry in recent:
                        web_tag = "🌐" if entry.get("web_used") else ""
                        src_tag = "🤝" if entry.get("source") == "orchestrator" else "🤖"
                        st.markdown(
                            f"{src_tag} **{entry.get('category_title', '')}** "
                            f"{web_tag} — {entry.get('provider', '') or '—'} "
                            f"— {entry.get('timestamp', '')[:19]}"
                        )
                        q = entry.get("question_preview", "")
                        if q:
                            st.caption(f"س: {q[:120]}{'…' if len(q) > 120 else ''}")

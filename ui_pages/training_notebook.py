"""
ui_pages/training_notebook.py — مختبر تدريب احترافي (Colab/Kaggle style)
"""
from __future__ import annotations

import json

import streamlit as st


def _ensure_nb():
    from ai.notebook_engine import create_notebook, list_notebooks, load_notebook

    if "nsm_nb_id" not in st.session_state:
        existing = list_notebooks()
        if existing:
            st.session_state.nsm_nb_id = existing[0]["id"]
        else:
            nb = create_notebook("NSM Training Lab", template="training")
            st.session_state.nsm_nb_id = nb.id
    nb = load_notebook(st.session_state.nsm_nb_id)
    if nb is None:
        nb = create_notebook("NSM Training Lab", template="training")
        st.session_state.nsm_nb_id = nb.id
    return nb


def _badge(ok: bool, yes: str = "جاهز", no: str = "غير جاهز") -> str:
    return f"{'🟢' if ok else '🔴'} {yes if ok else no}"


def _latest_job_banner():
    """شريط حالة مختصر لآخر مهمة تدريب — يظهر أعلى الواجهة بغض النظر عن
    التبويب المفتوح، بدل ما المستخدم يضطر يفتح تبويب «المهام» ليكتشف
    إن كانت آخر مهمة نجحت أو لسه شغّالة."""
    from ai.notebook_lab_service import list_jobs, refresh_job_status

    jobs = list_jobs(1)
    if not jobs:
        return
    j = jobs[0]
    job_id = j.get("job_id")
    ok = j.get("ok")
    title = j.get("preset_key") or j.get("type") or "مهمة"
    icon = "✅" if ok else ("❌" if ok is False else "🟡")

    with st.container():
        b1, b2, b3 = st.columns([4, 1.3, 1.3])
        with b1:
            st.markdown(
                f"{icon} **آخر مهمة:** {title} · `{job_id or '—'}` · "
                f"{(j.get('recorded_at') or '')[:19]}"
            )
        with b2:
            if job_id and st.button("🔄 تحديث الحالة", key="banner_job_refresh", use_container_width=True):
                st.session_state["lab_job_status"] = refresh_job_status(job_id)
                st.rerun()
        with b3:
            if j.get("kernel_url"):
                st.markdown(f"[↗ فتح على Kaggle]({j['kernel_url']})")

        live = st.session_state.get("lab_job_status")
        if live and live.get("job_id") == job_id:
            status_text = live.get("status") or live.get("cli_status") or live.get("raw")
            if status_text:
                st.caption(f"الحالة الحية: {str(status_text)[:200]}")
        if j.get("error"):
            st.caption(f"⚠️ {str(j['error'])[:200]}")
        st.markdown("---")


def render_training_notebook():
    """مختبر تدريب متكامل — سهل الاستخدام واحترافي."""
    st.markdown(
        '<div class="section-header">📓 مختبر التدريب الاحترافي</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "خلايا مثل Colab/Kaggle · إطلاق GPU عبر API · قوالب SurahChain · سجل مهام · "
        "التدريب الثقيل على Kaggle — التحرير والمراقبة من هنا"
    )

    from ai.notebook_engine import (
        add_cell,
        create_notebook,
        delete_cell,
        delete_notebook,
        detect_compute,
        export_ipynb,
        list_notebooks,
        load_notebook,
        move_cell,
        plan_remote_run,
        run_all,
        run_cell,
        save_notebook,
    )
    from ai.notebook_lab_service import (
        PRESETS,
        feature_list_ar,
        lab_health,
        launch_preset,
        list_jobs,
        refresh_job_status,
    )

    _latest_job_banner()

    # ── تبويبات فرعية للمختبر ──
    lab_tabs = st.tabs(
        [
            "🚀 إطلاق سريع",
            "📓 الدفتر",
            "📋 المهام",
            "🖥️ البيئة",
            "✨ الميزات",
        ]
    )

    # ═══════════ 1) إطلاق سريع ═══════════
    with lab_tabs[0]:
        st.markdown("### ابدأ تدريباً على Kaggle GPU")
        health = lab_health()
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(_badge(health.get("ready_to_launch_kaggle", False), "مفاتيح Kaggle", "أضف Secrets"))
        with c2:
            cli = (health.get("checks") or {}).get("kaggle_cli")
            st.markdown(_badge(bool(cli), "Kaggle CLI", "pip install kaggle"))
        with c3:
            gpu = (health.get("checks") or {}).get("local_gpu") or {}
            st.markdown(
                f"{'🟢' if gpu.get('cuda') else '🟡'} GPU محلي: "
                f"{'نعم' if gpu.get('cuda') else 'لا'} — التدريب الثقيل على Kaggle"
            )

        if not health.get("ready_to_launch_kaggle"):
            st.warning(
                "ضع في **Streamlit Secrets**:\n\n"
                "```toml\nKAGGLE_USERNAME = \"...\"\nKAGGLE_KEY = \"...\"\n```\n\n"
                "وعلى Kaggle Secrets: `GITHUB_TOKEN` للرفع بعد التدريب."
            )

        st.markdown("#### اختر قالباً")
        preset_keys = list(PRESETS.keys())
        labels = [f"{PRESETS[k]['label_ar']} · ~{PRESETS[k]['eta_hours']} ساعة" for k in preset_keys]
        pick = st.radio("القالب", labels, index=2, horizontal=False, key="lab_preset_radio")
        preset_key = preset_keys[labels.index(pick)]
        cfg = PRESETS[preset_key]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("d_model", cfg["d_model"])
        m2.metric("N", f"{cfg['n']:,}")
        m3.metric("epochs", cfg["epochs"])
        m4.metric("batch", cfg["batch"])
        m5.metric("تقدير", cfg["eta_hours"] + "س")
        st.caption(cfg["desc"])

        x1, x2 = st.columns(2)
        with x1:
            fresh = st.checkbox("من الصفر (FRESH)", value=True, key="lab_fresh")
        with x2:
            auto_push = st.checkbox("رفع تلقائي بعد النجاح", value=True, key="lab_autopush")

        if st.button(
            "▶ ابدأ التدريب على Kaggle الآن",
            type="primary",
            use_container_width=True,
            key="lab_launch_btn",
            disabled=not health.get("ready_to_launch_kaggle"),
        ):
            with st.spinner("تجهيز الـkernel ودفعه عبر Kaggle API…"):
                res = launch_preset(preset_key, fresh=fresh, auto_push=auto_push)
            st.session_state["lab_last_launch"] = res
            st.rerun()

        # تخصيص يدوي
        with st.expander("⚙️ تخصيص يدوي (يتجاوز القالب)"):
            d1, d2, d3, d4 = st.columns(4)
            with d1:
                man_preset = st.selectbox("SCN_PRESET", ["small", "medium", "large"], index=1, key="man_preset")
            with d2:
                man_ep = st.number_input("epochs", 1, 200, int(cfg["epochs"]), key="man_ep")
            with d3:
                man_n = st.number_input("N", 1000, 500000, int(cfg["n"]), step=1000, key="man_n")
            with d4:
                man_bs = st.number_input("batch", 4, 128, int(cfg["batch"]), key="man_bs")
            if st.button("▶ إطلاق بالإعدادات المخصصة", use_container_width=True, key="lab_launch_custom"):
                with st.spinner("دفع…"):
                    try:
                        from ai.kaggle_provider import start_surahchain_training_api
                        from ai.notebook_lab_service import append_job
                        res = start_surahchain_training_api(
                            preset=man_preset,
                            n=int(man_n),
                            epochs=int(man_ep),
                            batch=int(man_bs),
                            fresh=fresh,
                            auto_push=auto_push,
                        )
                        append_job({
                            "type": "surahchain_custom",
                            "ok": res.get("ok"),
                            "job_id": res.get("job_id"),
                            "kernel_url": res.get("kernel_url"),
                        })
                        st.session_state["lab_last_launch"] = res
                    except Exception as e:
                        st.session_state["lab_last_launch"] = {"ok": False, "error": str(e)}
                st.rerun()

        if st.session_state.get("lab_last_launch"):
            res = st.session_state["lab_last_launch"]
            if res.get("ok"):
                st.success(res.get("msg_ar") or "تم دفع المهمة إلى Kaggle")
                if res.get("kernel_url"):
                    st.markdown(f"**رابط المتابعة:** [{res['kernel_url']}]({res['kernel_url']})")
                st.info(
                    "بعد البدء: راقب Logs على Kaggle. ابحث عن `--- 2) التدريب ---`. "
                    "GITHUB_TOKEN في Kaggle Secrets مطلوب للرفع النهائي."
                )
            else:
                st.error(str(res.get("error") or res.get("msg_ar") or "فشل الإطلاق"))
                if res.get("need"):
                    st.code(str(res["need"]))
                pout = (res.get("push") or {}).get("output") or (res.get("push") or {}).get("error")
                if pout:
                    st.code(str(pout)[:2500])
            with st.expander("تفاصيل الاستجابة"):
                st.json(res)

    # ═══════════ 2) الدفتر ═══════════
    with lab_tabs[1]:
        notebooks = list_notebooks()
        c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
        with c1:
            options = {f"{n['name']} ({n['id']})": n["id"] for n in notebooks} or {}
            if options:
                labels_nb = list(options.keys())
                cur = st.session_state.get("nsm_nb_id")
                idx = 0
                for i, oid in enumerate(options.values()):
                    if oid == cur:
                        idx = i
                        break
                choice = st.selectbox("الدفتر", labels_nb, index=idx, key="nsm_nb_select")
                st.session_state.nsm_nb_id = options[choice]
            else:
                st.info("لا دفاتر — أنشئ واحداً")
        with c2:
            if st.button("➕ عام", use_container_width=True, key="nb_new"):
                nb = create_notebook("NSM Lab", template="training")
                st.session_state.nsm_nb_id = nb.id
                st.rerun()
            if st.button("📖 SurahChain", use_container_width=True, key="nb_surah"):
                nb = create_notebook("SurahChain Kaggle Lab", template="surahchain")
                st.session_state.nsm_nb_id = nb.id
                st.rerun()
            cur_id = st.session_state.get("nsm_nb_id")
            if st.session_state.get("nb_confirm_delete") == cur_id:
                st.warning("حذف الدفتر نهائي — تأكيد؟")
                dc1, dc2 = st.columns(2)
                with dc1:
                    if st.button("✅ احذف", use_container_width=True, key="nb_delete_confirm"):
                        delete_notebook(cur_id)
                        st.session_state.pop("nb_confirm_delete", None)
                        st.session_state.pop("nsm_nb_id", None)
                        st.rerun()
                with dc2:
                    if st.button("إلغاء", use_container_width=True, key="nb_delete_cancel"):
                        st.session_state.pop("nb_confirm_delete", None)
                        st.rerun()
            elif st.button("🗑 حذف الدفتر الحالي", use_container_width=True, key="nb_delete_ask"):
                st.session_state["nb_confirm_delete"] = cur_id
                st.rerun()
        with c3:
            provider = st.selectbox(
                "المزوّد",
                ["local", "kaggle", "modal", "lightning", "huggingface", "colab", "runpod", "vast"],
                key="nsm_nb_provider",
            )
        with c4:
            timeout = st.number_input("timeout ث", 30, 600, 120, key="nsm_nb_timeout")

        nb = _ensure_nb()
        nb.provider = provider
        save_notebook(nb)

        t1, t2, t3, t4, t5, t6 = st.columns(6)
        with t1:
            if st.button("▶️ Run All", use_container_width=True, key="nb_run_all"):
                with st.spinner("تشغيل…"):
                    results = run_all(nb, timeout=int(timeout), stop_on_error=True)
                st.toast(str(results)[:120], icon="✅")
                st.rerun()
        with t2:
            if st.button("➕ Code", use_container_width=True, key="nb_add_code"):
                add_cell(nb, "code", "# cell\nprint('ok')")
                st.rerun()
        with t3:
            if st.button("➕ MD", use_container_width=True, key="nb_add_md"):
                add_cell(nb, "markdown", "### عنوان")
                st.rerun()
        with t4:
            if st.button("➕ Bash", use_container_width=True, key="nb_add_bash"):
                add_cell(nb, "bash", "ls -la")
                st.rerun()
        with t5:
            if st.button("➕ Train", use_container_width=True, key="nb_add_train"):
                add_cell(nb, "train", "print('train')")
                st.rerun()
        with t6:
            st.download_button(
                "⬇️ ipynb",
                data=json.dumps(export_ipynb(nb), ensure_ascii=False, indent=2),
                file_name=f"{nb.name.replace(' ', '_')}.ipynb",
                mime="application/json",
                use_container_width=True,
                key="dl_ipynb2",
            )

        st.markdown(
            f"**{nb.name}** · `{nb.id}` · خلايا **{len(nb.cells)}** · "
            f"حُفظ `{nb.updated_at[:19] if nb.updated_at else '—'}`"
        )

        for i, cell in enumerate(list(nb.cells)):
            badge = {"markdown": "📝", "code": "🐍", "bash": "💻", "train": "🏋️"}.get(cell.type, "•")
            status_icon = {"ok": "✅", "error": "❌", "running": "⏳", "idle": "⚪"}.get(cell.status, "⚪")
            with st.container():
                h1, h2, h3, h4, h5, h6 = st.columns([1.4, 0.45, 0.45, 0.45, 0.45, 0.8])
                with h1:
                    st.markdown(f"**[{i}] {badge} {cell.type}** {status_icon} `{cell.id}`")
                with h2:
                    if st.button("▶", key=f"run_{cell.id}"):
                        with st.spinner("…"):
                            run_cell(nb, cell.id, timeout=int(timeout))
                        st.rerun()
                with h3:
                    if st.button("↑", key=f"up_{cell.id}"):
                        move_cell(nb, cell.id, -1)
                        st.rerun()
                with h4:
                    if st.button("↓", key=f"dn_{cell.id}"):
                        move_cell(nb, cell.id, 1)
                        st.rerun()
                with h5:
                    if st.button("🗑", key=f"del_{cell.id}"):
                        delete_cell(nb, cell.id)
                        st.rerun()
                with h6:
                    types = ["markdown", "code", "bash", "train"]
                    ti = types.index(cell.type) if cell.type in types else 1
                    new_type = st.selectbox("t", types, index=ti, key=f"typ_{cell.id}", label_visibility="collapsed")
                    if new_type != cell.type:
                        cell.type = new_type
                        save_notebook(nb)
                        st.rerun()

                height = 100 if cell.type == "markdown" else 150
                src = st.text_area("s", value=cell.source, height=height, key=f"src_{cell.id}", label_visibility="collapsed")
                if src != cell.source:
                    cell.source = src
                    save_notebook(nb)
                if cell.type == "markdown":
                    st.markdown(cell.source)
                for out in cell.outputs or []:
                    if out.get("stdout"):
                        st.code(out["stdout"], language="text")
                    if out.get("stderr"):
                        st.error(out["stderr"][:3000])
                    if out.get("exit_code") is not None:
                        st.caption(f"exit={out.get('exit_code')} · {out.get('duration_ms', 0)}ms")
                st.markdown("---")

    # ═══════════ 3) المهام ═══════════
    with lab_tabs[2]:
        st.markdown("### سجل مهام التدريب")
        jobs = list_jobs(40)
        if st.button("🔄 تحديث القائمة", key="jobs_refresh"):
            st.rerun()
        if not jobs:
            st.info("لا مهام مسجّلة بعد — أطلق تدريباً من «إطلاق سريع».")
        for j in jobs:
            ok = j.get("ok")
            icon = "✅" if ok else ("❌" if ok is False else "•")
            title = j.get("preset_key") or j.get("type") or "job"
            st.markdown(
                f"{icon} **{title}** · `{j.get('job_id') or '—'}` · "
                f"{(j.get('recorded_at') or '')[:19]}"
            )
            if j.get("kernel_url"):
                st.markdown(f"→ [{j['kernel_url']}]({j['kernel_url']})")
            cols = st.columns([1, 1, 3])
            with cols[0]:
                if j.get("job_id") and st.button("حالة", key=f"st_{j.get('job_id')}_{j.get('recorded_at')}"):
                    st.session_state["lab_job_status"] = refresh_job_status(j["job_id"])
            if j.get("error"):
                st.caption(str(j["error"])[:200])
            st.markdown("---")
        if st.session_state.get("lab_job_status"):
            st.json(st.session_state["lab_job_status"])

    # ═══════════ 4) البيئة ═══════════
    with lab_tabs[3]:
        st.markdown("### صحة البيئة")
        h = lab_health()
        checks = h.get("checks", {})
        kaggle_creds = checks.get("kaggle_creds") or {}
        local_gpu = checks.get("local_gpu") or {}
        api_keys = checks.get("api_keys") or {}

        e1, e2, e3 = st.columns(3)
        with e1:
            st.markdown(_badge(bool(kaggle_creds.get("ready")), "بيانات اعتماد Kaggle", "ناقصة"))
            st.markdown(_badge(bool(checks.get("kaggle_cli")), "Kaggle CLI", "غير مثبّت"))
        with e2:
            st.markdown(
                f"{'🟢' if local_gpu.get('cuda') else '🟡'} GPU محلي: "
                f"{local_gpu.get('name') or ('نعم' if local_gpu.get('cuda') else 'لا يوجد')}"
            )
            st.markdown(_badge(h.get("ready_to_launch_kaggle", False), "جاهز للإطلاق على Kaggle", "أضف Secrets"))
        with e3:
            ready_keys = [k for k, v in api_keys.items() if isinstance(v, dict) and v.get("ready")]
            st.markdown(f"🔑 مزوّدون بمفاتيح جاهزة: **{len(ready_keys)}** / {len(api_keys) or 0}")
            if ready_keys:
                st.caption(", ".join(ready_keys))

        with st.expander("📄 تفاصيل خام (JSON)"):
            st.json(h)

        st.markdown("### 📊 مراقبة GPU")
        if st.button("التقاط لقطة GPU الآن", key="gpu_snap_btn"):
            try:
                from ai.gpu_runtime import gpu_monitor_snapshot, device_report_md
                snap = gpu_monitor_snapshot()
                st.session_state["gpu_snap"] = snap
                st.markdown(device_report_md())
            except Exception as e:
                st.error(str(e))
        if st.session_state.get("gpu_snap"):
            snap = st.session_state["gpu_snap"]
            st.code(str(snap.get("nvidia_smi_csv") or "—"), language="text")
            st.json({k: snap[k] for k in snap if k != "tips_ar"})
            for tip in snap.get("tips_ar") or []:
                st.caption("• " + tip)
        with st.expander("detect_compute / خطة المزوّد"):
            nb = _ensure_nb()
            st.json(detect_compute())
            st.json(plan_remote_run(nb, st.session_state.get("nsm_nb_provider", "kaggle")))
        try:
            from ai.free_gpu_providers import recommended_stack_ar, list_free_gpu_providers
            st.markdown(recommended_stack_ar())
            for row in list_free_gpu_providers(include_paid=True)[:8]:
                st.markdown(
                    f"- **{row['name']}** · {row.get('gpu')} · "
                    f"[رابط]({row.get('signup_url')})"
                )
        except Exception as e:
            st.caption(str(e))

    # ═══════════ 5) الميزات ═══════════
    with lab_tabs[4]:
        st.markdown("### قدرات المختبر")
        for f in feature_list_ar():
            st.markdown(f"- ✅ {f}")
        st.markdown(
            """
### أفضل ممارسة
1. من **إطلاق سريع** اختر Medium (d=256) للتدريب الجاد  
2. تأكد من Streamlit Secrets + Kaggle `GITHUB_TOKEN`  
3. راقب Logs على Kaggle حتى `--- 2) التدريب ---`  
4. بعد النجاح راجع GitHub للـ checkpoints  

### ملاحظة
Streamlit = تحكم وتحرير. Kaggle/Modal = GPU الثقيل.
"""
        )

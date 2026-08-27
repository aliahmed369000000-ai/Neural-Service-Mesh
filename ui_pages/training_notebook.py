"""
ui_pages/training_notebook.py — مختبر تدريب احترافي (Colab/Kaggle style)
"""
from __future__ import annotations

import json
import re as _nre
import time

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


def nb_kernel_summary(nb_id: str):
    """🆕 معلومات مختصرة عن جلسة kernel الدفتر."""
    try:
        from ai.nb_kernel import session_summary
        return session_summary(nb_id)
    except Exception:
        return None


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


def _render_cell_outputs(outputs):
    """🆕 عرض مخرجات kernel بصيغة ipynb-native (stream/display/execute_result/
    error) مع بقاء التوافق مع البنية القديمة (stdout/stderr/exit_code)."""
    for out in outputs or []:
        otype = out.get("type", "")
        # البنية القديمة من subprocess
        if out.get("stdout"):
            st.code(out["stdout"], language="text")
        if out.get("stderr"):
            st.error(out["stderr"][:3000])
        if out.get("exit_code") is not None:
            st.caption(f"exit={out.get('exit_code')} · {out.get('duration_ms', 0)}ms")
        # البنية ipynb-native من kernel الحقيقي
        if otype == "stream":
            text = out.get("text", "")
            if text and not out.get("stdout"):
                st.code(text, language="text")
        elif otype == "display_data":
            data = out.get("data") or {}
            if "image/png" in data:
                import base64
                st.image(base64.b64decode(data["image/png"]))
            if "text/html" in data:
                st.markdown("".join(data["text/html"]), unsafe_allow_html=True)
            elif "text/plain" in data and "image/png" not in data and "text/html" not in data:
                st.text("".join(data["text/plain"]))
        elif otype == "execute_result":
            data = out.get("data") or {}
            if "image/png" in data:
                import base64
                st.image(base64.b64decode(data["image/png"]))
            elif "text/html" in data:
                st.markdown("".join(data["text/html"]), unsafe_allow_html=True)
            else:
                st.text("".join(data.get("text/plain") or [""]))
        elif otype == "error":
            st.error("\n".join(out.get("traceback") or [out.get("evalue", "")]))
        # 🆕 v4: الأنواع المتقدمة (SQL / HTTP / Markdown+)
        elif otype == "sql_result":
            for _res in out.get("results") or []:
                _cols = _res.get("cols") or []
                _rows = _res.get("rows") or []
                if _cols and _rows:
                    try:
                        import pandas as _pd
                        st.dataframe(_pd.DataFrame(_rows, columns=_cols),
                                     use_container_width=True)
                    except Exception:
                        st.code("\n".join(
                            " | ".join(str(_v) for _v in _r) for _r in _rows),
                            language="text")
                elif _cols:
                    st.caption("".join(_cols) + " — 0 صفوف")
                if _res.get("many"):
                    st.caption("⚠️ عُرضت أول 500 صف فقط")
            if out.get("error"):
                st.error(str(out["error"]))
            if out.get("duration_ms") is not None:
                st.caption(f"⏱️ {out.get('duration_ms')}ms")
        elif otype == "http_result":
            _st = out.get("status")
            _ok = isinstance(_st, int) and 200 <= _st < 300
            _icon = "✅" if _ok else ("⏳" if _st is None else "❌")
            st.markdown(f"{_icon} **HTTP {_st or '?'}** · "
                        f"{out.get('duration_ms', 0)}ms")
            if out.get("headers"):
                st.caption(" · ".join(
                    f"{str(_hk)}: {str(_hv)}" for _hk, _hv in
                    list(out["headers"].items())[:5]))
            if out.get("body"):
                st.code(out["body"][:3000], language="text")
            if out.get("error"):
                st.error(str(out["error"]))
        elif otype == "markdown_ex":
            _html = out.get("html", "")
            if _html:
                st.markdown(_html, unsafe_allow_html=True)
            if out.get("duration_ms") is not None:
                st.caption(f"⏱️ {out.get('duration_ms')}ms")


def _metrics_plot(nb):
    """🆕 v2: رسم مقاييس التدريب (loss/steps) المستخرجة من مخرجات خلايا الدفتر."""
    import re as _re
    loss_vals: List[float] = []
    steps: List[str] = []
    for c in nb.cells:
        for o in (c.outputs or []):
            txt = ""
            if o.get("stdout"):
                txt = o["stdout"]
            elif o.get("type") == "stream" and o.get("text"):
                txt = "".join(o["text"])
            for line in txt.splitlines():
                # loss: 0.1234 أو loss=0.1234 أو loss: 0.1234 أو epoch N loss X
                m = _re.search(r"loss\s*[=:)]\s*(\d+\.?\d*(?:e[+-]?\d+)?)", line, _re.I)
                if m:
                    try:
                        loss_vals.append(float(m.group(1)))
                        ep = _re.search(r"epoch\s*[=:)]?\s*(\d+)", line, _re.I)
                        steps.append(f"[{c.id[:4]}]" + (f"ep{ep.group(1)}" if ep else ""))
                    except ValueError:
                        continue
    if not loss_vals:
        st.info("لا مقاييس loss في مخرجات الخلايا بعد — نفّذ خلية تدريب أولًا")
        return
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(range(1, len(loss_vals) + 1), loss_vals, marker=".", color="#1f77b4")
        ax.set_xlabel("خطوة")
        ax.set_ylabel("loss")
        ax.set_title("منحنى فقدان التدريب (من مخرجات الدفتر)")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        st.caption(f"{len(loss_vals)} قيمة loss من مخرجات الخلايا")
    except Exception as e:
        st.code("\n".join(f"{s} {v}" for s, v in zip(steps[:50], loss_vals[:50])))
        st.caption(str(e)[:200])


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
        cell_version_list,
        create_notebook,
        delete_cell,
        delete_notebook,
        delete_scheduled_job,
        detect_compute,
        duplicate_notebook,
        export_ipynb,
        import_ipynb,
        import_shared_notebook,
        inject_secrets_into_kernel,
        insert_snippet,
        interrupt_cell,
        list_notebook_secrets,
        list_notebooks,
        list_scheduled_jobs,
        list_shared_notebooks,
        load_notebook,
        move_cell,
        plan_remote_run,
        run_all,
        run_cell,
        run_scheduled_jobs,
        save_cell_version,
        save_notebook,
        schedule_notebook,
        set_notebook_secret,
        undo_cell,
    )
    import os
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
            "🧬 المختبر العصبي",
            "🧿 مركز السيادة",
            "📋 المهام",
            "🖥️ البيئة",
            "✨ الميزات",
        ]
    )

    # ═══════════ 1) إطلاق سريع ═══════════
    with lab_tabs[0]:
        st.markdown("### ابدأ تدريباً على GPU سحابي (Kaggle)")
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

        with st.expander("⚙️ خيارات الإطلاق"):
            fresh = st.checkbox("من الصفر (FRESH)", value=True, key="lab_fresh")
            if fresh:
                st.caption("⚠️ سيبدأ التدريب من الصفر ويتجاوز أي checkpoint سابق محفوظ.")
            auto_push = st.checkbox("رفع تلقائي بعد النجاح (يتطلب GITHUB_TOKEN على المزود)", value=True, key="lab_autopush")

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
            # 🆕 استيراد دفتر .ipynb (Colab/Kaggle) إلى المختبر
            if st.button("⬆️ استيراد ipynb", use_container_width=True,
                         key="nb_import_ask"):
                st.session_state["nb_show_import"] = True
                st.rerun()
            if st.session_state.get("nb_show_import"):
                up = st.file_uploader(".ipynb",
                                      accept_multiple_files=False,
                                      type=["ipynb", "json"],
                                      key="nb_import_uploader",
                                      label_visibility="collapsed")
                if up is not None:
                    import tempfile as _tf
                    with _tf.NamedTemporaryFile(suffix=".ipynb", delete=False) as _f:
                        _f.write(up.read())
                        _tmp = _f.name
                    try:
                        nb = import_ipynb(_tmp, name=up.name.rsplit(".", 1)[0])
                        st.session_state.nsm_nb_id = nb.id
                        st.session_state.pop("nb_show_import", None)
                        st.toast("تم استيراد الدفتر", icon="✅")
                    except Exception as e:
                        st.error(f"فشل الاستيراد: {e}")
                    finally:
                        try:
                            os.unlink(_tmp)
                        except Exception:
                            pass
                    st.rerun()
                if st.button("إلغاء", use_container_width=True, key="nb_import_cancel"):
                    st.session_state.pop("nb_show_import", None)
                    st.rerun()
            # 🆕 v4: تصدير الدفتر كـ ipynb قياسي (يفتح في Colab/Jupyter)
            if st.button("⬇️ تصدير ipynb", use_container_width=True, key="nb_export_ask"):
                st.session_state["nb_show_export"] = True
                st.rerun()
            if st.session_state.get("nb_show_export"):
                from ai.notebook_engine import export_ipynb as _export
                ipynb_text = _export(nb)
                safe_name = re.sub(r"[^A-Za-z0-9_\u0600-\u06FF\-]", "_", nb.name)[:60] or "nsm_lab"
                st.download_button("⬇️ تنزيل {}.ipynb".format(safe_name),
                                   data=ipynb_text,
                                   file_name=f"{safe_name}.ipynb",
                                   mime="application/x-ipynb+json",
                                   use_container_width=True, key="nb_export_dl")
                if st.button("إغلاق", use_container_width=True, key="nb_export_close"):
                    st.session_state.pop("nb_show_export", None)
                    st.rerun()
            if st.button("⎘ استنساخ", use_container_width=True, key="nb_dup_ask"):
                dup = duplicate_notebook(nb)
                st.session_state.nsm_nb_id = dup.id
                st.toast(f"استنسخ الدفتر: {dup.name}", icon="✅")
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
                ["local", "kaggle", "modal", "lightning", "huggingface", "colab", "vast"],
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
                with st.spinner("تشغيل… الحالات تُحفظ بعد كل خلية"):
                    results = run_all(nb, timeout=int(timeout), stop_on_error=True)
                n_ok = sum(1 for r in results if r.get("status") == "ok")
                n_err = sum(1 for r in results if r.get("status") == "error")
                st.toast(f"✅ {n_ok} · ❌ {n_err} من {len(results)} خلايا", icon="✅")
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
        if st.button("➕ SQL 🗄️", use_container_width=True, key="nb_add_sql"):
            add_cell(nb, "sql",
                     "-- db demo.sqlite\n-- allow_writes\n"
                     "CREATE TABLE IF NOT EXISTS t (id INTEGER, v TEXT);\n"
                     "SELECT * FROM t;")
            st.rerun()
        if st.button("➕ HTTP 🌐", use_container_width=True, key="nb_add_http"):
            add_cell(nb, "http",
                     "GET https://api.github.com/rate_limit")
            st.rerun()
        if st.button("➕ MD+ 📋", use_container_width=True, key="nb_add_mdex"):
            add_cell(nb, "markdown_ex",
                     "### عنوان\n```mermaid\ngraph LR\n  A-->B\n```")
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

        # ── 🆕 v2: شريط أدوات موسّع (قوالب + أسرار + مشاركة + جدولة) ──
        tools_tabs = st.tabs(["📚 قوالب", "🔐 أسرار", "🤝 مشاركة", "⏰ جدولة", "📊 مقاييس", "⌨️ اختصارات"])
        # قوالب الكود الجاهزة (snippets)
        with tools_tabs[0]:
            from ai.notebook_engine import CODE_SNIPPETS
            snip_keys = list(CODE_SNIPPETS.keys())
            snip_labels = [f"{CODE_SNIPPETS[k]['label_ar']} ({k})" for k in snip_keys]
            if snip_labels:
                pick = st.selectbox("اختر قالبًا لإضافته للدفتر", snip_labels, key="nb_snip_pick")
                if pick:
                    k = snip_keys[snip_labels.index(pick)]
                    if st.button("➕ إدراج القالب في النهاية", use_container_width=True, key="nb_snip_insert"):
                        c = insert_snippet(nb, k)
                        st.toast(f"أُدرج القالب: {CODE_SNIPPETS[k]['label_ar']}", icon="📚")
                        st.rerun()
            # 🆕 اقتراح الخلية التالية (تكملة تلقائية/اقتراح سياقي)
            with st.expander("🪄 اقتراح الخلية التالية"):
                hist = [c.source.splitlines()[0] for c in nb.cells if c.source.strip()]
                if st.button("اقترح", use_container_width=True, key="nb_next_suggest"):
                    from ai.notebook_copilot import suggest_next
                    s = suggest_next("", hist)
                    st.markdown(f"💡 {s.get('text', '—')}")
                    if st.button("➕ أضف كخلية جديدة", use_container_width=True, key="nb_next_insert"):
                        add_cell(nb, "code", s.get("text", ""))
                        st.rerun()
        # أسرار الدفتر (متغيرات بيئة داخل kernel — لا تظهر في الكود)
        with tools_tabs[1]:
            st.caption("تُحقن كمتغيرات بيئة داخل kernel — لا تُعرض داخل المصادر")
            k1, k2 = st.columns([2, 1.4])
            with k1:
                sk = st.text_input("اسم المفتاح", placeholder="HF_TOKEN", key="nb_secret_key")
            with k2:
                sv = st.text_input("القيمة", type="password", placeholder="••••••", key="nb_secret_val")
            if sk and sv and st.button("💾 حفظ السر", key="nb_secret_save"):
                set_notebook_secret(nb, sk, sv)
                st.toast(f"حُفظ: {sk}", icon="🔐")
                st.rerun()
            keys_ = list_notebook_secrets(nb)
            if keys_:
                st.markdown("**الأسرار المحفوظة:** " + ", ".join(f"`{k}`" for k in keys_))
                if st.button("⚡ حقن في kernel", use_container_width=True, key="nb_secret_inject"):
                    n = inject_secrets_into_kernel(nb)
                    st.toast(f"حُقنت {n} سرًا في kernel", icon="⚡")
            # دفع الدفتر إلى Kaggle كـkernel
            with st.expander("🚀 دفع هذا الدفتر إلى Kaggle"):
                from ai.kaggle_provider import push_kaggle_kernel
                st.caption("يُصدّر الدفتر ويدفعه كـkernel Kaggle في مشروعك")
                if st.button("🚀 Push to Kaggle", use_container_width=True, key="nb_push_kaggle"):
                    with st.spinner("تصدير + دفع…"):
                        try:
                            from ai.notebook_engine import export_ipynb as _exp, ROOT as _ROOT
                            out = _exp(nb)
                            import tempfile as _tf
                            _p = _ROOT / "artifacts" / "model_training" / "notebooks" / f"{nb.id}.ipynb"
                            _p.parent.mkdir(parents=True, exist_ok=True)
                            _p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
                            res = push_kaggle_kernel(nb.id)
                            st.session_state["nb_push_res"] = res
                        except Exception as e:
                            st.session_state["nb_push_res"] = {"ok": False, "error": str(e)}
                    st.rerun()
                if st.session_state.get("nb_push_res"):
                    r = st.session_state["nb_push_res"]
                    if r.get("ok"):
                        st.success(f"دُفع: {r.get('kernel_url') or r.get('slug')}")
                    else:
                        st.error(str(r.get("error"))[:300])
        # مشاركة الدفاتر (مكتبة مشتركة بين الوكلاء والأعضاء)
        with tools_tabs[2]:
            sh1, sh2 = st.columns(2)
            with sh1:
                if st.button("📤 مشاركة هذا الدفتر", use_container_width=True, key="nb_share_this"):
                    from ai.notebook_engine import share_notebook as _share
                    res = _share(nb, "مشترك من مختبر NSM")
                    st.toast("تمت المشاركة", icon="🤝") if res.get("ok") else st.error(str(res))
                    st.rerun()
            with sh2:
                if st.button("🔄 تحديث المكتبة", use_container_width=True, key="nb_shared_refresh"):
                    st.session_state["nb_shared_list"] = list_shared_notebooks()
            for row in st.session_state.get("nb_shared_list") or list_shared_notebooks()[:10]:
                st.markdown(f"- **{row.get('name')}** ({row.get('cells')} خلية) — `{row.get('id')}`")
                sid = row.get("id")
                if sid and st.button("📥 استنساخ", key=f"nb_shared_imp_{sid}"):
                    c = import_shared_notebook(sid)
                    st.session_state.nsm_nb_id = c.id
                    st.toast("تم الاستنساخ", icon="📥")
                    st.rerun()
        # الجدولة (تشغيل دوري)
        with tools_tabs[3]:
            # فحص دوري خفيف للمهام المستحقة
            try:
                ran = run_scheduled_jobs()
            except Exception:
                ran = []
            if ran:
                for r in ran:
                    st.toast(f"⏰ مهمّة {r.get('name')}: {'✅' if r.get('ok') else '❌'}", icon="⏰")
            sch1, sch2, sch3, sch4 = st.columns([2, 2, 2, 1.5])
            with sch1:
                sch_at = st.text_input("وقت (HH:MM) أو now", value="now", key="nb_sch_at")
            with sch2:
                sch_int = st.number_input("تكرار كل (دقائق) — 0 = مرة واحدة", 0, 1440, 0, key="nb_sch_int")
            with sch3:
                sch_name = st.text_input("اسم المهمة", value=f"{nb.name}", key="nb_sch_name")
            with sch4:
                if st.button("⏰ جدولة", use_container_width=True, key="nb_sch_go"):
                    res = schedule_notebook(nb.id, sch_name, sch_at, int(sch_int))
                    st.toast("تمت الجدولة", icon="⏰") if res.get("ok") else st.error(str(res))
                    st.rerun()
            jobs_ = list_scheduled_jobs()
            if jobs_:
                for j in jobs_:
                    st.markdown(
                        f"- ⏰ **{j.get('name')}** · `{j.get('id')}` · "
                        f"التالي {j.get('next_run_at') or '—'} · "
                        f"دورية {j.get('interval_minutes')}د"
                    )
                    if st.button("🗑", key=f"nb_sch_del_{j.get('id')}"):
                        delete_scheduled_job(j["id"])
                        st.rerun()
            else:
                st.caption("لا مهام مجدولة")
        # مقاييس التدريب (استخراج loss/steps من مخرجات الخلايا)
        with tools_tabs[4]:
            st.caption("يُستخرج loss/خطوات من مخرجات الخلايا تلقائيًا ويُرسم")
            _metrics_plot(nb)
        # اختصارات لوحة المفاتيح
        with tools_tabs[5]:
            st.caption("💡 استخدم اختصارات المتصفح مع الأزرار: Shift+Enter داخل textarea لا يشغّل — اضغط ▶ يدويًا (Streamlit لا يدعم اختصارات مخصصة بالكامل)")
            st.markdown(
                "- **▶ تشغيل خلية**: زر ▶ بجوار كل خلية\n"
                "- **⏳ إيقاف**: زر ⏹ عند خلية جارية\n"
                "- **↺ تراجع**: زر ↺ بجوار كل خلية\n"
                "- **🤖 مساعد**: زر 🤖 بجوار كل خلية (شرح/إصلاح/تحسين)"
            )

        # 🆕 شريط حالة kernel (Colab/Kaggle style): ذاكرة مستمرة بين الخلايا
        try:
            from ai.notebook_engine import nb_kernel_health, restart_kernel_session
            from ai.nb_kernel import sessions_detail as _sd
            kbh = nb_kernel_health()
            _sessions = _sd() if kbh.get("backend") == "kernel" else []
        except Exception:
            kbh = {"ipykernel_available": False, "backend": "subprocess", "active_sessions": 0}
            _sessions = []
        kernel_ok = kbh.get("backend") == "kernel"
        sess = nb_kernel_summary(nb.id) if kernel_ok else None
        # 🆕 عرض جلسات kernel النشطة للأمانة
        if _sessions:
            for _s in _sessions:
                st.caption(
                    f"⚙️ جلسة kernel `{_s['session_id'][:8]}…`: "
                    f"{'حيّة 🟢' if _s['alive'] else 'ميتة 🔴'} · "
                    f"Uptime {(_s['uptime_s'] or 0) // 60}د {_s['uptime_s'] % 60}ث"
                )
        status_cols = st.columns([4, 1.4, 1.2])
        with status_cols[0]:
            st.markdown(
                f"**{nb.name}** · `{nb.id}` · خلايا **{len(nb.cells)}** · "
                f"حُفظ `{nb.updated_at[:19] if nb.updated_at else '—'}`"
            )
        with status_cols[1]:
            if kernel_ok:
                alive = sess.get("alive") if sess else False
                st.caption(
                    f"⚡ kernel حي: {'نعم' if alive else 'لا'} · "
                    f"ذاكرة مشتركة بين الخلايا {'🟢' if alive else '🟡'}"
                )
            else:
                st.caption("⚙️ وضع subprocess (ipykernel غير متوفر) — بدون ذاكرة مشتركة")
        with status_cols[2]:
            if kernel_ok and st.button("🔄 إعادة kernel", use_container_width=True, key="nb_restart_kernel"):
                with st.spinner("إعادة تشغيل kernel…"):
                    res = restart_kernel_session(nb.id)
                if res.get("ok"):
                    st.toast("ذاكرة kernel صُفّرت — مثل Reset في Colab", icon="✅")
                else:
                    st.toast(str(res.get("error")), icon="❌")
                st.rerun()

        # 🆕 v2: تشغيل دوري خفيف للمهام المجدولة (كل rerun للواجهة)
        try:
            _ran = run_scheduled_jobs()
        except Exception:
            _ran = []
        if _ran:
            for _r in _ran:
                st.toast(f"⏰ مهمة {_r.get('name')}: {'✅' if _r.get('ok') else '❌'}", icon="⏰")
            st.rerun()

        # 🆕 v2: live run — تشغيل الخلية في خيط جانبي (غير مسبب للواجهة) مع
        # تحديث مخرجاتها دوريًا (auto-refresh كل ~1.5s أثناء الجريان)
        def _nb_live_run(_cell_id: str) -> None:
            import threading as _t

            from ai.notebook_engine import run_cell_streaming
            st.session_state[f"nb_live_{_cell_id}"] = True
            def _w():
                try:
                    run_cell_streaming(nb, _cell_id, timeout=int(timeout))
                finally:
                    st.session_state.pop(f"nb_live_{_cell_id}", None)
            _t.Thread(target=_w, daemon=True).start()

        def _auto_refresh_nb() -> None:
            """🆕 polling دوري خفيف: أي خلية running → re-run تلقائي بعد مهلة قصيرة"""
            any_running = any(c.status == "running" for c in nb.cells)
            last = st.session_state.get("nb_last_poll", 0.0)
            if any_running and (time.time() - last) > 1.5:
                st.session_state.nb_last_poll = time.time()
                st.rerun()

        _auto_refresh_nb()

        # 🆕 v3: تشغيل كل الخلايا ترتيبًا (run-all) في خيط جانبي
        def _nb_run_all(stop_on_err: bool) -> None:
            import threading as _t

            from ai.notebook_engine import run_cell as _rc
            st.session_state.nb_runall_active = True
            def _w():
                try:
                    for c in list(nb.cells):
                        if c.type != "markdown":
                            # خلية ما زالت جارية من تنفيذ سابق أو من إيقاف يدوي
                            # — تجاهلها في run-all بدل إعادة تنفيذها
                            if c.status == "running":
                                continue
                            _rc(nb, c.id, timeout=int(timeout))
                            if stop_on_err and c.status == "error":
                                break
                finally:
                    st.session_state.pop("nb_runall_active", None)
            _t.Thread(target=_w, daemon=True).start()

        ac1, ac2 = st.columns(2)
        with ac1:
            if st.session_state.get("nb_runall_active"):
                st.button("🔄 تشغيل الكل…", disabled=True,
                          use_container_width=True, key="nb_runall_btn")
            elif st.button("▶️▶️ تشغيل الكل", use_container_width=True,
                           key="nb_runall_start", help="يشغّل كل الخلايا بالترتيب (يشمل train)"):
                _nb_run_all(stop_on_err=True)
        with ac2:
            if st.button("⏹ إيقاف أي خلية جارية", use_container_width=True,
                         key="nb_stop_any"):
                stopped = 0
                for c in nb.cells:
                    if c.status == "running":
                        from ai.notebook_engine import interrupt_cell as _ic2
                        _ic2(nb, c.id)
                        stopped += 1
                st.toast(f"أُرسل إيقاف لـ {stopped} خلية", icon="⏹")
                st.rerun()

        for i, cell in enumerate(list(nb.cells)):
            badge = {"markdown": "📝", "code": "🐍", "bash": "💻", "train": "🏋️",
                     "sql": "🗄️", "http": "🌐", "markdown_ex": "📋"}.get(cell.type, "•")
            status_icon = {"ok": "✅", "error": "❌", "running": "⏳", "idle": "⚪"}.get(cell.status, "⚪")
            with st.container():
                h1, h2, h3, h4, h5, h6, h7 = st.columns([1.5, 0.45, 0.45, 0.45, 0.45, 0.45, 0.6])
                _dur = None
                for _o in (cell.outputs or [])[::-1]:
                    if _o.get("duration_ms") is not None:
                        _dur = _o["duration_ms"]
                        break
                dur_text = f" · {(_dur / 1000):.1f}ث" if _dur else ""
                count_text = (f" · In[{cell.execution_count}]" if cell.execution_count else "")
                with h1:
                    st.markdown(
                        f"**[{i}] {badge} {cell.type}** {status_icon}{dur_text}{count_text} "
                        f"`{cell.id}`"
                    )
                with h2:
                    if cell.status == "running":
                        if st.button("⏹", key=f"int_{cell.id}"):
                            from ai.notebook_engine import interrupt_cell as _ic
                            with st.spinner("إيقاف…"):
                                res = _ic(nb, cell.id)
                            st.toast(
                                "أُوقفت الخلية" if res.get("ok") else str(res.get("error"))[:80],
                                icon="⏹" if res.get("ok") else "⚠️",
                            )
                            st.rerun()
                    elif st.button("▶", key=f"run_{cell.id}"):
                        with st.spinner("…"):
                            run_cell(nb, cell.id, timeout=int(timeout))
                        st.rerun()
                with h3:
                    # 🆕 v2: تشغيل حي في خيط جانبي (مخرجات تُحدّث خلال التنفيذ)
                    if st.button("⚡", key=f"live_{cell.id}", help="Live: مخرجات محدّثة أثناء التنفيذ"):
                        _nb_live_run(cell.id)
                with h4:
                    if st.button("↑", key=f"up_{cell.id}"):
                        move_cell(nb, cell.id, -1)
                        st.rerun()
                with h5:
                    if st.button("↓", key=f"dn_{cell.id}"):
                        move_cell(nb, cell.id, 1)
                        st.rerun()
                with h6:
                    # 🆕 v2: تراجع عن آخر تعديل (Undo)
                    if st.button("↺", key=f"undo_{cell.id}", help="تراجع عن آخر تعديل"):
                        res = undo_cell(nb, cell.id)
                        st.toast(
                            "تم التراجع" if res.get("ok") else str(res.get("error"))[:60],
                            icon="↺" if res.get("ok") else "⚠️",
                        )
                        st.rerun()
                with h7:
                    # 🆕 v2: المساعد الذكي للخلية (شرح/إصلاح/تحسين)
                    if st.button("🤖", key=f"cop_{cell.id}", help="مساعد: شرح / إصلاح / تحسين"):
                        st.session_state[f"nb_cop_open_{cell.id}"] = not st.session_state.get(f"nb_cop_open_{cell.id}")
                        st.rerun()
                    if st.session_state.get(f"nb_cop_open_{cell.id}"):
                        from ai.notebook_copilot import explain_cell as _explain, \
                            fix_cell as _fix, improve_cell as _improve
                        a1, a2, a3 = st.columns(3)
                        with a1:
                            if st.button("📖 شرح", use_container_width=True, key=f"cop_ex_{cell.id}"):
                                st.session_state[f"nb_cop_res_{cell.id}"] = {
                                    "n": "شرح", "r": _explain(cell.source)}
                        with a2:
                            if st.button("🛠 إصلاح", use_container_width=True, key=f"cop_fx_{cell.id}"):
                                last_err = ""
                                for _oe in (cell.outputs or [])[::-1]:
                                    if _oe.get("type") == "error":
                                        last_err = "\n".join(_oe.get("traceback") or [_oe.get("evalue", "")])[:1500]
                                        break
                                st.session_state[f"nb_cop_res_{cell.id}"] = {
                                    "n": "إصلاح", "r": _fix(cell.source, last_err)}
                        with a3:
                            if st.button("✨ تحسين", use_container_width=True, key=f"cop_im_{cell.id}"):
                                st.session_state[f"nb_cop_res_{cell.id}"] = {
                                    "n": "تحسين", "r": _improve(cell.source)}
                        cr = st.session_state.get(f"nb_cop_res_{cell.id}")
                        if cr:
                            st.markdown(f"**🤖 {cr['n']}:**")
                            st.markdown(str(cr["r"].get("text", "—")))
                types = ["markdown", "code", "bash", "train",
                         "sql", "http", "markdown_ex"]
                ti = types.index(cell.type) if cell.type in types else 1
                new_type = st.selectbox("t", types, index=ti, key=f"typ_{cell.id}", label_visibility="collapsed")
                if new_type != cell.type:
                    cell.type = new_type
                    save_notebook(nb)
                    st.rerun()
                # 🆕 v2: حفظ نسخة تلقائيًا قبل التعديل (Undo history)
                src = st.text_area("s", value=cell.source, height=150,
                                   key=f"src_{cell.id}", label_visibility="collapsed")
                if src != cell.source:
                    save_cell_version(nb, cell.id, note="تعديل من الواجهة")
                    cell.source = src
                    save_notebook(nb)
                # 🆕 v3: شريط تقدم داخل الخلية الجارية + استخراج loss/epoch لحظي
                if cell.status == "running":
                    _live_last = ""
                    for _lo in (cell.outputs or [])[::-1]:
                        if _lo.get("type") == "stream" and _lo.get("text"):
                            _live_last = "".join(_lo["text"])[-200:]
                            break
                    if not _live_last:
                        for _lo in (cell.outputs or [])[::-1]:
                            if _lo.get("stdout"):
                                _live_last = _lo["stdout"][-200:]
                                break
                    _lm = _nre.search(
                        r"(?:epoch|ep)\s*[=:)]?\s*(\d+).{0,20}?loss\s*[=:]?\s*([\d.e+-]+)",
                        _live_last, _nre.I)
                    if not _lm:
                        _lm = _nre.search(
                            r"loss\s*[=:)\s]\s*([\d.e+-]+)\s*(?:\S*\s*){0,2}?\b(?:epoch|ep)\b\s*[=:)]?\s*(\d+)",
                            _live_last, _nre.I)
                    if not _lm:
                        _lm = _nre.search(
                            r"loss\s*[=:)\s]\s*([\d.e+-]+)(?:[^\d]|$)",
                            _live_last, _nre.I)
                    if _lm:
                        _a, _b = _lm.group(1), _lm.group(2)
                        if _a is not None and _b is not None:
                            _ep, _loss = ((_a, _b) if _a.isdigit() else (_b, _a))
                        else:
                            _ep, _loss = _b, _a  # loss-only
                        st.progress(0.5,
                                    text=f"⏳ epoch {_ep or '…'} · loss {_loss or '…'} — راقب ⏹ للإيقاف")
                    else:
                        st.progress(0.5, text="⏳ الخلية تجري — راقب ⏹ للإيقاف")

                # (يُنفّذ أعلاه — src محفوظ + نسخة تاريخ)
                if cell.type == "markdown":
                    st.markdown(cell.source)
                else:
                    _render_cell_outputs(cell.outputs)
                st.markdown("---")

    # ═══════════ 3) المختبر العصبي ═══════════
    with lab_tabs[2]:
        st.markdown("### 🧬 المختبر العصبي (Neural Lab)")
        st.caption("بيئة تجريبية سريعة لمقارنة أداء محركات NumPy و TensorFlow على شبكة Surah 4096.")
        
        nb = _ensure_nb()
        
        c1, c2 = st.columns(2)
        with c1:
            engine_choice = st.radio("المحرك النشط", ["NumPy (الأصلي)", "TensorFlow (المسرع)"], horizontal=True, key="lab_engine_choice")
        with c2:
            st.metric("حالة الذاكرة", "مستقرة" if health.get("ready_to_launch_kaggle") else "محدودة")

        st.markdown("#### 🛠️ أدوات المختبر")
        tool_cols = st.columns(3)
        with tool_cols[0]:
            if st.button("📊 قياس سرعة التوليد", use_container_width=True):
                code = "from ai.benchmark_tf import run_comparison\nrun_comparison()"
                add_cell(nb.id, "code", code)
                st.rerun()
        with tool_cols[1]:
            if st.button("📽️ اختبار ميزات الفيديو", use_container_width=True):
                code = "import numpy as np\nfrom ai.arabic_transformer_tf import ArabicTransformerTF\nm = ArabicTransformerTF(d_model=64)\nv = np.random.rand(1, 16, 512).astype(np.float32)\nout = m._forward([1, 2, 3], video_feats=v)\nprint('Video fusion success:', out[0].shape)"
                add_cell(nb.id, "code", code)
                st.rerun()
        with tool_cols[2]:
            if st.button("🧪 فحص الـ KV Cache", use_container_width=True):
                code = "from tests.test_surah_multimodal_kv_cache import test_kv_cache_parity\ntest_kv_cache_parity()"
                add_cell(nb.id, "code", code)
                st.rerun()

    # ═══════════ 4) مركز السيادة ═══════════
    with lab_tabs[3]:
        st.markdown("### 🧿 مركز سيادة الوكيل (Agent Sovereignty)")
        st.caption("سجل أبحاث وتطوير الوكلاء الذاتي — توثيق المسيرة المعرفية.")
        
        nb = _ensure_nb()
        
        # 📊 مراقبة الموارد الحية
        st.markdown("#### 📉 مراقبة الموارد الحية")
        res_col1, res_col2, res_col3 = st.columns(3)
        try:
            import psutil
            cpu_usage = psutil.cpu_percent()
            ram_usage = psutil.virtual_memory().percent
            res_col1.metric("استهلاك المعالج", f"{cpu_usage}%")
            res_col2.metric("استهلاك الذاكرة", f"{ram_usage}%")
            res_col3.metric("الخلايا المنفذة", len([c for c in nb.cells if c.status == 'ok']))
        except Exception:
            st.caption("أدوات مراقبة النظام غير متوفرة في هذه البيئة.")

        # عرض سجل الأبحاث
        st.markdown("#### 📝 سجل أبحاث الوكيل")
        if hasattr(nb, 'agent_research_log') and nb.agent_research_log:
            for note in reversed(nb.agent_research_log):
                with st.expander(f"📌 {note['author']} · {note['timestamp'][:16]}", expanded=False):
                    st.markdown(note['content'])
                    if note.get('tags'):
                        st.caption("الوسوم: " + " · ".join(note['tags']))
        else:
            st.info("لا توجد ملاحظات بحثية بعد. الوكلاء سيقومون بتوثيق تجاربهم هنا.")

        # إضافة ملاحظة يدوية
        with st.expander("➕ إضافة ملاحظة بحثية جديدة"):
            with st.form("new_research_note"):
                author = st.text_input("الكاتب", value="NSM Bot")
                content = st.text_area("المحتوى (يدعم Markdown)")
                tags = st.text_input("الوسوم (مفصولة بفاصلة)")
                if st.form_submit_button("حفظ الملاحظة"):
                    from ai.notebook_engine import add_research_note
                    tag_list = [t.strip() for t in tags.split(",")] if tags else []
                    if add_research_note(nb.id, author, content, tag_list):
                        st.success("تم حفظ الملاحظة البحثية بنجاح.")
                        st.rerun()

    # ═══════════ 5) المهام ═══════════
    with lab_tabs[4]:
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

"""
ui_pages/training_notebook.py — مختبر تدريب بخلايا (Colab/Kaggle style)
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


def render_training_notebook():
    """دفتر تدريب تفاعلي — خلايا + مزوّدو GPU."""
    st.markdown(
        '<div class="section-header">📓 مختبر التدريب (Notebook)</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "خلايا مثل **Colab / Kaggle** · حفظ دائم على القرص · تنفيذ محلي · "
        "خطط إرسال لـ Kaggle / Colab / RunPod / Vast عبر المزوّدات الموجودة في المشروع"
    )

    from ai.notebook_engine import (
        add_cell,
        create_notebook,
        delete_cell,
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

    # ── شريط علوي: اختيار دفتر + مزوّد ──
    notebooks = list_notebooks()
    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    with c1:
        options = {f"{n['name']} ({n['id']})": n["id"] for n in notebooks} or {}
        if options:
            labels = list(options.keys())
            cur = st.session_state.get("nsm_nb_id")
            idx = 0
            for i, (lab, oid) in enumerate(options.items()):
                if oid == cur:
                    idx = i
                    break
            choice = st.selectbox("الدفتر", labels, index=idx, key="nsm_nb_select")
            st.session_state.nsm_nb_id = options[choice]
        else:
            st.info("لا دفاتر بعد")
    with c2:
        if st.button("➕ دفتر جديد", use_container_width=True, key="nb_new"):
            nb = create_notebook("NSM Lab " + st.session_state.get("nsm_nb_id", "")[:4], template="training")
            st.session_state.nsm_nb_id = nb.id
            st.rerun()
    with c3:
        provider = st.selectbox(
            "المزوّد",
            [
                "local",
                "kaggle",
                "modal",
                "lightning",
                "huggingface",
                "colab",
                "runpod",
                "vast",
                "generic_gpu",
            ],
            key="nsm_nb_provider",
        )
    with c4:
        timeout = st.number_input("timeout (ث)", 30, 600, 120, key="nsm_nb_timeout")

    nb = _ensure_nb()
    nb.provider = provider
    save_notebook(nb)

    # ── حالة الحوسبة ──
    with st.expander("🖥️ الحوسبة ومزوّدو GPU المجاني", expanded=True):
        try:
            from ai.free_gpu_providers import (
                list_free_gpu_providers,
                provider_env_status,
                recommended_stack_ar,
                plan_for_provider,
            )
            st.markdown(recommended_stack_ar())
            st.markdown("#### حالة مفاتيح API في البيئة")
            st.json(provider_env_status())
            st.markdown("#### الكتالوج")
            for row in list_free_gpu_providers(include_paid=True):
                keys = ", ".join(row.get("api_key_env") or ["—"])
                st.markdown(
                    f"- **{row['name']}** (`{row['id']}`) · {row.get('gpu')} · "
                    f"حصة: {row.get('quota_ar')} · مفاتيح: `{keys}` · "
                    f"[تسجيل]({row.get('signup_url')})"
                )
        except Exception as e:
            st.warning(f"كتالوج GPU: {e}")
        compute = detect_compute()
        with st.expander("تفاصيل detect_compute", expanded=False):
            st.json(compute)
        plan = plan_remote_run(nb, provider)
        st.markdown("**خطة التشغيل على المزوّد المختار**")
        st.json(plan)
        if provider != "local":
            st.info(
                "ضع المفاتيح في **Streamlit Secrets**. "
                "التدريب الطويل: Kaggle أو Modal/Lightning أفضل من Colab المجاني (أقل انقطاعاً)."
            )

    # ── أدوات الدفتر ──
    t1, t2, t3, t4, t5 = st.columns(5)
    with t1:
        if st.button("▶️ Run All", use_container_width=True, key="nb_run_all"):
            with st.spinner("تشغيل كل الخلايا…"):
                results = run_all(nb, timeout=int(timeout), stop_on_error=True)
            st.toast(f"انتهى: {results}", icon="✅")
            st.rerun()
    with t2:
        if st.button("➕ Code", use_container_width=True, key="nb_add_code"):
            add_cell(nb, "code", "# cell\nprint('ok')")
            st.rerun()
    with t3:
        if st.button("➕ Markdown", use_container_width=True, key="nb_add_md"):
            add_cell(nb, "markdown", "### عنوان")
            st.rerun()
    with t4:
        if st.button("➕ Bash", use_container_width=True, key="nb_add_bash"):
            add_cell(nb, "bash", "ls -la")
            st.rerun()
    with t5:
        if st.button("➕ Train", use_container_width=True, key="nb_add_train"):
            add_cell(nb, "train", "print('train cell')\n# epochs, model, data here")
            st.rerun()

    st.markdown(f"**{nb.name}** · `{nb.id}` · خلايا: **{len(nb.cells)}** · حُفظ: `{nb.updated_at[:19]}`")

    # ── الخلايا ──
    for i, cell in enumerate(list(nb.cells)):
        badge = {
            "markdown": "📝 MD",
            "code": "🐍 PY",
            "bash": "💻 SH",
            "train": "🏋️ TRAIN",
        }.get(cell.type, cell.type)
        status_icon = {"ok": "✅", "error": "❌", "running": "⏳", "idle": "⚪"}.get(cell.status, "⚪")

        with st.container():
            h1, h2, h3, h4, h5, h6 = st.columns([1.2, 0.5, 0.5, 0.5, 0.5, 0.7])
            with h1:
                st.markdown(f"**[{i}] {badge}** {status_icon} `{cell.id}`")
            with h2:
                if st.button("▶", key=f"run_{cell.id}", help="تشغيل"):
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
                new_type = st.selectbox(
                    "type",
                    ["markdown", "code", "bash", "train"],
                    index=["markdown", "code", "bash", "train"].index(cell.type)
                    if cell.type in ("markdown", "code", "bash", "train")
                    else 1,
                    key=f"typ_{cell.id}",
                    label_visibility="collapsed",
                )
                if new_type != cell.type:
                    cell.type = new_type
                    save_notebook(nb)
                    st.rerun()

            if cell.type == "markdown":
                src = st.text_area(
                    "md",
                    value=cell.source,
                    height=100,
                    key=f"src_{cell.id}",
                    label_visibility="collapsed",
                )
                if src != cell.source:
                    cell.source = src
                    save_notebook(nb)
                st.markdown(cell.source)
            else:
                src = st.text_area(
                    "src",
                    value=cell.source,
                    height=160,
                    key=f"src_{cell.id}",
                    label_visibility="collapsed",
                )
                if src != cell.source:
                    cell.source = src
                    save_notebook(nb)

            for out in cell.outputs or []:
                if out.get("type") == "markdown":
                    continue
                meta = f"exit={out.get('exit_code')} · {out.get('duration_ms', 0)}ms"
                if out.get("stdout"):
                    st.code(out["stdout"], language="text")
                if out.get("stderr"):
                    st.error(out["stderr"][:3000])
                st.caption(meta)

            st.markdown("---")

    # ── تصدير ──
    with st.expander("📦 تصدير Jupyter (.ipynb) / JSON"):
        ipynb = export_ipynb(nb)
        st.download_button(
            "تحميل .ipynb",
            data=json.dumps(ipynb, ensure_ascii=False, indent=2),
            file_name=f"{nb.name.replace(' ', '_')}.ipynb",
            mime="application/json",
            key="dl_ipynb",
        )
        st.download_button(
            "تحميل JSON خام",
            data=json.dumps(nb.to_dict(), ensure_ascii=False, indent=2),
            file_name=f"{nb.id}.json",
            mime="application/json",
            key="dl_json",
        )
        st.caption("يمكن رفع الـ ipynb إلى Colab أو Kaggle يدوياً، أو ربطه لاحقاً بـ remote_gpu_provider.")

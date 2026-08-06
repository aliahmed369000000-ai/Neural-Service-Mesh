"""
لوحة مراقبة موحّدة: مصنع + Registry + موافقات + صادرات CKG + AIaaS مختصر.
"""
from __future__ import annotations

import json
from pathlib import Path

from app_core import *  # noqa: F401,F403

ROOT = Path(__file__).resolve().parent.parent


def render_training_ops_dashboard():
    st.markdown("### 📡 مركز عمليات التدريب")
    st.caption("تشغيلات المصنع · السجل · الموافقات · صادرات CKG · ملخص AIaaS")

    tab_run, tab_reg, tab_apr, tab_ckg, tab_aiaas = st.tabs(
        ["تشغيل هدف", "Registry", "موافقات", "CKG", "AIaaS"]
    )

    with tab_run:
        goal = st.text_input(
            "هدف عام",
            value="هدف: حسّن تصنيف كيانات CKG بدقة أعلى من 70%",
            key="ops_goal",
        )
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🏭 تشغيل المصنع", type="primary"):
                with st.spinner("المصنع يعمل…"):
                    try:
                        from ai.training_factory import run_factory
                        st.markdown(run_factory(goal))
                    except Exception as e:
                        st.error(str(e))
        with c2:
            if st.button("📊 حالة المصنع"):
                try:
                    from ai.training_factory import factory_status
                    st.markdown(factory_status())
                except Exception as e:
                    st.error(str(e))

        runs_dir = ROOT / "artifacts" / "model_training" / "factory" / "runs"
        if runs_dir.is_dir():
            files = sorted(runs_dir.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:15]
            st.markdown("#### آخر التشغيلات")
            for p in files:
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                    with st.expander(f"{d.get('run_id')} — {d.get('status')} — {d.get('metric_value')}"):
                        st.json({
                            "goal": (d.get("goal") or {}).get("raw_text"),
                            "domain": (d.get("goal") or {}).get("domain_hint"),
                            "metric": f"{d.get('metric_name')}={d.get('metric_value')}",
                            "dataset": d.get("dataset"),
                            "steps": len(d.get("steps") or []),
                        })
                except Exception:
                    st.text(p.name)

    with tab_reg:
        try:
            from ai.training_feedback_loop import registry_report, load_registry
            st.markdown(registry_report())
            st.json(load_registry())
        except Exception as e:
            st.warning(str(e))

    with tab_apr:
        try:
            from ai.training_factory import list_approvals, resolve_approval
            st.markdown(list_approvals("pending"))
            aid = st.text_input("معرّف الموافقة apr_…", key="ops_apr")
            a1, a2 = st.columns(2)
            with a1:
                if st.button("وافق") and aid:
                    st.markdown(resolve_approval(aid.strip(), True))
            with a2:
                if st.button("ارفض") and aid:
                    st.markdown(resolve_approval(aid.strip(), False))
        except Exception as e:
            st.warning(str(e))

    with tab_ckg:
        try:
            from ai.ckg_training_export import export_status, export_entity_type_csv, export_root_category_csv
            st.markdown(export_status())
            if st.button("تصدير كيانات CKG الآن"):
                path, meta = export_entity_type_csv()
                st.success(path)
                st.json(meta)
            if st.button("تصدير جذور عربية"):
                path, meta = export_root_category_csv()
                st.success(path)
                st.json(meta)
        except Exception as e:
            st.warning(str(e))

    with tab_aiaas:
        try:
            from ai.aiaas_platform import platform_status
            st.markdown(platform_status())
        except Exception as e:
            st.warning(str(e))

"""
لوحة AIaaS — واجهة عملاء لتدريب النماذج كخدمة (معزولة حسب المستأجر).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403


def render_aiaas_console():
    st.markdown("### ☁️ منصة التدريب كخدمة (AIaaS)")
    st.caption(
        "رفع بيانات، تشغيل مهام معزولة لكل مستأجر، خطط اشتراك وقياس استهلاك. "
        "الفوترة تقديرية — ليست بوابة دفع."
    )

    try:
        from ai.aiaas_platform import (
            PLANS,
            DOMAINS,
            create_tenant,
            estimate_invoice,
            list_domains,
            platform_status,
            run_tenant_job,
            get_tenant,
            load_tenants_index,
            propose_self_evolution,
        )
    except Exception as e:
        st.error(f"تعذّر تحميل منصة AIaaS: {e}")
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        ["المستأجرون", "تشغيل مهمة", "الفوترة", "التطوير الذاتي"]
    )

    with tab1:
        st.markdown(platform_status())
        with st.form("new_tenant_form"):
            name = st.text_input("اسم المستأجر", value="demo")
            plan = st.selectbox("الخطة", list(PLANS.keys()), index=0)
            submitted = st.form_submit_button("إنشاء مستأجر")
            if submitted:
                rec = create_tenant(name=name, plan=plan)
                st.success(f"تم الإنشاء: `{rec['id']}`")
                st.warning(f"API Key (مرة واحدة): `{rec['api_key']}` — احفظه الآن")
                st.json({k: v for k, v in rec.items() if k != "api_key_hash"})

    with tab2:
        idx = load_tenants_index().get("tenants") or {}
        ids = list(idx.keys())
        if not ids:
            st.info("أنشئ مستأجراً أولاً من التبويب السابق.")
        else:
            tid = st.selectbox("المستأجر", ids)
            domain = st.selectbox(
                "المجال",
                [k for k, v in DOMAINS.items() if v.get("status") != "planned"],
            )
            epochs = st.slider("عدد الحقب", 5, int(PLANS.get((idx[tid] or {}).get("plan") or "free", PLANS["free"])["max_epochs"]), 15)
            if st.button("🚀 تشغيل مهمة تدريب", type="primary"):
                with st.spinner("جاري التدريب في بيئة المستأجر…"):
                    job = run_tenant_job(tid, domain=domain, epochs=epochs)
                if job.get("ok") is False or job.get("error"):
                    st.error(job.get("error") or "فشل")
                else:
                    st.success(f"اكتملت المهمة `{job.get('job_id')}` خلال {job.get('elapsed_s')}s")
                st.code((job.get("result_preview") or "")[:3000] or str(job), language=None)

    with tab3:
        idx = load_tenants_index().get("tenants") or {}
        ids = list(idx.keys())
        if ids:
            tid = st.selectbox("مستأجر للفاتورة", ids, key="inv_tid")
            st.json(estimate_invoice(tid))
            st.markdown("#### الخطط")
            st.json(PLANS)
        else:
            st.info("لا مستأجرين.")

    with tab4:
        if st.button("🧬 توليد اقتراح تطوير ذاتي"):
            with st.spinner("تحليل + بحث محكوم…"):
                st.markdown(propose_self_evolution())
        st.caption("التطوير الذاتي يسجّل اقتراحات فقط ولا يعدّل كود الإنتاج دون موافقة/تنفيذ بشري.")

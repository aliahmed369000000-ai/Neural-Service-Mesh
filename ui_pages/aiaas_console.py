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
        idx = load_tenants_index().get("tenants") or {}
        _m1, _m2, _m3 = st.columns(3)
        _m1.metric("👥 المستأجرون", len(idx))
        _m2.metric("📋 الخطط المتاحة", len(PLANS))
        _m3.metric("🧩 المجالات", len(DOMAINS))

        if idx:
            st.markdown("#### المستأجرون الحاليون")
            _rows = []
            for _tid, _rec in list(idx.items())[:50]:
                _u = _rec.get("usage") or {}
                _rows.append({
                    "المعرّف": _tid,
                    "الاسم": _rec.get("name", "—"),
                    "الخطة": (PLANS.get(_rec.get("plan") or "free") or {}).get("name", _rec.get("plan")),
                    "مهام اليوم": _u.get("jobs_today", 0),
                    "النماذج": _u.get("models_total", 0),
                })
            st.dataframe(_rows, use_container_width=True, hide_index=True)
        else:
            st.info("لا يوجد مستأجرون بعد — أنشئ واحداً من النموذج أدناه.")

        st.markdown("---")
        with st.form("new_tenant_form"):
            name = st.text_input("اسم المستأجر", value="demo")
            plan = st.selectbox("الخطة", list(PLANS.keys()), index=0,
                                 format_func=lambda k: f"{PLANS[k]['name']} — ${PLANS[k]['price_usd_month']}/شهر")
            submitted = st.form_submit_button("إنشاء مستأجر", type="primary")
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
            inv = estimate_invoice(tid)
            _i1, _i2, _i3, _i4 = st.columns(4)
            _i1.metric("💳 الاشتراك", f"${inv['subscription_usd']}/شهر")
            _i2.metric("⏱️ دقائق التدريب", f"{inv['train_minutes']:.0f}")
            _i3.metric("➕ تجاوز الحصة", f"${inv['overage_usd']}")
            _i4.metric("💰 الإجمالي التقديري", f"${inv['estimated_total_usd']}")
            st.caption(inv.get("note", ""))

            st.markdown("#### مقارنة الخطط")
            _plan_rows = [
                {
                    "الخطة": p["name"],
                    "السعر/شهر": f"${p['price_usd_month']}",
                    "مهام/يوم": p["max_jobs_per_day"],
                    "أقصى نماذج": p["max_models"],
                    "رفع (MB)": p["max_upload_mb"],
                    "أقصى حقب": p["max_epochs"],
                    "مهام متزامنة": p["concurrent_jobs"],
                }
                for p in PLANS.values()
            ]
            st.dataframe(_plan_rows, use_container_width=True, hide_index=True)
        else:
            st.info("لا مستأجرين.")

    with tab4:
        if st.button("🧬 توليد اقتراح تطوير ذاتي"):
            with st.spinner("تحليل + بحث محكوم…"):
                st.markdown(propose_self_evolution())
        st.caption("التطوير الذاتي يسجّل اقتراحات فقط ولا يعدّل كود الإنتاج دون موافقة/تنفيذ بشري.")

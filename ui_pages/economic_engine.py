"""
المحرك الاقتصادي — واجهة Streamlit للقنوات الأربع
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403


def render_economic_engine():
    st.markdown("### 💰 المحرك الاقتصادي")
    st.caption(
        "قنوات تشغيل تجريبية داخل NSM: AIaaS · سوق النماذج · مضاربة حوسبة · بيانات اصطناعية. "
        "الأرقام دفترية — ليست بوابة دفع ولا ضمان عوائد."
    )

    try:
        from ai.commercial_economy import (
            dashboard,
            publish_model,
            list_marketplace,
            sell_license,
            compute_arbitrage_quote,
            book_arbitrage_demo,
            SPOT_TABLE,
            price_synthetic_batch,
            sell_synthetic_demo,
            read_ledger,
            ledger_summary,
        )
    except Exception as e:
        st.error(f"تعذّر تحميل المحرك الاقتصادي: {e}")
        return

    try:
        from ai.aiaas_platform import PLANS, load_tenants_index, create_tenant, estimate_invoice
    except Exception:
        PLANS, load_tenants_index, create_tenant, estimate_invoice = {}, None, None, None

    dash = dashboard()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("إيراد دفتري إجمالي", f"${dash['ledger'].get('total_usd', 0):.2f}")
    c2.metric("عمليات", dash["ledger"].get("n_transactions", 0))
    c3.metric("نماذج في السوق", dash.get("marketplace_items", 0))
    c4.metric("MRR تقديري AIaaS", f"${(dash.get('aiaas') or {}).get('estimated_mrr_usd', 0):.0f}")

    t_dash, t_aiaas, t_market, t_compute, t_data = st.tabs(
        ["📊 اللوحة", "☁️ AIaaS", "🛒 سوق النماذج", "⚡ حوسبة", "🧪 بيانات"]
    )

    with t_dash:
        st.markdown("#### ملخص القنوات")
        by_ch = (dash.get("ledger") or {}).get("by_channel") or {}
        if by_ch:
            st.bar_chart({"usd": by_ch})
        else:
            st.info("لا حركات بعد — نفّذ عملية من التبويبات الأخرى.")
        st.markdown("#### آخر العمليات")
        rows = read_ledger(30)
        if rows:
            st.dataframe(
                [
                    {
                        "الوقت": str(r.get("at", ""))[:19],
                        "القناة": r.get("channel"),
                        "المبلغ $": r.get("amount_usd"),
                        "الوصف": (r.get("description") or "")[:60],
                    }
                    for r in reversed(rows)
                ],
                use_container_width=True,
                hide_index=True,
            )
        st.json(dash)

    with t_aiaas:
        st.markdown("#### الذكاء الاصطناعي كخدمة (اشتراكات)")
        st.caption("إدارة المستأجرين والخطط — نفس محرك `ai.aiaas_platform`.")
        if load_tenants_index is None:
            st.warning("وحدة AIaaS غير متاحة")
        else:
            idx = (load_tenants_index() or {}).get("tenants") or {}
            st.write(f"المستأجرون: **{len(idx)}** · الخطط: {', '.join(PLANS.keys()) if PLANS else '—'}")
            if idx:
                st.dataframe(
                    [
                        {
                            "id": tid,
                            "اسم": rec.get("name"),
                            "خطة": rec.get("plan"),
                            "مهام اليوم": (rec.get("usage") or {}).get("jobs_today", 0),
                        }
                        for tid, rec in list(idx.items())[:40]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            with st.form("eco_new_tenant"):
                name = st.text_input("اسم مستأجر", "startup_demo")
                plan = st.selectbox("خطة", list(PLANS.keys()) if PLANS else ["free"])
                if st.form_submit_button("إنشاء مستأجر", type="primary") and create_tenant:
                    rec = create_tenant(name=name, plan=plan)
                    st.success(f"تم: `{rec.get('id')}` — خطة {plan}")
                    st.rerun()
            if idx and estimate_invoice:
                tid = st.selectbox("فاتورة تقديرية لـ", list(idx.keys()))
                if st.button("احسب الفاتورة"):
                    st.json(estimate_invoice(tid))
        st.info("للواجهة التفصيلية (رفع بيانات + تشغيل مهمة) استخدم تبويب ☁️ AIaaS الرئيسي.")

    with t_market:
        st.markdown("#### سوق النماذج المعدَّلة (ترخيص API)")
        with st.form("publish_form"):
            n = st.text_input("اسم النموذج", "NSM-Arabic-Classifier")
            dom = st.text_input("المجال", "nlp_ar")
            price = st.number_input("السعر USD", min_value=0.0, value=49.0, step=1.0)
            lic = st.selectbox("نوع الترخيص", ["api", "weights", "saas_embed"])
            notes = st.text_input("ملاحظات", "نموذج تجريبي من منظومة NSM")
            if st.form_submit_button("نشر في الكتالوج", type="primary"):
                item = publish_model(n, dom, float(price), lic, notes)
                st.success(f"نُشر `{item['id']}`")
                st.rerun()
        items = list_marketplace()
        if items:
            st.dataframe(
                [
                    {
                        "id": it.get("id"),
                        "الاسم": it.get("name"),
                        "المجال": it.get("domain"),
                        "السعر": it.get("price_usd"),
                        "مبيعات": it.get("sales", 0),
                    }
                    for it in items
                ],
                use_container_width=True,
                hide_index=True,
            )
            sell_id = st.selectbox("بيع ترخيص تجريبي", [it["id"] for it in items])
            if st.button("تسجيل عملية بيع دفتري"):
                r = sell_license(sell_id)
                if r.get("ok"):
                    st.success(f"+${r.get('amount_usd')} في الدفتر")
                    st.rerun()
                else:
                    st.error(r.get("error"))
        else:
            st.info("الكتالوج فارغ — انشر نموذجاً أولاً.")

    with t_compute:
        st.markdown("#### مضاربة / هامش الحوسبة (Spot)")
        st.caption("يحسب فرق السعر بين Spot وإعادة البيع بسعر السوق — دون حجز سحابي فعلي.")
        prov = st.selectbox("المزوّد", list(SPOT_TABLE.keys()))
        hours = st.slider("ساعات", 1.0, 72.0, 10.0, 1.0)
        q = compute_arbitrage_quote(prov, hours)
        m1, m2, m3 = st.columns(3)
        m1.metric("تكلفة Spot", f"${q['cost_usd']:.2f}")
        m2.metric("إيراد تقديري", f"${q['revenue_usd']:.2f}")
        m3.metric("الهامش", f"${q['margin_usd']:.2f}", f"{q['margin_pct']}%")
        st.json(q)
        if st.button("تسجيل هامش دفتري", type="primary"):
            r = book_arbitrage_demo(prov, hours)
            if r.get("ok"):
                st.success("سُجّل في دفتر الإيرادات")
                st.rerun()
            else:
                st.warning(r.get("reason"))

    with t_data:
        st.markdown("#### تسعير وبيع بيانات اصطناعية")
        n = st.number_input("عدد العينات", min_value=100, value=1000, step=100)
        quality = st.selectbox("الجودة", ["standard", "curated", "domain_expert"])
        domain = st.selectbox("المجال", ["general", "medical", "finance", "nlp_ar"])
        pricing = price_synthetic_batch(int(n), quality, domain)
        st.metric("السعر التقديري", f"${pricing['price_usd']:.2f}")
        st.json(pricing)
        if st.button("توليد دفعة + تسجيل بيع دفتري", type="primary"):
            r = sell_synthetic_demo(int(n), quality, domain)
            st.success(f"تم — path=`{r.get('batch_path')}` · ${r['pricing']['price_usd']}")
            st.rerun()


def render_aiaas_economy_hub():
    """تجميع AIaaS + المحرك الاقتصادي في تبويب واحد."""
    sub = st.tabs(["☁️ منصة AIaaS", "💰 المحرك الاقتصادي"])
    with sub[0]:
        from ui_pages.aiaas_console import render_aiaas_console

        render_aiaas_console()
    with sub[1]:
        render_economic_engine()

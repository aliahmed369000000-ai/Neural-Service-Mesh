"""
ui_pages/scheduler_hub.py — لوحة المجدول متعدد الحسابات (24/7 مجانية)
======================================================================
تدير تدريب SurahChain المستمر عبر:
  - حتى 7 حسابات Kaggle (30 ساعة GPU لكل حساب أسبوعيًا = 210 ساعة/أسبوع)
  - فحص الكوتا تلقائيًا قبل كل دفع
  - مزودي Colab وLightning AI كفشلوفر
"""
from __future__ import annotations

import json

import streamlit as st


def _badge(ok: bool) -> str:
    return "🟢" if ok else "🔴"


def render_scheduler_hub():
    from ai import multi_account_scheduler as MAS
    from ai import free_providers as FP

    st.markdown("##### 🌐 المجدول متعدد الحسابات — تدريب 24/7 مجاني")
    st.caption(
        "يدور على حتى 7 حسابات Kaggle مع فحص الكوتا قبل كل تشغيل. "
        "عند نفاد كل الكوتا: ينتقل تلقائيًا إلى Colab المجاني ثم Lightning AI (22 ساعة L4/شهر)."
    )

    tab = st.tabs(["📊 الحالة والكوتا", "⚙️ إطلاق مهمة", "🔔 التنبيهات", "👥 الحسابات", "🔧 المزودات المجانية"])

    with tab[0]:
        _tab_status(MAS)

    with tab[1]:
        _tab_launch(MAS)

    with tab[2]:
        _tab_alerts()

    with tab[3]:
        _tab_accounts(MAS)

    with tab[4]:
        _tab_free_providers(FP, MAS)


def _tab_status(MAS):
    """شريط الحالة والكوتا والحسابات النشطة."""
    try:
        report = MAS.scheduler_report()
    except Exception as e:
        st.error(f"فشل تحميل التقرير: {e}")
        return

    # الكوتا لكل حساب
    st.markdown("###### كوتا الحسابات (يُعاد الفحص عند كل تحديث)")
    cols = st.columns(min(len(report["accounts"]) or 1, 4))
    for i, q in enumerate(report["accounts"]):
        with cols[i % len(cols)]:
            remaining = q.get("gpu_remaining_hours") or 0.0
            total = q.get("quota", {}).get("GPU", {}).get("total", 0.0) or 30.0
            st.metric(
                f"👤 {q.get('username')}",
                f"{remaining:.1f}h",
                delta=f"من {total:.0f}h",
                help="ساعات GPU المتبقية هذا الأسبوع",
            )
            flag = _badge(q.get("ok") and not q.get("gpu_exhausted"))
            st.caption(f"{flag} جاهز للتشغيل" if not q.get("gpu_exhausted") else f"{flag} الكوتا شبه فارغة")

    # المهام النشطة
    st.markdown("###### المهام النشطة")
    active = report.get("active_jobs") or []
    if not active:
        st.info("لا توجد مهام نشطة حاليًا — اضغط «أطلق دورة تدريب» أدناه.")
    else:
        for j in active:
            st.markdown(
                f"- **{j.get('job_id')}** — الحالة: `{j.get('status')}` · "
                f"الحساب: `{j.get('account')}` · بدأت: {(j.get('started_at') or '')[:19]}\n"
                f"  - الرابط: [{j.get('kernel_url') or '—'}]({j.get('kernel_url') or '#'})"
            )

    # السجل الأخير
    hist = report.get("history") or []
    if hist:
        with st.expander(f"📜 آخر الأحداث ({len(hist)})", expanded=False):
            for ev in hist[-10:]:
                st.caption(f"• [{(ev.get('at') or '')[:19]}] {ev.get('event')} — {ev.get('job_id', '')} @{ev.get('account', '')}")


def _tab_launch(MAS):
    """إطلاق دورة تدريب (tick) مع المعلمات."""
    st.markdown("###### إعداد المهمة")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        preset = st.selectbox("preset", ["small", "medium", "large"], index=1)
    with c2:
        n = st.number_input("N (حجم الشبكة)", min_value=1000, max_value=500000, value=60000, step=10000)
    with c3:
        epochs = st.number_input("epochs", min_value=1, max_value=300, value=30)
    with c4:
        batch = st.number_input("batch", min_value=4, max_value=128, value=24)

    c = st.columns(2)
    with c[0]:
        if st.button("▶️ أطلق دورة تدريب (يعمل عبر كل الحسابات)", use_container_width=True):
            with st.spinner("يفحص الكوتا ويطلق المهمة..."):
                try:
                    res = MAS.scheduler_tick(preset=preset, n=int(n), epochs=int(epochs), batch=int(batch))
                    st.session_state["sched_tick"] = res
                except Exception as e:
                    st.error(f"فشل: {e}")
                    st.rerun()
            res = st.session_state.get("sched_tick")
            if res:
                if res.get("action") == "job_launched":
                    st.success(f"تم الإطلاق! المهمة: `{res.get('job_id')}` على حساب `{res.get('account')}`")
                    if res.get("kernel_url"):
                        st.markdown(f"🔗 [{res.get('kernel_url')}]({res.get('kernel_url')})")
                elif res.get("action") == "all_accounts_exhausted":
                    st.warning("نفدت كوتا كل الحسابات — راجع تبويب «المزودات المجانية» للفشلوفر.")
                else:
                    st.warning(f"الإجراء: {res.get('action')}")
                    if res.get("error"):
                        st.caption(f"التفاصيل: {str(res.get('error'))[:400]}")
    with c[1]:
        if st.button("🔄 حدّث التقرير", use_container_width=True):
            st.rerun()


def _tab_alerts():
    """تبويب التنبيهات الذكية: سجل التنبيهات + الملخص + اختبار Discord."""
    try:
        from ai import training_alerts as TA
    except ImportError:
        st.warning("training_alerts غير متوفر — شغّل التطبيق من جذر المستودع")
        return

    try:
        summary = TA.alerts_summary()
    except Exception as e:
        st.error(f"فشل تحميل الملخص: {e}")
        return

    st.markdown("###### ملخص التنبيهات")
    c = st.columns(4)
    metrics = [
        ("📨 إجمالي", str(summary["total"])),
        ("🟢 info", str(summary["by_severity"].get("info", 0))),
        ("🟡 warning", str(summary["by_severity"].get("warning", 0))),
        ("🔴 critical", str(summary["by_severity"].get("critical", 0))),
    ]
    for col, (label, val) in zip(c, metrics):
        col.metric(label, val)

    disc_ok = summary.get("discord_enabled")
    st.markdown(
        f"**حالة Discord:** {_badge(disc_ok)} {'مفعّل (DISCORD_BOT_TOKEN مضبوط)' if disc_ok else 'غير مفعّل — اضبط Secrets'}\n\n"
        "التنبيهات تُرسَل تلقائيًا عند اكتمال/فشل مهمة تدريب أو اقتراب نفاد كوتا GPU المجانية. "
        "الإرسال من داخل kernels Kaggle (بيئتها غير محجوبة)."
    )

    cc = st.columns(2)
    with cc[0]:
        if st.button("🧪 جرّب تنبيه Discord الآن", use_container_width=True):
            with st.spinner("يرسل اختبارًا..."):
                res = TA.record_alert(
                    "ui_test", "info", "اختبار من لوحة المجدول",
                    "هذه رسالة اختبار من تبويب التنبيهات — إذا وصلت لقناة Discord فكل شيء يعمل",
                    subject="ui_test",
                )
                st.session_state["alert_test"] = res
            res = st.session_state.get("alert_test")
            if res and res.get("discord", {}).get("ok"):
                st.success("✅ وصل تنبيه Discord!")
            elif res:
                d = res.get("discord") or {}
                st.warning(
                    f"سُجّل محليًا لكن Discord فشل: {(d.get('error') or d.get('reason') or '—')[:200]}\n\n"
                    "ملاحظة: إرسال Discord من بيئة الاستضافة قد يكون محجوبًا — "
                    "التنبيهات الحرجة تُرسَل من داخل kernels نفسها وتصل فعليًا."
                )
    with cc[1]:
        if st.button("🔄 تحديث", use_container_width=True):
            st.rerun()

    st.markdown("###### سجل التنبيهات الأخير")
    try:
        alerts = TA.list_alerts(15)
    except Exception as e:
        st.error(f"فشل: {e}")
        alerts = []
    if not alerts:
        st.info("لا توجد تنبيهات بعد — أول تنبيه يصل عند اكتمال/فشل مهمة تدريب.")
    else:
        for a in alerts:
            sev = a.get("severity", "info")
            icon = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(sev, "📢")
            sent = "✅ Discord" if (a.get("discord") or {}).get("ok") else "⏸ محلي فقط"
            st.markdown(
                f"- {icon} **[{a.get('kind')}]** {a.get('title')} — {(a.get('at') or '')[:19]} · {sent}\n"
                f"  - {str(a.get('message', ''))[:200]}"
            )


def _tab_accounts(MAS):
    """إدارة قائمة الحسابات (7 حسابات Kaggle)."""
    st.markdown("###### الحسابات المسجلة")
    st.caption(
        "أضف الحسابات عبر Streamlit Secrets: مفتاح `NSM_KAGGLE_ACCOUNTS_JSON` يحتوي قائمة JSON "
        "`[{\"username\": \"...\", \"key\": \"...\", \"note\": \"...\"}]` — أو الملف المحلي "
        "`artifacts/model_training/scheduler/kaggle_accounts.json`"
    )
    accs = MAS.load_accounts()
    if not accs:
        st.info("لا توجد حسابات مسجلة.")
    else:
        for i, a in enumerate(accs):
            hidden_key = ("*" * 4) + (a["key"][-4:] if len(a.get("key", "")) > 4 else "") if a.get("key") else "—"
            st.markdown(f"**{i + 1}.** `{a.get('username')}` — مفتاح: `{hidden_key}` — ملاحظة: {a.get('note', '—')}")

    if st.button("🔄 إعادة تحميل القائمة", use_container_width=True):
        st.rerun()


def _tab_free_providers(FP, MAS):
    """المزودات المجانية: Colab + Lightning."""
    st.markdown("###### المزودات المجانية البديلة")
    try:
        status = FP.free_providers_status()
    except Exception as e:
        st.error(f"فشل التحميل: {e}")
        return

    # Colab
    colab = status.get("colab", {})
    st.markdown(
        f"**1️⃣ {colab.get('name')}** — {colab.get('cost')} · {colab.get('accelerator')} · {colab.get('quota')}\n\n"
        f"مهام جاهزة: {colab.get('jobs_count', 0)}\n\n"
        f"{colab.get('setup_needed')}"
    )
    c = st.columns(2)
    with c[0]:
        if st.button("📓 ولّد Notebook Colab جاهز", use_container_width=True):
            from ai.free_providers import colab_generate_notebook
            import uuid as _uuid
            try:
                res = colab_generate_notebook(f"colab_{_uuid.uuid4().hex[:8]}")
                st.session_state["colab_job"] = res
            except Exception as e:
                st.error(f"فشل: {e}")
                st.rerun()
            res = st.session_state.get("colab_job")
            if res and res.get("ok"):
                st.success("تم توليد notebook — افتحه وانقله إلى Colab (Runtime → T4 GPU)")
                st.code(json.dumps({
                    "notebook": res.get("notebook"),
                    "colab_open_url": res.get("colab_open_url"),
                }, indent=2, ensure_ascii=False), language="json")
                if res.get("colab_open_url"):
                    st.markdown(f"🔗 [افتح في Colab]({res['colab_open_url']})")

    # Lightning
    st.markdown("---")
    ln = status.get("lightning", {})
    creds = ln.get("credentials", {})
    st.markdown(
        f"**2️⃣ {ln.get('name')}** — {ln.get('cost')}\n\n"
        f"حالة الاعتمادات: {_badge(creds.get('ready'))} {'جاهز' if creds.get('ready') else creds.get('hint')}\n\n"
        f"مهام: {ln.get('jobs_count', 0)}"
    )
    with c[1]:
        if st.button("⚡ تحقق من رصيد Lightning", use_container_width=True):
            from ai.free_providers import lightning_check_balance
            try:
                bal = lightning_check_balance()
                st.session_state["lightning_balance"] = bal
            except Exception as e:
                st.error(f"فشل: {e}")
                st.rerun()
            bal = st.session_state.get("lightning_balance")
            if bal and bal.get("ok"):
                st.success(f"المستخدم: {bal.get('user')} — الخطة: {bal.get('plan')} — الرصيد: {bal.get('balance')}")
            elif bal:
                st.warning(str(bal.get("error") or bal.get("data"))[:300])

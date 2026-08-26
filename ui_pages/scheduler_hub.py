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
import re

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

    tab = st.tabs(["📊 الحالة والكوتا", "⚙️ إطلاق مهمة", "🔔 التنبيهات", "📡 Live Logs", "🎯 مركز القيادة", "👥 الحسابات", "🔧 المزودات المجانية"])

    with tab[0]:
        _tab_status(MAS)

    with tab[1]:
        _tab_launch(MAS)

    with tab[2]:
        _tab_alerts()

    with tab[3]:
        _tab_live_logs()

    with tab[4]:
        _tab_command_center()

    with tab[5]:
        _tab_accounts(MAS)

    with tab[6]:
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

    # سجل التبديل بين الحسابات (Checkpoint Handoff)
    handoffs = report.get("handoffs") or []
    last_ckpt = report.get("last_checkpoint")
    st.markdown("###### 🔗 التبديل بين الحسابات (Checkpoint Handoff)")
    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            "آخر نقطة تفتيش على GitHub",
            str(last_ckpt.get("job_id", "—") or "—"),
            delta=str(last_ckpt.get("at", "")[:16] or "لا نقل بعد"),
            help="آخر checkpoint رُفعت إلى GitHub قبل التبديل بين الحسابات (SCN_RESUME=auto يستأنفها تلقائيًا)",
        )
    with c2:
        st.metric("إجمالي عمليات النقل", str(len(handoffs)))
    if not handoffs:
        st.info("لا توجد عمليات نقل بين الحسابات بعد — تُنفَّذ تلقائيًا عند نفاد كوتا الحساب أو انتهاء المهمة.")
    else:
        for h in handoffs[-10:]:
            status = h.get("status") or ""
            icon = {"success": "✅", "failed": "❌", "warning": "⚠️"}.get(status, "🔄")
            reason = h.get("reason") or ("نفاد كوتا" if h.get("exhausted") else "اكتمال المهمة")
            st.markdown(
                f"- {icon} **{h.get('job_id')}**: `@{h.get('from_account')}` → `@{h.get('to_account')}` — "
                f"{(h.get('at') or '')[:19]} · {reason}"
            )
            if h.get("pause_url"):
                st.caption(f"  - kernel أُوقف: [{h['pause_url']}]({h['pause_url']})")

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


def _tab_command_center():
    """🎯 مركز القيادة الموحّد: كل kernels على كل المزودين، مع رسم loss حيّ وأزرار التحكم."""
    from ai import kaggle_provider as KP

    st.markdown("##### 🎯 مركز القيادة — كل kernels على كل المزودين")
    st.caption(
        "لوحة موحّدة حية: kernels التدريب وتجميع البيانات عبر كل حسابات Kaggle "
        "(والحساب الحالي تلقائيًا). تنبيه أحمر عند أي kernel فاشل."
    )

    auto = st.toggle("⚡ تحديث تلقائي كل 60 ثانية", value=True, key="sched_hub_auto_refresh_cmd_center")

    c_btn = st.columns(3)
    with c_btn[0]:
        if st.button("🔄 تحديث الآن", use_container_width=True):
            st.session_state.pop("cmd_center", None)
            st.rerun()
    with c_btn[1]:
        if st.button("📜 عرض كل الأنواع (يشمل other)", use_container_width=True):
            st.session_state["cmd_show_all"] = not st.session_state.get("cmd_show_all")
    with c_btn[2]:
        if st.button("🧹 إخفاء المكتمل", use_container_width=True):
            st.session_state["cmd_hide_done"] = not st.session_state.get("cmd_hide_done")

    try:
        overview = st.session_state.get("cmd_center") or KP.unified_kernel_overview()
        st.session_state["cmd_center"] = overview
    except Exception as e:
        st.error(f"فشل جلب kernels: {e}")
        return

    kernels = overview.get("kernels") or []
    show_all = st.session_state.get("cmd_show_all")
    hide_done = st.session_state.get("cmd_hide_done")
    kinds_filter = {"training", "tally", "other"} if show_all else {"training", "tally"}
    kernels = [k for k in kernels if k.get("kind", "other") in kinds_filter]
    if hide_done:
        kernels = [k for k in kernels if not re.match(r"^(complete|canceled|cancelled|error)$", (k.get("state") or "").lower())]

    if not kernels:
        st.info("لا توجد kernels تدريب/تجميع حاليًا.")
        return

    # شريط إحصاءات
    total = len(kernels)
    running = sum(1 for k in kernels if "RUN" in (k.get("state") or "").upper())
    failed = sum(1 for k in kernels if "ERROR" in (k.get("state") or "").upper() or "FAIL" in (k.get("state") or "").upper())
    comp = sum(1 for k in kernels if re.match(r"^(complete|canceled|cancelled)$", (k.get("state") or "").lower()))
    st.markdown(f"**{total} kernels** — 🟢 {running} جارية · 🔴 {failed} فاشلة · ✅ {comp} مكتملة/متوقفة")
    if failed:
        st.error("⚠️ يوجد kernels فاشلة — راجع الجدول أدناه (اللون الأحمر).")

    # جدول kernels
    rows = []
    for k in kernels:
        state = (k.get("state") or "").upper()
        icon = ("🟢" if "RUN" in state or "QUEU" in state
                else ("🔴" if "ERROR" in state or "FAIL" in state
                      else ("✅" if "COMP" in state else "🟡")))
        rows.append({
            "الحالة": f"{icon} {k.get('state', '—')[:28]}",
            "النوع": "تدريب" if k.get("kind") == "training" else "تجميع" if k.get("kind") == "tally" else "أخرى",
            "الحساب": k.get("username") or k.get("account") or "—",
            "آخر تشغيل": (k.get("last_run") or "")[11:19] if k.get("last_run") else "—",
            "الرابط": k.get("slug") or "—",
        })
    st.dataframe(rows, use_container_width=True, hide_index=True,
                 column_config={"الرابط": st.column_config.LinkColumn("kernel", width="medium")})

    # تفاعل: تفاصيل + control لـkernel محدد
    st.markdown("###### تفاصيل kernel محدد")
    slug_choice = st.selectbox(
        "اختر kernel",
        options=[k.get("slug") for k in kernels],
        format_func=lambda s: next((f"{k.get('state','—')[:20]} — {k.get('title','')[:50]} ({s})" for k in kernels if k.get("slug") == s), s),
    )

    if slug_choice:
        c_det, c_ctrl = st.columns([3, 2])
        with c_det:
            if st.button("📊 عرض التقدم ومنحنى loss", use_container_width=True):
                with st.spinner("يسحب kernel output..."):
                    try:
                        snap = KP.kernel_progress_snap(slug_choice)
                        st.session_state["cmd_snap"] = snap
                    except Exception as e:
                        st.error(f"فشل السحب: {e}")
            snap = st.session_state.get("cmd_snap")
            if snap and snap.get("ok") and snap.get("progress"):
                prog = snap["progress"]
                if isinstance(prog, dict):
                    cm = st.columns(3)
                    cm[0].metric("العصر", f"{prog.get('epoch','?')} / {prog.get('end_epoch','?')}")
                    loss = prog.get("loss")
                    cm[1].metric("Loss", f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss))
                    best = prog.get("best_loss")
                    cm[2].metric("Best Loss", f"{best:.4f}" if isinstance(best, (int, float)) else str(best))
                    # منحنى loss عبر epochs
                    series = prog.get("loss_series") or prog.get("history") or []
                    if series:
                        try:
                            import pandas as pd
                            df = pd.DataFrame(series)
                            if "loss" in df.columns and "epoch" in df.columns:
                                st.line_chart(df.set_index("epoch")["loss"], x_label="epoch", y_label="loss")
                        except Exception:
                            st.caption("لا يمكن رسم المنحنى (بيانات غير منظمة).")
                st.caption(f"المصدر: {snap.get('source')} — {snap.get('file')}")
            elif snap and not snap.get("ok"):
                st.info("لا يوجد ملف تقدم (progress.json) بعد — kernel في البداية أو لم تُرفع النقاط.")
        with c_ctrl:
            st.markdown("**التحكم**")
            if st.button("⏹ إيقاف kernel", use_container_width=True, type="secondary"):
                try:
                    res = KP.stop_surahchain_kernel(slug_choice)
                    st.success("تم إرسال طلب الإيقاف" if (res or {}).get("ok") else f"النتيجة: {res}")
                except Exception as e:
                    st.error(f"فشل الإيقاف: {e}")
            if st.button("📜 آخر السجلات", use_container_width=True):
                try:
                    res = KP.kernel_logs_raw(slug_choice) if hasattr(KP, "kernel_logs_raw") else {"ok": False, "error": "غير متوفر"}
                    if res.get("ok"):
                        st.code((res.get("logs") or "")[-3000:], language="text")
                    else:
                        st.warning(f"فشل: {(res.get('error') or '')[:200]}")
                except Exception as e:
                    st.error(f"فشل: {e}")

    if auto:
        st.empty()


# ════ مساعدات kernel control ═══════════════════════════════════════════════════

def _stop_kernel_by_slug(slug: str) -> dict:
    """إيقاف kernel محدد مباشرة (تستخدمه مركز القيادة)."""
    from ai import kaggle_provider as KP
    if hasattr(KP, "stop_surahchain_kernel"):
        return KP.stop_surahchain_kernel(slug)
    import subprocess
    try:
        proc = subprocess.run(["kaggle", "kernels", "status", slug], capture_output=True, text=True, timeout=60)
        return {"ok": proc.returncode == 0, "output": proc.stdout[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


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


def _tab_live_logs():
    """NSM Live Logs: عرض حيّ لحالة تدريب SurahChain جارٍ — يجمع kernel status + logs + progress."""
    from ai import kaggle_provider as KP

    st.markdown("##### 📡 المخرجات الحيّة للتدريب")
    st.caption(
        "تجميع لحظي: حالة الـkernel + آخر السجلات + ملف التقدم (progress.json يُكتب كل عصر "
        "بعد اكتمال التدريب). يُحدّث تلقائيًا كل 60 ثانية."
    )

    auto = st.toggle("⚡ تحديث تلقائي كل 60 ثانية", value=True, key="sched_hub_auto_refresh_live_logs")
    if auto:
        import time as _time
        st.markdown(f"_آخر تحديث: {_time.strftime('%H:%M:%S')}_")

    # قائمة jobs المتاحة
    jobs = KP.list_kaggle_jobs() if hasattr(KP, "list_kaggle_jobs") else []
    scn_jobs = [j for j in jobs if isinstance(j, dict) and j.get("type") in ("surahchain", "training")]
    if not scn_jobs:
        st.info("لا توجد مهمة تدريب SurahChain — ادفع مهمة من تبويب ⚙️ إطلاق مهمة أولًا")
        return
    job_id = st.selectbox(
        "اختر المهمة", options=[j.get("job_id") for j in scn_jobs],
        format_func=lambda jid: next((j.get("title") or jid for j in scn_jobs if j.get("job_id") == jid), jid),
    )
    if not job_id:
        return

    res = KP.live_training_status(job_id)
    col1, col2 = st.columns(2)
    with col1:
        state = (res.get("kernel_state") or "—").upper()
        if "RUN" in state:
            st.success(f"الـkernel: {state} 🟢")
        elif "COMP" in state:
            st.success(f"الـkernel: {state} ✅")
        elif "ERROR" in state or "FAIL" in state:
            st.error(f"الـkernel: {state} ❌")
        else:
            st.warning(f"الـkernel: {state or 'غير معروف'} 🟡")
    with col2:
        prog = res.get("progress")
        if prog:
            ep, end = prog.get("epoch"), prog.get("end_epoch")
            loss = prog.get("loss")
            best = prog.get("best_loss")
            st.metric("العصر", f"{ep} / {end}")
            st.metric("Loss (آخر)", f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss))
            st.metric("Best Loss", f"{best:.4f}" if isinstance(best, (int, float)) else str(best))
        else:
            st.info("لم تُرْفَع progress.json بعد — أول عصر لم يكتمل أو التدريب لم يبدأ")

    with st.expander("📜 آخر سجلات الـkernel (آخر 200 سطر)", expanded=True):
        logs = res.get("kernel_logs") or "لا سجلات بعد — kernel في الطابور أو logs buffering"
        st.code(logs[-6000:], language="text")

    with st.expander("🧬 تفاصيل raw"):
        st.json({k: v for k, v in res.items() if k not in ("kernel_logs",)})

    if auto:
        st.empty()


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

"""
ui_pages/social_agent.py
تم تفكيكه تلقائياً من streamlit_app.py الأصلي (تقسيم الكود لتحسين القابلية للصيانة والأداء).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — إعادة تصدير كل الاستيرادات والدوال المساعدة المشتركة



# ══════════════════════════════════════════════════════════════════════════
# تبويب وكلاء AI — صفحة مستقلة لكل فئة/تخصص
def render_social_agent():
    """يدير الوكيل الاجتماعي الموحّد (ai/social_agent.py): تشغيل/إيقاف
    الاستطلاع التلقائي، اختيار المنصات المفعّلة وكلمات المراقبة، النشر
    اليدوي الفوري، وعرض آخر الأحداث/الأخطاء لكل منصة."""
    st.markdown('<div class="section-header">📡 الوكيل الاجتماعي</div>', unsafe_allow_html=True)
    st.caption(
        "نشر + رد تلقائي + مراقبة عبر Discord وTelegram وInstagram "
        "وFacebook وYouTube وTikTok وReddit وThreads وWhatsApp، "
        "ونشر فقط عبر Pinterest (لا يوفّر API مراقبة/رد — راجع تلميح المنصة)، "
        "بنفس شخصية NSM الموحّدة — مع جدولة منشورات وتحليل مشاعر وردود تتذكّر كل شخص."
    )

    try:
        from ai.social_agent import (
            get_manager, get_config, set_config, get_recent_events,
            schedule_post, get_scheduled, cancel_scheduled, get_analytics_summary,
        )
        from ai.social_platforms import PLATFORM_LABELS_AR, PLATFORM_CHAR_LIMITS
    except Exception as _sa_err:
        st.error(f"⚠️ تعذّر تحميل وحدة الوكيل الاجتماعي: {_sa_err}")
        return

    mgr = get_manager()
    status = mgr.status()

    # ── شريط الحالة العلوي (ثابت خارج التبويبات — أهم معلومة تبقى مرئية دائماً) ──
    col_state, col_action = st.columns([2, 1])
    running = mgr.is_running()
    with col_state:
        n_ready = sum(1 for s in status.values() if s.configured)
        st.markdown(
            f"**حالة الخدمة:** {'🟢 تعمل' if running else '⚪ متوقفة'} "
            f"· {n_ready}/{len(status)} منصة مُهيّأة"
        )
    with col_action:
        if running:
            if st.button("⏹️ إيقاف", key="social_stop", use_container_width=True):
                with st.spinner("⟳ ..."):
                    mgr.stop()
                st.rerun()
        else:
            if st.button("▶️ تشغيل", key="social_start", use_container_width=True):
                with st.spinner("⟳ ..."):
                    mgr.start()
                st.rerun()

    tab_settings, tab_status, tab_publish, tab_insights = st.tabs(
        ["⚙️ الإعدادات", "📊 حالة المنصات", "✍️ نشر وجدولة", "📈 تحليلات وأحداث"]
    )

    # ═══════════════════════════════ ⚙️ الإعدادات ═══════════════════════════════
    with tab_settings:
        st.markdown("#### إعدادات المراقبة")
        selected = st.multiselect(
            "المنصات المفعّلة",
            options=list(PLATFORM_LABELS_AR.keys()),
            default=list(set(get_config("enabled_platforms", []))),
            format_func=lambda p: PLATFORM_LABELS_AR.get(p, p),
            key="social_enabled_platforms",
        )
        keywords_str = st.text_input(
            "كلمات مفتاحية للمراقبة (مفصولة بفاصلة، اتركه فارغاً لمراقبة كل شيء)",
            value=", ".join(get_config("keywords", [])),
            key="social_keywords",
        )
        auto_reply = st.checkbox(
            "🤖 رد تلقائي على الإشارات المطابقة",
            value=get_config("auto_reply", False), key="social_auto_reply",
        )
        poll_interval = st.slider(
            "فترة الاستطلاع (ثانية)", 30, 600,
            int(get_config("poll_interval", 90)), 10, key="social_poll_interval",
        )
        if st.button("💾 حفظ الإعدادات", key="social_save_settings", type="primary"):
            with st.spinner("⟳ يحفظ..."):
                set_config("enabled_platforms", selected)
                set_config("keywords", [k.strip() for k in keywords_str.split(",") if k.strip()])
                set_config("auto_reply", auto_reply)
                set_config("poll_interval", poll_interval)
            st.success("✅ تم الحفظ.")
            st.rerun()

        st.markdown("---")
        st.markdown("#### ⚡ Telegram: Webhook مقابل Polling")
        st.caption(
            "Webhook يدفع الرسائل فوراً بدل الاستطلاع الدوري، لكنه يتطلب "
            "endpoint HTTPS عام ثابت (خادم api_server.py، منفصل عن Streamlit) "
            "ومتغيرَي بيئة: TELEGRAM_WEBHOOK_BASE_URL وTELEGRAM_WEBHOOK_SECRET. "
            "بدونهما يبقى النظام يعمل بـpolling كالمعتاد — لا كسر لأي سلوك حالي."
        )
        webhook_platforms_cfg = set(get_config("webhook_enabled_platforms", []))
        tg_webhook_on = "telegram" in webhook_platforms_cfg
        base_url = os.environ.get("TELEGRAM_WEBHOOK_BASE_URL", "")
        tg_secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
        col_tg1, col_tg2 = st.columns([2, 1])
        with col_tg1:
            st.markdown(f"**الوضع الحالي:** {'🔗 Webhook' if tg_webhook_on else '🔁 Polling'}")
            if base_url and tg_secret:
                st.caption("عنوان الـwebhook (يُضبط تلقائياً عند الضغط على تفعيل):")
                st.code(f"{base_url.rstrip('/')}/webhook/telegram/{tg_secret}", language=None)
        with col_tg2:
            if not tg_webhook_on:
                if st.button("🔗 تفعيل Webhook", key="tg_webhook_enable", use_container_width=True):
                    if not base_url or not tg_secret:
                        st.error("يلزم ضبط TELEGRAM_WEBHOOK_BASE_URL وTELEGRAM_WEBHOOK_SECRET أولاً.")
                    else:
                        try:
                            with st.spinner("⟳ يفعّل..."):
                                url = f"{base_url.rstrip('/')}/webhook/telegram/{tg_secret}"
                                mgr.enable_webhook("telegram", url, secret_token=tg_secret)
                            st.success("✅ تم تفعيل webhook تيليجرام.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"فشل التفعيل: {e}")
            else:
                if st.button("🔁 العودة لـPolling", key="tg_webhook_disable", use_container_width=True):
                    try:
                        with st.spinner("⟳ يلغي..."):
                            mgr.disable_webhook("telegram")
                        st.success("✅ تم إلغاء webhook والعودة لـpolling.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"فشل الإلغاء: {e}")

        st.markdown("---")
        st.markdown("#### 💬 WhatsApp: رابط الـWebhook")
        st.caption(
            "واتساب لا يوفّر polling إطلاقاً — الربط يتم يدوياً من لوحة Meta "
            "Developer (وليس بزر هنا كتيليجرام): الصقي الرابط ورمز التحقق "
            "أدناه في إعدادات Webhook بتطبيق Meta الخاص بك."
        )
        wa_base = os.environ.get("TELEGRAM_WEBHOOK_BASE_URL", "")  # نفس الخادم عادة (api_server.py)
        wa_verify = os.environ.get("WHATSAPP_VERIFY_TOKEN", "")
        if wa_base and wa_verify:
            st.caption("Callback URL:")
            st.code(f"{wa_base.rstrip('/')}/webhook/whatsapp", language=None)
            st.caption("Verify Token:")
            st.code(wa_verify, language=None)
        else:
            st.caption("⚪ اضبطي TELEGRAM_WEBHOOK_BASE_URL وWHATSAPP_VERIFY_TOKEN لعرض الرابط جاهزاً للنسخ.")

    # ═══════════════════════════════ 📊 حالة المنصات ═══════════════════════════════
    with tab_status:
        col_h, col_r = st.columns([4, 1])
        with col_h:
            st.markdown("#### حالة كل منصة")
        with col_r:
            if st.button("🔄 تحديث", key="social_status_refresh", use_container_width=True):
                st.rerun()

        ready = [(p, s) for p, s in status.items() if s.configured]
        not_ready = [(p, s) for p, s in status.items() if not s.configured]

        def _render_platform_row(pid: str, s) -> None:
            label = PLATFORM_LABELS_AR.get(pid, pid)
            badge = "🟢 مُهيّأة" if s.configured else f"🔴 غير مُهيّأة (يلزم: {', '.join(s.missing_env) or '—'})"
            line = f"- **{label}** — {badge}"
            if not mgr.adapters[pid].supports_monitoring:
                line += " · ⚡ نشر فقط"
            if mgr.adapters[pid].supports_webhook and pid in webhook_platforms_cfg_for_status:
                line += " · 🔗 webhook مفعّل"
            if s.last_poll:
                line += f" · آخر استطلاع: {s.last_poll}"
            st.markdown(line)
            if s.last_error:
                st.caption(f"⚠️ آخر خطأ: {s.last_error}")

        webhook_platforms_cfg_for_status = set(get_config("webhook_enabled_platforms", []))

        if ready:
            st.markdown("**🟢 جاهزة**")
            for pid, s in ready:
                _render_platform_row(pid, s)
        if not_ready:
            st.markdown("**🔴 تحتاج إعداد**")
            for pid, s in not_ready:
                _render_platform_row(pid, s)

    # ═══════════════════════════════ ✍️ نشر وجدولة ═══════════════════════════════
    with tab_publish:
        st.markdown("#### نشر يدوي فوري")
        publish_text = st.text_area("النص", key="social_publish_text", height=100)
        publish_platforms = st.multiselect(
            "انشر على:", options=list(PLATFORM_LABELS_AR.keys()),
            format_func=lambda p: PLATFORM_LABELS_AR.get(p, p), key="social_publish_platforms",
        )
        if publish_text.strip() and publish_platforms:
            over_limit = []
            for pid in publish_platforms:
                limit = PLATFORM_CHAR_LIMITS.get(pid)
                if limit and len(publish_text.strip()) > limit:
                    over_limit.append((pid, limit))
            if over_limit:
                warn_lines = "، ".join(
                    f"{PLATFORM_LABELS_AR.get(p, p)} (الحد {lim} حرف)" for p, lim in over_limit
                )
                st.warning(f"⚠️ النص أطول من الحد المعروف لبعض المنصات: {warn_lines} — قد يُرفض أو يُقتطع.")
        if st.button("🚀 نشر الآن", key="social_publish_btn", type="primary"):
            if not publish_text.strip():
                st.warning("أدخل نصاً أولاً.")
            elif not publish_platforms:
                st.warning("اختر منصة واحدة على الأقل.")
            else:
                with st.spinner("⟳ ينشر..."):
                    results = mgr.publish_to(publish_platforms, publish_text.strip())
                _pub_ok = sum(1 for r in results.values() if not str(r).startswith("ERROR"))
                st.toast(f"🚀 تم النشر على {_pub_ok}/{len(results)} منصة", icon="🚀")
                for pid, res in results.items():
                    label = PLATFORM_LABELS_AR.get(pid, pid)
                    if str(res).startswith("ERROR"):
                        st.error(f"{label}: {res}")
                    else:
                        st.success(f"{label}: ✅ {res}")

        st.markdown("---")
        st.markdown("#### 📅 جدولة المنشورات (تقويم المحتوى)")
        st.caption("⏰ الأوقات بتوقيت UTC — الخادم يعالج المنشور المستحق في أقرب دورة استطلاع.")
        sch_col1, sch_col2 = st.columns(2)
        with sch_col1:
            sch_date = st.date_input("تاريخ النشر", key="social_sched_date")
        with sch_col2:
            sch_time = st.time_input("وقت النشر (UTC)", key="social_sched_time")
        sch_text = st.text_area("نص المنشور المجدول", key="social_sched_text", height=80)
        sch_platforms = st.multiselect(
            "المنصات", options=list(PLATFORM_LABELS_AR.keys()),
            format_func=lambda p: PLATFORM_LABELS_AR.get(p, p), key="social_sched_platforms",
        )
        if st.button("📌 جدولة المنشور", key="social_sched_btn"):
            if not sch_text.strip():
                st.warning("أدخل نص المنشور أولاً.")
            elif not sch_platforms:
                st.warning("اختر منصة واحدة على الأقل.")
            else:
                with st.spinner("⟳ يجدول..."):
                    sched_dt = datetime.combine(sch_date, sch_time).isoformat() + "+00:00"
                    schedule_post(sch_platforms, sch_text.strip(), sched_dt)
                st.toast(f"📌 تمت الجدولة على {sched_dt}", icon="📌")
                st.rerun()

        scheduled = get_scheduled(status="pending")
        if scheduled:
            st.caption(f"**{len(scheduled)} منشور مجدول قيد الانتظار:**")
            for sid, plats, text, sched_at, sstatus, pub_at, result in scheduled:
                plat_names = "، ".join(PLATFORM_LABELS_AR.get(p, p) for p in plats)
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.caption(f"🕐 {sched_at} — {plat_names} — {text[:60]}")
                with c2:
                    if st.button("❌", key=f"cancel_sched_{sid}"):
                        cancel_scheduled(sid)
                        st.rerun()
        else:
            st.caption("لا توجد منشورات مجدولة حالياً.")

    # ═══════════════════════════════ 📈 تحليلات وأحداث ═══════════════════════════════
    with tab_insights:
        st.markdown("#### لوحة التحليلات (آخر 7 أيام)")
        analytics = get_analytics_summary(days=7)
        if not analytics:
            st.caption("لا توجد بيانات كافية بعد.")
        else:
            for pid, s in analytics.items():
                label = PLATFORM_LABELS_AR.get(pid, pid)
                total_sent = s["positive"] + s["negative"] + s["neutral"]
                sent_str = (
                    f"😊 {s['positive']} · 😐 {s['neutral']} · 😠 {s['negative']}"
                    if total_sent else "لا بيانات مشاعر"
                )
                st.markdown(
                    f"**{label}** — إشارات: {s['monitor_hit']} · ردود: {s['reply']} "
                    f"(فشل: {s['reply_failed']}) · منشورات: {s['publish']} (فشل: {s['publish_failed']})"
                )
                st.caption(f"المشاعر: {sent_str}")

        st.markdown("---")
        st.markdown("#### 🧾 آخر الأحداث")
        _EVENT_TYPE_AR = {
            "monitor_hit": "👁️ إشارة رُصدت",
            "reply": "💬 رد",
            "publish": "📤 نشر",
            "monitor_error": "⚠️ خطأ مراقبة",
        }
        events = get_recent_events(20)
        if not events:
            st.caption("لا توجد أحداث بعد.")
        else:
            _event_export_lines = []
            for platform, event_type, author, content, reply_content, created_at, ok, sentiment, sentiment_score in events:
                label = PLATFORM_LABELS_AR.get(platform, platform)
                ev_label = _EVENT_TYPE_AR.get(event_type, event_type)
                status_icon = "✅" if ok else "❌"
                snippet = (content or "")[:80]
                line = f"{status_icon} {label} · {ev_label} · {created_at} — {snippet}"
                st.caption(line)
                _event_export_lines.append(line)
            st.download_button(
                "⬇️ تصدير سجل الأحداث", data="\n".join(_event_export_lines),
                file_name=f"nsm_social_events_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain", key="social_events_export",
            )

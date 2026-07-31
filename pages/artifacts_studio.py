"""
pages/artifacts_studio.py
تم تفكيكه تلقائياً من streamlit_app.py الأصلي (تقسيم الكود لتحسين القابلية للصيانة والأداء).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — إعادة تصدير كل الاستيرادات والدوال المساعدة المشتركة



# ══════════════════════════════════════════════════════════════════════════
# تبويب 🧩 الواجهات التفاعلية — Artifacts (HTML/SVG) + استدعاء API
# ══════════════════════════════════════════════════════════════════════════
def render_artifacts_studio():
    st.markdown('<div class="section-header">🧩 الواجهات التفاعلية (Artifacts)</div>', unsafe_allow_html=True)
    st.caption("أنشئ واعرض محتوى HTML/SVG تفاعلياً داخل التطبيق — رسوم بيانية، نماذج، بطاقات، إلخ.")

    try:
        from core.artifacts_store import (
            save_artifact, list_artifacts, get_artifact, delete_artifact,
        )
        _ART_STORE_OK = True
    except Exception as _art_err:
        _ART_STORE_OK = False
        st.error(f"⚠️ تعذّر تحميل مخزن الواجهات التفاعلية: {_art_err}")

    # تبويب "🔌 استدعاء API" أداة HTTP عامة بدون تحقق من الوجهة (خطر SSRF) —
    # لذلك لا يُضاف لقائمة التبويبات الفرعية أصلاً إلا بعد فتح وضع المالك
    # من الشريط الجانبي، تماماً كما فعلنا مع تبويب ⚙️ النظام.
    _art_tab_defs = [("🖼️ محرّر HTML/SVG", "editor")]
    if st.session_state.get("_dev_console_unlocked", False):
        _art_tab_defs.append(("🔌 استدعاء API", "api_caller"))

    _art_tabs = st.tabs([_label for _label, _kind in _art_tab_defs])
    art_tab1 = _art_tabs[0]
    art_tab2 = _art_tabs[1] if len(_art_tabs) > 1 else None

    # ── محرّر ومعرض الواجهات التفاعلية ───────────────────────────────────
    with art_tab1:
        _default_html = (
            "<div style=\"font-family:sans-serif;text-align:center;padding:2rem;"
            "background:linear-gradient(135deg,var(--gold),var(--emerald));color:#fff;border-radius:16px\">"
            "<h2>مرحباً من NSM 🧠</h2><p>هذا مثال بسيط — عدّل الكود وشاهد النتيجة فوراً.</p></div>"
        )
        col_edit, col_preview = st.columns([1, 1])
        with col_edit:
            art_title = st.text_input("عنوان الواجهة", value="واجهتي الجديدة", key="art_title")
            art_code = st.text_area(
                "كود HTML/SVG", value=_default_html, height=320, key="art_code",
                help="يمكنك كتابة HTML كامل مع <style> و<script> — يُعرض داخل إطار معزول.",
            )
            art_height = st.slider("ارتفاع العرض (px)", 200, 900, 420, 20, key="art_height")
            c1, c2 = st.columns(2)
            with c1:
                art_render_btn = st.button("🖥️ عرض", key="art_render_btn", use_container_width=True, type="primary")
            with c2:
                art_save_btn = st.button("💾 حفظ", key="art_save_btn", use_container_width=True,
                                          disabled=not _ART_STORE_OK)
            if art_save_btn and _ART_STORE_OK:
                if art_code.strip():
                    new_id = save_artifact(art_title, art_code, kind="html")
                    st.success(f"✅ تم الحفظ (رقم #{new_id})")
                else:
                    st.warning("أدخل كوداً أولاً.")

        with col_preview:
            st.markdown("**المعاينة:**")
            if art_render_btn or art_code.strip():
                try:
                    st.components.v1.html(art_code, height=art_height, scrolling=True)
                except Exception as _render_err:
                    st.error(f"❌ خطأ أثناء العرض: {_render_err}")

        if _ART_STORE_OK:
            st.markdown("---")
            st.markdown("#### 📚 الواجهات المحفوظة")
            saved = list_artifacts()
            if not saved:
                st.info("لا توجد واجهات محفوظة بعد.")
            else:
                for item in saved[:20]:
                    with st.expander(f"#{item['id']} — {item['title']} · {item['created_at'][:19].replace('T',' ')}"):
                        full = get_artifact(item["id"])
                        st.components.v1.html(full["content"], height=300, scrolling=True)
                        dcol1, dcol2 = st.columns(2)
                        with dcol1:
                            if st.button("📋 حمّل في المحرّر", key=f"art_load_{item['id']}"):
                                st.session_state["art_code"] = full["content"]
                                st.session_state["art_title"] = full["title"]
                                st.rerun()
                        with dcol2:
                            if st.button("🗑️ حذف", key=f"art_del_{item['id']}"):
                                delete_artifact(item["id"])
                                st.rerun()

    # ── استدعاء APIs مباشرة من الواجهة — للمالك فقط ──────────────────────
    if art_tab2 is not None:
      with art_tab2:
        st.warning("🔒 أداة داخلية للمالك — ترسل طلبات HTTP فعلية من الخادم لأي رابط تُدخله. لا تشاركها مع أحد.")
        st.markdown("""
        <div style="background:color-mix(in srgb, #38bdf8 12%, var(--surface2));border:1px solid color-mix(in srgb, #38bdf8 40%, var(--border));border-radius:10px;
                    padding:0.9rem 1.2rem;direction:rtl;margin-bottom:1rem;color:var(--text)">
            <strong>🔌 جرّب أي API مباشرة</strong><br>
            <small>أدخل رابط API، الطريقة، والترويسات/الجسم (JSON) — وشاهد الاستجابة فوراً.</small>
        </div>
        """, unsafe_allow_html=True)

        api_url = st.text_input("رابط الـ API", placeholder="https://api.example.com/data", key="api_tool_url")
        colm, colh = st.columns([1, 3])
        with colm:
            api_method = st.selectbox("الطريقة", ["GET", "POST", "PUT", "PATCH", "DELETE"], key="api_tool_method")
        with colh:
            api_headers_raw = st.text_input(
                "ترويسات (JSON، اختياري)", placeholder='{"Authorization": "Bearer ..."}', key="api_tool_headers"
            )
        api_body_raw = st.text_area(
            "جسم الطلب (JSON، اختياري — لـ POST/PUT/PATCH)", height=100, key="api_tool_body"
        )

        if st.button("▶️ استدعِ API", key="api_tool_run", type="primary"):
            if not api_url.strip():
                st.warning("أدخل رابط الـ API أولاً.")
            else:
                try:
                    headers = json.loads(api_headers_raw) if api_headers_raw.strip() else {}
                except Exception:
                    st.error("❌ الترويسات ليست JSON صالحاً.")
                    headers = None
                try:
                    body = json.loads(api_body_raw) if api_body_raw.strip() else None
                except Exception:
                    st.error("❌ جسم الطلب ليس JSON صالحاً.")
                    body = None
                    api_body_raw_invalid = True
                else:
                    api_body_raw_invalid = False

                if headers is not None and not api_body_raw_invalid:
                    try:
                        with st.spinner("⟳ جارٍ الاتصال..."):
                            resp = _requests.request(
                                api_method, api_url.strip(), headers=headers or None,
                                json=body if api_method in ("POST", "PUT", "PATCH") else None,
                                params=body if api_method in ("GET", "DELETE") and isinstance(body, dict) else None,
                                timeout=15,
                            )
                        st.markdown(f"**الحالة:** `{resp.status_code}`  ·  **الزمن:** `{resp.elapsed.total_seconds()*1000:.0f} ms`")
                        try:
                            st.json(resp.json())
                        except Exception:
                            st.text(resp.text[:3000])
                    except Exception as _api_err:
                        st.error(f"❌ فشل الاتصال: {_api_err}")

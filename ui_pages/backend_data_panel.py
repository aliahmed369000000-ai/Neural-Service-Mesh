"""
NSM Backend Data Panel
======================
لوحة مركز البيانات: واجهة Streamlit لطبقة Backend Layer (SQLite) —
الوكلاء، المهام، الذاكرة، الرسائل، مخزن KV — بالإضافة إلى
الموصلات الخارجية (دفع/خرائط/رسائل) وطبقة الخدمات المصغرة
(نمط الطلب/الاستجابة الثابت nsm-ms/1.0).

كل قراءة عبر طبقة `ai.backend_layer` و`connectors.external_services`
و`ai.microservices` — نفس الأكواد التي تستخدمها نقاط REST API،
فلا يوجد ازدواجية منطقية بين الواجهة والخلفية.

التحميل كسول (lazy) ضمن آلية Cold Start في streamlit_app.py.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict

import streamlit as st

_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ui_components import (
    render_agent_cards,
    render_alert_cards,
    render_kpi_cards,
    render_section_header,
)


def _backend() -> Any:
    from ai import backend_layer  # noqa: F401
    return backend_layer


def _connectors() -> Any:
    from connectors import external_services  # noqa: F401
    return external_services


def _microservices() -> Any:
    from ai import microservices  # noqa: F401
    return microservices


def _safe(fn, *args, **kwargs):
    """استدعاء محمي — خطأ واحد لا يكسر اللوحة."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - عرضي
        return {"ok": False, "error": str(exc)}


def _try(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - عرضي
        st.error(f"خطأ: {exc}")
        return None


def _show_result(title: str, payload: Any) -> None:
    if isinstance(payload, dict) and not payload.get("ok"):
        st.warning(f"{title}: {payload.get('error', 'فشل')}")
    else:
        st.success(f"{title}: OK")
    with st.expander(title, expanded=False):
        st.json(payload)


def render_backend_data_panel() -> None:
    render_section_header(
        "مركز البيانات والموصلات",
        "Backend Layer + External Connectors + Microservices",
    )

    bl = _backend()
    counts = _safe(bl.backend_counts) or {"ok": False}
    if counts.get("ok"):
        render_kpi_cards([
            {"label": "وكلاء", "value": counts["counts"].get("agents", 0)},
            {"label": "مهام", "value": counts["counts"].get("tasks", 0)},
            {"label": "ذاكرة", "value": counts["counts"].get("memories", 0)},
            {"label": "رسائل", "value": counts["counts"].get("messages", 0)},
            {"label": "مفاتيح KV",
             "value": counts["counts"].get("kv", 0)},
        ])

    # ── التبويبات الرئيسية ──────────────────────────────────────────────
    tabs = st.tabs([
        "🤖 الوكلاء", "📋 المهام", "🧠 الذاكرة",
        "📨 الرسائل", "🔑 مخزن KV",
        "🌐 الموصلات الخارجية", "⚙️ الخدمات المصغرة",
    ])

    # 1. الوكلاء
    with tabs[0]:
        render_section_header("سجل الوكلاء", "backend agents", live=True)
        agents = _try(bl.agent_list) or []
        if not agents:
            st.info("لا يوجد وكلاء مسجلون بعد — سجّل وكيلاً جديدًا:")
        else:
            st.dataframe(agents, use_container_width=True)
        st.caption("تسجيل وكيل جديد (يُحفظ في SQLite):")
        c1, c2 = st.columns([2, 3])
        with c1:
            new_id = st.text_input("agent_id", placeholder="agent-1")
            role = st.text_input("الدور", placeholder="researcher")
        with c2:
            if st.button("تسجيل الوكيل", use_container_width=True):
                if new_id:
                    _show_result(
                        "تسجيل الوكيل",
                        _safe(bl.agent_register, new_id, role or ""))
                    st.rerun()
                else:
                    st.warning("أدخل agent_id أولًا")
        if agents:
            with st.expander("حذف وكيل"):
                del_id = st.selectbox(
                    "الوكيل المراد حذفه", [a.get("agent_id") for a in agents])
                if st.button("حذف"):
                    if del_id:
                        _show_result(
                            "حذف الوكيل",
                            _safe(bl.agent_unregister, del_id))
                        st.rerun()

    # 2. المهام
    with tabs[1]:
        render_section_header("المهام", "backend tasks", live=True)
        tasks = _try(bl.task_list) or []
        st.metric("إجمالي المهام", len(tasks))
        if tasks:
            st.dataframe(tasks, use_container_width=True)
        st.caption("مهمة جديدة:")
        c1, c2 = st.columns([3, 2])
        with c1:
            title = st.text_input("عنوان المهمة", placeholder="جمع بيانات")
        with c2:
            kind = st.selectbox(
                "النوع", ["general", "training", "collection", "research"])
            if st.button("إنشاء المهمة", use_container_width=True):
                if title:
                    _show_result(
                        "إنشاء المهمة",
                        _safe(bl.task_create, title, kind))
                    st.rerun()
                else:
                    st.warning("أدخل عنوانًا")

    # 3. الذاكرة
    with tabs[2]:
        render_section_header("ذاكرة المشروع", "memories", live=True)
        c1, c2 = st.columns([2, 3])
        with c1:
            q = st.text_input("بحث دلالي في الذاكرة", placeholder="تدريب")
        with c2:
            st.write(" ")
            if st.button("بحث", use_container_width=True):
                if q:
                    mems = _safe(bl.memory_search, q) or {"memories": []}
                    st.session_state["_mem_search"] = mems.get("memories", [])
                else:
                    st.warning("أدخل نص بحث")
        if "_mem_search" in st.session_state:
            res = st.session_state["_mem_search"]
            st.dataframe(res, use_container_width=True) if res else (
                st.info("لا نتائج"))
        st.caption("إضافة ذكرى:")
        c1, c2 = st.columns(2)
        with c1:
            sub = st.text_input("الموضوع", placeholder="surah_chain")
        with c2:
            content = st.text_area("المحتوى", height=90)
        if st.button("حفظ الذاكرة"):
            if sub and content:
                _show_result(
                    "حفظ الذاكرة",
                    _safe(bl.memory_add, sub, content, None, 0.5))
                st.rerun()
            else:
                st.warning("الموضوع والمحتوى مطلوبان")

    # 4. الرسائل
    with tabs[3]:
        render_section_header("صندوق الرسائل", "inbox", live=True)
        inbox_recv = st.text_input(
            "صندوق الوارد لـ:", value="nsm_main", key="inbox_to")
        inbox = _safe(bl.message_inbox, inbox_recv) or {"messages": []}
        msgs = inbox.get("messages", [])
        if msgs:
            st.dataframe(msgs, use_container_width=True)
        else:
            st.info("لا رسائل واردة")
        st.caption("إرسال رسالة:")
        c1, c2 = st.columns(2)
        with c1:
            sender = st.text_input("المرسل", placeholder="agent-1")
            receiver = st.text_input(
                "المستقبل", placeholder=inbox_recv)
        with c2:
            subject = st.text_input("الموضوع", placeholder="تقرير جاهز")
        body = st.text_area("نص الرسالة", height=90)
        if st.button("إرسال"):
            if sender and receiver and subject and body:
                _show_result(
                    "إرسال الرسالة",
                    _safe(bl.message_send,
                          sender, receiver, subject, body))
                st.rerun()
            else:
                st.warning("أكمل الحقول الأربعة")

    # 5. مخزن KV
    with tabs[4]:
        render_section_header("مخزن المفاتيح/القيم", "key-value",
                              live=True)
        kv_domain = st.selectbox(
            "النطاق", ["general", "swarm", "training", "agents"])
        kv_list = _safe(bl.kv_list, kv_domain) or []
        if kv_list:
            st.dataframe(kv_list, use_container_width=True)
        c1, c2, c3 = st.columns([2, 3, 2])
        with c1:
            k = st.text_input("key", placeholder="last_run")
        with c2:
            v = st.text_input("value", placeholder="ok")
        with c3:
            st.write(" ")
            if st.button("حفظ", use_container_width=True):
                if k:
                    _show_result(
                        "حفظ KV", _safe(bl.kv_set, k, v, kv_domain))
                    st.rerun()
                else:
                    st.warning("أدخل key")

    # 6. الموصلات الخارجية
    with tabs[5]:
        render_section_header(
            "الموصلات الخارجية",
            "Payment / Maps / SMS (محاكاة الآن)",
            live=True,
        )
        ms = _microservices()
        conn_result = _safe(ms.call_service, "connectors", "list") or {}
        connectors = (conn_result.get("result") or {}).get("connectors", [])
        if not connectors:
            st.info("لا موصلات مسجّلة")
        else:
            for conn in connectors:
                cap = conn.get("capabilities", [])
                st.markdown(
                    f"**{conn.get('name')}** — "
                    f"{', '.join(str(c) for c in cap) if cap else '—'}")
        c1, c2, c3 = st.columns([2, 2, 3])
        with c1:
            svc = st.selectbox(
                "الموصل",
                [c.get("name") for c in connectors] if connectors else [])
        with c2:
            acts = {}
            for c_ in connectors:
                if c_.get("name") == svc:
                    acts = {a: str(a) for a in (c_.get("capabilities") or [])}
                    break
            act = st.selectbox("الإجراء", list(acts) if acts else [])
        with c3:
            st.write(" ")
            if st.button("استدعاء", use_container_width=True):
                _show_result(
                    f"call {svc}.{act}",
                    _safe(ms.call_service, "connectors", "call",
                          {"service": svc, "action": act, "payload": {}}))

    # 7. الخدمات المصغرة
    with tabs[6]:
        render_section_header(
            "الخدمات المصغرة",
            "نمط الطلب/الاستجابة الثابت nsm-ms/1.0",
            live=True,
        )
        svcs = _safe(ms.list_services) or []
        st.markdown("الخدمات المسجّلة: " +
                    ", ".join(f"`{s}`" for s in svcs) if svcs else
                    "لا خدمات")
        c1, c2, c3 = st.columns([2, 2, 3])
        with c1:
            svc2 = st.selectbox("الخدمة", svcs if svcs else [])
        with c2:
            act2 = st.text_input("الإجراء", placeholder="counts")
        with c3:
            st.write(" ")
            if st.button("استدعاء الخدمة", use_container_width=True):
                if svc2 and act2:
                    _show_result(
                        f"call {svc2}.{act2}",
                        _safe(ms.call_service, svc2, act2, {}))
                else:
                    st.warning("أكمل الخدمة والإجراء")
        # وصف خدمة
        c1, c2 = st.columns([2, 3])
        with c1:
            svc3 = st.selectbox(
                "وصف خدمة",
                svcs if svcs else [], key="svc_desc")
        with c2:
            st.write(" ")
            if st.button("عرض الوصف", use_container_width=True):
                if svc3:
                    _show_result(
                        f"describe {svc3}",
                        _safe(ms.call_service, "meta", "describe_service",
                              {"service": svc3}))
                else:
                    st.warning("اختر خدمة")

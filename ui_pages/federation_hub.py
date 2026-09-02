# -*- coding: utf-8 -*-
"""لوحة اتحاد NSM — قائد · ذاكرة جماعية · VCEN · Private FL · حماية القرارات."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent


def _safe_import():
    mods = {}
    try:
        from ai.living_mesh import LivingMeshNode, LIVING_MESH_DIR
        mods["LivingMeshNode"] = LivingMeshNode
        mods["LIVING_MESH_DIR"] = LIVING_MESH_DIR
    except Exception as e:
        mods["err_mesh"] = str(e)
    try:
        from ai.leader_election import LeaderElection
        mods["LeaderElection"] = LeaderElection
    except Exception as e:
        mods["err_le"] = str(e)
    try:
        from ai.byzantine_decision_guard import ByzantineDecisionGuard
        mods["ByzantineDecisionGuard"] = ByzantineDecisionGuard
    except Exception as e:
        mods["err_guard"] = str(e)
    try:
        from ai.collective_cognitive_ledger import CollectiveCognitiveLedger
        mods["CollectiveCognitiveLedger"] = CollectiveCognitiveLedger
    except Exception as e:
        mods["err_ccl"] = str(e)
    try:
        from ai.verifiable_cognitive_net import VerifiableCognitiveNet
        mods["VerifiableCognitiveNet"] = VerifiableCognitiveNet
    except Exception as e:
        mods["err_vcen"] = str(e)
    try:
        from ai.private_federated_learning import PrivateFederatedLearning
        mods["PrivateFederatedLearning"] = PrivateFederatedLearning
    except Exception as e:
        mods["err_pfl"] = str(e)
    return mods


def _get_or_create_node(mods):
    if "fed_node" in st.session_state and st.session_state["fed_node"] is not None:
        return st.session_state["fed_node"]
    LivingMeshNode = mods.get("LivingMeshNode")
    if not LivingMeshNode:
        return None
    try:
        node = LivingMeshNode(node_id=None, host="127.0.0.1", port=0)
        try:
            node.join_network()
        except Exception:
            pass
        st.session_state["fed_node"] = node
        return node
    except Exception as e:
        st.session_state["fed_node_error"] = str(e)
        return None


def render_federation_hub():
    st.markdown(
        '<div class="section-header">🏛️ اتحاد NSM — شبكة ذكاء بلا مركز دائم</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "انضمام · قيادة مؤقتة · تحقق VCEN · تعلم جماعي خاص · قرارات محمية من الانقسام"
    )

    mods = _safe_import()
    node = _get_or_create_node(mods)
    if node is None:
        st.error("تعذّر تهيئة عقدة الاتحاد: " + st.session_state.get("fed_node_error", mods.get("err_mesh", "?")))
        st.info("تأكد من توفر `ai/living_mesh.py` ومفاتيح التشفير.")
        _render_join_guide()
        return

    # أدوات الطبقة
    le = mods["LeaderElection"](node, lease_seconds=30) if mods.get("LeaderElection") else None
    guard = mods["ByzantineDecisionGuard"](node) if mods.get("ByzantineDecisionGuard") else None
    ccl = mods["CollectiveCognitiveLedger"](node, quorum=2) if mods.get("CollectiveCognitiveLedger") else None
    vcen = mods["VerifiableCognitiveNet"](node, quorum=1, require_independent=True) if mods.get("VerifiableCognitiveNet") else None
    pfl = mods["PrivateFederatedLearning"](node) if mods.get("PrivateFederatedLearning") else None
    if ccl and guard:
        ccl.guard = guard
    if le and guard:
        le.guard = guard

    tabs = st.tabs([
        "📡 الحالة",
        "👑 القيادة",
        "🧠 ذاكرة ونموذج",
        "🔐 Private FL",
        "✅ VCEN",
        "🗳️ قرارات",
        "🚀 انضم للاتحاد",
    ])

    with tabs[0]:
        _tab_status(node, le, guard, ccl, pfl)
    with tabs[1]:
        _tab_leader(node, le, guard)
    with tabs[2]:
        _tab_memory(ccl)
    with tabs[3]:
        _tab_pfl(node, pfl, vcen)
    with tabs[4]:
        _tab_vcen(vcen)
    with tabs[5]:
        _tab_decisions(ccl, guard)
    with tabs[6]:
        _render_join_guide()


def _tab_status(node, le, guard, ccl, pfl):
    c1, c2, c3, c4 = st.columns(4)
    ident = {}
    try:
        ident = node.identity_info()
    except Exception:
        ident = {"node_id": node.node_id}
    with c1:
        st.metric("هوية العقدة", str(ident.get("node_id", node.node_id))[:16])
    with c2:
        st.metric("بصمة المفتاح", str(ident.get("public_key_fingerprint", "—"))[:12])
    with c3:
        if le:
            st.metric("القائد الحالي", str(le.current_leader() or "لا يوجد"))
        else:
            st.metric("القائد", "—")
    with c4:
        if guard:
            st.metric("نصاب الاتحاد", guard.majority())
        else:
            st.metric("نصاب", "—")

    if le:
        st.json(le.status())
    if guard:
        with st.expander("الحارس البيزنطي"):
            st.json(guard.status())
    if pfl:
        st.caption("سياسة الخصوصية: " + str(pfl.stats()))
    if ccl:
        st.caption(f"ذاكرة جماعية: {ccl.memory_snapshot().get('count', 0)} إدخال")


def _tab_leader(node, le, guard):
    if not le:
        st.warning("LeaderElection غير متاح")
        return
    st.write("انتخاب قائد مؤقت — ليس منسّقاً ثابتاً")
    roster_raw = st.text_input(
        "أعضاء الاتحاد (مفصولة بفاصلة)",
        value=node.node_id,
        key="fed_roster_input",
    )
    if guard and st.button("تحديث Roster", key="fed_set_roster"):
        members = [x.strip() for x in roster_raw.split(",") if x.strip()]
        st.success(str(guard.set_roster(members)))

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("بدء انتخاب", key="fed_start_elec"):
            st.session_state["fed_election"] = le.start_election()
            st.json(st.session_state["fed_election"])
    with col2:
        if st.button("إعلان قيادة محلية (تجريبي)", key="fed_become"):
            if guard:
                g = guard.validate_leader_claim(
                    term=int(le.state.get("term") or 1),
                    leader_id=node.node_id,
                    vote_count=max(1, guard.majority()),
                )
                if not g.get("ok"):
                    st.error(g)
                else:
                    st.json(le.become_leader(le.state.get("term")))
            else:
                st.json(le.become_leader())
    with col3:
        if st.button("Heartbeat", key="fed_hb"):
            st.json(le.heartbeat())

    succ = st.text_input("خليفة للتسليم", value="", key="fed_handoff_id")
    if st.button("تسليم القيادة", key="fed_handoff") and succ:
        st.json(le.handoff(succ.strip()))

    st.subheader("حالة القيادة")
    st.json(le.status())


def _tab_memory(ccl):
    if not ccl:
        st.warning("CCL غير متاح")
        return
    snap = ccl.memory_snapshot(limit=20)
    st.metric("حجم الذاكرة", snap.get("count", 0))
    st.caption(f"chain tip: {snap.get('chain_tip')}")
    st.json(ccl.model_snapshot())
    integrity = ccl.verify_memory_integrity()
    if integrity.get("ok"):
        st.success("سلامة الذاكرة: OK")
    else:
        st.error(integrity)
    with st.expander("آخر الإدخالات"):
        st.json(snap.get("items") or [])
    with st.expander("تصدير تدقيق"):
        st.json(ccl.full_audit_export())


def _tab_pfl(node, pfl, vcen):
    if not pfl:
        st.warning("Private FL غير متاح")
        return
    st.write("لا تُرسل عيّنات خام — فقط تحديثات مقصوصة/مقنّعة")
    steps = st.slider("خطوات محلية", 1, 10, 3, key="fed_pfl_steps")
    if st.button("تشغيل خطوة تدريب خاصة محلية", key="fed_pfl_step"):
        out = pfl.local_private_train_step(steps=steps)
        st.session_state["fed_pfl_last"] = out
        st.json(out)
        if out.get("update_meta", {}).get("raw_data_included"):
            st.error("انتهاك خصوصية!")
        else:
            st.success("لا بيانات خام في المخرجات")

    if st.button("بناء مساهمة مقنّعة (share)", key="fed_pfl_share"):
        share = pfl.build_private_share(
            "ui_round",
            [node.node_id],
            steps=steps,
        )
        st.session_state["fed_pfl_share"] = share
        st.json({k: v for k, v in share.items() if k != "masked_update"} | {
            "masked_update_preview": (share.get("masked_update") or [])[:4]
        })

    if vcen and st.button("تجميع محلي → مطالبة VCEN", key="fed_pfl_vcen"):
        shares = []
        if "fed_pfl_share" in st.session_state:
            shares.append(st.session_state["fed_pfl_share"])
        else:
            shares.append(pfl.build_private_share("ui_round2", [node.node_id]))
        # مدقق = نفس العقدة لن يمرّ require_independent — نبني claim فقط
        out = pfl.private_round_to_vcen_claim(vcen, shares, verifier_vcens=None)
        st.json({
            "ok": out.get("ok"),
            "aggregate": out.get("aggregate"),
            "claim_id": (out.get("claim") or {}).get("claim_id"),
            "result_hash": (out.get("claim") or {}).get("result_hash"),
        })
        st.caption("للقبول النهائي أضف مدققاً مستقلاً (عقدة أخرى) — سياسة VCEN")


def _tab_vcen(vcen):
    if not vcen:
        st.warning("VCEN غير متاح")
        return
    st.write("لا تُقبل نتيجة بدون توقيع + Hash + مدقق مستقل + Quorum")
    st.metric("مطالبات مقبولة محلياً", len(getattr(vcen, "_accepted", {})))
    st.metric("مرفوضة", len(getattr(vcen, "_rejected", [])))
    if getattr(vcen, "_rejected", None):
        with st.expander("آخر الرفض"):
            st.json(vcen._rejected[-5:])
    if getattr(vcen, "_accepted", None):
        with st.expander("آخر القبول"):
            keys = list(vcen._accepted.keys())[-5:]
            st.json({k: vcen._accepted[k].get("verdict") for k in keys})


def _tab_decisions(ccl, guard):
    if not ccl:
        st.warning("CCL غير متاح")
        return
    title = st.text_input("عنوان القرار", value="اعتماد جولة اتحاد", key="fed_dec_title")
    payload_txt = st.text_area("الحمولة JSON", value='{"action": "accept_round"}', key="fed_dec_payload")
    if st.button("طرح قرار", key="fed_propose"):
        try:
            payload = json.loads(payload_txt)
        except Exception as e:
            st.error(e)
            payload = {"raw": payload_txt}
        thr = guard.majority() if guard else 2
        dec = ccl.propose_decision(title, payload, threshold=thr)
        st.session_state["fed_last_decision"] = dec["decision_id"]
        st.json(dec)

    did = st.text_input(
        "decision_id",
        value=st.session_state.get("fed_last_decision", ""),
        key="fed_dec_id",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("تصويت نعم", key="fed_vote_yes") and did:
            st.json(ccl.vote_decision(did, True))
    with c2:
        if st.button("إغلاق القرار", key="fed_finalize") and did:
            st.json(ccl.finalize_decision(did))

    if guard:
        st.caption(f"Majority الاتحاد: {guard.majority()} من {len(guard.roster())}")


def _render_join_guide():
    st.subheader("انضمام عقدة للاتحاد خلال دقيقة")
    st.markdown(
        """
**1. استنساخ وتشغيل**
```bash
git clone https://github.com/aliahmed369000000-ai/Neural-Service-Mesh.git
cd Neural-Service-Mesh
pip install -r requirements.txt   # أو الاعتمادات المتوفرة لديك
```

**2. تشغيل عقدة شبكة**
```bash
python -m ai.node_launcher --id my_node --host 0.0.0.0 --port 7860 --seed-host <SEED_IP> --seed-port 7860
```

**3. ما يحدث تلقائياً**
- هوية دائمة (`keys/node_identity.json` + RSA)
- اكتشاف أقران عبر البذرة
- Ping / مسارات / مهام قابلة للتحقق

**4. إثبات الاتحاد محلياً**
```bash
python3 scripts/prove_federation.py
```

**5. المبادئ**
- لا مركز دائم — قائد مؤقت منتخب
- لا نتيجة بدون VCEN (توقيع + Hash + مدقق + Quorum)
- لا عيّنات خام في التعلم الجماعي (Private FL)
- القرارات بنصاب اتحادي ومقاومة للانقسام

راجع أيضاً: `FEDERATION.md`
"""
    )


# alias شائع
render = render_federation_hub

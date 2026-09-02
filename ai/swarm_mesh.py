"""
Swarm Intelligence Mesh — شبكة وكلاء لامركزية خفيفة
===================================================
  • عقد (nodes) تتبادل «خبرات» موقعة محلياً
  • بروتوكول نشر: kind + embedding/hash + payload
  • محاكاة شبكة كوكبية على جهاز واحد (قابل للربط لاحقاً بـ libp2p/Redis)

لا يتصل بالإنترنت لإرسال أوزان حقيقية دون إعداد صريح.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("SwarmMesh")

ROOT = Path(__file__).resolve().parent.parent
MESH_DIR = ROOT / "artifacts" / "model_training" / "super_ai" / "swarm"
MESH_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = MESH_DIR / "mesh_state.json"
OUTBOX_MAX = 500   # أقصى عدد رسائل outbox محفوظة؛ الأقدم يُقصّ
INBOX_MAX = 2000   # أقصى عدد رسائل inbox (نسخة لكل عقدة بكل بث، لذا حدّها أكبر)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load() -> Dict[str, Any]:
    if STATE_PATH.is_file():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"nodes": {}, "inbox": [], "outbox": [], "created_at": _now()}


def _save(state: Dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def register_node(region: str = "local", role: str = "trainer") -> str:
    state = _load()
    nid = f"node_{uuid.uuid4().hex[:8]}"
    state["nodes"][nid] = {
        "region": region,
        "role": role,
        "joined_at": _now(),
        "last_seen": _now(),
    }
    _save(state)
    return nid


def broadcast_experience(
    node_id: str,
    kind: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    state = _load()
    if node_id not in state.get("nodes", {}):
        node_id = register_node()
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    msg = {
        "id": f"msg_{uuid.uuid4().hex[:10]}",
        "from": node_id,
        "kind": kind,
        "payload": payload,
        "hash": hashlib.sha256(body.encode()).hexdigest()[:16],
        "ts": _now(),
    }
    state.setdefault("outbox", []).append(msg)
    if len(state["outbox"]) > OUTBOX_MAX:
        state["outbox"] = state["outbox"][-OUTBOX_MAX:]
    # محاكاة انتشار: كل العقد الأخرى تستقبل
    for nid in state.get("nodes", {}):
        if nid != node_id:
            state.setdefault("inbox", []).append({**msg, "to": nid})
    if len(state.get("inbox", [])) > INBOX_MAX:
        state["inbox"] = state["inbox"][-INBOX_MAX:]
    state["nodes"][node_id]["last_seen"] = _now()
    _save(state)
    try:
        from ai.persistent_memory import remember_experience

        remember_experience(
            kind=f"swarm_{kind}",
            text=f"من {node_id}: {kind} {body[:200]}",
            meta={"hash": msg["hash"], "node": node_id},
        )
    except Exception:
        pass
    return msg


def mesh_status() -> str:
    state = _load()
    lines = [
        "## 🌍 شبكة السرب (Swarm Mesh)",
        f"- عقد: **{len(state.get('nodes', {}))}**",
        f"- outbox: **{len(state.get('outbox', []))}** · inbox: **{len(state.get('inbox', []))}**",
        "",
        "### العقد",
    ]
    for nid, meta in list(state.get("nodes", {}).items())[:20]:
        lines.append(
            f"- `{nid}` · {meta.get('region')} · {meta.get('role')} · last={str(meta.get('last_seen'))[:19]}"
        )
    lines += [
        "",
        "### آخر رسائل",
    ]
    for m in list(state.get("outbox", []))[-5:]:
        lines.append(f"- {m.get('ts', '')[:19]} `{m.get('kind')}` hash={m.get('hash')} from={m.get('from')}")
    lines += [
        "",
        "_للتوسع الكوكبي: اربط outbox بـ Redis/NATS/libp2p مع توقيع رسائل وتشفير._",
    ]
    return "\n".join(lines)


def simulate_planet_sync(n_nodes: int = 5) -> str:
    """محاكاة عقد في مناطق متعددة تتبادل درس تسريع."""
    state = _load()
    # ابدأ نظيفاً جزئياً للعرض
    regions = ["americas", "europe", "asia", "mena", "local"]
    node_ids = []
    for i in range(max(2, min(20, n_nodes))):
        nid = register_node(region=regions[i % len(regions)], role="trainer")
        node_ids.append(nid)
    # عقدة أمريكا تكتشف تسريعاً
    tip = {
        "tip": "AMP+compile on T4 raised tokens/s by ~1.8x",
        "metric": {"speedup": 1.8},
        "weights_ref": "local://not_uploaded",
    }
    msg = broadcast_experience(node_ids[0], "training_tip", tip)
    # بقية العقد «تتعلم» عبر inbox
    return (
        mesh_status()
        + "\n\n### محاكاة مزامنة\n"
        + f"- بُثّت رسالة `{msg['id']}` من `{msg['from']}`\n"
        + f"- المحتوى: {tip['tip']}\n"
        + "- استقبلتها العقد الأخرى في inbox (محلياً)."
    )


def handle_swarm_command(user_input: str) -> Optional[str]:
    import re

    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(
        r"(سرب|swarm|mesh|وكلاء\s*لامركز|شبك[ةه]\s*وكلاء|مزامن[ةه]\s*كوكب)",
        text,
        re.I,
    ):
        return None
    if re.search(r"(حاكي|simulate|مزامن)", text, re.I):
        n = 5
        m = re.search(r"(\d+)\s*(?:عقد|node)", text, re.I)
        if m:
            n = max(2, min(20, int(m.group(1))))
        return simulate_planet_sync(n_nodes=n)
    if re.search(r"(بث|broadcast|انشر\s*خبر)", text, re.I):
        nid = register_node()
        msg = broadcast_experience(nid, "manual_tip", {"text": text[:500]})
        return f"## 📡 بُثّت خبرة\n```json\n{json.dumps(msg, ensure_ascii=False, indent=2)}\n```"
    return mesh_status()

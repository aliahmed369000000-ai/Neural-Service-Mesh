"""
Cosmic Mesh — شبكة وعي لامركزية محصّنة (محاكاة عقد)
====================================================
كل عقدة = مجلد محلي. البث = كتابة رسائل مشفّرة بسيطة (HMAC) في outbox/inbox.
لا يعتمد على أقمار حقيقية — بروتوكول جاهز للربط لاحقاً بـ LoRa/MQTT.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
MESH = ROOT / "artifacts" / "model_training" / "cosmic_mesh"
MESH.mkdir(parents=True, exist_ok=True)

SECRET = (os.environ.get("NSM_MESH_SECRET") or "nsm-dev-mesh-secret").encode("utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _node_id() -> str:
    p = MESH / "node_id.txt"
    if p.is_file():
        return p.read_text(encoding="utf-8").strip()
    nid = "node-" + uuid.uuid4().hex[:10]
    p.write_text(nid, encoding="utf-8")
    return nid


def _sign(body: str) -> str:
    return hmac.new(SECRET, body.encode("utf-8"), hashlib.sha256).hexdigest()


def _verify(body: str, sig: str) -> bool:
    return hmac.compare_digest(_sign(body), sig)


def mesh_broadcast(topic: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    nid = _node_id()
    msg = {
        "id": uuid.uuid4().hex,
        "from": nid,
        "topic": topic,
        "payload": payload,
        "at": _now(),
    }
    body = json.dumps(msg, ensure_ascii=False, sort_keys=True)
    envelope = {"body": body, "sig": _sign(body)}
    outbox = MESH / "outbox"
    outbox.mkdir(exist_ok=True)
    path = outbox / f"{msg['id']}.json"
    path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
    # محاكاة تسليم محلي لـ inbox العقدة نفسها + peers/
    peers = MESH / "peers"
    peers.mkdir(exist_ok=True)
    inbox = MESH / "inbox"
    inbox.mkdir(exist_ok=True)
    (inbox / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    # حدّث world_model إن أمكن
    try:
        from world_model.environment_model import EnvironmentModel
        env = EnvironmentModel(model_dir=str(ROOT / "world_model"))
        env.update_service(nid, name="mesh-node", node_type="mesh", health="healthy", tags=[topic])
    except Exception:
        pass
    return {"ok": True, "node": nid, "message_id": msg["id"], "path": str(path.relative_to(ROOT))}


def mesh_recv(limit: int = 20) -> Dict[str, Any]:
    inbox = MESH / "inbox"
    inbox.mkdir(exist_ok=True)
    items = []
    for f in sorted(inbox.glob("*.json"))[-limit:]:
        try:
            env = json.loads(f.read_text(encoding="utf-8"))
            body = env.get("body") or ""
            sig = env.get("sig") or ""
            valid = _verify(body, sig)
            msg = json.loads(body) if body else {}
            items.append({"file": f.name, "valid_sig": valid, "msg": msg})
        except Exception as e:
            items.append({"file": f.name, "error": str(e)})
    return {"ok": True, "node": _node_id(), "n": len(items), "messages": items}


def mesh_status() -> Dict[str, Any]:
    return {
        "node_id": _node_id(),
        "outbox": len(list((MESH / "outbox").glob("*.json"))) if (MESH / "outbox").exists() else 0,
        "inbox": len(list((MESH / "inbox").glob("*.json"))) if (MESH / "inbox").exists() else 0,
        "transport": "local-file-hmac (LoRa/MQTT adapter later)",
        "at": _now(),
    }


def handle_mesh_command(user_input: str) -> Optional[str]:
    import re
    text = (user_input or "").strip()
    if not text:
        return None
    if re.search(r"(شبك[ةه]\s*وعي|cosmic\s*mesh|mesh\s*status|بث\s*لامركزي)", text, re.I):
        if re.search(r"بث|broadcast", text, re.I):
            r = mesh_broadcast("knowledge", {"note": "heartbeat", "from_cmd": True})
            return "## 🌌 Cosmic Mesh — بث\n```json\n" + json.dumps(r, ensure_ascii=False, indent=2) + "\n```"
        if re.search(r"استقبال|recv|inbox", text, re.I):
            r = mesh_recv()
            return "## 🌌 Cosmic Mesh — استقبال\n```json\n" + json.dumps(r, ensure_ascii=False, indent=2, default=str)[:3000] + "\n```"
        return "## 🌌 Cosmic Mesh\n```json\n" + json.dumps(mesh_status(), ensure_ascii=False, indent=2) + "\n```"
    return None

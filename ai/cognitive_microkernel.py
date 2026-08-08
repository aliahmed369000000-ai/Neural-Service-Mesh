"""
Cognitive Microkernel — واجهة نواة إدراكية سيادية (محاكاة برمجية)
================================================================
ليست بديلاً حقيقياً عن Linux. تُعرّف «نداءات نظام» إدراكية محدودة:
  ckg_lookup · reason · sense_poll · mesh_send · mesh_recv

الهدف: عزل مسار الاستدلال عن طبقة التطبيق وتقليل سطح الهجوم منطقياً.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
KLOG = ROOT / "artifacts" / "model_training" / "microkernel"
KLOG.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class KernelState:
    boots: int = 0
    calls: List[dict] = field(default_factory=list)
    offline: bool = True


_STATE = KernelState()


def k_boot() -> Dict[str, Any]:
    _STATE.boots += 1
    _STATE.offline = True
    rec = {"syscall": "boot", "at": _now(), "boots": _STATE.boots}
    _STATE.calls.append(rec)
    return {"ok": True, **rec, "note_ar": "نواة إدراكية محاكية — ليست OS عتادياً."}


def k_ckg_lookup(query: str) -> Dict[str, Any]:
    try:
        from ai.mcp_internal_gateway import search_ckg
        r = search_ckg(query)
    except Exception as e:
        r = {"ok": False, "error": str(e)}
    _STATE.calls.append({"syscall": "ckg_lookup", "query": query, "at": _now()})
    return r


def k_reason(question: str) -> Dict[str, Any]:
    try:
        from ai.mcp_internal_gateway import reason
        r = reason(question, train_on_query=False)
    except Exception as e:
        r = {"ok": False, "error": str(e)}
    _STATE.calls.append({"syscall": "reason", "q": question[:80], "at": _now()})
    return r


def k_sense_poll() -> Dict[str, Any]:
    try:
        from ai.sovereignty_loop import knowledge_pulse
        r = knowledge_pulse()
    except Exception as e:
        r = {"ok": False, "error": str(e)}
    _STATE.calls.append({"syscall": "sense_poll", "at": _now()})
    return r


def k_mesh_send(payload: Dict[str, Any], topic: str = "knowledge") -> Dict[str, Any]:
    try:
        from ai.cosmic_mesh import mesh_broadcast
        return mesh_broadcast(topic, payload)
    except Exception as e:
        return {"ok": False, "error": str(e)}


def k_status() -> Dict[str, Any]:
    return {
        "boots": _STATE.boots,
        "n_calls": len(_STATE.calls),
        "offline": _STATE.offline,
        "syscalls": ["boot", "ckg_lookup", "reason", "sense_poll", "mesh_send", "mesh_recv"],
        "at": _now(),
    }


def handle_kernel_command(user_input: str) -> Optional[str]:
    import re
    text = (user_input or "").strip()
    if not text:
        return None
    if re.search(r"(نظام\s*تشغيل\s*معرف|microkernel|نواة\s*ادراكي|cognitive\s*os)", text, re.I):
        k_boot()
        return "## 🧠 Cognitive Microkernel\n```json\n" + json.dumps(k_status(), ensure_ascii=False, indent=2) + "\n```\n" + (
            "نداءات: `ckg_lookup` · `reason` · `sense_poll` · `mesh_send` — محاكاة سيادية فوق Python حالياً."
        )
    return None

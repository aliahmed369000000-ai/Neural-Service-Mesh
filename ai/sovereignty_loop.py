"""
Sovereignty Loop — حلقة سيادة تشغيلية على المكوّنات الموجودة
=============================================================
Sensors → World Model → Continuous Training (gated) → Verifier signal

لا يزحف على الإنترنت بعدوانية؛ يعتمد مصادر محلية + إشارات حساسات.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("SovereigntyLoop")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "model_training" / "sovereignty"
OUT.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def knowledge_pulse() -> Dict[str, Any]:
    """استطلاع حساسات + تحديث نموذج العالم + مرشّحات فجوات."""
    events: List[dict] = []
    env_summary: Dict[str, Any] = {}
    try:
        from sensors.sensor_hub import SensorHub
        from sensors.filesystem_sensor import FilesystemSensor
        from sensors.log_sensor import LogSensor
        from world_model.environment_model import EnvironmentModel

        env = EnvironmentModel(model_dir=str(ROOT / "world_model"))
        hub = SensorHub()
        hub.register(
            FilesystemSensor(
                config={"watch_paths": [str(ROOT / "ai"), str(ROOT / "knowledge")]}
            )
        )
        hub.register(LogSensor(config={"log_paths": [str(ROOT / "logs")]}))
        hub.on_event(lambda e: env.ingest_sensor_event(e.to_dict()))
        polled = hub.poll_now()
        events = [e.to_dict() for e in polled[-20:]]
        env_summary = env.summary() if hasattr(env, "summary") else {}
    except Exception as e:
        logger.warning("pulse sensors: %s", e)
        env_summary = {"error": str(e)}

    # فجوات: أحداث تشير لأسئلة/ملفات معرفة ناقصة — heuristic
    gap_hints = []
    for ev in events:
        msg = str(ev.get("message") or ev.get("payload") or "")
        if any(k in msg.lower() for k in ("missing", "not found", "error", "فشل", "ناقص")):
            gap_hints.append(msg[:200])
    report = {
        "at": _now(),
        "n_events": len(events),
        "gap_hints": gap_hints[:10],
        "environment": env_summary,
        "whatsapp_note_ar": (
            "بوابة whatsapp_gateway منفصلة (Vercel) — اربط webhook لاحقاً "
            "ليستدعي knowledge_pulse عند تكرار سؤال بلا إجابة قوية."
        ),
    }
    (OUT / "last_pulse.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return report


def gated_continuous_training(prefer_remote: bool = False) -> Dict[str, Any]:
    """تدريب مستمر فقط إذا أعطى نموذج العالم ضوءاً أخضر."""
    safety: Dict[str, Any] = {}
    try:
        from world_model.environment_model import EnvironmentModel
        env = EnvironmentModel(model_dir=str(ROOT / "world_model"))
        safety = env.assess_training_safety("continuous_train", estimated_vram_mb=4096)
    except Exception as e:
        safety = {"green_light": False, "error": str(e), "decision": "deny_or_review"}

    if not safety.get("green_light"):
        return {
            "ok": False,
            "blocked": True,
            "safety": safety,
            "next_ar": "استخدم Kaggle أو خفّض الحجم — العالم توقّع خطراً.",
        }

    try:
        from ai.continuous_training_agent import run_continuous_cycle
        cycle = run_continuous_cycle(prefer_remote=prefer_remote)
    except Exception as e:
        cycle = {"error": str(e)}
    return {"ok": True, "blocked": False, "safety": safety, "cycle": cycle}


def sovereignty_status() -> Dict[str, Any]:
    st: Dict[str, Any] = {"at": _now(), "components": {}}
    for name, imp in (
        ("mcp_server", "mcp_server.server"),
        ("world_model", "world_model.environment_model"),
        ("sensors", "sensors.sensor_hub"),
        ("continuous_training", "ai.continuous_training_agent"),
        ("reasoning_pipeline", "ai.reasoning_pipeline"),
    ):
        try:
            __import__(imp)
            st["components"][name] = True
        except Exception as e:
            st["components"][name] = str(e)
    return st


def handle_sovereignty_command(user_input: str) -> Optional[str]:
    import re
    text = (user_input or "").strip()
    if not text:
        return None
    if re.search(r"(نبض[ةه]\s*معرف|knowledge\s*pulse|حساسات)", text, re.I):
        return "## 📡 نبضة معرفية\n\n```json\n" + json.dumps(knowledge_pulse(), ensure_ascii=False, indent=2, default=str)[:3500] + "\n```"
    if re.search(r"(تدريب\s*مبوّب|gated\s*train|سياد[ةه]\s*تدريب)", text, re.I):
        return "## 🛂 تدريب مبوّب بنموذج العالم\n\n```json\n" + json.dumps(gated_continuous_training(), ensure_ascii=False, indent=2, default=str)[:3500] + "\n```"
    if re.search(r"(سياد[ةه]\s*النظام|sovereignty|تفعيل\s*mcp|جسر\s*mcp)", text, re.I):
        return (
            "## 🏛️ سيادة التشغيل\n\n```json\n"
            + json.dumps(sovereignty_status(), ensure_ascii=False, indent=2)
            + "\n```\n\n"
            "MCP: `python mcp_server/server.py` · أدوات جديدة: reasoning_answer, training_safety_check, knowledge_pulse"
        )
    return None

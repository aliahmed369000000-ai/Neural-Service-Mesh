"""
Phase 7 – Environment Model (World Model)
==========================================
Maintains a live, structured map of the system and its environment.

Stored in /world_model/ directory as JSON.

The model tracks:
  known_services    – all registered nodes + their current health
  known_capabilities – aggregated from marketplace + sensor events
  known_failures    – recurring failure patterns from sensors + memory

Updated by:
  - SensorHub event callbacks
  - EvolutionPipeline cycle results
  - Direct mesh state snapshots

File: world_model/environment_model.py
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_STATE: Dict[str, Any] = {
    "_meta": {
        "schema_version": "7.0.0",
        "description": "Phase 7 Environment Model — live system world map",
        "created_at": None,
        "last_updated": None,
    },
    "known_services": {},       # node_id → {name, type, health, last_seen, tags}
    "known_capabilities": [],   # list of capability strings
    "known_failures": {},       # failure_key → {count, last_seen, pattern, severity}
    "sensor_alerts": [],        # recent critical/error sensor events (last 200)
    "known_concepts": {},       # concept_name → {cluster, strength, frequency}
    "concept_relations": [],    # [{source, target, weight, type}] top-100
    "system_metrics": {         # aggregate health metrics
        "total_services": 0,
        "healthy_services": 0,
        "failed_services": 0,
        "total_capabilities": 0,
        "active_failure_patterns": 0,
        "last_snapshot_at": None,
    },
}


class EnvironmentModel:
    """
    Phase 7: Live world model of the system's environment.

    Acts as the 'nervous system memory' — sensors write here,
    the EvolutionPipeline reads here to decide what to fix/build.
    """

    def __init__(self, model_dir: str = "./world_model"):
        self._dir = Path(model_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "environment.json"
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = self._load()

    # ── Persistence ────────────────────────────────────────────────────────

    def _load(self) -> Dict[str, Any]:
        if self._path.exists():
            try:
                with open(self._path, "r") as f:
                    state = json.load(f)
                logger.info("[EnvironmentModel] loaded from disk")
                return state
            except Exception as exc:
                logger.warning(f"[EnvironmentModel] load error: {exc}, using defaults")
        state = json.loads(json.dumps(_DEFAULT_STATE))
        state["_meta"]["created_at"] = datetime.now(timezone.utc).isoformat()
        return state

    def _save(self):
        self._state["_meta"]["last_updated"] = datetime.now(timezone.utc).isoformat()
        tmp = str(self._path) + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(self._state, f, indent=2)
            os.replace(tmp, self._path)
        except Exception as exc:
            logger.error(f"[EnvironmentModel] save error: {exc}")

    # ── Service registry ───────────────────────────────────────────────────

    def update_service(self, node_id: str, name: str, node_type: str = "node",
                       health: str = "healthy", tags: Optional[List[str]] = None,
                       extra: Optional[dict] = None):
        with self._lock:
            self._state["known_services"][node_id] = {
                "name": name,
                "type": node_type,
                "health": health,
                "tags": tags or [],
                "last_seen": datetime.now(timezone.utc).isoformat(),
                **(extra or {}),
            }
            self._update_metrics()
            self._save()

    def mark_service_failed(self, node_id: str, reason: str = ""):
        with self._lock:
            svc = self._state["known_services"].get(node_id, {})
            svc["health"] = "failed"
            svc["failure_reason"] = reason
            svc["last_failed"] = datetime.now(timezone.utc).isoformat()
            self._state["known_services"][node_id] = svc
            self._update_metrics()
            self._save()

    # ── Capability registry ────────────────────────────────────────────────

    def add_capability(self, capability: str):
        with self._lock:
            caps = self._state["known_capabilities"]
            if capability not in caps:
                caps.append(capability)
                self._update_metrics()
                self._save()

    def set_capabilities(self, capabilities: List[str]):
        with self._lock:
            self._state["known_capabilities"] = list(set(capabilities))
            self._update_metrics()
            self._save()

    # ── Failure tracking ───────────────────────────────────────────────────

    def record_failure(self, failure_key: str, pattern: str,
                       severity: str = "error", metadata: Optional[dict] = None):
        with self._lock:
            failures = self._state["known_failures"]
            entry = failures.get(failure_key, {"count": 0, "pattern": pattern, "severity": severity})
            entry["count"] += 1
            entry["last_seen"] = datetime.now(timezone.utc).isoformat()
            entry["pattern"] = pattern
            entry["severity"] = severity
            if metadata:
                entry.update(metadata)
            failures[failure_key] = entry
            self._update_metrics()
            self._save()

    # ── Sensor alert ingestion ─────────────────────────────────────────────

    def ingest_sensor_event(self, event_dict: dict):
        """Called by SensorHub callback for error/critical events."""
        severity = event_dict.get("severity", "info")
        if severity in ("error", "critical", "warning"):
            with self._lock:
                alerts = self._state["sensor_alerts"]
                alerts.append(event_dict)
                self._state["sensor_alerts"] = alerts[-200:]

                # Auto-record failure pattern
                event_type = event_dict.get("event_type", "")
                if severity in ("error", "critical"):
                    key = f"{event_type}:{event_dict.get('sensor_id', '')}"
                    payload = event_dict.get("payload", {})
                    pattern = payload.get("error") or payload.get("line", "")[:100] or event_type
                    self._record_failure_unlocked(key, pattern, severity)

                self._save()

    def _record_failure_unlocked(self, key: str, pattern: str, severity: str):
        failures = self._state["known_failures"]
        entry = failures.get(key, {"count": 0, "pattern": pattern, "severity": severity})
        entry["count"] += 1
        entry["last_seen"] = datetime.now(timezone.utc).isoformat()
        failures[key] = entry

    # ── Snapshot from live mesh ────────────────────────────────────────────

    def snapshot_from_mesh(self, mesh):
        """Pull current state from the mesh and update the world model."""
        try:
            # Services from registry
            for node_id, node_data in mesh.registry._nodes.items():
                self.update_service(
                    node_id=node_id,
                    name=node_data.get("name", node_id),
                    node_type=node_data.get("node_type", "node"),
                    tags=node_data.get("tags", []),
                )
            # Capabilities from marketplace
            try:
                caps = [c.get("capability", "") for c in mesh.marketplace.list_capabilities()]
                self.set_capabilities([c for c in caps if c])
            except Exception:
                pass

            with self._lock:
                self._state["system_metrics"]["last_snapshot_at"] = \
                    datetime.now(timezone.utc).isoformat()
                self._update_metrics()
                self._save()
        except Exception as exc:
            logger.error(f"[EnvironmentModel] snapshot_from_mesh error: {exc}")

    # ── Metrics ────────────────────────────────────────────────────────────

    def _update_metrics(self):
        svcs = self._state["known_services"]
        healthy = sum(1 for s in svcs.values() if s.get("health") == "healthy")
        failed = sum(1 for s in svcs.values() if s.get("health") == "failed")
        self._state["system_metrics"].update({
            "total_services": len(svcs),
            "healthy_services": healthy,
            "failed_services": failed,
            "total_capabilities": len(self._state["known_capabilities"]),
            "active_failure_patterns": len(self._state["known_failures"]),
        })

    # ── Read API ───────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self._state))

    def get_known_services(self) -> dict:
        with self._lock:
            return dict(self._state["known_services"])

    def get_known_failures(self) -> dict:
        with self._lock:
            return dict(self._state["known_failures"])

    def get_known_capabilities(self) -> List[str]:
        with self._lock:
            return list(self._state["known_capabilities"])

    def get_sensor_alerts(self, limit: int = 50) -> List[dict]:
        with self._lock:
            return list(self._state["sensor_alerts"])[-limit:]

    # ── CKG Integration ────────────────────────────────────────────────────

    def update_from_ckg(self, ckg) -> None:
        """
        تغذية الـ World Model من CKG بعد كل تحديث.
        يُستدعى تلقائياً بعد كل دورة ingestion.
        """
        try:
            stats = ckg.stats()
            top_relations = sorted(
                ckg.all_relations(),
                key=lambda r: r["weight"],
                reverse=True,
            )[:100]

            # مفاهيم مضغوطة (بدون sources الطويلة)
            concepts_compact = {
                name: {
                    "cluster":   c["cluster"],
                    "strength":  c["strength"],
                    "frequency": c["frequency"],
                }
                for name, c in {c["name"]: c for c in ckg.all_concepts()}.items()
            }

            with self._lock:
                self._state["known_concepts"]    = concepts_compact
                self._state["concept_relations"] = top_relations
                self._update_metrics()
                self._save()

            logger.info(
                f"[EnvironmentModel] CKG sync: "
                f"{len(concepts_compact)} concepts, {len(top_relations)} relations"
            )
        except Exception as exc:
            logger.error(f"[EnvironmentModel] update_from_ckg error: {exc}")

    def summary(self) -> dict:
        with self._lock:
            return {
                "world_model_path": str(self._path),
                **self._state["system_metrics"],
                "recent_alerts": len(self._state["sensor_alerts"]),
                "known_concepts": len(self._state.get("known_concepts", {})),
                "concept_relations": len(self._state.get("concept_relations", [])),
            }

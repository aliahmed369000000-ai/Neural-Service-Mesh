"""
Phase 5 – Service Generator Engine
====================================
When the system detects a gap (Node A → ??? → Node B),
it proposes and generates a new service definition automatically.

Workflow:
  1. GapDetector finds: Node A -> ??? -> Node B
  2. ServiceGenerator proposes: New Node Required: DataNormalizer
  3. Generates full node definition (schema, capability, tags)
  4. Registers the new node into the mesh autonomously
  5. AIGovernanceLayer validates before activation
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class GeneratedServiceSpec:
    """
    Full specification for an AI-generated service node.
    Includes schema, capability description, and confidence score.
    """

    def __init__(
        self,
        name: str,
        description: str,
        capability: str,
        input_fields: Dict[str, str],
        output_fields: Dict[str, str],
        required_inputs: List[str],
        tags: List[str],
        confidence: float = 0.0,
        gap_context: Optional[dict] = None,
    ):
        self.spec_id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.capability = capability
        self.input_fields = input_fields
        self.output_fields = output_fields
        self.required_inputs = required_inputs
        self.tags = tags
        self.confidence = confidence
        self.gap_context = gap_context or {}
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.status = "proposed"  # proposed / approved / rejected / active

    def to_dict(self) -> dict:
        return {
            "spec_id": self.spec_id,
            "name": self.name,
            "description": self.description,
            "capability": self.capability,
            "input_fields": self.input_fields,
            "output_fields": self.output_fields,
            "required_inputs": self.required_inputs,
            "tags": self.tags,
            "confidence": round(self.confidence, 4),
            "gap_context": self.gap_context,
            "created_at": self.created_at,
            "status": self.status,
        }


class ServiceGeneratorEngine:
    """
    Phase 5: Autonomous Service Generator.

    When the GapDetector identifies a missing capability between two nodes,
    this engine generates a complete service specification and optionally
    creates a live PassThroughNode with the correct semantic profile.
    """

    # Template patterns for common service types
    _SERVICE_TEMPLATES = {
        "transformer": {
            "tags": ["transform", "converter", "processing"],
            "input_fields": {"data": "Any", "format": "str"},
            "output_fields": {"result": "Any", "transformed": "bool"},
            "required_inputs": ["data"],
        },
        "normalizer": {
            "tags": ["normalize", "clean", "standardize"],
            "input_fields": {"raw_data": "Any", "schema": "dict"},
            "output_fields": {"normalized_data": "Any", "validation_report": "dict"},
            "required_inputs": ["raw_data"],
        },
        "aggregator": {
            "tags": ["aggregate", "combine", "merge"],
            "input_fields": {"items": "list", "strategy": "str"},
            "output_fields": {"aggregated": "Any", "count": "int"},
            "required_inputs": ["items"],
        },
        "validator": {
            "tags": ["validate", "check", "verify"],
            "input_fields": {"data": "Any", "rules": "dict"},
            "output_fields": {"valid": "bool", "errors": "list", "data": "Any"},
            "required_inputs": ["data"],
        },
        "enricher": {
            "tags": ["enrich", "augment", "enhance"],
            "input_fields": {"data": "Any", "context": "dict"},
            "output_fields": {"enriched_data": "Any", "added_fields": "list"},
            "required_inputs": ["data"],
        },
        "router": {
            "tags": ["route", "dispatch", "forward"],
            "input_fields": {"data": "Any", "routing_key": "str"},
            "output_fields": {"routed_data": "Any", "destination": "str"},
            "required_inputs": ["data"],
        },
        "filter": {
            "tags": ["filter", "select", "screen"],
            "input_fields": {"items": "list", "criteria": "dict"},
            "output_fields": {"filtered": "list", "removed_count": "int"},
            "required_inputs": ["items"],
        },
        "analyzer": {
            "tags": ["analyze", "inspect", "examine"],
            "input_fields": {"content": "Any", "analysis_type": "str"},
            "output_fields": {"analysis": "dict", "score": "float", "insights": "list"},
            "required_inputs": ["content"],
        },
    }

    def __init__(
        self,
        knowledge_store=None,
        semantic_matcher=None,
        governance=None,
    ):
        self._knowledge = knowledge_store
        self._semantic = semantic_matcher
        self._governance = governance
        self._generated: Dict[str, GeneratedServiceSpec] = {}
        self._generation_count = 0
        logger.info("ServiceGeneratorEngine initialised (Phase 5)")

    def set_knowledge_store(self, ks):
        self._knowledge = ks

    def set_semantic_matcher(self, sm):
        self._semantic = sm

    def set_governance(self, gov):
        self._governance = gov

    # ── Core generation logic ──────────────────────────────────────────────

    def generate_for_gap(self, gap: dict) -> Optional[GeneratedServiceSpec]:
        """
        Given a gap description (from GapDetector), generate a service spec.

        gap = {
            "source_node": {"name": ..., "capability": ...},
            "target_node": {"name": ..., "capability": ...},
            "missing_service": str,
            "confidence": float,
            "gap_type": str,
        }
        """
        missing_svc = gap.get("missing_service", "")
        source = gap.get("source_node", {})
        target = gap.get("target_node", {})

        # Determine service type from missing_service name and context
        svc_type = self._infer_service_type(missing_svc, source, target)
        template = self._SERVICE_TEMPLATES.get(svc_type, self._SERVICE_TEMPLATES["transformer"])

        # Build name and description
        name = self._build_name(missing_svc, source, target)
        description = self._build_description(name, source, target)
        capability = self._build_capability(svc_type, source, target)

        spec = GeneratedServiceSpec(
            name=name,
            description=description,
            capability=capability,
            input_fields=dict(template["input_fields"]),
            output_fields=dict(template["output_fields"]),
            required_inputs=list(template["required_inputs"]),
            tags=list(template["tags"]) + ["ai-generated", "phase5"],
            confidence=gap.get("confidence", 0.7),
            gap_context={
                "source_name": source.get("name", ""),
                "target_name": target.get("name", ""),
                "gap_type": gap.get("gap_type", ""),
                "missing_service": missing_svc,
            },
        )

        self._generated[spec.spec_id] = spec
        self._generation_count += 1
        logger.info(f"Generated service spec: '{name}' (type={svc_type}, confidence={spec.confidence:.2f})")

        # Persist to knowledge store
        self._persist_spec(spec)

        return spec

    def _infer_service_type(self, missing_svc: str, source: dict, target: dict) -> str:
        """Infer the best template type from context keywords."""
        combined = (
            missing_svc + " " +
            source.get("name", "") + " " + source.get("capability", "") + " " +
            target.get("name", "") + " " + target.get("capability", "")
        ).lower()

        priority = [
            ("normaliz", "normalizer"),
            ("validat", "validator"),
            ("aggregat", "aggregator"),
            ("enrich", "enricher"),
            ("filter", "filter"),
            ("rout", "router"),
            ("analyz", "analyzer"),
            ("analys", "analyzer"),
            ("transform", "transformer"),
            ("convert", "transformer"),
        ]
        for keyword, svc_type in priority:
            if keyword in combined:
                return svc_type
        return "transformer"

    def _build_name(self, missing_svc: str, source: dict, target: dict) -> str:
        """Build a clean CamelCase name for the generated service."""
        if missing_svc and missing_svc not in ("???", "unknown", ""):
            # Clean up and CamelCase
            parts = missing_svc.replace("-", " ").replace("_", " ").split()
            return "".join(p.capitalize() for p in parts if p)

        src_name = source.get("name", "Source").replace("Node", "").replace("Service", "")
        tgt_name = target.get("name", "Target").replace("Node", "").replace("Service", "")
        return f"{src_name}To{tgt_name}Bridge"

    def _build_description(self, name: str, source: dict, target: dict) -> str:
        src = source.get("name", "upstream service")
        tgt = target.get("name", "downstream service")
        return (
            f"AI-generated service '{name}' bridging {src} → {tgt}. "
            f"Auto-created by Phase 5 ServiceGenerator to fill detected capability gap."
        )

    def _build_capability(self, svc_type: str, source: dict, target: dict) -> str:
        src_cap = source.get("capability", "process data")
        tgt_cap = target.get("capability", "receive data")
        return f"{svc_type} data from ({src_cap}) to ({tgt_cap})"

    def _persist_spec(self, spec: GeneratedServiceSpec):
        """Save generated spec to knowledge store."""
        if not self._knowledge:
            return
        try:
            # Load existing generated services JSON or create new
            generated_key = "generated_services"
            existing = {}
            try:
                raw = self._knowledge.read_custom(generated_key)
                existing = raw if isinstance(raw, dict) else {}
            except Exception:
                existing = {}

            existing[spec.spec_id] = spec.to_dict()
            self._knowledge.write_custom(generated_key, existing)
        except Exception as e:
            logger.warning(f"Could not persist spec: {e}")

    # ── Instantiation ──────────────────────────────────────────────────────

    def instantiate_spec(self, spec: GeneratedServiceSpec):
        """
        Create a live PassThroughNode from a GeneratedServiceSpec.
        The node inherits the AI-generated semantic profile.
        Returns the live node object.
        """
        from services.dynamic_node import PassThroughNode

        node = PassThroughNode(
            name=spec.name,
            description=spec.description,
            tags=spec.tags,
        )
        spec.status = "active"
        logger.info(f"Instantiated AI-generated node: '{spec.name}' [{node.node_id[:8]}]")
        return node

    # ── Status & listing ───────────────────────────────────────────────────

    def list_generated(self, status_filter: Optional[str] = None) -> List[dict]:
        specs = list(self._generated.values())
        if status_filter:
            specs = [s for s in specs if s.status == status_filter]
        return [s.to_dict() for s in specs]

    def get_spec(self, spec_id: str) -> Optional[GeneratedServiceSpec]:
        return self._generated.get(spec_id)

    def approve_spec(self, spec_id: str) -> bool:
        spec = self._generated.get(spec_id)
        if spec:
            spec.status = "approved"
            return True
        return False

    def reject_spec(self, spec_id: str) -> bool:
        spec = self._generated.get(spec_id)
        if spec:
            spec.status = "rejected"
            return True
        return False

    def summary(self) -> dict:
        statuses = {}
        for s in self._generated.values():
            statuses[s.status] = statuses.get(s.status, 0) + 1
        return {
            "total_generated": self._generation_count,
            "in_memory": len(self._generated),
            "by_status": statuses,
        }

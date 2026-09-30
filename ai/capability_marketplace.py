"""
Phase 5 – Capability Marketplace
===================================
Every node declares its capabilities using a standardized vocabulary.
The system searches by CAPABILITY, not node name.

Instead of:
  "route to TextProcessor"
The system asks:
  "who can translate_text?"

Each node advertises:
  { "capability": "translate_text", "quality": 0.92, "latency_ms": 45 }

The marketplace maintains a live index and routes to the best provider.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CapabilityAdvertisement:
    """A node's declaration of what it can do."""

    def __init__(
        self,
        node_id: str,
        node_name: str,
        capability: str,
        quality_score: float = 0.8,
        avg_latency_ms: float = 100.0,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
    ):
        self.node_id = node_id
        self.node_name = node_name
        self.capability = capability.lower().replace(" ", "_")
        self.quality_score = min(1.0, max(0.0, quality_score))
        self.avg_latency_ms = avg_latency_ms
        self.tags = tags or []
        self.metadata = metadata or {}
        self.advertised_at = datetime.now(timezone.utc).isoformat()
        self.last_updated = self.advertised_at
        self.execution_count: int = 0
        self.is_active: bool = True

    @property
    def composite_score(self) -> float:
        """Combined quality+speed score for ranking."""
        latency_factor = max(0.0, 1.0 - (self.avg_latency_ms / 5000.0))
        return self.quality_score * 0.7 + latency_factor * 0.3

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "capability": self.capability,
            "quality_score": round(self.quality_score, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "composite_score": round(self.composite_score, 4),
            "tags": self.tags,
            "metadata": self.metadata,
            "advertised_at": self.advertised_at,
            "last_updated": self.last_updated,
            "execution_count": self.execution_count,
            "is_active": self.is_active,
        }


class CapabilityMarketplace:
    """
    Phase 5: Capability-based service discovery and routing.

    The Marketplace maintains an index of capability → [providers].
    When the system needs a capability, it queries the marketplace
    and gets the best available provider ranked by composite score.

    This decouples routing from node names entirely.
    """

    def __init__(self, knowledge_store=None):
        # capability -> list of advertisements
        self._index: Dict[str, List[CapabilityAdvertisement]] = {}
        # node_id -> list of capabilities it provides
        self._node_capabilities: Dict[str, List[str]] = {}
        self._knowledge = knowledge_store
        self._query_count = 0
        logger.info("CapabilityMarketplace initialised (Phase 5)")

    def set_knowledge_store(self, ks):
        self._knowledge = ks

    # ── Registration ───────────────────────────────────────────────────────

    def advertise(
        self,
        node_id: str,
        node_name: str,
        capability: str,
        quality_score: float = 0.8,
        avg_latency_ms: float = 100.0,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
    ) -> CapabilityAdvertisement:
        """Register a node's capability in the marketplace."""
        ad = CapabilityAdvertisement(
            node_id=node_id,
            node_name=node_name,
            capability=capability,
            quality_score=quality_score,
            avg_latency_ms=avg_latency_ms,
            tags=tags,
            metadata=metadata,
        )

        cap_key = ad.capability
        if cap_key not in self._index:
            self._index[cap_key] = []

        # Replace existing advertisement from same node
        self._index[cap_key] = [
            x for x in self._index[cap_key] if x.node_id != node_id
        ]
        self._index[cap_key].append(ad)

        if node_id not in self._node_capabilities:
            self._node_capabilities[node_id] = []
        if cap_key not in self._node_capabilities[node_id]:
            self._node_capabilities[node_id].append(cap_key)

        self._persist()
        logger.info(f"Marketplace: '{node_name}' advertised capability '{cap_key}'")
        return ad

    def advertise_from_node(self, node) -> List[CapabilityAdvertisement]:
        """
        Auto-extract and advertise capabilities from a BaseNode.
        Derives capability from description, tags, and name.
        """
        ads = []
        capabilities = self._extract_capabilities(node)
        for cap in capabilities:
            ad = self.advertise(
                node_id=node.node_id,
                node_name=node.name,
                capability=cap,
                tags=node.tags,
            )
            ads.append(ad)
        return ads

    def _extract_capabilities(self, node) -> List[str]:
        """Derive capability tokens from a node's metadata."""
        caps = []
        name_lower = node.name.lower()
        desc_lower = node.description.lower()
        combined = name_lower + " " + desc_lower

        # Keyword → capability mapping (order matters: most specific first)
        keyword_caps = [
            ("sentiment", "sentiment"),
            ("translat", "translate"),
            ("clean", "clean"),
            ("normaliz", "normalize"),
            ("validat", "validate"),
            ("aggregat", "aggregate"),
            ("enrich", "enrich"),
            ("filter", "filter"),
            ("format", "format"),
            ("summar", "summarize"),
            ("classif", "classify"),
            ("analyz", "analyze"),
            ("analys", "analyze"),
            ("transform", "transform"),
            ("convert", "transform"),
            ("process", "process"),
            ("input", "input"),
            ("output", "output"),
            ("route", "route"),
        ]

        for keyword, cap in keyword_caps:
            if keyword in combined:
                caps.append(cap)

        # Also use tags
        for tag in node.tags:
            normalized = tag.lower().replace("-", "_").replace(" ", "_")
            if normalized not in ("dynamic", "passthrough", "api", "phase5", "ai-generated"):
                caps.append(normalized)

        # Fallback: use node name cleaned
        if not caps:
            caps.append(name_lower.replace(" ", "_"))

        return list(dict.fromkeys(caps))  # deduplicate while preserving order

    # ── Discovery ──────────────────────────────────────────────────────────

    def find_providers(
        self,
        capability: str,
        top_k: int = 5,
        exclude_nodes: Optional[List[str]] = None,
    ) -> List[CapabilityAdvertisement]:
        """
        Find the best providers for a given capability.
        Returns ranked list (best first) by composite_score.
        """
        self._query_count += 1
        cap_key = capability.lower().replace(" ", "_")
        exclude = set(exclude_nodes or [])

        # Exact match
        providers = [
            ad for ad in self._index.get(cap_key, [])
            if ad.is_active and ad.node_id not in exclude
        ]

        # Fuzzy match if no exact match
        if not providers:
            for indexed_cap, ads in self._index.items():
                if cap_key in indexed_cap or indexed_cap in cap_key:
                    providers.extend([
                        ad for ad in ads
                        if ad.is_active and ad.node_id not in exclude
                    ])

        # Sort by composite score descending
        providers.sort(key=lambda x: x.composite_score, reverse=True)
        return providers[:top_k]

    def best_provider(
        self,
        capability: str,
        exclude_nodes: Optional[List[str]] = None,
    ) -> Optional[CapabilityAdvertisement]:
        """Return the single best provider for a capability."""
        providers = self.find_providers(capability, top_k=1, exclude_nodes=exclude_nodes)
        return providers[0] if providers else None

    def list_capabilities(self) -> List[str]:
        """List all advertised capabilities."""
        return sorted(self._index.keys())

    def capabilities_for_node(self, node_id: str) -> List[str]:
        """List all capabilities a specific node provides."""
        return self._node_capabilities.get(node_id, [])

    def all_advertisements(self) -> List[dict]:
        """Return all active advertisements."""
        result = []
        for ads in self._index.values():
            for ad in ads:
                if ad.is_active:
                    result.append(ad.to_dict())
        return result

    # ── Feedback & scoring ─────────────────────────────────────────────────

    def record_execution(
        self,
        node_id: str,
        capability: str,
        success: bool,
        latency_ms: float,
    ):
        """Update quality scores based on execution results.

        قبل هذا التعديل لم تكن تستدعي _persist() إطلاقاً — أي أن كل ما
        توثّقه restore() أعلاه عن 'الحفاظ على درجات الجودة/الكمون
        المتراكمة من record_execution()' لم يكن صحيحاً فعلياً: advertise()
        وحدها كانت تكتب على القرص، فالتحديثات الحقيقية القادمة من
        plan_and_execute_goal (أو أي مستدعٍ آخر لـrecord_execution) تبقى
        في الذاكرة فقط وتُفقَد عند إعادة التشغيل التالية، حتى لو استُعيد
        الفهرس نفسه بنجاح من نسخة advertise() الأصلية القديمة."""
        cap_key = capability.lower().replace(" ", "_")
        changed = False
        for ad in self._index.get(cap_key, []):
            if ad.node_id == node_id:
                ad.execution_count += 1
                # Exponential moving average for latency
                alpha = 0.2
                ad.avg_latency_ms = alpha * latency_ms + (1 - alpha) * ad.avg_latency_ms
                # Update quality score
                if success:
                    ad.quality_score = min(1.0, ad.quality_score + 0.01)
                else:
                    ad.quality_score = max(0.1, ad.quality_score - 0.05)
                ad.last_updated = datetime.now(timezone.utc).isoformat()
                changed = True
        if changed:
            self._persist()

    def deactivate_node(self, node_id: str):
        """Mark all advertisements for a node as inactive."""
        changed = False
        for ads in self._index.values():
            for ad in ads:
                if ad.node_id == node_id:
                    ad.is_active = False
                    changed = True
        if changed:
            self._persist()

    def reactivate_node(self, node_id: str):
        """Reactivate all advertisements for a node."""
        changed = False
        for ads in self._index.values():
            for ad in ads:
                if ad.node_id == node_id:
                    ad.is_active = True
                    changed = True
        if changed:
            self._persist()

    # ── Persistence ────────────────────────────────────────────────────────

    def restore(self) -> int:
        """أعِد بناء الفهرس من آخر نسخة محفوظة عبر KnowledgeStore.write_custom
        (الدالة أعلاه) بعد إعادة تشغيل العملية. قبل هذا التعديل لم يكن أي
        كائن في المشروع يستدعي read_custom("capability_marketplace") أبداً
        رغم أن _persist() تكتب snapshot كامل فعلياً على القرص عند كل
        advertise() — فيبدأ المتجر فارغاً من جديد بعد كل إعادة تشغيل،
        وتُفقَد كل درجات الجودة/الكمون المتراكمة من record_execution() ولو
        تراكمت آلاف التنفيذات الحقيقية. يُعيد عدد الإعلانات المُستعادة
        فعلياً (0 لو لا يوجد ملف محفوظ بعد، أو لا KnowledgeStore أصلاً)."""
        if not self._knowledge:
            return 0
        try:
            snapshot = self._knowledge.read_custom("capability_marketplace")
        except KeyError:
            return 0
        except Exception as e:
            logger.warning(f"Marketplace restore error: {e}")
            return 0
        if not isinstance(snapshot, dict):
            return 0

        restored = 0
        for cap_key, ads in snapshot.items():
            if not isinstance(ads, list):
                continue
            rebuilt: List[CapabilityAdvertisement] = []
            for ad_dict in ads:
                try:
                    ad = CapabilityAdvertisement(
                        node_id=ad_dict["node_id"],
                        node_name=ad_dict.get("node_name", ""),
                        capability=ad_dict.get("capability", cap_key),
                        quality_score=ad_dict.get("quality_score", 0.8),
                        avg_latency_ms=ad_dict.get("avg_latency_ms", 100.0),
                        tags=ad_dict.get("tags"),
                        metadata=ad_dict.get("metadata"),
                    )
                    # الحقول المتراكمة عبر الزمن (لا تُعاد بقيمة ابتدائية)
                    ad.execution_count = ad_dict.get("execution_count", 0)
                    ad.is_active = ad_dict.get("is_active", True)
                    ad.advertised_at = ad_dict.get("advertised_at", ad.advertised_at)
                    ad.last_updated = ad_dict.get("last_updated", ad.last_updated)
                except (KeyError, TypeError):
                    continue
                rebuilt.append(ad)
                node_caps = self._node_capabilities.setdefault(ad.node_id, [])
                if ad.capability not in node_caps:
                    node_caps.append(ad.capability)
                restored += 1
            if rebuilt:
                self._index[cap_key] = rebuilt

        if restored:
            logger.info(
                f"CapabilityMarketplace: restored {restored} capability "
                f"advertisement(s) across {len(self._index)} capability keys"
            )
        return restored

    def _persist(self):
        """Persist marketplace index to knowledge store."""
        if not self._knowledge:
            return
        try:
            snapshot = {
                cap: [ad.to_dict() for ad in ads]
                for cap, ads in self._index.items()
            }
            self._knowledge.write_custom("capability_marketplace", snapshot)
        except Exception as e:
            logger.warning(f"Marketplace persist error: {e}")

    # ── Summary ────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        total_ads = sum(len(ads) for ads in self._index.values())
        active_ads = sum(
            1 for ads in self._index.values()
            for ad in ads if ad.is_active
        )
        return {
            "total_capabilities": len(self._index),
            "total_advertisements": total_ads,
            "active_advertisements": active_ads,
            "registered_nodes": len(self._node_capabilities),
            "query_count": self._query_count,
            "capabilities": self.list_capabilities(),
        }

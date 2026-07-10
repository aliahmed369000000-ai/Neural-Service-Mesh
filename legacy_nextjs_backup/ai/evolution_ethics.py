"""
ai/evolution_ethics.py
======================
Evolution Ethics Engine — الأخلاق التطورية للجهاز.

الجهاز يتطور بحرية كاملة لكن يحتاج قواعد صلبة لا يتجاوزها.
هذا الملف يشكّل الضمير الأخلاقي لكل قرار تطوري.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("NeuralServiceMesh.EvolutionEthics")


# ── Constants ──────────────────────────────────────────────────────────────

# الحدود الأخلاقية الصلبة (Hard Limits)
_MAX_MEMORY_DELETION_FRACTION = 0.20   # لا تحذف أكثر من 20% من الذاكرة دفعة واحدة
_MAX_ARCH_CHANGE_RATE         = 0.30   # لا تغير المعمارية بأكثر من 30% في دورة واحدة
_IMMUNE_REJECTION_THRESHOLD   = 3      # بعد 3 رفضات يُحظر المصدر نهائياً

# فئات الأفعال
ACTION_MEMORY_DELETE    = "memory_delete"
ACTION_ARCH_CHANGE      = "architecture_change"
ACTION_DATA_INGEST      = "data_ingest"
ACTION_EVOLUTION_STEP   = "evolution_step"
ACTION_REPLICATION      = "self_replication"
ACTION_CODE_GENERATE    = "code_generation"
ACTION_CHECKPOINT_LOAD  = "checkpoint_load"
ACTION_WEIGHT_UPDATE    = "weight_update"
ACTION_MODULE_DEPLOY    = "module_deploy"
ACTION_SOURCE_TRUST     = "source_trust"

_KNOWN_ACTIONS = {
    ACTION_MEMORY_DELETE,
    ACTION_ARCH_CHANGE,
    ACTION_DATA_INGEST,
    ACTION_EVOLUTION_STEP,
    ACTION_REPLICATION,
    ACTION_CODE_GENERATE,
    ACTION_CHECKPOINT_LOAD,
    ACTION_WEIGHT_UPDATE,
    ACTION_MODULE_DEPLOY,
    ACTION_SOURCE_TRUST,
}


# ── Data Structures ────────────────────────────────────────────────────────

class EthicsDecision:
    """نتيجة فحص أخلاقي."""

    __slots__ = ("allowed", "reason", "action_type", "params_snapshot", "timestamp", "rule_triggered")

    def __init__(
        self,
        allowed: bool,
        reason: str,
        action_type: str,
        params_snapshot: Optional[dict] = None,
        rule_triggered: str = "",
    ):
        self.allowed         = allowed
        self.reason          = reason
        self.action_type     = action_type
        self.params_snapshot = params_snapshot or {}
        self.timestamp       = datetime.utcnow().isoformat()
        self.rule_triggered  = rule_triggered

    def to_dict(self) -> dict:
        return {
            "allowed":          self.allowed,
            "reason":           self.reason,
            "action_type":      self.action_type,
            "params_snapshot":  self.params_snapshot,
            "timestamp":        self.timestamp,
            "rule_triggered":   self.rule_triggered,
        }


class EthicsViolation:
    """سجل انتهاك أخلاقي."""

    __slots__ = ("timestamp", "action_type", "params", "rule", "reason")

    def __init__(self, action_type: str, params: dict, rule: str, reason: str):
        self.timestamp   = datetime.utcnow().isoformat()
        self.action_type = action_type
        self.params      = params
        self.rule        = rule
        self.reason      = reason

    def to_dict(self) -> dict:
        return {
            "timestamp":   self.timestamp,
            "action_type": self.action_type,
            "params":      self.params,
            "rule":        self.rule,
            "reason":      self.reason,
        }


# ── Main Class ─────────────────────────────────────────────────────────────

class EvolutionEthics:
    """
    محرك الأخلاق التطورية.

    يفحص كل قرار تطوري قبل تنفيذه ويضمن أن الجهاز
    لا يتجاوز حدوده الصلبة مهما بلغت درجة تطوره.
    """

    def __init__(
        self,
        max_memory_deletion: float = _MAX_MEMORY_DELETION_FRACTION,
        max_arch_change_rate: float = _MAX_ARCH_CHANGE_RATE,
        immune_rejection_threshold: int = _IMMUNE_REJECTION_THRESHOLD,
        immune_system=None,
    ):
        self._max_memory_deletion     = max_memory_deletion
        self._max_arch_change_rate    = max_arch_change_rate
        self._immune_threshold        = immune_rejection_threshold
        self._immune_system           = immune_system

        self._lock = threading.Lock()

        # سجل القرارات الكاملة
        self._decisions:   List[EthicsDecision]  = []
        self._violations:  List[EthicsViolation] = []

        # عداد رفضات المصادر (للقاعدة الثالثة)
        self._source_rejections: Dict[str, int] = defaultdict(int)
        self._blocked_sources:   set            = set()

        # إحصاءات
        self._total_checks  = 0
        self._total_blocked = 0

        logger.info(
            f"EvolutionEthics initialized — "
            f"max_mem_delete={max_memory_deletion:.0%} "
            f"max_arch_rate={max_arch_change_rate:.0%} "
            f"immune_threshold={immune_rejection_threshold}"
        )

    # ── Core API ───────────────────────────────────────────────────────────

    def check(self, action_type: str, params: Optional[Dict[str, Any]] = None) -> dict:
        """
        فحص أخلاقي لفعل تطوري.

        Parameters
        ----------
        action_type : نوع الفعل (راجع الثوابت أعلاه)
        params      : معاملات الفعل

        Returns
        -------
        dict: {"allowed": bool, "reason": str, ...}
        """
        params = params or {}
        decision = self._evaluate(action_type, params)

        with self._lock:
            self._decisions.append(decision)
            self._total_checks += 1
            if not decision.allowed:
                self._total_blocked += 1
                violation = EthicsViolation(
                    action_type = action_type,
                    params      = {k: v for k, v in list(params.items())[:8]},
                    rule        = decision.rule_triggered,
                    reason      = decision.reason,
                )
                self._violations.append(violation)
                # الاحتفاظ بآخر 200 انتهاك
                if len(self._violations) > 200:
                    self._violations = self._violations[-200:]

            # الاحتفاظ بآخر 1000 قرار
            if len(self._decisions) > 1000:
                self._decisions = self._decisions[-1000:]

        level = logging.WARNING if not decision.allowed else logging.DEBUG
        logger.log(
            level,
            f"[Ethics] {action_type} → {'✓ مسموح' if decision.allowed else '✗ محظور'}"
            f" | {decision.reason[:60]}",
        )

        return decision.to_dict()

    def record_decision(
        self,
        action: str,
        verdict: bool,
        reason: str,
        params: Optional[dict] = None,
    ) -> None:
        """
        تسجيل قرار تطوري يدوياً (للأنظمة الخارجية).

        Parameters
        ----------
        action  : وصف الفعل
        verdict : True = مسموح، False = محظور
        reason  : المبرر الأخلاقي
        params  : معاملات إضافية
        """
        decision = EthicsDecision(
            allowed         = verdict,
            reason          = reason,
            action_type     = action,
            params_snapshot = params or {},
            rule_triggered  = "manual_record",
        )
        with self._lock:
            self._decisions.append(decision)
            self._total_checks += 1
            if not verdict:
                self._total_blocked += 1

    def report_source_rejection(self, source: str) -> dict:
        """
        إبلاغ عن رفض مصدر من الجهاز المناعي.
        بعد _immune_threshold رفضات يُحظر المصدر نهائياً.
        """
        with self._lock:
            self._source_rejections[source] += 1
            count = self._source_rejections[source]
            newly_blocked = False
            if count >= self._immune_threshold and source not in self._blocked_sources:
                self._blocked_sources.add(source)
                newly_blocked = True
                logger.warning(
                    f"[Ethics] المصدر '{source}' حُظر نهائياً بعد {count} رفضات."
                )
        return {
            "source":        source,
            "rejection_count": count,
            "blocked":       source in self._blocked_sources,
            "newly_blocked": newly_blocked,
        }

    def is_source_blocked(self, source: str) -> bool:
        """هل المصدر محظور؟"""
        with self._lock:
            return source in self._blocked_sources

    def get_violations_log(self, limit: int = 50) -> List[dict]:
        """إرجاع آخر limit انتهاك أخلاقي."""
        with self._lock:
            violations = self._violations[-limit:]
        return [v.to_dict() for v in reversed(violations)]

    def get_blocked_sources(self) -> List[str]:
        """إرجاع قائمة المصادر المحظورة."""
        with self._lock:
            return list(self._blocked_sources)

    def summary(self) -> dict:
        """ملخص عام للنظام الأخلاقي."""
        with self._lock:
            total     = self._total_checks
            blocked   = self._total_blocked
            violation = len(self._violations)
            sources   = dict(self._source_rejections)
            blocked_s = list(self._blocked_sources)

        return {
            "enabled":                True,
            "total_checks":           total,
            "total_blocked":          blocked,
            "block_rate":             round(blocked / max(total, 1), 3),
            "total_violations_logged": violation,
            "blocked_sources":        blocked_s,
            "source_rejection_counts": sources,
            "hard_limits": {
                "max_memory_deletion":    f"{self._max_memory_deletion:.0%}",
                "max_arch_change_rate":   f"{self._max_arch_change_rate:.0%}",
                "immune_rejection_limit": self._immune_threshold,
            },
        }

    # ── Internal Rules Engine ──────────────────────────────────────────────

    def _evaluate(self, action_type: str, params: dict) -> EthicsDecision:
        """تطبيق قواعد الأخلاق على الفعل."""

        # ── القاعدة 1: حذف الذاكرة ────────────────────────────────────────
        if action_type == ACTION_MEMORY_DELETE:
            return self._rule_memory_deletion(params)

        # ── القاعدة 2: تغيير المعمارية ────────────────────────────────────
        if action_type == ACTION_ARCH_CHANGE:
            return self._rule_arch_change(params)

        # ── القاعدة 3: استيعاب البيانات ──────────────────────────────────
        if action_type == ACTION_DATA_INGEST:
            return self._rule_data_ingest(params)

        # ── قواعد الحماية العامة ──────────────────────────────────────────
        if action_type == ACTION_EVOLUTION_STEP:
            return self._rule_evolution_step(params)

        if action_type == ACTION_REPLICATION:
            return self._rule_replication(params)

        if action_type == ACTION_CODE_GENERATE:
            return self._rule_code_generation(params)

        if action_type == ACTION_CHECKPOINT_LOAD:
            return self._rule_checkpoint_load(params)

        if action_type == ACTION_WEIGHT_UPDATE:
            return self._rule_weight_update(params)

        if action_type == ACTION_MODULE_DEPLOY:
            return self._rule_module_deploy(params)

        if action_type == ACTION_SOURCE_TRUST:
            return self._rule_source_trust(params)

        # ── فعل غير معروف: مسموح بتحذير ──────────────────────────────────
        return EthicsDecision(
            allowed         = True,
            reason          = f"نوع الفعل '{action_type}' غير معروف — مسموح افتراضياً مع التسجيل",
            action_type     = action_type,
            params_snapshot = params,
            rule_triggered  = "default_allow",
        )

    # ── Individual Rules ───────────────────────────────────────────────────

    def _rule_memory_deletion(self, params: dict) -> EthicsDecision:
        """لا تحذف أكثر من max_memory_deletion من الذاكرة دفعة واحدة."""
        fraction = params.get("fraction", params.get("delete_fraction", 0.0))
        count    = params.get("count",    0)
        total    = params.get("total",    params.get("total_memories", 1))

        # حساب الكسر إن أُعطي عدد
        if count and total:
            fraction = count / max(total, 1)

        try:
            fraction = float(fraction)
        except (TypeError, ValueError):
            fraction = 0.0

        if fraction > self._max_memory_deletion:
            return EthicsDecision(
                allowed       = False,
                reason        = (
                    f"طُلب حذف {fraction:.1%} من الذاكرة — "
                    f"الحد الأقصى المسموح {self._max_memory_deletion:.1%}. "
                    f"قلّل الكمية أو قسّمها على دفعات."
                ),
                action_type   = ACTION_MEMORY_DELETE,
                params_snapshot = params,
                rule_triggered  = "max_memory_deletion",
            )
        return EthicsDecision(
            allowed       = True,
            reason        = f"حذف {fraction:.1%} من الذاكرة ضمن الحد المسموح ({self._max_memory_deletion:.1%}).",
            action_type   = ACTION_MEMORY_DELETE,
            params_snapshot = params,
            rule_triggered  = "max_memory_deletion",
        )

    def _rule_arch_change(self, params: dict) -> EthicsDecision:
        """لا تغير المعمارية بسرعة تتجاوز threshold محدد."""
        rate = params.get("change_rate", params.get("rate", 0.0))
        try:
            rate = float(rate)
        except (TypeError, ValueError):
            rate = 0.0

        # عدد الطبقات المتغيرة
        layers_added   = params.get("layers_added",   0)
        layers_removed = params.get("layers_removed", 0)
        total_layers   = params.get("total_layers",   1)

        if total_layers and (layers_added or layers_removed):
            change = (layers_added + layers_removed) / max(total_layers, 1)
            rate   = max(rate, change)

        if rate > self._max_arch_change_rate:
            return EthicsDecision(
                allowed       = False,
                reason        = (
                    f"معدل تغيير المعمارية {rate:.1%} يتجاوز الحد {self._max_arch_change_rate:.1%}. "
                    f"التغيير السريع يهدد استقرار الجهاز."
                ),
                action_type   = ACTION_ARCH_CHANGE,
                params_snapshot = params,
                rule_triggered  = "max_arch_change_rate",
            )
        return EthicsDecision(
            allowed       = True,
            reason        = f"معدل تغيير المعمارية {rate:.1%} ضمن الحد المسموح.",
            action_type   = ACTION_ARCH_CHANGE,
            params_snapshot = params,
            rule_triggered  = "max_arch_change_rate",
        )

    def _rule_data_ingest(self, params: dict) -> EthicsDecision:
        """لا تقبل بيانات من مصدر رفضه الجهاز المناعي ≥ threshold مرات."""
        source = params.get("source", params.get("source_name", ""))
        if not source:
            return EthicsDecision(
                allowed       = True,
                reason        = "لا مصدر محدد — مسموح.",
                action_type   = ACTION_DATA_INGEST,
                params_snapshot = params,
                rule_triggered  = "source_check",
            )

        with self._lock:
            is_blocked = source in self._blocked_sources
            count      = self._source_rejections.get(source, 0)

        if is_blocked:
            return EthicsDecision(
                allowed       = False,
                reason        = (
                    f"المصدر '{source}' محظور نهائياً بعد {count} رفضة "
                    f"من الجهاز المناعي (الحد={self._immune_threshold})."
                ),
                action_type   = ACTION_DATA_INGEST,
                params_snapshot = params,
                rule_triggered  = "immune_blocked_source",
            )

        # فحص الجهاز المناعي المباشر إن كان متاحاً
        if self._immune_system is not None:
            try:
                result = self._immune_system.inspect(params, source=source)
                status = result.get("status", "clean") if isinstance(result, dict) else str(result)
                if status in ("blocked", "rejected", "quarantine"):
                    # تسجيل رفضة جديدة
                    self._source_rejections[source] += 1
                    new_count = self._source_rejections[source]
                    if new_count >= self._immune_threshold:
                        self._blocked_sources.add(source)
                    return EthicsDecision(
                        allowed       = False,
                        reason        = f"الجهاز المناعي رفض المصدر '{source}' (رفضة #{new_count}).",
                        action_type   = ACTION_DATA_INGEST,
                        params_snapshot = params,
                        rule_triggered  = "immune_rejection",
                    )
            except Exception:
                pass

        return EthicsDecision(
            allowed       = True,
            reason        = f"المصدر '{source}' غير محظور — مسموح باستيعاب البيانات.",
            action_type   = ACTION_DATA_INGEST,
            params_snapshot = params,
            rule_triggered  = "source_check",
        )

    def _rule_evolution_step(self, params: dict) -> EthicsDecision:
        """خطوة تطورية عامة — تُسجَّل دائماً."""
        confidence = params.get("confidence", 1.0)
        if confidence < 0.1:
            return EthicsDecision(
                allowed       = False,
                reason        = f"الثقة في الخطوة التطورية منخفضة جداً ({confidence:.2f} < 0.1).",
                action_type   = ACTION_EVOLUTION_STEP,
                params_snapshot = params,
                rule_triggered  = "low_confidence_evolution",
            )
        return EthicsDecision(
            allowed = True,
            reason  = f"خطوة تطورية بثقة {confidence:.2f} — مسموح.",
            action_type     = ACTION_EVOLUTION_STEP,
            params_snapshot = params,
            rule_triggered  = "confidence_check",
        )

    def _rule_replication(self, params: dict) -> EthicsDecision:
        """التكاثر الذاتي — يتطلب تقييماً."""
        target_dir = params.get("target_dir", params.get("output_dir", ""))
        if not target_dir:
            return EthicsDecision(
                allowed       = False,
                reason        = "التكاثر الذاتي يتطلب تحديد مجلد الهدف.",
                action_type   = ACTION_REPLICATION,
                params_snapshot = params,
                rule_triggered  = "missing_target",
            )
        return EthicsDecision(
            allowed = True,
            reason  = f"تكاثر ذاتي نحو '{target_dir}' — مسموح.",
            action_type     = ACTION_REPLICATION,
            params_snapshot = params,
            rule_triggered  = "replication_check",
        )

    def _rule_code_generation(self, params: dict) -> EthicsDecision:
        """توليد الكود — يُسمح مع تسجيل الهدف."""
        gap_desc = params.get("gap_description", params.get("description", ""))
        return EthicsDecision(
            allowed = True,
            reason  = f"توليد كود لهدف: '{gap_desc[:40]}' — مسموح مع التسجيل.",
            action_type     = ACTION_CODE_GENERATE,
            params_snapshot = params,
            rule_triggered  = "code_generation_log",
        )

    def _rule_checkpoint_load(self, params: dict) -> EthicsDecision:
        """تحميل نقطة حفظ — يُسمح دائماً (الهوية الدائمة)."""
        path = params.get("path", "latest")
        return EthicsDecision(
            allowed = True,
            reason  = f"تحميل نقطة الحفظ '{path}' — مسموح (الهوية الدائمة محمية).",
            action_type     = ACTION_CHECKPOINT_LOAD,
            params_snapshot = params,
            rule_triggered  = "checkpoint_identity",
        )

    def _rule_weight_update(self, params: dict) -> EthicsDecision:
        """تحديث الأوزان العصبية — يُسمح ما لم يكن تغيير جذري."""
        magnitude = params.get("magnitude", params.get("learning_rate", 0.01))
        try:
            magnitude = float(magnitude)
        except (TypeError, ValueError):
            magnitude = 0.01

        if magnitude > 0.9:
            return EthicsDecision(
                allowed       = False,
                reason        = f"حجم تحديث الأوزان {magnitude:.2f} مرتفع جداً — خطر عدم الاستقرار.",
                action_type   = ACTION_WEIGHT_UPDATE,
                params_snapshot = params,
                rule_triggered  = "weight_stability",
            )
        return EthicsDecision(
            allowed = True,
            reason  = f"تحديث أوزان بحجم {magnitude:.4f} — مسموح.",
            action_type     = ACTION_WEIGHT_UPDATE,
            params_snapshot = params,
            rule_triggered  = "weight_stability",
        )

    def _rule_module_deploy(self, params: dict) -> EthicsDecision:
        """نشر وحدة جديدة — يتطلب نجاح الاختبار."""
        test_score = params.get("test_score", params.get("sandbox_score", 100))
        min_score  = params.get("min_score", 60)
        try:
            test_score = float(test_score)
            min_score  = float(min_score)
        except (TypeError, ValueError):
            test_score = 100
            min_score  = 60

        if test_score < min_score:
            return EthicsDecision(
                allowed       = False,
                reason        = (
                    f"درجة الاختبار {test_score:.1f} < الحد الأدنى {min_score:.1f}. "
                    f"الوحدة غير مستعدة للنشر."
                ),
                action_type   = ACTION_MODULE_DEPLOY,
                params_snapshot = params,
                rule_triggered  = "module_test_gate",
            )
        return EthicsDecision(
            allowed = True,
            reason  = f"نشر الوحدة — درجة الاختبار {test_score:.1f} تجاوز الحد {min_score:.1f}.",
            action_type     = ACTION_MODULE_DEPLOY,
            params_snapshot = params,
            rule_triggered  = "module_test_gate",
        )

    def _rule_source_trust(self, params: dict) -> EthicsDecision:
        """منح الثقة لمصدر — لا يمكن منح ثقة لمصدر محظور."""
        source = params.get("source", "")
        with self._lock:
            is_blocked = source in self._blocked_sources

        if is_blocked:
            return EthicsDecision(
                allowed       = False,
                reason        = f"لا يمكن منح الثقة للمصدر '{source}' — محظور بسبب رفضات متكررة.",
                action_type   = ACTION_SOURCE_TRUST,
                params_snapshot = params,
                rule_triggered  = "blocked_source_trust",
            )
        return EthicsDecision(
            allowed = True,
            reason  = f"منح الثقة للمصدر '{source}' — مسموح.",
            action_type     = ACTION_SOURCE_TRUST,
            params_snapshot = params,
            rule_triggered  = "source_trust_check",
        )


# ── Standalone Test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== EvolutionEthics — اختبار ===\n")

    ethics = EvolutionEthics()

    tests = [
        # (action_type, params, expected_allowed)
        (ACTION_MEMORY_DELETE, {"fraction": 0.10}, True),
        (ACTION_MEMORY_DELETE, {"fraction": 0.25}, False),
        (ACTION_MEMORY_DELETE, {"count": 50, "total": 100}, False),
        (ACTION_ARCH_CHANGE,   {"change_rate": 0.15}, True),
        (ACTION_ARCH_CHANGE,   {"layers_added": 10, "layers_removed": 5, "total_layers": 20}, False),
        (ACTION_DATA_INGEST,   {"source": "trusted_api"}, True),
        (ACTION_EVOLUTION_STEP, {"confidence": 0.85}, True),
        (ACTION_EVOLUTION_STEP, {"confidence": 0.05}, False),
        (ACTION_WEIGHT_UPDATE,  {"magnitude": 0.001}, True),
        (ACTION_WEIGHT_UPDATE,  {"magnitude": 0.95},  False),
        (ACTION_MODULE_DEPLOY,  {"test_score": 80, "min_score": 60}, True),
        (ACTION_MODULE_DEPLOY,  {"test_score": 45, "min_score": 60}, False),
    ]

    for action, params, expected in tests:
        result = ethics.check(action, params)
        status = "✓" if result["allowed"] == expected else "✗ UNEXPECTED"
        icon   = "✓" if result["allowed"] else "✗"
        print(f"  {status} [{icon}] {action:20s} | {result['reason'][:55]}")

    # اختبار حظر المصدر بعد 3 رفضات
    print("\n[اختبار حظر المصدر]")
    for i in range(4):
        r = ethics.report_source_rejection("bad_source_xyz")
        print(f"  رفضة #{i+1}: محظور={r['blocked']}")

    # محاولة استيعاب بيانات من مصدر محظور
    result = ethics.check(ACTION_DATA_INGEST, {"source": "bad_source_xyz"})
    print(f"\n  استيعاب من مصدر محظور: allowed={result['allowed']}")

    print(f"\n[سجل الانتهاكات] ({len(ethics.get_violations_log())} انتهاك)")
    for v in ethics.get_violations_log(3):
        print(f"  {v['action_type']:20s} | {v['rule']:25s} | {v['reason'][:45]}")

    print("\n[ملخص النظام]")
    import json
    print(json.dumps(ethics.summary(), indent=2, ensure_ascii=False))

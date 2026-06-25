"""
Safe Evolution — آلية Rollback الآمنة لدورات التطور الذاتي
============================================================
تضمن ألّا تُدمِّر دورة التطور أداءً جيداً قائماً.

المبدأ:
  1. قبل كل دورة تطور: خذ لقطة (Snapshot) من حالة النظام
  2. شَغِّل دورة التطور (EvolutionEngine.run_cycle)
  3. قيِّم الأداء بعد الدورة
  4. إذا تراجع الأداء عن العتبة: استعد اللقطة (Rollback)
  5. إذا تحسّن الأداء: ثبِّت اللقطة الجديدة كـ baseline

التكامل:
  - يلتف حول ai/evolution_engine.py دون تعديله (Wrapper Pattern)
  - يحفظ اللقطات في checkpoints/safe_evolution/
  - يحتفظ بسجل كامل لكل الدورات

الاستخدام:
    from ai.safe_evolution import SafeEvolutionWrapper
    safe_evo = SafeEvolutionWrapper(
        evolution_engine=engine,
        evaluator_fn=my_eval_fn,    # دالة تُعيد float (0-1)
        min_performance=0.65,
        max_snapshots=5,
    )
    result = safe_evo.run_safe_cycle()
    print(result.status)   # "improved" | "stable" | "rolled_back" | "error"
"""
from __future__ import annotations

import json
import logging
import shutil
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_NOW    = lambda: datetime.now(timezone.utc).isoformat()
_SNAP_DIR = Path("checkpoints/safe_evolution")


# ════════════════════════════════════════════════════════════════════════════
# Enums & Status
# ════════════════════════════════════════════════════════════════════════════

class CycleStatus(str, Enum):
    IMPROVED    = "improved"      # الأداء تحسّن — لقطة جديدة مُثبَّتة
    STABLE      = "stable"        # الأداء ثابت ضمن النطاق المقبول
    ROLLED_BACK = "rolled_back"   # الأداء تراجع — تمت استعادة اللقطة السابقة
    ERROR       = "error"         # خطأ في التطور أو التقييم
    SKIPPED     = "skipped"       # تجاوز الدورة (لا يوجد محرك تطور)


# ════════════════════════════════════════════════════════════════════════════
# SystemSnapshot — لقطة كاملة من حالة النظام
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SystemSnapshot:
    """لقطة كاملة من حالة النظام قبل/بعد دورة التطور."""
    snapshot_id: str
    cycle_number: int
    performance_score: float
    embedding_path: Optional[str]        # مسار ملف E المحفوظ
    weights_path: Optional[str]          # مسار ملف W (784×784)
    evolution_history_len: int
    metadata: Dict[str, Any]
    created_at: str = field(default_factory=_NOW)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "cycle_number": self.cycle_number,
            "performance_score": round(self.performance_score, 6),
            "embedding_path": self.embedding_path,
            "weights_path": self.weights_path,
            "evolution_history_len": self.evolution_history_len,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SystemSnapshot":
        return cls(**d)


# ════════════════════════════════════════════════════════════════════════════
# SafeCycleResult — نتيجة دورة آمنة
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class SafeCycleResult:
    """تقرير كامل من دورة تطور آمنة."""
    cycle_number: int
    status: CycleStatus
    performance_before: float
    performance_after: float
    performance_delta: float
    snapshot_id_before: str
    snapshot_id_after: Optional[str]
    was_rolled_back: bool
    evolution_result: Optional[Any]    # EvolutionCycleResult
    rollback_reason: str
    errors: List[str]
    duration_seconds: float
    timestamp: str = field(default_factory=_NOW)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_number": self.cycle_number,
            "status": self.status.value,
            "performance_before": round(self.performance_before, 6),
            "performance_after": round(self.performance_after, 6),
            "performance_delta": round(self.performance_delta, 6),
            "snapshot_id_before": self.snapshot_id_before,
            "snapshot_id_after": self.snapshot_id_after,
            "was_rolled_back": self.was_rolled_back,
            "rollback_reason": self.rollback_reason,
            "errors": self.errors,
            "duration_seconds": round(self.duration_seconds, 3),
            "timestamp": self.timestamp,
        }

    @property
    def summary(self) -> str:
        delta_sign = "+" if self.performance_delta >= 0 else ""
        return (
            f"Cycle #{self.cycle_number} [{self.status.value.upper()}] "
            f"perf: {self.performance_before:.3f} → {self.performance_after:.3f} "
            f"({delta_sign}{self.performance_delta:.3f})"
            + (f" | rolled_back: {self.rollback_reason}" if self.was_rolled_back else "")
        )


# ════════════════════════════════════════════════════════════════════════════
# SnapshotManager — إدارة اللقطات على القرص
# ════════════════════════════════════════════════════════════════════════════

class SnapshotManager:
    """يحفظ ويسترجع لقطات النظام على القرص."""

    def __init__(self, snap_dir: Path = _SNAP_DIR, max_snapshots: int = 5):
        self.snap_dir     = snap_dir
        self.max_snapshots = max_snapshots
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self._index_path  = self.snap_dir / "index.json"
        self._index: List[Dict[str, Any]] = self._load_index()

    def take_snapshot(
        self,
        cycle_number: int,
        performance_score: float,
        evolution_history_len: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SystemSnapshot:
        """يأخذ لقطة من الملفات الحيّة الحالية."""
        snap_id = f"snap_{uuid.uuid4().hex[:12]}"
        snap_subdir = self.snap_dir / snap_id
        snap_subdir.mkdir(parents=True, exist_ok=True)

        emb_dst = wgt_dst = None

        # نسخ nsm_embedding.npz
        emb_src = Path("nsm_embedding.npz")
        if emb_src.exists():
            emb_dst = str(snap_subdir / "nsm_embedding.npz")
            shutil.copy2(str(emb_src), emb_dst)

        # نسخ weights_784x784.csv إن وُجد
        wgt_src = Path("weights_784x784.csv")
        if wgt_src.exists():
            wgt_dst = str(snap_subdir / "weights_784x784.csv")
            shutil.copy2(str(wgt_src), wgt_dst)

        # نسخ EWC snapshots إن وُجدت
        for ewc_f in ("memory/ewc_fisher.npz", "memory/ewc_anchor.npz"):
            ewc_path = Path(ewc_f)
            if ewc_path.exists():
                shutil.copy2(str(ewc_path), str(snap_subdir / ewc_path.name))

        snap = SystemSnapshot(
            snapshot_id=snap_id,
            cycle_number=cycle_number,
            performance_score=performance_score,
            embedding_path=emb_dst,
            weights_path=wgt_dst,
            evolution_history_len=evolution_history_len,
            metadata=metadata or {},
        )

        # تحديث الفهرس
        self._index.append(snap.to_dict())
        self._cleanup_old_snapshots()
        self._save_index()

        logger.info(f"[SafeEvo] Snapshot taken: {snap_id} (perf={performance_score:.4f})")
        return snap

    def restore_snapshot(self, snapshot_id: str) -> bool:
        """يستعيد ملفات لقطة محددة."""
        snap_subdir = self.snap_dir / snapshot_id
        if not snap_subdir.exists():
            logger.error(f"[SafeEvo] Snapshot dir not found: {snapshot_id}")
            return False

        restored = []
        errors   = []

        file_map = {
            "nsm_embedding.npz": Path("nsm_embedding.npz"),
            "weights_784x784.csv": Path("weights_784x784.csv"),
            "ewc_fisher.npz": Path("memory/ewc_fisher.npz"),
            "ewc_anchor.npz": Path("memory/ewc_anchor.npz"),
        }

        for snap_name, live_path in file_map.items():
            snap_file = snap_subdir / snap_name
            if snap_file.exists():
                try:
                    live_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(snap_file), str(live_path))
                    restored.append(snap_name)
                except Exception as exc:
                    errors.append(f"{snap_name}: {exc}")

        if errors:
            logger.error(f"[SafeEvo] Restore errors: {errors}")

        logger.info(f"[SafeEvo] Restored {len(restored)} files from {snapshot_id}: {restored}")
        return len(errors) == 0

    def get_best_snapshot(self) -> Optional[Dict[str, Any]]:
        """يُعيد اللقطة ذات أعلى أداء."""
        if not self._index:
            return None
        return max(self._index, key=lambda x: x.get("performance_score", 0.0))

    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """يُعيد آخر لقطة."""
        return self._index[-1] if self._index else None

    def list_snapshots(self) -> List[Dict[str, Any]]:
        return list(self._index)

    # ── داخلي ──────────────────────────────────────────────────────────────

    def _cleanup_old_snapshots(self) -> None:
        """يحذف اللقطات الزائدة مع الاحتفاظ بـ max_snapshots."""
        if len(self._index) <= self.max_snapshots:
            return

        # احتفظ بأفضل واحدة دائماً
        best_id = self.get_best_snapshot()["snapshot_id"] if self._index else None

        # رتّب القديمة أولاً (باستثناء الأفضل)
        to_remove = []
        keep_ids  = {best_id} if best_id else set()

        for entry in self._index[: -self.max_snapshots]:
            if entry["snapshot_id"] not in keep_ids:
                to_remove.append(entry)

        for entry in to_remove:
            sid = entry["snapshot_id"]
            subdir = self.snap_dir / sid
            if subdir.exists():
                shutil.rmtree(str(subdir), ignore_errors=True)
            if entry in self._index:
                self._index.remove(entry)

    def _load_index(self) -> List[Dict[str, Any]]:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return []

    def _save_index(self) -> None:
        self._index_path.write_text(
            json.dumps(self._index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ════════════════════════════════════════════════════════════════════════════
# DefaultEvaluator — مُقيِّم الأداء الافتراضي
# ════════════════════════════════════════════════════════════════════════════

class DefaultEvaluator:
    """
    مُقيِّم أداء بسيط يعتمد على:
      1. جودة التمييز في مصفوفة Embedding (Cosine separation)
      2. عدد الحلقات المخزّنة في ExperienceStore
      3. صحة الـ CKG (عدد المفاهيم والعلاقات)
    """

    def __init__(self, embed_path: str = "nsm_embedding.npz"):
        self._embed_path = Path(embed_path)

    def evaluate(self) -> float:
        """
        يُقيِّم الحالة الحالية ويُعيد float [0, 1].
        0.0 = أداء ضعيف جداً، 1.0 = أداء مثالي.
        """
        scores: List[float] = []

        # المقياس 1: جودة Embedding (فصل المواضيع)
        emb_score = self._eval_embedding()
        if emb_score is not None:
            scores.append(emb_score)

        # المقياس 2: حجم ExperienceStore
        exp_score = self._eval_experience_store()
        if exp_score is not None:
            scores.append(exp_score)

        # المقياس 3: صحة CKG
        ckg_score = self._eval_ckg()
        if ckg_score is not None:
            scores.append(ckg_score)

        if not scores:
            return 0.5   # neutral إذا لم نتمكن من القياس

        return float(np.mean(scores))

    def _eval_embedding(self) -> Optional[float]:
        """يقيس فصل المواضيع في Embedding — أعلى = أفضل."""
        if not self._embed_path.exists():
            return None
        try:
            data = np.load(str(self._embed_path))
            E = data["E"].astype(np.float64)   # (784, 128)

            test_pairs = [
                ("القرآن الكريم سورة الفاتحة الصلاة",
                 "الشبكة العصبية Transformer attention"),
                ("الإسلام الإيمان التوحيد",
                 "Python برمجة قاعدة بيانات"),
                ("التوحيد الله النبي محمد",
                 "الجبر الخطي مصفوفة متجهات"),
            ]

            diffs: List[float] = []
            for t1, t2 in test_pairs:
                v1 = self._encode(E, t1)
                v2 = self._encode(E, t2)
                sim = float(v1 @ v2)
                # نريد sim منخفضة بين مواضيع مختلفة → نقيس (1 - sim)
                diffs.append(max(0.0, 1.0 - sim))

            # نسوي: diff > 0.5 → جيد، diff > 0.8 → ممتاز
            avg_diff = float(np.mean(diffs))
            return min(1.0, avg_diff / 0.8)

        except Exception as exc:
            logger.debug(f"[Evaluator] Embedding eval error: {exc}")
            return None

    def _eval_experience_store(self) -> Optional[float]:
        """يقيس غنى ExperienceStore — أكثر حلقات = أفضل (حتى 1000)."""
        exp_db = Path("memory/experience.db")
        if not exp_db.exists():
            return None
        try:
            import sqlite3
            conn = sqlite3.connect(str(exp_db))
            n = conn.execute(
                "SELECT COUNT(*) FROM neural_episodes"
            ).fetchone()[0]
            conn.close()
            return min(1.0, n / 1000.0)
        except Exception:
            return None

    def _eval_ckg(self) -> Optional[float]:
        """يقيس حجم الرسم البياني المعرفي."""
        ckg_path = Path("knowledge/cognitive_graph.json")
        if not ckg_path.exists():
            return None
        try:
            data = json.loads(ckg_path.read_text(encoding="utf-8"))
            nodes = len(data.get("nodes", data.get("concepts", {})))
            edges = len(data.get("edges", data.get("relations", {})))
            # هدف: 5000 مفهوم + 10000 علاقة
            node_score = min(1.0, nodes / 5000.0)
            edge_score = min(1.0, edges / 10000.0)
            return (node_score + edge_score) / 2.0
        except Exception:
            return None

    def _encode(self, E: np.ndarray, text: str) -> np.ndarray:
        vec = _text_to_vec_simple(text, E.shape[0])
        z   = E.T @ vec
        n   = np.linalg.norm(z) + 1e-8
        return z / n


# ════════════════════════════════════════════════════════════════════════════
# SafeEvolutionWrapper — المُحرّك الرئيسي
# ════════════════════════════════════════════════════════════════════════════

class SafeEvolutionWrapper:
    """
    يلتف حول EvolutionEngine ويُضيف:
      • لقطة تلقائية قبل كل دورة
      • تقييم أداء بعد كل دورة
      • استعادة تلقائية إن تراجع الأداء
      • سجل كامل لكل الدورات

    مثال:
        from ai.evolution_engine import EvolutionEngine
        from ai.safe_evolution import SafeEvolutionWrapper

        engine   = EvolutionEngine(mesh=mesh, ...)
        safe_evo = SafeEvolutionWrapper(evolution_engine=engine)
        result   = safe_evo.run_safe_cycle()
        print(result.summary)
    """

    def __init__(
        self,
        evolution_engine: Optional[Any] = None,
        evaluator_fn: Optional[Callable[[], float]] = None,
        min_performance: float = 0.60,
        max_regression: float = 0.05,
        snap_dir: Path | str = _SNAP_DIR,
        max_snapshots: int = 5,
        auto_rollback: bool = True,
    ):
        """
        Args:
            evolution_engine: كائن EvolutionEngine (يمكن أن يكون None للاختبار)
            evaluator_fn:     دالة تُعيد float [0,1] — إن لم تُعطَ يُستخدم DefaultEvaluator
            min_performance:  الحد الأدنى المقبول للأداء (0-1)
            max_regression:   أقصى تراجع مسموح قبل Rollback
            snap_dir:         مجلد اللقطات
            max_snapshots:    أقصى عدد لقطات محفوظة
            auto_rollback:    هل نُعيد تلقائياً عند التراجع؟
        """
        self._engine         = evolution_engine
        self._evaluator_fn   = evaluator_fn
        self._min_performance = min_performance
        self._max_regression  = max_regression
        self._auto_rollback   = auto_rollback

        self._snap_mgr  = SnapshotManager(Path(snap_dir), max_snapshots)
        self._evaluator = DefaultEvaluator()
        self._history: List[SafeCycleResult] = []
        self._cycle_count = 0

        logger.info("[SafeEvo] SafeEvolutionWrapper initialised "
                    f"(min_perf={min_performance}, max_regression={max_regression})")

    # ── واجهة عامة ──────────────────────────────────────────────────────────

    def run_safe_cycle(
        self,
        auto_register: bool = True,
        verbose: bool = True,
    ) -> SafeCycleResult:
        """
        تشغيل دورة تطور واحدة بشكل آمن.

        الخطوات:
          1. تقييم الأداء الحالي
          2. أخذ لقطة
          3. تشغيل دورة التطور
          4. تقييم الأداء الجديد
          5. قرار: تثبيت أم استعادة

        Returns:
            SafeCycleResult بكل التفاصيل
        """
        import time
        start_time = time.time()
        self._cycle_count += 1

        errors: List[str] = []
        evo_result = None
        snap_after = None

        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"  SafeEvolution Cycle #{self._cycle_count}")
            logger.info(f"{'='*60}")

        # ── الخطوة 1: تقييم قبل ────────────────────────────────────────────
        perf_before = self._evaluate(errors)
        if verbose:
            logger.info(f"  [Before] Performance = {perf_before:.4f}")

        # ── الخطوة 2: لقطة قبل ─────────────────────────────────────────────
        history_len = len(getattr(self._engine, "_history", []))
        snap_before = self._snap_mgr.take_snapshot(
            cycle_number=self._cycle_count,
            performance_score=perf_before,
            evolution_history_len=history_len,
            metadata={"phase": "before_evolution"},
        )

        # ── الخطوة 3: تشغيل التطور ─────────────────────────────────────────
        if self._engine is None:
            status = CycleStatus.SKIPPED
            perf_after = perf_before
            rollback_reason = ""
            if verbose:
                logger.warning("  [SafeEvo] No EvolutionEngine — skipping cycle")
        else:
            try:
                evo_result = self._engine.run_cycle(
                    auto_register=auto_register,
                    verbose=verbose,
                )
                if verbose:
                    logger.info(f"  Evolution complete: {evo_result.summary}")
            except Exception as exc:
                errors.append(f"Evolution error: {exc}")
                logger.error(f"  [SafeEvo] Evolution failed: {exc}")
                logger.debug(traceback.format_exc())
                evo_result = None

            # ── الخطوة 4: تقييم بعد ────────────────────────────────────────
            perf_after = self._evaluate(errors)
            if verbose:
                logger.info(f"  [After]  Performance = {perf_after:.4f}")

            # ── الخطوة 5: قرار ─────────────────────────────────────────────
            delta = perf_after - perf_before

            should_rollback, rollback_reason = self._should_rollback(
                perf_before, perf_after, delta, errors
            )

            if should_rollback and self._auto_rollback:
                logger.warning(
                    f"  [SafeEvo] ROLLING BACK — reason: {rollback_reason}"
                )
                success = self._snap_mgr.restore_snapshot(snap_before.snapshot_id)
                status = CycleStatus.ROLLED_BACK
                # أعِد قياس الأداء بعد الاستعادة
                perf_after = self._evaluate(errors)
                if not success:
                    errors.append("Rollback file restore had partial errors")
                if verbose:
                    logger.info(f"  [After Rollback] Performance = {perf_after:.4f}")
            else:
                rollback_reason = ""
                # خذ لقطة جديدة تُثبِّت الحالة المحسَّنة
                snap_after = self._snap_mgr.take_snapshot(
                    cycle_number=self._cycle_count,
                    performance_score=perf_after,
                    evolution_history_len=len(getattr(self._engine, "_history", [])),
                    metadata={"phase": "after_evolution", "delta": delta},
                )
                if delta > 0.01:
                    status = CycleStatus.IMPROVED
                    if verbose:
                        logger.info(f"  ✅ Performance IMPROVED by {delta:+.4f}")
                else:
                    status = CycleStatus.STABLE
                    if verbose:
                        logger.info(f"  ✓  Performance STABLE (delta={delta:+.4f})")

        duration = time.time() - start_time
        result = SafeCycleResult(
            cycle_number=self._cycle_count,
            status=status,
            performance_before=perf_before,
            performance_after=perf_after,
            performance_delta=perf_after - perf_before,
            snapshot_id_before=snap_before.snapshot_id,
            snapshot_id_after=snap_after.snapshot_id if snap_after else None,
            was_rolled_back=(status == CycleStatus.ROLLED_BACK),
            evolution_result=evo_result,
            rollback_reason=rollback_reason,
            errors=errors,
            duration_seconds=duration,
        )

        self._history.append(result)
        self._save_history()

        if verbose:
            logger.info(f"\n  {result.summary}")
            logger.info(f"  Duration: {duration:.2f}s")

        return result

    def force_rollback_to_best(self) -> bool:
        """يُعيد النظام إلى أفضل لقطة محفوظة بغض النظر عن الأداء الحالي."""
        best = self._snap_mgr.get_best_snapshot()
        if not best:
            logger.warning("[SafeEvo] No snapshots found — cannot rollback")
            return False
        snap_id = best["snapshot_id"]
        logger.info(f"[SafeEvo] Force rollback to best snapshot: {snap_id} "
                    f"(perf={best['performance_score']:.4f})")
        return self._snap_mgr.restore_snapshot(snap_id)

    def get_history(self, last_n: int = 20) -> List[Dict[str, Any]]:
        """سجل آخر n دورة."""
        return [r.to_dict() for r in self._history[-last_n:]]

    def get_stats(self) -> Dict[str, Any]:
        """ملخص إحصائي."""
        if not self._history:
            return {"total_cycles": 0}

        rollbacks  = sum(1 for r in self._history if r.was_rolled_back)
        improved   = sum(1 for r in self._history if r.status == CycleStatus.IMPROVED)
        avg_delta  = float(np.mean([r.performance_delta for r in self._history]))
        best_snap  = self._snap_mgr.get_best_snapshot()

        return {
            "total_cycles": self._cycle_count,
            "improved": improved,
            "stable": sum(1 for r in self._history if r.status == CycleStatus.STABLE),
            "rolled_back": rollbacks,
            "errors": sum(1 for r in self._history if r.status == CycleStatus.ERROR),
            "rollback_rate": round(rollbacks / max(len(self._history), 1), 3),
            "avg_performance_delta": round(avg_delta, 6),
            "current_performance": self._history[-1].performance_after,
            "best_known_performance": best_snap["performance_score"] if best_snap else None,
            "snapshots_saved": len(self._snap_mgr.list_snapshots()),
        }

    def list_snapshots(self) -> List[Dict[str, Any]]:
        return self._snap_mgr.list_snapshots()

    # ── داخلي ──────────────────────────────────────────────────────────────

    def _evaluate(self, errors: List[str]) -> float:
        """يستدعي دالة التقييم ويُعالج الأخطاء."""
        try:
            if self._evaluator_fn is not None:
                return float(self._evaluator_fn())
            return self._evaluator.evaluate()
        except Exception as exc:
            errors.append(f"Evaluation error: {exc}")
            logger.warning(f"[SafeEvo] Evaluation failed: {exc}")
            return 0.5   # neutral

    def _should_rollback(
        self,
        perf_before: float,
        perf_after: float,
        delta: float,
        errors: List[str],
    ) -> Tuple[bool, str]:
        """يقرر هل نُعيد الحالة السابقة."""
        # خطأ أثناء التطور
        if any("Evolution error" in e for e in errors):
            return True, "evolution_failed"

        # الأداء تحت الحد الأدنى المطلق
        if perf_after < self._min_performance:
            return True, f"performance_below_minimum({perf_after:.3f}<{self._min_performance})"

        # الأداء تراجع بأكثر من الحد المسموح
        if delta < -self._max_regression:
            return True, f"regression_too_large({delta:.3f}<-{self._max_regression})"

        return False, ""

    def _save_history(self) -> None:
        """يحفظ سجل الدورات في ملف JSON."""
        hist_path = _SNAP_DIR / "cycle_history.json"
        try:
            hist_path.parent.mkdir(parents=True, exist_ok=True)
            data = [r.to_dict() for r in self._history[-100:]]   # آخر 100 دورة
            hist_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning(f"[SafeEvo] Could not save history: {exc}")


# ════════════════════════════════════════════════════════════════════════════
# أداة مساعدة
# ════════════════════════════════════════════════════════════════════════════

def _fnv1a(s: str) -> int:
    h = 0x811c9dc5
    for ch in s:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h

def _text_to_vec_simple(text: str, dim: int = 784) -> np.ndarray:
    import math as _math
    vec = np.zeros(dim, dtype=np.float64)
    t   = text.strip().lower()
    for n in (1, 2, 3):
        for i in range(len(t) - n + 1):
            vec[_fnv1a(t[i:i+n]) % dim] += 1.0
    total = vec.sum()
    if total > 0:
        vec = np.log1p(vec * 10.0 / total) / _math.log1p(10.0)
    return vec

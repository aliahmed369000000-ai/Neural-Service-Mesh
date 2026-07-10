"""
Rollback Guard — آلية حماية من التطور الذاتي الخاطئ
=======================================================
الأولوية #3 من تقرير التحليل النهائي (الصورة المرفقة):
  "Rollback Mechanism — يحمي النظام من التطور الذاتي الخاطئ."

المشكلة:
  حالياً، أي تدريب جديد (EWC، fine-tuning، أو حتى nsm_trainer.py العادي)
  يكتب مباشرة فوق nsm_embedding.npz / ملفات الأوزان بدون أي فحص:
  لو خرج التدريب الجديد بنتيجة أسوأ (انهيار catastrophic forgetting،
  أو bug في البيانات الجديدة)، **لا توجد طريقة للرجوع** — الملف القديم
  ضاع للأبد.

الحل (هذا الملف):
  CheckpointGuard يبني "حارساً" حول أي عملية تحديث أوزان:
    1. ينسخ نسخة احتياطية من الملفات الحالية قبل التحديث (snapshot).
    2. يُنفّذ التحديث (update_fn) — تدريب، EWC، أي شيء يكتب على القرص.
    3. يُقيّم النتيجة الجديدة (eval_fn) ويقارنها بالنتيجة المحفوظة سابقاً.
    4. إن تراجعت النتيجة أكثر من الحد المسموح (tolerance) → **rollback تلقائي**
       يستعيد الملفات القديمة فوراً.
    5. إن نجح التحديث → يُسجَّل كـ "نسخة جيدة" جديدة ويُحتفظ بسجل من آخر
       N نسخ (rotation) لإمكانية الرجوع اليدوي لاحقاً.

  لا يعتمد على أي مكتبة خارجية — Python القياسية + NumPy فقط.
  يعمل مع أي نوع ملفات (.npz, .npy, .json) لأنه ينسخ بايتات الملفات مباشرة.

الاستخدام النموذجي (مثال متكامل):
    from ai.rollback_guard import CheckpointGuard, RollbackError

    guard = CheckpointGuard(asset="nsm_embedding")

    def do_training():
        E_new = my_training_function(...)
        np.savez("nsm_embedding.npz", E=E_new, dim=128)

    def evaluate_quality() -> float:
        # أعلى = أفضل. مثال: متوسط الفصل بين المواضيع (topic gap)
        return compute_topic_gap("nsm_embedding.npz")

    decision = guard.guarded_update(
        files=["nsm_embedding.npz"],
        update_fn=do_training,
        eval_fn=evaluate_quality,
        tolerance=-0.02,          # نسمح بتراجع طفيف 0.02 كحد أقصى
        label="إضافة موضوع جديد",
    )

    if decision.rolled_back:
        print(f"⚠ تم التراجع! النتيجة الجديدة ({decision.new_score:.4f}) "
              f"أسوأ من القديمة ({decision.old_score:.4f})")
    else:
        print(f"✓ التحديث مقبول: {decision.old_score:.4f} → {decision.new_score:.4f}")

يمكن أيضاً استخدامه يدوياً بدون update_fn (للأوزان العصبية، CKG، إلخ):
    guard.snapshot(["weights/quran_model/deep_network_state.npy"], label="قبل التدريب")
    ...  # تدريب يدوي
    guard.rollback()   # استرجاع يدوي عند الحاجة
"""
from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_NOW = lambda: datetime.now(timezone.utc).isoformat()

_DEFAULT_ROOT = Path("checkpoints/rollback")


# ════════════════════════════════════════════════════════════════════════════
# نتيجة القرار — تُرجَع من guarded_update
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class GuardDecision:
    """نتيجة عملية محمية — يُخبرك ماذا حدث بالضبط."""
    accepted:     bool
    rolled_back:  bool
    old_score:    Optional[float]
    new_score:    Optional[float]
    snapshot_id:  str
    label:        str
    error:        Optional[str] = None
    timestamp:    str = field(default_factory=_NOW)

    def summary(self) -> str:
        if self.error:
            return f"❌ فشل: {self.error} — تم التراجع للنسخة السابقة."
        if self.rolled_back:
            return (f"⚠ تراجع تلقائي: {self.old_score:.4f} → {self.new_score:.4f} "
                    f"(انخفاض غير مقبول) — استُعيدت النسخة السابقة.")
        return f"✓ تحديث مقبول: {self.old_score} → {self.new_score:.4f}"


class RollbackError(RuntimeError):
    """يُرفع عندما يفشل rollback نفسه (حالة نادرة وخطيرة)."""
    pass


# ════════════════════════════════════════════════════════════════════════════
# CheckpointGuard — المحرك الرئيسي
# ════════════════════════════════════════════════════════════════════════════

class CheckpointGuard:
    """
    يحرس مجموعة ملفات (asset) عبر نسخ احتياطية مُرقَّمة + فحص جودة تلقائي.

    كل asset له مجلد مستقل تحت checkpoints/rollback/<asset>/ يحتوي:
      - manifest.json   : سجل كل النسخ (snapshot_id, label, score, timestamp, files)
      - <snapshot_id>/   : نسخة كاملة من الملفات وقت اللقطة
    """

    def __init__(
        self,
        asset: str,
        root: Path | str = _DEFAULT_ROOT,
        max_snapshots: int = 8,
    ):
        self.asset         = asset
        self.root           = Path(root) / asset
        self.max_snapshots = max_snapshots
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / "manifest.json"
        self._manifest: List[Dict[str, Any]] = self._load_manifest()

    # ── واجهة عامة رفيعة المستوى ─────────────────────────────────────────────

    def guarded_update(
        self,
        files: List[Path | str],
        update_fn: Callable[[], Any],
        eval_fn: Callable[[], float],
        tolerance: float = 0.0,
        label: str = "update",
        require_baseline: bool = False,
    ) -> GuardDecision:
        """
        ينفّذ تحديثاً محمياً بالكامل: snapshot → update → eval → commit|rollback.

        Args:
            files:            مسارات الملفات التي سيُغيّرها update_fn (نسبية أو مطلقة)
            update_fn:        دالة بدون مدخلات تُنفّذ التحديث الفعلي (تكتب على القرص)
            eval_fn:          دالة بدون مدخلات تُرجع رقم جودة (أعلى = أفضل)
            tolerance:        أقل انخفاض مسموح به (سالب). مثلاً -0.02 يسمح بتراجع
                               0.02 كحد أقصى قبل اعتبار التحديث "فاشلاً".
                               0.0 يعني: يجب أن لا تقل النتيجة الجديدة عن القديمة.
            label:            وصف مختصر للتحديث (يُحفظ في السجل)
            require_baseline: إن True ولا توجد نسخة سابقة → يرفض التحديث بدل قبوله أعمى

        Returns:
            GuardDecision — تفاصيل كاملة عن القرار
        """
        paths = [Path(f) for f in files]

        # 1) تحديد النتيجة الأساسية (قبل التحديث)
        baseline = self._latest_snapshot_meta()
        old_score: Optional[float] = baseline["score"] if baseline else None

        if old_score is None and require_baseline:
            return GuardDecision(
                accepted=False, rolled_back=False,
                old_score=None, new_score=None,
                snapshot_id="", label=label,
                error="لا توجد نسخة أساسية (baseline) ويُطلب وجودها — رُفض التحديث",
            )

        # 2) لقطة احتياطية قبل أي تغيير
        pre_snapshot_id = self._snapshot(paths, label=f"pre::{label}", score=old_score)

        # 3) تنفيذ التحديث
        try:
            update_fn()
        except Exception as exc:
            logger.error(f"[RollbackGuard:{self.asset}] فشل التحديث نفسه: {exc}")
            self._restore(pre_snapshot_id, paths)
            return GuardDecision(
                accepted=False, rolled_back=True,
                old_score=old_score, new_score=None,
                snapshot_id=pre_snapshot_id, label=label,
                error=f"استثناء أثناء update_fn: {exc}",
            )

        # 4) تقييم النتيجة الجديدة
        try:
            new_score = float(eval_fn())
        except Exception as exc:
            logger.error(f"[RollbackGuard:{self.asset}] فشل التقييم: {exc} — تراجع احتياطي")
            self._restore(pre_snapshot_id, paths)
            return GuardDecision(
                accepted=False, rolled_back=True,
                old_score=old_score, new_score=None,
                snapshot_id=pre_snapshot_id, label=label,
                error=f"استثناء أثناء eval_fn: {exc}",
            )

        # 5) القرار: قبول أم تراجع؟
        if old_score is not None and (new_score - old_score) < tolerance:
            logger.warning(
                f"[RollbackGuard:{self.asset}] تراجع! {old_score:.4f} → {new_score:.4f} "
                f"(الحد المسموح: {tolerance}) — استعادة النسخة السابقة"
            )
            self._restore(pre_snapshot_id, paths)
            return GuardDecision(
                accepted=False, rolled_back=True,
                old_score=old_score, new_score=new_score,
                snapshot_id=pre_snapshot_id, label=label,
            )

        # 6) قبول: نسجّل النسخة الجديدة كـ "آخر نسخة جيدة"
        new_snapshot_id = self._snapshot(paths, label=label, score=new_score)
        self._prune()
        logger.info(
            f"[RollbackGuard:{self.asset}] ✓ مقبول: "
            f"{old_score if old_score is not None else '—'} → {new_score:.4f}"
        )
        return GuardDecision(
            accepted=True, rolled_back=False,
            old_score=old_score, new_score=new_score,
            snapshot_id=new_snapshot_id, label=label,
        )

    # ── واجهة يدوية (snapshot / rollback صريحان) ────────────────────────────

    def snapshot(
        self,
        files: List[Path | str],
        label: str = "manual",
        score: Optional[float] = None,
    ) -> str:
        """لقطة يدوية فورية — مفيدة قبل أي عملية خطرة لا تستخدم guarded_update."""
        return self._snapshot([Path(f) for f in files], label=label, score=score)

    def rollback(self, files: List[Path | str], to_snapshot: Optional[str] = None) -> bool:
        """
        تراجع يدوي إلى آخر نسخة جيدة، أو إلى snapshot_id محدد.

        Returns:
            True إن نجحت الاستعادة
        """
        paths = [Path(f) for f in files]
        if to_snapshot:
            return self._restore(to_snapshot, paths)

        meta = self._latest_snapshot_meta()
        if not meta:
            logger.warning(f"[RollbackGuard:{self.asset}] لا توجد نسخ للتراجع إليها")
            return False
        return self._restore(meta["snapshot_id"], paths)

    def history(self) -> List[Dict[str, Any]]:
        """سجل كل النسخ المحفوظة (الأحدث أولاً)."""
        return list(reversed(self._manifest))

    def current_score(self) -> Optional[float]:
        meta = self._latest_snapshot_meta()
        return meta["score"] if meta else None

    # ── الآليات الداخلية ──────────────────────────────────────────────────────

    def _snapshot(
        self, paths: List[Path], label: str, score: Optional[float]
    ) -> str:
        snap_id = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        snap_dir = self.root / snap_id
        snap_dir.mkdir(parents=True, exist_ok=True)

        saved_files: List[str] = []
        for p in paths:
            if p.exists():
                dest = snap_dir / p.name
                shutil.copy2(p, dest)
                saved_files.append(str(p))
            else:
                logger.debug(f"[RollbackGuard:{self.asset}] ملف غير موجود وقت اللقطة: {p}")

        entry = {
            "snapshot_id": snap_id,
            "label":       label,
            "score":       score,
            "files":       saved_files,
            "timestamp":   _NOW(),
        }
        self._manifest.append(entry)
        self._save_manifest()
        return snap_id

    def _restore(self, snapshot_id: str, paths: List[Path]) -> bool:
        snap_dir = self.root / snapshot_id
        if not snap_dir.exists():
            logger.error(f"[RollbackGuard:{self.asset}] لقطة غير موجودة: {snapshot_id}")
            return False

        ok = True
        for p in paths:
            src = snap_dir / p.name
            if src.exists():
                try:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, p)
                except Exception as exc:
                    logger.error(f"[RollbackGuard:{self.asset}] فشل استعادة {p}: {exc}")
                    ok = False
            # إن لم يكن الملف موجوداً في تلك اللقطة، نتركه كما هو (لم يكن موجوداً وقتها)
        if not ok:
            raise RollbackError(
                f"فشل استرجاع كامل اللقطة {snapshot_id} للأصل {self.asset} — "
                f"يُنصح بالتدخل اليدوي فوراً"
            )
        return ok

    def _latest_snapshot_meta(self) -> Optional[Dict[str, Any]]:
        """آخر لقطة بها score غير فارغ (نسخة 'مقبولة' حقيقية)."""
        for entry in reversed(self._manifest):
            if entry.get("score") is not None and not entry["label"].startswith("pre::"):
                return entry
        return None

    def _prune(self) -> None:
        """يحذف أقدم اللقطات إن تجاوز العدد max_snapshots (لا يحذف pre:: المرتبطة بآخر عملية)."""
        accepted = [e for e in self._manifest if not e["label"].startswith("pre::")]
        if len(accepted) <= self.max_snapshots:
            return
        to_remove = accepted[: len(accepted) - self.max_snapshots]
        remove_ids = {e["snapshot_id"] for e in to_remove}

        for snap_id in remove_ids:
            snap_dir = self.root / snap_id
            if snap_dir.exists():
                shutil.rmtree(snap_dir, ignore_errors=True)

        self._manifest = [e for e in self._manifest if e["snapshot_id"] not in remove_ids]
        self._save_manifest()

    def _load_manifest(self) -> List[Dict[str, Any]]:
        if self._manifest_path.exists():
            try:
                return json.loads(self._manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(f"[RollbackGuard] manifest تالف، يبدأ من جديد: {exc}")
        return []

    def _save_manifest(self) -> None:
        try:
            self._manifest_path.write_text(
                json.dumps(self._manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.error(f"[RollbackGuard] فشل حفظ manifest: {exc}")


# ════════════════════════════════════════════════════════════════════════════
# دالة مساعدة سريعة — لاستخدام لمرة واحدة بدون إنشاء الكائن يدوياً
# ════════════════════════════════════════════════════════════════════════════

def guarded(
    asset: str,
    files: List[Path | str],
    update_fn: Callable[[], Any],
    eval_fn: Callable[[], float],
    tolerance: float = 0.0,
    label: str = "update",
    root: Path | str = _DEFAULT_ROOT,
) -> GuardDecision:
    """اختصار: ينشئ CheckpointGuard ويُنفّذ guarded_update مباشرة."""
    guard = CheckpointGuard(asset=asset, root=root)
    return guard.guarded_update(
        files=files, update_fn=update_fn, eval_fn=eval_fn,
        tolerance=tolerance, label=label,
    )

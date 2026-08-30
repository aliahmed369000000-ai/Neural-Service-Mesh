"""اقتراحات وترقيعات كود محكومة وآمنة.

الوحدة لا تولّد الكود ولا تنفّذ أوامر Shell. تستقبل patch صريحاً من طبقة
توليد أعلى، ثم تتحقق من النسخة والمسار، وتطبّق التغيير ذرّياً، وتشغّل verifier
محقوناً، وتستعيد المحتوى السابق ذرّياً إذا فشل التحقق.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional


class PatchRejected(ValueError):
    """يرفع عند رفض الترقيع قبل الكتابة."""


@dataclass(frozen=True)
class PatchProposal:
    target: str
    old_text: str
    new_text: str
    reason: str
    expected_sha256: str
    tests: tuple[str, ...] = ()
    proposal_id: str = field(default_factory=lambda: f"patch-{uuid.uuid4().hex[:12]}")

    @classmethod
    def from_file(
        cls,
        root: Path | str,
        target: str,
        new_text: str,
        reason: str,
        tests: tuple[str, ...] = (),
    ) -> "PatchProposal":
        path = _safe_target(Path(root), target)
        old_text = path.read_text(encoding="utf-8")
        return cls(
            target=target,
            old_text=old_text,
            new_text=new_text,
            reason=reason,
            expected_sha256=_sha256(old_text),
            tests=tests,
        )


@dataclass(frozen=True)
class PatchResult:
    proposal_id: str
    status: str
    target: str
    changed: bool
    rolled_back: bool
    message: str


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_target(root: Path, target: str) -> Path:
    if not target or Path(target).is_absolute():
        raise PatchRejected("المسار يجب أن يكون نسبياً داخل جذر المشروع")
    candidate = (root / target).resolve()
    root_resolved = root.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise PatchRejected("المسار خارج جذر المشروع") from exc
    if candidate.is_symlink():
        raise PatchRejected("الروابط الرمزية غير مسموحة كهدف للترقيع")
    return candidate


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".patch-tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def apply_patch_atomically(
    root: Path | str,
    proposal: PatchProposal,
    verifier: Optional[Callable[[Path, PatchProposal], bool]] = None,
) -> PatchResult:
    """يطبق patch بعد تحقق optimistic-lock ثم يقبله أو يستعيده ذرّياً."""
    project_root = Path(root).resolve()
    path = _safe_target(project_root, proposal.target)
    if not path.is_file():
        raise PatchRejected("الملف الهدف غير موجود")

    current = path.read_text(encoding="utf-8")
    if _sha256(current) != proposal.expected_sha256:
        raise PatchRejected("تغير الملف منذ إنشاء الاقتراح؛ أُوقف التطبيق لتجنب الكتابة فوق تعديل أحدث")
    if current != proposal.old_text:
        raise PatchRejected("النص القديم لا يطابق الملف الحالي")
    if proposal.old_text == proposal.new_text:
        return PatchResult(proposal.proposal_id, "noop", proposal.target, False, False, "لا يوجد تغيير")

    _atomic_write(path, proposal.new_text)
    if verifier is not None:
        try:
            verified = bool(verifier(project_root, proposal))
        except Exception as exc:
            verified = False
            verification_error = str(exc)
        else:
            verification_error = ""
        if not verified:
            _atomic_write(path, proposal.old_text)
            message = "فشل التحقق؛ أُعيد المحتوى السابق ذرّياً"
            if verification_error:
                message += f": {verification_error}"
            return PatchResult(proposal.proposal_id, "rolled_back", proposal.target, True, True, message)

    return PatchResult(proposal.proposal_id, "accepted", proposal.target, True, False, "طُبق patch واجتاز التحقق")

"""توليد ترقيعات كود منظمة بواسطة نموذج لغوي مع بوابة أمان محلية.

الوحدة لا تطبق التعديل بنفسها؛ بل تنتج PatchProposal. التطبيق يمر حصراً عبر
apply_patch_atomically، وبوضع auto_apply لا يُسمح به إلا عند تفويض صريح.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

from .patch_proposal import PatchProposal


class PatchGenerationError(ValueError):
    """يرفع عند فشل تحويل خرج النموذج إلى اقتراح صالح."""


_PATCH_SYSTEM = """أنت مولد ترقيعات برمجية محافظ. أعد JSON فقط بلا Markdown أو شرح خارجه.
لا تغير إلا الملف والجزء المطلوب، ولا تضف أسراراً أو اتصالات شبكية أو أوامر Shell.
يجب أن يتضمن JSON المفاتيح: target, old_text, new_text, reason, tests.
يجب أن يكون old_text نسخة حرفية من النص المرسل، وnew_text الملف الكامل بعد التعديل.
"""


class LLMPatchGenerator:
    def __init__(self, llm: Optional[Any] = None, max_source_chars: int = 12000):
        self._llm = llm
        self.max_source_chars = max_source_chars

    @property
    def llm(self) -> Any:
        if self._llm is None:
            from .llm_fallback import LLMFallback
            self._llm = LLMFallback(max_tokens=1800, temperature=0.1, timeout=20)
        return self._llm

    def generate(
        self,
        root: Path | str,
        target: str,
        problem: str,
        tests: tuple[str, ...] = (),
    ) -> PatchProposal:
        project_root = Path(root).resolve()
        path = (project_root / target).resolve()
        try:
            path.relative_to(project_root)
        except ValueError as exc:
            raise PatchGenerationError("الهدف خارج جذر المشروع") from exc
        if not path.is_file() or path.is_symlink():
            raise PatchGenerationError("الهدف يجب أن يكون ملفاً عادياً موجوداً")
        source = path.read_text(encoding="utf-8")
        if len(source) > self.max_source_chars:
            raise PatchGenerationError("حجم الملف يتجاوز حد سياق مولد الترقيعات")

        request = json.dumps({
            "target": target,
            "problem": problem[:4000],
            "source": source,
            "tests": list(tests),
        }, ensure_ascii=False)
        result = self.llm.generate(request, system_prompt=_PATCH_SYSTEM)
        payload = self._parse_json(result.text)
        self._validate_payload(payload, target, source)
        return PatchProposal(
            target=payload["target"],
            old_text=payload["old_text"],
            new_text=payload["new_text"],
            reason=payload["reason"],
            expected_sha256=PatchProposal.from_file(
                project_root, target, payload["new_text"], payload["reason"],
                tuple(payload.get("tests", tests)),
            ).expected_sha256,
            tests=tuple(payload.get("tests", tests)),
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.I)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise PatchGenerationError("خرج النموذج ليس JSON صالحاً") from exc
        if not isinstance(data, dict):
            raise PatchGenerationError("خرج النموذج يجب أن يكون كائناً JSON")
        return data

    @staticmethod
    def _validate_payload(payload: dict[str, Any], target: str, source: str) -> None:
        required = {"target", "old_text", "new_text", "reason", "tests"}
        if not required.issubset(payload):
            raise PatchGenerationError("خرج النموذج يفتقد حقولاً إلزامية")
        if payload["target"] != target or payload["old_text"] != source:
            raise PatchGenerationError("خرج النموذج لا يطابق الملف الهدف الحالي")
        if not isinstance(payload["new_text"], str) or not payload["new_text"].strip():
            raise PatchGenerationError("new_text فارغ أو غير صالح")
        if not isinstance(payload["reason"], str) or not payload["reason"].strip():
            raise PatchGenerationError("reason فارغ أو غير صالح")
        if not isinstance(payload["tests"], list) or not all(isinstance(x, str) for x in payload["tests"]):
            raise PatchGenerationError("tests يجب أن تكون قائمة نصية")


def proposal_to_dict(proposal: PatchProposal) -> dict[str, Any]:
    return {
        "proposal_id": proposal.proposal_id,
        "target": proposal.target,
        "old_text": proposal.old_text,
        "new_text": proposal.new_text,
        "reason": proposal.reason,
        "expected_sha256": proposal.expected_sha256,
        "tests": list(proposal.tests),
    }

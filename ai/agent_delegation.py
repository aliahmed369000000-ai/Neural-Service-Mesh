# -*- coding: utf-8 -*-
"""
آلية التفويض الداخلي بين الوكلاء (Agent-to-Agent Delegation).

تتيح هذه الطبقة لأي وكيل في المسار الجماعي أن يفوّض مهمة فرعية
خارج اختصاصه إلى وكيل آخر، بحيث ينفّذ المنسّق (UnifiedAgentChat)
التفويض عبر reflecting_call ويعيد النتيجة ضمن ناتج الفريق النهائي.

التفويض يتم عبر وسم متفق عليه في رد الوكيل:
    ⤴ DELEGATE::<key>::<المهمة الفرعية>

القيود: عمق سلسلة التفويض محدود (يُمنع التفويض عن وكيل مُفوَّض
إليه أصلاً في نفس الجولة)، والوكيل الهدف يجب أن يكون ضمن الفهرس
المعروف، ولا يجوز تفويض الوكيل إلى نفسه.

أحداث ناقل الأحداث الجديدة:
    delegation_requested · delegation_rejected · delegation_started
    delegation_resolved
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("nsm.delegation")

# الحد الأقصى لعمق سلسلة التفويض في جولة واحدة (يمنع الحلقات)
MAX_DELEGATION_DEPTH: int = 2

# اسماء أنواع أحداث التفويض على ناقل الأحداث
DELEGATION_EVENTS = (
    "delegation_requested",
    "delegation_rejected",
    "delegation_started",
    "delegation_resolved",
)

# صيغة الوسم المتفق عليها في رد الوكيل
_DELEGATE_TAG = "⤴ DELEGATE::"
_DELEGATE_RE = re.compile(
    r"⤴\s*DELEGATE\s*::\s*([A-Za-z0-9_\-]+)\s*::\s*(.+?)(?=\n|$)",
    re.IGNORECASE | re.DOTALL,
)


def _emit(event_type: str, agent_id: str, title: str, status: str,
          detail: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """يطلق حدث تفويض على ناقل الأحداث؛ يفشل بصمت خارج سياق Streamlit."""
    try:
        from ai.agent_event_bus import emit_event  # noqa: WPS433 (استيراد كسول)
        return emit_event(event_type, agent_id=agent_id, title=title,
                          status=status, detail=detail, metadata=metadata)
    except Exception:  # pragma: no cover - حماية خارج Streamlit
        return None


def parse_delegation_requests(reply: str) -> List[Tuple[str, str]]:
    """يستخرج طلبات التفويض من رد وكيل بصيغة: ⤴ DELEGATE::<key>::<المهمة>."""
    if not reply:
        return []
    return [(key.strip(), subtask.strip()) for key, subtask in _DELEGATE_RE.findall(reply)]


def resolve_delegation_target(
    delegator_key: str,
    delegate_key: str,
    delegate_title: str,
    category_titles: Dict[str, str],
    running_delegates: Dict[str, str],
) -> Optional[str]:
    """
    يتحقق من صحة هدف التفويض ويعيد سبب الرفض إن وُجد.
    running_delegates: {agent_key: title} الوكلاء المنفذين الحاليين في الجولة.
    """
    if not category_titles or delegate_key not in category_titles:
        return f"الوكيل المستهدف غير معروف: {delegate_key}"
    if delegate_key == delegator_key:
        return "لا يجوز تفويض الوكيل إلى نفسه"
    if delegate_key in running_delegates and running_delegates.get(delegate_key) != delegate_title:
        # الوكيل الهدف يعمل أصلاً كمنفّذ مستقل في نفس الجولة — تفويض متداخل غير آمن
        return "الوكيل المستهدف يعمل على مهمة رئيسية أخرى"
    return None


class DelegationTracker:
    """يتتبع سلسلة طلبات التفويض في جولة واحدة مع حدود الأمان."""

    def __init__(self, max_depth: int = MAX_DELEGATION_DEPTH) -> None:
        self.max_depth: int = max_depth
        self.requests: List[Dict[str, Any]] = []
        self._depth_per_delegator: Dict[str, int] = {}

    def is_allowed(self, delegator_key: str, delegate_key: str,
                   delegate_title: str, category_titles: Dict[str, str],
                   running_delegates: Dict[str, str]) -> Optional[str]:
        """يعيد None إذا سُمح بالتفويض، أو سبب الرفض نصياً."""
        if delegate_key not in category_titles:
            return "الوكيل المستهدف غير معروف"
        if delegate_key == delegator_key:
            return "لا يجوز تفويض الوكيل إلى نفسه"
        # التفويض المتداخل: لا نسمح لوكيل مُفوَّض إليه أن يفوّض بدوره في نفس الجولة
        if delegator_key in {r.get("delegate_key") for r in self.requests}:
            return f"الوكيل {delegator_key} يعمل أصلاً كمستلم تفويض ولا يجوز له التفويض مجدداً"
        depth = self._depth_per_delegator.get(delegate_key, 0) + 1
        if depth > self.max_depth:
            return "تجاوز الحد المسموح لعمق سلسلة التفويض"
        reason = resolve_delegation_target(
            delegator_key, delegate_key, delegate_title, category_titles, running_delegates,
        )
        return reason

    def register_request(self, delegator_key: str, delegator_title: str,
                         delegate_key: str, delegate_title: str,
                         subtask: str) -> bool:
        """يسجل طلب تفويض جديد ويعيد True إن قُبل."""
        self.requests.append({
            "delegator_key": delegator_key,
            "delegator_title": delegator_title,
            "delegate_key": delegate_key,
            "delegate_title": delegate_title,
            "subtask": subtask,
            "status": "pending",
            "result": "",
        })
        depth = self._depth_per_delegator.get(delegate_key, 0) + 1
        self._depth_per_delegator[delegate_key] = depth
        return True

    def mark_result(self, delegate_key: str, status: str, result: str = "") -> None:
        """يحدّث حالة آخر طلب تفويض مفتوح لصالح وكيل مستلم معين."""
        for req in reversed(self.requests):
            if req["delegate_key"] == delegate_key and req["status"] == "pending":
                req["status"] = status
                req["result"] = result
                return

    def pending_count(self) -> int:
        return sum(1 for r in self.requests if r["status"] == "pending")

    def summary(self) -> Dict[str, Any]:
        return {
            "total": len(self.requests),
            "handled": sum(1 for r in self.requests if r["status"] != "pending"),
            "rejected": sum(1 for r in self.requests if r["status"] == "rejected"),
            "requests": self.requests,
        }


def announce_delegation_request(delegator_key: str, delegator_title: str,
                                delegate_key: str, delegate_title: str,
                                subtask: str, parent_task_id: str = "") -> None:
    """يعلن عن طلب تفويض جديد على ناقل الأحداث."""
    _emit(
        "delegation_requested",
        agent_id=delegator_key,
        title=f"{delegator_title} ➜ {delegate_title}",
        status="pending",
        detail=subtask,
        metadata={"delegate_key": delegate_key,
                  "delegator_key": delegator_key,
                  "parent_task_id": parent_task_id},
    )


def announce_delegation_started(delegate_key: str, delegate_title: str,
                                delegator_key: str, subtask: str) -> None:
    """يعلن أن المنسّق بدأ تنفيذ مهمة تفويضية."""
    _emit(
        "delegation_started",
        agent_id=delegate_key,
        title=f"{delegate_title} (بتفويض من {delegator_key})",
        status="running",
        detail=subtask,
        metadata={"delegate_key": delegate_key, "delegator_key": delegator_key},
    )


def announce_delegation_resolved(delegate_key: str, delegate_title: str,
                                 delegator_key: str, status: str,
                                 detail: str) -> None:
    """يعلن انتهاء مهمة تفويضية بنجاح أو فشل."""
    _emit(
        "delegation_resolved",
        agent_id=delegate_key,
        title=f"{delegate_title} (بتفويض من {delegator_key})",
        status=status,
        detail=detail,
        metadata={"delegate_key": delegate_key,
                  "delegator_key": delegator_key,
                  "final": status == "done"},
    )


def announce_delegation_rejected(delegator_key: str, delegator_title: str,
                                 delegate_key: str, reason: str) -> None:
    """يعلن رفض طلب تفويض لسبب أمان."""
    _emit(
        "delegation_rejected",
        agent_id=delegator_key,
        title=f"رفض تفويض: {delegator_title} ➜ {delegate_key}",
        status="error",
        detail=reason,
        metadata={"delegate_key": delegate_key,
                  "delegator_key": delegator_key,
                  "reason": reason},
    )


def strip_delegation_tags(reply: str) -> str:
    """يزيل أوسمة التفويض من النص قبل دمجه في الناتج النهائي."""
    if not reply:
        return reply
    cleaned = _DELEGATE_RE.sub("", reply)
    return cleaned.strip()

# -*- coding: utf-8 -*-
"""
core/node_hands.py — «يدان» للعقدة (NodeHands)
================================================
العقدة كانت تفكّر وتتواصل لكنها لا تستطيع *أن تفعل* شيئاً في العالم. هذه
الطبقة تعطيها يدين، كلٌّ بصلاحية مختلفة، كما يستعمل الإنسان يداً ليتحسّس
وأخرى ليعمل:

  • اليد اليسرى (left)  — «تتحسّس»: أدوات قراءة فقط، بلا أي أثر جانبي.
  • اليد اليمنى (right) — «تعمل»: أدوات لها أثر (إرسال رسالة، طلب تطوّر…).
                          افتراضياً **ممنوعة** حتى تمنحها سياسة صراحةً.

ضمانات أمان ثابتة (لا تُعطَّل بالإعداد):
  1. عقدة PAUSED (محجورة/موقوفة) لا تستخدم أي يد إطلاقاً.
  2. اليد اليمنى default-deny: بلا policy صريحة تُرفض كل الأفعال.
  3. حد معدّل لكل يد (نافذة منزلقة دقيقة) — اليمنى أضيق بكثير.
  4. مهلة لكل استدعاء، وقطع الناتج الطويل، والاستثناءات تُحوَّل لنتيجة خطأ.
  5. كل استخدام — حتى المرفوض — يُسجَّل في سجل تدقيق JSONL (مع حجب
     المفاتيح الحسّاسة مثل token/password/secret).
  6. لا أداة تُربَط تلقائياً: لا shell ولا git push ولا توكنات. ما لم
     يُربَط صراحةً بـ bind() لا وجود له بالنسبة للعقدة.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LEFT = "left"
RIGHT = "right"
HANDS = (LEFT, RIGHT)

_SENSITIVE_KEYS = ("token", "secret", "password", "passwd", "key", "auth", "credential")
_AUDIT_MAX_BYTES = 1_000_000
_audit_locks: Dict[str, threading.Lock] = {}
_audit_locks_guard = threading.Lock()

Policy = Callable[[Any, str, str, Dict[str, Any]], Tuple[bool, str]]


@dataclass
class HandResult:
    ok: bool
    hand: str
    tool: str
    output: Any = None
    error: Optional[str] = None
    denied: bool = False
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok, "hand": self.hand, "tool": self.tool,
            "output": self.output, "error": self.error,
            "denied": self.denied, "duration_ms": self.duration_ms,
        }


@dataclass
class _Tool:
    name: str
    fn: Callable[..., Any]
    description: str = ""


def default_policy(node: Any, hand: str, tool: str, kwargs: Dict[str, Any]) -> Tuple[bool, str]:
    """اليسرى مسموحة (قراءة فقط)، واليمنى مرفوضة حتى تُمنَح سياسة صراحةً."""
    if hand == LEFT:
        return True, "read-only hand"
    return False, "right hand is default-deny: no policy granted"


def _redact(kwargs: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in (kwargs or {}).items():
        if any(s in str(k).lower() for s in _SENSITIVE_KEYS):
            out[str(k)] = "***"
        else:
            out[str(k)] = repr(v)[:200]
    return out


class NodeHands:
    """يدا عقدة واحدة. انظر وثيقة الوحدة للضمانات."""

    def __init__(
        self,
        node: Any,
        *,
        policy: Optional[Policy] = None,
        audit_path: Optional[Path] = None,
        max_calls_per_minute: Optional[Dict[str, int]] = None,
        max_output_chars: int = 4000,
        timeout_s: float = 15.0,
    ):
        self._node = node
        self._policy: Policy = policy or default_policy
        self._audit_path = Path(audit_path) if audit_path else None
        self._limits = {LEFT: 30, RIGHT: 5}
        if max_calls_per_minute:
            self._limits.update(max_calls_per_minute)
        self._max_out = int(max_output_chars)
        self._timeout = float(timeout_s)
        self._tools: Dict[str, Dict[str, _Tool]] = {LEFT: {}, RIGHT: {}}
        self._calls: Dict[str, Deque[float]] = {LEFT: deque(), RIGHT: deque()}
        self._lock = threading.RLock()

    # ── ربط الأدوات (صريح فقط) ───────────────────────────────────────────
    def bind(self, hand: str, name: str, fn: Callable[..., Any], description: str = "") -> None:
        if hand not in HANDS:
            raise ValueError(f"hand must be one of {HANDS}, got {hand!r}")
        if not name or not isinstance(name, str):
            raise ValueError("tool name must be a non-empty string")
        if not callable(fn):
            raise TypeError("tool fn must be callable")
        with self._lock:
            self._tools[hand][name] = _Tool(name, fn, description)

    def unbind(self, hand: str, name: str) -> bool:
        with self._lock:
            return self._tools.get(hand, {}).pop(name, None) is not None

    def tools(self) -> Dict[str, List[Dict[str, str]]]:
        with self._lock:
            return {
                h: [{"name": t.name, "description": t.description} for t in self._tools[h].values()]
                for h in HANDS
            }

    # ── الاستخدام ────────────────────────────────────────────────────────
    def use(self, hand: str, tool: str, **kwargs: Any) -> HandResult:
        t0 = time.time()
        res = self._use(hand, tool, kwargs)
        res.duration_ms = round((time.time() - t0) * 1000, 2)
        self._audit(res, kwargs)
        return res

    def _use(self, hand: str, tool: str, kwargs: Dict[str, Any]) -> HandResult:
        def deny(reason: str) -> HandResult:
            return HandResult(False, hand, tool, error=reason, denied=True)

        if hand not in HANDS:
            return HandResult(False, hand, tool, error=f"unknown hand {hand!r}")
        with self._lock:
            spec = self._tools[hand].get(tool)
        if spec is None:
            return HandResult(False, hand, tool, error=f"tool {tool!r} is not bound to the {hand} hand")

        state = getattr(self._node, "state", None)
        if state == "paused":
            return deny("node is paused/quarantined: hands are disabled")

        try:
            allowed, reason = self._policy(self._node, hand, tool, dict(kwargs))
        except Exception as e:  # سياسة معطوبة = رفض (fail-closed)
            return deny(f"policy error (fail-closed): {e}")
        if not allowed:
            return deny(reason or "denied by policy")

        if not self._take_slot(hand):
            return deny(f"rate limit exceeded for {hand} hand ({self._limits[hand]}/min)")

        box: Dict[str, Any] = {}

        def run() -> None:
            try:
                box["out"] = spec.fn(**kwargs)
            except BaseException as e:  # noqa: BLE001 — نُحوّلها لنتيجة خطأ
                box["err"] = e

        th = threading.Thread(target=run, daemon=True, name=f"hand-{hand}-{tool}")
        th.start()
        th.join(self._timeout)
        if th.is_alive():
            return HandResult(False, hand, tool, error=f"timeout after {self._timeout}s")
        if "err" in box:
            e = box["err"]
            return HandResult(False, hand, tool, error=f"{type(e).__name__}: {e}")
        return HandResult(True, hand, tool, output=self._clip(box.get("out")))

    def _take_slot(self, hand: str) -> bool:
        now = time.time()
        with self._lock:
            q = self._calls[hand]
            while q and now - q[0] > 60.0:
                q.popleft()
            if len(q) >= self._limits[hand]:
                return False
            q.append(now)
            return True

    def _clip(self, out: Any) -> Any:
        try:
            text = json.dumps(out, ensure_ascii=False, default=str)
        except Exception:
            text = str(out)
        if len(text) > self._max_out:
            return text[: self._max_out] + "…[truncated]"
        return out

    # ── التدقيق ──────────────────────────────────────────────────────────
    def _audit(self, res: HandResult, kwargs: Dict[str, Any]) -> None:
        if self._audit_path is None:
            return
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "node_id": getattr(self._node, "node_id", None),
            "node": getattr(self._node, "name", None),
            "hand": res.hand, "tool": res.tool,
            "ok": res.ok, "denied": res.denied, "error": res.error,
            "duration_ms": res.duration_ms,
            "args": _redact(kwargs),
        }
        key = str(self._audit_path)
        with _audit_locks_guard:
            lock = _audit_locks.setdefault(key, threading.Lock())
        try:
            with lock:
                self._audit_path.parent.mkdir(parents=True, exist_ok=True)
                if self._audit_path.exists() and self._audit_path.stat().st_size > _AUDIT_MAX_BYTES:
                    self._audit_path.replace(self._audit_path.with_suffix(".jsonl.1"))
                with self._audit_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:  # التدقيق لا يُسقِط الفعل نفسه
            logger.warning("NodeHands audit write failed: %s", e)

    def __repr__(self) -> str:
        t = self.tools()
        return f"<NodeHands left={len(t[LEFT])} right={len(t[RIGHT])}>"

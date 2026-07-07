"""
Anthropic Advanced API — وحدة القدرات المتقدمة
=================================================
مستخلصة من Claude.ai System Prompt (2026).

توفر هذه الوحدة:
  1. استدعاء API مع Web Search Tool المدمج
  2. استدعاء API مع MCP Servers (Google Drive, Gmail, Calendar, ...)
  3. استخراج JSON منظّم (Structured Outputs)
  4. إرسال ملفات PDF وصور كـ base64
  5. إدارة المحادثات متعددة الأدوار بالسجل الكامل
  6. معالجة استجابات content blocks المتعددة الأنواع

النماذج المتاحة (2026):
  claude-sonnet-4-6         ← Sonnet 4 — الأفضل للاستخدام العام
  claude-opus-4-8           ← Opus 4  — للمهام المعقدة
  claude-haiku-4-5-20251001 ← Haiku 4 — للردود السريعة
  claude-sonnet-4-20250514  ← Sonnet 4 stable — للإنتاج

الاستخدام السريع:
    from ai.anthropic_advanced import AnthropicAdvanced

    client = AnthropicAdvanced()          # يقرأ ANTHROPIC_API_KEY تلقائياً

    # ❶ سؤال عادي
    text = client.ask("ما هو القرآن الكريم؟")

    # ❷ سؤال مع Web Search
    text = client.ask_with_search("ما آخر أخبار الذكاء الاصطناعي؟")

    # ❸ استخراج JSON منظّم
    data = client.ask_json("اعطني قائمة بأسماء الصحابة الثلاثة الأوائل بصيغة JSON")

    # ❹ تحليل صورة
    text = client.ask_with_image("ماذا يوجد في هذه الصورة؟", image_bytes, "image/jpeg")
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ai.llm_fallback import ANTHROPIC_MODELS, _post_json

logger = logging.getLogger("AnthropicAdvanced")

# ── الثوابت ──────────────────────────────────────────────────────────────────

_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MODEL     = ANTHROPIC_MODELS["sonnet"]
_DEFAULT_TIMEOUT   = 30
_DEFAULT_MAX_TOKENS = 1024


# ════════════════════════════════════════════════════════════════════════════
# نتيجة الاستدعاء
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class APIResult:
    """نتيجة استدعاء Anthropic API."""
    text:        str                      # النص المُستخرَج (مُجمَّع من كل blocks)
    raw:         Dict[str, Any] = field(default_factory=dict)  # الاستجابة الكاملة
    model:       str = ""
    input_tokens:  int = 0
    output_tokens: int = 0
    latency_ms:    float = 0.0
    error:         Optional[str] = None
    tool_calls:    List[Dict] = field(default_factory=list)  # mcp_tool_use blocks
    tool_results:  List[str]  = field(default_factory=list)  # mcp_tool_result blocks


# ════════════════════════════════════════════════════════════════════════════
# محلِّل content blocks
# ════════════════════════════════════════════════════════════════════════════

def _parse_content_blocks(content: List[Dict]) -> Tuple[str, List[Dict], List[str]]:
    """
    يُفرز blocks حسب النوع ويُعيد:
      (نص_مُجمَّع, قائمة_tool_use, قائمة_tool_results)

    الأنواع الممكنة: text | mcp_tool_use | mcp_tool_result | image | document
    """
    texts: List[str]        = []
    tool_calls: List[Dict]  = []
    tool_results: List[str] = []

    for block in content:
        btype = block.get("type", "")
        if btype == "text":
            texts.append(block.get("text", ""))
        elif btype == "mcp_tool_use":
            tool_calls.append({
                "name":  block.get("name", ""),
                "input": block.get("input", {}),
            })
        elif btype == "mcp_tool_result":
            inner = block.get("content", [])
            if isinstance(inner, list):
                for ib in inner:
                    if ib.get("type") == "text":
                        tool_results.append(ib.get("text", ""))
            elif isinstance(inner, str):
                tool_results.append(inner)

    return "\n".join(texts).strip(), tool_calls, tool_results


def _strip_json_fences(text: str) -> str:
    """يُزيل ```json ... ``` من حول نص JSON قبل التحليل."""
    text = text.strip()
    for fence in ("```json", "```JSON", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
            break
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


# ════════════════════════════════════════════════════════════════════════════
# الفئة الرئيسية
# ════════════════════════════════════════════════════════════════════════════

class AnthropicAdvanced:
    """
    واجهة متقدمة لـ Anthropic API تدعم:
      - Web Search Tool المدمج
      - MCP Servers (Google Drive, Gmail, Calendar, ...)
      - Structured JSON outputs
      - إرسال ملفات PDF/صور
      - Multi-turn بالسجل الكامل

    تتطلب: ANTHROPIC_API_KEY في متغيرات البيئة أو Streamlit Secrets.
    """

    def __init__(
        self,
        model:      str   = _DEFAULT_MODEL,
        max_tokens: int   = _DEFAULT_MAX_TOKENS,
        temperature: float = 0.5,
        timeout:    int   = _DEFAULT_TIMEOUT,
        api_key:    Optional[str] = None,
    ):
        self.model       = model
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.timeout     = timeout
        self._api_key    = api_key or os.getenv("ANTHROPIC_API_KEY", "").strip()

        if not self._api_key:
            logger.warning(
                "[AnthropicAdvanced] لا يوجد ANTHROPIC_API_KEY — "
                "ستفشل جميع الاستدعاءات."
            )

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    # ── بناء headers ─────────────────────────────────────────────────────

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key":         self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type":      "application/json",
        }

    # ── بناء messages من سجل المحادثة ───────────────────────────────────

    @staticmethod
    def build_messages(
        query:   str,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> List[Dict]:
        """
        يبني قائمة messages من سجل المحادثة + السؤال الحالي.
        history: [(user_msg, assistant_msg), ...]
        """
        messages: List[Dict] = []
        for u, a in (history or []):
            messages.append({"role": "user",      "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": query})
        return messages

    # ── الاستدعاء الأساسي ────────────────────────────────────────────────

    def _call(self, payload: Dict) -> APIResult:
        """يُرسل الطلب ويُعيد APIResult مُعبَّأ."""
        t0 = time.time()
        try:
            data = _post_json(
                _ANTHROPIC_API_URL,
                payload,
                self._headers(),
                self.timeout,
            )
            text, tool_calls, tool_results = _parse_content_blocks(
                data.get("content", [])
            )
            usage = data.get("usage", {})
            return APIResult(
                text=text,
                raw=data,
                model=data.get("model", self.model),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                latency_ms=round((time.time() - t0) * 1000, 1),
                tool_calls=tool_calls,
                tool_results=tool_results,
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            logger.error(f"[AnthropicAdvanced] HTTP {exc.code}: {body[:300]}")
            return APIResult(
                text="",
                latency_ms=round((time.time() - t0) * 1000, 1),
                error=f"HTTP {exc.code}: {body[:200]}",
            )
        except Exception as exc:
            logger.error(f"[AnthropicAdvanced] خطأ: {exc}")
            return APIResult(
                text="",
                latency_ms=round((time.time() - t0) * 1000, 1),
                error=str(exc),
            )

    # ════════════════════════════════════════════════════════════════════════
    # ❶  سؤال عادي
    # ════════════════════════════════════════════════════════════════════════

    def ask(
        self,
        query:         str,
        system:        str  = "",
        history:       Optional[List[Tuple[str, str]]] = None,
        model:         Optional[str] = None,
    ) -> str:
        """سؤال بسيط — يُعيد النص مباشرة."""
        payload: Dict[str, Any] = {
            "model":       model or self.model,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
            "messages":    self.build_messages(query, history),
        }
        if system:
            payload["system"] = system
        result = self._call(payload)
        return result.text or result.error or ""

    # ════════════════════════════════════════════════════════════════════════
    # ❷  سؤال مع Web Search Tool
    # ════════════════════════════════════════════════════════════════════════

    def ask_with_search(
        self,
        query:   str,
        system:  str = "",
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> APIResult:
        """
        يُفعّل Web Search Tool المدمج في Anthropic API.
        النموذج يقرر بنفسه متى يبحث.
        """
        payload: Dict[str, Any] = {
            "model":      self.model,
            "max_tokens": self.max_tokens,
            "messages":   self.build_messages(query, history),
            "tools": [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                }
            ],
        }
        if system:
            payload["system"] = system
        return self._call(payload)

    # ════════════════════════════════════════════════════════════════════════
    # ❸  استخراج JSON منظّم
    # ════════════════════════════════════════════════════════════════════════

    def ask_json(
        self,
        query:     str,
        json_schema_hint: str = "",
        system:    str = "",
        history:   Optional[List[Tuple[str, str]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        يطلب من النموذج إجابة JSON خالصة (بدون markdown).
        يُعيد dict مُحلَّل، أو None عند الفشل.

        json_schema_hint: وصف اختياري للبنية المطلوبة.
        """
        json_instruction = (
            "أجب فقط بـ JSON خالص بدون أي نص إضافي أو backticks. "
            "لا تضع ```json ... ```، أرسل JSON مباشرة."
        )
        if json_schema_hint:
            json_instruction += f" البنية المطلوبة: {json_schema_hint}"

        full_system = f"{system}\n\n{json_instruction}".strip() if system else json_instruction

        payload: Dict[str, Any] = {
            "model":       self.model,
            "max_tokens":  self.max_tokens,
            "temperature": 0.1,
            "system":      full_system,
            "messages":    self.build_messages(query, history),
        }
        result = self._call(payload)
        if result.error:
            logger.error(f"[ask_json] فشل: {result.error}")
            return None
        try:
            clean = _strip_json_fences(result.text)
            return json.loads(clean)
        except json.JSONDecodeError as exc:
            logger.error(f"[ask_json] خطأ JSON: {exc} | نص: {result.text[:200]}")
            return None

    # ════════════════════════════════════════════════════════════════════════
    # ❹  تحليل صورة أو ملف PDF
    # ════════════════════════════════════════════════════════════════════════

    def ask_with_image(
        self,
        query:      str,
        image_data: bytes,
        media_type: str = "image/jpeg",
        system:     str = "",
    ) -> str:
        """
        يُرسل صورة مع سؤال نصي.
        media_type: image/jpeg | image/png | image/gif | image/webp
        """
        b64 = base64.b64encode(image_data).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": media_type,
                            "data":       b64,
                        },
                    },
                    {"type": "text", "text": query},
                ],
            }
        ]
        payload: Dict[str, Any] = {
            "model":       self.model,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
            "messages":    messages,
        }
        if system:
            payload["system"] = system
        result = self._call(payload)
        return result.text or result.error or ""

    def ask_with_pdf(
        self,
        query:      str,
        pdf_data:   bytes,
        system:     str = "",
    ) -> str:
        """
        يُرسل ملف PDF مع سؤال نصي.
        """
        b64 = base64.b64encode(pdf_data).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type":       "base64",
                            "media_type": "application/pdf",
                            "data":       b64,
                        },
                    },
                    {"type": "text", "text": query},
                ],
            }
        ]
        payload: Dict[str, Any] = {
            "model":       self.model,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
            "messages":    messages,
        }
        if system:
            payload["system"] = system
        result = self._call(payload)
        return result.text or result.error or ""

    # ════════════════════════════════════════════════════════════════════════
    # ❺  استدعاء مع MCP Servers
    # ════════════════════════════════════════════════════════════════════════

    def ask_with_mcp(
        self,
        query:       str,
        mcp_servers: List[Dict[str, str]],
        system:      str = "",
        history:     Optional[List[Tuple[str, str]]] = None,
    ) -> APIResult:
        """
        يُفعّل MCP Servers المُمرَّرة.

        مثال على mcp_servers:
            [
                {"type": "url", "url": "https://drivemcp.googleapis.com/mcp/v1", "name": "Google Drive"},
                {"type": "url", "url": "https://gmailmcp.googleapis.com/mcp/v1",  "name": "Gmail"},
            ]

        خوادم MCP المتاحة من Claude.ai:
            Google Drive  → https://drivemcp.googleapis.com/mcp/v1
            Gmail         → https://gmailmcp.googleapis.com/mcp/v1
            Google Cal.   → https://calendarmcp.googleapis.com/mcp/v1
            Canva         → https://mcp.canva.com/mcp
            Figma         → https://mcp.figma.com/mcp
        """
        payload: Dict[str, Any] = {
            "model":       self.model,
            "max_tokens":  self.max_tokens,
            "messages":    self.build_messages(query, history),
            "mcp_servers": mcp_servers,
        }
        if system:
            payload["system"] = system
        return self._call(payload)

    # ════════════════════════════════════════════════════════════════════════
    # ❻  استدعاء stateful لتطبيقات/ألعاب
    # ════════════════════════════════════════════════════════════════════════

    def ask_stateful(
        self,
        state:       Dict[str, Any],
        last_action: str,
        system:      str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        يُرسل حالة التطبيق الكاملة ويُعيد الحالة المحدّثة كـ JSON.
        مناسب للألعاب وتطبيقات ذات حالة متغيرة.
        """
        query = (
            f"الحالة الحالية: {json.dumps(state, ensure_ascii=False)}\n"
            f"الإجراء الأخير: {last_action}\n\n"
            "أجب فقط بـ JSON خالص يحتوي على: updatedState, actionResult, availableActions"
        )
        return self.ask_json(query, system=system)

    # ════════════════════════════════════════════════════════════════════════
    # ❼  معلومات النموذج
    # ════════════════════════════════════════════════════════════════════════

    def info(self) -> Dict[str, str]:
        return {
            "model":      self.model,
            "api_key":    "✅ موجود" if self._api_key else "❌ غير موجود",
            "available":  "✅" if self.available else "❌",
        }

    @staticmethod
    def list_models() -> Dict[str, str]:
        """يُعيد قاموس أسماء النماذج المتاحة."""
        return ANTHROPIC_MODELS.copy()


# ════════════════════════════════════════════════════════════════════════════
# Singleton للاستخدام السريع
# ════════════════════════════════════════════════════════════════════════════

_client: Optional[AnthropicAdvanced] = None


def get_client(model: str = _DEFAULT_MODEL) -> AnthropicAdvanced:
    """يُعيد instance مشتركة (singleton) من AnthropicAdvanced."""
    global _client
    if _client is None or _client.model != model:
        _client = AnthropicAdvanced(model=model)
    return _client

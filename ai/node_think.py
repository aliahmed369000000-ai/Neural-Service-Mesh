# -*- coding: utf-8 -*-
"""ai/node_think.py — استخدام العقدة للشبكة العصبية المفتوحة المصدر.

Falcon-Arabic-7B-Instruct عبر Hugging Face Inference API (provider='hf' حصراً —
لا تسريب أبداً لمزوّد مدفوع حتى لو كانت مفاتيحه مُعدَّة في البيئة).

مشترك بين:
  • أداة think على اليد اليسرى (core/mesh_bundle.py و ai/living_mesh.py)
  • الاستخدام التلقائي داخل دورة التطوّر الذاتي (neural_refine_evolution_task)

التعطيل: NSM_NODE_NEURAL_AUTO=0 يوقف الاستخدام التلقائي فقط (الأداة اليدوية تبقى).
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

MAX_PROMPT_CHARS = 2000
MAX_TOKENS_CAP = 400
THINK_TIMEOUT = 11  # أقل من مهلة اليد (15ث) بهامش حقيقي

# حدّ أدنى بين استدعاءين تلقائيين لنفس العقدة + تراجع أسّي عند الفشل
AUTO_MIN_INTERVAL = 600.0
AUTO_MAX_BACKOFF = 3600.0
_auto_lock = threading.Lock()
_auto_state: Dict[str, Dict[str, float]] = {}


def neural_available() -> bool:
    """True فقط إن وُجد مفتاح HF فعلي (وإلا فـthink ستسقط لـCKG Synthesis
    وهو ليس شبكة عصبية — فلا فائدة من الاستدعاء التلقائي)."""
    return bool(
        os.getenv("HUGGINGFACE_API_KEY", "").strip()
        or os.getenv("HF_TOKEN", "").strip()
    )


def auto_enabled() -> bool:
    return os.getenv("NSM_NODE_NEURAL_AUTO", "1").strip().lower() not in ("0", "false", "no", "off")


def neural_think(prompt: str, max_tokens: int = 200) -> Dict[str, Any]:
    """نفس سلوك أداة think الأصلية بالضبط (تحقق إدخال + provider_override='hf')."""
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("prompt فارغ")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError("prompt أطول من 2000 حرف — اختصره قبل الاستدعاء")
    max_tokens = min(max(int(max_tokens), 1), MAX_TOKENS_CAP)

    from ai.llm_fallback import LLMFallback
    llm = LLMFallback(max_tokens=max_tokens, timeout=THINK_TIMEOUT, provider_override="hf")
    result = llm.generate(prompt)
    return {
        "text": result.text,
        "provider": result.provider.value,
        "model": result.model,
        "latency_ms": result.latency_ms,
        "used_open_source_model": result.provider.value == "huggingface",
    }


def _compact_diagnosis(diagnose: Optional[dict]) -> str:
    """ملخّص صغير آمن (بلا مفاتيح/توقيعات/مسارات) لتشخيص العقدة."""
    d = diagnose or {}
    st = d.get("status") if isinstance(d.get("status"), dict) else {}
    rep = d.get("reputation") if isinstance(d.get("reputation"), dict) else {}
    peers = d.get("peers") if isinstance(d.get("peers"), list) else []
    events = [e for e in (rep.get("events") or []) if isinstance(e, dict)][-5:]
    ev = [f"{e.get('reason', '?')}({e.get('delta', 0)})" for e in events]
    inbox = d.get("inbox") if isinstance(d.get("inbox"), dict) else {}
    out = {
        "evolution_score": st.get("evolution_score"),
        "peers": len(peers),
        "unread": inbox.get("unread_total", 0),
        "recent_reputation_events": ev,
    }
    return json.dumps(out, ensure_ascii=False)


def _clean_one_line(text: str, limit: int = 200) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text[:limit]


def neural_refine_evolution_task(
    node_id: str,
    base_task: str,
    diagnose: Optional[dict] = None,
    *,
    min_interval: float = AUTO_MIN_INTERVAL,
) -> Optional[str]:
    """استخدام تلقائي للشبكة العصبية داخل دورة التطوّر الذاتي: تقترح العقدة
    تركيزاً عملياً واحداً (سطر عربي) لدورة التحسين بناءً على تشخيصها.

    يُرجع None (بلا أي أثر) إن: معطَّل، لا مفتاح HF، ضمن فترة الانتظار،
    فشل الاستدعاء، أو كان الناتج ليس من النموذج المفتوح (CKG/خطأ).
    لا يرفع استثناءً أبداً — الشبكة العصبية إضافة اختيارية لا تعطّل التطوّر.
    النتيجة نصٌّ مُنظَّف فقط (سطر واحد ≤200 حرف) ولا تُنفَّذ كأمر."""
    try:
        if not (auto_enabled() and neural_available()):
            return None
        now = time.time()
        with _auto_lock:
            st = _auto_state.setdefault(node_id, {"last": 0.0, "fails": 0.0})
            wait = min(AUTO_MAX_BACKOFF, min_interval * (2 ** min(int(st["fails"]), 3)))
            if now - st["last"] < wait:
                return None
            st["last"] = now

        prompt = (
            "أنت عقدة في شبكة NSM تحسّن نفسها. المهمة الحالية: "
            f"{_clean_one_line(base_task, 300)}\n"
            f"تشخيصك: {_compact_diagnosis(diagnose)}\n"
            "اقترح تركيزاً عملياً واحداً للتحسين في سطر عربي واحد قصير فقط، بلا شرح."
        )
        res = neural_think(prompt, max_tokens=80)
        text = _clean_one_line(res.get("text", ""))
        ok = bool(res.get("used_open_source_model")) and bool(text) and not text.startswith(("❌", "⚠️"))
        with _auto_lock:
            st["fails"] = 0.0 if ok else st["fails"] + 1
        return text if ok else None
    except Exception as exc:  # noqa: BLE001 — إضافة اختيارية
        logger.debug("neural_refine_evolution_task failed for %s: %s", node_id, exc)
        try:
            with _auto_lock:
                _auto_state.setdefault(node_id, {"last": time.time(), "fails": 0.0})["fails"] += 1
        except Exception:
            pass
        return None


def neural_first_generate(prompt: str, max_tokens: int = 128) -> Optional[Dict[str, Any]]:
    """مسار «الشبكة العصبية أولاً» لمهام الاستدلال التي تستقبلها العقدة من
    الشبكة (ai/mesh_task_protocol.py::execute_inference): Falcon-Arabic-7B
    المفتوح المصدر حصراً (provider_override='hf').

    يُرجع None — بلا استثناء — إن: معطَّل (NSM_NODE_NEURAL_AUTO=0)، لا مفتاح HF،
    فشل الاستدعاء، أو لم يصدر الناتج من النموذج المفتوح؛ فيكمل المستدعي بمساره
    القديم بلا أي تغيير. بخلاف neural_refine_evolution_task لا حدّ معدّل هنا:
    كل مهمة استدلال حقيقية تستحق نموذجاً حقيقياً، ومزوّد HF نفسه يفرض حدوده
    (cooldown داخل LLMFallback عند الفشل)."""
    try:
        if not (auto_enabled() and neural_available()):
            return None
        prompt = (prompt or "").strip()
        if not prompt:
            return None
        res = neural_think(prompt[:MAX_PROMPT_CHARS], max_tokens=max_tokens)
        text = (res.get("text") or "").strip()
        if not res.get("used_open_source_model") or not text or text.startswith(("❌", "⚠️")):
            return None
        return {"text": text, "model": res.get("model"), "provider": res.get("provider")}
    except Exception as exc:  # noqa: BLE001
        logger.debug("neural_first_generate failed: %s", exc)
        return None

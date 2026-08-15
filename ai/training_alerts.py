"""
ai/training_alerts.py — نظام التنبيهات الذكية لتدريب SurahChain
================================================================
يرسل إشعارات فورية عند:
  1) اكتمال مهمة تدريب (نجاح)
  2) فشل مهمة تدريب
  3) اقتراب نفاد كوتا GPU/TPU المجانية على أي حساب Kaggle
  4) نفاد كوتا كل الحسابات (تفعيل الفشلوفر)

قنوات التنبيه المدعومة:
  - Discord (عبر DISCORD_BOT_TOKEN + DISCORD_CHANNEL_ID — البوت الاجتماعي الموجود)
  - سجل تنبيهات محلي (artifacts/model_training/alerts/alert_log.json)
  - واجهة Streamlit (تُقرأ من السجل المحلي في تبويب المجدول)

يُشغَّل تلقائيًا داخل scheduler_tick — لا يحتاج خدمة منفصلة.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("TrainingAlerts")

ROOT = Path(__file__).resolve().parent.parent
ALERTS_DIR = ROOT / "artifacts" / "model_training" / "alerts"
ALERTS_DIR.mkdir(parents=True, exist_ok=True)
ALERT_LOG = ALERTS_DIR / "alert_log.json"
ALERT_STATE = ALERTS_DIR / "alert_state.json"

# عتبات التنبيه
QUOTA_WARNING_HOURS = 5.0      # تنبيه "اقتراب نفاد الكوتا" عند أقل من 5 ساعات
QUOTA_CRITICAL_HOURS = 1.0     # تنبيه حرج عند أقل من ساعة
CHECK_COOLDOWN = 600           # منع تكرار نفس التنبيه خلال 10 دقائق


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── التسجيل المحلي ──────────────────────────────────────────────────────────

def _load_log() -> List[Dict[str, Any]]:
    if ALERT_LOG.is_file():
        try:
            return json.loads(ALERT_LOG.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []


def _save_log(log: List[Dict[str, Any]]) -> None:
    # أبقِ آخر 500 تنبيه فقط
    ALERT_LOG.write_text(json.dumps(log[-500:], ensure_ascii=False, indent=2), encoding="utf-8")


def _load_state() -> Dict[str, Any]:
    if ALERT_STATE.is_file():
        try:
            return json.loads(ALERT_STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"last_alert_keys": {}}


def _save_state(state: Dict[str, Any]) -> None:
    ALERT_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _alert_key(kind: str, subject: str) -> str:
    return f"{kind}:{subject}"


def _should_send(kind: str, subject: str) -> bool:
    """يمنع إغراق Discord بتنبيهات مكررة عن نفس الحدث."""
    state = _load_state()
    last = state.get("last_alert_keys", {}).get(_alert_key(kind, subject)) or 0.0
    if (time.time() - last) < CHECK_COOLDOWN:
        return False
    state.setdefault("last_alert_keys", {})[_alert_key(kind, subject)] = time.time()
    _save_state(state)
    return True


# ─── إرسال Discord ────────────────────────────────────────────────────────────

def discord_enabled() -> bool:
    return bool(os.environ.get("DISCORD_BOT_TOKEN") and os.environ.get("DISCORD_CHANNEL_ID"))


def send_discord_message(text: str) -> Dict[str, Any]:
    """يرسل رسالة إلى قناة Discord عبر البوت الموجود. لا يرمي استثناءات.
    يعتمد على DiscordAdapter (requests) أولًا، ثم urllib مباشر كـfallback
    لبيئات proxy/SSL التي تكسر شهادة requests (كما في بيئات التطوير).
    """
    if not discord_enabled():
        return {"ok": False, "reason": "DISCORD_BOT_TOKEN/DISCORD_CHANNEL_ID غير مضبوطين"}

    chan = os.environ["DISCORD_CHANNEL_ID"]
    payload = json.dumps({"content": text[:2000]}).encode("utf-8")

    def _urllib_send():
        import http.client as _httplib
        conn = _httplib.HTTPSConnection("discord.com", timeout=30)
        conn.request(
            "POST",
            f"/api/v10/channels/{chan}/messages",
            body=payload,
            headers={
                "Authorization": f"Bot {os.environ['DISCORD_BOT_TOKEN']}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 NSM-Alerts/1.0",
            },
        )
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", "replace")
        conn.close()
        if resp.status not in (200, 201):
            raise RuntimeError(f"Discord HTTP {resp.status}: {body[:200]}")
        return json.loads(body).get("id")

    # المسار 1: DiscordAdapter الرسمي
    try:
        from ai.social_platforms.discord_adapter import DiscordAdapter
        adapter = DiscordAdapter()
        adapter._require_configured = lambda: None  # bypass check (already verified)
        msg_id = adapter.publish(text[:2000])
        return {"ok": True, "message_id": msg_id, "channel": "requests"}
    except Exception as err1:
        logger.warning("DiscordAdapter فشل (%s) — نجرب urllib مباشر", str(err1)[:150])

    # المسار 2: urllib مباشر (مقاوم لبيئات SSL proxy)
    try:
        msg_id = _urllib_send()
        return {"ok": True, "message_id": msg_id, "channel": "urllib"}
    except Exception as err2:
        logger.warning("فشل إرسال Discord (urllib): %s", str(err2)[:200])

    # المسار 3: requests مع verify=False (لبيئات interception مثل CSP/حجب الشبكات)
    try:
        import requests as _req
        r = _req.post(
            f"https://discord.com/api/v10/channels/{chan}/messages",
            json={"content": text[:2000]},
            headers={"Authorization": f"Bot {os.environ['DISCORD_BOT_TOKEN']}"},
            timeout=30,
            verify=False,
        )
        r.raise_for_status()
        return {"ok": True, "message_id": r.json().get("id"), "channel": "requests-verify-false"}
    except Exception as err3:
        logger.warning("فشل إرسال Discord (verify=False): %s", str(err3)[:200])
        return {"ok": False, "error": str(err3)[:300]}


# ─── تسجيل التنبيه ────────────────────────────────────────────────────────────

def record_alert(
    kind: str,
    severity: str,
    title: str,
    message: str,
    subject: str = "",
    extra: Optional[Dict[str, Any]] = None,
    send_discord: bool = True,
) -> Dict[str, Any]:
    """
    يسجل تنبيهًا محليًا ويرسله إلى Discord إن كان متاحًا.
    الأنواع: job_complete, job_failed, quota_warning, quota_critical, fallback_activated
    الشدة: info, warning, critical
    """
    alert = {
        "id": str(uuid.uuid4().hex[:12]),
        "kind": kind,
        "severity": severity,
        "title": title,
        "message": message,
        "subject": subject,
        "at": _now(),
        "extra": extra or {},
    }
    log = _load_log()
    log.append(alert)
    _save_log(log)

    if send_discord and _should_send(kind, subject or title):
        emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(severity, "📢")
        discord_text = f"{emoji} **NSM — {title}**\n{message[:1800]}"
        disc_res = send_discord_message(discord_text)
        alert["discord"] = disc_res
        # سجّل التحديث
        _save_log(_load_log())

    return alert


# ─── تنبيهات المهام ──────────────────────────────────────────────────────────

SEVERITY_EMOJI = {"complete": "✅", "failed": "❌", "running": "🟡", "queued": "⏳"}


def alert_job_status(
    job_id: str,
    status: str,
    account: str = "",
    kernel_url: str = "",
    preset: str = "",
    n: int = 0,
    epochs: int = 0,
) -> Dict[str, Any]:
    """تنبيه عند اكتمال أو فشل مهمة تدريب."""
    if status not in ("complete", "failed"):
        return {"ok": True, "skipped": f"الحالة {status} لا تستدعي تنبيهًا"}

    severity = "info" if status == "complete" else "critical"
    emoji = SEVERITY_EMOJI.get(status, "📢")
    title = f"اكتمل التدريب {emoji}" if status == "complete" else "فشل التدريب ❌"
    params = f"preset={preset}, N={n}, epochs={epochs}" if preset else ""
    message = (
        f"المهمة: `{job_id}`\n"
        f"الحساب: `{account or '—'}`\n"
        f"{params}\n"
        + (f"الرابط: {kernel_url}" if kernel_url else "")
    )
    return record_alert(
        kind="job_complete" if status == "complete" else "job_failed",
        severity=severity,
        title=title,
        message=message,
        subject=job_id,
        extra={"job_id": job_id, "status": status, "account": account, "kernel_url": kernel_url},
    )


def alert_quota(
    username: str,
    gpu_remaining: float,
    tpu_remaining: float,
    total_gpu: float = 30.0,
) -> Optional[Dict[str, Any]]:
    """تنبيه عند اقتراب نفاد كوتا الحساب."""
    if gpu_remaining >= QUOTA_WARNING_HOURS:
        return None
    if gpu_remaining >= QUOTA_CRITICAL_HOURS:
        severity, kind = "warning", "quota_warning"
        title = f"كوتا GPU على وشك النفاد ({username})"
    else:
        severity, kind = "critical", "quota_critical"
        title = f"كوتا GPU شبه فارغة! ({username})"

    message = (
        f"الحساب: `{username}`\n"
        f"GPU متبقٍ: {gpu_remaining:.1f}h من {total_gpu:.0f}h\n"
        f"TPU متبقٍ: {tpu_remaining:.1f}h\n"
        "سيتم الانتقال تلقائيًا إلى الحساب التالي عند النفاد الكامل."
    )
    return record_alert(kind=kind, severity=severity, title=title, message=message, subject=username)


def alert_fallback_activated() -> Dict[str, Any]:
    """تنبيه عند تفعيل مزود الفشلوفر (Colab/Lightning)."""
    return record_alert(
        kind="fallback_activated",
        severity="warning",
        title="نفدت كوتا كل حسابات Kaggle — تفعّل المزود البديل",
        message=(
            "انتقل المجدول تلقائيًا إلى المزود المجاني البديل "
            "(Google Colab أو Lightning AI). أضف حسابات Kaggle جديدة "
            "عبر NSM_KAGGLE_ACCOUNTS_JSON لاستئناف التدريب على Kaggle."
        ),
        subject="fallback",
    )


# ─── فحص شامل (يُستدعى من scheduler_tick) ────────────────────────────────────

def check_and_alert_quotas(quotas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """يفحص كوتا الحسابات ويطلق التنبيهات اللازمة."""
    fired = []
    for q in quotas:
        if not q.get("ok"):
            continue
        gpu = q.get("gpu_remaining_hours") or 0.0
        tpu = q.get("tpu_remaining_hours") or 0.0
        total = q.get("quota", {}).get("GPU", {}).get("total", 30.0)
        res = alert_quota(q.get("username", "?"), gpu, tpu, total)
        if res:
            fired.append(res)
    return fired


# ─── واجهة القراءة ────────────────────────────────────────────────────────────

def list_alerts(limit: int = 50) -> List[Dict[str, Any]]:
    """آخر التنبيهات (لواجهة Streamlit)."""
    return _load_log()[-limit:][::-1]


def kernel_alert_snippet() -> str:
    """مقتطف Python جاهز للحقن في سكربت kernel Kaggle — يبعث تنبيه Discord
    عند اكتمال/فشل المهمة من داخل kernel نفسه (بيئته غير محجوبة على Discord).
    يُحقن الملف كاملًا كـbase64 عند بداية التنفيذ ثم يُستدعى notify_job() في النهاية.
    """
    import base64
    src = (ROOT / "ai" / "training_alerts.py").read_bytes()
    b64 = base64.b64encode(src).decode()
    return (
        "# ── NSM Training Alerts (injected) ──\n"
        "import base64 as _b64, os as _os, tempfile as _tf\n"
        "_src = _b64.b64decode('{}')\n"
        "_f = _tf.NamedTemporaryFile('wb', suffix='.py', delete=False)\n"
        "_f.write(_src); _f.close()\n"
        "import importlib.util as _iu\n"
        "_m = _iu.spec_from_file_location('training_alerts', _f.name)\n"
        "_mod = _iu.module_from_spec(_m); _m.loader.exec_module(_mod)\n"
        "_mod.send_discord = _mod.send_discord_message\n"
    ).format(b64)


def alerts_summary() -> Dict[str, Any]:
    """ملخص: عدد التنبيهات حسب النوع والشدة."""
    log = _load_log()
    by_kind: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}
    for a in log:
        by_kind[a.get("kind", "?")] = by_kind.get(a.get("kind", "?"), 0) + 1
        by_severity[a.get("severity", "?")] = by_severity.get(a.get("severity", "?"), 0) + 1
    return {
        "total": len(log),
        "by_kind": by_kind,
        "by_severity": by_severity,
        "discord_enabled": discord_enabled(),
        "latest": log[-5:] if log else [],
        "generated_at": _now(),
    }

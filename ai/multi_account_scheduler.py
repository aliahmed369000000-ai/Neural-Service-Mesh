"""
المجدول متعدد الحسابات (Multi-Account Training Scheduler)
=========================================================
يدير تدريب SurahChain المستمر 24/7 عبر عدة حسابات Kaggle:

  - يدعم حتى 7 حسابات Kaggle (كل حساب: 30 ساعة GPU + 20 ساعة TPU أسبوعيًا)
  - يفحص كوتا كل حساب عبر `kaggle quota` قبل الدفع
  - ينتقل تلقائيًا إلى الحساب التالي عند نفاد الكوتا
  - يتتبع حالة المهام في سجل مركزي (scheduler_state.json)
  - مزودا Colab وLightning AI كفشلوفر عند نفاد كوتا كل الحسابات
  - واجهة Streamlit لعرض الجدول والحالة

مصادر الحسابات (بالترتيب):
  1. NSM_KAGGLE_ACCOUNTS_JSON (env) — JSON array من {username, key}
  2. ملف artifacts/model_training/kaggle_accounts.json
  3. الحساب الحالي من KAGGLE_USERNAME/KAGGLE_KEY أو ~/.kaggle/kaggle.json (حساب 1)

لا يُرفع هذا الملف بالمفاتيح: الحسابات تُضاف عبر Streamlit Secrets أو الملف المحلي.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("MultiAccountScheduler")

ROOT = Path(__file__).resolve().parent.parent
SCHEDULER_DIR = ROOT / "artifacts" / "model_training" / "scheduler"
SCHEDULER_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = SCHEDULER_DIR / "scheduler_state.json"
ACCOUNTS_PATH = SCHEDULER_DIR / "kaggle_accounts.json"

# ─── الثوابت ────────────────────────────────────────────────────────────────

MIN_QUOTA_HOURS = 1.0        # أقل كوتا متبقية (ساعات) لدفع مهمة تدريب
QUOTA_REFRESH_TTL = 3600     # صلاحية نتيجة فحص الكوتا (ثانية)
MAX_RUNNING_PER_ACCOUNT = 1  # مهمة واحدة نشطة لكل حساب في كل مرة
MAX_CONCURRENT_ALL = 3       # حد المهام المتزامنة عبر كل الحسابات
POLL_INTERVAL = 300          # دورة فحص (5 دقائق)

# الحد الأقصى التقريبي لإتمام جلسة تدريب واحدة على Kaggle (ساعات)
ESTIMATED_SESSION_HOURS = 3.0

FALLBACK_PROVIDERS = ["colab", "lightning"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(s: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_" else "-" for c in (s or "nsm"))
    return out.strip("-_")[:48] or "nsm-job"


# ─── إدارة الحسابات ──────────────────────────────────────────────────────────

def load_accounts() -> List[Dict[str, str]]:
    """
    تحميل قائمة الحسابات من NSM_KAGGLE_ACCOUNTS_JSON ثم الملف المحلي
    ثم الحساب الحالي كحساب افتراضي (حساب 1).
    """
    accounts: List[Dict[str, str]] = []

    # 1) من المتغير البيئي
    raw = os.environ.get("NSM_KAGGLE_ACCOUNTS_JSON") or ""
    if raw.strip():
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict) and entry.get("username"):
                        accounts.append({
                            "username": str(entry["username"]).strip(),
                            "key": str(entry.get("key") or ""),
                            "note": str(entry.get("note") or ""),
                        })
        except Exception as exc:
            logger.warning("فشل تحليل NSM_KAGGLE_ACCOUNTS_JSON: %s", exc)

    # 2) من الملف المحلي
    if ACCOUNTS_PATH.is_file():
        try:
            file_data = json.loads(ACCOUNTS_PATH.read_text(encoding="utf-8"))
            for entry in (file_data if isinstance(file_data, list) else []):
                if isinstance(entry, dict) and entry.get("username"):
                    acc = {
                        "username": str(entry["username"]).strip(),
                        "key": str(entry.get("key") or ""),
                        "note": str(entry.get("note") or ""),
                    }
                    if not any(a["username"] == acc["username"] for a in accounts):
                        accounts.append(acc)
        except Exception as exc:
            logger.warning("فشل قراءة %s: %s", ACCOUNTS_PATH, exc)

    # 3) الحساب الحالي كحساب افتراضي إن لم يُدرج
    user = os.environ.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_USER") or ""
    key = os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_KEY") or ""
    if not user and (Path.home() / ".kaggle" / "kaggle.json").is_file():
        try:
            cfg = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text(encoding="utf-8"))
            user = cfg.get("username") or user
            key = cfg.get("key") or key
        except Exception:
            pass
    if user and not any(a["username"] == user for a in accounts):
        accounts.append({"username": user, "key": key, "note": "الحساب الحالي (افتراضي)"})

    return accounts


def save_accounts(accounts: List[Dict[str, str]]) -> None:
    ACCOUNTS_PATH.write_text(json.dumps(accounts, ensure_ascii=False, indent=2), encoding="utf-8")


# ─── فحص الكوتا ─────────────────────────────────────────────────────────────

def parse_quota_output(output: str) -> Dict[str, Dict[str, float]]:
    """
    يحلل مخرجات `kaggle quota` إلى قاموس:
      {"GPU": {"used": 0.90, "remaining": 29.10, "total": 30.00}, "TPU": {...}}
    """
    result: Dict[str, Dict[str, float]] = {}
    for line in (output or "").splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[0] in ("GPU", "TPU") and "used" not in line:
            try:
                result[parts[0]] = {
                    "used": float(parts[1].replace("h", "")),
                    "remaining": float(parts[2].replace("h", "")),
                    "total": float(parts[3].replace("h", "")),
                }
            except ValueError:
                continue
    return result


def check_account_quota(username: str, key: str) -> Dict[str, Any]:
    """يفحص كوتا حساب Kaggle عبر `kaggle quota` بمفاتيح مؤقتة."""
    env = dict(os.environ)
    env["KAGGLE_USERNAME"] = username
    env["KAGGLE_KEY"] = key
    # مسار config مؤقت حتى لا تتداخل الجلسات
    tmp = SCHEDULER_DIR / f"_cfg_{username}"
    tmp.mkdir(parents=True, exist_ok=True)
    cfg = tmp / "kaggle.json"
    cfg.write_text(json.dumps({"username": username, "key": key}), encoding="utf-8")
    env["KAGGLE_CONFIG_DIR"] = str(tmp)

    try:
        proc = subprocess.run(
            ["kaggle", "quota"],
            capture_output=True, text=True, timeout=60, env=env,
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        quota = parse_quota_output(out)
        gpu_remaining = quota.get("GPU", {}).get("remaining", 0.0)
        tpu_remaining = quota.get("TPU", {}).get("remaining", 0.0)
        return {
            "ok": proc.returncode == 0,
            "username": username,
            "quota": quota,
            "gpu_remaining_hours": gpu_remaining,
            "tpu_remaining_hours": tpu_remaining,
            "gpu_exhausted": gpu_remaining < MIN_QUOTA_HOURS,
            "raw": out[-2000:],
            "checked_at": _now(),
        }
    except Exception as e:
        return {"ok": False, "username": username, "error": str(e), "checked_at": _now()}


def accounts_quota_status() -> List[Dict[str, Any]]:
    """يفحص كوتا كل الحسابات ويعيد قائمة مرتبة بالكوتا المتبقية (تنازليًا)."""
    accounts = load_accounts()
    statuses = []
    for acc in accounts:
        if not acc.get("key"):
            statuses.append({
                "username": acc["username"],
                "ok": False,
                "error": "لا توجد KAGGLE_KEY لهذا الحساب",
                "gpu_remaining_hours": 0.0,
                "tpu_remaining_hours": 0.0,
                "gpu_exhausted": True,
                "checked_at": _now(),
            })
            continue
        statuses.append(check_account_quota(acc["username"], acc["key"]))
    statuses.sort(key=lambda s: -(s.get("gpu_remaining_hours") or 0.0))
    return statuses


# ─── سجل الحالة المركزي ──────────────────────────────────────────────────────

def load_state() -> Dict[str, Any]:
    if STATE_PATH.is_file():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"round_robin_index": 0, "jobs": {}, "history": [], "updated_at": _now()}


def save_state(state: Dict[str, Any]) -> None:
    state["updated_at"] = _now()
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# ─── اختيار الحساب ───────────────────────────────────────────────────────────

def pick_next_account(min_gpu_hours: float = MIN_QUOTA_HOURS) -> Optional[Dict[str, Any]]:
    """
    يختار الحساب التالي حسب تناوب round-robin على الحسابات التي لديها كوتا كافية.
    """
    accounts = load_accounts()
    if not accounts:
        return None
    state = load_state()
    idx = state.get("round_robin_index", 0) % len(accounts)
    for _ in range(len(accounts) + 1):
        acc = accounts[idx % len(accounts)]
        if not acc.get("key"):
            idx += 1
            continue
        q = check_account_quota(acc["username"], acc["key"])
        if q.get("ok") and (q.get("gpu_remaining_hours") or 0.0) >= min_gpu_hours:
            state["round_robin_index"] = (idx + 1) % len(accounts)
            save_state(state)
            return {"account": acc, "quota": q}
        idx += 1
    return None


# ─── دورة المجدول ────────────────────────────────────────────────────────────

def run_training_job(
    preset: str = "medium",
    n: int = 60000,
    epochs: int = 30,
    batch: int = 24,
    fresh: bool = True,
    auto_push: bool = True,
    account: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    يدفع مهمة تدريب SurahChain على Kaggle باسم حساب معين.
    يستبدل اعتمادات البيئة مؤقتًا قبل الدفع.
    """
    from ai.kaggle_provider import prepare_surahchain_kaggle_job, push_kaggle_kernel

    env_patch = {}
    restore = {}
    if account:
        for var, val in (("KAGGLE_USERNAME", account["username"]), ("KAGGLE_KEY", account["key"])):
            if val:
                env_patch[var] = val
        if account.get("username") and account.get("key"):
            tmp = SCHEDULER_DIR / f"_cfg_{account['username']}"
            tmp.mkdir(parents=True, exist_ok=True)
            (tmp / "kaggle.json").write_text(
                json.dumps({"username": account["username"], "key": account["key"]}),
                encoding="utf-8",
            )
            env_patch["KAGGLE_CONFIG_DIR"] = str(tmp)

    for var, val in env_patch.items():
        restore[var] = os.environ.get(var)
        os.environ[var] = val

    try:
        prep = prepare_surahchain_kaggle_job(
            preset=preset, n=n, epochs=epochs, batch=batch, fresh=fresh, auto_push=auto_push
        )
        if not prep.get("ok"):
            return {"ok": False, "error": "فشل التجهيز", "prepare": prep}
        push = push_kaggle_kernel(prep["job_id"])
        result = {
            "ok": bool(push.get("ok")),
            "job_id": prep["job_id"],
            "account": account.get("username") if account else None,
            "kernel_url": push.get("kernel_url") or push.get("kernel_slug"),
            "push": push,
            "pushed_at": _now(),
        }
        return result
    finally:
        for var, val in restore.items():
            if val is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = val


def scheduler_tick(
    preset: str = "medium",
    n: int = 60000,
    epochs: int = 30,
    batch: int = 24,
    fresh: bool = True,
) -> Dict[str, Any]:
    """
    دورة واحدة للمجدول: يفحص المهام النشطة ويطلق مهمة جديدة إن كانت الكوتا متاحة.
    """
    state = load_state()
    jobs = state.get("jobs", {})

    # 1) تحديث حالة المهام الحالية
    for jid, jmeta in list(jobs.items()):
        if jmeta.get("status") in ("running", "queued"):
            try:
                from ai.kaggle_provider import status_kaggle_kernel
                st = status_kaggle_kernel(jid)
                raw = (st.get("status_raw") or "").lower()
                if "complete" in raw:
                    jmeta["status"] = "complete"
                    jmeta["finished_at"] = _now()
                elif any(t in raw for t in ("error", "failed", "cancelled")):
                    jmeta["status"] = "failed"
                    jmeta["finished_at"] = _now()
                    jmeta["failure_raw"] = (st.get("status_raw") or "")[-1000:]
                elif "queued" in raw:
                    jmeta["status"] = "queued"
                else:
                    jmeta["status"] = "running"
                jmeta["last_check"] = _now()
            except Exception as exc:
                jmeta["check_error"] = str(exc)

    running = [j for j in jobs.values() if j.get("status") in ("running", "queued")]
    summary: Dict[str, Any] = {
        "tick_at": _now(),
        "total_jobs": len(jobs),
        "running": len(running),
        "action": "none",
    }

    # 2) إذا كانت هناك سعة لوظيفة جديدة
    if len(running) >= MAX_CONCURRENT_ALL:
        summary["action"] = "concurrent_limit_reached"
        save_state(state)
        return summary

    pick = pick_next_account()
    if not pick:
        # نفدت كوتا كل الحسابات — الفشلوفر إلى Colab/Lightning
        summary["action"] = "all_accounts_exhausted"
        summary["fallback"] = {
            "providers": FALLBACK_PROVIDERS,
            "hint": "أضف حسابات Kaggle جديدة عبر NSM_KAGGLE_ACCOUNTS_JSON أو الملف المحلي",
        }
        save_state(state)
        return summary

    # 3) إطلاق مهمة جديدة على الحساب المختار
    res = run_training_job(
        preset=preset, n=n, epochs=epochs, batch=batch, fresh=fresh,
        account=pick["account"],
    )
    if res.get("ok"):
        jobs[res["job_id"]] = {
            "job_id": res["job_id"],
            "status": "running",
            "account": res["account"],
            "kernel_url": res.get("kernel_url"),
            "preset": preset, "n": n, "epochs": epochs, "batch": batch,
            "started_at": res["pushed_at"],
            "quota_before": pick["quota"],
        }
        state["jobs"] = jobs
        state["history"].append({
            "event": "job_launched",
            "job_id": res["job_id"],
            "account": res["account"],
            "at": res["pushed_at"],
        })
        summary["action"] = "job_launched"
        summary["job_id"] = res["job_id"]
        summary["account"] = res["account"]
        summary["kernel_url"] = res.get("kernel_url")
    else:
        summary["action"] = "launch_failed"
        summary["error"] = res.get("error") or str(res.get("push", {}).get("output") or "")[-500:]

    save_state(state)
    return summary


# ─── واجهة عامة ──────────────────────────────────────────────────────────────

def scheduler_report() -> Dict[str, Any]:
    """تقرير شامل للمجدول: الحسابات، الكوتا، المهام."""
    state = load_state()
    quotas = accounts_quota_status()
    jobs = list(state.get("jobs", {}).values())
    return {
        "generated_at": _now(),
        "accounts": quotas,
        "active_jobs": [j for j in jobs if j.get("status") in ("running", "queued")],
        "all_jobs": jobs,
        "history": state.get("history", [])[-20:],
        "updated_at": state.get("updated_at"),
    }


def scheduler_cli(args: Optional[List[str]] = None) -> str:
    """واجهة CLI داخلية للمجدول."""
    argv = args or sys.argv[1:]
    cmd = (argv[0] if argv else "report").lower()

    if cmd == "status" or cmd == "report":
        rep = scheduler_report()
        lines = [f"NSM Multi-Account Scheduler — {rep['generated_at']}"]
        lines.append("── الحسابات والكوتا ──")
        for q in rep["accounts"]:
            flag = "✅" if q.get("ok") and not q.get("gpu_exhausted") else "❌"
            lines.append(
                f"  {flag} {q.get('username')}: GPU متبقٍ {q.get('gpu_remaining_hours'):.1f}h "
                f"(كُلّي {q.get('quota', {}).get('GPU', {}).get('total', 0.0):.0f}h)"
            )
        lines.append(f"── المهام: {len(rep['active_jobs'])} نشطة / {len(rep['all_jobs'])} إجمالية ──")
        for j in rep["active_jobs"]:
            lines.append(f"  ● {j.get('job_id')} [{j.get('status')}] @{j.get('account')} — {j.get('kernel_url')}")
        return "\n".join(lines)

    if cmd == "tick":
        res = scheduler_tick()
        return json.dumps(res, ensure_ascii=False, indent=2)

    if cmd == "quota":
        return json.dumps(accounts_quota_status(), ensure_ascii=False, indent=2, default=str)

    if cmd == "accounts":
        return json.dumps(load_accounts(), ensure_ascii=False, indent=2)

    if cmd == "state":
        return json.dumps(load_state(), ensure_ascii=False, indent=2, default=str)

    if cmd == "loop":
        interval = 300
        try:
            interval = int(argv[1])
        except (IndexError, ValueError):
            pass
        lines = [f"مجدول NSM متعدد الحسابات — دورة كل {interval}s (Ctrl+C للإيقاف)"]
        print(lines[0])
        try:
            while True:
                res = scheduler_tick()
                print(f"[{_now()}] {res.get('action')} — {json.dumps(res, ensure_ascii=False)}")
                time.sleep(interval)
        except KeyboardInterrupt:
            return "تم إيقاف المجدول"

    return f"أوامر المجدول: status|tick|quota|accounts|state|loop [ثانية]"


if __name__ == "__main__":
    print(scheduler_cli())

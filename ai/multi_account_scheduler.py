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

# Self-Healing: الحد الأقصى لمحاولات إعادة الإطلاق لكل سلسلة مهمة (الأصلية + المحاولات)
MAX_HEAL_ATTEMPTS = 3
# أقل فترة انتظار (دقائق) بعد فشل المهمة قبل إعادة إطلاقها (تحمي من حلقات
# إعادة الإطلاق الفورية عند فشل متكرر)
MIN_HEAL_COOLDOWN_MINUTES = 5


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


# ─── Handoff نقاط التفتيش بين الحسابات ────────────────────────────────────

def pull_kernel_checkpoints(job_id: str, into: Optional[Path] = None) -> Dict[str, Any]:
    """يسحب مخرجات kernel Kaggle (تشمل checkpoints/*.pt وprogress_*.json المرفوعة
    داخل kernel) إلى مجلد محلي — الخطوة الأولى في handoff قبل إيقاف kernel.
    """
    from ai.kaggle_provider import download_kaggle_output

    res = download_kaggle_output(job_id)
    if not res.get("ok"):
        return res
    out_dir = Path(res["output_dir"]) if res.get("output_dir") else None
    into_dir = into or (SCHEDULER_DIR / "handoff" / job_id)
    files_found = []
    if out_dir and out_dir.is_dir():
        try:
            import shutil

            for pat in ("checkpoints/*.pt", "checkpoints/*.json", "progress_*.json", "checkpoints/**/*"):
                for f in out_dir.glob(pat):
                    if f.is_file():
                        rel = f.relative_to(out_dir)
                        dest = into_dir / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(str(f), str(dest))
                        files_found.append(str(rel))
        except Exception as exc:
            logger.warning("نسخ checkpoints فشل: %s", exc)
    res["local_dir"] = str(into_dir) if into_dir else ""
    res["checkpoint_files"] = files_found
    return res


def upload_handoff_checkpoint(
    job_id: str,
    from_account: str,
    to_account: str,
    files_dir: str = "",
) -> Dict[str, Any]:
    """يرفع آخر checkpoint من مهمة منتهية/موقوفة إلى GitHub كـhandoff رسمی.

    يُستخدم قبل إيقاف kernel عند نفاد كوتا الحساب أو عند اكتشاف اكتمال/فشل مهمة،
    حتى يستأنف الحساب التالي من آخر عصر (SCN_RESUME=auto يسحبها تلقائيًا).
    """
    if not (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or os.environ.get("NSM_GITHUB_TOKEN")):
        return {"ok": False, "error": "لا GITHUB_TOKEN — تخطي رفع الـhandoff"}
    repo = os.environ.get("SCN_REPO", "aliahmed369000000-ai/Neural-Service-Mesh")
    branch = os.environ.get("SCN_BRANCH", "main")
    src_dir = Path(files_dir) if files_dir else (SCHEDULER_DIR / "handoff" / job_id)
    if not src_dir.is_dir():
        return {"ok": False, "error": f"لا مجلد checkpoints للحركة {src_dir}"}
    tmp = Path("/tmp/nsm_handoff_push")
    import shutil
    import subprocess

    shutil.rmtree(str(tmp), ignore_errors=True)
    try:
        token = (
            os.environ.get("GITHUB_TOKEN")
            or os.environ.get("GH_TOKEN")
            or os.environ.get("NSM_GITHUB_TOKEN")
            or ""
        )
        r = subprocess.run(
            ["git", "clone", "-q", "--branch", branch,
             f"https://x-access-token:{token}@github.com/{repo}.git", str(tmp)],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode != 0:
            return {"ok": False, "error": "clone فشل", "detail": r.stderr[-300:]}
        dest = tmp / "experiments" / "surah_chain_network" / "checkpoints"
        dest.mkdir(parents=True, exist_ok=True)
        copied = []
        # ملاحظة: pull_kernel_checkpoints ينسخ الملفات إلى local_dir/checkpoints/
        # لكن يمكن أيضًا أن تكون الملفات مباشرة في src_dir
        search_dirs = [src_dir]
        if (src_dir / "checkpoints").is_dir():
            search_dirs.append(src_dir / "checkpoints")
        for pat in ("*.pt", "*.json"):
            seen = set()
            for sdir in search_dirs:
                for f in sorted(sdir.glob(pat)):
                    if f.name in seen:
                        continue
                    shutil.copy(str(f), str(dest / f.name))
                    copied.append(f.name)
                    seen.add(f.name)
        subprocess.run(["git", "-C", str(tmp), "add", "-f", "experiments/surah_chain_network/checkpoints/"],
                       capture_output=True, check=False)
        st = subprocess.run(["git", "-C", str(tmp), "status", "--porcelain"], capture_output=True, text=True)
        if not st.stdout.strip():
            return {"ok": True, "skipped": True, "msg": "لا تغييرات جديدة"}
        msg = (
            f"NSM: checkpoint handoff من @{from_account} إلى @{to_account}"
            f" (job {job_id})"
        )
        subprocess.run(
            ["git", "-C", str(tmp), "-c", "user.email=nsm-bot@users.noreply.github.com",
             "-c", "user.name=NSM Bot", "commit", "-q", "-m", msg],
            capture_output=True, check=False,
        )
        r2 = subprocess.run(["git", "-C", str(tmp), "push", "-q", "origin", branch],
                            capture_output=True, text=True, timeout=600)
        if r2.returncode == 0:
            return {
                "ok": True,
                "uploaded": copied,
                "msg": msg,
                "from": from_account,
                "to": to_account,
            }
        return {"ok": False, "error": "push فشل", "detail": r2.stderr[-300:]}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


def record_handoff(
    from_account: str,
    to_account: str,
    job_id: str,
    status: str = "success",
    detail: str = "",
) -> Dict[str, Any]:
    """يسجّل عملية handoff في سجل مركزي scheduler_state.json".
    """
    state = load_state()
    handoffs = state.setdefault("handoffs", [])
    entry = {
        "at": _now(),
        "from_account": from_account,
        "to_account": to_account,
        "job_id": job_id,
        "status": status,
        "detail": detail or "",
    }
    handoffs.append(entry)
    state["handoffs"] = handoffs[-100:]
    save_state(state)
    return entry


def pause_kernel(job_id: str) -> Dict[str, Any]:
    """يوقف kernel Kaggle بلطف عبر Kaggle API (cancel) — الجزء الاختياري من handoff.
    الإيقاف ليس إلزاميًا: الـkernel قد يكون اكتمل أو فشل بالفعل، ونعالج ذلك بهدوء.
    """
    job_dir = ROOT / "artifacts" / "model_training" / "kaggle_jobs" / job_id
    meta_path = job_dir / "kernel-metadata.json"
    if not meta_path.is_file():
        return {"ok": False, "error": "لا metadata"}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    kernel_id = meta.get("id") or ""
    if not kernel_id or not _kaggle_cli_available_local():
        return {"ok": False, "error": "kaggle CLI غير متاح"}
    try:
        proc = subprocess.run(
            ["kaggle", "kernels", "status", kernel_id],
            capture_output=True, text=True, timeout=60,
        )
        st = (proc.stdout or "").strip()
        if any(t in st.lower() for t in ("complete", "cancelled", "error")):
            return {"ok": True, "already_done": True, "status": st}
        proc2 = subprocess.run(
            ["kaggle", "kernels", "stop", kernel_id],
            capture_output=True, text=True, timeout=60,
        )
        out = (proc2.stdout or "") + "\n" + (proc2.stderr or "")
        return {
            "ok": proc2.returncode == 0 or "cancelled" in (out or "").lower(),
            "cli_output": out[-500:],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _kaggle_cli_available_local() -> bool:
    """فحص سريع لوجود kaggle CLI محليًا (دون الاعتماد على ensure_kaggle_env)."""
    try:
        r = subprocess.run(["kaggle", "--version"], capture_output=True, text=True, timeout=20)
        return r.returncode == 0
    except Exception:
        return False


def perform_handoff(
    from_account: str,
    to_account: str,
    job_id: str,
    pause_kernel_first: bool = True,
) -> Dict[str, Any]:
    """الخطوة الكاملة: إيقاف kernel (إن كان يعمل) + سحب checkpoints + رفعها إلى GitHub.

    تُستدعى تلقائيًا من scheduler_tick عند نفاد كوتا حساب أو عند اكتمال/فشل مهمة
    — الحساب التالي يستأنف تلقائيًا بفضل SCN_RESUME=auto.
    """
    result: Dict[str, Any] = {"ok": False, "job_id": job_id, "from": from_account, "to": to_account}

    if pause_kernel_first:
        try:
            pa = pause_kernel(job_id)
            result["pause"] = pa
        except Exception as exc:
            result["pause_error"] = str(exc)

    try:
        pull = pull_kernel_checkpoints(job_id)
        result["pull"] = pull
        if not pull.get("ok"):
            record_handoff(from_account, to_account, job_id, "failed_pull", str(pull.get("error")))
            return result
    except Exception as exc:
        result["pull_error"] = str(exc)
        record_handoff(from_account, to_account, job_id, "failed_pull", str(exc))
        return result

    files_dir = pull.get("local_dir") or ""
    uploaded = upload_handoff_checkpoint(job_id, from_account, to_account, files_dir)
    result["upload"] = uploaded
    if uploaded.get("ok"):
        record_handoff(from_account, to_account, job_id, "success", ",".join(uploaded.get("uploaded") or []))
    else:
        record_handoff(from_account, to_account, job_id, "failed_upload", str(uploaded.get("error") or ""))
    result["ok"] = uploaded.get("ok")
    return result


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
                    if not jmeta.get("alerted_complete"):
                        try:
                            from ai.training_alerts import alert_job_status
                            alert_job_status(
                                jid, "complete",
                                account=jmeta.get("account") or "",
                                kernel_url=jmeta.get("kernel_url") or "",
                                preset=jmeta.get("preset") or "",
                                n=jmeta.get("n") or 0,
                                epochs=jmeta.get("epochs") or 0,
                            )
                        except Exception as exc:
                            logger.warning("فشل تنبيه الاكتمال: %s", exc)
                        jmeta["alerted_complete"] = True
                        # Handoff تلقائي: kernel اكتمل — تأكد أن آخر checkpoint
                        # مرفوعة على GitHub قبل إطلاق المهمة على الحساب التالي
                        try:
                            _handoff_on_job_end(jid, jmeta, reason="complete")
                        except Exception as hexc:
                            logger.warning("فشل handoff عند الاكتمال: %s", hexc)
                elif any(t in raw for t in ("error", "failed", "cancelled")):
                    jmeta["status"] = "failed"
                    jmeta["finished_at"] = _now()
                    jmeta["failure_raw"] = (st.get("status_raw") or "")[-1000:]
                    if not jmeta.get("alerted_failed"):
                        try:
                            from ai.training_alerts import alert_job_status
                            alert_job_status(
                                jid, "failed",
                                account=jmeta.get("account") or "",
                                kernel_url=jmeta.get("kernel_url") or "",
                                preset=jmeta.get("preset") or "",
                                n=jmeta.get("n") or 0,
                                epochs=jmeta.get("epochs") or 0,
                            )
                        except Exception as exc:
                            logger.warning("فشل تنبيه الفشل: %s", exc)
                        jmeta["alerted_failed"] = True
                        # Handoff تلقائي: kernel فشل/أُوقف — ارفع checkpoint الأخيرة
                        # قبل أن ينتقل التدريب إلى الحساب التالي
                        try:
                            _handoff_on_job_end(jid, jmeta, reason="failed")
                        except Exception as hexc:
                            logger.warning("فشل handoff عند الفشل: %s", hexc)
                        # Self-Healing: أعد الإطلاق تلقائيًا على حساب آخر
                        try:
                            healed = _heal_failed_job(jid, jmeta, state)
                            jmeta["healed"] = healed.get("ok", False)
                            jmeta["heal_result"] = healed
                        except Exception as hexc:
                            logger.warning("فشل الإصلاح التلقائي: %s", hexc)
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
        # نفدت كوتا كل الحسابات — قبل الفشلوفر إلى Colab/Lightning،
        # ارفع checkpoint آخر مهمة مكتملة/منتهية إلى GitHub إن وُجدت
        try:
            exhausted = _handoff_on_quota_exhaustion(jobs)
            summary["handoff_on_exhaustion"] = exhausted
        except Exception as exc:
            logger.warning("فشل handoff عند نفاد الكوتا: %s", exc)
            summary["handoff_on_exhaustion"] = {"ok": False, "error": str(exc)}
        summary["action"] = "all_accounts_exhausted"
        summary["fallback"] = {
            "providers": FALLBACK_PROVIDERS,
            "hint": "أضف حسابات Kaggle جديدة عبر NSM_KAGGLE_ACCOUNTS_JSON أو الملف المحلي",
        }
        # تنبيه ذكي عند تفعيل الفشلوفر
        try:
            from ai.training_alerts import alert_fallback_activated
            alert_fallback_activated()
        except Exception as exc:
            logger.warning("فشل تنبيه الفشلوفر: %s", exc)
        save_state(state)
        return summary

    # 2ب) الحساب الحالي نفدت كوتاه ولكن توجد حسابات أخرى صالحة؟
    #     (pick_next_account يتخطى الحساب المنفد تلقائيًا — لكن إن كانت المهمة
    #      النشطة الوحيدة على حساب ينفد قريبًا نسجّل تنبيه handoff وقائي)
    try:
        _handoff_warning_on_low_quota(jobs, pick)
    except Exception as exc:
        logger.warning("فشل فحص الكوتا المنخفضة: %s", exc)

    # 4) تنبيهات الكوتا (اقتراب النفاد)
    try:
        from ai.training_alerts import check_and_alert_quotas
        summary["quota_alerts"] = check_and_alert_quotas(
            [a["quota"] for a in [pick] if a.get("quota")]
        )
    except Exception as exc:
        logger.warning("فشل فحص تنبيهات الكوتا: %s", exc)

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


# ─── Self-Healing: إعادة إطلاق المهام الفاشلة تلقائيًا ───────────────────

def _pick_next_account_excluding(
    excluded_account: str,
) -> Optional[Dict[str, Any]]:
    """اختيار أغنى حساب بالكوتا يستبعد حسابًا محددًا (الحساب الفاشل)."""
    accounts = load_accounts()
    best: Optional[Dict[str, Any]] = None
    best_hours = -1.0
    for acc in accounts:
        if not acc.get("key"):
            continue
        if acc.get("username", "").strip() == (excluded_account or "").strip():
            continue
        q = check_account_quota(acc["username"], acc["key"])
        if not q.get("ok"):
            continue
        rem = q.get("gpu_remaining_hours") or 0.0
        if rem >= MIN_QUOTA_HOURS and rem > best_hours:
            best, best_hours = {"account": acc, "quota": q}, rem
    return best


def _heal_failed_job(
    jid: str,
    jmeta: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    """Self-Healing: بعد فشل kernel، يعيد إطلاق المهمة تلقائيًا على حساب آخر
    من آخر checkpoint متاح (SCN_RESUME=auto داخل kernel يستأنف تلقائيًا).
    يُسجَّل healed_from وheal_attempts ليعرف المراقب أن المهمة امتداد
    لمهمة سابقة. لا يعيد الإطلاق إن بلغت السلسلة سقف المحاولات، أو مرّ أقل
    من فترة الانتظار، أو لم يوجد حساب بديل غني بالكوتا."""
    res: Dict[str, Any] = {"ok": False, "reason": ""}
    root_id = jmeta.get("heal_root") or jid
    chain_attempts = jmeta.get("heal_attempts", 0) + 1
    if chain_attempts > MAX_HEAL_ATTEMPTS:
        res["reason"] = (
            f"بلعت المهمة الأم {root_id} سقف محاولات الإصلاح ({MAX_HEAL_ATTEMPTS})"
        )
        return res
    # فترة انتظار بعد الفشل (لا نعيد الإطلاق فورًا بنفس الخطأ)
    finished_at = jmeta.get("finished_at") or ""
    age_minutes = MIN_HEAL_COOLDOWN_MINUTES + 1
    if finished_at:
        try:
            age_minutes = (
                datetime.now(timezone.utc) - datetime.fromisoformat(finished_at)
            ).total_seconds() / 60.0
        except Exception:
            age_minutes = MIN_HEAL_COOLDOWN_MINUTES + 1
    if age_minutes < MIN_HEAL_COOLDOWN_MINUTES:
        res["reason"] = (
            f"فترة الانتظار {MIN_HEAL_COOLDOWN_MINUTES}د لم تنتهِ بعد (انقضت {age_minutes:.1f}د)"
        )
        return res
    # حساب بديل غني بالكوتا (لا نعيد الإطلاق على الحساب الفاشل نفسه)
    failed_account = jmeta.get("account") or ""
    pick = _pick_next_account_excluding(failed_account)
    if not pick:
        res["reason"] = "لا يوجد حساب بديل غني بالكوتا لإعادة الإطلاق"
        return res
    chosen = (pick.get("account") or {}).get("username", "")
    jparams = {
        "preset": jmeta.get("preset") or "medium",
        "n": jmeta.get("n") or 60000,
        "epochs": jmeta.get("epochs") or 30,
        "batch": jmeta.get("batch") or 24,
    }
    launch = run_training_job(
        preset=jparams["preset"], n=jparams["n"],
        epochs=jparams["epochs"], batch=jparams["batch"],
        fresh=False, auto_push=True, account=pick["account"],
    )
    if not launch.get("ok"):
        res["reason"] = "فشل إعادة الإطلاق: " + str(
            launch.get("error") or launch.get("push", {}).get("output") or ""
        )[-400:]
        return res
    new_jid = launch.get("job_id")
    state["jobs"][new_jid] = {
        "job_id": new_jid,
        "status": "running",
        "account": launch.get("account"),
        "kernel_url": launch.get("kernel_url"),
        "preset": jparams["preset"], "n": jparams["n"],
        "epochs": jparams["epochs"], "batch": jparams["batch"],
        "started_at": launch.get("pushed_at"),
        "quota_before": pick.get("quota"),
        "healed_from": jid,
        "heal_root": root_id,
        "heal_attempts": chain_attempts,
    }
    state["history"].append({
        "event": "job_healed",
        "new_job_id": new_jid,
        "healed_from": jid,
        "old_account": failed_account,
        "new_account": chosen,
        "at": _now(),
    })
    # تنبيه فوري بأن المهمة أُعيدت إطلاقها
    try:
        from ai.training_alerts import record_alert

        record_alert(
            "job_healed", "info",
            f"إعادة إطلاق تلقائي — {jid} → {new_jid}",
            (
                f"فشلت المهمة {jid} على حساب {failed_account} فأُعيد إطلاقها تلقائيًا "
                f"على حساب {chosen} من آخر checkpoint (محاولة {chain_attempts}/"
                f"{MAX_HEAL_ATTEMPTS}). المعاملات: {jparams}"
            ),
            subject=f"healed_{root_id}_{chain_attempts}",
        )
    except Exception as exc:
        logger.warning("فشل تنبيه الإصلاح: %s", exc)
    res.update({"ok": True, "new_job_id": new_jid, "new_account": chosen})
    return res


def _handoff_on_job_end(jid: str, jmeta: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Handoff داخلي عند انتهاء مهمة (اكتمال/فشل): kernel انتهى من نفسه فلا نوقفه
    بالقوة — نسحب checkpoints من مخرجاته ونرفعها إلى GitHub، ثم يسجّل الحدث.
    """
    acc = jmeta.get("account") or ""
    res = perform_handoff(acc, acc, jid, pause_kernel_first=False)
    try:
        from ai.training_alerts import record_alert

        record_alert(
            f"handoff_{reason}", "info",
            f"Handoff checkpoint — {jid} ({reason})",
            (
                f"اكتملت مهمة {jid} ({reason}) على حساب {acc}. "
                f"Checkpoint الأخيرة رُفعت إلى GitHub حتى يستأنف الحساب التالي تلقائيًا. "
                f"الحالة: {'نجح' if res.get('ok') else 'فشل'}"
            ),
            subject=f"handoff_{jid}",
        )
    except Exception as exc:
        logger.warning("فشل تنبيه handoff: %s", exc)
    return res


def _handoff_on_quota_exhaustion(jobs: Dict[str, Any]) -> Dict[str, Any]:
    """عند نفاد كوتا كل الحسابات: ابحث عن آخر مهمة منتهية حديثة وارفع checkpoint الأخيرة منها."""
    recently_done = [
        j for j in jobs.values()
        if j.get("status") in ("complete", "failed") and j.get("finished_at")
    ]
    if not recently_done:
        return {"ok": False, "reason": "لا مهام منتهية حديثة للـhandoff"}
    # الأحدث أولًا
    recently_done.sort(key=lambda j: j.get("finished_at", ""), reverse=True)
    return _handoff_on_job_end(
        recently_done[0]["job_id"],
        recently_done[0],
        reason="quota_exhausted",
    )


def _handoff_warning_on_low_quota(jobs: Dict[str, Any], pick: Dict[str, Any]) -> None:
    """إن كانت مهمة نشطة على حساب كوته منخفضة جدًا، سجّل تحذير handoff وقائي."""
    acc = pick.get("account") or {}
    quota = pick.get("quota") or {}
    rem = quota.get("gpu_remaining_hours") or 0.0
    if rem <= ESTIMATED_SESSION_HOURS * 0.5:
        active_on_account = [
            j for j in jobs.values()
            if j.get("status") in ("running", "queued") and (j.get("account") or "") == acc.get("username", "")
        ]
        for j in active_on_account:
            record_handoff(
                acc.get("username", ""), "<next>", j.get("job_id", ""),
                "warning_low_quota",
                f"كوتا منخفضة ({rem:.1f}h) — سيُنفَّذ handoff تلقائيًا عند اكتمال/فشل المهمة",
            )


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
        "handoffs": state.get("handoffs", [])[-10:],
        "last_checkpoint": _last_handoff_checkpoint(state),
        "updated_at": state.get("updated_at"),
    }


def _last_handoff_checkpoint(state: Dict[str, Any]) -> Dict[str, Any]:
    """آخر handoff ناجح مع تفاصيلها — لعرض حالة الاستئناف التلقائي في الواجهة."""
    for entry in reversed(state.get("handoffs", [])):
        if entry.get("status") == "success":
            return {
                "at": entry.get("at"),
                "job_id": entry.get("job_id"),
                "from_account": entry.get("from_account"),
                "to_account": entry.get("to_account"),
                "files": (entry.get("detail") or "").split(","),
            }
    return {}


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

    if cmd == "handoff":
        # handoff <from_account> <to_account> <job_id>
        if len(argv) < 4:
            return "الاستخدام: handoff <الحساب_القديم> <الحساب_الجديد> <job_id> [stop_kernel 0|1]"
        stop = "0"
        try:
            stop = argv[4]
        except IndexError:
            pass
        res = perform_handoff(argv[1], argv[2], argv[3], pause_kernel_first=stop != "0")
        return json.dumps(res, ensure_ascii=False, indent=2)

    if cmd == "handoffs":
        state = load_state()
        entries = state.get("handoffs", [])
        lines = ["── سجل handoffs ──"]
        for h in entries[-20:]:
            lines.append(
                f"  [{h.get('at', '')[:19]}] {h.get('status')}: "
                f"{h.get('job_id')} @{h.get('from_account')} → {h.get('to_account')} — {h.get('detail', '')}"
            )
        if len(entries) > 20:
            lines.append(f"  ... و{len(entries) - 20} أخرى")
        if len(entries) == 0:
            lines.append("  لا handoffs حتى الآن — تُسجّل عند اكتمال/فشل مهمة أو نفاد الكوتا")
        return "\n".join(lines)

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

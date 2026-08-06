"""
Training Sandbox & Guardrails — بيئة اختبار معزولة لوكيل تدريب النماذج
======================================================================
يوفر:
  • تقييد مسارات القراءة/الكتابة (لا تعديل لكود النظام الأساسي)
  • ميزانية زمن/حقب/عينات
  • اكتشاف GPU مع تفضيل CPU للمهام التجريبية
  • Early stopping
  • سجل مهام منظم (JSONL + ملخص)

لا يعتمد على Docker للتشغيل المحلي، لكنه يتوافق مع خدمة docker
`nsm-training` عند التشغيل داخل الحاوية.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger("TrainingSandbox")

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "training_guardrails.json"
LOG_DIR = ROOT / "artifacts" / "model_training" / "logs"
MISSION_DIR = ROOT / "artifacts" / "model_training" / "missions"
LOG_DIR.mkdir(parents=True, exist_ok=True)
MISSION_DIR.mkdir(parents=True, exist_ok=True)

_DEFAULT_CONFIG: Dict[str, Any] = {
    "sandbox": {
        "allowed_read_roots": ["data", "artifacts/model_training", "ai", "models"],
        "allowed_write_roots": [
            "artifacts/model_training",
            "artifacts/model_training/logs",
            "artifacts/model_training/uploads",
            "artifacts/model_training/missions",
        ],
        "forbidden_write_globs": ["ai/**", "ui_pages/**", ".git/**", "streamlit_app.py"],
        "max_runtime_seconds": 300,
        "max_epochs": 50,
        "max_samples": 5000,
        "max_upload_mb": 25,
    },
    "budget": {
        "max_missions_per_hour": 20,
        "max_gpu_minutes_per_day": 60,
        "allow_gpu": True,
        "prefer_cpu_for_toy": True,
    },
    "early_stopping": {
        "enabled": True,
        "patience": 5,
        "min_delta": 0.001,
        "monitor": "val_loss",
        "mode": "min",
    },
    "first_mission": {
        "id": "mission_001_toy_classification",
        "dataset": "data/samples/classification_demo.csv",
        "epochs": 15,
        "max_runtime_seconds": 120,
    },
}


def load_guardrails() -> Dict[str, Any]:
    try:
        if CONFIG_PATH.is_file():
            with open(CONFIG_PATH, encoding="utf-8") as f:
                cfg = json.load(f)
            # دمج سطحي مع الافتراضي
            out = json.loads(json.dumps(_DEFAULT_CONFIG))
            for k, v in cfg.items():
                if isinstance(v, dict) and k in out:
                    out[k].update(v)
                else:
                    out[k] = v
            return out
    except Exception as e:
        logger.warning("guardrails load failed: %s", e)
    return json.loads(json.dumps(_DEFAULT_CONFIG))


def _resolve_under_root(path: Path) -> Path:
    p = path if path.is_absolute() else (ROOT / path)
    return p.resolve()


def is_path_allowed(path: Path | str, mode: str = "read") -> bool:
    """mode: read | write"""
    cfg = load_guardrails()["sandbox"]
    try:
        target = _resolve_under_root(Path(path))
    except Exception:
        return False
    # يجب أن يبقى داخل ROOT
    try:
        target.relative_to(ROOT.resolve())
    except ValueError:
        return False

    if mode == "write":
        for g in cfg.get("forbidden_write_globs") or []:
            # مطابقة بسيطة على أجزاء المسار
            g_clean = g.replace("**/", "").replace("**", "").replace("*", "")
            if g_clean and g_clean in str(target.relative_to(ROOT)).replace("\\", "/"):
                # منع كتابة ملفات النظام الأساسية
                rel = str(target.relative_to(ROOT)).replace("\\", "/")
                if any(
                    rel == x.rstrip("/")
                    or rel.startswith(x.rstrip("/").replace("**", "").replace("*", "") + "/")
                    for x in (cfg.get("forbidden_write_globs") or [])
                    if not x.endswith("/**")
                ):
                    return False
                for bad in ("ai/", "ui_pages/", ".git/", "core/", "scripts/"):
                    if rel.startswith(bad) or rel in (
                        "streamlit_app.py",
                        "app_core.py",
                        "api_server.py",
                    ):
                        return False

        for root in cfg.get("allowed_write_roots") or []:
            base = _resolve_under_root(Path(root))
            try:
                target.relative_to(base)
                return True
            except ValueError:
                continue
        return False

    # read
    for root in cfg.get("allowed_read_roots") or []:
        base = _resolve_under_root(Path(root))
        try:
            target.relative_to(base)
            return True
        except ValueError:
            continue
    # السماح بقراءة العينات دائماً
    try:
        target.relative_to((ROOT / "data").resolve())
        return True
    except ValueError:
        return False


def clamp_epochs(epochs: int) -> int:
    mx = int(load_guardrails()["sandbox"].get("max_epochs", 50))
    return max(1, min(int(epochs), mx))


def clamp_samples(n: int) -> int:
    mx = int(load_guardrails()["sandbox"].get("max_samples", 5000))
    return max(1, min(int(n), mx))


def max_runtime_seconds(override: Optional[int] = None) -> int:
    if override is not None:
        return max(5, min(int(override), int(load_guardrails()["sandbox"].get("max_runtime_seconds", 300))))
    return int(load_guardrails()["sandbox"].get("max_runtime_seconds", 300))


def detect_compute() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "device": "cpu",
        "gpu_available": False,
        "gpu_name": None,
        "cuda_version": None,
        "prefer_cpu_for_toy": bool(load_guardrails()["budget"].get("prefer_cpu_for_toy", True)),
        "allow_gpu": bool(load_guardrails()["budget"].get("allow_gpu", True)),
    }
    try:
        import torch

        if torch.cuda.is_available() and info["allow_gpu"]:
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = getattr(torch.version, "cuda", None)
            if not info["prefer_cpu_for_toy"]:
                info["device"] = "cuda"
    except Exception:
        pass
    # رام
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    info["ram_available_gb"] = int(line.split()[1]) / (1024 * 1024)
                    break
    except Exception:
        info["ram_available_gb"] = None
    return info


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 0.001
    mode: str = "min"  # min loss or max metric
    best: Optional[float] = None
    bad_epochs: int = 0
    stopped: bool = False

    @classmethod
    def from_config(cls) -> "EarlyStopping":
        c = load_guardrails().get("early_stopping") or {}
        return cls(
            patience=int(c.get("patience", 5)),
            min_delta=float(c.get("min_delta", 0.001)),
            mode=str(c.get("mode", "min")),
        )

    def step(self, value: float) -> bool:
        """يعيد True إذا يجب إيقاف التدريب."""
        if self.best is None:
            self.best = value
            return False
        improved = (
            (value < self.best - self.min_delta)
            if self.mode == "min"
            else (value > self.best + self.min_delta)
        )
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.stopped = True
                return True
        return False


@dataclass
class MissionLog:
    mission_id: str
    name: str
    started_at: str
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "running"
    result_summary: Dict[str, Any] = field(default_factory=dict)
    finished_at: Optional[str] = None

    def event(self, level: str, message: str, **extra: Any) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "message": message,
            **extra,
        }
        self.events.append(entry)
        # JSONL append
        log_path = LOG_DIR / f"{self.mission_id}.jsonl"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("log write failed: %s", e)

    def finish(self, status: str, summary: Optional[Dict[str, Any]] = None) -> Path:
        self.status = status
        self.finished_at = datetime.now(timezone.utc).isoformat()
        if summary:
            self.result_summary.update(summary)
        out = MISSION_DIR / f"{self.mission_id}.json"
        payload = {
            "mission_id": self.mission_id,
            "name": self.name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "result_summary": self.result_summary,
            "events": self.events,
            "compute": detect_compute(),
            "guardrails": load_guardrails(),
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self.event("info", f"mission finished: {status}", summary=self.result_summary)
        return out


class SandboxTimeout(Exception):
    pass


def run_with_timeout(fn: Callable[[], Any], seconds: int) -> Any:
    """تشغيل دالة مع مهلة زمنية (خيط + علم). لا يقتل الخيط بقوة على كل المنصات."""
    box: Dict[str, Any] = {"done": False, "result": None, "error": None}

    def runner():
        try:
            box["result"] = fn()
        except Exception as e:
            box["error"] = e
        finally:
            box["done"] = True

    th = threading.Thread(target=runner, daemon=True)
    th.start()
    th.join(timeout=max(1, int(seconds)))
    if not box["done"]:
        raise SandboxTimeout(f"تجاوز المهلة ({seconds}s)")
    if box["error"] is not None:
        raise box["error"]
    return box["result"]


def assert_write_allowed(path: Path | str) -> Path:
    p = _resolve_under_root(Path(path))
    if not is_path_allowed(p, "write"):
        raise PermissionError(
            f"الكتابة مرفوضة خارج منطقة sandbox: {p}. "
            "المسموح: artifacts/model_training/** فقط تقريباً."
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def assert_read_allowed(path: Path | str) -> Path:
    p = _resolve_under_root(Path(path))
    if not is_path_allowed(p, "read"):
        raise PermissionError(f"القراءة مرفوضة لهذا المسار: {p}")
    return p


def sandbox_status_report() -> str:
    cfg = load_guardrails()
    comp = detect_compute()
    lines = [
        "## 🔒 حالة بيئة الاختبار (Sandbox + Guardrails)",
        "",
        "### الحوسبة",
        f"- الجهاز المفضّل: **{comp.get('device')}**",
        f"- GPU: {'✅ ' + str(comp.get('gpu_name')) if comp.get('gpu_available') else '❌ غير متاح / معطّل للمهام التجريبية'}",
        f"- CUDA: {comp.get('cuda_version') or '—'}",
        f"- رام متاحة: {comp.get('ram_available_gb') and round(comp['ram_available_gb'], 2)} GB",
        "",
        "### الصلاحيات",
        f"- قراءة: {', '.join(cfg['sandbox'].get('allowed_read_roots') or [])}",
        f"- كتابة: {', '.join(cfg['sandbox'].get('allowed_write_roots') or [])}",
        f"- ممنوع الكتابة على: ai/, ui_pages/, .git/, streamlit_app.py, …",
        "",
        "### الميزانية",
        f"- max_runtime: {cfg['sandbox'].get('max_runtime_seconds')}s",
        f"- max_epochs: {cfg['sandbox'].get('max_epochs')}",
        f"- max_samples: {cfg['sandbox'].get('max_samples')}",
        f"- max_missions/hour: {cfg['budget'].get('max_missions_per_hour')}",
        f"- GPU minutes/day: {cfg['budget'].get('max_gpu_minutes_per_day')}",
        "",
        "### Early stopping",
        f"- enabled={cfg['early_stopping'].get('enabled')} patience={cfg['early_stopping'].get('patience')} "
        f"min_delta={cfg['early_stopping'].get('min_delta')}",
        "",
        "### أول مهمة معرّفة",
        f"- id: `{cfg['first_mission'].get('id')}`",
        f"- dataset: `{cfg['first_mission'].get('dataset')}`",
        f"- epochs: {cfg['first_mission'].get('epochs')}",
        "",
        f"الإعدادات: `{CONFIG_PATH.relative_to(ROOT)}`",
        f"السجلات: `{LOG_DIR.relative_to(ROOT)}`",
    ]
    return "\n".join(lines)


def list_mission_logs(limit: int = 15) -> str:
    files = sorted(MISSION_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    lines = ["## 📜 سجل المهام التدريبية", ""]
    if not files:
        lines.append("لا توجد مهام مسجّلة بعد. شغّل: **أول مهمة تدريبية**")
        return "\n".join(lines)
    for p in files[:limit]:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            lines.append(
                f"- `{p.name}` — **{data.get('status')}** — {data.get('name', '')} "
                f"— {data.get('finished_at') or data.get('started_at')}"
            )
        except Exception:
            lines.append(f"- `{p.name}`")
    return "\n".join(lines)


def run_first_mission(dry_run: bool = False) -> str:
    """
    إطلاق أول مهمة تدريبية (Toy classification) تحت الحواجز.
    يستخدم data/samples/classification_demo.csv ونموذج MLP صغير.
    """
    cfg = load_guardrails()
    fm = cfg.get("first_mission") or {}
    mission_id = str(fm.get("id") or "mission_001_toy_classification")
    # لاحقة زمنية لتجنّب الكتابة فوق سجل سابق
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    mid = f"{mission_id}_{stamp}"
    name = str(fm.get("name") or "أول مهمة: تصنيف Toy")
    dataset = str(fm.get("dataset") or "data/samples/classification_demo.csv")
    epochs = clamp_epochs(int(fm.get("epochs") or 15))
    timeout = max_runtime_seconds(int(fm.get("max_runtime_seconds") or 120))

    log = MissionLog(
        mission_id=mid,
        name=name,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    log.event("info", "بدء المهمة الأولى", dataset=dataset, epochs=epochs, timeout=timeout)
    log.event("info", "فحص الحوسبة", **detect_compute())

    if dry_run:
        log.finish("dry_run", {"message": "لم يُنفَّذ تدريب فعلي"})
        return (
            f"## 🧪 أول مهمة (dry-run)\n\n"
            f"- mission_id: `{mid}`\n"
            f"- dataset: `{dataset}`\n"
            f"- epochs: {epochs} | timeout: {timeout}s\n"
            f"- الحواجز: مفعّلة\n"
            f"- السجل: `artifacts/model_training/missions/{mid}.json`"
        )

    # التحقق من المسارات
    try:
        ds_path = assert_read_allowed(dataset)
        log.event("info", "مسار البيانات مسموح", path=str(ds_path.relative_to(ROOT)))
    except Exception as e:
        log.finish("failed", {"error": str(e)})
        return f"❌ فشل فحص المسار: {e}"

    def _train():
        # استيراد محلي لتفادي دورات
        from ai.model_training_agent import train_from_csv

        return train_from_csv(
            str(ds_path.relative_to(ROOT)),
            target_col="label",
            epochs=epochs,
            prefer="torch",
        )

    t0 = time.time()
    try:
        result_text = run_with_timeout(_train, timeout)
        elapsed = time.time() - t0
        out_path = log.finish(
            "success",
            {
                "elapsed_s": round(elapsed, 2),
                "epochs": epochs,
                "dataset": dataset,
                "result_preview": (result_text or "")[:1500],
            },
        )
        return (
            f"## ✅ أول مهمة تدريبية اكتملت\n\n"
            f"- mission_id: `{mid}`\n"
            f"- المدة: {elapsed:.1f}s (حد {timeout}s)\n"
            f"- epochs: {epochs}\n"
            f"- السجل الكامل: `{out_path.relative_to(ROOT)}`\n"
            f"- JSONL: `artifacts/model_training/logs/{mid}.jsonl`\n\n"
            f"{result_text}"
        )
    except SandboxTimeout:
        log.finish("timeout", {"timeout_s": timeout})
        return f"❌ توقفت المهمة: تجاوز المهلة ({timeout}s). راجع الحواجز في config/training_guardrails.json"
    except Exception as e:
        log.event("error", str(e), traceback=traceback.format_exc()[-1500:])
        log.finish("failed", {"error": str(e)})
        return f"❌ فشلت المهمة: {type(e).__name__}: {e}"

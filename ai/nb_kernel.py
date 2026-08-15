"""
ai/nb_kernel.py — محرك خلايا حقيقي (Colab/Kaggle style)
==========================================================
المشكلة التي يعالجها:
  محرك notebook_engine القديم كان يشغّل كل خلية Python في عملية منفصلة
  (`subprocess.run(["python3","-c",source])`)، فلا يوجد أي استمرارية بين
  الخلايا — لا متغيرات مشتركة ولا استيراد محفوظ ولا matplotlib حية. هذا بعكس
  Colab/Kaggle حيث كل دفتر يعمل على kernel واحد طويل العمر تتشارك خلاياه نفس
  الذاكرة.

الحل:
  لكل دفتر (session_id) ندير KernelManager واحدًا من jupyter_client:
    - الخلايا تتشارك نفس الذاكرة (نفس الـnamespace)
    - نلتقط streams/display_data/execute_result/errors من قناة iopub
    - execute_reply من قناة shell يحدد ok/error/timeout
    - إمكانية restart kernel (إعادة تهيئة الذاكرة)
  التهيئة كسولة (lazy) وفشل ipykernel يعود للآلية القديمة (subprocess)
  تلقائيًا حتى لا يكسر أي شيء موجود.

الأمان:
  - لا تنفيذ تلقائي؛ كل خلية يشغّلها المستخدم يدويًا عبر الواجهة
  - timeout افتراضي 60ث مع إيقاف الخلية المعلقة (interrupt + restart)
  - kernel يعمل في عملية منفصلة (process isolation) عن Streamlit
  - PYTHONPATH يشمل جذر المشروع حتى تعمل استيرادات ai/* داخل الخلايا
"""
from __future__ import annotations

import os
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_TIMEOUT = 60
_MAX_KERNELS = 16  # حد عملي لمنع تسرّب kernels في بيئة مشاركة


def _ensure_ipykernel() -> bool:
    """التحقق من توفر ipykernel بدون استيراد صريح داخل مسار الإنتاج."""
    try:
        import ipykernel  # noqa: F401
        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# إدارة جلسات الكيرنل (singleton per session)
# ═══════════════════════════════════════════════════════════════════════════
_KERNEL_LOCK = threading.Lock()
_KERNELS: Dict[str, Dict[str, Any]] = {}  # session_id -> {km, kc, at, started_at}


def _available_for_kernel() -> bool:
    return _ensure_ipykernel() and len(_KERNELS) < _MAX_KERNELS


def get_kernel_client(session_id: str) -> Optional[Any]:
    """يعيد KernelClient حيًا للجلسة، ويؤسّس kernel جديدًا عند الحاجة.

    الجلسة تُنشأ مرة واحدة وتبقى حية طوال عمر العملية (مثل Colab).
    """
    if not _available_for_kernel():
        return None
    with _KERNEL_LOCK:
        entry = _KERNELS.get(session_id)
        if entry is not None:
            try:
                if entry["kc"].is_alive():
                    entry["at"] = time.time()
                    return entry["kc"]
            except Exception:
                pass
            # kernel مات أو تعطل — نظّف وأعد البناء
            _dispose_entry(session_id)
        try:
            from jupyter_client import KernelManager
        except Exception:
            return None
        try:
            km = KernelManager(kernel_name="python3")
            km.start_kernel(cwd=str(ROOT))
            kc = km.client()
            kc.start_channels()
            kc.wait_for_ready(timeout=30)
            # تهيئة PYTHONPATH داخل الkernel بحيث تعمل استيرادات المشروع
            env_pythonpath = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
            kc.execute(
                "import sys, os\n"
                f"_np = {env_pythonpath!r}\n"
                "if _np not in sys.path:\n"
                "    sys.path.insert(0, _np)",
                store_history=False,
                silent=True,
            )
            # 🆕 matplotlib inline (Colab/Kaggle style): الصور تُعرض مباشرة في الخلية
            try:
                kc.execute("%matplotlib inline", store_history=False, silent=True)
            except Exception:
                pass
            entry = {"km": km, "kc": kc, "at": time.time(), "started_at": time.time()}
            _KERNELS[session_id] = entry
            return kc
        except Exception:
            _dispose_entry(session_id)
            return None


def _dispose_entry(session_id: str) -> None:
    entry = _KERNELS.pop(session_id, None)
    if entry is None:
        return
    try:
        entry["kc"].stop_channels()
    except Exception:
        pass
    try:
        entry["km"].shutdown_kernel(now=True)
    except Exception:
        pass


def restart_kernel(session_id: str) -> Dict[str, Any]:
    """يقتل kernel الجلسة ويعيد بناءه — ذاكرة صافية مثل Reset في Colab."""
    with _KERNEL_LOCK:
        _dispose_entry(session_id)
    kc = get_kernel_client(session_id)
    ok = kc is not None
    return {"ok": ok, "session_id": session_id, "error": None if ok else "تعذّر إعادة تشغيل kernel"}


def shutdown_session(session_id: str) -> bool:
    with _KERNEL_LOCK:
        was = session_id in _KERNELS
        _dispose_entry(session_id)
    return was


def list_sessions() -> List[Dict[str, Any]]:
    with _KERNEL_LOCK:
        return [
            {
                "session_id": sid,
                "alive": bool(e.get("kc")) and e["kc"].is_alive(),
                "started_at": e.get("started_at"),
            }
            for sid, e in list(_KERNELS.items())
        ]


def kernel_health() -> Dict[str, Any]:
    return {
        "ipykernel_available": _ensure_ipykernel(),
        "active_sessions": len(_KERNELS),
        "sessions": list_sessions(),
        "backend": "kernel" if _available_for_kernel() else "subprocess",
    }


def sessions_detail() -> List[Dict[str, Any]]:
    """🆕 تفاصيل كل جلسات kernel النشطة (uptime/alive) للواجهة."""
    rows = []
    for sid, entry in _KERNELS.items():
        alive = False
        try:
            alive = entry["kc"].is_alive()
        except Exception:
            pass
        rows.append({
            "session_id": sid,
            "alive": alive,
            "started_at": entry.get("started_at"),
            "uptime_s": int(time.time() - entry["started_at"]) if entry.get("started_at") else None,
        })
    return rows


# ═══════════════════════════════════════════════════════════════════════════
# تنفيذ خلية عبر kernel
# ═══════════════════════════════════════════════════════════════════════════
def _collect_outputs(kc: Any, msg_id: str, timeout: int) -> Dict[str, Any]:
    """يجمع كل رسائل iopub الخاصة بالخلية حتى execute_reply أو انقضاء المهلة."""
    outputs: List[Dict[str, Any]] = []
    reply: Optional[Dict[str, Any]] = None
    t_deadline = time.time() + timeout
    idle_after_error = 0.0

    while time.time() < t_deadline:
        # 1) execute_reply يحدد نتيجة التنفيذ (ok/error/abort)
        if reply is None:
            try:
                reply = kc.get_shell_msg(timeout=min(1.0, max(0.1, t_deadline - time.time())))
                if reply.get("parent_header", {}).get("msg_id") != msg_id:
                    reply = None
                    continue
            except Exception:
                pass

        # 2) التقاط رسائل المخرجات المتبقية بسرعة بعد اكتمال التنفيذ
        try:
            m = kc.get_iopub_msg(timeout=min(0.25, max(0.05, t_deadline - time.time())))
        except Exception:
            m = None
        if m is None:
            # لا رسائل متبقية؛ إذا اكتمل التنفيذ فالخلية انتهت
            if reply is not None:
                break
            continue
        if m.get("parent_header", {}).get("msg_id") != msg_id:
            continue
        mt = m["msg_type"]
        content = m.get("content") or {}
        if mt == "stream":
            outputs.append({"type": "stream", "name": content.get("name", "stdout"),
                            "text": content.get("text", "")})
        elif mt == "display_data":
            outputs.append({"type": "display_data", "data": content.get("data") or {},
                            "metadata": content.get("metadata") or {}})
        elif mt == "execute_result":
            outputs.append({"type": "execute_result", "data": content.get("data") or {},
                            "metadata": content.get("metadata") or {},
                            "execution_count": content.get("execution_count")})
        elif mt == "error":
            outputs.append({
                "type": "error",
                "ename": content.get("ename", "Error"),
                "evalue": content.get("evalue", ""),
                "traceback": content.get("traceback") or [],
            })
            # بعد الخطأ نلتقط أي بقايا بسرعة ثم ننتهي
            idle_after_error = time.time()
        elif mt in ("status", "execute_input", "iopub_welcome"):
            pass

        # حد أمان: بعد خطأ، التقت 1.5ث إضافية ثم ننتهي
        if idle_after_error and time.time() - idle_after_error > 1.5:
            break

    # انقضاء المهلة دون execute_reply
    if reply is None:
        return {
            "ok": False,
            "outputs": outputs,
            "timeout": True,
            "error": f"timeout {timeout}s",
            "execution_count": None,
        }

    status = (reply.get("content") or {}).get("status", "error")
    ok = status == "ok"
    exec_count = (reply.get("content") or {}).get("execution_count")
    if not ok:
        # خطأ غير معلَن عبر رسائل iopub (abort) — أضف مخرجاً توضيحيًا إن لم يوجد
        if not any(o["type"] == "error" for o in outputs):
            outputs.append({"type": "error", "ename": "ExecutionError",
                            "evalue": "execution aborted", "traceback": []})
    return {
        "ok": ok,
        "outputs": outputs,
        "timeout": False,
        "error": None,
        "execution_count": exec_count,
    }


def run_cell_kernel(session_id: str, source: str, timeout: int = _DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """يشغّل خلية في kernel الجلسة. يعيد dict جاهزًا للتخزين في Cell.outputs.

    الشكل متوافق مع notebook_engine.outputs بحيث يمكن عرض النتائج بنفس منطق
    الواجهة الحالي.
    """
    t0 = time.time()
    if not source.strip():
        return {"ok": True, "outputs": [], "timeout": False, "error": None,
                "execution_count": None, "duration_ms": 0}
    kc = get_kernel_client(session_id)
    if kc is None:
        return {"ok": False, "outputs": [], "timeout": False,
                "error": "تعذّر تشغيل kernel (ipykernel غير متوفر) — الكود لم يُنفَّذ",
                "execution_count": None, "duration_ms": int((time.time() - t0) * 1000)}
    try:
        msg_id = kc.execute(source, store_history=True)
    except Exception as e:
        return {"ok": False, "outputs": [], "timeout": False, "error": str(e),
                "execution_count": None, "duration_ms": int((time.time() - t0) * 1000)}
    result = _collect_outputs(kc, msg_id, timeout)
    result["duration_ms"] = int((time.time() - t0) * 1000)
    return result


def interrupt_kernel(session_id: str) -> Dict[str, Any]:
    """يوقف خلية معلّقة (Ctrl+C) دون قتل kernel والذاكرة."""
    with _KERNEL_LOCK:
        entry = _KERNELS.get(session_id)
    if entry is None:
        return {"ok": False, "error": "لا kernel نشط للجلسة"}
    try:
        entry["km"].interrupt_kernel()
        return {"ok": True, "error": None}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# أدوات مساعدة للواجهة
# ═══════════════════════════════════════════════════════════════════════════
def output_to_text(parts: List[Dict[str, Any]]) -> Dict[str, str]:
    """يحوّل outputs إلى (stdout, stderr, rich) لعرضها في الواجهة القديمة."""
    stdout, stderr, rich = [], [], []
    for o in parts:
        if o.get("type") == "stream":
            (stdout if o.get("name") != "stderr" else stderr).append(o.get("text", ""))
        elif o.get("type") == "error":
            tb = o.get("traceback") or []
            # strip ANSI من التتبع لأن Streamlit لا يدعمه
            clean = [line.replace("\u001b[0m", "") for line in tb]
            stderr.append("\n".join(clean))
        else:
            rich.append(o)
    return {"stdout": "".join(stdout), "stderr": "".join(stderr),
            "rich": rich, "has_rich": bool(rich)}


def session_summary(session_id: str) -> Dict[str, Any]:
    """معلومات مختصرة عن جلسة kernel للواجهة."""
    with _KERNEL_LOCK:
        entry = _KERNELS.get(session_id)
    if entry is None:
        return {"alive": False}
    alive = False
    try:
        alive = entry["kc"].is_alive()
    except Exception:
        pass
    return {
        "alive": alive,
        "started_at": entry.get("started_at"),
        "uptime_s": int(time.time() - entry["started_at"]) if entry.get("started_at") else None,
    }

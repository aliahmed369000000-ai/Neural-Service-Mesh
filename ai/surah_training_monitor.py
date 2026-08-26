"""
surah_training_monitor — مراقبة حيّة لتدريب SurahChain
=====================================================
تجلب حالة التدريب الحيّة من GitHub (live progress JSON المرفوع كل عصر من
كيرنلات Kaggle مثل `nsm-surahchain-xlarge-12h-fix2`) ومن الكيرنل نفسه عبر
Kaggle API، وتعرضها في لوحة Streamlit:

  • fetch_live_state()   — آخر تقدم مسجّل في GitHub (live) + حالة الكيرنل من Kaggle API
  • render_live_training_dashboard() — لوحة Streamlit جاهزة للإدراج في تبويب

المصدر:
  1) GitHub raw: checkpoints/progress_{TAG}.json أو progress_torch.json
     (يُرفع كل عصر من train_pretrain_torch.py عبر _write_progress + _upload_checkpoint)
  2) Kaggle API: حالة الكيرنل المباشر (QUEUED/RUNNING/COMPLETE/ERROR)

لا يعتمد على أي مفاتيح للقراءة (GitHub raw علني، Kaggle kernels_status يحتاج
~/.kaggle/kaggle.json الموجود لدى المستخدم على Kaggle فقط — هنا نقرأ الحالة
العلنية عبر GitHub فقط داخل Streamlit Community Cloud، وتُترك Kaggle API
اختيارية عند توفر التوكن في الأسرار).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import requests  # noqa: F401
    _REQ_OK = True
except Exception:
    _REQ_OK = False

_ROOT = Path(__file__).resolve().parent.parent
REPO_OWNER = "aliahmed369000000-ai"
REPO_NAME = "Neural-Service-Mesh"
BRANCH = "main"
CKPT_PREFIX = "experiments/surah_chain_network/checkpoints"

# ملف التقدم الافتراضي (التجارب القديمة) + قائمة ملفات محتملة أخرى تُفحص
# بالتسلسل حتى نجد أحدث ملف تقدم فعلي.
PROGRESS_CANDIDATES = [
    "progress_d8192_s1.json",       # تجربة xlarge الحالية (SCN_TAG=d8192_s1)
    "progress_torch.json",          # التسمية القديمة
    "progress_d256_s1.json",
    "progress_d512_s1.json",
]


def _raw_url(filename: str) -> str:
    return (
        f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/"
        f"{BRANCH}/{CKPT_PREFIX}/{filename}"
    )


def _api_contents() -> Optional[List[Dict[str, Any]]]:
    """سرد محتويات مجلد checkpoints عبر GitHub API (علني)."""
    if not _REQ_OK:
        return None
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/"
            f"contents/{CKPT_PREFIX}",
            timeout=15,
            headers={"Accept": "application/vnd.github+json"},
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def fetch_progress_file(candidate: str) -> Optional[Dict[str, Any]]:
    """جلب ملف تقدم معين من GitHub raw."""
    if not _REQ_OK:
        return None
    try:
        resp = requests.get(_raw_url(candidate), timeout=15)
        if resp.status_code == 200:
            data = json.loads(resp.text)
            if isinstance(data, dict) and data.get("epoch") is not None:
                return data
        return None
    except Exception:
        return None


def fetch_latest_progress() -> Optional[Dict[str, Any]]:
    """جلب أحدث ملف تقدم متاح — يجرب المرشحين من الأحدث افتراضيًا."""
    best: Optional[Dict[str, Any]] = None
    best_ts: float = -1.0
    for cand in PROGRESS_CANDIDATES:
        data = fetch_progress_file(cand)
        if data is None:
            continue
        ts = float(data.get("updated_at") or data.get("started_at") or 0)
        if ts > best_ts:
            best, best_ts = data, ts
    return best


def _try_kaggle_status(kernel_slug: str) -> Optional[str]:
    """حالة الكيرنل عبر Kaggle API — تتطلب ~/.kaggle/kaggle.json (محلية فقط).
    داخل Streamlit Community Cloud تُفعل فقط إن وضع المستخدم التوكن في الأسرار."""
    kaggle_user = os.environ.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_USERNAME_SECRET")
    if not kaggle_user:
        return None
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        status = api.kernels_status(kernel_slug)
        return str(status.status).split(".")[-1] if status and status.status else None
    except Exception:
        return None


def kaggle_kernel_status(kernel_slug: str) -> str:
    """حالة الكيرنل مع fallback: Kaggle API أولًا ثم GitHub (آخر تقدم يدل أن الكيرنل يعمل)."""
    status = _try_kaggle_status(kernel_slug)
    if status:
        return status
    prog = fetch_latest_progress()
    if prog is not None:
        # آخر تحديث قبل أقل من ساعة → مرجّح أن الكيرنل ما زال يعمل
        if (time.time() - float(prog.get("updated_at") or 0)) < 3600:
            return "RUNNING (استدلال من آخر تقدم مرفوع)"
        return "غير معروف — آخر تقدم قديم"
    return "لا يوجد تقدم مرفوع بعد"


def fetch_live_state(kernel_slug: str = "aliahmedmo/nsm-surahchain-xlarge-12h-fix2") -> Dict[str, Any]:
    """الحالة الحيّة الكاملة: تقدم + حالة كيرنل + قائمة ملفات checkpoints على GitHub."""
    state: Dict[str, Any] = {
        "kernel_slug": kernel_slug,
        "kernel_status": kaggle_kernel_status(kernel_slug),
        "progress": fetch_latest_progress(),
        "checkpoints": [],
        "fetched_at": time.time(),
    }
    items = _api_contents()
    if items:
        state["checkpoints"] = [
            {"name": it.get("name"), "size": it.get("size")}
            for it in items
            if it.get("type") == "file"
        ]
    return state


# ═══════════════════════════════════════════════════════════════════════════
# واجهة Streamlit
# ═══════════════════════════════════════════════════════════════════════════

def render_live_training_dashboard() -> None:
    """لوحة تقدم التدريب الحي — تُستدعى من تبويب training_ops_hub."""
    import streamlit as st

    st.markdown("### 📡 التدريب الحي — SurahChain على Kaggle")
    st.caption(
        "تُحدَّث مباشرة من ملفات التقدم المرفوعة إلى GitHub كل عصر "
        "(live progress JSON من كيرنلات Kaggle)"
    )

    slug = st.text_input(
        "كيرنل Kaggle",
        value="aliahmedmo/nsm-surahchain-xlarge-12h-fix2",
        key="surah_monitor_slug",
    )

    auto = st.session_state.get("surah_monitor_auto", False)
    if st.checkbox("تحديث تلقائي كل 30 ثانية", value=auto, key="surah_monitor_auto_cb"):
        st.session_state.surah_monitor_auto = True
        st.rerun()

    import time as _time
    with st.spinner("جلب الحالة الحيّة…"):
        live = fetch_live_state(slug)

    # ── صف الحالة العامة ──────────────────────────────────────────────────
    status = live.get("kernel_status") or "—"
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.metric("حالة الكيرنل", status)
    prog = live.get("progress") or {}
    with kpi_cols[1]:
        st.metric("العصر الحالي", f"{prog.get('epoch', '—')}/{prog.get('end_epoch', '—')}")
    with kpi_cols[2]:
        st.metric("الخسارة الأخيرة", f"{prog.get('loss', '—') if prog.get('loss') is not None else '—'}")
    with kpi_cols[3]:
        # حساب معدل التوكنات (تقديري بناءً على global_step و elapsed و batch)
        # N = global_step * batch * max_len
        batch = int(prog.get("batch") or 24)
        max_len = int(prog.get("max_len") or 96)
        step = int(prog.get("global_step") or 0)
        elapsed = float(prog.get("elapsed") or 0)
        tps = (step * batch * max_len) / elapsed if elapsed > 0 else 0
        st.metric("سرعة المعالجة", f"{tps:.0f} T/s", help="معدل التوكنات في الثانية (تقديري)")

    if prog:
        elapsed = float(prog.get("elapsed") or 0)
        if elapsed:
            h, rem = divmod(int(elapsed), 3600)
            m, s = divmod(rem, 60)
            st.caption(
                f"زمن الجلسة: {h}س {m}د {s}ث | "
                f"learning_rate={prog.get('lr', '—')} | "
                f"global_step={prog.get('global_step', '—')}"
            )
    else:
        st.warning(
            "لم يُرفع ملف تقدم بعد — الكيرنل غالبًا في مرحلة التحميل "
            "(تسجيل Kaggle buffered). سيظهر التقدم مع أول رفع تلقائي بعد عصر."
        )

    # ── حالة الكيرنل من Kaggle API — تحديث يدوي ──────────────────────────
    col_refresh, _ = st.columns([1, 6])
    with col_refresh:
        if st.button("🔄 تحديث حالة Kaggle API", key="surah_monitor_refresh"):
            with st.spinner("Kaggle API…"):
                api_status = _try_kaggle_status(slug)
                st.session_state["surah_monitor_kaggle_status"] = api_status
    k_api = st.session_state.get("surah_monitor_kaggle_status")
    if k_api:
        st.info(f"Kaggle API مباشر: **{k_api}**")

    # ── قائمة الملفات على GitHub ─────────────────────────────────────────
    ckpts = live.get("checkpoints") or []
    if ckpts:
        with st.expander(f"🗂️ ملفات checkpoints على GitHub ({len(ckpts)})"):
            for it in ckpts:
                size_mb = (it.get("size") or 0) / (1024 * 1024)
                st.text(f"• {it['name']}  ({size_mb:.2f} MB)")

    # ── رسم منحنى الخسارة إن توفّر تاريخ ─────────────────────────────────
    hist = prog.get("history") if prog else None
    if hist and len(hist) >= 2:
        import streamlit as st
        st.markdown("#### 📉 منحنى الخسارة")
        import pandas as pd
        df = pd.DataFrame(hist)
        st.line_chart(df.set_index("epoch")[["loss", "lr"]])
    st.caption("آخر جلب: " + _time.strftime("%H:%M:%S", _time.localtime(live.get("fetched_at") or 0)))


def live_training_state_text(kernel_slug: str = "aliahmedmo/nsm-surahchain-xlarge-12h-fix2") -> str:
    """نسخة نصية من الحالة الحيّة — للاستخدام داخل الوكلاء (Terminal/Agent)."""
    live = fetch_live_state(kernel_slug)
    prog = live.get("progress") or {}
    lines = [
        "## 📡 حالة تدريب SurahChain الحيّ",
        f"- الكيرنل: `{live.get('kernel_slug')}` — {live.get('kernel_status')}",
    ]
    if prog:
        lines += [
            f"- العصر: {prog.get('epoch')}/{prog.get('end_epoch')}",
            f"- الخسارة: {prog.get('loss')}",
            f"- أفضل خسارة: {prog.get('best_loss')}",
            f"- زمن الجلسة: {prog.get('elapsed', 0):.0f} ثانية",
        ]
    else:
        lines.append("- لا تقدم مرفوعًا بعد (مرحلة تحميل الكيرنل)")
    return "\n".join(lines)

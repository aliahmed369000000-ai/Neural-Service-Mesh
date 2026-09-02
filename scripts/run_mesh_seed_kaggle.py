#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""تشغيل عقدة بذرة Living Mesh داخل Kaggle Notebook + تعريضها للإنترنت
عبر Cloudflare Tunnel (اتصال صادر فقط — متوافق مع قيود شبكة Kaggle).

لماذا هذا السكربت؟
-------------------
Kaggle Notebooks (تماماً مثل Streamlit Community Cloud — راجع docs/real_p2p_design.md
وسجل commit c46e1ac) تسمح بالاتصال **الصادر** فقط؛ لا يوجد منفذ عام وارد يمكن لعقد
أخرى الاتصال به مباشرة. لذلك عقدة `ai/node_launcher.py` وحدها لن تكون مرئية للشبكة
إن شُغّلت هنا كما هي. هذا السكربت يحل المشكلة بنفس أسلوب
`scripts/run_mesh_seed_termux.sh`: يشغّل العقدة محلياً على 127.0.0.1، ثم يفتح
Cloudflare Quick Tunnel (اتصال صادر) يعرّض المنفذ برابط عام *.trycloudflare.com.

الاستخدام داخل خلية Kaggle (بعد استنساخ المستودع):
    !python scripts/run_mesh_seed_kaggle.py

أو بمدة/منفذ/معرّف مخصص:
    PORT=7860 NODE_ID=mesh_seed_kaggle DURATION_HOURS=12 !python scripts/run_mesh_seed_kaggle.py

ملاحظات مهمة:
- لتشغيل حقيقي لمدة 12 ساعة بدون إبقاء المتصفح مفتوحاً: احفظ نسخة الدفتر
  واستخدم "Save Version → Save & Run All" (نفس أسلوب notebooks/README_KAGGLE.md
  لجلسات SurahChain الليلية) بدل الاعتماد على جلسة تفاعلية.
- رابط النفق (*.trycloudflare.com) عشوائي ويتغيّر مع كل تشغيل جديد — بعد ظهوره
  حدّث قيمة SEED_NODE_URL في أسرار Streamlit Cloud إن أردت ربط العقد ببعضها.
- Kaggle يفرض حد أقصى لطول الجلسة (~12 ساعة للـGPU)؛ هذا السكربت يوقف نفسه
  تلقائياً عند بلوغ `DURATION_HOURS` لتفادي انقطاع مفاجئ من المنصة نفسها.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
import stat
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-linux-amd64"
)


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def ensure_cloudflared(bin_dir: Path) -> Path:
    """يحمّل الثنائي الرسمي لـ cloudflared إن لم يكن متوفراً (اتصال صادر فقط)."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    target = bin_dir / "cloudflared"
    if target.exists() and os.access(target, os.X_OK):
        _log(f"✅ cloudflared موجود مسبقاً: {target}")
        return target
    _log("⬇️ تحميل cloudflared (ثنائي رسمي من GitHub Releases)...")
    urllib.request.urlretrieve(CLOUDFLARED_URL, target)
    target.chmod(target.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    _log(f"✅ تم التحميل: {target}")
    return target


def wait_port_ready(host: str, port: int, timeout: float = 30.0) -> bool:
    import socket

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def start_node(node_id: str, port: int, data_dir: Path, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["NODE_ID"] = node_id
    env["PORT"] = str(port)
    env["NSM_NODE_DATA_DIR"] = str(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "a", buffering=1, encoding="utf-8")
    _log(f"🚀 تشغيل العقدة (NODE_ID={node_id}, PORT={port})...")
    return subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "ai" / "node_launcher.py"),
            "--id", node_id,
            "--host", "0.0.0.0",
            "--port", str(port),
            "--data-dir", str(data_dir),
        ],
        cwd=str(ROOT),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )


def start_tunnel(cloudflared_bin: Path, port: int, log_path: Path) -> subprocess.Popen:
    log_f = open(log_path, "a", buffering=1, encoding="utf-8")
    _log("🌐 تشغيل Cloudflare Quick Tunnel (--protocol http2)...")
    # القيمة الافتراضية لـcloudflared (auto) تجرّب QUIC عبر UDP أولاً، وشبكة
    # Kaggle تحجب/تُسقط UDP الصادر بصمت — فيعلق cloudflared بلا أي رابط ولا
    # أي رسالة خطأ واضحة (يتكرر هذا كل ~35 دقيقة حسب مهلات إعادة المحاولة
    # الداخلية لديه). --protocol http2 يفرض النقل عبر TCP:443 مباشرة ويتفادى
    # محاولة QUIC كلياً — هذا هو الإصلاح الموثّق لهذه الحالة تحديداً في بيئات
    # تحجب UDP الصادر (راجع: github.com/cloudflare/cloudflared/issues/758).
    return subprocess.Popen(
        [
            str(cloudflared_bin),
            "tunnel",
            "--protocol", "http2",
            "--url", f"http://localhost:{port}",
        ],
        cwd=str(ROOT),
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )


def read_tunnel_url(log_path: Path, timeout: float = 30.0) -> str:
    deadline = time.time() + timeout
    pattern = re.compile(r"https://[a-zA-Z0-9.-]+\.trycloudflare\.com")
    while time.time() < deadline:
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            m = pattern.search(text)
            if m:
                return m.group(0)
        time.sleep(1.0)
    return ""


def main() -> None:
    node_id = os.environ.get("NODE_ID", "mesh_seed_kaggle")
    port = int(os.environ.get("PORT", "7860"))
    duration_hours = float(os.environ.get("DURATION_HOURS", "12"))
    duration_seconds = duration_hours * 3600.0

    log_dir = ROOT / "artifacts" / "living_mesh" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    node_log = log_dir / f"node_{node_id}.log"
    tunnel_log = log_dir / f"cloudflared_{node_id}.log"
    data_dir = ROOT / "artifacts" / "living_mesh" / "nodes" / node_id
    url_file = ROOT / "mesh_seed_tunnel_url.txt"

    cloudflared_bin = ensure_cloudflared(ROOT / "artifacts" / "bin")

    node_proc = start_node(node_id, port, data_dir, node_log)

    if not wait_port_ready("127.0.0.1", port, timeout=30.0):
        _log(f"❌ المنفذ {port} لم يصبح جاهزاً خلال 30 ثانية — راجع السجل: {node_log}")
    else:
        _log(f"✅ المنفذ {port} جاهز محلياً.")

    tunnel_proc = start_tunnel(cloudflared_bin, port, tunnel_log)
    tunnel_url = read_tunnel_url(tunnel_log, timeout=30.0)

    if not tunnel_url:
        _log(f"❌ لم يظهر رابط النفق بعد 30 ثانية. راجع السجل: {tunnel_log}")
    else:
        host_only = tunnel_url.replace("https://", "")
        url_file.write_text(tunnel_url + "\n", encoding="utf-8")
        _log("✅ العقدة شغّالة وظاهرة للإنترنت على:")
        _log(f"   {tunnel_url}/status")
        _log("📋 بإعدادات Streamlit Cloud (Secrets) أضف:")
        _log("   NSM_ENABLE_NODE = true")
        _log(f"   SEED_NODE_URL = {host_only}:443")

    _log(f"📄 سجل العقدة: {node_log}")
    _log(f"📄 سجل النفق:  {tunnel_log}")
    _log(f"⏱️ سيستمر التشغيل حتى {duration_hours:g} ساعة أو حتى إيقاف الدفتر يدوياً.")

    stop = {"flag": False}

    def _handle_sigterm(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    start_time = time.time()
    heartbeat_interval = 300.0  # كل 5 دقائق: تحقق من صحة العمليتين واطبع نبضاً
    last_heartbeat = 0.0

    try:
        while not stop["flag"]:
            elapsed = time.time() - start_time
            if elapsed >= duration_seconds:
                _log(f"⏰ بلغ التشغيل الحد الأقصى ({duration_hours:g} ساعة) — إيقاف آمن.")
                break

            if node_proc.poll() is not None:
                _log(f"❌ عملية العقدة توقفت (rc={node_proc.returncode}) — راجع {node_log}")
                break

            if tunnel_proc.poll() is not None:
                _log("⚠️ نفق Cloudflare توقف — إعادة تشغيل النفق...")
                tunnel_proc = start_tunnel(cloudflared_bin, port, tunnel_log)
                new_url = read_tunnel_url(tunnel_log, timeout=30.0)
                if new_url and new_url != tunnel_url:
                    tunnel_url = new_url
                    url_file.write_text(tunnel_url + "\n", encoding="utf-8")
                    host_only = tunnel_url.replace("https://", "")
                    _log(f"🔄 رابط نفق جديد: {tunnel_url}/status")
                    _log(f"   SEED_NODE_URL = {host_only}:443")

            if elapsed - last_heartbeat >= heartbeat_interval:
                last_heartbeat = elapsed
                remaining_min = max(0.0, (duration_seconds - elapsed) / 60.0)
                _log(f"💓 نبض: العقدة والنفق يعملان — الوقت المتبقي ≈ {remaining_min:.0f} دقيقة.")

            time.sleep(5.0)
    finally:
        _log("🛑 إيقاف العقدة والنفق...")
        for proc in (node_proc, tunnel_proc):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
        if url_file.exists():
            url_file.unlink()
        _log("👋 تم إيقاف الجلسة بأمان.")


if __name__ == "__main__":
    main()

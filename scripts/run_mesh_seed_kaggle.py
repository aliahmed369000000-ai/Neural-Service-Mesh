#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""تشغيل عقدة بذرة Living Mesh داخل Kaggle Notebook + تعريضها للإنترنت
عبر سلسلة أنفاق بديلة (بلا Cloudflare — راجع "لماذا لا Cloudflare" أدناه).

لماذا هذا السكربت؟
-------------------
Kaggle Notebooks (تماماً مثل Streamlit Community Cloud — راجع docs/real_p2p_design.md
وسجل commit c46e1ac) تسمح بالاتصال **الصادر** فقط؛ لا يوجد منفذ عام وارد يمكن لعقد
أخرى الاتصال به مباشرة. لذلك عقدة `ai/node_launcher.py` وحدها لن تكون مرئية للشبكة
إن شُغّلت هنا كما هي. هذا السكربت يحل المشكلة بتشغيل العقدة محلياً على 127.0.0.1،
ثم فتح نفق صادر يعرّضها برابط عام.

لماذا لا Cloudflare (v12 — أُزيل بالكامل):
--------------------------------------------
جُرِّب Cloudflare Quick Tunnel أولاً (commit fa7a5a6)، ثم أُصلح ليفرض النقل عبر
TCP:443 بدل QUIC/UDP المحجوب على شبكة Kaggle (commit b45eae2). حتى بعد هذا
الإصلاح، تبيّن عملياً أن الاتصال الصادر نفسه إلى `api.trycloudflare.com` يتوقف
بمهلة زمنية (timeout) بلا أي رابط على شبكة Kaggle تحديداً — أي أن الحجب أوسع من
QUIC وحده. لذلك أُزيل Cloudflare من هذا السكربت كلياً، واستُبدل بسلسلة بدائل
لا تعتمد على نطاق Cloudflare إطلاقاً:

  1. bore.pub  — نفق TCP خام عبر ثنائي `bore` الرسمي (github.com/ekzhang/bore).
  2. ngrok     — إن وُجد NGROK_AUTHTOKEN في البيئة (موثوق أكثر عند توفر التوكن).
  3. serveo.net — نفق عبر SSH العادي (منفذ 22 صادر)، بلا أي تثبيت إضافي.
  4. localhost.run — نفس أسلوب SSH، كبديل أخير إن حُجب serveo.net أيضاً.

يُجرَّب كل بديل بالترتيب أعلاه لمدة محدودة، وأول من ينجح يُستخدم لبقية الجلسة
(مع محاولة استئناف نفس البديل الناجح إن انقطع، والانتقال للتالي بعد عدة فشل).

v12.1 (إصلاح مهم): رفض الإعلان الكاذب عن نجاح bore عندما تظهر كلمة
"bore.pub:PORT" داخل رسالة خطأ (timed out / could not connect). كان هذا
السبب الذي جعل تشغيل Kaggle السابق يتوقف عند bore الفاشل ولا يجرّب SSH.
الآن يُشترط ظهور العبارة الإيجابية "listening at bore.pub:..." + بقاء
العملية حيّة قبل اعتبار البديل ناجحاً.

v12.2: إضافة ngrok كبديل (يُفعَّل فقط عند وجود NGROK_AUTHTOKEN).

الاستخدام داخل خلية Kaggle (بعد استنساخ المستودع):
    !python scripts/run_mesh_seed_kaggle.py

أو بمدة/منفذ/معرّف مخصص:
    PORT=7860 NODE_ID=mesh_seed_kaggle DURATION_HOURS=12 !python scripts/run_mesh_seed_kaggle.py

ملاحظات مهمة:
- لتشغيل حقيقي لمدة 12 ساعة بدون إبقاء المتصفح مفتوحاً: احفظ نسخة الدفتر
  واستخدم "Save Version → Save & Run All" (نفس أسلوب notebooks/README_KAGGLE.md
  لجلسات SurahChain الليلية) بدل الاعتماد على جلسة تفاعلية.
- الرابط العام عشوائي ويتغيّر مع كل تشغيل جديد (ولكل بديل صيغة مختلفة) — بعد
  ظهوره حدّث قيمة SEED_NODE_URL في أسرار Streamlit Cloud إن أردت ربط العقد ببعضها.
- Kaggle يفرض حد أقصى لطول الجلسة (~12 ساعة للـGPU)؛ هذا السكربت يوقف نفسه
  تلقائياً عند بلوغ `DURATION_HOURS` لتفادي انقطاع مفاجئ من المنصة نفسها.
- إن فشلت الأنفاق الثلاثة كلها (شبكات مؤسسية/جامعية أحياناً تحجب SSH الصادر
  أيضاً)، آخر خيار عملي هو تشغيل العقدة البذرة خارج Kaggle (VPS رخيص أو جهازك
  الشخصي عبر scripts/run_mesh_seed_termux.sh أو scripts/run_local_mesh.py).
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

from ai.tunnel_providers import TunnelCandidate, verify_tunnel

BORE_RELEASES_API = "https://api.github.com/repos/ekzhang/bore/releases/latest"
BORE_ASSET_PATTERNS = (
    re.compile(r"^bore-v[\d.]+-x86_64-unknown-linux-musl\.tar\.gz$"),
    re.compile(r"^bore-v[\d.]+-x86_64-unknown-linux-gnu\.tar\.gz$"),
)
# احتياطي أخير إن فشل استعلام GitHub API (حد الطلبات 60/ساعة لكل IP بلا توكن،
# وقد يكون IP خلية Kaggle مشتركاً مع مستخدمين آخرين يستهلكونه). قد يصبح هذا
# الإصدار قديماً مع الوقت — لذا هو آخر محاولة فقط بعد فشل الاستعلام الديناميكي.
BORE_FALLBACK_URL = (
    "https://github.com/ekzhang/bore/releases/download/v0.6.0/"
    "bore-v0.6.0-x86_64-unknown-linux-musl.tar.gz"
)


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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


# ---------------------------------------------------------------------------
# البديل 1: bore.pub (نفق TCP خام — لا يحتاج SSH، ثنائي مستقل)
# ---------------------------------------------------------------------------

def ensure_bore(bin_dir: Path) -> Optional[Path]:
    """يحمّل ثنائي bore الرسمي (اتصال صادر فقط) إن لم يكن متوفراً."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    target = bin_dir / "bore"
    if target.exists() and os.access(target, os.X_OK):
        _log(f"✅ bore موجود مسبقاً: {target}")
        return target
    try:
        _log("⬇️ جلب معلومات آخر إصدار bore من GitHub API...")
        headers = {"Accept": "application/vnd.github+json"}
        # اختياري: GITHUB_TOKEN يرفع حد GitHub API من 60 لـ5000 طلب/ساعة —
        # مفيد إن كانت شبكة Kaggle تشارك IP مع مستخدمين آخرين يستهلكون الحد.
        gh_token = os.environ.get("GITHUB_TOKEN") or os.environ.get("NSM_GITHUB_TOKEN")
        if gh_token:
            headers["Authorization"] = f"Bearer {gh_token}"
        req = urllib.request.Request(BORE_RELEASES_API, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as resp:
            release = json.loads(resp.read().decode("utf-8"))
        assets = release.get("assets") or []
        asset_url = None
        for pattern in BORE_ASSET_PATTERNS:
            for asset in assets:
                if pattern.match(asset.get("name", "")):
                    asset_url = asset.get("browser_download_url")
                    break
            if asset_url:
                break
        if not asset_url:
            _log("❌ لم يُعثر على ثنائي bore مناسب (x86_64 linux) في آخر إصدار.")
            return None
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError) as e:
        _log(f"⚠️ فشل استعلام GitHub API ({e}) — تجربة رابط احتياطي ثابت...")
        asset_url = BORE_FALLBACK_URL
    try:
        _log(f"⬇️ تحميل bore: {asset_url}")
        with urllib.request.urlopen(asset_url, timeout=60) as resp:
            data = resp.read()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            member = next((m for m in tf.getmembers() if m.name.split("/")[-1] == "bore"), None)
            if member is None:
                _log("❌ لم يُعثر على ملف bore داخل الأرشيف المحمّل.")
                return None
            extracted = tf.extractfile(member)
            if extracted is None:
                return None
            target.write_bytes(extracted.read())
        target.chmod(target.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        _log(f"✅ تم تحميل bore: {target}")
        return target
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, tarfile.TarError, json.JSONDecodeError) as e:
        _log(f"❌ فشل تحميل bore: {e}")
        return None


def start_bore(bore_bin: Path, port: int, log_path: Path) -> subprocess.Popen:
    log_f = open(log_path, "a", buffering=1, encoding="utf-8")
    _log("🌐 تشغيل نفق bore.pub...")
    return subprocess.Popen(
        [str(bore_bin), "local", str(port), "--to", "bore.pub"],
        cwd=str(ROOT),
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )


def read_bore_url(log_path: Path, timeout: float = 30.0) -> str:
    """يستخرج رابط bore فقط عند ظهور العبارة الإيجابية الحقيقية.
    يرفض صراحةً رسائل الخطأ التي تحتوي على bore.pub:PORT (كان هذا سبب
    الإعلان الكاذب عن نجاح في تشغيل Kaggle السابق).
    """
    deadline = time.time() + timeout
    positive = re.compile(r"listening at bore\.pub:(\d+)", re.IGNORECASE)
    # كلمات تدل على فشل الاتصال — إن وُجدت قرب المطابقة نرفضها
    negative = re.compile(
        r"(could not connect|timed out|Error:|connection refused|failed to|unable to)",
        re.IGNORECASE,
    )
    while time.time() < deadline:
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            m = positive.search(text)
            if m:
                # افحص نافذة حول المطابقة بحثاً عن كلمات فشل
                start = max(0, m.start() - 150)
                snippet = text[start : m.end() + 60]
                if negative.search(snippet):
                    # مطابقة داخل رسالة خطأ — تجاهل وانتظر أكثر
                    pass
                else:
                    return f"bore.pub:{m.group(1)}"
        time.sleep(1.0)
    return ""


# ---------------------------------------------------------------------------
# البديل 2: ngrok (يتطلب NGROK_AUTHTOKEN — يُتخطى تلقائياً إن غاب)
# ---------------------------------------------------------------------------

NGROK_DOWNLOAD_URL = (
    "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"
)


def ensure_ngrok(bin_dir: Path) -> Optional[Path]:
    """يحمّل ثنائي ngrok الرسمي إن لم يكن متوفراً."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    target = bin_dir / "ngrok"
    if target.exists() and os.access(target, os.X_OK):
        _log(f"✅ ngrok موجود مسبقاً: {target}")
        return target
    try:
        _log(f"⬇️ تحميل ngrok: {NGROK_DOWNLOAD_URL}")
        with urllib.request.urlopen(NGROK_DOWNLOAD_URL, timeout=90) as resp:
            data = resp.read()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            member = next(
                (m for m in tf.getmembers() if m.name.split("/")[-1] == "ngrok"),
                None,
            )
            if member is None:
                _log("❌ لم يُعثر على ملف ngrok داخل الأرشيف.")
                return None
            extracted = tf.extractfile(member)
            if extracted is None:
                return None
            target.write_bytes(extracted.read())
        target.chmod(target.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        _log(f"✅ تم تحميل ngrok: {target}")
        return target
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, tarfile.TarError) as e:
        _log(f"❌ فشل تحميل ngrok: {e}")
        return None


def write_ngrok_config_file(cfg_dir: Path, authtoken: str) -> Optional[Path]:
    """يكتب ملف تهيئة ngrok الرسمي (v3) بشكل صريح، إضافةً لأمر add-authtoken
    (وليس بديلاً عنه) — يفيد في نسخ ngrok/بيئات Kaggle التي لا تلتقط أمر
    config add-authtoken بشكل موثوق، عبر تمرير --config صراحةً لاحقاً."""
    try:
        cfg_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = cfg_dir / "ngrok.yml"
        # صيغة v3 الرسمية (agent.authtoken) — بدون أي اعتماد على مكتبة yaml خارجية
        cfg_path.write_text(
            "version: 3\nagent:\n  authtoken: " + authtoken + "\n",
            encoding="utf-8",
        )
        _log(f"📝 كُتب ملف تهيئة ngrok الإضافي: {cfg_path}")
        return cfg_path
    except OSError as e:
        _log(f"⚠️ تعذّرت كتابة ملف تهيئة ngrok الإضافي (لن يوقف التشغيل): {e}")
        return None


def start_ngrok(ngrok_bin: Path, port: int, log_path: Path, authtoken: str) -> Optional[subprocess.Popen]:
    """يشغّل ngrok http بعد ضبط الـ authtoken."""
    # ضبط التوكن (مرة واحدة — يكتب في ~/.ngrok2 أو config محلي)
    try:
        cfg_dir = ROOT / "artifacts" / "ngrok_cfg"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        # نمرّر التوكن عبر متغير بيئة + أمر config لتجنب الاعتماد على ملف المستخدم
        env = os.environ.copy()
        env["NGROK_AUTHTOKEN"] = authtoken
        # config add-authtoken (صامت إن نجح سابقاً)
        subprocess.run(
            [str(ngrok_bin), "config", "add-authtoken", authtoken],
            cwd=str(cfg_dir),
            env=env,
            capture_output=True,
            timeout=20,
            check=False,
        )
        # إضافي (لا يستبدل السطر أعلاه): ملف تهيئة v3 صريح + تمريره بـ --config
        # كطبقة أمان ثانية إن لم يلتقط ngrok أمر add-authtoken لأي سبب.
        explicit_cfg = write_ngrok_config_file(cfg_dir, authtoken)
        log_f = open(log_path, "a", buffering=1, encoding="utf-8")
        _log("🌐 تشغيل نفق ngrok...")
        cmd = [str(ngrok_bin)]
        if explicit_cfg is not None:
            cmd += ["--config", str(explicit_cfg)]
        cmd += ["http", str(port), "--log=stdout", "--log-format=logfmt"]
        # --log=stdout يجعل الرسائل تظهر في الملف الذي نراقبه
        return subprocess.Popen(
            cmd,
            cwd=str(cfg_dir),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
    except Exception as e:
        _log(f"❌ فشل بدء ngrok: {e}")
        return None


def read_ngrok_url(log_path: Path, timeout: float = 35.0) -> str:
    """يستخرج رابط https://xxxx.ngrok-free.app أو ngrok.io من سجل ngrok."""
    deadline = time.time() + timeout
    # صيغ شائعة: url=https://xxxx.ngrok-free.app  أو  Forwarding  https://...
    patterns = [
        re.compile(r"url=(https://[a-zA-Z0-9.-]+\.ngrok(?:-free)?\.(?:app|io|dev)[^\s]*)"),
        re.compile(r"https://[a-zA-Z0-9.-]+\.ngrok(?:-free)?\.(?:app|io|dev)"),
        re.compile(r"Forwarding\s+(https://[a-zA-Z0-9.-]+\.ngrok[^\s]*)"),
    ]
    while time.time() < deadline:
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            for pat in patterns:
                m = pat.search(text)
                if m:
                    url = m.group(1) if m.lastindex else m.group(0)
                    # نظّف أي محارف زائدة
                    url = url.strip().rstrip('"').rstrip("'")
                    if url.startswith("https://"):
                        return url
        time.sleep(1.0)
    return ""


# ---------------------------------------------------------------------------
# البديل 3 و4: serveo.net و localhost.run (كلاهما نفق عبر SSH القياسي)
# ---------------------------------------------------------------------------

def start_ssh_tunnel(remote_host: str, remote_user: str, port: int, log_path: Path) -> Optional[subprocess.Popen]:
    if shutil.which("ssh") is None:
        _log("❌ أمر ssh غير متوفر على هذه البيئة — تخطّي هذا البديل.")
        return None
    log_f = open(log_path, "a", buffering=1, encoding="utf-8")
    target = f"{remote_user}@{remote_host}" if remote_user else remote_host
    _log(f"🌐 تشغيل نفق SSH عبر {remote_host}...")
    return subprocess.Popen(
        [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ServerAliveInterval=30",
            "-o", "ExitOnForwardFailure=yes",
            "-R", f"80:localhost:{port}",
            target,
        ],
        cwd=str(ROOT),
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )


def read_ssh_tunnel_url(log_path: Path, domain_suffixes: tuple, timeout: float = 25.0) -> str:
    deadline = time.time() + timeout
    suffix_pattern = "|".join(re.escape(s) for s in domain_suffixes)
    pattern = re.compile(rf"https://([a-zA-Z0-9.-]+(?:{suffix_pattern}))")
    while time.time() < deadline:
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            m = pattern.search(text)
            if m:
                return m.group(1)
        time.sleep(1.0)
    return ""


TUNNEL_PROVIDERS = ("bore", "ngrok", "serveo", "localhost_run")


def _verified_tunnel(provider: str, proc: subprocess.Popen, url: str):
    """تحقق فعلي من /health قبل إعلان نجاح أي نفق."""
    candidate = TunnelCandidate(provider, url, lambda: proc.poll() is None)
    if verify_tunnel(candidate, timeout=float(os.environ.get("TUNNEL_HEALTH_TIMEOUT", "5"))):
        return proc, url
    _log(f"❌ الرابط {url} ظهر لكن /health لم يثبت صلاحية النفق — رفض الإعلان.")
    try:
        proc.terminate()
    except OSError:
        pass
    return None, ""


def try_start_tunnel(provider: str, port: int, log_dir: Path, node_id: str):
    """يحاول تشغيل بديل نفق واحد. يُعيد (proc, public_url) أو (None, "") عند الفشل.
    يتحقق أن العملية ما زالت حيّة بعد استخراج الرابط لتجنب الإعلان الكاذب.
    """
    if provider == "bore":
        bore_bin = ensure_bore(ROOT / "artifacts" / "bin")
        if bore_bin is None:
            return None, ""
        log_path = log_dir / f"bore_{node_id}.log"
        proc = start_bore(bore_bin, port, log_path)
        url = read_bore_url(log_path)
        if not url or proc.poll() is not None:
            if not url:
                _log(f"❌ لم يظهر رابط bore.pub الإيجابي خلال المهلة. راجع السجل: {log_path}")
            else:
                _log("❌ ظهرت عبارة bore لكن العملية توقفت فوراً — اعتباره فشلاً.")
            try:
                proc.terminate()
            except Exception:
                pass
            return None, ""
        return _verified_tunnel(provider, proc, f"https://{url}")

    if provider == "ngrok":
        authtoken = (
            os.environ.get("NGROK_AUTHTOKEN")
            or os.environ.get("NGROK_TOKEN")
            or os.environ.get("NSM_NGROK_AUTHTOKEN")
            or ""
        ).strip()
        if not authtoken:
            _log("⏭️ تخطّي ngrok — لا يوجد NGROK_AUTHTOKEN في البيئة.")
            return None, ""
        ngrok_bin = ensure_ngrok(ROOT / "artifacts" / "bin")
        if ngrok_bin is None:
            return None, ""
        log_path = log_dir / f"ngrok_{node_id}.log"
        proc = start_ngrok(ngrok_bin, port, log_path, authtoken)
        if proc is None:
            return None, ""
        url = read_ngrok_url(log_path)
        if not url or proc.poll() is not None:
            _log(f"❌ لم يظهر رابط ngrok خلال المهلة (أو توقفت العملية). راجع السجل: {log_path}")
            try:
                proc.terminate()
            except Exception:
                pass
            return None, ""
        # لا نعلن الرابط قبل نجاح فحص /health فعلياً.
        return _verified_tunnel(provider, proc, url)

    if provider == "serveo":
        log_path = log_dir / f"serveo_{node_id}.log"
        proc = start_ssh_tunnel("serveo.net", "", port, log_path)
        if proc is None:
            return None, ""
        host = read_ssh_tunnel_url(log_path, (".serveo.net",), timeout=35.0)
        if not host or proc.poll() is not None:
            _log(f"❌ لم يظهر رابط serveo.net خلال المهلة (أو انقطع SSH). راجع السجل: {log_path}")
            try:
                proc.terminate()
            except Exception:
                pass
            return None, ""
        return _verified_tunnel(provider, proc, f"https://{host}:443")

    if provider == "localhost_run":
        log_path = log_dir / f"localhost_run_{node_id}.log"
        proc = start_ssh_tunnel("localhost.run", "nokey", port, log_path)
        if proc is None:
            return None, ""
        host = read_ssh_tunnel_url(log_path, (".lhr.life", ".localhost.run"), timeout=35.0)
        if not host or proc.poll() is not None:
            _log(f"❌ لم يظهر رابط localhost.run خلال المهلة (أو انقطع SSH). راجع السجل: {log_path}")
            try:
                proc.terminate()
            except Exception:
                pass
            return None, ""
        return _verified_tunnel(provider, proc, f"https://{host}:443")

    return None, ""


def start_any_tunnel(port: int, log_dir: Path, node_id: str, skip: str = ""):
    """يجرّب البدائل بالترتيب (مع تخطي `skip` إن كان قد فشل للتو) ويُعيد أول ناجح."""
    order = [p for p in TUNNEL_PROVIDERS if p != skip] + ([skip] if skip else [])
    for provider in order:
        _log(f"🔎 تجربة البديل: {provider}")
        proc, url = try_start_tunnel(provider, port, log_dir, node_id)
        if proc is not None and url:
            _log(f"✅ نجح البديل {provider}: {url}")
            return provider, proc, url
    return None, None, ""


def main() -> None:
    node_id = os.environ.get("NODE_ID", "mesh_seed_kaggle")
    port = int(os.environ.get("PORT", "7860"))
    duration_hours = float(os.environ.get("DURATION_HOURS", "12"))
    duration_seconds = duration_hours * 3600.0

    log_dir = ROOT / "artifacts" / "living_mesh" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    node_log = log_dir / f"node_{node_id}.log"
    data_dir = ROOT / "artifacts" / "living_mesh" / "nodes" / node_id
    url_file = ROOT / "mesh_seed_tunnel_url.txt"

    node_proc = start_node(node_id, port, data_dir, node_log)

    if not wait_port_ready("127.0.0.1", port, timeout=30.0):
        _log(f"❌ المنفذ {port} لم يصبح جاهزاً خلال 30 ثانية — راجع السجل: {node_log}")
    else:
        _log(f"✅ المنفذ {port} جاهز محلياً.")

    provider, tunnel_proc, public_url = start_any_tunnel(port, log_dir, node_id)

    if not public_url:
        _log("❌ فشلت كل البدائل (bore.pub / ngrok / serveo.net / localhost.run).")
        _log("   إن أردت تفعيل ngrok: صدّر NGROK_AUTHTOKEN قبل التشغيل.")
        _log("   إذا لم يظهر الرابط هنا، آخر خيار عملي هو تشغيل عقدة البذرة خارج")
        _log("   Kaggle (VPS أو جهازك عبر scripts/run_mesh_seed_termux.sh).")
        _log("   راجع سجلات artifacts/living_mesh/logs/ للتفاصيل.")
    else:
        url_file.write_text(public_url + "\n", encoding="utf-8")
        prov = (provider or "unknown").upper()
        _log(f"✅ v12 (NO Cloudflare) — العقدة شغّالة وظاهرة عبر {prov}")
        _log("📋 بإعدادات Streamlit Cloud (Secrets) أضف:")
        _log("   NSM_ENABLE_NODE = true")
        _log(f"   TUNNEL_URL = {provider}")
        _log(f"   SEED_NODE_URL = {public_url}")
        print(f"\n*** COPY ***\nSEED_NODE_URL={public_url}\nMODE={prov}\n", flush=True)

    _log(f"📄 سجل العقدة: {node_log}")
    _log(f"⏱️ سيستمر التشغيل حتى {duration_hours:g} ساعة أو حتى إيقاف الدفتر يدوياً.")

    stop = {"flag": False}

    def _handle_sigterm(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    start_time = time.time()
    heartbeat_interval = 300.0  # كل 5 دقائق: تحقق من صحة العمليتين واطبع نبضاً
    last_heartbeat = 0.0
    consecutive_tunnel_failures = 0

    try:
        while not stop["flag"]:
            elapsed = time.time() - start_time
            if elapsed >= duration_seconds:
                _log(f"⏰ بلغ التشغيل الحد الأقصى ({duration_hours:g} ساعة) — إيقاف آمن.")
                break

            if node_proc.poll() is not None:
                _log(f"❌ عملية العقدة توقفت (rc={node_proc.returncode}) — راجع {node_log}")
                break

            if tunnel_proc is not None and tunnel_proc.poll() is not None:
                _log(f"⚠️ نفق {provider} توقف — محاولة استئنافه (أو الانتقال لبديل آخر)...")
                new_provider, new_proc, new_url = start_any_tunnel(
                    port, log_dir, node_id, skip=provider if consecutive_tunnel_failures >= 2 else ""
                )
                if new_url:
                    provider, tunnel_proc, public_url = new_provider, new_proc, new_url
                    consecutive_tunnel_failures = 0
                    url_file.write_text(public_url + "\n", encoding="utf-8")
                    _log(f"🔄 رابط نفق جديد عبر {provider.upper()}: {public_url}")
                    _log(f"   SEED_NODE_URL = {public_url}")
                else:
                    consecutive_tunnel_failures += 1
                    tunnel_proc = None
                    _log("❌ فشلت إعادة تشغيل النفق — سيُعاد المحاولة عند النبض التالي.")

            if elapsed - last_heartbeat >= heartbeat_interval:
                last_heartbeat = elapsed
                remaining_min = max(0.0, (duration_seconds - elapsed) / 60.0)
                _log(f"💓 نبض: العقدة تعمل (نفق: {provider or 'لا يوجد'}) — الوقت المتبقي ≈ {remaining_min:.0f} دقيقة.")
                if tunnel_proc is None:
                    provider, tunnel_proc, public_url = start_any_tunnel(port, log_dir, node_id)
                    if public_url:
                        url_file.write_text(public_url + "\n", encoding="utf-8")
                        _log(f"🔄 استُعيد النفق عبر {provider.upper()}: {public_url}")

            time.sleep(5.0)
    finally:
        _log("🛑 إيقاف العقدة والنفق...")
        for proc in (node_proc, tunnel_proc):
            if proc is not None and proc.poll() is None:
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

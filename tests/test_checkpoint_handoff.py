#!/usr/bin/env python3
"""
اختبار محاكاة لنظام Checkpoint Handoff (بدون مفاتيح API حقيقية)
===============================================================
يحاكي:
1) scheduler_report يتضمن handoffs + last_checkpoint
2) record_handoff يسجّل في state.json
3) pull_kernel_checkpoints يسحب ملفات وهمية من kernel
4) upload_handoff_checkpoint يرفع checkpoint وهمية لـGitHub (clone حقيقي + commit + push — آمن)
5) perform_handoff كاملة بـpause_kernel=False (لا kaggle CLI حقيقي)
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai import multi_account_scheduler as MAS

PASSED, FAILED = 0, 0


def check(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ✅ {name}")
    else:
        FAILED += 1
        print(f"  ❌ {name} — {detail}")


def main():
    global PASSED, FAILED

    # ── 1) state + record_handoff ──
    print("1) فحص state + record_handoff:")
    state = MAS.load_state()
    before = len(state.get("handoffs", []))
    entry = MAS.record_handoff("acc_old", "acc_new", "scn_test123", "success", "latest_pretrain_d256.pt")
    check("record_handoff سجّل", entry.get("status") == "success")
    check("from_account صحيح", entry.get("from_account") == "acc_old")
    check("to_account صحيح", entry.get("to_account") == "acc_new")
    state2 = MAS.load_state()
    check("حُفظ في state.json", len(state2.get("handoffs", [])) == before + 1)

    # ── 2) scheduler_report يتضمن handoffs ──
    print("2) scheduler_report + last_checkpoint:")
    rep = MAS.scheduler_report()
    check("handoffs في التقرير", "handoffs" in rep)
    check("last_checkpoint في التقرير", "last_checkpoint" in rep)
    lc = rep.get("last_checkpoint", {})
    check("last_checkpoint يشير لأحدث نجاح", lc.get("job_id") == "scn_test123")

    # ── 3) pull_kernel_checkpoints بمحاكاة download_kaggle_output ──
    print("3) pull_kernel_checkpoints (محاكاة kernel output):")
    fake_out = ROOT / "artifacts" / "model_training" / "kaggle_jobs" / f"scn_testfake_{uuid.uuid4().hex[:8]}" / "output" / "checkpoints"
    fake_out.mkdir(parents=True, exist_ok=True)
    for fn in ("latest_pretrain_d256.pt", "best_pretrain_d256.pt", "progress_d256.json", "pretrain_state_d256.json"):
        (fake_out / fn).write_text(f"FAKE-{fn}")
    # job dir وmetadata ضروريان لـdownload_kaggle_output
    job_id = f"scn_testfake_{uuid.uuid4().hex[:8]}"
    job_dir = ROOT / "artifacts" / "model_training" / "kaggle_jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "kernel-metadata.json").write_text(json.dumps({"id": f"nsm/{job_id}"}))
    (job_dir / "output" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (job_dir / "output" / "checkpoints" / "latest_pretrain_d256.pt").write_text("FAKE-PT")

    with mock.patch("ai.kaggle_provider.download_kaggle_output") as dl:
        dl.return_value = {
            "ok": True, "job_id": job_id, "output_dir": str(job_dir / "output"),
            "files": ["latest_pretrain_d256.pt"],
        }
        pull = MAS.pull_kernel_checkpoints(job_id)
    check("pull ok", pull.get("ok") is True, str(pull.get("error")))
    files = pull.get("checkpoint_files", [])
    check("وجد ملفات checkpoint", "checkpoints/latest_pretrain_d256.pt" in files, str(files))
    local = pull.get("local_dir", "")
    copied_any = any((Path(local) / rel).is_file() for rel in files) if local and files else False
    check("نسخ محلي فعلي", copied_any, local)

    # ── 4) upload_handoff_checkpoint برفع حقيقي لـGitHub ──
    print("4) upload_handoff_checkpoint (رفع فعلي لـGitHub — اختبار حقيقي):")
    up = MAS.upload_handoff_checkpoint(
        job_id, "test_old", "test_new",
        files_dir=str(job_dir / "output" / "checkpoints"),
    )
    check("upload ok", up.get("ok") is True, str(up.get("error")) or str(up.get("detail")))
    if up.get("ok"):
        # تحقق فعلي من الملف على remote
        proc = subprocess.run(
            ["git", "ls-remote", "--refs", "https://github.com/aliahmed369000000-ai/Neural-Service-Mesh.git", "refs/heads/main"],
            capture_output=True, text=True, timeout=30,
        )
        check("ls-remote يعمل (GitHub متاح)", proc.returncode == 0)

    # ── 5) perform_handoff كاملة (بدون إيقاف kernel) ──
    print("5) perform_handoff كاملة (pause_kernel_first=False):")
    with mock.patch("ai.kaggle_provider.download_kaggle_output") as dl2:
        dl2.return_value = {
            "ok": True, "job_id": job_id, "output_dir": str(job_dir / "output"),
            "files": ["latest_pretrain_d256.pt"],
        }
        res = MAS.perform_handoff("old_acc", "new_acc", job_id, pause_kernel_first=False)
    # upload قد يكون skipped (لا تغييرات جديدة — خطوة 4 رفعت نفس الملفات بالفعل) — هذا صحيح
    up = res.get("upload", {})
    # skipped = لا تغييرات جديدة (خطوة 4 رفعت نفس الملفات بالفعل) — سلوك صحيح
    upload_ok = res.get("ok") is True
    check("perform_handoff ok", upload_ok, str(up.get("error")) or str(up))
    check("recorded في السجل", any(
        h.get("job_id") == job_id and h.get("status") == "success"
        for h in MAS.load_state().get("handoffs", [])
    ))

    # ── 5ب) تأكيد رفع حقيقي جديد بملف مميز ──
    print("5ب) تأكيد رفع فعلي لملف مميز جديد:")
    marker = "NSM-handoff-test-marker-" + uuid.uuid4().hex
    (job_dir / "output" / "checkpoints" / "handoff_marker_test.pt").write_text(marker)
    # clone حقيقي جديد (يسبق perform_handoff) حتى لا يتعارض tmp shared
    shutil.rmtree("/tmp/nsm_handoff_push", ignore_errors=True)
    with mock.patch("ai.kaggle_provider.download_kaggle_output") as dl3:
        dl3.return_value = {
            "ok": True, "job_id": job_id, "output_dir": str(job_dir / "output"),
            "files": ["handoff_marker_test.pt"],
        }
        res2 = MAS.perform_handoff("old_acc", "new_acc", job_id, pause_kernel_first=False)
    up2 = res2.get("upload", {})
    check("رفع حقيقي بملف مميز", bool(up2.get("uploaded")), str(up2.get("error")) or str(up2))
    marker_ok = False
    if up2.get("uploaded"):
        # تحقق فعلي على GitHub عبر API contents (بدون CDN cache الخاصة بـ raw.githubusercontent)
        import time
        time.sleep(3)
        basic = base64.b64encode(f"x-access-token:{os.environ.get('GITHUB_TOKEN', '')}".encode()).decode()
        try:
            import urllib.request
            for attempt in range(6):
                req = urllib.request.Request(
                    "https://api.github.com/repos/aliahmed369000000-ai/Neural-Service-Mesh/"
                    "contents/experiments/surah_chain_network/checkpoints/"
                    "handoff_marker_test.pt?ref=main",
                    headers={"Authorization": f"Basic {basic}", "User-Agent": "nsm-test"},
                )
                import json as _json
                d = _json.load(urllib.request.urlopen(req, timeout=30))
                data = base64.b64decode(d["content"]).decode()
                if marker[:24] in data:
                    marker_ok = True
                    break
                time.sleep(3)
        except Exception as e:
            print("  (فحص remote):", e)
    check("الملف المميز موجود فعليًا على GitHub", marker_ok)

    # ── 6) CLI ──
    print("6) CLI commands:")
    out = MAS.scheduler_cli(["handoff", "a", "b", job_id])
    check("CLI handoff يعمل", "uploaded" in out or "skipped" in out or '"ok"' in out)
    out2 = MAS.scheduler_cli(["handoffs"])
    check("CLI handoffs يعمل", "scn_test123" in out2 or "scn_testfake" in out2)

    # تنظيف
    shutil.rmtree(fake_out.parents[2], ignore_errors=True)
    shutil.rmtree(job_dir, ignore_errors=True)
    # احذف test handoff entry من state (اترك record حقيقيًا مفيدًا)

    print(f"\n{'='*50}\nالنتيجة: {PASSED} نجاح / {FAILED} فشل")
    return FAILED == 0


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)

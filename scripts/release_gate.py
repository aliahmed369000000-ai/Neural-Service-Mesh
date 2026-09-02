#!/usr/bin/env python3
"""
NSM Production Release Gate
===========================
يشغّل حزمة اختبارات الاستقرار للإصدار الإنتاجي ويخرج PASS/FAIL.
لا يضيف ميزات — يجمّد الثقة بما هو مدعوم في RELEASE_NOTES.md
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(REPO))

# اختبارات وحدة — بدون شبكة حقيقية
UNIT_TESTS = [
    "tests/test_verifiable_cognitive_net.py",
    "tests/test_collective_cognitive_ledger.py",
    "tests/test_byzantine_decision_guard.py",
    "tests/test_leader_election.py",
    "tests/test_private_federated_learning.py",
    "tests/test_identity_select_failover.py",
    "tests/test_mesh_task_protocol.py",
    "tests/test_nsm_node_v2_slice.py",
    "tests/test_mesh_storage_protocol.py",
    "tests/test_mesh_health_relay_multisig.py",
]

# سكربتات إثبات (قد تحتاج aiohttp)
PROOF_SCRIPTS = [
    ("prove_federation", [sys.executable, "scripts/prove_federation.py"]),
]

# إثبات انضمام حي اختياري إذا aiohttp متوفر
LIVE_JOIN = ("live_join_worker_task", [sys.executable, "scripts/run_live_join_demo.py"])


def run_cmd(cmd: list, timeout: int = 180) -> dict:
    t0 = time.time()
    try:
        p = subprocess.run(
            cmd,
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "ok": p.returncode == 0,
            "returncode": p.returncode,
            "elapsed_s": round(time.time() - t0, 2),
            "stdout_tail": (p.stdout or "")[-1500:],
            "stderr_tail": (p.stderr or "")[-800:],
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "returncode": -1, "elapsed_s": timeout, "error": "timeout"}
    except Exception as e:
        return {"ok": False, "returncode": -2, "error": str(e)}


def has_aiohttp() -> bool:
    try:
        import aiohttp  # noqa: F401
        return True
    except ImportError:
        return False


def compile_check(paths: list) -> dict:
    failed = []
    for rel in paths:
        p = REPO / rel
        if not p.exists():
            failed.append(f"missing:{rel}")
            continue
        r = subprocess.run(
            [sys.executable, "-m", "py_compile", str(p)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            failed.append(rel)
    return {"ok": len(failed) == 0, "failed": failed}


def main():
    print("=" * 64)
    print("🔒 NSM Production Release Gate")
    print("=" * 64)
    results = []
    t_all = time.time()

    # 0) py_compile للوحدات الحرجة
    critical = [
        "ai/living_mesh.py",
        "ai/node_launcher.py",
        "ai/node_health_layer.py",
        "ai/verifiable_cognitive_net.py",
        "ai/collective_cognitive_ledger.py",
        "ai/leader_election.py",
        "ai/byzantine_decision_guard.py",
        "ai/private_federated_learning.py",
        "ai/mesh_task_protocol.py",
    ]
    print("\n[0] py_compile critical modules")
    c0 = compile_check(critical)
    results.append({"name": "py_compile_critical", **c0})
    print("  ", "✅" if c0["ok"] else "❌", c0)

    # 1) unit tests
    print("\n[1] unit tests")
    for rel in UNIT_TESTS:
        path = REPO / rel
        if not path.exists():
            results.append({"name": rel, "ok": False, "error": "missing"})
            print(f"  ❌ {rel} missing")
            continue
        r = run_cmd([sys.executable, rel], timeout=120)
        results.append({"name": rel, **r})
        print(f"  {'✅' if r['ok'] else '❌'} {rel} ({r.get('elapsed_s')}s)")
        if not r["ok"] and r.get("stderr_tail"):
            print("     stderr:", r["stderr_tail"][:200].replace("\n", " "))

    # 2) federation proof
    print("\n[2] federation proof")
    for name, cmd in PROOF_SCRIPTS:
        r = run_cmd(cmd, timeout=180)
        results.append({"name": name, **r})
        print(f"  {'✅' if r['ok'] else '❌'} {name} ({r.get('elapsed_s')}s)")

    # 3) live join (optional)
    print("\n[3] live join worker task")
    if has_aiohttp():
        r = run_cmd(LIVE_JOIN[1], timeout=120)
        results.append({"name": LIVE_JOIN[0], **r, "optional": False})
        print(f"  {'✅' if r['ok'] else '❌'} {LIVE_JOIN[0]} ({r.get('elapsed_s')}s)")
    else:
        results.append({
            "name": LIVE_JOIN[0],
            "ok": True,
            "skipped": True,
            "reason": "aiohttp_not_installed",
        })
        print("  ⚠️ skipped (aiohttp not installed) — لا يفشل البوابة")

    # خلاصة
    # الاختبارات الإلزامية: كل شيء ما عدا skipped
    required = [x for x in results if not x.get("skipped")]
    passed = sum(1 for x in required if x.get("ok"))
    failed = [x["name"] for x in required if not x.get("ok")]
    gate_ok = len(failed) == 0

    summary = {
        "gate": "PASS" if gate_ok else "FAIL",
        "passed": passed,
        "total_required": len(required),
        "failed": failed,
        "elapsed_s": round(time.time() - t_all, 2),
        "protocol": "nsm-join-v1",
        "results": [
            {
                "name": x["name"],
                "ok": x.get("ok"),
                "skipped": x.get("skipped"),
                "elapsed_s": x.get("elapsed_s"),
                "error": x.get("error"),
            }
            for x in results
        ],
    }

    out_dir = REPO / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "release_gate_report.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n" + "=" * 64)
    print(f"GATE: {summary['gate']}  ({passed}/{len(required)})  {summary['elapsed_s']}s")
    if failed:
        print("FAILED:", ", ".join(failed))
    print(f"📁 {out_path}")
    print("=" * 64)
    return 0 if gate_ok else 1


if __name__ == "__main__":
    sys.exit(main())

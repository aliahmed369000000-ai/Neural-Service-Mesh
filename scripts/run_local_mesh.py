#!/usr/bin/env python3
"""
تشغيل مجموعة محلية محافظة من عقد Living Mesh.
NSM_NODE_COUNT=1 → بذرة فقط
NSM_NODE_COUNT=3 → بذرة + عاملين
لا يشغّل عشرات العقد افتراضياً.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED_PORT = int(os.getenv("PORT", "7860"))
COUNT = max(1, min(int(os.getenv("NSM_NODE_COUNT", "1")), 10))  # سقف آمن 10


def main() -> int:
    procs: list[subprocess.Popen] = []
    env_base = os.environ.copy()
    seed_data = ROOT / "artifacts" / "living_mesh" / "nodes" / "mesh_seed"
    seed_data.mkdir(parents=True, exist_ok=True)

    seed_env = env_base.copy()
    seed_env["NODE_ID"] = "mesh_seed"
    seed_env["PORT"] = str(SEED_PORT)
    seed_env["NSM_NODE_DATA_DIR"] = str(seed_data)
    print(f"🌱 starting mesh_seed on :{SEED_PORT} data={seed_data}")
    procs.append(subprocess.Popen(
        [sys.executable, str(ROOT / "ai" / "node_launcher.py"),
         "--id", "mesh_seed", "--host", "0.0.0.0", "--port", str(SEED_PORT),
         "--data-dir", str(seed_data)],
        cwd=str(ROOT), env=seed_env,
    ))
    time.sleep(2.0)

    for i in range(1, COUNT):
        wid = f"worker_{i}"
        port = SEED_PORT + i
        wdata = ROOT / "artifacts" / "living_mesh" / "nodes" / wid
        wdata.mkdir(parents=True, exist_ok=True)
        wenv = env_base.copy()
        wenv["NODE_ID"] = wid
        wenv["PORT"] = str(port)
        wenv["SEED_NODE_URL"] = f"127.0.0.1:{SEED_PORT}"
        wenv["NSM_NODE_DATA_DIR"] = str(wdata)
        print(f"👷 starting {wid} on :{port} seed=127.0.0.1:{SEED_PORT}")
        procs.append(subprocess.Popen(
            [sys.executable, str(ROOT / "ai" / "node_launcher.py"),
             "--id", wid, "--host", "0.0.0.0", "--port", str(port),
             "--data-dir", str(wdata)],
            cwd=str(ROOT), env=wenv,
        ))
        time.sleep(0.8)

    def _stop(signum, frame):
        print("🛑 stopping local mesh...")
        for p in procs:
            try:
                p.send_signal(signal.SIGTERM)
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    print(f"✅ local mesh running: {COUNT} process(es). Ctrl+C to stop.")
    while True:
        alive = [p for p in procs if p.poll() is None]
        if not alive:
            print("all processes exited")
            return 1
        time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())

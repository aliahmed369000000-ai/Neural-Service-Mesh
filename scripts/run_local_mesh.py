#!/usr/bin/env python3
"""
تشغيل مجموعة محلية محافظة من عقد Living Mesh.
NSM_NODE_COUNT=1 → بذرة فقط
NSM_NODE_COUNT=3 → بذرة + عاملين
لا يشغّل عشرات العقد افتراضياً.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED_PORT = int(os.getenv("PORT", "7860"))
MANIFEST = ROOT / "config" / "mesh_nodes_28.json"
with MANIFEST.open(encoding="utf-8") as handle:
    NODE_MANIFEST = json.load(handle)
MAX_NODES = int(NODE_MANIFEST["node_count"])
COUNT = max(1, min(int(os.getenv("NSM_NODE_COUNT", "1")), MAX_NODES))


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from ai.mesh_supervisor import NodeSpec, Supervisor, http_probe

    env_base = os.environ.copy()
    specs: list[NodeSpec] = []

    seed_data = ROOT / "artifacts" / "living_mesh" / "nodes" / "mesh_seed"
    seed_data.mkdir(parents=True, exist_ok=True)
    seed_env = env_base.copy()
    seed_env["NODE_ID"] = "mesh_seed"
    seed_env["PORT"] = str(SEED_PORT)
    seed_env["NSM_NODE_DATA_DIR"] = str(seed_data)
    specs.append(NodeSpec(
        "mesh_seed",
        [sys.executable, str(ROOT / "ai" / "node_launcher.py"),
         "--id", "mesh_seed", "--host", "0.0.0.0", "--port", str(SEED_PORT),
         "--data-dir", str(seed_data)],
        seed_env, str(ROOT), SEED_PORT,
    ))

    for worker in NODE_MANIFEST["workers"][: COUNT - 1]:
        wid = worker["id"]
        port = SEED_PORT + int(worker["port_offset"])
        wdata = ROOT / "artifacts" / "living_mesh" / "nodes" / wid
        wdata.mkdir(parents=True, exist_ok=True)
        wenv = env_base.copy()
        wenv["NODE_ID"] = wid
        wenv["PORT"] = str(port)
        wenv["SEED_NODE_URL"] = f"127.0.0.1:{SEED_PORT}"
        wenv["NSM_NODE_DATA_DIR"] = str(wdata)
        specs.append(NodeSpec(
            wid,
            [sys.executable, str(ROOT / "ai" / "node_launcher.py"),
             "--id", wid, "--host", "0.0.0.0", "--port", str(port),
             "--data-dir", str(wdata)],
            wenv, str(ROOT), port,
        ))

    # الإشراف مفعّل افتراضياً (إعادة تشغيل ما يتوقف/يتجمّد)؛ NSM_SUPERVISE=0 للسلوك القديم.
    supervise = os.getenv("NSM_SUPERVISE", "1").strip().lower() not in ("0", "false", "no", "off")
    sup = Supervisor(specs, probe=http_probe if supervise else None)
    for st in sup.states:
        print(f"🌱 starting {st.spec.name} on :{st.spec.port}")
    sup.start_all(stagger=0.8)

    def _stop(signum, frame):
        print("🛑 stopping local mesh...")
        sup.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    mode = "supervised (auto-restart)" if supervise else "unsupervised"
    print(f"✅ local mesh running: {COUNT} process(es) from {MANIFEST.name} [{mode}]. Ctrl+C to stop.")
    while True:
        if supervise:
            sup.tick()
        elif sup.alive_count() == 0:
            print("all processes exited")
            return 1
        time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())

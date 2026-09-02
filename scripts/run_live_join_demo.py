#!/usr/bin/env python3
"""بذرة حية على منفذ محلي + مسار انضمام خارجي عبر --seed مع تقرير."""
from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
SEED_PORT = int(__import__("os").environ.get("NSM_SEED_PORT", "19876"))


async def main():
    from aiohttp import web
    import ai.living_mesh as lm
    from ai.living_mesh import LivingMeshNode
    from ai.node_health_layer import NodeHealthLayer
    from ai import node_launcher as nl

    tmp = Path(tempfile.mkdtemp(prefix="nsm_seed_live_"))
    lm.LIVING_MESH_DIR = tmp
    lm.NETWORK_STATE = tmp / "network_state.json"
    lm.CONTENT_DIR = tmp / "content"
    lm.CONTENT_DIR.mkdir(exist_ok=True)
    seed = LivingMeshNode(node_id="seed_live", host="127.0.0.1", port=SEED_PORT)
    health = NodeHealthLayer(seed)
    app = web.Application()
    app["node"] = seed
    app["health"] = health
    for method, path, handler in [
        ("GET", "/health", nl.handle_health),
        ("GET", "/v2/join-info", nl.handle_join_info),
        ("POST", "/v2/join", nl.handle_join),
        ("POST", "/v2/first-task", nl.handle_first_verified_task),
        ("POST", "/v2/task", nl.handle_submit_task),
        ("GET", "/v2/tasks", nl.handle_tasks),
    ]:
        if method == "GET":
            app.router.add_get(path, handler)
        else:
            app.router.add_post(path, handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", SEED_PORT).start()
    base = f"http://127.0.0.1:{SEED_PORT}"
    for _ in range(50):
        try:
            urllib.request.urlopen(f"{base}/health", timeout=1)
            break
        except Exception:
            await asyncio.sleep(0.1)
    print(f"SEED {base}")
    cmd = [
        sys.executable, str(REPO / "scripts/join_external_path.py"),
        "--seed", base, "--node-id", "external_live_1",
        "--host", "127.0.0.1", "--port", "19901",
    ]
    loop = asyncio.get_event_loop()
    rc = await loop.run_in_executor(None, lambda: subprocess.call(cmd, cwd=str(REPO)))
    await runner.cleanup()
    return rc


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

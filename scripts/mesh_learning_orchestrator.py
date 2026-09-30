#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""منسّق تعلّم موزَّع عبر Living Mesh — يوزّع موضوعاً مختلفاً على كل عقدة حية.

لماذا هذا السكربت؟
-------------------
كانت كل الأدوات موجودة (بحث/جلب ويب، تدريب جزئي، تعلّم ذاتي عبر
ai/self_feed_learner.py) لكن بلا طرف خارجي يقرر "أي عقدة تبحث عن أي موضوع".
هذا السكربت هو تلك الطبقة الناقصة: يأخذ قائمة عقد حية + قائمة مواضيع،
ويوزّع موضوعاً مختلفاً لكل عقدة عبر مهمة `self_feed_learn` الجديدة
(ai/mesh_task_protocol.py) — بروتوكول RPC موقّع حقيقي (LivingMeshNode.
request_from_peer)، لا HTTP خام. كل ابتلاع ناجح يغذّي ai/unified_semantic_memory.py
تلقائياً، فيفيد أي سؤال لاحق بمحادثة أي عقدة بالشبكة (راجع commit ربط
self_feed_learner).

الاستخدام:
    # عقد محلية معرَّفة بـconfig/mesh_nodes_28.json (127.0.0.1 + port_offset):
    python scripts/mesh_learning_orchestrator.py --local --base-port 19900

    # عقد حية بعناوين صريحة (bore.pub/serveo/localhost.run/إلخ):
    python scripts/mesh_learning_orchestrator.py --nodes-file live_nodes.json

    # اكتشاف تلقائي لأقران عقدة بذرة حية عبر /v2/routes:
    python scripts/mesh_learning_orchestrator.py --discover-from http://SEED_HOST:PORT

نسق ملف --nodes-file (JSON):
    [{"id": "worker_1", "host": "bore.pub", "port": 41234}, ...]

نسق ملف --topics-file (JSON، سطر لكل موضوع أو قائمة نصوص):
    ["نظرية الأعداد الأولية", "معلقة امرئ القيس", "دورة الماء في الطبيعة"]

بلا --topics-file: يُستخدَم DEFAULT_TOPICS أدناه (توسعة لمواضيع classic_showcase).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.living_mesh import LivingMeshNode  # noqa: E402
from ai import mesh_task_protocol as mt  # noqa: E402

DEFAULT_TOPICS = [
    "نظرية الأعداد الأولية",
    "متتالية فيبوناتشي",
    "عدد π ومتسلسلة ليبنيز",
    "معلقة امرئ القيس",
    "دورة الماء في الطبيعة",
    "الخلية النباتية والحيوانية",
    "الطاقة الشمسية وتحويلها كهرباء",
    "تاريخ الخط العربي",
    "نظرية فيثاغورس",
    "المد والجزر",
]


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_local_roster(base_port: int, config_path: Path) -> List[Dict[str, Any]]:
    """يبني قائمة عقد محلية من config/mesh_nodes_28.json (127.0.0.1 + port_offset)."""
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    roster = []
    seed = cfg.get("seed") or {}
    roster.append({
        "id": seed.get("id", "mesh_seed"),
        "host": "127.0.0.1",
        "port": base_port + int(seed.get("port_offset", 0)),
    })
    for w in cfg.get("workers") or []:
        roster.append({
            "id": w.get("id"),
            "host": "127.0.0.1",
            "port": base_port + int(w.get("port_offset", 0)),
        })
    return roster


async def discover_roster(seed_host: str, seed_port: int, timeout: float = 10.0) -> List[Dict[str, Any]]:
    """يكتشف الأقران المعروفين لدى عقدة بذرة حية عبر /v2/routes (HTTP)."""
    import aiohttp
    url = f"http://{seed_host}:{seed_port}/v2/routes"
    roster = [{"id": "seed", "host": seed_host, "port": seed_port}]
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout) as resp:
                data = await resp.json()
        for row in (data.get("rows") if isinstance(data, dict) else data) or []:
            h, p = row.get("host"), row.get("port")
            if h and p:
                roster.append({"id": row.get("peer_id") or f"{h}:{p}", "host": h, "port": int(p)})
    except Exception as e:
        _log(f"⚠️ فشل اكتشاف الأقران عبر {url}: {e} — سيُستخدَم البذرة فقط.")
    return roster


async def dispatch_one(
    client: LivingMeshNode, node: Dict[str, Any], topic: str, deep: bool, timeout: float,
) -> Dict[str, Any]:
    t0 = time.time()
    try:
        res = await client.request_from_peer(
            node["host"], int(node["port"]), mt.KIND_SELF_FEED_LEARN,
            {"topic": topic, "deep": deep, "max_results": 6},
            timeout=timeout,
        )
    except Exception as e:
        res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    result = res.get("result") or {}
    return {
        "node_id": node.get("id"),
        "host": node["host"],
        "port": node["port"],
        "topic": topic,
        "acked": bool(res.get("acked")),
        "ok": bool(res.get("ok") and result.get("ok")),
        "error": res.get("error") or result.get("error"),
        "elapsed_ms": round((time.time() - t0) * 1000, 2),
        "ingest_mode": result.get("mode"),
    }


async def run(roster: List[Dict[str, Any]], topics: List[str], deep: bool, timeout: float) -> Dict[str, Any]:
    if not roster:
        return {"ok": False, "error": "empty_roster"}
    if not topics:
        topics = DEFAULT_TOPICS

    client = LivingMeshNode(
        node_id="mesh_learning_orchestrator",
        host="127.0.0.1",
        port=0,
        data_dir=str(ROOT / "artifacts" / "living_mesh" / "nodes" / "mesh_learning_orchestrator"),
    )

    _log(f"🎯 توزيع {len(roster)} عقدة × مواضيع ({len(topics)} متاحة، دوّار) — deep={deep}")
    tasks = []
    for i, node in enumerate(roster):
        topic = topics[i % len(topics)]
        tasks.append(dispatch_one(client, node, topic, deep, timeout))
    results = await asyncio.gather(*tasks)

    oks = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    for r in results:
        mark = "✅" if r["ok"] else "❌"
        _log(f"  {mark} {r['node_id']} ({r['host']}:{r['port']}) ← \"{r['topic']}\" "
             f"[{r['elapsed_ms']}ms]" + (f" — {r['error']}" if r["error"] else ""))

    summary = {
        "ok": True,
        "total": len(results),
        "succeeded": len(oks),
        "failed": len(failed),
        "results": results,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", action="store_true", help="استخدم config/mesh_nodes_28.json محلياً (127.0.0.1)")
    ap.add_argument("--base-port", type=int, default=19900, help="منفذ القاعدة للنشر المحلي (مع --local)")
    ap.add_argument("--config", type=str, default=str(ROOT / "config" / "mesh_nodes_28.json"))
    ap.add_argument("--nodes-file", type=str, default=None, help="ملف JSON بقائمة عقد صريحة")
    ap.add_argument("--discover-from", type=str, default=None, help="http://host:port لعقدة بذرة حية لاكتشاف أقرانها")
    ap.add_argument("--topics-file", type=str, default=None, help="ملف JSON بقائمة مواضيع")
    ap.add_argument("--deep", action="store_true", help="بحث عميق (deep_research) بدل بحث سريع")
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--out", type=str, default=str(ROOT / "artifacts" / "mesh_learning_orchestrator_report.json"))
    args = ap.parse_args()

    roster: List[Dict[str, Any]] = []
    if args.nodes_file:
        roster = json.loads(Path(args.nodes_file).read_text(encoding="utf-8"))
    elif args.discover_from:
        u = args.discover_from.replace("http://", "").replace("https://", "").rstrip("/")
        host, _, port_s = u.partition(":")
        roster = asyncio.run(discover_roster(host, int(port_s or "80")))
    elif args.local:
        roster = load_local_roster(args.base_port, Path(args.config))
    else:
        _log("❌ حدّد أحد: --local أو --nodes-file أو --discover-from")
        sys.exit(2)

    topics = None
    if args.topics_file:
        topics = json.loads(Path(args.topics_file).read_text(encoding="utf-8"))

    summary = asyncio.run(run(roster, topics, args.deep, args.timeout))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"📄 تقرير كامل: {out_path}")
    _log(f"✅ {summary['succeeded']}/{summary['total']} عقدة نجحت")
    if summary["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

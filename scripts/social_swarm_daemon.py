#!/usr/bin/env python3
"""
Social Swarm Daemon — تشغيل دوري للمنظومة الاجتماعية
لا ينشر فعلياً بدون اعتمادات المنصات؛ يجهّز منشورات + نبضة حساسات.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "artifacts" / "model_training" / "social_swarm"
OUT.mkdir(parents=True, exist_ok=True)


def once(topic: str | None = None) -> dict:
    from ai.social_swarm import run_social_swarm
    from ai.sensors_training_bridge import bridge_cycle
    from world_model.predictive_sim import full_campaign_sim

    swarm = run_social_swarm(topic=topic)
    sample = next(iter((swarm.get("safe_posts") or swarm.get("posts") or {}).values()), "")
    sim = full_campaign_sim(sample or "معرفة نافعة") if sample else {}
    bridge = bridge_cycle()
    report = {
        "swarm_topic": swarm.get("topic"),
        "safe_posts": len(swarm.get("safe_posts") or {}),
        "sim_go": sim.get("go") if isinstance(sim, dict) else None,
        "bridge_weak": len(bridge.get("weak_for_training") or []),
        "publish_note_ar": (
            "النشر الفعلي يتطلب مفاتيح المنصات في البيئة؛ "
            "daemon يجهّز المحتوى ويمرره عبر درع الأزمة والمحاكاة."
        ),
    }
    (OUT / "daemon_last.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--interval", type=int, default=3600)
    ap.add_argument("--topic", type=str, default="")
    args = ap.parse_args()
    if args.loop:
        while True:
            r = once(args.topic or None)
            print(json.dumps(r, ensure_ascii=False))
            time.sleep(max(60, args.interval))
    else:
        print(json.dumps(once(args.topic or None), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

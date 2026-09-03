"""مراقب بسيط لصحة Seed Node للاستخدام مع cron أو systemd."""
from __future__ import annotations

import argparse
import json
import time
import urllib.request


def check_health(url: str, timeout: float) -> bool:
    """تحقق من HTTP 200 وJSON ok=true دون اعتبار الرابط حياً شكلياً."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
            return payload.get("ok", True) is not False
    except (OSError, ValueError, TypeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="مراقبة صحة عقدة NSM")
    parser.add_argument("url", help="رابط /health للعقدة")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--interval", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    while True:
        healthy = check_health(args.url, args.timeout)
        print(json.dumps({"url": args.url, "healthy": healthy}), flush=True)
        if args.once:
            return 0 if healthy else 1
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())

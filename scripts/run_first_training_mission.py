#!/usr/bin/env python3
"""إطلاق مهام تدريبية معزولة لوكيل NSM Model Training Agent."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="NSM sandboxed training mission")
    parser.add_argument("--dry-run", action="store_true", help="فحص الحواجز دون تدريب")
    parser.add_argument("--status", action="store_true", help="عرض حالة sandbox فقط")
    parser.add_argument(
        "--mission",
        default="1",
        help="1/first أو 2/second أو first_mission/second_mission",
    )
    args = parser.parse_args()

    from ai.training_sandbox import (
        list_mission_logs,
        run_mission,
        sandbox_status_report,
    )

    if args.status:
        print(sandbox_status_report())
        print()
        print(list_mission_logs())
        return 0

    print(sandbox_status_report())
    print("\n" + "=" * 60 + "\n")
    key = args.mission.strip().lower()
    if key in ("1", "first", "first_mission", "الأولى"):
        key = "first_mission"
    elif key in ("2", "second", "second_mission", "الثانية"):
        key = "second_mission"
    print(run_mission(key, dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

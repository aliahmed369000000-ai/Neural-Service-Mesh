#!/usr/bin/env python3
"""إطلاق أول مهمة تدريبية معزولة لوكيل NSM Model Training Agent."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="NSM first training mission (sandboxed)")
    parser.add_argument("--dry-run", action="store_true", help="فحص الحواجز دون تدريب")
    parser.add_argument("--status", action="store_true", help="عرض حالة sandbox فقط")
    args = parser.parse_args()

    from ai.training_sandbox import list_mission_logs, run_first_mission, sandbox_status_report

    if args.status:
        print(sandbox_status_report())
        print()
        print(list_mission_logs())
        return 0

    print(sandbox_status_report())
    print("\n" + "=" * 60 + "\n")
    print(run_first_mission(dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

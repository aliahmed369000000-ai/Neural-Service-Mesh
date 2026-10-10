#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI للجسر اليدوي: list | answer <id> <نص> (أو النص من stdin بـ -)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ai import claude_bridge as cb  # noqa: E402


def main(argv):
    if len(argv) < 2 or argv[1] == "list":
        pend = cb.list_pending()
        print(f"pending: {len(pend)}  dir: {cb.bridge_dir()}")
        for d in pend:
            print(f"- {d['id']} [{d.get('node_id','')}] {d['prompt']!r}")
        return 0
    if argv[1] == "answer" and len(argv) >= 3:
        text = sys.stdin.read() if (len(argv) == 3 or argv[3] == "-") else " ".join(argv[3:])
        ok = cb.answer(argv[2], text)
        print("answered" if ok else "failed (معرّف غير صالح/غير معلّق/نص فارغ)")
        return 0 if ok else 1
    print("usage: claude_bridge.py list | answer <id> <text|->")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))

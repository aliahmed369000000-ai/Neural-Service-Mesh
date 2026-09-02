# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from ai import mesh_task_protocol as mt

def test_blocks_localhost():
    r = mt.execute_web_fetch({"url": "https://localhost/admin"})
    assert r["ok"] is False
    print("✅ block localhost", r.get("error"))

def test_blocks_http():
    r = mt.execute_web_fetch({"url": "http://example.com/"})
    assert r["ok"] is False and r.get("error") == "https_only"
    print("✅ https_only")

def test_blocks_private_literal():
    r = mt.execute_web_fetch({"url": "https://127.0.0.1/"})
    assert r["ok"] is False
    print("✅ block 127.0.0.1", r.get("error"))

def test_public_example():
    r = mt.execute_web_fetch({"url": "https://example.com/", "max_chars": 1500, "timeout": 15})
    # الشبكة قد تُحجب في بعض البيئات
    if r.get("ok"):
        assert r.get("content_hash") and r.get("text")
        print("✅ fetch example.com chars", r.get("chars"))
    else:
        print("⚠️ skip live fetch", r.get("error"))

if __name__ == "__main__":
    test_blocks_localhost()
    test_blocks_http()
    test_blocks_private_literal()
    test_public_example()
    print("🎉 web_fetch tests done")

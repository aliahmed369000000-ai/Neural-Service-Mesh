"""تحصينان في ai/living_mesh.py (بلا شبكة ولا مفاتيح):

1. مهام الخلفية (غوسيب/تنفيذ تطوّر) كانت asyncio.create_task بلا مرجع: تُجمَّع
   بالـGC وتضيع أخطاؤها. _spawn_bg تمسك مرجعاً قوياً حتى الانتهاء وتسجّل الفشل.
2. ملفات JSON (حالة الشبكة/الهوية/الميتا) تُكتب UTF-8 لكنها كانت تُقرأ بترميز
   النظام الافتراضي — على نظام غير UTF-8 يُعدّ ملف عربي سليم «تالفاً» فيُنسخ
   ويُصفَّر. القراءة/الكتابة الآن UTF-8 صريحة.
"""
import asyncio
import gc
import logging
from pathlib import Path

import pytest

import ai.living_mesh as lm
from ai.living_mesh import LivingMeshNode, _spawn_bg


def test_spawn_bg_holds_reference_until_done_and_cleans_up():
    async def main():
        gate = asyncio.Event()
        done = []

        async def work():
            await gate.wait()
            done.append(1)

        t = _spawn_bg(work(), "w")
        assert t in lm._BG_TASKS
        del t
        gc.collect()
        gate.set()
        await asyncio.sleep(0.05)
        assert done == [1]
        assert not lm._BG_TASKS

    asyncio.run(main())


def test_spawn_bg_logs_failure(caplog):
    async def main():
        async def boom():
            raise RuntimeError("boom-xyz")

        _spawn_bg(boom(), "boomer")
        await asyncio.sleep(0.05)

    with caplog.at_level(logging.ERROR, logger="LivingMesh"):
        asyncio.run(main())
    assert any("boomer" in r.getMessage() and "boom-xyz" in r.getMessage() for r in caplog.records)
    assert not lm._BG_TASKS


def test_state_and_identity_read_write_use_explicit_utf8(tmp_path, monkeypatch):
    node = LivingMeshNode(node_id="A", host="127.0.0.1", port=9001, data_dir=tmp_path)
    seen = []
    real_read = Path.read_text
    real_write = Path.write_text

    def spy_read(self, *a, **kw):
        seen.append(kw.get("encoding"))
        return real_read(self, *a, **kw)

    def spy_write(self, *a, **kw):
        seen.append(kw.get("encoding"))
        return real_write(self, *a, **kw)

    state = {"nodes": {"ع": {"id": "ع", "note": "السلام عليكم"}}, "global_experience": []}
    node._save_state(state)
    monkeypatch.setattr(Path, "read_text", spy_read)
    monkeypatch.setattr(Path, "write_text", spy_write)
    assert node._load_state()["nodes"]["ع"]["note"] == "السلام عليكم"
    node.identity_info()
    assert seen and all(e == "utf-8" for e in seen), seen

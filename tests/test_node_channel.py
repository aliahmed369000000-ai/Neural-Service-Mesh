"""
اختبار: core/node_channel.py (قناة التواصل بين العُقد) + storage/file_storage.py.

يغطي التطوير الجديد:
1. NodeChannel.send() يرفض ValueError عند from_id/to_id فارغين — بدل حفظ
   رسالة صامتة في صندوق بريد بمفتاح فارغ.
2. reply_to يُحفَظ ويُقرأ فعلياً في الرسالة (ربط طلب/استجابة حقيقي).
3. تزامن الخيوط: عشرات النداءات المتزامنة لـsend() من خيوط متعددة على
   نفس القناة لا تفقد أي رسالة (قبل القفل، هذا كان عرضة لتسابق
   قراءة-تعديل-كتابة على self._inboxes/self._log).
4. FileStorage.save() ذرّية: لا يُترَك ملف .tmp متبقٍ بعد حفظ ناجح،
   والمحتوى المقروء لاحقاً صحيح ومطابق تماماً.
"""
from __future__ import annotations

import shutil
import tempfile
import threading

import pytest

from core.node_channel import NodeChannel
from storage.file_storage import FileStorage


@pytest.fixture()
def channel():
    tmp_dir = tempfile.mkdtemp()
    try:
        yield NodeChannel(FileStorage(storage_dir=tmp_dir))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_send_rejects_empty_ids(channel):
    with pytest.raises(ValueError):
        channel.send(from_id="", to_id="node-b", topic="t")
    with pytest.raises(ValueError):
        channel.send(from_id="node-a", to_id="", topic="t")


def test_reply_to_round_trips(channel):
    original = channel.send(from_id="a", to_id="b", topic="request", payload={"q": 1})
    reply = channel.send(
        from_id="b", to_id="a", topic="response",
        payload={"ok": True}, reply_to=original["message_id"],
    )
    assert reply["reply_to"] == original["message_id"]
    stored = channel.inbox("a")[-1]
    assert stored["reply_to"] == original["message_id"]


def test_broadcast_skips_invalid_but_sends_valid(channel):
    sent = channel.broadcast(from_id="root", to_ids=["", "root", "peer1", "peer2"], topic="hi")
    assert {m["to_id"] for m in sent} == {"peer1", "peer2"}


def test_concurrent_sends_do_not_lose_messages(channel):
    n_threads = 20
    per_thread = 25

    def worker(i):
        for j in range(per_thread):
            channel.send(from_id=f"sender-{i}", to_id="hub", topic="ping", payload={"i": i, "j": j})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert channel.stats()["total_messages"] == n_threads * per_thread


def test_file_storage_atomic_save_no_leftover_tmp():
    tmp_dir = tempfile.mkdtemp()
    try:
        storage = FileStorage(storage_dir=tmp_dir)
        for i in range(10):
            assert storage.save("thing.json", {"n": i}) is True
        loaded = storage.load("thing.json")
        assert loaded == {"n": 9}
        leftovers = [p for p in __import__("pathlib").Path(tmp_dir).iterdir() if p.name.endswith(".tmp")]
        assert leftovers == []
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

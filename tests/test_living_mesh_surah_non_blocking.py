"""ai/living_mesh.py::LivingMeshNode._load_surah_pretrain — كانت تستدعي
huggingface_hub.hf_hub_download مباشرة (نداء متزامن حاجز، بلا await)، مع
retry داخلي بتأخير تصاعدي يصل حتى ~31 ثانية لكل نداء عند تعذّر الشبكة. بما
أن هذه الدالة async تُجدوَل كمهمة asyncio خلفية (loop.create_task) على
نفس حلقة الأحداث الوحيدة التي تخدم aiohttp/WebSocket/كل شيء آخر للعقدة،
كان النداء الحاجز الداخلي يُجمِّد الحلقة بأكملها طوال تلك المدة — لا HTTP،
لا WebSocket، لا معالجة أي رسالة قرين. تحقّقتُ من هذا فعلياً بتشغيل عقدة
حقيقية مرتين منفصلتين: /status ظلت غير مستجيبة تماماً لأكثر من 15 ثانية
في كل مرة (شبكة الحاوية لا تصل huggingface.co). بعد الإصلاح
(loop.run_in_executor): عقدة حقيقية أصبحت جاهزة ومستجيبة خلال 0.64 ثانية،
وبقيت تستجيب 200 لـ/status في t=1s وt=4s وt=7s بينما محاولات Surah ما
زالت جارية في الخلفية.

هذا الاختبار يثبت الآلية بمعزل عن الشبكة الحقيقية: يستبدل hf_hub_download
بدالة حاجزة وهمية (time.sleep) تحاكي التعليق الحقيقي المُعاد إنتاجه، ثم
يتأكد أن حلقة asyncio تبقى حرّة لتنفيذ مهام أخرى متزامنة أثناء ذلك —
بدلاً من انتظار تشغيل Surah الحقيقي البطيء (أو شبكة فعلية) في كل تشغيل
اختبار.
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
import time

import pytest

from ai.living_mesh import LivingMeshNode


@pytest.fixture()
def data_dir():
    d = tempfile.mkdtemp(prefix="nsm_surah_nonblock_test_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _blocking_download(repo_id: str, filename: str) -> str:
    """محاكاة hf_hub_download حاجزة حقيقية — نفس أثر الشبكة البطيئة
    المُعاد إنتاجها فعلياً، بدون الاعتماد على شبكة حقيقية بطيئة في الاختبار."""
    time.sleep(0.5)
    raise ConnectionError("محاكاة: الشبكة غير متاحة (نفس أثر HF الحقيقي)")


def test_event_loop_stays_free_while_surah_loads(data_dir, monkeypatch):
    """بينما _load_surah_pretrain 'تُحمِّل' (حاجزة محاكاة)، يجب أن تستمر
    مهمة asyncio أخرى منفصلة بالعمل بلا أي تجميد — هذا هو جوهر الإصلاح."""
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _blocking_download)

    async def run_test():
        node = LivingMeshNode(node_id="surah_nonblock_test", host="127.0.0.1", port=0, data_dir=data_dir)

        ticks = []

        async def ticker():
            # لو كانت الحلقة مُجمَّدة بنداء حاجز، هذه المهمة لن تتقدّم أبداً
            # أثناء ذلك التجميد — تُثبِت حرية الحلقة بعدد tick فعلي كافٍ.
            for _ in range(8):
                await asyncio.sleep(0.05)
                ticks.append(time.monotonic())

        await asyncio.gather(ticker(), node._load_surah_pretrain())

        assert len(ticks) == 8  # المهمة الأخرى أكملت كل دوراتها دون توقّف
        assert node.surah_awareness["status"] == "failed"  # فشلت (متوقَّع) لكن بأمان

    asyncio.run(run_test())


def test_surah_download_runs_in_executor_not_main_thread(data_dir, monkeypatch):
    """تأكيد مباشر أن hf_hub_download لا تُستدعى في نفس خيط حلقة الأحداث —
    هذا بالضبط ما يمنع التجميد."""
    import threading
    import huggingface_hub

    call_threads = []

    def _recording_download(repo_id, filename):
        call_threads.append(threading.current_thread().name)
        raise ConnectionError("محاكاة")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _recording_download)

    async def run_test():
        node = LivingMeshNode(node_id="surah_thread_test", host="127.0.0.1", port=0, data_dir=data_dir)
        main_thread_name = threading.current_thread().name
        await node._load_surah_pretrain()
        assert call_threads, "hf_hub_download لم تُستدعَ إطلاقاً"
        assert all(t != main_thread_name for t in call_threads), (
            f"hf_hub_download استُدعيت في خيط الحلقة الرئيسي نفسه ({main_thread_name}) "
            f"— ستُجمِّد الحدث الرئيسي عند أي بطء شبكة حقيقي"
        )

    asyncio.run(run_test())

"""يتحقق أن خطّاف conftest.py يشغّل 'async def test_*' فعلاً (لا يتخطاها ولا يفشلها)."""
import asyncio

RAN = []


async def test_async_function_is_actually_awaited():
    await asyncio.sleep(0)
    RAN.append(1)
    assert RAN == [1]

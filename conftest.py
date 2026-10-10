"""
conftest.py — جذر المشروع
==========================
يضيف جذر المستودع لـ sys.path تلقائياً بناءً على موقع هذا الملف نفسه (لا
مسارات مُثبَّتة يدوياً مثل "/home/claude/build" الموجودة في بعض اختبارات
ai/*.py القديمة — تلك تعمل فقط على جهاز كاتبها الأصلي، وتفشل على أي جهاز
آخر أو في أي CI). هذا الملف يجعل `import ai.xxx` و`import knowledge.xxx`
يعملان من أي مكان يُشغَّل منه pytest، بما في ذلك جهازك المحلي وGitHub Actions.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── اختبارات async بلا اعتماد على pytest-asyncio ─────────────────────────
# pytest-asyncio غير موجود في requirements، فكانت أي دالة 'async def test_*'
# تفشل بـ"async def functions are not natively supported" وتُهمَل فعلياً من
# أي تشغيل تلقائي (وُجدت هكذا: test_unified_memory_perf وtest_distributed_mesh).
# هذا الخطّاف يشغّل الـcoroutine عبر asyncio.run فقط حين لا يوجد مكوّن إضافي
# async مفعّل (pytest-asyncio/anyio) يتولّاها، فلا يتعارض معها.
import asyncio
import inspect


def pytest_pyfunc_call(pyfuncitem):
    func = pyfuncitem.obj
    if not inspect.iscoroutinefunction(func):
        return None
    pm = pyfuncitem.config.pluginmanager
    if pm.hasplugin("asyncio") or pm.hasplugin("pytest_asyncio"):
        return None  # pytest-asyncio يتولّاها
    if pyfuncitem.get_closest_marker("anyio") is not None:
        return None  # anyio يتولّاها
    argnames = pyfuncitem._fixtureinfo.argnames
    kwargs = {name: pyfuncitem.funcargs[name] for name in argnames}
    asyncio.run(func(**kwargs))
    return True

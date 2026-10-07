"""
محاكاة تواصل موزّع بين عقدتين (alpha بذرة + zeta تنضم إليها) — نسخة مُصحَّحة.

النسخة القديمة كانت تستخدم `websockets.serve(node._handle_ws_connection, ...)`
وهذه الدالة أُزيلت من LivingMeshNode منذ زمن طويل (commit 1ad85c1 فأقدم) لصالح
مسار aiohttp الموحّد في ai/node_launcher.py (handle_ws). لأن الاستثناء كان
يُطلَق داخل asyncio.create_task() غير المُنتظَر، كان يُبتلَع بصمت، فتفشل
الاختبار بفشل التأكيد النهائي فقط (discovered=False) بدل خطأ واضح — هذا
الإصلاح يستخدم نفس handle_ws الإنتاجي الحقيقي بدل الدالة المحذوفة.
"""
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai.living_mesh import LivingMeshNode
from ai.node_launcher import handle_ws
from aiohttp import web


async def _serve(node, port):
    app = web.Application()
    app["node"] = node
    app.add_routes([web.get("/ws", handle_ws)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    return runner


async def _simulate_distributed():
    """محاكاة تواصل بين عقدتين بعناوين IP مختلفة (افتراضية)، بمجلدي بيانات معزولين."""
    with tempfile.TemporaryDirectory(prefix="nsm_dist_test_") as tmp:
        alpha = LivingMeshNode(node_id="mesh_alpha_dist", host="127.0.0.1", port=9001,
                                data_dir=str(Path(tmp) / "alpha"))
        zeta = LivingMeshNode(node_id="mesh_zeta_dist", host="127.0.0.1", port=9002,
                               data_dir=str(Path(tmp) / "zeta"))

        runners = []
        try:
            return await _simulate_with(alpha, zeta, runners)
        finally:
            # join_network() يبدأ خيطَي daemon (قدرات + تطوّر ذاتي) يكتبان في
            # data_dir؛ إيقافهما قبل خروج TemporaryDirectory يمنع سباق
            # التنظيف (OSError: Directory not empty). وrunners تُنظَّف حتى لو
            # فشل أي تأكيد قبل cleanup الأصلي.
            for r in runners:
                try:
                    await r.cleanup()
                except Exception:
                    pass
            alpha.mark_offline()
            zeta.mark_offline()


async def _simulate_with(alpha, zeta, runners):
    if True:
        alpha.join_network()
        runner_alpha = await _serve(alpha, 9001)
        runners.append(runner_alpha)

        seed = {"id": "mesh_alpha_dist", "host": "127.0.0.1", "port": 9001}
        zeta.join_network(seed_nodes=[seed])
        runner_zeta = await _serve(zeta, 9002)
        runners.append(runner_zeta)

        # انضمام zeta فعلياً عبر الشبكة (لا محاكاة)
        ok = await zeta.request_peers(seed["host"], seed["port"])
        await asyncio.sleep(0.5)

        state_zeta = zeta._load_state()
        discovered = any(n.get("id") == "mesh_alpha_dist" for n in state_zeta["nodes"].values())
        print(f"Discovered Alpha: {discovered} (request_peers ok={ok})")

        # إرسال خبرة مباشرة من alpha إلى zeta والتحقق من وصولها فعلياً
        exp_data = {"test": "distributed_logic", "value": 1.0}
        sent = await alpha.send_to_peer("127.0.0.1", 9002, "distributed_test", exp_data)
        await asyncio.sleep(0.5)

        state_zeta2 = zeta._load_state()
        received = [e for e in state_zeta2.get("global_experience", [])
                    if e.get("kind") == "distributed_test"]
        print(f"send_to_peer ok={sent}, zeta received distributed_test: {bool(received)}")

        assert discovered, "Zeta should discover Alpha in the distributed simulation"
        assert sent, "alpha.send_to_peer to zeta should succeed"
        assert received, "Zeta should have received the distributed_test experience"
        print("🏆 test_distributed_mesh passed")


def test_distributed_simulation():
    # الاختبار coroutine يُشغَّل هنا عبر asyncio.run بدل الاعتماد على
    # pytest-asyncio (غير موجود في requirements) — بدون هذا يرفضه pytest
    # بـ "async def functions are not natively supported".
    asyncio.run(_simulate_distributed())


if __name__ == "__main__":
    asyncio.run(_simulate_distributed())

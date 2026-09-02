"""
اختبار: كل بث تدرجات (broadcast_gradients) يصل لكل قرين مرة واحدة بالضبط
===========================================================================
اكتُشف سابقاً بالقياس الفعلي: عقدة بثّت 5 رسائل gradient_push مباشرة إلى
قرينين، لكن أحد القرينين استقبل 13 نسخة (تضخيم عبر ترحيل Gossip المحدود
الموجود أصلاً بالتصميم — hops<3 — لأن حمولة gradient_push لم تكن تحمل
_gossip_id ثابت، فكل عقدة تستقبلها كانت تفترضها "جديدة" دوماً).

الإصلاح: gradient_mesh.py الآن يُرفق _gossip_id ثابت لكل بث منطقي واحد
(نفس المعرّف لكل الأقران المستهدَفين بنفس الاستدعاء)، فتعمل آلية
_seen_gossip_ids الموجودة أصلاً بـ living_mesh.py على امتصاص أي نسخة
مكرّرة تصل عبر مسار آخر (ترحيل Gossip) بدل تخزينها/إعادة بثّها من جديد.

هذا الاختبار يشغّل 3 عقد معزولة بعمليات OS منفصلة تماماً (نفس منهجية
test_real_p2p.py)، gamma تبثّ N تدرجات مباشرة لـ alpha وbeta، ويتحقق أن
كل واحدة منهما استقبلت بالضبط N نسخة — لا أكثر ولا أقل.
"""
import asyncio
import json
import multiprocessing
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

ALPHA_PORT, BETA_PORT, GAMMA_PORT = 8931, 8932, 8933
HOST = "127.0.0.1"
N_BROADCASTS = 5


def _run_node(role: str, data_dir: str, port: int, seed: dict, result_path: str):
    async def _main():
        import torch
        from aiohttp import web
        from ai.living_mesh import LivingMeshNode
        from ai.node_launcher import handle_ws
        from ai.gradient_mesh import GradientExchangeProtocol

        node = LivingMeshNode(node_id=f"node_{role}", host=HOST, port=port, data_dir=data_dir)
        node.join_network(seed_nodes=[seed] if seed else None)

        app = web.Application()
        app["node"] = node
        app.add_routes([web.get("/ws", handle_ws)])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, HOST, port)
        await site.start()

        result = {"role": role}
        model = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.Linear(32, 8))
        x = torch.randn(4, 64)
        model(x).sum().backward()

        proto = GradientExchangeProtocol(node_id=node.node_id, mesh_node=node)
        await proto.connect()

        if seed:
            await node.request_peers(seed["host"], seed["port"])
            await asyncio.sleep(0.5)

        if role == "gamma":
            await asyncio.sleep(2.0)
            await node.request_peers(seed["host"], seed["port"])
            await asyncio.sleep(0.3)
            for _ in range(N_BROADCASTS):
                await proto.broadcast_gradients(model)

        # امنح وقتاً كافياً لأي ترحيل Gossip محتمل يصل (لو الإصلاح فشل بيظهر هنا)
        # + تبقى alpha وbeta حيّتين طوال عمر الاختبار (وقت الاكتشاف بالمحاولات
        # المتكررة retries=3 قد يمتد لعدة ثوانٍ قبل ما gamma تبدأ البث فعلياً)
        hold = 14.0 if role in ("alpha", "beta") else 4.0
        await asyncio.sleep(hold)

        state = node._load_state()
        received = [e for e in state.get("global_experience", []) if e.get("kind") == "gradient_push"]
        unique_gossip_ids = {e["data"].get("_gossip_id") for e in received if e.get("data")}
        result["received_count"] = len(received)
        result["unique_gossip_ids_count"] = len(unique_gossip_ids)

        Path(result_path).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        await runner.cleanup()

    asyncio.run(_main())


def run_test():
    print(f"🔬 اختبار عدم تكرار gradient_push — gamma تبثّ {N_BROADCASTS} تدرجات مباشرة لألفا وبيتا\n")
    with tempfile.TemporaryDirectory(prefix="nsm_dedup_test_") as tmp:
        results_dir = Path(tmp) / "results"; results_dir.mkdir()
        ctx = multiprocessing.get_context("spawn")

        p_alpha = ctx.Process(target=_run_node, args=(
            "alpha", str(Path(tmp) / "alpha"), ALPHA_PORT, None, str(results_dir / "alpha.json")))
        p_alpha.start(); time.sleep(1.5)

        p_beta = ctx.Process(target=_run_node, args=(
            "beta", str(Path(tmp) / "beta"), BETA_PORT,
            {"id": "seed", "host": HOST, "port": ALPHA_PORT}, str(results_dir / "beta.json")))
        p_beta.start(); time.sleep(2.0)

        p_gamma = ctx.Process(target=_run_node, args=(
            "gamma", str(Path(tmp) / "gamma"), GAMMA_PORT,
            {"id": "seed", "host": HOST, "port": ALPHA_PORT}, str(results_dir / "gamma.json")))
        p_gamma.start()

        for p in (p_gamma, p_beta, p_alpha):
            p.join(timeout=30)
        for p in (p_alpha, p_beta, p_gamma):
            if p.is_alive(): p.terminate()

        results = {}
        for name in ("alpha", "beta", "gamma"):
            f = results_dir / f"{name}.json"
            results[name] = json.loads(f.read_text()) if f.exists() else None

        ok = True
        for name in ("alpha", "beta"):
            r = results.get(name)
            if not r:
                print(f"❌ {name}: العملية فشلت (لا نتيجة)")
                ok = False
                continue
            got = r["received_count"]
            uniq = r["unique_gossip_ids_count"]
            status = "✅" if got == N_BROADCASTS == uniq else "❌"
            print(f"{status} {name}: استقبلت {got} رسالة gradient_push (متوقَّع {N_BROADCASTS}), معرّفات فريدة={uniq}")
            if got != N_BROADCASTS or uniq != N_BROADCASTS:
                ok = False

        print("\n" + ("🏆 نجح: لا تضخيم — كل بث وصل مرة واحدة بالضبط لكل قرين"
                       if ok else "💥 فشل: لسّه فيه تضخيم أو نقص بالرسائل المستلمة"))
        return ok


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)

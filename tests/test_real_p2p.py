"""
اختبار Mesh حقيقي متعدد العقد — ثلاث عقد سيادية معزولة فعلياً
================================================================
كل عقدة تعمل بعملية نظام تشغيل (process) منفصلة تماماً، بمجلد بيانات
(artifacts/keys) خاص بها لوحدها — عشان نضمن أن أي "نجاح" ناتج فعلاً عن
اتصال شبكي حقيقي عبر WebSocket، وليس مجرد ملف حالة مشترك بنفس العملية
(وهو خطأ منهجي كان موجود بالنسخة السابقة من هذا الاختبار).

السيناريو:
  1. Alpha تُقلَع كعقدة بذرة (seed) — لا تعرف أحداً.
  2. Beta تُقلَع، تتصل بـ Alpha وتكتشفها (peer_discovery)، Alpha تسجّل Beta كذلك (تسجيل متبادل).
  3. Gamma تُقلَع، تتصل بـ Alpha، فتكتشف Alpha *و* Beta معاً (اكتشاف متعدد القفزات
     عبر Alpha، دون أي اتصال مباشر سابق بين Gamma وBeta).
  4. Gamma ترسل رسالة Gossip موقّعة مباشرة إلى Beta (P2P حقيقي، تمر فوق الشبكة
     مباشرة بين Gamma وBeta بدون المرور عبر Alpha).
  5. نتحقق أن Beta استقبلت الرسالة وتحقّقت من توقيعها بمفتاح Gamma العام —
     المفتاح اللي تعلّمته Beta بشكل غير مباشر (عبر Alpha)، لا بشكل مباشر من Gamma.

نجاح هذا الاختبار يثبت: اكتشاف لامركزي حقيقي + تبادل مفاتيح متعدد القفزات +
رسائل P2P موقّعة مشفّرة تعمل فعلاً عبر عمليات/منافذ منعزلة.
"""
import asyncio
import json
import logging
import multiprocessing
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.WARNING)

ALPHA_PORT = 8901
BETA_PORT = 8902
GAMMA_PORT = 8903
HOST = "127.0.0.1"


def _isolate_node_storage(tmp_root: str, node_id: str):
    """يعزل هذه العملية بمجلد artifacts/keys خاص بها فقط — قبل أي استيراد لـ
    ai.living_mesh — عشان نضمن عدم مشاركة أي حالة بين العقد الثلاث."""
    import ai.living_mesh as lm

    node_dir = Path(tmp_root) / node_id
    node_dir.mkdir(parents=True, exist_ok=True)
    lm.LIVING_MESH_DIR = node_dir
    lm.NETWORK_STATE = node_dir / "network_state.json"
    return lm


def _run_node_process(tmp_root: str, node_id: str, port: int, seed: dict, result_path: str, hold_seconds: float):
    """نقطة دخول العملية المعزولة لكل عقدة. تُشغَّل بعملية OS منفصلة تماماً."""
    async def _main():
        lm = _isolate_node_storage(tmp_root, node_id)
        from aiohttp import web

        node = lm.LivingMeshNode(node_id=node_id, host=HOST, port=port)
        node.join_network()

        async def handle_ws(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await node._handle_aiohttp_ws_msg(ws, data)
            return ws

        app = web.Application()
        app.add_routes([web.get("/ws", handle_ws)])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, HOST, port)
        await site.start()

        result = {"node_id": node_id, "steps": {}}
        t0 = time.time()

        if seed:
            await node.request_peers(seed["host"], seed["port"])
            await asyncio.sleep(1.0)
            state = node._load_state()
            known = sorted(k for k in state.get("nodes", {}).keys() if k != node_id)
            result["steps"]["discovered_peers_round1"] = known
            result["steps"]["discovery_latency_ms"] = round((time.time() - t0) * 1000, 1)

        # Beta فقط: تعيد الاكتشاف (round 2) بعد إمهال Gamma وقت تسجّل بيه لدى Alpha —
        # هذا يثبت الاكتشاف متعدد القفزات لمفتاح Gamma العام (لا اتصال مباشر سابق بينهما)
        if node_id == "node_beta":
            await asyncio.sleep(3.0)
            await node.request_peers(seed["host"], seed["port"])
            await asyncio.sleep(1.0)
            state = node._load_state()
            known2 = sorted(k for k in state.get("nodes", {}).keys() if k != node_id)
            result["steps"]["discovered_peers_round2"] = known2

        # Gamma فقط: تنتظر لحين تأكّد اكتمال الجولة الثانية عند Beta، ثم ترسل
        # رسالة P2P مباشرة إليها بدون المرور عبر Alpha إطلاقاً
        if node_id == "node_gamma":
            await asyncio.sleep(3.0)
            t1 = time.time()
            await node.send_to_peer(HOST, BETA_PORT, "sovereign_gossip", {
                "message": "سلام من Gamma إلى Beta مباشرة، بدون وسيط",
                "sent_at": t1,
            })
            result["steps"]["direct_send_to_beta"] = True

        # انتظر لحين استقبال أي رسائل واردة (لو كانت هذي العقدة الهدف)
        await asyncio.sleep(hold_seconds)

        final_state = node._load_state()
        gossip = [e for e in final_state.get("global_experience", [])
                  if e.get("kind") == "sovereign_gossip"]
        result["steps"]["received_gossip"] = gossip
        result["steps"]["known_keys_on_disk"] = sorted(
            p.stem for p in node.keys_dir.glob("*.pub")
        )

        Path(result_path).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        await runner.cleanup()

    asyncio.run(_main())


def run_test():
    print("🚀 اختبار Mesh حقيقي — ثلاث عقد معزولة بعمليات OS منفصلة\n")

    with tempfile.TemporaryDirectory(prefix="nsm_p2p_test_") as tmp_root:
        results_dir = Path(tmp_root) / "results"
        results_dir.mkdir()

        ctx = multiprocessing.get_context("spawn")

        # 1) Alpha: عقدة بذرة، لا تعرف أحداً مسبقاً
        p_alpha = ctx.Process(
            target=_run_node_process,
            args=(tmp_root, "node_alpha", ALPHA_PORT, None, str(results_dir / "alpha.json"), 12.0),
        )
        p_alpha.start()
        time.sleep(1.5)  # امنح Alpha وقت كافي لتشغيل الخادم قبل أي اتصال بها

        # 2) Beta: تكتشف Alpha
        p_beta = ctx.Process(
            target=_run_node_process,
            args=(tmp_root, "node_beta", BETA_PORT,
                  {"host": HOST, "port": ALPHA_PORT}, str(results_dir / "beta.json"), 3.0),
        )
        p_beta.start()
        time.sleep(2.0)  # امنح Beta وقت تسجّل بيه Alpha قبل ما تدخل Gamma

        # 3) Gamma: تكتشف Alpha، وتتعلم عن Beta بشكل غير مباشر (متعدد القفزات)
        #    ثم ترسل رسالة مباشرة لـ Beta
        p_gamma = ctx.Process(
            target=_run_node_process,
            args=(tmp_root, "node_gamma", GAMMA_PORT,
                  {"host": HOST, "port": ALPHA_PORT}, str(results_dir / "gamma.json"), 2.0),
        )
        p_gamma.start()

        p_gamma.join(timeout=25)
        p_beta.join(timeout=25)
        p_alpha.join(timeout=25)

        for p in (p_alpha, p_beta, p_gamma):
            if p.is_alive():
                p.terminate()

        results = {}
        for name in ("alpha", "beta", "gamma"):
            f = results_dir / f"{name}.json"
            results[name] = json.loads(f.read_text()) if f.exists() else None

        # ---- التقرير ----
        ok = True

        gamma = results.get("gamma")
        beta = results.get("beta")

        print("── الاكتشاف اللامركزي ──")
        if beta and "node_alpha" in beta["steps"].get("discovered_peers_round1", []):
            print(f"✅ Beta اكتشفت Alpha مباشرة خلال {beta['steps']['discovery_latency_ms']} ms")
        else:
            print("❌ Beta لم تكتشف Alpha")
            ok = False

        if gamma:
            discovered = gamma["steps"].get("discovered_peers_round1", [])
            print(f"   Gamma اكتشفت عبر Alpha: {discovered}")
            if "node_alpha" in discovered and "node_beta" in discovered:
                print(f"✅ اكتشاف متعدد القفزات نجح: Gamma تعرفت على Beta *بدون* اتصال مباشر سابق (خلال {gamma['steps']['discovery_latency_ms']} ms)")
            else:
                print("❌ فشل الاكتشاف متعدد القفزات — Gamma ما تعرفت على Beta عبر Alpha")
                ok = False
        else:
            print("❌ Gamma process فشلت")
            ok = False

        if beta:
            round2 = beta["steps"].get("discovered_peers_round2", [])
            print(f"   Beta أعادت الاكتشاف (جولة 2) وعرفت: {round2}")
            if "node_gamma" not in round2:
                print("❌ Beta ما تعرفت على Gamma عبر إعادة الاكتشاف")
                ok = False

        print("\n── رسالة P2P مباشرة (Gamma → Beta، بلا وسيط) ──")
        if beta:
            gossip = beta["steps"].get("received_gossip", [])
            if gossip:
                print(f"✅ Beta استقبلت رسالة موقّعة من Gamma مباشرة: {gossip[0]['data']['message']}")
                print(f"   المُرسِل: {gossip[0]['node']}")
            else:
                print("❌ Beta لم تستقبل رسالة Gossip من Gamma")
                ok = False

            keys = beta["steps"].get("known_keys_on_disk", [])
            print(f"\n── تبادل المفاتيح ──")
            print(f"   مفاتيح محفوظة فعلياً على قرص Beta المعزول: {keys}")
            if "node_gamma" in keys:
                print("✅ Beta تحقّقت من توقيع Gamma بمفتاح تعلّمته بشكل غير مباشر (عبر Alpha) — لا اتصال مباشر سابق")
            else:
                print("❌ Beta ما عندها مفتاح Gamma — التحقق من التوقيع كان سيفشل")
                ok = False

        print("\n" + ("🏆 النتيجة: كل خطوات التواصل P2P اللامركزي تعمل فعلياً (اكتشاف + تعدد قفزات + رسائل موقّعة مباشرة)"
                       if ok else "💥 النتيجة: فشل جزء من سلسلة التواصل P2P — راجع التفاصيل أعلاه"))
        return ok


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)

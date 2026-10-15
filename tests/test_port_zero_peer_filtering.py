"""وُجد بتكليف شبكة حقيقية من 3 عقد بمهمة تلخيص جماعي فعلية
(submit_collective_summary): عقدة عميل انضمت بـport=0 (شائع لعقدة/سكربت
منظِّم لا خادم له يستقبل اتصالات) انتشرت عبر gossip كقرين عادي، ثم
اختيرت فعلياً كـ'عامل' لمهمة حقيقية — محاولة الاتصال بـhost:0 مضمونة
الفشل استهلكت timeout_per_task كاملاً (15 ثانية حقيقية) قبل أن يُعاد
توجيه المهمة لعامل آخر. السبب: `port is None` لا يستبعد port=0 (0 is
not None). نفس النمط المعيب كان مكرَّراً في 9 مواضع عبر 4 ملفات."""
import pytest


def _peer(id_, host="127.0.0.1", port=9999):
    return {"id": id_, "host": host, "port": port, "capabilities": ["text"], "status": "online"}


def test_rank_workers_excludes_port_zero():
    from ai.mesh_job_orchestrator import MeshJobOrchestrator

    class FakeNode:
        node_id = "me"
        def _get_active_peers_list(self, require_capabilities=None):
            return [_peer("good", port=9001), _peer("zero_port", port=0), _peer("me", port=9002)]

    orch = MeshJobOrchestrator(FakeNode())
    workers = orch._rank_workers()
    ids = [w["peer_id"] for w in workers]
    assert ids == ["good"]  # لا zero_port (منفذ غير صالح)، ولا me (الذات)


def test_rank_workers_excludes_none_port_too():
    from ai.mesh_job_orchestrator import MeshJobOrchestrator

    class FakeNode:
        node_id = "me"
        def _get_active_peers_list(self, require_capabilities=None):
            return [_peer("good", port=9001), {"id": "no_port", "host": "127.0.0.1", "port": None}]

    orch = MeshJobOrchestrator(FakeNode())
    ids = [w["peer_id"] for w in orch._rank_workers()]
    assert ids == ["good"]


def test_node_health_layer_routes_table_excludes_port_zero():
    """المسار الأساسي (routes_table → rank_workers) — لا مسار fallback
    فرعي فقط. كان قرين port=0 يمرّ عبر routes_table() بلا أي تصفية
    إطلاقاً (الفحص القديم كان في مسار fallback لا يُستخدَم أبداً طالما
    routes_table() ترجع صفاً واحداً على الأقل)، فيظهر كمسار صالح
    ويُختار فعلياً لمهمة حقيقية."""
    from ai.node_health_layer import NodeHealthLayer

    class FakeNode:
        node_id = "me"
        data_dir = None
        def _get_active_peers_list(self, require_capabilities=None):
            return [_peer("good", port=9001), _peer("zero_port", port=0)]
        def get_reputation(self):
            return {}

    layer = NodeHealthLayer(FakeNode())
    table = layer.routes_table()["routes"]
    ids = {r["peer_id"] for r in table}
    assert ids == {"good"}

    ranked = layer.rank_workers()
    ranked_ids = {w.get("peer_id") for w in ranked}
    assert ranked_ids == {"good"}


"""mark_offline() يجب أن يوقف فعلياً كلا خيطَي المراقبة (capability_watch
وself_evolution_watch) التي يبدأها join_network() — لا يترك العقدة
تكتب في الخلفية بعد مغادرتها الشبكة. اكتشفتُ غياب هذا التحقق أثناء
تتبّع عطل متقطّع حقيقي في tests/test_nsm_node_v2_slice.py: تنظيف
tempfile.TemporaryDirectory() كان يتسابق مع أول تكة كتابة من هذين
الخيطين (OSError: Directory not empty)."""
import tempfile
from pathlib import Path


def _isolate(tmp, node_id):
    import ai.living_mesh as lm
    d = Path(tmp) / node_id
    d.mkdir(parents=True, exist_ok=True)
    lm.LIVING_MESH_DIR = d
    lm.NETWORK_STATE = d / "network_state.json"
    lm.CONTENT_DIR = d / "content"
    lm.CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    return lm


def test_mark_offline_stops_both_watch_threads(tmp_path):
    lm = _isolate(tmp_path, "n1")
    n = lm.LivingMeshNode(node_id="n1", host="127.0.0.1", port=0)
    n.join_network()
    cap_thread = n._capability_watch_thread
    evo_thread = n._self_evolution_thread
    assert cap_thread is not None and cap_thread.is_alive()
    assert evo_thread is not None and evo_thread.is_alive()

    n.mark_offline()

    assert n._capability_watch_thread is None
    assert n._self_evolution_thread is None
    # الخيطان المحدَّدان بالذات (لا تخمين بالاسم) توقّفا فعلياً — join()
    # داخل stop_* يضمن هذا قبل أن يعود mark_offline.
    assert not cap_thread.is_alive()
    assert not evo_thread.is_alive()


def test_mark_offline_updates_persisted_status(tmp_path):
    lm = _isolate(tmp_path, "n2")
    n = lm.LivingMeshNode(node_id="n2", host="127.0.0.1", port=0)
    n.join_network()
    n.mark_offline()
    state = n._load_state()
    assert state["nodes"]["n2"]["status"] == "offline"


def test_mark_offline_without_join_is_safe_noop():
    with tempfile.TemporaryDirectory() as tmp:
        lm = _isolate(tmp, "n3")
        n = lm.LivingMeshNode(node_id="n3", host="127.0.0.1", port=0)
        n.mark_offline()  # لم ينضم للشبكة أصلاً — لا يجب أن يرفع استثناء

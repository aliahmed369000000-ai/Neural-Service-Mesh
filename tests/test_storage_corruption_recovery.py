"""تلف ملف التخزين (توقف مفاجئ/قرص) لا يجب أن يمحو تاريخ العُقد بصمت،
والتسجيل من خيوط متعددة لا يفسد السجل."""
import threading
from core.node import BaseNode, NodeSchema
from core.registry import NodeRegistry
from storage.file_storage import FileStorage


class N(BaseNode):
    input_schema = NodeSchema(fields={}, required=[])
    output_schema = NodeSchema(fields={}, required=[])
    def process(self, data):
        return {}


def test_corrupt_file_falls_back_to_backup_and_is_preserved(tmp_path):
    st = FileStorage(str(tmp_path))
    st.save("a.json", {"v": 1})
    st.save("a.json", {"v": 2})           # الآن .bak = v1
    (tmp_path / "a.json").write_text("{ corrupt", encoding="utf-8")
    assert st.load("a.json") == {"v": 1}   # استُعيد من النسخة الاحتياطية
    assert any(p.name.startswith("a.json.corrupt-") for p in tmp_path.iterdir())
    assert not (tmp_path / "a.json").exists()


def test_corrupt_without_backup_returns_none_but_keeps_evidence(tmp_path):
    st = FileStorage(str(tmp_path))
    (tmp_path / "b.json").write_text("not json", encoding="utf-8")
    assert st.load("b.json") is None
    assert any(p.name.startswith("b.json.corrupt-") for p in tmp_path.iterdir())


def test_normal_save_load_unchanged_and_list_files_ignores_backups(tmp_path):
    st = FileStorage(str(tmp_path))
    st.save("c.json", {"x": 1}); st.save("c.json", {"x": 2})
    assert st.load("c.json") == {"x": 2}
    assert st.list_files() == ["c.json"]
    assert st.load("missing.json") is None


def test_registry_recovers_nodes_from_backup_after_corruption(tmp_path):
    st = FileStorage(str(tmp_path))
    reg = NodeRegistry(st)
    n1 = N("one"); reg.register(n1)
    n2 = N("two"); reg.register(n2)         # .bak يحوي n1 فقط
    (tmp_path / "nodes.json").write_text("{trunc", encoding="utf-8")
    reg2 = NodeRegistry(FileStorage(str(tmp_path)))
    assert reg2.get_meta_by_name("one") is not None


def test_concurrent_register_keeps_all_nodes(tmp_path):
    reg = NodeRegistry(FileStorage(str(tmp_path)))
    errors = []
    def work(i):
        try:
            for j in range(10):
                reg.register(N(f"n{i}_{j}"))
                reg.refresh_meta(next(iter(reg._nodes)))
        except Exception as e:
            errors.append(e)
    ts = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not errors
    assert reg.count() == 80
    assert NodeRegistry(FileStorage(str(tmp_path))).list_metadata().__len__() == 80

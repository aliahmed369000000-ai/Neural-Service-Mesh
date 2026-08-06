"""
اختبارات ai/code_agent.py — حماية المسارات (_safe_path).

المشكلة: read_file/edit_file/create_file/list_files/fix_file/
summarize_file كانت تمرّر path (نص خام يصل مباشرة من رسالة دردشة
المستخدم عبر nsm_chat._handle_code_command، بلا أي تحقق) إلى
ROOT / path مباشرة. مسار مثل "../../../etc/passwd" أو مسار مطلق
"/etc/passwd" كان يهرب فعلياً من مجلد المشروع، فيسمح لأي رسالة دردشة
عادية بقراءة أو كتابة أي ملف على القرص (مثل secrets.toml على
Streamlit Cloud). هذا الملف يتحقق أن كل دوال path محمية الآن، وأن
الاستخدام الطبيعي داخل المشروع لم يتأثر.
"""
from ai.code_agent import (
    ROOT, _safe_path, _UNSAFE_PATH_MSG,
    read_file, edit_file, create_file, list_files, fix_file, summarize_file,
)


class TestSafePath:
    def test_relative_path_inside_root_allowed(self):
        assert _safe_path("ai/code_agent.py") == (ROOT / "ai/code_agent.py").resolve()

    def test_dotdot_traversal_rejected(self):
        assert _safe_path("../../../etc/passwd") is None

    def test_absolute_path_rejected(self):
        assert _safe_path("/etc/passwd") is None

    def test_mixed_traversal_rejected(self):
        """مسار يبدأ بمجلد صحيح ثم يهرب لاحقاً بـ '..' يجب أن يُرفض أيضاً."""
        assert _safe_path("ai/../../../etc/hosts") is None

    def test_empty_or_blank_rejected(self):
        assert _safe_path("") is None
        assert _safe_path("   ") is None

    def test_dot_root_itself_allowed(self):
        assert _safe_path(".") == ROOT.resolve()


class TestPathFunctionsBlockTraversal:
    """كل دالة تستقبل path من نص الدردشة يجب أن ترفض الهروب برسالة
    _UNSAFE_PATH_MSG بدل تنفيذ العملية فعلياً خارج المشروع."""

    def test_read_file_blocks_traversal(self):
        assert read_file("../../../etc/passwd") == _UNSAFE_PATH_MSG

    def test_read_file_blocks_absolute(self):
        assert read_file("/etc/passwd") == _UNSAFE_PATH_MSG

    def test_edit_file_blocks_traversal(self):
        assert edit_file("../../../etc/hostname", "a", "b") == _UNSAFE_PATH_MSG

    def test_create_file_blocks_traversal(self, tmp_path):
        """يتحقق أيضاً أن الملف فعلياً لم يُنشَأ خارج المشروع، لا فقط أن
        الرسالة صحيحة."""
        target = "../../../tmp/nsm_traversal_probe_should_not_exist.py"
        assert create_file(target, "print('pwned')") == _UNSAFE_PATH_MSG
        import pathlib
        assert not (ROOT / target).resolve().exists()

    def test_list_files_blocks_traversal(self):
        assert list_files("../../../etc") == _UNSAFE_PATH_MSG

    def test_fix_file_blocks_traversal(self):
        assert fix_file("/etc/passwd") == _UNSAFE_PATH_MSG

    def test_summarize_file_blocks_traversal(self):
        assert summarize_file("../../../etc/passwd") == _UNSAFE_PATH_MSG


class TestPathFunctionsStillWorkNormally:
    """الإصلاح لا يجب أن يكسر أي استخدام طبيعي داخل مجلد المشروع."""

    def test_read_file_normal_usage(self):
        content = read_file("ai/code_agent.py")
        assert "def read_file" in content

    def test_create_edit_read_roundtrip(self, tmp_path):
        rel = "ai/_tmp_path_safety_probe.py"
        try:
            assert create_file(rel, "x = 1\n").startswith("✅")
            assert edit_file(rel, "x = 1", "x = 2").startswith("✅")
            assert read_file(rel).strip() == "x = 2"
        finally:
            full = (ROOT / rel).resolve()
            if full.exists():
                full.unlink()

    def test_list_files_root_still_works(self):
        out = list_files(".")
        assert "code_agent.py" in out

    def test_summarize_file_normal_usage(self):
        out = summarize_file("ai/code_agent.py")
        assert "read_file" in out

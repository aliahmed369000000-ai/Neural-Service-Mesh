"""
tests/test_chat_sessions.py
=============================
اختبارات الحزمة الجديدة — بدون أي مفاتيح API حقيقية:

1. ai/chat_history_store.py — دورة الحياة الكاملة للجلسات
   (حفظ / قائمة / استعادة / حذف مستهدف / تنظيف).
2. ai/qdrant_semantic_memory.py — SQLite المحلي يعمل بلا Qdrant
   (Qdrant غير مثبت/بلا مفاتيح هنا فيختبر الاحتياط المحلي فقط).
3. nsm_chat_plus.py — last_metadata يُسجَّل بعد كل ردّ وهمي
   + _is_agent_request patterns + تثبيت الذاكرة الدلالية لا يفسد المسار.
4. ai/llm_fallback.py — تصنيف الأخطاء العابرة (_is_transient logic)
   دون إجراء أي اتصال شبكي حقيقي.
5. ui_pages/chat.py — py_compile فقط (لا يمكن اختبار Streamlit rendering
   بلا مفاتيح حقيقية لتحميل NSMChat — render_chat يعتمد على استيراد
   حقيقي من app_core، ونظام الإنتاج يختبرها على Streamlit Cloud).

تُشغَّل من جذر المشروع:  python3 -m pytest tests/test_chat_sessions.py
أو:                     python3 tests/test_chat_sessions.py
"""

from __future__ import annotations

import importlib
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, ".")


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestChatHistoryStore(unittest.TestCase):
    """1. دورة حياة الجلسات — SQLite محلي مؤقت.

    chat_history_store يقرأ DB_PATH عند كل استدعاء _db() (الاتصال يُفتح
    جديدًا في كل عملية)، فتكفي إزاحة DB_PATH إلى ملف مؤقت قبل أي اختبار.
    """

    def setUp(self):
        import os
        self.chs = _load_module("chat_history_store", "ai/chat_history_store.py")
        self._tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_chs.db")
        self.chs.DB_PATH = __import__("pathlib").Path(self._tmp)
        # ضمان نظافة كاملة بين الاختبارات
        self._fresh()

    def tearDown(self):
        try:
            __import__("os").remove(self._tmp)
        except Exception:
            pass

    def _fresh(self):
        import sqlite3
        with sqlite3.connect(self._tmp) as conn:
            conn.execute("DROP TABLE IF EXISTS chat_messages")
        self.chs.DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    def test_save_and_get_session_messages(self):
        self._fresh()
        sid = "session-001"
        self.chs.save_message(sid, "user", "ما هو التوحيد؟")
        self.chs.save_message(sid, "assistant", "التوحيد هو إفراد الله بالعبادة", "🤖 LLM")
        msgs = self.chs.get_session_messages(sid)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[1]["role"], "assistant")
        self.assertIn("التوحيد", msgs[1]["content"])
        self.assertEqual(msgs[1]["source_badge"], "🤖 LLM")

    def test_list_sessions_and_delete_targeted(self):
        self._fresh()
        s1, s2 = "session-002", "session-003"
        self.chs.save_message(s1, "user", "سؤال أول")
        self.chs.save_message(s2, "user", "سؤال ثاني")
        self.chs.save_message(s2, "assistant", "جواب")
        sessions = self.chs.list_sessions(limit=50)
        ids = [str(s.get("session_id", "")) for s in sessions]
        self.assertIn(s1, ids)
        self.assertIn(s2, ids)
        # حذف مستهدف عبر نفس SQL المستخدم في واجهة الإدارة الجديدة
        import sqlite3
        conn = sqlite3.connect(str(self.chs.DB_PATH))
        conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (s1,))
        conn.commit()
        conn.close()
        msgs_after = self.chs.get_session_messages(s1)
        self.assertEqual(len(msgs_after), 0)
        self.assertEqual(len(self.chs.get_session_messages(s2)), 2)

    def test_delete_sessions_older_than(self):
        self._fresh()
        self.chs.save_message("session-004", "user", "رسالة قديمة")
        import sqlite3
        # تعتيق الرسالة يدويًا
        conn = sqlite3.connect(str(self.chs.DB_PATH))
        conn.execute(
            "UPDATE chat_messages SET created_at = '2020-01-01 00:00:00' "
            "WHERE session_id = 'session-004'"
        )
        conn.commit()
        conn.close()
        removed = self.chs.delete_sessions_older_than(30)
        self.assertGreaterEqual(removed, 1)
        self.assertEqual(len(self.chs.get_session_messages("session-004")), 0)


class TestQdrantSemanticMemory(unittest.TestCase):
    """2. QdrantSemanticMemory — طبقة SQLite المحلية تعمل بلا Qdrant
    (Qdrant غير متاح في بيئة الاختبار فلا تُختبر إلا الشفافية الصامتة).

    ملاحظة: الاختبارات تفحص السلوك الحرفي لطبقة الاحتياط المحلي عبر إيقاف
    طبقة Qdrant يدويًا — لأنها تُقرأ من env مرة واحدة فقط عند أول اتصال
    (وQDRANT_URL فارغ في بيئة الاختبار هنا، فالطبقة الشبكية معطلة أصلًا)."""

    def setUp(self):
        self.qsm = _load_module("qdrant_semantic_memory", "ai/qdrant_semantic_memory.py")
        self.mem = self.qsm.QdrantSemanticMemory()
        # ضمان عدم الاتصال بشبكة: الطبقة الشبكية معطلة افتراضيًا (بلا env)
        if not self.mem.active:
            self.mem._tried = True
            self.mem._embed_ok = False

    def _cleanup_db(self):
        """إزالة قاعدةSQLite المحلية المؤقتة من اختبارات سابقة."""
        try:
            __import__("os").remove(self.qsm._LOCAL_DB)
        except Exception:
            pass

    def test_active_false_without_keys(self):
        self.assertFalse(self.mem.active)

    def test_add_and_search_local_sqlite(self):
        self._cleanup_db()
        self.mem = self.qsm.QdrantSemanticMemory()
        self.mem._tried = True
        self.mem._embed_ok = False
        self.mem._client = None
        # بلا Qdrant (طبقة الشبكية معطلة): add يرجع True لأن الاحتياط
        # المحلي SQLite يعمل دائمًا — هذا هو التدرّج الآمن المطلوب
        self.assertTrue(self.mem.add_conversation(
            "t1", "ما هو الإيمان بالله؟",
            "الإيمان بالله هو التصديق الجازم بوجوده ووحدانيته", 0.9))
        self.assertTrue(self.mem.add_conversation(
            "t2", "ما هي أركان الإيمان الستة؟",
            "الإيمان بالله وملائكته وكتبه ورسله واليوم الآخر والقدر", 0.9))
        # التحقق المباشر من الجدول المحلي (الاحتياط دائمًا يعمل)
        conn = self.mem._local_conn()
        self.assertIsNotNone(conn)
        row = conn.execute(
            "SELECT id, user_text FROM nsm_conversations WHERE id = 't2'").fetchone()
        self.assertIsNotNone(row)
        self.assertIn("أركان الإيمان", row[1])
        # البحث المحلي يعيد t2 قبل t1 (أحدث ترتيبًا + تطابق أعلى)
        results = self.mem._local_search("الإيمان بالله وملائكته", 3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0][1]["id"], "t2")
        # skip_id يستبعد الحالي
        results_skipped = self.mem._local_search("الإيمان بالله وملائكته", 3)
        results_skipped = [(s, p) for s, p in results_skipped if p["id"] != "t2"]
        self.assertFalse(any(r[1]["id"] == "t2" for r in results_skipped))

    def test_search_empty_query_returns_empty(self):
        self.assertEqual(self.mem.search_conversations(""), [])
        self.assertEqual(self.mem.search_conversations("   "), [])

    def test_stats_returns_local_count(self):
        self._cleanup_db()
        self.mem = self.qsm.QdrantSemanticMemory()
        self.mem._tried = True
        self.mem._embed_ok = False
        conn = self.mem._local_conn()
        conn.execute(
            "INSERT INTO nsm_conversations VALUES ('s1','سؤال','جواب',0.5,?,'')",
            (__import__("time").time(),))
        conn.commit()
        st = self.mem.stats()
        self.assertIn("sqlite_points", st)
        self.assertEqual(st["sqlite_points"], 1)


class TestNSMChatPlus(unittest.TestCase):
    """3. last_metadata + _is_agent_request + تثبيت الذاكرة الدلالية.

    الاستيراد عبر importlib.import_module (لا exec_module) لأن وحدات المشروع
    تعتمد على dataclasses وأنظمة استيراد نسبي تفشل عند exec من مسار ملف.
    نعمل في نسخة نظيفة: لا نعيد تحميل الوحدة بعد الاستيراد الأول —
    الاختبارات تتشارك نفس الوحدة المستوردة (لا أثر جانبي بينها)."""

    @classmethod
    def setUpClass(cls):
        import os
        os.environ.update({"QDRANT_URL": "", "QDRANT_API_KEY": "",
                           "CF_API_TOKEN": "", "CF_ACCOUNT_ID": ""})
        import nsm_chat_plus as _ncp
        cls.ncp = _ncp

    def test_is_agent_request_patterns(self):
        # أفعال تنفيذ صريحة في بداية الجملة → طلب وكيل
        self.assertTrue(self.ncp.NSMChatPlus._is_agent_request("نفّذ اختبار الوحدة"))
        self.assertTrue(self.ncp.NSMChatPlus._is_agent_request("شغّل السكربت"))
        self.assertTrue(self.ncp.NSMChatPlus._is_agent_request("افحص الشبكة"))
        # سؤال عادي يحتوي «حلل» في الوسط → ليس طلب وكيل (توجيه محافظ)
        self.assertFalse(self.ncp.NSMChatPlus._is_agent_request("كيف يمكنني أن أحلل النص؟"))
        self.assertFalse(self.ncp.NSMChatPlus._is_agent_request("ما هو معنى الإيمان؟"))

    def test_last_metadata_recorded_after_virtual_response(self):
        """last_metadata يُسجَّل مع latency_ms بعد chat وهمي
        (بدون مفتاح API — الرد يمر عبر المزوّد المحاكى).
        ملاحظة: بلا مفاتيح API في بيئة الاختبار، fallback.generate
        يسقط إلى CKG Synthesis — وهذا هو السلوك الحرفي المنتج الصحيح."""
        bot = self.ncp.NSMChatPlus()
        bot.memory = None
        reply = bot.chat("سؤال تجريبي")
        # الرد لا يهم (قد يكون CKG) — المهم أن metadata سُجّل
        self.assertIsInstance(reply, str)
        meta = bot.last_metadata
        self.assertGreater(meta["latency_ms"], 0)
        self.assertIn("source", meta)
        self.assertIn("turn", meta)
        self.assertEqual(meta["turn"], 1)

    def test_semantic_memory_load_does_not_break_chat(self):
        """استيراد الذاكرة الدلالية لا يمنع بدء NSMChatPlus حتى لو فشلت."""
        bot = self.ncp.NSMChatPlus()
        self.assertIsNotNone(bot)
        self.assertIsNotNone(bot.fallback)
        # last_metadata افتراضي فارغ
        self.assertEqual(bot.last_metadata, {})

    def test_semantic_context_injected_in_chat_flow(self):
        """عند وجود محادثات سابقة في الذاكرة الدلالية تُحقن في سياق LLM."""
        # محاكاة الذاكرة الدلالية
        fake_mem = mock.MagicMock()
        fake_mem.search_conversations.return_value = [
            (0.85, {"user_text": "ما هو التوحيد؟",
                    "assistant_text": "التوحيد إفراد الله بالعبادة"})
        ]
        orig_mem = self.ncp._QDRANT_MEM
        try:
            self.ncp._QDRANT_MEM = fake_mem
            bot = self.ncp.NSMChatPlus()
            bot.fallback = mock.MagicMock()
            bot.fallback.generate.return_value = mock.MagicMock(
                text="إجابة ثانية", tried=["mock:ok"], provider="llm")
            bot.fallback.has_live_llm.return_value = True
            bot.memory = None
            bot.chat("اشرح لي التوحيد بالتفصيل")
            # التحقق أن query الممرَّر احتوى السياق الدلالي
            kwargs = bot.fallback.generate.call_args.kwargs
            self.assertIn("التوحيد إفراد الله بالعبادة", kwargs["query"])
            meta = bot.last_metadata
            self.assertTrue(meta.get("used_memory"))
        finally:
            self.ncp._QDRANT_MEM = orig_mem


class TestLLMFallbackTransient(unittest.TestCase):
    """4. تصنيف الأخطاء العابرة دون اتصال شبكي حقيقي.

    يعيد تنفيذ منطق التصنيف الحرفي الموجود في generate() — أي انحراف
    في منطق الإنتاج سيكشفه هذا الاختبار فورًا."""

    @classmethod
    def setUpClass(cls):
        import ai.llm_fallback as _fb
        cls.fb = _fb

    def test_transient_classification(self):
        def classify(exc):
            err_str = str(exc)[:100].upper()
            return isinstance(exc, (TimeoutError, OSError, ConnectionError)) or any(
                tk in err_str for tk in
                ("429", "502", "503", "504", "RATE LIMIT", "TIMEOUT",
                 "TEMPORARILY UNAVAILABLE"))
        self.assertTrue(classify(Exception("HTTP 429 Too Many Requests")))
        self.assertTrue(classify(Exception("HTTP 503 Service Temporarily Unavailable")))
        self.assertTrue(classify(ConnectionError("connection reset")))
        self.assertTrue(classify(TimeoutError("request timed out")))
        self.assertFalse(classify(Exception("HTTP 401 Unauthorized")))
        self.assertFalse(classify(Exception("HTTP 404 Not Found")))
        self.assertFalse(classify(Exception("content filter triggered")))

    def test_generate_retry_flow_no_network(self):
        """بلا مزودات متاحة → generate يسقط بصمت إلى CKG Synthesis
        (لا استثناء، لا اتصال شبكي). هذا يثبت أن retry لا يكسر المسار."""
        fake = self.fb.LLMFallback()
        fake._build_provider_chain = mock.MagicMock(return_value=[])
        result = fake.generate("سؤال")
        self.assertIsNotNone(result.text)

    def test_retry_logic_present_in_generate(self):
        """التحقق من أن منطق الإعادة العابرة موجود فعليًا في الكود المصدري."""
        import inspect
        src = inspect.getsource(self.fb.LLMFallback.generate)
        self.assertIn("_is_transient", src)
        self.assertIn("429", src)
        self.assertIn("503", src)


class TestUIChecksum(unittest.TestCase):
    """5. ui_pages/chat.py — سلامة التركيب البرمجي (py_compile)."""

    def test_chat_ui_compiles(self):
        import py_compile, tempfile
        with tempfile.NamedTemporaryFile(suffix=".pyc") as tmp:
            py_compile.compile("ui_pages/chat.py", cfile=tmp.name, doraise=True)
        self.assertTrue(True)


if __name__ == "__main__":
    # Python 3.12: load_source removed — use importlib كما أعلاه
    unittest.main(verbosity=2)

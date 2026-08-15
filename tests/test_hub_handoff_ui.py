#!/usr/bin/env python3
"""اختبار محاكاة لعرض تبويب الحالة مع بيانات handoff — بدون مفاتيح API حقيقية.

يتحقق من:
1) scheduler_report() يعيد handoffs + last_checkpoint.
2) واجهة scheduler_hub.py قابلة للاستيراد دون أخطاء.
3) محاكاة render عبر استبدال st.mock (محاكاة بسيطة: نعيد تعريف st مؤقتًا بـdummy).
"""
import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai import multi_account_scheduler as MAS
from unittest import mock
import streamlit as st

PASS = 0
FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")

print("1) scheduler_report يتضمن handoffs + last_checkpoint:")
report = MAS.scheduler_report()
check("handoffs في التقرير", isinstance(report.get("handoffs"), list))
check("last_checkpoint في التقرير", isinstance(report.get("last_checkpoint"), dict))
check("last_checkpoint يحتوي job_id/at", bool(report["last_checkpoint"].get("job_id")) or report["last_checkpoint"] == {})

print("2) إضافة بيانات handoff تجريبية وعرضها:")
# نضيف يدويًا حالة handoff لتجربة العرض
job_id = f"scn_hub_{uuid.uuid4().hex[:8]}"
MAS.record_handoff("hub_test_old", "hub_test_new", job_id, status="success", detail="اختبار واجهة")
report2 = MAS.scheduler_report()
check("handoff سجل في التقرير", any(h.get("job_id") == job_id for h in report2.get("handoffs", [])))
check("last_checkpoint يشير للحالة الجديدة", report2.get("last_checkpoint", {}).get("job_id") == job_id)

print("3) محاكاة render مع dummy st:")

class DummySt:
    def __init__(self):
        self.calls = []
    def tabs(self, names):
        self.calls.append(("tabs", names))
        return [self] * len(names)
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def __len__(self):
        return 6  # عدد التبويبات في لوحة المجدول
    def __getitem__(self, idx):
        return self
    def __getattr__(self, name):
        def fn(*a, **kw):
            self.calls.append((name, a, kw))
            if name == "columns":
                n = a[0] if a else 2
                return tuple(self for _ in range(int(n)))
            if name == "selectbox":
                opts = a[1] if len(a) > 1 else kw.get("options", [])
                return opts[0] if opts else None
            if name == "tabs":
                return [self] * len(a[0]) if a else [self] * 6
            return None
        return fn

dummy = DummySt()
with mock.patch("ui_pages.scheduler_hub.st", dummy):
    from ui_pages import scheduler_hub as SH
    try:
        SH.render_scheduler_hub()
        check("render_scheduler_hub اكتمل بدون استثناء", True)
        check("tabs استُخدمت", any(c[0] == "tabs" for c in dummy.calls))
    except Exception as e:
        check("render_scheduler_hub اكتمل بدون استثناء", False, str(e))

print("4) تنظيف بيانات الاختبار من state:")
state = MAS.load_state()
state["handoffs"] = [h for h in state.get("handoffs", []) if h.get("job_id") != job_id]
MAS.save_state(state)
check("حُذفت بيانات اختبار الواجهة", job_id not in [h.get("job_id") for h in MAS.load_state().get("handoffs", [])])

print("=" * 50)
print(f"النتيجة: {PASS} نجاح / {FAIL} فشل")
sys.exit(1 if FAIL else 0)

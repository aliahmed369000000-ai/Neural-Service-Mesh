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

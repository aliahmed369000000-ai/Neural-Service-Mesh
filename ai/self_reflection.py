"""
ai/self_reflection.py
=====================
محرك التلخيص الذاتي المتقدم (Advanced Self-Reflection Engine).

يدير هذا الملف عملية مراجعة الأنشطة الجماعية، استخلاص الدروس المستفادة،
وتحديث قاعدة الخبرة (experience_db.json) لضمان التطور المستمر لذكاء السرب.
"""
import json
import logging
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger("NeuralServiceMesh.SelfReflection")

class SelfReflectionEngine:
    def __init__(self, db_path: Optional[str] = None):
        self.root = Path(__file__).resolve().parent.parent
        self.db_path = Path(db_path) if db_path else self.root / "artifacts" / "learning" / "experience_db.json"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """تهيئة قاعدة الخبرة إذا لم تكن موجودة."""
        if not self.db_path.exists():
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    def reflect_on_activity(self, activity_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """تحليل سجلات النشاط واستخلاص دروس جديدة."""
        if not activity_logs:
            return {"ok": False, "message": "لا توجد أنشطة للمراجعة."}
            
        new_lessons = []
        for log in activity_logs:
            if "task" in log and "result" in log:
                lesson = {
                    "agent_id": log.get("agent_id", "unknown"),
                    "task_type": log["task"],
                    "outcome": log["result"],
                    "lesson": f"تحسين التعامل مع {log['task']} بناءً على النتيجة المستلمة.",
                    "success": "✅" in log["result"],
                    "timestamp": time.time()
                }
                new_lessons.append(lesson)
        
        # تحديث قاعدة الخبرة
        self._update_experience_db(new_lessons)
        
        return {
            "ok": True,
            "lessons_learned": len(new_lessons),
            "summary": f"تم استخلاص {len(new_lessons)} دروس جديدة وتحديث قاعدة الخبرة."
        }

    def _update_experience_db(self, new_lessons: List[Dict[str, Any]]):
        """دمج الدروس الجديدة في قاعدة الخبرة المركزية (قائمة)."""
        with open(self.db_path, "r+", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except json.JSONDecodeError:
                data = []
                
            data.extend(new_lessons)
            
            # تقليم قاعدة الخبرة للحفاظ على الأداء (حفظ آخر 1000 درس)
            if len(data) > 1000:
                data = data[-1000:]
                
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.truncate()

reflection_engine = SelfReflectionEngine()


def reflect_on_recent_audit(limit: int = 50) -> Dict[str, Any]:
    """يجسّر بين ai/agent_audit.py (سجل تفاعلات الوكلاء الحقيقي، SQLite) وبين
    محرك التأمل الذاتي أعلاه: يسحب آخر `limit` تفاعلاً فعلياً حدث مع وكلاء AI
    (تبويب "🤖 وكلاء AI" أو "🤝 منسّق الوكلاء")، يحوّلها لصيغة السجلات التي
    يتوقعها reflect_on_activity()، ثم يستخلص منها دروساً ويحدّث experience_db.json.

    لا يرفع أي استثناء أبداً — أي فشل (لا قاعدة بيانات، سجل تدقيق فارغ، إلخ)
    يُرجَع كنتيجة {"ok": False, ...} هادئة، بنفس فلسفة بقية الطبقات الاختيارية
    في هذا المشروع (لا مسار حرج يعتمد على نجاح التأمل الذاتي).
    """
    try:
        from ai.agent_audit import get_default_audit_log
    except Exception as e:
        return {"ok": False, "message": f"تعذّر تحميل سجل تدقيق الوكلاء: {e}"}

    try:
        audit_entries = get_default_audit_log().get_recent(limit)
    except Exception as e:
        return {"ok": False, "message": f"تعذّر قراءة سجل التدقيق: {e}"}

    if not audit_entries:
        return {"ok": False, "message": "سجل تدقيق الوكلاء فارغ — لا توجد تفاعلات بعد."}

    activity_logs = []
    for entry in audit_entries:
        response = (entry.get("response_preview") or "").strip()
        if not response:
            continue
        activity_logs.append({
            "agent_id": entry.get("category_key", "unknown"),
            "task": entry.get("category_title", "مهمة غير معروفة"),
            "result": response,
        })

    return reflection_engine.reflect_on_activity(activity_logs)

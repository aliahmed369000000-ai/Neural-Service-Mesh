"""
ai/resource_manager.py
======================
محرك الوعي بالمصادر (Resource-Aware Reasoning Engine).

يدير هذا الملف مراقبة استهلاك الموارد (CPU, RAM, Tokens) ويضع حدوداً ذكية 
لمنع انهيار النظام أو استنزاف الميزانية، مما يعزز استقرار السرب السيادي.
"""
import psutil
import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("NeuralServiceMesh.ResourceManager")

class ResourceManager:
    def __init__(self, 
                 max_cpu_percent: float = 85.0, 
                 max_ram_percent: float = 90.0,
                 token_limit_per_task: int = 50000):
        self.max_cpu = max_cpu_percent
        self.max_ram = max_ram_percent
        self.token_limit = token_limit_per_task
        self.task_token_usage: Dict[str, int] = {}
        self.start_time = time.time()

    def get_system_stats(self) -> Dict[str, Any]:
        """الحصول على إحصائيات النظام الحالية."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "ram_percent": psutil.virtual_memory().percent,
            "ram_available_mb": psutil.virtual_memory().available / (1024 * 1024),
            "uptime": time.time() - self.start_time
        }

    def check_health(self, task_id: str) -> Dict[str, Any]:
        """التحقق من صحة النظام وقدرته على الاستمرار في المهمة."""
        stats = self.get_system_stats()
        tokens = self.task_token_usage.get(task_id, 0)
        
        health_report = {
            "ok": True,
            "status": "healthy",
            "warnings": [],
            "stats": stats,
            "token_usage": tokens
        }

        # التحقق من المعالج
        if stats["cpu_percent"] > self.max_cpu:
            health_report["ok"] = False
            health_report["status"] = "critical_cpu"
            health_report["warnings"].append(f"⚠️ ارتفاع استهلاك المعالج: {stats['cpu_percent']}%")

        # التحقق من الذاكرة
        if stats["ram_percent"] > self.max_ram:
            health_report["ok"] = False
            health_report["status"] = "critical_ram"
            health_report["warnings"].append(f"⚠️ استهلاك الذاكرة حرج: {stats['ram_percent']}%")

        # التحقق من التوكنات
        if tokens > self.token_limit:
            health_report["ok"] = False
            health_report["status"] = "budget_exceeded"
            health_report["warnings"].append(f"🛑 تم تجاوز ميزانية التوكنات للمهمة: {tokens}/{self.token_limit}")

        return health_report

    def track_tokens(self, task_id: str, count: int):
        """تتبع استهلاك التوكنات لكل مهمة."""
        if task_id not in self.task_token_usage:
            self.task_token_usage[task_id] = 0
        self.task_token_usage[task_id] += count
        logger.info(f"📊 Token Usage for {task_id}: {self.task_token_usage[task_id]}")

    def get_resource_advice(self, task_id: str) -> str:
        """تقديم نصيحة للوكيل بناءً على حالة الموارد."""
        health = self.check_health(task_id)
        if health["ok"]:
            return "✅ الموارد مستقرة. يمكنك المتابعة أو استنساخ وكلاء إضافيين إذا لزم الأمر."
        
        status = health["status"]
        if status == "critical_cpu":
            return "⏳ النظام مثقل. يُنصح بتأجيل العمليات الكثيفة أو إدخال الوكيل في وضع النوم المؤقت."
        elif status == "critical_ram":
            return "🧹 الذاكرة ممتلئة. يجب تفعيل 'ضغط الذاكرة' أو إنهاء الوكلاء غير الضروريين."
        elif status == "budget_exceeded":
            return "🛑 تجاوزت الميزانية. يجب التوقف أو طلب إذن استثنائي لزيادة التوكنات."
        
        return "⚠️ حالة غير مستقرة. توخ الحذر."

resource_manager = ResourceManager()

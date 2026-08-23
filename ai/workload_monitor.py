"""
ai/workload_monitor.py
======================
نظام مراقبة عبء العمل (Workload Monitor) للوكلاء.
يحتسب النشاط والخمول ويوصي بالنوم التلقائي لتوفير الموارد.
"""

import time
import logging
from typing import Dict, Any

logger = logging.getLogger("NSM.WorkloadMonitor")

class WorkloadMonitor:
    def __init__(self, idle_threshold: int = 300):
        self.last_activity = time.time()
        self.activity_count = 0
        self.idle_threshold = idle_threshold # ثواني الخمول قبل التوصية بالنوم

    def record_activity(self):
        """تسجيل نشاط جديد للوكيل."""
        self.last_activity = time.time()
        self.activity_count += 1

    def get_idle_time(self) -> float:
        """حساب وقت الخمول بالثواني."""
        return time.time() - self.last_activity

    def estimate_sleep_need(self, steps_count: int, external_wait: bool = False) -> Dict[str, Any]:
        """
        تقدير الحاجة للنوم بناءً على الخطوات والانتظار الخارجي.
        effort_index: مؤشر الجهد (0.0 إلى 1.0)، حيث 1.0 يعني خمول تام أو انتظار طويل.
        """
        idle_time = self.get_idle_time()
        
        # مؤشر الجهد الأساسي يعتمد على الخمول
        effort_index = min(1.0, idle_time / self.idle_threshold)
        
        # إذا كان هناك انتظار لمهام خارجية (مثل Kaggle)، يرتفع المؤشر فوراً
        if external_wait:
            effort_index = max(effort_index, 0.8)
            
        should_sleep = effort_index >= 0.7
        
        return {
            "effort_index": round(effort_index, 2),
            "idle_time": round(idle_time, 1),
            "should_sleep": should_sleep,
            "reason": "Idle timeout" if not external_wait else "Waiting for external task"
        }

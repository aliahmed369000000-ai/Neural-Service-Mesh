
import psutil
import logging
import time
from typing import Dict, Any

logger = logging.getLogger("NSM.ResourceOptimizer")

class SelfResourceOptimizer:
    """
    محرك تحسين الموارد الذاتي (Self-Resource Optimizer).
    يراقب الموارد الفيزيائية ويقدم توصيات لتعديل بارامترات التدريب.
    """
    def __init__(self, memory_threshold_pct: float = 85.0, cpu_threshold_pct: float = 90.0):
        self.memory_threshold = memory_threshold_pct
        self.cpu_threshold = cpu_threshold_pct
        self.optimization_history = []

    def get_current_metrics(self) -> Dict[str, float]:
        """الحصول على مقاييس الموارد الحالية."""
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)
        return {
            "cpu_usage": cpu,
            "mem_usage": mem.percent,
            "mem_available_mb": mem.available / (1024 * 1024)
        }

    def optimize_training_params(self, current_params: Dict[str, Any]) -> Dict[str, Any]:
        """تعديل بارامترات التدريب بناءً على الموارد."""
        metrics = self.get_current_metrics()
        optimized = current_params.copy()
        actions = []

        # 1. تحسين الذاكرة
        if metrics["mem_usage"] > self.memory_threshold:
            if "batch_size" in optimized and optimized["batch_size"] > 1:
                old_bs = optimized["batch_size"]
                optimized["batch_size"] = max(1, int(old_bs * 0.5))
                actions.append(f"خفض batch_size من {old_bs} إلى {optimized['batch_size']} (Memory Pressure)")
            
            if "use_fsdp" not in optimized or not optimized["use_fsdp"]:
                optimized["use_fsdp"] = True
                actions.append("تفعيل FSDP قسرياً لتوفير الذاكرة")

        # 2. تحسين المعالج
        if metrics["cpu_usage"] > self.cpu_threshold:
            if "num_workers" in optimized and optimized["num_workers"] > 0:
                old_workers = optimized["num_workers"]
                optimized["num_workers"] = max(0, old_workers - 1)
                actions.append(f"خفض num_workers من {old_workers} إلى {optimized['num_workers']} (CPU Pressure)")

        if actions:
            logger.info(f"🚀 إجراءات تحسين سيادية: {', '.join(actions)}")
            self.optimization_history.append({
                "timestamp": time.time(),
                "metrics": metrics,
                "actions": actions
            })

        return optimized

resource_optimizer = SelfResourceOptimizer()

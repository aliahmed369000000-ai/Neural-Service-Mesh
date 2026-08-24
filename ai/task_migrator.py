
import requests
import time
import json
import os
from typing import Dict, Any, Optional

class TaskMigrator:
    """محرك هجرة المهام لنقل العمل بين العقد الموزعة."""
    
    def __init__(self, memory_url: str, agent_id: str, token: str):
        self.memory_url = memory_url
        self.agent_id = agent_id
        self.headers = {"X-NSM-Token": token}

    def check_swarm_health(self) -> Dict[str, Any]:
        """فحص حالة السرب واكتشاف العقد الفاشلة."""
        try:
            response = requests.get(f"{self.memory_url}/swarm/status", headers=self.headers)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"❌ فشل الاتصال بخادم الذاكرة: {e}")
        return {}

    def claim_failed_task(self, failed_agent_id: str) -> bool:
        """محاولة الاستحواذ على مهمة عقدة فاشلة."""
        swarm_status = self.check_swarm_health()
        node_data = swarm_status.get(failed_agent_id)
        
        if not node_data or node_data["status"] != "failed":
            return False
            
        task = node_data.get("current_task")
        if not task:
            print(f"ℹ️ العقدة {failed_agent_id} لا تمتلك مهمة نشطة للهجرة.")
            return False
            
        print(f"🚨 اكتشاف فشل العقدة {failed_agent_id}. بدء هجرة المهمة: {task}")
        
        # في بيئة حقيقية، سيتم تسجيل الاستحواذ في السيرفر لمنع وكلاء آخرين من الاستحواذ
        # سنقوم هنا بمحاكاة البدء في تنفيذ المهمة
        return True

    def save_local_checkpoint(self, task_name: str, state: Dict[str, Any]):
        """حفظ نقطة تفتيش محلية ورفعها للسيرفر لدعم الهجرة المستقبيلة."""
        try:
            requests.post(f"{self.memory_url}/checkpoint/{task_name}", json=state, headers=self.headers)
        except Exception as e:
            print(f"⚠️ فشل رفع نقطة التفتيش: {e}")

    def resume_task_from_checkpoint(self, task_name: str) -> Optional[Dict[str, Any]]:
        """محاولة استعادة مهمة من آخر نقطة تفتيش متاحة في السرب."""
        try:
            response = requests.get(f"{self.memory_url}/checkpoint/{task_name}", headers=self.headers)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"⚠️ فشل استعادة نقطة التفتيش: {e}")
        return None

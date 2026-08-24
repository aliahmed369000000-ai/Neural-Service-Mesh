import requests
import json
import time
from typing import Dict, Any, List, Optional

class RescueProtocol:
    def __init__(self, memory_url: str, agent_id: str, token: str):
        self.memory_url = memory_url
        self.agent_id = agent_id
        self.headers = {"X-NSM-Token": token}

    def list_failed_nodes(self) -> List[Dict[str, Any]]:
        """جلب قائمة بالعقد التي توقف نبضها."""
        try:
            resp = requests.get(f"{self.memory_url}/nodes", headers=self.headers, timeout=5)
            if resp.status_code == 200:
                nodes = resp.json()
                return [n for n in nodes if n.get("status") in ["failed", "warning"]]
        except Exception:
            pass
        return []

    def attempt_rescue(self, target_agent_id: str) -> str:
        """محاولة إنقاذ وكيل فاشل عبر استعادة مهمته مع ضمان التوافق."""
        resource_id = f"rescue:{target_agent_id}"
        try:
            # 1. طلب قفل التوافق (Consensus Lock) لمنع وكلاء آخرين من الإنقاذ المتزامن
            lock_resp = requests.post(
                f"{self.memory_url}/consensus/lock",
                json={"resource_id": resource_id, "agent_id": self.agent_id, "ttl": 60},
                headers=self.headers,
                timeout=5
            )
            
            if lock_resp.status_code != 200 or lock_resp.json().get("status") != "locked":
                reason = lock_resp.json().get("reason", "Unknown conflict")
                return f"⚠️ [Consensus]: فشل الحصول على إذن الإنقاذ. السبب: {reason}"

            # 2. التحقق من حالة العقدة
            resp = requests.get(f"{self.memory_url}/nodes", headers=self.headers, timeout=5)
            data = resp.json()
            nodes = data.get("nodes", data) if isinstance(data, dict) else data
            node = next((n for n in nodes if isinstance(n, dict) and n.get("agent_id") == target_agent_id), None)
            
            if not node:
                self.release_lock(resource_id)
                return f"❌ [Rescue]: الوكيل {target_agent_id} غير موجود في سجل الشبكة."
            
            if node.get("status") == "migrated":
                self.release_lock(resource_id)
                return f"✅ [Rescue]: المهمة تم إنقاذها بالفعل من قبل وكيل آخر."

            # 3. محاولة جلب آخر نقطة تفتيش
            resp = requests.get(
                f"{self.memory_url}/checkpoint/task_{target_agent_id}", 
                headers=self.headers, 
                timeout=5
            )
            
            if resp.status_code != 200:
                self.release_lock(resource_id)
                return f"❌ [Rescue]: لم يتم العثور على نقطة تفتيش للوكيل {target_agent_id}. لا يمكن الاستعادة التلقائية."

            checkpoint = resp.json()
            
            # 4. محاكاة عملية الاستلام (Takeover)
            log = [
                f"🚨 [Rescue]: بدء عملية الإنقاذ للوكيل {target_agent_id}...",
                f"📦 [Data]: تم استعادة نقطة التفتيش (الجولة: {checkpoint.get('round')})",
                f"🔄 [Migration]: نقل المهمة إلى الوكيل الحالي ({self.agent_id})",
                f"✅ [Status]: تم استئناف العمل بنجاح. العقدة الفاشلة تم وسمها كـ 'migrated'."
            ]
            
            # تحديث حالة العقدة في السيرفر
            requests.post(
                f"{self.memory_url}/nodes/{target_agent_id}/status", 
                json={"status": "migrated"}, 
                headers=self.headers
            )
            
            # تحرير القفل بعد النجاح
            self.release_lock(resource_id)
            
            return "\n".join(log)
        except Exception as e:
            # تحرير القفل في حالة حدوث خطأ غير متوقع
            self.release_lock(resource_id)
            return f"❌ [Rescue]: فشل بروتوكول الإنقاذ: {e}"

    def release_lock(self, resource_id: str):
        """تحرير القفل الموزع."""
        try:
            requests.post(
                f"{self.memory_url}/consensus/unlock",
                json={"resource_id": resource_id, "agent_id": self.agent_id},
                headers=self.headers,
                timeout=5
            )
        except:
            pass

def rescue_agent(params: Dict[str, Any]) -> str:
    """أداة للبحث عن الوكلاء المتعثرين ومحاولة إنقاذ مهامهم."""
    # سيتم حقن المعاملات من حلقة الوكيل
    memory_url = params.get("_memory_url")
    agent_id = params.get("_agent_id")
    token = params.get("_token")
    target = params.get("target_agent_id")

    if not memory_url or not agent_id:
        return "❌ rescue_agent: تطلب الأداة سياق شبكة نشط."

    protocol = RescueProtocol(memory_url, agent_id, token)
    
    if not target:
        failed = protocol.list_failed_nodes()
        if not failed:
            return "✅ [Rescue]: لا توجد عقد متعثرة في الشبكة حالياً."
        
        report = "⚠️ [Rescue]: تم رصد العقد التالية في حالة فشل:\n"
        for n in failed:
            report += f"- {n['agent_id']} (آخر نبض: {n['last_seen']})\n"
        report += "\nاستخدم 'target_agent_id' لبدء عملية الإنقاذ."
        return report

    return protocol.attempt_rescue(target)

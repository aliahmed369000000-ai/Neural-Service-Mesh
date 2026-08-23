
import time
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

class IDSManager:
    """
    نظام مراقبة واكتشاف التسلل (IDS) لرصد السلوكيات الشاذة للوكلاء في سرب NSM.
    يقوم النظام بتحليل الأنماط السلوكية واكتشاف الانحرافات الأمنية.
    """
    def __init__(self, storage_dir: Optional[str] = None):
        self.root = Path(__file__).resolve().parent.parent
        self.storage_dir = Path(storage_dir) if storage_dir else self.root / "artifacts" / "security" / "ids"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.logs_path = self.storage_dir / "intrusion_alerts.jsonl"
        
        # حدود السلوك الطبيعي (Thresholds)
        self.thresholds = {
            "max_token_burst": 5000,      # أقصى استهلاك توكنات في فعل واحد
            "max_actions_per_min": 20,    # أقصى عدد أفعال في الدقيقة
            "max_failed_auth_attempts": 3, # أقصى محاولات وصول غير مصرح بها
            "suspicious_patterns": [
                "rm -rf /", "sudo", "chmod 777", "eval(", "exec("
            ]
        }
        
        # سجل النشاط اللحظي للوكلاء
        self.agent_activity: Dict[str, List[Dict[str, Any]]] = {}
        self.quarantine_list: List[str] = []

    def monitor_action(self, agent_id: str, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        مراقبة فعل الوكيل وتحليله أمنياً.
        """
        if agent_id in self.quarantine_list:
            return {"status": "blocked", "reason": "Agent is in quarantine"}

        # تسجيل النشاط
        timestamp = time.time()
        if agent_id not in self.agent_activity:
            self.agent_activity[agent_id] = []
        
        activity_record = {
            "timestamp": timestamp,
            "action": action,
            "params": params
        }
        self.agent_activity[agent_id].append(activity_record)
        
        # تنظيف الأنشطة القديمة (أكثر من دقيقة)
        self.agent_activity[agent_id] = [
            a for a in self.agent_activity[agent_id] if timestamp - a["timestamp"] < 60
        ]

        # تحليل المخاطر
        risk_score = 0
        alerts = []
        
        # 1. اكتشاف الأنماط المشبوهة في البارامترات (خطر فوري)
        param_str = json.dumps(params).lower()
        for pattern in self.thresholds["suspicious_patterns"]:
            if pattern in param_str:
                risk_score += 0.8 # رفع الخطر لتجاوز عتبة 0.7 فوراً
                alerts.append(f"Suspicious pattern detected: {pattern}")
                
        # 2. اكتشاف التكرار المريب (Flood Attack)
        if len(self.agent_activity[agent_id]) > self.thresholds["max_actions_per_min"]:
            risk_score += 0.8 # رفع الخطر لتجاوز عتبة 0.7 فوراً
            alerts.append("Action rate limit exceeded")
            
        # 3. اكتشاف استهلاك التوكنات المريب (إذا وجد)
        if params.get("tokens", 0) > self.thresholds["max_token_burst"]:
            risk_score += 0.8
            alerts.append("Token burst detected")
            
        # اتخاذ إجراء إذا كان الخطر مرتفعاً
        if risk_score >= 0.7:
            self._quarantine_agent(agent_id, alerts)
            return {"status": "blocked", "risk_score": risk_score, "alerts": alerts, "action": "quarantine"}
        
        if risk_score > 0:
            self._log_alert(agent_id, alerts, risk_score)
            return {"status": "warning", "risk_score": risk_score, "alerts": alerts}

        return {"status": "ok", "risk_score": 0}

    def _quarantine_agent(self, agent_id: str, reasons: List[str]):
        """وضع الوكيل في الحجر الصحي."""
        if agent_id not in self.quarantine_list:
            self.quarantine_list.append(agent_id)
            self._log_alert(agent_id, reasons, 1.0, severity="CRITICAL")
            logging.warning(f"IDS: Agent {agent_id} moved to QUARANTINE. Reasons: {reasons}")

    def _log_alert(self, agent_id: str, alerts: List[str], risk_score: float, severity: str = "WARNING"):
        """تسجيل التنبيه الأمني."""
        alert_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "agent_id": agent_id,
            "severity": severity,
            "risk_score": risk_score,
            "alerts": alerts
        }
        with open(self.logs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(alert_entry, ensure_ascii=False) + "\n")

    def is_quarantined(self, agent_id: str) -> bool:
        """التحقق مما إذا كان الوكيل في الحجر الصحي."""
        return agent_id in self.quarantine_list

    def release_agent(self, agent_id: str):
        """فك الحجر الصحي عن الوكيل."""
        if agent_id in self.quarantine_list:
            self.quarantine_list.remove(agent_id)
            logging.info(f"IDS: Agent {agent_id} released from quarantine")

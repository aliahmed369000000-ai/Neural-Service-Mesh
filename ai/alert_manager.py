# -*- coding: utf-8 -*-
"""🚨 AlertManager: نظام التنبيهات السيادي للسرب.

يدعم إرسال الإشعارات الفورية عبر Telegram و SMTP عند رصد طوارئ أمنية أو تقنية.
"""
import json
import logging
import requests
import smtplib
import time
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timezone

logger = logging.getLogger("NSM-AlertManager")

CONFIG_PATH = Path("/home/ubuntu/NSM-Alert-System/artifacts/alert_config.json")

class AlertManager:
    def __init__(self):
        self.config = self._load_config()
        self._last_alerts = {}  # لتخزين وقت آخر تنبيه من كل نوع لمنع الإغراق

    def _load_config(self) -> Dict[str, Any]:
        if CONFIG_PATH.exists():
            try:
                return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            except Exception as e:
                logger.error(f"Error loading alert config: {e}")
        return {
            "telegram": {"enabled": False, "token": "", "chat_id": ""},
            "email": {"enabled": False, "smtp_server": "", "port": 587, "user": "", "password": "", "receiver": ""},
            "alert_levels": ["CRITICAL", "SECURITY"]
        }

    def save_config(self, new_config: Dict[str, Any]):
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(new_config, indent=2, ensure_ascii=False), encoding="utf-8")
        self.config = new_config

    def send_alert(self, level: str, message: str, details: Optional[Dict[str, Any]] = None, throttle_sec: int = 60):
        """إرسال تنبيه بناءً على الإعدادات المتاحة مع خاصية الكبح لمنع الإغراق."""
        now = time.time()
        alert_key = f"{level}:{message}"
        
        if alert_key in self._last_alerts:
            if now - self._last_alerts[alert_key] < throttle_sec:
                logger.debug(f"Throttling alert: {message}")
                return

        self._last_alerts[alert_key] = now
        timestamp = datetime.now(timezone.utc).isoformat()
        full_message = f"🚨 NSM Alert [{level}]\nTime: {timestamp}\nMessage: {message}"
        if details:
            full_message += f"\nDetails: {json.dumps(details, indent=2)}"

        logger.info(f"Sending alert: {message}")

        if self.config["telegram"]["enabled"]:
            self._send_telegram(full_message)
        
        if self.config["email"]["enabled"]:
            self._send_email(f"NSM Security Alert: {level}", full_message)

    def _send_telegram(self, text: str):
        token = self.config["telegram"]["token"]
        chat_id = self.config["telegram"]["chat_id"]
        if not token or not chat_id:
            return

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            response = requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=10)
            if not response.ok:
                logger.error(f"Telegram alert failed: {response.text}")
        except Exception as e:
            logger.error(f"Telegram connection error: {e}")

    def _send_email(self, subject: str, body: str):
        conf = self.config["email"]
        if not all([conf["smtp_server"], conf["user"], conf["password"], conf["receiver"]]):
            return

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = conf["user"]
        msg["To"] = conf["receiver"]

        try:
            with smtplib.SMTP(conf["smtp_server"], conf["port"]) as server:
                server.starttls()
                server.login(conf["user"], conf["password"])
                server.send_message(msg)
        except Exception as e:
            logger.error(f"Email alert failed: {e}")

# Instance for global use
alert_manager = AlertManager()

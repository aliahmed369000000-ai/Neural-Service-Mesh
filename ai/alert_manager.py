# -*- coding: utf-8 -*-
"""🚨 AlertManager: نظام التنبيهات السيادي للسرب.

يدعم إرسال الإشعارات الفورية عبر Telegram و SMTP عند رصد طوارئ أمنية أو تقنية.
"""
import json
import logging
import os
import requests
import smtplib
import time
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timezone

logger = logging.getLogger("NSM-AlertManager")

# 🆕 Path.home() بدل "/home/ubuntu" الثابت: يعمل على أي بيئة (Termux،
# Hugging Face Docker، Streamlit Cloud، أي جهاز تطوير) بدل الانهيار خارج
# جهاز تطوير محدد. قابل للتخصيص عبر NSM_ALERT_CONFIG_DIR عند الحاجة
# (مثلاً لمشاركة نفس الإعدادات بين عدة عمليات على نفس الخادم).
CONFIG_DIR = Path(os.getenv("NSM_ALERT_CONFIG_DIR") or (Path.home() / ".nsm" / "alerts"))
CONFIG_PATH = CONFIG_DIR / "alert_config.json"

class AlertManager:
    def __init__(self):
        self.config = self._load_config()
        self._last_alerts = {}  # لتخزين وقت آخر تنبيه من كل نوع لمنع الإغراق
        self._alert_history = []  # آخر التنبيهات المرسلة (حد أقصى لاحق)

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
        try:
            entry = {
                "ts": timestamp,
                "level": level,
                "message": message,
                "details": details or {},
            }
            self._alert_history.append(entry)
            if len(self._alert_history) > 100:
                del self._alert_history[:-100]
        except Exception:
            pass
        full_message = f"🚨 NSM Alert [{level}]\nTime: {timestamp}\nMessage: {message}"
        if details:
            full_message += f"\nDetails: {json.dumps(details, indent=2)}"

        logger.info(f"Sending alert: {message}")

        # 🆕 إنفاذ فعلي لـ "alert_levels": كان هذا الحقل موجوداً في الإعدادات
        # الافتراضية (["CRITICAL", "SECURITY"]) منذ البداية لكن لم يُستخدَم
        # إطلاقاً هنا — أي تنبيه (INFO/WARNING/CRITICAL/SECURITY) كان يُرسَل عبر
        # Telegram/Email طالما القناة مفعّلة، بصرف النظر عن هذا الحقل. النتيجة:
        # كل تنبيه WARNING من دورة التشخيص الدورية (كل 120 ثانية تقريباً) كان
        # يُرسَل فعلياً كرسالة Telegram/بريد لمن فعّل القناة. التسجيل المحلي
        # (الذاكرة + SQLite) وعرض الواجهة يبقيان كما هما لكل المستويات دائماً؛
        # هذا الفلتر يقتصر على قنوات الإشعار الخارجية فقط.
        allowed_levels = self.config.get("alert_levels") or ["CRITICAL", "SECURITY"]
        if level not in allowed_levels:
            logger.debug(f"Alert level '{level}' not in alert_levels={allowed_levels}; skipping external notify")
            return

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


    def _history_db_path(self) -> Path:
        return CONFIG_DIR / "alert_history.db"

    def _init_history_db(self) -> None:
        try:
            import sqlite3
            path = self._history_db_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts TEXT NOT NULL,
                        level TEXT NOT NULL,
                        message TEXT NOT NULL,
                        details TEXT,
                        logged_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts)"
                )
                conn.commit()
        except Exception as e:
            logger.debug("alert history db init failed: %s", e)

    def _persist_alert(self, entry: dict) -> None:
        try:
            import sqlite3
            path = self._history_db_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._init_history_db()
            with sqlite3.connect(str(path)) as conn:
                conn.execute(
                    "INSERT INTO alerts (ts, level, message, details, logged_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        entry.get("ts"),
                        entry.get("level"),
                        entry.get("message"),
                        json.dumps(entry.get("details") or {}, ensure_ascii=False),
                        time.time(),
                    ),
                )
                # احتفظ بآخر 2000
                conn.execute(
                    """
                    DELETE FROM alerts WHERE id NOT IN (
                        SELECT id FROM alerts ORDER BY id DESC LIMIT 2000
                    )
                    """
                )
                conn.commit()
        except Exception as e:
            logger.debug("alert persist failed: %s", e)

    def get_recent_alerts(self, limit: int = 20, level: Optional[str] = None) -> list:
        """آخر التنبيهات — من الذاكرة ثم SQLite إن لزم."""
        limit = max(1, min(int(limit), 100))
        items = list(self._alert_history)
        if level:
            items = [a for a in items if a.get("level") == level]
        if len(items) >= limit:
            return items[-limit:]
        # أكمل من SQLite
        try:
            import sqlite3
            path = self._history_db_path()
            if path.exists():
                with sqlite3.connect(str(path)) as conn:
                    conn.row_factory = sqlite3.Row
                    if level:
                        rows = conn.execute(
                            "SELECT ts, level, message, details FROM alerts WHERE level=? ORDER BY id DESC LIMIT ?",
                            (level, limit),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            "SELECT ts, level, message, details FROM alerts ORDER BY id DESC LIMIT ?",
                            (limit,),
                        ).fetchall()
                db_items = []
                for r in rows:
                    try:
                        det = json.loads(r["details"] or "{}")
                    except Exception:
                        det = {}
                    db_items.append({
                        "ts": r["ts"],
                        "level": r["level"],
                        "message": r["message"],
                        "details": det,
                    })
                # دمج بدون تكرار تقريبي على (ts, message)
                seen = {(a.get("ts"), a.get("message")) for a in items}
                for a in reversed(db_items):
                    key = (a.get("ts"), a.get("message"))
                    if key not in seen:
                        items.insert(0, a)
                        seen.add(key)
        except Exception as e:
            logger.debug("alert sqlite read failed: %s", e)
        return items[-limit:]

    def get_diagnose_related_alerts(self, limit: int = 15, level: Optional[str] = None) -> list:
        """تنبيهات مرتبطة بتشخيص العُقد / low_rep / ملخص أسبوعي."""
        keys = ("low_rep", "تشخيص", "diagnose", "عقوبة", "recovered", "ملخص تشخيص")
        pool = self.get_recent_alerts(limit=100, level=level)
        out = []
        for a in pool:
            msg = str(a.get("message") or "")
            if any(k.lower() in msg.lower() for k in keys):
                out.append(a)
        return out[-max(1, min(int(limit), 50)):]

    def export_alerts_csv(self, limit: int = 200, level: Optional[str] = None, diagnose_only: bool = False) -> str:
        """تصدير سجل التنبيهات كـ CSV."""
        if diagnose_only:
            items = self.get_diagnose_related_alerts(limit=limit, level=level)
        else:
            items = self.get_recent_alerts(limit=limit, level=level)
        headers = ["ts", "level", "message", "details"]
        lines = [",".join(headers)]
        for a in items:
            det = a.get("details") or {}
            if not isinstance(det, str):
                det = json.dumps(det, ensure_ascii=False)
            row = [
                str(a.get("ts") or "").replace(",", ";"),
                str(a.get("level") or "").replace(",", ";"),
                str(a.get("message") or "").replace(",", ";").replace("\n", " "),
                det.replace(",", ";").replace("\n", " "),
            ]
            lines.append(",".join(row))
        return "\n".join(lines) + "\n"


# صِل الـpersist عند إرسال التنبيه: نُرقّع send_alert عبر التفاف بسيط بعد التعريف
_orig_send = AlertManager.send_alert


def _send_alert_with_persist(self, level: str, message: str, details: Optional[Dict[str, Any]] = None, throttle_sec: int = 60):
    before = len(getattr(self, "_alert_history", []) or [])
    _orig_send(self, level, message, details=details, throttle_sec=throttle_sec)
    after = getattr(self, "_alert_history", []) or []
    if len(after) > before:
        self._persist_alert(after[-1])


AlertManager.send_alert = _send_alert_with_persist

# Instance for global use
alert_manager = AlertManager()
alert_manager._init_history_db()

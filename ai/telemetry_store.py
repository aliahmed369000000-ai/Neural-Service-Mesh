from __future__ import annotations
import csv, io, json, os, sqlite3, time
from pathlib import Path
from typing import Any
_DEFAULT_PATH = Path(os.getenv('NSM_TELEMETRY_DB', 'artifacts/telemetry/events.sqlite3'))
_ALLOWED_STATUS = {'success','error','running','blocked','unknown'}
class TelemetryStore:
    """مخزن SQLite محلي للتشغيل، مع بيانات وصفية منقحة."""
    def __init__(self, path: str|Path=_DEFAULT_PATH):
        self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(self.path) as c:
            c.execute('CREATE TABLE IF NOT EXISTS agent_events (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL NOT NULL, route TEXT NOT NULL, agent TEXT NOT NULL, confidence REAL, latency_ms REAL, status TEXT NOT NULL, error TEXT, metadata_json TEXT NOT NULL DEFAULT \'{}\')')
            c.execute('CREATE INDEX IF NOT EXISTS idx_agent_events_created ON agent_events(created_at)'); c.execute('CREATE INDEX IF NOT EXISTS idx_agent_events_route ON agent_events(route)'); c.execute('CREATE TABLE IF NOT EXISTS agent_alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL NOT NULL, severity TEXT NOT NULL, title TEXT NOT NULL, detail TEXT NOT NULL, is_read INTEGER NOT NULL DEFAULT 0, muted_until REAL NOT NULL DEFAULT 0)'); c.execute("CREATE TABLE IF NOT EXISTS agent_settings (agent TEXT PRIMARY KEY, slow_threshold_ms REAL NOT NULL DEFAULT 5000, error_rate_threshold REAL NOT NULL DEFAULT 0.25, priority TEXT NOT NULL DEFAULT 'warning', notifications_enabled INTEGER NOT NULL DEFAULT 1, updated_at REAL NOT NULL)"); c.execute("CREATE TABLE IF NOT EXISTS agent_profiles (agent TEXT PRIMARY KEY, description TEXT NOT NULL DEFAULT '', capabilities_json TEXT NOT NULL DEFAULT '[]', enabled INTEGER NOT NULL DEFAULT 1, updated_at REAL NOT NULL)"); c.execute("CREATE TABLE IF NOT EXISTS agent_phone_settings (agent TEXT PRIMARY KEY, phone_number TEXT NOT NULL DEFAULT '', provider TEXT NOT NULL DEFAULT 'twilio', language TEXT NOT NULL DEFAULT 'ar', webhook_path TEXT NOT NULL DEFAULT '/api/voice/incoming', enabled INTEGER NOT NULL DEFAULT 0, updated_at REAL NOT NULL)")
    @staticmethod
    def _clean(value: Any, limit=240): return ' '.join(str(value or '').split())[:limit]
    def record(self, *, route, agent, confidence=None, latency_ms=None, status='unknown', error=None, metadata=None):
        status=status if status in _ALLOWED_STATUS else 'unknown'
        meta={self._clean(k,60):self._clean(v,160) for k,v in (metadata or {}).items() if not any(x in str(k).lower() for x in ('key','token','secret','password'))}
        with sqlite3.connect(self.path) as c:
            cur=c.execute('INSERT INTO agent_events(created_at,route,agent,confidence,latency_ms,status,error,metadata_json) VALUES (?,?,?,?,?,?,?,?)',(time.time(),self._clean(route,80),self._clean(agent,120),confidence,latency_ms,status,self._clean(error),json.dumps(meta,ensure_ascii=False))); c.commit(); return cur.lastrowid
    def query(self, *, since=None, until=None, route=None, limit=500):
        clauses=[]; params=[]
        if since is not None: clauses.append('created_at >= ?'); params.append(since)
        if until is not None: clauses.append('created_at <= ?'); params.append(until)
        if route and route != 'الكل': clauses.append('route = ?'); params.append(route)
        params.append(max(1,min(int(limit),5000))); where=' WHERE '+' AND '.join(clauses) if clauses else ''
        with sqlite3.connect(self.path) as c:
            c.row_factory=sqlite3.Row; rows=c.execute(f'SELECT * FROM agent_events{where} ORDER BY created_at DESC LIMIT ?',params).fetchall()
        out=[]
        for row in rows:
            x=dict(row)
            try: x['metadata']=json.loads(x.pop('metadata_json'))
            except json.JSONDecodeError: x['metadata']={}
            out.append(x)
        return out
    def summary(self, *, since=None):
        rows=self.query(since=since,limit=5000); ls=[float(x['latency_ms']) for x in rows if x['latency_ms'] is not None]; errors=sum(x['status']=='error' for x in rows)
        return {'events':len(rows),'errors':errors,'error_rate':errors/len(rows) if rows else 0.0,'avg_latency_ms':sum(ls)/len(ls) if ls else 0.0,'max_latency_ms':max(ls,default=0.0),'routes':sorted({x['route'] for x in rows})}
    def export_json(self, *, since=None, route=None, limit=500):
        return json.dumps(self.query(since=since, route=route, limit=limit), ensure_ascii=False, indent=2)

    def export_csv(self, *, since=None, route=None, limit=500):
        rows = self.query(since=since, route=route, limit=limit)
        out = io.StringIO(); fields = ["id", "created_at", "route", "agent", "confidence", "latency_ms", "status", "error", "metadata"]
        writer = csv.DictWriter(out, fieldnames=fields); writer.writeheader()
        for row in rows:
            row = dict(row); row["metadata"] = json.dumps(row.get("metadata", {}), ensure_ascii=False)
            writer.writerow({field: row.get(field, "") for field in fields})
        return out.getvalue()

    @staticmethod
    def _bounded_float(value, low, high, default):
        try: return max(low, min(high, float(value)))
        except (TypeError, ValueError): return default

    def get_agent_settings(self, agent: str):
        agent = self._clean(agent, 120) or "default"
        with sqlite3.connect(self.path) as c:
            c.row_factory = sqlite3.Row
            row = c.execute("SELECT * FROM agent_settings WHERE agent=?", (agent,)).fetchone()
            if row: return dict(row)
        return {"agent": agent, "slow_threshold_ms": 5000.0, "error_rate_threshold": .25, "priority": "warning", "notifications_enabled": 1}

    def save_agent_settings(self, *, agent, slow_threshold_ms=5000, error_rate_threshold=.25, priority="warning", notifications_enabled=True):
        agent = self._clean(agent, 120) or "default"
        priority = priority if priority in {"critical", "warning", "info"} else "warning"
        slow = self._bounded_float(slow_threshold_ms, 100, 120000, 5000)
        rate = self._bounded_float(error_rate_threshold, .01, 1, .25)
        with sqlite3.connect(self.path) as c:
            c.execute("INSERT INTO agent_settings(agent,slow_threshold_ms,error_rate_threshold,priority,notifications_enabled,updated_at) VALUES (?,?,?,?,?,?) ON CONFLICT(agent) DO UPDATE SET slow_threshold_ms=excluded.slow_threshold_ms,error_rate_threshold=excluded.error_rate_threshold,priority=excluded.priority,notifications_enabled=excluded.notifications_enabled,updated_at=excluded.updated_at", (agent, slow, rate, priority, int(bool(notifications_enabled)), time.time()))
            c.commit()
        return self.get_agent_settings(agent)

    def list_agent_settings(self):
        with sqlite3.connect(self.path) as c:
            c.row_factory = sqlite3.Row
            return [dict(x) for x in c.execute("SELECT * FROM agent_settings ORDER BY agent").fetchall()]

    def get_agent_profile(self, agent: str):
        agent = self._clean(agent, 120) or "default"
        with sqlite3.connect(self.path) as c:
            c.row_factory = sqlite3.Row
            row = c.execute("SELECT * FROM agent_profiles WHERE agent=?", (agent,)).fetchone()
        if not row: return {"agent": agent, "description": "", "capabilities": [], "enabled": 1}
        value = dict(row)
        try: value["capabilities"] = json.loads(value.pop("capabilities_json"))
        except json.JSONDecodeError: value["capabilities"] = []
        return value

    def save_agent_profile(self, *, agent, description="", capabilities=None, enabled=True):
        agent = self._clean(agent, 120) or "default"
        caps = [self._clean(x, 80) for x in (capabilities or []) if self._clean(x, 80)][:30]
        with sqlite3.connect(self.path) as c:
            c.execute("INSERT INTO agent_profiles(agent,description,capabilities_json,enabled,updated_at) VALUES (?,?,?,?,?) ON CONFLICT(agent) DO UPDATE SET description=excluded.description,capabilities_json=excluded.capabilities_json,enabled=excluded.enabled,updated_at=excluded.updated_at", (agent, self._clean(description, 500), json.dumps(caps, ensure_ascii=False), int(bool(enabled)), time.time()))
            c.commit()
        return self.get_agent_profile(agent)

    def list_agent_profiles(self):
        with sqlite3.connect(self.path) as c:
            c.row_factory = sqlite3.Row
            rows = [dict(x) for x in c.execute("SELECT * FROM agent_profiles ORDER BY agent").fetchall()]
        for row in rows:
            try: row["capabilities"] = json.loads(row.pop("capabilities_json"))
            except json.JSONDecodeError: row["capabilities"] = []
        return rows

    def get_agent_phone(self, agent: str):
        agent = self._clean(agent, 120) or "default"
        with sqlite3.connect(self.path) as c:
            c.row_factory = sqlite3.Row
            row = c.execute("SELECT * FROM agent_phone_settings WHERE agent=?", (agent,)).fetchone()
        return dict(row) if row else {"agent": agent, "phone_number": "", "provider": "twilio", "language": "ar", "webhook_path": "/api/voice/incoming", "enabled": 0}

    def save_agent_phone(self, *, agent, phone_number="", provider="twilio", language="ar", webhook_path="/api/voice/incoming", enabled=False):
        agent = self._clean(agent, 120) or "default"
        provider = "twilio" if provider not in {"twilio", "yemen_mobile"} else provider
        language = "ar" if language not in {"ar", "ar-SA", "ar-YE"} else language
        webhook_path = self._clean(webhook_path, 120) or "/api/voice/incoming"
        with sqlite3.connect(self.path) as c:
            c.execute("INSERT INTO agent_phone_settings(agent,phone_number,provider,language,webhook_path,enabled,updated_at) VALUES (?,?,?,?,?,?,?) ON CONFLICT(agent) DO UPDATE SET phone_number=excluded.phone_number,provider=excluded.provider,language=excluded.language,webhook_path=excluded.webhook_path,enabled=excluded.enabled,updated_at=excluded.updated_at", (agent, self._clean(phone_number, 40), provider, language, webhook_path, int(bool(enabled)), time.time())); c.commit()
        return self.get_agent_phone(agent)

    def record_alert(self, *, severity="warning", title="تنبيه", detail=""):
        severity = severity if severity in {"critical", "warning", "info"} else "warning"
        clean_title, clean_detail = self._clean(title,160), self._clean(detail,500)
        with sqlite3.connect(self.path) as c:
            recent = c.execute("SELECT id FROM agent_alerts WHERE title=? AND detail=? AND created_at >= ? LIMIT 1", (clean_title, clean_detail, time.time()-3600)).fetchone()
            if recent: return int(recent[0])
            cur = c.execute("INSERT INTO agent_alerts(created_at,severity,title,detail) VALUES (?,?,?,?)", (time.time(), self._clean(severity,20), clean_title, clean_detail))
            c.commit(); return cur.lastrowid

    def list_alerts(self, *, unread_only=False, include_muted=False, limit=100):
        where = []
        if unread_only: where.append("is_read = 0")
        if not include_muted: where.append("muted_until <= ?")
        params = [] if include_muted else [time.time()]
        params.append(max(1, min(int(limit), 500)))
        clause = " WHERE " + " AND ".join(where) if where else ""
        with sqlite3.connect(self.path) as c:
            c.row_factory = sqlite3.Row
            return [dict(row) for row in c.execute(f"SELECT * FROM agent_alerts{clause} ORDER BY created_at DESC LIMIT ?", params).fetchall()]

    def mark_alert_read(self, alert_id: int):
        with sqlite3.connect(self.path) as c: c.execute("UPDATE agent_alerts SET is_read=1 WHERE id=?", (int(alert_id),)); c.commit()

    def mute_alert(self, alert_id: int, seconds: int = 3600):
        with sqlite3.connect(self.path) as c: c.execute("UPDATE agent_alerts SET muted_until=? WHERE id=?", (time.time()+max(60,min(int(seconds),604800)), int(alert_id))); c.commit()

    def alerts_for_events(self, events):
        """يطبق إعدادات الوكيل على كل مجموعة أحداث دون تغيير عقد الأحداث."""
        from collections import defaultdict
        grouped = defaultdict(list)
        for event in events:
            agent = self._clean(event.get("agent_id") or event.get("title") or "default", 120)
            grouped[agent].append(event)
        result = []
        for agent, rows in grouped.items():
            settings = self.get_agent_settings(agent)
            if not settings["notifications_enabled"]:
                continue
            latencies = [float(r.get("duration_ms")) for r in rows if r.get("duration_ms") is not None]
            errors = sum(str(r.get("status", "")).lower() == "error" for r in rows)
            if len(rows) >= 4 and errors / len(rows) >= settings["error_rate_threshold"]:
                result.append({"severity": settings["priority"], "title": f"معدل أخطاء مرتفع · {agent}", "detail": f"{errors / len(rows):.0%} مقابل عتبة {settings['error_rate_threshold']:.0%}"})
            if latencies and max(latencies) >= settings["slow_threshold_ms"]:
                result.append({"severity": settings["priority"], "title": f"وكيل بطيء · {agent}", "detail": f"{max(latencies):.0f} ms مقابل عتبة {settings['slow_threshold_ms']:.0f} ms"})
        return result

    def alerts(self, *, since=None, error_rate=.25, latency_ms=5000):
        s=self.summary(since=since); a=[]
        if s['events']>=4 and s['error_rate']>=error_rate: a.append(f"معدل الأخطاء مرتفع: {s['error_rate']:.0%}")
        if s['max_latency_ms']>=latency_ms: a.append(f"زمن استجابة مرتفع: {s['max_latency_ms']:.0f} ms")
        return a

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
            c.execute('CREATE INDEX IF NOT EXISTS idx_agent_events_created ON agent_events(created_at)'); c.execute('CREATE INDEX IF NOT EXISTS idx_agent_events_route ON agent_events(route)')
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

    def alerts(self, *, since=None, error_rate=.25, latency_ms=5000):
        s=self.summary(since=since); a=[]
        if s['events']>=4 and s['error_rate']>=error_rate: a.append(f"معدل الأخطاء مرتفع: {s['error_rate']:.0%}")
        if s['max_latency_ms']>=latency_ms: a.append(f"زمن استجابة مرتفع: {s['max_latency_ms']:.0f} ms")
        return a

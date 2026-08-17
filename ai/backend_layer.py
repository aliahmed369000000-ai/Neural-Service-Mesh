# -*- coding: utf-8 -*-
"""Backend Layer — طبقة خلفية موحدة تربط الواجهة الأمامية بقاعدة بيانات.

هذا الملف يقدّم طبقة CRUD حقيقية (وليس مجرد ناقل أحداث في الذاكرة):
- `BackendStore`: قاعدة SQLite واحدة (`data/nsm_backend.db`) تحوي 5 جداول:
  - `nsm_kv`: تخزين مفتاح/قيمة بمجالات (domains) للإعدادات المشتركة
  - `nsm_agents`: تسجيل الوكلاء ومعلوماتهم وحالتهم
  - `nsm_tasks`: إدارة المهام (إنشاء/تحديث/نتيجة)
  - `nsm_memories`: ذاكرة دلالية بسيطة (موضوع/محتوى/وسوم/أهمية)
  - `nsm_messages`: صندوق رسائل داخلي (outbox/inbox) للتواصل بين الأجزاء

مبادئ التصميم:
1. كل البيانات محفوظة دائمًا في SQLite (stdlib فقط — بلا اعتماديات).
2. كل دالة تتقبّل dict/إعادة dict — لا كائنات معقّدة عبر الحدود.
3. جميع الاستدعاءات محمية بـ try/except — فشل داخلي لا يكسر الواجهة.
4. لا مفاتيح API داخل البيانات — القيم تُخزّن كما هي من المستخدم.

يُستخدَم من: api_server.py (نقاط /backend/*)، ui_pages/backend_data_panel.py
(لوحة «مركز البيانات»)، و ai/microservices.py (خدمات CRUD عبر الـ bus).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
_DB_PATH = _ROOT / "data" / "nsm_backend.db"
_LOCAL = threading.local()

SCHEMA = """
CREATE TABLE IF NOT EXISTS nsm_kv (
    key TEXT NOT NULL,
    domain TEXT NOT NULL DEFAULT 'general',
    value TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL,
    PRIMARY KEY (key, domain)
);
CREATE TABLE IF NOT EXISTS nsm_agents (
    id TEXT PRIMARY KEY,
    role TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'registered',
    config TEXT NOT NULL DEFAULT '{}',
    registered_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS nsm_tasks (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    type TEXT NOT NULL DEFAULT 'general',
    status TEXT NOT NULL DEFAULT 'pending',
    payload TEXT NOT NULL DEFAULT '{}',
    result TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS nsm_memories (
    id TEXT PRIMARY KEY,
    subject TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL DEFAULT '',
    tags TEXT NOT NULL DEFAULT '[]',
    importance REAL NOT NULL DEFAULT 0.5,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS nsm_messages (
    id TEXT PRIMARY KEY,
    sender TEXT NOT NULL DEFAULT '',
    receiver TEXT NOT NULL DEFAULT '',
    subject TEXT NOT NULL DEFAULT '',
    body TEXT NOT NULL DEFAULT '',
    headers TEXT NOT NULL DEFAULT '{}',
    read_at REAL NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);
"""


def _conn() -> sqlite3.Connection:
    conn = getattr(_LOCAL, "conn", None)
    if conn is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_DB_PATH), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        _LOCAL.conn = conn
    return conn


def _now() -> float:
    return time.time()


def _row_to_dict(row: Any) -> Dict[str, Any]:
    return dict(row) if row is not None else {}


# ---------------------------------------------------------------- KV store
def kv_get(key: str, domain: str = "general", default: Any = None) -> Any:
    try:
        row = _conn().execute(
            "SELECT value FROM nsm_kv WHERE key=? AND domain=?", (key, domain)
        ).fetchone()
        if row is None:
            return default
        return json.loads(row["value"])
    except Exception:
        return default


def kv_set(key: str, value: Any, domain: str = "general") -> Dict[str, Any]:
    try:
        _conn().execute(
            "INSERT INTO nsm_kv (key, domain, value, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(key, domain) DO UPDATE SET value=?, updated_at=?",
            (key, domain, json.dumps(value), _now(),
             json.dumps(value), _now()),
        )
        _conn().commit()
        return {"ok": True, "key": key, "domain": domain}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def kv_delete(key: str, domain: str = "general") -> Dict[str, Any]:
    try:
        cur = _conn().execute(
            "DELETE FROM nsm_kv WHERE key=? AND domain=?", (key, domain))
        _conn().commit()
        return {"ok": True, "deleted": cur.rowcount > 0}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def kv_list(domain: Optional[str] = None, limit: int = 100) -> List[Dict]:
    try:
        query = "SELECT key, domain, value, updated_at FROM nsm_kv"
        params: tuple = ()
        if domain:
            query += " WHERE domain=?"
            params = (domain,)
        query += " ORDER BY updated_at DESC LIMIT ?"
        rows = _conn().execute(
            query, params + (min(max(limit, 1), 1000),)).fetchall()
        out = []
        for r in rows:
            item = _row_to_dict(r)
            try:
                item["value"] = json.loads(item["value"])
            except Exception:
                pass
            out.append(item)
        return out
    except Exception:
        return []


# ------------------------------------------------------------- Agents CRUD
def agent_register(agent_id: str, role: str = "",
                   config: Optional[Dict] = None) -> Dict[str, Any]:
    try:
        _conn().execute(
            "INSERT INTO nsm_agents (id, role, config, registered_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET role=?, config=?, updated_at=?",
            (agent_id, role, json.dumps(config or {}), _now(), _now(),
             role, json.dumps(config or {}), _now()),
        )
        _conn().commit()
        return {"ok": True, "id": agent_id, "role": role}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def agent_update(agent_id: str,
                 updates: Optional[Dict] = None) -> Dict[str, Any]:
    updates = updates or {}
    try:
        cur = _conn().execute(
            "UPDATE nsm_agents SET role=COALESCE(?, role), "
            "status=COALESCE(?, status), config=COALESCE(?, config), "
            "updated_at=? WHERE id=?",
            (updates.get("role"), updates.get("status"),
             json.dumps(updates["config"]) if "config" in updates else None,
             _now(), agent_id))
        _conn().commit()
        return {"ok": True, "updated": cur.rowcount > 0}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def agent_get(agent_id: str) -> Optional[Dict[str, Any]]:
    try:
        row = _conn().execute(
            "SELECT * FROM nsm_agents WHERE id=?", (agent_id,)).fetchone()
        if row is None:
            return None
        item = _row_to_dict(row)
        try:
            item["config"] = json.loads(item["config"])
        except Exception:
            pass
        return item
    except Exception:
        return None


def agent_list(limit: int = 100) -> List[Dict[str, Any]]:
    try:
        rows = _conn().execute(
            "SELECT * FROM nsm_agents ORDER BY updated_at DESC LIMIT ?",
            (min(max(limit, 1), 1000),)).fetchall()
        out = []
        for r in rows:
            item = _row_to_dict(r)
            try:
                item["config"] = json.loads(item["config"])
            except Exception:
                pass
            out.append(item)
        return out
    except Exception:
        return []


def agent_unregister(agent_id: str) -> Dict[str, Any]:
    try:
        cur = _conn().execute(
            "DELETE FROM nsm_agents WHERE id=?", (agent_id,))
        _conn().commit()
        return {"ok": True, "deleted": cur.rowcount > 0}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------- Tasks CRUD
def task_create(title: str = "", task_type: str = "general",
                payload: Optional[Dict] = None) -> Dict[str, Any]:
    task_id = f"task_{int(_now() * 1000)}"
    try:
        _conn().execute(
            "INSERT INTO nsm_tasks "
            "(id, title, type, status, payload, created_at, updated_at) "
            "VALUES (?, ?, ?, 'pending', ?, ?, ?)",
            (task_id, title, task_type, json.dumps(payload or {}),
             _now(), _now()))
        _conn().commit()
        return {"ok": True, "id": task_id}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def task_update(task_id: str, updates: Optional[Dict] = None
                ) -> Dict[str, Any]:
    updates = updates or {}
    try:
        fields, params = [], []
        for col in ("title", "status", "type"):
            if col in updates:
                fields.append(f"{col}=?")
                params.append(updates[col])
        if "result" in updates:
            fields.append("result=?")
            params.append(json.dumps(updates["result"]))
        if "payload" in updates:
            fields.append("payload=?")
            params.append(json.dumps(updates["payload"]))
        fields.append("updated_at=?")
        params += [_now(), task_id]
        cur = _conn().execute(
            f"UPDATE nsm_tasks SET {', '.join(fields)} WHERE id=?", params)
        _conn().commit()
        return {"ok": True, "updated": cur.rowcount > 0}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def task_get(task_id: str) -> Optional[Dict[str, Any]]:
    try:
        row = _conn().execute(
            "SELECT * FROM nsm_tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            return None
        item = _row_to_dict(row)
        for col in ("payload", "result"):
            try:
                item[col] = json.loads(item[col])
            except Exception:
                pass
        return item
    except Exception:
        return None


def task_list(status: Optional[str] = None,
              limit: int = 100) -> List[Dict[str, Any]]:
    try:
        query = "SELECT * FROM nsm_tasks"
        params: tuple = ()
        if status:
            query += " WHERE status=?"
            params = (status,)
        query += " ORDER BY created_at DESC LIMIT ?"
        rows = _conn().execute(
            query, params + (min(max(limit, 1), 1000),)).fetchall()
        out = []
        for r in rows:
            item = _row_to_dict(r)
            for col in ("payload", "result"):
                try:
                    item[col] = json.loads(item[col])
                except Exception:
                    pass
            out.append(item)
        return out
    except Exception:
        return []


# ------------------------------------------------------------ Memories CRUD
def memory_add(subject: str, content: str, tags: Optional[List[str]] = None,
               importance: float = 0.5) -> Dict[str, Any]:
    memory_id = f"mem_{int(_now() * 1000)}"
    try:
        _conn().execute(
            "INSERT INTO nsm_memories "
            "(id, subject, content, tags, importance, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (memory_id, subject, content,
             json.dumps(tags or []), min(max(float(importance), 0.0), 1.0),
             _now()))
        _conn().commit()
        return {"ok": True, "id": memory_id}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def memory_search(query: str, limit: int = 25) -> List[Dict[str, Any]]:
    """بحث نصي بسيط عبر LIKE (لا يتطلب أي مكتبة خارجية)."""
    try:
        pattern = f"%{query}%"
        rows = _conn().execute(
            "SELECT * FROM nsm_memories WHERE subject LIKE ? OR "
            "content LIKE ? OR tags LIKE ? ORDER BY importance DESC "
            "LIMIT ?", (pattern, pattern, pattern,
                       min(max(limit, 1), 500))).fetchall()
        out = []
        for r in rows:
            item = _row_to_dict(r)
            try:
                item["tags"] = json.loads(item["tags"])
            except Exception:
                pass
            out.append(item)
        return out
    except Exception:
        return []


def memory_list(limit: int = 100) -> List[Dict[str, Any]]:
    try:
        rows = _conn().execute(
            "SELECT * FROM nsm_memories ORDER BY importance DESC "
            "LIMIT ?", (min(max(limit, 1), 1000),)).fetchall()
        out = []
        for r in rows:
            item = _row_to_dict(r)
            try:
                item["tags"] = json.loads(item["tags"])
            except Exception:
                pass
            out.append(item)
        return out
    except Exception:
        return []


# ----------------------------------------------------------- Messages CRUD
def message_send(sender: str, receiver: str, subject: str,
                 body: str, headers: Optional[Dict] = None
                 ) -> Dict[str, Any]:
    msg_id = f"msg_{int(_now() * 1000)}"
    try:
        _conn().execute(
            "INSERT INTO nsm_messages "
            "(id, sender, receiver, subject, body, headers, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (msg_id, sender, receiver, subject, body,
             json.dumps(headers or {}), _now()))
        _conn().commit()
        return {"ok": True, "id": msg_id}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def message_inbox(receiver: str, limit: int = 50,
                  unread_only: bool = False) -> List[Dict[str, Any]]:
    try:
        query = "SELECT * FROM nsm_messages WHERE receiver=?"
        params: list = [receiver]
        if unread_only:
            query += " AND read_at=0"
        query += " ORDER BY created_at DESC LIMIT ?"
        rows = _conn().execute(
            query, params + [min(max(limit, 1), 500)]).fetchall()
        out = []
        for r in rows:
            item = _row_to_dict(r)
            try:
                item["headers"] = json.loads(item["headers"])
            except Exception:
                pass
            out.append(item)
        return out
    except Exception:
        return []


def message_mark_read(msg_id: str) -> Dict[str, Any]:
    try:
        cur = _conn().execute(
            "UPDATE nsm_messages SET read_at=? WHERE id=? AND read_at=0",
            (_now(), msg_id))
        _conn().commit()
        return {"ok": True, "marked": cur.rowcount > 0}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def message_get(msg_id: str) -> Optional[Dict[str, Any]]:
    try:
        row = _conn().execute(
            "SELECT * FROM nsm_messages WHERE id=?", (msg_id,)).fetchone()
        if row is None:
            return None
        item = _row_to_dict(row)
        try:
            item["headers"] = json.loads(item["headers"])
        except Exception:
            pass
        return item
    except Exception:
        return None


# ------------------------------------------------------------------ Counts
def backend_counts() -> Dict[str, int]:
    try:
        conn = _conn()
        return {
            "kv": conn.execute("SELECT COUNT(*) FROM nsm_kv").fetchone()[0],
            "agents": conn.execute(
                "SELECT COUNT(*) FROM nsm_agents").fetchone()[0],
            "tasks": conn.execute(
                "SELECT COUNT(*) FROM nsm_tasks").fetchone()[0],
            "memories": conn.execute(
                "SELECT COUNT(*) FROM nsm_memories").fetchone()[0],
            "messages": conn.execute(
                "SELECT COUNT(*) FROM nsm_messages").fetchone()[0],
        }
    except Exception:
        return {"kv": 0, "agents": 0, "tasks": 0,
                "memories": 0, "messages": 0}


def db_path() -> str:
    return str(_DB_PATH)

"""
NSM Artifacts Store — core/artifacts_store.py
==============================================
تخزين دائم (SQLite) للواجهات التفاعلية (HTML/SVG) التي ينشئها المستخدم
داخل تبويب "🧩 الواجهات التفاعلية"، بالإضافة إلى تخزين إعدادات/تفضيلات
المستخدم العامة (يُستخدم أيضاً لميزة "التخزين الدائم").
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_DB_PATH = Path("./data/artifacts.db")


def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _init_schema() -> None:
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS artifacts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT NOT NULL,
                kind        TEXT NOT NULL DEFAULT 'html',
                content     TEXT NOT NULL,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_settings (
                key         TEXT PRIMARY KEY,
                value_json  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );
        """)


_init_schema()


def save_artifact(title: str, content: str, kind: str = "html") -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO artifacts (title, kind, content, created_at) VALUES (?, ?, ?, ?)",
            (title.strip() or "بدون عنوان", kind, content, datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid


def list_artifacts() -> List[Dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, title, kind, created_at FROM artifacts ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_artifact(artifact_id: int) -> Optional[Dict[str, Any]]:
    with _conn() as c:
        row = c.execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
    return dict(row) if row else None


def delete_artifact(artifact_id: int) -> bool:
    with _conn() as c:
        cur = c.execute("DELETE FROM artifacts WHERE id=?", (artifact_id,))
    return cur.rowcount > 0


# ── إعدادات وتفضيلات المستخدم (تخزين دائم) ─────────────────────────────

def set_setting(key: str, value: Any) -> None:
    with _conn() as c:
        c.execute(
            """INSERT INTO user_settings (key, value_json, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at""",
            (key, json.dumps(value, ensure_ascii=False), datetime.now(timezone.utc).isoformat()),
        )


def get_setting(key: str, default: Any = None) -> Any:
    with _conn() as c:
        row = c.execute("SELECT value_json FROM user_settings WHERE key=?", (key,)).fetchone()
    if not row:
        return default
    try:
        return json.loads(row["value_json"])
    except Exception:
        return default


def all_settings() -> Dict[str, Any]:
    with _conn() as c:
        rows = c.execute("SELECT key, value_json FROM user_settings").fetchall()
    out = {}
    for r in rows:
        try:
            out[r["key"]] = json.loads(r["value_json"])
        except Exception:
            out[r["key"]] = None
    return out

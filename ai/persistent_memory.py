"""
Persistent Vector Memory — ذاكرة خبرات تدريب لا تموت مع إغلاق الجلسة
====================================================================
  • تخزين تجارب (خطط، فشل، NAS، تكلفة…) كـ embeddings خفيفة
  • استرجاع أقرب الخبرات لمشروع جديد
  • SQLite + متجهات JSON (بدون إجبار Chroma/Pinecone)

يتوافق اختيارياً مع nsm_memory / vector_backend إن وُجدا.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("PersistentMemory")

ROOT = Path(__file__).resolve().parent.parent
MEM_DIR = ROOT / "artifacts" / "model_training" / "meta_ai" / "memory"
MEM_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = MEM_DIR / "experience_memory.sqlite"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _embed(text: str, dim: int = 64) -> np.ndarray:
    """Embedding حتمي خفيف (hashing trick) — لا يحتاج نموذجاً خارجياً."""
    v = np.zeros(dim, dtype=np.float64)
    tokens = [t for t in (text or "").lower().split() if t]
    if not tokens:
        return v
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        v[idx] += sign
    n = np.linalg.norm(v)
    if n > 0:
        v /= n
    return v


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT,
            text TEXT,
            meta_json TEXT,
            embedding_json TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    return conn


def remember_experience(
    kind: str,
    text: str,
    meta: Optional[Dict[str, Any]] = None,
) -> int:
    emb = _embed(text)
    conn = _connect()
    cur = conn.execute(
        "INSERT INTO experiences (kind, text, meta_json, embedding_json, created_at) VALUES (?,?,?,?,?)",
        (
            kind,
            text[:4000],
            json.dumps(meta or {}, ensure_ascii=False),
            json.dumps(emb.tolist()),
            _now(),
        ),
    )
    conn.commit()
    rid = int(cur.lastrowid)
    conn.close()
    return rid


def recall_similar(query: str, top_k: int = 5, kind: Optional[str] = None) -> List[Dict[str, Any]]:
    q = _embed(query)
    conn = _connect()
    if kind:
        rows = conn.execute(
            "SELECT id, kind, text, meta_json, embedding_json, created_at FROM experiences WHERE kind=?",
            (kind,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, kind, text, meta_json, embedding_json, created_at FROM experiences"
        ).fetchall()
    conn.close()
    scored = []
    for rid, k, text, meta_j, emb_j, created in rows:
        try:
            emb = np.asarray(json.loads(emb_j), dtype=np.float64)
            score = _cos(q, emb)
        except Exception:
            score = 0.0
        scored.append(
            {
                "id": rid,
                "kind": k,
                "text": text,
                "meta": json.loads(meta_j or "{}"),
                "score": score,
                "created_at": created,
            }
        )
    scored.sort(key=lambda x: -x["score"])
    return scored[:top_k]


def memory_stats() -> Dict[str, Any]:
    conn = _connect()
    n = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
    kinds = conn.execute(
        "SELECT kind, COUNT(*) FROM experiences GROUP BY kind"
    ).fetchall()
    conn.close()
    return {"total": n, "by_kind": {k: c for k, c in kinds}, "db": str(DB_PATH.relative_to(ROOT))}


def handle_memory_command(user_input: str) -> Optional[str]:
    import re

    text = (user_input or "").strip()
    if not text:
        return None
    if re.search(r"(احصاء\s*الذاكر|إحصاء\s*الذاكر|memory\s*stats|حجم\s*الذاكر)", text, re.I):
        st = memory_stats()
        return (
            "## 🗄️ إحصاء الذاكرة المستمرة\n\n```json\n"
            + json.dumps(st, ensure_ascii=False, indent=2)
            + "\n```"
        )
    if re.search(r"(تذكر|احفظ\s*خبر|remember)", text, re.I):
        m = re.search(r"(?:تذكر|احفظ|remember)[:\s]+(.+)$", text, re.I)
        body = m.group(1).strip() if m else text
        rid = remember_experience("manual", body)
        return f"## 💾 تم الحفظ\n- id=`{rid}`\n- {body[:300]}"
    if re.search(r"(استرجع|recall|ذاكر[ةه]\s*مشابه|خبرات\s*سابق)", text, re.I):
        m = re.search(r"(?:استرجع|recall|عن)[:\s]+(.+)$", text, re.I)
        q = m.group(1).strip() if m else text
        hits = recall_similar(q, top_k=5)
        lines = ["## 🔎 استرجاع من الذاكرة المتجهة", f"- الاستعلام: {q}", ""]
        if not hits:
            lines.append("_لا خبرات بعد — نفّذ دورة تفكير/NAS/فشل لملء الذاكرة._")
        for h in hits:
            lines.append(
                f"- **{h['score']:.3f}** `{h['kind']}` — {h['text'][:180]}…"
            )
        return "\n".join(lines)
    return None

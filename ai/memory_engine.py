"""
Phase 3 – Memory Engine
Persistent SQLite memory of successful/failed routes,
node performance, and learned patterns.

Knowledge Layer Integration (Phase 3 completion):
  Writes route records and node execution stats to:
    knowledge/route_memory.json
    knowledge/node_profiles.json
  via KnowledgeStore (injected via set_knowledge_store()).
"""
from __future__ import annotations
import sqlite3
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from knowledge.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)


class RouteMemory:
    """Summary of a remembered route (path + performance stats)."""

    def __init__(self, path_key: str, path: List[str]):
        self.path_key = path_key
        self.path = path
        self.runs: int = 0
        self.successes: int = 0
        self.failures: int = 0
        self.total_latency_ms: float = 0.0
        self.first_seen: str = datetime.utcnow().isoformat()
        self.last_seen: str = self.first_seen
        self.is_promoted: bool = False   # Marked as a "golden" route

    @property
    def success_rate(self) -> float:
        return self.successes / self.runs if self.runs > 0 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.runs if self.runs > 0 else 0.0

    @property
    def health(self) -> str:
        sr = self.success_rate
        if sr >= 0.9:
            return "excellent"
        if sr >= 0.75:
            return "good"
        if sr >= 0.5:
            return "degraded"
        return "critical"

    @property
    def memory_score(self) -> float:
        """Overall desirability (0-100) for route selection."""
        sr_component = self.success_rate * 70.0
        boost = 15.0 if self.is_promoted else 0.0
        # Latency penalty: 15 pts at 0 ms, 0 pts at 10 s
        lat_component = max(0.0, 15.0 * (1.0 - self.avg_latency_ms / 10000.0))
        return round(min(100.0, sr_component + boost + lat_component), 2)

    def record(self, success: bool, latency_ms: float):
        self.runs += 1
        self.total_latency_ms += latency_ms
        if success:
            self.successes += 1
        else:
            self.failures += 1
        self.last_seen = datetime.utcnow().isoformat()
        self._path_json = None  # إعادة تجهيز نص path المُجمّد بعد التحديث

    def path_json(self) -> str:
        """نص JSON المُجمّد للمسار — يُعاد بناؤه مرة واحدة بعد كل تحديث."""
        if self._path_json is None:
            self._path_json = json.dumps(self.path)
        return self._path_json

    def to_dict(self) -> dict:
        return {
            "path_key": self.path_key,
            "path": self.path,
            "runs": self.runs,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": round(self.success_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "health": self.health,
            "memory_score": self.memory_score,
            "is_promoted": self.is_promoted,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


class NodeMemory:
    """Execution statistics per individual node."""

    def __init__(self, node_id: str, name: str):
        self.node_id = node_id
        self.name = name
        self.executions: int = 0
        self.successes: int = 0
        self.failures: int = 0
        self.total_latency_ms: float = 0.0
        self.last_executed: Optional[str] = None

    @property
    def success_rate(self) -> float:
        return self.successes / self.executions if self.executions > 0 else 0.5

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.executions if self.executions > 0 else 0.0

    def record(self, success: bool, latency_ms: float):
        self.executions += 1
        self.total_latency_ms += latency_ms
        if success:
            self.successes += 1
        else:
            self.failures += 1
        self.last_executed = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "executions": self.executions,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": round(self.success_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "last_executed": self.last_executed,
        }


class MemoryEngine:
    """
    Phase 3 Memory Engine.
    Learns from every execution and stores knowledge in SQLite.
    Provides recall API for routing decisions.
    """

    def __init__(self, db_path: str = "./data/mesh.db"):
        self._db_path = Path(db_path)
        self._routes: Dict[str, RouteMemory] = {}
        self._nodes: Dict[str, NodeMemory] = {}
        self._knowledge = None   # KnowledgeStore — injected via set_knowledge_store()
        self._conn_lock = threading.Lock()
        self._shared_conn: Optional[sqlite3.Connection] = None
        # دفتر ملاحظات لتجميع الكتابات في دفقة واحدة (batch flush) —
        # بدل قفل SQLite وكتابة منفصلة لكل خطوة في كل تنفيذ.
        self._dirty_routes: Dict[str, RouteMemory] = {}
        self._dirty_nodes: Dict[str, NodeMemory] = {}
        self._flush_timer: Optional[threading.Timer] = None
        self._flush_interval: float = 2.0
        self._init_schema()
        self._load()
        logger.info("MemoryEngine initialised (Phase 3)")

    def set_knowledge_store(self, ks) -> None:
        """Inject the KnowledgeStore so MemoryEngine can persist to JSON knowledge files."""
        self._knowledge = ks
        logger.info("MemoryEngine: KnowledgeStore connected")

    # ── Schema ─────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        """يعيد استخدام اتصال SQLite واحد بدل فتح اتصال جديد في كل استدعاء
        (تحسين أداء: connection pooling بسيط + آمن للخيوط عبر قفل)."""
        if self._shared_conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            self._shared_conn = conn
        return self._shared_conn

    def close(self) -> None:
        """يغلق الاتصال المشترك صراحة (اختياري عند إيقاف التشغيل) بعد تفريغ الدفقات."""
        try:
            self.flush()
        except Exception:
            pass
        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None
        with self._conn_lock:
            if self._shared_conn is not None:
                try:
                    self._shared_conn.close()
                finally:
                    self._shared_conn = None

    def _init_schema(self):
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn_lock, self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS route_memory (
                    path_key        TEXT PRIMARY KEY,
                    path_json       TEXT NOT NULL,
                    runs            INTEGER DEFAULT 0,
                    successes       INTEGER DEFAULT 0,
                    failures        INTEGER DEFAULT 0,
                    total_latency_ms REAL DEFAULT 0.0,
                    is_promoted     INTEGER DEFAULT 0,
                    first_seen      TEXT NOT NULL,
                    last_seen       TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS node_memory (
                    node_id         TEXT PRIMARY KEY,
                    name            TEXT NOT NULL,
                    executions      INTEGER DEFAULT 0,
                    successes       INTEGER DEFAULT 0,
                    failures        INTEGER DEFAULT 0,
                    total_latency_ms REAL DEFAULT 0.0,
                    last_executed   TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_route_score
                    ON route_memory(successes, runs);
            """)

    def _load(self):
        try:
            with self._conn_lock, self._conn() as conn:
                for row in conn.execute("SELECT * FROM route_memory").fetchall():
                    rm = RouteMemory(row["path_key"], json.loads(row["path_json"]))
                    rm.runs = row["runs"]
                    rm.successes = row["successes"]
                    rm.failures = row["failures"]
                    rm.total_latency_ms = row["total_latency_ms"]
                    rm.is_promoted = bool(row["is_promoted"])
                    rm.first_seen = row["first_seen"]
                    rm.last_seen = row["last_seen"]
                    self._routes[rm.path_key] = rm

                for row in conn.execute("SELECT * FROM node_memory").fetchall():
                    nm = NodeMemory(row["node_id"], row["name"])
                    nm.executions = row["executions"]
                    nm.successes = row["successes"]
                    nm.failures = row["failures"]
                    nm.total_latency_ms = row["total_latency_ms"]
                    nm.last_executed = row["last_executed"]
                    self._nodes[nm.node_id] = nm

            logger.info(f"MemoryEngine loaded {len(self._routes)} routes, {len(self._nodes)} nodes")
        except Exception as e:
            logger.warning(f"MemoryEngine load warning: {e}")

    # ── Learning ───────────────────────────────────────────────────────────

    def learn_from_run(self, run_result: dict):
        """Primary learning method. Called after every execution."""
        path = run_result.get("path", [])
        if not path:
            return

        success = run_result.get("status") == "success"
        total_ms = run_result.get("total_duration_ms") or 0.0

        # Update route memory
        path_key = "->".join(p[:8] for p in path)
        if path_key not in self._routes:
            self._routes[path_key] = RouteMemory(path_key, path)
        self._routes[path_key].record(success, total_ms)
        self._dirty_routes[path_key] = self._routes[path_key]

        # Update node memory from steps — تجميع في الدفتر بدل الكتابة الفورية لكل خطوة.
        for step in run_result.get("steps", []):
            nid = step.get("node_id")
            nname = step.get("node_name", "unknown")
            if not nid:
                continue
            if nid not in self._nodes:
                self._nodes[nid] = NodeMemory(nid, nname)
            step_ok = step.get("status") == "success"
            step_ms = step.get("duration_ms") or 0.0
            self._nodes[nid].record(step_ok, step_ms)
            self._dirty_nodes[nid] = self._nodes[nid]

        self._schedule_flush()

        logger.debug(f"MemoryEngine learned from run {run_result.get('run_id','?')[:8]}")

        # ── Knowledge layer: persist to JSON files ─────────────────────────
        if self._knowledge:
            rm = self._routes[path_key]
            try:
                self._knowledge.upsert_route(rm.to_dict())
                run_id = run_result.get('run_id', '')
                self._knowledge.append_route_execution(
                    path_key, run_id, success, total_ms
                )
            except Exception as ke:
                logger.warning(f"MemoryEngine: knowledge route write failed: {ke}")

            for step in run_result.get('steps', []):
                nid = step.get('node_id')
                if not nid:
                    continue
                step_ok = step.get('status') == 'success'
                step_ms = step.get('duration_ms') or 0.0
                try:
                    self._knowledge.update_node_execution_stats(nid, step_ok, step_ms)
                except Exception as ke:
                    logger.warning(f"MemoryEngine: knowledge node stats write failed: {ke}")

    def _persist_route(self, rm: RouteMemory):
        """كتابة فورية لمسار واحد (تُستخدم عند الترقية/الإلغاء يدوياً)."""
        try:
            with self._conn_lock, self._conn() as conn:
                conn.execute("""
                    INSERT INTO route_memory
                        (path_key, path_json, runs, successes, failures,
                         total_latency_ms, is_promoted, first_seen, last_seen)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(path_key) DO UPDATE SET
                        runs=excluded.runs, successes=excluded.successes,
                        failures=excluded.failures,
                        total_latency_ms=excluded.total_latency_ms,
                        is_promoted=excluded.is_promoted,
                        last_seen=excluded.last_seen
                """, (
                    rm.path_key, rm.path_json(),
                    rm.runs, rm.successes, rm.failures,
                    rm.total_latency_ms, int(rm.is_promoted),
                    rm.first_seen, rm.last_seen,
                ))
        except Exception as e:
            logger.error(f"MemoryEngine persist_route error: {e}")

    def _persist_node(self, nm: NodeMemory):
        """كتابة فورية لعقدة واحدة (تُستخدم عند الحاجة للضمان الفوري)."""
        try:
            with self._conn_lock, self._conn() as conn:
                conn.execute("""
                    INSERT INTO node_memory
                        (node_id, name, executions, successes, failures,
                         total_latency_ms, last_executed)
                    VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        executions=excluded.executions,
                        successes=excluded.successes,
                        failures=excluded.failures,
                        total_latency_ms=excluded.total_latency_ms,
                        last_executed=excluded.last_executed
                """, (
                    nm.node_id, nm.name,
                    nm.executions, nm.successes, nm.failures,
                    nm.total_latency_ms, nm.last_executed,
                ))
        except Exception as e:
            logger.error(f"MemoryEngine persist_node error: {e}")

    # ── Batched flush — تجميع الكتابات في دفقة واحدة ─────────────────────

    def _schedule_flush(self) -> None:
        """يحدد دفقة مجمّعة قادمة؛ الكتابة الفعلية تتم مرة واحدة لكل نافذة زمنية."""
        if self._flush_timer is not None:
            return
        self._flush_timer = threading.Timer(self._flush_interval, self.flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def flush(self) -> None:
        """يكتب كل المسارات والعقد المتسخة دفعة واحدة داخل قفل واحد."""
        try:
            with self._conn_lock:
                routes = self._dirty_routes
                nodes = self._dirty_nodes
                self._dirty_routes = {}
                self._dirty_nodes = {}
                conn = self._conn()
                if routes:
                    conn.executemany("""
                        INSERT INTO route_memory
                            (path_key, path_json, runs, successes, failures,
                             total_latency_ms, is_promoted, first_seen, last_seen)
                        VALUES (?,?,?,?,?,?,?,?,?)
                        ON CONFLICT(path_key) DO UPDATE SET
                            runs=excluded.runs, successes=excluded.successes,
                            failures=excluded.failures,
                            total_latency_ms=excluded.total_latency_ms,
                            is_promoted=excluded.is_promoted,
                            last_seen=excluded.last_seen
                    """, (
                        (rm.path_key, rm.path_json(), rm.runs, rm.successes, rm.failures,
                         rm.total_latency_ms, int(rm.is_promoted), rm.first_seen, rm.last_seen)
                        for rm in routes.values()
                    ))
                if nodes:
                    conn.executemany("""
                        INSERT INTO node_memory
                            (node_id, name, executions, successes, failures,
                             total_latency_ms, last_executed)
                        VALUES (?,?,?,?,?,?,?)
                        ON CONFLICT(node_id) DO UPDATE SET
                            executions=excluded.executions,
                            successes=excluded.successes,
                            failures=excluded.failures,
                            total_latency_ms=excluded.total_latency_ms,
                            last_executed=excluded.last_executed
                    """, (
                        (nm.node_id, nm.name, nm.executions, nm.successes, nm.failures,
                         nm.total_latency_ms, nm.last_executed)
                        for nm in nodes.values()
                    ))
                conn.commit()
        except Exception as e:
            logger.error(f"MemoryEngine flush error: {e}")
        finally:
            self._flush_timer = None

    # ── Recall API ─────────────────────────────────────────────────────────

    def recall_best_routes(self, start_id: str, end_id: str,
                           top_k: int = 3) -> List[RouteMemory]:
        """Return top-k remembered routes between start and end."""
        matching = [
            rm for rm in self._routes.values()
            if rm.path and rm.path[0] == start_id and rm.path[-1] == end_id
        ]
        matching.sort(key=lambda r: r.memory_score, reverse=True)
        return matching[:top_k]

    def recall_route(self, path: List[str]) -> Optional[RouteMemory]:
        key = "->".join(p[:8] for p in path)
        return self._routes.get(key)

    def get_node_memory(self, node_id: str) -> Optional[NodeMemory]:
        return self._nodes.get(node_id)

    def best_nodes(self, n: int = 10) -> List[dict]:
        return sorted(
            [nm.to_dict() for nm in self._nodes.values()],
            key=lambda x: x["success_rate"], reverse=True
        )[:n]

    def worst_nodes(self, n: int = 10) -> List[dict]:
        return sorted(
            [nm.to_dict() for nm in self._nodes.values() if nm.executions > 0],
            key=lambda x: x["success_rate"]
        )[:n]

    def promote_route(self, path: List[str]):
        """Mark a route as promoted (golden path)."""
        key = "->".join(p[:8] for p in path)
        if key in self._routes:
            self._routes[key].is_promoted = True
            self._persist_route(self._routes[key])
            if self._knowledge:
                self._knowledge.promote_route(key)

    def demote_route(self, path: List[str]):
        """Remove promotion from a route."""
        key = "->".join(p[:8] for p in path)
        if key in self._routes:
            self._routes[key].is_promoted = False
            self._persist_route(self._routes[key])
            if self._knowledge:
                self._knowledge.demote_route(key)

    def summary(self) -> dict:
        promoted = [r for r in self._routes.values() if r.is_promoted]
        failed = [r for r in self._routes.values() if r.health == "critical"]
        return {
            "total_routes": len(self._routes),
            "total_nodes_tracked": len(self._nodes),
            "promoted_routes": len(promoted),
            "critical_routes": len(failed),
        }

    def all_routes(self) -> List[dict]:
        return [rm.to_dict() for rm in self._routes.values()]

    def __repr__(self):
        return f"<MemoryEngine routes={len(self._routes)} nodes={len(self._nodes)}>"

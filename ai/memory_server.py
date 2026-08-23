from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import json
import time
import uvicorn
import asyncio
import threading
from pathlib import Path

app = FastAPI(title="NSM Shared Memory Server")

# أمان بسيط باستخدام API Key
API_KEY = "nsm_secret_key_2026"
api_key_header = APIKeyHeader(name="X-NSM-Token")

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate credentials")

# تخزين الذاكرة (في الذاكرة حالياً مع دعم التحميل من ملف)
STORAGE_PATH = Path("artifacts/learning/distributed_knowledge.json")
STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

class MemoryState:
    def __init__(self):
        self.data = self._load()
        self._lock = threading.Lock()
        self._save_pending = False
    
    def _load(self):
        if STORAGE_PATH.exists():
            try:
                with open(STORAGE_PATH, "r") as f:
                    return json.load(f)
            except: pass
        return {"shared_facts": {}, "active_queries": {}, "trust_scores": {}}
    
    def save(self):
        """حفظ البيانات بشكل غير متزامن لتجنب حظر الطلبات."""
        if self._save_pending: return
        self._save_pending = True
        threading.Thread(target=self._save_worker).start()

    def _save_worker(self):
        time.sleep(1) # تجميع الطلبات (Debounce)
        with self._lock:
            with open(STORAGE_PATH, "w") as f:
                json.dump(self.data, f, indent=2)
        self._save_pending = False

memory = MemoryState()

class MetricsState:
    def __init__(self):
        self.request_counts = {"share": 0, "sync": 0, "ask": 0, "answer": 0}
        self.agent_stats = {}
        self.start_time = time.time()

    def log_request(self, type: str, agent_id: str):
        self.request_counts[type] = self.request_counts.get(type, 0) + 1
        if agent_id not in self.agent_stats:
            self.agent_stats[agent_id] = {"requests": 0, "last_seen": 0}
        self.agent_stats[agent_id]["requests"] += 1
        self.agent_stats[agent_id]["last_seen"] = time.time()

metrics = MetricsState()

class Fact(BaseModel):
    agent_id: str
    content: str
    importance: float
    semantic_hash: Optional[str] = None

class Query(BaseModel):
    agent_id: str
    query: str
    context: str = ""

class Answer(BaseModel):
    agent_id: str
    query_id: str
    answer: str

@app.get("/health")
def health():
    return {"status": "online", "version": "1.0.0"}

@app.post("/share")
def share_fact(fact: Fact, api_key: str = Depends(get_api_key)):
    metrics.log_request("share", fact.agent_id)
    # استخدام معرف ثابت يعتمد على المحتوى
    import hashlib
    fact_id = f"shared_{hashlib.md5(fact.content.encode()).hexdigest()[:8]}"
    memory.data["shared_facts"][fact_id] = {
        "content": fact.content,
        "origin_agent": fact.agent_id,
        "shared_at": time.time(),
        "importance": fact.importance,
        "semantic_hash": fact.semantic_hash
    }
    memory.save()
    return {"status": "success", "fact_id": fact_id}

@app.get("/sync")
def sync_facts(agent_id: Optional[str] = None, api_key: str = Depends(get_api_key)):
    if agent_id: metrics.log_request("sync", agent_id)
    return memory.data["shared_facts"]

@app.post("/queries/ask")
def ask_query(q: Query, api_key: str = Depends(get_api_key)):
    import uuid
    query_id = f"q_{uuid.uuid4().hex[:6]}"
    memory.data["active_queries"][query_id] = {
        "query": q.query,
        "context": q.context,
        "asker": q.agent_id,
        "timestamp": time.time(),
        "status": "open",
        "answers": []
    }
    memory.save()
    return {"status": "success", "query_id": query_id}

@app.get("/queries/pending")
def get_queries(api_key: str = Depends(get_api_key)):
    return memory.data["active_queries"]

@app.post("/queries/answer")
def answer_query(a: Answer, api_key: str = Depends(get_api_key)):
    metrics.log_request("answer", a.agent_id)
    if a.query_id in memory.data["active_queries"]:
        memory.data["active_queries"][a.query_id]["answers"].append({
            "answer": a.answer,
            "provider": a.agent_id,
            "timestamp": time.time()
        })
        memory.data["active_queries"][a.query_id]["status"] = "answered"
        memory.save()
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Query not found")

@app.get("/metrics")
def get_metrics(api_key: str = Depends(get_api_key)):
    uptime = time.time() - metrics.start_time
    return {
        "uptime_seconds": uptime,
        "total_requests": sum(metrics.request_counts.values()),
        "request_breakdown": metrics.request_counts,
        "active_agents_count": len(metrics.agent_stats),
        "agent_details": metrics.agent_stats,
        "memory_usage": {
            "facts_count": len(memory.data["shared_facts"]),
            "queries_count": len(memory.data["active_queries"])
        }
    }

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

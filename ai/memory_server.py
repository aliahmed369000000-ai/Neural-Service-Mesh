from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import json
import time
import uvicorn
import asyncio
import threading
import gc
import os
import psutil
from pathlib import Path

app = FastAPI(title="NSM Shared Memory Server")

# نظام RBAC: تعريف الأدوار والصلاحيات
ROLES = {
    "admin": ["read", "write", "delete", "manage"],
    "expert": ["read", "write"],
    "viewer": ["read"],
    "worker": ["read", "write"]
}

# سجل الوكلاء المعتمدين (يمكن نقله لقاعدة بيانات لاحقاً)
# في الإنتاج، يجب تحميل هذه البيانات من قاعدة بيانات مؤمنة أو متغيرات بيئة
AGENT_REGISTRY = {
    "Admin_Agent_01": {"role": "admin", "token": os.environ.get("NSM_ADMIN_TOKEN", "admin_dev_token")},
    "Expert_Agent_Alpha": {"role": "expert", "token": os.environ.get("NSM_EXPERT_TOKEN", "expert_dev_token")},
    "Worker_Agent_Beta": {"role": "worker", "token": os.environ.get("NSM_WORKER_TOKEN", "worker_dev_token")},
    "Viewer_Agent_Gamma": {"role": "viewer", "token": os.environ.get("NSM_VIEWER_TOKEN", "viewer_dev_token")}
}

api_key_header = APIKeyHeader(name="X-NSM-Token")

def get_current_agent(api_key: str = Security(api_key_header)):
    for agent_id, data in AGENT_REGISTRY.items():
        if data["token"] == api_key:
            return {"agent_id": agent_id, "role": data["role"]}
    raise HTTPException(status_code=403, detail="Invalid Agent Token")

def check_permission(agent: dict, required_permission: str):
    permissions = ROLES.get(agent["role"], [])
    if required_permission not in permissions:
        raise HTTPException(
            status_code=403, 
            detail=f"Role '{agent['role']}' does not have '{required_permission}' permission"
        )
    return True

# تخزين الذاكرة (في الذاكرة حالياً مع دعم التحميل من ملف)
STORAGE_PATH = Path("artifacts/learning/distributed_knowledge.json")
STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)

class MemoryState:
    def __init__(self):
        self.data = self._load()
        self._lock = threading.Lock()
        self._save_pending = False
        self.max_facts = 50000  # حد أقصى للحقائق في الذاكرة
        self.ttl_seconds = 86400 * 7  # 7 أيام للحقائق منخفضة الأهمية
    
    def _load(self):
        if STORAGE_PATH.exists():
            try:
                with open(STORAGE_PATH, "r") as f:
                    return json.load(f)
            except: pass
        return {
            "shared_facts": {}, 
            "active_queries": {}, 
            "trust_scores": {},
            "shared_tools": {} # 🆕 سجل الأدوات المشترك
        }
    
    def save(self):
        """حفظ البيانات بشكل غير متزامن لتجنب حظر الطلبات."""
        if self._save_pending: return
        self._save_pending = True
        threading.Thread(target=self._save_worker).start()

    def _save_worker(self):
        time.sleep(1) # تجميع الطلبات (Debounce)
        self.prune() # تنظيف الذاكرة قبل الحفظ
        with self._lock:
            with open(STORAGE_PATH, "w") as f:
                json.dump(self.data, f, indent=2)
        self._save_pending = False

    def prune(self):
        """تنظيف الحقائق القديمة أو منخفضة الأهمية عند تجاوز السعة."""
        with self._lock:
            facts = self.data["shared_facts"]
            if len(facts) <= self.max_facts:
                return

            print(f"🧹 بدء تنظيف الذاكرة... الحجم الحالي: {len(facts)}")
            # ترتيب حسب الأهمية (الأقل أولاً) ثم الوقت (الأقدم أولاً)
            sorted_keys = sorted(
                facts.keys(),
                key=lambda k: (facts[k].get("importance", 0.5), facts[k].get("shared_at", 0))
            )
            
            # حذف أقدم/أقل 20% من البيانات
            to_remove = sorted_keys[:int(len(facts) * 0.2)]
            for k in to_remove:
                del facts[k]
            
            print(f"✅ تم حذف {len(to_remove)} حقيقة قديمة.")
            gc.collect() # تحفيز جمع النفايات

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
    is_encrypted: bool = False

class ToolDefinition(BaseModel):
    name: str
    description: str
    code: str
    params_schema: Dict[str, Any]
    agent_id: str
    version: str = "1.0.0"

class Query(BaseModel):
    agent_id: str
    query: str
    context: str = ""

class Answer(BaseModel):
    agent_id: str
    query_id: str
    answer: str

class ToolVote(BaseModel):
    tool_id: str
    agent_id: str
    vote: str # "up" or "down"
    comment: Optional[str] = None
    trust_score: float = 1.0

@app.get("/health")
def health():
    return {"status": "online", "version": "1.0.0"}

@app.post("/share")
def share_fact(fact: Fact, agent: dict = Depends(get_current_agent)):
    check_permission(agent, "write")
    metrics.log_request("share", agent["agent_id"])
    # استخدام معرف ثابت يعتمد على المحتوى
    import hashlib
    fact_id = f"shared_{hashlib.md5(fact.content.encode()).hexdigest()[:8]}"
    memory.data["shared_facts"][fact_id] = {
        "content": fact.content,
        "origin_agent": fact.agent_id,
        "shared_at": time.time(),
        "importance": fact.importance,
        "semantic_hash": fact.semantic_hash,
        "is_encrypted": fact.is_encrypted
    }
    memory.save()
    return {"status": "success", "fact_id": fact_id}

@app.get("/sync")
def sync_facts(agent: dict = Depends(get_current_agent)):
    check_permission(agent, "read")
    metrics.log_request("sync", agent["agent_id"])
    return memory.data["shared_facts"]

@app.post("/queries/ask")
def ask_query(q: Query, agent: dict = Depends(get_current_agent)):
    check_permission(agent, "write")
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
def get_queries(agent: dict = Depends(get_current_agent)):
    check_permission(agent, "read")
    return memory.data["active_queries"]

@app.post("/queries/answer")
def answer_query(a: Answer, agent: dict = Depends(get_current_agent)):
    check_permission(agent, "write")
    metrics.log_request("answer", agent["agent_id"])
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

# 🆕 نقاط نهاية سجل الأدوات (Tool Registry Endpoints)

@app.post("/tools/publish")
def publish_tool(tool: ToolDefinition, agent: dict = Depends(get_current_agent)):
    """نشر أداة جديدة للسجل المركزي بحالة معلقة (Pending)."""
    check_permission(agent, "write")
    if "shared_tools" not in memory.data:
        memory.data["shared_tools"] = {}
        
    tool_id = tool.name
    memory.data["shared_tools"][tool_id] = {
        "name": tool.name,
        "description": tool.description,
        "code": tool.code,
        "params_schema": tool.params_schema,
        "author": tool.agent_id,
        "version": tool.version,
        "published_at": time.time(),
        "status": "pending", # 🆕 الحالة الافتراضية: معلق
        "votes": {"up": 0, "down": 0},
        "reviews": []
    }
    memory.save()
    return {"status": "success", "tool_id": tool_id}

@app.post("/tools/vote")
def vote_tool(v: ToolVote, agent: dict = Depends(get_current_agent)):
    """التصويت على أداة ومراجعتها من قبل الأقران."""
    check_permission(agent, "write")
    tools = memory.data.get("shared_tools", {})
    if v.tool_id not in tools:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    tool = tools[v.tool_id]
    
    # تحديث التصويت
    if v.vote == "up":
        tool["votes"]["up"] += 1
    else:
        tool["votes"]["down"] += 1
        
    # إضافة مراجعة
    tool["reviews"].append({
        "reviewer": v.agent_id,
        "vote": v.vote,
        "comment": v.comment,
        "timestamp": time.time(),
        "trust_score": v.trust_score
    })
    
    # 🆕 معيار الاعتماد التلقائي: 3 أصوات إيجابية
    if tool["votes"]["up"] >= 3 and tool["status"] == "pending":
        tool["status"] = "approved"
        
    memory.save()
    return {"status": "success", "current_status": tool["status"]}

@app.get("/tools/list")
def list_tools(agent: dict = Depends(get_current_agent)):
    """قائمة بجميع الأدوات المتاحة في السجل."""
    check_permission(agent, "read")
    return memory.data.get("shared_tools", {})

@app.get("/tools/get/{tool_id}")
def get_tool(tool_id: str, agent: dict = Depends(get_current_agent)):
    """جلب تفاصيل أداة محددة."""
    check_permission(agent, "read")
    tools = memory.data.get("shared_tools", {})
    if tool_id in tools:
        return tools[tool_id]
    raise HTTPException(status_code=404, detail="Tool not found")

@app.get("/metrics")
def get_metrics(agent: dict = Depends(get_current_agent)):
    check_permission(agent, "manage")
    uptime = time.time() - metrics.start_time
    process = psutil.Process(os.getpid())
    ram_usage_mb = process.memory_info().rss / (1024 * 1024)
    
    return {
        "uptime_seconds": uptime,
        "total_requests": sum(metrics.request_counts.values()),
        "request_breakdown": metrics.request_counts,
        "active_agents_count": len(metrics.agent_stats),
        "agent_details": metrics.agent_stats,
        "system_resources": {
            "ram_usage_mb": round(ram_usage_mb, 2),
            "cpu_percent": process.cpu_percent(interval=0.1),
            "gc_objects": len(gc.get_objects())
        },
        "memory_usage": {
            "facts_count": len(memory.data["shared_facts"]),
            "queries_count": len(memory.data["active_queries"]),
            "max_capacity": memory.max_facts
        }
    }

@app.post("/system/gc")
def trigger_gc(agent: dict = Depends(get_current_agent)):
    check_permission(agent, "manage")
    initial_obj = len(gc.get_objects())
    gc.collect()
    final_obj = len(gc.get_objects())
    return {
        "status": "GC Triggered",
        "objects_collected": initial_obj - final_obj,
        "current_objects": final_obj
    }

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

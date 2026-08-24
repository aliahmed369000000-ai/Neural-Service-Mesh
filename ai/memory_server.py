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
# سجل الوكلاء المعتمدين (يتم تحميله من البيئة لدعم التوزيع)
def load_agent_registry():
    registry = {}
    # تحميل التوكن الأساسي من البيئة (NSM_AUTH_TOKEN)
    base_token = os.environ.get("NSM_AUTH_TOKEN", "nsm-secure-token-2026")
    
    # إضافة الوكلاء الافتراضيين
    registry["Admin_Agent_01"] = {"role": "admin", "token": os.environ.get("NSM_ADMIN_TOKEN", base_token)}
    registry["agent-alpha"] = {"role": "expert", "token": base_token}
    registry["agent-beta"] = {"role": "worker", "token": base_token}
    registry["test-agent"] = {"role": "admin", "token": "test_token_recovery"}
    
    # إمكانية إضافة وكلاء ديناميكياً عبر متغيرات البيئة AGENT_ID_TOKEN
    for key, value in os.environ.items():
        if key.startswith("NSM_AGENT_"):
            agent_id = key.replace("NSM_AGENT_", "").lower()
            registry[agent_id] = {"role": "worker", "token": value}
            
    return registry

AGENT_REGISTRY = load_agent_registry()

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
        self.request_counts = {"share": 0, "sync": 0, "ask": 0, "answer": 0, "heartbeat": 0}
        self.agent_stats = {}
        self.start_time = time.time()
        self.node_status = {} # 🆕 حالة العقد الموزعة
        self.active_locks = {} # 🆕 الأقفال الموزعة {resource_id: {agent_id, expires_at}}
        self._lock = threading.Lock()

    def acquire_lock(self, resource_id: str, agent_id: str, ttl: int = 30) -> bool:
        """محاولة الحصول على قفل لمورد معين مع ضمان الذرية."""
        with self._lock:
            now = time.time()
            # التحقق من وجود قفل نشط وغير منتهي الصلاحية لوكيل آخر
            if resource_id in self.active_locks:
                lock = self.active_locks[resource_id]
                if now < lock["expires_at"]:
                    if lock["agent_id"] == agent_id:
                        # تجديد القفل لنفس الوكيل
                        lock["expires_at"] = now + ttl
                        return True
                    return False # محجوز لوكيل آخر
            
            # منح القفل (لا يوجد قفل أو انتهت صلاحيته)
            self.active_locks[resource_id] = {
                "agent_id": agent_id,
                "expires_at": now + ttl
            }
            return True

    def release_lock(self, resource_id: str, agent_id: str):
        """تحرير قفل مورد."""
        with self._lock:
            if resource_id in self.active_locks:
                if self.active_locks[resource_id]["agent_id"] == agent_id:
                    del self.active_locks[resource_id]

    def log_request(self, type: str, agent_id: str):
        self.request_counts[type] = self.request_counts.get(type, 0) + 1
        if agent_id not in self.agent_stats:
            self.agent_stats[agent_id] = {"requests": 0, "last_seen": 0}
        self.agent_stats[agent_id]["requests"] += 1
        self.agent_stats[agent_id]["last_seen"] = time.time()
        
    def update_heartbeat(self, agent_id: str, node_info: Dict[str, Any]):
        """تحديث نبض القلب وحالة العقدة مع تطبيق التعافي التدريجي للثقة."""
        now = time.time()
        print(f"💓 Heartbeat received from: {agent_id}")
        self.request_counts["heartbeat"] += 1
        
        if agent_id not in self.node_status:
            self.node_status[agent_id] = {"agent_id": agent_id, "last_recovery": now}
        
        # آلية التعافي التدريجي: زيادة الثقة بمقدار 0.01 كل 60 ثانية من النشاط المستمر
        last_recovery = self.node_status[agent_id].get("last_recovery", now)
        if now - last_recovery >= 60: # التعافي التدريجي كل دقيقة من النشاط
            current_score = memory.data["trust_scores"].get(agent_id, 1.0)
            if current_score < 1.0: # التعافي يعمل فقط للعودة للمستوى الطبيعي
                memory.data["trust_scores"][agent_id] = min(current_score + 0.01, 1.0)
                print(f"📈 [Recovery]: زيادة ثقة الوكيل {agent_id} إلى {memory.data['trust_scores'][agent_id]}")
                memory.save() # ضمان حفظ نقاط الثقة الجديدة
            self.node_status[agent_id]["last_recovery"] = now

        self.node_status[agent_id].update({
            "last_seen": now,
            "info": node_info,
            "status": "online"
        })

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

class LockRequest(BaseModel):
    resource_id: str
    agent_id: str
    ttl: int = 30

class Heartbeat(BaseModel):
    agent_id: str
    node_info: Dict[str, Any]
    current_task: Optional[str] = None

@app.get("/health")
def health():
    return {"status": "online", "version": "1.0.0"}

@app.post("/heartbeat")
def post_heartbeat(hb: Heartbeat, agent: dict = Depends(get_current_agent)):
    """استقبال نبض القلب من الوكلاء وتحديث حالتهم."""
    print(f"💓 نبض قلب مستلم من: {hb.agent_id}")
    metrics.update_heartbeat(hb.agent_id, hb.node_info)
    # تخزين المهمة الحالية لدعم التعافي
    if hb.current_task:
        metrics.node_status[hb.agent_id]["current_task"] = hb.current_task
    return {"status": "alive", "timestamp": time.time()}

@app.get("/swarm/status")
def get_swarm_status(agent: dict = Depends(get_current_agent)):
    """عرض حالة السرب بالكامل واكتشاف العقد الفاشلة."""
    check_permission(agent, "read")
    now = time.time()
    swarm_report = {}
    for agent_id, data in metrics.node_status.items():
        # إذا لم يرسل نبضاً لأكثر من 15 ثانية، نعتبره "فاشلاً"
        status = "online"
        if now - data["last_seen"] > 15:
            status = "failed"
        elif now - data["last_seen"] > 10:
            status = "warning"
            
        swarm_report[agent_id] = {
            "status": status,
            "last_seen_seconds_ago": round(now - data["last_seen"], 2),
            "info": data.get("info"),
            "current_task": data.get("current_task")
        }
    return swarm_report

@app.get("/nodes")
def get_nodes(agent: dict = Depends(get_current_agent)):
    """جلب قائمة العقد وحالتها (متوافق مع Rescue Protocol)."""
    check_permission(agent, "read")
    now = time.time()
    nodes_list = []
    for agent_id, data in metrics.node_status.items():
        last_seen_diff = now - data["last_seen"]
        status = "online"
        if last_seen_diff > 15:
            status = "failed"
        elif last_seen_diff > 10:
            status = "warning"
            
        nodes_list.append({
            "agent_id": agent_id,
            "status": status,
            "last_seen": round(last_seen_diff, 2),
            "current_task": data.get("current_task")
        })
    return nodes_list

@app.post("/nodes/{target_id}/status")
def update_node_status(target_id: str, payload: Dict[str, str], agent: dict = Depends(get_current_agent)):
    """تحديث حالة عقدة معينة (مثلاً عند الهجرة)."""
    check_permission(agent, "write")
    if target_id in metrics.node_status:
        metrics.node_status[target_id]["status"] = payload.get("status", "unknown")
        return {"status": "updated"}
    raise HTTPException(status_code=404, detail="Node not found")

@app.post("/checkpoint/{task_id}")
def save_checkpoint(task_id: str, checkpoint: Dict[str, Any], agent: dict = Depends(get_current_agent)):
    """حفظ نقطة تفتيش لمهمة معينة."""
    check_permission(agent, "write")
    if "checkpoints" not in memory.data:
        memory.data["checkpoints"] = {}
    memory.data["checkpoints"][task_id] = {
        "data": checkpoint,
        "ts": time.time(),
        "agent_id": agent["agent_id"]
    }
    memory.save()
    return {"status": "saved"}

@app.get("/checkpoint/{task_id}")
def get_checkpoint(task_id: str, agent: dict = Depends(get_current_agent)):
    """جلب آخر نقطة تفتيش لمهمة معينة."""
    check_permission(agent, "read")
    checkpoints = memory.data.get("checkpoints", {})
    if task_id in checkpoints:
        return checkpoints[task_id]["data"]
    raise HTTPException(status_code=404, detail="Checkpoint not found")

@app.post("/consensus/lock")
def acquire_resource_lock(req: LockRequest, agent: dict = Depends(get_current_agent)):
    """طلب قفل موزّع لمورد (مثل إنقاذ وكيل أو تعديل كود)."""
    check_permission(agent, "write")
    success = metrics.acquire_lock(req.resource_id, req.agent_id, req.ttl)
    if success:
        return {"status": "locked", "resource": req.resource_id}
    return {"status": "denied", "reason": "Resource is currently locked by another agent"}

@app.post("/consensus/unlock")
def release_resource_lock(req: LockRequest, agent: dict = Depends(get_current_agent)):
    """تحرير قفل موزّع."""
    check_permission(agent, "write")
    metrics.release_lock(req.resource_id, req.agent_id)
    return {"status": "unlocked"}

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
    """التصويت المرجح بالثقة على أداة ومراجعتها من قبل الأقران."""
    check_permission(agent, "write")
    tools = memory.data.get("shared_tools", {})
    if v.tool_id not in tools:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    tool = tools[v.tool_id]
    
    # منع التصويت المتكرر من نفس الوكيل
    if any(r["reviewer"] == agent["agent_id"] for r in tool.get("reviews", [])):
        raise HTTPException(status_code=400, detail="Agent has already voted for this tool")

    # جلب نقاط الثقة للوكيل (افتراضي 1.0 للأدوار العادية، 5.0 للأدمن)
    agent_trust = 5.0 if agent["role"] == "admin" else memory.data["trust_scores"].get(agent["agent_id"], 1.0)
    
    # تحديث التصويت المرجح
    vote_weight = agent_trust
    if v.vote == "up":
        tool["votes"]["up"] += vote_weight
    else:
        tool["votes"]["down"] += vote_weight
        
    # إضافة مراجعة مفصلة
    tool["reviews"].append({
        "reviewer": agent["agent_id"],
        "vote": v.vote,
        "comment": v.comment,
        "timestamp": time.time(),
        "trust_score": agent_trust
    })
    
    # 🆕 معيار الاعتماد الموزون: مجموع أوزان الثقة >= 10.0
    if tool["status"] == "pending":
        author_id = tool.get("author")
        if tool["votes"]["up"] >= 10.0:
            tool["status"] = "approved"
            # 🎁 مكافأة المؤلف: زيادة الثقة بمقدار 0.5 عند قبول أداة
            if author_id:
                current_score = memory.data["trust_scores"].get(author_id, 1.0)
                memory.data["trust_scores"][author_id] = min(current_score + 0.5, 10.0)
        elif tool["votes"]["down"] >= 5.0:
            tool["status"] = "rejected"
            # ⚠️ جزاء المؤلف: خفض الثقة بمقدار 0.2 عند رفض أداة
            if author_id:
                current_score = memory.data["trust_scores"].get(author_id, 1.0)
                memory.data["trust_scores"][author_id] = max(current_score - 0.2, 0.1)
        
    memory.save()
    return {
        "status": "success", 
        "current_status": tool["status"], 
        "weighted_votes": tool["votes"],
        "your_trust_weight": agent_trust,
        "author_new_trust": memory.data["trust_scores"].get(tool.get("author"), 1.0) if tool.get("author") else None
    }

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

def start_memory_server(port: int = 8080):
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    start_memory_server(port)

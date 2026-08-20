"""
Neural Service Mesh (NSM) — Vercel FastAPI Edition
واجهة خفيفة وسريعة بدون Streamlit
"""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import sys
import json
import time
import asyncio
import httpx

# ── Configuration ────────────────────────────────────────────────
GITHUB_TOKEN = os.getenv("NSM_GITHUB_TOKEN", "")
GITHUB_API = "https://api.github.com"
REPO = "aliahmed369000000-ai/Neural-Service-Mesh"
BRANCH = "main"

app = FastAPI(title="NSM Lite", version="2.0")

# ── Models ───────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    message: str
    history: List[Dict[str, str]] = []

class AgentRequest(BaseModel):
    agent: str
    action: str
    params: Optional[Dict] = {}

# ── LLM Provider (Groq - مجاني وسريع) ──────────────────────────
GROQ_API_KEY = os.getenv("NSM_GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

async def call_groq(message: str, history: List[Dict] = None) -> str:
    """استدعاء Groq API (gpt-oss-120b مجاني)"""
    if not GROQ_API_KEY:
        return "Groq API key not configured"
    
    messages = [
        {"role": "system", "content": "أنت مساعد ذكي في مشروع Neural Service Mesh (NSM) — ذكاء اصطناعي عربي. أجب بالعربية."},
    ]
    
    if history:
        for h in history[-5:]:
            messages.append(h)
    messages.append({"role": "user", "content": message})
    
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/gpt-oss-120b",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024,
            },
        )
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "لا توجد استجابة")

# ── GitHub API ───────────────────────────────────────────────────
async def get_repo_info() -> Dict:
    """جلب معلومات المستودع"""
    if not GITHUB_TOKEN:
        return {"error": "GitHub token not configured"}
    
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{GITHUB_API}/repos/{REPO}",
            headers={"Authorization": f"Bearer {GITHUB_TOKEN}"},
        )
        if resp.status_code != 200:
            return {"error": f"GitHub API error: {resp.status_code}"}
        data = resp.json()
        return {
            "name": data.get("name"),
            "stars": data.get("stargazers_count"),
            "forks": data.get("forks_count"),
            "last_commit": data.get("pushed_at"),
            "description": data.get("description"),
        }

async def get_recent_commits(count: int = 10) -> List[Dict]:
    """جلب آخر commits"""
    if not GITHUB_TOKEN:
        return [{"message": "GitHub token not configured"}]
    
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{GITHUB_API}/repos/{REPO}/commits?per_page={count}",
            headers={"Authorization": f"Bearer {GITHUB_TOKEN}"},
        )
        if resp.status_code != 200:
            return [{"message": f"Error: {resp.status_code}"}]
        commits = resp.json()
        return [
            {
                "sha": c["sha"][:7],
                "message": c["commit"]["message"],
                "author": c["commit"]["author"]["name"],
                "date": c["commit"]["author"]["date"],
            }
            for c in commits
        ]

async def get_file_tree() -> Dict:
    """جلب بنية المستودع"""
    if not GITHUB_TOKEN:
        return {"error": "GitHub token not configured"}
    
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            f"{GITHUB_API}/repos/{REPO}/git/trees/{BRANCH}?recursive=1",
            headers={"Authorization": f"Bearer {GITHUB_TOKEN}"},
        )
        if resp.status_code != 200:
            return {"error": f"Error: {resp.status_code}"}
        data = resp.json()
        files = [t["path"] for t in data.get("tree", []) if t["type"] == "blob"]
        
        # تنظيم الملفات في مجلدات
        tree = {}
        for f in files:
            parts = f.split("/")
            current = tree
            for p in parts[:-1]:
                if p not in current:
                    current[p] = {}
                current = current[p]
            if parts[-1] not in current:
                current[parts[-1]] = None
        
        return {"files_count": len(files), "tree": tree}

# ── Routes ───────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """الصفحة الرئيسية"""
    return FileResponse("public/index.html")

@app.post("/api/chat")
async def chat(msg: ChatMessage):
    """المحادثة مع NSM"""
    start = time.time()
    response = await call_groq(msg.message, msg.history)
    elapsed = time.time() - start
    return {
        "response": response,
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "elapsed_ms": round(elapsed * 1000),
    }

@app.get("/api/status")
async def status():
    """حالة المشروع"""
    repo_info = await get_repo_info()
    return {
        "status": "online",
        "platform": "vercel",
        "version": "2.0",
        "repo": repo_info,
        "timestamp": time.time(),
    }

@app.get("/api/commits")
async def commits(count: int = 10):
    """آخر commits"""
    return await get_recent_commits(count)

@app.get("/api/files")
async def files():
    """بنية المستودع"""
    return await get_file_tree()

@app.get("/api/agents")
async def agents():
    """قائمة الوكلاء"""
    return {
        "agents": [
            {"name": "ResearchAgent", "desc": "بحث وتحليل", "status": "نشط"},
            {"name": "TranslationAgent", "desc": "ترجمة بين اللغات", "status": "نشط"},
            {"name": "CodeAgent", "desc": "كتابة وتعديل الكود", "status": "نشط"},
            {"name": "GitAgent", "desc": "إدارة GitHub", "status": "نشط"},
            {"name": "KaggleAgent", "desc": "إدارة Kaggle kernels", "status": "نشط"},
            {"name": "WebAgent", "desc": "تصفح الإنترنت", "status": "نشط"},
            {"name": "AutoHealAgent", "desc": "إصلاح الأخطاء تلقائياً", "status": "نشط"},
            {"name": "AutoReplyAgent", "desc": "رد تلقائي على الإيميلات", "status": "نشط"},
            {"name": "NotificationAgent", "desc": "إرسال التنبيهات", "status": "نشط"},
            {"name": "TrainingAgent", "desc": "إدارة التدريب", "status": "نشط"},
        ]
    }

@app.post("/api/agent")
async def agent_action(req: AgentRequest):
    """تنفيذ إجراء وكيل"""
    return {
        "agent": req.agent,
        "action": req.action,
        "status": "executing",
        "message": f"تنفيذ {req.action} بواسطة {req.agent}...",
    }

@app.get("/api/training")
async def training_status():
    """حالة التدريب"""
    return {
        "kernels": [
            {"slug": "nsm-d6144-tpu-v15", "status": "ERROR", "note": "Kaggle TPU DNS محظور"},
            {"slug": "nsm-d256-gpu-v9", "status": "completed", "note": "53 epochs, loss=0.5372"},
            {"slug": "nsm-d8192-tpu-v2", "status": "ERROR", "note": "needs fix"},
        ],
        "note": "خط أنابيب التدريب يحتاج إصلاح — Kaggle TPU يمنع DNS الصادرة",
    }

@app.get("/api/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}

"""
Neural Service Mesh — FastAPI Backend
خادم API لمشروع النظام المعرفي العربي
يعمل على المنفذ 5000 عبر Uvicorn
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# إضافة مسار المشروع
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# محاولة استيراد المكونات الداخلية
try:
    from core.engine import Engine
    from core.registry import Registry
    _CORE_OK = True
except Exception as _e:
    _CORE_OK = False
    _CORE_ERR = str(_e)

app = FastAPI(
    title="Neural Service Mesh API",
    description="واجهة برمجية للنظام المعرفي العربي",
    version="1.0.0",
)

# السماح بجميع الاتصالات (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "service": "Neural Service Mesh API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "core_available": _CORE_OK,
        "core_error": None if _CORE_OK else _CORE_ERR,
    }


@app.post("/process")
async def process(payload: dict):
    """معالجة النص عبر شبكة الخدمات"""
    if not _CORE_OK:
        return JSONResponse(
            status_code=503,
            content={"error": "Core engine not available", "detail": _CORE_ERR},
        )
    try:
        engine = Engine()
        result = engine.process(payload)
        return {"status": "ok", "result": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=5000, reload=True)

"""
mcp_server.py
==============
خادم Neural Service Mesh كمزوّد MCP (Model Context Protocol) عبر HTTP/SSE.

يتيح لأي عميل MCP خارجي (Claude، Gemini، أو تطبيق مخصص) الاتصال بمحرك
الأسئلة والأجوبة القرآني في NSM عبر الإنترنت واستدعائه كأداة (tool).

هذا خادم مستقل عن تطبيق Streamlit — يعمل كعملية (process) منفصلة على
منفذها الخاص، ويحتاج استضافة دائمة التشغيل (مثل Replit Deployments أو
Render/Railway)؛ Streamlit Community Cloud وحده لا يكفي لاستضافته.

التشغيل محلياً:
    pip install -r requirements.txt
    python mcp_server.py

المتغيرات البيئية:
    MCP_HOST  — عنوان الاستماع (افتراضي 0.0.0.0)
    MCP_PORT  — المنفذ (افتراضي 8800)
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from mcp_auth import check_api_key

# ── إعداد مسارات الاستيراد (نفس منهجية streamlit_app.py) ───────────────────
BASE = Path(__file__).parent
KNOWLEDGE_DIR = BASE / "knowledge"

_KNOWLEDGE_MODULE_DIR = str(KNOWLEDGE_DIR)
if _KNOWLEDGE_MODULE_DIR not in sys.path:
    sys.path.insert(0, _KNOWLEDGE_MODULE_DIR)

from qa_engine import answer_question  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# تحميل البيانات المعرفية — نفس منطق streamlit_app.py لكن بدون @st.cache_data
# (خادم MCP يعمل بمعزل عن Streamlit)، مع ذاكرة تخزين مؤقتة يدوية بمهلة TTL
# ═══════════════════════════════════════════════════════════════════════════

_CACHE_TTL_SECONDS = 300
_cache_lock = threading.Lock()
_cache: Dict[str, Any] = {"ckg": None, "ayat": None, "entities": None, "loaded_at": 0.0}


def _load_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_ckg() -> Dict[str, Any]:
    """تحميل الـ CKG — يعود بقاموس فارغ إذا كان الملف فارغاً أو Git LFS pointer."""
    empty = {"concepts": {}, "relations": {}}
    path = KNOWLEDGE_DIR / "cognitive_graph.json"
    try:
        content = path.read_text(encoding="utf-8").strip()
        if not content or content.startswith("version https://git-lfs"):
            return empty
        data = json.loads(content)
        if not isinstance(data, dict):
            return empty
        data.setdefault("concepts", {})
        data.setdefault("relations", {})
        return data
    except Exception:
        return empty


def _load_all_quran_ayat() -> List[Dict[str, Any]]:
    ayat: List[Dict[str, Any]] = []
    for cf in sorted(KNOWLEDGE_DIR.glob("quran_chunk_*.json")):
        chunk = _load_json(cf)
        if isinstance(chunk, list):
            ayat.extend(chunk)
    return ayat


def _load_entities() -> Dict[str, Any]:
    path = KNOWLEDGE_DIR / "entities.json"
    try:
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return {}
        data = json.loads(content)
        return data.get("entities", {}) if isinstance(data, dict) else {}
    except Exception:
        return {}


def _get_knowledge_data() -> tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    """يعيد (ckg, ayat, entities) من الذاكرة المؤقتة، ويعيد التحميل عند انتهاء TTL."""
    with _cache_lock:
        now = time.time()
        if now - _cache["loaded_at"] > _CACHE_TTL_SECONDS or _cache["ckg"] is None:
            _cache["ckg"] = _load_ckg()
            _cache["ayat"] = _load_all_quran_ayat()
            _cache["entities"] = _load_entities()
            _cache["loaded_at"] = now
        return _cache["ckg"], _cache["ayat"], _cache["entities"]


# ═══════════════════════════════════════════════════════════════════════════
# خادم MCP
# ═══════════════════════════════════════════════════════════════════════════

mcp = FastMCP(
    name="neural-service-mesh",
    instructions=(
        "خادم Neural Service Mesh (NSM) — يوفر أداة للإجابة على أسئلة "
        "المعرفة الإسلامية القرآنية بالاعتماد على رسم معرفي (CKG) "
        "وآيات القرآن الكريم مع درجة ثقة للإجابة."
    ),
    host=os.environ.get("MCP_HOST", "0.0.0.0"),
    port=int(os.environ.get("MCP_PORT", "8800")),
)


@mcp.tool()
def ask_islamic_knowledge(question: str, max_verses: int = 5, max_related: int = 8) -> dict:
    """
    يجيب على سؤال معرفة إسلامي/قرآني بالاعتماد على الرسم المعرفي (CKG)
    وآيات القرآن الكريم.

    Args:
        question: السؤال بالعربية (مثال: "ما هو الصبر؟" أو "من هو نوح؟").
        max_verses: أقصى عدد آيات داعمة تُعاد (افتراضي 5).
        max_related: أقصى عدد مفاهيم مرتبطة تُعاد (افتراضي 8).

    Returns:
        قاموس يحتوي الإجابة، الآيات الداعمة، المفاهيم المرتبطة، ودرجة الثقة.
    """
    ckg, ayat, entities = _get_knowledge_data()
    if not ckg.get("concepts"):
        return {
            "error": "الرسم المعرفي (CKG) غير محمّل على هذا الخادم — تحقق من ملف cognitive_graph.json",
        }
    return answer_question(
        question=question,
        ckg=ckg,
        ayat=ayat,
        max_verses=max_verses,
        max_related=max_related,
        entities=entities,
    )


@mcp.tool()
def nsm_knowledge_stats() -> dict:
    """يعيد إحصاءات سريعة عن حجم قاعدة المعرفة المحمّلة حالياً على الخادم."""
    ckg, ayat, entities = _get_knowledge_data()
    return {
        "concepts_count": len(ckg.get("concepts", {})),
        "relations_count": len(ckg.get("relations", {})),
        "ayat_count": len(ayat),
        "entities_count": len(entities),
    }


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """
    يتحقق من ترويسة X-API-Key قبل السماح بالوصول لأي مسار من مسارات MCP
    (/sse، /messages). يعيد 401 لمفتاح مفقود/غير صالح، و429 عند تجاوز
    الحد اليومي للخطة.
    """

    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get("x-api-key") or request.query_params.get("api_key")
        result = check_api_key(api_key)
        if not result.ok:
            return JSONResponse(
                {"error": result.reason}, status_code=result.status_code
            )
        return await call_next(request)


def build_app():
    """يبني تطبيق SSE مع طبقة المصادقة مثبّتة فوقه."""
    app = mcp.sse_app()
    app.add_middleware(ApiKeyMiddleware)
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        build_app(),
        host=os.environ.get("MCP_HOST", "0.0.0.0"),
        port=int(os.environ.get("MCP_PORT", "8800")),
    )

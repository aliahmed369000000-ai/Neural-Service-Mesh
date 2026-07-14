"""
NSM MCP Server
==============
يعرض أدوات Neural Service Mesh (NSM) كأدوات MCP قياسية، بحيث يقدر أي
عميل MCP (Claude Desktop، Claude Code، أو أي IDE يدعم MCP) يستخدمها
مباشرة بدون المرور عبر واجهة Streamlit.

الأدوات المعروضة حالياً:
  - quran_lookup   : جلب نص آية بعينها عبر رقم السورة ورقم الآية.
  - quran_search    : بحث نصّي عن آيات تحتوي كلمة/عبارة معيّنة.
  - classify_harm   : تصنيف نص عربي/إنجليزي حسب نطاق الأذى (مبني على
                       ai/harm_classifier.py الموجود بالفعل في المشروع).

التشغيل محلياً (stdio transport):
    python mcp_server/server.py

الإضافة إلى Claude Desktop (مثال ضمن claude_desktop_config.json):
    {
      "mcpServers": {
        "nsm": {
          "command": "python",
          "args": ["/path/to/Neural-Service-Mesh/mcp_server/server.py"]
        }
      }
    }
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

# إتاحة استيراد حزمة ai/ الموجودة في جذر المشروع
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mcp.server.fastmcp import FastMCP

from ai.harm_classifier import classify_prompt

_KNOWLEDGE_DIR = _ROOT / "knowledge"
_INDEX_FILE = _KNOWLEDGE_DIR / "quran_index.json"
_CHUNK_SIZE = 100  # يطابق chunk_size المخزّن في quran_index.json

mcp = FastMCP("nsm")


def _load_surah_index() -> dict:
    with open(_INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["surah_index"]


def _chunk_path(chunk_id: int) -> Path:
    return _KNOWLEDGE_DIR / f"quran_chunk_{chunk_id:04d}.json"


def _load_chunk(chunk_id: int) -> list:
    path = _chunk_path(chunk_id)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_all_ayat():
    """Generator كسول يمر على كل الآيات عبر كل الـchunks بالترتيب."""
    chunk_id = 0
    while True:
        path = _chunk_path(chunk_id)
        if not path.exists():
            break
        for item in _load_chunk(chunk_id):
            yield item
        chunk_id += 1


@mcp.tool()
def quran_lookup(surah: int, ayah: int) -> str:
    """جلب نص آية قرآنية بعينها عبر رقم السورة ورقم الآية.

    Args:
        surah: رقم السورة (1-114).
        ayah: رقم الآية داخل السورة.
    """
    surah_index = _load_surah_index()
    meta = surah_index.get(str(surah))
    if meta is None:
        return json.dumps({"error": f"رقم سورة غير صالح: {surah}"}, ensure_ascii=False)

    if ayah < 1 or ayah > meta["ayah_count"]:
        return json.dumps(
            {"error": f"رقم آية غير صالح لسورة {surah} (المدى المتاح: 1-{meta['ayah_count']})"},
            ensure_ascii=False,
        )

    # نمسح تسلسلياً بدءاً من أول chunk تظهر فيه السورة حتى نلقى الآية،
    # لأن آيات السورة الواحدة قد تمتد لأكثر من chunk واحد.
    chunk_id = meta["first_chunk"]
    while True:
        data = _load_chunk(chunk_id)
        if not data:
            break
        for item in data:
            if item["surah"] == surah and item["ayah"] == ayah:
                return json.dumps(
                    {
                        "surah": surah,
                        "ayah": ayah,
                        "text": item["text"],
                        "found": True,
                    },
                    ensure_ascii=False,
                )
        # لو تجاوزنا رقم السورة المطلوبة في نهاية الـchunk، توقف
        if data[-1]["surah"] > surah:
            break
        chunk_id += 1

    return json.dumps({"error": "لم يتم العثور على الآية", "found": False}, ensure_ascii=False)


@mcp.tool()
def quran_search(query: str, limit: int = 5) -> str:
    """بحث نصّي عن آيات قرآنية تحتوي كلمة أو عبارة معيّنة (بحث حرفي في النص المُطبَّع).

    Args:
        query: النص أو الكلمة المراد البحث عنها.
        limit: أقصى عدد نتائج تُرجَع (افتراضي 5).
    """
    if not query.strip():
        return json.dumps({"error": "النص المطلوب البحث عنه فارغ"}, ensure_ascii=False)

    results = []
    for item in _iter_all_ayat():
        if query in item.get("text_norm", "") or query in item.get("text", ""):
            results.append(
                {"surah": item["surah"], "ayah": item["ayah"], "text": item["text"]}
            )
            if len(results) >= limit:
                break

    return json.dumps({"query": query, "count": len(results), "results": results}, ensure_ascii=False)


@mcp.tool()
def classify_harm(text: str) -> str:
    """تصنيف نص حسب نطاق الأذى المحتمل (باستخدام مصنّف NSM الحالي المبني على regex).

    Args:
        text: النص المراد تصنيفه.
    """
    result = classify_prompt(text)
    return json.dumps(
        {
            "domain": result.domain,
            "subcategory": getattr(result, "subcategory", None),
            "confidence": getattr(result, "confidence", None),
            "is_sensitive": result.domain != "benign",
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")

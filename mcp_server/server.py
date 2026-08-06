"""
NSM MCP Server
==============
يعرض أدوات Neural Service Mesh (NSM) كأدوات MCP قياسية، بحيث يقدر أي
عميل MCP (Claude Desktop، Claude Code، أو أي IDE يدعم MCP) يستخدمها
مباشرة بدون المرور عبر واجهة Streamlit.

الأدوات المعروضة حالياً:
  - quran_lookup        : جلب نص آية بعينها عبر رقم السورة ورقم الآية.
  - quran_search         : بحث نصّي عن آيات تحتوي كلمة/عبارة معيّنة.
  - classify_harm        : تصنيف نص عربي/إنجليزي حسب نطاق الأذى (مبني على
                            ai/harm_classifier.py الموجود بالفعل في المشروع).
  - ask_nsm               : إرسال سؤال لوكيل NSM الكامل (ai/nsm_agent_core.py)
                            والحصول على رد — نفس الوكيل المستخدم في واجهة
                            Streamlit، بدون المرور عبرها.
  - search_ckg            : بحث دلالي في قاعدة المعرفة المعرفية (CKG) الخاصة
                            بالمشروع (7000+ مفهوم)، بمطابقة كلمة كاملة
                            (ai/ckg_text_encoder_v2.py) بدل الاحتواء الجزئي.
  - check_project_health  : تقرير جاهزية سريع عن حالة الكود (Phase6Validator):
                            نسبة تغطية المراحل، الكود الميت، عدد الملفات.
  - reasoning_answer      : مسار ReasoningPipeline + DeepRouting على سؤال عربي.
  - training_safety_check : بوابة نموذج العالم قبل run_training_loop.
  - knowledge_pulse       : نبضة حساسات + فجوات معرفية محتملة.

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


@mcp.tool()
def ask_nsm(query: str) -> str:
    """إرسال سؤال إلى وكيل NSM الكامل (نفس الوكيل المستخدم في واجهة Streamlit)
    والحصول على رده الكامل دفعة واحدة.

    ملاحظة: يحتاج الوكيل مفتاح API واحداً على الأقل مُعرَّفاً كمتغيّر بيئة
    (مثلاً GROQ_API_KEY أو GOOGLE_API_KEY) ليعمل فعلياً؛ بدون ذلك يرجع رسالة
    توضّح عدم التوفر بدل الفشل الصامت.

    Args:
        query: السؤال أو الطلب المراد إرساله للوكيل.
    """
    if not query.strip():
        return json.dumps({"error": "السؤال فارغ"}, ensure_ascii=False)

    try:
        from ai.nsm_agent_core import NSMAgent
    except Exception as e:
        return json.dumps({"error": f"تعذّر تحميل وكيل NSM: {e}"}, ensure_ascii=False)

    agent = NSMAgent()
    if not agent.available:
        return json.dumps(
            {
                "error": "لا يوجد مفتاح API مُعرَّف كمتغيّر بيئة "
                         "(GROQ_API_KEY / GOOGLE_API_KEY / OPENROUTER_API_KEY / "
                         "OPENAI_API_KEY / CF_API_TOKEN+CF_ACCOUNT_ID)",
                "available": False,
            },
            ensure_ascii=False,
        )

    try:
        answer = agent.run(query)
    except Exception as e:
        return json.dumps({"error": f"فشل الوكيل أثناء التنفيذ: {e}"}, ensure_ascii=False)

    return json.dumps({"query": query, "answer": answer}, ensure_ascii=False)


@mcp.tool()
def search_ckg(query: str, limit: int = 5) -> str:
    """بحث دلالي في قاعدة المعرفة المعرفية الخاصة بـ NSM (الجراف المعرفي CKG،
    آلاف المفاهيم المستخرجة من القرآن ومصادر عامة)، بمطابقة كلمة كاملة
    (لا احتواء جزئي — مثلاً البحث عن "علم" لا يطابق "يعلمون" خطأً).

    Args:
        query: كلمة أو عبارة عربية للبحث عن مفاهيم مرتبطة بها.
        limit: أقصى عدد نتائج تُرجَع (افتراضي 5).
    """
    if not query.strip():
        return json.dumps({"error": "نص البحث فارغ"}, ensure_ascii=False)

    try:
        from knowledge.cognitive_graph import get_ckg
        from ai.ckg_text_encoder_v2 import _tokenize, _word_level_score
    except Exception as e:
        return json.dumps({"error": f"تعذّر تحميل قاعدة المعرفة: {e}"}, ensure_ascii=False)

    ckg = get_ckg()
    q_words = _tokenize(query)

    scored = []
    for c in ckg.all_concepts():
        score = _word_level_score(q_words, c["name"])
        if score > 0:
            scored.append((score, c))

    scored.sort(key=lambda t: (t[0], t[1].get("strength", 0)), reverse=True)
    results = [
        {
            "concept": c["name"],
            "cluster": c.get("cluster"),
            "strength": c.get("strength"),
            "frequency": c.get("frequency"),
            "match_score": round(score, 3),
        }
        for score, c in scored[:limit]
    ]

    return json.dumps(
        {"query": query, "total_concepts": ckg.concept_count(), "count": len(results), "results": results},
        ensure_ascii=False,
    )


@mcp.tool()
def sensor_hub_status() -> str:
    """يشغّل استطلاعاً حياً (single poll) لطبقة الحساسات (Phase 7 Sensory
    Layer): FilesystemSensor (تغيّرات في ai/ وservices/) وLogSensor (أنماط
    ERROR/CRITICAL/Exception في logs/). كل حدث يُغذّى فعلياً إلى
    EnvironmentModel عبر ingest_sensor_event، ثم يُعاد ملخّص الحالة.

    لا يحتاج مفاتيح API ولا اتصال شبكة — قراءة قرص محلية فقط.
    """
    try:
        from sensors import SensorHub, FilesystemSensor, LogSensor
        from world_model import EnvironmentModel
    except Exception as e:
        return json.dumps({"error": f"تعذّر تحميل طبقة الحساسات: {e}"}, ensure_ascii=False)

    try:
        env_model = EnvironmentModel(model_dir=str(_ROOT / "world_model"))
        hub = SensorHub()
        hub.register(FilesystemSensor(config={
            "watch_paths": [str(_ROOT / "ai"), str(_ROOT / "services")],
        }))
        hub.register(LogSensor(config={"log_paths": [str(_ROOT / "logs")]}))
        hub.on_event(lambda e: env_model.ingest_sensor_event(e.to_dict()))

        events = hub.poll_now()

        summary = hub.summary()
        summary["latest_events"] = [e.to_dict() for e in events[-10:]]
        summary["environment_alerts_after_ingest"] = len(env_model.get_sensor_alerts(limit=10))
        return json.dumps(summary, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"فشل تشغيل استطلاع الحساسات: {e}"}, ensure_ascii=False)


@mcp.tool()
def check_project_health() -> str:
    """تقرير جاهزية سريع عن حالة كود مشروع NSM: عدد الملفات، نسبة تغطية
    المراحل (Phase coverage)، نسبة الكود الميت (وحدات غير مستوردة من أي
    مكان)، وتوصيات. لا يحتاج أي بيانات تشغيل حيّة (mesh) — تحليل كود ثابت.
    """
    try:
        from ai.validator import Phase6Validator
    except Exception as e:
        return json.dumps({"error": f"تعذّر تحميل المُدقّق: {e}"}, ensure_ascii=False)

    # 🆕 تمرير mesh bundle الحي الفعلي (registry/memory/scoring/reputation/
    # agent_factory/swarm/DNA المشترَكة مع واجهة Streamlit) بدل mesh=None —
    # كان هذا يجعل قسم "Live system" في التقرير يُظهر أصفاراً دائماً حتى لو
    # كانت هناك عُقد ووكلاء مسجَّلون فعلياً.
    try:
        from core.mesh_bundle import get_mesh_bundle
        _mesh = get_mesh_bundle()
    except Exception:
        _mesh = None

    validator = Phase6Validator(mesh=_mesh, project_root=str(_ROOT))
    report = validator.generate()

    summary = {
        "files_total": report.get("files", {}).get("total_py_files"),
        "lines_of_code": report.get("files", {}).get("total_lines_of_code"),
        "phase_coverage_pct": report.get("phase_coverage", {}).get("overall_coverage_pct"),
        "dead_code_pct": report.get("dead_code", {}).get("dead_pct"),
        "dead_files_count": report.get("dead_code", {}).get("dead_count"),
        "phase7_readiness_score": report.get("phase7_readiness", {}).get("score"),
        "verdict": report.get("phase7_readiness", {}).get("verdict"),
        "recommendations": report.get("phase7_readiness", {}).get("recommendations", []),
    }
    return json.dumps(summary, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")


@mcp.tool()
def reasoning_answer(question: str, train_on_query: bool = False) -> str:
    """يشغّل ReasoningPipeline (مع DeepRouting إن توفّر) على سؤال عربي
    ويعيد ملخص الإجابة والمفاهيم المرتبة. لا يستبدل ask_nsm بل يفعّل
    مسار CKG→NeuralCore مباشرة للعملاء الخارجيين (Claude/Cursor).
    """
    try:
        from ai.reasoning_pipeline import ReasoningPipeline
        pipe = ReasoningPipeline(train_on_query=bool(train_on_query), use_deep_routing=True)
        result = pipe.answer(question or "")
        payload = {
            "answer": getattr(result, "answer_text", None) or str(result),
            "weights": getattr(result, "decision_weights", {}),
            "ranked": (getattr(result, "ranked_concepts", None) or [])[:8],
            "deep_routing": bool((getattr(result, "decision_weights", {}) or {}).get("_deep_routing")),
        }
        return json.dumps(payload, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": f"reasoning_answer failed: {e}"}, ensure_ascii=False)


@mcp.tool()
def training_safety_check(action: str = "run_training_loop", estimated_vram_mb: int = 4096) -> str:
    """يسأل نموذج العالم: هل تنفيذ تدريب ثقيل آمن الآن؟"""
    try:
        from world_model.environment_model import EnvironmentModel
        env = EnvironmentModel(model_dir=str(_ROOT / "world_model"))
        report = env.assess_training_safety(action=action, estimated_vram_mb=int(estimated_vram_mb))
        return json.dumps(report, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e), "green_light": False}, ensure_ascii=False)


@mcp.tool()
def knowledge_pulse() -> str:
    """نبضة حساسات + إشارة فجوات معرفية لحلقة التعلم النشط."""
    try:
        from ai.sovereignty_loop import knowledge_pulse as _pulse
        return json.dumps(_pulse(), ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def campaign_simulate(text: str, platforms: str = "twitter,linkedin,telegram") -> str:
    """محاكاة تفاعل الجمهور وتكلفة الميزانية قبل النشر/الحملة."""
    try:
        from world_model.predictive_sim import full_campaign_sim
        plats = [p.strip() for p in (platforms or "").split(",") if p.strip()]
        return json.dumps(full_campaign_sim(text, plats or None), ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


@mcp.tool()
def mcp_auth_check(api_key: str = "") -> str:
    """التحقق من مفتاح مستأجر AIaaS لاستخدام MCP المدفوع."""
    try:
        from mcp_server.monetization import authenticate_mcp_key
        ok, meta = authenticate_mcp_key(api_key or None)
        return json.dumps({"ok": ok, **meta}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False)

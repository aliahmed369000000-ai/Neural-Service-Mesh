"""
Higgsfield Explainer Engine — NSM v1.0
=======================================
يُنتج فيديو وثائقياً (حتى 10 دقائق) من موضوع نصي عبر pipeline ثلاثي المراحل:

  المرحلة 1 — Gemini Omni Flash:
      يبحث في المعلومات ويبني هيكل مشاهد الوثائقي (outline + حقائق رئيسية).

  المرحلة 2 — Nova Fable 5 (Claude Fable 5):
      يصيغ نص السرد الصوتي + video prompt احترافي لكل مشهد بأسلوب سينمائي.

  المرحلة 3 — Higgsfield API:
      يُرسل كل مشهد إلى Higgsfield لتوليد مقطع فيديو قصير (3-8 ثوانٍ).
      يستطلع حالة الجلسة بشكل غير متزامن حتى تكتمل.

الاستخدام:
    from ai.higgsfield_engine import HiggsfieldEngine, build_gemini_llm, build_fable_llm

    engine = HiggsfieldEngine(
        gemini_llm=build_gemini_llm(),
        fable_llm=build_fable_llm(),
    )
    result = engine.create_documentary("تاريخ طريق الحرير", target_minutes=5)
    for scene in result.scenes:
        print(scene.narration)
        print(scene.video_url or scene.video_status)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional

logger = logging.getLogger("HiggsfieldEngine")


# ══════════════════════════════════════════════════════════════════════════
# نماذج اللغة المتخصصة
# ══════════════════════════════════════════════════════════════════════════

_GEMINI_FLASH_MODEL = "gemini-2.0-flash"       # Gemini Omni Flash — البحث
_FABLE_MODEL_KEY    = "fable"                   # claude-fable-5 — السرد الإبداعي


class _PinnedLLM:
    """
    غلاف يُثبّت مزوّداً بعينه (Gemini أو Claude Fable) بدلاً من الاعتماد
    على ترتيب الـ chain الاعتيادي في LLMFallback.generate().

    المنطق:
    - يحتفظ بنسخة LLMFallback مُهيَّأة للمزوّد المستهدف.
    - يستدعي دالة _call_XXX مباشرةً (بدلاً من generate() التي تُعيد بناء
      الـ chain من متغيرات البيئة في كل مرة).
    - إن فشل المزوّد الأوّل يسقط إلى fallback_llm العادي.
    """

    def __init__(self, primary_fn, fallback_llm, provider_label: str):
        # primary_fn: callable(query, history, system_prompt) → FallbackResult | None
        self._primary       = primary_fn
        self._fallback      = fallback_llm
        self._provider_label = provider_label

    def generate(self, query: str, history: list, system_prompt: str = ""):
        if self._primary is not None:
            try:
                return self._primary(query, history, system_prompt)
            except Exception as exc:
                logger.warning(
                    "المزوّد المثبَّت [%s] فشل: %s — ينتقل إلى الـ fallback",
                    self._provider_label, exc,
                )
        return self._fallback.generate(query, history=history, system_prompt=system_prompt)

    @property
    def provider_label(self) -> str:
        return self._provider_label


def build_gemini_llm() -> _PinnedLLM:
    """
    يُنشئ _PinnedLLM يستدعي Gemini Flash مباشرةً إن كان GOOGLE_API_KEY
    متاحاً، وإلا يسقط إلى سلسلة الـ fallback الاعتيادية.
    """
    from ai.llm_fallback import LLMFallback
    fallback = LLMFallback()
    key = os.getenv("GOOGLE_API_KEY", "").strip()

    primary_fn = None
    if key:
        # نسخة LLMFallback خاصة بـ Gemini فقط — لا يُستخدم generate() عليها
        _gemini_instance       = LLMFallback()
        _gemini_instance._api_key = key
        _gemini_instance._model   = _GEMINI_FLASH_MODEL
        primary_fn = _gemini_instance._call_gemini   # ← استدعاء مباشر، يتجاوز الـ chain

    return _PinnedLLM(primary_fn, fallback, "gemini-omni-flash")


def build_fable_llm() -> _PinnedLLM:
    """
    يُنشئ _PinnedLLM يستدعي claude-fable-5 (Nova Fable 5) مباشرةً إن
    كان ANTHROPIC_API_KEY متاحاً، وإلا يسقط إلى سلسلة الـ fallback.
    """
    from ai.llm_fallback import LLMFallback, ANTHROPIC_MODELS
    fallback = LLMFallback(model_key=_FABLE_MODEL_KEY)
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()

    primary_fn = None
    if key:
        _fable_instance          = LLMFallback(model_key=_FABLE_MODEL_KEY)
        _fable_instance._api_key = key
        _fable_instance._model   = ANTHROPIC_MODELS[_FABLE_MODEL_KEY]  # claude-fable-5
        primary_fn = _fable_instance._call_anthropic   # ← استدعاء مباشر

    return _PinnedLLM(primary_fn, fallback, "nova-fable-5")


# ══════════════════════════════════════════════════════════════════════════
# Dataclasses
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class VideoScene:
    """مشهد واحد في الوثائقي — نص + فيديو"""
    index:         int
    title:         str          = ""     # عنوان مختصر للمشهد
    narration:     str          = ""     # نص السرد الصوتي (عربي فصيح)
    visual_notes:  str          = ""     # توجيه مرئي عام
    video_prompt:  str          = ""     # prompt سينمائي لـ Higgsfield (إنجليزي)
    est_seconds:   int          = 30     # المدة التقديرية للمشهد
    job_id:        str          = ""     # معرّف مهمة Higgsfield
    video_url:     str          = ""     # رابط الفيديو المُولَّد
    video_status:  str          = "pending"  # pending / processing / completed / failed
    video_error:   str          = ""     # رسالة الخطأ إن فشل التوليد


@dataclass
class DocumentaryScript:
    """نتيجة المرحلتين 1+2: السيناريو الكامل قبل توليد الفيديو"""
    topic:        str
    title:        str
    synopsis:     str          = ""
    scenes:       List[VideoScene] = field(default_factory=list)
    research_provider: str     = ""
    narrative_provider: str    = ""
    error:        str          = ""

    @property
    def total_seconds(self) -> int:
        return sum(s.est_seconds for s in self.scenes)

    @property
    def full_narration(self) -> str:
        return "\n\n".join(
            f"[المشهد {s.index}: {s.title}]\n{s.narration}"
            for s in self.scenes
        )


@dataclass
class HiggsfieldResult:
    """النتيجة النهائية بعد توليد الفيديو"""
    script:       DocumentaryScript
    scenes_done:  int = 0
    scenes_total: int = 0
    api_used:     bool = False

    @property
    def scenes(self) -> List[VideoScene]:
        return self.script.scenes

    @property
    def progress_pct(self) -> float:
        if not self.scenes_total:
            return 0.0
        return self.scenes_done / self.scenes_total * 100


# ══════════════════════════════════════════════════════════════════════════
# Higgsfield API Client
# ══════════════════════════════════════════════════════════════════════════

_HF_BASE_URL      = "https://api.higgsfield.ai"
_HF_TIMEOUT       = 30   # ثانية
_HF_POLL_INTERVAL = 5    # استطلاع كل 5 ثوانٍ
_HF_MAX_WAIT      = 180  # حد أقصى للانتظار لكل مشهد (3 دقائق)
_HF_CLIP_DURATION = 6    # مدة المقطع الواحد بالثواني


def _hf_post(endpoint: str, body: dict, api_key: str) -> dict:
    url  = f"{_HF_BASE_URL}{endpoint}"
    data = json.dumps(body).encode("utf-8")
    req  = urllib.request.Request(
        url, data=data,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept":        "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_HF_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _hf_get(endpoint: str, api_key: str) -> dict:
    url = f"{_HF_BASE_URL}{endpoint}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept":        "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=_HF_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


class HiggsfieldClient:
    """
    عميل HTTP لـ Higgsfield API.
    يُغلّف استدعاءات توليد الفيديو واستطلاع الحالة.
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("HIGGSFIELD_API_KEY غير موجود في البيئة")
        self._key = api_key

    def submit_job(self, prompt: str, duration: int = _HF_CLIP_DURATION) -> str:
        """
        يُرسل طلب توليد فيديو. يعيد job_id.
        يرفع استثناءً فورياً إن لم يعُد الخادم معرّف مهمة صالحاً.
        """
        duration = max(3, min(int(duration), 8))
        body = {
            "prompt":       prompt,
            "model":        "higgsfield-video-1",
            "duration":     duration,
            "style":        "cinematic",
            "aspect_ratio": "16:9",
        }
        resp   = _hf_post("/api/v1/videos", body, self._key)
        job_id = resp.get("id") or resp.get("job_id") or ""
        if not job_id:
            raise RuntimeError(
                f"Higgsfield لم يُعِد job_id صالحاً — الرد الكامل: {resp}"
            )
        return job_id

    def poll_job(self, job_id: str, max_wait: int = _HF_MAX_WAIT) -> VideoScene:
        """
        يستطلع حالة مهمة حتى تكتمل أو تفشل أو ينتهي الوقت.

        - أخطاء 4xx (auth/not-found/bad-request) → فشل فوري (terminal).
        - أخطاء شبكة/5xx → يُعيد المحاولة حتى انتهاء المهلة.
        - يعيد VideoScene جزئي (video_url + video_status فقط).
        """
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                resp   = _hf_get(f"/api/v1/videos/{job_id}", self._key)
                status = resp.get("status", "pending").lower()

                if status in ("completed", "done", "success"):
                    url = (
                        resp.get("output", {}).get("url")
                        or resp.get("video_url")
                        or resp.get("url", "")
                    )
                    return VideoScene(
                        index=0, video_url=url,
                        video_status="completed", job_id=job_id,
                    )
                if status in ("failed", "error"):
                    err = resp.get("error") or resp.get("message", "خطأ غير محدد")
                    return VideoScene(
                        index=0, video_status="failed",
                        video_error=str(err), job_id=job_id,
                    )
                # حالة "processing / pending / running" — نواصل الاستطلاع

            except urllib.error.HTTPError as http_exc:
                # 4xx = خطأ في الطلب نفسه، لا فائدة من إعادة المحاولة
                if 400 <= http_exc.code < 500:
                    err_body = ""
                    try:
                        err_body = http_exc.read().decode("utf-8", errors="replace")
                    except Exception:
                        pass
                    return VideoScene(
                        index=0, video_status="failed",
                        video_error=f"HTTP {http_exc.code}: {err_body[:200]}",
                        job_id=job_id,
                    )
                # 5xx أو مشاكل شبكة → نُعيد المحاولة
                logger.debug("Higgsfield poll_job HTTP %d — سنُعيد المحاولة", http_exc.code)

            except Exception as exc:
                logger.debug("Higgsfield poll_job خطأ شبكة: %s — سنُعيد المحاولة", exc)

            time.sleep(_HF_POLL_INTERVAL)

        return VideoScene(
            index=0, video_status="timeout",
            video_error=f"انتهت مهلة الانتظار ({max_wait}s)",
            job_id=job_id,
        )


# ══════════════════════════════════════════════════════════════════════════
# System Prompts
# ══════════════════════════════════════════════════════════════════════════

_RESEARCH_SP = """\
أنت باحث ومنظّم محتوى وثائقي متخصص. مهمتك: استقبال موضوع من المستخدم
وتوليد بنية مشاهد وثائقية مُفصَّلة تصلح لفيديو وثائقي فاخر.

قواعد صارمة:
- لا تختلق حقائق غير مؤكدة — قل "ادّعي المؤرخون" أو "تشير الروايات" إن لزم.
- ركّز على المعلومات المثيرة للاهتمام والأقل شهرة.
- كل مشهد يجب أن يكون وحدة قصصية مكتملة — لا مجرد جمل معلّقة.

التنسيق المطلوب (التزم به حرفياً):
### المشهد N
العنوان: <عنوان قصير للمشهد بالعربية>
المحتوى: <3-5 جمل عربية تصف المعلومة/الحدث/الفكرة الرئيسية للمشهد>
المشهد المرئي: <وصف مختصر بالعربية للصورة المرئية المناسبة (طبيعة، بنية، خريطة، إلخ)>
المدة: <مدة المشهد بالثواني — بين 20-60 ثانية>
"""

_NARRATIVE_SP = """\
أنت كاتب سيناريو وثائقي إبداعي بارع — أسلوبك يجمع بين دقة الوثائقي وحرارة القصص.
مهمتك: تحويل مخطط المشهد إلى نص سرد صوتي فصيح + video prompt سينمائي بالإنجليزية.

للنص السردي:
- فصحى رشيقة، إيقاع حيّ، كثافة معلومات عالية — بلا حشو
- ابدأ بجملة خاطفة تُشعل فضول المشاهد
- ~130 كلمة عربية لكل 60 ثانية

للـ video prompt (بالإنجليزية):
- أسلوب سينمائي cinematic documentary
- صِف: الزاوية (overhead/close-up/wide)، الإضاءة، الحركة (slow motion/timelapse/aerial)
- 1-2 جملة مكثّفة — لا قوائم طويلة

التنسيق (التزم به حرفياً):
السرد: <نص السرد الصوتي بالعربية الفصحى>
الـ Prompt: <cinematic video prompt in English>
"""


# ══════════════════════════════════════════════════════════════════════════
# HiggsfieldEngine — المحرك الرئيسي
# ══════════════════════════════════════════════════════════════════════════

class HiggsfieldEngine:
    """
    محرك Higgsfield Explainer — ثلاث مراحل:
      1. Gemini Flash   → بحث وتنظيم المشاهد
      2. Nova Fable 5   → صياغة السرد + video prompts
      3. Higgsfield API → توليد الفيديو لكل مشهد
    """

    def __init__(self, gemini_llm, fable_llm, higgsfield_key: str = ""):
        self._gemini  = gemini_llm
        self._fable   = fable_llm
        self._hf_key  = higgsfield_key or os.getenv("HIGGSFIELD_API_KEY", "").strip()
        self._hf_client: Optional[HiggsfieldClient] = None
        if self._hf_key:
            try:
                self._hf_client = HiggsfieldClient(self._hf_key)
            except ValueError:
                pass

    # ── المرحلة 1: البحث والتنظيم ─────────────────────────────────────

    def research_outline(self, topic: str, target_minutes: int) -> tuple[str, str]:
        """
        يستدعي Gemini Flash لتوليد هيكل المشاهد.
        يعيد (raw_text, provider_name).
        """
        n_scenes = max(3, target_minutes * 2)  # ~30 ثانية لكل مشهد
        query = (
            f"موضوع الوثائقي: {topic}\n"
            f"المدة المستهدفة: {target_minutes} دقيقة\n"
            f"عدد المشاهد المطلوب: {n_scenes} مشهداً تقريباً"
        )
        result = self._gemini.generate(query, history=[], system_prompt=_RESEARCH_SP)
        provider = getattr(result.provider, "value", str(result.provider))
        return result.text.strip(), provider

    # ── المرحلة 2: صياغة السرد ────────────────────────────────────────

    def craft_narration(self, scene_outline: str, scene_index: int) -> tuple[str, str, str]:
        """
        يستدعي Claude Fable 5 لصياغة نص السرد + video prompt لمشهد واحد.
        يعيد (narration, video_prompt, provider_name).
        """
        query = (
            f"المشهد {scene_index}:\n{scene_outline}\n\n"
            "اكتب نص السرد الصوتي العربي والـ video prompt بالإنجليزية للمشهد أعلاه."
        )
        result = self._fable.generate(query, history=[], system_prompt=_NARRATIVE_SP)
        provider = getattr(result.provider, "value", str(result.provider))

        raw = result.text.strip()
        narration   = _extract_field(raw, "السرد")
        video_prompt = _extract_field(raw, "الـ Prompt")
        if not narration:
            narration = raw  # في حال لم يلتزم النموذج بالتنسيق
        if not video_prompt:
            video_prompt = f"cinematic documentary, {scene_outline[:120]}"

        return narration, video_prompt, provider

    # ── المرحلة 3: توليد الفيديو ──────────────────────────────────────

    def generate_video_for_scene(self, scene: VideoScene) -> VideoScene:
        """
        يُرسل scene.video_prompt إلى Higgsfield ويُحدّث scene بالنتيجة.
        إن لم يكن المفتاح متاحاً يُعيد scene بحالة no_api.
        """
        if not self._hf_client:
            scene.video_status = "no_api"
            scene.video_error  = "HIGGSFIELD_API_KEY غير موجود"
            return scene

        try:
            job_id = self._hf_client.submit_job(
                scene.video_prompt,
                duration=min(_HF_CLIP_DURATION, scene.est_seconds),
            )
            scene.job_id      = job_id
            scene.video_status = "processing"

            poll_result = self._hf_client.poll_job(job_id)
            scene.video_url    = poll_result.video_url
            scene.video_status = poll_result.video_status
            scene.video_error  = poll_result.video_error

        except Exception as exc:
            scene.video_status = "failed"
            scene.video_error  = str(exc)
            logger.warning("Higgsfield توليد فيديو فشل للمشهد %d: %s", scene.index, exc)

        return scene

    # ── الدالة الرئيسية ───────────────────────────────────────────────

    def create_documentary(
        self,
        topic: str,
        target_minutes: int = 5,
        style: str = "وثائقي",
        generate_video: bool = True,
        progress_cb: Optional[Callable[[str, float], None]] = None,
    ) -> HiggsfieldResult:
        """
        يُنشئ الوثائقي الكامل.

        Args:
            topic:           موضوع الوثائقي
            target_minutes:  المدة المستهدفة (1-10 دقائق)
            style:           النوع (وثائقي / تاريخي / علمي / ثقافي / طبيعي)
            generate_video:  هل نستدعي Higgsfield لتوليد الفيديو؟
            progress_cb:     دالة callback(message, pct) للتقدم المباشر

        Returns:
            HiggsfieldResult
        """
        target_minutes = max(1, min(int(target_minutes), 10))

        def _prog(msg: str, pct: float):
            if progress_cb:
                progress_cb(msg, pct)
            logger.info("[%.0f%%] %s", pct, msg)

        # ── المرحلة 1: البحث والتنظيم ────────────────────────────
        _prog("🔍 Gemini Flash يبحث ويُنظّم المشاهد...", 5)
        full_topic = f"[{style}] {topic}"
        outline_text, research_provider = self.research_outline(full_topic, target_minutes)

        raw_scenes = _parse_outline_scenes(outline_text)
        if not raw_scenes:
            # fallback: قسّم النص إلى فقرات
            raw_scenes = _fallback_split(outline_text, target_minutes)

        _prog(f"✅ تم تحديد {len(raw_scenes)} مشهداً", 15)

        # ── المرحلة 2: صياغة السرد لكل مشهد ─────────────────────
        scenes: List[VideoScene] = []
        narrative_provider = ""
        total = len(raw_scenes)

        for i, raw in enumerate(raw_scenes):
            pct = 15 + (i / total) * 45
            _prog(f"✍️ Nova Fable 5 يصيغ المشهد {i+1}/{total}...", pct)

            narration, video_prompt, np = self.craft_narration(raw["content"], i + 1)
            if not narrative_provider:
                narrative_provider = np

            scene = VideoScene(
                index        = i + 1,
                title        = raw.get("title", f"المشهد {i+1}"),
                narration    = narration,
                visual_notes = raw.get("visual", ""),
                video_prompt = video_prompt,
                est_seconds  = raw.get("duration", 30),
                video_status = "pending",
            )
            scenes.append(scene)

        script = DocumentaryScript(
            topic              = topic,
            title              = topic.strip(),
            scenes             = scenes,
            research_provider  = research_provider,
            narrative_provider = narrative_provider,
        )
        result = HiggsfieldResult(
            script       = script,
            scenes_total = len(scenes),
            api_used     = bool(self._hf_client),
        )

        # ── المرحلة 3: توليد الفيديو ─────────────────────────────
        if generate_video and self._hf_client:
            for i, scene in enumerate(scenes):
                pct = 60 + (i / total) * 38
                _prog(f"🎬 Higgsfield يُولّد المشهد {i+1}/{total}...", pct)
                self.generate_video_for_scene(scene)
                result.scenes_done = i + 1
        else:
            for scene in scenes:
                scene.video_status = "no_api" if not self._hf_client else "skipped"
            result.scenes_done = 0

        _prog("✅ اكتمل!", 100)
        return result

    # ── توليد مشهد واحد (للتحديث التدريجي في UI) ─────────────────

    def generate_videos_iter(
        self, result: HiggsfieldResult
    ) -> Iterator[VideoScene]:
        """
        يُولّد مقاطع الفيديو مشهداً بمشهد ويُعيد كل مشهد فور اكتماله.
        مفيد لتحديث واجهة Streamlit بشكل تدريجي.
        """
        for scene in result.scenes:
            self.generate_video_for_scene(scene)
            result.scenes_done += 1
            yield scene


# ══════════════════════════════════════════════════════════════════════════
# دوال مساعدة للتحليل
# ══════════════════════════════════════════════════════════════════════════

def _extract_field(text: str, field_name: str) -> str:
    """يستخرج قيمة حقل بصيغة 'الحقل: القيمة'"""
    pattern = rf"{re.escape(field_name)}\s*:\s*(.+?)(?=\n\s*(?:السرد|الـ Prompt|العنوان|المحتوى|المشهد المرئي|المدة)\s*:|$)"
    m = re.search(pattern, text, re.S | re.DOTALL)
    return m.group(1).strip() if m else ""


def _parse_outline_scenes(text: str) -> List[dict]:
    """
    يُحلّل نص الـ outline الخام إلى قائمة مشاهد.
    يتوقع تنسيق:
        ### المشهد N
        العنوان: ...
        المحتوى: ...
        المشهد المرئي: ...
        المدة: ...
    """
    scenes = []
    blocks = re.split(r"###\s*المشهد\s*\d+", text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        title   = _extract_field(block, "العنوان") or block.splitlines()[0][:60]
        content = _extract_field(block, "المحتوى") or block
        visual  = _extract_field(block, "المشهد المرئي")
        dur_m   = re.search(r"المدة\s*:\s*(\d+)", block)
        duration = int(dur_m.group(1)) if dur_m else 30

        scenes.append({
            "title":    title.strip(),
            "content":  content.strip(),
            "visual":   visual.strip(),
            "duration": max(20, min(duration, 90)),
        })
    return scenes


def _fallback_split(text: str, target_minutes: int) -> List[dict]:
    """
    Fallback: يقسّم النص إلى فقرات إن لم يلتزم النموذج بالتنسيق.
    """
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paras:
        paras = [text]
    n_target = max(3, target_minutes * 2)
    # دمج الفقرات لتقترب من العدد المستهدف
    chunk_size = max(1, len(paras) // n_target)
    scenes = []
    for i in range(0, len(paras), chunk_size):
        chunk = "\n".join(paras[i:i + chunk_size])
        scenes.append({
            "title":    f"المشهد {len(scenes)+1}",
            "content":  chunk,
            "visual":   "",
            "duration": 30,
        })
        if len(scenes) >= n_target:
            break
    return scenes

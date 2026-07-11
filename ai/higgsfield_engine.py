"""
Higgsfield Explainer Engine — NSM v2.0 (Professional Edition)
==============================================================
ينتج فيديو وثائقياً احترافياً (حتى 10 دقائق) من موضوع نصي عبر pipeline ثلاثي المراحل:

  المرحلة 1 — Gemini Omni Flash:
      يبحث في المعلومات ويبني هيكل مشاهد وثائقية بمستوى National Geographic.

  المرحلة 2 — NSM Agent Fable 5 (NSM):
      يصيغ نص سرد شاعري-وثائقي فصيح + video prompt سينمائي احترافي لكل مشهد.

  المرحلة 3 — Higgsfield API:
      يُرسل كل مشهد لتوليد مقطع فيديو عالي الجودة (5-8 ثوانٍ).
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

_GEMINI_FLASH_MODEL = "gemini-2.0-flash"
_FABLE_MODEL_KEY    = "fable"


class _PinnedLLM:
    def __init__(self, primary_fn, fallback_llm, provider_label: str):
        self._primary        = primary_fn
        self._fallback       = fallback_llm
        self._provider_label = provider_label

    def generate(self, query: str, history: list, system_prompt: str = ""):
        if self._primary is not None:
            try:
                return self._primary(query, history, system_prompt)
            except Exception as exc:
                logger.warning(
                    "المزوّد [%s] فشل: %s — ينتقل إلى الـ fallback",
                    self._provider_label, exc,
                )
        return self._fallback.generate(query, history=history, system_prompt=system_prompt)

    @property
    def provider_label(self) -> str:
        return self._provider_label


def build_gemini_llm() -> _PinnedLLM:
    from ai.llm_fallback import LLMFallback
    fallback = LLMFallback()
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    primary_fn = None
    if key:
        _gemini_instance          = LLMFallback()
        _gemini_instance._api_key = key
        _gemini_instance._model   = _GEMINI_FLASH_MODEL
        primary_fn = _gemini_instance._call_gemini
    return _PinnedLLM(primary_fn, fallback, "gemini-omni-flash")


def build_fable_llm() -> _PinnedLLM:
    from ai.llm_fallback import LLMFallback, ANTHROPIC_MODELS
    fallback = LLMFallback(model_key=_FABLE_MODEL_KEY)
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    primary_fn = None
    if key:
        _fable_instance          = LLMFallback(model_key=_FABLE_MODEL_KEY)
        _fable_instance._api_key = key
        _fable_instance._model   = ANTHROPIC_MODELS[_FABLE_MODEL_KEY]
        primary_fn = _fable_instance._call_anthropic
    return _PinnedLLM(primary_fn, fallback, "nsm-agent")


# ══════════════════════════════════════════════════════════════════════════
# Dataclasses
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class VideoScene:
    index:         int
    title:         str          = ""
    narration:     str          = ""
    visual_notes:  str          = ""
    video_prompt:  str          = ""
    est_seconds:   int          = 35
    job_id:        str          = ""
    video_url:     str          = ""
    video_status:  str          = "pending"
    video_error:   str          = ""


@dataclass
class DocumentaryScript:
    topic:        str
    title:        str
    synopsis:     str               = ""
    scenes:       List[VideoScene]  = field(default_factory=list)
    research_provider: str          = ""
    narrative_provider: str         = ""
    error:        str               = ""

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
    script:       DocumentaryScript
    scenes_done:  int  = 0
    scenes_total: int  = 0
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
# Higgsfield API Client — محسّن مع دعم endpoints متعددة
# ══════════════════════════════════════════════════════════════════════════

_HF_BASE_URL      = "https://api.higgsfield.ai"
_HF_TIMEOUT       = 45
_HF_POLL_INTERVAL = 8
_HF_MAX_WAIT      = 300    # 5 دقائق لكل مشهد
_HF_CLIP_DURATION = 8      # مدة المقطع بالثواني


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
    """عميل Higgsfield API محسّن — يجرّب endpoints متعددة للتوافق الأقصى."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("HIGGSFIELD_API_KEY غير موجود في البيئة")
        self._key = api_key

    def submit_job(self, prompt: str, duration: int = _HF_CLIP_DURATION) -> str:
        """يُرسل طلب توليد فيديو بجودة احترافية عالية."""
        duration = max(3, min(int(duration), 8))

        # قائمة endpoints مع body مناسب لكل منها
        attempts = [
            ("/api/v1/videos/generate", {
                "prompt":       prompt,
                "model":        "higgsfield-1",
                "duration":     duration,
                "aspect_ratio": "16:9",
                "quality":      "high",
            }),
            ("/api/v1/videos", {
                "prompt":       prompt,
                "model":        "higgsfield-video-1",
                "duration":     duration,
                "style":        "cinematic",
                "aspect_ratio": "16:9",
                "quality":      "high",
                "fps":          24,
            }),
            ("/v1/generations", {
                "prompt":       prompt,
                "duration":     duration,
                "aspect_ratio": "16:9",
                "model":        "higgsfield-video-1",
            }),
        ]

        last_exc = None
        for endpoint, body in attempts:
            try:
                resp   = _hf_post(endpoint, body, self._key)
                job_id = (
                    resp.get("id")
                    or resp.get("job_id")
                    or resp.get("generation_id")
                    or resp.get("video_id")
                    or ""
                )
                if job_id:
                    logger.info("Higgsfield job submitted → endpoint=%s job_id=%s", endpoint, job_id)
                    return str(job_id)
                logger.warning("Higgsfield %s لم يُعِد job_id: %s", endpoint, resp)
            except urllib.error.HTTPError as e:
                last_exc = e
                if e.code in (404, 405):
                    continue
                if e.code in (401, 403):
                    raise RuntimeError(
                        f"خطأ مصادقة Higgsfield (HTTP {e.code}) — تحقّق من صحة API Key"
                    )
                if 400 <= e.code < 500:
                    body_txt = ""
                    try:
                        body_txt = e.read().decode("utf-8", errors="replace")
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"Higgsfield رفض الطلب (HTTP {e.code}): {body_txt[:300]}"
                    )
            except Exception as exc:
                last_exc = exc
                logger.warning("Higgsfield endpoint %s فشل: %s", endpoint, exc)

        raise RuntimeError(
            f"تعذّر إرسال الطلب إلى Higgsfield — آخر خطأ: {last_exc}"
        )

    def poll_job(self, job_id: str, max_wait: int = _HF_MAX_WAIT) -> VideoScene:
        """يستطلع حالة المهمة حتى اكتمالها أو انتهاء المهلة."""
        poll_paths    = [
            f"/api/v1/videos/{job_id}",
            f"/api/v1/generations/{job_id}",
            f"/v1/generations/{job_id}",
        ]
        working_path = poll_paths[0]
        deadline     = time.time() + max_wait

        while time.time() < deadline:
            try:
                resp   = _hf_get(working_path, self._key)
                status = (
                    resp.get("status")
                    or resp.get("state")
                    or "pending"
                ).lower()

                if status in ("completed", "done", "success", "succeeded", "finished"):
                    url = (
                        resp.get("output", {}).get("url")
                        or resp.get("video_url")
                        or resp.get("url")
                        or resp.get("download_url")
                        or ""
                    )
                    return VideoScene(
                        index=0, video_url=url,
                        video_status="completed", job_id=job_id,
                    )

                if status in ("failed", "error", "cancelled"):
                    err = (
                        resp.get("error")
                        or resp.get("message")
                        or resp.get("reason")
                        or "خطأ غير محدد"
                    )
                    return VideoScene(
                        index=0, video_status="failed",
                        video_error=str(err), job_id=job_id,
                    )

                logger.debug("Higgsfield job %s → %s", job_id, status)

            except urllib.error.HTTPError as e:
                if 400 <= e.code < 500:
                    # جرّب path بديل
                    try:
                        idx = poll_paths.index(working_path)
                        if idx + 1 < len(poll_paths):
                            working_path = poll_paths[idx + 1]
                            continue
                    except ValueError:
                        pass
                    err_body = ""
                    try:
                        err_body = e.read().decode("utf-8", errors="replace")
                    except Exception:
                        pass
                    return VideoScene(
                        index=0, video_status="failed",
                        video_error=f"HTTP {e.code}: {err_body[:200]}",
                        job_id=job_id,
                    )
                logger.debug("Higgsfield poll HTTP %d — سنُعيد المحاولة", e.code)

            except Exception as exc:
                logger.debug("Higgsfield poll خطأ: %s — سنُعيد المحاولة", exc)

            time.sleep(_HF_POLL_INTERVAL)

        return VideoScene(
            index=0, video_status="timeout",
            video_error=f"انتهت مهلة الانتظار ({max_wait}s)",
            job_id=job_id,
        )


# ══════════════════════════════════════════════════════════════════════════
# System Prompts — بجودة إنتاجية استثنائية
# ══════════════════════════════════════════════════════════════════════════

_RESEARCH_SP = """\
أنت كبير منتجي الوثائقيات في شبكة National Geographic وBBC Earth — خبرتك 20 عاماً
في صنع وثائقيات فائزة بجوائز إيمي وبافتا. مهمتك: استقبال موضوع وتصميم بنية مشاهد
وثائقية استثنائية بمستوى إنتاجي عالمي.

مبادئ الإنتاج الاحترافي:
- ابدأ بمشهد "الخطّاف" (Hook Scene) يُفجّر الدهشة ويُشعل الفضول في الثواني العشر الأولى
- خطّط للقوس الدرامي: افتتاح جذّاب → بناء درامي → ذروة مُذهلة → خاتمة لا تُنسى
- اختر الحقائق التي تُذهل الخبير قبل المبتدئ — الغريبة، المثيرة، غير المتوقعة
- المشهد البصري يجب أن يكون من أجمل ما يمكن تصويره — جمال طبيعي، بنية معمارية، لحظة إنسانية
- تنوّع في أنواع المشاهد: مشاهد جوية، ميكروسكوبية، تاريخية، بشرية، تجريدية

قواعد صارمة:
- لا تختلق حقائق — قل "تشير الدراسات" أو "يُعتقد" إن كنت غير متأكد
- المدة بين 25 و75 ثانية للمشهد الواحد
- المشهد البصري يجب أن يكون بصرياً قابلاً للتصوير (لا "شخص يتكلم" أو "خريطة")

التنسيق المطلوب (التزم به حرفياً):
### المشهد N
العنوان: <عنوان شعري يُثير الفضول — جملة واحدة>
المحتوى: <4-6 جمل عربية — الأكثر إثارة وغير المتوقعة، بأسلوب حيوي ومكثّف>
المشهد المرئي: <وصف تصويري مُفصَّل — زاوية الكاميرا، الإضاءة، الحركة، الألوان، العناصر البصرية>
المدة: <المدة بالثواني — بين 25 و 75>
القيمة العاطفية: <دهشة/رهبة/حنين/إلهام/توتر/فرح/حزن — إحساس واحد مُسيطر>
"""

_NARRATIVE_SP = """\
أنت كاتب سيناريو وثائقي من أفضل خمسة في العالم — أسلوبك يُضاهي كتّاب BBC Earth
وNational Geographic وAttenborough ومارتن سكورسيزي. مهمتك: تحويل مخطط المشهد
إلى تحفة أدبية-سينمائية تترك أثراً لا يُمحى.

للنص السردي العربي:
- فصحى رشيقة تنبض بالحياة — بعيدة عن الجمود الأكاديمي
- ابدأ بجملة خاطفة تُحطّم التوقع أو تُعيد تعريف المشهد كلياً
- إيقاع متنفّس: جمل قصيرة مُدوّية يعقبها وصف مطوّل يُغرق القارئ
- كثافة معلومات عالية مع صور شعرية — كل كلمة مُبرَّرة، كل جملة لها دور
- ~130-150 كلمة لكل 60 ثانية من المشهد
- اختم بجملة تُبقي المشاهد متعلقاً — سؤال مُعلَّق، مفارقة، أو وصف حسّي عميق

للـ video prompt الاحترافي (بالإنجليزية):
أنت الآن المخرج السينمائي الأول في الفيلم. اكتب prompt بمستوى Higgsfield × Hollywood:

بنية الـ prompt المثالي (3-5 جمل، لا قوائم):
[Shot type + camera movement] of [specific subject with details],
[environment + lighting details], [atmosphere + mood],
[color grade + visual style], [technical quality reference].

أنواع اللقطات الاحترافية:
- extreme close-up macro / close-up / medium shot / wide establishing / aerial drone
- POV / over-the-shoulder / dutch angle / bird's-eye / worm's-eye
- tracking shot / dolly zoom / crane / steadicam / handheld / gimbal stabilized

حركات الكاميرا:
- slow push-in / dramatic pull-back / circular orbit / lateral tracking
- crane rising / helicopter banking / microscopic zoom-in

الإضاءة والألوان:
- golden hour warm glow / blue-hour twilight / harsh midday sun
- bioluminescent underwater / neon-lit urban / candlelit interior
- rim-lit silhouette / fog-diffused soft light / storm-lit dramatic

المزاج والأجواء:
- epic awe-inspiring / intimate melancholic / mysterious mystical
- raw visceral / serene transcendent / tense foreboding / joyful euphoric

مثال على prompt احترافي ممتاز:
"Extreme macro close-up of a single dewdrop on an ancient Arabic calligraphy manuscript,
slowly pulling back to reveal the full illuminated page with golden ink glistening in candlelight,
camera continuing to pull back through an arched window into a vast medieval library at dusk,
warm amber lighting with deep indigo shadows, rich color grade with Kodak-inspired tones,
cinematic 4K documentary, National Geographic production quality, anamorphic lens, ultra-realistic."

قواعد الـ prompt الاحترافي:
- 3-5 جمل وصفية مكثّفة — لا قوائم نقطية إطلاقاً
- ابدأ بنوع اللقطة وحركة الكاميرا بدقة
- أضِف تفاصيل الإضاءة والألوان والأجواء
- اذكر reference style: "National Geographic quality" أو "BBC Earth cinematic"
- صِف ما تراه بعينيك — لا مفاهيم مجردة

التنسيق (التزم به حرفياً، لا تضف أي نص آخر):
السرد: <نص السرد الصوتي بالعربية الفصحى — غني ومكثّف وحيوي>
الـ Prompt: <cinematic video prompt in English — 3-5 sentences, highly specific and detailed>
"""


# ══════════════════════════════════════════════════════════════════════════
# HiggsfieldEngine — المحرك الرئيسي المحسّن
# ══════════════════════════════════════════════════════════════════════════

class HiggsfieldEngine:
    """
    محرك Higgsfield Explainer المحسّن — جودة National Geographic:
      1. Gemini Flash   → بحث وتصميم مشاهد وثائقية استثنائية
      2. NSM Agent Fable 5   → سرد شاعري + video prompts سينمائية احترافية
      3. Higgsfield API → توليد فيديو عالي الجودة لكل مشهد
    """

    def __init__(self, gemini_llm, fable_llm, higgsfield_key: str = ""):
        self._gemini    = gemini_llm
        self._fable     = fable_llm
        self._hf_key    = higgsfield_key or os.getenv("HIGGSFIELD_API_KEY", "").strip()
        self._hf_client: Optional[HiggsfieldClient] = None
        if self._hf_key:
            try:
                self._hf_client = HiggsfieldClient(self._hf_key)
            except ValueError:
                pass

    # ── المرحلة 1: البحث والتصميم ─────────────────────────────────────

    def research_outline(self, topic: str, target_minutes: int) -> tuple[str, str]:
        n_scenes = max(4, int(target_minutes * 2.5))
        query = (
            f"موضوع الوثائقي: {topic}\n"
            f"المدة المستهدفة: {target_minutes} دقيقة\n"
            f"عدد المشاهد المطلوب: {n_scenes} مشهداً تقريباً\n\n"
            "تذكّر: ابدأ بمشهد 'الخطّاف' الذي يُفجّر الدهشة فوراً، "
            "وخطّط للقوس الدرامي الكامل من الافتتاح حتى الخاتمة."
        )
        result   = self._gemini.generate(query, history=[], system_prompt=_RESEARCH_SP)
        provider = getattr(result.provider, "value", str(result.provider))
        return result.text.strip(), provider

    # ── المرحلة 2: صياغة السرد والـ prompt ────────────────────────────

    def craft_narration(
        self,
        scene_outline: str,
        scene_index: int,
        visual_notes: str = "",
        emotional_value: str = "",
    ) -> tuple[str, str, str]:
        emotion_hint = f"\nالقيمة العاطفية المستهدفة: {emotional_value}" if emotional_value else ""
        visual_hint  = f"\nالمشهد المرئي المقترح: {visual_notes}" if visual_notes else ""
        query = (
            f"المشهد {scene_index}:\n{scene_outline}"
            f"{visual_hint}{emotion_hint}\n\n"
            "اكتب نص السرد الصوتي العربي الاحترافي والـ video prompt السينمائي بالإنجليزية.\n"
            "الـ prompt يجب أن يكون 3-5 جمل وصفية مُفصَّلة — بمستوى إنتاج National Geographic."
        )
        result   = self._fable.generate(query, history=[], system_prompt=_NARRATIVE_SP)
        provider = getattr(result.provider, "value", str(result.provider))
        raw      = result.text.strip()

        narration    = _extract_field(raw, "السرد")
        video_prompt = _extract_field(raw, "الـ Prompt")

        if not narration:
            narration = raw
        if not video_prompt:
            video_prompt = _build_fallback_prompt(scene_outline, visual_notes)

        # تعزيز الـ prompt إن كان قصيراً أو مُبهَماً
        if len(video_prompt.split()) < 25:
            video_prompt = _enhance_prompt(video_prompt, visual_notes)

        return narration, video_prompt, provider

    # ── المرحلة 3: توليد الفيديو ──────────────────────────────────────

    def generate_video_for_scene(self, scene: VideoScene) -> VideoScene:
        if not self._hf_client:
            scene.video_status = "no_api"
            scene.video_error  = "HIGGSFIELD_API_KEY غير موجود — أضِفه في الأسرار أو أدخله أعلاه"
            return scene

        try:
            job_id = self._hf_client.submit_job(
                scene.video_prompt,
                duration=_HF_CLIP_DURATION,
            )
            scene.job_id       = job_id
            scene.video_status = "processing"

            poll_result        = self._hf_client.poll_job(job_id)
            scene.video_url    = poll_result.video_url
            scene.video_status = poll_result.video_status
            scene.video_error  = poll_result.video_error

        except Exception as exc:
            scene.video_status = "failed"
            scene.video_error  = str(exc)
            logger.warning("Higgsfield فشل للمشهد %d: %s", scene.index, exc)

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
        target_minutes = max(1, min(int(target_minutes), 10))

        def _prog(msg: str, pct: float):
            if progress_cb:
                progress_cb(msg, pct)
            logger.info("[%.0f%%] %s", pct, msg)

        # ── المرحلة 1 ────────────────────────────────────────────────
        _prog("🔍 Gemini يُصمّم مشاهد وثائقية بمستوى National Geographic...", 5)
        full_topic   = f"[{style}] {topic}"
        outline_text, research_provider = self.research_outline(full_topic, target_minutes)

        raw_scenes = _parse_outline_scenes(outline_text)
        if not raw_scenes:
            raw_scenes = _fallback_split(outline_text, target_minutes)

        _prog(f"✅ تم تصميم {len(raw_scenes)} مشهداً وثائقياً احترافياً", 15)

        # ── المرحلة 2 ────────────────────────────────────────────────
        scenes: List[VideoScene] = []
        narrative_provider = ""
        total = len(raw_scenes)

        for i, raw in enumerate(raw_scenes):
            pct = 15 + (i / total) * 45
            _prog(f"✍️ صياغة المشهد {i+1}/{total} بأسلوب سينمائي احترافي...", pct)

            narration, video_prompt, np = self.craft_narration(
                raw["content"], i + 1,
                visual_notes    = raw.get("visual", ""),
                emotional_value = raw.get("emotion", ""),
            )
            if not narrative_provider:
                narrative_provider = np

            scene = VideoScene(
                index        = i + 1,
                title        = raw.get("title", f"المشهد {i+1}"),
                narration    = narration,
                visual_notes = raw.get("visual", ""),
                video_prompt = video_prompt,
                est_seconds  = raw.get("duration", 35),
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

        # ── المرحلة 3 ────────────────────────────────────────────────
        if generate_video and self._hf_client:
            for i, scene in enumerate(scenes):
                pct = 60 + (i / total) * 38
                _prog(f"🎬 Higgsfield يُولّد الفيديو للمشهد {i+1}/{total}...", pct)
                self.generate_video_for_scene(scene)
                result.scenes_done = i + 1
        else:
            for scene in scenes:
                scene.video_status = "no_api" if not self._hf_client else "skipped"
            result.scenes_done = 0

        _prog("✅ اكتمل الوثائقي بجودة احترافية!", 100)
        return result

    # ── توليد تدريجي للـ UI ──────────────────────────────────────────

    def generate_videos_iter(
        self, result: HiggsfieldResult
    ) -> Iterator[VideoScene]:
        """يُولّد الفيديو مشهداً بمشهد ويُعيده فور اكتماله — مثالي لـ Streamlit."""
        for scene in result.scenes:
            self.generate_video_for_scene(scene)
            result.scenes_done += 1
            yield scene


# ══════════════════════════════════════════════════════════════════════════
# دوال مساعدة
# ══════════════════════════════════════════════════════════════════════════

def _extract_field(text: str, field_name: str) -> str:
    pattern = (
        rf"{re.escape(field_name)}\s*:\s*"
        rf"(.+?)(?=\n\s*(?:السرد|الـ Prompt|العنوان|المحتوى|المشهد المرئي"
        rf"|المدة|القيمة العاطفية)\s*:|$)"
    )
    m = re.search(pattern, text, re.S | re.DOTALL)
    return m.group(1).strip() if m else ""


def _build_fallback_prompt(content: str, visual: str = "") -> str:
    """يبني prompt احترافياً افتراضياً عند غياب الـ prompt من النموذج."""
    subject = (visual or content)[:100]
    return (
        f"Cinematic wide establishing shot capturing {subject}, "
        "camera slowly pushing in with a smooth crane descent, "
        "dramatic golden-hour lighting casting long warm shadows across the scene, "
        "rich color grade with deep indigo shadows and warm amber highlights, "
        "atmospheric depth of field with soft bokeh background, "
        "4K ultra-realistic cinematic documentary, National Geographic production quality, "
        "anamorphic lens flare, ultra-detailed textures."
    )


def _enhance_prompt(prompt: str, visual: str = "") -> str:
    """يُثري prompt قصيراً أو مُبهَماً بتفاصيل سينمائية احترافية."""
    additions = (
        " Camera movement: smooth dolly push-in with subtle gimbal stabilization. "
        "Lighting: cinematic three-point setup with warm golden key light and cool blue fill. "
        "Color grade: rich Kodak Portra-inspired tones, deep lifted blacks, "
        "luminous highlights. Visual style: 4K cinematic documentary, "
        "National Geographic and BBC Earth quality, "
        "anamorphic lens characteristics, ultra-realistic textures."
    )
    return prompt.rstrip(".") + additions


def _parse_outline_scenes(text: str) -> List[dict]:
    scenes = []
    blocks = re.split(r"###\s*المشهد\s*\d+", text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        title   = _extract_field(block, "العنوان") or block.splitlines()[0][:60]
        content = _extract_field(block, "المحتوى") or block
        visual  = _extract_field(block, "المشهد المرئي")
        emotion = _extract_field(block, "القيمة العاطفية")
        dur_m   = re.search(r"المدة\s*:\s*(\d+)", block)
        duration = int(dur_m.group(1)) if dur_m else 35

        if not content or len(content) < 10:
            continue

        scenes.append({
            "title":    title.strip(),
            "content":  content.strip(),
            "visual":   visual.strip(),
            "emotion":  emotion.strip(),
            "duration": max(20, min(duration, 90)),
        })
    return scenes


def _fallback_split(text: str, target_minutes: int) -> List[dict]:
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paras:
        paras = [text]
    n_target   = max(4, int(target_minutes * 2.5))
    chunk_size = max(1, len(paras) // n_target)
    scenes     = []
    for i in range(0, len(paras), chunk_size):
        chunk = "\n".join(paras[i:i + chunk_size])
        scenes.append({
            "title":    f"المشهد {len(scenes)+1}",
            "content":  chunk,
            "visual":   "",
            "emotion":  "دهشة وإلهام",
            "duration": 35,
        })
        if len(scenes) >= n_target:
            break
    return scenes

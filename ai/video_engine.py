"""
Video Engine — محرك رندر الفيديو الفعلي — NSM
=================================================
يركّب سيناريو ExplainerScript (من FableEngine.generate_short/generate_explainer)
+ الصوت المولَّد عبر TTSEngine → فيديو mp4 عمودي فعلي بأسلوب الترجمات
المتحركة كلمة-بكلمة (Kinetic Captions بنمط CapCut/Submagic/Opus Clip) —
كل عبارة قصيرة تظهر بحاجز (pill) ملوّن ونص عريض بحدّ أبيض وتأثير "نبضة"
عند الظهور + زووم Ken-Burns مستمر عبر المشهد بالكامل، بدون أي اعتماد على
ImageMagick (كل النص يُرسم عبر Pillow مباشرة).

🎬 خلفيات سينمائية احترافية (اختياري — VideoEngine(use_cinematic_backgrounds=True)):
    بدل الخلفية المتدرّجة الافتراضية، يمكن توليد خلفية فيديو حقيقية لكل
    مشهد عبر Higgsfield API (نفس مزوّد ai/higgsfield_engine.py، بجودة
    "National Geographic/BBC Earth") ثم قصّها من 16:9 لتغطية الإطار
    العمودي 9:16، مع تركيب الترجمات المتحركة فوقها كطبقة شفافة منفصلة.
    مُعطَّلة افتراضياً (opt-in) لأن Higgsfield مزوّد مدفوع بعكس بقية
    مسار NSM المجاني — تُفعَّل فقط بطلب صريح من المستخدم بالواجهة، وتتراجع
    تلقائياً وبصمت للخلفية المتدرّجة المجانية عند غياب HIGGSFIELD_API_KEY
    أو فشل التوليد لأي مشهد (لا يوقف الفيديو بالكامل أبداً).

المتطلبات (requirements.txt):
    moviepy>=2.0
    imageio-ffmpeg>=0.4.9   # يحمل ثنائي ffmpeg تلقائياً، بدون حاجة لـ apt
    pillow                  # موجود أصلاً بالمشروع (يلزم Pillow>=8.0 لدعم stroke_width في draw.text)
    arabic-reshaper         # موجود أصلاً بالمشروع
    python-bidi             # موجود أصلاً بالمشروع

اختياري (Streamlit Cloud) — packages.txt:
    ffmpeg
    fonts-noto-core

الاستخدام:
    from ai.fable_engine import FableEngine
    engine = FableEngine(llm_fallback=my_llm_fallback)
    script = engine.generate_short(source_text, target_seconds=60)
    engine.render_audio(script)          # يملأ الصوت الفعلي لكل مشهد
    mp4_bytes = engine.render_video(script)   # فيديو mp4 فعلي جاهز (خلفية متدرّجة)
    mp4_bytes = engine.render_video(script, use_cinematic_backgrounds=True)  # خلفيات Higgsfield
    with open("short.mp4", "wb") as f:
        f.write(mp4_bytes)
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import tempfile
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("VideoEngine")

# مهلة زمنية قصوى (ثوانٍ) لكل مشهد عند توليد خلفية سينمائية — أقصر من
# مهلة الوثائقي الطويل (_HF_MAX_WAIT=300 في higgsfield_engine) عمداً،
# لأن Shorts فيديو قصير ولا يجب أن يُعلَّق المستخدم دقائق طويلة انتظاراً
# لكل مشهد؛ عند تجاوز المهلة نتراجع فوراً للخلفية المتدرّجة لهذا المشهد
# فقط دون فشل الفيديو بالكامل.
_HF_SHORT_MAX_WAIT = 90

# أبعاد فيديو رأسي قياسي (9:16) — نفس نسبة NotebookLM Shorts
FRAME_W, FRAME_H = 1080, 1920
FPS = 30

# لوحة ألوان تدرّجية تتناوب بين المشاهد (RGB)
_GRADIENT_PAIRS = [
    ((20, 24, 38), (58, 33, 92)),
    ((15, 32, 39), (32, 58, 67)),
    ((44, 20, 60), (90, 30, 60)),
    ((10, 30, 50), (40, 70, 110)),
    ((30, 15, 45), (75, 40, 90)),
]

# مسارات خطوط عربية شائعة في بيئات Linux (Replit / Streamlit Cloud / Debian)
# ترتيب الأولوية: خطوط عريضة (Bold) أولاً — أوضح وأقوى بصرياً لأسلوب
# الترجمات المتحركة (Kinetic Captions) المستخدم بمنصات مثل CapCut/Submagic —
# ثم الأوزان العادية، وDejaVuSans (لا يدعم العربي) آخر خيار مطلق فقط عشان
# لا يفشل الرسم بالكامل.
_SYSTEM_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoKufiArabic-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf",
    "/usr/share/fonts/opentype/noto/NotoKufiArabic-Bold.ttf",
    "/usr/share/fonts/opentype/noto/NotoNaskhArabic-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoNaskhArabicUI-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/truetype/kacst/KacstOne.ttf",
]
_LAST_RESORT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # لا يدعم العربي

# مصدر تنزيل احتياطي (GitHub raw، مُتحقَّق منه) إن لم يوجد الخط بالنظام —
# نفضّل Noto Kufi Arabic Bold (خط عرض هندسي عريض، مثالي للعناوين/الترجمات
# المتحركة)، ويُخزَّن محلياً بعد أول تنزيل فلا يُعاد الطلب كل مرة.
_FONT_FALLBACK_URL = (
    "https://raw.githubusercontent.com/notofonts/notofonts.github.io/main/"
    "fonts/NotoKufiArabic/hinted/ttf/NotoKufiArabic-Bold.ttf"
)
_FONT_CACHE_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
_FONT_CACHE_PATH = _FONT_CACHE_DIR / "NotoKufiArabic-Bold.ttf"


class VideoEngineError(RuntimeError):
    pass


# ── تكامل اختياري مع Higgsfield لخلفيات سينمائية حقيقية ──────────────────

def _build_cinematic_prompt(narration: str, visual_notes: str) -> str:
    """يبني video prompt سينمائياً احترافياً (بالإنجليزية) من سرد/وصف
    المشهد العربي، بإعادة استخدام نفس منطق higgsfield_engine (بدل تكرار
    الأسلوب) حتى تتوافق جودة الخلفية مع بقية NSM."""
    from ai.higgsfield_engine import _build_fallback_prompt

    base = _build_fallback_prompt(narration, visual_notes)
    # تلميحات جودة للمسار المجاني (ZeroGPU) + تكوين عمودي للشورتس
    return (
        f"{base}. Photorealistic cinematic short, vertical 9:16 framing, "
        "shallow depth of field, natural volumetric light, rich color grade, "
        "smooth stabilized camera, high detail texture, filmic contrast, "
        "no text, no watermark, no logo, no subtitles"
    )


def _fetch_cinematic_clip(narration: str, visual_notes: str, tmp_dir: str, seg_index: int) -> Optional[str]:
    """يطلب مقطع فيديو سينمائي حقيقي من Higgsfield لمشهد واحد، وينزّله
    محلياً. يُرجِع مسار الملف عند النجاح، أو None عند أي فشل/غياب مفتاح
    (تراجع صامت — لا يرفع استثناء أبداً حتى لا يُسقط الفيديو بالكامل)."""
    api_key = os.getenv("HIGGSFIELD_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        from ai.higgsfield_engine import HiggsfieldClient

        client = HiggsfieldClient(api_key)
        prompt = _build_cinematic_prompt(narration, visual_notes)
        job_id = client.submit_job(prompt)
        result = client.poll_job(job_id, max_wait=_HF_SHORT_MAX_WAIT)

        if result.video_status != "completed" or not result.video_url:
            logger.info(
                "خلفية Higgsfield للمشهد %d غير جاهزة (%s) — استخدام الخلفية "
                "المتدرّجة كبديل لهذا المشهد فقط.", seg_index, result.video_status,
            )
            return None

        clip_path = os.path.join(tmp_dir, f"hf_bg_{seg_index}.mp4")
        req = urllib.request.Request(result.video_url, headers={"User-Agent": "NSM-VideoEngine/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(clip_path, "wb") as f:
            f.write(resp.read())
        return clip_path

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "تعذّر جلب خلفية Higgsfield للمشهد %d (%s) — استخدام الخلفية "
            "المتدرّجة كبديل لهذا المشهد فقط.", seg_index, exc,
        )
        return None


def _wan_free_negative_prompt() -> str:
    """برومبت سلبي أقوى للمسار المجاني — يقلّل الضبابية والنص والتشوه."""
    return (
        "blurry, low quality, worst quality, jpeg artifacts, noise, grain, "
        "watermark, logo, text, subtitles, caption, title, letters, "
        "distorted, deformed, mutated, disfigured, extra limbs, bad anatomy, "
        "static image, still frame, frozen, slideshow, low resolution, "
        "overexposed, underexposed, washed out, cartoonish (unless requested)"
    )



def _enhance_free_clip(src_path: str, dst_path: str) -> str:
    """رفع جودة مقطع مجاني بعد التوليد: تغطية 9:16 + حدة + ألوان + 30fps.
    إن فشل ffmpeg يُعاد المسار الأصلي دون كسر المسار.
    """
    import shutil
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not src_path or not os.path.isfile(src_path):
        return src_path
    vf = (
        f"scale={FRAME_W}:{FRAME_H}:force_original_aspect_ratio=increase,"
        f"crop={FRAME_W}:{FRAME_H},"
        "fps=30,"
        "eq=contrast=1.10:saturation=1.15:brightness=0.03:gamma=1.02,"
        "unsharp=5:5:0.7:5:5:0.0"
    )
    cmd = [
        ffmpeg, "-y", "-i", src_path,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "slow", "-crf", "15",
        "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-tune", "film", "-an", "-movflags", "+faststart",
        dst_path,
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if p.returncode == 0 and os.path.isfile(dst_path) and os.path.getsize(dst_path) > 1000:
            return dst_path
        logger.debug("enhance free clip failed: %s", (p.stderr or "")[-300:])
    except Exception as exc:  # noqa: BLE001
        logger.debug("enhance free clip error: %s", exc)
    return src_path


# قائمة مساحات Hugging Face المجتمعية المجانية — بترتيب الأولوية (الأسرع
# أولاً). كل عنصر يحمل توقيع الاستدعاء الخاص بمساحته (كل مساحة مبنية
# بواجهة مختلفة قليلاً)، فلا يوجد شكل موحّد واحد. fffiloni/Wan2.1 آخر
# القائمة لأنه يُشغّل generate.py كعملية فرعية كاملة (أبطأ بكثير من
# النموذجين المُقطَّرين قبله).
_WAN_FREE_CANDIDATES = [
    {
        # LTX-Video 13B المُقطَّر (Lightricks) — تحقّقنا من app.py الفعلي
        # للمساحة (Running on Zero — GPU مجاني حقيقي، ليس وسيطاً مدفوعاً).
        # ⚠️ اسم نقطة النهاية "/generate" أفضل تخمين ممكن دون اتصال حي
        # بالمساحة (لها 3 أزرار توليد بنفس الدالة لأوضاع نص/صورة/فيديو
        # مختلفة) — لو تبيّن أنه خطأ، هذه المساحة تُستبعَد بصمت تلقائياً
        # (fetch يمسكها كاستثناء عادي) وتُجرَّب المساحة التالية بالقائمة،
        # دون أي كسر. يُفضَّل التأكد من الاسم الدقيق عبر لوحة "Use via
        # API" بصفحة المساحة نفسها بعد أول رندر حي من Streamlit Cloud.
        "space": "Lightricks/ltx-video-distilled",
        "api_name": "/generate",
        "timeout": 100,
        # ترتيب المدخلات يطابق دالة generate() الفعلية: prompt,
        # negative_prompt, input_image_filepath, input_video_filepath,
        # height_ui, width_ui, mode, duration_ui, ui_frames_to_use,
        # seed_ui, randomize_seed, ui_guidance_scale, improve_texture_flag
        # دقة أعلى قليلاً + improve_texture + guidance أقوى (ما زال ضمن حدود ZeroGPU)
        "build_args": lambda prompt: (
            prompt, _wan_free_negative_prompt(), None, None,
            768, 512, "text-to-video", 4.0, 12, 0, True, 4.5, True,
        ),
        "extract": lambda result: result[0] if isinstance(result, (list, tuple)) else result,
    },
    {
        "space": "KingNish/wan2-2-fast",
        "api_name": "/generate_video",
        "timeout": 100,
        # ترتيب المدخلات يطابق app.py الفعلي للمساحة (تحقّقنا منه):
        # image, prompt, height, width, negative_prompt, duration_seconds,
        # guidance_scale, steps, seed, randomize_seed
        # خطوات أكثر + guidance معتدل لجودة أوضح مع بقاء الوقت مقبولاً
        "build_args": lambda prompt: (
            None, prompt, 832, 480, _wan_free_negative_prompt(), 4.0, 5.0, 10, 0, True,
        ),
        "extract": lambda result: result[0] if isinstance(result, (list, tuple)) else result,
    },
    {
        "space": "fffiloni/Wan2.1",
        "api_name": "/infer",
        "timeout": 110,
        "build_args": lambda prompt: (prompt,),
        "extract": lambda result: result,
    },
]

# حالات "stage" الممكنة من Hugging Face Spaces Runtime API، مُصنَّفة
# لعرض مبسّط للمستخدم (راجع check_wan_free_space_status أدناه). أي
# قيمة غير مذكورة هنا تُعرَض كما هي مع علامة "❔".
_WAN_STAGE_LABELS = {
    "RUNNING": ("🟢 يعمل الآن — جاهزة فوراً", True),
    "SLEEPING": ("🟡 نائمة (تستيقظ تلقائياً عند أول طلب، قد يستغرق قليلاً)", True),
    "PAUSED": ("🟡 موقوفة مؤقتاً (تستيقظ تلقائياً عند أول طلب)", True),
    "BUILDING": ("🟡 قيد الإقلاع الآن", True),
    "RUNNING_BUILDING": ("🟡 تعمل ويُبنى إصدار جديد بالخلفية", True),
    "STARTING": ("🟡 قيد الإقلاع الآن", True),
    "RUNTIME_ERROR": ("🔴 بها عطل حالياً — سيُستبعَد تلقائياً لهذا الرندر", False),
    "BUILD_ERROR": ("🔴 فشل البناء — سيُستبعَد تلقائياً لهذا الرندر", False),
    "DELETED": ("🔴 المساحة محذوفة", False),
    "STOPPED": ("🔴 متوقفة", False),
    "CONFIG_ERROR": ("🔴 خطأ إعداد بالمساحة", False),
}


def check_wan_free_space_status(timeout: float = 6.0) -> List[dict]:
    """يتحقّق من الحالة الحيّة الفعلية لكل مساحة Hugging Face مجانية
    (Running on Zero) مُستخدَمة لتوليد الخلفيات السينمائية، عبر REST
    API الرسمي لـ Hugging Face (huggingface.co/api/spaces/{id}/runtime)
    — طلب HTTP خفيف بدون الحاجة لتحميل gradio_client كاملاً، فقط لغرض
    الفحص السريع قبل إطلاق رندر قد يستغرق دقائق.

    يُستخدَم بالواجهة (زر "🔍 تحقّق من توفّر GPU المجاني الآن") ليُطلِع
    المستخدم مسبقاً إن كانت المساحة نائمة/بها عطل، بدل اكتشاف ذلك بعد
    انتظار طويل أثناء الرندر الفعلي.

    يُرجِع قائمة (بترتيب _WAN_FREE_CANDIDATES) من قواميس:
        {"space": "KingNish/wan2-2-fast", "stage": "RUNNING",
         "label": "🟢 يعمل الآن — جاهزة فوراً", "ok": True}

    لا يرفع أي استثناء أبداً مهما حدث (فشل شبكة/مهلة/JSON غير صالح) —
    أي خلل يُترجَم لحالة "❔ تعذّر الفحص" مع ok=False، لأن هذا فحص
    تقديري بحت ولا يجب أن يُسقط الواجهة أو يمنع محاولة الرندر الفعلي
    (الذي له تراجعه التلقائي المستقل عبر _WanFreeProvider.fetch).
    """
    results: List[dict] = []
    for candidate in _WAN_FREE_CANDIDATES:
        space = candidate["space"]
        entry = {"space": space, "stage": "UNKNOWN", "label": "❔ تعذّر الفحص (شبكة)", "ok": False}
        try:
            req = urllib.request.Request(
                f"https://huggingface.co/api/spaces/{space}/runtime",
                headers={"User-Agent": "NSM-VideoEngine/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            stage = str(data.get("stage") or "UNKNOWN").upper()
            label, ok = _WAN_STAGE_LABELS.get(stage, (f"❔ حالة غير معروفة ({stage})", False))
            entry.update(stage=stage, label=label, ok=ok)
        except Exception as exc:  # noqa: BLE001
            logger.debug("تعذّر فحص حالة مساحة Wan المجانية '%s': %s", space, exc)
        results.append(entry)
    return results


class _WanFreeProvider:
    """يدير الاتصال بمزوّدي Wan2.1/2.2 المجانيين (مساحات Hugging Face
    مجتمعية، راجع _WAN_FREE_CANDIDATES) لكل رندر فيديو واحد. نسخة واحدة
    لكل VideoEngine.render() (يُنشئها fable_engine محرّكاً جديداً بكل
    استدعاء رندر، فلا تتراكم الحالة بين فيديوهات مختلفة):

    - يعيد استخدام نفس اتصال Client لكل المشاهد بدل إعادة الاتصال من
      الصفر بكل مشهد (كل اتصال جديد يستهلك جولة شبكة إضافية لجلب
      إعدادات الواجهة) — تحسين سرعة مباشر.
    - إن فشلت مساحة معيّنة لأي سبب بمشهد واحد (تعطّل/مهلة/تغيّر واجهة)،
      تُستبعَد تلقائياً لبقية مشاهد نفس الفيديو بدل إعادة تجربتها ودفع
      نفس مهلة الانتظار الطويلة مجدداً لكل مشهد لاحق — تحسين استقرار
      وسرعة معاً، خصوصاً بالوثائقيات الطويلة (حتى 10 مشاهد أو أكثر).
    """

    def __init__(self, initial_dead: Optional[set] = None) -> None:
        self._clients: dict = {}
        # مساحات معروف تعطّلها مسبقاً (مثلاً من check_wan_free_space_status
        # عبر زر «تحقّق من التوفّر» بالواجهة) — تُستبعَد فوراً من أول مشهد
        # بدل انتظار فشلها الفعلي (مهلة قد تصل 70-110 ثانية) لاكتشاف ما
        # كان معروفاً مسبقاً. لا يمنع هذا أي مساحة لم تُفحَص أو فُحصت
        # ووُجدت سليمة من العمل بشكل طبيعي.
        self._dead: set = set(initial_dead or ())

    def fetch(self, narration: str, visual_notes: str, tmp_dir: str, seg_index: int) -> Optional[str]:
        prompt = _build_cinematic_prompt(narration, visual_notes)
        for candidate in _WAN_FREE_CANDIDATES:
            space = candidate["space"]
            if space in self._dead:
                continue
            try:
                result_path = self._call(candidate, prompt)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "مساحة Wan المجانية '%s' فشلت (%s) — تُستبعَد لبقية "
                    "مشاهد هذا الفيديو، وتُجرَّب المساحة التالية إن وُجدت.",
                    space, exc,
                )
                self._dead.add(space)
                continue

            if result_path and os.path.isfile(str(result_path)):
                clip_path = os.path.join(tmp_dir, f"wan_bg_{seg_index}.mp4")
                import shutil
                shutil.copyfile(str(result_path), clip_path)
                enhanced = os.path.join(tmp_dir, f"wan_bg_{seg_index}_hq.mp4")
                return _enhance_free_clip(clip_path, enhanced)
            # لا استثناء لكن لا ملف صالح (مثلاً خرج فارغ) — لا نستبعد
            # المساحة نهائياً (قد ينجح مشهد آخر)، فقط نجرّب المساحة
            # التالية لهذا المشهد تحديداً.

        return None

    def _call(self, candidate: dict, prompt: str):
        import concurrent.futures
        from gradio_client import Client

        space = candidate["space"]
        client = self._clients.get(space)
        if client is None:
            hf_token = os.getenv("HF_TOKEN", "").strip() or None
            client = Client(space, token=hf_token, verbose=False)
            self._clients[space] = client

        args = candidate["build_args"](prompt)

        def _predict():
            return client.predict(*args, api_name=candidate["api_name"])

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_predict)
            result = future.result(timeout=candidate["timeout"])

        return candidate["extract"](result)


def _fetch_stock_background_image(
    narration: str, visual_notes: str, seg_index: int, professional_mode: bool = False,
) -> Optional["Image.Image"]:
    """يجلب صورة خلفية مجانية من Pexels تطابق مضمون المشهد، وتُرجَع كصورة
    PIL جاهزة (بعد قصّ/تكبير 'cover' لملء الإطار 9:16 + نفس تأثير
    Vignette السينمائي المُطبَّق على التدرّج اللوني الافتراضي، لاتساق
    بصري كامل بين المشاهد بغض النظر عن مصدر الخلفية).

    يُرجِع None عند أي فشل/غياب مفتاح (تراجع صامت تماماً كـ
    _fetch_cinematic_clip — لا يرفع استثناء أبداً حتى لا يُسقط الفيديو
    بالكامل؛ في هذه الحالة النتيجة النهائية تبقى التدرّج اللوني القديم).

    Pexels API مجاني بالكامل (200 طلب/ساعة، 20,000/شهر) — يتطلب فقط
    التسجيل المجاني على https://www.pexels.com/api/ للحصول على مفتاح.
    """
    api_key = os.getenv("PEXELS_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        import numpy as np
        from PIL import Image

        # استعلام بحث: visual_notes (وصف اللقطة المقترح لهذا المشهد تحديداً)
        # أدق من narration الكاملة؛ Pexels يدعم بحثاً متعدد اللغات معقولاً،
        # لكن نوفّر احتياطاً عاماً لو جاءت النتائج فارغة (استعلام غامض/نادر).
        query = (visual_notes or narration or "").strip()[:80] or "cinematic abstract light"

        req = urllib.request.Request(
            f"https://api.pexels.com/v1/search?query={urllib.parse.quote(query)}"
            f"&per_page=1&orientation=portrait",
            headers={"Authorization": api_key, "User-Agent": "NSM-VideoEngine/1.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        photos = data.get("photos") or []
        if not photos:
            # احتياط: استعلام عام لضمان خلفية بدل الفشل الكامل لهذا المشهد
            req = urllib.request.Request(
                "https://api.pexels.com/v1/search?query=cinematic+abstract+light"
                "&per_page=1&orientation=portrait",
                headers={"Authorization": api_key, "User-Agent": "NSM-VideoEngine/1.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            photos = data.get("photos") or []
            if not photos:
                return None

        img_url = (
            photos[0].get("src", {}).get("portrait")
            or photos[0].get("src", {}).get("large2x")
            or photos[0].get("src", {}).get("original")
        )
        if not img_url:
            return None

        img_req = urllib.request.Request(img_url, headers={"User-Agent": "NSM-VideoEngine/1.0"})
        with urllib.request.urlopen(img_req, timeout=20) as resp:
            raw = Image.open(io.BytesIO(resp.read())).convert("RGB")

        # قصّ/تكبير "cover" لملء 9:16 كاملاً (نفس أسلوب _prepare_cinematic_bg_clip)
        scale = max(FRAME_W / raw.width, FRAME_H / raw.height)
        resized = raw.resize((int(raw.width * scale) + 1, int(raw.height * scale) + 1))
        left = (resized.width - FRAME_W) // 2
        top = (resized.height - FRAME_H) // 2
        cropped = resized.crop((left, top, left + FRAME_W, top + FRAME_H))

        # نفس تظليل الحواف (Vignette) المُطبَّق على التدرّج اللوني الافتراضي
        # — اتساق بصري كامل بين المشاهد بغض النظر عن مصدر الخلفية.
        arr = np.array(cropped, dtype=np.float32)
        yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W]
        cx, cy = FRAME_W / 2.0, FRAME_H / 2.0
        dist = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2).astype(np.float32)
        strength = 0.42 if professional_mode else 0.30
        floor = 0.55 if professional_mode else 0.62
        vignette = np.clip(1.0 - strength * np.clip(dist - 0.50, 0, None), floor, 1.0)
        arr *= vignette[:, :, None]
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "تعذّر جلب صورة Pexels للمشهد %d (%s) — استخدام التدرّج "
            "اللوني كبديل لهذا المشهد فقط.", seg_index, exc,
        )
        return None


def _font_actually_supports_arabic(path: str) -> bool:
    """يتحقق فعلياً (وليس افتراضاً) أن ملف الخط صالح للفتح ويحتوي فعلياً
    على تغطية Unicode لحروف عربية أساسية — لا يكفي وجود الملف على القرص:
    قد يكون تالفاً، فارغاً، أو خطاً لاتينياً محضاً بامتداد .ttf. هذا هو
    خط الدفاع الثاني بعد الإصلاح السابق (081e57a) الذي حلّ مشكلة اسم
    الملف فقط، وليس صحة محتواه أو دعمه الفعلي للعربية."""
    if not path or not os.path.isfile(path):
        return False
    try:
        from fontTools.ttLib import TTFont  # type: ignore

        tt = TTFont(path, lazy=True, fontNumber=0)
        cmap = tt.getBestCmap() or {}
        # 'ا' (ألف، U+0627) حرف عربي أساسي — أي خط عربي حقيقي يدعمه.
        return 0x0627 in cmap
    except Exception as exc:  # noqa: BLE001
        # fontTools غير متاح أو الملف تالف — نتراجع لفحص أخفّ عبر Pillow:
        # نرسم الحرف فعلياً ونتأكد أن عرضه (bbox) غير صفري (أي ليس .notdef
        # فارغاً)، بدل الافتراض الأعمى أن وجود الملف يعني أنه صالح.
        try:
            from PIL import ImageFont, Image, ImageDraw

            font = ImageFont.truetype(path, 40)
            img = Image.new("RGB", (10, 10))
            bbox = ImageDraw.Draw(img).textbbox((0, 0), "ا", font=font)
            return (bbox[2] - bbox[0]) > 0
        except Exception as exc2:  # noqa: BLE001
            logger.warning("فشل التحقق من دعم الخط %s للعربية (%s / %s)", path, exc, exc2)
            return False


def _resolve_arabic_font() -> Optional[str]:
    """يبحث عن خط عربي صالح بترتيب أولوية صارم، **مع التحقق الفعلي** من
    دعم كل مرشّح للعربية قبل قبوله (لا يكفي وجود الملف — راجع
    _font_actually_supports_arabic):
    1) خطوط عربية بالنظام  2) نسخة مخبأة محلياً من تنزيل سابق
    3) أي .ttf عربي مُضمَّن فعلياً بـ assets/fonts
    4) محاولة تنزيل من GitHub (مرة واحدة، تُخزَّن للمرات القادمة)
    5) DejaVuSans كخيار أخير مطلق (لن يعرض العربية بشكل صحيح، لكن أفضل من فشل الرسم)."""
    candidates: List[str] = list(_SYSTEM_FONT_CANDIDATES)
    if _FONT_CACHE_PATH.is_file():
        candidates.append(str(_FONT_CACHE_PATH))
    if _FONT_CACHE_DIR.is_dir():
        candidates.extend(str(p) for p in sorted(_FONT_CACHE_DIR.glob("*.ttf")))

    rejected: List[str] = []
    for path in candidates:
        if _font_actually_supports_arabic(path):
            return path
        rejected.append(path)

    if rejected:
        logger.warning(
            "وُجدت %d ملف(ات) خط لكن رُفضت جميعها (تالفة أو لا تدعم العربية "
            "فعلياً رغم وجودها بالقرص): %s", len(rejected), ", ".join(rejected),
        )

    try:
        import urllib.request

        _FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_FONT_FALLBACK_URL, _FONT_CACHE_PATH)
        if _font_actually_supports_arabic(str(_FONT_CACHE_PATH)):
            logger.info("تم تنزيل خط عربي احتياطي والتحقق من صلاحيته: %s", _FONT_CACHE_PATH)
            return str(_FONT_CACHE_PATH)
        logger.warning("الخط المُنزَّل احتياطياً فشل فحص دعم العربية أيضاً: %s", _FONT_CACHE_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "تعذّر إيجاد/تنزيل خط عربي صالح (%s) — سيُستخدم خط بديل لا يدعم "
            "العربية بشكل صحيح. أضِف 'fonts-noto-core' لـ packages.txt "
            "(Streamlit Cloud) لحل هذا بشكل دائم.", exc,
        )

    return _LAST_RESORT_FONT if os.path.isfile(_LAST_RESORT_FONT) else None


def _ease_out_cubic(x: float) -> float:
    """منحنى تسارع/تباطؤ (easing) بدل الحركة الخطّية — أساس أي موشن جرافيك
    احترافي (After Effects/CapCut تستخدم نفس المنطق لكل انتقال): البداية
    سريعة والنهاية ناعمة، بعكس الحركة الخطّية الميكانيكية الملحوظة سابقاً."""
    x = min(1.0, max(0.0, x))
    return 1.0 - (1.0 - x) ** 3


def _make_particles_layer(duration: float, seed: int, size=(FRAME_W, FRAME_H), count: int = 26):
    """طبقة "بوكيه" من جزيئات ضوء ناعمة تطفو للأعلى ببطء طوال المشهد —
    لمسة موشن جرافيك خفيفة (بنفس روح خلفيات After Effects الاحترافية)
    تعمل فوق أي خلفية (متدرّجة أو سينمائية) دون التأثير على وضوح النص،
    لأنها شفافة جداً ومحدودة العدد."""
    import numpy as np
    from moviepy import VideoClip

    rng = np.random.default_rng(seed)
    w, h = size
    n = count
    x0 = rng.uniform(0, w, n)
    y0 = rng.uniform(0, h, n)
    speed = rng.uniform(14, 34, n)          # بكسل/ثانية صعوداً
    # أقطار صغيرة جداً (غبار/بريق) بدل "كرات" واضحة — أسلوب bokeh خفيف
    # يُلاحَظ بزاوية العين لا يتصادم مع النص أبداً.
    radius = rng.uniform(1.3, 3.2, n)
    phase = rng.uniform(0, 2 * np.pi, n)
    drift_amp = rng.uniform(3, 10, n)
    base_alpha = rng.uniform(0.05, 0.14, n)
    # نطاق التصحيح المحلي (patch) حول كل جزيء — أوسع من نصف قطره بكثير
    # لضمان انتقال ناعم بلا حواف مقصوصة، لكن أصغر بكثير جداً من الإطار
    # الكامل. ⚠️ حاسم للأداء: حساب مسافة Gaussian على الإطار الكامل
    # (1080×1920) لكل جزيء وكل إطار كان يستغرق ~0.84 ثانية/إطار (أبطأ
    # بمقدار ~380× من هذا التصحيح المحلي) — كافٍ لجعل رندر فيديو Shorts
    # كامل يتجاوز دقائق طويلة بلا فائدة بصرية إضافية تُذكر خارج هذا النطاق
    # الصغير أصلاً (قيمة Gaussian تقترب من الصفر خارجه).
    box_half = 10

    def make_alpha_frame(t):
        alpha = np.zeros((h, w), dtype=np.float32)
        y_t = (y0 - speed * t) % h
        x_t = x0 + drift_amp * np.sin(phase + t * 0.6)
        for i in range(n):
            cx, cy, r, a = x_t[i], y_t[i], radius[i], base_alpha[i]
            x_lo, x_hi = max(0, int(cx - box_half)), min(w, int(cx + box_half) + 1)
            y_lo, y_hi = max(0, int(cy - box_half)), min(h, int(cy + box_half) + 1)
            if x_lo >= x_hi or y_lo >= y_hi:
                continue
            yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
            dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
            alpha[y_lo:y_hi, x_lo:x_hi] += np.exp(-dist2 / (2 * (r * 1.8) ** 2)) * a
        return np.clip(alpha, 0, 0.32)

    def make_rgb_frame(_t):
        return np.full((h, w, 3), 255, dtype=np.uint8)

    rgb_clip = VideoClip(make_rgb_frame, duration=duration).with_fps(FPS)
    mask_clip = VideoClip(make_alpha_frame, duration=duration, is_mask=True).with_fps(FPS)
    return rgb_clip.with_mask(mask_clip)


def _synthesize_ambient_bed(duration: float, seed: int = 0):
    """يولّد "سجادة" صوتية محيطية (ambient pad) داخلياً عبر numpy — بدون أي
    ملف موسيقى خارجي أو استدعاء API — لتفادي أي إشكال حقوق ملكية أو
    اعتماد على مزوّد صوتي مدفوع (بنفس فلسفة الخلفية المتدرّجة اللونية:
    حل مضمون ومجاني دائماً). النتيجة نغمة/طبقات هادئة بلا إيقاع أو لحن
    واضح (drone/pad) — أقرب لضجيج محيطي منظّم من "موسيقى" بمعناها
    التقليدي، وتُبقى منخفضة جداً عبر music_volume قبل مزجها مع السرد.

    يعيد moviepy AudioClip (ستيريو) بطول duration بالضبط."""
    import numpy as np
    from moviepy.audio.AudioClip import AudioClip

    rng = np.random.default_rng(seed)
    sr = 44100
    # طبقتان/ثلاث نغمات متجاورة (chord) بترددات منخفضة هادئة — بلا
    # نغمة "لحنية" متحركة، فقط طبقة ثابتة تتنفّس ببطء عبر LFO خفيف.
    base_freq = float(rng.uniform(110, 146))  # نطاق A2-D3 تقريباً
    partials = [1.0, 1.5, 2.0]  # أساس + خامسة + أوكتاف — تناغم بسيط محايد
    partial_gains = [0.55, 0.28, 0.17]
    lfo_freq = float(rng.uniform(0.06, 0.11))  # تنفّس بطيء جداً (~10-16 ثانية)
    lfo_phase = float(rng.uniform(0, 2 * np.pi))
    # اهتزاز طفيف جداً بالتردد (Detune) لإحساس "حي" بدل نغمة رقمية جامدة
    detune = float(rng.uniform(0.997, 1.003))

    def make_frame(t):
        t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
        signal = np.zeros_like(t_arr)
        for mult, gain in zip(partials, partial_gains):
            freq = base_freq * mult * detune
            signal += gain * np.sin(2 * np.pi * freq * t_arr)
        # مغلّف سعة بطيء (breathing envelope) — يمنع إحساس "طنين" ثابت مزعج
        envelope = 0.75 + 0.25 * np.sin(2 * np.pi * lfo_freq * t_arr + lfo_phase)
        signal *= envelope
        # تطبيع لطيف لحماية من أي تجاوز قبل تخفيض الحجم النهائي بالمزج
        signal = np.clip(signal * 0.28, -1.0, 1.0)
        stereo = np.stack([signal, signal], axis=-1)
        return stereo if stereo.shape[0] > 1 else stereo[0]

    return AudioClip(make_frame, duration=duration, fps=sr)


def _shape_arabic(text: str) -> str:
    """يهيّئ النص العربي للعرض الصحيح (اتصال الحروف + اتجاه RTL)."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:  # noqa: BLE001
        return text  # نص لاتيني أو فشل التشكيل — يُعرض كما هو


class VideoEngine:
    """يحوّل ExplainerScript (مع صوت مُولَّد مسبقاً) إلى فيديو mp4 فعلي."""

    # ── قوالب تصميم النصوص والعناوين الجاهزة (Shorts/TikTok) ────────
    # كل قالب: style = نمط الرسم داخل _draw_caption ("pill"/"bottom"/
    # "neon"/"headline"/"top"/"highlight"/"dramatic")، color = اللون
    # الفاقع الافتراضي إن لم تُمرَّر accent_color، name/desc = العرض
    # في الواجهة (عربية RTL) — قائمة CAPTION_TEMPLATES هي المرجع الوحيد
    # للقوالب المتاحة؛ أي قالب جديد يُضاف هنا ويظهر تلقائيًا.
    CAPTION_TEMPLATES = {
        "classic_pill": {
            "name": "حاجز ملوّن (الافتراضي)",
            "desc": "نص عريض فوق حاجز ملوّن بالمنتصف — أعلى قابلية قراءة، أقرب لأسلوب CapCut/Submagic.",
            "style": "pill", "color": (255, 199, 0),
        },
        "classic_lower": {
            "name": "كلاسيكي أسفل الشاشة",
            "desc": "نص أبيض بحد أسود أسفل الفيديو — نفس أسلوب اليوتيوب الكلاسيكي، لا يغطي وسط اللقطة.",
            "style": "bottom", "color": None,
        },
        "neon": {
            "name": "متوهّج (نيون)",
            "desc": "نص أبيض متوهّج ملوّن بلا حاجز — تيرند تيك توك الأشهر، يعطي إحساس حيوي على الخلفيات الداكنة.",
            "style": "neon", "color": (255, 92, 92),
        },
        "headline": {
            "name": "عنوان ضخم",
            "desc": "عنوان ثخين جدًّا (112) بحد أبيض وأسود بالمنتصف — Hook قوي يحبس العين في أول ثانية.",
            "style": "headline", "color": (255, 255, 255),
        },
        "top_banner": {
            "name": "شريط علوي",
            "desc": "شريط داكن نصف شفاف أعلى الشاشة مع نص أبيض — مناسب للتحليل والأخبار.",
            "style": "top", "color": None,
        },
        "highlighted": {
            "name": "الكلمة البارزة",
            "desc": "الكلمة الأولى بلون فاقع داخل حاجز + بقية النص أبيض — نمط الـHook السريع المنتشر.",
            "style": "highlight", "color": (94, 211, 255),
        },
        "dramatic": {
            "name": "دراماتيكي",
            "desc": "نص أحمر دموي بحد أبيض سميك وظل أسود — قصص الغموض والإثارة والرعب.",
            "style": "dramatic", "color": (205, 35, 35),
        },
    }

    def __init__(
        self,
        use_cinematic_backgrounds: bool = False,
        cinematic_provider: str = "higgsfield",
        use_stock_backgrounds: bool = True,
        use_background_music: bool = False,
        music_volume: float = 0.10,
        wan_skip_spaces: Optional[set] = None,
        professional_mode: bool = False,
        cinematic_strategy: str = "hero",
        caption_template: str = "classic_pill",
    ) -> None:
        self._font_path = _resolve_arabic_font()
        # قالب التصميم النصي المستخدم فوق الفيديو (Shorts/TikTok):
        #   classic_pill = حاجز ملوّن خلف النص بالمنتصف (الأسلوب الحالي)
        #   classic_lower = نص أبيض بحد أسود أسفل الشاشة (يوتيوب الكلاسيكي)
        #   neon = نص متوهّج ملوّن بلا حاجز (تيرند تيك توك)
        #   headline = عنوان ضخم ثخين وسط الشاشة (Hook قوي)
        #   top_banner = شريط علوي نصف شفاف مع نص (أخبار/تحليل)
        #   highlighted = الكلمة الأولى بلون فاقع داخل حاجز صغير (CapCut)
        #   dramatic = نص أحمر-أبيض دراماتيكي بحدّ سميك
        # أي قيمة غير معروفة → يتراجع silently إلى classic_pill.
        if caption_template not in VideoEngine.CAPTION_TEMPLATES:
            caption_template = "classic_pill"
        self._caption_template = caption_template
        # وضع احترافي لـ Shorts: جودة ترميز أعلى، شريط تقدّم، انتقالات أنعم،
        # تظليل vignette، وموسيقى محيطية خفيفة افتراضياً إن طُلب.
        self._professional_mode = bool(professional_mode)
        # استراتيجية المشاهد السينمائية (مهمّة للمسار المجاني البطيء):
        #   hero       = أول + أوسط + آخر مشهد فقط (افتراضي — توازن جودة/وقت)
        #   first_last = أول وآخر فقط
        #   all        = كل المشاهد (أبطأ، طوابير GPU أطول)
        self._cinematic_strategy = (
            cinematic_strategy if cinematic_strategy in ("hero", "first_last", "all") else "hero"
        )
        # اختياري (opt-in) — راجع شرح الميزة في رأس الملف. لا يُفعَّل أبداً
        # ضمنياً حتى لا يستهلك رصيد Higgsfield المدفوع دون طلب صريح.
        self._use_cinematic_backgrounds = use_cinematic_backgrounds
        # المزوّد عند تفعيل الخلفيات السينمائية: "higgsfield" (مدفوع،
        # أسرع وأدق) أو "wan_free" (Wan2.1 مفتوح المصدر عبر مساحة Hugging
        # Face مجتمعية مجانية — أبطأ وأقل ثباتاً، لكن بدون أي تكلفة).
        _prov = (cinematic_provider or "wan_free").strip().lower()
        if _prov in ("free", "auto_free", "wan", "wan_free"):
            _prov = "wan_free"
        elif _prov not in ("higgsfield", "wan_free"):
            _prov = "wan_free"  # تفضيل المسار المجاني افتراضياً
        self._cinematic_provider = _prov
        # مثيل _WanFreeProvider واحد يُنشأ عند الحاجة فقط (lazy) ويُعاد
        # استخدامه لكل مشاهد نفس الفيديو — راجع شرح الكلاس أعلاه.
        self._wan_free_provider: Optional["_WanFreeProvider"] = None
        # مساحات معروف تعطّلها مسبقاً (من فحص حيّ سابق بالواجهة) — تُمرَّر
        # لـ_WanFreeProvider عند إنشائه لتُستبعَد فوراً بدل انتظار فشلها.
        self._wan_skip_spaces = set(wan_skip_spaces or ())
        # صور stock مجانية (Pexels) بديلة للتدرّج اللوني الفارغ — مفعَّلة
        # افتراضياً (بعكس Higgsfield) لأنها مجانية بالكامل ولا خطر تكلفة؛
        # تتراجع تلقائياً وبصمت للتدرّج اللوني القديم عند غياب PEXELS_API_KEY
        # أو أي فشل شبكي/نتائج فارغة — لا تأثير على المسار القديم إطلاقاً.
        self._use_stock_backgrounds = use_stock_backgrounds
        # موسيقى خلفية — اختياري (opt-in)، مُعطَّلة افتراضياً عمداً: محتوى
        # المشروع معرفي إسلامي وبعض المستخدمين/الجمهور يُفضّل عدم وجود
        # موسيقى آلية إطلاقاً، فلا يجب تفعيلها ضمنياً أبداً. عند التفعيل،
        # تُولَّد "سجادة" صوتية محيطية (ambient bed) داخلياً عبر numpy —
        # وليست مقطوعة موسيقية جاهزة مُحمَّلة من الإنترنت — لتفادي أي
        # إشكال حقوق ملكية أو اعتماد على مزوّد خارجي.
        self._use_background_music = use_background_music
        # نطاق آمن: منخفضة بما يكفي حتى لا تطغى على السرد الصوتي أبداً
        # (نمط "الدَك" duck تحت الصوت الرئيسي بالإنتاج الاحترافي).
        if self._professional_mode:
            # رفع جودة الإحساس البصري دون إجبار موسيقى أو مزوّد مدفوع
            if use_stock_backgrounds:
                self._use_stock_backgrounds = True
        self._music_volume = max(0.0, min(0.35, music_volume))

    # ── بناء صورة خلفية متدرّجة للمشهد رقم N ─────────────────────────
    def _build_background(self, seg_index: int) -> "Image.Image":
        import numpy as np
        from PIL import Image

        top, bottom = _GRADIENT_PAIRS[seg_index % len(_GRADIENT_PAIRS)]

        # تدرّج رأسي فعلي لكل بكسل (بدل حلقة يدوية بخطوة 4px) — أدق وأسرع
        # عبر numpy المُتَّجه (vectorized)، بلا أي تكلفة إضافية.
        t = np.linspace(0.0, 1.0, FRAME_H, dtype=np.float32).reshape(FRAME_H, 1)
        top_arr = np.array(top, dtype=np.float32)
        bottom_arr = np.array(bottom, dtype=np.float32)
        row_colors = top_arr + (bottom_arr - top_arr) * t          # (FRAME_H, 3)
        arr = np.broadcast_to(row_colors[:, None, :], (FRAME_H, FRAME_W, 3)).copy()

        # تظليل خفيف بالحواف (Vignette) — يعطي عمقاً سينمائياً بسيطاً
        # للخلفية المتدرّجة المجانية (بدون الحاجة لخلفيات Higgsfield المدفوعة)،
        # بنفس أسلوب أدوات الفيديو الاحترافية (CapCut/Submagic/Premiere).
        yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W]
        cx, cy = FRAME_W / 2.0, FRAME_H / 2.0
        dist = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2).astype(np.float32)
        strength = 0.42 if getattr(self, "_professional_mode", False) else 0.30
        floor = 0.55 if getattr(self, "_professional_mode", False) else 0.62
        vignette = np.clip(1.0 - strength * np.clip(dist - 0.50, 0, None), floor, 1.0)
        arr *= vignette[:, :, None]

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    # ── تقسيم النص لعبارات قصيرة (كلمة-بكلمة/عبارة-بعبارة) ──────────
    @staticmethod
    def _split_into_chunks(text: str, max_words: int = 3) -> List[str]:
        words = text.split()
        if not words:
            return [text] if text else [""]
        return [
            " ".join(words[i:i + max_words])
            for i in range(0, len(words), max_words)
        ]

    # ── تجميع الترجمات حسب توقيت الكلمات الحقيقي (WordBoundary من Edge
    #    TTS) بدل التقدير التناسبي حسب عدد الحروف — مزامنة فعلية للنص مع
    #    الصوت المنطوق (بالضبط كما تفعل أدوات professional captioning مثل
    #    CapCut/Submagic التي تعتمد على ASR/TTS timestamps حقيقية) ───────
    @staticmethod
    def _group_word_timings(
        word_timings: List[tuple], duration: float, max_words: int = 3,
    ) -> Optional[List[Tuple[str, float, float]]]:
        """يُرجِع [(نص المجموعة, بداية بالثانية, مدة بالثانية), ...] تغطي
        كامل المدة [0, duration] بلا فجوات، أو None إن كانت التوقيتات غير
        صالحة (عدد غير منطقي أو توقيتات معكوسة) — VideoEngine يتراجع
        فوراً للتقدير التناسبي القديم في هذه الحالة."""
        if not word_timings:
            return None
        try:
            words = [(str(w[0]), float(w[1]), float(w[2])) for w in word_timings]
        except (TypeError, ValueError, IndexError):
            return None
        if any(d < 0 or s < 0 for _, s, d in words):
            return None

        groups: List[Tuple[str, float, float]] = []
        for i in range(0, len(words), max_words):
            batch = words[i:i + max_words]
            text = " ".join(w[0] for w in batch)
            start = batch[0][1]
            end = batch[-1][1] + batch[-1][2]
            groups.append([text, start, end])  # end مؤقتاً بدل المدة

        if not groups:
            return None

        # لا فجوات: كل مجموعة تبدأ حيث انتهت السابقة، أول مجموعة من 0،
        # وآخر مجموعة تمتد حتى نهاية الصوت الفعلية بالكامل.
        groups[0][1] = 0.0
        for i in range(1, len(groups)):
            groups[i][1] = groups[i - 1][2]
        groups[-1][2] = max(duration, groups[-1][2])

        result = [
            (text, start, max(0.15, end - start))
            for text, start, end in groups
        ]
        return result

    # ── رسم عبارة نصية واحدة بأسلوب الترجمات المتحركة (Kinetic Caption):
    #    حاجز (pill) ملوّن خلف نص عريض بحدّ أبيض، أعلى قابلية للقراءة
    #    وأقرب لأسلوب CapCut/Submagic الاحترافي ──────────────────────
    def _draw_caption(
        self,
        img: "Image.Image",
        text: str,
        font_size: int = 84,
        accent_color: Optional[Tuple[int, int, int]] = None,
        template: Optional[str] = None,
    ) -> "Image.Image":
        """يرسم عبارة نصية واحدة بأسلوب قالب التصميم المحدد (Shorts/TikTok).
        template: معرّف قالب من CAPTION_TEMPLATES (مثل "neon", "headline").
        إن لم يُحدَّد يُستخدم قالب الفيديو الافتراضي (self._caption_template).
        accent_color: اللون الفاقع للقالب (يُختار تلقائيًا بالدوران على
        _ACCENT_COLORS إن تركته None)."""
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(img, "RGBA")
        try:
            font = ImageFont.truetype(self._font_path, font_size) if self._font_path else ImageFont.load_default()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "فشل تحميل الخط العربي المُختار (%s) أثناء الرسم الفعلي — "
                "تراجع لخط Pillow الافتراضي (لا يدعم العربية): %s",
                self._font_path, exc,
            )
            font = ImageFont.load_default()

        _ct = VideoEngine.CAPTION_TEMPLATES
        tpl = _ct.get(template or self._caption_template) or _ct["classic_pill"]
        style = tpl["style"]
        color = accent_color or tpl.get("color") or (255, 199, 0)
        # ⚠️ مهم جداً — ترتيب العمليات هنا يمنع مشكلة النص المشوّه/المبعثر:
        # يجب لفّ السطور بالترتيب المنطقي الأصلي (حسب الكلمات) *قبل* تطبيق
        # التشكيل (reshape) وBiDi. تطبيق get_display (الذي يعكس النص لترتيب
        # العرض البصري) ثم تمرير الناتج إلى textwrap.wrap لاحقاً يقسّم سطراً
        # مُعاد ترتيبه بصرياً بالفعل حسب عدّ الأحرف، فتُقطَّع الكلمات في
        # منتصف تسلسلها البصري وتظهر متكسّرة/معكوسة — بالضبط الخلل السابق.
        stroke_w = max(4, font_size // 14)
        # سماكة الحد الموحّدة للقوالب (تُستخدم في highlight/dramatic).
        sw1 = max(4, font_size // 14)
        logical_lines = textwrap.wrap(text, width=16) or [text]
        wrapped_lines = [_shape_arabic(line) for line in logical_lines]

        line_heights: List[int] = []
        line_widths: List[int] = []
        for line in wrapped_lines:
            bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_w)
            line_heights.append(bbox[3] - bbox[1])
            line_widths.append(bbox[2] - bbox[0])
        total_h = sum(line_heights) + max(0, len(wrapped_lines) - 1) * 22

        # ── قوالب التصميم الجاهزة (Shorts/TikTok) ────────────────────
        # كل قالب يحدد مكان النص (y) وطريقة رسمه؛ القالب الافتراضي
        # classic_pill هو سلوك الرسم الأصلي للحفاظ على التوافق التام.
        if style == "bottom":
            # كلاسيكي أسفل الشاشة: نص أبيض بحد أسود (بدون حاجز) — نفس أسلوب
            # اليوتيوب الكلاسيكي، لا يغطي مركز الفيديو.
            y = FRAME_H - total_h - max(180, int(FRAME_H * 0.12))
            for line, lh, lw in zip(wrapped_lines, line_heights, line_widths):
                x = (FRAME_W - lw) // 2
                draw.text(
                    (x, y), line, font=font,
                    fill=(255, 255, 255), stroke_width=stroke_w + 2,
                    stroke_fill=(0, 0, 0),
                )
                y += lh + 22
            return img
        if style == "neon":
            # متوهّج: طبقة ظل ملونة (glow) خلف النص ثم نص أبيض نظيف —
            # أسلوب منتشِر في تيرندات تيك توك، بلا أي حاجز.
            y = (FRAME_H - total_h) // 2
            glow = (color[0], color[1], color[2], 230)
            for line, lh, lw in zip(wrapped_lines, line_heights, line_widths):
                x = (FRAME_W - lw) // 2
                draw.text((x + 2, y + 2), line, font=font, fill=glow)
                draw.text((x - 2, y - 2), line, font=font, fill=glow)
                draw.text((x + 2, y - 2), line, font=font, fill=glow)
                draw.text((x - 2, y + 2), line, font=font, fill=glow)
                draw.text(
                    (x, y), line, font=font,
                    fill=(255, 255, 255), stroke_width=max(2, font_size // 28),
                    stroke_fill=color,
                )
                y += lh + 22
            return img
        if style == "headline":
            # عنوان ضخم ثخين: خط أكبر (112) حدّ سميك أسود + ظل أبيض —
            # Hook قوي لأول ثانية من الـShort.
            big_font = font
            if self._font_path:
                try:
                    big_font = ImageFont.truetype(self._font_path, 112)
                except Exception:
                    pass
            big_lines = textwrap.wrap(text, width=10) or [text]
            big_wrapped = [_shape_arabic(l) for l in big_lines]
            b_heights, b_widths = [], []
            for bl in big_wrapped:
                bb = draw.textbbox((0, 0), bl, font=big_font, stroke_width=6)
                b_heights.append(bb[3] - bb[1])
                b_widths.append(bb[2] - bb[0])
            b_total = sum(b_heights) + max(0, len(big_wrapped) - 1) * 26
            y = (FRAME_H - b_total) // 2
            for bl, bh, bw in zip(big_wrapped, b_heights, b_widths):
                x = (FRAME_W - bw) // 2
                draw.text((x, y), bl, font=big_font, fill=(0, 0, 0))
                draw.text(
                    (x, y), bl, font=big_font,
                    fill=(255, 255, 255), stroke_width=6, stroke_fill=(0, 0, 0),
                )
                y += bh + 26
            return img
        if style == "top":
            # شريط علوي نصف شفاف + نص داكن (أسلوب الأخبار والتحليل) —
            # يترك ثلثي الشاشة سفلاً لللقطات.
            pad_x, pad_y = 36, 22
            block_w = max(line_widths) + pad_x * 2
            block_h = total_h + pad_y * 2
            bx = (FRAME_W - block_w) // 2
            by = max(120, int(FRAME_H * 0.10)) - pad_y
            draw.rounded_rectangle(
                [bx, by, bx + block_w, by + block_h],
                radius=24, fill=(15, 15, 20, 205),
            )
            y = by + pad_y
            for line, lh, lw in zip(wrapped_lines, line_heights, line_widths):
                x = (FRAME_W - lw) // 2
                draw.text(
                    (x, y), line, font=font,
                    fill=(255, 255, 255), stroke_width=max(2, font_size // 30),
                    stroke_fill=(0, 0, 0),
                )
                y += lh + 22
            return img
        if style == "highlight":
            # الكلمة الأولى ملونة داخل حاجز صغير + بقية النص أبيض —
            # نمط CapCut السائد في فيديوهات الـHook السريعة.
            words = text.strip().split()
            first = words[0] if words else ""
            rest = " ".join(words[1:]) if len(words) > 1 else ""
            first_wrapped = [_shape_arabic(first)]
            rest_wrapped = [_shape_arabic(l) for l in textwrap.wrap(rest, width=16)] if rest else []
            fb = draw.textbbox((0, 0), first_wrapped[0], font=font, stroke_width=sw1)
            fw, fh = fb[2] - fb[0], fb[3] - fb[1]
            pad_x, pad_y = 28, 20
            block_w = fw + pad_x * 2
            block_h = fh + pad_y * 2
            bx = (FRAME_W - block_w) // 2
            by = (FRAME_H - block_h) // 2 - 60
            shadow_offset = 10
            draw.rounded_rectangle(
                [bx + shadow_offset, by + shadow_offset,
                 bx + block_w + shadow_offset, by + block_h + shadow_offset],
                radius=28, fill=(0, 0, 0, 90),
            )
            draw.rounded_rectangle(
                [bx, by, bx + block_w, by + block_h],
                radius=28, fill=(*color, 240),
            )
            draw.text(
                (bx + (block_w - fw) // 2, by + pad_y), first_wrapped[0],
                font=font, fill=(18, 14, 10), stroke_width=sw1, stroke_fill=(255, 255, 255),
            )
            y = by + block_h + 30
            for line in rest_wrapped:
                bb = draw.textbbox((0, 0), line, font=font, stroke_width=sw1)
                lw = bb[2] - bb[0]
                draw.text(
                    ((FRAME_W - lw) // 2, y), line, font=font,
                    fill=(255, 255, 255), stroke_width=sw1, stroke_fill=(0, 0, 0),
                )
                y += (bb[3] - bb[1]) + 22
            return img
        if style == "dramatic":
            # دراماتيكي: نص أحمر دموي بحدّ أبيض سميك + ظل أسود —
            # مناسب لقصص الغموض والإثارة (ترند الرعب).
            y = (FRAME_H - total_h) // 2
            for line, lh, lw in zip(wrapped_lines, line_heights, line_widths):
                x = (FRAME_W - lw) // 2
                draw.text((x + 3, y + 3), line, font=font, fill=(0, 0, 0, 180))
                draw.text(
                    (x, y), line, font=font,
                    fill=(205, 35, 35), stroke_width=sw1 + 4, stroke_fill=(255, 255, 255),
                )
                y += lh + 22
            return img

        # classic_pill (الافتراضي — السلوك الأصلي):
        y = (FRAME_H - total_h) // 2

        pad_x, pad_y = 44, 26
        block_w = max(line_widths) + pad_x * 2
        block_h = total_h + pad_y * 2
        bx = (FRAME_W - block_w) // 2
        by = y - pad_y
        # ظل ناعم أسفل الحاجز (drop shadow) — يفصله بصرياً عن الخلفية
        # ويعطي إحساس عمق/ارتفاع (elevation) بنفس أسلوب تصميم الحركة
        # الاحترافي (Material/After Effects)، بدل الحاجز المستوي تماماً
        # على الخلفية سابقاً.
        shadow_offset = 10
        draw.rounded_rectangle(
            [bx + shadow_offset, by + shadow_offset,
             bx + block_w + shadow_offset, by + block_h + shadow_offset],
            radius=32, fill=(0, 0, 0, 90),
        )
        draw.rounded_rectangle(
            [bx, by, bx + block_w, by + block_h],
            radius=32, fill=(*color, 235),
        )
        text_fill = (18, 14, 10)
        stroke_fill = (255, 255, 255)

        for line, lh, lw in zip(wrapped_lines, line_heights, line_widths):
            x = (FRAME_W - lw) // 2
            draw.text(
                (x, y), line, font=font,
                fill=text_fill, stroke_width=stroke_w, stroke_fill=stroke_fill,
            )
            y += lh + 22
        return img

    # ── بناء مقطع فيديو واحد (مشهد) بالصوت المرافق — بأسلوب الترجمات
    #    المتحركة كلمة/عبارة-بعبارة (Kinetic Captions) ─────────────────
    _ACCENT_COLORS = [
        (255, 199, 0),    # أصفر ذهبي
        (255, 92, 92),    # أحمر مرجاني
        (94, 211, 255),   # سماوي
        (178, 130, 255),  # بنفسجي
        (110, 231, 172),  # أخضر نعناعي
    ]

    # ── تحضير مقطع خلفية سينمائي (Higgsfield، 16:9) ليغطي الإطار العمودي
    #    9:16 بالكامل (تكبير حسب الارتفاع ثم قصّ العرض الزائد من المنتصف
    #    — أسلوب "cover" المعتاد بالتصميم الاحترافي)، ويطابق مدة المشهد
    #    (تكرار إن كان أقصر، أو قصّ إن كان أطول) ──────────────────────
    @staticmethod
    def _prepare_cinematic_bg_clip(clip_path: str, duration: float):
        from moviepy import VideoFileClip, vfx

        src = VideoFileClip(clip_path).without_audio()

        scale = FRAME_H / src.h
        resized = src.resized(scale)
        if resized.w >= FRAME_W:
            resized = resized.cropped(x_center=resized.w / 2, width=FRAME_W, height=FRAME_H)
        else:
            # نادر (فيديو أعرض بمناسبة غير 16:9) — نكبّر حسب العرض بدلاً
            # من الارتفاع لضمان تغطية الإطار كاملاً دون أشرطة سوداء.
            resized = src.resized(FRAME_W / src.w)
            resized = resized.cropped(y_center=resized.h / 2, width=FRAME_W, height=FRAME_H)

        if resized.duration < duration:
            resized = resized.with_effects([vfx.Loop(duration=duration)])
        else:
            resized = resized.subclipped(0, duration)
        return resized


    def _should_fetch_cinematic(self, index: int) -> bool:
        """هل نطلب خلفية مولَّدة لهذا المشهد؟ (يوفر وقت/طابور في المسار المجاني)."""
        if not self._use_cinematic_backgrounds:
            return False
        total = max(1, int(getattr(self, "_pro_total_segments", 1)))
        strat = getattr(self, "_cinematic_strategy", "hero")
        if strat == "all":
            return True
        if strat == "first_last":
            return index == 0 or index >= total - 1
        # hero
        if total <= 2:
            return True
        return index in (0, total // 2, total - 1)

    def _build_segment_clip(self, segment, index: int, tmp_dir: str):
        from moviepy import AudioFileClip, ImageClip, concatenate_videoclips
        import numpy as np

        if not segment.audio_bytes:
            raise VideoEngineError(
                f"المشهد {segment.index} بدون صوت — استدعِ render_audio() أولاً."
            )

        audio_path = os.path.join(tmp_dir, f"seg_{index}.{segment.audio_format or 'mp3'}")
        with open(audio_path, "wb") as f:
            f.write(segment.audio_bytes)

        audio_clip = AudioFileClip(audio_path)
        # تطبيع مستوى الصوت (loudness normalize) لكل مشهد — مزوّدو TTS
        # المختلفين (Edge/gTTS/Gemini/ElevenLabs) قد يُنتِجون مستويات صوت
        # متفاوتة بين مقطع وآخر بنفس السيناريو، فيُحسّ المستخدم بقفزات
        # حجم مزعجة عند التنقّل بين المشاهد. AudioNormalize يوحّد الذروة
        # لكل مقطع على حدة قبل الدمج. اختياري وآمن تماماً: أي فشل (مثلاً
        # مقطع صامت بالكامل) يتراجع بصمت للصوت الأصلي دون كسر الفيديو.
        try:
            from moviepy import afx
            audio_clip = audio_clip.with_effects([afx.AudioNormalize()])
        except Exception as exc:  # noqa: BLE001
            logger.debug("تعذّر تطبيع صوت المشهد %d (%s) — استخدام الصوت الأصلي.", index, exc)
        duration = max(1.2, audio_clip.duration)

        # خلفية سينمائية حقيقية (Higgsfield، اختياري) إن كانت مفعّلة ومتاحة
        # لهذا المشهد تحديداً — وإلا نتراجع فوراً للخلفية المتدرّجة المجانية
        # دون أي تأثير على بقية الفيديو.
        cinematic_bg = None
        if self._use_cinematic_backgrounds and self._should_fetch_cinematic(index):
            if self._cinematic_provider == "wan_free":
                if self._wan_free_provider is None:
                    self._wan_free_provider = _WanFreeProvider(initial_dead=self._wan_skip_spaces)
                clip_path = self._wan_free_provider.fetch(
                    segment.narration, segment.visual_notes, tmp_dir, index,
                )
            else:
                clip_path = _fetch_cinematic_clip(
                    segment.narration, segment.visual_notes, tmp_dir, index,
                )
            if clip_path:
                try:
                    cinematic_bg = self._prepare_cinematic_bg_clip(clip_path, duration)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "تعذّر تجهيز خلفية Higgsfield للمشهد %d (%s) — "
                        "استخدام الخلفية المتدرّجة كبديل.", index, exc,
                    )
                    cinematic_bg = None

        # عند وجود خلفية سينمائية: الترجمات تُرسم على طبقة شفافة منفصلة
        # (بدل الدمج بالخلفية مباشرة) ثم تُركَّب فوق الفيديو الحقيقي.
        # الأولوية: Higgsfield (فيديو حقيقي متحرك، إن فُعِّل ونجح) → صورة
        # Pexels مجانية (أفضل من التدرّج الفارغ، مفعَّلة افتراضياً) →
        # التدرّج اللوني كحل أخير مضمون دائماً.
        stock_bg_image = None
        if cinematic_bg is None and self._use_stock_backgrounds:
            stock_bg_image = _fetch_stock_background_image(
                segment.narration, segment.visual_notes, index,
                professional_mode=getattr(self, "_professional_mode", False),
            )
        bg_base = (
            stock_bg_image if stock_bg_image is not None
            else (None if cinematic_bg is not None else self._build_background(index))
        )

        # نقسّم سرد المشهد إلى عبارات قصيرة (2-3 كلمات) تظهر تباعاً —
        # نفس منطق CapCut/Submagic لترجمات كلمة-بكلمة أكثر جاذبية من فقرة
        # ثابتة كاملة طوال المشهد. الأولوية لتوقيت الكلمات الحقيقي (Edge
        # TTS WordBoundary) إن توفّر — مزامنة فعلية دقيقة بالصوت المنطوق؛
        # وإلا تراجع للتقدير التناسبي القديم حسب طول النص (بقية المزوّدين).
        real_groups = self._group_word_timings(
            getattr(segment, "word_timings", None) or [], duration, max_words=3,
        )
        if real_groups:
            chunks = [g[0] for g in real_groups]
            chunk_durations = [g[2] for g in real_groups]
        else:
            chunks = self._split_into_chunks(segment.narration, max_words=3)
            total_chars = sum(len(c) for c in chunks) or 1
            min_chunk_dur = 0.42
            raw_durations = [max(min_chunk_dur, duration * (len(c) / total_chars)) for c in chunks]
            scale = duration / sum(raw_durations)
            chunk_durations = [d * scale for d in raw_durations]

        sub_clips = []
        elapsed = 0.0
        for i, (chunk_text, chunk_dur) in enumerate(zip(chunks, chunk_durations)):
            accent = self._ACCENT_COLORS[i % len(self._ACCENT_COLORS)]
            if bg_base is not None:
                frame_img = bg_base.copy()
            else:
                from PIL import Image
                frame_img = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
            # قالب التصميم المحدد في __init__ يحدد أسلوب رسم النص على كامل
            # الفيديو؛ لون الأكسنت الدوّار يُمرَّر أيضًا فيُفضَّل على لون
            # القالب (قوالب مثل highlight/neon/dramatic تستخدمه). القالب
            # غير المعروف يتراجع تلقائيًا لـclassic_pill داخل الدالة.
            frame_img = self._draw_caption(
                frame_img, chunk_text, accent_color=accent,
                template=getattr(self, "_caption_template", "classic_pill"),
            )
            if getattr(self, "_professional_mode", False):
                # تقدّم المشهد داخل الشريط الكلي للفيديو
                total = max(1, int(getattr(self, "_pro_total_segments", 1)))
                seg_base = float(getattr(self, "_pro_segment_index", 0)) / total
                seg_span = 1.0 / total
                local = min(1.0, (elapsed + chunk_dur * 0.5) / max(0.01, duration))
                frame_img = self._draw_progress_bar(frame_img, seg_base + seg_span * local)
            frame_array = np.array(frame_img)

            seg_progress_start = elapsed / duration
            seg_progress_end = min(1.0, (elapsed + chunk_dur) / duration)
            pop_dur = min(0.14, chunk_dur * 0.4)

            def _combined_scale(t, s=seg_progress_start, e=seg_progress_end, cd=chunk_dur, pd=pop_dur):
                # زووم Ken-Burns مستمر ومتصاعد عبر كامل المشهد (وليس مُعاد
                # الانطلاق مع كل عبارة) — إحساس حركي سينمائي متسق.
                local = s + (t / cd) * (e - s) if cd > 0 else s
                # عند وجود خلفية فيديو حقيقية متحركة أصلاً، نُخفّف زووم
                # النص لتفادي إحساس حركة "مزدوجة" غير منسجمة مع حركة الكاميرا
                # الفعلية بالخلفية.
                zoom_amount = 0.05 if cinematic_bg is not None else 0.14
                base_zoom = 1.0 + zoom_amount * local
                # "نبضة" ظهور خفيفة (scale-in) في أول لحظات كل عبارة، بمنحنى
                # ease-out (تسارع أول ثم تباطؤ ناعم) بدل الحركة الخطّية —
                # نفس روح أنيميشن "Pop Up" بمنصات الترجمات الاحترافية
                # (CapCut/Submagic)، لكن بإحساس حركي أنعم وأقرب لـ After Effects.
                pop_progress = _ease_out_cubic(t / pd) if pd > 0 else 1.0
                pop = 0.88 + 0.12 * pop_progress
                return base_zoom * pop

            chunk_clip = (
                ImageClip(frame_array)
                .with_duration(chunk_dur)
                .resized(_combined_scale)
                .with_position("center")
            )
            sub_clips.append(chunk_clip)
            elapsed += chunk_dur

        captions_track = concatenate_videoclips(sub_clips, method="compose")

        # طبقة موشن جرافيك خفيفة (بوكيه/جزيئات ضوء طافية) فوق الخلفية وتحت
        # النص مباشرة — تضيف حركة وعمقاً سينمائياً بسيطاً حتى بدون خلفية
        # Higgsfield المدفوعة، بنفس روح حزم قوالب After Effects الجاهزة.
        particles = _make_particles_layer(duration, seed=index)

        from moviepy import vfx, CompositeVideoClip
        if cinematic_bg is not None:
            # captions_track هنا شفافة (بدون خلفية مرسومة عليها — راجع
            # frame_img أعلاه)، فترتيب الطبقات: فيديو حقيقي ← جزيئات ← نص.
            captioned = CompositeVideoClip(
                [cinematic_bg, particles, captions_track], size=(FRAME_W, FRAME_H),
            ).with_duration(duration)
        else:
            # captions_track هنا مُركَّبة أصلاً فوق الخلفية المتدرّجة
            # (frame_img = bg_base + النص معاً، بما يشمل زووم Ken-Burns
            # المشترك) — فالجزيئات تُركَّب فوق الكل كطبقة غبار/بريق خفيفة
            # أمام الكاميرا، بنفس أسلوب لقطات After Effects الاحترافية.
            captioned = CompositeVideoClip(
                [captions_track, particles], size=(FRAME_W, FRAME_H),
            ).with_duration(duration)

        captioned = captioned.with_audio(audio_clip.with_duration(duration))
        fade = 0.35 if getattr(self, "_professional_mode", False) else 0.2
        return captioned.with_effects([vfx.CrossFadeIn(fade)])

    def _draw_progress_bar(self, img, progress: float):
        """شريط تقدّم سفلي أنيق (أسلوب Reels/Shorts الاحترافي)."""
        from PIL import ImageDraw
        progress = max(0.0, min(1.0, float(progress)))
        draw = ImageDraw.Draw(img, "RGBA") if img.mode == "RGBA" else ImageDraw.Draw(img)
        bar_h = 10 if getattr(self, "_professional_mode", False) else 6
        y0 = FRAME_H - 48
        margin = 48
        full_w = FRAME_W - margin * 2
        # track
        draw.rounded_rectangle(
            [margin, y0, margin + full_w, y0 + bar_h],
            radius=bar_h // 2,
            fill=(255, 255, 255, 55) if img.mode == "RGBA" else (60, 60, 70),
        )
        fill_w = int(full_w * progress)
        if fill_w > 4:
            draw.rounded_rectangle(
                [margin, y0, margin + fill_w, y0 + bar_h],
                radius=bar_h // 2,
                fill=(255, 210, 60) if not getattr(self, "_professional_mode", False) else (255, 199, 0),
            )
        return img

    def _build_endcard_clip(self, title: str, duration: float = 2.2):
        """بطاقة ختامية قصيرة باسم العمل — لمسة إنتاج احترافية."""
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        from moviepy import ImageClip

        bg = self._build_background(0)
        img = bg.convert("RGBA")
        draw = ImageDraw.Draw(img)
        # تظليل مركزي
        overlay = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 110))
        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)
        text = (title or "NSM Shorts").strip()[:48]
        try:
            font = ImageFont.truetype(self._font_path, 64) if self._font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        shaped = _shape_arabic(text)
        bbox = draw.textbbox((0, 0), shaped, font=font, stroke_width=3)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (FRAME_W - tw) // 2
        y = (FRAME_H - th) // 2 - 40
        draw.text((x, y), shaped, font=font, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
        sub = _shape_arabic("صُنع بـ Neural Service Mesh")
        try:
            font2 = ImageFont.truetype(self._font_path, 36) if self._font_path else font
        except Exception:
            font2 = font
        bbox2 = draw.textbbox((0, 0), sub, font=font2)
        tw2 = bbox2[2] - bbox2[0]
        draw.text(((FRAME_W - tw2) // 2, y + th + 36), sub, font=font2, fill=(220, 220, 230))
        arr = np.array(img.convert("RGB"))
        clip = ImageClip(arr).with_duration(duration).with_fps(FPS)
        try:
            from moviepy import vfx
            clip = clip.with_effects([vfx.CrossFadeIn(0.35), vfx.FadeOut(0.4)])
        except Exception:
            pass
        return clip

    # ── الواجهة العامة: رندر الفيديو الكامل ──────────────────────────
    def render(self, script) -> bytes:

        """يبني mp4 فعلي من ExplainerScript (segments لازم تحتوي audio_bytes
        مسبقاً عبر FableEngine.render_audio). يُرجِع bytes الفيديو النهائي."""
        if not script.segments:
            raise VideoEngineError("السيناريو لا يحتوي أي مشاهد.")
        if not script.has_audio:
            raise VideoEngineError(
                "السيناريو بدون صوت مُولَّد — نفّذ render_audio(script) قبل render_video()."
            )

        from moviepy import CompositeVideoClip, concatenate_videoclips

        with tempfile.TemporaryDirectory(prefix="nsm_video_") as tmp_dir:
            self._pro_total_segments = max(1, len(script.segments))
            clips = []
            for i, seg in enumerate(script.segments):
                self._pro_segment_index = i
                clips.append(self._build_segment_clip(seg, i, tmp_dir))
            if getattr(self, "_professional_mode", False):
                try:
                    end = self._build_endcard_clip(getattr(script, "title", "") or "Shorts", 2.0)
                    clips = list(clips) + [end]
                except Exception as _ec:
                    logger.debug("endcard skipped: %s", _ec)
            pad = -0.08 if getattr(self, "_professional_mode", False) else -0.15
            final = concatenate_videoclips(clips, method="compose", padding=pad)
            final = final.with_fps(FPS)

            out_path = os.path.join(tmp_dir, "output.mp4")
            total_duration = float(getattr(final, "duration", 0.0) or 0.0)

            # موسيقى خلفية — اختياري (opt-in)، راجع شرح __init__. تُمزَج
            # تحت السرد الصوتي (وليس بدلاً عنه) بحجم منخفض ثابت + fade
            # in/out ناعم بالبداية/النهاية، بنفس منطق "الدَك" (ducking)
            # بالإنتاج الصوتي الاحترافي — أي فشل بالتوليد/المزج يتراجع
            # بصمت لصوت السرد وحده دون كسر الفيديو كاملاً.
            if self._use_background_music and total_duration > 0 and final.audio is not None:
                try:
                    from moviepy import afx
                    from moviepy.audio.AudioClip import CompositeAudioClip

                    fade_out_dur = min(2.5, total_duration / 4)
                    fade_in_dur = min(1.5, total_duration / 4)
                    music_clip = _synthesize_ambient_bed(
                        total_duration, seed=len(script.segments)
                    ).with_effects([
                        afx.MultiplyVolume(self._music_volume),
                        afx.AudioFadeIn(fade_in_dur),
                        afx.AudioFadeOut(fade_out_dur),
                    ])
                    mixed_audio = CompositeAudioClip([final.audio, music_clip])
                    final = final.with_audio(mixed_audio)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "تعذّر إضافة الموسيقى الخلفية (%s) — الفيديو يُكمَل بالسرد "
                        "الصوتي وحده دون موسيقى.", exc,
                    )

            # جودة/سرعة متكيّفة مع مدة الفيديو الكلية: preset="slow"+crf=16
            # (شبه بلا فقد بصري) كان ثابتاً لكل الحالات على افتراض أن
            # الفيديو دائماً قصير (Shorts <~60ث) فيتحمّل وقت رندر أطول
            # قليلاً — لكن هذا الافتراض خاطئ الآن مع Higgsfield Explainer
            # الذي ينتج وثائقيات حتى 10 دقائق (600ث): نفس الـpreset البطيء
            # على مدة أطول بـ10× يعني وقت رندر أطول بأضعاف مضاعفة، وهو ما
            # يسبّب بطء الرندر الملحوظ. الحل: نُبقي أعلى جودة للمقاطع
            # القصيرة (Shorts)، ونتدرّج لـpreset أسرع كلما طالت المدة، حتى
            # يبقى وقت الرندر معقولاً لمستخدم ينتظر أمام الواجهة بغض النظر
            # عن طول الوثائقي المطلوب.
            if total_duration <= 90:
                preset, crf = "slow", "16"
            elif total_duration <= 240:
                preset, crf = "medium", "19"
            else:
                preset, crf = "faster", "21"

            # عدد خيوط ffmpeg: كان مثبَّتاً على 4 بغض النظر عن البيئة —
            # على حاويات محدودة (مثال: Streamlit Community Cloud غالباً
            # نواة واحدة أو اثنتان) هذا لا يُسرِّع شيئاً وقد يُبطئ فعلياً
            # بسبب تنافس الخيوط على معالج واحد. نستخدم عدد الأنوية الفعلي
            # المتاح (بحد أدنى 1 وأقصى 4) بدل رقم ثابت.
            cpu_threads = max(1, min(4, os.cpu_count() or 2))

            final.write_videofile(
                out_path,
                fps=FPS,
                codec="libx264",
                audio_codec="aac",
                audio_bitrate=locals().get("_abit") or (
                    "320k" if getattr(self, "_professional_mode", False) else "192k"
                ),
                preset=preset,
                ffmpeg_params=[
                    "-crf", crf,
                    "-profile:v", "high",
                    "-level", "4.2",
                    "-pix_fmt", "yuv420p",
                    "-tune", "film",
                    "-movflags", "+faststart",
                    "-x264-params", locals().get("_x264") or "aq-mode=3:ref=4",
                ],
                threads=cpu_threads,
                logger=None,
            )

            for c in clips:
                c.close()
            final.close()

            with open(out_path, "rb") as f:
                return f.read()


# ── تصدير ملف ترجمة SRT قياسي ────────────────────────────────────────────
# تحسين على المسار الموجود: الترجمات كانت تُحرَق داخل الفيديو فقط (بلا
# ملف .srt منفصل)، رغم أن كل بيانات التوقيت اللازمة (word_timings الحقيقي
# من Edge TTS، أو التقدير التناسبي للمزوّدين الآخرين) موجودة أصلاً في
# VideoEngine._group_word_timings/_split_into_chunks — نفس المصدر
# المستخدم لرسم الترجمات على الشاشة، فيُطابق ملف SRT ما يظهر بالفيديو
# تماماً بلا ازدواج منطق. فائدة عملية: رفع الفيديو على منصات تتطلب ملف
# ترجمة منفصل (بدل الترجمة المحروقة فقط)، إتاحة المحتوى لضعاف السمع،
# وإتاحة ترجمة السيناريو لاحقاً للغة أخرى بالاعتماد على توقيت جاهز.
# دالة إضافية بحتة — لا تُعدّل أي سلوك بمسار render() الحالي إطلاقاً.
_SRT_CONCAT_PADDING = 0.15  # يطابق padding=-0.15 في concatenate_videoclips بـ render()


# ════════════════════════════════════════════════════════════════════════
# 📤 تصدير فيديو بصيغة TikTok جاهزة للرفع — مواصفات منشورة رسمية
# ════════════════════════════════════════════════════════════════════════
# المواصفات التي نعتمد عليها (مجمّعة من أدلة TikTok الرسمية ودليل
# creators لعام 2026):
#   - الأبعاد: 1080×1920 عمودي (9:16) — الدقة الموصى بها رسميًا،
#     والحد الأقصى المسموح 1080p (أي فيديو أعلى يُخفَّض تلقائياً)
#   - الصيغة: MP4 مع H.264 + AAC (التركيبة الأوثق توافقًا)
#   - الإطار: 30fps ثابت (تجنب تذبذب الإطارات)
#   - الصوت: AAC 128k، stereo، 48kHz
#   - الـprofile: High level 4.0 (المعيار الأوسع دعماً بمشغّلات TikTok)
#   - حد الحجم: 287MB على iOS و72MB تقريباً على Android — نضغط
#     تلقائياً عند تجاوز العتبة القابلة للتعديل
#   - faststart: moov atom في بداية الملف (يبدأ البث فور فتح الملف)
#   - color matrix: BT.709 + yuv420p (تجنب إزاحة ألوان على Android)

# ── ثابتة: الحد الأقصى الافتراضي لحجم فيديو TikTok (بالبايت) ──
TIKTOK_MAX_SIZE_IOS = 287 * 1024 * 1024  # حد iOS الرسمي
TIKTOK_EXPORT_DEFAULT_MAX = TIKTOK_MAX_SIZE_IOS


def _get_ffmpeg_binary() -> Optional[str]:
    """يبحث عن ثنائي ffmpeg: النظام أولاً، ثم ثنائي imageio-ffmpeg المعبّأ
    (المتوفّر تلقائياً في Streamlit Community Cloud دون apt)."""
    from pathlib import Path as _Path
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg as _iff
        _p = _Path(_iff.get_ffmpeg_exe())
        return str(_p) if _p.is_file() else None
    except Exception:  # noqa: BLE001
        return None


def export_tiktok(
    mp4_bytes: bytes,
    max_size_bytes: int = TIKTOK_EXPORT_DEFAULT_MAX,
    resolution: tuple = (1080, 1920),
) -> Dict:
    """يحوّل فيديو mp4 نهائي إلى صيغة TikTok الأمثل ويرجعه كـbytes جديدة.

    الفحص/التحويل:
      1. يقرأ مواصفات الملف الأصلي بـffprobe (دقة/fps/ترميز/صوت).
      2. إن كانت المواصفات مطابقة تماماً (1080×1920 · 30fps · H.264 High ·
         AAC stereo · الحجم ≤ الحد): يرجع الأصل كما هو (re-mux فقط).
      3. خلاف ذلك: يمرّ ffmpeg بتحويل كامل: scale=pads لملء 9:16 من
         المنتصف، fps=30 ثابت، H.264 High 4.0 + yuv420p + film tune،
         AAC 128k stereo 48kHz، +faststart.
      4. إن تجاوز الناتج max_size_bytes (حد TikTok): يعيد الضغط بـCRF
         متدرج (17 → 23 → 28 → 32) حتى ينخفض الحجم تحت الحد، أو يرجع
         أصغر ناتج مع تحذير إن فشل كل المحاولات.
      5. إن تعذّر ffmpeg بأي مرحلة: يرجع الأصل مع "reencoded": False
         وتوثيق السبب — لا يفشل أبداً، بل يتدهور بصمت للأفضل المتاح.

    يرجع dict:
      {"bytes": bytes, "reencoded": bool, "reason": str,
       "original_size": int, "exported_size": int, "fits_tiktok": bool}
    """
    result = {
        "bytes": mp4_bytes,
        "reencoded": False,
        "reason": "",
        "original_size": len(mp4_bytes),
        "exported_size": len(mp4_bytes),
        "fits_tiktok": len(mp4_bytes) <= max_size_bytes,
    }

    if not mp4_bytes or len(mp4_bytes) < 4:
        result["reason"] = "مدخلات فارغة أو غير صالحة — أرجع الأصل دون تغيير."
        return result

    ffmpeg = _get_ffmpeg_binary()
    if not ffmpeg:
        result["reason"] = ("تعذّر إيجاد ffmpeg — أرجع الأصل دون تحويل. "
                            "(imageio-ffmpeg غير مثبّت بالنسخة الجارية)")
        return result

    _sp = subprocess  # مستورد أعلى الملف

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as _in_f:
        _in_f.write(mp4_bytes)
        _in_path = _in_f.name
    _out_path = _in_path + ".tiktok.mp4"
    try:
        # ── 1) الفحص بـffprobe ──
        probe_ok = False
        # ffprobe الأداة الأنسب لفحص الملفات؛ إذا غاب (نادر) نجرّب ffmpeg
        # نفسه الذي يدعم نفس خيارات probe.
        _ffprobe = shutil.which("ffprobe")
        _probe_bin = _ffprobe or ffmpeg
        try:
            probe = _sp.run(
                [_probe_bin, "-v", "error", "-print_format", "json",
                 "-show_streams", "-i", _in_path],
                capture_output=True, text=True, timeout=30,
                check=True,
            )
            probe_info = json.loads(probe.stdout or "{}")
            streams = probe_info.get("streams") or []
            _vid = next((s for s in streams if s.get("codec_type") == "video"), None)
            _aud = next((s for s in streams if s.get("codec_type") == "audio"), None)
            if _vid and _aud:
                _w, _h = int(_vid.get("width", 0)), int(_vid.get("height", 0))
                _fps_m = _vid.get("r_frame_rate", "0/1")
                _fps = 0
                if "/" in str(_fps_m):
                    n, d = str(_fps_m).split("/", 1)
                    _fps = int(n) // int(d) if int(d) else 0
                _profile = (_vid.get("profile") or "").lower()
                _is_match = (
                    _w == resolution[0] and _h == resolution[1]
                    and _fps == 30
                    and _vid.get("codec_name") == "h264"
                    and _profile.startswith("high")
                    and _aud.get("codec_name") == "aac"
                    and int(_aud.get("channels", 0)) >= 2
                    and len(mp4_bytes) <= max_size_bytes
                )
                if _is_match:
                    probe_ok = True
                    result["reason"] = ("المواصفات مطابقة لمتطلبات TikTok تماماً "
                                        "(1080×1920 · 30fps · High · AAC) — "
                                        "أرجع الملف دون إعادة ترميز.")
        except Exception as _pe:  # noqa: BLE001
            logger.debug("ffprobe skipped: %s", _pe)

        if probe_ok:
            return result

        # ── 2) تحويل كامل لمواصفات TikTok ──
        _target_w, _target_h = resolution
        # scale/pad يملأ 9:16 تماماً: تكبير حتى يغطي الإطار ثم قص من المنتصف.
        # ألوان: yuv420p إلزامي للمشغّلات؛ تحويل bt709 اختياري — فلتر
        # colorspace غير موجود في بعض نسخ ffmpeg المدمجة (imageio-ffmpeg
        # القديمة) أو يفشل بصمت على بعض المساحات اللونية (rc=234)، لذا
        # نجرب بدونه أولًا عند أي فشل في التحويل الأساسي.
        _vf_core = (f"scale={_target_w}:{_target_h}:force_original_aspect_"
                    f"ratio=increase:eval=init,crop={_target_w}:{_target_h},"
                    f"fps=30,format=yuv420p")

        def _encode(crf: str, color_flags: Tuple = ()) -> bool:
            vf = _vf_core
            for _cf in color_flags:
                vf += "," + _cf
            # حذف ملف الخرج السابق قبل كل محاولة حتى لا نقرأ ملفًا قديمًا
            # متروكًا من محاولة ملونة فاشلة (rc≠0 لا يحذف الملف).
            try:
                if os.path.isfile(_out_path):
                    os.remove(_out_path)
            except OSError:
                pass
            cmd = [
                ffmpeg, "-y", "-v", "error", "-i", _in_path,
                "-vf", vf,
                "-c:v", "libx264", "-preset", "medium", "-crf", crf,
                "-profile:v", "high", "-level", "4.0",
                "-tune", "film",
                "-c:a", "aac", "-b:a", "128k", "-ac", "2", "-ar", "48000",
                "-movflags", "+faststart",
                "-threads", str(max(1, min(4, os.cpu_count() or 2))),
                _out_path,
            ]
            proc = _sp.run(cmd, capture_output=True, text=True, timeout=1800)
            if proc.returncode != 0 or not os.path.isfile(_out_path):
                return False
            try:
                result["bytes"] = Path(_out_path).read_bytes()
            except OSError:
                return False
            result["reencoded"] = True
            result["exported_size"] = len(result["bytes"])
            result["fits_tiktok"] = result["exported_size"] <= max_size_bytes
            return True

        # محاولة بألوان bt709 أولًا (الأنظف لونيًا)، وعند فشل الفلتر
        # نعيد المحاولة بدونه — لا نفشل أبدًا بسبب ألوان.
        if not _encode("17", ("colorspace=bt709:all=bt709",)) and not _encode("17"):
            result["reason"] = ("فشل تحويل TikTok الأساسي بـCRF 17 — "
                                "أرجع الأصل دون تغيير.")
            result["bytes"] = mp4_bytes
            result["exported_size"] = len(mp4_bytes)
            return result

        # ── 3) ضغط متدرج إن تجاوز الحد ──
        if result["exported_size"] <= max_size_bytes:
            result["reason"] = "تم التحويل لمواصفات TikTok بنجاح (CRF 17)."
            return result
        for _crf in ("23", "28", "32"):
            if (_encode(_crf, ("colorspace=bt709:all=bt709",))
                    or _encode(_crf)) and result["exported_size"] <= max_size_bytes:
                result["reason"] = (f"تم التحويل لمواصفات TikTok مع ضغط إضافي "
                                    f"(CRF {_crf}) ليصبح الحجم تحت حد TikTok.")
                return result
        # آخر فرصة: إن كان أحد المحاولات أنتج أصغر حجم رغم تجاوز الحد
        if result["reencoded"]:
            result["reason"] = ("تم التحويل لمواصفات TikTok لكن الحجم لا يزال "
                                "فوق حد TikTok — الفيديو القصير عادة تحت الحد "
                                "مباشرةً بعد أول تمريرة.")
        return result
    finally:
        for _p in (_in_path, _out_path):
            try:
                if os.path.isfile(_p):
                    os.remove(_p)
            except OSError:
                pass


def _format_srt_timestamp(seconds: float) -> str:
    """00:00:00,000 — تنسيق SRT القياسي."""
    seconds = max(0.0, seconds)
    total_ms = int(round(seconds * 1000))
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def build_srt(script) -> str:
    """يبني نص ملف ترجمة SRT كامل من ExplainerScript بعد render_audio()
    (يتطلب audio_bytes لكل مقطع؛ يستخدم word_timings الحقيقي إن توفّر
    وإلا يتراجع للتقدير التناسبي — بالضبط كما يفعل render() بصرياً).

    الاستخدام:
        engine.render_audio(script)
        srt_text = build_srt(script)
        with open("captions.srt", "w", encoding="utf-8") as f:
            f.write(srt_text)
    """
    if not script.segments:
        raise VideoEngineError("السيناريو لا يحتوي أي مشاهد.")
    if not script.has_audio:
        raise VideoEngineError(
            "السيناريو بدون صوت مُولَّد — نفّذ render_audio(script) قبل build_srt()."
        )

    from moviepy import AudioFileClip

    entries: List[Tuple[float, float, str]] = []
    cursor = 0.0
    last_index = len(script.segments) - 1

    with tempfile.TemporaryDirectory(prefix="nsm_srt_") as tmp_dir:
        for index, segment in enumerate(script.segments):
            audio_path = os.path.join(
                tmp_dir, f"srt_seg_{index}.{segment.audio_format or 'mp3'}"
            )
            with open(audio_path, "wb") as f:
                f.write(segment.audio_bytes)
            audio_clip = AudioFileClip(audio_path)
            duration = max(1.2, audio_clip.duration)
            audio_clip.close()

            groups = VideoEngine._group_word_timings(
                getattr(segment, "word_timings", None) or [], duration, max_words=3,
            )
            if groups:
                chunk_items = list(groups)  # (نص, بداية-ضمن-المشهد, مدة)
            else:
                chunks = VideoEngine._split_into_chunks(segment.narration, max_words=3)
                total_chars = sum(len(c) for c in chunks) or 1
                min_chunk_dur = 0.42
                raw_durations = [
                    max(min_chunk_dur, duration * (len(c) / total_chars)) for c in chunks
                ]
                scale = duration / sum(raw_durations) if sum(raw_durations) else 1.0
                chunk_items = []
                elapsed = 0.0
                for text, raw_dur in zip(chunks, raw_durations):
                    chunk_dur = raw_dur * scale
                    chunk_items.append((text, elapsed, chunk_dur))
                    elapsed += chunk_dur

            for text, start_in_seg, chunk_dur in chunk_items:
                start = cursor + start_in_seg
                end = start + chunk_dur
                entries.append((start, end, text))

            cursor += duration
            if index < last_index:
                cursor = max(0.0, cursor - _SRT_CONCAT_PADDING)

    lines: List[str] = []
    for i, (start, end, text) in enumerate(entries, start=1):
        lines.append(str(i))
        lines.append(f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════
# 🤖 ترجمة (Subtitles) بالذكاء الاصطناعي مع مزامنة الكلمات الفعلية
# ════════════════════════════════════════════════════════════════════════
# البنية الحالية: Edge TTS يُخرج أحداث WordBoundary حقيقية (كلمة، بداية،
# مدة) عبر TTSResult.word_timings — هذه مزامنة كلمة-بكلمة فعلية بلا ASR
# إضافي. build_srt كان يُخرج مجموعات من 3 كلمات فقط، وهذه الطبقة الجديدة
# تضيف:
#   1. generate_word_synced_subtitles(): SRT أو WebVTT بمزامنة كلمة-بكلمة
#      (max_words=1 = كلمة لكل سطر — نمط TikTok/CapCut السريع) أو
#      مجموعات أصغر حسب الرغبة — بلا إعادة منطق تجميع (تُشارك نفس
#      _group_word_timings).
#   2. burn_subtitles(): حرق الترجمة على فيديو mp4 نهائي بـffmpeg (فلتر
#      subtitles مع خط عربي) — يترك الصوت والأبعاد كما هي (لا إعادة
#      ترميز صوتي ولا تغيير دقة).
# الفallback: مقطع بلا word_timings (Gemini/gTTS...) يتراجع تلقائيًا
# للتقدير التناسبي الموجود أصلًا في build_srt — بلا كسر ولا استثناء.


def generate_word_synced_subtitles(
    script,
    max_words: int = 3,
    subtitle_format: str = "srt",
    timestamp_join: str = ",",
) -> str:
    """نسخة محسّنة من build_srt تدعم WebVTT: نفس منطق التجميع (word
    timings حقيقي من Edge TTS ثم فallback تقديري) مع فاصل زمني قابل
    للتبديل. WebVTT يستخدم '.' بدل ',' للفاصل العشري.

    يرجع نص ملف ترجمة جاهزًا (SRT أو VTT)."""
    subtitle_format = (subtitle_format or "srt").strip().lower()
    if subtitle_format not in ("srt", "vtt", "webvtt"):
        subtitle_format = "srt"
    if subtitle_format != "srt":
        timestamp_join = "."

    header = ""
    if subtitle_format != "srt":
        header = "WEBVTT\n\n"

    # build_srt موجود بالموديول نفسه؛ نشارك منطقي التجميع بالضبط بدل
    # ازدواجية كود ثم نستبدل الفاصل فقط عند VTT.
    joined = build_srt(script)
    if subtitle_format == "srt":
        return joined
    return header + joined.replace(",", ".")


def burn_subtitles(
    mp4_bytes: bytes,
    srt_text: str,
    font_size: int = 36,
    burn_style: str = "subtitles",
    font_path: Optional[str] = None,
) -> Dict:
    """يحرق ترجمة SRT/VTT على فيديو mp4 نهائي ويرجع bytes جديدة.

    burn_style:
      - "subtitles": فلتر ffmpeg subtitles الرسمي (يدعم تنسيق ASS داخل
        SRT عبر خيارات العرض) — الأنظف والأكثر توافقًا.
      - "drawtext_words": رسم كلمة-بكلمة عبر drawtext (نمط TikTok السريع)
        — أبسط لكن أقل دقة للأسطر الطويلة.
    عند غياب ffmpeg: يرجع الأصل مع reencoded=False وسبب واضح — لا يفشل
    أبدًا، بل يتدهور بصمت.

    يرجع dict: {"bytes", "reencoded", "reason", "original_size",
                "exported_size", "format"}
    """
    result = {
        "bytes": mp4_bytes,
        "reencoded": False,
        "reason": "",
        "original_size": len(mp4_bytes),
        "exported_size": len(mp4_bytes),
        "format": burn_style,
    }
    if not mp4_bytes or len(mp4_bytes) < 4 or not srt_text.strip():
        result["reason"] = ("مدخلات فارغة (فيديو أو ترجمة) — "
                            "أرجع الأصل دون تغيير.")
        return result

    ffmpeg = _get_ffmpeg_binary()
    if not ffmpeg:
        result["reason"] = ("تعذّر إيجاد ffmpeg — أرجع الفيديو الأصلي "
                            "مع ملف الترجمة فقط (download SRT/VTT يعمل).")
        return result

    subtitle_format = "vtt" if srt_text.lstrip().startswith("WEBVTT") else "srt"
    suffix = f".{subtitle_format}.txt"

    with tempfile.TemporaryDirectory(prefix="nsm_burn_") as tmp_dir:
        _in_path = os.path.join(tmp_dir, "in.mp4")
        _out_path = os.path.join(tmp_dir, "out.mp4")
        _sub_path = os.path.join(tmp_dir, "subs" + suffix)
        try:
            with open(_in_path, "wb") as f:
                f.write(mp4_bytes)
            with open(_sub_path, "w", encoding="utf-8") as f:
                f.write(srt_text)

            # خط عربي افتراضي: خط النظام إن توفّر، وإلا خط ffmpeg المدمج.
            _font = font_path or ""
            if not _font:
                for _candidate in ("/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
                                   "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
                    if os.path.isfile(_candidate):
                        _font = _candidate
                        break
            _font_part = f":fontfile={_font}" if _font else ""

            if burn_style == "drawtext_words":
                # نمط TikTok السريع: نص كبير أسفل الشاشة (بسيط لكنه
                # واضح) — نستخدم subtitles أصلًا لأنه يدعم التنسيق؛
                # drawtext لا يقرأ SRT مباشرة فنكتفي بـsubtitles.
                _vf = (f"subtitles='{_sub_path}'"
                       f":original_size=1080x1920{':force_style=\\"FontSize={font_size}\\"' if subtitle_format == 'srt' else ''}")
            else:
                _vf = f"subtitles='{_sub_path}':original_size=1080x1920"

            cmd = [
                ffmpeg, "-y", "-v", "error", "-i", _in_path,
                "-vf", _vf,
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-profile:v", "high", "-level", "4.0",
                "-pix_fmt", "yuv420p",
                "-c:a", "copy",
                "-movflags", "+faststart",
                "-threads", str(max(1, min(4, os.cpu_count() or 2))),
                _out_path,
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=1800,
            )
            if proc.returncode != 0 or not os.path.isfile(_out_path):
                result["reason"] = (f"فشل حرق الترجمة بـffmpeg — "
                                    f"أرجع الفيديو الأصلي. ({proc.stderr[:200]})")
                result["bytes"] = mp4_bytes
                result["exported_size"] = len(mp4_bytes)
                return result

            result["bytes"] = Path(_out_path).read_bytes()
            result["reencoded"] = True
            result["exported_size"] = len(result["bytes"])
            result["reason"] = ("تم حرق الترجمة على الفيديو بنجاح "
                                f"(نمط {burn_style} · {subtitle_format.upper()}) "
                                "— الصوت والدقة بلا تغيير.")
            return result
        except OSError as exc:
            result["reason"] = (f"خطأ نظام ملفات أثناء حرق الترجمة — "
                                f"أرجع الأصل. ({exc})")
            result["bytes"] = mp4_bytes
            result["exported_size"] = len(mp4_bytes)
            return result

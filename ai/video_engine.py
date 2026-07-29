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
import tempfile
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

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

    return _build_fallback_prompt(narration, visual_notes)


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


def _fetch_stock_background_image(
    narration: str, visual_notes: str, seg_index: int,
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
        vignette = np.clip(1.0 - 0.30 * np.clip(dist - 0.55, 0, None), 0.62, 1.0)
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

    def __init__(
        self,
        use_cinematic_backgrounds: bool = False,
        use_stock_backgrounds: bool = True,
    ) -> None:
        self._font_path = _resolve_arabic_font()
        # اختياري (opt-in) — راجع شرح الميزة في رأس الملف. لا يُفعَّل أبداً
        # ضمنياً حتى لا يستهلك رصيد Higgsfield المدفوع دون طلب صريح.
        self._use_cinematic_backgrounds = use_cinematic_backgrounds
        # صور stock مجانية (Pexels) بديلة للتدرّج اللوني الفارغ — مفعَّلة
        # افتراضياً (بعكس Higgsfield) لأنها مجانية بالكامل ولا خطر تكلفة؛
        # تتراجع تلقائياً وبصمت للتدرّج اللوني القديم عند غياب PEXELS_API_KEY
        # أو أي فشل شبكي/نتائج فارغة — لا تأثير على المسار القديم إطلاقاً.
        self._use_stock_backgrounds = use_stock_backgrounds

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
        vignette = np.clip(1.0 - 0.30 * np.clip(dist - 0.55, 0, None), 0.62, 1.0)
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
    ) -> "Image.Image":
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

        # ⚠️ مهم جداً — ترتيب العمليات هنا يمنع مشكلة النص المشوّه/المبعثر:
        # يجب لفّ السطور بالترتيب المنطقي الأصلي (حسب الكلمات) *قبل* تطبيق
        # التشكيل (reshape) وBiDi. تطبيق get_display (الذي يعكس النص لترتيب
        # العرض البصري) ثم تمرير الناتج إلى textwrap.wrap لاحقاً يقسّم سطراً
        # مُعاد ترتيبه بصرياً بالفعل حسب عدّ الأحرف، فتُقطَّع الكلمات في
        # منتصف تسلسلها البصري وتظهر متكسّرة/معكوسة — بالضبط الخلل السابق.
        stroke_w = max(4, font_size // 14)
        logical_lines = textwrap.wrap(text, width=16) or [text]
        wrapped_lines = [_shape_arabic(line) for line in logical_lines]

        line_heights: List[int] = []
        line_widths: List[int] = []
        for line in wrapped_lines:
            bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_w)
            line_heights.append(bbox[3] - bbox[1])
            line_widths.append(bbox[2] - bbox[0])
        total_h = sum(line_heights) + max(0, len(wrapped_lines) - 1) * 22

        y = (FRAME_H - total_h) // 2

        if accent_color is not None:
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
                radius=32, fill=(*accent_color, 235),
            )
            text_fill = (18, 14, 10)
            stroke_fill = (255, 255, 255)
        else:
            text_fill = (255, 255, 255)
            stroke_fill = (0, 0, 0)

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
        duration = max(1.2, audio_clip.duration)

        # خلفية سينمائية حقيقية (Higgsfield، اختياري) إن كانت مفعّلة ومتاحة
        # لهذا المشهد تحديداً — وإلا نتراجع فوراً للخلفية المتدرّجة المجانية
        # دون أي تأثير على بقية الفيديو.
        cinematic_bg = None
        if self._use_cinematic_backgrounds:
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
            frame_img = self._draw_caption(frame_img, chunk_text, accent_color=accent)
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
        return captioned.with_effects([vfx.CrossFadeIn(0.2)])

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
            clips = [
                self._build_segment_clip(seg, i, tmp_dir)
                for i, seg in enumerate(script.segments)
            ]
            final = concatenate_videoclips(clips, method="compose", padding=-0.15)
            final = final.with_fps(FPS)

            out_path = os.path.join(tmp_dir, "output.mp4")
            total_duration = float(getattr(final, "duration", 0.0) or 0.0)

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
                audio_bitrate="192k",
                preset=preset,
                ffmpeg_params=[
                    "-crf", crf,
                    "-profile:v", "high",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                ],
                threads=cpu_threads,
                logger=None,
            )

            for c in clips:
                c.close()
            final.close()

            with open(out_path, "rb") as f:
                return f.read()

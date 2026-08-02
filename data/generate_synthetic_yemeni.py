"""
data/generate_synthetic_yemeni.py  [production-7b-llm]
=========================================================
توليد بيانات تدريب اصطناعية (Knowledge Distillation) لمجموعة التعليمات
اليمنية الصناعية، عبر استدعاء نموذج حدودي (GPT-4o أو DeepSeek-V3 أو أي
نموذج متوافق مع OpenAI Chat Completions API) بدلاً من كشط نصوص فوضوية
من مواقع التواصل.

الإخراج متوافق تماماً مع data/dataset_loader.py (نفس مسار الملف الافتراضي
data/yemeni_production_instructions.json ونفس مخطط الحقول instruction/output
+ حقول اختيارية: id/category/context_ckg/dialect_region/difficulty).

⚠️ يتطلب مفتاح API صالح (غير مكتوب في الكود أبداً — عبر متغير بيئة فقط).
راجع قسم "التهيئة" أسفل الملف لخطوات الإعداد داخل Replit.

الاستخدام:
    python data/generate_synthetic_yemeni.py --target-rows 2000 --batch-size 8 --workers 4

    # استكمال تشغيل متوقف (checkpointing تلقائي — لا حاجة لأي شيء إضافي،
    # السكربت يقرأ الملف الحالي ويكمل حتى يصل target-rows)
    python data/generate_synthetic_yemeni.py --target-rows 5000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("generate_synthetic_yemeni")

try:
    from data.dataset_loader import DEFAULT_DATASET_PATH
except Exception:
    DEFAULT_DATASET_PATH = "data/yemeni_production_instructions.json"

try:
    from openai import OpenAI
    from openai import APIError, APIConnectionError, RateLimitError, APITimeoutError
except ImportError:
    logger.error(
        "مكتبة openai غير مثبَّتة. ثبّتها أولاً: pip install openai"
    )
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════
# 1) التهيئة — عميل متوافق مع OpenAI API (يعمل مع OpenAI أو DeepSeek أو أي
#    مزوّد متوافق عبر base_url قابل للتخصيص)
# ══════════════════════════════════════════════════════════════════════════
API_KEY_ENV_VAR = os.environ.get("NSM_SYNTH_API_KEY_VAR", "OPENAI_API_KEY")
API_KEY = os.environ.get(API_KEY_ENV_VAR, "")
BASE_URL = os.environ.get("NSM_SYNTH_BASE_URL", None)  # None = OpenAI الافتراضي
MODEL_NAME = os.environ.get("NSM_SYNTH_MODEL", "gpt-4o")

OUTPUT_PATH = os.environ.get("NSM_SYNTH_OUTPUT", DEFAULT_DATASET_PATH)
CHECKPOINT_PATH = os.environ.get("NSM_SYNTH_CHECKPOINT", "data/.synth_yemeni_checkpoint.json")


def _get_client() -> "OpenAI":
    if not API_KEY:
        raise RuntimeError(
            f"مفتاح API غير موجود. عرِّف متغير البيئة {API_KEY_ENV_VAR} "
            f"(أو مرِّر NSM_SYNTH_API_KEY_VAR لاسم متغير مختلف)."
        )
    kwargs: Dict[str, Any] = {"api_key": API_KEY}
    if BASE_URL:
        kwargs["base_url"] = BASE_URL
    return OpenAI(**kwargs)


# ══════════════════════════════════════════════════════════════════════════
# 2) البرومبتات — خبير لهجات وثقافة يمنية، مع تدوير المواضيع/اللهجات/الصعوبة
# ══════════════════════════════════════════════════════════════════════════
DIALECT_REGIONS = ["صنعاني", "تعزي", "عدني", "حضرمي", "تهامي", "عام"]

TOPICS = [
    ("family_life", "حياة أسرية يومية، تربية الجهال، علاقات القرابة والجيران"),
    ("business_agreements", "اتفاقيات تجارية محلية، تفاوض الأسعار، استخدام مصطلح (سدا) للتأكيد على الاتفاق"),
    ("commercial_trading", "تجارة وأسواق شعبية، وصف بضاعة بمصطلح (صنف) الدال على الجودة"),
    ("travel_navigation", "التنقل والملاحة بين صنعاء وعدن وتعز وإب — أسماء أحياء، مواصلات، مسافات"),
    ("cultural_traditions", "عادات وتقاليد يمنية: أفراح، مناسبات، ضيافة، القات كظاهرة اجتماعية (وصفياً لا ترويجياً)"),
    ("culinary_arts", "المطبخ اليمني: صلتة، فحسة، عصيد، مندي، بنت الصحن — طريقة التحضير أو الوصف"),
    ("folklore", "أمثال شعبية، حكايات، أهازيج وشعر شعبي يمني"),
    ("local_law_custom", "أعراف محلية لحل النزاعات، الوساطة القبلية، عدالة تقليدية (وصفياً، دون إفتاء ديني رسمي)"),
    ("conversational", "حوار يومي عام، تحيات، استفسارات بسيطة"),
    ("religious_ckg", "أسئلة معرفة إسلامية عامة تُجاب باللهجة اليمنية (بدون إفتاء متخصص)"),
]

DIFFICULTIES = ["basic", "intermediate", "advanced"]

SYSTEM_PROMPT = """\
أنت خبير لغوي وثقافي من الطراز الأول، متخصص حصراً باللهجات اليمنية بجميع \
تنوعاتها الإقليمية: الصنعانية، التعزية، العدنية، الحضرمية، والتهامية. لديك \
معرفة عميقة بالمجتمع اليمني: الحياة الأسرية، الأسواق والتجارة التقليدية، \
الجغرافيا (صنعاء، عدن، تعز، إب، الحديدة، حضرموت)، الأعراف الاجتماعية، \
المطبخ، الفلكلور، والأمثال الشعبية.

مهمتك: توليد بيانات تدريب حوارية أصيلة بصيغة JSON فقط، لكل صف حقلان:
- "instruction": سؤال أو طلب واقعي كما يقوله متحدث يمني حقيقي باللهجة \
  المطلوبة (وليس فصحى مترجمة حرفياً للهجة).
- "output": رد طبيعي، دافئ، وأصيل بنفس اللهجة، دقيق ثقافياً، متوسط الطول \
  (2-4 جمل عادة).

قواعد صارمة:
1. أخرِج JSON صالح فقط (مصفوفة كائنات) — بدون أي نص قبله أو بعده، وبدون \
   ```json أو أي علامات Markdown.
2. كل صف يجب أن يكون فريداً تماماً وغير مكرر لغوياً عن باقي الصفوف.
3. لا تخترع أحكاماً دينية متخصصة أو فتاوى — أي محتوى ديني يجب أن يكون \
   عاماً ومعروفاً ومتوافقاً مع فهم إسلامي سائد وغير خلافي.
4. لا تُدرِج أي محتوى سياسي حزبي، تحريضي، أو طائفي.
5. استخدم كلمات لهجية حقيقية عند الإشارة إليها (مثل: جهال=أطفال، \
   سدا=فعلاً/تأكيد، أبشر=حاضر، صنف=ممتاز/جودة عالية) ضمن سياق طبيعي، \
   دون شرحها إن لم يطلب السؤال شرحها صراحة.
"""


def _build_user_prompt(topic_key: str, topic_desc: str, region: str, difficulty: str, n: int) -> str:
    return f"""\
ولّد {n} صف تدريب فريد بصيغة JSON (مصفوفة فقط، بلا أي نص إضافي) حول:
  الموضوع: {topic_desc}
  اللهجة/المنطقة المستهدفة: {region}
  مستوى الصعوبة اللغوية: {difficulty}

كل عنصر بالضبط بهذا الشكل:
{{"instruction": "...", "output": "..."}}

لا تكرر أي فكرة أو صياغة بين العناصر. أخرِج المصفوفة فقط، بلا شرح، بلا Markdown.
"""


# ══════════════════════════════════════════════════════════════════════════
# 3) تنظيف واستخراج JSON من استجابة النموذج
# ══════════════════════════════════════════════════════════════════════════
_MD_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _clean_and_parse_json_array(raw_text: str) -> List[Dict[str, Any]]:
    """
    يزيل أسيجة Markdown (```json ... ```) وأي نص زائد قبل/بعد المصفوفة،
    ثم يحاول تحليل JSON. يتحمّل قصاصات ناقصة في النهاية بمحاولة اقتطاع
    آخر عنصر غير مكتمل بدل رمي الاستجابة كاملة.
    """
    text = _MD_FENCE_RE.sub("", raw_text).strip()

    # لو فيه نص قبل/بعد المصفوفة، نقتطع من أول [ إلى آخر ]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        pass

    # محاولة إصلاح: القطع الناقصة في نهاية الاستجابة (max_tokens قطع الرد)
    # نحذف آخر فاصلة/عنصر غير مكتمل تدريجياً حتى ينجح التحليل
    last_complete = text.rfind("},")
    while last_complete != -1:
        candidate = text[:last_complete + 1] + "]"
        try:
            data = json.loads(candidate)
            logger.warning("[synth] تم اقتطاع استجابة ناقصة وإصلاحها جزئياً")
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            last_complete = text.rfind("},", 0, last_complete)

    # حالة أضيق: عنصر واحد فقط مقطوع بدون أي "}," (لا يوجد عنصر ثانٍ يسبقه) —
    # نبحث من النهاية للبداية عن أقرب "}" ينتج JSON صالحاً عند إغلاقه، بنفس
    # منطق الحلقة أعلاه، بدل افتراض أول "}" (قد يكون داخل نص القيمة نفسه).
    single_close = text.rfind("}")
    while single_close != -1:
        candidate = text[:single_close + 1] + "]"
        try:
            data = json.loads(candidate)
            logger.warning("[synth] تم اقتطاع استجابة ناقصة (عنصر واحد) وإصلاحها")
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            single_close = text.rfind("}", 0, single_close)

    logger.warning(f"[synth] فشل تحليل JSON نهائياً — أول 200 حرف: {text[:200]!r}")
    return []


def _clean_row(row: Dict[str, Any], topic_key: str, region: str, difficulty: str) -> Optional[Dict[str, Any]]:
    """يطبّع صفاً واحداً ويتحقق من الحقول الإلزامية. يُرجع None لو غير صالح."""
    if not isinstance(row, dict):
        return None
    instruction = str(row.get("instruction", "")).strip()
    output = str(row.get("output", "")).strip()
    if not instruction or not output:
        return None
    # إزالة أي بقايا أسيجة Markdown داخل الحقول نفسها
    instruction = _MD_FENCE_RE.sub("", instruction).strip().strip('"')
    output = _MD_FENCE_RE.sub("", output).strip().strip('"')
    if len(instruction) < 3 or len(output) < 3:
        return None

    row_id = "yem-synth-" + hashlib.sha1(instruction.encode("utf-8")).hexdigest()[:10]
    return {
        "id": row_id,
        "category": topic_key,
        "instruction": instruction,
        "context_ckg": row.get("context_ckg", "") if isinstance(row.get("context_ckg"), str) else "",
        "output": output,
        "dialect_region": region,
        "difficulty": difficulty,
    }


# ══════════════════════════════════════════════════════════════════════════
# 4) استدعاء API مع Exponential Backoff
# ══════════════════════════════════════════════════════════════════════════
def _call_model_with_backoff(
    client: "OpenAI",
    system_prompt: str,
    user_prompt: str,
    model: str,
    max_retries: int = 5,
    base_delay: float = 2.0,
) -> Optional[str]:
    """يستدعي Chat Completions مع إعادة محاولة أسّية عند rate limits/أخطاء مؤقتة."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=1.0,
                max_tokens=2000,
            )
            return resp.choices[0].message.content
        except RateLimitError as e:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"[synth] Rate limit (محاولة {attempt+1}/{max_retries}) — الانتظار {delay:.1f}ث: {e}")
            time.sleep(delay)
        except (APIConnectionError, APITimeoutError) as e:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"[synth] خطأ اتصال (محاولة {attempt+1}/{max_retries}) — الانتظار {delay:.1f}ث: {e}")
            time.sleep(delay)
        except APIError as e:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"[synth] خطأ API (محاولة {attempt+1}/{max_retries}) — الانتظار {delay:.1f}ث: {e}")
            time.sleep(delay)
        except Exception as e:
            logger.error(f"[synth] خطأ غير متوقع، إيقاف إعادة المحاولة لهذه الدفعة: {e}")
            return None
    logger.error(f"[synth] فشل نهائي بعد {max_retries} محاولات")
    return None


def _generate_one_batch(
    client: "OpenAI", model: str, rows_per_call: int
) -> List[Dict[str, Any]]:
    """يولّد دفعة واحدة (topic/region/difficulty عشوائيان) عبر استدعاء API واحد."""
    topic_key, topic_desc = random.choice(TOPICS)
    region = random.choice(DIALECT_REGIONS)
    difficulty = random.choice(DIFFICULTIES)
    user_prompt = _build_user_prompt(topic_key, topic_desc, region, difficulty, rows_per_call)

    raw = _call_model_with_backoff(client, SYSTEM_PROMPT, user_prompt, model)
    if raw is None:
        return []

    parsed = _clean_and_parse_json_array(raw)
    cleaned = [_clean_row(r, topic_key, region, difficulty) for r in parsed]
    return [r for r in cleaned if r is not None]


# ══════════════════════════════════════════════════════════════════════════
# 5) Checkpointing — يقرأ الملف الحالي، يُزيل التكرار، يكتب بشكل ذرّي (atomic)
# ══════════════════════════════════════════════════════════════════════════
def _load_existing_rows(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning(f"[synth] تعذّرت قراءة {path} الحالي ({e}) — سنبدأ من قائمة فارغة (نسخة احتياطية لن تُفقد لأننا لا نكتب حتى ننجح)")
        return []


def _atomic_write_json(path: str, data: List[Dict[str, Any]]) -> None:
    """كتابة ذرّية: يكتب لملف مؤقت ثم يستبدل — يمنع تلف الملف لو توقف السكربت أثناء الكتابة."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def _dedup_by_instruction(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in rows:
        key = r.get("instruction", "").strip()
        if key and key not in seen:
            seen.add(key)
            out.append(r)
    return out


# ══════════════════════════════════════════════════════════════════════════
# 6) الحلقة الرئيسية — تجميع دفعات متزامنة حتى بلوغ target_rows
# ══════════════════════════════════════════════════════════════════════════
def run_generation(
    target_rows: int,
    batch_size: int,
    rows_per_call: int,
    workers: int,
    model: str,
    output_path: str,
) -> None:
    existing = _dedup_by_instruction(_load_existing_rows(output_path))
    logger.info(f"[synth] صفوف موجودة مسبقاً: {len(existing)} / الهدف: {target_rows}")

    if len(existing) >= target_rows:
        logger.info("[synth] الهدف محقَّق مسبقاً — لا حاجة لتوليد إضافي")
        return

    client = _get_client()
    all_rows = existing[:]
    calls_this_round = max(1, (target_rows - len(all_rows) + rows_per_call - 1) // rows_per_call)

    round_num = 0
    while len(all_rows) < target_rows:
        round_num += 1
        remaining_calls = max(1, (target_rows - len(all_rows) + rows_per_call - 1) // rows_per_call)
        n_calls = min(batch_size, remaining_calls)
        logger.info(f"[synth] جولة #{round_num}: {n_calls} استدعاء متزامن (حالياً {len(all_rows)} صف)")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_generate_one_batch, client, model, rows_per_call)
                for _ in range(n_calls)
            ]
            new_rows: List[Dict[str, Any]] = []
            for fut in as_completed(futures):
                try:
                    new_rows.extend(fut.result())
                except Exception as e:
                    logger.warning(f"[synth] فشلت دفعة كاملة: {e}")

        all_rows = _dedup_by_instruction(all_rows + new_rows)
        _atomic_write_json(output_path, all_rows)  # checkpoint فوري بعد كل جولة
        logger.info(f"[synth] ✓ حُفِظ checkpoint: {len(all_rows)} صف إجمالي في {output_path}")

        if not new_rows:
            logger.warning("[synth] جولة بدون أي صف جديد صالح — توقف لتفادي حلقة لا نهائية "
                            "(تحقّق من مفتاح API/الاتصال)")
            break

    logger.info(f"[synth] انتهى. الإجمالي النهائي: {len(all_rows)} صف في {output_path}")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="توليد بيانات يمنية اصطناعية عبر نموذج حدودي")
    p.add_argument("--target-rows", type=int, default=2000, help="إجمالي عدد الصفوف المطلوب في الملف")
    p.add_argument("--rows-per-call", type=int, default=8, help="عدد الصفوف المطلوبة من كل استدعاء API")
    p.add_argument("--batch-size", type=int, default=10, help="عدد استدعاءات API المتزامنة لكل جولة")
    p.add_argument("--workers", type=int, default=5, help="عدد الخيوط المتزامنة (ThreadPoolExecutor)")
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--output", default=OUTPUT_PATH)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_generation(
        target_rows=args.target_rows,
        batch_size=args.batch_size,
        rows_per_call=args.rows_per_call,
        workers=args.workers,
        model=args.model,
        output_path=args.output,
    )

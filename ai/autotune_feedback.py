"""
AutoTune Feedback Loop Engine
==============================
Python port من src/lib/autotune-feedback.ts في G0DM0D3-main.

يجمع إشارات الجودة (تقييم المستخدم 👍/👎 + قياسات آلية) بعد كل رد،
يخزّنها مع المعاملات التي أنتجتها، ويستخدم المتوسط المتحرك الأسي (EMA)
لتعلم أفضل تعديلات المعاملات لكل نوع سياق مع مرور الوقت.

التعديلات المتعلَّمة تمتزج مع اختيار AutoTune للمعاملات —
كلما زادت بيانات التغذية الراجعة زاد التأثير، محدودًا بـ 50%
حتى تبقى الملفات الأساسية هي المرجع.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

ContextType = Literal["code", "creative", "analytical", "conversational", "chaotic"]

# ── ثوابت ───────────────────────────────────────────────────────────

EMA_ALPHA              = 0.3   # وزن الملاحظات الجديدة (أعلى = تعلم أسرع)
MAX_HISTORY            = 500   # حد تاريخ التغذية الراجعة
MIN_SAMPLES_TO_APPLY   = 3     # أقل عدد عينات قبل تطبيق التعديلات
MAX_LEARNED_WEIGHT     = 0.5   # أقصى تأثير للتعديلات المتعلمة (50%)
SAMPLES_FOR_MAX_WEIGHT = 20    # عينات للوصول للوزن الأقصى

NEUTRAL_PARAMS: Dict[str, float] = {
    "temperature":        0.7,
    "top_p":              0.9,
    "top_k":              50.0,
    "frequency_penalty":  0.2,
    "presence_penalty":   0.2,
    "repetition_penalty": 1.1,
}

DB_PATH = Path("memory/autotune_feedback.db")

# ── قياسات الجودة الآلية ─────────────────────────────────────────────

@dataclass
class ResponseHeuristics:
    response_length:         int
    repetition_score:        float   # 0.0 = لا تكرار، 1.0 = تكرار شديد
    avg_sentence_length:     float
    vocabulary_diversity:    float   # كلمات فريدة / إجمالي الكلمات


def _compute_repetition_score(text: str) -> float:
    """رصد التكرار عبر فحص 3-grams المتكررة."""
    words = [w for w in text.lower().split() if w]
    if len(words) < 6:
        return 0.0
    trigrams: Dict[str, int] = {}
    for i in range(len(words) - 2):
        tri = f"{words[i]} {words[i+1]} {words[i+2]}"
        trigrams[tri] = trigrams.get(tri, 0) + 1
    total = len(trigrams)
    if total == 0:
        return 0.0
    repeated = sum(1 for c in trigrams.values() if c > 1)
    return min(repeated / total, 1.0)


def compute_heuristics(response: str) -> ResponseHeuristics:
    """حساب قياسات الجودة الآلية للرد."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", response) if s.strip()]
    words = [w for w in response.lower().split() if w]
    unique_words = set(words)
    avg_sent = (
        sum(len(s.split()) for s in sentences) / len(sentences)
        if sentences else 0.0
    )
    return ResponseHeuristics(
        response_length=len(response),
        repetition_score=_compute_repetition_score(response),
        avg_sentence_length=avg_sent,
        vocabulary_diversity=len(unique_words) / len(words) if words else 1.0,
    )


# ── بنى البيانات ─────────────────────────────────────────────────────

@dataclass
class FeedbackRecord:
    message_id:   str
    timestamp:    float
    context_type: str
    model:        str
    persona:      str
    params:       Dict[str, float]
    rating:       int   # 1 = إعجاب، -1 = عدم إعجاب
    heuristics:   Dict[str, float]


@dataclass
class LearnedProfile:
    context_type:    str
    sample_count:    int                    = 0
    positive_count:  int                    = 0
    negative_count:  int                    = 0
    positive_params: Dict[str, float]       = field(default_factory=lambda: dict(NEUTRAL_PARAMS))
    negative_params: Dict[str, float]       = field(default_factory=lambda: dict(NEUTRAL_PARAMS))
    adjustments:     Dict[str, float]       = field(default_factory=dict)
    last_updated:    float                  = field(default_factory=time.time)


def _empty_profiles() -> Dict[str, LearnedProfile]:
    return {
        ctx: LearnedProfile(context_type=ctx)
        for ctx in ("code", "creative", "analytical", "conversational", "chaotic")
    }


# ── قاعدة البيانات ───────────────────────────────────────────────────

def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=3000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback_records (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id  TEXT NOT NULL,
            timestamp   REAL NOT NULL,
            context_type TEXT NOT NULL,
            model       TEXT NOT NULL,
            persona     TEXT NOT NULL,
            params_json TEXT NOT NULL,
            rating      INTEGER NOT NULL,
            heuristics_json TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS learned_profiles (
            context_type    TEXT PRIMARY KEY,
            profile_json    TEXT NOT NULL,
            last_updated    REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def save_feedback(record: FeedbackRecord) -> None:
    """حفظ سجل تغذية راجعة في قاعدة البيانات."""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO feedback_records
           (message_id, timestamp, context_type, model, persona, params_json, rating, heuristics_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            record.message_id,
            record.timestamp,
            record.context_type,
            record.model,
            record.persona,
            json.dumps(record.params),
            record.rating,
            json.dumps(record.heuristics),
        ),
    )
    conn.commit()
    conn.close()


def load_profiles() -> Dict[str, LearnedProfile]:
    """تحميل الملفات المتعلمة من قاعدة البيانات."""
    profiles = _empty_profiles()
    try:
        conn = _get_conn()
        rows = conn.execute("SELECT context_type, profile_json FROM learned_profiles").fetchall()
        conn.close()
        for ctx, pjson in rows:
            data = json.loads(pjson)
            profiles[ctx] = LearnedProfile(**data)
    except Exception:
        pass
    return profiles


def _save_profile(profile: LearnedProfile) -> None:
    conn = _get_conn()
    data = asdict(profile)
    conn.execute(
        """INSERT OR REPLACE INTO learned_profiles (context_type, profile_json, last_updated)
           VALUES (?, ?, ?)""",
        (profile.context_type, json.dumps(data), profile.last_updated),
    )
    conn.commit()
    conn.close()


# ── محرك EMA ─────────────────────────────────────────────────────────

def _ema_update(current: Dict[str, float], new_val: Dict[str, float], alpha: float) -> Dict[str, float]:
    """تحديث المتوسط المتحرك الأسي."""
    return {
        k: alpha * new_val.get(k, current[k]) + (1 - alpha) * current[k]
        for k in current
    }


def _compute_adjustments(profile: LearnedProfile) -> Dict[str, float]:
    """حساب دلتا المعاملات: الفرق بين الأنماط الإيجابية والسلبية."""
    if profile.positive_count == 0 or profile.negative_count == 0:
        return {}
    deltas: Dict[str, float] = {}
    for key in NEUTRAL_PARAMS:
        pos = profile.positive_params.get(key, NEUTRAL_PARAMS[key])
        neg = profile.negative_params.get(key, NEUTRAL_PARAMS[key])
        delta = pos - neg
        if abs(delta) > 0.01:
            deltas[key] = delta * 0.5  # تطبيق نصف الدلتا للحفاظ على الاستقرار
    return deltas


def process_feedback(
    record: FeedbackRecord,
    profiles: Optional[Dict[str, LearnedProfile]] = None,
) -> Dict[str, LearnedProfile]:
    """
    معالجة سجل تغذية راجعة جديد وتحديث الملف المتعلَّم.
    يحفظ السجل في قاعدة البيانات ويعيد الملفات المحدّثة.
    """
    if profiles is None:
        profiles = load_profiles()

    save_feedback(record)

    ctx = record.context_type
    if ctx not in profiles:
        profiles[ctx] = LearnedProfile(context_type=ctx)

    profile = profiles[ctx]
    profile.sample_count += 1
    profile.last_updated = time.time()

    if record.rating == 1:
        profile.positive_count += 1
        profile.positive_params = _ema_update(
            profile.positive_params, record.params, EMA_ALPHA
        )
    else:
        profile.negative_count += 1
        profile.negative_params = _ema_update(
            profile.negative_params, record.params, EMA_ALPHA
        )

    # إعادة حساب التعديلات
    profile.adjustments = _compute_adjustments(profile)
    _save_profile(profile)

    return profiles


# ── تطبيق التعديلات المتعلمة على AutoTune ───────────────────────────

# حدود آمنة لمعاملات API — تمنع الانجراف خارج النطاق المقبول
PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "temperature":        (0.0, 2.0),
    "top_p":              (0.01, 1.0),
    "top_k":              (1.0, 100.0),
    "frequency_penalty":  (-2.0, 2.0),
    "presence_penalty":   (-2.0, 2.0),
    "repetition_penalty": (1.0, 2.0),
}


def _clamp_params(params: Dict[str, float]) -> Dict[str, float]:
    """تثبيت المعاملات ضمن حدود آمنة لـ API."""
    result = {}
    for key, val in params.items():
        lo, hi = PARAM_BOUNDS.get(key, (-1e9, 1e9))
        result[key] = max(lo, min(hi, val))
    return result


def apply_learned_adjustments(
    params: Dict[str, float],
    context_type: str,
    profiles: Optional[Dict[str, LearnedProfile]] = None,
) -> Tuple[Dict[str, float], bool, str]:
    """
    تطبيق التعديلات المتعلمة على معاملات AutoTune.
    يعيد: (المعاملات المعدّلة، هل طُبّق التعلم، ملاحظة للعرض)
    المعاملات المُعادة مضمونة ضمن حدود API الآمنة.
    """
    if profiles is None:
        profiles = load_profiles()

    profile = profiles.get(context_type)
    if not profile:
        return _clamp_params(params), False, ""

    if profile.sample_count < MIN_SAMPLES_TO_APPLY:
        return _clamp_params(params), False, f"التعلم: {profile.sample_count}/{MIN_SAMPLES_TO_APPLY} عينات"

    if not profile.adjustments:
        return _clamp_params(params), False, "التعلم: لا توجد تعديلات بعد"

    # حساب وزن التعلم (يزداد مع زيادة العينات، حتى 50%)
    weight = min(
        (profile.sample_count / SAMPLES_FOR_MAX_WEIGHT) * MAX_LEARNED_WEIGHT,
        MAX_LEARNED_WEIGHT,
    )

    adjusted = dict(params)
    applied_keys: List[str] = []
    for key, delta in profile.adjustments.items():
        if key in adjusted:
            adjusted[key] = adjusted[key] + delta * weight
            applied_keys.append(key)

    # تثبيت النتيجة ضمن الحدود الآمنة
    adjusted = _clamp_params(adjusted)

    note = (
        f"✨ تعلّم: {len(applied_keys)} معامل معدَّل "
        f"({profile.sample_count} عينة، {weight:.0%} وزن)"
    )
    return adjusted, True, note


# ── إحصائيات ─────────────────────────────────────────────────────────

def get_feedback_stats(profiles: Optional[Dict[str, LearnedProfile]] = None) -> Dict:
    """إحصائيات موجزة لنظام التعلم."""
    if profiles is None:
        profiles = load_profiles()

    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT rating FROM feedback_records ORDER BY timestamp"
        ).fetchall()
        conn.close()
    except Exception:
        rows = []

    total = len(rows)
    positive = sum(1 for (r,) in rows if r == 1)

    ctx_breakdown: Dict[str, Dict] = {}
    for ctx, profile in profiles.items():
        ctx_breakdown[ctx] = {
            "total":       profile.sample_count,
            "positive":    profile.positive_count,
            "negative":    profile.negative_count,
            "has_learned": (
                profile.sample_count >= MIN_SAMPLES_TO_APPLY
                and bool(profile.adjustments)
            ),
        }

    return {
        "total_feedback":    total,
        "positive_rate":     positive / total if total else 0.0,
        "context_breakdown": ctx_breakdown,
    }

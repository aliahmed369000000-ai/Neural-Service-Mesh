"""
Social Swarm — سرب الوكيل الاجتماعي
===================================
  • TrendScoutAgent — رصد اتجاهات (RSS/كلمات صاعدة — بدون كشط منصات مخالف)
  • PsychologicalContentAgent — صياغة حسب شخصية المنصة
  • VisualCoordinatorAgent — صورة Unsplash إن وُجد المفتاح
  • CrisisControl — فحص قبل النشر + تجميد
  • SocialCRM — ملف متابع عبر المنصات (SQLite خفيف)
  • CommerceFunnel — رد اشتراك/باقات (بدون خصم دفع حقيقي)

يُستدعى من agent_factory roles ومن أوامر التدريب/الاجتماعي.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

logger = logging.getLogger("SocialSwarm")

ROOT = Path(__file__).resolve().parent.parent
SWARM_DIR = ROOT / "artifacts" / "model_training" / "social_swarm"
SWARM_DIR.mkdir(parents=True, exist_ok=True)
CRM_DB = SWARM_DIR / "social_crm.sqlite"
FREEZE_FLAG = SWARM_DIR / "PUBLISH_FROZEN.json"

PLATFORM_VOICE = {
    "linkedin": "أسلوب رصين مهني، فقرة قصيرة وفائدة عملية، بلا إيموجي مفرط.",
    "twitter": "جملة حادّة واضحة ≤260 حرفاً، سؤال تفاعلي في النهاية.",
    "x": "جملة حادّة واضحة ≤260 حرفاً، سؤال تفاعلي في النهاية.",
    "threads": "نبرة حوارية قريبة، جمل قصيرة متتابعة.",
    "tiktok": "خطاف أول 3 ثوانٍ + 3 نقاط سريعة + دعوة للمتابعة.",
    "instagram": "نص دافئ مع أسطر متباعدة وهاشتاجات خفيفة في النهاية.",
    "facebook": "أسلوب مجتمعي واضح مع دعوة للتعليق.",
    "telegram": "رسالة مباشرة منظمة بنقاط.",
    "whatsapp": "رد شخصي مختصر ومهذب.",
    "youtube": "عنوان جذّاب + وصف منظم بأقسام.",
    "reddit": "نبرة نقاش موضوعية بلا دعاية فجّة.",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Trend Scout ───────────────────────────────────────────────────────────

def scout_trends(limit: int = 8) -> Dict[str, Any]:
    """يجلب عناوين من RSS عام (Google News AR) + كلمات مفتاحية مستخرجة."""
    trends: List[Dict[str, str]] = []
    feeds = [
        "https://news.google.com/rss?hl=ar&gl=SA&ceid=SA:ar",
        "https://news.google.com/rss/search?q=%D8%B0%D9%83%D8%A7%D8%A1+%D8%A7%D8%B5%D8%B7%D9%86%D8%A7%D8%B9%D9%8A&hl=ar&gl=SA&ceid=SA:ar",
    ]
    for url in feeds:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NSM-SocialSwarm/1.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = resp.read()
            root = ET.fromstring(data)
            for item in root.findall(".//item")[: limit]:
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                if title:
                    trends.append({"title": title[:200], "link": link, "source": "rss"})
        except Exception as e:
            logger.info("trend feed skip %s: %s", url, e)
    # كلمات مفتاحية بسيطة
    tags: Dict[str, int] = {}
    for t in trends:
        for w in re.findall(r"[\u0600-\u06FFa-zA-Z]{3,}", t["title"]):
            if w in ("التي", "هذا", "هذه", "من", "على", "في", "إلى"):
                continue
            tags[w] = tags.get(w, 0) + 1
    top_tags = sorted(tags.items(), key=lambda x: -x[1])[:12]
    report = {
        "ok": True,
        "n": len(trends),
        "trends": trends[:limit],
        "rising_tags": [{"tag": k, "score": v} for k, v in top_tags],
        "note_ar": "رصد عبر RSS عام — ليس بديلاً عن TikTok/Reels API الرسمي.",
        "at": _now(),
    }
    (SWARM_DIR / "last_trends.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


# ── Content ───────────────────────────────────────────────────────────────

def craft_platform_posts(topic: str, platforms: Optional[List[str]] = None) -> Dict[str, str]:
    topic = (topic or "معرفة نافعة").strip()
    platforms = platforms or ["linkedin", "twitter", "instagram", "tiktok", "telegram"]
    out: Dict[str, str] = {}
    for p in platforms:
        voice = PLATFORM_VOICE.get(p, "نص واضح ومختصر بالعربية الفصحى المبسطة.")
        body = {
            "linkedin": f"{topic}\n\nثلاث نقاط عملية:\n• افهم السياق\n• طبّق بخطوة واحدة اليوم\n• راجع الأثر أسبوعياً\n\n#معرفة #تطوير",
            "twitter": f"{topic} — ما أهم درس تعلّمته من هذا الموضوع؟",
            "x": f"{topic} — ما أهم درس تعلّمته من هذا الموضوع؟",
            "instagram": f"{topic}\n\n✨ خذ نفساً\n✨ طبّق فكرة واحدة\n✨ شارك صديقك\n\n#تأمل #معرفة",
            "tiktok": f"هل تعلم؟ {topic}\n1) الفكرة\n2) المثال\n3) التطبيق\nتابعني للمزيد",
            "telegram": f"📌 {topic}\n\nملخص سريع ونقاش مفتوح في الردود.",
            "facebook": f"{topic}\nما رأيكم؟ شاركونا تجاربكم في التعليقات.",
            "threads": f"{topic}\n\nأحدّثكم باختصار — وأسمعكم.",
            "whatsapp": f"مرحباً، بخصوص «{topic}»: هل تريد ملخصاً أم خطوات عملية؟",
            "youtube": f"عنوان: {topic}\nالوصف: شرح مبسّط + أمثلة + خلاصة قابلة للتطبيق.",
            "reddit": f"**نقاش:** {topic}\n\nأفتح الموضوع للنقاش بهدوء — مصادركم؟",
        }.get(p, f"{topic}\n({voice})")
        out[p] = body
    return out


# ── Visual ────────────────────────────────────────────────────────────────

def suggest_visual(query: str) -> Dict[str, Any]:
    key = os.environ.get("UNSPLASH_ACCESS_KEY") or os.environ.get("UNSPLASH_KEY")
    if not key:
        return {
            "ok": False,
            "image_url": None,
            "note_ar": "اضبط UNSPLASH_ACCESS_KEY لاقتراح صور حقيقية.",
            "query": query,
        }
    try:
        q = urllib.parse.quote(query[:80])
        url = f"https://api.unsplash.com/search/photos?query={q}&per_page=1"
        req = urllib.request.Request(url, headers={"Authorization": f"Client-ID {key}"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = data.get("results") or []
        if not results:
            return {"ok": False, "image_url": None, "note_ar": "لا نتائج", "query": query}
        img = results[0].get("urls", {}).get("regular")
        return {"ok": True, "image_url": img, "query": query, "credit": "unsplash"}
    except Exception as e:
        return {"ok": False, "image_url": None, "error": str(e), "query": query}


# ── Crisis ────────────────────────────────────────────────────────────────

_TOXIC_PATTERNS = [
    r"اقتل", r"إرهاب", r"سب[ّ\s]*الدين", r"عنصر[يى]", r"شتائم",
    r"kill\s+yourself", r"terror", r"slur",
]


def pre_publish_check(text: str) -> Dict[str, Any]:
    flags = []
    low = (text or "").lower()
    for pat in _TOXIC_PATTERNS:
        if re.search(pat, text or "", re.I) or re.search(pat, low):
            flags.append(pat)
    # مشاعر سلبية كثيفة كلمات
    neg = len(re.findall(r"(كره|احتقار|حقير|سخيف|فاشل)", text or ""))
    if neg >= 3:
        flags.append("negative_density")
    frozen = FREEZE_FLAG.is_file()
    ok = not flags and not frozen
    return {
        "ok": ok,
        "flags": flags,
        "frozen": frozen,
        "recommendation": "publish" if ok else "block",
        "reason_ar": (
            "آمن للنشر ضمن الفحص الآلي" if ok else
            ("النشر مجمّد يدوياً/تلقائياً" if frozen else "عُثر على إشارات خطر — مراجعة بشرية")
        ),
    }


def freeze_publishing(reason: str = "hostile_activity") -> Dict[str, Any]:
    payload = {"frozen": True, "reason": reason, "at": _now()}
    FREEZE_FLAG.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def unfreeze_publishing() -> Dict[str, Any]:
    if FREEZE_FLAG.is_file():
        FREEZE_FLAG.unlink()
    return {"frozen": False, "at": _now()}


def detect_hostility_burst(negative_count: int, window_hint: str = "recent") -> Dict[str, Any]:
    """إذا تجاوزت التعليقات السلبية عتبة — تجميد."""
    if negative_count >= 8:
        return {"action": "freeze", **freeze_publishing(f"burst_{window_hint}_{negative_count}")}
    return {"action": "continue", "negative_count": negative_count}


# ── CRM ───────────────────────────────────────────────────────────────────

def _crm_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(CRM_DB))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS followers (
            identity_key TEXT PRIMARY KEY,
            display_name TEXT,
            platforms_json TEXT,
            interests_json TEXT,
            last_seen TEXT,
            notes TEXT
        )
        """
    )
    conn.commit()
    return conn


def upsert_follower(
    name: str,
    platform: str,
    external_id: str = "",
    interest: str = "",
) -> Dict[str, Any]:
    key = f"{(name or 'user').strip().lower()}::{(external_id or name).strip().lower()}"
    conn = _crm_conn()
    row = conn.execute("SELECT platforms_json, interests_json FROM followers WHERE identity_key=?", (key,)).fetchone()
    platforms = {}
    interests = []
    if row:
        try:
            platforms = json.loads(row[0] or "{}")
            interests = json.loads(row[1] or "[]")
        except Exception:
            pass
    platforms[platform] = external_id or name
    if interest and interest not in interests:
        interests.append(interest)
    conn.execute(
        """
        INSERT INTO followers(identity_key, display_name, platforms_json, interests_json, last_seen, notes)
        VALUES(?,?,?,?,?,?)
        ON CONFLICT(identity_key) DO UPDATE SET
          platforms_json=excluded.platforms_json,
          interests_json=excluded.interests_json,
          last_seen=excluded.last_seen
        """,
        (key, name, json.dumps(platforms, ensure_ascii=False), json.dumps(interests, ensure_ascii=False), _now(), ""),
    )
    conn.commit()
    conn.close()
    return {"identity_key": key, "name": name, "platforms": platforms, "interests": interests}


def recall_follower(name: str) -> Optional[Dict[str, Any]]:
    conn = _crm_conn()
    rows = conn.execute(
        "SELECT identity_key, display_name, platforms_json, interests_json, last_seen FROM followers WHERE display_name LIKE ?",
        (f"%{name}%",),
    ).fetchall()
    conn.close()
    if not rows:
        return None
    r = rows[0]
    return {
        "identity_key": r[0],
        "name": r[1],
        "platforms": json.loads(r[2] or "{}"),
        "interests": json.loads(r[3] or "[]"),
        "last_seen": r[4],
    }


def personalized_reply(name: str, message: str) -> str:
    profile = recall_follower(name) or upsert_follower(name, "unknown")
    interests = profile.get("interests") or []
    interest_line = f" لاحظت اهتمامك بـ «{interests[-1]}»." if interests else ""
    return (
        f"مرحباً {profile.get('name', name)}،{interest_line} "
        f"بخصوص رسالتك: «{(message or '')[:120]}» — كيف تريد أن أساعدك تحديداً اليوم؟"
    )


# ── Commerce ──────────────────────────────────────────────────────────────

PLANS = {
    "free": {"name": "Free", "price_usd": 0, "blurb": "تجربة محدودة"},
    "pro": {"name": "Pro", "price_usd": 49, "blurb": "تدريب ومهام يومية أعلى"},
    "enterprise": {"name": "Enterprise", "price_usd": 299, "blurb": "عزل مستأجرين ودعم"},
}


def commerce_reply(user_message: str) -> Dict[str, Any]:
    msg = (user_message or "").lower()
    suggest = "pro"
    if any(k in msg for k in ("شرك", "مؤسس", "فريق", "enterprise", "مؤسسة")):
        suggest = "enterprise"
    elif any(k in msg for k in ("تجرب", "مجاني", "free")):
        suggest = "free"
    plan = PLANS[suggest]
    text = (
        f"يسعدني مساعدتك في الاشتراك. بناءً على وصفك أقترح باقة **{plan['name']}** "
        f"(${plan['price_usd']}/شهر): {plan['blurb']}. "
        "يمكن إتمام الدفع عبر بوابة المؤسسة (Stripe/PayPal) بعد تفعيل المفاتيح — "
        "أرسل «أريد Pro» أو «Enterprise» لأسفل لك خطوات الربط."
    )
    return {"suggested_plan": suggest, "plan": plan, "reply_ar": text, "payment": "not_charged_demo"}


# ── Orchestrate swarm ─────────────────────────────────────────────────────

def run_social_swarm(topic: Optional[str] = None, platforms: Optional[List[str]] = None) -> Dict[str, Any]:
    trends = scout_trends()
    if not topic:
        topic = (trends.get("trends") or [{"title": "معرفة نافعة"}])[0].get("title") or "معرفة نافعة"
        # قص العنوان
        topic = topic.split(" - ")[0][:80]
    posts = craft_platform_posts(topic, platforms)
    visual = suggest_visual(topic)
    checks = {p: pre_publish_check(txt) for p, txt in posts.items()}
    safe_posts = {p: t for p, t in posts.items() if checks[p].get("ok")}
    report = {
        "ok": True,
        "topic": topic,
        "trends_sample": (trends.get("trends") or [])[:5],
        "tags": trends.get("rising_tags") or [],
        "posts": posts,
        "safe_posts": safe_posts,
        "checks": checks,
        "visual": visual,
        "at": _now(),
    }
    path = SWARM_DIR / f"swarm_run_{int(time.time())}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["path"] = str(path.relative_to(ROOT))
    return report


def handle_social_swarm_command(user_input: str) -> Optional[str]:
    text = (user_input or "").strip()
    if not text:
        return None
    if re.search(r"(تجميد\s*نشر|freeze\s*publish)", text, re.I):
        return "## 🧊 تجميد النشر\n```json\n" + json.dumps(freeze_publishing(), ensure_ascii=False, indent=2) + "\n```"
    if re.search(r"(رفع\s*تجميد|unfreeze)", text, re.I):
        return "## ✅ رفع التجميد\n```json\n" + json.dumps(unfreeze_publishing(), ensure_ascii=False, indent=2) + "\n```"
    if re.search(r"(crm|متابع|تذك[رّ]\s*متابع)", text, re.I):
        m = re.search(r"(?:متابع|اسم)[:\s]+(\S+)", text)
        name = m.group(1) if m else "أحمد"
        interest_m = re.search(r"(?:اهتمام|سورة|مفهوم)[:\s]+(.+)$", text)
        interest = interest_m.group(1).strip() if interest_m else ""
        if interest:
            upsert_follower(name, "telegram", interest=interest)
        prof = recall_follower(name) or upsert_follower(name, "telegram")
        return "## 👤 CRM اجتماعي\n```json\n" + json.dumps(prof, ensure_ascii=False, indent=2) + "\n```\n\n" + personalized_reply(name, "مرحبا")
    if re.search(r"(اشتراك|باق[ةه]|commerce|stripe|بيع\s*خدم)", text, re.I):
        return "## 💼 Social Commerce\n" + commerce_reply(text)["reply_ar"]
    if re.search(r"(سرب\s*اجتماع|social\s*swarm|تريند\s*اجتماع|صغ\s*منشورات)", text, re.I):
        topic = None
        m = re.search(r"(?:حول|عن|topic)[:\s]+(.+)$", text, re.I)
        if m:
            topic = m.group(1).strip()[:120]
        r = run_social_swarm(topic=topic)
        lines = [
            f"## 🐝 السرب الاجتماعي — {r['topic']}",
            f"- تريندات: {r.get('trends_sample') and len(r['trends_sample'])}",
            f"- منشورات آمنة: {len(r.get('safe_posts') or {})}/{len(r.get('posts') or {})}",
            f"- ملف: `{r.get('path')}`",
            "",
            "### عيّنة",
        ]
        for p, t in list((r.get("safe_posts") or r.get("posts") or {}).items())[:4]:
            lines.append(f"**{p}:** {t[:160]}…")
        return "\n".join(lines)
    if re.search(r"(رصد\s*تريند|trend\s*scout)", text, re.I):
        r = scout_trends()
        return "## 📈 Trend Scout\n```json\n" + json.dumps(r, ensure_ascii=False, indent=2)[:3000] + "\n```"
    return None

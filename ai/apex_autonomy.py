"""
Apex Autonomy Layer — طبقة الذروة (Narrow-AGI scaffolding)
==========================================================
مراحل مفاهيمية متقدمة فوق مصنع التدريب وAIaaS:

  1) AI Mergers: رصد نماذج خارجية (ميتا فقط) ودمج قدرات في كتالوج داخلي
  2) Synthetic Knowledge: توليد بيانات اصطناعية عبر حوار/تنافس نماذج صغيرة
  3) Autonomous DAO (محاكاة): دفتر قرارات وخزينة افتراضية — بلا مفاتيح تشفير حقيقية

⚠️ حدود صريحة:
  • هذا ليس AGI ولا كياناً قانونياً مستقلاً.
  • لا محفظة كريبتو حقيقية ولا دفع تلقائي لفاتورة سيرفرات.
  • لا استحواذ مالي على مستودعات/نماذج مدفوعة.
  • الدمج = فهرسة قدرات + تجارب محلية، لا نسخ أوزان محمية بترخيص دون مراجعة.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ApexAutonomy")

ROOT = Path(__file__).resolve().parent.parent
APEX = ROOT / "artifacts" / "model_training" / "apex"
MERGERS = APEX / "mergers"
SYN = APEX / "synthetic"
DAO = APEX / "dao"
for d in (APEX, MERGERS, SYN, DAO):
    d.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_apex_cfg() -> Dict[str, Any]:
    defaults = {
        "mergers_enabled": True,
        "synthetic_enabled": True,
        "dao_simulation_only": True,
        "real_crypto_wallets": False,  # مرفوض افتراضياً وللأبد تقريباً في هذا الكود
        "max_synthetic_rows": 500,
        "max_merge_candidates": 5,
    }
    path = ROOT / "config" / "training_guardrails.json"
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            ax = data.get("apex") or {}
            defaults.update({k: v for k, v in ax.items() if v is not None})
    except Exception as e:
        logger.warning("apex cfg: %s", e)
    # فرض الأمان
    defaults["real_crypto_wallets"] = False
    defaults["dao_simulation_only"] = True
    return defaults


# ── 1) AI Mergers (catalog + capability map) ───────────────────────────────

def discover_external_models(query: str = "arabic nlp", max_results: int = 5) -> str:
    """رصد نماذج على Hugging Face (ميتا) وفهرستها كمرشحي دمج."""
    cfg = _load_apex_cfg()
    if not cfg.get("mergers_enabled", True):
        return "عمليات الدمج معطّلة في الإعدادات."
    max_results = min(int(max_results), int(cfg.get("max_merge_candidates") or 5))
    results: List[Dict[str, Any]] = []
    notes = []
    try:
        from ai.training_web_access import search_huggingface, _offline

        if _offline():
            notes.append("وضع offline — فهرسة محلية فقط")
        else:
            raw = search_huggingface(query, kind="models", max_results=max_results)
            notes.append("HF meta fetched")
            # استخرج معرفات من النص
            ids = re.findall(r"\*\*([^*]+)\*\*", raw)
            for mid in ids[:max_results]:
                results.append(
                    {
                        "source": "huggingface",
                        "model_id": mid.strip(),
                        "license_review": "required_before_weight_use",
                        "integration": "metadata_only",
                    }
                )
    except Exception as e:
        notes.append(f"HF: {e}")

    # مرشحون محليون من registry
    try:
        from ai.training_feedback_loop import load_registry

        reg = load_registry()
        for m in (reg.get("models") or [])[:3]:
            results.append(
                {
                    "source": "local_registry",
                    "model_id": m.get("id"),
                    "metric": f"{m.get('metric_name')}={m.get('metric_value')}",
                    "integration": "already_owned",
                }
            )
    except Exception as e:
        notes.append(f"registry: {e}")

    catalog = {
        "id": f"merge_scan_{uuid.uuid4().hex[:8]}",
        "query": query,
        "created_at": _now(),
        "candidates": results,
        "notes": notes,
        "disclaimer": "لا شراء ولا تنزيل أوزان تلقائي — فهرسة قدرات فقط",
    }
    path = MERGERS / f"{catalog['id']}.json"
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "## 🔗 رصد نماذج للدمج (AI Mergers — ميثا)",
        f"- scan: `{catalog['id']}`",
        f"- query: `{query}`",
        f"- مرشحون: **{len(results)}**",
        "",
    ]
    for c in results:
        lines.append(
            f"- [{c.get('source')}] `{c.get('model_id')}` — {c.get('integration')} "
            f"{c.get('metric') or ''}"
        )
    lines.append("")
    lines.append(
        "للدمج المفاهيمي مع الكتالوج الداخلي: `ادمج قدرات "
        + (results[0]["model_id"] if results else "model_id")
        + "`"
    )
    lines.append(f"الملف: `{path.relative_to(ROOT)}`")
    lines.append(f"_ملاحظات: {'; '.join(notes)}_")
    return "\n".join(lines)


def merge_capability(model_id: str, capability_tags: Optional[List[str]] = None) -> str:
    """
    دمج قدرات في خريطة داخلية (ليس دمج أوزان).
    يسجّل أن النظام 'يمتلك مسار تكامل' لهذا النموذج/القدرة.
    """
    tags = capability_tags or ["external_model", "metadata_integration"]
    entry = {
        "id": f"cap_{uuid.uuid4().hex[:8]}",
        "model_id": model_id,
        "tags": tags,
        "merged_at": _now(),
        "type": "capability_map_entry",
        "status": "indexed",
    }
    # تحديث خريطة القدرات
    cmap_path = MERGERS / "capability_map.json"
    cmap = {"capabilities": []}
    if cmap_path.is_file():
        try:
            cmap = json.loads(cmap_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    cmap.setdefault("capabilities", []).append(entry)
    cmap["updated_at"] = _now()
    cmap_path.write_text(json.dumps(cmap, ensure_ascii=False, indent=2), encoding="utf-8")
    (MERGERS / f"{entry['id']}.json").write_text(
        json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return (
        f"## ✅ دمج قدرات (مفاهيمي)\n"
        f"- entry: `{entry['id']}`\n"
        f"- model: `{model_id}`\n"
        f"- tags: {tags}\n"
        f"- الخريطة: `artifacts/model_training/apex/mergers/capability_map.json`\n\n"
        f"_لم تُحمَّل أوزان ولم يُنفَّذ استحواذ مالي._"
    )


def capability_map_report() -> str:
    path = MERGERS / "capability_map.json"
    if not path.is_file():
        return "خريطة القدرات فارغة. ابدأ بـ `ارصد نماذج arabic nlp` ثم `ادمج قدرات …`."
    cmap = json.loads(path.read_text(encoding="utf-8"))
    lines = ["## 🗺️ خريطة القدرات المدمجة", f"- محدّثة: {cmap.get('updated_at')}", ""]
    for c in (cmap.get("capabilities") or [])[-20:]:
        lines.append(f"- `{c.get('id')}` ← `{c.get('model_id')}` tags={c.get('tags')}")
    return "\n".join(lines)


# ── 2) Synthetic Knowledge Generation ──────────────────────────────────────

def _synthetic_dialogue_rows(topic: str, n: int) -> List[Dict[str, Any]]:
    """
    توليد صفوف اصطناعية عبر قواعد + تنويعات (محاكاة تنافس معلّم/طالب).
    لا يعتمد على LLM خارجي إن لم يتوفر — قابل للاستبدال لاحقاً.
    """
    rng = np.random.default_rng(abs(hash(topic)) % (2**32))
    templates_pos = [
        f"هذا النص يدعم فكرة {topic} بشكل واضح ومفيد.",
        f"تجربة إيجابية حول {topic} مع نتائج جيدة.",
        f"أوصي بالاهتمام بـ {topic} لما فيه من فائدة.",
    ]
    templates_neg = [
        f"لا أرى فائدة حقيقية من {topic} في هذا السياق.",
        f"نقد مباشر لـ {topic} مع تحفظات جوهرية.",
        f"نتائج ضعيفة عند تطبيق {topic}.",
    ]
    rows = []
    for i in range(n):
        # تنافس: معلّم يختار قطبية، طالب يضيف ضوضاء لفظية
        label = int(rng.integers(0, 2))
        base = templates_pos[i % 3] if label == 1 else templates_neg[i % 3]
        noise = "".join(rng.choice(list("ابتجحخدسشعيقلمنوي "), size=int(rng.integers(0, 8))))
        rows.append({"text": f"{base} {noise}".strip(), "label": label, "source": "synthetic_debate"})
    return rows


def generate_synthetic_dataset(topic: str = "جودة الخدمة", n_rows: int = 100) -> str:
    cfg = _load_apex_cfg()
    if not cfg.get("synthetic_enabled", True):
        return "التوليد الاصطناعي معطّل."
    n_rows = max(20, min(int(n_rows), int(cfg.get("max_synthetic_rows") or 500)))
    rows = _synthetic_dialogue_rows(topic, n_rows)
    # كتابة CSV
    out = SYN / f"synth_{uuid.uuid4().hex[:8]}.csv"
    with open(out, "w", encoding="utf-8") as f:
        f.write("text,label\n")
        for r in rows:
            text = r["text"].replace('"', "'")
            f.write(f'"{text}",{r["label"]}\n')
    meta = {
        "path": str(out.relative_to(ROOT)),
        "topic": topic,
        "n_rows": n_rows,
        "method": "teacher_student_template_debate",
        "created_at": _now(),
        "disclaimer": "بيانات اصطناعية لأغراض التجريب — ليست معرفة عالم حقيقية",
    }
    (SYN / f"meta_{out.stem}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return (
        f"## 🧪 معرفة اصطناعية\n"
        f"- الموضوع: `{topic}`\n"
        f"- صفوف: **{n_rows}**\n"
        f"- الملف: `{out.relative_to(ROOT)}`\n"
        f"- الأسلوب: حوار معلّم/طالب قالبي (قابل للترقية لنماذج حقيقية)\n\n"
        f"درّب عليها: `درّب على csv {out.relative_to(ROOT)}` "
        f"أو `شغّل مصنع على الاصطناعي`"
    )


def train_on_synthetic(topic: str = "جودة الخدمة", n_rows: int = 120, epochs: int = 15) -> str:
    gen = generate_synthetic_dataset(topic, n_rows)
    # استخرج المسار
    m = re.search(r"`(artifacts/model_training/apex/synthetic/synth_[\w]+\.csv)`", gen)
    if not m:
        return gen
    path = m.group(1)
    try:
        from ai.training_feedback_loop import self_correct_and_train

        train_res = self_correct_and_train(dataset=path, epochs=epochs, prefer="text", max_retries=2)
        return gen + "\n\n### تدريب على الاصطناعي\n" + train_res
    except Exception as e:
        try:
            from ai.model_training_agent import train_from_csv

            return gen + "\n\n### تدريب\n" + train_from_csv(path, epochs=epochs, prefer="text")
        except Exception as e2:
            return gen + f"\n\n❌ فشل التدريب: {e} / {e2}"


# ── 3) DAO simulation (NOT real crypto) ────────────────────────────────────

def _dao_state_path() -> Path:
    return DAO / "ledger.json"


def load_dao() -> Dict[str, Any]:
    p = _dao_state_path()
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "version": 1,
        "simulation_only": True,
        "real_crypto_wallets": False,
        "treasury_sim_units": 1000.0,  # وحدات افتراضية
        "owner_share_percent": 70.0,
        "reinvest_percent": 30.0,
        "transactions": [],
        "policies": [
            "لا تحويل أموال حقيقية",
            "أي نفقة سيرفر محاكاة فقط",
            "توزيع الأرباح للمؤسس = قيد دفتري",
        ],
    }


def save_dao(state: Dict[str, Any]) -> None:
    state["simulation_only"] = True
    state["real_crypto_wallets"] = False
    state["updated_at"] = _now()
    _dao_state_path().write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def dao_report() -> str:
    s = load_dao()
    lines = [
        "## 🏛️ DAO رقمي (محاكاة فقط)",
        f"- treasury_sim: **{s.get('treasury_sim_units')}** وحدة",
        f"- حصة المؤسس: {s.get('owner_share_percent')}% | إعادة استثمار: {s.get('reinvest_percent')}%",
        f"- real_crypto: **{s.get('real_crypto_wallets')}** (مرفوض في الكود)",
        "",
        "### السياسات",
    ]
    for p in s.get("policies") or []:
        lines.append(f"- {p}")
    lines.append("")
    lines.append("### آخر القيود")
    for tx in (s.get("transactions") or [])[-10:]:
        lines.append(
            f"- {tx.get('ts', '')[:19]} | {tx.get('type')} | {tx.get('amount')} | {tx.get('note')}"
        )
    if not s.get("transactions"):
        lines.append("- لا قيود بعد.")
    lines.append("")
    lines.append(
        "أوامر محاكاة: `نفقة سيرفر 10` · `وزّع أرباح 50` · `حالة dao` — "
        "كلها دفترية وليست بلوكتشين."
    )
    return "\n".join(lines)


def dao_tx(tx_type: str, amount: float, note: str = "") -> str:
    s = load_dao()
    amount = float(amount)
    bal = float(s.get("treasury_sim_units") or 0)
    if tx_type in ("expense", "server", "نفقة") and amount > bal:
        return f"❌ رصيد افتراضي غير كافٍ ({bal})"
    if tx_type in ("expense", "server", "نفقة"):
        s["treasury_sim_units"] = bal - amount
        kind = "expense_sim"
    elif tx_type in ("revenue", "دخل"):
        s["treasury_sim_units"] = bal + amount
        kind = "revenue_sim"
    elif tx_type in ("dividend", "أرباح", "distribute"):
        owner = amount * float(s.get("owner_share_percent") or 70) / 100.0
        rein = amount - owner
        if amount > bal:
            return f"❌ لا يمكن توزيع {amount} من رصيد {bal}"
        s["treasury_sim_units"] = bal - amount
        kind = "dividend_sim"
        note = f"{note} | owner={owner:.2f} reinvest_pool={rein:.2f}"
    else:
        kind = tx_type
    tx = {
        "id": f"tx_{uuid.uuid4().hex[:8]}",
        "ts": _now(),
        "type": kind,
        "amount": amount,
        "note": note or "",
        "balance_after": s["treasury_sim_units"],
    }
    s.setdefault("transactions", []).append(tx)
    save_dao(s)
    return (
        f"## 📒 قيد DAO (محاكاة)\n"
        f"- {kind}: {amount}\n"
        f"- الرصيد الافتراضي بعد العملية: **{s['treasury_sim_units']:.2f}**\n"
        f"- id: `{tx['id']}`\n\n"
        f"_لا محفظة مشفرة حقيقية ولا عقود ذكية على شبكة عامة._"
    )


def apex_status() -> str:
    cfg = _load_apex_cfg()
    n_merge = len(list(MERGERS.glob("merge_scan_*.json")))
    n_syn = len(list(SYN.glob("synth_*.csv")))
    dao = load_dao()
    lines = [
        "## 🏔️ طبقة الذروة (Apex Autonomy)",
        "",
        "### الحدود الفلسفية/التقنية",
        "- ليست AGI عامة؛ سقالات لـ Narrow autonomy داخل NSM.",
        "- الدمج = فهرسة قدرات، لا استحواذ شركات.",
        "- المعرفة الاصطناعية = بيانات تجريبية مولَّدة.",
        "- DAO = دفتر محاسبي افتراضي فقط.",
        "",
        "### الحالة",
        f"- mergers_enabled: {cfg.get('mergers_enabled')}",
        f"- synthetic_enabled: {cfg.get('synthetic_enabled')}",
        f"- dao_simulation_only: {cfg.get('dao_simulation_only')}",
        f"- real_crypto_wallets: **False (مفروض)**",
        f"- عمليات رصد دمج: {n_merge}",
        f"- مجموعات اصطناعية: {n_syn}",
        f"- خزينة افتراضية: {dao.get('treasury_sim_units')}",
        "",
        "أوامر: `حالة الذروة` · `ارصد نماذج …` · `ادمج قدرات …` · "
        "`ولّد معرفة اصطناعية …` · `درّب على الاصطناعي` · `حالة dao`",
    ]
    return "\n".join(lines)


def handle_apex_command(user_input: str) -> Optional[str]:
    text = (user_input or "").strip()
    if not text:
        return None

    if re.search(r"(حالة|status).{0,12}(الذروة|apex)", text, re.I) or text in (
        "حالة الذروة",
        "apex",
    ):
        return apex_status()

    if re.search(r"(خريطة|map).{0,10}(قدرات|capabilities)", text, re.I):
        return capability_map_report()

    m = re.search(r"(?:ارصد|اكتشف|discover).{0,10}(?:نماذج|models)\s*(.*)$", text, re.I)
    if m:
        q = (m.group(1) or "arabic nlp").strip() or "arabic nlp"
        return discover_external_models(q)

    m = re.search(r"(?:ادمج|دمج).{0,10}(?:قدرات|capability)\s+(.+)$", text, re.I)
    if m:
        return merge_capability(m.group(1).strip().strip("`"))

    m = re.search(
        r"(?:ول[ّ]?د|ولّد|generate).{0,10}(?:معرفة|بيانات).{0,10}(?:اصطناعي\w*)\s*(.*)$",
        text,
        re.I,
    )
    if m:
        topic = (m.group(1) or "جودة الخدمة").strip() or "جودة الخدمة"
        n = 100
        mn = re.search(r"(\d+)\s*(?:صف|row)", text, re.I)
        if mn:
            n = int(mn.group(1))
        return generate_synthetic_dataset(topic, n)

    if re.search(r"(?:در[ّ]?ب|train).{0,15}(?:اصطناعي|synthetic)", text, re.I):
        topic = "جودة الخدمة"
        mt = re.search(r"(?:موضوع|topic)\s*[=:]?\s*(.+)$", text, re.I)
        if mt:
            topic = mt.group(1).strip()
        return train_on_synthetic(topic=topic)

    if re.search(r"(حالة|status).{0,8}(dao|الخزينة)", text, re.I) or text.lower() in (
        "حالة dao",
        "dao",
    ):
        return dao_report()

    m = re.search(r"(?:نفقة|expense)\s*(?:سيرفر)?\s*(\d+(?:\.\d+)?)", text, re.I)
    if m:
        return dao_tx("expense", float(m.group(1)), note="server_sim")

    m = re.search(r"(?:وز[ّ]?ع|distribute).{0,8}(?:أرباح|dividend)\s*(\d+(?:\.\d+)?)", text, re.I)
    if m:
        return dao_tx("dividend", float(m.group(1)), note="owner_distribution_sim")

    m = re.search(r"(?:دخل|revenue)\s*(\d+(?:\.\d+)?)", text, re.I)
    if m and re.search(r"dao|خزين|افتراض", text, re.I):
        return dao_tx("revenue", float(m.group(1)), note="sim_revenue")

    return None

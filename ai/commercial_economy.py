"""
Commercial Economy — محرك تجاري فوق NSM (دفتر داخلي)
====================================================
قنوات: AIaaS · Marketplace · Compute Arbitrage · Synthetic Data
لا بوابة دفع ولا ضمان أرباح — تشغيل تجريبي داخل المشروع.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("CommercialEconomy")

ROOT = Path(__file__).resolve().parent.parent
ECO = ROOT / "artifacts" / "model_training" / "commercial"
MARKET = ECO / "marketplace"
LEDGER = ECO / "ledger"
for d in (ECO, MARKET, LEDGER):
    d.mkdir(parents=True, exist_ok=True)

CATALOG_PATH = MARKET / "catalog.json"
LEDGER_PATH = LEDGER / "revenue_ledger.jsonl"

SPOT_TABLE = {
    "vast_rtx3090": {"spot": 0.15, "ondemand": 0.35},
    "runpod_rtx4090": {"spot": 0.29, "ondemand": 0.44},
    "aws_g4dn": {"spot": 0.16, "ondemand": 0.53},
    "kaggle_t4": {"spot": 0.0, "ondemand": 0.0},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_catalog() -> Dict[str, Any]:
    if CATALOG_PATH.is_file():
        try:
            return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"items": [], "updated_at": _now()}


def _save_catalog(cat: Dict[str, Any]) -> None:
    cat["updated_at"] = _now()
    CATALOG_PATH.write_text(json.dumps(cat, ensure_ascii=False, indent=2), encoding="utf-8")


def append_ledger(entry: Dict[str, Any]) -> None:
    entry = {**entry, "id": entry.get("id") or f"tx_{uuid.uuid4().hex[:10]}", "at": _now()}
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_ledger(limit: int = 100) -> List[Dict[str, Any]]:
    if not LEDGER_PATH.is_file():
        return []
    rows = []
    for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows[-limit:]


def ledger_summary() -> Dict[str, Any]:
    rows = read_ledger(5000)
    by_channel: Dict[str, float] = {}
    total = 0.0
    for r in rows:
        ch = str(r.get("channel") or "other")
        amt = float(r.get("amount_usd") or 0)
        by_channel[ch] = by_channel.get(ch, 0.0) + amt
        total += amt
    return {
        "total_usd": round(total, 2),
        "by_channel": {k: round(v, 2) for k, v in by_channel.items()},
        "n_transactions": len(rows),
    }


def publish_model(name: str, domain: str, price_usd: float, license_type: str = "api", notes: str = "") -> Dict[str, Any]:
    cat = _load_catalog()
    item = {
        "id": f"mdl_{uuid.uuid4().hex[:8]}",
        "name": (name or "unnamed").strip(),
        "domain": domain or "general",
        "price_usd": float(max(0.0, price_usd)),
        "license_type": license_type,
        "notes": (notes or "")[:500],
        "status": "listed",
        "created_at": _now(),
        "sales": 0,
    }
    cat.setdefault("items", []).append(item)
    _save_catalog(cat)
    return item


def list_marketplace() -> List[Dict[str, Any]]:
    return list(_load_catalog().get("items") or [])


def sell_license(item_id: str, buyer: str = "demo_buyer") -> Dict[str, Any]:
    cat = _load_catalog()
    for it in cat.get("items") or []:
        if it.get("id") == item_id:
            it["sales"] = int(it.get("sales") or 0) + 1
            _save_catalog(cat)
            tx = {
                "channel": "marketplace",
                "amount_usd": float(it.get("price_usd") or 0),
                "ref": item_id,
                "buyer": buyer,
                "description": f"ترخيص {it.get('name')}",
            }
            append_ledger(tx)
            return {"ok": True, **tx, "item": it}
    return {"ok": False, "error": "item_not_found"}


def compute_arbitrage_quote(provider: str = "runpod_rtx4090", hours: float = 10.0, sell_rate_usd_h: Optional[float] = None) -> Dict[str, Any]:
    row = SPOT_TABLE.get(provider) or {"spot": 0.3, "ondemand": 0.5}
    buy = float(row["spot"])
    market = float(sell_rate_usd_h if sell_rate_usd_h is not None else row["ondemand"])
    hours = max(0.1, float(hours))
    cost = buy * hours
    revenue = market * hours
    margin = revenue - cost
    return {
        "provider": provider,
        "hours": hours,
        "buy_rate": buy,
        "sell_rate": market,
        "cost_usd": round(cost, 3),
        "revenue_usd": round(revenue, 3),
        "margin_usd": round(margin, 3),
        "margin_pct": round(100.0 * margin / revenue, 1) if revenue else 0.0,
        "note_ar": "تقدير دفتري — التنفيذ السحابي يحتاج مفاتيح مزوّد.",
    }


def book_arbitrage_demo(provider: str, hours: float) -> Dict[str, Any]:
    q = compute_arbitrage_quote(provider, hours)
    if q["margin_usd"] <= 0:
        return {"ok": False, "quote": q, "reason": "لا هامش موجب"}
    append_ledger({
        "channel": "compute_arbitrage",
        "amount_usd": q["margin_usd"],
        "ref": provider,
        "description": f"هامش تقديري {hours}h على {provider}",
        "meta": q,
    })
    return {"ok": True, "quote": q}


def price_synthetic_batch(n_samples: int, quality: str = "standard", domain: str = "general") -> Dict[str, Any]:
    base = 0.002
    mult = {"standard": 1.0, "curated": 1.8, "domain_expert": 3.0}.get(quality, 1.0)
    if domain in ("medical", "طب", "finance", "مالية"):
        mult *= 1.5
    n = max(1, int(n_samples))
    return {
        "n_samples": n,
        "quality": quality,
        "domain": domain,
        "unit_usd": round(base * mult, 5),
        "price_usd": round(n * base * mult, 2),
    }


def sell_synthetic_demo(n_samples: int = 1000, quality: str = "curated", domain: str = "general") -> Dict[str, Any]:
    pricing = price_synthetic_batch(n_samples, quality, domain)
    batch_path = ""
    try:
        from ai.synthetic_data_factory import run_factory
        rep = run_factory(n_texts=min(200, max(20, n_samples // 10)))
        batch_path = rep.output_path
    except Exception as e:
        logger.info("synthetic skip: %s", e)
    append_ledger({
        "channel": "synthetic_data",
        "amount_usd": pricing["price_usd"],
        "ref": batch_path or f"n={n_samples}",
        "description": f"بيع بيانات اصطناعية {domain}/{quality}",
        "meta": pricing,
    })
    return {"ok": True, "pricing": pricing, "batch_path": batch_path}


def aiaas_revenue_snapshot() -> Dict[str, Any]:
    try:
        from ai.aiaas_platform import load_tenants_index, PLANS
        idx = load_tenants_index().get("tenants") or {}
        mrr = sum(float((PLANS.get(rec.get("plan") or "free") or {}).get("price_usd_month") or 0) for rec in idx.values())
        return {"tenants": len(idx), "estimated_mrr_usd": round(mrr, 2), "plans": list(PLANS.keys())}
    except Exception as e:
        return {"tenants": 0, "estimated_mrr_usd": 0.0, "error": str(e)}


def dashboard() -> Dict[str, Any]:
    return {
        "ledger": ledger_summary(),
        "marketplace_items": len(list_marketplace()),
        "aiaas": aiaas_revenue_snapshot(),
        "spot_table": SPOT_TABLE,
        "generated_at": _now(),
    }


def handle_economic_command(user_input: str) -> Optional[str]:
    import re
    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(r"(محرك\s*اقتصاد|economic|سوق\s*نماذج|marketplace|arbitrage|بيع\s*بيانات|ايرادات|أرباح\s*دفتر|لوح[ةه]\s*الاقتصاد)", text, re.I):
        return None
    if re.search(r"(لوح[ةه]|dashboard|ايرادات|ملخص)", text, re.I):
        return "## 💰 لوحة المحرك الاقتصادي\n\n```json\n" + json.dumps(dashboard(), ensure_ascii=False, indent=2)[:4000] + "\n```"
    if re.search(r"(انشر\s*نموذج|publish\s*model)", text, re.I):
        item = publish_model("NSM-Demo-Classifier", "tabular", 29.0, notes="تجريبي")
        return f"## 📦 نُشر في السوق\n```json\n{json.dumps(item, ensure_ascii=False, indent=2)}\n```"
    if re.search(r"(arbitrage|مضارب[ةه]\s*حوسب|هامش\s*spot)", text, re.I):
        q = compute_arbitrage_quote("runpod_rtx4090", 10)
        return f"## 📉 عرض Arbitrage\n```json\n{json.dumps(q, ensure_ascii=False, indent=2)}\n```"
    if re.search(r"(بيع\s*بيانات|synthetic\s*sell)", text, re.I):
        r = sell_synthetic_demo(500, "curated", "general")
        return f"## 🧪 بيع بيانات (دفتري)\n```json\n{json.dumps(r, ensure_ascii=False, indent=2)[:2000]}\n```"
    return "## 💰 المحرك الاقتصادي — أوامر\n- `لوحة الاقتصاد`\n- `انشر نموذج`\n- `هامش spot`\n- `بيع بيانات`"

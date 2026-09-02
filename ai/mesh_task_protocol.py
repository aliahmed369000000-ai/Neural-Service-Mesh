# -*- coding: utf-8 -*-
"""
NSM Mesh Task Protocol — بروتوكول المهام الموزّعة فوق Living Mesh
================================================================
أنواع المهام:
  1) AI / Deep Learning:
     - submodel_train   : تدريب طبقة/جزء من النموذج ثم إرجاع أوزان جزئية
     - inference        : توليد رد (نصي أو وصفي) موقّع من عقدة قادرة
     - model_eval       : تقييم دقة/خسارة على شريحة بيانات اختبار
  2) Scientific / Heavy compute:
     - map_reduce       : Map على شريحة ثم Reduce عند الطالب
     - sim_chunk        : شريحة محاكاة (معادلات تفاضلية مبسّطة)
     - keyspace_scan    : مسح نطاق مفاتيح/هششر (لاختبار أمان شرعي)
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("MeshTaskProtocol")

# أنواع الرسائل
KIND_SUBMODEL_TRAIN = "submodel_train"
KIND_SUBMODEL_RESULT = "submodel_train_result"
KIND_INFERENCE = "inference_request"
KIND_INFERENCE_RESULT = "inference_result"
KIND_MODEL_EVAL = "model_eval"
KIND_MODEL_EVAL_RESULT = "model_eval_result"
KIND_MAP = "map_reduce_map"
KIND_MAP_RESULT = "map_reduce_map_result"
KIND_SIM = "sim_chunk"
KIND_SIM_RESULT = "sim_chunk_result"
KIND_KEYSPACE = "keyspace_scan"
KIND_KEYSPACE_RESULT = "keyspace_scan_result"
KIND_SUMMARIZE = "summarize_chunk"
KIND_SUMMARIZE_RESULT = "summarize_chunk_result"
KIND_SEARCH = "search_chunk"
KIND_SEARCH_RESULT = "search_chunk_result"

# إدارة دورة حياة المهمة (v1.1+)
KIND_TASK_ACK = "task_ack"
KIND_TASK_CANCEL = "task_cancel"
KIND_TASK_STATUS = "task_status"
KIND_TASK_STATUS_RESULT = "task_status_result"

ALL_TASK_KINDS = {
    KIND_SUBMODEL_TRAIN, KIND_SUBMODEL_RESULT,
    KIND_INFERENCE, KIND_INFERENCE_RESULT,
    KIND_MODEL_EVAL, KIND_MODEL_EVAL_RESULT,
    KIND_MAP, KIND_MAP_RESULT,
    KIND_SIM, KIND_SIM_RESULT,
    KIND_KEYSPACE, KIND_KEYSPACE_RESULT,
    KIND_SUMMARIZE, KIND_SUMMARIZE_RESULT,
    KIND_SEARCH, KIND_SEARCH_RESULT,
    KIND_TASK_ACK, KIND_TASK_CANCEL,
    KIND_TASK_STATUS, KIND_TASK_STATUS_RESULT,
}

# حالات دورة حياة المهمة
TASK_STATUS_PENDING = "pending"
TASK_STATUS_ACKED = "acked"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"
TASK_STATUS_CANCELLED = "cancelled"
TASK_STATUS_TIMEOUT = "timeout"
TASK_STATUS_DUPLICATE = "duplicate_rejected"


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# 1) Sub-model training — تدريب جزء من الطبقات محلياً
# ---------------------------------------------------------------------------
def execute_submodel_train(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    يحاكي تدريب طبقة فرعية:
    task = {
      "layer_name": "encoder.block_3",
      "layer_index": 3,
      "steps": 10,
      "lr": 1e-3,
      "seed_weights": optional list/dict,
      "data_shard": optional list of {x,y}
    }
    يُرجع أوزاناً جزئية + loss نهائي (بدون الاعتماد على GPU إلزامياً).
    """
    t0 = time.time()
    layer_name = task.get("layer_name") or f"layer_{task.get('layer_index', 0)}"
    steps = max(1, min(int(task.get("steps") or 5), 100))
    lr = _safe_float(task.get("lr"), 1e-3)
    seed = task.get("seed_weights")
    shard = task.get("data_shard") or []

    # أوزان أولية بسيطة
    if isinstance(seed, dict) and "w" in seed:
        w = [float(v) for v in seed["w"][:64]]
    elif isinstance(seed, list):
        w = [float(v) for v in seed[:64]]
    else:
        h = hashlib.sha256(layer_name.encode()).digest()
        w = [((b / 255.0) * 2 - 1) for b in h[:16]]

    loss = 1.0
    for step in range(steps):
        # تدرج تقريبي على شريحة أو هدف ثابت
        if shard:
            err = 0.0
            for sample in shard[:32]:
                x = sample.get("x", 0.5) if isinstance(sample, dict) else 0.5
                y = sample.get("y", 0.0) if isinstance(sample, dict) else 0.0
                pred = sum(w) / max(len(w), 1) * _safe_float(x, 0.5)
                err += (pred - _safe_float(y)) ** 2
            loss = err / max(len(shard[:32]), 1)
        else:
            target = 0.1
            pred = sum(w) / max(len(w), 1)
            loss = (pred - target) ** 2
        grad = 2.0 * math.sqrt(max(loss, 1e-12)) / max(len(w), 1)
        w = [wi - lr * grad for wi in w]
        lr *= 0.99

    return {
        "ok": True,
        "layer_name": layer_name,
        "layer_index": task.get("layer_index"),
        "steps": steps,
        "final_loss": round(loss, 6),
        "partial_weights": [round(v, 6) for v in w],
        "elapsed_ms": round((time.time() - t0) * 1000, 2),
        "task_id": task.get("task_id"),
    }


def merge_submodel_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """دمج نتائج تدريب طبقات فرعية من عدة عقد."""
    layers = {}
    total_loss = 0.0
    n = 0
    for r in results:
        if not r or not r.get("ok"):
            continue
        name = r.get("layer_name") or f"layer_{r.get('layer_index')}"
        layers[name] = {
            "weights": r.get("partial_weights"),
            "loss": r.get("final_loss"),
            "steps": r.get("steps"),
        }
        if r.get("final_loss") is not None:
            total_loss += float(r["final_loss"])
            n += 1
    return {
        "ok": True,
        "merged_layers": layers,
        "mean_loss": round(total_loss / n, 6) if n else None,
        "layers_count": len(layers),
    }


# ---------------------------------------------------------------------------
# 2) Inference — توليد محتوى موقّع (بدون إجبار نماذج ثقيلة)
# ---------------------------------------------------------------------------
def execute_inference(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    task = {
      "prompt": str,
      "modality": "text" | "image_desc",
      "model_hint": "llama" | "sd" | "local",
      "max_tokens": int
    }
    يُنتج رداً حتمياً قابلاً للتوقيع (يمكن لاحقاً ربطه بنموذج حقيقي).
    """
    t0 = time.time()
    prompt = (task.get("prompt") or "").strip()
    modality = (task.get("modality") or "text").lower()
    model_hint = (task.get("model_hint") or "local").lower()
    max_tokens = max(16, min(int(task.get("max_tokens") or 128), 512))

    if not prompt:
        return {"ok": False, "error": "empty_prompt", "task_id": task.get("task_id")}

    # مولّد محلي حتمي يعتمد على الهاش (بديل آمن حتى ربط Llama/SD)
    digest = hashlib.sha256(f"{model_hint}:{modality}:{prompt}".encode()).hexdigest()
    if modality.startswith("image"):
        output = {
            "type": "image_descriptor",
            "prompt": prompt[:500],
            "seed": int(digest[:8], 16),
            "note": f"synthetic SD-style descriptor via {model_hint}",
            "palette": [digest[i:i+6] for i in range(0, 18, 6)],
        }
        text_out = json.dumps(output, ensure_ascii=False)
    else:
        words = prompt.split()
        cont = []
        for i in range(min(max_tokens // 4, 40)):
            cont.append(digest[(i * 2) % 60:(i * 2) % 60 + 4])
        text_out = (
            f"[mesh-inference/{model_hint}] {prompt[:200]} "
            f"→ {' '.join(cont)}"
        )[: max_tokens * 4]

    return {
        "ok": True,
        "modality": modality,
        "model_hint": model_hint,
        "prompt_preview": prompt[:120],
        "output": text_out,
        "elapsed_ms": round((time.time() - t0) * 1000, 2),
        "task_id": task.get("task_id"),
    }


# ---------------------------------------------------------------------------
# 3) Model Evaluation
# ---------------------------------------------------------------------------
def execute_model_eval(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    task = {
      "samples": [{"x":..., "y":...}, ...],
      "metric": "accuracy" | "loss" | "both",
      "weights": optional list
    }
    """
    t0 = time.time()
    samples = task.get("samples") or []
    metric = (task.get("metric") or "both").lower()
    weights = task.get("weights")
    if isinstance(weights, list) and weights:
        w_mean = sum(float(v) for v in weights[:64]) / max(len(weights[:64]), 1)
    else:
        w_mean = 0.5

    if not samples:
        return {"ok": False, "error": "no_samples", "task_id": task.get("task_id")}

    correct = 0
    loss_sum = 0.0
    for s in samples[:1000]:
        if not isinstance(s, dict):
            continue
        x = _safe_float(s.get("x"), 0.5)
        y = _safe_float(s.get("y"), 0.0)
        pred = w_mean * x
        loss_sum += (pred - y) ** 2
        # عتبة تصنيف بسيطة
        if (pred >= 0.5 and y >= 0.5) or (pred < 0.5 and y < 0.5):
            correct += 1

    n = max(len(samples[:1000]), 1)
    acc = correct / n
    loss = loss_sum / n
    out: Dict[str, Any] = {
        "ok": True,
        "n_samples": n,
        "elapsed_ms": round((time.time() - t0) * 1000, 2),
        "task_id": task.get("task_id"),
    }
    if metric in ("accuracy", "both"):
        out["accuracy"] = round(acc, 6)
    if metric in ("loss", "both"):
        out["loss"] = round(loss, 6)
    return out


def merge_eval_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_n = 0
    weighted_acc = 0.0
    weighted_loss = 0.0
    has_acc = has_loss = False
    for r in results:
        if not r or not r.get("ok"):
            continue
        n = int(r.get("n_samples") or 0)
        if n <= 0:
            continue
        total_n += n
        if "accuracy" in r:
            weighted_acc += float(r["accuracy"]) * n
            has_acc = True
        if "loss" in r:
            weighted_loss += float(r["loss"]) * n
            has_loss = True
    out: Dict[str, Any] = {"ok": total_n > 0, "n_samples": total_n}
    if has_acc and total_n:
        out["accuracy"] = round(weighted_acc / total_n, 6)
    if has_loss and total_n:
        out["loss"] = round(weighted_loss / total_n, 6)
    return out


# ---------------------------------------------------------------------------
# 4) Map-Reduce
# ---------------------------------------------------------------------------
def execute_map(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    task = {
      "chunk_id": str,
      "lines": [str, ...] | "text": str,
      "op": "wordcount" | "sum" | "filter",
      "filter_contains": optional str
    }
    """
    t0 = time.time()
    op = (task.get("op") or "wordcount").lower()
    lines = task.get("lines")
    if lines is None and task.get("text"):
        lines = str(task["text"]).splitlines()
    lines = lines or []
    if not isinstance(lines, list):
        lines = [str(lines)]

    if op == "sum":
        total = 0.0
        for ln in lines:
            for tok in str(ln).replace(",", " ").split():
                try:
                    total += float(tok)
                except Exception:
                    pass
        partial = {"sum": total, "count": len(lines)}
    elif op == "filter":
        needle = str(task.get("filter_contains") or "")
        kept = [str(ln) for ln in lines if needle in str(ln)]
        partial = {"kept": kept[:500], "kept_count": len(kept), "in_count": len(lines)}
    else:  # wordcount
        counts: Dict[str, int] = {}
        for ln in lines:
            for w in str(ln).split():
                w = w.strip().lower()
                if not w:
                    continue
                counts[w] = counts.get(w, 0) + 1
        # أعلى 200 كلمة
        top = dict(sorted(counts.items(), key=lambda kv: -kv[1])[:200])
        partial = {"counts": top, "lines": len(lines)}

    return {
        "ok": True,
        "chunk_id": task.get("chunk_id"),
        "op": op,
        "partial": partial,
        "elapsed_ms": round((time.time() - t0) * 1000, 2),
        "task_id": task.get("task_id"),
    }


def reduce_map_results(op: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    op = (op or "wordcount").lower()
    if op == "sum":
        total = 0.0
        count = 0
        for r in results:
            if not r or not r.get("ok"):
                continue
            p = r.get("partial") or {}
            total += _safe_float(p.get("sum"))
            count += int(p.get("count") or 0)
        return {"ok": True, "op": op, "sum": total, "count": count}
    if op == "filter":
        kept_all = []
        in_count = 0
        for r in results:
            if not r or not r.get("ok"):
                continue
            p = r.get("partial") or {}
            kept_all.extend(p.get("kept") or [])
            in_count += int(p.get("in_count") or 0)
        return {
            "ok": True,
            "op": op,
            "kept_count": len(kept_all),
            "in_count": in_count,
            "kept_preview": kept_all[:100],
        }
    # wordcount merge
    merged: Dict[str, int] = {}
    lines = 0
    for r in results:
        if not r or not r.get("ok"):
            continue
        p = r.get("partial") or {}
        lines += int(p.get("lines") or 0)
        for w, c in (p.get("counts") or {}).items():
            merged[w] = merged.get(w, 0) + int(c)
    top = dict(sorted(merged.items(), key=lambda kv: -kv[1])[:300])
    return {"ok": True, "op": "wordcount", "counts": top, "lines": lines, "unique": len(merged)}


# ---------------------------------------------------------------------------
# 5) Simulation chunk — معادلات تفاضلية مبسّطة (Euler)
# ---------------------------------------------------------------------------
def execute_sim_chunk(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    task = {
      "x0": float, "y0": float,
      "t0": float, "t1": float, "dt": float,
      "params": {"k": 0.1, "mode": "decay"|"osc"|"logistic"}
    }
    """
    t0w = time.time()
    x = _safe_float(task.get("x0"), 1.0)
    y = _safe_float(task.get("y0"), 0.0)
    t = _safe_float(task.get("t0"), 0.0)
    t1 = _safe_float(task.get("t1"), 1.0)
    dt = max(1e-4, min(_safe_float(task.get("dt"), 0.01), 0.5))
    params = task.get("params") or {}
    k = _safe_float(params.get("k"), 0.1)
    mode = (params.get("mode") or "decay").lower()

    steps = 0
    max_steps = 5000
    trajectory = []
    while t < t1 and steps < max_steps:
        if mode == "osc":
            # x' = y, y' = -k*x
            dx, dy = y, -k * x
        elif mode == "logistic":
            # x' = k*x*(1-x)
            dx, dy = k * x * (1.0 - x), 0.0
        else:
            # decay: x' = -k*x
            dx, dy = -k * x, 0.0
        x += dx * dt
        y += dy * dt
        t += dt
        steps += 1
        if steps % max(1, steps // 20 or 1) == 0 or t >= t1:
            trajectory.append({"t": round(t, 6), "x": round(x, 6), "y": round(y, 6)})

    return {
        "ok": True,
        "final": {"t": round(t, 6), "x": round(x, 6), "y": round(y, 6)},
        "steps": steps,
        "mode": mode,
        "trajectory_sample": trajectory[:30],
        "elapsed_ms": round((time.time() - t0w) * 1000, 2),
        "task_id": task.get("task_id"),
        "chunk_id": task.get("chunk_id"),
    }


# ---------------------------------------------------------------------------
# 6) Keyspace scan — اختبار أمان شرعي (نطاق محدود)
# ---------------------------------------------------------------------------
def execute_keyspace_scan(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    task = {
      "start": int, "end": int,          # نطاق شامل-جزئي [start, end)
      "target_hash": str,                # sha256 hex للهدف
      "prefix": str optional,            # بادئة تُهشّ مع العدد
      "max_checks": int                  # سقف أمان محلي
    }
    يبحث عن عدد n في النطاق بحيث sha256(prefix+str(n)) == target_hash
    أو يطابق بادئة الهاش إن target_hash قصير.
    """
    t0 = time.time()
    start = int(task.get("start") or 0)
    end = int(task.get("end") or start)
    if end < start:
        start, end = end, start
    # سقف صارم لمنع إساءة الاستخدام في عقدة واحدة
    max_checks = max(1, min(int(task.get("max_checks") or 50000), 200000))
    end = min(end, start + max_checks)
    target = (task.get("target_hash") or "").strip().lower()
    prefix = str(task.get("prefix") or "")
    found = None
    checked = 0
    for n in range(start, end):
        checked += 1
        h = hashlib.sha256(f"{prefix}{n}".encode()).hexdigest()
        if not target:
            continue
        if h == target or (len(target) < 64 and h.startswith(target)):
            found = {"n": n, "hash": h}
            break

    return {
        "ok": True,
        "found": found,
        "checked": checked,
        "range": [start, end],
        "elapsed_ms": round((time.time() - t0) * 1000, 2),
        "task_id": task.get("task_id"),
        "chunk_id": task.get("chunk_id"),
    }



def execute_summarize_chunk(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    تلخيص قطعة مصدر محلية (آمن — بلا شبكة).
    task: {source_id, text, query?, max_chars?}
    يُرجع ملخصاً حتمياً + hash المصدر لإثبات provenance.
    """
    t0 = time.time()
    text = (task.get("text") or task.get("content") or "").strip()
    source_id = task.get("source_id") or task.get("chunk_id") or "src_unknown"
    query = (task.get("query") or "").strip()
    max_chars = max(40, min(int(task.get("max_chars") or 240), 2000))
    if not text:
        return {"ok": False, "error": "empty_source", "task_id": task.get("task_id"), "source_id": source_id}
    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    # ملخص حتمي: جمل/أجزاء مرتبطة بالاستعلام إن وُجد
    parts = [p.strip() for p in text.replace("\n", " ").split(".") if p.strip()]
    if query:
        q = query.lower()
        ranked = sorted(parts, key=lambda s: (0 if q in s.lower() else 1, -len(s)))
        picked = ranked[:3] if ranked else [text[:max_chars]]
    else:
        picked = parts[:3] if parts else [text[:max_chars]]
    summary = ". ".join(picked)
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3] + "..."
    return {
        "ok": True,
        "source_id": source_id,
        "source_hash": source_hash,
        "query": query[:200] if query else None,
        "summary": summary,
        "output": summary,  # للتوافق مع semantic digest
        "chars_in": len(text),
        "chars_out": len(summary),
        "elapsed_ms": round((time.time() - t0) * 1000, 2),
        "task_id": task.get("task_id"),
    }



def execute_search_chunk(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    بحث دلالي خفيف داخل مصدر/مصادر محلية مُمرَّرة (بلا شبكة).
    task: {query, documents:[{source_id,text}], top_k?}
    """
    t0 = time.time()
    query = (task.get("query") or "").strip().lower()
    docs = task.get("documents") or task.get("sources") or []
    top_k = max(1, min(int(task.get("top_k") or 5), 20))
    if not query:
        return {"ok": False, "error": "empty_query", "task_id": task.get("task_id")}
    if not docs:
        return {"ok": False, "error": "empty_documents", "task_id": task.get("task_id")}
    tokens = [tok for tok in query.replace("،", " ").split() if tok]
    hits = []
    for d in docs:
        text = (d.get("text") or d.get("content") or "").strip()
        if not text:
            continue
        sid = d.get("source_id") or d.get("id") or "src"
        low = text.lower()
        score = 0.0
        for tok in tokens:
            if tok in low:
                score += 1.0 + low.count(tok) * 0.1
        # مكافأة ظهور العبارة كاملة
        if query in low:
            score += 2.0
        if score <= 0:
            continue
        # مقتطف حول أول تطابق
        pos = low.find(tokens[0]) if tokens else 0
        if pos < 0:
            pos = 0
        start = max(0, pos - 40)
        snippet = text[start: start + 180].strip()
        if start > 0:
            snippet = "…" + snippet
        if start + 180 < len(text):
            snippet = snippet + "…"
        sh = hashlib.sha256(text.encode("utf-8")).hexdigest()
        hits.append({
            "source_id": sid,
            "source_hash": sh,
            "score": round(score, 3),
            "snippet": snippet,
            "chars": len(text),
        })
    hits.sort(key=lambda h: h["score"], reverse=True)
    hits = hits[:top_k]
    return {
        "ok": True,
        "query": task.get("query"),
        "hits": hits,
        "hit_count": len(hits),
        "output": json.dumps([h.get("source_id") for h in hits], ensure_ascii=False),
        "elapsed_ms": round((time.time() - t0) * 1000, 2),
        "task_id": task.get("task_id"),
    }


def dispatch_task(kind: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """موجّه مركزي لتنفيذ مهمة حسب النوع."""
    data = data or {}
    if kind == KIND_SUBMODEL_TRAIN:
        return execute_submodel_train(data)
    if kind == KIND_INFERENCE:
        return execute_inference(data)
    if kind == KIND_MODEL_EVAL:
        return execute_model_eval(data)
    if kind == KIND_MAP:
        return execute_map(data)
    if kind == KIND_SIM:
        return execute_sim_chunk(data)
    if kind == KIND_KEYSPACE:
        return execute_keyspace_scan(data)
    if kind == KIND_SUMMARIZE:
        return execute_summarize_chunk(data)
    if kind == KIND_SEARCH:
        return execute_search_chunk(data)
    return None


def result_kind_for(request_kind: str) -> str:
    return {
        KIND_SUBMODEL_TRAIN: KIND_SUBMODEL_RESULT,
        KIND_INFERENCE: KIND_INFERENCE_RESULT,
        KIND_MODEL_EVAL: KIND_MODEL_EVAL_RESULT,
        KIND_MAP: KIND_MAP_RESULT,
        KIND_SIM: KIND_SIM_RESULT,
        KIND_KEYSPACE: KIND_KEYSPACE_RESULT,
        KIND_SUMMARIZE: KIND_SUMMARIZE_RESULT,
        KIND_SEARCH: KIND_SEARCH_RESULT,
        KIND_TASK_STATUS: KIND_TASK_STATUS_RESULT,
    }.get(request_kind, request_kind + "_result")

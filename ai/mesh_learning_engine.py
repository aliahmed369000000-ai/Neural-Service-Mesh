# -*- coding: utf-8 -*-
"""
محرك تعلّم الشبكة — يستخلص معرفة ومهارات من المهام الموثّقة.
لا يعتمد على torch؛ يعمل على أي عقدة خفيفة (Colab/Kaggle/VPS).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


LEARN_STATE_KEY = "mesh_learning_v1"
DEFAULT_SKILLS = {
    "map_reduce_map": 0.5,
    "sim_chunk": 0.5,
    "submodel_train": 0.5,
    "model_eval": 0.5,
    "keyspace_scan": 0.5,
    "summarize_chunk": 0.5,
    "search_chunk": 0.5,
    "web_fetch": 0.5,
    "classic_showcase": 0.5,
    "predict": 0.5,
    "temporal_forecast": 0.5,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


class MeshLearningEngine:
    """يحافظ على مهارات العقدة + بنك أوزان + مقتطفات معرفة من المهام الناجحة."""

    def __init__(self, node):
        self.node = node
        self.path = Path(node.data_dir) / "mesh_learning.json"
        self.state = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
        # توافق مع network_state إن وُجد
        try:
            st = self.node._load_state()
            if isinstance(st.get(LEARN_STATE_KEY), dict):
                return st[LEARN_STATE_KEY]
        except Exception:
            pass
        return {
            "version": 1,
            "skills": dict(DEFAULT_SKILLS),
            "skill_counts": {},
            "knowledge": [],
            "weight_bank": {},
            "stats": {
                "lessons": 0,
                "successes": 0,
                "failures": 0,
                "last_learn_at": None,
            },
            "history_digest": [],
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            st = self.node._load_state()
            st[LEARN_STATE_KEY] = {
                "skills": self.state.get("skills"),
                "skill_counts": self.state.get("skill_counts"),
                "stats": self.state.get("stats"),
                "knowledge_count": len(self.state.get("knowledge") or []),
                "updated_at": _now(),
            }
            self.node._save_state(st)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def learn_from_task(
        self,
        kind: str,
        result: Dict[str, Any],
        receipt: Optional[Dict[str, Any]] = None,
        task_id: str = None,
    ) -> Dict[str, Any]:
        """درس واحد من نتيجة مهمة موثّقة."""
        kind = (kind or "unknown").strip()
        ok = bool((result or {}).get("ok", True))
        skills = self.state.setdefault("skills", dict(DEFAULT_SKILLS))
        counts = self.state.setdefault("skill_counts", {})
        stats = self.state.setdefault("stats", {})
        counts[kind] = int(counts.get(kind) or 0) + 1
        prev = float(skills.get(kind, 0.5))

        # تحديث مهارة (متوسط أسي)
        target = 1.0 if ok else 0.0
        # مكافأة إضافية لجودة quantifiable
        bonus = 0.0
        if ok:
            if kind in ("model_eval",) and result.get("accuracy") is not None:
                bonus = 0.15 * float(result.get("accuracy") or 0)
            if kind in ("classic_showcase", "map_reduce_map") and result.get("abs_error") is not None:
                # خطأ أصغر → مهارة أعلى
                err = float(result.get("abs_error") or 1.0)
                bonus = 0.1 * _clip01(1.0 - min(err * 1000, 1.0))
            if kind == "submodel_train" and result.get("final_loss") is not None:
                loss = float(result.get("final_loss") or 1.0)
                bonus = 0.1 * _clip01(1.0 / (1.0 + loss * 10))

        new_skill = _clip01(prev * 0.85 + (target + bonus) * 0.15)
        skills[kind] = round(new_skill, 6)

        stats["lessons"] = int(stats.get("lessons") or 0) + 1
        if ok:
            stats["successes"] = int(stats.get("successes") or 0) + 1
        else:
            stats["failures"] = int(stats.get("failures") or 0) + 1
        stats["last_learn_at"] = _now()

        knowledge_item = self._extract_knowledge(kind, result, task_id)
        if knowledge_item:
            kn = self.state.setdefault("knowledge", [])
            kn.append(knowledge_item)
            if len(kn) > 200:
                del kn[:-200]

        # بنك أوزان من التدريب الجزئي
        if ok and kind == "submodel_train":
            weights = result.get("partial_weights")
            if isinstance(weights, list) and weights:
                layer = result.get("layer_name") or "default"
                self.state.setdefault("weight_bank", {})[layer] = {
                    "weights": weights[:256],
                    "final_loss": result.get("final_loss"),
                    "updated_at": _now(),
                    "task_id": task_id,
                }

        # ذاكرة موحّدة (نصية) إن توفرت
        self._store_memory(kind, result, task_id, ok)

        # سمعة محلية خفيفة
        try:
            delta = 1 if ok else -1
            self.node.update_reputation(self.node.node_id, delta=delta, reason=f"learn:{kind}")
        except Exception:
            pass

        # مشاركة خبرة مختصرة عبر gossip إن أمكن
        try:
            self.node.sync_experience(
                "mesh_learning_lesson",
                {
                    "kind": kind,
                    "ok": ok,
                    "skill": skills[kind],
                    "task_id": task_id,
                    "headline": (knowledge_item or {}).get("headline"),
                },
            )
        except Exception:
            pass

        digest = hashlib.sha256(
            json.dumps({"kind": kind, "task_id": task_id, "ok": ok}, sort_keys=True).encode()
        ).hexdigest()[:16]
        hist = self.state.setdefault("history_digest", [])
        hist.append({"ts": _now(), "d": digest, "kind": kind, "ok": ok})
        if len(hist) > 100:
            del hist[:-100]

        self._save()
        return {
            "ok": True,
            "kind": kind,
            "skill_before": prev,
            "skill_after": skills[kind],
            "lessons": stats["lessons"],
            "knowledge_added": bool(knowledge_item),
        }

    def _extract_knowledge(self, kind: str, result: Dict[str, Any], task_id: str) -> Optional[Dict[str, Any]]:
        if not result or not result.get("ok", True):
            return None
        item: Dict[str, Any] = {
            "ts": _now(),
            "kind": kind,
            "task_id": task_id,
        }
        if kind == "classic_showcase" or result.get("pi_estimate") is not None:
            item["headline"] = result.get("headline") or f"π≈{result.get('pi_estimate')}"
            item["pi_estimate"] = result.get("pi_estimate")
            item["abs_error"] = result.get("abs_error")
        elif kind == "map_reduce_map":
            partial = result.get("partial") or {}
            if partial.get("sum") is not None:
                # قد تكون حدود Leibniz
                s = float(partial["sum"])
                item["headline"] = f"sum={s:.8f} → π≈{4*s:.8f}"
                item["sum"] = s
                item["pi_from_sum"] = 4 * s
            elif partial.get("counts"):
                top = sorted(partial["counts"].items(), key=lambda kv: -kv[1])[:8]
                item["headline"] = "wordcount:" + ",".join(f"{w}:{c}" for w, c in top[:5])
                item["top_tokens"] = top
        elif kind == "web_fetch":
            item["headline"] = f"fetched {result.get('host') or result.get('url')}"
            item["url"] = result.get("url")
            item["content_hash"] = result.get("content_hash")
            text = (result.get("text") or result.get("output") or "")[:280]
            item["snippet"] = text
        elif kind == "summarize_chunk":
            item["headline"] = (result.get("summary") or "")[:160]
            item["source_hash"] = result.get("source_hash")
        elif kind == "model_eval":
            item["headline"] = f"acc={result.get('accuracy')} loss={result.get('loss')}"
            item["accuracy"] = result.get("accuracy")
            item["loss"] = result.get("loss")
        elif kind == "submodel_train":
            item["headline"] = f"train loss={result.get('final_loss')} steps={result.get('steps')}"
        elif kind == "sim_chunk":
            item["headline"] = f"sim steps={result.get('steps')} final={result.get('final')}"
        else:
            item["headline"] = f"{kind} ok"
        return item

    def _store_memory(self, kind: str, result: Dict[str, Any], task_id: str, ok: bool) -> None:
        try:
            mem = getattr(self.node, "memory", None)
            if mem is None:
                return
            text = json.dumps(
                {
                    "type": "mesh_task_lesson",
                    "kind": kind,
                    "task_id": task_id,
                    "ok": ok,
                    "summary": (result or {}).get("headline")
                    or (result or {}).get("summary")
                    or str((result or {}).get("partial") or "")[:200],
                },
                ensure_ascii=False,
            )
            mem.store_experience(
                {
                    "type": "mesh_learning",
                    "kind": kind,
                    "task_id": task_id,
                    "ok": ok,
                    "text": text,
                    "ts": _now(),
                }
            )
        except Exception:
            pass

    def consolidate(self, task_log: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """يمر على سجل المهام ويعلّم ما لم يُستوعب بعد."""
        learned = 0
        seen = {h.get("d") for h in (self.state.get("history_digest") or [])}
        for entry in task_log or []:
            kind = entry.get("kind") or ""
            result = entry.get("result") or {}
            task_id = entry.get("task_id")
            digest = hashlib.sha256(
                json.dumps({"kind": kind, "task_id": task_id, "ok": entry.get("ok")}, sort_keys=True).encode()
            ).hexdigest()[:16]
            if digest in seen:
                continue
            if entry.get("mode") == "remote" and not result:
                continue
            self.learn_from_task(kind, result if result else {"ok": entry.get("ok")}, entry.get("receipt"), task_id)
            learned += 1
        return {"ok": True, "learned_new": learned, "skills": self.skills_snapshot()}

    def skills_snapshot(self) -> Dict[str, Any]:
        skills = self.state.get("skills") or {}
        ranked = sorted(skills.items(), key=lambda kv: -kv[1])
        return {
            "skills": skills,
            "top_skills": ranked[:8],
            "counts": self.state.get("skill_counts") or {},
            "stats": self.state.get("stats") or {},
            "knowledge_count": len(self.state.get("knowledge") or []),
            "weight_layers": list((self.state.get("weight_bank") or {}).keys()),
            "power_score": round(self.power_score(), 4),
        }

    def power_score(self) -> float:
        """مؤشر قوة تقريبي: متوسط المهارات × نجاح نسبي × تنوع."""
        skills = list((self.state.get("skills") or {}).values()) or [0.5]
        avg = sum(skills) / len(skills)
        stats = self.state.get("stats") or {}
        lessons = max(1, int(stats.get("lessons") or 0))
        succ = int(stats.get("successes") or 0)
        success_rate = succ / lessons if lessons else 0.5
        diversity = min(1.0, len(self.state.get("skill_counts") or {}) / 8.0)
        knowledge = min(1.0, len(self.state.get("knowledge") or {}) / 50.0)
        return _clip01(0.4 * avg + 0.3 * success_rate + 0.15 * diversity + 0.15 * knowledge)

    def best_weights(self, layer: str = None) -> Optional[List[float]]:
        bank = self.state.get("weight_bank") or {}
        if layer and layer in bank:
            return bank[layer].get("weights")
        if not bank:
            return None
        # أقل خسارة
        best = min(bank.values(), key=lambda x: float(x.get("final_loss") or 1e9))
        return best.get("weights")

    def recent_knowledge(self, limit: int = 10) -> List[Dict[str, Any]]:
        kn = self.state.get("knowledge") or []
        return kn[-limit:]

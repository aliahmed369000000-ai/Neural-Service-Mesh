"""
ai/agent_learning_feedback.py
=============================
🆕 نظام التغذية الراجعة والتعلم (Learning & Feedback Loop).

يجمع ملاحظات المستخدم ويحسّن سلوك الوكلاء تلقائيًا:
  • تسجيل feedback (مفيد/غير مفيد)
  • تحليل الأنماط (أي أفعال تُقدَّر، أي تُرفض)
  • تكييف system prompt بناءً على التاريخ
  • تصنيف جودة الردود

الاستخدام:
    from ai.agent_learning_feedback import FeedbackLearner
    learner = FeedbackLearner()
    learner.record_feedback("response_id", score=5, comment="ممتاز")
    learner.record_feedback("response_id2", score=1, comment="بطيء")
    insights = learner.get_insights()
"""
from __future__ import annotations
import json
import logging
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger("nsm.feedback")

ROOT = Path(__file__).resolve().parent.parent
FEEDBACK_DIR = ROOT / "artifacts" / "agent_feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


class FeedbackLearner:
    """نظام التعلم من التغذية الراجعة."""

    def __init__(self, feedback_file: Optional[Path] = None):
        self._feedback_file = feedback_file or (FEEDBACK_DIR / "feedback.jsonl")
        self._feedbacks: List[Dict[str, Any]] = []
        self._load()

    def record_feedback(
        self,
        response_id: str,
        score: int,
        comment: str = "",
        agent_type: str = "nsm_agent",
        action_type: str = "answer",
        extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        """تسجيل ملاحظة المستخدم."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "response_id": response_id,
            "score": max(1, min(5, score)),
            "comment": comment[:500],
            "agent_type": agent_type,
            "action_type": action_type,
            "extra": extra or {},
        }

        self._feedbacks.append(entry)

        # حفظ
        try:
            with open(self._feedback_file, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Feedback save error: {e}")

        return len(self._feedbacks)

    def get_insights(self, min_samples: int = 5) -> Dict[str, Any]:
        """تحليل الأنماط من الملاحظات."""
        if len(self._feedbacks) < min_samples:
            return {
                "status": "insufficient_data",
                "count": len(self._feedbacks),
                "needed": min_samples,
            }

        # متوسط التقييم
        scores = [f["score"] for f in self._feedbacks]
        avg_score = sum(scores) / len(scores)

        # أفضل/أسوأ أنواع الإجراءات
        action_scores = defaultdict(list)
        for f in self._feedbacks:
            action_scores[f["action_type"]].append(f["score"])

        action_averages = {
            action: sum(s) / len(s)
            for action, s in action_scores.items()
        }
        best_action = max(action_averages, key=action_averages.get)
        worst_action = min(action_averages, key=action_averages.get)

        # أفضل/أسوأ أنواع الوكلاء
        agent_scores = defaultdict(list)
        for f in self._feedbacks:
            agent_scores[f["agent_type"]].append(f["score"])
        agent_averages = {
            agent: sum(s) / len(s)
            for agent, s in agent_scores.items()
        }

        # الكلمات الشائعة في التعليقات السلبية
        negative_comments = [f["comment"] for f in self._feedbacks if f["score"] <= 2]
        word_freq = Counter()
        for comment in negative_comments:
            for word in comment.lower().split():
                if len(word) > 3:
                    word_freq[word] += 1

        # اتجاه زمني (تحسّن أم تدهور)
        recent = self._feedbacks[-min(10, len(self._feedbacks)):]
        recent_avg = sum(f["score"] for f in recent) / len(recent) if recent else 0
        older = self._feedbacks[:-min(10, len(self._feedbacks))]
        older_avg = sum(f["score"] for f in older) / len(older) if older else recent_avg
        trend = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"

        return {
            "status": "ok",
            "total_feedbacks": len(self._feedbacks),
            "avg_score": round(avg_score, 2),
            "best_action": best_action,
            "worst_action": worst_action,
            "best_agent": max(agent_averages, key=agent_averages.get) if agent_averages else "n/a",
            "negative_keywords": dict(word_freq.most_common(10)),
            "trend": trend,
            "recent_avg": round(recent_avg, 2),
        }

    def get_adapted_prompt(self, base_prompt: str) -> str:
        """تكييف system prompt بناءً على الملاحظات."""
        insights = self.get_insights(min_samples=3)
        if insights.get("status") != "ok":
            return base_prompt

        additions = []

        # تفضيل الإجراءات الجيدة
        if insights.get("best_action"):
            additions.append(
                f"📊 من خبراتك: إجراءات نوع '{insights['best_action']}' "
                f"تحصل على تقييم أعلى — ركّز عليها."
            )

        # تجنب الإجراءات السيئة
        if insights.get("worst_action"):
            additions.append(
                f"⚠️ إجراءات نوع '{insights['worst_action']}' تحصل على تقييم منخفض — "
                f"قلّل استخدامها أو حسّن جودتها."
            )

        # الكلمات السلبية
        neg_kw = insights.get("negative_keywords", {})
        if neg_kw:
            top_neg = list(neg_kw.keys())[:5]
            additions.append(
                f"🔧 المستخدم يشتكي من: {', '.join(top_neg)} — تجنّبها."
            )

        if additions:
            return base_prompt + "\n\n" + "\n".join(additions)
        return base_prompt

    def clear(self) -> int:
        """مسح كل الملاحظات."""
        count = len(self._feedbacks)
        self._feedbacks.clear()
        if self._feedback_file.exists():
            self._feedback_file.unlink()
        return count

    def _load(self):
        """تحميل الملاحظات من disk."""
        if not self._feedback_file.exists():
            return
        try:
            with open(self._feedback_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._feedbacks.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Feedback load error: {e}")

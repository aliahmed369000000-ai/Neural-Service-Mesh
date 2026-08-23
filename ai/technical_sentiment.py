
import logging
from typing import Dict, Any, List

logger = logging.getLogger("NSM.TechnicalSentiment")

class TechnicalSentimentEngine:
    """
    محرك تحليل المشاعر التقنية (Technical Sentiment Engine).
    يقيم الحالة العاطفية للوكيل بناءً على التفاعلات التقنية.
    """
    def __init__(self):
        self.sentiment_history = []
        self.current_sentiment = "Stable" # [Stable, Confident, Frustrated, Exhausted]
        self.confidence_score = 0.7

    def analyze_steps(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """تحليل خطوات الوكيل لتحديد حالته العاطفية."""
        if not steps: return {"sentiment": self.current_sentiment, "confidence": self.confidence_score}
        
        recent_steps = steps[-5:]
        success_count = sum(1 for s in recent_steps if s.get("type") == "result" and "✅" in str(s.get("output", "")))
        failure_count = sum(1 for s in recent_steps if s.get("type") == "result" and "❌" in str(s.get("output", "")))
        
        # منطق تحديد الحالة
        if failure_count >= 2:
            self.current_sentiment = "Frustrated"
            self.confidence_score = max(0.2, self.confidence_score - 0.15)
        elif success_count >= 3:
            self.current_sentiment = "Confident"
            self.confidence_score = min(1.0, self.confidence_score + 0.1)
        else:
            self.current_sentiment = "Stable"
            
        return {
            "sentiment": self.current_sentiment,
            "confidence": round(self.confidence_score, 2),
            "alert": self.current_sentiment in ["Frustrated", "Exhausted"]
        }

    def get_swarm_sentiment(self, agent_sentiments: Dict[str, str]) -> str:
        """تحليل المناخ العام للسرب."""
        sentiments = list(agent_sentiments.values())
        if not sentiments: return "Neutral"
        
        if sentiments.count("Frustrated") > len(sentiments) / 2:
            return "Crisis"
        if sentiments.count("Confident") > len(sentiments) / 2:
            return "Optimal"
        return "Stable"

sentiment_engine = TechnicalSentimentEngine()

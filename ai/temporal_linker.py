"""
ai/temporal_linker.py
=====================
الرابط الزمني-الدلالي الموحد (Unified Temporal-Semantic Linker).
يربط بين الأحداث النصية، لقطات الفيديو، والمشاعر في خط زمني واحد.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("NSM.TemporalLinker")

class TemporalLinker:
    def __init__(self):
        pass

    def link_video_to_memory(self, video_id: str, memory_manager: Any) -> List[Dict[str, Any]]:
        """ربط الذاكرة الدلالية للوكيل بلقطات فيديو محددة."""
        from ai.multimodal_sync import multimodal_sync
        
        # 1. جلب بيانات الفيديو المزامنة
        video_context = multimodal_sync.query_context(video_id, "", semantic=False)
        if not video_context:
            logger.warning(f"⚠️ لا توجد بيانات مزامنة للفيديو {video_id}")
            return []
            
        linked_timeline = []
        
        # 2. لكل حدث في الفيديو، ابحث عن حقائق مرتبطة في ذاكرة الوكيل
        for item in video_context:
            ts = item["timestamp"]
            desc = item["visual_description"]
            
            # البحث الدلالي في ذاكرة الوكيل عن كلمات مفتاحية من وصف الفيديو
            related_facts = memory_manager.search(desc, limit=2)["semantic"]
            
            linked_entry = {
                "timestamp": ts,
                "visual": desc,
                "audio": item.get("spoken_text"),
                "sentiment": item.get("sentiment", {}).get("label"),
                "related_agent_facts": [f["content"] for f in related_facts],
                "frame_path": item.get("frame_path")
            }
            linked_timeline.append(linked_entry)
            
        logger.info(f"🔗 تم إنشاء رابط زمني موحد لـ {len(linked_timeline)} نقطة في الفيديو {video_id}")
        return linked_timeline

    def generate_chronological_report(self, timeline: List[Dict[str, Any]]) -> str:
        """توليد تقرير نصي مرتب زمنياً يدمج الفيديو والذاكرة."""
        report = ["📅 تقرير الخط الزمني الموحد (فيديو + ذاكرة الوكيل):"]
        for entry in timeline:
            ts_str = time.strftime('%M:%S', time.gmtime(entry['timestamp']))
            report.append(f"[{ts_str}] 🎬 رؤية: {entry['visual']}")
            if entry['audio']:
                report.append(f"      🎙️ صوت: {entry['audio']}")
            if entry['related_agent_facts']:
                report.append(f"      🧠 ذاكرة مرتبطة: {', '.join(entry['related_agent_facts'])}")
            report.append("-" * 20)
            
        return "\n".join(report)

temporal_linker = TemporalLinker()

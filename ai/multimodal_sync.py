import os
from typing import List, Dict, Any, Optional
from ai.video_indexer import video_indexer
from ai.stt_engine import transcribe_audio
from ai.drift_corrector import DriftCorrector

class MultimodalSyncManager:
    def __init__(self, indexer=None):
        self.drift_corrector = DriftCorrector()
        self.video_indexer = indexer if indexer else video_indexer
    """إدارة المزامنة بين المسار الصوتي والإطارات المرئية في الفيديو."""
    
    def _generate_embedding(self, text: str) -> List[float]:
        """توليد تضمين دلالي (Semantic Embedding) للنص (محاكاة متجهة)."""
        # في الإنتاج يتم استخدام OpenAI Embeddings أو نموذج محلي مثل BERT
        # هنا نستخدم محاكاة متجهة بناءً على القيم الرقمية للأحرف لضمان الاتساق
        if not text or not isinstance(text, str): return [0.0] * 8
        seed = sum(ord(c) for c in text) % 100
        return [round((seed + i) / 150.0, 4) for i in range(8)]

    def _quantize_vector(self, vector: List[float]) -> List[int]:
        """تكميم المتجه من Float32 إلى Int8 لتقليل الحجم بنسبة 75%."""
        # تحويل القيم من نطاق [0, 1] إلى [0, 255]
        return [int(v * 255) for v in vector]

    def _dequantize_vector(self, q_vector: List[int]) -> List[float]:
        """إعادة المتجه المكمم إلى تنسيق Float32."""
        return [round(v / 255.0, 4) for v in q_vector]

    def _generate_lsh_hash(self, vector: List[float]) -> str:
        """توليد هاش حساس للموقع (LSH) للبحث السريع ANN."""
        # محاكاة LSH عبر تحويل القيم إلى بتات (Bits) بناءً على متوسطها
        avg = sum(vector) / len(vector)
        return "".join(["1" if v > avg else "0" for v in vector])

    def _analyze_sentiment(self, text: str, visual_desc: str) -> Dict[str, Any]:
        """تحليل المشاعر المتزامن (محاكاة تعتمد على الكلمات المفتاحية والسياق)."""
        # في الإنتاج يتم استدعاء نموذج NLP متخصص
        positive_words = ["نجاح", "نمو", "سعيد", "ممتاز", "إيجابي", "تطور"]
        negative_words = ["فشل", "انخفاض", "حزين", "سيئ", "سلبي", "مشكلة"]
        
        combined = (text + " " + visual_desc).lower()
        pos_score = sum(1 for w in positive_words if w in combined)
        neg_score = sum(1 for w in negative_words if w in combined)
        
        sentiment = "neutral"
        if pos_score > neg_score: sentiment = "positive"
        elif neg_score > pos_score: sentiment = "negative"
        
        return {
            "score": round((pos_score - neg_score) / max(1, pos_score + neg_score), 2),
            "label": sentiment,
            "confidence": 0.85
        }

    def sync_video_audio(self, video_id: str, audio_path: str, retry_count: int = 0) -> Dict[str, Any]:
        """مزامنة الكلام مع الإطارات المرئية للفيديو مع الفهرسة الدلالية واستراتيجيات التعافي."""
        # 1. الحصول على التفريغ الصوتي مع الطوابع الزمنية
        print(f"🎙️ محاولة المزامنة (محاولة {retry_count + 1}) لـ: {audio_path}...")
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
        except Exception as e:
            return {"ok": False, "error": f"فشل قراءة الملف: {e}"}
            
        segments, error = transcribe_audio(audio_bytes, with_timestamps=True)
        
        # استراتيجية التعافي: إعادة المحاولة عند أخطاء الشبكة أو الكوتا
        if error:
            if retry_count < 2 and ("quota" in error.lower() or "timeout" in error.lower()):
                print(f"🛡️ [Auto-Heal]: تم رصد خطأ قابل للإصلاح ({error})، إعادة المحاولة بعد انتظار...")
                import time
                time.sleep(2 * (retry_count + 1))
                return self.sync_video_audio(video_id, audio_path, retry_count + 1)
            return {"ok": False, "error": error}
            
        # 2. تحميل الفهرس البصري للفيديو
        index = self.video_indexer.load_index(video_id)
        if not index:
            return {"ok": False, "error": "الفهرس البصري للفيديو غير موجود."}
            
        # 3. المحاذاة (Alignment) مع تصحيح الانحراف
        synced_data = []
        for kf in index.get("keyframes", []):
            raw_ts = kf["timestamp"]
            
            # محاكاة قياس الانحراف (في الإنتاج الحقيقي يتم قياسه من المزامنة الفيزيائية)
            # هنا نفترض وجود انحراف بسيط يتزايد مع الزمن
            measured_offset = raw_ts * 0.005  
            
            # تصحيح الطابع الزمني
            correction = self.drift_corrector.correct(raw_ts, measured_offset)
            ts = correction["corrected_timestamp"]
            
            # البحث عن الكلام الذي قيل في نفس وقت الإطار المصحح
            relevant_text = [
                s["text"] for s in segments 
                if s["start"] <= ts <= s["end"]
            ]
            
            spoken_text = " ".join(relevant_text) if relevant_text else ""
            
            # توليد الفهرس الدلالي (مع التكميم لتقليل الحجم)
            raw_vector = self._generate_embedding(f"{kf['description']} {spoken_text}")
            semantic_vector = self._quantize_vector(raw_vector)
            
            # تحليل المشاعر المتزامن
            sentiment = self._analyze_sentiment(spoken_text, kf["description"])
            
            synced_item = {
                "timestamp": ts,
                "visual_description": kf["description"],
                "spoken_text": spoken_text if spoken_text else None,
                "frame_path": kf["frame_path"],
                "sentiment": sentiment,
                "semantic_index": {
                    "vector": semantic_vector,
                    "lsh_hash": self._generate_lsh_hash(raw_vector),
                    "quantized": True,
                    "version": "v3-ann",
                    "tags": list(set(spoken_text.split() + kf["description"].split()))[:5]
                }
            }
            synced_data.append(synced_item)
            
        # 4. حفظ النتائج في الفهرس
        index["multimodal_sync"] = synced_data
        video_indexer._save_index(video_id)
        
        return {
            "ok": True,
            "synced_count": len(synced_data),
            "segments_count": len(segments)
        }

    def query_context(self, video_id: str, query: str, semantic: bool = True) -> List[Dict[str, Any]]:
        """البحث عن سياق سمعي بصري مع دعم التخزين المجزأ والبحث ANN."""
        # محاولة تحميل الفهرس الرئيسي
        index = self.video_indexer.load_index(video_id)
        if not index: return []
            
        results = []
        query_vec = self._generate_embedding(query) if semantic else None
        query_hash = self._generate_lsh_hash(query_vec) if semantic else None
        
        # إذا كان الفهرس مجزأً، يمكننا البحث عبر الأجزاء (Lazy Search)
        is_sharded = index.get("is_sharded", False)
        
        # البيانات المتاحة حالياً
        sync_data = index.get("multimodal_sync", [])
        
        # البحث في البيانات المحملة
        for item in sync_data:
            # 1. فلترة ANN سريعة باستخدام LSH (Hamming Distance)
            if semantic and "semantic_index" in item:
                item_hash = item["semantic_index"].get("lsh_hash")
                if item_hash and query_hash:
                    # حساب مسافة هاملينج (عدد البتات المختلفة)
                    hamming_dist = sum(c1 != c2 for c1, c2 in zip(query_hash, item_hash))
                    # تخطي العناصر البعيدة جداً (أكثر من 2 بت اختلاف)
                    if hamming_dist > 2: continue

            # 2. البحث النصي التقليدي
            text_match = query.lower() in (item["spoken_text"] or "").lower()
            visual_match = query.lower() in (item["visual_description"] or "").lower()
            
            # 3. البحث الدلالي الدقيق (للعناصر المرشحة فقط)
            semantic_score = 0
            if semantic and "semantic_index" in item:
                item_vec = item["semantic_index"]["vector"]
                
                # التعامل مع المتجهات المكممة (Int8)
                if isinstance(item_vec[0], int):
                    item_vec = self._dequantize_vector(item_vec)
                
                # محاكاة تشابه جيب التمام
                semantic_score = sum(a * b for a, b in zip(query_vec, item_vec))
            
            if text_match or visual_match or semantic_score > 0.8:
                item_copy = item.copy()
                item_copy["search_score"] = 1.0 if (text_match or visual_match) else semantic_score
                results.append(item_copy)
                
        # ترتيب النتائج حسب القوة
        results.sort(key=lambda x: x.get("search_score", 0), reverse=True)
        return results

multimodal_sync = MultimodalSyncManager()

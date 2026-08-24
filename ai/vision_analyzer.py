import base64
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from openai import OpenAI

logger = logging.getLogger("NeuralServiceMesh.VisionAnalyzer")

class VisionAnalyzer:
    """محرك تحليل الرؤية للوكلاء باستخدام نماذج LLM المتعددة الوسائط."""
    
    def __init__(self, model: str = "gemini-3-flash-preview"):
        self.model = model
        try:
            self.client = OpenAI()
        except Exception as e:
            logger.error(f"فشل تهيئة OpenAI client: {e}")
            self.client = None

    def _encode_image(self, image_path: str) -> Optional[str]:
        """تحويل الصورة إلى base64."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"خطأ في ترميز الصورة {image_path}: {e}")
            return None

    def analyze_image(self, image_path: str, prompt: str = "صف محتوى هذه الصورة بدقة لاستخدامها في فهرس فيديو.") -> Dict[str, Any]:
        """إرسال الصورة للنموذج والحصول على وصف نصي."""
        if not self.client:
            return {"ok": False, "error": "OpenAI client not initialized"}
            
        if not os.path.exists(image_path):
            return {"ok": False, "error": f"Image path not found: {image_path}"}
            
        base64_image = self._encode_image(image_path)
        if not base64_image:
            return {"ok": False, "error": "Failed to encode image"}
            
        extension = Path(image_path).suffix.lower().replace(".", "")
        mime_type = f"image/{extension}" if extension in ["png", "jpeg", "jpg", "webp"] else "image/jpeg"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                    "detail": "auto"
                                }
                            },
                        ],
                    }
                ],
                max_tokens=1024 # مناسب لـ Gemini
            )
            
            description = response.choices[0].message.content
            return {
                "ok": True,
                "description": description,
                "model": self.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                }
            }
        except Exception as e:
            logger.error(f"خطأ أثناء استدعاء نموذج الرؤية: {e}")
            return {"ok": False, "error": str(e)}

# نسخة عالمية للاستخدام السريع
vision_analyzer = VisionAnalyzer()


from PIL import Image
import os

class ImageProcessorNode:
    def process(self, data):
        path = data.get("image_path")
        if not os.path.exists(path):
            return {"error": "File not found"}
            
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode
            
        return {
            "status": "success",
            "dimensions": f"{width}x{height}",
            "mode": mode,
            "file_size_kb": os.path.getsize(path) / 1024
        }

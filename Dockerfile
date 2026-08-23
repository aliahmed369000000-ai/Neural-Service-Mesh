# Neural Service Mesh (NSM) — صورة Docker لتطبيق Streamlit
# جزء من إعداد "النشر المغلق بدون إنترنت خارجي" — استخدم مع docker-compose.yml
# الذي يشغّل أيضاً خادم Ollama للنموذج المحلي ويسحب النموذج تلقائياً.
#
# التشغيل الموصى به (يشغّل التطبيق + Ollama معاً بأمر واحد):
#   docker compose up -d
#
# بناء هذه الصورة منفردة (بدون compose) لأغراض الاختبار فقط:
#   docker build -t nsm-app .
#   docker run -p 8501:8501 -e NSM_OFFLINE_MODE=1 nsm-app

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# اعتماديات نظام مطلوبة فعلياً بواسطة مكتبات requirements.txt (وليست تخميناً):
#   - tesseract-ocr + tesseract-ocr-ara : pytesseract (OCR عربي)
#   - ffmpeg                            : moviepy / librosa / imageio-ffmpeg
#   - libsndfile1                       : librosa
#   - build-essential                   : تجميع numba/llvmlite إن لم تتوفر wheel جاهزة
#   - curl                              : فحص healthcheck لخادم Streamlit
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-ara \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# تثبيت متطلبات بايثون في طبقة منفصلة (كاش أسرع عند تعديل الكود فقط)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn requests aiohttp

# نسخ بقية المشروع
COPY . .

# إنشاء المجلدات اللازمة للبيانات
RUN mkdir -p artifacts/learning artifacts/memory artifacts/video_index

# .streamlit/config.toml في المستودع مضبوط مسبقاً: headless=true, address=0.0.0.0
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# القيمة الافتراضية هي تشغيل Streamlit، ولكن يمكن تجاوزها في docker-compose
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

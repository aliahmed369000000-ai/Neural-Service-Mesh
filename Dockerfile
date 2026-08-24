FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# اعتماديات نظام مطلوبة
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# تثبيت متطلبات بايثون
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir websockets cryptography aiohttp

# نسخ بقية المشروع
COPY . .

# إنشاء المجلدات اللازمة للبيانات
RUN mkdir -p artifacts/living_mesh artifacts/learning artifacts/memory

# Hugging Face يخصص المنفذ 7860 افتراضياً
ENV PORT=7860
ENV NODE_ID=mesh_alpha_seed
EXPOSE 7860

# تشغيل العقدة الموزعة
CMD ["python3", "ai/node_launcher.py", "--id", "mesh_alpha_seed", "--host", "0.0.0.0"]

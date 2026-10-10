"""استمرارية أسرار العقدة (GITHUB_TOKEN/HF_TOKEN ومفاتيح مزوّدي LLM) عبر
إعادة التشغيل — بدون تمريرها كمتغيّرات بيئة في كل مرة يُشغَّل فيها
node_launcher.py.

قبل هذا: ai/git_manager.py::GitManager وai/llm_fallback.py::LLMFallback
يقرآن GITHUB_TOKEN/HF_TOKEN/GROQ_API_KEY/... عبر os.getenv() مباشرة —
إن لم تُمرَّر كمتغيّر بيئة في سطر تشغيل العملية، تبقى العقدة بلا مصادقة
Git (عرضة لتعليق git clone، انظر إصلاح ai/git_manager.py السابق) وبلا
استدلال LLM حقيقي (يسقط دائماً إلى CKG Synthesis، انظر
ai/mesh_task_protocol.py::execute_inference) — حتى لو أُعطيت التوكن مرة
واحدة، تُفقَد عند أول إعادة تشغيل.

الحل: عند توفّر أي من هذه المتغيّرات في env عند الإقلاع، تُشفَّر وتُحفظ
في data_dir/secrets.enc.json — مشفَّرة بالمفتاح العام لهوية العقدة نفسها
(RSA-OAEP+AES-256-GCM عبر ai/e2e_crypto.py، نفس الآلية المستخدمة أصلاً
للتشفير الطرفي بين الأقران، مُطبَّقة هنا كـ"تشفير للذات" بدل لقرين). في
أي تشغيل لاحق بلا تلك المتغيّرات، تُقرأ وتُفَك وتُحقَن في os.environ
تلقائياً قبل بناء GitManager/أي استدعاء LLMFallback — فتعمل العقدة بنفس
المصادقة التي أُعطيت لها أول مرة، من نفس data_dir، دون تكرارها يدوياً.

أمان: الملف يبقى محليًا داخل data_dir الخاص بالعقدة (خارج المستودع —
انظر .gitignore)، مشفَّر بمفتاح لا يُستخرَج إلا من ملف الهوية الخاص
لنفس العقدة (keys/<node_id>.pem)، وبصلاحيات 0600. لا قيمة فعلية لأي سرّ
تُسجَّل في أي سطر log هنا — فقط أسماء المتغيّرات النشطة.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List

from ai.e2e_crypto import decrypt_payload, encrypt_payload

logger = logging.getLogger("NodeSecrets")

SECRETS_FILENAME = "secrets.enc.json"

# نفس أسماء المتغيّرات التي تقرأها GitManager (ai/git_manager.py) وLLMFallback
# (ai/llm_fallback.py) عبر os.getenv() مباشرة — مصدر واحد للحقيقة هنا حتى لا
# يُنسى مزوّد جديد يُضاف لاحقاً في أحد الملفين دون الآخر.
SECRET_ENV_VARS: List[str] = [
    "GITHUB_TOKEN", "HF_TOKEN", "HUGGINGFACE_API_KEY",
    "GROQ_API_KEY", "CEREBRAS_API_KEY",
    "CF_API_TOKEN", "CF_ACCOUNT_ID",
    "GOOGLE_API_KEY", "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TOGETHER_API_KEY",
]


def load_and_persist_secrets(data_dir: Path, private_key, public_key_pem: str) -> List[str]:
    """تُستدعى مرة واحدة عند إقلاع العقدة (بعد تحميل هوية RSA الخاصة بها).

    ترجع أسماء المتغيّرات (لا قيمها) النشطة الآن في os.environ لهذه العقدة،
    سواء جاءت من env الحالي أو من ملف محفوظ من تشغيل سابق."""
    secrets_path = Path(data_dir) / SECRETS_FILENAME
    persisted: dict = {}

    if secrets_path.exists():
        try:
            envelope = json.loads(secrets_path.read_text(encoding="utf-8"))
            persisted = decrypt_payload(envelope, private_key) or {}
        except Exception as e:
            # ملف تالف أو هوية مختلفة (نادر: نُسِخ data_dir بدون مفتاحه) — لا
            # نرفع استثناءً يعطّل إقلاع العقدة كاملة؛ نكمل كأنه لا يوجد شيء
            # محفوظ، ومتغيّرات env الحالية (إن وُجدت) ستُحفَظ من جديد أدناه.
            logger.warning(f"NodeSecrets: تعذّر قراءة/فك تشفير {SECRETS_FILENAME}: {e}")
            persisted = {}

    changed = False
    active: List[str] = []
    for key in SECRET_ENV_VARS:
        env_val = os.getenv(key)
        if env_val:
            active.append(key)
            if persisted.get(key) != env_val:
                persisted[key] = env_val
                changed = True
        elif persisted.get(key):
            os.environ[key] = persisted[key]
            active.append(key)

    if changed and persisted:
        try:
            envelope = encrypt_payload(persisted, public_key_pem)
            secrets_path.write_text(json.dumps(envelope), encoding="utf-8")
            os.chmod(secrets_path, 0o600)
        except Exception as e:
            logger.warning(f"NodeSecrets: تعذّر حفظ الأسرار المحدَّثة: {e}")

    if active:
        logger.info(f"NodeSecrets: متغيّرات نشطة لهذه العقدة: {sorted(set(active))}")

    return sorted(set(active))

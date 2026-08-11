# -*- coding: utf-8 -*-
"""
Checkpoint Storage — مزامنة اختيارية لنقاط تفتيش النموذج مع تخزين خارجي
========================================================================

المشكلة التي تحلّها:
  على Streamlit Community Cloud (ومعظم بيئات الحاويات المؤقتة) القرص المحلي
  لا يبقى بين إعادة تشغيل التطبيق/إعادة النشر. أي أوزان يحفظها التدريب محلياً
  (artifacts/model_training/checkpoints/*.pt) تُفقد، فيُعاد التدريب من الصفر
  رغم أن عداد المواضع (epoch) يبدو وكأنه تقدّم في الجلسة السابقة.

الحل هنا:
  طبقة تخزين اختيارية (pluggable) تدفع/تسحب ملفات الـ checkpoint من
  Hugging Face Hub، بدون أي تأثير على من لا يفعّلها. كل شيء best-effort:
  لو فشلت المزامنة الخارجية (لا مفاتيح، لا اتصال، مكتبة غير مثبّتة) يستمر
  التدريب محلياً تماماً كما كان قبل هذا التعديل — لا يُرفع أي استثناء للأعلى.

التفعيل (اختياري تماماً، معطّل افتراضياً):
  NSM_CHECKPOINT_REMOTE=hf
  HF_TOKEN=<توكن Hugging Face بصلاحية write>
  HF_CHECKPOINT_REPO=<username>/<repo-name>   (repo من نوع "model" أو "dataset")

لا تُضاف أي بيانات اعتماد هنا؛ تُقرأ فقط من متغيرات البيئة وقت التشغيل.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger("CheckpointStorage")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class CheckpointStorageBackend:
    """واجهة أساسية. أي backend جديد يطبّق push/pull ولا يرفع استثناءات."""

    name = "base"

    def push(self, local_path: Path, remote_key: str) -> bool:
        raise NotImplementedError

    def pull(self, remote_key: str, local_path: Path) -> bool:
        raise NotImplementedError


class LocalOnlyBackend(CheckpointStorageBackend):
    """الافتراضي: لا يوجد تخزين خارجي. push/pull لا تفعل شيئاً وتُعيد False
    بأمان تام (السلوك يطابق ما كان عليه المشروع قبل هذا التعديل)."""

    name = "local_only"

    def push(self, local_path: Path, remote_key: str) -> bool:
        return False

    def pull(self, remote_key: str, local_path: Path) -> bool:
        return False


class HFHubBackend(CheckpointStorageBackend):
    """يرفع/يسحب ملفات checkpoint من مستودع خاص على Hugging Face Hub.

    يتطلب: pip install huggingface_hub  (اختياري — مذكور في requirements.txt)
    وتفعيل عبر متغيرات البيئة الموصوفة أعلى الملف.
    """

    name = "huggingface_hub"

    def __init__(self, repo_id: str, token: str):
        self._repo_id = repo_id
        self._token = token

    def push(self, local_path: Path, remote_key: str) -> bool:
        try:
            from huggingface_hub import HfApi  # استيراد كسول — لا يكسر الاستيراد لو غير مثبّتة
        except Exception as e:
            logger.warning(f"huggingface_hub غير متاحة، تُهمَل المزامنة الخارجية: {e}")
            return False
        try:
            api = HfApi(token=self._token)
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_key,
                repo_id=self._repo_id,
                repo_type="model",
            )
            logger.info(f"تم رفع checkpoint خارجياً: {remote_key}")
            return True
        except Exception as e:
            logger.warning(f"فشل رفع checkpoint إلى HF Hub ({remote_key}): {e}")
            return False

    def pull(self, remote_key: str, local_path: Path) -> bool:
        try:
            from huggingface_hub import hf_hub_download
        except Exception as e:
            logger.warning(f"huggingface_hub غير متاحة، تُهمَل الاستعادة الخارجية: {e}")
            return False
        try:
            downloaded = hf_hub_download(
                repo_id=self._repo_id,
                filename=remote_key,
                token=self._token,
                repo_type="model",
            )
            local_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(downloaded, local_path)
            logger.info(f"تم استرجاع checkpoint من التخزين الخارجي: {remote_key}")
            return True
        except Exception as e:
            # طبيعي جداً لو الملف غير موجود بعد على المستودع البعيد
            logger.info(f"لا يوجد checkpoint خارجي لـ {remote_key} ({e})")
            return False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_backend() -> CheckpointStorageBackend:
    """يقرأ متغيرات البيئة ويُعيد أفضل backend متاح. لا يرفع استثناءات أبداً؛
    أي خطأ في الإعداد → رجوع آمن لـ LocalOnlyBackend."""
    try:
        remote = (os.environ.get("NSM_CHECKPOINT_REMOTE") or "").strip().lower()
        if remote == "hf":
            repo = (os.environ.get("HF_CHECKPOINT_REPO") or "").strip()
            token = (os.environ.get("HF_TOKEN") or "").strip()
            if repo and token:
                return HFHubBackend(repo_id=repo, token=token)
            logger.info(
                "NSM_CHECKPOINT_REMOTE=hf لكن HF_CHECKPOINT_REPO أو HF_TOKEN ناقصين — "
                "سيُستخدم التخزين المحلي فقط."
            )
    except Exception as e:
        logger.warning(f"تعذّرت قراءة إعداد التخزين الخارجي، رجوع للمحلي: {e}")
    return LocalOnlyBackend()


# ---------------------------------------------------------------------------
# High-level helpers (يستخدمها model_training_agent.py)
# ---------------------------------------------------------------------------

def sync_checkpoint_after_save(run_id: str, files: Iterable[Path]) -> List[str]:
    """يُستدعى بعد حفظ checkpoint محلياً. يحاول رفع كل ملف للتخزين الخارجي
    إن كان مفعّلاً. لا يرفع استثناءات مطلقاً — فشل المزامنة لا يوقف التدريب.
    يُعيد قائمة أسماء الملفات التي رُفعت بنجاح (فارغة لو محلي فقط أو فشل)."""
    uploaded: List[str] = []
    try:
        backend = get_backend()
        if isinstance(backend, LocalOnlyBackend):
            return uploaded
        for f in files:
            try:
                f = Path(f)
                if not f.is_file():
                    continue
                remote_key = f"{run_id}/{f.name}"
                if backend.push(f, remote_key):
                    uploaded.append(remote_key)
            except Exception as e:
                logger.warning(f"تعذّر مزامنة {f}: {e}")
    except Exception as e:
        logger.warning(f"sync_checkpoint_after_save: خطأ غير متوقع تم تجاهله: {e}")
    return uploaded


def restore_checkpoint_if_missing(run_id: str, checkpoint_dir: Path, which: str = "latest") -> bool:
    """لو الملف المحلي latest.pt/best.pt غير موجود، يحاول استرجاعه من التخزين
    الخارجي (إن كان مفعّلاً) قبل أن يُعامَل كـ 'لا يوجد checkpoint'.
    لا يرفع استثناءات أبداً."""
    try:
        checkpoint_dir = Path(checkpoint_dir)
        local_path = checkpoint_dir / f"{which}.pt"
        if local_path.is_file():
            return True  # موجود محلياً أصلاً، لا حاجة للاستعادة
        backend = get_backend()
        if isinstance(backend, LocalOnlyBackend):
            return False
        remote_key = f"{run_id}/{which}.pt"
        return backend.pull(remote_key, local_path)
    except Exception as e:
        logger.warning(f"restore_checkpoint_if_missing: خطأ غير متوقع تم تجاهله: {e}")
        return False

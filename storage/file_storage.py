from __future__ import annotations
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class FileStorage:
    def __init__(self, storage_dir: str = "./data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"FileStorage: {self.storage_dir.resolve()}")

    def save(self, filename: str, data: Any) -> bool:
        """يكتب filename بشكل ذرّي: JSON كامل لملف مؤقت في نفس المجلد ثم
        os.replace() (ذرّية على نفس نظام الملفات). قبل هذا كانت الكتابة
        مباشرة على الملف النهائي — أي انقطاع (تعطّل العملية، أو كتابتان
        متزامنتان من خيطين لنفس filename كما يحدث فعلياً مع
        core/node_channel.py المستخدَمة من مسارات متعددة الخيوط عبر
        SwarmCoordinator) تترك JSON غير مكتمل/تالف يفشل load() في قراءته
        لاحقاً، فتضيع كل الرسائل/السجلّ المحفوظ بصمت. os.replace() يضمن
        أن أي قارئ متزامن يرى إما المحتوى القديم كاملاً أو الجديد كاملاً،
        لا حالة وسيطة أبداً."""
        target = self._path(filename)
        tmp_path = None
        try:
            fd, tmp_name = tempfile.mkstemp(
                dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
            )
            tmp_path = Path(tmp_name)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target)
            return True
        except Exception as e:
            logger.error(f"save failed '{filename}': {e}")
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            return False

    def load(self, filename: str) -> Optional[Any]:
        p = self._path(filename)
        if not p.exists():
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"load failed '{filename}': {e}")
            return None

    def delete(self, filename: str) -> bool:
        p = self._path(filename)
        if p.exists():
            p.unlink()
            return True
        return False

    def exists(self, filename: str) -> bool:
        return self._path(filename).exists()

    def list_files(self) -> List[str]:
        return [f.name for f in self.storage_dir.glob("*.json")]

    def stats(self) -> dict:
        files = self.list_files()
        size = sum((self.storage_dir / f).stat().st_size for f in files)
        return {"storage_dir": str(self.storage_dir), "files": files,
                "file_count": len(files), "total_bytes": size}

    def _path(self, filename: str) -> Path:
        return self.storage_dir / Path(filename).name

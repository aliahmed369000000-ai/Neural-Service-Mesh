from __future__ import annotations
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
import threading
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class FileStorage:
    def __init__(self, storage_dir: str = "./data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        # يُسلسل الكتابات على نفس الملف داخل العملية: كتابتان متزامنتان من
        # خيطين كانتا تنتهيان بأن يفوز الأبطأ بلقطة أقدم (last-writer-wins).
        self._lock = threading.RLock()
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
            with self._lock:
                self._keep_backup(target)
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

    @staticmethod
    def _backup_path(target: Path) -> Path:
        return target.with_name(target.name + ".bak")

    def _keep_backup(self, target: Path) -> None:
        """يحتفظ بآخر نسخة سليمة معروفة كـ <name>.bak عبر hard link (بلا نسخ
        بيانات) قبل استبدال الملف — best-effort، فشله لا يمنع الحفظ."""
        if not target.exists():
            return
        bak = self._backup_path(target)
        try:
            bak.unlink(missing_ok=True)
            os.link(target, bak)
        except Exception:
            pass

    def _read_json(self, p: Path) -> Any:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def load(self, filename: str) -> Optional[Any]:
        """عند تلف الملف (JSON غير صالح) لا يُهمَل بصمت: يُنقل كما هو إلى
        <name>.corrupt-<وقت> (حتى لا تكتب أول save لاحقة فوق الدليل وتضيع
        بياناته للأبد)، ثم تُجرَّب آخر نسخة سليمة <name>.bak. قبل هذا كان
        الفشل يرجع None فيبدأ NodeRegistry/reputation فارغاً ثم تمحو أول
        save كل تاريخ العُقد المحفوظ."""
        p = self._path(filename)
        if not p.exists():
            return None
        try:
            return self._read_json(p)
        except Exception as e:
            logger.error(f"load failed '{filename}': {e}")
        with self._lock:
            try:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
                p.replace(p.with_name(f"{p.name}.corrupt-{stamp}"))
            except Exception as e2:
                logger.error(f"تعذّر حفظ الملف التالف '{filename}': {e2}")
        bak = self._backup_path(p)
        if bak.exists():
            try:
                data = self._read_json(bak)
                logger.warning(f"'{filename}' استُعيد من النسخة الاحتياطية .bak")
                return data
            except Exception as e3:
                logger.error(f"النسخة الاحتياطية '{filename}.bak' تالفة أيضاً: {e3}")
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

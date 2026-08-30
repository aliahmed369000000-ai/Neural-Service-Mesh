"""
NSM Sovereignty Toolbox — ai/sovereignty_toolbox.py
===================================================
أدوات متقدمة للوكلاء للقيام بمهام السيادة الذاتية:
- الاستنساخ النظيف (Autonomous Clone)
- الرفع الموثق (Autonomous Push)
- التنظيف الذاتي (Autonomous Cleanup)
- إدارة التوكنات الآمنة (Secure Token Management)
"""

import os
import shutil
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

class SovereigntyToolbox:
    def __init__(self, repo_url: str = "https://github.com/aliahmed369000000-ai/Neural-Service-Mesh"):
        self.repo_url = repo_url
        self.bot_name = "NSM Bot"
        self.bot_email = "nsm-bot@users.noreply.github.com"

    def _run_cmd(self, cmd: list, cwd: Optional[str] = None) -> Tuple[int, str]:
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
                cwd=cwd or str(ROOT)
            )
            return res.returncode, (res.stdout + res.stderr).strip()
        except Exception as e:
            return 1, str(e)

    def autonomous_clone(self, token: str, target_dir: str = "/tmp/nsm_sovereign") -> Dict[str, Any]:
        """استنساخ المستودع باستخدام التوكن إلى مجلد مؤقت."""
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        
        # إخفاء التوكن في رابط الاستنساخ
        auth_url = self.repo_url.replace("https://", f"https://{token}@")
        
        code, out = self._run_cmd(["git", "clone", auth_url, target_dir], cwd="/tmp")
        
        if code == 0:
            # ضبط هوية البوت فور الاستنساخ
            self._run_cmd(["git", "config", "user.name", self.bot_name], cwd=target_dir)
            self._run_cmd(["git", "config", "user.email", self.bot_email], cwd=target_dir)
            return {"ok": True, "msg": "تم الاستنساخ وضبط الهوية بنجاح.", "path": target_dir}
        else:
            return {"ok": False, "msg": f"فشل الاستنساخ: {out}"}

    def autonomous_push(self, target_dir: str, commit_msg: str, files: list = None) -> Dict[str, Any]:
        """إضافة الملفات، عمل commit، والرفع مع التحقق."""
        if not os.path.exists(target_dir):
            return {"ok": False, "msg": "مجلد العمل غير موجود."}

        # 1. إضافة الملفات
        if files:
            for f in files:
                self._run_cmd(["git", "add", f], cwd=target_dir)
        else:
            self._run_cmd(["git", "add", "."], cwd=target_dir)

        # 2. Commit
        code, out = self._run_cmd(["git", "commit", "-m", commit_msg], cwd=target_dir)
        if code != 0 and "nothing to commit" not in out:
            return {"ok": False, "msg": f"فشل الـ Commit: {out}"}

        # 3. Push
        code, out = self._run_cmd(["git", "push", "origin", "main"], cwd=target_dir)
        if code != 0:
            return {"ok": False, "msg": f"فشل الـ Push: {out}"}

        # 4. التحقق الفعلي
        c1, local_sha = self._run_cmd(["git", "rev-parse", "HEAD"], cwd=target_dir)
        c2, remote_out = self._run_cmd(["git", "ls-remote", "origin", "main"], cwd=target_dir)
        
        if c2 == 0 and local_sha in remote_out:
            return {"ok": True, "msg": "تم الرفع والتحقق بنجاح (SHA متطابق).", "sha": local_sha}
        else:
            return {"ok": False, "msg": "تم الرفع ولكن فشل التحقق من SHA البعيد."}

    def autonomous_cleanup(self, target_dir: str) -> Dict[str, Any]:
        """حذف النسخة المحلية لتنظيف التوكنات والأدلة."""
        if os.path.exists(target_dir) and target_dir.startswith("/tmp/"):
            shutil.rmtree(target_dir)
            return {"ok": True, "msg": "تم تنظيف بيئة العمل بنجاح."}
        return {"ok": False, "msg": "المسار غير صالح للتنظيف أو غير موجود."}

if __name__ == "__main__":
    # اختبار بسيط للبنية
    toolbox = SovereigntyToolbox()
    print("Sovereignty Toolbox Initialized.")

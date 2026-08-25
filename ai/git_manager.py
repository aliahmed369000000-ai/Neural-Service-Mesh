import os
import subprocess
import shutil
import logging
from typing import Optional, List

logger = logging.getLogger("NSM-GitManager")

class GitManager:
    """
    مدير عمليات Git لوكلاء NSM.
    يسمح للوكلاء بالاستنساخ، التعديل، والرفع بشكل آمن ومستقل.
    """
    
    def __init__(self, token: Optional[str] = None, repo_url: str = "github.com/aliahmed369000000-ai/Neural-Service-Mesh.git"):
        self.token = token or os.getenv("HF_TOKEN") or os.getenv("GITHUB_TOKEN")
        self.repo_url = repo_url
        self.base_dir = "/tmp/nsm_evolution"
        
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def _get_auth_url(self) -> str:
        """بناء رابط الاستنساخ مع التوكن للمصادقة."""
        if self.token:
            return f"https://{self.token}@{self.repo_url}"
        return f"https://{self.repo_url}"

    def clone(self, target_name: str = "clone_temp") -> str:
        """استنساخ المستودع إلى مجلد مؤقت."""
        target_path = os.path.join(self.base_dir, target_name)
        if os.path.exists(target_path):
            shutil.rmtree(target_path)
            
        logger.info(f"🚀 Cloning repository to {target_path}...")
        cmd = ["git", "clone", self._get_auth_url(), target_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise Exception(f"❌ Git Clone Failed: {result.stderr}")
            
        return target_path

    def commit_and_push(self, repo_path: str, message: str, files: List[str] = ["."]):
        """تنفيذ التغييرات ورفعها إلى المستودع."""
        try:
            # إعداد الهوية
            subprocess.run(["git", "config", "user.email", "nsm-bot@users.noreply.github.com"], cwd=repo_path)
            subprocess.run(["git", "config", "user.name", "NSM Bot"], cwd=repo_path)
            
            # إضافة الملفات
            for file in files:
                subprocess.run(["git", "add", file], cwd=repo_path)
            
            # Commit
            result = subprocess.run(["git", "commit", "-m", message], cwd=repo_path, capture_output=True, text=True)
            if "nothing to commit" in result.stdout:
                logger.info("⚠️ Nothing to commit.")
                return
            
            # Push
            logger.info("📤 Pushing changes to GitHub...")
            push_result = subprocess.run(["git", "push", "origin", "main"], cwd=repo_path, capture_output=True, text=True)
            
            if push_result.returncode != 0:
                raise Exception(f"❌ Git Push Failed: {push_result.stderr}")
            
            logger.info("✅ Push successful!")
            
        finally:
            # تنظيف النسخة المحلية دائماً
            self.cleanup(repo_path)

    def cleanup(self, repo_path: str):
        """حذف المجلد المؤقت والتوكن الملحق به."""
        if os.path.exists(repo_path):
            logger.info(f"🧹 Cleaning up local clone at {repo_path}...")
            shutil.rmtree(repo_path)

    def apply_evolution(self, task_description: str):
        """
        دالة تجريبية: تسمح للوكيل بتعديل نفسه بناءً على وصف المهمة.
        (سيتم ربطها بـ LLM في المراحل المتقدمة).
        """
        repo_path = self.clone("self_evolution_task")
        # هنا يتم تنفيذ منطق التعديل البرمجي
        # كمثال: إضافة تعليق في ملف README
        readme_path = os.path.join(repo_path, "README.md")
        with open(readme_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n### 🧬 Evolution Log: {task_description}\n")
        
        self.commit_and_push(repo_path, f"🧬 NSM Evolution: {task_description}")

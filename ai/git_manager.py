import os
import subprocess
import shutil
import logging
import tempfile
from typing import Optional, List

logger = logging.getLogger("NSM-GitManager")

# 🆕 لا مهلة زمنية افتراضية لأي subprocess.run هنا (git clone/push) — جرّبتها
# فعلياً: شغّلت عقدة حقيقية عبر ai/node_launcher.py بلا GITHUB_TOKEN/HF_TOKEN
# (سيناريو نشر طبيعي تماماً، لا حالة حافة)، فتوقّفت دورة التطوّر الذاتي
# الأولى إلى الأبد عند "git clone" (استنساخ مجهول الهوية لمستودع يبدو أنه
# يتطلّب مصادقة يتعلّق بلا استجابة بدل فشل سريع — تحقّقت مباشرة: نفس الأمر
# بمعزل عن العقدة علّق أكثر من 15 ثانية بلا أي تقدّم ولا أي خطأ). بما أن
# maybe_self_evolve() في ai/living_mesh.py تُستدعى من خيط مراقب خلفي واحد في
# حلقة (_self_evolution_watch_loop)، فتعليق استدعاء git واحد يُسكِت دورة
# التطوّر الذاتي للعقدة بالكامل للأبد — بصمت، بلا أي استثناء يُسجَّل، لأن
# الخيط نفسه عالق داخل subprocess.run لا يصل أبداً لأي except. مهلة صريحة
# تحوّل هذا من تعليق أبدي صامت إلى فشل سريع وواضح يلتقطه
# _execute_evolution's except الموجود أصلاً (نفس مسار "❌ Git Clone Failed").
_GIT_TIMEOUT_SECONDS = 60


class GitOperationTimeout(Exception):
    """أمر git (clone/add/commit/push) تجاوز _GIT_TIMEOUT_SECONDS بلا استجابة
    — غالباً مصادقة ناقصة (GITHUB_TOKEN/HF_TOKEN) تنتظر إدخالاً تفاعلياً لن
    يصل أبداً داخل subprocess.run، أو شبكة متعطّلة لا ترفض الاتصال بوضوح."""
    pass


class GitManager:
    """
    مدير عمليات Git لوكلاء NSM.
    يسمح للوكلاء بالاستنساخ، التعديل، والرفع بشكل آمن ومستقل.
    """
    
    def __init__(self, token: Optional[str] = None, repo_url: str = "github.com/aliahmed369000000-ai/Neural-Service-Mesh.git",
                 timeout_seconds: float = _GIT_TIMEOUT_SECONDS):
        self.token = token or os.getenv("HF_TOKEN") or os.getenv("GITHUB_TOKEN")
        self.repo_url = repo_url
        self.timeout_seconds = timeout_seconds
        # 🆕 tempfile.gettempdir() بدل "/tmp" الثابت: يحترم TMPDIR/TEMP/TMP
        # ويسقط لبدائل قابلة للكتابة تلقائياً — "/tmp" وحدها غير موجودة
        # أو غير قابلة للكتابة عند الجذر على بيئات مثل Termux/أندرويد،
        # ما كان يُسقط GitManager() (ومن ثمّ LivingMeshNode كاملاً) بخطأ
        # "OSError: [Errno 30] Read-only file system: '/tmp'" فور الإنشاء.
        self.base_dir = os.path.join(tempfile.gettempdir(), "nsm_evolution")
        
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
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired as e:
            # تنظيف أي بقايا استنساخ جزئي قبل الرفع — نفس سلوك cleanup() في
            # المسارات الناجحة، حتى لا يتراكم target_path فاسداً عبر محاولات
            # لاحقة (self.clone يحذفه أصلاً في بداية الاستدعاء، لكن التنظيف
            # هنا فوري بدل الانتظار للمحاولة التالية).
            shutil.rmtree(target_path, ignore_errors=True)
            raise GitOperationTimeout(
                f"❌ Git Clone timed out after {self.timeout_seconds}s — تحقّق من "
                f"GITHUB_TOKEN/HF_TOKEN إن كان المستودع يتطلّب مصادقة، أو من الشبكة"
            ) from e

        if result.returncode != 0:
            raise Exception(f"❌ Git Clone Failed: {result.stderr}")
            
        return target_path

    def commit_and_push(self, repo_path: str, message: str, files: List[str] = ["."]):
        """تنفيذ التغييرات ورفعها إلى المستودع.

        🆕 كل subprocess.run هنا الآن له timeout=self.timeout_seconds (انظر
        تعليق _GIT_TIMEOUT_SECONDS أعلى الملف) — git push تحديداً نفس خطورة
        git clone: بلا مصادقة أو عند تعليق الشبكة يعلّق بلا استجابة، وهذه
        الدالة تُستدعى من نفس خيط التطوّر الذاتي الخلفي الوحيد. finally أدناه
        (تنظيف repo_path) يبقى يعمل بغض النظر — سواء انتهت العملية بنجاح، أو
        برفع استثناء TimeoutExpired/غيره؛ الفرق الوحيد الآن هو أن الاستثناء
        يصل فعلاً خلال ثوانٍ معدودة بدل تعليق الخيط إلى الأبد.
        """
        try:
            try:
                # إعداد الهوية
                subprocess.run(["git", "config", "user.email", "nsm-bot@users.noreply.github.com"],
                                cwd=repo_path, timeout=self.timeout_seconds)
                subprocess.run(["git", "config", "user.name", "NSM Bot"],
                                cwd=repo_path, timeout=self.timeout_seconds)

                # إضافة الملفات
                for file in files:
                    subprocess.run(["git", "add", file], cwd=repo_path, timeout=self.timeout_seconds)

                # Commit
                result = subprocess.run(["git", "commit", "-m", message], cwd=repo_path,
                                         capture_output=True, text=True, timeout=self.timeout_seconds)
                if "nothing to commit" in result.stdout:
                    logger.info("⚠️ Nothing to commit.")
                    return

                # Push
                logger.info("📤 Pushing changes to GitHub...")
                push_result = subprocess.run(["git", "push", "origin", "main"], cwd=repo_path,
                                              capture_output=True, text=True, timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired as e:
                # نفس الاستثناء الموحَّد لكل أمر git هنا (config/add/commit/
                # push) — جميعها نفس المخاطرة بالضبط (مصادقة ناقصة أو شبكة
                # معطوبة)، فلا داعٍ لتخصيص try/except منفصل لكل أمر.
                raise GitOperationTimeout(
                    f"❌ Git command timed out after {self.timeout_seconds}s "
                    f"({' '.join(e.cmd)}) — تحقّق من GITHUB_TOKEN/HF_TOKEN أو من الشبكة"
                ) from e

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

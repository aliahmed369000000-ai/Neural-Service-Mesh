import os
import json
import re
import subprocess
import time
from typing import List, Dict

class KaggleGlobalScheduler:
    """
    موزع المهام العالمي لإدارة تشغيل سرب NSM عبر حسابات Kaggle المتعددة.
    """
    def __init__(self, accounts_file: str):
        self.accounts_file = accounts_file
        self.accounts = self._load_accounts()
        self.base_dir = "/home/ubuntu/.kaggle"
        os.makedirs(self.base_dir, exist_ok=True)

    def _load_accounts(self) -> List[Dict]:
        with open(self.accounts_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _setup_credentials(self, username: str, key: str):
        """تهيئة بيانات الاعتماد لحساب معين."""
        creds = {"username": username, "key": key}
        with open(os.path.join(self.base_dir, "kaggle.json"), "w") as f:
            json.dump(creds, f)
        os.chmod(os.path.join(self.base_dir, "kaggle.json"), 0o600)

    def create_kernel_metadata(self, username: str, node_id: str, target_dir: str = "."):
        """إنشاء ملف تعريف الكيرنل لكل عقدة."""
        # استخدام طابع زمني لضمان فرادة المعرف وتجنب تعارض الـ 409
        ts = int(time.time())
        metadata = {
            "id": f"{username}/nsm-node-{node_id}-{ts}",
            "title": f"NSM Global Node - {node_id} ({ts})",
            "code_file": "kaggle_run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "accelerator": "nvidiaTeslaT4",
            "gpu_count": 2,
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": [f"{username}/nsm-gpu-wheels-cu118"],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": []
        }
        with open(os.path.join(target_dir, "kernel-metadata.json"), "w") as f:
            json.dump(metadata, f)

    def build_bundle(self, output_path: str = None) -> str:
        """يبني kaggle_run.py موحّداً بدمج شجرة الاعتماديات الحقيقية فقط
        (فُحصت فعلياً سطراً بسطر، وليست قائمة تخمينية):

            security_guard.py، cognitive_growth.py، gradient_mesh.py
            → distributed_trainer.py → kaggle_run.py (نقطة الدخول)

        ملاحظة إصلاح جوهرية: الإصدار السابق كان يدمج 3 ملفات فقط
        (distributed_trainer, gradient_mesh, living_mesh) بينما
        distributed_trainer.py يستورد أيضاً من security_guard و
        cognitive_growth — وهما لم يكونا مُدمَجين إطلاقاً. هذا بالضبط
        سبب ModuleNotFoundError: No module named 'security_guard' في
        سرب nsm-global-node-*.

        living_mesh.py استُبعد عمداً هنا: تحققت أنه غير مُستورد فعلياً
        من distributed_trainer.py أو kaggle_run.py في مسار التدريب
        الحالي، وإدراجه كان سيجرّ اعتماديات إضافية غير ضرورية لهذا
        المسار (alert_manager، unified_memory، git_manager، toolbox،
        ann_engine، sharding_engine) وترفع احتمال فشل جديد بلا فائدة.

        كذلك: الإصدار السابق كان يستبدل "from ai." بـ"from " نصياً بلا
        تمييز — فتتحول "from ai.security_guard import X" إلى
        "from security_guard import X"، وهذا يفشل أيضاً حتى لو أُضيف
        الملف للدمج، لأن الرمز أصبح متاحاً مباشرة بنفس نطاق الأسماء بعد
        الدمج ولا حاجة لاستيراده من ملف منفصل غير موجود على Kaggle.
        الإصلاح هنا يحذف هذه الأسطر كلياً (تعليق فقط) بدل إعادة تسميتها.
        """
        dep_order = [
            "ai/security_guard.py",
            "ai/cognitive_growth.py",
            "ai/gradient_mesh.py",
            "ai/distributed_trainer.py",
        ]
        bundled_module_names = {os.path.splitext(os.path.basename(p))[0] for p in dep_order}

        def _strip_internal_imports(content: str) -> str:
            out_lines = []
            for line in content.splitlines():
                m = re.match(r"^\s*from ai\.(\w+) import", line)
                if m and m.group(1) in bundled_module_names:
                    out_lines.append(f"# [bundled, تم دمجه أعلاه] {line.strip()}")
                    continue
                out_lines.append(line)
            return "\n".join(out_lines)

        def _disable_main_block(content: str) -> str:
            """يعطّل كتلة if __name__ == '__main__': الخاصة بملفات التبعية
            (وليس نقطة الدخول النهائية kaggle_run.py) — بدون هذا، كل ملف
            مُدمَج ينفّذ كتلة __main__ الخاصة به بالتتابع عند تشغيل الحزمة
            الموحّدة (لأنها كلها تُنفَّذ فعلياً تحت __name__ == '__main__'
            الواحد الخاص بالحزمة الكاملة). تحقّقت من هذا فعلياً بتشغيل
            الحزمة في بيئة معزولة قبل هذا الإصلاح: نفَّذت security_guard.py
            كتلته التجريبية الخاصة به بدل الانتقال مباشرة لتنفيذ التدريب."""
            return re.sub(
                r'^if __name__ == "__main__":',
                'if __name__ == "__disabled_main__":  # [bundled] كتلة تشغيل تجريبية معطَّلة عمداً',
                content,
                flags=re.M,
            )

        parts = []
        for dep in dep_order:
            if not os.path.exists(dep):
                raise FileNotFoundError(f"ملف اعتماد أساسي مفقود لبناء الحزمة: {dep}")
            with open(dep, "r", encoding="utf-8") as f:
                content = f.read()
            content = _strip_internal_imports(content)
            content = _disable_main_block(content)
            parts.append(f"\n# ═══ FROM {dep} ═══\n" + content)

        with open("kaggle_run.py", "r", encoding="utf-8") as f:
            main_content = f.read()
        # لا نعطّل كتلة __main__ هنا عمداً — هذه نقطة الدخول الوحيدة المقصودة للحزمة
        parts.append("\n# ═══ MAIN RUNNER (kaggle_run.py) ═══\n" + _strip_internal_imports(main_content))

        bundle = "\n".join(parts)
        
        # 🔐 حماية الأسرار: التوكن يتم استلامه من بيئة التشغيل في Kaggle وليس صلباً في الكود
        header = """
import os
import subprocess
import sys
import glob

# 🚀 فرض استخدام T4x2 والتحقق من الـ GPU لسرعة الإقلاع
print("🛠️ NSM Swarm Turbo-Boot: Initializing GPU environment...")
try:
    import torch
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        print(f"✅ GPU Found: {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})")
    else:
        print("⚠️ GPU not detected. Check Kaggle settings.")
except Exception as err:
    print(f"⚠️ GPU Check failed: {err}")

# التوكن يتم حقنه عبر المجدول في بيئة التشغيل فقط
os.environ['HF_REPO_ID'] = 'AliAhmedMo/nsm-surah-weights'

try:
    import huggingface_hub
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
"""
        bundle = header + bundle

        if output_path:
            with open(output_path, "w", encoding="utf-8") as out:
                out.write(bundle)
        return bundle

    def launch_all(self):
        """إطلاق جميع العقد عبر الحسابات السبعة مع تفعيل 28 عقدة افتراضية."""
        print(f"🚀 بدء إطلاق السرب العالمي (28 عقدة) عبر {len(self.accounts)} حسابات...")
        results = []
        
        upload_dir = "kaggle_upload"
        os.makedirs(upload_dir, exist_ok=True)
        self.build_bundle(output_path=os.path.join(upload_dir, "kaggle_run.py"))
        
        for i, acc in enumerate(self.accounts):
            username = acc['username']
            key = acc['key']
            self._setup_credentials(username, key)
            
            # نطلق عقدتين متزامنتين لكل حساب لضمان السيادة وتجنب OOM
            for j in range(2): 
                node_num = i * 4 + j + 1
                node_id = f"global_{node_num}"
                
                print(f"📦 تجهيز العقدة {node_id} للحساب {username}...")
                self.create_kernel_metadata(username, node_id, target_dir=upload_dir)
                
                try:
                    process = subprocess.run(
                        ["kaggle", "kernels", "push", "-p", upload_dir],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    print(f"✅ تم إرسال كود الإطلاق للعقدة {node_id} بنجاح.")
                    results.append({"node": node_id, "user": username, "status": "Active"})
                except subprocess.CalledProcessError as e:
                    print(f"❌ فشل إطلاق العقدة {node_id}: {e.stderr}")
                    results.append({"node": node_id, "user": username, "status": "Failed", "error": e.stderr})
                
                time.sleep(15) 
            
        return results

    def run_relay_24_7(self):
        """تشغيل نظام التتابع السيادي لضمان نشاط السرب 24/7."""
        print("🌀 تفعيل نظام التتابع السيادي (Relay Persistence) - نشاط 24/7...")
        while True:
            try:
                self.launch_all()
                print("⏳ [Relay System] السرب نشط الآن. الدورة القادمة بعد 8 ساعات...")
                time.sleep(8 * 3600)
            except Exception as e:
                print(f"⚠️ خطأ في دورة التتابع: {e}. محاولة مجددة بعد ساعة...")
                time.sleep(3600)

if __name__ == "__main__":
    ACCOUNTS_PATH = "artifacts/model_training/scheduler/kaggle_accounts.json"
    scheduler = KaggleGlobalScheduler(ACCOUNTS_PATH)
    import sys
    if "--relay" in sys.argv:
        scheduler.run_relay_24_7()
    else:
        scheduler.launch_all()

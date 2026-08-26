import os
import json
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
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": []
        }
        with open(os.path.join(target_dir, "kernel-metadata.json"), "w") as f:
            json.dump(metadata, f)

    def launch_all(self):
        """إطلاق جميع العقد عبر الحسابات السبعة."""
        print(f"🚀 بدء إطلاق السرب العالمي عبر {len(self.accounts)} حسابات...")
        results = []
        
        # إنشاء ملف تشغيل واحد يحتوي على كافة التبعيات لتجنب مشاكل الاستيراد
        upload_dir = "kaggle_upload"
        os.makedirs(upload_dir, exist_ok=True)
        
        with open(os.path.join(upload_dir, "kaggle_run.py"), "w") as out:
            # إضافة التبعيات الأساسية
            for dep in ["ai/distributed_trainer.py", "ai/gradient_mesh.py", "ai/living_mesh.py"]:
                if os.path.exists(dep):
                    with open(dep, "r") as f:
                        content = f.read()
                        # إزالة الاستيرادات المحلية التي ستسبب مشاكل
                        content = content.replace("from ai.", "from ")
                        out.write(f"\n# --- FROM {dep} ---\n")
                        out.write(content)
            
            # إضافة كود التشغيل الأساسي
            with open("kaggle_run.py", "r") as f:
                content = f.read()
                content = content.replace("from ai.", "from ")
                out.write("\n# --- MAIN RUNNER ---\n")
                out.write(content)
        
        for i, acc in enumerate(self.accounts):
            username = acc['username']
            key = acc['key']
            node_id = f"global_{i+1}"
            
            print(f"📦 تجهيز العقدة {node_id} للحساب {username}...")
            self._setup_credentials(username, key)
            self.create_kernel_metadata(username, node_id, target_dir=upload_dir)
            
            try:
                # الإطلاق الفعلي عبر Kaggle CLI
                process = subprocess.run(
                    ["kaggle", "kernels", "push", "-p", upload_dir],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print(f"✅ تم إرسال كود الإطلاق للعقدة {node_id} بنجاح.")
                results.append({"node": node_id, "user": username, "status": "Active", "url": f"https://www.kaggle.com/{username}/nsm-node-{node_id}"})
            except subprocess.CalledProcessError as e:
                print(f"❌ فشل إطلاق العقدة {node_id} للحساب {username}: {e.stderr}")
                results.append({"node": node_id, "user": username, "status": "Failed", "error": e.stderr})
            
            time.sleep(5) # فاصل زمني لضمان معالجة Kaggle للطلبات
            
        return results

if __name__ == "__main__":
    # مسار ملف الحسابات الذي تم اكتشافه
    ACCOUNTS_PATH = "artifacts/model_training/scheduler/kaggle_accounts.json"
    scheduler = KaggleGlobalScheduler(ACCOUNTS_PATH)
    scheduler.launch_all()

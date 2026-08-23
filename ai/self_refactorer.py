
import os
import shutil
import py_compile
from datetime import datetime

class SelfRefactorer:
    """🛠️ محرك التطوير الذاتي: يسمح للوكيل بتعديل الكود المصدري للمشروع بأمان."""
    
    def __init__(self, repo_path):
        self.repo_path = repo_path
        self.backups_dir = os.path.join(repo_path, "artifacts", "backups")
        os.makedirs(self.backups_dir, exist_ok=True)

    def refactor_file(self, file_path, new_content, reason):
        """تعديل ملف مع النسخ الاحتياطي والتحقق من الصحة."""
        full_path = os.path.join(self.repo_path, file_path)
        if not os.path.exists(full_path):
            return {"status": "error", "message": f"File {file_path} not found."}

        # 1. نسخة احتياطية
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backups_dir, f"{os.path.basename(file_path)}.{timestamp}.bak")
        shutil.copy2(full_path, backup_path)

        # 2. تطبيق التعديل مؤقتاً للتحقق
        temp_path = full_path + ".tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            
            # التحقق من Syntax
            py_compile.compile(temp_path, doraise=True)
            
            # 3. اعتماد التعديل
            shutil.move(temp_path, full_path)
            return {
                "status": "success", 
                "message": f"Refactored {file_path} successfully.",
                "backup": backup_path,
                "reason": reason
            }
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return {"status": "error", "message": f"Refactoring failed: {str(e)}"}

self_refactorer = SelfRefactorer("/home/ubuntu/Neural-Service-Mesh")

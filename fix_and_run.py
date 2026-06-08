"""
شغّل هذا الملف بدلاً من main.py مباشرة:
!python fix_and_run.py --mode api
"""
import shutil, os, sys

# 1. امسح الـ cache القديم
for root, dirs, files in os.walk("."):
    for d in dirs:
        if d == "__pycache__":
            shutil.rmtree(os.path.join(root, d))
            print(f"🗑️  Cleared: {os.path.join(root, d)}")

# 2. تحقق من الملف المكسور وأصلحه تلقائياً
init_path = "knowledge_sources/__init__.py"
with open(init_path, "r") as f:
    content = f.read()

if "score_manager" in content:
    content = content.replace(
        "from knowledge_sources.score_manager",
        "from knowledge_sources.source_manager"
    )
    with open(init_path, "w") as f:
        f.write(content)
    print("✅ Fixed: knowledge_sources/__init__.py")
else:
    print("✅ __init__.py already correct")

# 3. شغّل main.py بنفس الـ arguments
sys.argv[0] = "main.py"
exec(open("main.py").read())

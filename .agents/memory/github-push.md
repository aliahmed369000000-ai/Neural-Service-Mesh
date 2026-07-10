---
name: GitHub Push Blocked
description: git add/commit محظوران في المُوكِّل الرئيسي، والحلول المتاحة
---

# قيود Git في المُوكِّل الرئيسي

## المشكلة

`git add` و`git commit` ترمي خطأ "Destructive git operations are not allowed in the main agent."
`git push` بدون force يتجاوز المهلة الزمنية.

## الحلول

1. **push_to_github.sh** — سكريبت في جذر المشروع يستخدم `GITHUB_PERSONAL_ACCESS_TOKEN`
   ```bash
   bash push_to_github.sh "رسالة الـ commit"
   ```
   لكنه يحتاج git add وgit commit اللتين هما محظورتان أيضاً.

2. **project_tasks skill** — المسار الصحيح لعمليات git في بيئة معزولة

3. **Auto-commit** — Replit يُنفّذ commit تلقائياً عند mark_task_complete؛ بعده يمكن الدفع.

## Remote URL

`https://github.com/aliahmed369000000-ai/Neural-Service-Mesh`
Token: `GITHUB_PERSONAL_ACCESS_TOKEN` (موجود في الـ secrets)

**Why:** sandbox الـ main agent يعترض git calls الخطرة لحماية النظام.
**How to apply:** للدفع لـ GitHub، استخدم project_tasks أو اطلب من المستخدم تشغيل push_to_github.sh بعد auto-commit.

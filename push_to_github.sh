#!/usr/bin/env bash
# push_to_github.sh — يدفع التغييرات إلى GitHub باستخدام GITHUB_PERSONAL_ACCESS_TOKEN
# الاستخدام: bash push_to_github.sh "رسالة الـ commit"

set -e

COMMIT_MSG="${1:-auto: update Nova system}"
REPO="aliahmed369000000-ai/Neural-Service-Mesh"
TOKEN="${GITHUB_PERSONAL_ACCESS_TOKEN}"

if [ -z "$TOKEN" ]; then
  echo "❌ GITHUB_PERSONAL_ACCESS_TOKEN غير موجود في البيئة"
  exit 1
fi

# ضبط الهوية
git config user.email "nova@aurora-labs.ai" 2>/dev/null || true
git config user.name "Nova System" 2>/dev/null || true

# ضبط الـ remote مع التوكن
git remote set-url origin "https://${TOKEN}@github.com/${REPO}.git"

# إضافة وحفظ
git add -A
git commit -m "$COMMIT_MSG" || echo "⚠ لا يوجد تغييرات جديدة"
git push origin main

echo "✅ تم الدفع إلى GitHub بنجاح"

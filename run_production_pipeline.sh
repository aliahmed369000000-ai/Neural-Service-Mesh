#!/usr/bin/env bash
#
# run_production_pipeline.sh  [production-7b-llm]
# =================================================
# أنبوب الإنتاج الكامل من طرف لطرف لتدريب YemeniDecoder (Qwen2.5-7B + LoRA):
#
#   1) جلب/تحديث بيانات HF وتحويلها لصيغة NSM      (data/fetch_hf_yemeni_dataset.py)
#   2) تدريب QLoRA على النموذج الأساسي 7B            (train_production_yemeni.py)
#   3) الأوزان تُحفظ مباشرة في models/yemeni_qwen7b_lora/ (مسار السكربت الحالي)
#   4) حذف نهائي للبيانات الخام وكاش HF
#   5) commit + push تلقائي لفرع production-7b-llm
#
# ⚠️ متطلبات إلزامية: GPU ≥24GB VRAM، اتصال شبكة بـ huggingface.co،
#    بيئة Python مع: torch transformers accelerate peft bitsandbytes trl datasets
#    لن يعمل على Streamlit Community Cloud أو أي بيئة CPU فقط.
#
# ⚠️ تحذير تشغيلي مهم: هذا السكربت يحذف نهائياً (rm -rf) البيانات الخام
#    وكاش HF بعد التدريب مباشرة، ثم يدفع الأوزان تلقائياً بدون مراجعة بشرية.
#    لا يوجد نسخ احتياطي تلقائي للبيانات الخام قبل حذفها ولا نقطة توقف
#    قبل الدفع النهائي — هذا سلوك مقصود حسب طلبك للأتمتة الكاملة، لكن:
#      - استخدم --keep-raw-data لو تريد الاحتفاظ بالبيانات لإعادة استخدامها.
#      - استخدم --no-push لو تريد مراجعة الأوزان محلياً قبل الدفع يدوياً.
#      - استخدم --dry-run لطباعة الخطوات فقط بدون تنفيذ فعلي.
#
# الاستخدام:
#   HF_TOKEN=xxx GITHUB_TOKEN=xxx ./run_production_pipeline.sh \
#       --hf-dataset org/yemeni-dialect-instructions
#
set -euo pipefail

# ══════════════════════════════════════════════════════════════════════════
# الإعدادات الافتراضية
# ══════════════════════════════════════════════════════════════════════════
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

HF_DATASETS=()
SPLIT="train"
DATASET_OUT="data/yemeni_production_instructions.jsonl"
CACHE_DIR="${NSM_FETCH_CACHE_DIR:-.hf_cache}"
OUTPUT_DIR="models/yemeni_qwen7b_lora"   # مطابق لمسار train_production_yemeni.py الحالي
EPOCHS=3
BASE_BRANCH="production-7b-llm"
KEEP_RAW_DATA=0
NO_PUSH=0
DRY_RUN=0
GIT_USER_NAME="NSM Bot"
GIT_USER_EMAIL="nsm-bot@users.noreply.github.com"

log() { echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { echo -e "[FATAL] $*" >&2; exit 1; }
run() {
  # يطبع الأمر دائماً؛ ينفّذه فقط لو DRY_RUN=0
  log "▶ $*"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    eval "$@"
  fi
}

# ══════════════════════════════════════════════════════════════════════════
# تحليل المعطيات
# ══════════════════════════════════════════════════════════════════════════
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-dataset) HF_DATASETS+=("$2"); shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --branch) BASE_BRANCH="$2"; shift 2 ;;
    --keep-raw-data) KEEP_RAW_DATA=1; shift ;;
    --no-push) NO_PUSH=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) die "معطى غير معروف: $1 (استخدم --help)" ;;
  esac
done

[[ ${#HF_DATASETS[@]} -gt 0 ]] || die "يجب تحديد --hf-dataset واحد على الأقل"

log "════════════════════════════════════════════════════"
log " NSM Production Pipeline — production-7b-llm"
log "════════════════════════════════════════════════════"
log "مجموعات البيانات : ${HF_DATASETS[*]}"
log "مجلد الإخراج      : $OUTPUT_DIR"
log "الفرع الهدف       : $BASE_BRANCH"
log "DRY_RUN           : $DRY_RUN"

# ══════════════════════════════════════════════════════════════════════════
# فحص مبكر للاعتماديات — فشل واضح بدل تتبع غامض في منتصف الأنبوب
# ══════════════════════════════════════════════════════════════════════════
command -v python3 >/dev/null 2>&1 || die "python3 غير موجود في PATH"
command -v git >/dev/null 2>&1 || die "git غير موجود في PATH"
if [[ "$DRY_RUN" -eq 0 ]]; then
  python3 -c "import torch" >/dev/null 2>&1 || die "حزمة torch غير مثبتة — pip install torch"
  python3 -c "import datasets" >/dev/null 2>&1 || die "حزمة datasets غير مثبتة — pip install datasets"
  python3 -c "import peft, trl, bitsandbytes" >/dev/null 2>&1 || \
    die "حزم peft/trl/bitsandbytes غير مثبتة — pip install peft trl bitsandbytes"
  python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || \
    die "لا يوجد GPU متاح (torch.cuda.is_available() == False). التدريب على 7B يتطلب GPU."
else
  log "(--dry-run) تخطي فحص حزم torch/datasets/peft/trl/bitsandbytes وGPU"
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[[ "$CURRENT_BRANCH" == "$BASE_BRANCH" ]] || \
  die "أنت على الفرع '$CURRENT_BRANCH' وليس '$BASE_BRANCH'. نفّذ: git checkout $BASE_BRANCH"

git diff --quiet && git diff --cached --quiet || \
  die "توجد تغييرات غير محفوظة (uncommitted) في المستودع — احفظها أو تراجع عنها قبل تشغيل الأنبوب."

# ══════════════════════════════════════════════════════════════════════════
# الخطوة ١: جلب البيانات
# ══════════════════════════════════════════════════════════════════════════
log "── الخطوة 1/5: جلب بيانات Hugging Face ──"
HF_ARGS=()
for ds in "${HF_DATASETS[@]}"; do HF_ARGS+=(--hf-dataset "$ds"); done
run python3 data/fetch_hf_yemeni_dataset.py \
  "${HF_ARGS[@]}" \
  --split "$SPLIT" \
  --output "$DATASET_OUT" \
  --cache-dir "$CACHE_DIR"

# ══════════════════════════════════════════════════════════════════════════
# الخطوة ٢: التدريب (QLoRA على 7B)
# ══════════════════════════════════════════════════════════════════════════
log "── الخطوة 2/5: تدريب QLoRA (Qwen2.5-7B) ──"
run python3 train_production_yemeni.py \
  --dataset "$DATASET_OUT" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS"

# ══════════════════════════════════════════════════════════════════════════
# الخطوة ٣: التحقق من وجود الأوزان الناتجة فعلياً قبل أي حذف/دفع
# ══════════════════════════════════════════════════════════════════════════
log "── الخطوة 3/5: التحقق من الأوزان المُصدَّرة ──"
if [[ "$DRY_RUN" -eq 0 ]]; then
  [[ -d "$OUTPUT_DIR" ]] || die "مجلد الأوزان '$OUTPUT_DIR' غير موجود — فشل التدريب على الأرجح. توقف قبل الحذف."
  find "$OUTPUT_DIR" -type f \( -iname "*.safetensors" -o -iname "adapter_model*" \) -print -quit | \
    grep -q . || die "لا توجد ملفات أوزان LoRA داخل '$OUTPUT_DIR' — توقف قبل الحذف والدفع."
  log "✓ أوزان LoRA موجودة في $OUTPUT_DIR"
fi

# ══════════════════════════════════════════════════════════════════════════
# الخطوة ٤: حذف نهائي للبيانات الخام وكاش HF
# ══════════════════════════════════════════════════════════════════════════
log "── الخطوة 4/5: حذف البيانات الخام وكاش HF ──"
if [[ "$KEEP_RAW_DATA" -eq 1 ]]; then
  log "تم تفعيل --keep-raw-data — تخطي الحذف."
else
  run rm -rf "\"$CACHE_DIR\""
  run rm -f "\"$DATASET_OUT\""
  run rm -f "\"data/yemeni_production_instructions.chatml.jsonl\""
  log "✓ حُذفت البيانات الخام وكاش HF نهائياً"
fi

# ══════════════════════════════════════════════════════════════════════════
# الخطوة ٥: commit + push تلقائي
# ══════════════════════════════════════════════════════════════════════════
log "── الخطوة 5/5: commit + push إلى $BASE_BRANCH ──"
if [[ "$NO_PUSH" -eq 1 ]]; then
  log "تم تفعيل --no-push — الأوزان محفوظة محلياً في $OUTPUT_DIR فقط. لا commit ولا push."
else
  run git config user.name "\"$GIT_USER_NAME\""
  run git config user.email "\"$GIT_USER_EMAIL\""
  run git add "\"$OUTPUT_DIR\""
  if [[ "$DRY_RUN" -eq 0 ]] && git diff --cached --quiet; then
    log "لا توجد تغييرات جديدة في $OUTPUT_DIR — لا شيء يُرفع."
  else
    COMMIT_MSG="تدريب إنتاجي جديد: أوزان LoRA محدّثة لـ YemeniDecoder (Qwen2.5-7B, $EPOCHS epochs)"
    run git commit -m "\"$COMMIT_MSG\""
    run git push origin "$BASE_BRANCH"

    log "التحقق الفعلي من نجاح الدفع عبر git ls-remote..."
    LOCAL_SHA="$(git rev-parse HEAD)"
    REMOTE_SHA="$(git ls-remote origin "refs/heads/$BASE_BRANCH" | awk '{print $1}')"
    if [[ "$DRY_RUN" -eq 0 ]]; then
      [[ "$LOCAL_SHA" == "$REMOTE_SHA" ]] || die "فشل التحقق: SHA المحلي ($LOCAL_SHA) لا يطابق البعيد ($REMOTE_SHA)"
      log "✓ تم التحقق: الفرع البعيد $BASE_BRANCH يطابق $LOCAL_SHA"
    fi
  fi
fi

log "════════════════════════════════════════════════════"
log " ✓ اكتمل الأنبوب بنجاح"
log "════════════════════════════════════════════════════"

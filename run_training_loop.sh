#!/bin/bash
# حلقة تدريب كاملة حتى DONE_ALL
# تُشغَّل في الخلفية: nohup bash run_training_loop.sh > logs/training_full.log 2>&1 &

LOG="logs/training_full.log"
mkdir -p logs

echo "[$(date '+%H:%M:%S')] بدء حلقة التدريب الكاملة — النموذج 200M" | tee -a "$LOG"

RUN=0
while true; do
    RUN=$((RUN + 1))
    echo "" | tee -a "$LOG"
    echo "════════════════════════════════════════" | tee -a "$LOG"
    echo "[$(date '+%H:%M:%S')] تشغيل رقم $RUN" | tee -a "$LOG"
    echo "════════════════════════════════════════" | tee -a "$LOG"

    output=$(python3 train_batch_v3.py 2>&1)
    echo "$output" | tee -a "$LOG"

    if echo "$output" | grep -q "DONE_ALL"; then
        echo "" | tee -a "$LOG"
        echo "✅ [$(date '+%H:%M:%S')] التدريب مكتمل 100% — DONE_ALL" | tee -a "$LOG"
        break
    fi

    if echo "$output" | grep -q "ABORT"; then
        echo "❌ [$(date '+%H:%M:%S')] التدريب توقف بخطأ — راجع السجل" | tee -a "$LOG"
        break
    fi
done

echo "[$(date '+%H:%M:%S')] انتهت الحلقة" | tee -a "$LOG"

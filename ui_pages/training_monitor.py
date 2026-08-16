"""
لوحة تقدم التدريب الحي — SurahChain على Kaggle
==============================================
تبويب فرعي داخل «التدريب والعمليات» يعرض حالة التدريب الفعلي من كيرنلات Kaggle
مباشرة في Streamlit: العصر/الخسارة/أفضل خسارة/زمن الجلسة + منحنى خسارة +
حالة الكيرنل من Kaggle API — دون الحاجة لفتح Kaggle أو قراءة سجلاته (buffered).

المصدر: ملفات progress_{TAG}.json المرفوعة تلقائيًا كل عصر من
train_pretrain_torch.py (عبر _write_progress + _upload_checkpoint) إلى
experiments/surah_chain_network/checkpoints/ على GitHub.
"""
from __future__ import annotations


def render_training_monitor() -> None:
    try:
        from ai.surah_training_monitor import render_live_training_dashboard
        render_live_training_dashboard()
    except Exception as e:
        import streamlit as st
        st.error(f"تعذّر تحميل لوحة التدريب الحي: {type(e).__name__}: {e}")

# -*- coding: utf-8 -*-
"""🚀 NSM Kaggle Runner — سكربت تشغيل السرب السيادي على Kaggle.

يقوم هذا السكربت بتنسيق الـ 7 بطاقات GPU، تحميل البيانات، وبدء جلسة التدريب الجماعية.
"""
import os
import torch
import sys

# إضافة المسار الحالي للمشروع
sys.path.append(os.getcwd())

from ai.distributed_trainer import NSMDistributedTrainer
from ai.arabic_transformer import ArabicTransformer # يفترض وجود بنية Surah 4096 هنا

def start_swarm_session():
    print("🌟 Starting NSM Sovereign Swarm Session on Kaggle...")
    
    # 1. إعداد النموذج (Surah 4096)
    # ملاحظة: نستخدم بارامترات مصغرة للاختبار إذا لم تتوفر الأوزان الضخمة
    model_config = {
        "d_model": 4096,
        "n_layers": 114,
        "n_heads": 32,
        "vocab_size": 50257
    }
    
    print(f"🏗️ Building Surah 4096 Model ({model_config['n_layers']} layers)...")
    model = ArabicTransformer(**model_config)
    
    # 2. تحميل البيانات من Kaggle Input
    # افترض وجود بيانات في /kaggle/input/nsm-data/
    data_path = "/kaggle/input/nsm-data/train_data.pt"
    if os.path.exists(data_path):
        print(f"📂 Loading training data from {data_path}...")
        dataset = torch.load(data_path)
    else:
        print("⚠️ Training data not found. Using synthetic data for swarm initialization.")
        dataset = [(torch.randn(1, 1024), torch.randint(0, 50257, (1, 1024))) for _ in range(100)]

    # 3. تهيئة محرك التدريب الموزع
    trainer = NSMDistributedTrainer(model, dataset)
    trainer.setup()
    
    # 4. بدء الجلسة الجماعية
    print("🚀 Swarm Integrated. Beginning Collective Training Session...")
    for epoch in range(10):
        for i, batch in enumerate(dataset):
            loss = trainer.train_step(batch, step_idx=i)
            if i % 10 == 0:
                print(f"📊 Epoch {epoch} | Step {i} | Loss: {loss:.4f}")
        
        # حفظ الأوزان في نهاية كل Epoch
        trainer.save_checkpoint(tag=f"surah_4096_epoch_{epoch}")

if __name__ == "__main__":
    start_swarm_session()

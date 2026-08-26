# -*- coding: utf-8 -*-
"""🚀 NSM Distributed Trainer — محرك التدريب الموزع لـ Surah 4096.

يستخدم هذا المحرك DeepSpeed ZeRO-3 لدمج قوة 7 بطاقات GPU وتوزيع الذاكرة
بشكل يسمح بتدريب نماذج عملاقة تتجاوز سعة البطاقة الواحدة.
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# محاولة استيراد DeepSpeed (يجب تثبيته في بيئة التدريب)
try:
    import deepspeed
    from deepspeed.ops.adam import FusedAdam
except ImportError:
    deepspeed = None

class NSMDistributedTrainer:
    def __init__(self, model, train_dataset, config=None):
        self.model = model
        self.train_dataset = train_dataset
        self.config = config or self._default_config()
        self.world_size = torch.cuda.device_count()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        print(f"📡 Initializing NSM Distributed Swarm with {self.world_size} GPUs...")

    def _default_config(self):
        """إعدادات DeepSpeed ZeRO-3 المثالية لـ 7 بطاقات GPU."""
        return {
            "train_batch_size": 32,
            "steps_per_print": 10,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 3e-7
                }
            },
            "zero_optimization": {
                "stage": 3,  # ZeRO-3 لتقسيم الأوزان والذاكرة بالكامل
                "offload_optimizer": {"device": "cpu"},
                "offload_param": {"device": "cpu"},
                "overlap_comm": True,
                "contiguous_gradients": True,
                "reduce_bucket_size": 5e8,
                "stage3_prefetch_bucket_size": 5e8,
                "stage3_param_persistence_threshold": 1e6
            },
            "fp16": {"enabled": True}
        }

    def setup(self):
        """إعداد بيئة التدريب الموزع."""
        if deepspeed:
            # استخدام DeepSpeed للتدريب العملاق
            self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
                config=self.config,
                model=self.model,
                model_parameters=self.model.parameters()
            )
        else:
            # التراجع إلى DDP التقليدي إذا لم يتوفر DeepSpeed
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
            self.model = self.model.to(self.local_rank)
            self.model = DDP(self.model, device_ids=[self.local_rank])
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def train_step(self, batch):
        """خطوة تدريب واحدة موزعة."""
        inputs, labels = batch
        
        if deepspeed:
            outputs = self.model_engine(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            self.model_engine.backward(loss)
            self.model_engine.step()
        else:
            inputs, labels = inputs.to(self.local_rank), labels.to(self.local_rank)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
        return loss.item()

    def save_checkpoint(self, path):
        """حفظ الأوزان الموزعة."""
        if self.local_rank == 0:
            if deepspeed:
                self.model_engine.save_checkpoint(path)
            else:
                torch.save(self.model.state_dict(), path)
            print(f"💾 Checkpoint saved at {path}")

if __name__ == "__main__":
    # مثال توضيحي للتشغيل
    print("🛠️ NSM Distributed Trainer Ready for Surah 4096.")
    print("Usage: deepspeed ai/distributed_trainer.py --num_gpus 7")

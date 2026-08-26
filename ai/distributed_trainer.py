# -*- coding: utf-8 -*-
"""🚀 NSM Distributed Trainer — محرك التدريب الموزع لـ Surah 4096.

يستخدم هذا المحرك DeepSpeed ZeRO-3 لدمج قوة 7 بطاقات GPU وتوزيع الذاكرة
بشكل يسمح بتدريب نماذج عملاقة تتجاوز سعة البطاقة الواحدة.
"""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from ai.security_guard import NSMSecurityGuard

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
        self.security = NSMSecurityGuard()
        
        print(f"🛡️ Security Protocol Active. Initializing NSM Distributed Swarm with {self.world_size} GPUs...")

    def _default_config(self):
        """إعدادات DeepSpeed ZeRO-3 للتحسين الهجومي (Aggressive Optimization) لـ 7 بطاقات GPU."""
        return {
            "train_batch_size": 128,
            "train_micro_batch_size_per_gpu": 16,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 1e-4,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": 1e-6
                }
            },
            "zero_optimization": {
                "stage": 3,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "none"
                },
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "gather_16bit_weights_on_model_save": True
            },
            "gradient_clipping": 1.0,
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "wall_clock_breakdown": False
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
        """حفظ الأوزان الموزعة وتأمينها."""
        if self.local_rank == 0:
            if deepspeed:
                self.model_engine.save_checkpoint(path)
                # تشفير الأوزان بعد الحفظ
                self.security.encrypt_weights(os.path.join(path, "mp_rank_00_model_states.pt"))
            else:
                torch.save(self.model.state_dict(), path)
                self.security.encrypt_weights(path)
            print(f"💾 Checkpoint saved and secured at {path}")

if __name__ == "__main__":
    # مثال توضيحي للتشغيل
    print("🛠️ NSM Distributed Trainer Ready for Surah 4096.")
    print("Usage: deepspeed ai/distributed_trainer.py --num_gpus 7")

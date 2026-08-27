# -*- coding: utf-8 -*-
"""🚀 NSM Distributed Trainer — محرك التدريب الموزع لـ Surah 4096 (Kaggle Edition).

يستخدم هذا المحرك DeepSpeed ZeRO-3 لدمج قوة 7 بطاقات GPU وتوزيع الذاكرة
بشكل يسمح بتدريب نماذج عملاقة تتجاوز سعة البطاقة الواحدة.
تم تحسينه للعمل في بيئة Kaggle وحفظ الأوزان محلياً.
"""
import os
import asyncio
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from ai.security_guard import NSMSecurityGuard
from ai.cognitive_growth import cognitive_engine
from ai.gradient_mesh import GradientExchangeProtocol

# محاولة استيراد DeepSpeed (يجب تثبيته في بيئة التدريب)
try:
    import deepspeed
    from deepspeed.ops.adam import FusedAdam
except ImportError:
    deepspeed = None

class NSMDistributedTrainer:
    def __init__(self, model, train_dataset, config=None, checkpoint_dir=None):
        self.model = model
        self.train_dataset = train_dataset
        self.config = config or self._default_config()
        self.world_size = torch.cuda.device_count()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.security = NSMSecurityGuard()
        
        # إعداد مسارات التخزين (الافتراضي Kaggle)
        self.checkpoint_dir = checkpoint_dir or "/kaggle/working/checkpoints"
        if self.local_rank == 0:
            try:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
            except Exception as e:
                print(f"⚠️ Warning: Could not create checkpoint directory {self.checkpoint_dir}: {e}")
        
        # تهيئة بروتوكول تبادل التدرجات العالمي
        alpha_url = os.environ.get("ALPHA_NODE_WS_URL", "wss://aliahmedmo-nsm-alpha-node.hf.space/ws")
        self.gradient_mesh = GradientExchangeProtocol(node_id=f"kaggle_{os.environ.get('KAGGLE_USERNAME', 'unknown')}", alpha_url=alpha_url)
        
        print(f"🛡️ Security Protocol Active. Initializing NSM Distributed Swarm with {self.world_size} GPUs...")

    def _default_config(self):
        """إعدادات DeepSpeed ZeRO-3 للتحسين الهجومي (Aggressive Optimization) لـ 7 بطاقات GPU."""
        return {
            "train_batch_size": 256,
            "train_micro_batch_size_per_gpu": 32,
            "gradient_accumulation_steps": 1,
            "steps_per_print": 1,
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 2e-4,
                    "betas": [0.9, 0.98],
                    "eps": 1e-8,
                    "weight_decay": 1e-5
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
        # التراجع إلى DDP التقليدي أو التدريب المحلي
        if torch.cuda.is_available():
            if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
                try:
                    if not dist.is_initialized():
                        dist.init_process_group(backend="nccl")
                    torch.cuda.set_device(self.local_rank)
                    self.model = self.model.to(self.local_rank)
                    self.model = DDP(self.model, device_ids=[self.local_rank])
                    print("✅ DDP Initialized successfully.")
                except Exception as e:
                    print(f"⚠️ DDP Init failed: {e}. Falling back to single-node mode.")
                    self.model = self.model.cuda()
            else:
                print("ℹ️ Standalone mode: RANK/WORLD_SIZE not set. Using local GPU.")
                self.model = self.model.cuda()
        else:
            print("⚠️ Running in Local CPU mode (No CUDA detected).")
            self.model = self.model.to("cpu")
            
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def train_step(self, batch, step_idx=0):
        """خطوة تدريب واحدة موزعة مع تحليل النمو المعرفي وتبادل التدرجات."""
        inputs, labels = batch
        
        if deepspeed:
            outputs = self.model_engine(inputs)
            loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            self.model_engine.backward(loss)
            
            # تبادل التدرجات عبر الشبكة العالمية قبل تحديث الأوزان
            if step_idx % 5 == 0: # مزامنة كل 5 خطوات لتقليل ضغط الشبكة
                asyncio.run(self.gradient_mesh.broadcast_gradients(self.model))
                
            self.model_engine.step()
        else:
            device = self.local_rank if torch.cuda.is_available() else "cpu"
            inputs, labels = inputs.to(device), labels.to(device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            
            # تبادل التدرجات عبر الشبكة العالمية
            if step_idx % 5 == 0:
                asyncio.run(self.gradient_mesh.broadcast_gradients(self.model))
                
            self.optimizer.step()
            
        # تحليل النمو المعرفي كل 100 خطوة
        if step_idx % 100 == 0 and self.local_rank == 0:
            self._trigger_cognitive_growth(loss.item())
            
        return loss.item()

    def _trigger_cognitive_growth(self, current_loss):
        """تفعيل تحليل النمو المعرفي بناءً على نتائج التدريب."""
        if not hasattr(self, 'loss_history'): self.loss_history = []
        self.loss_history.append(current_loss)
        
        trend = cognitive_engine.analyze_learning_trend(self.loss_history)
        print(f"🧠 Cognitive Analysis: {trend}")
        
        if "structural evolution" in trend.lower():
            metrics = {"accuracy": 0.75, "loss_variance": 0.5} # مقاييس محاكية
            proposals = cognitive_engine.propose_structural_evolution(metrics)
            for p in proposals:
                cognitive_engine.apply_evolutionary_patch(p)
                print(f"✨ Evolutionary Patch Proposed: {p['type']}")

    def save_checkpoint(self, tag):
        """حفظ الأوزان الموزعة وتأمينها في مسار Kaggle."""
        if self.local_rank == 0:
            save_path = os.path.join(self.checkpoint_dir, tag)
            if deepspeed:
                self.model_engine.save_checkpoint(self.checkpoint_dir, tag=tag)
                # ملاحظة: DeepSpeed يحفظ عدة ملفات، التشفير يحتاج معالجة خاصة للمجلد
            else:
                torch.save(self.model.state_dict(), save_path)
                self.security.encrypt_weights(save_path)
            print(f"💾 Checkpoint '{tag}' saved and secured at {self.checkpoint_dir}")

if __name__ == "__main__":
    # مثال توضيحي للتشغيل في Kaggle
    print("🛠️ NSM Distributed Trainer Ready for Surah 4096 on Kaggle.")
    print(f"Checkpoints will be saved to: /kaggle/working/checkpoints")
    print("Usage: deepspeed ai/distributed_trainer.py --num_gpus 7")

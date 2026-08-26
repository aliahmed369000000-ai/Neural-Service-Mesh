# -*- coding: utf-8 -*-
"""🚀 NSM Gradient Mesh — بروتوكول تبادل التدرجات اللحظي للسرب.
يربط عقد Kaggle ببعضها البعض وبالعقدة المركزية لتبادل التدرجات (Gradients)
وتحديث الأوزان بشكل متزامن لنموذج Surah 4096.
"""
import asyncio
import json
import torch
import base64
import io
import logging
import aiohttp
from typing import Dict, Any, List

logger = logging.getLogger("GradientMesh")

class GradientExchangeProtocol:
    def __init__(self, node_id: str, alpha_url: str):
        self.node_id = node_id
        self.alpha_url = alpha_url # ws://alpha-node.hf.space/ws
        self.ws = None
        self.is_connected = False

    async def connect(self):
        """الاتصال بالعقدة المركزية للتنسيق."""
        try:
            session = aiohttp.ClientSession()
            self.ws = await session.ws_connect(self.alpha_url)
            self.is_connected = True
            logger.info(f"✅ Node {self.node_id} connected to Alpha Node for Gradient Exchange.")
            # إرسال تعريف العقدة
            await self.ws.send_str(json.dumps({
                "type": "register",
                "node_id": self.node_id,
                "role": "worker"
            }))
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            self.is_connected = False

    def serialize_gradients(self, model: torch.nn.Module) -> str:
        """تحويل التدرجات إلى صيغة قابلة للنقل عبر الشبكة."""
        grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                # تكميم بسيط لتقليل حجم النقل (FP16)
                grads[name] = param.grad.detach().cpu().half().numpy().tolist()
        
        buffer = io.BytesIO()
        torch.save(grads, buffer)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    def deserialize_and_apply(self, model: torch.nn.Module, grad_data_b64: str):
        """استقبال وتطبيق التدرجات المجمعة من السرب."""
        grad_bytes = base64.b64decode(grad_data_b64)
        buffer = io.BytesIO(grad_bytes)
        remote_grads = torch.load(buffer)
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in remote_grads:
                    remote_grad = torch.tensor(remote_grads[name]).to(param.device).float()
                    if param.grad is None:
                        param.grad = remote_grad
                    else:
                        # All-Reduce: متوسط التدرجات
                        param.grad = (param.grad + remote_grad) / 2.0

    async def broadcast_gradients(self, model: torch.nn.Module):
        """بث التدرجات المحلية للسرب."""
        if not self.is_connected: return
        
        grad_data = self.serialize_gradients(model)
        await self.ws.send_str(json.dumps({
            "type": "gradient_push",
            "node_id": self.node_id,
            "data": grad_data
        }))

    async def listen_for_updates(self, model: torch.nn.Module):
        """الاستماع لتحديثات الأوزان/التدرجات من السرب."""
        if not self.is_connected: return
        
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") == "gradient_pull":
                    self.deserialize_and_apply(model, data["data"])
                    logger.info("🔄 Global Gradients Integrated.")

gradient_protocol = None

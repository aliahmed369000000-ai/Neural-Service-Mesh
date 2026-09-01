# -*- coding: utf-8 -*-
"""🚀 NSM Gradient Mesh — بروتوكول تبادل التدرجات اللحظي للسرب (P2P).
يربط عقد التدريب ببعضها عبر بروتوكول living_mesh اللامركزي (send_to_peer)
بدل الاعتماد على نقطة مركزية واحدة (alpha_url). يبقى دعم alpha_url اختيارياً
للتوافق الخلفي فقط.
"""
import asyncio
import json
import torch
import base64
import io
import logging
import time
import random
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger("GradientMesh")

# استيراد متأخر لتجنب مشاكل الاستيراد الدائري عند التحميل
def _get_living_mesh_node_class():
    try:
        from ai.living_mesh import LivingMeshNode
        return LivingMeshNode
    except Exception as e:
        logger.warning(f"⚠️ LivingMeshNode unavailable ({e}); P2P path disabled")
        return None


class GradientExchangeProtocol:
    def __init__(
        self,
        node_id: str,
        alpha_url: str = None,
        mesh_node=None,
        host: str = "127.0.0.1",
        port: int = None,
    ):
        """
        node_id: معرف العقدة.
        alpha_url: (اختياري/قديم) رابط WebSocket مركزي — يُستخدم فقط إذا لم يتوفر mesh_node.
        mesh_node: كائن LivingMeshNode جاهز (مفضل). إن لم يُمرَّر يُنشأ واحد خفيف عند الحاجة.
        host/port: يُستخدمان عند إنشاء mesh_node داخلياً.
        """
        self.node_id = node_id
        self.alpha_url = alpha_url
        self.ws = None
        self.is_connected = False
        self._model_callback: Optional[Callable] = None
        self._pending_grads: List[Dict[str, Any]] = []

        self.mesh_node = mesh_node
        if self.mesh_node is None and (host is not None or port is not None):
            # إنشاء عقدة living mesh خفيفة لاستخدام بروتوكول send_to_peer
            LivingMeshNode = _get_living_mesh_node_class()
            if LivingMeshNode is not None:
                try:
                    self.mesh_node = LivingMeshNode(node_id=node_id, host=host, port=port or 0)
                    self.mesh_node.join_network()
                except Exception as e:
                    logger.warning(f"⚠️ LivingMesh join_network soft-fail: {e}")
                    self.mesh_node = None

        # تسجيل معالج التدرجات داخل living_mesh إن وُجدت العقدة
        if self.mesh_node is not None:
            self._register_gradient_handler()

    def _register_gradient_handler(self):
        """يربط استقبال رسائل gradient_push بمعالج محلي داخل الـ mesh_node."""
        original = getattr(self.mesh_node, "_process_secure_message", None)
        if original is None or getattr(self.mesh_node, "_gradient_handler_installed", False):
            return

        protocol = self

        async def wrapped_process(data, websocket=None):
            try:
                msg = json.loads(data) if isinstance(data, str) else data
                payload = msg.get("payload") or msg
                kind = payload.get("kind") if isinstance(payload, dict) else None
                if kind == "gradient_push":
                    exp_data = payload.get("data") or {}
                    sender = payload.get("from")
                    logger.info(f"📥 Gradient push received from {sender} (via living_mesh)")
                    protocol._pending_grads.append(exp_data)
                    if protocol._model_callback is not None:
                        try:
                            protocol._model_callback(exp_data)
                        except Exception as cb_err:
                            logger.error(f"❌ Gradient callback error: {cb_err}")
                    # لا نعيد البث هنا؛ living_mesh الأصلي يتولى الـ gossip عند الحاجة
            except Exception:
                pass
            # استدعاء المعالج الأصلي دائماً
            return await original(data, websocket=websocket)

        self.mesh_node._process_secure_message = wrapped_process
        self.mesh_node._gradient_handler_installed = True
        logger.info("✅ Gradient handler registered on LivingMeshNode")

    def set_model_callback(self, callback: Callable):
        """تسجيل دالة تُستدعى عند استقبال تدرجات جديدة (data dict)."""
        self._model_callback = callback

    async def connect(self):
        """الاتصال (مركزي قديم أو عبر living_mesh)."""
        if self.mesh_node is not None:
            self.is_connected = True
            logger.info(f"✅ Node {self.node_id} ready for P2P Gradient Exchange via LivingMesh.")
            return

        if not self.alpha_url:
            logger.warning("⚠️ No mesh_node and no alpha_url — gradient exchange disabled.")
            self.is_connected = False
            return

        try:
            import aiohttp
            session = aiohttp.ClientSession()
            self.ws = await session.ws_connect(self.alpha_url)
            self.is_connected = True
            logger.info(f"✅ Node {self.node_id} connected to Alpha Node (legacy) for Gradient Exchange.")
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
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

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
        """بث التدرجات المحلية عبر بروتوكول living_mesh P2P (أو القناة المركزية القديمة)."""
        if not self.is_connected and self.mesh_node is None:
            # محاولة اتصال تلقائي خفيفة
            await self.connect()
            if not self.is_connected and self.mesh_node is None:
                return

        start_time = time.time()
        grad_data = self.serialize_gradients(model)
        serialization_time = time.time() - start_time

        payload_data = {
            "node_id": self.node_id,
            "data": grad_data,
            "timestamp": start_time,
            "serialization_ms": serialization_time * 1000,
        }

        # ---- المسار الجديد: P2P عبر living_mesh ----
        if self.mesh_node is not None:
            try:
                active_peers = self.mesh_node._get_active_peers_list()
                targets = [
                    p for p in active_peers
                    if p.get("id") != self.node_id and p.get("host") and p.get("port") is not None
                ]
                if not targets:
                    logger.info("📡 No active peers yet — gradient stored locally only.")
                    return

                # Gossip محدود: نرسل لعينة صغيرة (مثل living_mesh.sync_experience)
                sample_size = min(len(targets), 3)
                chosen = random.sample(targets, sample_size)
                for peer in chosen:
                    host = peer.get("host")
                    port = peer.get("port")
                    try:
                        await self.mesh_node.send_to_peer(
                            host, port, "gradient_push", payload_data, hops=0
                        )
                    except Exception as e:
                        logger.warning(f"⚠️ send_to_peer gradient failed for {host}:{port}: {e}")
                logger.info(
                    f"📤 Gradients pushed via P2P to {len(chosen)} peer(s). "
                    f"Serialization: {serialization_time*1000:.2f}ms"
                )
                return
            except Exception as e:
                logger.error(f"❌ P2P gradient broadcast failed: {e}")
                # لا نسقط إلى المركزي تلقائياً إلا إذا كان متاحاً

        # ---- المسار القديم (توافق خلفي) ----
        if self.ws is not None and self.is_connected:
            try:
                import aiohttp
                await self.ws.send_str(json.dumps({
                    "type": "gradient_push",
                    "node_id": self.node_id,
                    "data": grad_data,
                    "timestamp": start_time,
                    "serialization_ms": serialization_time * 1000,
                }))
                logger.info(f"📤 Gradients pushed (legacy alpha). Serialization: {serialization_time*1000:.2f}ms")
            except Exception as e:
                logger.error(f"❌ Legacy gradient push failed: {e}")

    async def listen_for_updates(self, model: torch.nn.Module):
        """الاستماع لتحديثات التدرجات.
        عند استخدام living_mesh يكون الاستماع حدثياً عبر المعالج المسجّل؛
        هذه الدالة تطبق أي تدرجات معلّقة ثم (إن وُجدت قناة مركزية) تستمع عليها.
        """
        # تطبيق أي تدرجات وصلت عبر living_mesh
        while self._pending_grads:
            exp = self._pending_grads.pop(0)
            data_b64 = exp.get("data")
            if data_b64:
                try:
                    self.deserialize_and_apply(model, data_b64)
                    latency = (time.time() - exp.get("timestamp", time.time())) * 1000
                    logger.info(f"🔄 Global Gradients Integrated (P2P). Approx latency: {latency:.2f}ms")
                except Exception as e:
                    logger.error(f"❌ Failed to apply pending gradient: {e}")

        # المسار القديم فقط إذا كان متصلًا بقناة مركزية
        if self.ws is None or not self.is_connected or self.mesh_node is not None:
            return

        import aiohttp
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") == "gradient_pull":
                    recv_time = time.time()
                    push_time = data.get("timestamp", recv_time)
                    latency = (recv_time - push_time) * 1000
                    self.deserialize_and_apply(model, data["data"])
                    logger.info(f"🔄 Global Gradients Integrated (legacy). E2E Latency: {latency:.2f}ms")


# مثيل عام اختياري (يُهيَّأ لاحقاً من المتصل)
gradient_protocol = None

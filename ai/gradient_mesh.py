# -*- coding: utf-8 -*-
"""🚀 NSM Gradient Mesh — تبادل التدرجات عبر بروتوكول P2P الحقيقي.
يحوّل الاتصال من WebSocket مركزي دائم إلى LivingMeshNode.send_to_peer.
alpha_url يُستخدم كبذرة لاكتشاف الأقران فقط (وليس مسار تجميع مركزي دائم).
"""
import asyncio
import json
import torch
import base64
import io
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Callable
from urllib.parse import urlparse

logger = logging.getLogger("GradientMesh")


def _get_living_mesh_node_class():
    try:
        from ai.living_mesh import LivingMeshNode
        return LivingMeshNode
    except Exception as e:
        logger.warning(f"⚠️ LivingMeshNode unavailable ({e}); P2P path disabled")
        return None


def _parse_seed_from_alpha_url(alpha_url: Optional[str]):
    """استخراج host/port من alpha_url لاستخدامه كبذرة اكتشاف فقط."""
    if not alpha_url:
        return None
    try:
        raw = alpha_url.strip()
        if "://" not in raw:
            raw = "ws://" + raw
        parsed = urlparse(raw)
        host = parsed.hostname
        if not host:
            return None
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme in ("wss", "https") else 80
        return {"host": host, "port": int(port)}
    except Exception as e:
        logger.warning(f"⚠️ Could not parse alpha_url as seed: {e}")
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
        alpha_url: بذرة لاكتشاف الأقران فقط (seed)، وليس مسار تجميع مركزي دائم.
        mesh_node: LivingMeshNode جاهز (مفضل).
        host/port: عند إنشاء mesh_node داخلياً.
        """
        self.node_id = node_id
        self.alpha_url = alpha_url
        self.is_connected = False
        self.p2p_ready = False
        self._model_callback: Optional[Callable] = None
        self._pending_grads: List[Dict[str, Any]] = []
        self._last_broadcast_stats: Dict[str, Any] = {}

        self.mesh_node = mesh_node
        if self.mesh_node is None:
            LivingMeshNode = _get_living_mesh_node_class()
            if LivingMeshNode is not None:
                try:
                    self.mesh_node = LivingMeshNode(
                        node_id=node_id,
                        host=host or "127.0.0.1",
                        port=port or 0,
                    )
                    seed_nodes = []
                    seed = _parse_seed_from_alpha_url(alpha_url)
                    if seed:
                        seed_nodes.append({
                            "id": "alpha_seed",
                            "host": seed["host"],
                            "port": seed["port"],
                        })
                    self.mesh_node.join_network(seed_nodes=seed_nodes or None)
                    self.p2p_ready = True
                except Exception as e:
                    logger.warning(f"⚠️ LivingMesh init soft-fail: {e}")
                    self.mesh_node = None

        if self.mesh_node is not None:
            self.p2p_ready = True
            self._register_gradient_handler()

    def _register_gradient_handler(self):
        """معالج موحّد وغير متزامن لاستقبال gradient_push."""
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
                    logger.info(f"📥 gradient_push from {sender} (unified async handler)")
                    protocol._pending_grads.append(exp_data)
                    if protocol._model_callback is not None:
                        try:
                            protocol._model_callback(exp_data)
                        except Exception as cb_err:
                            logger.error(f"❌ Gradient callback error: {cb_err}")
            except Exception:
                pass
            return await original(data, websocket=websocket)

        self.mesh_node._process_secure_message = wrapped_process
        self.mesh_node._gradient_handler_installed = True
        logger.info("✅ Unified gradient_push handler registered on LivingMeshNode")

    def set_model_callback(self, callback: Callable):
        self._model_callback = callback

    async def connect(self):
        """جاهزية P2P + اكتشاف أقران عبر بذرة alpha_url إن وُجدت."""
        if self.mesh_node is None:
            self.is_connected = False
            logger.warning("⚠️ No LivingMeshNode — gradient P2P disabled.")
            return

        self.p2p_ready = True
        self.is_connected = True
        seed = _parse_seed_from_alpha_url(self.alpha_url)
        if seed:
            ok = await self.mesh_node.request_peers(seed["host"], seed["port"])
            if ok:
                logger.info(f"✅ Peer discovery via alpha seed {seed['host']}:{seed['port']} succeeded")
            else:
                logger.warning(f"⚠️ Peer discovery via alpha seed failed (will retry on broadcast)")
        else:
            logger.info(f"✅ Node {self.node_id} P2P ready (no alpha seed configured)")

    def serialize_gradients(self, model: torch.nn.Module) -> str:
        grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.detach().cpu().half().numpy().tolist()
        buffer = io.BytesIO()
        torch.save(grads, buffer)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def deserialize_and_apply(self, model: torch.nn.Module, grad_data_b64: str):
        """دمج التدرجات الواردة مع تدرجات النموذج المحلي (متوسط All-Reduce بسيط)."""
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
                        param.grad = (param.grad + remote_grad) / 2.0

    async def broadcast_gradients(self, model: torch.nn.Module) -> Dict[str, Any]:
        """بث التدرجات مباشرة إلى جميع الأقران المكتشفين عبر send_to_peer.
        يُرجع إحصائيات النجاح/الفشل صراحةً (لا نجاح صامت).
        """
        if self.mesh_node is None:
            stats = {"ok": 0, "failed": 0, "peers": 0, "error": "no_mesh_node"}
            self._last_broadcast_stats = stats
            logger.warning("⚠️ broadcast_gradients: no mesh_node")
            return stats

        if not self.is_connected and not self.p2p_ready:
            await self.connect()

        start_time = time.time()
        grad_data = self.serialize_gradients(model)
        serialization_ms = (time.time() - start_time) * 1000

        payload_data = {
            "node_id": self.node_id,
            "data": grad_data,
            "timestamp": start_time,
            "serialization_ms": serialization_ms,
            # معرّف ثابت لهذه اللقطة المنطقية الواحدة من التدرجات — يُستخدَم بواسطة
            # sync_experience() في living_mesh.py (آلية _seen_gossip_ids الموجودة
            # فعلاً) لمنع تكرار تطبيق/إعادة بث نفس التدرجات لو وصلت لعقدة عبر أكثر
            # من مسار (إرسال مباشر + ترحيل Gossip محدود hops<3). بدون هذا المعرّف،
            # كل عقدة تستقبل الرسالة (مباشرة أو مُرحَّلة) كانت تفترضها "جديدة" فتُعيد
            # بثّها، ما يضخّم عدد النسخ المستلمة فعلياً (تحقّقنا: 5 بث → 13 استقبال).
            "_gossip_id": f"grad_{uuid.uuid4().hex[:12]}",
        }

        active_peers = self.mesh_node._get_active_peers_list()
        targets = [
            p for p in active_peers
            if p.get("id") != self.node_id and p.get("host") and p.get("port") is not None
        ]

        # إن لم يوجد أقران بعد، أعد محاولة الاكتشاف عبر بذرة alpha
        if not targets:
            seed = _parse_seed_from_alpha_url(self.alpha_url)
            if seed:
                await self.mesh_node.request_peers(seed["host"], seed["port"])
                active_peers = self.mesh_node._get_active_peers_list()
                targets = [
                    p for p in active_peers
                    if p.get("id") != self.node_id and p.get("host") and p.get("port") is not None
                ]

        ok_count = 0
        fail_count = 0
        failures: List[str] = []

        # إرسال مباشر إلى جميع الأقران المكتشفين (وليس عينة عشوائية)
        for peer in targets:
            host = peer.get("host")
            port = peer.get("port")
            peer_id = peer.get("id", f"{host}:{port}")
            try:
                success = await self.mesh_node.send_to_peer(
                    host, port, "gradient_push", payload_data, hops=0
                )
                if success:
                    ok_count += 1
                else:
                    fail_count += 1
                    failures.append(str(peer_id))
            except Exception as e:
                fail_count += 1
                failures.append(f"{peer_id}:{e}")
                logger.warning(f"⚠️ gradient send failed for {peer_id}: {e}")

        stats = {
            "ok": ok_count,
            "failed": fail_count,
            "peers": len(targets),
            "failures": failures,
            "serialization_ms": round(serialization_ms, 2),
        }
        self._last_broadcast_stats = stats

        if ok_count == 0 and fail_count == 0:
            logger.warning("⚠️ Gradients not sent — no discovered peers yet.")
        elif fail_count > 0:
            logger.warning(
                f"⚠️ Gradients partial: ok={ok_count} failed={fail_count}/{len(targets)} "
                f"failures={failures}"
            )
        else:
            logger.info(
                f"📤 Gradients pushed to ALL {ok_count} peer(s) via P2P. "
                f"serialization={serialization_ms:.2f}ms"
            )
        return stats

    async def listen_for_updates(self, model: torch.nn.Module):
        """تطبيق التدرجات الواردة عبر المعالج الموحّد (غير متزامن / قائمة انتظار)."""
        applied = 0
        while self._pending_grads:
            exp = self._pending_grads.pop(0)
            data_b64 = exp.get("data")
            if not data_b64:
                continue
            try:
                self.deserialize_and_apply(model, data_b64)
                latency = (time.time() - exp.get("timestamp", time.time())) * 1000
                logger.info(f"🔄 Merged remote gradients into local model. latency≈{latency:.2f}ms")
                applied += 1
            except Exception as e:
                logger.error(f"❌ Failed to merge pending gradient: {e}")
        return applied


gradient_protocol = None

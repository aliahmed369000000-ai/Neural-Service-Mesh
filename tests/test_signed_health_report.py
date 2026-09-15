import base64
import json

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from ai.node_health_layer import NodeHealthLayer


class FakeNode:
    node_id = "node-test"

    def __init__(self):
        self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.public_key = self.private_key.public_key()
        self.state = {"nodes": {}}

    def _load_state(self):
        return dict(self.state)

    def _save_state(self, state):
        self.state = state

    def network_health_snapshot(self):
        return {
            "node_id": self.node_id,
            "online_peers": 1,
            "known_nodes": 2,
            "reputation_self": 0,
            "receipts": 0,
            "content_objects": 0,
            "identity_pub_fingerprint": "test-fp",
        }

    def get_reputation(self, node_id):
        return {"score": 3}

    def sign_message(self, message):
        signature = self.private_key.sign(
            message.encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode()

    def public_pem(self):
        return self.public_key.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )


def test_signed_health_report_rejects_tampering():
    node = FakeNode()
    layer = NodeHealthLayer(node)
    report = layer.health()
    assert NodeHealthLayer.verify_health_report(node.public_pem(), report)
    report["known_nodes"] = 999
    assert not NodeHealthLayer.verify_health_report(node.public_pem(), report)

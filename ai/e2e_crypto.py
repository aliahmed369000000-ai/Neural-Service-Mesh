"""Hybrid E2E encryption for NSM peer payloads.

RSA-OAEP wraps one fresh AES-256-GCM key per message.  The returned envelope
contains no plaintext and is JSON serialisable.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ALGORITHM = "RSA-OAEP-SHA256+AES-256-GCM"


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"), validate=True)


def encrypt_payload(payload: Dict[str, Any], recipient_public_key_pem: str | bytes) -> Dict[str, str]:
    """Encrypt a JSON object for the recipient's RSA public key."""
    if isinstance(recipient_public_key_pem, str):
        recipient_public_key_pem = recipient_public_key_pem.encode()
    public_key = serialization.load_pem_public_key(recipient_public_key_pem)
    if not isinstance(public_key, rsa.RSAPublicKey):
        raise TypeError("recipient key must be an RSA public key")
    plaintext = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    aes_key = AESGCM.generate_key(bit_length=256)
    nonce = os.urandom(12)
    ciphertext = AESGCM(aes_key).encrypt(nonce, plaintext, None)
    wrapped_key = public_key.encrypt(
        aes_key,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                     algorithm=hashes.SHA256(), label=None),
    )
    return {
        "algorithm": ALGORITHM,
        "wrapped_key": _b64(wrapped_key),
        "nonce": _b64(nonce),
        "ciphertext": _b64(ciphertext),
    }


def decrypt_payload(envelope: Dict[str, str], recipient_private_key) -> Dict[str, Any]:
    """Decrypt and validate an envelope using the node's RSA private key."""
    if envelope.get("algorithm") != ALGORITHM:
        raise ValueError("unsupported E2E algorithm")
    aes_key = recipient_private_key.decrypt(
        _unb64(envelope["wrapped_key"]),
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                     algorithm=hashes.SHA256(), label=None),
    )
    plaintext = AESGCM(aes_key).decrypt(
        _unb64(envelope["nonce"]), _unb64(envelope["ciphertext"]), None
    )
    value = json.loads(plaintext.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("decrypted payload must be an object")
    return value

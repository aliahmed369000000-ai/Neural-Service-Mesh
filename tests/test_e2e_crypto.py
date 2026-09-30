import unittest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

from ai.e2e_crypto import encrypt_payload, decrypt_payload


def pem(key):
    return key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )


class E2ECryptoTests(unittest.TestCase):
    def test_round_trip_and_no_plaintext(self):
        private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        payload = {"kind": "inference_request", "data": {"text": "سرّي"}}
        envelope = encrypt_payload(payload, pem(private))
        self.assertNotIn("سرّي", str(envelope))
        self.assertEqual(decrypt_payload(envelope, private), payload)

    def test_tampering_or_wrong_key_fails(self):
        private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        wrong = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        envelope = encrypt_payload({"x": 1}, pem(private))
        envelope["ciphertext"] = envelope["ciphertext"][:-2] + "AA"
        with self.assertRaises(Exception):
            decrypt_payload(envelope, private)
        envelope = encrypt_payload({"x": 1}, pem(private))
        with self.assertRaises(Exception):
            decrypt_payload(envelope, wrong)


if __name__ == "__main__":
    unittest.main()

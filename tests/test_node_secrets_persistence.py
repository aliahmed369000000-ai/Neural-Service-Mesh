"""ai/node_secrets.py — قبل هذا، GitManager وLLMFallback يقرآن GITHUB_TOKEN/
HF_TOKEN/مفاتيح LLM عبر os.getenv() مباشرة فقط: أي توكن أُعطي عند تشغيل
واحد يُفقَد تماماً عند أول إعادة تشغيل بلا تمرير نفس متغيّر env يدوياً من
جديد. هذا الملف يثبت أن العقدة تتذكّر الأسرار التي أُعطيت لها مرة واحدة،
عبر RSA+AES (ai/e2e_crypto.py) باستخدام مفتاح هويتها الخاص، دون تخزين أي
قيمة صريحة على القرص، وأن ذلك مربوط فعلاً داخل LivingMeshNode.__init__
فتستفيد منه GitManager/LLMFallback تلقائياً دون أي تغيير فيهما.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from ai.node_secrets import SECRET_ENV_VARS, load_and_persist_secrets


@pytest.fixture()
def keypair():
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    from cryptography.hazmat.primitives import serialization
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return priv, pub_pem


@pytest.fixture()
def data_dir():
    d = tempfile.mkdtemp(prefix="nsm_secrets_test_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _clear_secret_env(monkeypatch):
    for k in SECRET_ENV_VARS:
        monkeypatch.delenv(k, raising=False)


def test_token_in_env_gets_persisted_encrypted(data_dir, keypair, monkeypatch):
    _clear_secret_env(monkeypatch)
    priv, pub_pem = keypair
    monkeypatch.setenv("GITHUB_TOKEN", "secret_value_abc123")

    active = load_and_persist_secrets(data_dir, priv, pub_pem)
    assert "GITHUB_TOKEN" in active

    secrets_file = os.path.join(data_dir, "secrets.enc.json")
    assert os.path.exists(secrets_file)
    raw = open(secrets_file, encoding="utf-8").read()
    assert "secret_value_abc123" not in raw  # لا نص صريح إطلاقاً على القرص
    # envelope صالح (نفس بنية ai/e2e_crypto.py)
    envelope = json.loads(raw)
    assert envelope["algorithm"].startswith("RSA-OAEP")


def test_token_restored_on_restart_without_env_var(data_dir, keypair, monkeypatch):
    priv, pub_pem = keypair

    # الجولة الأولى: التوكن موجود في env
    _clear_secret_env(monkeypatch)
    monkeypatch.setenv("GITHUB_TOKEN", "secret_value_xyz789")
    load_and_persist_secrets(data_dir, priv, pub_pem)

    # الجولة الثانية: نفس data_dir ونفس المفتاح الخاص، لكن بلا أي env على الإطلاق
    _clear_secret_env(monkeypatch)
    assert "GITHUB_TOKEN" not in os.environ
    active = load_and_persist_secrets(data_dir, priv, pub_pem)

    assert "GITHUB_TOKEN" in active
    assert os.environ["GITHUB_TOKEN"] == "secret_value_xyz789"


def test_wrong_identity_key_cannot_decrypt_another_nodes_secrets(data_dir, keypair, monkeypatch):
    """ملف أسرار عقدة أخرى (مفتاح خاص مختلف) يُرفض بأمان بدل كسر الإقلاع —
    يُعامَل كأنه لا يوجد شيء محفوظ."""
    priv_a, pub_a = keypair
    _clear_secret_env(monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "node_a_secret")
    load_and_persist_secrets(data_dir, priv_a, pub_a)

    priv_b = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    from cryptography.hazmat.primitives import serialization
    pub_b = priv_b.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()

    _clear_secret_env(monkeypatch)
    # لا يرفع استثناء — يتعامل مع الملف كأنه غير قابل للقراءة ويكمل بأمان
    active = load_and_persist_secrets(data_dir, priv_b, pub_b)
    assert "HF_TOKEN" not in active
    assert "HF_TOKEN" not in os.environ


def test_no_secrets_in_env_or_disk_is_a_clean_noop(data_dir, keypair, monkeypatch):
    _clear_secret_env(monkeypatch)
    priv, pub_pem = keypair
    active = load_and_persist_secrets(data_dir, priv, pub_pem)
    assert active == []
    assert not os.path.exists(os.path.join(data_dir, "secrets.enc.json"))


def test_living_mesh_node_git_manager_picks_up_persisted_token(monkeypatch):
    """تحقّق التكامل الحقيقي: LivingMeshNode تبني GitManager تلقائياً بتوكن
    مُستعاد من تشغيل سابق، بلا أي تعديل على GitManager نفسها."""
    from ai.living_mesh import LivingMeshNode

    tmp = tempfile.mkdtemp(prefix="nsm_secrets_integ_")
    try:
        for k in SECRET_ENV_VARS:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "integ_test_token_999")
        LivingMeshNode(node_id="integ_node", host="127.0.0.1", port=0, data_dir=tmp)

        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        node2 = LivingMeshNode(node_id="integ_node", host="127.0.0.1", port=0, data_dir=tmp)
        assert node2.git_manager.token == "integ_test_token_999"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

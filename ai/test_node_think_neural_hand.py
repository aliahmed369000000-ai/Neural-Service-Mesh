# -*- coding: utf-8 -*-
"""اختبارات لإضافة جديدة لليد اليسرى: think — العقدة تستخدم شبكة عصبية
حقيقية مفتوحة المصدر (Falcon-Arabic-7B-Instruct عبر Hugging Face
Inference API المجانية)، وليس فقط قواعد مكتوبة سلفاً.

الآلية: provider_override جديد في ai/llm_fallback.py.LLMFallback.__init__
يفرض مزوّداً واحداً فقط على نسخة بعينها (بصرف النظر عن
NSM_LLM_PROVIDER_PREF العام) — يضمن أن think() لا تستخدم أبداً أي مزوّد
غير المفتوح المصدر 'hf'، حتى لو كانت مفاتيح مزوّدين مدفوعين (Anthropic...)
مُعدَّة في بيئة التشغيل نفسها.
"""
from __future__ import annotations

import shutil
import tempfile
from unittest.mock import patch

import pytest

from ai.llm_fallback import LLMFallback, Provider
from core.mesh_bundle import MeshBundle
from core.node_hands import LEFT


# ── 1) provider_override على LLMFallback مباشرة (وحدة) ───────────────────

def test_provider_override_hf_ignores_other_configured_keys(monkeypatch):
    """حتى مع مفتاح Anthropic مُعدّاً (مزوّد مدفوع كان سيُختار عادةً أولاً
    لو تُرك الأمر لـ auto)، provider_override='hf' يجب أن يقصر السلسلة
    على Hugging Face فقط."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf-key")
    monkeypatch.delenv("NSM_LLM_PROVIDER_PREF", raising=False)

    llm = LLMFallback(provider_override="hf")
    chain = llm._build_provider_chain()
    assert chain, "يجب أن تحتوي السلسلة على Hugging Face على الأقل"
    assert all(p[0] is Provider.HUGGINGFACE for p in chain), (
        f"provider_override='hf' سرّب مزوّداً آخر: {[p[0] for p in chain]}"
    )
    assert llm._provider is Provider.HUGGINGFACE


def test_provider_override_falls_back_to_ckg_synth_when_no_hf_key(monkeypatch):
    """بلا مفتاح HF إطلاقاً (حتى لو كانت مزوّدات أخرى مُعدَّة)،
    provider_override='hf' يجب أن يسقط بلطف لـCKG Synthesis، لا أن يرفع
    استثناءً أو يتسرّب لمزوّد آخر."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("NSM_LLM_PROVIDER_PREF", raising=False)

    llm = LLMFallback(provider_override="hf")
    assert llm._provider is Provider.CKG_SYNTH


def test_provider_override_none_preserves_old_env_based_behavior(monkeypatch):
    """بلا provider_override (القيمة الافتراضية None)، يجب ألا يتغيّر
    السلوك القديم المعتمِد على NSM_LLM_PROVIDER_PREF إطلاقاً."""
    monkeypatch.setenv("NSM_LLM_PROVIDER_PREF", "huggingface")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")

    llm = LLMFallback()  # provider_override=None صراحة بالتصميم
    assert llm._provider is Provider.HUGGINGFACE


# ── 2) أداة think المربوطة باليد اليسرى (تكامل) ───────────────────────────

@pytest.fixture()
def bundle():
    tmp = tempfile.mkdtemp(prefix="nsm_think_test_")
    try:
        yield MeshBundle(storage_dir=tmp, db_path=f"{tmp}/mesh.db")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _a_node(b):
    ids = list(b.role_node_ids.values())
    return b.registry.get(ids[0])


def test_think_bound_to_left_hand(bundle):
    a = _a_node(bundle)
    names = {t["name"] for t in a.hands.tools()[LEFT]}
    assert "think" in names


def test_think_rejects_empty_prompt(bundle):
    a = _a_node(bundle)
    res = a.use_hand(LEFT, "think", prompt="   ")
    assert not res.ok
    assert res.denied is False  # خطأ تحقق من الإدخال، وليس رفضاً أمنياً


def test_think_rejects_overlong_prompt(bundle):
    a = _a_node(bundle)
    res = a.use_hand(LEFT, "think", prompt="س" * 2001)
    assert not res.ok


def test_think_uses_real_open_source_model_end_to_end(bundle, monkeypatch):
    """محاكاة استجابة HF Inference API حقيقية الشكل (نفس صيغة
    [{"generated_text": ...}] الفعلية) — تتحقق أن think() تصل فعلاً حتى
    نهاية المسار الحقيقي (بناء prompt → استدعاء HF → تفسير الاستجابة)
    لا مجرد mock سطحي على طبقة أعلى."""
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def fake_post_json(url, payload, headers, timeout=15):
        assert "messages" in payload and payload["model"]
        return [{"generated_text": "جواب حقيقي من فالكون"}]

    a = _a_node(bundle)
    with patch("ai.llm_fallback._post_json", side_effect=fake_post_json):
        res = a.use_hand(LEFT, "think", prompt="ما عاصمة اليمن؟")

    assert res.ok, res.error
    assert res.output["text"] == "جواب حقيقي من فالكون"
    assert res.output["provider"] == "huggingface"
    assert res.output["used_open_source_model"] is True


def test_think_never_leaks_to_paid_provider_even_if_configured(bundle, monkeypatch):
    """الضمان الجوهري: حتى مع مفتاح Anthropic مُعدّاً في بيئة التشغيل،
    think() يجب ألا تستدعيه إطلاقاً — فقط Hugging Face أو CKG fallback."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("think() استدعت مزوّداً غير مفتوح المصدر!")

    a = _a_node(bundle)
    with patch("ai.llm_fallback.LLMFallback._call_anthropic", side_effect=fail_if_called):
        res = a.use_hand(LEFT, "think", prompt="سؤال بسيط")

    # بلا مفتاح HF، يسقط لـCKG Synthesis — لا خطأ، ولا استدعاء لـAnthropic
    assert res.ok, res.error
    assert res.output["provider"] != "anthropic"
    assert res.output["used_open_source_model"] is False


def test_think_denied_when_node_is_paused():
    """الضمان الثابت 'عقدة محجورة لا تستخدم أي يد إطلاقاً' يسبق think
    أيضاً — عقدة لا يمكنها 'التفكير' عبر يدها وهي محجورة."""
    tmp = tempfile.mkdtemp(prefix="nsm_think_test2_")
    try:
        b = MeshBundle(storage_dir=tmp, db_path=f"{tmp}/mesh.db")
        a = _a_node(b)
        a.pause(reason="manual")
        res = a.use_hand(LEFT, "think", prompt="سؤال")
        assert res.denied
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

"""
اختبار: دالتان حُذفتا بإعادة كتابة سابقة بينما ما زال كود الإنتاج يستدعيهما:
1) ai/video_engine._fetch_cinematic_clip كان يستدعي HiggsfieldClient.submit_job/
   poll_job (حُذفتا في 64bb484) فيرفع AttributeError يبتلعه except، فتُستخدم
   الخلفية المتدرّجة بصمت دائماً ولا يعمل Higgsfield أبداً.
2) LivingMeshNode.get_evolutionary_updates (حُذفت في e46e43b) تستدعيها
   ai/arabic_transformer_tf.py::evolve_live.
"""
from __future__ import annotations

import io
import shutil
import tempfile

from ai import video_engine
from ai.living_mesh import LivingMeshNode


class _FakeResp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def test_cinematic_clip_uses_current_higgsfield_api(monkeypatch):
    calls = {}

    class FakeClient:
        def __init__(self, key): calls["key"] = key
        def generate_video_from_prompt(self, prompt, duration=8, aspect_ratio="9:16"):
            calls["prompt"], calls["ar"] = prompt, aspect_ratio
            return "https://example.invalid/clip.mp4"

    import ai.higgsfield_engine as hf
    monkeypatch.setattr(hf, "HiggsfieldClient", FakeClient)
    monkeypatch.setenv("HIGGSFIELD_API_KEY", "ID:SECRET")
    monkeypatch.setattr(video_engine.urllib.request, "urlopen",
                        lambda req, timeout=60: _FakeResp(b"mp4-bytes"))
    d = tempfile.mkdtemp()
    try:
        path = video_engine._fetch_cinematic_clip("نص", "مشهد", d, 0)
        assert path and open(path, "rb").read() == b"mp4-bytes"
        assert calls["key"] == "ID:SECRET" and calls["ar"] == "9:16"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_cinematic_clip_falls_back_when_no_url(monkeypatch):
    class FakeClient:
        def __init__(self, key): pass
        def generate_video_from_prompt(self, *a, **k): return ""

    import ai.higgsfield_engine as hf
    monkeypatch.setattr(hf, "HiggsfieldClient", FakeClient)
    monkeypatch.setenv("HIGGSFIELD_API_KEY", "ID:SECRET")
    assert video_engine._fetch_cinematic_clip("a", "b", tempfile.gettempdir(), 0) is None


def test_get_evolutionary_updates_returns_only_evolution_sync():
    d = tempfile.mkdtemp()
    node = LivingMeshNode("evo-test", data_dir=d)
    try:
        node.join_network()
        node.sync_experience("evolution_sync", {"change_log": "v2"})
        node.sync_experience("other_kind", {"x": 1})
        updates = node.get_evolutionary_updates()
        assert updates and all(u["kind"] == "evolution_sync" for u in updates)
        assert updates[-1]["data"]["change_log"] == "v2"
    finally:
        node.mark_offline()
        shutil.rmtree(d, ignore_errors=True)

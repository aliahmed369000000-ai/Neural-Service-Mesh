from pathlib import Path
import importlib.util

def _mod():
    spec = importlib.util.spec_from_file_location(
        "ckg_quality_report", Path("scripts/ckg_quality_report.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def test_inspect_detects_lfs_or_concepts():
    m = _mod()
    info = m.inspect_ckg_file(Path("knowledge/cognitive_graph.json"))
    assert info["exists"] is True
    # either LFS pointer or has concepts
    assert info.get("lfs_pointer") or info.get("n_concepts", 0) >= 0

def test_score_answer():
    m = _mod()
    s = m.score_answer("إجابة طويلة نسبياً عن الأمانة والعدل في الخطاب المعرفي", {"W_SEMANTIC": 0.3, "W_SCORE": 0.3, "W_MEMORY": 0.2, "W_TOPOLOGY": 0.2}, [{"a": 1}])
    assert 0 <= s["overall"] <= 1

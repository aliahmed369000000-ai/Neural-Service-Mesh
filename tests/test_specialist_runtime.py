from ai.specialist_runtime import (
    format_research_context,
    get_specialist_protocol,
    normalize_sources,
    specialist_capability_report,
)


def test_three_specialists_have_operating_protocols():
    report = specialist_capability_report()
    assert report["specialists"] == ["research", "coding", "maintenance"]
    assert all(report["protocols"].values())
    assert report["deterministic"] is True
    assert report["writes_files"] is False


def test_source_normalization_accepts_structured_search_response():
    payload = {
        "ok": True,
        "results": [
            {"title": "مصدر أول", "url": "https://example.com/a", "snippet": "نص"},
            {"title": "مكرر", "url": "https://example.com/a", "snippet": "آخر"},
            {"title": "بدون رابط", "url": "", "snippet": "لا يُقبل"},
            {"title": "مصدر ثان", "url": "https://example.org/b", "snippet": "نص آخر"},
        ],
    }
    records = normalize_sources(payload, max_results=5)
    assert [record.title for record in records] == ["مصدر أول", "مصدر ثان"]
    assert records[0].domain == "example.com"
    assert len(records[0].fingerprint) == 12


def test_research_context_has_citations_and_no_empty_entries():
    context = format_research_context(
        {"results": [{"title": "مرجع موثوق", "url": "https://example.net/ref", "snippet": "ملخص"}]}
    )
    assert "[1] مرجع موثوق (example.net)" in context
    assert "الرابط: https://example.net/ref" in context
    assert "ملخص" in context
    assert "بدون رابط" not in context


def test_unknown_category_is_safe():
    assert get_specialist_protocol("unknown") == ""
    assert format_research_context({"results": []}) == ""

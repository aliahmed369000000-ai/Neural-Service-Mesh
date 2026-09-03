from ai.mesh_network_config import parse_seed_urls, public_url


def test_parse_seed_urls_normalizes_deduplicates_and_rejects_invalid_values():
    result = parse_seed_urls("https://seed-a.example/, https://seed-a.example/\nftp://bad,not-a-url")
    assert [endpoint.url for endpoint in result] == ["https://seed-a.example"]
    assert result[0].health_url == "https://seed-a.example/health"


def test_public_url_rejects_bind_addresses(monkeypatch):
    monkeypatch.setenv("PUBLIC_NODE_URL", "http://0.0.0.0:8765")
    assert public_url("PUBLIC_NODE_URL") == ""


def test_public_url_accepts_external_http_url(monkeypatch):
    monkeypatch.setenv("PUBLIC_NODE_URL", "https://seed.example/mesh/")
    assert public_url("PUBLIC_NODE_URL") == "https://seed.example/mesh"

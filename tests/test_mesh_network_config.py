from ai.mesh_network_config import reconnect_delay, parse_seed_urls, public_url, seed_retry_order


def test_parse_seed_urls_normalizes_deduplicates_and_rejects_invalid_values():
    result = parse_seed_urls("https://seed-a.example/, https://seed-a.example/\nftp://bad,not-a-url")
    assert [endpoint.url for endpoint in result] == ["https://seed-a.example"]
    assert result[0].health_url == "https://seed-a.example/health"


def test_seed_retry_order_prefers_healthy_alternatives():
    seeds = parse_seed_urls("https://a.example,https://b.example")
    assert [seed.url for seed in seed_retry_order(seeds, "https://a.example")] == [
        "https://b.example", "https://a.example"
    ]


def test_reconnect_delay_is_exponential_and_capped():
    assert reconnect_delay(0) == 1.0
    assert reconnect_delay(3) == 8.0
    assert reconnect_delay(99) == 60.0


def test_public_url_rejects_bind_addresses(monkeypatch):
    monkeypatch.setenv("PUBLIC_NODE_URL", "http://0.0.0.0:8765")
    assert public_url("PUBLIC_NODE_URL") == ""


def test_public_url_accepts_external_http_url(monkeypatch):
    monkeypatch.setenv("PUBLIC_NODE_URL", "https://seed.example/mesh/")
    assert public_url("PUBLIC_NODE_URL") == "https://seed.example/mesh"

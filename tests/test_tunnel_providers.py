from ai.tunnel_providers import TunnelCandidate, health_url, verify_tunnel


def test_health_url():
    assert health_url("https://seed.example/mesh/") == "https://seed.example/mesh/health"


def test_verify_tunnel_fails_closed_when_process_is_dead():
    candidate = TunnelCandidate("fake", "https://seed.example", lambda: False)
    assert verify_tunnel(candidate) is False

from scripts.monitor_seed_health import check_health


def test_check_health_fails_closed_for_unreachable_url():
    assert check_health("http://127.0.0.1:1/health", 0.1) is False

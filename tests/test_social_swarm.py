from ai.social_swarm import pre_publish_check, craft_platform_posts, upsert_follower, recall_follower
from ai.agent_factory import AgentFactory

def test_pre_publish_safe():
    r = pre_publish_check("منشور معرفي هادئ عن الأمانة")
    assert r["ok"] is True

def test_platform_posts():
    posts = craft_platform_posts("العلم", ["linkedin", "twitter"])
    assert "linkedin" in posts and "twitter" in posts

def test_crm_roundtrip():
    upsert_follower("سارة", "whatsapp", interest="سورة النور")
    p = recall_follower("سارة")
    assert p is not None
    assert "سورة النور" in (p.get("interests") or [])

def test_factory_social_roles():
    roles = AgentFactory.available_roles()
    assert "TrendScoutAgent" in roles
    assert "SocialContentAgent" in roles

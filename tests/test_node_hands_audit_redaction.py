"""حجب الأسرار في سجل تدقيق NodeHands: كان الحجب بحسب اسم المفتاح الأعلى فقط،
فسرّ داخل dict/قائمة متداخلة أو داخل نص (URL، أمر curl، Bearer) يتسرّب إلى
node_hands_audit.jsonl كما هو. الآن يُحجَب بشكل متداخل وبأنماط النصوص، دون
المساس بالنص العادي (عربي/أرقام) ودون تعديل kwargs الأصلية."""
import json

from core.node_hands import NodeHands, _redact


def test_nested_dict_and_list_secrets_redacted():
    r = _redact({"headers": {"Authorization": "Bearer abc123SECRETxyz"},
                 "opts": [{"password": "hunter2"}]})
    blob = json.dumps(r, ensure_ascii=False)
    assert "abc123SECRETxyz" not in blob and "hunter2" not in blob


def test_secret_patterns_inside_strings_redacted():
    r = _redact({"url": "https://api.x.com/v1?token=SECRET999&q=1",
                 "cmd": "curl -H 'Authorization: Bearer ghp_AAAABBBBCCCCDDDD'"})
    blob = json.dumps(r, ensure_ascii=False)
    assert "SECRET999" not in blob and "ghp_AAAABBBBCCCCDDDD" not in blob
    assert "q=1" in blob  # الباقي سليم


def test_plain_values_unchanged_and_original_kwargs_untouched():
    kw = {"q": "ما حكم الصلاة؟", "n": 3, "cfg": {"mode": "fast", "password": "p"}}
    r = _redact(kw)
    assert "ما حكم الصلاة؟" in r["q"] and r["n"] == "3" and "fast" in r["cfg"]
    assert kw["cfg"]["password"] == "p"  # لا تعديل للأصل


def test_deep_and_huge_structures_do_not_blow_up():
    deep = cur = {}
    for _ in range(50):
        cur["x"] = {}
        cur = cur["x"]
    huge = {"items": list(range(100000))}
    assert len(_redact({"d": deep, "h": huge})["h"]) <= 200


def test_end_to_end_audit_file_has_no_secret(tmp_path):
    class N:
        state, node_id, name = "active", "n1", "n"

    audit = tmp_path / "a.jsonl"
    h = NodeHands(N(), audit_path=audit)
    h.bind("left", "fetch", lambda **kw: "ok")
    h.use("left", "fetch", url="https://x/y?api_key=TOPSECRET1", opts={"auth": {"t": "zzz"}})
    text = audit.read_text(encoding="utf-8")
    assert "TOPSECRET1" not in text and "zzz" not in text

import json
from ai.nsm_agent_core import NSMAgent

agent = NSMAgent("safety-test")
unsafe = agent._dispatch_action({"action": "read_file", "path": "/etc/passwd"})
assert "غير مسموح" in unsafe

safe = agent._dispatch_action({"action": "run_file", "cmd": "python -m py_compile ai/nsm_agent_core.py"})
safe_data = json.loads(safe)
assert safe_data["ok"] is True
assert safe_data["automatic"] is True

denied = agent._dispatch_action({"action": "run_file", "cmd": "rm -rf /tmp/nsm-agent-should-not-run"})
denied_data = json.loads(denied)
assert denied_data["ok"] is False
assert denied_data["requires_approval"] is True
print("AGENT_SAFETY_OK")

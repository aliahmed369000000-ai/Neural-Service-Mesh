# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.agent_user_assist import handle_user_assist, welcome_card
from ai.agent_project_bridge import dispatch_agent_message


def test_welcome():
    w = welcome_card()
    assert "NSM" in w or "مساعد" in w


def test_help_intent():
    r = handle_user_assist("مساعدة")
    assert r is not None
    assert "صنّف" in r or "مساعدة" in r or "ماذا" in r


def test_bridge_help():
    r = dispatch_agent_message("ماذا تستطيع؟")
    assert r is not None
    assert len(r) > 40


def test_explain_moe():
    r = handle_user_assist("ما هو moe")
    assert r is not None
    assert "MoE" in r or "خبراء" in r

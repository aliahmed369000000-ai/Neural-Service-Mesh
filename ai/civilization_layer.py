"""
Civilization Layer — ما بعد قمة الإنتاج
======================================
  1) Grand Knowledge Mesh
  2) Cognitive Swarm Simulation
  3) Cognitive Edge Runtime (لا ASIC حقيقي)
  4) Civilizational Guardrails
"""
from __future__ import annotations

import re
from typing import Optional


def civilization_status() -> str:
    lines = [
        "## 🌍 طبقة الحضارة / ما بعد القمة",
        "",
        "1. **Grand Mesh** — `شبكة معرفية كونية` / `grand mesh`",
        "2. **محاكاة عقول** — `محاكاة عقول حول: …`",
        "3. **مسار طرفي** — `نظام تشغيل معرفي` / `edge runtime` (ملف تشغيل لا OS كامل)",
        "4. **حارس معرفي** — `تحقق ادعاء: …`",
        "",
        "حدود: لا تصنيع رقاقات، لا نشر تلقائي على الشبكات، لا فتوى آلية.",
    ]
    for mod in (
        "ai.grand_knowledge_mesh",
        "ai.cognitive_swarm_sim",
        "ai.cognitive_os_edge",
        "ai.civilizational_guardrails",
    ):
        try:
            __import__(mod)
            lines.append(f"- `{mod}`: ✅")
        except Exception as e:
            lines.append(f"- `{mod}`: ❌ {e}")
    return "\n".join(lines)


def handle_civilization_command(user_input: str) -> Optional[str]:
    text = (user_input or "").strip()
    if not text:
        return None
    if re.search(r"(طبق[ةه]\s*حضار|civilization\s*layer|ما\s*بعد\s*القمه|حارس\s*حضار)", text, re.I) or text.lower() in (
        "civilization",
        "حضارة",
    ):
        return civilization_status()
    for mod, fn in (
        ("ai.grand_knowledge_mesh", "handle_mesh_command"),
        ("ai.cognitive_swarm_sim", "handle_swarm_sim_command"),
        ("ai.cognitive_os_edge", "handle_edge_command"),
        ("ai.civilizational_guardrails", "handle_guard_command"),
    ):
        try:
            m = __import__(mod, fromlist=[fn])
            r = getattr(m, fn)(text)
            if r is not None:
                return r
        except Exception:
            continue
    return None

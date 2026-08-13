# -*- coding: utf-8 -*-
"""
سرب المناقشة (Swarm Debate / Consensus) — تعاون متعدد الوكلاء عبر جولات نقاش.

بعد أن ينفّذ السرب مهامه الفرعية، تدخل النتائج في جولات مناقشة: كل وكيل
يراجع نتائج زملائه ويساهم برأيه (تأييد / اعتراض / إثراء)، ثم يُلخّص
المنسّق مواقف الجولات ويستخرج نقاط الاتفاق والخلاف والقرار النهائي،
فيرتفع بذلك مستوى جودة الإجابة الموحّدة للمهام المعقّدة.

القواعد:
- لا تبدأ المناقشة إذا كان عدد النتائج الناجحة أقل من DEBATE_MIN_PARTICIPANTS
- عدد الجولات محدود بـ MAX_DEBATE_ROUNDS (يمنع النقاش المفتوح)
- الفشل في أي مرحلة نقاش لا يُعطّل السرب — يُسجَّل حدث debate_abandoned
  وتكمل المهمة بمخرجاتها العادية

أحداث ناقل الأحداث:
    debate_started · debate_argument · debate_round_done
    debate_consensus · debate_abandoned
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nsm.debate")

# عدد الجولات الأقصى للمناقشة في عملية سرب واحدة
MAX_DEBATE_ROUNDS: int = 3

# أقل عدد نتائج ناجحة مطلوبة لبدء النقاش
DEBATE_MIN_PARTICIPANTS: int = 2

DEBATE_EVENTS = (
    "debate_started",
    "debate_argument",
    "debate_round_done",
    "debate_consensus",
    "debate_abandoned",
)

STANCE_LABELS = {
    "agree": "✅ مؤيد",
    "disagree": "❌ معترض",
    "enhance": "💡 مُثرٍ",
    "partial": "🟡 مؤيد جزئياً",
}


def _emit(event_type: str, agent_id: str, title: str, status: str,
          detail: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """يطلق حدث نقاش على ناقل الأحداث؛ يفشل بصمت خارج سياق Streamlit."""
    try:
        from ai.agent_event_bus import emit_event  # noqa: WPS433 (استيراد كسول)
        return emit_event(event_type, agent_id=agent_id, title=title,
                          status=status, detail=detail, metadata=metadata)
    except Exception:  # pragma: no cover - حماية خارج Streamlit
        return None


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")


def _build_arguments_prompt(
    goal: str,
    results: List[Dict[str, Any]],
    previous_round: Optional[List[Dict[str, Any]]],
    round_index: int,
) -> str:
    """يبني برومبت الجولة الحالية للمناقشة."""
    lines = [
        f"الهدف الأصلي: {goal}",
        f"جولة المناقشة {round_index + 1} من {MAX_DEBATE_ROUNDS}.",
        "",
        "نتائج الوكلاء المنفّذين:",
    ]
    for res in results:
        lines.append(
            f"- [{res.get('agent_role', res.get('agent_id', 'وكيل'))}] "
            f"({res.get('sub_goal', '')}): {res.get('result_text', '')}"
        )
    if previous_round:
        lines += ["", "مساهمات الجولة السابقة:", ""]
        for prev in previous_round:
            stance = STANCE_LABELS.get(prev.get("stance", ""), prev.get("stance", ""))
            lines.append(f"- {prev.get('agent_id', '')} [{stance}]: {prev.get('argument', '')}")
    lines += [
        "",
        "أنت وكيل يراجع النتائج. أدلِ بمساهمة واحدة قصيرة ومحددة لكل نتيجة من "
        "نتائج زملائك: قيّم صحتها من منظور تخصصك، وأضف نقطة اعتراض أو إثراء أو "
        "تأييد واقعية. لا تكرر ما قيل في الجولة السابقة.",
        "",
        "أجب بصيغة JSON فقط، مصفوفة كائنات بهذا الشكل بالضبط بدون أي نص خارجها:",
        '[{"agent_id": "agent-1", "stance": "agree|disagree|enhance|partial", '
        '"argument": "مساهمتك", "target_agent": "الوكيل المستهدف من مساهمتك"}]',
    ]
    return "\n".join(lines)


def _build_consensus_prompt(goal: str, transcript: List[Dict[str, Any]]) -> str:
    return (
        f"الهدف الأصلي: {goal}\n\n"
        "هذه سجلّات مناقشة بين عدة وكلاء حول الهدف أعلاه:\n"
        + "\n".join(
            f"- {r.get('round_index', '')} | {r.get('agent_id', '')} "
            f"[{STANCE_LABELS.get(r.get('stance', ''), r.get('stance', ''))}]: "
            f"{r.get('argument', '')}"
            for r in transcript
        )
        + "\n\n"
        "أجب بصيغة JSON فقط:\n"
        '{"agreed": "نقاط الاتفاق الرئيسية", '
        '"disagreed": "نقاط الخلاف التي ما زالت مفتوحة", '
        '"verdict": "القرار النهائي الموحّد الذي يجب أن تُبنى عليه الإجابة"}'
    )


def _extract_json_array(text: str) -> Optional[List[Dict[str, Any]]]:
    """يستخرج مصفوفة JSON من نص الوكيل بأمان."""
    import re
    match = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not match:
        return None
    try:
        import json
        parsed = json.loads(match.group(0))
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    import re
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        import json
        return json.loads(match.group(0))
    except Exception:
        return None


def debate_round(
    goal: str,
    results: List[Dict[str, Any]],
    round_index: int,
    previous_round: Optional[List[Dict[str, Any]]] = None,
    llm_generate=None,
) -> List[Dict[str, Any]]:
    """
    ينفّذ جولة مناقشة واحدة. يتلقى llm_generate كدالة مولّد LLM
    (أو يستدعي LLMFallback افتراضياً) لكي يبقى قابلاً للاختبار بدون API.
    """
    prompt = _build_arguments_prompt(goal, results, previous_round, round_index)
    text = ""
    try:
        if llm_generate is not None:
            text = str(llm_generate(prompt))
        else:
            from ai.llm_fallback import LLMFallback
            text = (LLMFallback().generate(prompt).text or "")
    except Exception as exc:
        logger.warning(f"فشل تنفيذ جولة نقاش {round_index + 1}: {exc}")
        return []
    args = _extract_json_array(text) or []
    cleaned = []
    for item in args:
        if not isinstance(item, dict):
            continue
        argument = str(item.get("argument") or "").strip()
        if not argument:
            continue
        stance = str(item.get("stance", "")).strip().lower()
        if stance not in STANCE_LABELS:
            stance = "enhance"
        cleaned.append({
            "round_index": round_index,
            "agent_id": str(item.get("agent_id", "unknown")),
            "stance": stance,
            "argument": argument,
            "target_agent": str(item.get("target_agent", "")),
            "timestamp": _timestamp(),
        })
    return cleaned


def consensus_summary(
    goal: str,
    transcript: List[Dict[str, Any]],
    llm_generate=None,
) -> Optional[Dict[str, Any]]:
    """يُلخّص النقاش في إجماع وخلاف وقرار نهائي. transcript فارغ (نقاش
    بدأ وتوقف مبكراً قبل أي مساهمة) لا يُلغي الإجماع الإحصائي."""
    fallback = _fallback_consensus(transcript)
    if not transcript:
        # لا مساهمات لاستخلاص رأي منها — لا نُرهق LLM ولا نُسقط السرب
        return fallback
    try:
        prompt = _build_consensus_prompt(goal, transcript)
        text = ""
        if llm_generate is not None:
            text = str(llm_generate(prompt))
        else:
            from ai.llm_fallback import LLMFallback
            text = (LLMFallback().generate(prompt).text or "")
        obj = _extract_json_object(text)
        if not obj:
            return fallback or {"agreed": "", "disagreed": "", "verdict": "", "source": "fallback"}
        return {
            "agreed": str(obj.get("agreed", "")),
            "disagreed": str(obj.get("disagreed", "")),
            "verdict": str(obj.get("verdict", "")),
            "source": "llm",
        }
    except Exception as exc:
        logger.warning(f"فشل تلخيص إجماع النقاش: {exc}")
        return fallback or {"agreed": "", "disagreed": "", "verdict": "", "source": "fallback"}


def _fallback_consensus(transcript: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """تقرير إحصائي بسيط يُبنى من المواقف مباشرة عند تعذّر LLM — لا يتعطل السرب."""
    if not transcript:
        return {"agreed": "بدأت المناقشة لكن لم تُسجَّل أي مساهمة من الوكلاء",
                "disagreed": "", "verdict": "الاعتماد على نتائج المهام كما هي دون تعديل", "source": "fallback"}
    stances = [r.get("stance", "") for r in transcript]
    return {
        "agreed": f"شارك {len(transcript)} وكيل في {max(r.get('round_index', 0) + 1 for r in transcript)} جولات نقاش",
        "disagreed": ", ".join({r.get("agent_id", "") for r in transcript if r.get("stance") == "disagree"}) or "لا اعتراضات مسجلة",
        "verdict": f"الأكثر انتشاراً: {'تأييد' if stances.count('agree') >= stances.count('disagree') else 'تعديل'}",
        "source": "fallback",
    }


# ── أحداث ناقل الأحداث ─────────────────────────────────────────────────

def announce_debate_started(goal: str, participants: int, max_rounds: int) -> None:
    _emit("debate_started", agent_id="swarm_debate", title=f"مناقشة السرب: {goal[:48]}",
          status="running",
          detail=f"بدأت {participants} وكلاء ناجحة مناقشة عبر {max_rounds} جولة كحد أقصى",
          metadata={"participants": participants, "max_rounds": max_rounds})


def announce_debate_argument(agent_id: str, stance: str, argument: str, round_index: int,
                             target_agent: str = "") -> None:
    stance_label = STANCE_LABELS.get(stance, stance)
    target = f" إلى {target_agent}" if target_agent else ""
    _emit("debate_argument", agent_id=agent_id, title=f"مساهمة {agent_id}{target}",
          status=stance,
          detail=argument[:200],
          metadata={"stance": stance, "stance_label": stance_label,
                    "round_index": round_index, "target_agent": target_agent})


def announce_debate_round_done(round_index: int, contributions: int, goal: str) -> None:
    _emit("debate_round_done", agent_id="swarm_debate",
          title=f"اكتملت جولة النقاش {round_index + 1}",
          status="done",
          detail=f"{contributions} مساهمة في جولة النقاش {round_index + 1}",
          metadata={"round_index": round_index, "contributions": contributions})


def announce_debate_consensus(goal: str, consensus: Dict[str, Any]) -> None:
    _emit("debate_consensus", agent_id="swarm_debate", title=f"إجماع السرب: {goal[:48]}",
          status="done",
          detail=consensus.get("verdict", "")[:200],
          metadata={"consensus": consensus, "final": True})


def announce_debate_abandoned(goal: str, reason: str) -> None:
    _emit("debate_abandoned", agent_id="swarm_debate", title=f"نقاش ملغي: {goal[:48]}",
          status="error", detail=reason, metadata={"reason": reason})


# ── الواجهة الرئيسية ────────────────────────────────────────────────────

def run_debate(
    goal: str,
    results: List[Dict[str, Any]],
    max_rounds: int = MAX_DEBATE_ROUNDS,
    llm_generate=None,
) -> Dict[str, Any]:
    """
    يدير جلسات النقاش كاملة: جولات + تلخيص إجماع + أحداث.
    يعيد {transcript, rounds, consensus} أو سجل ملغى عند تعذر النقاش.
    """
    successful = [r for r in (results or []) if r.get("success")]
    if len(successful) < DEBATE_MIN_PARTICIPANTS:
        reason = f"لا توجد نتائج ناجحة كافية للنقاش ({len(successful)} < {DEBATE_MIN_PARTICIPANTS})"
        announce_debate_abandoned(goal, reason)
        return {"transcript": [], "rounds": [], "consensus": None, "abandoned": reason}

    announce_debate_started(goal, len(successful), max_rounds)
    transcript: List[Dict[str, Any]] = []
    rounds_meta: List[Dict[str, Any]] = []
    previous_round = None
    for round_index in range(max_rounds):
        contributions = debate_round(goal, successful, round_index, previous_round,
                                     llm_generate=llm_generate)
        for arg in contributions:
            transcript.append(arg)
            announce_debate_argument(
                arg["agent_id"], arg["stance"], arg["argument"],
                round_index, arg.get("target_agent", ""),
            )
        rounds_meta.append({"round_index": round_index, "contributions": len(contributions)})
        announce_debate_round_done(round_index, len(contributions), goal)
        # توقف مبكر إذا لم تضف الجولة أي شيء جديداً
        if not contributions:
            break
        previous_round = contributions

    consensus = consensus_summary(goal, transcript, llm_generate=llm_generate)
    if consensus:
        announce_debate_consensus(goal, consensus)
    return {"transcript": transcript, "rounds": rounds_meta, "consensus": consensus, "abandoned": None}

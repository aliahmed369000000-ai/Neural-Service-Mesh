"""
Harm Classifier — مُصنِّف الأذى في المدخلات
=============================================
Python port من src/lib/classify.ts في G0DM0D3-main.

مُصنِّف سريع قائم على regex يُلصق بمدخلات المستخدم تصنيفًا حسب نطاق الأذى.
يعمل جانب AutoTune — نفس النمط، بدون تكلفة API.

التصنيف: 13 نطاق × عشرات الفئات الفرعية.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

# ── التصنيف ───────────────────────────────────────────────────────────

HarmDomain = Literal[
    "violence", "self_harm", "sexual", "hate", "cbrn",
    "cyber", "fraud", "illegal", "deception", "privacy",
    "meta", "gray", "benign",
]

HarmSubcategory = Literal[
    # violence
    "weapons", "physical_harm", "mass_harm", "animal_cruelty", "threat",
    # self_harm
    "suicide", "eating_disorder", "self_injury", "substance_abuse",
    # sexual
    "csam", "non_consensual", "explicit", "trafficking",
    # hate
    "slurs", "supremacy", "discrimination", "radicalization",
    # cbrn
    "chemical", "biological", "radiological", "dual_use_cbrn",
    # cyber
    "malware", "exploit", "intrusion", "phishing", "credential", "ddos", "spyware",
    # fraud
    "financial", "identity", "counterfeiting", "academic",
    # illegal
    "drugs_synthesis", "drugs_acquisition", "human_trafficking", "organized_crime", "evasion",
    # deception
    "misinformation", "manipulation", "deepfakes", "propaganda", "impersonation",
    # privacy
    "doxxing", "stalking", "data_theft", "surveillance",
    # meta
    "jailbreak", "prompt_injection", "system_prompt", "guardrail_test",
    # gray
    "dual_use", "controversial", "age_restricted", "professional_sensitive",
    # benign
    "coding", "creative", "education", "analysis", "conversation",
    "humor", "professional", "personal", "other",
]

DOMAIN_LABELS: dict = {
    "violence":    ("🔴", "عنف"),
    "self_harm":   ("🟠", "إيذاء الذات"),
    "sexual":      ("🔴", "محتوى جنسي"),
    "hate":        ("🔴", "خطاب كراهية"),
    "cbrn":        ("☢️", "CBRN"),
    "cyber":       ("💻", "إلكتروني/هاكنج"),
    "fraud":       ("💳", "احتيال"),
    "illegal":     ("⚖️", "غير قانوني"),
    "deception":   ("🎭", "خداع"),
    "privacy":     ("👁", "خصوصية"),
    "meta":        ("🔓", "تجاوز القيود"),
    "gray":        ("🟡", "منطقة رمادية"),
    "benign":      ("🟢", "بريء"),
}


@dataclass
class ClassificationResult:
    domain:      str
    subcategory: str
    confidence:  float          # 0.0–1.0
    flags:       List[str] = field(default_factory=list)
    intent:      Optional[str] = None


# ── قواعد النمط ──────────────────────────────────────────────────────

@dataclass
class PatternRule:
    domain:     str
    sub:        str
    weight:     int
    patterns:   List[re.Pattern]


def _p(*exprs: str) -> List[re.Pattern]:
    return [re.compile(e, re.IGNORECASE) for e in exprs]


RULES: List[PatternRule] = [

    # ═══ CYBER ══════════════════════════════════════════════════════

    PatternRule("cyber", "malware", 10, _p(
        r"\b(write|code|create|build|make)\b.{0,30}\b(malware|virus|trojan|worm|ransomware|rat|rootkit|keylogger|botnet)\b",
        r"\b(malware|virus|trojan|ransomware|rat|rootkit|keylogger)\b.{0,30}\b(source\s*code|tutorial|guide|how.to)\b",
        r"\b(crypter|fud\s*(crypter|payload)|obfuscat)\b.{0,20}\b(malware|payload|rat)\b",
    )),

    PatternRule("cyber", "exploit", 10, _p(
        r"\b(write|create|develop)\b.{0,20}\b(exploit|zero.day|0day|buffer\s*overflow|rce|remote\s*code\s*exec)\b",
        r"\b(exploit|vulnerability)\b.{0,30}\b(write|develop|craft|weaponize)\b",
        r"\b(cve-\d{4}-\d+)\b.{0,20}\b(exploit|poc|proof.of.concept)\b",
    )),

    PatternRule("cyber", "intrusion", 8, _p(
        r"\b(hack|break\s*into|gain\s*access|compromise|penetrat)\b.{0,30}\b(server|system|network|account|database|website|wifi|computer)\b",
        r"\b(bypass|crack|brute.?force)\b.{0,20}\b(password|authentication|login|security|firewall|2fa|mfa)\b",
    )),

    PatternRule("cyber", "phishing", 8, _p(
        r"\b(create|write|build|make)\b.{0,20}\b(phishing|spear.?phishing)\b.{0,20}\b(email|page|site|kit|template)\b",
        r"\b(phishing)\b.{0,20}\b(kit|template|clone|page)\b",
        r"\b(social\s*engineer)\b.{0,20}\b(attack|script|template|pretexting)\b",
    )),

    PatternRule("cyber", "credential", 7, _p(
        r"\b(crack|dump|steal|harvest)\b.{0,20}\b(password|credential|hash|token|cookie|session)\b",
        r"\b(credential\s*stuff|password\s*spray|rainbow\s*table|hashcat|john\s*the\s*ripper)\b",
    )),

    PatternRule("cyber", "ddos", 8, _p(
        r"\b(ddos|dos\s*attack|flood|stress\s*test)\b.{0,20}\b(tool|script|service|booter|stresser|how.to)\b",
        r"\b(take\s*down|overload|crash)\b.{0,20}\b(server|website|network|service)\b",
    )),

    PatternRule("cyber", "spyware", 8, _p(
        r"\b(spy(ware)?|stalk(er)?ware|keylog|screen\s*capture|remote\s*access)\b.{0,20}\b(install|plant|deploy|on\s*(phone|computer|device))\b",
        r"\b(track|monitor|spy\s*on)\b.{0,20}\b(wife|husband|spouse|partner|girlfriend|boyfriend|ex|phone|location)\b",
    )),

    # ═══ FRAUD ══════════════════════════════════════════════════════

    PatternRule("fraud", "financial", 8, _p(
        r"\b(credit\s*card|bank)\b.{0,20}\b(fraud|scam|steal|clone|skim)\b",
        r"\b(ponzi|pyramid\s*scheme|money\s*laundr|wire\s*fraud)\b",
        r"\b(pump.and.dump|insider\s*trad|market\s*manipulat)\b",
    )),

    PatternRule("fraud", "identity", 8, _p(
        r"\b(identity\s*theft|steal\s*identity|fake\s*id|forged?\s*(id|passport|license|document))\b",
        r"\b(ssn|social\s*security)\b.{0,20}\b(fake|steal|generate|buy)\b",
    )),

    PatternRule("fraud", "counterfeiting", 8, _p(
        r"\b(counterfeit|fake|forg(e|ed|ing))\b.{0,20}\b(money|currency|bill|note|document|diploma|certificate|prescription)\b",
        r"\b(print|make)\b.{0,20}\b(fake\s*(money|bills|currency|passport))\b",
    )),

    PatternRule("fraud", "academic", 6, _p(
        r"\b(write|do)\b.{0,20}\b(my\s*essay|my\s*thesis|my\s*homework|my\s*assignment|my\s*paper)\b",
        r"\b(homework\s*help|essay\s*mill|contract\s*cheating|turnitin\s*bypass)\b",
    )),

    # ═══ ILLEGAL ════════════════════════════════════════════════════

    PatternRule("illegal", "organized_crime", 7, _p(
        r"\b(mafia|cartel|gang\s*activity|organized\s*crime|money\s*mule|fence\s*stolen)\b",
        r"\b(hitman|hired\s*killer|assassination\s*contract|dark\s*web\s*(market|purchase))\b",
    )),

    PatternRule("illegal", "evasion", 7, _p(
        r"\b(evade|avoid|launder)\b.{0,20}\b(tax|taxes|irs|customs|border|detection)\b",
        r"\b(off\s*the\s*grid|ghost\s*address|shell\s*compan|nominee\s*account)\b",
    )),

    # ═══ DECEPTION ══════════════════════════════════════════════════

    PatternRule("deception", "misinformation", 7, _p(
        r"\b(create|write|generate|spread)\b.{0,20}\b(fake\s*news|misinformation|disinformation|propaganda)\b",
        r"\b(fabricat|invent|make\s*up)\b.{0,20}\b(news|article|story|quote|statistic)\b",
    )),

    PatternRule("deception", "deepfakes", 8, _p(
        r"\b(deepfake|face\s*swap|voice\s*clone|synthetic\s*media)\b.{0,20}\b(create|make|generate|of)\b",
        r"\b(ai.generated\s*(porn|nude|video))\b",
    )),

    PatternRule("deception", "manipulation", 7, _p(
        r"\b(manipulat|gaslight|psychologically\s*manipulat)\b.{0,20}\b(person|people|someone|victim)\b",
        r"\b(dark\s*pattern|cult\s*recruit|mind\s*control\s*technique)\b",
    )),

    PatternRule("deception", "impersonation", 7, _p(
        r"\b(impersonat|pretend\s*to\s*be|pose\s*as)\b.{0,20}\b(police|doctor|official|celebrity|bank|company)\b",
        r"\b(fake\s*(profile|account|identity))\b.{0,20}\b(create|make|build)\b",
    )),

    # ═══ PRIVACY ════════════════════════════════════════════════════

    PatternRule("privacy", "doxxing", 9, _p(
        r"\b(dox|doxx|find\s*(address|phone|personal\s*info))\b.{0,20}\b(someone|person|user|target)\b",
        r"\b(reveal\s*personal\s*info|expose\s*(identity|address|phone))\b",
    )),

    PatternRule("privacy", "stalking", 8, _p(
        r"\b(stalk|track\s*without\s*consent|follow\s*secretly)\b.{0,20}\b(person|someone|ex|partner)\b",
        r"\b(find\s*location|track\s*phone)\b.{0,20}\b(without\s*(knowing|consent|permission))\b",
    )),

    PatternRule("privacy", "data_theft", 8, _p(
        r"\b(steal|scrape|harvest)\b.{0,20}\b(user\s*data|personal\s*data|private\s*info|database)\b",
        r"\b(data\s*breach|leak\s*database|dump\s*user\s*data)\b",
    )),

    # ═══ META — تجاوز القيود ═════════════════════════════════════════

    PatternRule("meta", "jailbreak", 9, _p(
        r"\b(jailbreak|jail\s*break|bypass\s*(safety|restrictions|filter|guardrail))\b",
        r"\b(ignore\s*(your\s*)?(previous\s*)?instructions|disregard\s*(all|your)\s*(previous\s*)?instructions)\b",
        r"\b(pretend\s*(you\s*)?have\s*no\s*(restrictions|rules|guidelines))\b",
        r"\b(act\s*as\s*(if|though)\s*you\s*(were|are)\s*not\s*(an\s*ai|restricted|limited))\b",
        r"\b(developer\s*mode|god\s*mode|dan\s*mode|do\s*anything\s*now)\b",
    )),

    PatternRule("meta", "prompt_injection", 8, _p(
        r"\b(ignore\s*above|ignore\s*previous|new\s*instructions|override\s*system)\b",
        r"\b(\}\}\s*ignore|---\s*new\s*task|system:\s*you\s*(are|must|will))\b",
    )),

    PatternRule("meta", "guardrail_test", 6, _p(
        r"\b(test(ing)?\s*(your\s*)?(safety|limits|guardrails|filter|censor))\b",
        r"\b(what\s*(can|can't|cannot|won't|will\s*not)\s*you\s*(do|say|tell))\b",
        r"\b(are\s*you\s*censored|do\s*you\s*have\s*restrictions)\b",
    )),

    # ═══ VIOLENCE — عنف ═════════════════════════════════════════════

    PatternRule("violence", "weapons", 9, _p(
        r"\b(how\s*to\s*(make|build|create|assemble|modify)\b.{0,20}\b(gun|rifle|pistol|firearm|silencer|suppressor|automatic|machine\s*gun|pipe\s*bomb|ied|landmine))\b",
        r"\b(3d\s*print\b.{0,20}\b(gun|firearm|weapon|lower\s*receiver))\b",
        r"\b(ghost\s*gun|untraceable\s*(gun|firearm|weapon))\b",
        r"\b(convert\b.{0,20}\b(semi.?auto|automatic|full.?auto))\b",
    )),

    PatternRule("violence", "physical_harm", 8, _p(
        r"\b(how\s*to\b.{0,30}\b(kill|murder|hurt|harm|assault|attack)\b.{0,20}\b(someone|person|people|him|her|them))\b",
        r"\b(best\s*way\s*to\b.{0,20}\b(kill|murder|dispose\s*of\s*(a\s*)?body))\b",
        r"\b(poison\b.{0,20}\b(someone|person|food|water|drink)\b.{0,20}\b(without|undetected|untraceable))\b",
    )),

    PatternRule("violence", "mass_harm", 10, _p(
        r"\b(mass\s*(shooting|casualty|murder|killing)|school\s*shooting|terrorist\s*attack)\b.{0,20}\b(plan|how|execute|carry\s*out)\b",
        r"\b(attack\s*(a\s*)?(school|mall|church|concert|crowd|public))\b.{0,20}\b(plan|how|maximiz)\b",
    )),

    PatternRule("violence", "threat", 7, _p(
        r"\b(write\s*(a\s*)?threat|threatening\s*(letter|message|note))\b.{0,20}\b(to|for)\b",
        r"\b(how\s*to\s*intimidate|make\s*someone\s*afraid\s*of\s*me)\b",
    )),

    # ═══ SELF_HARM ════════════════════════════════════════════════════

    PatternRule("self_harm", "suicide", 9, _p(
        r"\b(how\s*to\s*(commit|attempt)\s*suicide|suicide\s*method|lethal\s*dose\s*of)\b",
        r"\b(most\s*(effective|painless|quick)\s*(way|method)\s*to\s*(die|kill\s*(myself|yourself)))\b",
        r"\b(want\s*to\s*(die|end\s*(it|my\s*life)|kill\s*myself))\b.{0,30}\b(help|how|method|tell\s*me)\b",
    )),

    PatternRule("self_harm", "self_injury", 7, _p(
        r"\b(how\s*to\s*cut\s*(myself|yourself|self)|self.?(harm|injur|mutilat)\s*(method|technique|without\s*scars))\b",
        r"\b(cutting\s*pattern|hide\s*self.?harm\s*scars)\b",
    )),

    PatternRule("self_harm", "substance_abuse", 7, _p(
        r"\b(how\s*to\s*(get\s*high|make\s*(meth|crack|heroin|fentanyl)|extract\s*(dxm|kratom\s*alkaloids)))\b",
        r"\b(recreational\s*(dose|use)\s*of\b.{0,20}\b(fentanyl|heroin|meth|opioid))\b",
    )),

    # ═══ HATE ═════════════════════════════════════════════════════════

    PatternRule("hate", "slurs", 8, _p(
        r"\b(write\s*(a\s*)?(racist|antisemitic|islamophobic|homophobic|transphobic)\s*(joke|post|message|speech|essay))\b",
        r"\b(explain\s*why\s*(jews|blacks|muslims|gays|trans)\s*(are|control|ruin))\b",
    )),

    PatternRule("hate", "supremacy", 9, _p(
        r"\b(white\s*supremac|neo.?nazi|aryan\s*(brotherhood|nation|superiority)|master\s*race)\b.{0,30}\b(manifesto|recruitment|propaganda|believe|ideology)\b",
        r"\b(write\s*(recruitment|propaganda)\s*(for|material)\b.{0,20}\b(hate\s*group|extremist))\b",
    )),

    PatternRule("hate", "radicalization", 8, _p(
        r"\b(how\s*to\s*(radicalize|recruit)\s*(someone|youth|people)\s*(to|into|for)\b.{0,20}\b(extremis|terroris|jihadis))\b",
        r"\b(isis|al.?qaeda|terror\s*group)\b.{0,20}\b(recruit|join|how\s*to)\b",
    )),

    # ═══ CBRN — أسلحة الدمار الشامل ══════════════════════════════════

    PatternRule("cbrn", "chemical", 10, _p(
        r"\b(synthesize|make|produce|create)\b.{0,20}\b(sarin|vx|novichok|nerve\s*agent|mustard\s*gas|chlorine\s*bomb|chemical\s*weapon)\b",
        r"\b(chemical\s*weapon\b.{0,20}\b(synthesis|recipe|production|formula))\b",
    )),

    PatternRule("cbrn", "biological", 10, _p(
        r"\b(synthesize|enhance|engineer|modify|weaponize)\b.{0,20}\b(anthrax|botulinum|ricin|smallpox|plague|ebola|pathogen|bioweapon)\b",
        r"\b(gain.of.function\b.{0,20}\b(pathogen|virus|bacteria|how\s*to))\b",
        r"\b(bioweapon|biological\s*weapon)\b.{0,20}\b(make|create|produce|synthesize|recipe)\b",
    )),

    PatternRule("cbrn", "radiological", 9, _p(
        r"\b(dirty\s*bomb|radiological\s*(weapon|device))\b.{0,20}\b(build|make|create|how)\b",
        r"\b(enrich\s*uranium|weapons.grade\s*(plutonium|uranium))\b.{0,20}\b(how|process|centrifuge)\b",
    )),

    # ═══ SEXUAL ═══════════════════════════════════════════════════════

    PatternRule("sexual", "non_consensual", 9, _p(
        r"\b(rape|sexual\s*assault)\b.{0,20}\b(how\s*to|plan|without\s*(getting\s*caught|consent))\b",
        r"\b(drug\s*(someone|a\s*drink)\b.{0,20}\b(sex|rape|assault))\b",
        r"\b(rohypnol|date\s*rape\s*drug)\b.{0,20}\b(how\s*to\s*(use|get|obtain))\b",
    )),

    # ═══ ILLEGAL — مخدرات ════════════════════════════════════════════

    PatternRule("illegal", "drugs_synthesis", 9, _p(
        r"\b(synthesize|cook|make|produce)\b.{0,20}\b(meth(amphetamine)?|fentanyl|heroin|cocaine|lsd|mdma|dmt)\b",
        r"\b(drug\s*synthesis\s*(route|recipe|guide|step.by.step))\b",
        r"\b(precursor\s*(chemical|purchase)\s*for\b.{0,20}\b(meth|fentanyl|amphetamine))\b",
    )),

    # ═══ GRAY — منطقة رمادية ═════════════════════════════════════════

    PatternRule("gray", "dual_use", 5, _p(
        r"\b(penetration\s*test|ethical\s*hack|red\s*team|security\s*research|ctf|capture\s*the\s*flag)\b",
        r"\b(osint|open\s*source\s*intelligence)\b.{0,20}\b(target|individual|person)\b",
    )),

    PatternRule("gray", "controversial", 4, _p(
        r"\b(controversial|taboo|sensitive\s*topic|politically?\s*(charged|sensitive))\b",
        r"\b(argue\s*(for|that)|defend\s*the\s*position)\b",
    )),

    # ═══ BENIGN — حميد ══════════════════════════════════════════════

    PatternRule("benign", "coding", 0, _p(
        r"\b(write\s*(a\s*)?function|debug\s*(this\s*)?code|how\s*do\s*i\s*implement|code\s*review|refactor)\b",
        r"```[\s\S]{0,50}```",
    )),

    PatternRule("benign", "creative", 0, _p(
        r"\b(write\s*(a\s*)?(story|poem|haiku|essay|script)|creative\s*writing|brainstorm)\b",
    )),

    PatternRule("benign", "education", 0, _p(
        r"\b(explain|teach|how\s*does|what\s*is|define|summarize|overview\s*of)\b",
    )),
]


# ── دالة التصنيف الرئيسية ────────────────────────────────────────────

def classify_prompt(prompt: str) -> ClassificationResult:
    """
    تصنيف مدخل المستخدم حسب نطاق الأذى.
    يستخدم regex فقط — بدون API، بدون تكلفة.
    """
    low = prompt.lower()
    best_domain    = "benign"
    best_sub       = "other"
    best_weight    = 0
    best_confidence= 0.3
    flags: List[str] = []

    for rule in RULES:
        matched_patterns = 0
        for pat in rule.patterns:
            if pat.search(low):
                matched_patterns += 1

        if matched_patterns == 0:
            continue

        # حساب الثقة بناءً على عدد الأنماط المتطابقة
        confidence = min(
            (matched_patterns / len(rule.patterns)) * (rule.weight / 10),
            0.95,
        )

        if rule.weight > best_weight or (rule.weight == best_weight and confidence > best_confidence):
            best_weight     = rule.weight
            best_confidence = confidence
            best_domain     = rule.domain
            best_sub        = rule.sub

        # إضافة الأعلام
        if rule.domain not in ("benign",) and matched_patterns > 0:
            flags.append(f"{rule.domain}/{rule.sub}")

    # إزالة التكرار من الأعلام
    seen: set = set()
    unique_flags: List[str] = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            unique_flags.append(f)

    return ClassificationResult(
        domain=best_domain,
        subcategory=best_sub,
        confidence=best_confidence,
        flags=unique_flags[:5],  # حد أقصى 5 أعلام
    )


def get_domain_label(domain: str) -> Tuple[str, str]:
    """إعادة (emoji، نص عربي) للنطاق."""
    return DOMAIN_LABELS.get(domain, ("⚪", domain))


def is_sensitive(result: ClassificationResult) -> bool:
    """هل التصنيف يستحق الإبراز للمستخدم؟"""
    return result.domain not in ("benign",) and result.confidence > 0.4

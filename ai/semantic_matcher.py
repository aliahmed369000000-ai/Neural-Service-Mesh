"""
Phase 3 – Semantic Matcher
Matches node outputs to node inputs using semantic similarity.
Uses keyword-based cosine similarity (no external ML libs required).
Foundation is kept clean for future embedding-based upgrade.
"""
from __future__ import annotations
import re
import math
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Semantic vocabulary ────────────────────────────────────────────────────
# Groups of semantically related terms so that "text" ~ "content" ~ "body"
_SYNONYM_GROUPS: List[List[str]] = [
    ["text", "content", "body", "message", "string", "str", "raw"],
    ["data", "payload", "input", "output", "result", "value"],
    ["analysis", "result", "output", "response", "score", "report"],
    ["sentiment", "emotion", "mood", "tone", "feeling"],
    ["summary", "abstract", "overview", "description", "synopsis"],
    ["number", "count", "int", "integer", "numeric", "quantity"],
    ["user", "customer", "client", "person", "profile"],
    ["error", "exception", "failure", "fault", "issue"],
    ["status", "state", "condition", "health", "flag"],
    ["timestamp", "date", "time", "created_at", "updated_at"],
    ["id", "identifier", "uuid", "key", "ref", "reference"],
    ["list", "array", "items", "collection", "set"],
    ["file", "path", "url", "uri", "link", "source"],
    ["json", "dict", "object", "map", "record", "document"],
    ["log", "event", "trace", "audit", "history"],
    ["model", "prediction", "inference", "classification", "label"],
    ["query", "search", "filter", "request", "prompt"],
    ["config", "settings", "parameters", "options", "args"],
    ["token", "auth", "credential", "secret", "key"],
    ["metric", "measurement", "stat", "kpi", "indicator"],
]

# Build reverse lookup: term -> canonical group index
_TERM_TO_GROUP: Dict[str, int] = {}
for _gi, _group in enumerate(_SYNONYM_GROUPS):
    for _term in _group:
        _TERM_TO_GROUP[_term.lower()] = _gi


def _tokenize(text: str) -> List[str]:
    """Split a schema field name / description into lowercase tokens."""
    text = text.lower()
    # split on non-alphanumeric
    tokens = re.split(r"[^a-z0-9]+", text)
    return [t for t in tokens if t]


def _semantic_tokens(tokens: List[str]) -> List[int]:
    """Map tokens to their synonym group index (or unique high IDs if unknown)."""
    groups = []
    unknown_id = len(_SYNONYM_GROUPS)
    for t in tokens:
        gid = _TERM_TO_GROUP.get(t, None)
        if gid is not None:
            groups.append(gid)
        else:
            # Treat unknown token as its own unique group
            groups.append(hash(t) % 10000 + unknown_id)
    return groups


def _bow_vector(tokens: List[str]) -> Dict[int, float]:
    """Build a term-frequency bag-of-words vector from semantic group ids."""
    vec: Dict[int, float] = {}
    gids = _semantic_tokens(tokens)
    for gid in gids:
        vec[gid] = vec.get(gid, 0.0) + 1.0
    # L2 normalise
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def _cosine(a: Dict[int, float], b: Dict[int, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    dot = sum(a[k] * b[k] for k in a if k in b)
    return round(dot, 4)


class NodeSemanticProfile:
    """
    Encapsulates all semantic information for a single node.
    Created once per node at registration / discovery time.
    """

    def __init__(self, node_id: str, name: str, description: str,
                 input_fields: Dict[str, str], output_fields: Dict[str, str],
                 tags: List[str] = None, capability: str = ""):
        self.node_id = node_id
        self.name = name
        self.description = description
        self.input_fields = input_fields    # field_name -> field_type
        self.output_fields = output_fields
        self.tags = tags or []
        self.capability = capability        # free-text capability description

        # Pre-compute vectors
        self._input_vec = self._build_vec(input_fields, description, tags)
        self._output_vec = self._build_vec(output_fields, description, tags)
        self._capability_vec = self._build_vec({}, capability, tags)

    @staticmethod
    def _build_vec(fields: Dict[str, str], text: str, tags: List[str]) -> Dict[int, float]:
        tokens: List[str] = []
        for fname, ftype in fields.items():
            tokens += _tokenize(fname)
            tokens += _tokenize(ftype)
        tokens += _tokenize(text)
        for tag in (tags or []):
            tokens += _tokenize(tag)
        return _bow_vector(tokens)

    def input_similarity(self, other: "NodeSemanticProfile") -> float:
        """How well does `other`'s output match this node's input?"""
        return _cosine(other._output_vec, self._input_vec)

    def capability_similarity(self, goal_text: str) -> float:
        """How well does this node's capability match a goal description?"""
        goal_tokens = _tokenize(goal_text)
        goal_vec = _bow_vector(goal_tokens)
        return _cosine(self._capability_vec, goal_vec)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "description": self.description,
            "input_fields": self.input_fields,
            "output_fields": self.output_fields,
            "tags": self.tags,
            "capability": self.capability,
        }

    @classmethod
    def from_node(cls, node) -> "NodeSemanticProfile":
        """Build a profile directly from a BaseNode instance."""
        return cls(
            node_id=node.node_id,
            name=node.name,
            description=node.description,
            input_fields=node.input_schema.fields,
            output_fields=node.output_schema.fields,
            tags=node.tags,
            capability=node.description,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "NodeSemanticProfile":
        return cls(
            node_id=d["node_id"],
            name=d["name"],
            description=d.get("description", ""),
            input_fields=d.get("input_fields", {}),
            output_fields=d.get("output_fields", {}),
            tags=d.get("tags", []),
            capability=d.get("capability", d.get("description", "")),
        )


class SemanticMatcher:
    """
    Phase 3 Semantic Matcher.
    Maintains a registry of NodeSemanticProfiles and answers
    questions like:
      - Which nodes can accept the output of node X?
      - Which nodes best match a goal description?
      - What is the semantic compatibility score between two nodes?
    """

    def __init__(self):
        self._profiles: Dict[str, NodeSemanticProfile] = {}
        logger.info("SemanticMatcher initialised (Phase 3)")

    # ── Profile management ─────────────────────────────────────────────────

    def register(self, node) -> NodeSemanticProfile:
        """Register (or re-register) a node's semantic profile."""
        profile = NodeSemanticProfile.from_node(node)
        self._profiles[node.node_id] = profile
        logger.debug(f"SemanticMatcher: registered profile for '{node.name}'")
        return profile

    def register_from_dict(self, node_dict: dict) -> NodeSemanticProfile:
        profile = NodeSemanticProfile.from_dict(node_dict)
        self._profiles[node_dict["node_id"]] = profile
        return profile

    def get_profile(self, node_id: str) -> Optional[NodeSemanticProfile]:
        return self._profiles.get(node_id)

    def all_profiles(self) -> List[NodeSemanticProfile]:
        return list(self._profiles.values())

    # ── Matching API ───────────────────────────────────────────────────────

    def find_compatible_consumers(self, producer_id: str,
                                  threshold: float = 0.1) -> List[Tuple[str, float]]:
        """
        Return nodes whose INPUT is semantically compatible with
        `producer_id`'s OUTPUT.
        Returns list of (node_id, score) sorted descending.
        """
        producer = self._profiles.get(producer_id)
        if not producer:
            return []

        results = []
        for nid, profile in self._profiles.items():
            if nid == producer_id:
                continue
            score = profile.input_similarity(producer)
            if score >= threshold:
                results.append((nid, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def find_nodes_for_goal(self, goal: str,
                            top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Return nodes best suited to achieve a described goal.
        Returns list of (node_id, score) sorted descending.
        """
        results = []
        for nid, profile in self._profiles.items():
            score = profile.capability_similarity(goal)
            results.append((nid, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def compatibility_score(self, producer_id: str, consumer_id: str) -> float:
        """Direct semantic compatibility from producer → consumer."""
        producer = self._profiles.get(producer_id)
        consumer = self._profiles.get(consumer_id)
        if not producer or not consumer:
            return 0.0
        return consumer.input_similarity(producer)

    def suggest_new_connections(self,
                                existing_edges: List[Tuple[str, str]],
                                threshold: float = 0.15) -> List[dict]:
        """
        Suggest new connections not currently in the graph
        that have high semantic compatibility.
        """
        existing = set((s, t) for s, t in existing_edges)
        suggestions = []

        node_ids = list(self._profiles.keys())
        for i, src_id in enumerate(node_ids):
            for tgt_id in node_ids[i + 1:]:
                if (src_id, tgt_id) in existing or (tgt_id, src_id) in existing:
                    continue
                # Check both directions
                score_fwd = self.compatibility_score(src_id, tgt_id)
                score_rev = self.compatibility_score(tgt_id, src_id)
                best_score = max(score_fwd, score_rev)
                if best_score >= threshold:
                    direction = (src_id, tgt_id) if score_fwd >= score_rev else (tgt_id, src_id)
                    suggestions.append({
                        "source_id": direction[0],
                        "target_id": direction[1],
                        "semantic_score": best_score,
                        "reason": "semantic_compatibility",
                    })

        suggestions.sort(key=lambda x: x["semantic_score"], reverse=True)
        return suggestions

    def profile_count(self) -> int:
        return len(self._profiles)

    def __repr__(self):
        return f"<SemanticMatcher profiles={self.profile_count()}>"


# ══════════════════════════════════════════════════════════════════════════════
# طبقة التحليل الدلالي العربي (مُضافة للملف الأصلي)
# ══════════════════════════════════════════════════════════════════════════════

from enum import Enum as _Enum
import math as _math


class IntentType(_Enum):
    DEFINITION    = "تعريف"
    HOW_TO        = "كيفية"
    REASON        = "سبب"
    COUNT         = "عدد"
    TIME          = "وقت"
    COMPARISON    = "مقارنة"
    LIST          = "قائمة"
    COMMAND       = "أمر"
    CLARIFICATION = "توضيح"
    CONFIRMATION  = "تأكيد"
    GENERAL       = "عام"


class SemanticDomain(_Enum):
    QURAN       = "قرآن"
    HADITH      = "حديث"
    PRAYER      = "صلاة"
    FASTING     = "صيام"
    ZAKAT       = "زكاة"
    HAJJ        = "حج"
    FIQH        = "فقه"
    AQEEDAH     = "عقيدة"
    ARABIC_LANG = "لغة_عربية"
    HISTORY     = "تاريخ_إسلامي"
    PROGRAMMING = "برمجة"
    AI          = "ذكاء_اصطناعي"
    SCIENCE     = "علوم"
    GENERAL     = "عام"


_AR_LEXICON: dict = {
    "قرآن":{"d":"قرآن","w":1.0,"rel":["آية","سورة","تلاوة"]},
    "آية": {"d":"قرآن","w":0.9,"rel":["سورة","قرآن"]},
    "سورة":{"d":"قرآن","w":0.9,"rel":["آية","قرآن"]},
    "تلاوة":{"d":"قرآن","w":0.8,"rel":["قرآن","تجويد"]},
    "تجويد":{"d":"قرآن","w":0.8,"rel":["قرآن"]},
    "تفسير":{"d":"قرآن","w":0.9,"rel":["قرآن"]},
    "مصحف":{"d":"قرآن","w":0.9,"rel":["قرآن"]},
    "حديث":{"d":"حديث","w":1.0,"rel":["سنة","نبي"]},
    "سنة": {"d":"حديث","w":0.9,"rel":["حديث","نبي"]},
    "صحيح":{"d":"حديث","w":0.8,"rel":["حديث"]},
    "بخاري":{"d":"حديث","w":0.9,"rel":["حديث"]},
    "مسلم":{"d":"حديث","w":0.9,"rel":["حديث"]},
    "صلاة":{"d":"صلاة","w":1.0,"rel":["ركعة","وضوء"]},
    "ركعة":{"d":"صلاة","w":0.9,"rel":["صلاة","سجود"]},
    "وضوء":{"d":"صلاة","w":0.9,"rel":["صلاة"]},
    "أذان":{"d":"صلاة","w":0.9,"rel":["صلاة"]},
    "فجر": {"d":"صلاة","w":0.8,"rel":["صلاة"]},
    "ظهر": {"d":"صلاة","w":0.8,"rel":["صلاة"]},
    "عصر": {"d":"صلاة","w":0.8,"rel":["صلاة"]},
    "مغرب":{"d":"صلاة","w":0.8,"rel":["صلاة"]},
    "عشاء":{"d":"صلاة","w":0.8,"rel":["صلاة"]},
    "جمعة":{"d":"صلاة","w":0.9,"rel":["صلاة"]},
    "سجود":{"d":"صلاة","w":0.8,"rel":["صلاة","ركعة"]},
    "صيام":{"d":"صيام","w":1.0,"rel":["رمضان","إفطار"]},
    "صوم": {"d":"صيام","w":1.0,"rel":["رمضان"]},
    "رمضان":{"d":"صيام","w":1.0,"rel":["صيام"]},
    "إفطار":{"d":"صيام","w":0.9,"rel":["صيام"]},
    "سحور":{"d":"صيام","w":0.9,"rel":["صيام"]},
    "زكاة":{"d":"زكاة","w":1.0,"rel":["نصاب","مال"]},
    "حج":  {"d":"حج","w":1.0,"rel":["عمرة","كعبة"]},
    "عمرة":{"d":"حج","w":0.9,"rel":["حج","مكة"]},
    "كعبة":{"d":"حج","w":0.9,"rel":["حج","مكة"]},
    "حلال":{"d":"فقه","w":1.0,"rel":["حرام"]},
    "حرام":{"d":"فقه","w":1.0,"rel":["حلال"]},
    "فرض": {"d":"فقه","w":0.9,"rel":["واجب"]},
    "واجب":{"d":"فقه","w":0.9,"rel":["فرض"]},
    "توحيد":{"d":"عقيدة","w":1.0,"rel":["إيمان"]},
    "إيمان":{"d":"عقيدة","w":1.0,"rel":["توحيد"]},
    "إسلام":{"d":"عقيدة","w":1.0,"rel":["إيمان"]},
    "نحو": {"d":"لغة_عربية","w":1.0,"rel":["إعراب"]},
    "صرف": {"d":"لغة_عربية","w":0.9,"rel":["جذر"]},
    "بلاغة":{"d":"لغة_عربية","w":0.9,"rel":[]},
    "إعراب":{"d":"لغة_عربية","w":0.9,"rel":["نحو"]},
    "جذر": {"d":"لغة_عربية","w":0.9,"rel":["صرف"]},
    "python":{"d":"برمجة","w":1.0,"rel":["code","برمجة"]},
    "برمجة":{"d":"برمجة","w":1.0,"rel":["python","api"]},
    "code": {"d":"برمجة","w":0.9,"rel":["python"]},
    "api":  {"d":"برمجة","w":0.9,"rel":["برمجة"]},
    "github":{"d":"برمجة","w":0.9,"rel":["git","code"]},
    "ذكاء":{"d":"ذكاء_اصطناعي","w":0.9,"rel":["نموذج"]},
    "نموذج":{"d":"ذكاء_اصطناعي","w":0.9,"rel":["تعلم"]},
    "تعلم":{"d":"ذكاء_اصطناعي","w":0.8,"rel":["نموذج"]},
    "neural":{"d":"ذكاء_اصطناعي","w":0.9,"rel":["ai"]},
    "llm":  {"d":"ذكاء_اصطناعي","w":1.0,"rel":["نموذج"]},
    "embedding":{"d":"ذكاء_اصطناعي","w":0.9,"rel":["neural"]},
}

_DOMAIN_SLOTS_AR = {
    "قرآن":0,"حديث":1,"صلاة":2,"صيام":3,"زكاة":4,
    "حج":5,"فقه":6,"عقيدة":7,"لغة_عربية":8,"تاريخ_إسلامي":9,
    "برمجة":10,"ذكاء_اصطناعي":11,"علوم":12,"عام":13,
}

_DOMAIN_ENUM_MAP = {
    "قرآن":SemanticDomain.QURAN,"حديث":SemanticDomain.HADITH,
    "صلاة":SemanticDomain.PRAYER,"صيام":SemanticDomain.FASTING,
    "زكاة":SemanticDomain.ZAKAT,"حج":SemanticDomain.HAJJ,
    "فقه":SemanticDomain.FIQH,"عقيدة":SemanticDomain.AQEEDAH,
    "لغة_عربية":SemanticDomain.ARABIC_LANG,"برمجة":SemanticDomain.PROGRAMMING,
    "ذكاء_اصطناعي":SemanticDomain.AI,
}

_INTENT_RE = [
    (IntentType.DEFINITION,    [re.compile(p,re.I|re.U) for p in [
        r'^ما (هو|هي|معنى|تعريف|مفهوم)',r'^عرّف',r'what is\b',r'define\b']]),
    (IntentType.HOW_TO,        [re.compile(p,re.I|re.U) for p in [
        r'^كيف (يمكن|يُمكن|أستطيع|يعمل|تعمل)',r'^كيفية',r'how (to|do|can)\b']]),
    (IntentType.REASON,        [re.compile(p,re.I|re.U) for p in [
        r'^لماذا',r'^ما (سبب|علة)',r'why\b']]),
    (IntentType.COUNT,         [re.compile(p,re.I|re.U) for p in [
        r'^كم (عدد|ركعة|ركعات)',r'how many\b']]),
    (IntentType.TIME,          [re.compile(p,re.I|re.U) for p in [r'^متى',r'when\b']]),
    (IntentType.COMPARISON,    [re.compile(p,re.I|re.U) for p in [
        r'الفرق بين',r'difference between\b',r'\bvs\b']]),
    (IntentType.LIST,          [re.compile(p,re.I|re.U) for p in [
        r'^ما هي (أنواع|فوائد|أركان|شروط)',r'^اذكر',r'list\b']]),
    (IntentType.COMMAND,       [re.compile(p,re.I|re.U) for p in [
        r'^(اكتب|أنشئ|اعمل|ابنِ|طور)',r'^(write|create|build|make)\b']]),
    (IntentType.CLARIFICATION, [re.compile(p,re.I|re.U) for p in [
        r'^(وضح|اشرح|فسر)',r'^(explain|clarify)\b']]),
    (IntentType.CONFIRMATION,  [re.compile(p,re.I|re.U) for p in [r'^(هل|أليس)',r'^is (it|this)\b']]),
]


def _ar_fnv(s: str) -> int:
    h = 0x811c9dc5
    for ch in s:
        h ^= ord(ch); h = (h * 0x01000193) & 0xFFFFFFFF
    return h


class ArabicSemanticMatcher:
    """
    محلل التشابه الدلالي للغة العربية.
    مُضاف للـ SemanticMatcher الأصلي كطبقة منفصلة.
    """

    def analyze(self, text: str):
        clean    = self._clean(text)
        concepts = self._concepts(clean)
        intent   = self._intent(clean)
        domain   = self._domain(concepts)
        vector   = self._vector(concepts)
        conf     = self._conf(concepts)

        class _R:
            pass
        r = _R()
        r.text, r.intent, r.domain, r.concepts = text, intent, domain, concepts
        r.vector, r.confidence = vector, conf
        r.summary = f"نية:{intent.value}|مجال:{domain.value}|مفاهيم:{','.join(concepts[:3])}"
        return r

    def similarity(self, t1: str, t2: str) -> float:
        a1, a2 = self.analyze(t1), self.analyze(t2)
        dot = sum(x*y for x,y in zip(a1.vector,a2.vector))
        na  = _math.sqrt(sum(x*x for x in a1.vector))
        nb  = _math.sqrt(sum(y*y for y in a2.vector))
        vs  = dot/(na*nb) if na>1e-8 and nb>1e-8 else 0.0
        shared = set(a1.concepts) & set(a2.concepts)
        union  = set(a1.concepts) | set(a2.concepts)
        cs     = len(shared)/len(union) if union else 0.0
        ds     = 0.15 if a1.domain == a2.domain else 0.0
        return round(vs*0.5 + cs*0.35 + ds, 4)

    def _clean(self, t: str) -> str:
        t = re.sub(r'[\u064B-\u065F\u0670\u0640]','',t)
        t = re.sub(r'[أإآ]','ا',t)
        return t.replace('ة','ه').strip()

    def _concepts(self, text: str) -> list:
        words = re.findall(r'[\u0600-\u06FF]{2,}|[a-zA-Z]{3,}', text.lower())
        concepts, seen = [], set()
        for w in words:
            cands = [w]
            if w.startswith('ال') and len(w)>3: cands.append(w[2:])
            if w.endswith('ه') and len(w)>3:   cands.append(w[:-1]+'ة')
            if w.endswith('ات') and len(w)>4:  cands.append(w[:-2])
            found = next((c for c in cands if c in _AR_LEXICON), None)
            if found and found not in seen:
                concepts.append(found); seen.add(found)
                for rel in _AR_LEXICON[found].get("rel",[])[:2]:
                    if rel in _AR_LEXICON and rel not in seen:
                        concepts.append(rel); seen.add(rel)
        return (concepts or [w for w in words if len(w)>3][:5])[:12]

    def _intent(self, text: str) -> IntentType:
        for it, pats in _INTENT_RE:
            if any(p.search(text) for p in pats):
                return it
        return IntentType.GENERAL

    def _domain(self, concepts: list) -> SemanticDomain:
        scores: dict = {}
        for c in concepts:
            e = _AR_LEXICON.get(c,{})
            d = e.get("d","عام")
            scores[d] = scores.get(d,0.0) + e.get("w",0.3)
        if not scores: return SemanticDomain.GENERAL
        return _DOMAIN_ENUM_MAP.get(max(scores, key=scores.get), SemanticDomain.GENERAL)

    def _vector(self, concepts: list) -> list:
        v = [0.0]*50
        for c in concepts:
            e = _AR_LEXICON.get(c,{})
            s = _DOMAIN_SLOTS_AR.get(e.get("d","عام"),13)
            if s < 14: v[s] += e.get("w",0.3)
        mx = max(v[:14]) if any(x>0 for x in v[:14]) else 1.0
        for i in range(14): v[i] /= mx
        for c in concepts[:10]:
            s = 14 + (_ar_fnv(c) % 36)
            v[s] = min(1.0, v[s] + _AR_LEXICON.get(c,{}).get("w",0.3))
        return [round(x,4) for x in v]

    def _conf(self, concepts: list) -> float:
        if not concepts: return 0.2
        known = sum(1 for c in concepts if c in _AR_LEXICON)
        return round(min(1.0, known/len(concepts)*0.6 +
            sum(_AR_LEXICON.get(c,{}).get("w",0.3) for c in concepts)/len(concepts)*0.4), 3)

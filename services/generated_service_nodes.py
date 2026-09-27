"""
Phase 5 — منطق حقيقي للعُقد المُولَّدة تلقائياً
=================================================
قبل هذا التعديل، `ServiceGeneratorEngine.instantiate_spec` كانت تُنشئ
`PassThroughNode` لكل أنواع الخدمات الثمانية في `_SERVICE_TEMPLATES`
(transformer/normalizer/aggregator/validator/enricher/router/filter/
analyzer) بغض النظر عن نوعها — أي أن كل عقدة "مولَّدة ذاتياً" كانت في
الواقع تمرّر البيانات كما هي بلا أي معالجة حقيقية، رغم أن اسمها ومخطط
مدخلاتها/مخرجاتها يوحيان بمعالجة فعلية (`normalized_data`,
`validation_report`, `errors`, `score`...).

كل صنف هنا ينفّذ منطقاً حقيقياً وعاماً (لا يعتمد على نوع بيانات محدّد
مسبقاً، لأن هذه عُقد تُولَّد ديناميكياً لسدّ فجوة اكتُشفت تلقائياً بين
عقدتين لا نعرف طبيعة بياناتهما وقت التوليد) يطابق تماماً مخطط الحقول
(input_fields/output_fields/required_inputs) المُعرَّف في
`ai/service_generator.py._SERVICE_TEMPLATES` لكل نوع.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from core.node import BaseNode, NodeSchema

_NUMERIC = (int, float)


def _deep_normalize(value: Any, issues: List[str], path: str = "$") -> Any:
    """تطبيع عميق حقيقي: يشذّب الفراغات، يوحّد مفاتيح القواميس لأحرف
    صغيرة، ويُسقط قيم None — ويسجّل كل تعديل فعلي في issues."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped != value:
            issues.append(f"{path}: تمّ تشذيب الفراغات")
        return stripped
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if v is None:
                issues.append(f"{path}.{k}: أُسقطت قيمة فارغة (None)")
                continue
            nk = k.strip().lower() if isinstance(k, str) else k
            if nk != k:
                issues.append(f"{path}.{k}: تمّ توحيد اسم المفتاح إلى '{nk}'")
            out[nk] = _deep_normalize(v, issues, f"{path}.{nk}")
        return out
    if isinstance(value, list):
        return [_deep_normalize(v, issues, f"{path}[{i}]") for i, v in enumerate(value)]
    return value


class NormalizerNode(BaseNode):
    """تطبيع بيانات خام حقيقي: تشذيب نصوص، توحيد مفاتيح، إسقاط قيم فارغة،
    وتحقّق اختياري من مطابقة schema (قائمة مفاتيح مطلوبة)."""

    def __init__(self, name: str, description: str = "", tags: list = None,
                 node_id: str = None):
        super().__init__(name=name, description=description or "Auto-generated data normalizer",
                          tags=tags or ["generated", "normalizer"], node_id=node_id)

    @property
    def input_schema(self) -> NodeSchema:
        return NodeSchema(fields={"raw_data": "Any", "schema": "dict"}, required=["raw_data"])

    @property
    def output_schema(self) -> NodeSchema:
        return NodeSchema(fields={"normalized_data": "Any", "validation_report": "dict"}, required=[])

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        issues: List[str] = []
        normalized = _deep_normalize(data["raw_data"], issues)

        schema = data.get("schema") or {}
        missing: List[str] = []
        if isinstance(schema, dict) and schema.get("required_keys") and isinstance(normalized, dict):
            for key in schema["required_keys"]:
                lk = key.strip().lower() if isinstance(key, str) else key
                if lk not in normalized:
                    missing.append(key)

        return {
            "normalized_data": normalized,
            "validation_report": {
                "changes_made": len(issues),
                "issues": issues,
                "missing_required_keys": missing,
                "schema_satisfied": not missing,
            },
        }


class ValidatorNode(BaseNode):
    """تحقّق حقيقي من `data` وفق قواعد `rules`: required/types/min/max/pattern."""

    def __init__(self, name: str, description: str = "", tags: list = None,
                 node_id: str = None):
        super().__init__(name=name, description=description or "Auto-generated data validator",
                          tags=tags or ["generated", "validator"], node_id=node_id)

    @property
    def input_schema(self) -> NodeSchema:
        return NodeSchema(fields={"data": "Any", "rules": "dict"}, required=["data"])

    @property
    def output_schema(self) -> NodeSchema:
        return NodeSchema(fields={"valid": "bool", "errors": "list", "data": "Any"}, required=[])

    _TYPE_MAP = {"str": str, "int": int, "float": (int, float), "bool": bool, "list": list, "dict": dict}

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        payload = data["data"]
        rules = data.get("rules") or {}
        errors: List[str] = []

        required = rules.get("required") or []
        if isinstance(payload, dict):
            for field_name in required:
                if field_name not in payload:
                    errors.append(f"حقل مطلوب مفقود: '{field_name}'")

        types = rules.get("types") or {}
        if isinstance(payload, dict) and isinstance(types, dict):
            for field_name, type_name in types.items():
                if field_name not in payload:
                    continue
                py_type = self._TYPE_MAP.get(type_name)
                if py_type and not isinstance(payload[field_name], py_type):
                    errors.append(
                        f"نوع خاطئ للحقل '{field_name}': متوقَّع {type_name}, "
                        f"موجود {type(payload[field_name]).__name__}"
                    )

        ranges = rules.get("ranges") or {}
        if isinstance(payload, dict) and isinstance(ranges, dict):
            for field_name, bounds in ranges.items():
                v = payload.get(field_name)
                if isinstance(v, _NUMERIC) and isinstance(bounds, dict):
                    if "min" in bounds and v < bounds["min"]:
                        errors.append(f"'{field_name}'={v} أقل من الحد الأدنى {bounds['min']}")
                    if "max" in bounds and v > bounds["max"]:
                        errors.append(f"'{field_name}'={v} أكبر من الحد الأقصى {bounds['max']}")

        patterns = rules.get("patterns") or {}
        if isinstance(payload, dict) and isinstance(patterns, dict):
            for field_name, pattern in patterns.items():
                v = payload.get(field_name)
                if isinstance(v, str) and not re.match(pattern, v):
                    errors.append(f"'{field_name}' لا يطابق النمط المطلوب")

        return {"valid": len(errors) == 0, "errors": errors, "data": payload}


class AggregatorNode(BaseNode):
    """تجميع حقيقي لقائمة `items` وفق `strategy`: sum/avg/concat/merge/unique/count."""

    def __init__(self, name: str, description: str = "", tags: list = None,
                 node_id: str = None):
        super().__init__(name=name, description=description or "Auto-generated data aggregator",
                          tags=tags or ["generated", "aggregator"], node_id=node_id)

    @property
    def input_schema(self) -> NodeSchema:
        return NodeSchema(fields={"items": "list", "strategy": "str"}, required=["items"])

    @property
    def output_schema(self) -> NodeSchema:
        return NodeSchema(fields={"aggregated": "Any", "count": "int"}, required=[])

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        items = list(data["items"] or [])
        strategy = (data.get("strategy") or "auto").lower()
        numeric = [i for i in items if isinstance(i, _NUMERIC)]

        if strategy == "sum" or (strategy == "auto" and numeric and len(numeric) == len(items)):
            aggregated: Any = sum(numeric)
        elif strategy == "avg" and numeric:
            aggregated = sum(numeric) / len(numeric)
        elif strategy == "unique":
            seen = []
            for i in items:
                if i not in seen:
                    seen.append(i)
            aggregated = seen
        elif strategy == "merge" and all(isinstance(i, dict) for i in items):
            merged: Dict[str, Any] = {}
            for i in items:
                merged.update(i)
            aggregated = merged
        elif strategy == "concat":
            aggregated = "".join(str(i) for i in items)
        else:
            aggregated = items

        return {"aggregated": aggregated, "count": len(items)}


_OPS = {
    "$gt": lambda v, x: v is not None and v > x,
    "$gte": lambda v, x: v is not None and v >= x,
    "$lt": lambda v, x: v is not None and v < x,
    "$lte": lambda v, x: v is not None and v <= x,
    "$ne": lambda v, x: v != x,
    "$in": lambda v, x: v in x,
}


def _matches_criteria(item: Any, criteria: Dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return item in criteria.values() if criteria else True
    for key, expected in criteria.items():
        actual = item.get(key)
        if isinstance(expected, dict) and any(op in expected for op in _OPS):
            for op, op_val in expected.items():
                fn = _OPS.get(op)
                if fn and not fn(actual, op_val):
                    return False
        elif actual != expected:
            return False
    return True


class FilterNode(BaseNode):
    """تصفية حقيقية لقائمة `items` وفق `criteria` (مطابقة مباشرة أو
    مشغّلات $gt/$gte/$lt/$lte/$ne/$in)."""

    def __init__(self, name: str, description: str = "", tags: list = None,
                 node_id: str = None):
        super().__init__(name=name, description=description or "Auto-generated item filter",
                          tags=tags or ["generated", "filter"], node_id=node_id)

    @property
    def input_schema(self) -> NodeSchema:
        return NodeSchema(fields={"items": "list", "criteria": "dict"}, required=["items"])

    @property
    def output_schema(self) -> NodeSchema:
        return NodeSchema(fields={"filtered": "list", "removed_count": "int"}, required=[])

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        items = list(data["items"] or [])
        criteria = data.get("criteria") or {}
        filtered = [i for i in items if _matches_criteria(i, criteria)] if criteria else items
        return {"filtered": filtered, "removed_count": len(items) - len(filtered)}


class EnricherNode(BaseNode):
    """إثراء حقيقي: يدمج `context` داخل `data` بدون الكتابة فوق حقول
    موجودة مسبقاً، ويُرجع فعلياً أسماء الحقول المُضافة."""

    def __init__(self, name: str, description: str = "", tags: list = None,
                 node_id: str = None):
        super().__init__(name=name, description=description or "Auto-generated data enricher",
                          tags=tags or ["generated", "enricher"], node_id=node_id)

    @property
    def input_schema(self) -> NodeSchema:
        return NodeSchema(fields={"data": "Any", "context": "dict"}, required=["data"])

    @property
    def output_schema(self) -> NodeSchema:
        return NodeSchema(fields={"enriched_data": "Any", "added_fields": "list"}, required=[])

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        payload = data["data"]
        context = data.get("context") or {}
        if not isinstance(payload, dict) or not isinstance(context, dict):
            return {"enriched_data": payload, "added_fields": []}

        enriched = dict(payload)
        added: List[str] = []
        for k, v in context.items():
            if k not in enriched:
                enriched[k] = v
                added.append(k)
        return {"enriched_data": enriched, "added_fields": added}


class RouterNode(BaseNode):
    """توجيه حقيقي: يحدّد وجهة `data` بناءً على قيمة `routing_key` فعلياً
    الموجودة داخل البيانات — وليس قيمة ثابتة."""

    def __init__(self, name: str, description: str = "", tags: list = None,
                 node_id: str = None):
        super().__init__(name=name, description=description or "Auto-generated data router",
                          tags=tags or ["generated", "router"], node_id=node_id)

    @property
    def input_schema(self) -> NodeSchema:
        return NodeSchema(fields={"data": "Any", "routing_key": "str"}, required=["data"])

    @property
    def output_schema(self) -> NodeSchema:
        return NodeSchema(fields={"routed_data": "Any", "destination": "str"}, required=[])

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        payload = data["data"]
        routing_key = data.get("routing_key")

        if routing_key and isinstance(payload, dict) and routing_key in payload:
            destination = str(payload[routing_key])
        elif isinstance(payload, dict) and payload:
            destination = f"dict:{sorted(payload.keys())[0]}"
        else:
            destination = f"type:{type(payload).__name__}"

        return {"routed_data": payload, "destination": destination}


class AnalyzerNode(BaseNode):
    """تحليل حقيقي لـ `content` (نص/قائمة/قاموس/رقم) بإحصاءات فعلية
    محسوبة من المحتوى نفسه، وليست قيماً وهمية ثابتة."""

    def __init__(self, name: str, description: str = "", tags: list = None,
                 node_id: str = None):
        super().__init__(name=name, description=description or "Auto-generated content analyzer",
                          tags=tags or ["generated", "analyzer"], node_id=node_id)

    @property
    def input_schema(self) -> NodeSchema:
        return NodeSchema(fields={"content": "Any", "analysis_type": "str"}, required=["content"])

    @property
    def output_schema(self) -> NodeSchema:
        return NodeSchema(fields={"analysis": "dict", "score": "float", "insights": "list"}, required=[])

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        content = data["content"]
        insights: List[str] = []

        if isinstance(content, str):
            words = content.split()
            unique_words = set(w.lower().strip(".,!?؛،") for w in words)
            analysis = {
                "type": "text",
                "char_count": len(content),
                "word_count": len(words),
                "unique_word_count": len(unique_words),
                "avg_word_length": round(sum(len(w) for w in words) / len(words), 2) if words else 0.0,
            }
            score = min(1.0, len(content) / 1000.0)
            if len(words) and analysis["unique_word_count"] / len(words) < 0.4:
                insights.append("تكرار عالٍ للكلمات — تنوّع مفرداتي منخفض")
            if not words:
                insights.append("محتوى نصي فارغ")
        elif isinstance(content, list):
            type_counts: Dict[str, int] = {}
            for item in content:
                t = type(item).__name__
                type_counts[t] = type_counts.get(t, 0) + 1
            analysis = {"type": "list", "length": len(content), "type_distribution": type_counts}
            score = min(1.0, len(content) / 100.0)
            if len(type_counts) > 1:
                insights.append("القائمة تحتوي أنواع بيانات مختلطة")
        elif isinstance(content, dict):
            analysis = {"type": "dict", "key_count": len(content), "keys": list(content.keys())[:20]}
            score = min(1.0, len(content) / 50.0)
        elif isinstance(content, _NUMERIC):
            analysis = {"type": "number", "value": content, "is_negative": content < 0}
            score = 1.0
        else:
            analysis = {"type": type(content).__name__}
            score = 0.0
            insights.append("نوع محتوى غير معروف — تحليل محدود")

        return {"analysis": analysis, "score": round(score, 4), "insights": insights}


class TransformerNode(BaseNode):
    """تحويل حقيقي لـ `data` إلى الصيغة المطلوبة في `format`
    (json/str/list/dict) — تحويل فعلي وليس تمريراً بلا تغيير."""

    def __init__(self, name: str, description: str = "", tags: list = None,
                 node_id: str = None):
        super().__init__(name=name, description=description or "Auto-generated data transformer",
                          tags=tags or ["generated", "transformer"], node_id=node_id)

    @property
    def input_schema(self) -> NodeSchema:
        return NodeSchema(fields={"data": "Any", "format": "str"}, required=["data"])

    @property
    def output_schema(self) -> NodeSchema:
        return NodeSchema(fields={"result": "Any", "transformed": "bool"}, required=[])

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        value = data["data"]
        fmt = (data.get("format") or "").lower()

        if fmt == "json":
            result = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            changed = not isinstance(value, str)
        elif fmt == "str":
            result = value if isinstance(value, str) else str(value)
            changed = not isinstance(value, str)
        elif fmt == "list":
            if isinstance(value, list):
                result, changed = value, False
            elif isinstance(value, (tuple, set)):
                result, changed = list(value), True
            else:
                result, changed = [value], True
        elif fmt == "dict":
            if isinstance(value, dict):
                result, changed = value, False
            else:
                result, changed = {"value": value}, True
        else:
            result, changed = value, False

        return {"result": result, "transformed": changed}


# خريطة نوع الخدمة (كما يحسبها ServiceGeneratorEngine._infer_service_type)
# → الصنف الحقيقي المقابل له.
NODE_CLASS_BY_SERVICE_TYPE = {
    "normalizer": NormalizerNode,
    "validator": ValidatorNode,
    "aggregator": AggregatorNode,
    "filter": FilterNode,
    "enricher": EnricherNode,
    "router": RouterNode,
    "analyzer": AnalyzerNode,
    "transformer": TransformerNode,
}

# خريطة اسم الصنف (BaseNode.metadata.node_type = self.__class__.__name__،
# محفوظ فعلياً في meta_cache/node.to_dict() منذ إنشاء العقدة) → الصنف
# نفسه. تُستخدم في MeshBundle._restore_dynamic_nodes بعد إعادة تشغيل
# العملية لإحياء عقدة self_evolved بنوعها الحقيقي (NormalizerNode مثلاً)
# بدل تنزيلها جميعاً إلى PassThroughNode عام — كان هذا مقبولاً قبل هذا
# الملف لأن كل عقدة self_evolved كانت PassThroughNode أصلاً، لكن أصبح
# سيفقد المنطق الحقيقي بعد كل إعادة تشغيل لولا هذه الخريطة.
NODE_CLASS_BY_NAME = {cls.__name__: cls for cls in NODE_CLASS_BY_SERVICE_TYPE.values()}

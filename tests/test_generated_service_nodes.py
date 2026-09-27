"""
قبل هذا التعديل: `ServiceGeneratorEngine.instantiate_spec` كانت تُنشئ
`PassThroughNode` (تمرير بلا تغيير) لكل الأنواع الثمانية في
`_SERVICE_TEMPLATES`، رغم أن output_fields توحي بمعالجة حقيقية
(`validation_report`, `errors`, `score`, `destination`...). هذا الملف
يبني عبر `ServiceGeneratorEngine.generate_for_gap` + `instantiate_spec`
عقدة حقيقية لكل نوع، وينفّذها فعلياً عبر `node.execute()` (وليس فقط
`process()` مباشرة) للتأكد أن schema validation تمرّ أيضاً، ويتحقق أن
الناتج محسوب فعلياً من المدخلات لا قيمة ثابتة.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.service_generator import ServiceGeneratorEngine
from services.dynamic_node import PassThroughNode
from services.generated_service_nodes import (
    AggregatorNode, AnalyzerNode, EnricherNode, FilterNode,
    NormalizerNode, RouterNode, TransformerNode, ValidatorNode,
)


def _instantiate(missing_svc: str):
    gen = ServiceGeneratorEngine()
    spec = gen.generate_for_gap({
        "source_node": {"name": "SourceX", "capability": "produce raw data"},
        "target_node": {"name": "TargetY", "capability": "consume clean data"},
        "missing_service": missing_svc,
        "confidence": 0.9,
        "gap_type": "capability_gap",
    })
    node = gen.instantiate_spec(spec)
    assert spec.status == "active"
    return node


class TestServiceTypeDispatch:
    def test_all_eight_types_get_real_classes_not_passthrough(self):
        expected = {
            "Normalizer": NormalizerNode,
            "Validator": ValidatorNode,
            "Aggregator": AggregatorNode,
            "Filter": FilterNode,
            "Enricher": EnricherNode,
            "Router": RouterNode,
            "Analyzer": AnalyzerNode,
            "Transformer": TransformerNode,
        }
        for missing_svc, cls in expected.items():
            node = _instantiate(missing_svc)
            assert isinstance(node, cls), f"{missing_svc} -> {type(node).__name__}, expected {cls.__name__}"
            assert not isinstance(node, PassThroughNode)


class TestNormalizerNode:
    def test_strips_and_lowercases_and_drops_none(self):
        node = _instantiate("Normalizer")
        out = node.execute({"raw_data": {" Name ": "  Ali  ", "Age": 30, "Note": None}})
        assert out["normalized_data"] == {"name": "Ali", "age": 30}
        assert out["validation_report"]["changes_made"] > 0

    def test_reports_missing_required_keys(self):
        node = _instantiate("Normalizer")
        out = node.execute({
            "raw_data": {"name": "Ali"},
            "schema": {"required_keys": ["name", "email"]},
        })
        assert out["validation_report"]["missing_required_keys"] == ["email"]
        assert out["validation_report"]["schema_satisfied"] is False


class TestValidatorNode:
    def test_detects_missing_field_and_wrong_type(self):
        node = _instantiate("Validator")
        out = node.execute({
            "data": {"age": "not-a-number"},
            "rules": {"required": ["name"], "types": {"age": "int"}},
        })
        assert out["valid"] is False
        assert any("name" in e for e in out["errors"])
        assert any("age" in e for e in out["errors"])

    def test_valid_data_passes(self):
        node = _instantiate("Validator")
        out = node.execute({
            "data": {"name": "Ali", "age": 30},
            "rules": {"required": ["name"], "types": {"age": "int"}, "ranges": {"age": {"min": 0, "max": 120}}},
        })
        assert out["valid"] is True
        assert out["errors"] == []


class TestAggregatorNode:
    def test_sum_strategy(self):
        node = _instantiate("Aggregator")
        out = node.execute({"items": [1, 2, 3], "strategy": "sum"})
        assert out["aggregated"] == 6
        assert out["count"] == 3

    def test_merge_strategy(self):
        node = _instantiate("Aggregator")
        out = node.execute({"items": [{"a": 1}, {"b": 2}], "strategy": "merge"})
        assert out["aggregated"] == {"a": 1, "b": 2}


class TestFilterNode:
    def test_exact_match_and_operator(self):
        node = _instantiate("Filter")
        items = [{"age": 10}, {"age": 20}, {"age": 30}]
        out = node.execute({"items": items, "criteria": {"age": {"$gt": 15}}})
        assert out["filtered"] == [{"age": 20}, {"age": 30}]
        assert out["removed_count"] == 1


class TestEnricherNode:
    def test_adds_only_missing_keys(self):
        node = _instantiate("Enricher")
        out = node.execute({"data": {"a": 1}, "context": {"a": 999, "b": 2}})
        assert out["enriched_data"] == {"a": 1, "b": 2}
        assert out["added_fields"] == ["b"]


class TestRouterNode:
    def test_routes_by_actual_field_value(self):
        node = _instantiate("Router")
        out = node.execute({"data": {"lang": "ar", "text": "hi"}, "routing_key": "lang"})
        assert out["destination"] == "ar"


class TestAnalyzerNode:
    def test_text_stats_are_computed_not_fixed(self):
        node = _instantiate("Analyzer")
        out = node.execute({"content": "hello world hello"})
        assert out["analysis"]["word_count"] == 3
        assert out["analysis"]["unique_word_count"] == 2


class TestTransformerNode:
    def test_converts_dict_to_json_string(self):
        node = _instantiate("Transformer")
        out = node.execute({"data": {"a": 1}, "format": "json"})
        assert isinstance(out["result"], str)
        assert out["transformed"] is True

    def test_noop_when_already_target_format(self):
        node = _instantiate("Transformer")
        out = node.execute({"data": "already a string", "format": "str"})
        assert out["transformed"] is False


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

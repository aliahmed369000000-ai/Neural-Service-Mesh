"""
سجل التقنيات المفعّلة في NSM — مرجع سريع للمطورين والواجهة.
"""
from __future__ import annotations

from typing import Any, Dict, List


def list_techniques() -> List[Dict[str, Any]]:
    return [
        {
            "id": "arabic_text_clean",
            "name": "تنظيف وتطبيع عربي",
            "module": "ai.arabic_text_clean",
            "status": "core",
        },
        {
            "id": "modern_bbpe",
            "name": "Modern BBPE (GPT-4/tiktoken style)",
            "module": "ai.modern_bbpe_tokenizer",
            "status": "core",
        },
        {
            "id": "tokenizer_factory",
            "name": "مصنع التوكنايزر الموحّد",
            "module": "ai.tokenizer_factory",
            "status": "core",
        },
        {
            "id": "nucleus_sampling",
            "name": "Nucleus / top-k / repetition penalty",
            "module": "ai.sampling_utils",
            "status": "core",
        },
        {
            "id": "yemeni_rag",
            "name": "RAG BM25 على جمل يمنية",
            "module": "ai.yemeni_rag",
            "status": "core",
        },
        {
            "id": "query_expand",
            "name": "توسيع استعلام فصيح↔يمني",
            "module": "ai.query_expand",
            "status": "core",
        },
        {
            "id": "dialect_boost",
            "name": "تعزيز لهجي + حقن RAG في QA",
            "module": "ai.dialect_boost",
            "status": "core",
        },
        {
            "id": "nlp_pipeline",
            "name": "خط أنابيب NLP موحّد",
            "module": "ai.nlp_pipeline",
            "status": "core",
        },
        {
            "id": "camel_optional",
            "name": "CAMeL Tools اختياري (DID SAN + صرف MSA)",
            "module": "ai.camel_optional",
            "status": "optional",
        },
        {
            "id": "yemeni_voice",
            "name": "مسار صوتي يمني (STT/TTS)",
            "module": "ai.yemeni_voice",
            "status": "optional",
        },
        {
            "id": "diarization_optional",
            "name": "فصل متحدثين (pyannote) اختياري",
            "module": "ai.diarization_optional",
            "status": "optional",
        },
        {
            "id": "sequence_packing",
            "name": "Sequence packing في تدريب ArabicTransformer",
            "module": "train_batch_v3",
            "status": "core",
        },
    ]


def techniques_summary() -> str:
    lines = ["تقنيات NSM النشطة:", ""]
    for t in list_techniques():
        lines.append(f"- [{t['status']}] {t['name']} — `{t['module']}`")
    return "\n".join(lines)

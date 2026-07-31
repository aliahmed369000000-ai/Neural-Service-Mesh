"""
اختبارات Experience Learning:
  - EpisodeStore (تخزين/استرجاع/تنوع)
  - score_episode (تقييم الجودة)
  - ExperienceTrainer (replay top/recent/diverse + دورة كاملة)
  - ReasoningPipeline integration (إنشاء Episode تلقائياً)
  - إصلاح CKG <-> ArabicNLPEngine (ckg._concepts)
"""
import os
import shutil
import tempfile

import numpy as np
import pytest

from ai.experience_store import Episode, EpisodeStore
from ai.experience_trainer import (
    ExperienceTrainer, score_episode,
    score_concept_coverage, score_relation_coverage,
    score_memory_recall_quality, score_answer_confidence,
)
from ai.neural_core import NeuralCore
from ai.knowledge_trainer import CKGManager
from ai.arabic_nlp import ArabicNLPEngine
from ai.reasoning_pipeline import ReasoningPipeline, DEFAULT_INPUT_DIM


CLUSTERS = ["توحيد", "إيمان", "آخرة", "فقه", "عبادة"]


def _make_episode(i):
    return Episode(
        question=f"سؤال رقم {i}",
        matched_concepts=[{"name": f"concept_{i}", "cluster": CLUSTERS[i % len(CLUSTERS)],
                            "strength": 0.1 * (i % 10 + 1), "frequency": i}],
        related_concepts=[{"name": f"rel_{i}", "cluster": CLUSTERS[(i + 1) % len(CLUSTERS)],
                            "relation_type": "co_occurrence", "relation_weight": 0.05 * (i % 10 + 1), "via": f"concept_{i}"}],
        decision_weights={"W_SEMANTIC": 0.3, "W_SCORE": 0.25, "W_MEMORY": 0.2, "W_TOPOLOGY": 0.25},
        confidence=0.05 * (i % 10 + 1),
        answer=f"إجابة {i}",
        context_vector=[0.1 * i % 1.0] * 7,
        target_used=[0.3, 0.35, 0.25, 0.10],
        train_loss=0.1 - 0.005 * i,
        memory_hits=[],
        quality={
            "concept_coverage": 0.1 * (i % 10 + 1),
            "relation_coverage": 0.05 * (i % 10 + 1),
            "memory_recall_quality": 0.0,
            "answer_confidence": 0.05 * (i % 10 + 1),
            "overall_quality": 0.05 * (i % 10 + 1),
        },
    )


@pytest.fixture()
def populated_store(tmp_path):
    db_path = str(tmp_path / "experience.db")
    store = EpisodeStore(db_path)
    for i in range(15):
        store.add(_make_episode(i))
    return store


def test_ckg_arabic_nlp_engine_alignment():
    ckg = CKGManager()
    assert hasattr(ckg, "_concepts")
    assert len(ckg._concepts) == len(ckg._data.get("concepts", {})) and len(ckg._concepts) > 0

    engine = ArabicNLPEngine(ckg=ckg)
    res = engine.analyse("من هو الله ومن هو الرحمن؟")
    assert res.semantic.ckg_aligned is True
    assert res.feature_vector.semantic_concept_score == 1.0


def test_episode_store_basic(populated_store):
    store = populated_store
    assert store.count() == 15

    recent = store.get_recent(limit=5)
    assert len(recent) == 5
    assert all(recent[i].timestamp >= recent[i + 1].timestamp for i in range(len(recent) - 1))

    top = store.get_top_by_quality(limit=5)
    assert len(top) == 5
    qualities = [ep.quality["overall_quality"] for ep in top]
    assert all(qualities[i] >= qualities[i + 1] for i in range(len(qualities) - 1))

    diverse = store.get_diverse_sample(limit=5, seed=42)
    assert len(diverse) <= 5
    diverse_clusters = [ep.matched_concepts[0]["cluster"] for ep in diverse]
    assert len(set(diverse_clusters)) >= 2

    store.mark_replayed([ep.episode_id for ep in recent[:2]])
    stats = store.stats()
    assert stats["total_replays"] == 2
    assert stats["total_episodes"] == 15


def test_score_episode_components():
    assert score_concept_coverage([]) == 0.0
    assert score_concept_coverage([{"name": f"c{i}"} for i in range(5)], max_expected=5) == 1.0
    assert score_concept_coverage([{"name": f"c{i}"} for i in range(10)], max_expected=5) == 1.0
    assert abs(score_concept_coverage([{"name": "a"}, {"name": "b"}], max_expected=5) - 0.4) < 1e-9

    assert score_relation_coverage([]) == 0.0
    rc = score_relation_coverage([{"relation_weight": 0.8}, {"relation_weight": 0.6}], max_expected=10)
    assert rc > 0.0

    assert score_memory_recall_quality([]) == 0.0
    mrq_self_only = score_memory_recall_quality([{"similarity": 1.0}])
    assert mrq_self_only == 1.0
    mrq_mixed = score_memory_recall_quality([{"similarity": 1.0}, {"similarity": 0.6}, {"similarity": 0.4}])
    assert abs(mrq_mixed - 0.5) < 1e-6

    ac_no_match = score_answer_confidence({"W_SEMANTIC": 0.4}, [])
    assert abs(ac_no_match - 0.2) < 1e-6
    ac_match = score_answer_confidence({"W_SEMANTIC": 0.4}, [{"strength": 0.8}])
    assert abs(ac_match - (0.5 * 0.4 + 0.5 * 0.8)) < 1e-6

    full = score_episode(
        matched_concepts=[{"name": "a", "strength": 0.9}, {"name": "b", "strength": 0.7}],
        related_concepts=[{"relation_weight": 0.5}],
        memory_hits=[{"similarity": 1.0}, {"similarity": 0.7}],
        decision_weights={"W_SEMANTIC": 0.3, "W_SCORE": 0.25, "W_MEMORY": 0.2, "W_TOPOLOGY": 0.25},
    )
    assert set(full.keys()) == {"concept_coverage", "relation_coverage", "memory_recall_quality",
                                 "answer_confidence", "overall_quality"}
    assert abs(full["overall_quality"] - np.mean([
        full["concept_coverage"], full["relation_coverage"],
        full["memory_recall_quality"], full["answer_confidence"]
    ])) < 1e-9
    assert all(0.0 <= v <= 1.0 for v in full.values())


def test_experience_trainer_replay_strategies(populated_store, tmp_path):
    store = populated_store
    core = NeuralCore(input_dim=7, hidden_dims=[10], output_dim=4, seed=123)
    trainer = ExperienceTrainer(core, store)

    rep_top = trainer.replay_top(limit=5)
    assert rep_top.episodes_used == 5
    assert rep_top.avg_loss_before is not None and rep_top.avg_loss_after is not None

    rep_recent = trainer.replay_recent(limit=5)
    assert rep_recent.episodes_used == 5

    rep_diverse = trainer.replay_diverse(limit=5, seed=7)
    assert 0 < rep_diverse.episodes_used <= 5

    cycle = trainer.run_training_cycle(top_limit=3, recent_limit=3, diverse_limit=3, save=False, seed=1)
    assert cycle["status"] == "ok"
    assert all(k in cycle for k in ("top", "recent", "diverse"))

    empty_store = EpisodeStore(str(tmp_path / "empty.db"))
    empty_trainer = ExperienceTrainer(core, empty_store)
    empty_cycle = empty_trainer.run_training_cycle(save=False)
    assert empty_cycle["status"] == "no_episodes"


def test_reasoning_pipeline_creates_episode(tmp_path):
    cwd = os.getcwd()
    pipe_dir = tempfile.mkdtemp()
    os.chdir(pipe_dir)
    try:
        shutil.copytree(os.path.join(cwd, "knowledge"), os.path.join(pipe_dir, "knowledge"))

        pipe_store = EpisodeStore(os.path.join(pipe_dir, "memory", "experience.db"))
        pipeline = ReasoningPipeline(
            train_on_query=True,
            core_save_path=None,  # تجنب الكتابة على models/ المشتركة في الاختبار
            episode_store=pipe_store,
            # ArabicTransformer الحالي (D_MODEL=2304/D_FF=8384/N_LAYERS=16) يبني
            # ~995M باراميتر فعلياً (وليس ~40M كما موثّق) ويستهلك 8-24GB رام عند
            # البناء — يُسبّب OOM في أي بيئة CI عادية. مُعطَّل هنا عمداً حتى تُحسم
            # معمارية الأبعاد لاحقاً؛ هذا لا يغيّر سلوك الإنتاج لأن المعامل
            # اختياري أصلاً في ReasoningPipeline.
            transformer_weights_path=None,
        )

        result = pipeline.answer("من هو الله ومن هو الرحمن؟")
        assert result.episode_id is not None
        assert result.quality is not None and "overall_quality" in result.quality
        assert pipe_store.count() == 1

        stored = pipe_store.get_recent(limit=1)[0]
        assert stored.question == "من هو الله ومن هو الرحمن؟"
        assert len(stored.context_vector) == DEFAULT_INPUT_DIM
        assert len(stored.matched_concepts) > 0
        assert stored.quality == result.quality

        pipeline.answer("ما هو الإيمان؟")
        assert pipe_store.count() == 2

        pipe_trainer = ExperienceTrainer(pipeline.core, pipe_store)
        pipe_cycle = pipe_trainer.run_training_cycle(top_limit=2, recent_limit=2, diverse_limit=2, save=False)
        assert pipe_cycle["status"] == "ok"
        assert any(pipe_cycle[s]["improved"] for s in ("top", "recent", "diverse"))
    finally:
        os.chdir(cwd)
        shutil.rmtree(pipe_dir, ignore_errors=True)

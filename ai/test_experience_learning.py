"""
اختبارات Experience Learning:
  - EpisodeStore (تخزين/استرجاع/تنوع)
  - score_episode (تقييم الجودة)
  - ExperienceTrainer (replay top/recent/diverse + دورة كاملة)
  - ReasoningPipeline integration (إنشاء Episode تلقائياً)
  - إصلاح CKG <-> ArabicNLPEngine (ckg._concepts)
"""
import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from ai.experience_store import Episode, EpisodeStore
from ai.experience_trainer import (
    ExperienceTrainer, score_episode,
    score_concept_coverage, score_relation_coverage,
    score_memory_recall_quality, score_answer_confidence,
)
from ai.neural_core import NeuralCore
from ai.knowledge_trainer import CKGManager
from ai.arabic_nlp import ArabicNLPEngine
from ai.reasoning_pipeline import ReasoningPipeline


PASS = []
FAIL = []


def check(name, condition, detail=""):
    if condition:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("1) إصلاح CKG <-> ArabicNLPEngine (ckg._concepts)")
print("=" * 70)

ckg = CKGManager()
check("CKGManager has _concepts property", hasattr(ckg, "_concepts"))
check("_concepts returns real concepts dict",
      len(ckg._concepts) == len(ckg._data.get("concepts", {})) and len(ckg._concepts) > 0,
      f"len={len(ckg._concepts)}")

engine = ArabicNLPEngine(ckg=ckg)
res = engine.analyse("من هو الله ومن هو الرحمن؟")
check("ArabicNLPEngine.semantic.ckg_aligned == True", res.semantic.ckg_aligned is True)
check("semantic_concept_score boosted by ckg_bonus (==1.0 here)",
      res.feature_vector.semantic_concept_score == 1.0,
      f"got={res.feature_vector.semantic_concept_score}")


# ════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("2) EpisodeStore: تخزين/استرجاع/تنوع")
print("=" * 70)

tmpdir = tempfile.mkdtemp()
db_path = os.path.join(tmpdir, "experience.db")
store = EpisodeStore(db_path)

check("store starts empty", store.count() == 0)

clusters = ["توحيد", "إيمان", "آخرة", "فقه", "عبادة"]
for i in range(15):
    ep = Episode(
        question=f"سؤال رقم {i}",
        matched_concepts=[{"name": f"concept_{i}", "cluster": clusters[i % len(clusters)],
                            "strength": 0.1 * (i % 10 + 1), "frequency": i}],
        related_concepts=[{"name": f"rel_{i}", "cluster": clusters[(i+1) % len(clusters)],
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
    store.add(ep)

check("store.count() == 15", store.count() == 15, f"got={store.count()}")

recent = store.get_recent(limit=5)
check("get_recent returns 5", len(recent) == 5, f"got={len(recent)}")
check("get_recent is DESC by timestamp",
      all(recent[i].timestamp >= recent[i+1].timestamp for i in range(len(recent)-1)))

top = store.get_top_by_quality(limit=5)
check("get_top_by_quality returns 5", len(top) == 5)
qualities = [ep.quality["overall_quality"] for ep in top]
check("get_top_by_quality is sorted DESC",
      all(qualities[i] >= qualities[i+1] for i in range(len(qualities)-1)),
      f"qualities={qualities}")

diverse = store.get_diverse_sample(limit=5, seed=42)
check("get_diverse_sample returns <= 5", len(diverse) <= 5, f"got={len(diverse)}")
diverse_clusters = [ep.matched_concepts[0]["cluster"] for ep in diverse]
check("get_diverse_sample has multiple distinct clusters",
      len(set(diverse_clusters)) >= 2, f"clusters={diverse_clusters}")

store.mark_replayed([ep.episode_id for ep in recent[:2]])
stats = store.stats()
check("stats() total_replays == 2 after marking 2", stats["total_replays"] == 2, f"stats={stats}")
check("stats() total_episodes == 15", stats["total_episodes"] == 15)


# ════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("3) score_episode: مكوّنات تقييم الجودة")
print("=" * 70)

# concept_coverage
check("concept_coverage([]) == 0.0", score_concept_coverage([]) == 0.0)
check("concept_coverage(5 concepts, max=5) == 1.0",
      score_concept_coverage([{"name": f"c{i}"} for i in range(5)], max_expected=5) == 1.0)
check("concept_coverage(10 concepts, max=5) capped at 1.0",
      score_concept_coverage([{"name": f"c{i}"} for i in range(10)], max_expected=5) == 1.0)
check("concept_coverage(2 concepts, max=5) == 0.4",
      abs(score_concept_coverage([{"name": "a"}, {"name": "b"}], max_expected=5) - 0.4) < 1e-9)

# relation_coverage
check("relation_coverage([]) == 0.0", score_relation_coverage([]) == 0.0)
rc = score_relation_coverage(
    [{"relation_weight": 0.8}, {"relation_weight": 0.6}], max_expected=10)
check("relation_coverage > 0 for non-empty", rc > 0.0, f"got={rc}")

# memory_recall_quality
check("memory_recall_quality([]) == 0.0", score_memory_recall_quality([]) == 0.0)
mrq_self_only = score_memory_recall_quality([{"similarity": 1.0}])
check("memory_recall_quality single self-match == 1.0", mrq_self_only == 1.0, f"got={mrq_self_only}")
mrq_mixed = score_memory_recall_quality([{"similarity": 1.0}, {"similarity": 0.6}, {"similarity": 0.4}])
check("memory_recall_quality excludes self-match (==avg(0.6,0.4))",
      abs(mrq_mixed - 0.5) < 1e-6, f"got={mrq_mixed}")

# answer_confidence
ac_no_match = score_answer_confidence({"W_SEMANTIC": 0.4}, [])
check("answer_confidence with no matches == 0.5*W_SEMANTIC",
      abs(ac_no_match - 0.2) < 1e-6, f"got={ac_no_match}")
ac_match = score_answer_confidence({"W_SEMANTIC": 0.4}, [{"strength": 0.8}])
check("answer_confidence with matches == 0.5*Wsem + 0.5*avg(strength)",
      abs(ac_match - (0.5*0.4 + 0.5*0.8)) < 1e-6, f"got={ac_match}")

# score_episode overall
full = score_episode(
    matched_concepts=[{"name": "a", "strength": 0.9}, {"name": "b", "strength": 0.7}],
    related_concepts=[{"relation_weight": 0.5}],
    memory_hits=[{"similarity": 1.0}, {"similarity": 0.7}],
    decision_weights={"W_SEMANTIC": 0.3, "W_SCORE": 0.25, "W_MEMORY": 0.2, "W_TOPOLOGY": 0.25},
)
check("score_episode returns all 5 keys",
      set(full.keys()) == {"concept_coverage", "relation_coverage", "memory_recall_quality",
                            "answer_confidence", "overall_quality"},
      f"keys={set(full.keys())}")
check("overall_quality is mean of 4 components",
      abs(full["overall_quality"] - np.mean([
          full["concept_coverage"], full["relation_coverage"],
          full["memory_recall_quality"], full["answer_confidence"]
      ])) < 1e-9)
check("all quality scores in [0,1]",
      all(0.0 <= v <= 1.0 for v in full.values()), f"full={full}")


# ════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("4) ExperienceTrainer: replay top/recent/diverse")
print("=" * 70)

core = NeuralCore(input_dim=7, hidden_dims=[10], output_dim=4, seed=123)
trainer = ExperienceTrainer(core, store)

rep_top = trainer.replay_top(limit=5)
check("replay_top uses up to 5 episodes", rep_top.episodes_used == 5, f"got={rep_top.episodes_used}")
check("replay_top reports avg_loss_before/after", rep_top.avg_loss_before is not None and rep_top.avg_loss_after is not None)

rep_recent = trainer.replay_recent(limit=5)
check("replay_recent uses up to 5 episodes", rep_recent.episodes_used == 5, f"got={rep_recent.episodes_used}")

rep_diverse = trainer.replay_diverse(limit=5, seed=7)
check("replay_diverse uses up to 5 episodes", rep_diverse.episodes_used <= 5 and rep_diverse.episodes_used > 0,
      f"got={rep_diverse.episodes_used}")

# دورة كاملة
cycle = trainer.run_training_cycle(top_limit=3, recent_limit=3, diverse_limit=3, save=False, seed=1)
check("run_training_cycle status == ok", cycle["status"] == "ok", f"cycle={cycle}")
check("run_training_cycle has top/recent/diverse reports",
      all(k in cycle for k in ("top", "recent", "diverse")))

# دورة على مخزن فارغ
empty_store = EpisodeStore(os.path.join(tmpdir, "empty.db"))
empty_trainer = ExperienceTrainer(core, empty_store)
empty_cycle = empty_trainer.run_training_cycle(save=False)
check("run_training_cycle on empty store returns no_episodes",
      empty_cycle["status"] == "no_episodes", f"got={empty_cycle}")


# ════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("5) ReasoningPipeline integration: إنشاء Episode تلقائياً")
print("=" * 70)

pipe_dir = tempfile.mkdtemp()
cwd = os.getcwd()
os.chdir(pipe_dir)
try:
    shutil.copytree(os.path.join(cwd, "knowledge"), os.path.join(pipe_dir, "knowledge"))

    pipe_store = EpisodeStore(os.path.join(pipe_dir, "memory", "experience.db"))
    pipeline = ReasoningPipeline(
        train_on_query=True,
        core_save_path=None,  # تجنب الكتابة على models/ المشتركة في الاختبار
        episode_store=pipe_store,
    )

    result = pipeline.answer("من هو الله ومن هو الرحمن؟")
    check("answer() returns episode_id", result.episode_id is not None)
    check("answer() returns quality dict",
          result.quality is not None and "overall_quality" in result.quality)
    check("episode stored in EpisodeStore", pipe_store.count() == 1, f"count={pipe_store.count()}")

    stored = pipe_store.get_recent(limit=1)[0]
    check("stored episode question matches", stored.question == "من هو الله ومن هو الرحمن؟")
    check("stored episode has context_vector of len 7", len(stored.context_vector) == 7,
          f"len={len(stored.context_vector)}")
    check("stored episode has non-empty matched_concepts",
          len(stored.matched_concepts) > 0, f"matched={stored.matched_concepts}")
    check("stored episode quality matches result.quality", stored.quality == result.quality)

    # سؤال ثانٍ + دورة تدريب تجريبية كاملة عبر pipeline.core
    pipeline.answer("ما هو الإيمان؟")
    check("episode store now has 2 episodes", pipe_store.count() == 2, f"count={pipe_store.count()}")

    pipe_trainer = ExperienceTrainer(pipeline.core, pipe_store)
    pipe_cycle = pipe_trainer.run_training_cycle(top_limit=2, recent_limit=2, diverse_limit=2, save=False)
    check("pipeline-driven training cycle status == ok", pipe_cycle["status"] == "ok", f"{pipe_cycle}")
    check("NeuralCore improves (loss decreases) on at least one strategy",
          any(pipe_cycle[s]["improved"] for s in ("top", "recent", "diverse")),
          f"{pipe_cycle}")
finally:
    os.chdir(cwd)
    shutil.rmtree(pipe_dir, ignore_errors=True)
    shutil.rmtree(tmpdir, ignore_errors=True)


# ════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print(f"النتيجة الإجمالية: {len(PASS)} PASS / {len(FAIL)} FAIL")
print("=" * 70)
if FAIL:
    print("الاختبارات الفاشلة:")
    for f in FAIL:
        print("  -", f)
    sys.exit(1)
else:
    print("ALL TESTS PASSED")

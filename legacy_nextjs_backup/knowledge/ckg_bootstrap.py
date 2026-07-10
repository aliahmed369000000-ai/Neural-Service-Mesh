"""
knowledge/ckg_bootstrap.py — الأولوية 5
=========================================
يُشغَّل مرة واحدة عند بدء الخادم:
  1. يُحمِّل آيات القرآن من quran_source
  2. يُدرِّب ConceptExtractor (TF-IDF)
  3. يبني CognitiveKnowledgeGraph كاملاً
  4. يُشغِّل RelationInferencer
  5. يُحدِّث EnvironmentModel بالمفاهيم
  6. يحفظ cognitive_graph.json
  7. يُسجِّل في self_narrative

يعمل في background thread حتى لا يُعطِّل الـ API.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from main import NeuralServiceMesh

logger = logging.getLogger(__name__)

_BOOTSTRAP_DONE  = False
_BOOTSTRAP_LOCK  = threading.Lock()
_BOOTSTRAP_THREAD: Optional[threading.Thread] = None


# ═══════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════

def bootstrap_ckg(
    mesh:        "NeuralServiceMesh",
    background:  bool = True,
    force:       bool = False,
    graph_file:  Optional[Path] = None,
    max_ayahs:   Optional[int]  = None,
) -> Optional[threading.Thread]:
    """
    بناء CKG من بيانات القرآن.

    Parameters
    ----------
    mesh        : NeuralServiceMesh instance
    background  : شغّل في خيط خلفي (لا يُعطِّل الـ API)
    force       : أعد البناء حتى لو الملف موجود
    graph_file  : مسار الجراف (افتراضي: knowledge/cognitive_graph.json)
    max_ayahs   : حدّ اختياري للاختبار
    """
    global _BOOTSTRAP_DONE, _BOOTSTRAP_THREAD

    with _BOOTSTRAP_LOCK:
        if _BOOTSTRAP_DONE and not force:
            logger.info("[CKG Bootstrap] already done — skipping")
            return None

    if background:
        t = threading.Thread(
            target=_run_bootstrap,
            args=(mesh, graph_file, max_ayahs, force),
            daemon=True,
            name="CKGBootstrap",
        )
        t.start()
        _BOOTSTRAP_THREAD = t
        logger.info("[CKG Bootstrap] started in background thread")
        return t
    else:
        _run_bootstrap(mesh, graph_file, max_ayahs, force)
        return None


def is_bootstrap_done() -> bool:
    return _BOOTSTRAP_DONE


def wait_for_bootstrap(timeout: float = 120.0) -> bool:
    """انتظر حتى ينتهي البناء (أقصاه timeout ثانية)."""
    global _BOOTSTRAP_THREAD
    if _BOOTSTRAP_DONE:
        return True
    if _BOOTSTRAP_THREAD and _BOOTSTRAP_THREAD.is_alive():
        _BOOTSTRAP_THREAD.join(timeout=timeout)
    return _BOOTSTRAP_DONE


# ═══════════════════════════════════════════════════════════════════════════
# Internal bootstrap logic
# ═══════════════════════════════════════════════════════════════════════════

def _run_bootstrap(
    mesh,
    graph_file:  Optional[Path],
    max_ayahs:   Optional[int],
    force:       bool,
) -> None:
    global _BOOTSTRAP_DONE

    t0 = time.time()
    logger.info("[CKG Bootstrap] ═══ بدء بناء الجراف المعرفي ═══")

    try:
        import importlib.util, sys

        # ── Load modules safely (avoid circular imports) ──────────────────
        def _load(mod_name, path):
            if mod_name in sys.modules:
                return sys.modules[mod_name]
            spec = importlib.util.spec_from_file_location(mod_name, path)
            mod  = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
            return mod

        ce_mod  = _load('knowledge_sources.concept_extractor',
                        'knowledge_sources/concept_extractor.py')
        ckg_mod = _load('knowledge.cognitive_graph',
                        'knowledge/cognitive_graph.py')
        ri_mod  = _load('knowledge.relation_inferencer',
                        'knowledge/relation_inferencer.py')

        ConceptExtractor          = ce_mod.ConceptExtractor
        CognitiveKnowledgeGraph   = ckg_mod.CognitiveKnowledgeGraph
        RelationInferencer        = ri_mod.RelationInferencer

        # ── 1. Load Quran data ────────────────────────────────────────────
        logger.info("[CKG Bootstrap] 1/6 تحميل بيانات القرآن…")
        import json
        from pathlib import Path as P

        data_paths = [
            P("knowledge_sources/quran/data/quran.json"),
            P("knowledge_sources/quran/data/quran_sample.json"),
        ]
        raw_surahs = []
        for dp in data_paths:
            if dp.exists():
                data = json.loads(dp.read_text(encoding='utf-8'))
                s = data.get('surahs', {})
                raw_surahs = s.get('references', s) if isinstance(s, dict) else s
                logger.info(f"[CKG Bootstrap] loaded from {dp}: {len(raw_surahs)} surahs")
                break

        if not raw_surahs:
            logger.warning("[CKG Bootstrap] لا توجد بيانات قرآن — البناء متوقف")
            return

        texts, refs, snames = [], [], []
        for s in raw_surahs:
            s_num  = s.get('number', 0)
            s_name = s.get('name', f'سورة {s_num}')
            for a in s.get('ayahs', []):
                txt = a.get('text', '').strip()
                if txt:
                    texts.append(txt)
                    refs.append(f"{s_num}:{a.get('numberInSurah', 0)}")
                    snames.append(s_name)
                    if max_ayahs and len(texts) >= max_ayahs:
                        break
            if max_ayahs and len(texts) >= max_ayahs:
                break

        logger.info(f"[CKG Bootstrap] آيات: {len(texts)}")

        # ── 2. Train ConceptExtractor ─────────────────────────────────────
        logger.info("[CKG Bootstrap] 2/6 تدريب ConceptExtractor…")
        extractor = ConceptExtractor(max_concepts=12, min_score=0.12)
        if len(texts) > 1:
            extractor.fit(texts)
            vocab_size = len(extractor._vectorizer.vocabulary_) if extractor._fitted else 0
            logger.info(f"[CKG Bootstrap] TF-IDF: vocab={vocab_size}")
        else:
            logger.warning("[CKG Bootstrap] عدد الآيات قليل — TF-IDF معطّل")

        # ── 3. Extract concepts (batch) ───────────────────────────────────
        logger.info("[CKG Bootstrap] 3/6 استخراج المفاهيم…")
        all_matches = extractor.extract_batch(
            texts, references=refs, surah_names=snames
        )
        total_semantic = sum(
            len([m for m in matches if m.cluster != 'هيكل'])
            for matches in all_matches
        )
        logger.info(f"[CKG Bootstrap] مفاهيم دلالية مُستخرجة: {total_semantic}")

        # ── 4. Build CKG ──────────────────────────────────────────────────
        logger.info("[CKG Bootstrap] 4/6 بناء الجراف المعرفي…")
        gf   = graph_file or P("knowledge/cognitive_graph.json")
        ckg  = CognitiveKnowledgeGraph(graph_file=gf)

        result = ckg.ingest_batch(all_matches, refs, auto_save=False)
        logger.info(
            f"[CKG Bootstrap] ingest: "
            f"concepts={result['total_concepts']} "
            f"relations={result['total_relations']}"
        )

        # ── 5. Run RelationInferencer ─────────────────────────────────────
        logger.info("[CKG Bootstrap] 5/6 استنتاج العلاقات…")
        inferencer = RelationInferencer(ckg)
        report     = inferencer.run(verbose=False)
        logger.info(
            f"[CKG Bootstrap] inference: "
            f"affinity={report['cluster_affinity']['relations_inferred']} "
            f"chains={report['chain_inference']['chains_inferred']}"
        )

        # ── 6. Save ───────────────────────────────────────────────────────
        ckg.save()
        stats = ckg.stats()
        logger.info(
            f"[CKG Bootstrap] saved → {gf}\n"
            f"  مفاهيم    : {stats['total_concepts']}\n"
            f"  علاقات    : {stats['total_relations']}\n"
            f"  كثافة     : {stats['relation_density']:.4f}\n"
            f"  clusters  : {stats['cluster_distribution']}\n"
            f"  أقوى 5    : {[c['name'] for c in stats['top_concepts']]}"
        )

        # ── 7. Update EnvironmentModel ────────────────────────────────────
        try:
            if hasattr(mesh, 'env_model') and mesh.env_model is not None:
                mesh.env_model.update_from_ckg(ckg)
                logger.info("[CKG Bootstrap] EnvironmentModel updated ✓")
        except Exception as e:
            logger.warning(f"[CKG Bootstrap] env_model update failed: {e}")

        # ── 8. Record in SelfNarrative ────────────────────────────────────
        try:
            if hasattr(mesh, 'self_narrative') and mesh.self_narrative is not None:
                mesh.self_narrative.record_event(
                    event_type     = "world_knowledge",
                    data           = {
                        "message":   f"الجراف المعرفي جاهز: {stats['total_concepts']} مفهوم، {stats['total_relations']} علاقة",
                        "concepts":  stats['total_concepts'],
                        "relations": stats['total_relations'],
                        "clusters":  stats['cluster_distribution'],
                    },
                    surprise_score = 0.4,
                    importance     = 1.0,
                )
        except Exception as e:
            logger.warning(f"[CKG Bootstrap] narrative record failed: {e}")

        elapsed = round(time.time() - t0, 2)
        logger.info(f"[CKG Bootstrap] ═══ مكتمل في {elapsed}s ═══")

        with _BOOTSTRAP_LOCK:
            _BOOTSTRAP_DONE = True

    except Exception as exc:
        logger.error(f"[CKG Bootstrap] FAILED: {exc}", exc_info=True)

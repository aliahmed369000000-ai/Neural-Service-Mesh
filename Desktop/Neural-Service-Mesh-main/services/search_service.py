from __future__ import annotations

from typing import List, Dict, Any
import os
import time
import re
import json
from services.backend import get_trainer


def _norm(text: str) -> str:
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    return text.strip()


def _question_terms(question: str) -> List[str]:
    STOP = {
        'ما','من','كيف','لماذا','متى','أين','هل','ماذا','ما هو','ما هي',
        'ما معنى','تعريف','شرح','اشرح','عرف','هي','هو','في','إلى','على',
        'عن','مع','يعني','معنى','تعني','بماذا','وما','وهو','وهي','هو','هي',
        'ال','و','ف','ب','ك','ل',
    }
    words = re.split(r'[\s،,؟?\.]+' , question.strip())
    terms = [w.strip() for w in words if len(w.strip()) >= 2 and w.strip() not in STOP]
    norm_terms = [_norm(t) for t in terms]
    return list(dict.fromkeys(terms + norm_terms))


def _search_quran(concept_names: List[str]) -> List[Dict[str, Any]]:
    hits = []
    quran_dir = 'knowledge'
    try:
        files = sorted(f for f in os.listdir(quran_dir) if f.startswith('quran_chunk_') and f.endswith('.json'))
        for fname in files:
            if len(hits) >= 5:
                break
            fpath = os.path.join(quran_dir, fname)
            try:
                chunks = json.loads(open(fpath, encoding='utf-8').read())
                for chunk in chunks:
                    text_norm = chunk.get('text_norm', '') or _norm(chunk.get('text', ''))
                    text_orig = chunk.get('text', '')
                    for cn in concept_names[:6]:
                        cn_norm = _norm(cn)
                        if cn_norm in text_norm or cn in text_orig:
                            hits.append({
                                'surah': chunk.get('surah', ''),
                                'ayah': chunk.get('ayah', ''),
                                'ref': f"{chunk.get('surah','')}:{chunk.get('ayah','')}",
                                'text': text_orig,
                                'concept': cn,
                            })
                            break
                    if len(hits) >= 5:
                        break
            except Exception:
                continue
    except Exception:
        pass
    return hits[:5]


def ask(concept: str) -> Dict[str, Any]:
    question = concept.strip()
    if not question:
        return {"error": "question required"}

    trainer = get_trainer()
    try:
        from knowledge.cognitive_graph import get_ckg
        ckg = get_ckg()
    except Exception:
        ckg = getattr(trainer, 'ckg', None)

    t0 = time.time()
    terms = _question_terms(question)
    nterms = [_norm(t) for t in terms]

    found = []
    if ckg:
        try:
            for c in ckg.all_concepts():
                cname = c['name']
                cnorm = _norm(cname)
                score = 0.0
                if cname == question or cnorm == _norm(question):
                    score = 1.0
                elif cname in question or question in cname:
                    score = 0.9
                elif any((cname in t or t in cname) for t in terms if len(t) >= 3):
                    score = 0.72
                elif any((cnorm in nt or nt in cnorm) for nt in nterms if len(nt) >= 3):
                    score = 0.58
                if score > 0:
                    found.append({**c, '_ms': score})
            found.sort(key=lambda c: c['_ms'] * (c.get('strength', 0) + 0.01), reverse=True)
            found = found[:10]
        except Exception:
            found = []

    related = []
    all_sources = []
    if found and ckg:
        try:
            for fc in found[:3]:
                if hasattr(ckg, 'query_related'):
                    for rname, rw in ckg.query_related(fc['name'], top_k=8):
                        if rname not in {r['name'] for r in related}:
                            rc = ckg.get_concept(rname)
                            if rc:
                                related.append({'name': rname, 'cluster': rc.cluster, 'strength': rc.strength, 'relation_weight': round(rw,3)})
                for src in fc.get('sources', [])[:5]:
                    if src and src not in all_sources:
                        all_sources.append(src)
        except Exception:
            pass

    concept_names = [fc['name'] for fc in found[:5]] + terms[:3]
    quran_hits = _search_quran(concept_names)

    confidence = 0.0
    if found:
        mc = found[0]
        base_c = min(1.0, len(found) / 5.0)
        str_c = mc.get('strength', 0.0)
        src_c = min(1.0, len(all_sources) / 10.0)
        rel_c = min(1.0, len(related) / 8.0)
        confidence = round(base_c * 0.3 + str_c * 0.3 + src_c * 0.2 + rel_c * 0.2, 3)

    sources = [str(s) for s in all_sources[:8]]
    q_refs = [q['ref'] for q in quran_hits[:5]]
    cross = []

    return {
        'concept': found[0]['name'] if found else question,
        'found_in_ckg': bool(found),
        'confidence_score': confidence,
        'related_concepts': [r['name'] for r in related[:8]],
        'sources': sources,
        'cross_domain_connections': cross,
        'quran_references': q_refs,
        'elapsed_s': round(time.time() - t0, 3),
    }

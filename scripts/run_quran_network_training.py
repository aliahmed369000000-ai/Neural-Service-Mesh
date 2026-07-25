"""
scripts/run_quran_network_training.py
======================================
تشغيل تدريب فعلي حقيقي (ليس محاكاة) لـ DeepRoutingNetwork على بيانات
القرآن الحقيقية (6,236 آية) عبر إعادة استخدام منطق الإنتاج نفسه من
knowledge/quran_continuous_trainer.py — بدون إعادة اختراعه.

هذا سكربت CLI مستقل لتشغيل خطوة واحدة (أو عدة دفعات محدودة) يدوياً،
بعكس QuranContinuousTrainer الأصلي المصمَّم كـ background thread داخل
تطبيق Streamlit الحي (يحتاج كائن `mesh` حقيقي من التطبيق).

هنا نبني "mesh" بديل بسيط (SimpleNamespace) يحمل فقط الحقل الذي يقرأه
KnowledgeTrainer فعلياً (`deep_network`)، ونحمّله من الأوزان المحفوظة
فعلياً على القرص في weights/quran_model/ عبر get_default_deep_network()
— نفس دالة الإنتاج، لا شبكة مزيّفة ولا أوزان عشوائية.

الاستخدام:
    python3 scripts/run_quran_network_training.py --max-ayahs 500
    python3 scripts/run_quran_network_training.py --max-ayahs 500 --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("run_quran_network_training")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--max-ayahs", type=int, default=500,
        help="أقصى عدد آيات تُعالَج في هذا التشغيل (الأنبوب الكامل 6236 آية "
             "— نُقسّمه لدفعات يدوية بدل تشغيل واحد طويل)",
    )
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument(
        "--restart-from-zero", action="store_true",
        help="تجاهل المؤشر (cursor) الحالي وابدأ من الآية 0 حتى لو كان "
             "المرور السابق قد اكتمل بالفعل",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="حمّل البيانات والشبكة واطبع الحالة بدون أي تدريب أو حفظ فعلي",
    )
    return p


def run(args: argparse.Namespace) -> int:
    # نستورد من قلب منطق الإنتاج الفعلي — لا إعادة اختراع
    from knowledge.quran_continuous_trainer import (
        _load_all_ayahs, _load_cursor, _save_cursor, _process_batch,
    )
    from knowledge.cognitive_graph import get_ckg
    from knowledge.relation_inferencer import RelationInferencer
    from ai.knowledge_trainer import KnowledgeTrainer
    from ai.deep_routing_network import get_default_deep_network, WEIGHTS_DIR

    QURAN_WEIGHTS_DIR = "weights/quran_model"

    logger.info("── تحميل الشبكة العميقة الحقيقية من %s ──", QURAN_WEIGHTS_DIR)
    deep_network = get_default_deep_network(weights_dir=QURAN_WEIGHTS_DIR)
    logger.info(
        "✓ الشبكة محمّلة: %s | خطوات تدريب سابقة: %d | آخر loss: %s",
        deep_network.architecture_str() if hasattr(deep_network, "architecture_str") else deep_network.name,
        deep_network._train_steps,
        deep_network._last_loss,
    )

    # mesh بديل يحمل فقط ما يقرأه KnowledgeTrainer._get_layer() فعلياً
    mesh = SimpleNamespace(deep_network=deep_network)

    logger.info("── تحميل CKG الحقيقي (knowledge/cognitive_graph.json) ──")
    ckg = get_ckg()
    logger.info(
        "✓ CKG محمّل: %d مفهوم، %d علاقة (قبل هذا التشغيل)",
        ckg.concept_count(), ckg.relation_count(),
    )

    logger.info("── تحميل آيات القرآن من knowledge/quran_chunk_*.json ──")
    ayahs = _load_all_ayahs()
    logger.info("✓ %d آية محمّلة", len(ayahs))
    if not ayahs:
        logger.error("لا توجد آيات محمّلة — توقف.")
        return 1

    cursor = _load_cursor()
    start = 0 if args.restart_from_zero else cursor.get("last_ayah_processed", 0)
    if start >= len(ayahs):
        logger.info("المؤشر عند %d (تجاوز/يساوي إجمالي الآيات) — إعادة البدء من 0 لتقوية العلاقات", start)
        start = 0
    end = min(start + args.max_ayahs, len(ayahs))
    logger.info("نطاق هذا التشغيل: آيات [%d:%d] من أصل %d (batch_size=%d)",
                start, end, len(ayahs), args.batch_size)

    if args.dry_run:
        logger.info("(--dry-run) توقف هنا بدون تدريب أو حفظ فعلي.")
        return 0

    trainer = KnowledgeTrainer(mesh)

    pos = start
    total_concepts_added = 0
    total_relations_added = 0
    total_train_steps = 0
    batches_done = 0

    while pos < end:
        batch = ayahs[pos: min(pos + args.batch_size, end)]
        if not batch:
            break
        stats = _process_batch(batch, ckg, RelationInferencer, trainer, cursor)

        deep_network.save(QURAN_WEIGHTS_DIR)

        pos += len(batch)
        batches_done += 1
        total_concepts_added += stats["concepts_added"]
        total_relations_added += stats["relations_added"]
        total_train_steps += stats["train_steps_added"]

        cursor["last_ayah_processed"] = pos
        cursor["total_concepts"] = stats["total_concepts"]
        cursor["total_relations"] = stats["total_relations"]
        cursor["total_training_steps"] = cursor.get("total_training_steps", 0) + stats["train_steps_added"]
        from datetime import datetime, timezone
        cursor["last_run"] = datetime.now(timezone.utc).isoformat()
        _save_cursor(cursor)

        logger.info(
            "دفعة %d: pos=%d/%d | مفاهيم إجمالي=%d | علاقات إجمالي=%d | "
            "خطوات تدريب جديدة=%d | آخر loss=%s",
            batches_done, pos, end, stats["total_concepts"], stats["total_relations"],
            stats["train_steps_added"], deep_network._last_loss,
        )

    logger.info("════════════════════════════════════════════")
    logger.info("✓ اكتمل هذا التشغيل")
    logger.info("  دفعات مُنفَّذة        : %d", batches_done)
    logger.info("  آيات مُعالَجة         : %d (من %d إلى %d)", pos - start, start, pos)
    logger.info("  مفاهيم مضافة (هذا التشغيل) : %d", total_concepts_added)
    logger.info("  علاقات مضافة (هذا التشغيل) : %d", total_relations_added)
    logger.info("  خطوات تدريب شبكة جديدة     : %d", total_train_steps)
    logger.info("  إجمالي مفاهيم CKG الآن     : %d", ckg.concept_count())
    logger.info("  إجمالي علاقات CKG الآن     : %d", ckg.relation_count())
    logger.info("  إجمالي خطوات تدريب الشبكة (تراكمي) : %d", deep_network._train_steps)
    logger.info("  المؤشر (cursor) التالي     : %d/%d", pos, len(ayahs))
    logger.info("════════════════════════════════════════════")
    return 0


def main() -> None:
    args = build_arg_parser().parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()

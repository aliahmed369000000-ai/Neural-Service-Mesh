"""
اختبار مستقل (لا يوجد ملف اختبار مخصص لـ ai/multi_goal_planner.py أو
لتغليف ai/capability_marketplace.py في المستودع) للتحقق من الإصلاح:

قبل هذا التعديل:
  1. CapabilityMarketplace كانت تُبنى في core/mesh_bundle.py بلا
     knowledge_store، فتصبح _persist() no-op دائماً رغم أنها مكتوبة، ولا
     يوجد استدعاء لـ read_custom() المقابل — فيبدأ المتجر فارغاً بعد كل
     إعادة تشغيل.
  2. advertise_from_node() كانت تُستدعى فقط من ai/evolution_engine.py عند
     اعتماد عقدة self_evolved جديدة — الأدوار السبعة الثابتة (AGENT_CATALOGUE)
     وأدوات MCP لم يكن لها أي إعلان قدرات في المتجر أبداً.
  3. ai/multi_goal_planner.py::MultiGoalPlanner كانت مكتوبة بالكامل
     ومصمَّمة صراحة لتعمل فوق CapabilityMarketplace، لكن لا يوجد أي مكان
     في المشروع يبنيها فعلياً (بحث عن "MultiGoalPlanner(" يرجع صفر نتائج
     خارج تعريف الكلاس نفسه) — لا مسار تنفيذ حقيقي لأي هدف مركّب متعدد
     الخطوات.

هذا الاختبار يبني MeshBundle حقيقياً (بمجلد بيانات مؤقت) ويتحقق:
  1. الأدوار/الأدوات الثابتة معلَنة فعلاً في marketplace فور الإقلاع
     (بلا انتظار أي دورة تطوّر ذاتي).
  2. plan_and_execute_goal() تحلّ هدفاً مركّباً إلى عقد حقيقية وتنفّذه
     فعلياً عبر ExecutionEngine (لا محاكاة)، وتُحدِّث درجات المتجر بعد
     التنفيذ الحقيقي (record_execution).
  3. إعادة بناء MeshBundle على نفس مجلد التخزين تستعيد نفس الإعلانات
     (marketplace.restore() تعمل فعلياً، لا مجرد كتابة بلا قراءة).
"""
import json
import shutil
import tempfile

from core.mesh_bundle import MeshBundle


def run_test():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_marketplace_test_")
    try:
        bundle = MeshBundle(storage_dir=f"{tmp_dir}/data", db_path=f"{tmp_dir}/data/mesh.db")

        # 1) الأدوار/الأدوات الثابتة (الموجودة منذ أول إقلاع، ليست
        #    self_evolved) يجب أن تكون معلَنة في المتجر فوراً.
        mp_summary = bundle.marketplace.summary()
        assert mp_summary["registered_nodes"] >= bundle.registry.count(), (
            "توقّعنا أن كل عقدة مسجَّلة (أدوار + أدوات MCP) معلَنة في المتجر "
            f"فوراً — registered_nodes={mp_summary['registered_nodes']} "
            f"لكن عدد عُقد الـregistry={bundle.registry.count()}"
        )
        assert mp_summary["total_capabilities"] > 0, (
            "المتجر فارغ تماماً — الأدوار الثابتة لم تُعلَن أبداً (نفس علّة ما قبل الإصلاح)"
        )

        # 2) plan_and_execute_goal: هدف يُحلَّل إلى أكثر من عقدة (قالب
        #    process_data) وينفَّذ فعلياً — لا محاكاة.
        result = bundle.plan_and_execute_goal("process the data", data={"task": "hello"})
        plan = result.get("multi_goal_plan", {})
        assert plan.get("sub_goals"), "لم يُبنَ أي هدف فرعي إطلاقاً"
        resolved = [sg for sg in plan["sub_goals"] if sg["status"] == "resolved"]
        assert len(resolved) >= 2, (
            f"توقّعنا حلّ هدفين فرعيين على الأقل إلى عقد حقيقية، لكن حصلنا على "
            f"{len(resolved)} فقط: {json.dumps(plan['sub_goals'], ensure_ascii=False)}"
        )
        assert "steps" in result, "لا يوجد أي تنفيذ فعلي (result.steps مفقودة) — التخطيط وحده بلا تنفيذ"
        assert len(result["steps"]) >= 1, "لم يُنفَّذ أي مسار فعلياً"

        # انحسار (regression): MultiGoalPlanner.execute_plan كانت تُنشئ مرجعاً
        # دائرياً (plan.result = result_dict ثم result_dict["multi_goal_plan"]
        # = plan.to_dict() التي تتضمّن نفس result_dict عبر self.result) —
        # اكتُشف عملياً حين فشل json.dumps/FastAPI jsonable_encoder
        # بـRecursionError لا نهائي عند تسلسل النتيجة عبر /plan في api_server.py.
        # هذا الفحص يمنع رجوع نفس العلّة مستقبلاً.
        json.dumps(result, ensure_ascii=False)

        # كل خطوة نُفِّذت فعلياً يجب أن تكون حدَّثت رقم الاستدعاءات في المتجر
        # (query_count يزيد مع كل best_provider() أثناء plan())
        assert bundle.marketplace.summary()["query_count"] > 0, (
            "المتجر لم يُستعلَم منه إطلاقاً أثناء plan() — best_provider() لم تُستدعَ فعلياً"
        )

        # 3) إعادة بناء MeshBundle على نفس مجلد التخزين: يجب أن تُستعاد نفس
        #    الإعلانات (restore() تعمل فعلياً)، لا متجر فارغ من جديد.
        bundle2 = MeshBundle(storage_dir=f"{tmp_dir}/data", db_path=f"{tmp_dir}/data/mesh.db")
        mp_summary_2 = bundle2.marketplace.summary()
        assert mp_summary_2["total_capabilities"] == mp_summary["total_capabilities"], (
            "بعد إعادة التشغيل، عدد القدرات المُستعادة يختلف — restore() لا تعمل "
            f"كما هو متوقّع: قبل={mp_summary['total_capabilities']} "
            f"بعد={mp_summary_2['total_capabilities']}"
        )
        assert mp_summary_2["registered_nodes"] == mp_summary["registered_nodes"]

        print("OK: CapabilityMarketplace + MultiGoalPlanner wiring — كل الفحوص نجحت")
        print(json.dumps({
            "before_restart": mp_summary,
            "after_restart": mp_summary_2,
            "plan_status": plan.get("status"),
            "execution_status": result.get("status"),
        }, ensure_ascii=False, indent=2))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    run_test()

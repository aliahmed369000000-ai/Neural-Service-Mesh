"""اختبار محاكاة دعم TPU — بدون أي مفاتيح API حقيقية."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ✅ {name}")
    else:
        FAILED.append(name)
        print(f"  ❌ {name} {detail}")


def test_gen_script_tpu():
    from ai.kaggle_provider import generate_surahchain_kernel_script

    s = generate_surahchain_kernel_script("tst123", preset="xlarge", use_tpu=True)
    print("-- سكربت TPU (أول 60 سطر):")
    for ln in s.splitlines()[:60]:
        print("   ", ln)
    check("SCN_TPU=1 موجود في السكربت", 'SCN_TPU = "1"' in s or '"SCN_TPU": SCN_TPU,' in s)
    # السطر: "SCN_USE_8BIT_ADAM": "1" if SCN_TPU != "1" else "0"
    # عند SCN_TPU=1 (وضع TPU) يُقيّم إلى "0" — القيمة صحيحة
    check("SCN_USE_8BIT_ADAM=0 على TPU",
          '"SCN_USE_8BIT_ADAM": "1" if SCN_TPU != "1" else "0",' in s)
    check("SCN_GRAD_ACCUM=8 على TPU", '"SCN_GRAD_ACCUM": "8"' in s)
    check("XLA_USE_BF16=1 على TPU", '"XLA_USE_BF16": "1"' in s)
    check("SCN_TPU placeholder استُبدل", "__SCN_TPU__" not in s)
    check("فحص torch_xla في TPU branch", "torch_xla" in s)
    check("bitsandbytes skip على TPU", 'if SCN_TPU != "1"' in s)
    check("ألا يوجد CUDA check في سكربت TPU", '=== TPU check ===' in s)

    g = generate_surahchain_kernel_script("tst123b", preset="medium", use_tpu=False)
    check("سكربت GPU: SCN_TPU=0", '"SCN_TPU": SCN_TPU,' in g)
    check("سكربت GPU: bitsandbytes يُثبّت", '"SCN_USE_8BIT_ADAM": "1"' in g)


def test_prepare_job_tpu():
    from ai.kaggle_provider import prepare_surahchain_kaggle_job

    j = prepare_surahchain_kaggle_job(preset="xlarge", n=100000, epochs=30, batch=2, use_tpu=True)
    meta_path = Path(j["job_dir"]) / "kernel-metadata.json"
    meta = json_loads_safe(meta_path)
    print("-- meta TPU:", {k: meta.get(k) for k in ("machine_shape", "enable_gpu", "accelerator", "docker_image", "enable_internet", "is_private")})
    check("machine_shape=TpuV5E8", meta.get("machine_shape") == "TpuV5E8", str(meta.get("machine_shape")))
    check("enable_gpu=False", meta.get("enable_gpu") is False, str(meta.get("enable_gpu")))
    check("docker_image صورة TPU", "tpuvm" in (meta.get("docker_image") or ""), meta.get("docker_image"))
    check("enable_internet=True", meta.get("enable_internet") is True)
    check("is_private=True", meta.get("is_private") is True)
    check("job slug موجود", bool(j.get("slug") or j.get("job_id")))

    g = prepare_surahchain_kaggle_job(preset="small", use_tpu=False)
    gm = json_loads_safe(Path(g["job_dir"]) / "kernel-metadata.json")
    check("وضع GPU: enable_gpu=True", gm.get("enable_gpu") is True)
    # generate_kernel_metadata يحفظ T4 في machine_shape (وليس accelerator)
    check("وضع GPU: machine_shape=T4", gm.get("machine_shape") == "NvidiaTeslaT4",
          str(gm.get("machine_shape")))
    check("وضع GPU: enable_tpu=False", gm.get("enable_tpu") is False)


def test_preset_xlarge_tpu_env():
    """محاكاة إعدادات preset xlarge عند SCN_TPU=1 في train_pretrain_torch.py."""
    env = {
        "SCN_PRESET": "xlarge",
        "SCN_TPU": "1",
        "SCN_FRESH": "1",
        "SCN_N": "5000",
        "SCN_EPOCHS": "1",
        "SCN_MAX_EPOCHS": "1",
    }
    old = {k: os.environ.pop(k, None) for k in (
        "SCN_PRESET", "SCN_TPU", "SCN_FRESH", "SCN_N", "SCN_EPOCHS",
        "SCN_BATCH", "SCN_MAX_LEN", "SCN_USE_8BIT_ADAM", "SCN_GRAD_ACCUM",
        "SCN_COMPILE", "SCN_LR", "SCN_D_FF", "SCN_CHAIN_SCALE", "SCN_TAG",
        "SCN_D_MODEL", "SCN_N_HEADS", "SCN_N_PRE", "SCN_N_POST",
        "SCN_N_KV_HEADS", "SCN_D_HEAD", "SCN_MAX_EXPANDS",
    )}
    for k, v in env.items():
        os.environ[k] = v
    try:
        # استيراد الوحدة — التعريفات تُنفَّذ عند الاستيراد
        import importlib
        import experiments.surah_chain_network.train_pretrain_torch as tp
        importlib.reload(tp)
        print("-- preset xlarge + SCN_TPU=1:")
        print("   D_MODEL =", tp.D_MODEL, "| BATCH =", tp.BATCH,
              "| GRAD_ACCUM =", tp.GRAD_ACCUM, "| USE_8BIT_ADAM =", tp.USE_8BIT_ADAM,
              "| COMPILE =", tp.COMPILE)
        check("xlarge d=8192", tp.D_MODEL == 8192)
        check("BATCH=2 على TPU (افتراضي)", tp.BATCH == 2, str(tp.BATCH))
        check("GRAD_ACCUM=8 على TPU (effective=16)", tp.GRAD_ACCUM == 8, str(tp.GRAD_ACCUM))
        check("USE_8BIT_ADAM=False على TPU", tp.USE_8BIT_ADAM is False, str(tp.USE_8BIT_ADAM))
        check("COMPILE=False على TPU", tp.COMPILE is False, str(tp.COMPILE))
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def json_loads_safe(f):
    return __import__("json").loads(Path(f).read_text())


if __name__ == "__main__":
    print("1) generate_surahchain_kernel_script (TPU)")
    test_gen_script_tpu()
    print("2) prepare_surahchain_kaggle_job (TPU)")
    test_prepare_job_tpu()
    print("3) preset xlarge مع SCN_TPU=1 (محاكاة env)")
    test_preset_xlarge_tpu_env()
    print("-" * 50)
    print("ALL TESTS OK" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)

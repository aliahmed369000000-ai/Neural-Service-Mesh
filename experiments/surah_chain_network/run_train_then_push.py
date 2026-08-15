#!/usr/bin/env python3
"""
SurahChain: تدريب ثم رفع تلقائي عند النجاح
==========================================
  1) prepare_pretrain_data.py (اختياري)
  2) train_pretrain_torch.py
  3) git add checkpoints + state → commit → push

البيئة:
  GITHUB_TOKEN أو GH_TOKEN
  SCN_* كما في train_pretrain_torch
  AUTO_PUSH=1 (افتراضي) — عطّل بـ AUTO_PUSH=0 أو --skip-push

الاستخدام:
  python experiments/surah_chain_network/run_train_then_push.py
  python experiments/surah_chain_network/run_train_then_push.py --skip-prepare
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EXP = Path(__file__).resolve().parent
ROOT = EXP.parent.parent


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None, check: bool = True) -> int:
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(cwd or ROOT), env=env)
    if check and r.returncode != 0:
        raise SystemExit(r.returncode)
    return r.returncode


def _token() -> str:
    return (
        os.environ.get("GITHUB_TOKEN")
        or os.environ.get("GH_TOKEN")
        or os.environ.get("NSM_GITHUB_TOKEN")
        or ""
    ).strip()


def push_artifacts(
    repo: str = "aliahmed369000000-ai/Neural-Service-Mesh",
    branch: str = "main",
    message: str | None = None,
) -> dict:
    token = _token()
    if not token:
        return {"ok": False, "error": "لا GITHUB_TOKEN — تخطّي الرفع"}

    files = [
        EXP / "checkpoints/best_pretrain_torch.pt",
        EXP / "checkpoints/latest_pretrain_torch.pt",
        EXP / "checkpoints/pretrain_torch_state.json",
        EXP / "tokenizer_vocab_pretrain.json",
    ]
    # أي pretrain_state_*.json إضافية
    ckpt = EXP / "checkpoints"
    if ckpt.is_dir():
        files.extend(sorted(ckpt.glob("pretrain_state_*.json")))

    existing = [f for f in files if f.is_file()]
    if not existing:
        return {"ok": False, "error": "لا ملفات نتائج للرفع"}

    env = os.environ.copy()
    # لا نطبع التوكن
    run(["git", "config", "user.email", "nsm-bot@users.noreply.github.com"], cwd=ROOT, check=False)
    run(["git", "config", "user.name", "NSM Bot"], cwd=ROOT, check=False)

    url = f"https://x-access-token:{token}@github.com/{repo}.git"
    # لا نغيّر origin بشكل دائم إذا فشل — نستخدم push URL مرة واحدة
    for f in existing:
        rel = str(f.relative_to(ROOT))
        run(["git", "add", "-f", rel], cwd=ROOT, check=False)

    st = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if not (st.stdout or "").strip():
        return {"ok": True, "skipped": True, "msg": "لا تغييرات للرفع"}

    preset = os.environ.get("SCN_PRESET", "?")
    epochs = os.environ.get("SCN_EPOCHS", "?")
    n = os.environ.get("SCN_N", "?")
    msg = message or (
        f"SurahChain pretrain auto-push preset={preset} N={n} epochs={epochs} "
        f"at={datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')}"
    )
    run(["git", "commit", "-m", msg], cwd=ROOT)
    r = subprocess.run(
        ["git", "push", url, f"HEAD:{branch}"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "")[-800:]
        # أخفِ التوكن إن ظهر
        err = err.replace(token, "***")
        return {"ok": False, "error": err, "files": [str(f.name) for f in existing]}
    return {
        "ok": True,
        "pushed": True,
        "branch": branch,
        "files": [str(f.relative_to(ROOT)) for f in existing],
        "msg": msg,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="تدريب SurahChain ثم رفع تلقائي")
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-push", action="store_true")
    ap.add_argument("--repo", default=os.environ.get("SCN_REPO", "aliahmed369000000-ai/Neural-Service-Mesh"))
    ap.add_argument("--branch", default=os.environ.get("SCN_BRANCH", "main"))
    args = ap.parse_args()

    auto_push = os.environ.get("AUTO_PUSH", "1").strip().lower() not in ("0", "false", "no")
    if args.skip_push:
        auto_push = False

    env = os.environ.copy()
    # قيم افتراضية معقولة إن لم تُضبط
    env.setdefault("SCN_PRESET", "medium")
    env.setdefault("SCN_N", "60000")
    env.setdefault("SCN_EPOCHS", "30")
    env.setdefault("SCN_BATCH", "24")
    env.setdefault("SCN_FRESH", "1")
    # استئناف تلقائي من checkpoint المرفوعة على GitHub (حتى مع FRESH=1)
    env.setdefault("SCN_RESUME", "auto")
    env.setdefault("SCN_CHECKPOINT_EVERY", "2")
    env.setdefault("PYTHONUNBUFFERED", "1")

    print("=" * 60)
    print("SurahChain: تدريب → رفع تلقائي" if auto_push else "SurahChain: تدريب (بدون رفع)")
    print(
        f"preset={env.get('SCN_PRESET')} N={env.get('SCN_N')} "
        f"epochs={env.get('SCN_EPOCHS')} batch={env.get('SCN_BATCH')}"
    )
    print("=" * 60)

    if not args.skip_prepare:
        print("\n--- 1) تحضير البيانات ---")
        env = dict(env)
        env.setdefault("HF_DATASETS_NUM_PROC", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        r = subprocess.run(
            [sys.executable, "-u", str(EXP / "prepare_pretrain_data.py")],
            cwd=str(ROOT),
            env=env,
        )
        cache = EXP / "data" / "pretrain_sentences.pkl"
        need = int(env.get("SCN_N", "8000"))
        cache_ok = False
        if cache.is_file():
            try:
                import pickle
                with cache.open("rb") as f:
                    data = pickle.load(f)
                cache_ok = isinstance(data, list) and len(data) >= min(need, 1000)
                print(f"فحص الكاش: {len(data) if isinstance(data, list) else '?'} مقطع (مطلوب ≈{need}) ok={cache_ok}")
            except Exception as e:
                print("قراءة الكاش:", e)
        if r.returncode != 0 and not cache_ok:
            print("فشل التحضير — لن يُرفع")
            return r.returncode
        if r.returncode != 0 and cache_ok:
            print("⚠ رمز خروج التحضير غير صفري لكن الكاش جاهز — نتابع التدريب")

    if not args.skip_train:
        print("\n--- 2) التدريب ---")
        r = subprocess.run(
            [sys.executable, "-u", str(EXP / "train_pretrain_torch.py")],
            cwd=str(ROOT),
            env=env,
        )
        if r.returncode != 0:
            print("فشل التدريب — لن يُرفع")
            return r.returncode

    if auto_push:
        print("\n--- 3) رفع تلقائي للنتائج ---")
        # مرّر env إلى os.environ للرسالة
        for k, v in env.items():
            if k.startswith("SCN_"):
                os.environ[k] = str(v)
        result = push_artifacts(repo=args.repo, branch=args.branch)
        print(result)
        if not result.get("ok"):
            print("⚠ فشل الرفع التلقائي — التدريب نفسه نجح. حمّل المخرجات من Kaggle Output.")
            print("سبب الرفع:", result.get("error") or result)
            return 0
        print("✅ تم الرفع تلقائياً بعد انتهاء التدريب")
    else:
        print("\n(تخطي الرفع)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

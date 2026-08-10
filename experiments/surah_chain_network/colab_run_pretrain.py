#!/usr/bin/env python3
"""
تشغيل SurahChain Pre-train من طرف إلى طرف على Colab (أو أي جهاز):
  تحضير بيانات → تدريب (resume) → رفع إلى GitHub

الاستخدام في Colab بعد استنساخ المستودع:
  export GITHUB_TOKEN=ghp_xxx
  export SCN_N=30000 SCN_EPOCHS=10 SCN_D_MODEL=128 SCN_BATCH=32
  python experiments/surah_chain_network/colab_run_pretrain.py

أو دفعة واحدة بدون clone مسبق:
  python experiments/surah_chain_network/colab_run_pretrain.py --clone
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> int:
    print(">>", " ".join(cmd))
    p = subprocess.run(cmd, cwd=str(cwd or _REPO))
    if check and p.returncode != 0:
        raise SystemExit(p.returncode)
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", action="store_true", help="استنساخ المستودع إلى /content إن لزم")
    ap.add_argument("--skip-prepare", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-push", action="store_true")
    ap.add_argument("--repo", default=os.environ.get("NSM_REPO", "aliahmed369000000-ai/Neural-Service-Mesh"))
    ap.add_argument("--branch", default=os.environ.get("NSM_BRANCH", "main"))
    args = ap.parse_args()

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
    n = os.environ.get("SCN_N", "30000")
    epochs = os.environ.get("SCN_EPOCHS", "10")
    d_model = os.environ.get("SCN_D_MODEL", "128")
    batch = os.environ.get("SCN_BATCH", "32")
    lr = os.environ.get("SCN_LR", "1e-3")
    fresh = os.environ.get("SCN_FRESH", "0")

    repo_root = _REPO
    if args.clone:
        if not token:
            print("GITHUB_TOKEN مطلوب مع --clone")
            sys.exit(1)
        work = Path("/content/Neural-Service-Mesh")
        url = f"https://{token}@github.com/{args.repo}.git"
        if work.exists():
            run(["git", "remote", "set-url", "origin", url], cwd=work)
            run(["git", "pull", "origin", args.branch], cwd=work, check=False)
        else:
            run(["git", "clone", "--depth", "1", "-b", args.branch, url, str(work)])
        repo_root = work
        os.chdir(work)

    print("=" * 60)
    print(f"SurahChain Colab runner | N={n} EPOCHS={epochs} D={d_model} B={batch}")
    print("=" * 60)

    env = os.environ.copy()
    env["SCN_N"] = str(n)
    env["SCN_EPOCHS"] = str(epochs)
    env["SCN_D_MODEL"] = str(d_model)
    env["SCN_BATCH"] = str(batch)
    env["SCN_LR"] = str(lr)
    env["SCN_FRESH"] = str(fresh)

    if not args.skip_prepare:
        print("\n--- تحضير البيانات ---")
        r = subprocess.run(
            [sys.executable, str(repo_root / "experiments/surah_chain_network/prepare_pretrain_data.py")],
            cwd=str(repo_root),
            env=env,
        )
        if r.returncode != 0:
            sys.exit(r.returncode)

    if not args.skip_train:
        print("\n--- التدريب ---")
        r = subprocess.run(
            [sys.executable, str(repo_root / "experiments/surah_chain_network/train_pretrain_torch.py")],
            cwd=str(repo_root),
            env=env,
        )
        if r.returncode != 0:
            sys.exit(r.returncode)

    if not args.skip_push:
        if not token:
            print("لا GITHUB_TOKEN — تخطّي الرفع")
            return
        print("\n--- رفع النتائج ---")
        exp = repo_root / "experiments/surah_chain_network"
        files = [
            exp / "checkpoints/best_pretrain_torch.pt",
            exp / "checkpoints/latest_pretrain_torch.pt",
            exp / "checkpoints/pretrain_torch_state.json",
            exp / "tokenizer_vocab_pretrain.json",
        ]
        run(["git", "config", "user.email", "nsm-bot@users.noreply.github.com"], cwd=repo_root)
        run(["git", "config", "user.name", "NSM Bot"], cwd=repo_root)
        url = f"https://{token}@github.com/{args.repo}.git"
        run(["git", "remote", "set-url", "origin", url], cwd=repo_root)
        for f in files:
            if f.exists():
                run(["git", "add", "-f", str(f)], cwd=repo_root)
        st = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo_root), capture_output=True, text=True)
        if not st.stdout.strip():
            print("لا تغييرات للرفع")
            return
        run(
            [
                "git",
                "commit",
                "-m",
                f"Colab SurahChain pretrain N={n} epochs={epochs} d_model={d_model}",
            ],
            cwd=repo_root,
        )
        run(["git", "push", "origin", args.branch], cwd=repo_root)
        print("تم الرفع إلى GitHub")


if __name__ == "__main__":
    main()

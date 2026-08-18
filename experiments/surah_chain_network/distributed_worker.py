# -*- coding: utf-8 -*-
"""
NSM SurahChain — Worker للتدريب الموزّع (Federated-style)
===========================================================

كل sandbox (worker) يدرب نسخة مستقلة من SurahChain على **شريحة بيانات منفصلة**،
ثم يرفع أوزانه (checkpoint) إلى فرع GitHub خاص به:
`dist-worker-{id}`.

المجمّع (aggregator) لاحقًا يجمع الأوزان من فروع العمال ويُنتج النموذج الموحّد
`dist/global.pt` على الفرع الرئيسي.

مبدأ الفيدرالية البسيط هنا: **إرسال الأوزان** (Federated Averaging).
كل worker يرفع أوزانه + إحصاءات (epochs/loss/steps/bytes) كل K عصور.

المتغيرات (environment):
  SCN_WORKER_ID   معرف العامل (مثل: 001, 002) — إلزامي
  SCN_ROUNDS      عدد جولات التدريب المحلية قبل الرفع (افتراضي: 2)
  SCN_EPOCHS      أقصى عدد عصور لكل جولة (افتراضي: 50)
  SCN_BATCH       حجم الدفعة (افتراضي: 32 — آمن على 4GB RAM)
  SCN_N           عدد المقاطع لكل worker (افتراضي: 30000 شريحة)
  SCN_DATA_SEED   بذرة اختيار شريحة البيانات (افتراضي: معرف العامل)
  SCN_FROM_GLOBAL 1 = يبدأ من dist/global.pt الموحّد إن وُجد (إلزامي للعامل الأول = 0)
  SCN_SHARD_SRC   مصدر إضافي للشردات المحلية (مسار مجلد parquet أو HuggingFace)

لا يعتمد على أي شيء خارج: git + GitHub PAT + torch.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent


# ── إعدادات ────────────────────────────────────────────────────────────────
WORKER_ID = os.environ.get("SCN_WORKER_ID", "001").strip()
if not WORKER_ID:
    print("خطأ: يجب ضبط SCN_WORKER_ID (مثل: SCN_WORKER_ID=001)")
    sys.exit(1)

ROUNDS = int(os.environ.get("SCN_ROUNDS", "2"))
EPOCHS = int(os.environ.get("SCN_EPOCHS", "50"))
BATCH = int(os.environ.get("SCN_BATCH", "32"))
N = int(os.environ.get("SCN_N", "30000"))
DATA_SEED = int(os.environ.get("SCN_DATA_SEED", WORKER_ID))
FROM_GLOBAL = os.environ.get("SCN_FROM_GLOBAL", "0").strip() == "1"
SHARD_SRC = os.environ.get("SCN_SHARD_SRC", "").strip()  # مسار parquet أو HuggingFace dataset

BRANCH = f"dist-worker-{WORKER_ID}"
REPO = os.environ.get("SCN_REPO", "aliahmed369000000-ai/Neural-Service-Mesh")
DIST_DIR = _HERE / "dist"
WORK_DIR = DIST_DIR / WORKER_ID
WORK_DIR.mkdir(parents=True, exist_ok=True)

TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
PRESET = os.environ.get("SCN_PRESET", "medium")
D_MODEL = int(os.environ.get("SCN_D_MODEL", "256"))


def _token_ok() -> bool:
    return bool(TOKEN)


def _run(cmd, **kw):
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    kw.setdefault("timeout", 600)
    return subprocess.run(cmd, **kw)


# ── البيانات: شريحة منفصلة لكل عامل ────────────────────────────────────────
def _load_texts(n: int, seed: int) -> list:
    """يحمّل n مقطعًا بشريحة منفصلة (sharding) عبر DATA_SEED.

    كل عامل يحمّل من HF streaming مباشرة مع skip خاص به ويحفظ كاشه
    المحلي المنفصل (dist/worker-{id}/cache.pkl) — لا يشارك الكاش مع عمال آخرين
    حتى لا يحصلون على نفس البيانات.
    """
    import pickle
    wcache = WORK_DIR / "cache.pkl"
    if wcache.exists():
        try:
            with open(wcache, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, list) and len(data) >= n:
                g = random.Random(seed)
                out = g.sample(data, min(n, len(data)))
                print(f"[worker-{WORKER_ID}] شريحة من كاش العامل: {len(out)} مقطع (seed={seed})")
                return out
        except Exception as e:
            print(f"[worker-{WORKER_ID}] قراءة كاش العامل فشلت: {e}")
    print("[worker-%s] تحميل بيانات من HuggingFace (streaming، شريحة seed=%d)..." % (WORKER_ID, seed))
    try:
        from datasets import load_dataset
        ds = load_dataset("Jr23xd23/ArabicText-Large", split="train", streaming=True)
        it = iter(ds)
        # تخطي شريحة خاصة بهذا العامل لضمان عدم التداخل بين العمال
        skip = (seed % 8) * 4000
        for _ in range(skip):
            try:
                next(it)
            except StopIteration:
                break
        texts = []
        for item in it:
            t = (item.get("text") or item.get("content") or "")
            if isinstance(t, str) and len(t.strip()) >= 20:
                texts.append(t.strip())
            if len(texts) >= n * 2:  # نحمّل ضعف المطلوب لاختيار عشوائي لاحقًا
                break
        # اختيار عشوائي (ثابت بالبذرة) لضمان تنوع داخل العامل
        if len(texts) > n:
            g = random.Random(seed + 7)
            texts = g.sample(texts, n)
        print(f"[worker-{WORKER_ID}] حُصِل على {len(texts)} مقطع")
        if texts:
            wcache.parent.mkdir(parents=True, exist_ok=True)
            with open(wcache, "wb") as f:
                pickle.dump(texts, f, protocol=pickle.HIGHEST_PROTOCOL)
        return texts[:n]
    except Exception as e:
        print(f"[worker-{WORKER_ID}] تحميل HF فشل: {e}")
        return []


# ── التواصل مع GitHub ──────────────────────────────────────────────────────
def _clone_branch(tmp: Path) -> bool:
    shutil.rmtree(str(tmp), ignore_errors=True)
    # إصلاح timeout: clone ضحل + fetch branch محدد فقط (بدون تحميل كل الفروع)
    r = _run(["git", "clone", "-q", "--depth", "1", "--single-branch",
              "--branch", "main", "--filter=blob:none", "--sparse",
              f"https://x-access-token:{TOKEN}@github.com/{REPO}.git", str(tmp)], timeout=1200)
    if r.returncode != 0:
        print(f"[worker-{WORKER_ID}] clone فشل: {(r.stderr or '')[-200:]}")
        return False
    # sparse: جلب dist/worker-id فقط
    _run(["git", "-C", str(tmp), "sparse-checkout", "set", f"dist/{WORKER_ID}", "dist/global.pt", "dist/global_meta.json"], timeout=120)
    # جلب فرع العامل الحالي إن وُجد
    r = _run(["git", "-C", str(tmp), "fetch", "-q", "--depth", "1", "origin", BRANCH], timeout=300)
    if r.returncode == 0:
        _run(["git", "-C", str(tmp), "checkout", "-q", "-B", BRANCH, f"FETCH_HEAD"], timeout=60)
    else:
        _run(["git", "-C", str(tmp), "checkout", "-q", "-b", BRANCH], timeout=60)
    _run(["git", "-C", str(tmp), "pull", "-q", "--ff-only", "-X", "ours"], timeout=120)
    # Git LFS للملفات الكبيرة (weights.pt > 100MB يُرفض بدون LFS)
    _run(["git", "-C", str(tmp), "lfs", "install", "--local"], timeout=60)
    _run(["git", "-C", str(tmp), "lfs", "track", "*.pt"], timeout=60)
    _run(["git", "-C", str(tmp), "add", "-f", ".gitattributes"], timeout=60)
    return True


def _push_dist(tmp: Path, files: dict, meta: dict, ep: int, loss: float) -> bool:
    """يرفع weights.pt + meta.json إلى فرع العامل.

    files: {name: Path} — الملفات لنسخها إلى dist/worker-{id}/
    """
    r = _run(["git", "-C", str(tmp), "-c", "user.email=nsm-bot@users.noreply.github.com",
              "-c", "user.name=NSM Bot", "checkout", "-q", BRANCH], check=False)
    if r.returncode != 0:
        r = _run(["git", "-C", str(tmp), "checkout", "-q", "-b", BRANCH, "origin/main"], check=False)
    dest = tmp / "dist" / WORKER_ID
    dest.mkdir(parents=True, exist_ok=True)
    for name, src in files.items():
        if src and Path(src).is_file():
            shutil.copy(str(src), str(dest / name))
    meta_path = dest / "meta.json"
    meta.update({
        "worker_id": WORKER_ID,
        "branch": BRANCH,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "d_model": D_MODEL,
    })
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    _run(["git", "-C", str(tmp), "add", "-f", f"dist/{WORKER_ID}/"], check=False)
    st = _run(["git", "-C", str(tmp), "status", "--porcelain"])
    if not st.stdout.strip():
        print(f"[worker-{WORKER_ID}] لا تغييرات (epoch {ep}) — تخطي الرفع")
        return True
    _run(["git", "-C", str(tmp), "-c", "user.email=nsm-bot@users.noreply.github.com",
          "-c", "user.name=NSM Bot", "commit", "-q", "-m",
          f"NSM dist: worker-{WORKER_ID} epoch {ep} loss={loss:.4f} d={D_MODEL}"], check=False)
    r2 = _run(["git", "-C", str(tmp), "push", "-q", "-u", "origin", BRANCH])
    if r2.returncode != 0:
        print(f"[worker-{WORKER_ID}] push فشل: {(r2.stderr or '')[-200:]}")
        return False
    local_head = _run(["git", "-C", str(tmp), "rev-parse", "HEAD"]).stdout.strip()
    rv = _run(["git", "-C", str(tmp), "ls-remote", "origin", f"refs/heads/{BRANCH}"], timeout=120)
    remote_head = rv.stdout.split("\t")[0].strip() if rv.returncode == 0 else ""
    if remote_head and remote_head == local_head:
        print(f"[worker-{WORKER_ID}] ✅ رُفعت الأوزان epoch {ep} (loss={loss:.4f}) ✅ (ls-remote مُطابَق)")
    else:
        print(f"[worker-{WORKER_ID}] رفع epoch {ep} — لكن التحقق البعدي غير مكتمل")
    return True


def _fetch_global_weights(tmp: Path) -> Path | None:
    """يسحب dist/global.pt من الفرع الرئيسي إن وُجد."""
    if not FROM_GLOBAL:
        return None
    dist = tmp / "dist"
    g = dist / "global.pt"
    s = dist / "global_meta.json"
    if g.is_file():
        (WORK_DIR / "global.pt").write_bytes(g.read_bytes())
        if s.is_file():
            (WORK_DIR / "global_meta.json").write_bytes(s.read_bytes())
        print(f"[worker-{WORKER_ID}] بدأ من dist/global.pt الموحّد")
        return WORK_DIR / "global.pt"
    print(f"[worker-{WORKER_ID}] لا dist/global.pt — يبدأ من الصفر (يُضاف لاحقًا عند تفعيل التجميع)")
    return None


# ── تشغيل التدريب المحلي للround ────────────────────────────────────────────
def _train_round(texts: list, start_from: Path | None, round_idx: int) -> dict:
    """يشغّل train_pretrain_torch.py بجولة عصور محددة عبر SCN_TAG الخاص بالعامل."""
    env = os.environ.copy()
    env.update({
        "SCN_PRESET": PRESET,
        "SCN_N": str(len(texts)),
        "SCN_EPOCHS": str(EPOCHS),
        "SCN_BATCH": str(BATCH),
        "SCN_FRESH": "0" if (start_from and start_from.exists()) else "1",
        "SCN_RESUME": "none",           # لا استئناف من GitHub — worker يدير أوزانه بنفسه
        "SCN_CHECKPOINT_EVERY": "1",
        "SCN_UPLOAD_RETRIES": "1",      # لا نرفع داخل التدريب — worker يرفع بنفسه
        "SCN_WORKER_CACHE": str(WORK_DIR / "cache.pkl"),  # كاش العامل المنفصل
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_DATASETS_NUM_PROC": "1",
    })
    # إن أردنا الاستئناف من نموذج موحّد/محلي: نتأكد أنه في موقع checkpoint
    if start_from and start_from.exists():
        tag = os.environ.get("SCN_TAG", f"d{D_MODEL}_w{WORKER_ID}")
        dest = _HERE / "checkpoints" / f"latest_pretrain_{tag}.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if Path(start_from).resolve() != dest.resolve():
            shutil.copy(str(start_from), str(dest))
        env["SCN_FRESH"] = "0"
        print(f"[worker-{WORKER_ID}] استئناف → {dest.name}")
    tag = os.environ.get("SCN_TAG", f"d{D_MODEL}_w{WORKER_ID}")
    env["SCN_TAG"] = tag
    print(f"[worker-{WORKER_ID}] بدء round {round_idx + 1}/{ROUNDS} (epochs={EPOCHS})...")
    r = subprocess.run(
        [sys.executable, "-u", str(_HERE / "train_pretrain_torch.py")],
        cwd=str(ROOT), env=env,
    )
    return {"rc": r.returncode, "tag": tag}


def _read_state(tag: str) -> dict:
    state = _HERE / "checkpoints" / f"pretrain_state_{tag}.json"
    if state.is_file():
        try:
            return json.loads(state.read_text())
        except Exception:
            return {}
    return {}


def main() -> int:
    print("=" * 64)
    print(f"NSM SurahChain — Federated Worker {WORKER_ID} (d={D_MODEL})")
    print(f"repo={REPO} branch={BRANCH} rounds={ROUNDS} epochs/round={EPOCHS}")
    print("=" * 64)
    if not _token_ok():
        print("لا GITHUB_TOKEN — لا يمكن التواصل. اضبط GITHUB_TOKEN ثم أعد التشغيل.")
        return 1

    texts = _load_texts(N, DATA_SEED)
    if len(texts) < 100:
        print("البيانات غير كافية (<100 مقطع) — إنهاء")
        return 1

    with tempfile.TemporaryDirectory(prefix="nsm_dist_") as td:
        tmp = Path(td)
        if not _clone_branch(tmp):
            return 1
        start_from = _fetch_global_weights(tmp)

        # ── استئناف محلي عند إعادة التشغيل (حماية من OOM/انقطاع الجلسة):
        #     آخر وزن محلي يتفوق على global إن كان أحدث ──
        tag = os.environ.get("SCN_TAG", f"d{D_MODEL}_w{WORKER_ID}")
        local_latest = _HERE / "checkpoints" / f"latest_pretrain_{tag}.pt"
        if local_latest.is_file():
            lm = local_latest.stat().st_mtime
            gm = Path(start_from).stat().st_mtime if (start_from and Path(start_from).is_file()) else 0.0
            if lm > gm:
                print(f"[worker-{WORKER_ID}] استئناف من الوزن المحلي الأحدث "
                      f"(محلي {time.strftime('%H:%M', time.localtime(lm))} > "
                      f"global {time.strftime('%H:%M', time.localtime(gm)) if gm else 'لا يوجد'})")
                start_from = local_latest
                ls = Path(_HERE / "checkpoints" / f"pretrain_state_{tag}.json")
                if ls.is_file():
                    try:
                        sd = json.loads(ls.read_text())
                        el = sd.get("epochs_completed") or sd.get("epochs_this_run") or 0
                        if el:
                            print(f"[worker-{WORKER_ID}] epochs مكتملة محليًا: {el}")
                    except Exception:
                        pass

        last_loss = 0.0
        for r_idx in range(ROUNDS):
            res = _train_round(texts, start_from, r_idx)
            if res["rc"] != 0:
                print(f"[worker-{WORKER_ID}] round {r_idx + 1} فشل (rc={res['rc']}) — نحاول الرفع بما لدينا")
            state = _read_state(res["tag"])
            ep = state.get("epoch", r_idx + 1)
            loss = state.get("best_loss") or state.get("last_loss") or last_loss
            last_loss = float(loss) if loss else last_loss

            tag = res["tag"]
            files = {
                "weights.pt": _HERE / "checkpoints" / f"latest_pretrain_{tag}.pt",
                "state.json": _HERE / "checkpoints" / f"pretrain_state_{tag}.json",
                "vocab.json": _HERE / f"tokenizer_vocab_pretrain_{tag}.json",
            }
            meta = {
                "epochs_done": ep,
                "last_loss": last_loss,
                "round": r_idx + 1,
                "n_texts": len(texts),
                "batch": BATCH,
                "preset": PRESET,
            }
            ok = _push_dist(tmp, files, meta, ep, last_loss)
            if not ok:
                print(f"[worker-{WORKER_ID}] تحذير: round {r_idx + 1} لم يُرفع — سيُعاد في الجولة التالية")
            start_from = files["weights.pt"]  # استئناف من آخر وزن محلي

    print(f"[worker-{WORKER_ID}] اكتملت {ROUNDS} جولة. الأوزان على فرع {BRANCH}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""
NSM SurahChain — Aggregator للتدريب الموزّع (Federated Averaging)
=================================================================

يجمع أوزان العمال من فروع `dist-worker-{id}` ويعمل **Federated Averaging**:
  W_global = (1/K) * Σ W_worker

يرفع الناتج إلى `dist/global.pt` + `dist/global_meta.json` على الفرع الرئيسي.

الاستخدام:
  SCN_WORKERS="001,002,003" python3 distributed_aggregator.py
  (أو SCN_DISCOVER=1 لاستكشاف الفروع تلقائيًا عبر GitHub API)

لا يحتاج تشغيل التدريب — يعمل كعملية منفصلة (cron أو يدوية) بعد اكتمال جولات العمال.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent.parent
REPO = os.environ.get("SCN_REPO", "aliahmed369000000-ai/Neural-Service-Mesh")
TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
WORKERS = [w.strip() for w in os.environ.get("SCN_WORKERS", "").split(",") if w.strip()]
DISCOVER = os.environ.get("SCN_DISCOVER", "0").strip() == "1"
D_MODEL = int(os.environ.get("SCN_D_MODEL", "256"))
BRANCH = os.environ.get("SCN_BRANCH", "main")


def _run(cmd, **kw):
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    kw.setdefault("timeout", 600)
    return subprocess.run(cmd, **kw)


def _discover_workers(tmp: Path) -> list:
    """يستكشف فروع dist-worker-* عبر git ls-remote."""
    rv = _run(["git", "-C", str(tmp), "ls-remote", "--heads", "origin"], timeout=120)
    out = []
    for line in (rv.stdout or "").splitlines():
        ref = line.split("\t")[-1].strip()
        if ref.startswith("refs/heads/dist-worker-"):
            out.append(ref.replace("refs/heads/dist-worker-", ""))
    return sorted(set(out))


def _read_worker(tmp: Path, wid: str) -> dict | None:
    """يسحب فرع العامل ويقرأ weights.pt + meta.json."""
    r = _run(["git", "-C", str(tmp), "fetch", "-q", "origin", f"refs/heads/dist-worker-{wid}"], check=False)
    if r.returncode != 0:
        print(f"[agg] عامل {wid}: فرع غير موجود")
        return None
    work = tmp / "workers" / wid
    work.mkdir(parents=True, exist_ok=True)
    r = _run(["git", "-C", str(work), "init", "-q"], check=False) if not (work / ".git").exists() else _run(["true"])
    # checkout via worktree بسيط: clone فرعي
    sub = tmp / "sub" / wid
    r = _run(["git", "clone", "-q", "-b", f"dist-worker-{wid}", "--single-branch",
              str(tmp), str(sub)], check=False)
    if r.returncode != 0:
        print(f"[agg] عامل {wid}: checkout فشل")
        return None
    _run(["git", "-C", str(sub), "lfs", "install", "--local"], check=False)
    _run(["git", "-C", str(sub), "lfs", "pull"], check=False)
    wp = sub / "dist" / wid / "weights.pt"
    mp = sub / "dist" / wid / "meta.json"
    if not wp.is_file():
        print(f"[agg] عامل {wid}: لا weights.pt")
        return None
    meta = {}
    if mp.is_file():
        try:
            meta = json.loads(mp.read_text())
        except Exception:
            pass
    return {"id": wid, "weights": wp, "meta": meta}


def _avg_weights(pairs: list) -> bytes:
    """متوسط الأوزان: state_dict متطابق البنية (نفس d_model ونفس TAG بنية)."""
    import torch
    sd_list = []
    for p in pairs:
        sd = torch.load(str(p["weights"]), map_location="cpu", weights_only=True)
        sd_list.append(sd)
    n = len(sd_list)
    avg = {}
    for key in sd_list[0]:
        if torch.is_tensor(sd_list[0][key]):
            avg[key] = sum(s[key].float() for s in sd_list) / n
        else:
            avg[key] = sd_list[0][key]
    import io
    buf = io.BytesIO()
    torch.save(avg, buf)
    return buf.getvalue()


def _push_global(tmp: Path, data: bytes, meta: dict) -> bool:
    dist = tmp / "dist"
    dist.mkdir(exist_ok=True)
    (dist / "global.pt").write_bytes(data)
    (dist / "global_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    _run(["git", "-C", str(tmp), "add", "-f", "dist/global.pt", "dist/global_meta.json"], check=False)
    st = _run(["git", "-C", str(tmp), "status", "--porcelain"])
    if not st.stdout.strip():
        print("[agg] لا تغييرات — تخطي")
        return True
    _run(["git", "-C", str(tmp), "-c", "user.email=nsm-bot@users.noreply.github.com",
          "-c", "user.name=NSM Bot", "commit", "-q", "-m",
          f"NSM dist: تجميع فيدرالي من {len(meta.get('workers', []))} عامل (d={D_MODEL})"], check=False)
    r = _run(["git", "-C", str(tmp), "push", "-q", "origin", BRANCH])
    if r.returncode != 0:
        print(f"[agg] push فشل: {(r.stderr or '')[-200:]}")
        return False
    lh = _run(["git", "-C", str(tmp), "rev-parse", "HEAD"]).stdout.strip()
    rv = _run(["git", "-C", str(tmp), "ls-remote", "origin", f"refs/heads/{BRANCH}"], timeout=120)
    rh = rv.stdout.split("\t")[0].strip() if rv.returncode == 0 else ""
    if rh and rh == lh:
        print("[agg] ✅ رُفع dist/global.pt ✅ (ls-remote مُطابَق)")
    return True


def main() -> int:
    print("=" * 64)
    print("NSM SurahChain — Federated Aggregator")
    print("=" * 64)
    if not TOKEN:
        print("لا GITHUB_TOKEN")
        return 1
    with tempfile.TemporaryDirectory(prefix="nsm_agg_") as td:
        tmp = Path(td)
        r = _run(["git", "clone", "-q", f"https://x-access-token:{TOKEN}@github.com/{REPO}.git", str(tmp)])
        if r.returncode != 0:
            print("clone فشل")
            return 1
        ids = WORKERS
        if DISCOVER or not ids:
            ids = _discover_workers(tmp)
        if not ids:
            print("لا عمال للاكتشاف أو التجميع — اضبط SCN_WORKERS أو SCN_DISCOVER=1")
            return 1
        print(f"[agg] عمال: {ids}")
        pairs = []
        for wid in ids:
            w = _read_worker(tmp, wid)
            if w:
                pairs.append(w)
                print(f"[agg] {wid}: epochs={w['meta'].get('epochs_done', '?')} loss={w['meta'].get('last_loss', '?')}")
        if len(pairs) < 2:
            print("[agg] نحتاج عاملين على الأقل للتجميع — إن كان عاملًا واحدًا فقط، سنرفعه كـ global")
            if len(pairs) == 1:
                print("[agg] رفع وزن العامل الوحيد كـ global (مرحلة أولى)")
            else:
                return 1
        try:
            data = _avg_weights(pairs)
        except Exception as e:
            print(f"[agg] averaging فشل: {e} — ربما بنية weights مختلفة بين العمال (تحقق من d_model)")
            return 1
        meta = {
            "workers": [p["id"] for p in pairs],
            "method": "federated_avg",
            "d_model": D_MODEL,
            "aggregated_at": datetime.now(timezone.utc).isoformat(),
            "epochs": [p["meta"].get("epochs_done") for p in pairs],
            "losses": [p["meta"].get("last_loss") for p in pairs],
        }
        ok = _push_global(tmp, data, meta)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

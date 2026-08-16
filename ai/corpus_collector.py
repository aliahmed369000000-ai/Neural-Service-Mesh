from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
import numpy as np

def _hf_available():
    try:
        from datasets import load_dataset, IterableDataset  # noqa: F401
        return True
    except ImportError:
        return False

REPOS = {
    "arabic101b": {"repo": "ClusterlabAi/101_billion_arabic_words_dataset", "split": "train", "tokens_total": 0.1e12, "language": "AR"},
    "fineweb_ar": {"repo": "kaust-generative-ai/fineweb-edu-ar", "split": "train", "tokens_total": 0.2e12, "language": "AR", "gated": True},
    "arabicweb24": {"repo": "kaust-generative-ai/arabicweb24", "split": "train", "tokens_total": 0.3e12, "language": "AR", "gated": True},
    "fineweb": {"repo": "HuggingFaceFW/fineweb", "split": "train", "tokens_total": 18.5e12, "language": None},
    "redpajama": {"repo": "togethercomputer/RedPajama-Data-V2", "split": "train", "tokens_total": 30e12, "language": "EN"},
}

CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
WS_RE = re.compile(r"\s+")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_RE.sub("", text)
    if len(text) > 8000:
        text = text[:8000]
    return WS_RE.sub(" ", text).strip()

def is_arabic_heuristic(text):
    if not text:
        return False
    sample = text[:2000]
    if not sample:
        return False
    arabic = sum(1 for ch in sample if "\u0600" <= ch <= "\u06FF")
    return arabic >= len(sample) * 0.35

def load_resume(state_path):
    if state_path.exists():
        return json.loads(state_path.read_text(encoding="utf-8"))
    return {"shard_idx": 0, "global_tokens": 0.0, "bytes_on_disk": 0, "sources": {}, "done": []}

def save_resume(state_path, state):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

def _merge_resume_states(out_dir):
    """يدمج كل meta/resume_*.json في ملف واحد: meta/state_global.json (يُرفَع مع الـdataset)."""
    meta_dir = out_dir / "meta"
    merged = {"shards_done": {}, "global_tokens": 0.0, "sources": {}, "updated_at": ""}
    for f in sorted(meta_dir.glob("resume_*.json")):
        try:
            st = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        repo_name = f.name.replace("resume_", "").replace(".json", "").replace("_", "/", 1)
        merged["shards_done"][repo_name] = st.get("shard_idx", 0)
        merged["global_tokens"] += st.get("global_tokens", 0.0)
        if st.get("sources"):
            for r, v in st["sources"].items():
                merged["sources"][r] = merged["sources"].get(r, 0.0) + v
    import datetime
    merged["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    state_global = meta_dir / "state_global.json"
    state_global.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return state_global


def _flush_resume_state(out_dir, ds_ref):
    """يرفع meta/state_global.json (وأي resume_*.json موجودة) إلى الـdataset."""
    if not ds_ref:
        return
    import kaggle as _kg
    try:
        state_global = _merge_resume_states(out_dir)
        files = [f for f in (out_dir / "meta").glob("resume_*.json") if f.is_file()] + [state_global]
        if not files:
            return
        for f in files:
            try:
                _kg.api.dataset_upload_file(ds_ref, str(f))
                print(f"[state] رُفِع {f.name} إلى {ds_ref}")
            except Exception as e:
                print(f"[state] رفع {f.name} فشل (متابع): {e}", file=sys.stderr)
    except Exception as e:
        print(f"[state] flush_resume_state فشل (متابع): {e}", file=sys.stderr)


def load_remote_resume_state(out_dir, ds_refs):
    """يسحب meta/state_global.json من آخر دفعة مرفوعة ويحفظه في out_dir ليُستخدم للاستئناف.
    يُرجع قاموس {repo: shard_idx} أو فارغًا إن لم يوجد."""
    import kaggle as _kg
    fetched = {}
    for ds_ref in ds_refs:
        try:
            tmp = out_dir / "meta" / "_remote"
            tmp.mkdir(parents=True, exist_ok=True)
            _kg.api.dataset_download_files(ds_ref, path=str(tmp), unzip=True, force=True)
            sg = tmp / "state_global.json"
            if sg.is_file():
                data = json.loads(sg.read_text(encoding="utf-8"))
                for repo_name, idx in (data.get("shards_done") or {}).items():
                    fetched[repo_name] = max(fetched.get(repo_name, 0), idx)
                print(f"[resume] حالة مرفوعة من {ds_ref}: {data.get('global_tokens', 0)/1e9:.2f}B توكن — "
                      f"{', '.join(f'{r}: shard {i}' for r, i in data.get('shards_done', {}).items())}")
                break
        except Exception as e:
            print(f"[resume] فشل سحب الحالة من {ds_ref} (متابع): {e}", file=sys.stderr)
    return fetched


def write_shard(out_dir, repo, idx, records):
    import pandas as pd
    shards_dir = out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    safe = repo.replace("/", "_").replace(":", "_")
    path = shards_dir / f"{safe}_shard_{idx:05d}.parquet"
    df.to_parquet(path, index=False, compression="zstd", compression_level=9)
    return path.stat().st_size if path.exists() else 0

def _bytes_on_disk(out_dir):
    sd = out_dir / "shards"
    return sum(p.stat().st_size for p in sd.glob("*.parquet") if p.is_file()) if sd.is_dir() else 0

def compact_shards(out_dir):
    """يدمج الشردات المؤقتة في ملف زبدي نهائي في meta/final/ ثم يحذف المؤقتات (يحرر القرص)."""
    import pandas as pd
    shards = sorted(p for p in (out_dir / "shards").glob("*.parquet") if p.is_file())
    if not shards:
        return 0, 0
    meta_dir = out_dir / "meta" / "final"
    meta_dir.mkdir(parents=True, exist_ok=True)
    n_final = len(list(meta_dir.glob("*.parquet.zst")))
    final = meta_dir / f"nsm_corpus_final_{n_final:03d}.parquet.zst"
    frames = [pd.read_parquet(p) for p in shards]
    merged = pd.concat(frames, ignore_index=True)
    merged.to_parquet(final, index=False, compression="zstd", compression_level=9)
    del frames, merged
    size = final.stat().st_size
    total_bytes = sum(p.stat().st_size for p in shards)
    for p in shards:
        p.unlink()
    print(f"[compact] دُمج {len(shards)} shard ({total_bytes/1e9:.2f}GB) -> {final.name} ({size/1e9:.2f}GB)")
    return size, total_bytes

def upload_to_kaggle_dataset(files, ds_ref):
    from kaggle.api import KaggleApi
    api = KaggleApi()
    api.authenticate()
    try:
        api.dataset_initialize(ds_ref)
    except Exception:  # noqa: BLE001
        pass
    for f in files:
        print(f"[upload] رفع {f.name} ({f.stat().st_size/1e9:.2f}GB)...")
        api.dataset_upload_file(ds_ref, str(f))
        f.unlink()
        print(f"[upload] حُذف {f.name} من القرص")
    return True

def flush_to_dataset(out_dir, ds_ref, budget_bytes):
    """يفرّغ الشردات: دمج -> رفع إلى Kaggle Dataset -> حذف، عند تجاوز ربع الميزانية أو 4 شردات.
    يُرفَع أيضًا ملف meta/state_global.json (كل حالات الاستئناف مجمعة) مع كل عملية رفع
    حتى تبقى نقطة الاستئناف محفوظة داخل الـdataset حتى لو فُرّغ /kaggle/working."""
    shards = list((out_dir / "shards").glob("*.parquet")) if (out_dir / "shards").is_dir() else []
    need_flush = _bytes_on_disk(out_dir) > budget_bytes * 0.25 or len(shards) >= 4
    if not need_flush:
        return
    # رفع ملف حالة الاستئناف المجمع مع كل دفعة
    _flush_resume_state(out_dir, ds_ref)
    compact_shards(out_dir)
    meta_final = out_dir / "meta" / "final"
    files = sorted(p for p in meta_final.glob("*.parquet.zst") if p.is_file())
    if not files:
        return
    ok = False
    for attempt in range(3):
        try:
            upload_to_kaggle_dataset(files, ds_ref)
        except Exception as e:  # noqa: BLE001
            print(f"[upload] محاولة {attempt+1}/3 فشلت: {e}", file=sys.stderr)
            time.sleep(30)
        else:
            ok = True
            break
    if not ok:
        print("[upload] الرفع فشل 3 مرات — نحافظ على الملفات ونتابع بحذر")

def collect_source(repo_def, out_dir, shard_size_tokens=50_000_000, max_tokens=float("inf"),
                   language_only=None, hf_token=None, state=None, disk_budget_gb=60.0,
                   ds_ref=None, report_every=5000):
    from datasets import load_dataset
    from huggingface_hub import HfApi, login
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
    api = HfApi()
    repo = repo_def["repo"]
    if not hf_token and repo_def.get("gated"):
        print(f"[{repo}] مستودع gated — تخطّي بدون HF_TOKEN")
        return 0.0
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "meta" / f"resume_{repo.replace('/', '_')}.json"
    state = load_resume(state_path) if state is None else state
    # استئناف من الحالة المرفوعة داخل الـdataset إن لم توجد حالة محلية حديثة
    if not state_path.exists() and ds_ref and state.get("shard_idx", 0) == 0:
        try:
            remote = load_remote_resume_state(out_dir, [ds_ref])
            if remote.get(repo, 0) > 0:
                state["shard_idx"] = remote[repo]
                print(f"[{repo}] استئناف من الحالة المرفوعة — آخر shard: {remote[repo]}")
                save_resume(state_path, state)
        except Exception as e:
            print(f"[{repo}] فحص الحالة المرفوعة فشل (متابع): {e}", file=sys.stderr)
    try:
        print(f"[{repo}] info: {api.dataset_info(repo).id}")
    except Exception as e:  # noqa: BLE001
        print(f"[{repo}] info error (متابع): {e}", file=sys.stderr)
    ds = load_dataset(repo, split=repo_def["split"], streaming=True)
    shard_idx = state.get("shard_idx", 0)
    tokens_written = 0.0
    records = []
    rec_tokens = 0.0
    rows_seen = 0
    budget_bytes = disk_budget_gb * 1e9
    t0 = time.time()
    for row in ds:
        rows_seen += 1
        text = clean_text(row.get("text", ""))
        if not text or len(text) < 50:
            continue
        if language_only == "AR" and not is_arabic_heuristic(text):
            continue
        if language_only == "EN" and is_arabic_heuristic(text):
            continue
        est = len(text.split()) * 1.3
        records.append({"text": text, "source": repo, "est_tokens": est})
        rec_tokens += est
        if rec_tokens >= shard_size_tokens or len(records) >= 200_000:
            write_shard(out_dir, repo, shard_idx, records)
            tokens_written += rec_tokens
            state["shard_idx"] = shard_idx + 1
            state["bytes_on_disk"] = _bytes_on_disk(out_dir)
            state["global_tokens"] = state.get("global_tokens", 0.0) + rec_tokens
            state["sources"] = state.get("sources", {})
            state["sources"][repo] = state["sources"].get(repo, 0.0) + rec_tokens
            save_resume(state_path, state)
            records = []
            rec_tokens = 0.0
            shard_idx += 1
            print(f"[{repo}] shard {shard_idx-1} — إجمالي {state['global_tokens']/1e9:.2f}B توكن، قرص: {state['bytes_on_disk']/1e9:.2f}GB، {(time.time()-t0)/60:.0f}min")
        flush_to_dataset(out_dir, ds_ref or f"nsm-corpus-ar-{repo.replace('/', '-')}", budget_bytes)
        if rows_seen % report_every == 0:
            print(f"[{repo}] progress: {rows_seen} rows")
        if tokens_written >= max_tokens:
            break
    if records:
        write_shard(out_dir, repo, shard_idx, records)
        tokens_written += rec_tokens
        state["shard_idx"] = shard_idx + 1
        save_resume(state_path, state)
        flush_to_dataset(out_dir, ds_ref or f"nsm-corpus-ar-{repo.replace('/', '-')}", budget_bytes)
    state["done"] = state.get("done", []) + [repo]
    save_resume(state_path, state)
    print(f"[{repo}] DONE — {tokens_written/1e9:.2f}B توكن من {rows_seen} صف")
    return tokens_written

def parse_tokens(s):
    s = s.strip().upper()
    for suffix, mult in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if s.endswith(suffix):
            return float(s[:-1]) * mult
    return float(s)

def main():
    if not _hf_available():
        print("تثبيت: pip install -q datasets huggingface_hub pyarrow pandas kaggle", file=sys.stderr)
        return 1
    p = argparse.ArgumentParser(description="NSM Corpus Collector — قرص مقيد")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--sources", nargs="+", default=["arabic101b", "fineweb_ar", "arabicweb24"])
    p.add_argument("--target-tokens", default="50B")
    p.add_argument("--per-source-tokens", default=None)
    p.add_argument("--language", default=None, choices=["AR", "EN"])
    p.add_argument("--shard-size", type=int, default=50_000_000)
    p.add_argument("--out-dir", default="/kaggle/working/nsm_corpus")
    p.add_argument("--disk-budget-gb", type=float, default=60.0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    target = parse_tokens(a.target_tokens)
    print(f"== NSM Corpus Collector (b3 قرص مقيد) — هدف: {target/1e12:.2f}T توكن ==")
    if a.dry_run:
        for src in a.sources:
            if src in REPOS:
                print(f"  {src}: {REPOS[src]['repo']}")
        return 0
    out = Path(a.out_dir)
    total = 0.0
    per = parse_tokens(a.per_source_tokens) if a.per_source_tokens else target / max(len(a.sources), 1)
    for src in a.sources:
        if src not in REPOS:
            print(f"مصدر غير معروف: {src}", file=sys.stderr)
            continue
        limit = min(per, target - total)
        print(f"\n━━━ {src} — حد: {limit/1e12:.2f}T ━━━")
        written = collect_source(REPOS[src], out, shard_size_tokens=a.shard_size,
                                 max_tokens=limit, language_only=a.language,
                                 hf_token=a.token, disk_budget_gb=a.disk_budget_gb)
        total += written
        if total >= target:
            print("\nالهدف اكتمل.")
            break
    print(f"\nإجمالي: {total/1e12:.2f}T توكن في {out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

---
name: SourceValidator Hash Deduplication
description: Validator uses hash-based dedup stored in data/ks_seen_hashes.json; must clear to re-ingest.
---

## The Rule
To re-ingest already-seen items, clear `data/ks_seen_hashes.json` before calling sync.

**Why:** SourceValidator stores SHA fingerprints of every validated item. On re-sync, items with known hashes are "rejected" (not duplicated). This is correct behavior but confusing — 6200 "rejected" does NOT mean errors, it means "already ingested".

**How to apply:** `json.dump({"hashes": []}, open("data/ks_seen_hashes.json","w"))` resets dedup. The `/sources/quran/sync` endpoint could optionally accept `force=true` to clear hashes before sync.

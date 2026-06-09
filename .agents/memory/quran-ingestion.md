---
name: Quran Ingestion Batch Fix
description: SourceManager was slow/timing out for 6236-item Quran sync; fixed with batch node registration.
---

## The Rule
Always use `register_nodes_batch()` for large knowledge source ingestions, not per-item `register_node()`.

**Why:** `register_node()` calls `upsert_node_profile()` which reads+writes `node_profiles.json` for EACH item. For 6236 Quran ayahs this caused ~6236 file writes, leading to HTTP timeout (120s+) and a race condition (node_profiles.tmp rename error) when another process held the file.

**How to apply:** `SourceManager._ingest_items()` now checks for `register_nodes_batch()` first and falls back to per-item. KnowledgeStore has both methods. For any new source with >100 items, use the batch path.

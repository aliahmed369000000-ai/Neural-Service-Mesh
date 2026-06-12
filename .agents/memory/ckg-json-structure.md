---
name: CKG JSON Key Names
description: cognitive_graph.json uses "concepts"/"relations" not "nodes"/"edges".
---

## The Rule
When reading `knowledge/cognitive_graph.json`, use `d["concepts"]` (dict) and `d["relations"]` (dict), NOT `d["nodes"]` or `d["edges"]`.

**Why:** CognitiveKnowledgeGraph.save() writes these as "concepts" and "relations". Old code assumed "nodes"/"edges" (like a generic graph), returning 0 counts.

**How to apply:** Use `.get("concepts", d.get("nodes", {}))` as a safe fallback when reading the file in case schema evolves.

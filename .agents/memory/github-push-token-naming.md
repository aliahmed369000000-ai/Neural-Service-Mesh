---
name: GitHub push token env var naming
description: Handling mismatches between an app's expected git-token env var name and the platform-provided secret name.
---

An imported project's own git-sync code may hardcode a specific env var name for the
GitHub token (e.g. `GITHUB_TOKEN`), while the secret actually available/requested in the
environment has a different name (e.g. `GITHUB_PERSONAL_ACCESS_TOKEN`).

**Why:** Renaming or duplicating the secret is unnecessary risk/clutter; the simplest fix
is to make the code accept either name.

**How to apply:** When a project's push/sync helper can't find its token, check for a
naming mismatch first. Patch the helper to fall back through the alternate env var name
rather than creating a second secret with the same value. Non-sensitive companion values
the sync needs (git username, remote URL) can be set as plain shared env vars, not secrets,
since they aren't credentials.

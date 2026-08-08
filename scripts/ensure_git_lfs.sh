#!/usr/bin/env bash
set -euo pipefail
if ! command -v git-lfs >/dev/null 2>&1; then
  echo "ثبّت git-lfs أولاً (apt install git-lfs / brew install git-lfs)"
  exit 1
fi
git lfs install
echo "Git LFS جاهز."
git lfs ls-files | head -30 || true

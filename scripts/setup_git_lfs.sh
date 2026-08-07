#!/usr/bin/env bash
# إعداد Git LFS وسحب الملفات الكبيرة للجميع (CKG، أوزان، pkl…)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "=== NSM Git LFS Setup ==="

have_lfs() { command -v git-lfs >/dev/null 2>&1; }

install_lfs_linux() {
  local ver="3.5.1" tmp
  tmp="$(mktemp -d)"
  echo "تنزيل git-lfs ${ver}…"
  curl -fsSL \
    "https://github.com/git-lfs/git-lfs/releases/download/v${ver}/git-lfs-linux-amd64-v${ver}.tar.gz" \
    -o "$tmp/git-lfs.tgz"
  tar -xzf "$tmp/git-lfs.tgz" -C "$tmp"
  mkdir -p "$ROOT/.tools/bin"
  cp "$tmp"/git-lfs-*/git-lfs "$ROOT/.tools/bin/git-lfs"
  chmod +x "$ROOT/.tools/bin/git-lfs"
  export PATH="$ROOT/.tools/bin:$PATH"
  rm -rf "$tmp"
  echo "ثُبّت في .tools/bin"
}

if [[ "${1:-}" != "--pull-only" ]]; then
  if ! have_lfs; then
    case "$(uname -s)" in
      Linux) install_lfs_linux ;;
      Darwin) command -v brew >/dev/null && brew install git-lfs || { echo "ثبّت git-lfs من https://git-lfs.com"; exit 1; } ;;
      *) echo "ثبّت git-lfs من https://git-lfs.com"; exit 1 ;;
    esac
  fi
  [[ -x "$ROOT/.tools/bin/git-lfs" ]] && export PATH="$ROOT/.tools/bin:$PATH"
  git lfs install
  echo "git lfs install OK: $(git lfs version 2>/dev/null | head -1)"
fi

[[ -x "$ROOT/.tools/bin/git-lfs" ]] && export PATH="$ROOT/.tools/bin:$PATH"
have_lfs || { echo "Git LFS غير متوفر"; exit 1; }

echo "سحب كائنات LFS…"
git lfs pull

echo "=== تحقق CKG ==="
for f in knowledge/cognitive_graph.json knowledge/cognitive_graph_general_ar.json; do
  [[ -f "$f" ]] || { echo "✗ مفقود $f"; continue; }
  sz=$(wc -c < "$f" | tr -d ' ')
  head1=$(head -c 40 "$f" 2>/dev/null || true)
  if echo "$head1" | grep -qE 'git-lfs.github.com|version https://git-lfs'; then
    echo "✗ $f ما زال مؤشر LFS (${sz} بايت)"
  else
    echo "✓ $f جاهز (${sz} بايت)"
  fi
done
echo "اختياري: python3 scripts/ckg_quality_report.py --ckg-only"

#!/usr/bin/env bash
# Shallow-clone REPO at an exact COMMIT into DEST (used by the prepare_*.sh scripts).
# Usage: bash scripts/_pinned_clone.sh <https-url> <commit-sha> <dest>
set -euo pipefail
URL="$1"; SHA="$2"; DEST="$3"
mkdir -p "$DEST"
git -C "$DEST" init -q
git -C "$DEST" remote add origin "$URL" 2>/dev/null || git -C "$DEST" remote set-url origin "$URL"
git -C "$DEST" fetch -q --depth 1 origin "$SHA"
git -C "$DEST" checkout -q FETCH_HEAD
echo "[pinned] $URL @ $(git -C "$DEST" rev-parse --short=12 HEAD) -> $DEST"

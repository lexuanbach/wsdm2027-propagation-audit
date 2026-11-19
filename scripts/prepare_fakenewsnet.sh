#!/usr/bin/env bash
# FakeNewsNet PolitiFact and GossipCop (Shu et al., 2020), pinned commit.
# Only the four public CSVs (news URL, title, tweet IDs) are used; no tweets are
# hydrated. They are copied to data/raw/fakenewsnet/, where the loader expects them.
set -euo pipefail
DEST="${1:-data/raw/fakenewsnet}"
if [ -f "$DEST/politifact_fake.csv" ]; then echo "[fakenewsnet] $DEST already present"; exit 0; fi
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
bash "$(dirname "$0")/_pinned_clone.sh" https://github.com/KaiDMML/FakeNewsNet.git \
  654361e1c8d5baa751baf1dac5032df621652280 "$TMP/FakeNewsNet"
mkdir -p "$DEST"
cp "$TMP"/FakeNewsNet/dataset/{politifact,gossipcop}_{fake,real}.csv "$DEST/"
ls -l "$DEST"

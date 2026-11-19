#!/usr/bin/env bash
# ReCOVery (Zhou et al., 2020), pinned to the commit used in the paper.
# ~50 MB, no API keys. dataset/ has been unchanged upstream since 2021-09-25.
set -euo pipefail
DEST="${1:-data/raw/recovery}"
if [ -f "$DEST/dataset/recovery-news-data.csv" ]; then echo "[recovery] $DEST already present"; exit 0; fi
bash "$(dirname "$0")/_pinned_clone.sh" https://github.com/apurvamulay/ReCOVery.git \
  f0ddb61a7c02523975dca65db0f1a89e72478a85 "$DEST"

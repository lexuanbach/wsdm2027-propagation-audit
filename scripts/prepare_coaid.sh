#!/usr/bin/env bash
# CoAID (Cui and Lee, 2020), pinned to the commit used in the paper.
# ~30 MB, no API keys. The CSVs hold tweet IDs, not tweet text or user data;
# the loader recovers event times from the tweet-ID snowflake encoding.
set -euo pipefail
DEST="${1:-data/raw/coaid}"
if [ -d "$DEST/05-01-2020" ]; then echo "[coaid] $DEST already present"; exit 0; fi
bash "$(dirname "$0")/_pinned_clone.sh" https://github.com/cuilimeng/CoAID.git \
  d238224346781255e1e7e6ed8bc410c2b2e6329e "$DEST"

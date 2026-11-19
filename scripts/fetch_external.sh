#!/usr/bin/env bash
# Retweet-tree releases used by Table 3 and the graph-input controls, into external/.
#   1) UPFD PolitiFact/GossipCop (Dou et al., 2021) through PyTorch Geometric,
#      'profile' node features and the official splits (needs torch_geometric).
#   2) Twitter15/16 and Weibo propagation trees from the BiGCN release
#      (Bian et al., 2020), pinned to commit 5779f2c.
# About 1 GB in total. Nothing here is redistributed by this repository.
set -euo pipefail
EXT="${1:-external}"; mkdir -p "$EXT"
python - "$EXT" <<'PY'
import sys
from torch_geometric.datasets import UPFD
root = sys.argv[1] + "/upfd"
for name in ("politifact", "gossipcop"):
    for split in ("train", "val", "test"):
        UPFD(root, name, "profile", split)
print("UPFD ok")
PY
if [ ! -d "$EXT/BiGCN/data" ]; then
  bash "$(dirname "$0")/_pinned_clone.sh" https://github.com/TianBian95/BiGCN.git \
    5779f2cf86a4649651f2466b43c73caa6ed7463e "$EXT/BiGCN"
fi
if [ ! -f "$EXT/BiGCN/data/Weibo/weibotree.txt" ]; then
  unzip -q -o "$EXT/BiGCN/data/Weibo/weibotree.txt.zip" -d "$EXT/BiGCN/data/Weibo"
fi
echo "done: $EXT"

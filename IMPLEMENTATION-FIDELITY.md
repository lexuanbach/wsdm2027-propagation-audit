# Implementation fidelity and provenance

This is a source audit, not a matched-input official rerun. Status: **partial**.
The official BiGCN Twitter source is pinned to `5779f2cf86a4649651f2466b43c73caa6ed7463e`.
Six retrieved files match the git blob IDs in the archived repository tree.
`FIDELITY-SOURCES.json` gives upstream URLs, git blob IDs, SHA-256 digests and
local source hashes. Current source defaults do not prove historical run settings.

## BiGCN: three distinct implementations

| Component | Pinned official Twitter BiGCN | Local retweet-tree BiGCN-style | Local event-list BiGCN-style |
|---|---|---|---|
| Source | `model/Twitter/BiGCN_Twitter.py`: `TDrumorGCN`, `BUrumorGCN`, `train_GCN` | `scripts/rev6_graph_informative.py`: `BiGCNStruct`, `_train_graph_once` | `src/docless/baselines/bigcn.py`: `_BiPropGCN` |
| Input | 5,000-dimensional node text, observed tree | UPFD profiles (training-node standardization), or structural features on Twitter/Weibo | 16-dimensional constructed hypergraph node features |
| Direction | Top-down tree and separately sampled bottom-up reverse edges | Tree and its exact edge reversal | SHARE/BURST versus REPLY/COEXPOSE incidence subsets; these are not reverse trees |
| Layers | Two GCNConv layers per direction; first root input concatenated before layer 2 | Two GCNConv layers per direction; raw root concatenated before layer 2 | Linear projection plus two residual mean node-edge-node passes per direction |
| Root after layer 2 | First-convolution root embedding | Raw root features again | No corresponding tree-root enhancement |
| Readout | Mean pool each direction; concatenate; linear four-class head | Mean pool each direction; concatenate; linear corpus-class head | Mean over burst members per direction; concatenate; binary head |
| Regularization | Feature dropout uses PyTorch default 0.5; TD/BU edge-drop arguments 0.2 | Feature dropout 0.2; no edge drop in this driver | No explicit dropout in this class |
| Optimizer in inspected source | Adam, base lr 0.0005; BU convolution lr divided by 5; weight decay 0.0001 | Adam, lr search {0.001, 0.005}; weight decay 0.0001 | Standalone function uses Adam; selection driver `rev1_gnn.py` has its own grid |
| Training in inspected source | Batch 128, cap 200 epochs, patience 10 | Batch 64, patience 10; cap supplied to driver | Full-batch fixed epochs supplied by caller |
| Selection | Official five-fold train/test lists; test-based early stopping | Validation checkpoint and lr selection; family selection is `test` or `val` depending on recorded arm | Do not infer selected settings from standalone defaults; matched-grid driver selects on validation |
| Historical evidence | Earlier execution audit establishes input/protocol compatibility, not a completed official accuracy baseline | Saved `r15/graph_valselect/upfd_*.json` records seeds 0–4, cap 60 and `selection_rule=val`; per-trial chosen lr/epoch is absent | Source and aggregate outputs retained; exact per-trial configuration linkage incomplete |

Pinned upstream source: [BiGCN Twitter implementation](https://github.com/TianBian95/BiGCN/blob/5779f2cf86a4649651f2466b43c73caa6ed7463e/model/Twitter/BiGCN_Twitter.py),
[edge preprocessing](https://github.com/TianBian95/BiGCN/blob/5779f2cf86a4649651f2466b43c73caa6ed7463e/Process/dataset.py),
[fold construction](https://github.com/TianBian95/BiGCN/blob/5779f2cf86a4649651f2466b43c73caa6ed7463e/Process/rand5fold.py).

## Other local baselines

| Label / source | Verified local operations | Official comparison status |
|---|---|---|
| GCNFN-style / `src/docless/baselines/gcnfn.py` | Two residual mean incidence passes; burst-member mean; linear head. No ChebConv is instantiated. | Exact upstream commit/operator/preprocessing/optimizer/training mapping unavailable; do not claim operator identity. |
| UPFD-prop-style / `src/docless/baselines/upfd_prop.py` | Input LayerNorm, linear projection, two mean incidence passes with GELU, normalized linear head after burst-member mean. | Exact official mapping unavailable; not established as official UPFD with a preference branch zeroed. |
| EBGCN-style / `src/docless/baselines/ebgcn.py` | Per-incidence softmax weights, directional residual passes, learned scalar mixing gate, burst-member mean. | Exact official mapping unavailable; not a fidelity-certified EBGCN reproduction. |
| Tree GCN/GATv2 / `scripts/rev6_graph_informative.py` | Two library convolutions; learned root embedding concatenated with mean pool. | Library-layer comparators; not additional official misinformation implementations. |

The source docstrings now match these boundaries; computational statements
are unchanged. No scores, predictions, split assignments or numerical outputs
were altered. Missing historical sampler flags, optimizer trials and official
matched-input runs remain unresolved. See the paper's limitations and stored
output provenance before interpreting a model name as implementation identity.

Verify the shipped local source bindings with
`python scripts/verify_fidelity_sources.py`. The upstream descriptors were
verified on retrieval; upstream code is not vendored into this artifact.

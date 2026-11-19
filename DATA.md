# Data

This repository ships no corpus files, tweet text or user data. Each corpus is
downloaded from its public release at the commit used for the paper. The stored
results in `outputs/` contain only aggregate metrics, per-seed predictions keyed by
hashed story identifiers, and split manifests of hashed story identifiers.

## Sources and pinned versions

| Corpus | Regime | Source | Pinned version | Download | Local path |
| --- | --- | --- | --- | --- | --- |
| CoAID | event list | https://github.com/cuilimeng/CoAID (Cui and Lee, 2020) | commit `d238224346781255e1e7e6ed8bc410c2b2e6329e` | `bash scripts/prepare_coaid.sh` | `data/raw/coaid/` |
| ReCOVery | event list | https://github.com/apurvamulay/ReCOVery (Zhou et al., 2020) | commit `f0ddb61a7c02523975dca65db0f1a89e72478a85` | `bash scripts/prepare_recovery.sh` | `data/raw/recovery/` |
| FakeNewsNet PolitiFact, GossipCop | event list | https://github.com/KaiDMML/FakeNewsNet (Shu et al., 2020) | commit `654361e1c8d5baa751baf1dac5032df621652280`, files `dataset/{politifact,gossipcop}_{fake,real}.csv` | `bash scripts/prepare_fakenewsnet.sh` | `data/raw/fakenewsnet/` |
| UPFD PolitiFact, GossipCop | retweet tree | PyTorch Geometric `torch_geometric.datasets.UPFD` (Dou et al., 2021), `profile` features, official splits | files hashed in the dataset card (PyG 2.8 was used) | `bash scripts/fetch_external.sh` | `external/upfd/` |
| Twitter15, Twitter16, Weibo | retweet tree | BiGCN release https://github.com/TianBian95/BiGCN (Bian et al., 2020; trees from Ma et al., 2016, 2017) | commit `5779f2cf86a4649651f2466b43c73caa6ed7463e` | `bash scripts/fetch_external.sh` | `external/BiGCN/data/` |

`make data` runs all five downloads. Sizes: about 75 MB for `data/raw/` and about
5 GB unpacked for `external/`.

## Checking the inputs

`outputs/r15/dataset_card/dataset_card.json` records the size and SHA-256 of every
input file the loaders opened when the paper's results were produced (52 files under
`data/raw/`, 7 under `external/BiGCN/data/`, 20 under `external/upfd/`). After
downloading, run

```bash
python scripts/check_data.py
```

It reports, per tree, how many files match, differ or are missing. We confirmed
that the pinned commits above reproduce the recorded hashes.

## How the inputs are used

* **Event-list corpora.** The loaders read tweet IDs only. Event times are decoded
  from the tweet-ID snowflake encoding; no tweet is hydrated and no Twitter/X API is
  used. Each tweet ID is an event proxy; no stable user identity exists in these
  releases. GossipCop is subsampled to 1,500 stories per class with a fixed seed.
* **UPFD.** Graphs, 10-dimensional profile features and the official
  train/validation/test indices come from the PyG release.
* **Twitter15/16, Weibo.** Trees are parsed from `data.TD_RvNN.vol_5000.txt` and
  `weibotree.txt` (unzipped from `weibotree.txt.zip` by `fetch_external.sh`); node
  features are structural only. Counts of labeled and retained trees are in Table 1
  of the paper.

Splits are deterministic given the seed. `outputs/r15/split_manifest/` lists the
hashed story IDs of every train/validation/test fold for the event-list corpora,
and `scripts/rev9_split_manifest.py` regenerates it.

## Terms of use

Each corpus remains under its provider's terms; this repository's MIT license
does not apply to them.

* ReCOVery is distributed under CC BY-NC-SA 4.0 (its `License.md`).
* UPFD's code repository (safe-graph/GNN-FakeNews) is Apache-2.0; the data are
  derived from FakeNewsNet.
* CoAID, FakeNewsNet and the BiGCN release state no license in their repositories;
  please follow their README files and cite the original papers.
* Tweet-ID releases are subject to the X (Twitter) developer terms. We use the IDs
  only as anonymous event proxies and do not attempt re-identification.

# When Do Propagation Models Beat Cascade Summaries?

Code, stored results and reproduction instructions for

> Duy Nguyen\*, Phat T. Tran-Truong\*, Son Ha Xuan, and Xuan-Bach Le. 2027.
> **When Do Propagation Models Beat Cascade Summaries? A Released-Signal Audit of
> Text-Free Misinformation Detection.** In *Proceedings of the Twentieth ACM
> International Conference on Web Search and Data Mining (WSDM '27)*, Hong Kong.
> ACM. https://doi.org/10.1145/3862322.3863898
>
> \* Equal contribution. Corresponding author: Xuan-Bach Le.

The paper compares text-free misinformation detectors that see the same released
signals: a seven-number cascade summary with a small MLP (PropMLP), tabular models
(logistic regression, RandomForest, HistGBT), propagation GNNs (GCNFN-, BiGCN-,
UPFD-prop- and EBGCN-style, RGCN, GATv2) and a diagnostic test bed (DLF-Full:
hypergraph, next-slice pretraining, adjustment head, abstention). It covers four
event-list corpora (CoAID, ReCOVery, FakeNewsNet PolitiFact and GossipCop) and five
retweet-tree corpora (UPFD PolitiFact and GossipCop, Twitter15, Twitter16, Weibo).
The supplementary material is [`supplement.pdf`](supplement.pdf).

Everything runs on CPU. No GPU and no model checkpoints are needed.

## Contents

| Path | What it holds |
| --- | --- |
| `src/docless/` | Python package: data loaders and cascade schema, hypergraph encoder, next-slice pretraining, adjustment head, abstention, all baselines, evaluation and paired statistics |
| `scripts/` | One script per experiment; each writes JSON under `outputs/` (see the map below) |
| `outputs/` | Stored results behind every number in the paper and supplement (JSON, about 10 MB) |
| `tests/` | Unit tests (`pytest`) |
| `Makefile` | Named targets for every experiment, check and download (`make help`) |
| `DATA.md` | Where each corpus comes from, the pinned versions, SHA-256 checks and terms of use |
| `supplement.pdf` | Supplementary material of the paper |
| `requirements-lock.txt` | Exact package versions of the tested environment |
| `Dockerfile` | CPU image with the locked environment |

The folder names `outputs/r4` to `outputs/r15` and the `rev*` script prefixes are
the names the supplement cites, so they are kept unchanged.

## Installation

Python 3.10 or newer. Tested with Python 3.12 on macOS (Apple Silicon, CPU).

```bash
git clone https://github.com/lexuanbach/wsdm2027-propagation-audit.git
cd wsdm2027-propagation-audit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt   # exact tested versions
pip install -e ".[dev,graph]"          # package, pytest, PyTorch Geometric
```

`graph` (PyTorch Geometric) is needed only for the retweet-tree experiments and the
UPFD download. Docker alternative:

```bash
docker build -t propagation-audit .
docker run --rm -it propagation-audit make test verify figures
```

## Quick check (no data download, about one minute)

```bash
make test      # unit tests; loader tests skip until data/raw/ is populated
make verify    # checks headline numbers of the paper against outputs/
make figures   # rebuilds the Figure 2 plot data and the Table 7 cells into generated/
```

`make verify` asserts, among others, the PropMLP, DLF-Full, RGCN, GATv2, RandomForest
and HistGBT cells of Table 2, the paired p-values, the validation-selected UPFD
comparison of Table 3, the tree-corpus power analysis, the label-permutation and
graph-input controls, and the mitigation estimates of Figure 2. `make figures` writes
`generated/figures/data/*.dat` (the exact values plotted in Figure 2) and
`generated/tables/controls_table.tex` (the cells of Table 7).

## Data

No corpus is redistributed here. `make data` downloads each public release at the
exact commit used for the paper; `make check-data` then compares every input file
with the SHA-256 recorded when the paper's results were produced
(`outputs/r15/dataset_card/dataset_card.json`). Details, sizes and terms of use are
in [DATA.md](DATA.md).

```bash
make data-event-list   # CoAID, ReCOVery, FakeNewsNet -> data/raw/ (about 75 MB)
make data-trees        # UPFD and Twitter15/16/Weibo -> external/ (about 5 GB unpacked)
make check-data
```

## Reproduce the paper

There are two levels.

1. **From stored results (seconds).** `outputs/` holds the per-seed and aggregated
   JSON written by the runs reported in the paper. `make verify` and `make figures`
   read them. Every table cell can be read from the JSON listed below.
2. **From scratch (CPU hours).** Each `make` target below retrains the models of one
   table or analysis and overwrites its `outputs/` folder; compare the new JSON with
   the stored one (for example with `git diff`). All runs use seeds 0 to 4 unless
   the paper states otherwise. Hyperparameters are selected on validation folds;
   the retweet-tree "best graph vs. best summary" comparison of Table 3 is
   test-selected and labeled exploratory in the paper, and `make graph-valselect`
   runs its validation-selected counterpart.

Bit-identical retraining across machines is not guaranteed: PyTorch and
scikit-learn versions, thread counts and CPU type can change low-order digits.
Splits are deterministic given the seed, and `outputs/r15/split_manifest` lists the
story IDs of every split so that a rerun can be checked against ours.

### Main paper

| Paper element | Targets (retrain) | Stored results |
| --- | --- | --- |
| Figure 1 (overview) | conceptual diagram; the example strip is `make running-example` | `outputs/r12/example/example.json` |
| Table 1 (released signals, sizes, splits) | `make signal-matrix dataset-card split-manifest` | `outputs/r13/signal_matrix/`, `outputs/r15/dataset_card/`, `outputs/r15/split_manifest/` |
| Table 2 (event-list macro-F1) | `make main-table rgcn gatv2 ebgcn tabular tabular-strong rich dlf-plus-gossipcop` | GNN rows on CoAID/ReCOVery: `outputs/r7/main_table5/`; PropMLP, DLF-Full: `outputs/r9/propmlp_table/`, `outputs/r9/politifact/`, `outputs/r9/gossipcop_3k/`; RGCN: `outputs/r9/rgcn/`; GATv2: `outputs/r10/gatv2/`; EBGCN-style: `outputs/r11/ebgcn/`; RandomForest: `outputs/r9/tabular_baselines/`; HistGBT: `outputs/r9/stanford2/`; PropMLP+, DLF-Full+: `outputs/r9/rich_summary/`, `outputs/r13/gossipcop_dlf_plus.json` |
| Table 2 paired tests and multiplicity | `make revision-stats paired-rf rev-stats seed-paired gnn-parity propmlp-parity ssl-tree power` | `outputs/r9/revision_stats/`, `outputs/r9/paired_rf/`, `outputs/r10/rev_stats/`, `outputs/r11/seed_paired_gnn/`, `outputs/r10/gnn_parity/`, `outputs/r11/gnn_parity_politifact/`, `outputs/r10/propmlp_parity/`, `outputs/r10/ssl_tree/`, `outputs/r10/power/` |
| Table 3 (retweet trees) | `make graph-informative graph-valselect tree-power` | `outputs/r11/graph_informative/`, `outputs/r11/graph_informative_clustered/`, `outputs/r15/graph_valselect/`, `outputs/r15/tree_power/` |
| Figure 2 (feature shift, label-free corrections) | `make temporal-drift mitig-delta`, then `make figures` | `outputs/r10/temporal_drift/summary_v2.json`, `outputs/r10/mitig_delta/` |
| Table 4A (random vs. future-test split) | `make temporal-split temporal-histgbt` | `outputs/r9/temporal_split/`, `outputs/r11/temporal/` |
| Table 4B (cross-corpus transfer) | `make transfer` | `outputs/r11/transfer/transfer.json` |
| Table 5 (ranking and calibration at natural prevalence) | `make calibration calib-ci` | `outputs/r10/calibration/`, `outputs/r10/calib_ci/` |
| Table 6 (label-permutation control) | `make label-perm` | `outputs/r13/label_permutation/` |
| Table 7 (graph-input controls) | `make edge-rewire profile-permute`, then `make controls-table` | `outputs/r13/edge_rewire/`, `outputs/r15/profile_permute/` |
| Section 5.4 (power and equivalence) | `make power tree-power` | `outputs/r10/power/`, `outputs/r15/tree_power/` |
| Section 5.5 (early detection, leakage, observation windows) | `make early-det censoring censoring-fine size-only`; leakage: `PYTHONPATH=src python -m docless.eval.user_disjoint` | `outputs/r9/early_det_fused/`, `outputs/r8/user_disjoint/`, `outputs/r10/censoring/`, `outputs/r15/censoring_fine/`, `outputs/r15/size_only/` |
| Section 5.6 (title-text reference point) | `make text-bound` | `outputs/r11/text_bound/` |
| Section 5.1 (validation-only selection) | `PYTHONPATH=src python outputs/r14/ensemble/valselect/verify_no_leakage.py` re-derives every pick with all test fields blanked; the per-seed dumps come from `run_cheap.py`, `run_gnn.py` and `run_dlf.py` in the same folder | `outputs/r14/ensemble/valselect/` (`SELECTION_RULE.json`, `MANIFEST.json`, per-seed dumps in `preds/`, regrets in `published_leader_check.json`); folds in `outputs/r14/ensemble/inventory/fold_manifest.json` |
| Publisher-disjoint and size-stratified splits | `make source-disjoint size-split` | `outputs/r9/source_disjoint/`, `outputs/r11/size_split/` |
| Official BiGCN code, as released | `make official-bigcn` (runs the authors' unmodified code through a one-symbol shim) | `outputs/r15/official_bigcn/twitter15_asreleased.log` |

The validation-only selection in Section 5.1 used eight components on CoAID, ReCOVery and PolitiFact and four on GossipCop (140 corpus, model and seed outputs). `preds/gossipcop` also holds DLF-Full and DLF-Full+ dumps that finished later, so rerunning `published_leader_check.py` lists six GossipCop components; the GossipCop pick and its regret (0.000) do not change. The stored `published_leader_check.json` is the version the paper reports.

### Supplement

| Supplement element | Targets | Stored results |
| --- | --- | --- |
| Table S1 (cross-dataset transfer) | `make transfer` | `outputs/r11/transfer/` |
| Table S2 (importance weighting on the future-test era) | `make temporal-histgbt` | `outputs/r11/temporal/` |
| Table S3 (log vs. raw features) | `make rawlog` | `outputs/r11/rawlog/` |
| Table S4 (point-process features) | `make hawkes` | `outputs/r11/hawkes/` |
| Table S5 (AUPRC and Brier) | `make calibration` | `outputs/r10/calibration/` |
| Table S6 (size-stratified split) | `make size-split` | `outputs/r11/size_split/` |
| Table S7 (future-test macro-F1 under corrections) | `make temporal-drift` | `outputs/r10/temporal_drift/summary_v2.json` |
| Table S8 (structure vs. summary) | `make fusion-probe scaling` | `outputs/r9/fusion_probe/`, `outputs/r9/strong_config/` |
| Table S11 (label permutation, full) | `make label-perm` | `outputs/r13/label_permutation/` |
| Table S12 (degree-preserving rewire, full) | `make edge-rewire` | `outputs/r13/edge_rewire/` |
| Adjustment head, abstention, KL sweep | `make revision-stats abstention lambda-sweep` | `outputs/r9/revision_stats/`, `outputs/r9/abstention_baseline/`, `outputs/r10/lambda_sweep/`, `outputs/r6/kl_sweep/` |
| Stratum calibration, interaction ablation, resource profile | `python -m docless.eval.stratum_calibration`, `interaction_ablation`, `resource_profile` (with `PYTHONPATH=src`) | `outputs/r7/` |
| Negative-source pretraining, media ablation, stress tests | `python -m docless.eval.ssl_negsource`; `make media-ablation`; `python -m docless.eval.robustness` | `outputs/r8/ssl_negsource/`, `outputs/r9/media_ablation/`, `outputs/r4/robustness/`, `outputs/r5/robustness_recov/` |
| Rolling-origin temporal checks, feature-drop ablation | `make temporal-rigor feature-drop` | `outputs/r10/temporal_rigor/`, `outputs/r11/feature_drop/` |
| Earlier three-seed sweeps (coverage, confidence shift) | `bash scripts/run_tier25.sh` | `outputs/tier2/`, `outputs/tier25/` |

Tables S9 (inputs of the original evaluations) and S10 (notation) contain no
computed values.

### Running time

Times are from a laptop CPU (Apple Silicon) and only indicate scale. The
no-training checks take seconds. Summary and tabular targets take minutes per
corpus. GNN and DLF-Full targets take from tens of minutes to a few hours each;
`make dlf-plus-gossipcop` takes about 8 hours. Retraining every target takes on the
order of a day of CPU time. `make demo` (about 10 minutes) downloads CoAID and
ReCOVery and runs a reduced two-seed version of Table 2 into `outputs/demo/`; it
checks the pipeline but does not use the paper configuration.

## Citation

```bibtex
@inproceedings{nguyen2027propagation,
  author    = {Nguyen, Duy and Tran-Truong, Phat T. and Ha Xuan, Son and Le, Xuan-Bach},
  title     = {When Do Propagation Models Beat Cascade Summaries? {A} Released-Signal
               Audit of Text-Free Misinformation Detection},
  booktitle = {Proceedings of the Twentieth ACM International Conference on Web Search
               and Data Mining (WSDM '27)},
  year      = {2027},
  publisher = {ACM},
  address   = {Hong Kong},
  doi       = {10.1145/3862322.3863898}
}
```

## License and contact

Code and stored results: MIT License (see [LICENSE](LICENSE)). The corpora keep
their own terms (see [DATA.md](DATA.md)). Questions: please open a GitHub issue or
contact the corresponding author, Xuan-Bach Le (Ho Chi Minh City University of
Technology, VNU-HCM).

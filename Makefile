# Reproduction targets for "When Do Propagation Models Beat Cascade Summaries?"
# (WSDM 2027). Every target is CPU-only. Run `make help` for the list.
#
# Two kinds of target:
#   * check / rebuild targets read the stored results in outputs/ (seconds);
#   * experiment targets retrain models and overwrite their outputs/ folder
#     (minutes to hours; they need the public data, see DATA.md).

.PHONY: help install dev test verify figures check-data \
        data data-event-list data-trees \
        main-table rgcn gatv2 ebgcn gnn-parity seed-paired tabular tabular-strong rich \
        dlf-plus-gossipcop revision-stats paired-rf rev-stats propmlp-parity ssl-tree \
        signal-matrix dataset-card split-manifest running-example \
        graph-informative graph-valselect tree-power \
        temporal-split temporal-histgbt temporal-rigor temporal-drift mitig-delta \
        censoring censoring-fine transfer source-disjoint size-split \
        calibration calib-ci power label-perm edge-rewire profile-permute controls-table \
        size-only feature-drop rawlog hawkes text-bound fusion-probe scaling early-det \
        media-ablation abstention lambda-sweep official-bigcn demo clean

PY  ?= python
RUN  = PYTHONPATH=src $(PY)

help:
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | sort

# ---------------------------------------------------------------- setup
install:            ## install the package (pyproject.toml)
	$(PY) -m pip install -e .

dev:                ## install the package plus test tools
	$(PY) -m pip install -e ".[dev]"

test:               ## unit tests (data-dependent tests skip if data/raw is empty)
	$(RUN) -m pytest -q

# ---------------------------------------------------------------- no-training checks
verify:             ## check headline paper numbers against stored outputs (seconds)
	$(RUN) scripts/verify_paper_claims.py

figures:            ## rebuild Figure 2 plot data and the Table 7 cells into generated/
	$(RUN) scripts/rev7_make_paper_figures.py
	$(RUN) scripts/rev9_make_controls_table.py

check-data:         ## compare downloaded inputs with the SHA-256 values used in the paper
	$(RUN) scripts/check_data.py

# ---------------------------------------------------------------- data (see DATA.md)
data: data-event-list data-trees  ## download every public input at the pinned versions

data-event-list:    ## CoAID, ReCOVery, FakeNewsNet into data/raw/
	bash scripts/prepare_coaid.sh
	bash scripts/prepare_recovery.sh
	bash scripts/prepare_fakenewsnet.sh

data-trees:         ## UPFD (via PyG) and BiGCN Twitter15/16/Weibo into external/
	bash scripts/fetch_external.sh

# ---------------------------------------------------------------- Table 1
signal-matrix:      ## Table 1 signal matrix -> outputs/r13/signal_matrix
	$(RUN) scripts/rev8_signal_matrix.py
dataset-card:       ## input hashes and counts -> outputs/r15/dataset_card
	$(RUN) scripts/rev9_dataset_card.py
split-manifest:     ## story IDs of every split -> outputs/r15/split_manifest
	$(RUN) scripts/rev9_split_manifest.py
running-example:    ## Section 2 running example -> outputs/r12/example
	$(RUN) scripts/rev7_running_example.py

# ---------------------------------------------------------------- Table 2 (event lists)
main-table:         ## majority/GCNFN/BiGCN/UPFD-prop/PropMLP/DLF-Full on all four corpora
	$(RUN) -m docless.eval.main_table --datasets coaid recovery \
		--methods majority gcnfn bigcn rgcn propmlp upfd_prop docless \
		--seeds 0 1 2 3 4 --hidden-dim 64 --out outputs/r9/propmlp_table
	$(RUN) scripts/politifact_eval.py
	$(RUN) scripts/gossipcop_eval.py
rgcn:               ## RGCN rows -> outputs/r9/rgcn
	$(RUN) scripts/rgcn_baseline.py
gatv2:              ## GATv2 rows + paired test -> outputs/r10/gatv2
	$(RUN) scripts/gatv2_baseline.py
ebgcn:              ## EBGCN-style rows -> outputs/r11/ebgcn
	$(RUN) scripts/rev6_ebgcn_all.py recovery politifact gossipcop
gnn-parity:         ## matched GNN tuning -> outputs/r10/gnn_parity, outputs/r11/gnn_parity_politifact
	$(RUN) scripts/rev1_gnn.py coaid recovery
	GNN_PARITY_OUT=outputs/r11/gnn_parity_politifact $(RUN) scripts/rev1_gnn.py politifact
seed-paired:        ## seed-paired GCNFN/BiGCN tests -> outputs/r11/seed_paired_gnn
	$(RUN) scripts/seed_paired_gnn_tests.py
tabular:            ## LogReg/RandomForest/GBT + importances -> outputs/r9/tabular_baselines
	$(RUN) scripts/tabular_baselines.py
tabular-strong:     ## tuned HistGBT + early-window features -> outputs/r9/stanford2
	$(RUN) scripts/stanford2_baselines.py
rich:               ## PropMLP+ / DLF-Full+ -> outputs/r9/rich_summary
	$(RUN) scripts/rich_summary_eval.py
	$(RUN) scripts/rev9_rich_summary_rest.py
dlf-plus-gossipcop: ## DLF-Full+ on GossipCop, five seeds (~8 h) -> outputs/r13
	for s in 0 1 2 3 4; do $(RUN) scripts/rev9_gossipcop_dlf_plus_seed.py --seed $$s; done
	$(RUN) scripts/rev9_gossipcop_aggregate.py
revision-stats:     ## DLF-Full vs PropMLP paired bootstrap + permuted strata -> outputs/r9/revision_stats
	$(RUN) scripts/revision_stats.py
paired-rf:          ## RandomForest vs PropMLP paired bootstrap -> outputs/r9/paired_rf
	$(RUN) scripts/paired_rf_vs_propmlp.py
rev-stats:          ## Holm/BH and seed-level t-tests -> outputs/r10/rev_stats
	$(RUN) scripts/rev1_stats.py
propmlp-parity:     ## PropMLP tuning parity -> outputs/r10/propmlp_parity
	$(RUN) scripts/rev3_propmlp_parity.py
ssl-tree:           ## neural vs boosted-tree paired test -> outputs/r10/ssl_tree
	$(RUN) scripts/rev2_ssl_tree.py

# ---------------------------------------------------------------- Table 3 (retweet trees)
graph-informative:  ## test-selected tree comparison -> outputs/r11/graph_informative
	$(RUN) scripts/rev6_graph_informative.py \
		--datasets upfd_politifact upfd_gossipcop twitter15 twitter16 weibo \
		--seeds 0 1 2 3 4 --epochs 60
graph-valselect:    ## validation-selected arm -> outputs/r15/graph_valselect
	$(RUN) scripts/rev6_graph_informative.py --datasets upfd_politifact upfd_gossipcop \
		--seeds 0 1 2 3 4 --epochs 60 --selection val --outdir outputs/r15/graph_valselect
tree-power:         ## story-clustered intervals, MDE, TOST -> outputs/r15/tree_power
	$(RUN) scripts/rev9_tree_power.py

# ---------------------------------------------------------------- Figure 2 and Table 4
temporal-split:     ## future-test split -> outputs/r9/temporal_split
	$(RUN) scripts/temporal_split_eval.py
	$(RUN) scripts/dlf_temporal_gossipcop.py
temporal-histgbt:   ## future-test trees + importance weighting -> outputs/r11/temporal
	$(RUN) scripts/rev6_temporal_histgbt.py
temporal-rigor:     ## rolling-origin splits + prior shift -> outputs/r10/temporal_rigor
	$(RUN) scripts/rev1_temporal.py
temporal-drift:     ## feature drift + label-free corrections -> outputs/r10/temporal_drift
	$(RUN) scripts/temporal_drift_v2.py
mitig-delta:        ## paired correction deltas (Figure 2b,c) -> outputs/r10/mitig_delta
	$(RUN) scripts/rev2_mitig.py
censoring:          ## fixed observation windows (1, 2, 7 days) -> outputs/r10/censoring
	$(RUN) scripts/rev5_censoring.py
censoring-fine:     ## 1 h, 6 h, 3 d windows -> outputs/r15/censoring_fine
	CENSORING_FINE=1 CENSORING_OUT=outputs/r15/censoring_fine $(RUN) scripts/rev5_censoring.py
transfer:           ## cross-corpus transfer (Table 4B, Table S1) -> outputs/r11/transfer
	$(RUN) scripts/rev6_transfer.py
source-disjoint:    ## publisher-disjoint split -> outputs/r9/source_disjoint
	$(RUN) scripts/source_disjoint_eval.py
size-split:         ## cascade-size-stratified split (Table S6) -> outputs/r11/size_split
	$(RUN) scripts/size_stratified_split.py

# ---------------------------------------------------------------- Tables 5-7
calibration:        ## natural-prevalence ROC/PR/Brier/ECE (Table 5, Table S5) -> outputs/r10/calibration
	$(RUN) scripts/rev1_calib.py
calib-ci:           ## ECE bootstrap intervals -> outputs/r10/calib_ci
	$(RUN) scripts/rev2_calib_ci.py
power:              ## event-list MDE and TOST -> outputs/r10/power
	$(RUN) scripts/rev2_power.py
label-perm:         ## label-permutation control (Table 6, Table S11) -> outputs/r13/label_permutation
	$(RUN) scripts/rev8_label_permutation.py
edge-rewire:        ## degree-preserving rewire (Table 7, Table S12); needs PyTorch Geometric
	$(RUN) scripts/rev8_edge_rewire.py --datasets upfd_gossipcop upfd_politifact twitter15 twitter16 \
		--seeds 0 1 2 3 4 --graph-models bigcn
	$(RUN) scripts/rev8_edge_rewire_analysis.py
profile-permute:    ## UPFD profile permutation (Table 7) -> outputs/r15/profile_permute
	$(RUN) scripts/rev9_profile_permute.py --variant within --datasets upfd_gossipcop upfd_politifact
	$(RUN) scripts/rev9_profile_permute.py --variant global --datasets upfd_gossipcop upfd_politifact
	$(RUN) scripts/rev9_permute_analysis.py
controls-table:     ## Table 7 cells from stored outputs -> generated/tables/controls_table.tex
	$(RUN) scripts/rev9_make_controls_table.py

# ---------------------------------------------------------------- other cited analyses
size-only:          ## one-feature cascade-size baseline -> outputs/r15/size_only
	$(RUN) scripts/rev9_size_only.py
feature-drop:       ## leave-one-feature-out -> outputs/r11/feature_drop
	$(RUN) scripts/feature_drop_ablation.py
rawlog:             ## raw vs log features (Table S3) -> outputs/r11/rawlog
	$(RUN) scripts/rev6_rawlog.py
hawkes:             ## point-process features (Table S4) -> outputs/r11/hawkes
	$(RUN) scripts/rev6_hawkes.py
text-bound:         ## title-text reference point -> outputs/r11/text_bound
	$(RUN) scripts/rev6_text_bound.py
fusion-probe:       ## structure vs summary ablation (Table S8) -> outputs/r9/fusion_probe
	$(RUN) scripts/strong_config_probe.py
scaling:            ## hidden-128 structure vs summary -> outputs/r9/strong_config
	$(RUN) scripts/scaling_extend.py
early-det:          ## truncated cascades -> outputs/r9/early_det_fused
	$(RUN) -m docless.eval.early_detection --datasets coaid recovery \
		--seeds 0 1 2 3 4 --out outputs/r9/early_det_fused
media-ablation:     ## media-node ablation -> outputs/r9/media_ablation
	$(RUN) scripts/media_ablation.py
abstention:         ## TV vs confidence abstention -> outputs/r9/abstention_baseline
	$(RUN) scripts/abstention_baseline.py
lambda-sweep:       ## adjustment-head KL sweep -> outputs/r10/lambda_sweep
	$(RUN) scripts/rev3_lambda_sweep.py
official-bigcn:     ## run the authors' BiGCN code unmodified on Twitter15 (needs external/BiGCN)
	$(RUN) scripts/official_bigcn_shim.py Twitter15 1

# ---------------------------------------------------------------- quick demo
demo:               ## ~10 min: download CoAID+ReCOVery, reduced 2-seed run -> outputs/demo
	bash scripts/prepare_coaid.sh
	bash scripts/prepare_recovery.sh
	$(RUN) -m docless.eval.main_table --datasets coaid recovery \
		--methods majority gcnfn bigcn rgcn propmlp docless --seeds 0 1 \
		--hidden-dim 32 --epochs-short 10 --epochs-docless-pre 2 --epochs-docless-ft 8 \
		--out outputs/demo/main_table
	@echo "Reduced demo written to outputs/demo/main_table (not the paper configuration)."

clean:              ## remove caches and generated/ (keeps stored outputs)
	rm -rf build dist *.egg-info .pytest_cache generated outputs/demo
	find . -name __pycache__ -prune -exec rm -rf {} +

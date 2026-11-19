"""Golden-value regression test (paper §VI reproducibility commitment).

This test is the load-bearing check that everything assembles deterministically
on CPU: it runs Algorithm 1 on a fixed synthetic dataset + fixed seed and
checks macro-F1 and coverage against a broad synthetic smoke-test tolerance.
This is not numerical reproduction of the empirical paper tables.

If you change the random number stream (synthetic loader, hypergraph builder,
mask sampler, anywhere), you must rerun
    PYTHONPATH=src python3 -m pytest tests/test_golden_values.py -s
to capture the new golden value, then update GOLDEN below.
"""
from __future__ import annotations

from docless.repro import set_deterministic
from docless.train import DataConfig, ModelConfig, TrainConfig, train_one_seed


# Captured by running this same code path on macOS Darwin / torch 2.8.0 / CPU.
# Tolerance is only a smoke-test diagnostic.
GOLDEN_F1 = 0.451
GOLDEN_COV = 0.938
TOL_F1 = 0.10    # synthetic smoke test only; not a paper reproduction guarantee
TOL_COV = 0.10


def test_golden_values_stable():
    set_deterministic(seed=0)
    data_cfg = DataConfig(n_stories=300)
    model_cfg = ModelConfig(hidden_dim=32, num_layers=1, num_heads=4)
    train_cfg = TrainConfig(
        pretrain_epochs=1, finetune_epochs=5,
        lambda_kl_max=0.05, coverage_budget=0.9,
    )
    m = train_one_seed(0, data_cfg, model_cfg, train_cfg, verbose=False)
    print(f"observed: f1={m['f1_all']:.4f} cov={m['coverage']:.4f}")
    assert abs(m["f1_all"] - GOLDEN_F1) <= TOL_F1, m
    assert abs(m["coverage"] - GOLDEN_COV) <= TOL_COV, m

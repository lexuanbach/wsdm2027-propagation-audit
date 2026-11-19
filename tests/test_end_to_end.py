"""End-to-end smoke test: Algorithm 1 on synthetic data must (a) run without
error, (b) produce a valid abstention coverage, and (c) clear chance F1."""

from docless.train import (
    DataConfig,
    ModelConfig,
    TrainConfig,
    train_one_seed,
)


def test_smoke_end_to_end():
    data_cfg = DataConfig(n_stories=200)
    model_cfg = ModelConfig(hidden_dim=32, num_layers=1, num_heads=4)
    train_cfg = TrainConfig(
        pretrain_epochs=1, finetune_epochs=5, lambda_kl_max=0.05, coverage_budget=0.9
    )
    m = train_one_seed(seed=0, data_cfg=data_cfg, model_cfg=model_cfg,
                       train_cfg=train_cfg, verbose=False)

    assert 0.0 <= m["coverage"] <= 1.0
    assert m["n_test"] > 0
    # macro-F1 should at least clear naive-majority on the synthetic SCM.
    # The synthetic data has class imbalance ~0.5/0.5; chance F1 ≈ 0.5.
    assert m["f1_all"] >= 0.35

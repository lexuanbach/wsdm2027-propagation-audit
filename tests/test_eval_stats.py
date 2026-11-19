import numpy as np

from docless.eval import (
    fpr_fnr_eo_gap,
    macro_f1,
    paired_bootstrap,
    run_main_table,
)


def test_paired_bootstrap_detects_true_difference():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=0.10, scale=0.05, size=400)  # method A loss
    b = rng.normal(loc=0.20, scale=0.05, size=400)  # method B loss (higher = worse)
    res = paired_bootstrap(a, b, n_resamples=2000, seed=1)
    # B - A is positive, so A wins.
    assert res.mean_diff > 0.05
    assert res.p_value < 0.01


def test_paired_bootstrap_no_diff_when_equal():
    rng = np.random.default_rng(2)
    a = rng.normal(0.15, 0.05, size=300)
    b = a.copy()
    res = paired_bootstrap(a, b, n_resamples=1000, seed=2)
    assert abs(res.mean_diff) < 1e-9
    assert res.p_value > 0.5


def test_macro_f1_known_value():
    # Class 0: TP=5, FP=1, FN=0 -> F1=10/11
    # Class 1: TP=4, FP=0, FN=1 -> F1=8/9
    # Macro   = (10/11 + 8/9) / 2
    preds  = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 0])
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    expected = 0.5 * (10 / 11 + 8 / 9)
    assert abs(macro_f1(preds, labels) - expected) < 1e-9


def test_fpr_fnr_eo_gap_two_groups():
    preds  = np.array([1, 0, 0, 1, 1, 0, 0, 1])
    labels = np.array([0, 0, 0, 1, 1, 0, 1, 1])
    groups = np.array(["en", "en", "en", "en", "ar", "ar", "ar", "ar"])
    out = fpr_fnr_eo_gap(preds, labels, groups)
    # en (4): neg=[0,1,2] preds [1,0,0] -> FPR=1/3; pos=[3] preds [1] -> FNR=0
    # ar (4): neg=[5]     preds [0]     -> FPR=0;   pos=[4,6,7] preds [1,0,1] -> FNR=1/3
    assert abs(out["per_group"]["en"]["fpr"] - 1 / 3) < 1e-9
    assert abs(out["per_group"]["en"]["fnr"] - 0.0) < 1e-9
    assert abs(out["per_group"]["ar"]["fpr"] - 0.0) < 1e-9
    assert abs(out["per_group"]["ar"]["fnr"] - 1 / 3) < 1e-9
    # EO-gap = max(1/3) - min(0)
    assert abs(out["eo_gap"] - 1 / 3) < 1e-9


def test_runner_writes_json(tmp_path):
    def fake_train(seed: int) -> dict:
        return {"f1_all": 0.8 + 0.01 * seed, "acc_all": 0.85, "coverage": 0.9}

    res = run_main_table(
        method="DocLessFake",
        dataset="synthetic",
        regime="without_text",
        seeds=[0, 1, 2],
        train_fn=fake_train,
        out_dir=tmp_path,
    )
    assert abs(res.f1_mean() - 0.81) < 1e-6
    path = tmp_path / "DocLessFake__synthetic__without_text.json"
    assert path.exists()

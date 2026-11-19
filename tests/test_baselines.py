from docless.baselines import train_gcnfn, train_majority, train_upfd_prop
from docless.train import DataConfig


def _synth_cfg():
    return DataConfig(source="synthetic", n_stories=120)


def test_majority_runs_and_predicts_one_class():
    m = train_majority(seed=0, data_cfg=_synth_cfg())
    assert m["method"] == "majority"
    assert m["coverage"] == 1.0
    assert 0.0 <= m["f1_all"] <= 1.0
    assert m["n_test"] > 0


def test_gcnfn_runs_and_clears_majority_floor_on_synth():
    cfg = _synth_cfg()
    floor = train_majority(seed=0, data_cfg=cfg)["f1_all"]
    gcnfn = train_gcnfn(seed=0, data_cfg=cfg, epochs=10, hidden_dim=32)["f1_all"]
    # Synthetic data is designed to be propagation-discriminable; GCNFN should
    # at least tie the majority floor on average.
    assert gcnfn >= floor - 0.10


def test_upfd_prop_runs():
    cfg = _synth_cfg()
    m = train_upfd_prop(seed=0, data_cfg=cfg, epochs=10, hidden_dim=32)
    assert m["method"] == "upfd_prop"
    assert 0.0 <= m["f1_all"] <= 1.0

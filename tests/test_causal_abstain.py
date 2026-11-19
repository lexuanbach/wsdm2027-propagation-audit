import numpy as np
import torch

from docless.abstain import calibrate_threshold, selective_predict, tv_disagreement
from docless.causal import DocLessHead, StratumEstimator, observational_and_interventional
from docless.data.loaders import generate_synthetic


def test_stratum_estimator_recovers_kp_ke_bins():
    cascades = generate_synthetic(n_stories=200, seed=4)
    est = StratumEstimator(kp=4, ke=4).fit(cascades)
    p, e = est.transform(cascades)
    assert int(p.max()) <= 3 and int(p.min()) >= 0
    assert int(e.max()) <= 3 and int(e.min()) >= 0
    assert est.q_hat.shape == (4, 4)
    np.testing.assert_allclose(est.q_hat.sum(), 1.0, atol=1e-6)


def test_observational_and_interventional_shapes_and_validity():
    torch.manual_seed(0)
    head = DocLessHead(hidden_dim=16, kp=4, ke=4)
    B, H = 7, 16
    h = torch.randn(B, H)
    p = torch.tensor([0, 1, 2, 3, 0, 1, 2])
    e = torch.tensor([1, 1, 2, 0, 3, 0, 3])
    q = torch.ones(4, 4) / 16.0
    f_obs, f_do = observational_and_interventional(head, h, p, e, q)
    assert f_obs.shape == (B, 2)
    assert f_do.shape == (B, 2)
    # f_do should be the log of a valid prob (rows of exp(f_do) ≈ 1)
    assert torch.allclose(f_do.exp().sum(-1), torch.ones(B), atol=1e-4)


def test_tv_disagreement_bounds():
    a = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    b = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
    d = tv_disagreement(a, b)
    assert d[0].item() < 1e-4
    assert d[1].item() > 0.9


def test_threshold_calibration_meets_coverage():
    rng = np.random.default_rng(0)
    deltas = torch.from_numpy(rng.uniform(size=1000).astype("float32"))
    tau = calibrate_threshold(deltas, coverage=0.8)
    frac = (deltas <= tau).float().mean().item()
    # Quantile guarantees fraction ≥ coverage within ±1/n.
    assert 0.79 <= frac <= 0.81


def test_selective_predict_obeys_threshold():
    obs = torch.tensor([[5.0, -5.0], [-5.0, 5.0]])
    do = torch.tensor([[5.0, -5.0], [5.0, -5.0]])   # row 1 disagrees hard, row 0 agrees
    preds, keep = selective_predict(obs, do, tau=0.5)
    assert bool(keep[0].item()) is True
    assert bool(keep[1].item()) is False

"""Causal-inertness at the probability level, swept over lambda_2.

We test whether 'the causal head is inert' is an artifact of
a deliberately weak KL regulariser. We therefore (a) sweep lambda_kl_max upward
and (b) measure inertness on PROBABILITIES, not only argmax:
  * mean |p_do - p_obs| (the actual divergence the do-head induces on test);
  * the fraction of test stories whose argmax flips between obs and do heads;
  * macro-F1 of the obs vs do head (does the adjusted head ever help or hurt?);
  * the same under permuted (P,E_hash) strata (which tests whether the head uses
    the implemented diagnostic variables).
If predictions stay put even as lambda is driven up until the heads visibly
diverge, inertness is not a weak-regulariser artifact. CoAID + ReCOVery, 3 seeds.
"""
from __future__ import annotations
import json, os
import numpy as np

from docless.train import train_one_seed, DataConfig, ModelConfig, TrainConfig

OUT = "outputs/r10/lambda_sweep"; os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2]
LAMBDAS = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
DATASETS = {
    "coaid":    dict(source="coaid", balanced=True, data_root="data/raw/coaid"),
    "recovery": dict(source="recovery", balanced=True, data_root="data/raw/recovery"),
}


def _macro_f1(pred, y):
    pred, y = np.asarray(pred), np.asarray(y); f = []
    for c in (0, 1):
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
        d = 2 * tp + fp + fn; f.append(2 * tp / d if d else 0.0)
    return float(np.mean(f))


def _cell(kw, lam, permute):
    tv, flip, f1o, f1d = [], [], [], []
    for s in SEEDS:
        tc = TrainConfig(lambda_kl_max=lam, permute_confounders=permute)
        r = train_one_seed(s, DataConfig(**kw), ModelConfig(), tc, verbose=False)
        p_do, p_obs = np.array(r["p_do"]), np.array(r["p_obs"])
        y = np.array(r["labels"]); ao, ad = np.array(r["argmax_obs"]), np.array(r["argmax_do"])
        tv.append(float(np.mean(np.abs(p_do - p_obs))))      # TV per story (binary => |dp|)
        flip.append(float(np.mean(ao != ad)))
        f1o.append(_macro_f1(ao, y)); f1d.append(_macro_f1(ad, y))
    return dict(mean_tv=round(float(np.mean(tv)), 4), argmax_flip=round(float(np.mean(flip)), 4),
                f1_obs=round(float(np.mean(f1o)), 3), f1_do=round(float(np.mean(f1d)), 3))


def run():
    out = {}
    for name, kw in DATASETS.items():
        out[name] = {}
        for lam in LAMBDAS:
            base = _cell(kw, lam, permute=False)
            perm = _cell(kw, lam, permute=True)
            out[name][f"lam{lam}"] = dict(base=base, permuted=perm,
                                          do_minus_obs_f1=round(base["f1_do"] - base["f1_obs"], 3))
            print(f"[{name} lam={lam}] TV={base['mean_tv']} flip={base['argmax_flip']} "
                  f"f1_obs={base['f1_obs']} f1_do={base['f1_do']} | permuted TV={perm['mean_tv']} f1_do={perm['f1_do']}", flush=True)
            json.dump(out, open(f"{OUT}/lambda_sweep.json", "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    run()

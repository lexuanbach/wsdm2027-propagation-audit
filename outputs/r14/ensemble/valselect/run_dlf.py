"""Regenerate DLF-Full (7d) / DLF-Full+ (15d) and dump val AND test per-story
predictions and scores.

train_one_seed() already computes a val-fold forward pass (it needs it to
calibrate tau) and then throws the val logits away.  Rather than edit
src/docless/train.py while other jobs are running, we wrap DocLessFake.forward
and keep the last two eval-mode forwards: [-2] is the val pass, [-1] is test.
The training path is untouched, so f1_all must reproduce the published cell.

Usage: run_dlf.py <corpus> <variant: dlf|dlf15> <seed>
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import torch
torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import folds, check_alignment, dump, ROOT
from docless.eval.stats import macro_f1
import docless.model as M
from docless.train import DataConfig, ModelConfig, TrainConfig, train_one_seed

_CAP = []
_orig = M.DocLessFake.forward


def _wrapped(self, *a, **kw):
    out = _orig(self, *a, **kw)
    if not self.training:
        _CAP.append(out)
    return out


M.DocLessFake.forward = _wrapped

corpus, variant, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
rich = variant == "dlf15"
dcfg, tr, va, te = folds(corpus, seed)
check_alignment(corpus, seed, va, te)
t0 = time.time()
_CAP.clear()
r = train_one_seed(seed, dcfg,
                   ModelConfig(hidden_dim=64, rich_summary=rich),
                   TrainConfig(pretrain_epochs=4, finetune_epochs=25),
                   verbose=False, device=torch.device("cpu"))
assert len(_CAP) >= 2, f"expected >=2 eval forwards, got {len(_CAP)}"
out_va, out_te = _CAP[-2], _CAP[-1]
assert out_va["logits_do"].shape[0] == len(va), \
    f"val capture size {out_va['logits_do'].shape[0]} != {len(va)}"
assert out_te["logits_do"].shape[0] == len(te)

vp = out_va["logits_do"].argmax(-1).cpu().numpy()
vs = out_va["logits_do"].exp()[:, 1].detach().cpu().numpy()
tp = np.asarray(r["preds"])
ts = np.asarray(r["p_do"])
# sanity: the captured test forward must agree with what train_one_seed returned
assert (out_te["logits_do"].argmax(-1).cpu().numpy() == tp).all()
assert [c.label for c in te] == r["labels"]

y_va = np.asarray([c.label for c in va])
rec = dump(variant, corpus, seed, va, te, vp, vs, tp, ts,
           extra=dict(tau=r["tau"], coverage=r["coverage"],
                      test_p_obs=r["p_obs"], test_argmax_obs=r["argmax_obs"],
                      val_p_obs=out_va["logits_obs"].softmax(-1)[:, 1].detach().cpu().tolist(),
                      wall_s=round(time.time() - t0, 1)))
print(json.dumps(dict(corpus=corpus, variant=variant, seed=seed,
                      val_f1=round(float(macro_f1(vp, y_va)), 4),
                      test_f1=round(float(r["f1_all"]), 4),
                      published_check=round(float(macro_f1(tp, np.asarray(r["labels"]))), 4),
                      wall_s=round(time.time() - t0, 1))), flush=True)

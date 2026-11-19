"""Run the authors' released BiGCN source unmodified on this machine.

external/BiGCN (commit 5779f2c) pins torch_geometric==1.3.2 / torch_scatter==1.4.0.
Those wheels do not build here, but the released source only needs ONE symbol
from torch_scatter -- scatter_mean(x, batch, dim=0), at
model/Twitter/BiGCN_Twitter.py:44,78 and model/Weibo/BiGCN_Weibo.py:43 -- and
modern torch_geometric ships an equivalent reduction.  This module installs a
recorded three-line shim under the name `torch_scatter` and then executes the
authors' file with runpy, so no line of the released source is edited.

The shim is the ONLY substitution.  Everything else -- the model, the optimizer
groups, the drop rates, the 5-fold split code, the early-stopping rule -- is the
authors' own.

Usage:
  PYTHONPATH=src python3 scripts/official_bigcn_shim.py Twitter15 1
"""
from __future__ import annotations

import os
import pathlib
import runpy
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
BIGCN = ROOT / "external" / "BiGCN"


def install_shim():
    import torch
    from torch_geometric.utils import scatter

    m = types.ModuleType("torch_scatter")

    def scatter_mean(src, index, dim=0, out=None, dim_size=None):
        return scatter(src, index, dim=dim, dim_size=dim_size, reduce="mean")

    def scatter_add(src, index, dim=0, out=None, dim_size=None):
        return scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")

    def scatter_max(src, index, dim=0, out=None, dim_size=None):
        return scatter(src, index, dim=dim, dim_size=dim_size, reduce="max"), None

    m.scatter_mean, m.scatter_add, m.scatter_max = scatter_mean, scatter_add, scatter_max
    m.__doc__ = "recorded shim over torch_geometric.utils.scatter; see official_bigcn_shim.py"
    sys.modules["torch_scatter"] = m

    # NumPy 2.0 removed the `np.Inf` alias that tools/earlystopping.py uses.
    # Restoring the alias is a rename, not a change to the authors' method.
    import numpy as np
    if not hasattr(np, "Inf"):
        np.Inf = np.inf
    if not hasattr(np, "float_"):
        np.float_ = np.float64

    import torch_geometric
    return {"scatter_shim": "torch_scatter.scatter_mean -> torch_geometric.utils.scatter(reduce='mean')",
            "numpy_alias_restored": ["np.Inf", "np.float_"],
            "torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
            "numpy": np.__version__}


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(__doc__)
    target = os.environ.get("BIGCN_ENTRY", "model/Twitter/BiGCN_Twitter.py")
    # The released code passes num_workers=5 to DataLoader and has no
    # `if __name__ == "__main__"` guard, so it requires the fork start method
    # that was the Linux default when it was published.  macOS/py3.14 defaults
    # to spawn, which would re-execute the driver in every worker.
    import multiprocessing as mp
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    info = install_shim()
    info["mp_start_method"] = mp.get_start_method()
    print(f"[official-bigcn] {info}", flush=True)
    os.chdir(BIGCN)                      # the released code uses os.getcwd() for data paths
    sys.path.insert(0, str(BIGCN))
    sys.argv = [str(BIGCN / target)] + args
    runpy.run_path(str(BIGCN / target), run_name="__main__")


if __name__ == "__main__":
    main()

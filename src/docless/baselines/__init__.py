"""Baseline implementations for the Table I comparison.

Each baseline implements the same `train_one_seed(seed, data_cfg, ...) -> dict`
shape as `docless.train.train_one_seed`, so the eval runners are baseline-agnostic.

Implemented locally (no external deps):
    majority         class-majority floor; verifies the metric pipeline
    gcnfn            geometric-DL propagation GCN, no text (Monti et al. 2019)
    upfd_prop        propagation-only readout matching the UPFD paper's `prop` variant
    bigcn            bidirectional propagation GCN (Bian et al. 2020)
    rgcn             relational/heterogeneous GCN over typed hyperedges (Schlichtkrull 2018)
    gatv2            modern dynamic-attention GNN, type-aware (Brody et al. 2022)

Not implemented (need external repos / data / API):
    dEFEND, GCAN, FANG, CSI, EBGCN, HG-SL
    DELL, MMFND, HypergraphMisinfo-2024, CausalMisinfo  (2024-2025 SOTA)
"""
from .majority import train_majority
from .gcnfn import train_gcnfn
from .upfd_prop import train_upfd_prop
from .bigcn import train_bigcn
from .propmlp import train_propmlp
from .rgcn import train_rgcn
from .gatv2 import train_gatv2

__all__ = ["train_majority", "train_gcnfn", "train_upfd_prop", "train_bigcn",
           "train_propmlp", "train_rgcn", "train_gatv2"]

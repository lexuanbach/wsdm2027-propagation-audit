"""Device selection. Order: explicit > CUDA > MPS > CPU.

`use_amp` returns False on MPS even if requested — torch 2.x autocast on MPS
is unstable for the ops the HGNN layer relies on.
"""
from __future__ import annotations

import os

import torch


def select_device(prefer: str | None = None) -> torch.device:
    """Pick the best device.

    Note: for our CoAID-sized hypergraphs (~10k nodes), MPS is *slower* than
    CPU on Apple Silicon because (i) the scatter_reduce(amax) op has no native
    MPS kernel in torch 2.x and we round-trip to CPU for it, (ii) host↔device
    copies dominate small-graph compute. We therefore stay on CPU by default
    on Apple Silicon and only escalate to MPS for larger graphs (override via
    `prefer='mps'` or `DOCLESS_DEVICE=mps`). On a CUDA host we default to CUDA.
    """
    if prefer:
        return torch.device(prefer)
    env = os.environ.get("DOCLESS_DEVICE")
    if env:
        return torch.device(env)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_hdata(h, device: torch.device):
    """Move HData tensors to device in place and return."""
    h.x = h.x.to(device)
    h.node_type = h.node_type.to(device)
    h.edge_index = h.edge_index.to(device)
    h.edge_type = h.edge_type.to(device)
    if h.story_labels is not None:
        h.story_labels = h.story_labels.to(device)
    return h

"""Diagnostic stratum-adjusted classifier head (legacy module path).

Two outputs:
  - observational:  f_obs(h, p_s, e_s)
  - interventional: f_do(h)  =  Σ_{p,e} q̂(p,e) · f(h, p, e)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DocLessHead(nn.Module):
    """MLP head conditioned on one-hot (P, E_hash) strata."""

    def __init__(self, hidden_dim: int, kp: int, ke: int, num_classes: int = 2) -> None:
        super().__init__()
        self.kp, self.ke = kp, ke
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + kp + ke, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, h: Tensor, p_bin: Tensor, e_bin: Tensor) -> Tensor:
        # h: [B, H]; p_bin/e_bin: [B]
        p_oh = F.one_hot(p_bin, self.kp).float()
        e_oh = F.one_hot(e_bin, self.ke).float()
        z = torch.cat([h, p_oh, e_oh], dim=-1)
        return self.mlp(z)                              # logits [B, C]


def observational_and_interventional(
    head: DocLessHead,
    h_story: Tensor,                # [B, H]
    p_bin: Tensor,                  # [B]
    e_bin: Tensor,                  # [B]
    q_hat: Tensor,                  # [kp, ke]
) -> tuple[Tensor, Tensor]:
    """Return (f_obs_logits, f_do_logits).

    f_obs conditions on the story's own (p_s, e_s).
    f_do marginalizes head outputs over the empirical q̂(p, e) (Eq. 7).
    """
    B, H = h_story.shape
    kp, ke = head.kp, head.ke

    f_obs = head(h_story, p_bin, e_bin)                 # [B, C]

    # Vectorized marginalization: build all (kp*ke) strata and average probs.
    # Probabilities (not logits) are averaged so the result is a valid
    # distribution; we return logits via log of the averaged probability.
    p_grid = torch.arange(kp, device=h_story.device).repeat_interleave(ke)         # [kp*ke]
    e_grid = torch.arange(ke, device=h_story.device).repeat(kp)                    # [kp*ke]
    h_rep = h_story.unsqueeze(1).expand(B, kp * ke, H).reshape(B * kp * ke, H)
    p_rep = p_grid.repeat(B)
    e_rep = e_grid.repeat(B)

    logits = head(h_rep, p_rep, e_rep).view(B, kp * ke, -1)                         # [B, kp*ke, C]
    probs = logits.softmax(-1)
    w = q_hat.flatten().to(h_story.device).view(1, kp * ke, 1)                      # [1, kp*ke, 1]
    p_do = (probs * w).sum(1)                                                        # [B, C]
    f_do_logits = (p_do.clamp_min(1e-12)).log()
    return f_obs, f_do_logits

"""Synthetic cascade generator.

Produces Cascade objects from latent popularity/community bins so the pipeline
can be unit-tested without external data. These latent bins are not the same as
the production event-proxy/hash-concentration features; the generator is a
software stress test, not validation of the paper's causal assumptions.

    P, E   ~  marginal q(p, e)
    T      ~  h(P, E, U_T)
    Y      ~  g(T, U_Y)

with an added latent perturbation controlled by Γ.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..schema import Cascade, Event


def _stratum_label_logit(p: int, e: int, kp: int, ke: int) -> float:
    """A non-trivial dependence of E[Y|P,E] on the stratum."""
    pn, en = p / max(kp - 1, 1), e / max(ke - 1, 1)
    return 2.0 * (pn - 0.5) + 1.5 * (en - 0.5) - 0.5


def generate_synthetic(
    n_stories: int = 200,
    n_users: int = 500,
    n_urls: int = 80,
    n_media: int = 40,
    kp: int = 4,
    ke: int = 4,
    gamma: float = 1.5,
    seed: int = 0,
) -> list[Cascade]:
    """Generate `n_stories` cascades with a known causal structure."""
    rng = np.random.default_rng(seed)
    cascades: list[Cascade] = []

    # Per-user propensity for joining viral cascades (drives popularity confounder).
    user_pop = rng.beta(2.0, 5.0, size=n_users)
    # Per-user community assignment (drives echo-chamber confounder).
    user_com = rng.integers(0, 4, size=n_users)

    for sid in range(n_stories):
        # Sample stratum
        p_bin = int(rng.integers(0, kp))
        e_bin = int(rng.integers(0, ke))

        # Cascade size grows with popularity bin
        size = int(rng.poisson(lam=8 + 10 * p_bin / max(kp - 1, 1))) + 3
        # Echo-chamber bin biases the community distribution
        com_pref = rng.dirichlet(np.ones(4) * (1.0 + e_bin))
        users = rng.choice(n_users, size=size, replace=True, p=_renorm(user_pop))

        # Echo-chamber re-sampling: drop a fraction that don't match the
        # preferred community, replace with in-community users.
        for i in range(size):
            if rng.random() < e_bin / max(ke - 1, 1) * 0.6:
                # bias toward preferred community
                tgt = int(rng.choice(4, p=com_pref))
                same = np.where(user_com == tgt)[0]
                if len(same):
                    users[i] = int(rng.choice(same))

        # Time stamps (one cascade ~ 60 minutes)
        t0 = float(rng.uniform(0, 1_000_000))
        deltas = np.sort(rng.exponential(scale=60.0, size=size))
        timestamps = t0 + deltas

        # URLs + media — most events share a small set
        urls = rng.choice(n_urls, size=size, p=_uniform_skewed(n_urls, rng))
        media = np.where(
            rng.random(size) < 0.6,
            rng.choice(n_media, size=size),
            -1,
        )

        events = [
            Event(int(u), int(r), int(m), float(t))
            for u, r, m, t in zip(users, urls, media, timestamps)
        ]

        # Causal mechanism: label ~ logistic(stratum_logit + hidden_conf)
        hidden = float(rng.normal(0, 1)) * np.log(gamma)
        logit = _stratum_label_logit(p_bin, e_bin, kp, ke) + hidden
        prob = 1.0 / (1.0 + np.exp(-logit))
        label = int(rng.random() < prob)

        early_end = float(timestamps[0] + 600.0)  # 10-minute early window
        cascades.append(
            Cascade(
                story_id=f"syn-{sid:05d}",
                label=label,
                events=events,
                language="en",
                early_window_end=early_end,
                timestamp=t0,
            )
        )
    return cascades


def _renorm(x: np.ndarray) -> np.ndarray:
    return x / x.sum()


def _uniform_skewed(n: int, rng: np.random.Generator) -> np.ndarray:
    p = rng.gamma(2.0, 1.0, size=n)
    return _renorm(p)


class SyntheticLoader:
    """File-backed loader matching the same interface real dataset loaders will use."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def load(self) -> list[Cascade]:
        path = self.root / "cascades.jsonl"
        out: list[Cascade] = []
        with path.open() as f:
            for line in f:
                obj = json.loads(line)
                events = [Event(**e) for e in obj.pop("events")]
                out.append(Cascade(events=events, **obj))
        return out

    def save(self, cascades: list[Cascade]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "cascades.jsonl"
        with path.open("w") as f:
            for c in cascades:
                obj = asdict(c)
                f.write(json.dumps(obj) + "\n")


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-stories", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cascades = generate_synthetic(n_stories=args.n_stories, seed=args.seed)
    SyntheticLoader(args.out).save(cascades)
    print(f"wrote {len(cascades)} synthetic cascades to {args.out}/cascades.jsonl")


if __name__ == "__main__":
    _cli()

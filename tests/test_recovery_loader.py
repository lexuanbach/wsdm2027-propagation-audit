from __future__ import annotations

from pathlib import Path

import pytest

from docless.data.loaders import load_recovery


RECOVERY_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "recovery"
pytestmark = pytest.mark.skipif(
    not (RECOVERY_ROOT / "dataset" / "recovery-news-data.csv").exists(),
    reason="ReCOVery data not present (run scripts/prepare_recovery.sh first)",
)


def test_recovery_loads_both_labels():
    cs = load_recovery(RECOVERY_ROOT, min_events_per_story=3)
    assert len(cs) > 500
    labels = {c.label for c in cs}
    assert labels == {0, 1}
    # Cascades are time-ordered.
    for c in cs[:5]:
        ts = [e.timestamp for e in c.events]
        assert ts == sorted(ts)


def test_recovery_balanced_flag_exact():
    cs = load_recovery(RECOVERY_ROOT, balanced=True, seed=0)
    n_pos = sum(1 for c in cs if c.label == 1)
    n_neg = sum(1 for c in cs if c.label == 0)
    assert n_pos == n_neg
    assert n_pos > 100  # ReCOVery has 453 fakes; balanced ≈ 906


def test_recovery_event_cap_respected():
    cs = load_recovery(RECOVERY_ROOT, max_events_per_story=50)
    assert all(len(c.events) <= 50 for c in cs)

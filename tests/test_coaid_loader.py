"""Tests against the real CoAID checkout under data/raw/coaid/.

These are skipped if the data isn't present (so CI without the checkout
still passes), but on a developer machine that ran the clone they run
real assertions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from docless.data.loaders import load_coaid
from docless.data.loaders.coaid import _decode_snowflake_ts, _stable_url_id


COAID_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "coaid"
pytestmark = pytest.mark.skipif(
    not COAID_ROOT.exists() or not any(COAID_ROOT.glob("*-*-2020")),
    reason="CoAID data not present (run scripts/prepare_coaid.sh first)",
)


def test_snowflake_decode_matches_known_id():
    # Known tweet: 1255263076087185413 -> 2020-04-28T22:30:13 UTC
    ts = _decode_snowflake_ts(1255263076087185413)
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    assert dt.year == 2020 and dt.month == 4 and dt.day == 28


def test_stable_url_id_is_deterministic_and_positive():
    a = _stable_url_id("https://example.com/a")
    b = _stable_url_id("https://example.com/a")
    c = _stable_url_id("https://example.com/b")
    assert a == b
    assert a != c
    assert a > 0


def test_coaid_loads_with_both_labels():
    cs = load_coaid(COAID_ROOT, min_events_per_story=3, max_events_per_story=200)
    assert len(cs) > 100
    labels = {c.label for c in cs}
    assert 0 in labels and 1 in labels
    # Cascade is ordered by timestamp
    for c in cs[:5]:
        ts = [e.timestamp for e in c.events]
        assert ts == sorted(ts)


def test_coaid_balanced_flag_balances_class_counts():
    cs = load_coaid(COAID_ROOT, balanced=True, seed=0)
    n_pos = sum(1 for c in cs if c.label == 1)
    n_neg = sum(1 for c in cs if c.label == 0)
    assert n_pos == n_neg
    assert n_pos > 50      # CoAID has ~107 fakes; balanced -> ~214 stories

"""FakeNewsNet / Politifact loader (Shu et al., 2020 -- KaiDMML/FakeNewsNet).

The public repo ships, per source, two CSVs with columns
    id, news_url, title, tweet_ids
where `tweet_ids` is a TAB-separated list of the tweet ids that engaged with
the article. The tweet *ids* are public (no hydration / Twitter API needed);
as for CoAID and ReCOVery, we recover a per-event timestamp from the snowflake
encoding and treat each tweet id as a unique event proxy. This provides no
stable user identity or shipped retweet edges.

Politifact is a political fact-checking domain -- distinct from the COVID-19
topic of CoAID/ReCOVery -- so it serves as an out-of-domain third dataset.

Label convention: fake -> 1 (misinformation), real -> 0 (credible).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from ..schema import Cascade, Event
from .coaid import _decode_snowflake_ts, _stable_url_id

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def _read_source(path: Path, label: int) -> list[tuple[str, list[int], int]]:
    """Return [(news_url, [tweet_id, ...], label)] for one source CSV."""
    rows: list[tuple[str, list[int], int]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = (row.get("news_url") or "").strip()
            raw = (row.get("tweet_ids") or "").strip()
            if not url or not raw:
                continue
            tids: list[int] = []
            for tok in raw.replace("\t", " ").split():
                try:
                    tids.append(int(tok))
                except ValueError:
                    continue
            if tids:
                rows.append((url, tids, label))
    return rows


def load_fakenewsnet(
    root: str | Path,
    source: str = "politifact",
    min_events_per_story: int = 3,
    max_events_per_story: int | None = 200,
    balanced: bool = False,
    seed: int = 0,
    max_per_class: int | None = None,
) -> list[Cascade]:
    """Load a FakeNewsNet source (default Politifact) into a list of Cascade."""
    root = Path(root)
    cascades: list[Cascade] = []
    seen_urls: set[str] = set()

    for kind, label in (("fake", 1), ("real", 0)):
        path = root / f"{source}_{kind}.csv"
        if not path.exists():
            continue
        for url, tids, lab in _read_source(path, label):
            if url in seen_urls:
                continue
            seen_urls.add(url)
            url_id = _stable_url_id(url)
            events: list[Event] = []
            for tid in tids:
                try:
                    ts = _decode_snowflake_ts(tid)
                except Exception:
                    continue
                events.append(Event(user_id=int(tid), url_id=int(url_id), media_id=-1, timestamp=float(ts)))
            if len(events) < min_events_per_story:
                continue
            events.sort(key=lambda e: e.timestamp)
            if max_events_per_story is not None and len(events) > max_events_per_story:
                events = events[:max_events_per_story]
            t0 = events[0].timestamp
            early_end = min(events[min(9, len(events) - 1)].timestamp, t0 + 3600.0)
            cascades.append(Cascade(
                story_id=f"{source}:{url_id:016x}",
                label=lab,
                events=events,
                language="en",
                early_window_end=early_end,
                timestamp=t0,
            ))

    if balanced and cascades:
        import random as _r
        rng = _r.Random(seed)
        pos = [c for c in cascades if c.label == 1]
        neg = [c for c in cascades if c.label == 0]
        n = min(len(pos), len(neg))
        if max_per_class is not None:
            n = min(n, max_per_class)
        if n > 0:
            rng.shuffle(neg); rng.shuffle(pos)
            cascades = pos[:n] + neg[:n]
            rng.shuffle(cascades)
    return cascades

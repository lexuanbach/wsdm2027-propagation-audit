"""ReCOVery loader (Zhou et al. 2020 — https://github.com/apurvamulay/ReCOVery).

ReCOVery ships two CSVs:
    recovery-news-data.csv          news_id -> url, publisher, reliability (0/1)
    recovery-social-media-data.csv  news_id -> tweet_id

Like CoAID, no user IDs or timestamps live in the CSVs (Twitter ToS); we
recover per-event timestamps from the tweet_id snowflake encoding.

Label convention here matches paper:
    reliability == 0  ->  misinformation  (our label = 1)
    reliability == 1  ->  credible       (our label = 0)
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..schema import Cascade, Event
from .coaid import _decode_snowflake_ts, _stable_url_id


def load_recovery(
    root: str | Path,
    min_events_per_story: int = 3,
    max_events_per_story: int | None = 200,
    balanced: bool = False,
    seed: int = 0,
) -> list[Cascade]:
    """Parse the ReCOVery checkout at `root/dataset/`."""
    root = Path(root)
    news_csv = root / "dataset" / "recovery-news-data.csv"
    social_csv = root / "dataset" / "recovery-social-media-data.csv"
    if not news_csv.exists() or not social_csv.exists():
        raise FileNotFoundError(
            f"ReCOVery files missing under {root}/dataset/. "
            f"Run: bash scripts/prepare_recovery.sh"
        )

    # 1. news_id -> (label, url)
    nid_to: dict[str, tuple[int, str]] = {}
    with news_csv.open(newline="", encoding="utf-8", errors="replace") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            nid = row.get("news_id", "").strip()
            url = (row.get("url") or "").strip()
            rel = (row.get("reliability") or "").strip()
            if not nid or not url or rel not in ("0", "1"):
                continue
            # reliability=0 -> misinformation -> label=1
            label = 1 if rel == "0" else 0
            nid_to[nid] = (label, url)

    # 2. collect tweet ids per news_id
    by_nid: dict[str, list[int]] = {}
    with social_csv.open(newline="", encoding="utf-8", errors="replace") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            nid = (row.get("news_id") or "").strip()
            tid = (row.get("tweet_id") or "").strip()
            if not nid or not tid:
                continue
            try:
                tid_int = int(tid)
            except ValueError:
                continue
            by_nid.setdefault(nid, []).append(tid_int)

    # 3. assemble Cascades
    cascades: list[Cascade] = []
    for nid, tids in by_nid.items():
        if nid not in nid_to:
            continue
        if len(tids) < min_events_per_story:
            continue
        label, url = nid_to[nid]
        url_id = _stable_url_id(url)
        # de-dupe tweet ids
        tids = list(dict.fromkeys(tids))
        events: list[Event] = []
        for tid in tids:
            try:
                ts = _decode_snowflake_ts(tid)
            except Exception:
                continue
            events.append(Event(
                user_id=int(tid),
                url_id=int(url_id),
                media_id=-1,
                timestamp=float(ts),
            ))
        if len(events) < min_events_per_story:
            continue
        events.sort(key=lambda e: e.timestamp)
        if max_events_per_story is not None and len(events) > max_events_per_story:
            events = events[:max_events_per_story]
        t0 = events[0].timestamp
        early_end = min(
            events[min(9, len(events) - 1)].timestamp,
            t0 + 3600.0,
        )
        cascades.append(Cascade(
            story_id=f"recovery:{nid}",
            label=label,
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
        if n > 0:
            rng.shuffle(neg); rng.shuffle(pos)
            cascades = pos[:n] + neg[:n]
            rng.shuffle(cascades)

    return cascades

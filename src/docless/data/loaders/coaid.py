"""CoAID loader (Cui & Lee, 2020 — https://github.com/cuilimeng/CoAID).

CoAID ships:
    News(Fake|Real)COVID-19.csv               news_id (column 0) -> news_url, title, ...
    News(Fake|Real)COVID-19_tweets.csv        news_id -> tweet_id
    News(Fake|Real)COVID-19_tweets_replies.csv  news_id -> tweet_id -> reply_id

The CSVs intentionally do *not* contain user ids or timestamps (Twitter ToS).
We recover timestamps from the snowflake encoding of the tweet/reply id
(top 41 bits = ms since the Twitter epoch 2010-11-04) and treat each
tweet_id as a unique event proxy. This recovers an event list, not stable user
identity or retweet-tree structure, and makes the release usable without an API.

Four dated snapshots ship in the repo; we merge them and de-duplicate by
(news_url, tweet_id).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

from ..schema import Cascade, Event


# Twitter snowflake epoch (ms): "the first second of 2010-11-04 UTC"
_TWITTER_EPOCH_MS = 1_288_834_974_657


def _decode_snowflake_ts(tid: int) -> float:
    """Return unix seconds (UTC) for a Twitter snowflake id."""
    return (_TWITTER_EPOCH_MS + (int(tid) >> 22)) / 1000.0


def _stable_url_id(url: str) -> int:
    """Deterministic 63-bit positive int from a URL string."""
    # FNV-1a 64-bit — deterministic across Python builds (PYTHONHASHSEED-free).
    h = 0xcbf29ce484222325
    for b in url.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001b3) & 0xffffffffffffffff
    return h & 0x7fffffffffffffff  # keep positive


def _read_news_index_to_url(path: Path) -> dict[int, str]:
    """Read NewsX.csv and return {news_index_in_csv: news_url}."""
    out: dict[int, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        # The header has a blank first column; news_url lives at position 4.
        try:
            url_col = header.index("news_url")
        except ValueError:
            # Some snapshots use slightly different headers
            url_col = 4
        for row in reader:
            if not row or not row[0]:
                continue
            try:
                idx = int(row[0])
            except ValueError:
                continue
            url = row[url_col] if url_col < len(row) else ""
            if url:
                out[idx] = url
    return out


def _read_tweets(path: Path, idx_to_url: dict[int, str]) -> list[tuple[str, int]]:
    """Yield (news_url, tweet_id) pairs from a NewsX_tweets.csv."""
    out: list[tuple[str, int]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ni = int(row.get("index") or row.get("﻿index") or "")
                tid = int(row.get("tweet_id") or "")
            except ValueError:
                continue
            url = idx_to_url.get(ni)
            if url:
                out.append((url, tid))
    return out


def _read_replies(path: Path, idx_to_url: dict[int, str]) -> list[tuple[str, int, int]]:
    """Yield (news_url, tweet_id, reply_id) triples from a *_tweets_replies.csv."""
    out: list[tuple[str, int, int]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ni = int(row.get("news_id") or row.get("﻿news_id") or "")
                tid = int(row.get("tweet_id") or "")
                rid = int(row.get("reply_id") or "")
            except ValueError:
                continue
            url = idx_to_url.get(ni)
            if url:
                out.append((url, tid, rid))
    return out


def load_coaid(
    root: str | Path,
    snapshots: list[str] | None = None,
    min_events_per_story: int = 3,
    include_replies: bool = True,
    max_events_per_story: int | None = 200,
    balanced: bool = False,
    seed: int = 0,
) -> list[Cascade]:
    """Load CoAID into a list of `Cascade`.

    Parameters
    ----------
    root
        Path to the cloned CoAID repo (containing `05-01-2020/`, `07-01-2020/`, ...).
    snapshots
        Subset of date folders to read. Defaults to all that exist.
    min_events_per_story
        Drop stories with fewer than this many engagement events. The default
        of 3 keeps anything with at least one tweet + one reply or three tweets,
        matching what the propagation literature uses.
    """
    root = Path(root)
    if snapshots is None:
        snapshots = sorted([p.name for p in root.iterdir() if re.match(r"\d{2}-\d{2}-\d{4}", p.name)])

    # url -> dict
    by_url: dict[str, dict] = {}  # url -> {label, events: set[(tid, src)]}
    seen = set()                  # (url, event_id) dedupe across snapshots

    def _ensure(url: str, label: int) -> dict:
        if url not in by_url:
            by_url[url] = {"label": label, "events": []}
        return by_url[url]

    for snap in snapshots:
        snap_dir = root / snap
        if not snap_dir.exists():
            continue
        for kind, label in [("Fake", 1), ("Real", 0)]:
            news_csv = snap_dir / f"News{kind}COVID-19.csv"
            tweets_csv = snap_dir / f"News{kind}COVID-19_tweets.csv"
            replies_csv = snap_dir / f"News{kind}COVID-19_tweets_replies.csv"
            if not news_csv.exists():
                continue
            idx2url = _read_news_index_to_url(news_csv)

            if tweets_csv.exists():
                for url, tid in _read_tweets(tweets_csv, idx2url):
                    key = (url, "t", tid)
                    if key in seen:
                        continue
                    seen.add(key)
                    bucket = _ensure(url, label)
                    bucket["events"].append(("t", tid))

            if include_replies and replies_csv.exists():
                for url, tid, rid in _read_replies(replies_csv, idx2url):
                    key = (url, "r", rid)
                    if key in seen:
                        continue
                    seen.add(key)
                    bucket = _ensure(url, label)
                    bucket["events"].append(("r", rid))

    # Build Cascade objects.
    cascades: list[Cascade] = []
    for url, info in by_url.items():
        if len(info["events"]) < min_events_per_story:
            continue
        events: list[Event] = []
        url_id = _stable_url_id(url)
        for src, tid in info["events"]:
            try:
                ts = _decode_snowflake_ts(tid)
            except Exception:
                continue
            events.append(Event(
                user_id=int(tid),       # proxy: each tweet/reply is its own sharer
                url_id=int(url_id),
                media_id=-1,            # CoAID has no media metadata
                timestamp=float(ts),
            ))
        if len(events) < min_events_per_story:
            continue
        events.sort(key=lambda e: e.timestamp)
        if max_events_per_story is not None and len(events) > max_events_per_story:
            # Stratified-truncate around the early window: keep the first
            # max_events events so the cascade still ends well past the
            # P/E measurement horizon.
            events = events[:max_events_per_story]
        # Early window: first 10 events or 1 hour, whichever is smaller.
        t0 = events[0].timestamp
        early_end = min(events[min(9, len(events) - 1)].timestamp, t0 + 3600.0)
        cascades.append(Cascade(
            story_id=f"coaid:{url_id:016x}",
            label=info["label"],
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

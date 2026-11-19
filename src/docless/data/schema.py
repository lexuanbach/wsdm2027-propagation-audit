"""Canonical data schema. Paper §III-A: nodes ∈ users ∪ urls ∪ media ∪ cascade-slices."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True)
class Event:
    """A single engagement event in a story's cascade."""
    user_id: int
    url_id: int
    media_id: int           # -1 if no media
    timestamp: float        # unix seconds


@dataclass
class Cascade:
    """A story instantiated as an ordered cascade of events.

    The encoder NEVER sees article text — story is referenced only through
    structural and behavioural statistics of its events (paper §III-B).
    """
    story_id: str
    label: int                              # 0 = credible, 1 = misinformation
    events: list[Event] = field(default_factory=list)
    language: str = "en"                    # ISO-639-1
    early_window_end: Optional[float] = None  # for P, E measurement (Assumption 2)
    timestamp: Optional[float] = None       # story creation time, for temporal splits

    def num_users(self) -> int:
        return len({e.user_id for e in self.events})

    def num_urls(self) -> int:
        return len({e.url_id for e in self.events})

    def num_media(self) -> int:
        return len({e.media_id for e in self.events if e.media_id >= 0})

    def early_events(self) -> list[Event]:
        if self.early_window_end is None:
            return self.events
        return [e for e in self.events if e.timestamp <= self.early_window_end]

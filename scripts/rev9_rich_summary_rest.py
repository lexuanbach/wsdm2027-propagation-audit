"""Fill the two unevaluated DocLessFake+ (15d) cells of the main table.

`rich_summary_eval.py` ran the four-variant rich-feature audit on the two
powered event-list corpora only (ReCOVery, PolitiFact), which is why the
DocLessFake+ row of the main table carries dashes on CoAID and GossipCop.
This script runs the identical protocol -- same seeds, same DataConfig, same
ModelConfig/TrainConfig -- on the remaining two corpora.  Re-running the
`docless` (7d) arm is deliberate: its mean must reproduce the published
main-table cell (CoAID 0.654, GossipCop 0.842) for the new DocLessFake+ cell
to be protocol-comparable with the rest of the row.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "scripts")

from rich_summary_eval import run_dataset  # noqa: E402

DATASETS = {
    "coaid": dict(source="coaid", balanced=True, data_root="data/raw/coaid"),
    "gossipcop": dict(source="gossipcop", balanced=True,
                      data_root="data/raw/fakenewsnet", max_per_class=1500),
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(DATASETS)
    for name in names:
        run_dataset(name, DATASETS[name])
    print("ALL DONE", flush=True)

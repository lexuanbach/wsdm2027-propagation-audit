"""Text reference point: a title-only TF-IDF + logistic-regression
classifier and a title+summary fusion on the FakeNewsNet corpora, under the
identical stratified splits and seeds as the text-free tables. This is *not*
a text-free method; it is upper-bound context
and an illustration of why text-free detection is a different problem: the
public releases contain only the headline, and headline signal is exactly what
an adversary deletes or paraphrases.

Outputs: outputs/r11/text_bound/text_bound.json
Usage: PYTHONPATH=src python3 scripts/rev6_text_bound.py
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from docless.baselines.propmlp import _features  # noqa: E402
from docless.data.loaders.coaid import _stable_url_id  # noqa: E402
from docless.eval.stats import macro_f1  # noqa: E402
from docless.train import DataConfig, _load_cascades, _set_seed, _stratified_split  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "r11" / "text_bound"
SEEDS = [0, 1, 2, 3, 4]
DATASETS = {
    "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
    "gossipcop": dict(source="gossipcop", balanced=True, data_root="data/raw/fakenewsnet", max_per_class=1500),
}


def titles_for(source: str) -> dict[str, str]:
    out = {}
    for kind in ("fake", "real"):
        p = ROOT / "data" / "raw" / "fakenewsnet" / f"{source}_{kind}.csv"
        with p.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                url = (row.get("news_url") or "").strip()
                title = (row.get("title") or "").strip()
                if url:
                    out[f"{source}:{_stable_url_id(url):016x}"] = title
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, kw in DATASETS.items():
        dcfg = DataConfig(**kw)
        titles = titles_for(name)
        res = {"title_tfidf_lr": [], "title_plus_summary15_lr": [], "summary15_lr": [], "histgbt15_ref": []}
        missing = 0
        for s in SEEDS:
            _set_seed(s)
            casc = _load_cascades(dcfg, s)
            tr, va, te = _stratified_split(casc, dcfg.split_ratios, s)
            def txt(cs):
                nonlocal missing
                t = []
                for c in cs:
                    v = titles.get(c.story_id, "")
                    if not v:
                        missing += 1
                    t.append(v)
                return t
            y_tr = np.array([c.label for c in tr]); y_te = np.array([c.label for c in te])
            vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True).fit(txt(tr))
            T_tr, T_te = vec.transform(txt(tr)), vec.transform(txt(te))
            x_tr, x_te = _features(tr, rich=True), _features(te, rich=True)
            sc = StandardScaler().fit(x_tr)
            S_tr, S_te = csr_matrix(sc.transform(x_tr)), csr_matrix(sc.transform(x_te))
            lr_t = LogisticRegression(max_iter=3000, C=4.0, random_state=s).fit(T_tr, y_tr)
            res["title_tfidf_lr"].append(macro_f1(lr_t.predict(T_te), y_te))
            lr_ts = LogisticRegression(max_iter=3000, C=4.0, random_state=s).fit(hstack([T_tr, S_tr]), y_tr)
            res["title_plus_summary15_lr"].append(macro_f1(lr_ts.predict(hstack([T_te, S_te])), y_te))
            lr_s = LogisticRegression(max_iter=3000, random_state=s).fit(sc.transform(x_tr), y_tr)
            res["summary15_lr"].append(macro_f1(lr_s.predict(sc.transform(x_te)), y_te))
            hg = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, random_state=0).fit(sc.transform(x_tr), y_tr)
            res["histgbt15_ref"].append(macro_f1(hg.predict(sc.transform(x_te)), y_te))
            print(f"[{name}] seed {s}: " + " ".join(f"{k}={v[-1]:.3f}" for k, v in res.items()), flush=True)
        result[name] = {k: {"f1_mean": round(float(np.mean(v)), 4), "f1_std": round(float(np.std(v)), 4)}
                        for k, v in res.items()}
        result[name]["stories_without_title"] = missing
    (OUT / "text_bound.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()

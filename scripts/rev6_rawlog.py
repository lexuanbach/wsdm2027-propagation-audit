"""Raw-vs-log feature ablation : the 7-d summary with raw counts/durations
instead of log-transformed ones, for PropMLP, RandomForest and HistGBT, 5 seeds,
identical stratified splits. Output: outputs/r11/rawlog/rawlog.json."""
import json, pathlib, sys
import numpy as np, torch, torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, "src")
from docless.baselines.propmlp import _MLP, _features
from docless.eval.stats import macro_f1
from docless.train import DataConfig, _load_cascades, _set_seed, _stratified_split
OUT = pathlib.Path("outputs/r11/rawlog"); OUT.mkdir(parents=True, exist_ok=True)
DATASETS = {"coaid": dict(source="coaid", balanced=True, data_root="data/raw/coaid"),
            "recovery": dict(source="recovery", balanced=True, data_root="data/raw/recovery"),
            "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
            "gossipcop": dict(source="gossipcop", balanced=True, data_root="data/raw/fakenewsnet", max_per_class=1500)}
LOG_COLS = [0, 1, 2, 3, 4, 6]  # log n, log u, log r, log m, log P, (E raw), log spread
def unlog(x):
    x = x.copy(); x[:, LOG_COLS] = np.expm1(np.clip(x[:, LOG_COLS], 0, 50)); return x
def mlp(x_tr, y_tr, x_te, seed):
    torch.manual_seed(seed); m = _MLP(in_dim=x_tr.shape[1], hidden=64); opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    xt, yt = torch.tensor(x_tr, dtype=torch.float32), torch.tensor(y_tr)
    for _ in range(25):
        m.train(); opt.zero_grad(); F.cross_entropy(m(xt), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return m(torch.tensor(x_te, dtype=torch.float32)).argmax(-1).numpy()
res = {}
for name, kw in DATASETS.items():
    dcfg = DataConfig(**kw); r = {f"{m}_{v}": [] for m in ("propmlp", "rf", "histgbt") for v in ("log", "raw")}
    for s in range(5):
        _set_seed(s); casc = _load_cascades(dcfg, s); tr, va, te = _stratified_split(casc, dcfg.split_ratios, s)
        y_tr = np.array([c.label for c in tr]); y_te = np.array([c.label for c in te])
        for v in ("log", "raw"):
            x_tr, x_te = _features(tr), _features(te)
            if v == "raw": x_tr, x_te = unlog(x_tr), unlog(x_te)
            sc = StandardScaler().fit(x_tr); a, b = sc.transform(x_tr), sc.transform(x_te)
            r[f"propmlp_{v}"].append(macro_f1(mlp(a, y_tr, b, s), y_te))
            r[f"rf_{v}"].append(macro_f1(RandomForestClassifier(200, max_depth=8, random_state=s).fit(a, y_tr).predict(b), y_te))
            r[f"histgbt_{v}"].append(macro_f1(HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(a, y_tr).predict(b), y_te))
    res[name] = {k: round(float(np.mean(v)), 4) for k, v in r.items()}; print(name, res[name], flush=True)
json.dump(res, open(OUT / "rawlog.json", "w"), indent=2)

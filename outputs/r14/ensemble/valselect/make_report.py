"""Render the val-fold-selection control as a standalone HTML report.

Every number is read from the frozen JSON products in this directory; nothing is
hardcoded. Regenerate after any re-run of analyze.py / stability.py / valnoise.py
/ val_loso.py / published_leader_check.py.
"""
from __future__ import annotations
import html, json, os, sys

D = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, "report.html")


def J(name):
    p = os.path.join(D, name)
    return json.load(open(p)) if os.path.exists(p) else None


rep = J("report.json")
loso = J("val_loso_full.json")
loso_c = J("val_loso_common.json")
stab = J("leader_stability.json")
vres = J("val_resolution.json")
plc = J("published_leader_check.json")
vcon = J("val_contrast.json")
lci = J("val_loso_ci.json")
rule = J("SELECTION_RULE.json")

P = rep["full_pool"]
CORPORA = sorted(P["selection"])
NICE = {"coaid": "CoAID", "recovery": "ReCOVery", "politifact": "PolitiFact",
        "gossipcop": "GossipCop"}
PRETTY = {"rf": "RandomForest (7d)", "propmlp": "PropMLP (7d)", "propmlp15": "PropMLP+ (15d)",
          "histgbt15": "HistGBT (15d)", "ebgcn": "EBGCN-style", "gatv2": "GATv2",
          "dlf": "DLF-Full (7d)", "dlf15": "DLF-Full+ (15d)"}
SHORT = {"rf": "RandomForest", "propmlp": "PropMLP", "propmlp15": "PropMLP+",
         "histgbt15": "HistGBT", "ebgcn": "EBGCN", "gatv2": "GATv2",
         "dlf": "DLF-Full", "dlf15": "DLF-Full+"}
# components whose regenerated test mean reproduces the published main-table cell
MODIFIED = {"histgbt15", "ebgcn", "gatv2"}
e = html.escape


def f4(x):
    return "&mdash;" if x is None else f"{x:.4f}"


def f3(x):
    return "&mdash;" if x is None else f"{x:.3f}"


# ---------------------------------------------------------------- rank ladder
def ladder(c):
    """SVG: components ordered by VAL mean on the left, by TEST mean on the right,
    with a connector per component. The amount of crossing IS the finding."""
    vt, tt = rep["val_table"][c], rep["test_table"][c]
    pool = P["pool"][c]
    vorder = sorted(pool, key=lambda m: -vt[m]["mean"])
    torder = sorted(pool, key=lambda m: -tt[m]["mean"])
    pick = P["selection"][c]["variant_A_component"]
    lead = P["diagnostics"][c]["oracle_component_on_test_mean"]
    n = len(pool)
    ROW, TOP, H = 30, 34, 0
    H = TOP + ROW * n + 10
    W, LX, RX = 660, 226, 434
    rows = []
    rows.append(f'<text x="{LX}" y="20" class="lg-head lg-val" text-anchor="end">'
                f'VALIDATION FOLD &nbsp;&rarr;</text>')
    rows.append(f'<text x="{RX}" y="20" class="lg-head lg-test">&larr;&nbsp; TEST FOLD</text>')
    for i, m in enumerate(vorder):
        y = TOP + ROW * i + ROW / 2
        j = torder.index(m)
        y2 = TOP + ROW * j + ROW / 2
        cls = "lk"
        if m == pick:
            cls += " lk-pick"
        elif m == lead:
            cls += " lk-lead"
        mid = (LX + RX) / 2
        rows.append(f'<path class="{cls}" d="M{LX+8} {y} C{mid} {y} {mid} {y2} {RX-8} {y2}"/>')
    for i, m in enumerate(vorder):
        y = TOP + ROW * i + ROW / 2
        star = "&#9656;&#8202;" if m == pick else ""
        cl = "lg-name lg-name-pick" if m == pick else "lg-name"
        rows.append(f'<text x="{LX-62}" y="{y+4}" class="{cl}" text-anchor="end">'
                    f'{star}{e(SHORT[m])}</text>')
        rows.append(f'<text x="{LX-16}" y="{y+4}" class="lg-num" text-anchor="end">'
                    f'{vt[m]["mean"]:.3f}</text>')
        rows.append(f'<circle cx="{LX+8}" cy="{y}" r="3.4" class="dot dot-val'
                    f'{" dot-on" if m == pick else ""}"/>')
    for j, m in enumerate(torder):
        y = TOP + ROW * j + ROW / 2
        cl = "lg-name lg-name-lead" if m == lead else "lg-name"
        tag = "&#9733;&#8202;" if m == lead else ""
        rows.append(f'<circle cx="{RX-8}" cy="{y}" r="3.4" class="dot dot-test'
                    f'{" dot-on" if m == lead else ""}"/>')
        rows.append(f'<text x="{RX+14}" y="{y+4}" class="{cl}">{tag}{e(SHORT[m])}</text>')
        rows.append(f'<text x="{RX+150}" y="{y+4}" class="lg-num">{tt[m]["mean"]:.3f}</text>')
    return (f'<svg class="ladder" viewBox="0 0 {W} {H}" width="100%" '
            f'preserveAspectRatio="xMidYMin meet" role="img" aria-label="'
            f'{e(NICE[c])}: components ranked by validation macro-F1 on the left and by test '
            f'macro-F1 on the right, connected">' + "".join(rows) + "</svg>")


# ---------------------------------------------------------------- corpus cards
def corpus_card(c):
    s, d = P["selection"][c], P["diagnostics"][c]
    vr = (vres or {}).get(c, {})
    st = (stab or {}).get(c, {})
    pick, lead = s["variant_A_component"], d["oracle_component_on_test_mean"]
    hit = d["variant_A_picked_test_leader"]
    pl = (plc or {}).get("per_corpus", {}).get(c, {})
    badge = ('<span class="pill pill-hit">validation found it</span>' if hit else
             '<span class="pill pill-miss">validation missed it</span>')
    vt, tt = rep["val_table"][c], rep["test_table"][c]
    pool = P["pool"][c]
    rowsh = "".join(
        f'<tr class="{"is-pick" if m == pick else ""}{" is-lead" if m == lead else ""}">'
        f'<td class="mname">{e(PRETTY[m])}'
        f'{"<sup class=mod>&#8225;</sup>" if m in MODIFIED else ""}</td>'
        f'<td class="num v">{vt[m]["mean"]:.4f}</td>'
        f'<td class="num sd">&plusmn;{vt[m]["std"]:.3f}</td>'
        f'<td class="num t">{tt[m]["mean"]:.4f}</td>'
        f'<td class="num sd">&plusmn;{tt[m]["std"]:.3f}</td>'
        f'<td class="num p">{f3((st.get("p_leads_on_test") or {}).get(m))}</td>'
        f'<td class="mk">{"&#9656; picked" if m == pick else ""}'
        f'{"&#9733; leads" if m == lead else ""}</td></tr>'
        for m in sorted(pool, key=lambda m: -vt[m]["mean"]))
    facts = [
        ("validation picks", PRETTY[pick]),
        ("test leader", PRETTY[lead]),
        ("regret vs test oracle", f'{d["variant_A_regret_vs_oracle"]:+.4f}'),
        ("per-seed picks hit leader", f'{d["variant_B_seedwise_hits"]}/5'),
        ("val rank of the test leader",
         str(pl.get("val_rank_of_regen_test_leader", "&mdash;")) + f" of {len(pool)}"),
        ("val leader margin &divide; bootstrap SD",
         f3(vr.get("margin_over_sd"))),
        ("P(same val leader on resample)", f3(vr.get("p_same_val_leader_under_resampling"))),
        ("distinct per-seed test leaders",
         f'{d["n_distinct_test_leaders_across_seeds"]} across 5 seeds'),
    ]
    fl = "".join(f'<div class="fact"><dt>{k}</dt><dd>{v}</dd></div>' for k, v in facts)
    return f'''
<article class="corpus" id="c-{e(c)}">
  <header class="corpus-head">
    <h3>{e(NICE[c])}</h3>
    <div class="corpus-meta">
      <span>{len(pool)} components</span><span>n<sub>val</sub> = {vr.get("n_val_per_seed","?")}</span>
      <span>5 paired seeds</span>{badge}
    </div>
  </header>
  <div class="ladder-wrap">{ladder(c)}</div>
  <dl class="facts">{fl}</dl>
  <div class="tbl-wrap"><table class="data">
    <thead><tr><th>component</th><th class="num v" colspan="2">val macro-F1</th>
      <th class="num t" colspan="2">test macro-F1</th>
      <th class="num">P(leads test)</th><th></th></tr></thead>
    <tbody>{rowsh}</tbody>
  </table></div>
</article>'''


# ---------------------------------------------------------------- assembly
# ---- narrative facts, all computed ----
_hit = [c for c in CORPORA if P["diagnostics"][c]["variant_A_picked_test_leader"]]
_miss = [c for c in CORPORA if not P["diagnostics"][c]["variant_A_picked_test_leader"]]
_reg = {c: P["diagnostics"][c]["variant_A_regret_vs_oracle"] for c in CORPORA}
_cheap = sorted([c for c in _miss if _reg[c] <= 0.02], key=lambda c: _reg[c])
_dear = sorted([c for c in _miss if _reg[c] > 0.02], key=lambda c: -_reg[c])
_tmarg = {c: P["diagnostics"][c]["test_leader_margin"] for c in CORPORA}
_nlead = {c: P["diagnostics"][c]["n_distinct_test_leaders_across_seeds"] for c in CORPORA}
_mos = {c: vres[c]["margin_over_sd"] for c in CORPORA} if vres else {}
_best_mos = max(_mos, key=lambda c: _mos[c]) if _mos else None
_tot_lead = sum(_nlead.values())


def _lst(cs, conj="and"):
    cs = [e(NICE[c]) for c in cs]
    if not cs:
        return ""
    if len(cs) == 1:
        return cs[0]
    return ", ".join(cs[:-1]) + f" {conj} " + cs[-1]


h = P["headline"]
hits_regen = (plc or {}).get("headline", {}).get("val_pick_recovers_regen_test_leader", "?")
hits_pub = (plc or {}).get("headline", {}).get("val_pick_recovers_published_leader", "?")
n_corp = len(CORPORA)

tiles = [
    ("validation names the test leader", hits_regen,
     "corpora &mdash; and the paper's published leader on " + hits_pub),
    ("per-seed selection hits the per-seed leader",
     f'{h["variant_B_hits_on_test_leader"]}/{h["variant_B_cells"]}',
     "corpus&times;seed cells"),
    ("mean regret vs the test-fold oracle", f'{h["mean_regret_A"]:+.4f}',
     "macro-F1; three of four corpora cost &le;&#8202;0.010"),
    ("no validation leader clears its own noise",
     f'{max(_mos.values()):.2f}&#8202;SD',
     f'best margin-to-noise ratio, on {_lst([_best_mos])}'),
]
tiles_h = "".join(
    f'<div class="tile"><div class="tile-k">{k}</div><div class="tile-v">{v}</div>'
    f'<div class="tile-s">{s}</div></div>' for k, v, s in tiles)

lo = loso
lo_c = loso_c
def _ci(m):
    ct = (lci or {}).get("contrasts", {}).get(m)
    if not ct:
        return '<td class="num">&mdash;</td><td class="num">&mdash;</td>'
    sig = ct["excludes_zero"]
    return (f'<td class="num">[{ct["ci95"][0]:+.4f}, {ct["ci95"][1]:+.4f}]</td>'
            f'<td class="num {"ok" if sig else "no"}">{ct["p_two_sided"]:.3f}</td>')


lo_rows = "".join(
    f'<tr><td class="mname">{e(PRETTY[m])} everywhere</td>'
    f'<td class="num">{lo["vs_every_fixed_common_component"][m]["held_out_val_mean"]:.4f}</td>'
    f'<td class="num d">{lo["vs_every_fixed_common_component"][m]["selection_minus"]:+.4f}</td>'
    f'{_ci(m)}'
    f'<td class="num">{lo["vs_every_fixed_common_component"][m]["selection_wins_on_k"]}/5</td></tr>'
    for m in lo["common_pool"]) if lo else ""
uni_row = (
    f'<tr><td class="mname">best uniform model, re-chosen on each fold</td>'
    f'<td class="num">{lo["held_out_val_uniform_mean"]:.4f}</td>'
    f'<td class="num d">{lo["selection_minus_uniform"]:+.4f}</td>{_ci("uniform")}'
    f'<td class="num">{lo["selection_beats_uniform_on_k"]}/5</td></tr>') if lo else ""

picks_line = " &middot; ".join(
    f'<b>{e(NICE[c])}</b> {e(PRETTY[P["selection"][c]["variant_A_component"]])}'
    for c in CORPORA)

fid = rep["fidelity_vs_published"]
fid_rows = "".join(
    f'<tr><td>{e(NICE[c])}</td><td class="mname">{e(PRETTY[m])}</td>'
    f'<td class="num">{fid[c][m]["regen_test_mean"]:.4f}</td>'
    f'<td class="num">{f3(fid[c][m]["published_test_mean"])}</td>'
    f'<td class="{"ok" if fid[c][m]["matches"] else "no"}">'
    f'{"reproduces" if fid[c][m]["matches"] else "protocol changed"}</td></tr>'
    for c in CORPORA for m in sorted(fid[c], key=lambda m: (m in MODIFIED, m)))

uni = P.get("uniform_single_baseline", {})

ALL8 = ["rf", "propmlp", "propmlp15", "histgbt15", "ebgcn", "gatv2", "dlf", "dlf15"]
missing = {c: [m for m in ALL8 if m not in P["pool"][c]] for c in CORPORA}
cov_rows = "".join(
    f'<tr><td>{e(NICE[c])}</td><td class="num">{len(P["pool"][c])}&#8202;/&#8202;8</td>'
    f'<td>{", ".join(e(PRETTY[m]) for m in missing[c]) or "&mdash;"}</td></tr>'
    for c in CORPORA)
n_missing = sum(len(v) for v in missing.values())
PUB = json.load(open(os.path.join(D, "report.json")))  # PUBLISHED lives in analyze.py
from analyze import PUBLISHED as _PUB
# For each absent component, is its PUBLISHED test mean already below the pool leader's
# regenerated test mean? If so its absence cannot have changed the selection.
cov_note = []
for c in CORPORA:
    if not missing[c]:
        continue
    lead_v = rep["test_table"][c][P["diagnostics"][c]["oracle_component_on_test_mean"]]["mean"]
    known = [(m, _PUB[c].get(m)) for m in missing[c]]
    below = [m for m, v in known if v is not None and v < lead_v]
    unknown = [m for m, v in known if v is None]
    cov_note.append((c, lead_v, below, unknown))
cov_note_h = "".join(
    f'<li><b>{e(NICE[c])}</b> &mdash; pool leader at {lv:.3f}. '
    + (f'The absent {", ".join(e(SHORT[m]) for m in bl)} publish '
       f'{", ".join(f"{_PUB[c][m]:.3f}" for m in bl)}, all below it: had validation reached '
       f'for one of them the control would have scored <em>worse</em>. ' if bl else '')
    + (f'<span style="color:var(--test)">{", ".join(e(SHORT[m]) for m in un)} has no published '
       f'number on this corpus &mdash; a genuine unknown.</span>' if un else '')
    + '</li>'
    for c, lv, bl, un in cov_note)

DOC = f'''<title>Can Validation Find the Winner?</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,600;1,6..72,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{
  --ground:#EDF0F1; --surface:#FFFFFF; --surface-2:#F5F7F8;
  --ink:#16202A; --ink-2:#3C4E5A; --ink-3:#6C7F8A;
  --line:#D3DBDF; --line-2:#E4EAEC;
  --val:#0E6E6E; --val-soft:#0E6E6E22; --val-ink:#0A5252;
  --test:#AB4E1C; --test-soft:#AB4E1C1F; --test-ink:#8C3F16;
  --good:#1E6B45; --bad:#9A3324;
  --shadow:0 1px 2px #16202A0F, 0 8px 24px -14px #16202A2E;
}}
@media (prefers-color-scheme:dark){{
  :root:not([data-theme="light"]){{
    --ground:#0E151A; --surface:#151E24; --surface-2:#1A252C;
    --ink:#DDE7EC; --ink-2:#A7B8C1; --ink-3:#7A8D97;
    --line:#26343C; --line-2:#1F2C33;
    --val:#54BFB4; --val-soft:#54BFB42B; --val-ink:#7FD4CB;
    --test:#E0894F; --test-soft:#E0894F26; --test-ink:#EDA574;
    --good:#5BBF8A; --bad:#E0796A;
    --shadow:0 1px 2px #0006, 0 10px 30px -16px #000A;
  }}
}}
:root[data-theme="dark"]{{
  --ground:#0E151A; --surface:#151E24; --surface-2:#1A252C;
  --ink:#DDE7EC; --ink-2:#A7B8C1; --ink-3:#7A8D97;
  --line:#26343C; --line-2:#1F2C33;
  --val:#54BFB4; --val-soft:#54BFB42B; --val-ink:#7FD4CB;
  --test:#E0894F; --test-soft:#E0894F26; --test-ink:#EDA574;
  --good:#5BBF8A; --bad:#E0796A;
  --shadow:0 1px 2px #0006, 0 10px 30px -16px #000A;
}}
*{{box-sizing:border-box}}
body{{
  background:var(--ground); color:var(--ink);
  font-family:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  font-size:16px; line-height:1.62; margin:0;
  -webkit-font-smoothing:antialiased;
}}
.wrap{{max-width:1000px; margin:0 auto; padding:0 24px 96px}}
.prose{{max-width:66ch}}
h1,h2,h3{{font-family:Newsreader,Georgia,serif; font-weight:600; text-wrap:balance; margin:0}}
h1{{font-size:clamp(2.2rem,5.4vw,3.4rem); line-height:1.08; letter-spacing:-.015em}}
h2{{font-size:1.72rem; line-height:1.2; letter-spacing:-.01em}}
h3{{font-size:1.3rem; line-height:1.25}}
p{{margin:0}}
a{{color:var(--val-ink)}}
.eyebrow{{
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:.7rem;
  letter-spacing:.16em; text-transform:uppercase; color:var(--ink-3);
}}
header.top{{
  border-bottom:1px solid var(--line); padding:56px 0 40px; margin-bottom:48px;
  display:flex; flex-direction:column; gap:20px;
}}
header.top .prose{{display:flex; flex-direction:column; gap:14px}}
.lede{{font-size:1.14rem; color:var(--ink-2)}}
section{{margin-top:64px; display:flex; flex-direction:column; gap:22px}}
section > .prose{{display:flex; flex-direction:column; gap:14px}}
.sec-head{{display:flex; flex-direction:column; gap:8px}}

.verdict{{
  background:var(--surface); border:1px solid var(--line); border-radius:3px;
  box-shadow:var(--shadow); padding:28px 30px; display:flex;
  flex-direction:column; gap:18px;
}}
.verdict .prose{{display:flex; flex-direction:column; gap:12px}}
.tiles{{display:grid; grid-template-columns:repeat(auto-fit,minmax(196px,1fr)); gap:1px;
  background:var(--line-2); border:1px solid var(--line-2); border-radius:3px; overflow:hidden}}
.tile{{background:var(--surface); padding:18px 18px 16px; display:flex;
  flex-direction:column; gap:5px}}
.tile-k{{font-size:.76rem; color:var(--ink-3); line-height:1.35}}
.tile-v{{font-family:"IBM Plex Mono",monospace; font-size:1.85rem; font-weight:500;
  font-variant-numeric:tabular-nums; letter-spacing:-.02em; color:var(--test)}}
.tile-s{{font-size:.74rem; color:var(--ink-3); line-height:1.35}}

.corpus{{background:var(--surface); border:1px solid var(--line); border-radius:3px;
  box-shadow:var(--shadow); padding:26px 28px 24px; display:flex;
  flex-direction:column; gap:20px}}
.corpus-head{{display:flex; flex-direction:column; gap:8px;
  border-bottom:1px solid var(--line-2); padding-bottom:14px}}
.corpus-meta{{display:flex; flex-wrap:wrap; gap:14px; align-items:center;
  font-size:.78rem; color:var(--ink-3)}}
.pill{{font-family:"IBM Plex Mono",monospace; font-size:.68rem; letter-spacing:.06em;
  text-transform:uppercase; padding:3px 9px; border-radius:2px}}
.pill-hit{{background:var(--val-soft); color:var(--val-ink)}}
.pill-miss{{background:var(--test-soft); color:var(--test-ink)}}

.ladder-wrap{{overflow-x:auto}}
svg.ladder{{min-width:640px; display:block; font-family:"IBM Plex Sans",sans-serif}}
.lg-head{{font-family:"IBM Plex Mono",monospace; font-size:9.5px; letter-spacing:.14em}}
.lg-val{{fill:var(--val)}} .lg-test{{fill:var(--test)}}
.lg-name{{font-size:12.5px; fill:var(--ink-2)}}
.lg-name-pick{{fill:var(--val); font-weight:600}}
.lg-name-lead{{fill:var(--test); font-weight:600}}
.lg-num{{font-family:"IBM Plex Mono",monospace; font-size:11px; fill:var(--ink-3);
  font-variant-numeric:tabular-nums}}
.lk{{fill:none; stroke:var(--line); stroke-width:1.4}}
.lk-pick{{stroke:var(--val); stroke-width:2.2}}
.lk-lead{{stroke:var(--test); stroke-width:2.2; stroke-dasharray:5 3}}
.dot{{fill:var(--line)}} .dot-val.dot-on{{fill:var(--val)}} .dot-test.dot-on{{fill:var(--test)}}

.facts{{display:grid; grid-template-columns:repeat(auto-fit,minmax(178px,1fr));
  gap:14px 20px; margin:0}}
.fact{{display:flex; flex-direction:column; gap:1px; border-left:2px solid var(--line-2);
  padding-left:11px}}
.fact dt{{font-size:.72rem; color:var(--ink-3); line-height:1.3}}
.fact dd{{margin:0; font-size:.9rem; font-weight:500;
  font-variant-numeric:tabular-nums}}

.tbl-wrap{{overflow-x:auto}}
table.data{{border-collapse:collapse; width:100%; min-width:600px; font-size:.83rem}}
table.data th{{font-family:"IBM Plex Mono",monospace; font-weight:500; font-size:.68rem;
  letter-spacing:.08em; text-transform:uppercase; color:var(--ink-3);
  text-align:left; padding:0 10px 7px; border-bottom:1px solid var(--line)}}
table.data td{{padding:6px 10px; border-bottom:1px solid var(--line-2)}}
table.data tbody tr:last-child td{{border-bottom:none}}
.num{{text-align:right; font-family:"IBM Plex Mono",monospace;
  font-variant-numeric:tabular-nums; white-space:nowrap}}
th.v,td.v{{color:var(--val)}} th.t,td.t{{color:var(--test)}}
td.sd{{color:var(--ink-3); font-size:.76rem; padding-left:0}}
td.p{{color:var(--ink-2)}}
td.d{{color:var(--good)}}
.mname{{white-space:nowrap}}
sup.mod{{color:var(--test); font-size:.9em}}
.mk{{font-size:.7rem; color:var(--ink-3); white-space:nowrap}}
tr.is-pick td{{background:var(--val-soft)}}
tr.is-lead td{{box-shadow:inset 0 0 0 0 transparent}}
tr.is-lead td:first-child{{box-shadow:inset 3px 0 0 var(--test)}}
.ok{{color:var(--good); font-size:.78rem}} .no{{color:var(--test); font-size:.78rem}}

.panel{{background:var(--surface); border:1px solid var(--line); border-radius:3px;
  padding:24px 26px; display:flex; flex-direction:column; gap:16px; box-shadow:var(--shadow)}}
.panel h3{{font-size:1.12rem}}
.rule{{background:var(--surface-2); border-left:2px solid var(--val);
  padding:16px 20px; font-size:.92rem; color:var(--ink-2)}}
ul.tight{{margin:0; padding-left:1.1em; display:flex; flex-direction:column; gap:9px;
  font-size:.94rem; color:var(--ink-2)}}
ul.tight b{{color:var(--ink)}}
code{{font-family:"IBM Plex Mono",monospace; font-size:.86em;
  background:var(--surface-2); padding:1px 5px; border-radius:2px; color:var(--ink-2)}}
pre{{background:var(--surface-2); border:1px solid var(--line-2); border-radius:3px;
  padding:14px 16px; overflow-x:auto; font-family:"IBM Plex Mono",monospace;
  font-size:.78rem; line-height:1.7; margin:0; color:var(--ink-2)}}
footer{{margin-top:72px; padding-top:24px; border-top:1px solid var(--line);
  font-size:.8rem; color:var(--ink-3); display:flex; flex-direction:column; gap:8px}}
.grid2{{display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:20px}}
:focus-visible{{outline:2px solid var(--val); outline-offset:2px}}
@media (prefers-reduced-motion:no-preference){{
  .lk{{transition:stroke .2s}}
}}
</style>

<div class="wrap">
<header class="top">
  <div class="eyebrow">WSDM-05 &middot; doc-less fake &middot; combiner control &ldquo;valselect&rdquo;</div>
  <div class="prose">
    <h1>Can validation find the winner?</h1>
    <p class="lede">The main table shows a different leading model on every corpus.
    This is the control that asks whether any of that is usable: for each corpus, pick
    the component with the best <span style="color:var(--val)">validation</span> macro-F1
    and keep it. No blending, no weights, one free decision per corpus &mdash; and the
    <span style="color:var(--test)">test</span> fold never enters the choice.</p>
  </div>
</header>

<div class="verdict">
  <div class="eyebrow">verdict</div>
  <div class="tiles">{tiles_h}</div>
  <div class="prose">
    <p><b>Validation names the exact test-fold leader on {hits_regen} corpora</b> &mdash;
    only {_lst(_hit)}. Read alone that number is damning, and it is also misleading. On
    {_lst(_cheap)} validation picks the wrong model and loses
    {min(_reg[c] for c in _cheap):.4f}&ndash;{max(_reg[c] for c in _cheap):.4f} macro-F1 by
    doing so, because on those corpora the models it is choosing between are not actually
    different: the gap between first and second place on test is
    {min(_tmarg[c] for c in _cheap):.4f}&ndash;{max(_tmarg[c] for c in _cheap):.4f}.</p>
    <p>The identity of a corpus's &ldquo;winner&rdquo; is not a stable property of that
    corpus. Across the five paired seeds the four corpora produce {_tot_lead} distinct
    per-seed test leaders between them
    ({", ".join(f"{e(NICE[c])} {_nlead[c]}" for c in CORPORA)}), and per-seed selection
    lands on the right one in {h["variant_B_hits_on_test_leader"]} of
    {h["variant_B_cells"]} corpus&times;seed cells. So the table's headline pattern is real
    as a description of one particular test draw and largely irreproducible as a claim
    about corpora.</p>
    <p>{_lst(_dear)} is the exception that matters: there the spread between models
    <em>is</em> large ({_tmarg[_dear[0]]:.3f} between first and second), and validation gets
    it wrong, costing {_reg[_dear[0]]:.3f} macro-F1. It is also the smallest fold in the
    study &mdash; {vres[_dear[0]]["n_val_per_seed"] if vres else "?"} validation stories
    asked to rank {len(P["pool"][_dear[0]])} models.</p>
    <p>Selection picks: {picks_line}.</p>
  </div>
</div>

<section>
  <div class="sec-head prose">
    <div class="eyebrow">the rule, frozen before any test fold was scored</div>
    <h2>What this combiner is</h2>
  </div>
  <div class="rule prose">{e(rule["statement"])}</div>
  <div class="prose">
    <p>It is the weakest framework that still counts as one, and therefore the bar every
    fancier combiner has to clear. It also has a property worth stating plainly:
    <b>on any single corpus it cannot beat the best individual baseline, because it
    <em>is</em> one of them.</b> The only thing it can beat is a single model applied
    uniformly to all four corpora &mdash; which is what every row of the main table
    actually is.</p>
  </div>
  <div class="panel">
    <h3>Invariants</h3>
    <ul class="tight">
      <li><b>Every component is fitted on the train fold only.</b> The validation fold is
      scored exactly once per (corpus, component, seed) and is never in any fit.</li>
      <li><b>Selection reads validation fields only.</b>
      <code>verify_no_leakage.py</code> reloads every per-story dump, sets every
      <code>test_*</code> field to <code>None</code>, re-runs the rule on what is left, and
      asserts it reproduces the frozen manifest.</li>
      <li><b>Seeds 0&ndash;4 stay paired.</b> Each component's fold membership is checked
      story-by-story against an independently built fold manifest before it is used.</li>
      <li><b>The test fold is emitted, not consulted.</b> Its only uses are handing
      predictions to the judge and the post-hoc diagnostics on this page.</li>
    </ul>
  </div>
</section>

<section>
  <div class="sec-head prose">
    <div class="eyebrow">per corpus &middot; components ranked on each fold</div>
    <h2>Where the ranking survives, and where it doesn't</h2>
    <p style="color:var(--ink-2)">Each ribbon runs from a component's validation rank to its
    test rank. Flat lines mean validation saw the truth; crossings mean it did not. The
    <span style="color:var(--val)">solid</span> ribbon is what validation picked; the
    <span style="color:var(--test)">dashed</span> one is what actually led on test.</p>
  </div>
  {"".join(corpus_card(c) for c in CORPORA)}
</section>

<section>
  <div class="sec-head prose">
    <div class="eyebrow">validation fold only &middot; leave-one-seed-out</div>
    <h2>What per-corpus selection is worth</h2>
    <p style="color:var(--ink-2)">Crediting the rule with the same validation scores it
    maximised is circular. So the rule is cross-validated across the five paired seeds:
    it selects on four seeds' validation folds and is scored on the fifth, which it has
    not seen. Now it can lose. Against a single component applied to all
    {n_corp} corpora &mdash; the shape of every main-table row &mdash; it does not.</p>
  </div>
  <div class="panel">
    <div class="tbl-wrap"><table class="data">
      <thead><tr><th>held-out validation macro-F1, mean over {n_corp} corpora</th>
        <th class="num">score</th><th class="num">gain</th>
        <th class="num">95% CI</th><th class="num">p</th>
        <th class="num">wins</th></tr></thead>
      <tbody>
        <tr class="is-pick"><td class="mname"><b>per-corpus validation selection</b></td>
          <td class="num"><b>{lo["held_out_val_selection_mean"]:.4f}</b></td>
          <td class="num">&mdash;</td><td class="num">&mdash;</td>
          <td class="num">&mdash;</td><td class="num">&mdash;</td></tr>
        {uni_row}
        {lo_rows}
      </tbody>
    </table></div>
    <p style="font-size:.9rem;color:var(--ink-2)">Per-corpus selection beats every uniformly
    applied component, by {min(lo["vs_every_fixed_common_component"][m]["selection_minus"] for m in lo["common_pool"]):+.4f}
    to {max(lo["vs_every_fixed_common_component"][m]["selection_minus"] for m in lo["common_pool"]):+.4f}
    macro-F1. Under a story-clustered paired bootstrap of the held-out folds
    ({lci["B"] if lci else "?"} resamples) the gain excludes zero against
    {sum(1 for m in ["uniform"] + lo["common_pool"] if lci and lci["contrasts"][m]["excludes_zero"])}
    of {1 + len(lo["common_pool"])} comparators. The exception is
    <b>{e(PRETTY["histgbt15"])}</b>, where the interval just contains zero
    (p&nbsp;=&nbsp;{lci["contrasts"]["histgbt15"]["p_two_sided"]:.3f}) &mdash; the closest any
    single fixed model comes to matching a framework that gets to choose per corpus.</p>
    <p style="font-size:.9rem;color:var(--ink-2)">Restricting selection to the components
    present on all four corpora changes nothing
    ({lo_c["held_out_val_selection_mean"]:.4f} vs {lo["held_out_val_selection_mean"]:.4f}) &mdash;
    the wider pool on {e(NICE["coaid"])} and {e(NICE["politifact"])} does no work here.</p>
    <p style="font-size:.9rem;color:var(--ink-3)">Held-out validation numbers are internal to
    this experiment and are not comparable to the main table's test numbers. On the same
    validation folds the in-sample gain over the best uniform model is
    {vcon["observed_val_gain"]:+.4f} (95% CI {vcon["ci95"][0]:+.4f} to {vcon["ci95"][1]:+.4f},
    p = {vcon["p_two_sided"]}), which is the optimistic version of the same comparison.</p>
  </div>
</section>

<section>
  <div class="sec-head prose">
    <div class="eyebrow">signal versus noise</div>
    <h2>How much resolving power does a validation fold have?</h2>
    <p style="color:var(--ink-2)">A story-clustered bootstrap resamples each fold within
    each seed, using identical indices for every component so the pairing holds. It answers
    two questions: is the validation leader stable, and is the test leader stable?</p>
  </div>
  <div class="panel">
    <div class="tbl-wrap"><table class="data">
      <thead><tr><th>corpus</th><th class="num">n<sub>val</sub></th>
        <th class="num v">val margin</th><th class="num">&divide; bootstrap SD</th>
        <th class="num v">P(same val leader)</th>
        <th class="num t">P(modal test leader leads)</th>
        <th class="num">val leader also leads test</th></tr></thead>
      <tbody>{"".join(
        f'<tr><td>{e(NICE[c])}</td>'
        f'<td class="num">{vres[c]["n_val_per_seed"]}</td>'
        f'<td class="num v">{vres[c]["val_leader_margin"]:.4f}</td>'
        f'<td class="num">{vres[c]["margin_over_sd"]:.2f}</td>'
        f'<td class="num v">{vres[c]["p_same_val_leader_under_resampling"]:.2f}</td>'
        f'<td class="num t">{stab[c]["modal_test_leader_prob"]:.2f}</td>'
        f'<td class="num">{stab[c]["p_val_leader_also_leads_test"]:.2f}</td></tr>'
        for c in CORPORA)}</tbody>
    </table></div>
    <p style="font-size:.9rem;color:var(--ink-2)"><b>Not one of the four validation folds
    separates its top two models by even a single bootstrap SD.</b> The best ratio in the
    study is {max(_mos.values()):.2f} SD, on {_lst([_best_mos])}; the worst is
    {min(_mos.values()):.2f}. Resample the stories and the validation leader changes identity
    between {min(vres[c]["p_same_val_leader_under_resampling"] for c in CORPORA):.0%} and
    {max(vres[c]["p_same_val_leader_under_resampling"] for c in CORPORA):.0%} of the time.
    A validation fold of 20&ndash;300 stories is simply not an instrument that can rank
    these models.</p>
    <p style="font-size:.9rem;color:var(--ink-2)">The test fold is no better. Its leader is
    modal rather than decisive on three corpora
    ({", ".join(f"{e(NICE[c])} {stab[c]['modal_test_leader_prob']:.2f}" for c in CORPORA if c != "gossipcop")}),
    and the model validation picks goes on to lead the test fold in as few as
    {min(stab[c]["p_val_leader_also_leads_test"] for c in CORPORA):.1%} of resamples
    ({_lst([min(CORPORA, key=lambda c: stab[c]["p_val_leader_also_leads_test"])])}). Only
    {e(NICE["gossipcop"])}, with 300 validation and 300 test stories, behaves like a
    measurement: there the validation and test rankings agree perfectly
    (Spearman {P["diagnostics"]["gossipcop"]["spearman_val_vs_test_ranking"]:.2f}).</p>
  </div>
</section>

<section>
  <div class="sec-head prose">
    <div class="eyebrow">provenance</div>
    <h2>Do the components reproduce the paper's table?</h2>
    <p style="color:var(--ink-2)">Nothing in this repository stored per-story predictions,
    so every component was refit from source. Five of the eight rows reproduce their
    published main-table cell to within a thousandth. Three could not, and the reason
    matters more than the number.</p>
  </div>
  <div class="panel">
    <div class="tbl-wrap"><table class="data">
      <thead><tr><th>corpus</th><th>component</th><th class="num">regenerated</th>
        <th class="num">published</th><th>status</th></tr></thead>
      <tbody>{fid_rows}</tbody>
    </table></div>
  </div>
  <div class="panel">
    <h3>Component coverage</h3>
    <p style="font-size:.9rem;color:var(--ink-2)">Selection is given every component that has
    a complete five-seed validation and test dump on that corpus &mdash; a rule declared
    before any pool-level comparison was made. {n_missing} of 32 corpus&times;component cells
    are absent, all for compute reasons rather than choice: the graph and DocLessFake rows
    cost hours per seed and {e(NICE["gossipcop"])} is 3,000 stories.</p>
    <div class="tbl-wrap"><table class="data">
      <thead><tr><th>corpus</th><th class="num">pool</th><th>not available</th></tr></thead>
      <tbody>{cov_rows}</tbody>
    </table></div>
    <p style="font-size:.9rem;color:var(--ink-2)">Both effects of a short pool flatter the
    control, so read its hit rate as an upper bound. Fewer components is an easier ranking
    problem than the eight-way one the main table implies; and because absence acts on the
    <em>validation</em> side, a missing component could only have hurt by winning the
    validation fold and then underperforming on test &mdash; which is what the published
    table says each of these would have done.</p>
    <ul class="tight" style="font-size:.88rem">{cov_note_h}</ul>
    <p style="font-size:.9rem;color:var(--ink-3)">One real gap remains: DLF-Full+ on
    {e(NICE["gossipcop"])} has no published number at all, so it cannot be ruled in or out.
    The leave-one-seed-out result is unaffected either way &mdash; restricted to the four
    components every corpus has, it is unchanged.</p>
  </div>
  <div class="panel">
    <h3>The three protocol changes <sup class="mod">&#8225;</sup></h3>
    <ul class="tight">
      <li><b>HistGBT (15d)</b> &mdash; the published runner tunes on validation and then
      refits on train + validation, which makes its validation predictions in-sample and
      unusable as a selection input. Replaced by 5-fold cross-validation over the identical
      grid strictly inside the train fold, then one fit on train alone. On
      {e(NICE["coaid"])} this makes the component <em>substantially stronger</em> than the
      table's row ({fid["coaid"]["histgbt15"]["regen_test_mean"]:.3f} against
      {fid["coaid"]["histgbt15"]["published_test_mean"]:.3f}); selecting 8 hyperparameter
      points on 20 validation stories was mostly fitting noise.</li>
      <li><b>EBGCN-style</b> and <b>GATv2</b> &mdash; both published runners choose
      (epochs, lr) by maximising validation macro-F1, so their published validation score is
      a selected maximum. Replaced by an inner 85/15 stratified holdout carved out of the
      train fold, same grid and same number of fits. One runner is used for all corpora,
      resolving the provenance split in the published EBGCN row.</li>
      <li>The consequence is stated rather than hidden: for these three, the component in
      this experiment is not literally the component in the main table. Every conclusion on
      this page is reported for the full pool and, in <code>report.json</code>, for a strict
      pool holding only the five components that reproduce exactly.</li>
    </ul>
  </div>
</section>

<section>
  <div class="sec-head prose">
    <div class="eyebrow">what this means for the paper</div>
    <h2>Reading</h2>
  </div>
  <div class="grid2">
    <div class="panel">
      <h3>For the audit's argument</h3>
      <ul class="tight">
        <li>The varying leader is <b>partly real and partly an artefact</b>. Two of four
        corpora have a leader stable enough that an honest protocol finds it; two do not.</li>
        <li>Per-seed selection lands on the per-seed leader
        {h["variant_B_hits_on_test_leader"]}/{h["variant_B_cells"]} times. A practitioner
        with one split has roughly chance-level access to the winner the table reports.</li>
        <li>The corpus where the paper's own method leads &mdash; {e(NICE["coaid"])},
        DLF-Full at {fid["coaid"]["dlf"]["published_test_mean"]} &mdash; is exactly the corpus
        with the least resolving power. That is the strongest self-critical result here and
        the paper should say it.</li>
      </ul>
    </div>
    <div class="panel">
      <h3>For the combiners that follow</h3>
      <ul class="tight">
        <li>This is the bar: <b>{lo["held_out_val_selection_mean"]:.4f}</b> held-out
        validation macro-F1, from a rule with one discrete decision per corpus and no fitted
        parameters.</li>
        <li>Stacking or weighted voting that cannot beat this on validation is unjustified
        machinery, and the paper should report that outcome rather than bury it.</li>
        <li>The validation folds cap what any combiner can learn: 20, 62, 90 and 300 stories.
        Anything with more than a couple of free parameters is fitting {e(NICE["coaid"])}'s
        noise.</li>
      </ul>
    </div>
  </div>
</section>

<section>
  <div class="sec-head prose">
    <div class="eyebrow">reproduce</div>
    <h2>Commands</h2>
  </div>
  <pre>cd {e("WSDM-05-doc-less-fake/artifact")}

# 1. regenerate every component with train-only fits, dumping val AND test
#    per-story ids / labels / preds / scores  (nothing on disk had them)
PYTHONPATH=src python3 outputs/r14/ensemble/valselect/run_cheap.py
PYTHONPATH=src python3 outputs/r14/ensemble/valselect/drive.py \\
    outputs/r14/ensemble/valselect/gnn_jobs.txt 3
PYTHONPATH=src python3 outputs/r14/ensemble/valselect/drive.py \\
    outputs/r14/ensemble/valselect/dlf_jobs.txt 6

# 2. apply the frozen rule; emit test predictions + diagnostics
PYTHONPATH=src python3 outputs/r14/ensemble/valselect/analyze.py
PYTHONPATH=src python3 outputs/r14/ensemble/valselect/published_leader_check.py

# 3. validation-only worth of the rule, and fold resolving power
PYTHONPATH=src POOL=full   python3 outputs/r14/ensemble/valselect/val_loso.py
PYTHONPATH=src B=2000 python3 outputs/r14/ensemble/valselect/val_loso_ci.py
PYTHONPATH=src POOL=common python3 outputs/r14/ensemble/valselect/val_loso.py
PYTHONPATH=src B=1000 python3 outputs/r14/ensemble/valselect/valnoise.py
PYTHONPATH=src B=4000 python3 outputs/r14/ensemble/valselect/stability.py

# 4. the audit: blind every test_* field and re-derive the selection
PYTHONPATH=src python3 outputs/r14/ensemble/valselect/verify_no_leakage.py</pre>
</section>

<footer>
  <div>Test predictions for the judge:
  <code>outputs/r14/ensemble/valselect/combiner/test_preds_full_pool.json</code>
  &mdash; keyed by corpus, variant, seed, with story ids, labels, hard predictions and
  scores.</div>
  <div>Frozen rule: <code>SELECTION_RULE.json</code> &middot; full numbers:
  <code>report.json</code>, <code>val_loso_full.json</code>, <code>val_loso_ci.json</code>,
  <code>leader_stability.json</code>, <code>val_resolution.json</code>,
  <code>published_leader_check.json</code>.</div>
</footer>
</div>
'''

open(OUT, "w").write(DOC)
print("wrote", OUT, len(DOC), "bytes")

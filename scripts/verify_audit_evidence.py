#!/usr/bin/env python3
"""Verify historical table SDs, multiplicity arithmetic and sampler provenance.
No inference or training; never overwrite saved experimental outputs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]

def load(path):
    return json.loads((ROOT / path).read_text())

def verify():
    cells = load('scripts/table2_bindings.json')['cells']
    for cell in cells:
        if cell['status'] == 'unavailable':
            assert cell['sd'] is None
            continue
        source = ROOT / cell['source']
        assert hashlib.sha256(source.read_bytes()).hexdigest() == cell['source_sha256'], cell['source']
        data = json.loads(source.read_text())
        if 'row_filter' in cell:
            rows = sorted((row for row in data['results'] if all(row.get(k)==v for k,v in cell['row_filter'].items())), key=lambda row:row['seed'])
            assert [row['seed'] for row in rows] == list(range(5))
            values = [row['f1_all'] for row in rows]
        else:
            for key in cell['field'].split('/')[1:]: data = data[key]
            values = data['per_seed']
        assert len(values) == 5
        assert abs(statistics.mean(values)-cell['mean']) < .00051, cell
        assert abs(statistics.pstdev(values)-cell['sd']) < .00055, cell
    assert sum(c['status']=='verified' for c in cells) == 51
    manuscript = ROOT / 'paper/main.tex'
    if manuscript.exists():
        text = manuscript.read_text(); start=text.index('majority class      &'); end=text.index('\\bottomrule', start)
        printed=[]
        for line in text[start:end].splitlines():
            if '&' in line:
                printed += [(float(m), None if sd=='--' else float(sd)) for m,sd in re.findall(r'([0-9]+\.[0-9]+).*?\\sd\{([0-9.]+|--)\}',line)]
        assert printed == [(c['mean'],c['sd']) for c in cells], 'Table 2 differs from source bindings'
    family = load('outputs/r10/rev_stats/rev_stats.json')
    rows=family['multiplicity']; n=len(rows); assert n==23
    rows=sorted(rows,key=lambda row:row['raw_p'])
    holm=[];prev=0
    for i,row in enumerate(rows):
        prev=max(prev,min(1,(n-i)*row['raw_p']));holm.append(prev)
    bh=[0.]*n;prev=1
    for i in reversed(range(n)):
        prev=min(prev,n*rows[i]['raw_p']/(i+1));bh[i]=prev
    ledger=[]
    for i,row in enumerate(rows):
        assert abs(holm[i]-row['holm_p'])<.000051
        assert abs(bh[i]-row['bh_p'])<.000051
        unit='story cluster' if row['test'] in family['comparisons'] else 'pooled story-seed row'
        ledger.append(dict(row,resampling_unit=unit))
    assert sum(x['resampling_unit']=='story cluster' for x in ledger)==12
    neg=load('outputs/r8/ssl_negsource/ssl_negsource.json')['rows']
    full=load('outputs/r7/interaction_ablation/interaction_ablation.json')['results']
    scores={mode:[row['f1'] for row in sorted(neg,key=lambda x:x['seed']) if row['mode']==mode] for mode in ['label_free','label_aware']}
    r7=[row['f1_all'] for row in sorted(full,key=lambda x:x['seed']) if row['dataset']=='coaid' and row['variant']=='full']
    assert r7==scores['label_aware']
    assert round(statistics.mean(scores['label_free']),3)==.563
    assert round(statistics.mean(scores['label_aware']),3)==.587
    return {'table2_verified_cells':51,'table2_sd_unavailable':1,'ddof':0,'multiplicity':ledger,
            'interpretation':'Recomputed arithmetic on archived rounded p-values, not newly estimated or uniformly story-clustered inference. Stored zeros remain bootstrap resolution-limited estimates.',
            'sampler_diagnostic_means':{m:statistics.mean(v) for m,v in scores.items()},
            'sampler_provenance':'r7 full matches r8 label-aware seed for seed; sampler flags for historical main-table DLF runs are absent. Equality alone does not prove their configuration.'}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path);args=parser.parse_args()
    report=verify()
    if args.out: args.out.write_text(json.dumps(report,indent=2)+'\n')
    print('Verified: 51 Table 2 mean/population-SD cells; 1 explicitly unavailable SD; 23 Holm/BH entries (12 cluster, 11 row); pretraining diagnostic provenance.')

import json, numpy as np, torch, time, os
torch.set_num_threads(6)
from docless.train import DataConfig, ModelConfig, TrainConfig, train_one_seed, _temporal_split
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs/r9/temporal_split/docless_gossipcop.json")
dc = DataConfig(source='gossipcop', balanced=True,
                data_root=os.path.join(os.path.dirname(__file__), "..", "data/raw/fakenewsnet"),
                max_per_class=1500)
vals = []
for s in [0,1,2,3,4]:
    t0 = time.time()
    r = train_one_seed(s, dc, ModelConfig(hidden_dim=64, rich_summary=False),
                       TrainConfig(pretrain_epochs=4, finetune_epochs=25),
                       verbose=False, device=torch.device('cpu'), split_fn=_temporal_split)
    vals.append(r['f1_all'])
    json.dump(dict(mean=round(float(np.mean(vals)),3), std=round(float(np.std(vals)),3),
                   per_seed=[round(v,3) for v in vals], n_done=len(vals)), open(OUT,'w'), indent=2)
    print('seed', s, round(r['f1_all'],4), round(time.time()-t0,1), 's', flush=True)
print('DONE', flush=True)

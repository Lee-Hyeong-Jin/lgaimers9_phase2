"""Train NN on all training seasons and export for inference. usage:
python src/nn_export.py --tag nn_noids --no_ids --epochs 7 --lr 1e-3 --seeds 0,1,2"""
import sys, os, json, time, argparse, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, joblib
import torch, torch.nn.functional as Fnn
from sklearn.preprocessing import QuantileTransformer
from common import load_train, ROOT
import features2 as f2
from features import add_change_features, _recover_indicators, RATE_COLS_P
from exp2 import feature_sets
from nn_core import CAT_SPECS, build_net, transform_numeric, transform_cats

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='nn_noids'); ap.add_argument('--out', default=os.path.join(ROOT, 'submit', 'model'))
ap.add_argument('--epochs', type=int, default=7); ap.add_argument('--bs', type=int, default=4096)
ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--drop', type=float, default=0.25)
ap.add_argument('--id_drop', type=float, default=0.15); ap.add_argument('--seeds', default='0,1,2')
ap.add_argument('--aux_w', type=float, default=0.5); ap.add_argument('--hidden', default='512,256,128')
ap.add_argument('--wd', type=float, default=1e-5); ap.add_argument('--no_ids', action='store_true')
ap.add_argument('--weight', type=float, default=1.0)
ap.add_argument('--offset', action='store_true')
ap.add_argument('--ctx_offset', action='store_true')  # add CTX platoon/2K prior adjustments to the offset
ap.add_argument('--src_bayes', action='store_true')  # label-source-aware Bayes level
ap.add_argument('--wexp', type=float, default=0.0)
a = ap.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
tr = load_train(); max_season = int(tr.season.max())
stats = f2.build_stats(tr)
F = add_change_features(f2.build_features(tr, stats, mode='prev'))
fs = feature_sets(F)
cats = [c for c in CAT_SPECS if not (a.no_ids and c in ('pitcher_id', 'batter_id'))]
num_feats = [c for c in fs['V1'] if not c.startswith(('ROLE_', 'TM_')) and c not in list(CAT_SPECS) + ['pitcher_id', 'batter_id', 'season']]
F['season_rel'] = np.clip(max_season - F['season'].values, 0, 10)
num_feats = num_feats + ['season_rel']
X = F[num_feats].astype(np.float32).values; nanmask = np.isnan(X)
qt = QuantileTransformer(n_quantiles=200, output_distribution='normal', subsample=200000, random_state=0).fit(np.nan_to_num(X, nan=0.0))
nan_cols = [j for j in range(X.shape[1]) if nanmask[:, j].mean() > 0.001]
Xn = transform_numeric(F, num_feats, qt, nan_cols)
maps = {c: {v: k + 1 for k, v in enumerate(pd.unique(F[c].values))} for c in cats}
sizes = [(len(maps[c]), CAT_SPECS[c]) for c in cats]
Xc = transform_cats(F, cats, maps)
ind = _recover_indicators(tr, 'pitcher_id', 'asof_pitcher_n', RATE_COLS_P)
far = (1 - ind['s'] - ind['rev'] - ind['mid']).clip(0, 1)
y4 = torch.tensor(np.stack([ind['s'].values, ind['rev'].values, ind['mid'].values, far.values], 1).argmax(1))
ybs = torch.tensor(np.stack([ind['ball'].values, ind['strike'].values, (1 - ind['ball'] - ind['strike']).clip(0, 1).values], 1).argmax(1))
ypt = torch.tensor(np.stack([ind['fb'].values, ind['br'].values, ind['os'].values], 1).argmax(1))
init4 = None; fshare = None
if a.offset:
    from features_dev import bayes_features, BAYES_HP
    bay = bayes_features(tr, F, stats, src=a.src_bayes, **BAYES_HP)
    mu = dict(zip(stats['league'].season, stats['league'].league_rate)); mu_row = pd.Series(tr.season.values).map(mu).values
    pm = np.clip(mu_row + bay['bayes_dev'] + ((F['CTX_hand_adj'].values + F['CTX_2k_adj'].values) if a.ctx_offset else 0.0), 0.05, 0.95)
    fshare = np.bincount(y4.numpy(), minlength=4)[1:].astype(float); fshare /= fshare.sum()
    init4 = torch.tensor(np.stack([np.log(pm)] + [np.log((1 - pm) * fshare[k]) for k in range(3)], 1).astype(np.float32))
Xn_t = torch.tensor(Xn); Xc_t = torch.tensor(Xc); n = len(Xn)
W_np = (np.exp(-a.wexp * (max_season - tr.season.values)).astype(np.float32))
if os.environ.get('F_W'): W_np = W_np * np.where(tr.game_type.values == 'F', float(os.environ['F_W']), 1.0).astype(np.float32)
W_t = torch.tensor(W_np)
hidden = tuple(int(h) for h in a.hidden.split(','))
os.makedirs(a.out, exist_ok=True); files = []
for seed in [int(s) for s in a.seeds.split(',')]:
    torch.manual_seed(seed); np.random.seed(seed)
    net = build_net(Xn.shape[1], sizes, hidden, a.drop).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * ((n + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(n); tot = 0; t = time.time()
        for i in range(0, n, a.bs):
            b = perm[i:i + a.bs]; xn = Xn_t[b].to(dev); xc = Xc_t[b].to(dev)
            if a.id_drop > 0 and not a.no_ids:
                m = (torch.rand(xc.shape[0], 2, device=dev) < a.id_drop); xc = xc.clone(); xc[:, :2][m] = 0
            o4, obs, opt_ = net(xn, xc)
            if init4 is not None: o4 = o4 + init4[b].to(dev)
            if a.wexp > 0 or os.environ.get('F_W'):
                wb = W_t[b].to(dev)
                loss = (Fnn.cross_entropy(o4, y4[b].to(dev), reduction='none') * wb).sum() / wb.sum() + a.aux_w * ((Fnn.cross_entropy(obs, ybs[b].to(dev), reduction='none') * wb).sum() / wb.sum() + (Fnn.cross_entropy(opt_, ypt[b].to(dev), reduction='none') * wb).sum() / wb.sum())
            else:
                loss = Fnn.cross_entropy(o4, y4[b].to(dev)) + a.aux_w * (Fnn.cross_entropy(obs, ybs[b].to(dev)) + Fnn.cross_entropy(opt_, ypt[b].to(dev)))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(b)
        print(f'seed {seed} ep {ep} loss {tot/n:.4f} ({time.time()-t:.0f}s)', flush=True)
    fn = f'{a.tag}_seed{seed}.pt'
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, os.path.join(a.out, fn)); files.append(fn)
joblib.dump({'qt': qt, 'nan_cols': nan_cols, 'num_feats': num_feats, 'cats': cats, 'maps': maps, 'sizes': sizes, 'hidden': hidden, 'drop': a.drop, 'max_season': max_season}, os.path.join(a.out, f'{a.tag}_prep.pkl'), compress=3)
meta = {'type': 'nn', 'prep': f'{a.tag}_prep.pkl', 'boosters': files, 'mode': 'prev', 'level_guess': None, 'logit_shift': 0.0, 'weight': a.weight, 'offset': ((('bayes_ctx' if a.ctx_offset else 'bayes') + ('_src' if a.src_bayes else '')) if a.offset else ''), 'fshare': (fshare.tolist() if fshare is not None else None)}
json.dump(meta, open(os.path.join(a.out, f'{a.tag}_meta.json'), 'w'))
ens_path = os.path.join(a.out, 'ensemble.json')
ens = json.load(open(ens_path)) if os.path.exists(ens_path) else {'models': [], 'blend': 'logit'}
if f'{a.tag}_meta.json' not in ens['models']: ens['models'].append(f'{a.tag}_meta.json')
json.dump(ens, open(ens_path, 'w')); print('exported', files)

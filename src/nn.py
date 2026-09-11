"""Multi-task MLP with entity embeddings. Fold evaluation + OOF saving.
usage: python src/nn.py --name zoo_nn --folds 2024,2023 --epochs 6
"""
import sys, os, json, time, argparse, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as Fnn
from sklearn.preprocessing import QuantileTransformer
from exp2 import fold_data, feature_sets, oracle_shift
from features import _recover_indicators, RATE_COLS_P
from common import load_train, bss_score, log_experiment, EXP

CAT_SPECS = {'pitcher_id': 64, 'batter_id': 32, 'pitcher_team': 4, 'batter_team': 4, 'home_team': 4, 'count_state': 4, 'base_state': 3}


def prep_numeric(F, feats, fit_idx):
    X = F[feats].astype(np.float32).values
    nanmask = np.isnan(X)
    qt = QuantileTransformer(n_quantiles=200, output_distribution='normal', subsample=200000, random_state=0)
    Xf = X.copy(); Xf[nanmask] = np.nan
    qt.fit(np.nan_to_num(X[fit_idx], nan=0.0))
    Xq = qt.transform(np.nan_to_num(X, nan=0.0)).astype(np.float32)
    Xq[nanmask] = 0.0
    nan_cols = [j for j in range(X.shape[1]) if nanmask[fit_idx, j].mean() > 0.001]
    Xn = np.concatenate([Xq, nanmask[:, nan_cols].astype(np.float32)], 1)
    return Xn, qt, nan_cols


class Net(nn.Module):
    def __init__(self, n_num, cat_sizes, hidden=(512, 256, 128), drop=0.2, n_heads=1):
        super().__init__(); self.n_heads = n_heads
        self.embs = nn.ModuleList([nn.Embedding(n + 1, d) for n, d in cat_sizes])  # +1 unknown
        d_in = n_num + sum(d for _, d in cat_sizes)
        layers = []
        for h in hidden:
            layers += [nn.Linear(d_in, h), nn.BatchNorm1d(h), nn.SiLU(), nn.Dropout(drop)]; d_in = h
        self.mlp = nn.Sequential(*layers)
        self.head4 = nn.Linear(d_in, 4 * n_heads); self.head_bs = nn.Linear(d_in, 3); self.head_pt = nn.Linear(d_in, 3)

    def forward(self, xn, xc):
        e = [emb(xc[:, i]) for i, emb in enumerate(self.embs)]
        h = self.mlp(torch.cat([xn] + e, 1))
        o = self.head4(h)
        if self.n_heads > 1:  # label-process-specific output heads; head index = last column of xc (not embedded)
            o = o.view(-1, self.n_heads, 4)[torch.arange(h.shape[0], device=h.device), xc[:, -1]]
        return o, self.head_bs(h), self.head_pt(h)


def encode_cats(F, cats, fit_idx):
    codes = np.zeros((len(F), len(cats)), dtype=np.int64); sizes = []
    maps = {}
    for i, c in enumerate(cats):
        vals = F[c].values
        uniq = pd.unique(vals[fit_idx])
        mp = {v: k + 1 for k, v in enumerate(uniq)}  # 0 = unknown
        codes[:, i] = pd.Series(vals).map(mp).fillna(0).astype(int).values
        sizes.append((len(uniq), CAT_SPECS[c])); maps[c] = mp
    return codes, sizes, maps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='zoo_nn'); ap.add_argument('--folds', default='2024,2023')
    ap.add_argument('--epochs', type=int, default=6); ap.add_argument('--bs', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--drop', type=float, default=0.25)
    ap.add_argument('--id_drop', type=float, default=0.15); ap.add_argument('--seeds', default='0')
    ap.add_argument('--aux_w', type=float, default=0.5); ap.add_argument('--feats', default='V1ONLY')
    ap.add_argument('--hidden', default='512,256,128'); ap.add_argument('--wd', type=float, default=1e-5)
    ap.add_argument('--no_ids', action='store_true')
    ap.add_argument('--no_batter_id', action='store_true')
    ap.add_argument('--h2', action='store_true')
    ap.add_argument('--ctx_offset', action='store_true')
    ap.add_argument('--pseason', type=int, default=0)
    ap.add_argument('--no_bayes', action='store_true')
    ap.add_argument('--src_heads', action='store_true')  # separate output heads per label process (nonLG-R / LG-R / F)  # with --offset: drop the Bayes level term (keep league mean + CTX adj)  # >0: add pitcher x season embedding of this dim (test season -> pitcher's last train season)  # add CTX platoon/2K prior adjustments into the class-logit offset
    ap.add_argument('--offset', action='store_true')  # add class-wise Bayes log-prior to the 4-class logits
    ap.add_argument('--emb_scale', type=float, default=1.0)  # multiply ID embedding dims
    ap.add_argument('--wexp', type=float, default=0.0)  # recency sample weight exp(-wexp*(last_train_season-season))
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tr = load_train(); y = tr.control_success.values
    ind = _recover_indicators(tr, 'pitcher_id', 'asof_pitcher_n', RATE_COLS_P)
    far = (1 - ind['s'] - ind['rev'] - ind['mid']).clip(0, 1)
    y4 = np.stack([ind['s'].values, ind['rev'].values, ind['mid'].values, far.values], 1).argmax(1)
    ybs = np.stack([ind['ball'].values, ind['strike'].values, (1 - ind['ball'] - ind['strike']).clip(0, 1).values], 1).argmax(1)
    ypt = np.stack([ind['fb'].values, ind['br'].values, ind['os'].values], 1).argmax(1)
    OOF = os.path.join(EXP, 'oof'); os.makedirs(OOF, exist_ok=True)
    res = {'folds': {}}
    for vs in [int(x) for x in a.folds.split(',')]:
        F, stats_for_offset = fold_data(tr, vs, 'prev'); fs = feature_sets(F)
        base = fs['V1'] if a.feats == 'V1' else [c for c in fs['V1'] if not c.startswith(('ROLE_', 'TM_'))] if a.feats == 'V1ONLY' else fs[a.feats]
        cats = [c for c in CAT_SPECS if not ((a.no_ids and c in ('pitcher_id', 'batter_id')) or (a.no_batter_id and c == 'batter_id'))]
        num_feats = [c for c in base if c not in cats + ['pitcher_id', 'batter_id', 'season']]
        if os.environ.get('NN_NUM_FEATS'): num_feats = [c for c in os.environ['NN_NUM_FEATS'].split(',') if c in F.columns]
        tr_idx = np.where(tr.season.values < vs)[0]; va_idx = np.where(tr.season.values == vs)[0]
        if a.h2: k = len(va_idx) // 2; tr_idx = np.concatenate([tr_idx, va_idx[:k]]); va_idx = va_idx[k:]
        ref = vs if a.h2 else vs - 1; ftag = f'{vs}H2' if a.h2 else f'{vs}'
        # season relative index (0 = latest train season); val rows -> 0
        F = F.copy(); F['season_rel'] = np.clip(ref - F['season'].values, 0, 10)
        n_id_cols = 2
        if a.pseason > 0:
            seas_ = F['season'].values; pid_ = F['pitcher_id'].values
            last_active = pd.Series(seas_[tr_idx]).groupby(pid_[tr_idx]).max()
            key_season = np.where(np.isin(np.arange(len(F)), tr_idx), seas_, pd.Series(pid_).map(last_active).fillna(0).values)
            F['pseason_key'] = (pid_.astype(np.int64) * 100 + (key_season.astype(np.int64) - 2000)) * (key_season > 0)
            CAT_SPECS['pseason_key'] = a.pseason; cats = cats[:2] + ['pseason_key'] + cats[2:]; n_id_cols = 3
        num_feats = num_feats + ['season_rel']
        init4 = None
        if a.offset:
            from features_dev import bayes_features, BAYES_HP
            bay = bayes_features(tr, F, stats_for_offset, **BAYES_HP)
            mu = dict(zip(stats_for_offset['league'].season, stats_for_offset['league'].league_rate)); mu_row = pd.Series(tr.season.values).map(mu).fillna(mu[max(mu)]).values
            adj_cols = [c for c in os.environ.get('NN_CTX_ADJ', 'CTX_hand_adj,CTX_2k_adj').split(',') if c in F.columns]
            pm = np.clip(mu_row + (0.0 if a.no_bayes else bay['bayes_dev']) + (sum(F[c].values for c in adj_cols) if a.ctx_offset else 0.0), 0.05, 0.95)
            fsh = np.bincount(y4[tr_idx], minlength=4)[1:].astype(float); fsh /= fsh.sum()
            init4 = torch.tensor(np.stack([np.log(pm)] + [np.log((1 - pm) * fsh[k]) for k in range(3)], 1).astype(np.float32))
        Xn, qt, nan_cols = prep_numeric(F, num_feats, tr_idx)
        Xc, sizes, maps = encode_cats(F, cats, tr_idx)
        n_heads = 1
        if a.src_heads:
            isF_ = (tr.game_type.values == 'F'); lg_ = ((tr.pitcher_team_id.values == 13) | (tr.batter_team_id.values == 13)) & (~isF_)
            src_idx = np.where(isF_, 2, np.where(lg_, 1, 0)).astype(np.int64); Xc = np.concatenate([Xc, src_idx[:, None]], 1); n_heads = 3
            print('src heads: counts', np.bincount(src_idx[tr_idx]), flush=True)
        if a.emb_scale != 1.0: sizes = [(n, int(d * a.emb_scale)) if c in ('pitcher_id', 'batter_id') else (n, d) for (n, d), c in zip(sizes, cats)]
        print(f'fold {vs}: num {Xn.shape[1]} cats {sizes}', flush=True)
        Xn_t = torch.tensor(Xn); Xc_t = torch.tensor(Xc)
        W_np = np.exp(-a.wexp * (ref - tr.season.values)).astype(np.float32)
        if os.environ.get('F_W'): W_np = W_np * np.where(tr.game_type.values == 'F', float(os.environ['F_W']), 1.0).astype(np.float32)
        W_t = torch.tensor(W_np)
        Y4 = torch.tensor(y4); YBS = torch.tensor(ybs); YPT = torch.tensor(ypt)
        preds = np.zeros(len(va_idx)); yv = y[va_idx]; r = yv.mean()
        for seed in [int(s) for s in a.seeds.split(',')]:
            torch.manual_seed(seed); np.random.seed(seed)
            net = Net(Xn.shape[1], sizes, tuple(int(h) for h in a.hidden.split(',')), a.drop, n_heads=n_heads).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
            n_tr = len(tr_idx); steps = a.epochs * ((n_tr + a.bs - 1) // a.bs)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
            tr_idx_t = torch.tensor(tr_idx)
            best = None
            for ep in range(a.epochs):
                net.train(); perm = tr_idx_t[torch.randperm(n_tr)]; t = time.time(); tot = 0
                for i in range(0, n_tr, a.bs):
                    b = perm[i:i + a.bs]
                    xn = Xn_t[b].to(dev); xc = Xc_t[b].to(dev)
                    if a.id_drop > 0:  # randomly mask entity ids -> unknown (cold-start robustness)
                        m = (torch.rand(xc.shape[0], n_id_cols, device=dev) < a.id_drop)
                        xc = xc.clone(); xc[:, :n_id_cols][m] = 0
                    o4, obs, opt_ = net(xn, xc)
                    if init4 is not None: o4 = o4 + init4[b].to(dev)
                    if a.wexp > 0 or os.environ.get('F_W'):
                        wb = W_t[b].to(dev)
                        loss = (Fnn.cross_entropy(o4, Y4[b].to(dev), reduction='none') * wb).sum() / wb.sum() + a.aux_w * ((Fnn.cross_entropy(obs, YBS[b].to(dev), reduction='none') * wb).sum() / wb.sum() + (Fnn.cross_entropy(opt_, YPT[b].to(dev), reduction='none') * wb).sum() / wb.sum())
                    else:
                        loss = Fnn.cross_entropy(o4, Y4[b].to(dev)) + a.aux_w * (Fnn.cross_entropy(obs, YBS[b].to(dev)) + Fnn.cross_entropy(opt_, YPT[b].to(dev)))
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(b)
                net.eval(); ps = []
                with torch.no_grad():
                    for i in range(0, len(va_idx), 32768):
                        b = torch.tensor(va_idx[i:i + 32768])
                        o4, _, _ = net(Xn_t[b].to(dev), Xc_t[b].to(dev))
                        if init4 is not None: o4 = o4 + init4[b].to(dev)
                        ps.append(torch.softmax(o4, 1)[:, 0].cpu().numpy())
                p = np.concatenate(ps)
                sc, so = bss_score(yv, p), bss_score(yv, oracle_shift(p, r)[0])
                print(f'  seed {seed} ep {ep}: loss={tot/n_tr:.4f} score={sc:.1f} oracle={so:.1f} pm={p.mean():.4f} ({time.time()-t:.0f}s)', flush=True)
                best = p
            preds += best
        preds /= len(a.seeds.split(','))
        sc, so = bss_score(yv, preds), bss_score(yv, oracle_shift(preds, r)[0])
        res['folds'][ftag] = {'score': sc, 'oracle': so, 'pm': float(preds.mean())}
        print(f'[{a.name}] fold {ftag}: score={sc:.1f} oracle={so:.1f} pm={preds.mean():.4f}', flush=True)
        np.save(os.path.join(OOF, f'{a.name}_{ftag}.npy'), preds)
    log_experiment(a.name, {**res, 'args': vars(a)})


if __name__ == '__main__':
    main()

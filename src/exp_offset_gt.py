"""Offset multiclass with a GAME-TYPE-SPECIFIC dynamic-Bayesian level: prior seasons' totals restricted to the row's game type
(F rows use F history + F league means; R rows use R history + R league means); season-to-date (mixed) update as before."""
import sys, os, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.special import softmax
from exp2 import fold_data, feature_sets, oracle_shift
from features import _recover_indicators, RATE_COLS_P, CAT_FEATURES
from bayes_level import season_priors, row_posterior
from common import load_train, bss_score, EXP
from exp import DEFAULT_PARAMS
vs = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.03; ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 180
tr = load_train(); y = tr.control_success.values; seas = tr.season.values; gt = tr.game_type.values
F, stats = fold_data(tr, vs, 'prev'); fs = feature_sets(F); feats = fs['V1']
cat = [c for c in CAT_FEATURES + ['home_team'] if c in feats]
ind = _recover_indicators(tr, 'pitcher_id', 'asof_pitcher_n', RATE_COLS_P)
far = (1 - ind['s'] - ind['rev'] - ind['mid']).clip(0, 1)
y4 = np.stack([ind['s'].values, ind['rev'].values, ind['mid'].values, far.values], 1).argmax(1)
ps_tab = stats['ps']; Lgt = stats['L_gt']
pm = np.zeros(len(tr))
for g in ('R', 'F'):
    if g == 'F':
        tab = ps_tab[['pitcher_id', 'season', 'n_F', 's_F']].rename(columns={'n_F': 'n', 's_F': 's'})
    else:
        tab = ps_tab[['pitcher_id', 'season']].copy(); tab['n'] = ps_tab.n - ps_tab.n_F; tab['s'] = ps_tab.s - ps_tab.s_F
    tab = tab[tab.n > 0]
    league = Lgt[g].dropna().to_dict()
    pri, mu = season_priors(tab, league, 0.6, 0.004, 0.004)
    m = gt == g
    pm_g, pv, m0, v0 = row_posterior(tr[m], F.iloc[np.where(m)[0]], pri, mu, 0.004)
    pm[m] = pm_g
pm = np.clip(pm, 0.05, 0.95)
tr_idx = np.where(seas < vs)[0]; va = np.where(seas == vs)[0]; r = y[va].mean(); isR = gt[va] == 'R'
fshare = np.bincount(y4[tr_idx], minlength=4)[1:].astype(float); fshare /= fshare.sum()
init = np.stack([np.log(pm)] + [np.log((1 - pm) * fshare[k]) for k in range(3)], 1)
wts = np.exp(-0.15 * (vs - 1 - seas[tr_idx])).astype(float)
p = {**DEFAULT_PARAMS, 'objective': 'multiclass', 'num_class': 4, 'num_threads': 10, 'learning_rate': LR}
ps_ = np.zeros(len(va))
for sd in (0, 1):
    dtr = lgb.Dataset(F[feats].iloc[tr_idx], y4[tr_idx], weight=wts, init_score=init[tr_idx], categorical_feature=cat)
    m = lgb.train({**p, 'seed': sd, 'bagging_seed': sd, 'feature_fraction_seed': sd}, dtr, num_boost_round=ROUNDS)
    ps_ += softmax(m.predict(F[feats].iloc[va], raw_score=True) + init[va], axis=1)[:, 0] / 2
rr = y[va][isR].mean()
print(f'[fold {vs}] multiclass + GAME-TYPE bayes offset (lr={LR}, rounds={ROUNDS}): raw={bss_score(y[va], ps_):.1f} oracle={bss_score(y[va], oracle_shift(ps_, r)[0]):.1f} | R-only oracle={bss_score(y[va][isR], oracle_shift(ps_[isR], rr)[0]):.1f} | F-only oracle={bss_score(y[va][~isR], oracle_shift(ps_[~isR], y[va][~isR].mean())[0]):.1f}', flush=True)
np.save(os.path.join(EXP, 'oof', f'mcs_offgt_{vs}.npy'), ps_)

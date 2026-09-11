"""Multiclass LGB with class-wise init_score derived from the Bayes posterior mean of success:
init = [log(pm), log((1-pm)*f_rev), log((1-pm)*f_mid), log((1-pm)*f_far)] (f = failure-mode shares in training)."""
import sys, os, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.special import logit, expit, softmax
from exp2 import fold_data, feature_sets, oracle_shift
from features import _recover_indicators, RATE_COLS_P, CAT_FEATURES
from features_dev import bayes_features
from common import load_train, bss_score, EXP
from exp import DEFAULT_PARAMS
vs = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
seeds = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else '0,1').split(',')]
use_batter = len(sys.argv) > 3 and sys.argv[3] == 'batter'
hp = dict(a=float(sys.argv[4]), tau2=float(sys.argv[5]), v_new=float(sys.argv[6])) if len(sys.argv) > 6 else dict(a=0.6, tau2=0.004, v_new=0.004)
tag = sys.argv[7] if len(sys.argv) > 7 else ('off_b' if use_batter else 'off')
ROUNDS = int(sys.argv[8]) if len(sys.argv) > 8 else 180
WEXP = float(sys.argv[9]) if len(sys.argv) > 9 else 0.15
tr = load_train(); y = tr.control_success.values; seas = tr.season.values
F, stats = fold_data(tr, vs, 'prev'); fs = feature_sets(F); feats = fs['V1']
cat = [c for c in CAT_FEATURES + ['home_team'] if c in feats] + (['pitcher_id', 'batter_id'] if os.environ.get('OFF_CATIDS') else [])
ind = _recover_indicators(tr, 'pitcher_id', 'asof_pitcher_n', RATE_COLS_P)
far = (1 - ind['s'] - ind['rev'] - ind['mid']).clip(0, 1)
y4 = np.stack([ind['s'].values, ind['rev'].values, ind['mid'].values, far.values], 1).argmax(1)
bay = bayes_features(tr, F, stats, src=os.environ.get('SRC_BAYES') == '1', **hp)
if use_batter:
    from features_dev import batter_bayes_features
    bb = batter_bayes_features(tr, F, stats)
mu = dict(zip(stats['league'].season, stats['league'].league_rate)); mu_row = pd.Series(seas).map(mu).fillna(mu[max(mu)]).values
pm = np.clip(mu_row + bay['bayes_dev'] + (bb['bbayes_dev'] if use_batter else 0.0), 0.05, 0.95)
tr_idx = np.where(seas < vs)[0]; va = np.where(seas == vs)[0]
H2 = bool(os.environ.get('H2'))
if H2: k = len(va) // 2; tr_idx = np.concatenate([tr_idx, va[:k]]); va = va[k:]
r = y[va].mean(); isR = tr.game_type.values[va] == 'R'; ref = vs if H2 else vs - 1; ftag = f'{vs}H2' if H2 else f'{vs}'
fshare = np.bincount(y4[tr_idx], minlength=4)[1:].astype(float); fshare /= fshare.sum()
init = np.stack([np.log(pm)] + [np.log((1 - pm) * fshare[k]) for k in range(3)], 1)
wts = np.exp(-WEXP * (ref - seas[tr_idx])).astype(float)
if os.environ.get('F_W'): wts = wts * np.where(tr.game_type.values[tr_idx] == 'F', float(os.environ['F_W']), 1.0)
rs = os.environ.get('ROWSET')
if rs:
    lgi = (tr.pitcher_team_id.values == 13) | (tr.batter_team_id.values == 13); isF = tr.game_type.values == 'F'; mo = tr.game_month.values
    newreg = (seas > 2023) | ((seas == 2023) & ((mo >= 5) | isF))
    keep = {'lg': (lgi & newreg), 'lgall': lgi, 'nonlg': (~lgi) & (~isF)}[rs]
    m_ = keep[tr_idx]; tr_idx = tr_idx[m_]; wts = wts[m_]; print(f'ROWSET={rs}: training rows {len(tr_idx)}', flush=True)
if os.environ.get('LG_W'):
    lgw_ = ((tr.pitcher_team_id.values[tr_idx] == 13) | (tr.batter_team_id.values[tr_idx] == 13)) & (tr.game_type.values[tr_idx] == 'R') & ((seas[tr_idx] > 2023) | ((seas[tr_idx] == 2023) & (tr.game_month.values[tr_idx] >= 5)))
    wts = wts * np.where(lgw_, float(os.environ['LG_W']), 1.0)
if os.environ.get('DROP_LGOLD') == '1':
    lg_old = (((tr.pitcher_team_id.values[tr_idx] == 13) | (tr.batter_team_id.values[tr_idx] == 13)) & (seas[tr_idx] < 2023))
    tr_idx = tr_idx[~lg_old]; wts = wts[~lg_old]; print('dropped pre-2023 LG rows:', int(lg_old.sum()), flush=True)
p = {**DEFAULT_PARAMS, 'objective': 'multiclass', 'num_class': 4, 'num_threads': 6, 'learning_rate': float(os.environ.get('OFF_LR', '0.03'))}
if os.environ.get('OFF_CATIDS'): p.update({'cat_l2': 50, 'cat_smooth': 100, 'min_data_per_group': 500})
ps_ = np.zeros(len(va))
for sd in seeds:
    dtr = lgb.Dataset(F[feats].iloc[tr_idx], y4[tr_idx], weight=wts, init_score=init[tr_idx], categorical_feature=cat)
    m = lgb.train({**p, 'seed': sd, 'bagging_seed': sd, 'feature_fraction_seed': sd}, dtr, num_boost_round=ROUNDS)
    raw = m.predict(F[feats].iloc[va], raw_score=True) + init[va]; ps_ += softmax(raw, axis=1)[:, 0] / len(seeds)
rr = y[va][isR].mean()
print(f'[fold {ftag}] multiclass + bayes offset [{tag} hp={hp}] ({len(seeds)} seeds): raw={bss_score(y[va], ps_):.1f} oracle={bss_score(y[va], oracle_shift(ps_, r)[0]):.1f} | R-only oracle={bss_score(y[va][isR], oracle_shift(ps_[isR], rr)[0]):.1f}', flush=True)
np.save(os.path.join(EXP, 'oof', f'mcs_{tag}_{ftag}.npy'), ps_)

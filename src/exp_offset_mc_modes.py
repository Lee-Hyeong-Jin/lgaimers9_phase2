"""Multiclass LGB with class-wise init from PITCHER-SPECIFIC dynamic-Bayesian posteriors of success, reverse and middle rates."""
import sys, os, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.special import softmax
from exp2 import fold_data, feature_sets, oracle_shift
from features import _recover_indicators, RATE_COLS_P, CAT_FEATURES
from bayes_level import season_priors, row_posterior
from common import load_train, bss_score, EXP
from exp import DEFAULT_PARAMS
vs = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
tr = load_train(); y = tr.control_success.values; seas = tr.season.values
F, stats = fold_data(tr, vs, 'prev'); fs = feature_sets(F); feats = fs['V1']
cat = [c for c in CAT_FEATURES + ['home_team'] if c in feats]
ind = _recover_indicators(tr, 'pitcher_id', 'asof_pitcher_n', RATE_COLS_P)
far = (1 - ind['s'] - ind['rev'] - ind['mid']).clip(0, 1)
y4 = np.stack([ind['s'].values, ind['rev'].values, ind['mid'].values, far.values], 1).argmax(1)
tr_idx = np.where(seas < vs)[0]; va = np.where(seas == vs)[0]; r = y[va].mean(); isR = tr.game_type.values[va] == 'R'
ps_tab = stats['ps']
def posterior(col_s, cur_rate_col, mu_dict, a, tau2, v_new):
    tab = ps_tab[['pitcher_id', 'season', 'n', col_s]].rename(columns={col_s: 's'})
    pri, mu = season_priors(tab, mu_dict, a, tau2, v_new)
    Fx = F[['p_cur_n']].copy(); Fx['p_cur_s'] = np.nan_to_num(F[cur_rate_col].values) * F['p_cur_n'].values
    pm, pv, m0, v0 = row_posterior(tr, Fx, pri, mu, v_new); return pm
league_s = dict(zip(stats['league'].season, stats['league'].league_rate))
# season means of rev/mid from training rows
fm = pd.DataFrame({'season': seas[seas < vs], 'rev': ind['rev'].values[seas < vs], 'mid': ind['mid'].values[seas < vs]}).groupby('season').mean()
pm_s = np.clip(posterior('s', 'p_cur_s_rate', league_s, 0.6, 0.004, 0.004), 0.05, 0.95)
pm_rev = np.clip(posterior('rev', 'p_cur_rev_rate', fm['rev'].to_dict(), 0.6, 0.002, 0.002), 0.02, 0.8)
pm_mid = np.clip(posterior('mid', 'p_cur_mid_rate', fm['mid'].to_dict(), 0.6, 0.001, 0.001), 0.02, 0.6)
# renormalize failure modes to sum to (1 - pm_s); far = remainder (>= 0.02)
fail = np.maximum(1 - pm_s, 0.05); pm_far = np.clip(fail - pm_rev - pm_mid, 0.02, None)
tot = pm_rev + pm_mid + pm_far; pm_rev, pm_mid, pm_far = pm_rev / tot * fail, pm_mid / tot * fail, pm_far / tot * fail
init = np.log(np.stack([pm_s, pm_rev, pm_mid, pm_far], 1))
wts = np.exp(-0.15 * (vs - 1 - seas[tr_idx])).astype(float)
p = {**DEFAULT_PARAMS, 'objective': 'multiclass', 'num_class': 4, 'num_threads': 8}
ps_ = np.zeros(len(va))
for sd in (0, 1):
    dtr = lgb.Dataset(F[feats].iloc[tr_idx], y4[tr_idx], weight=wts, init_score=init[tr_idx], categorical_feature=cat)
    m = lgb.train({**p, 'seed': sd, 'bagging_seed': sd, 'feature_fraction_seed': sd}, dtr, num_boost_round=180)
    ps_ += softmax(m.predict(F[feats].iloc[va], raw_score=True) + init[va], axis=1)[:, 0] / 2
rr = y[va][isR].mean()
print(f'[fold {vs}] multiclass + mode-wise bayes offsets: raw={bss_score(y[va], ps_):.1f} oracle={bss_score(y[va], oracle_shift(ps_, r)[0]):.1f} | R-only oracle={bss_score(y[va][isR], oracle_shift(ps_[isR], rr)[0]):.1f}', flush=True)
np.save(os.path.join(EXP, 'oof', f'mcs_offmodes_{vs}.npy'), ps_)

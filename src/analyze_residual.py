"""Residual analysis of the fold-2024 OOF blend: (1) residual-GBM with pitcher-grouped 2-fold to detect unexplained
feature structure, (2) binned residual means for key features, (3) per-pitcher residual autocorrelation."""
import sys, os, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.special import logit, expit
from exp2 import fold_data, feature_sets, oracle_shift
from common import load_train, bss_score
tr = load_train(); y = tr.control_success.values
va = np.where(tr.season.values == 2024)[0]; yv = y[va]; r = yv.mean()
F, _ = fold_data(tr, 2024, 'prev'); fs = feature_sets(F); Fv = F.iloc[va].reset_index(drop=True)
P = {k: np.load(f'experiments/oof/{k}_2024.npy') for k in ['mcs_pb', 'zoo_catids_lr02_s3', 'zoo_cat_mc']}
p = expit(0.47 * logit(P['mcs_pb']) + 0.41 * logit(P['zoo_catids_lr02_s3']) + 0.12 * logit(P['zoo_cat_mc']))
p, _ = oracle_shift(p, r); base = bss_score(yv, p); res = yv - p
print(f'blend oracle score {base:.1f}; residual mean {res.mean():.5f} sd {res.std():.4f}')
# (1) residual GBM, grouped 2-fold by pitcher
feats = [c for c in fs['V1'] if c not in ('pitcher_id', 'batter_id')]
pid = Fv.pitcher_id.values; rng = np.random.RandomState(0); up = np.unique(pid); rng.shuffle(up)
half = set(up[:len(up) // 2]); g1 = np.array([q in half for q in pid]); folds = [(g1, ~g1), (~g1, g1)]
pred_res = np.zeros(len(va))
params = dict(objective='regression', learning_rate=0.03, num_leaves=31, min_data_in_leaf=2000, feature_fraction=0.7, lambda_l2=50, verbose=-1, num_threads=4)
imp = None
for trm, tem in folds:
    m = lgb.train(params, lgb.Dataset(Fv[feats][trm], res[trm]), num_boost_round=150)
    pred_res[tem] = m.predict(Fv[feats][tem])
    imp = pd.Series(m.feature_importance('gain'), index=feats) if imp is None else imp + pd.Series(m.feature_importance('gain'), index=feats)
p2 = np.clip(p + pred_res, 1e-4, 1 - 1e-4)
print(f'residual-GBM (pitcher-grouped 2-fold) corrected score: {bss_score(yv, p2):.1f} (gain {bss_score(yv, p2) - base:+.1f}); corr(pred_res, res) = {np.corrcoef(pred_res, res)[0,1]:.4f}')
print('top residual-GBM features:', imp.sort_values(ascending=False).head(12).round(0).to_dict())
# (2) binned residual means for key features (|mean| > 2 se flagged)
def binned(col, bins=10):
    x = Fv[col].values; ok = ~np.isnan(x)
    q = pd.qcut(x[ok], bins, labels=False, duplicates='drop')
    d = pd.DataFrame({'q': q, 'res': res[ok], 'x': x[ok]}).groupby('q').agg(n=('res', 'size'), mres=('res', 'mean'), x=('x', 'mean'))
    d['se'] = 0.5 / np.sqrt(d.n); d['z'] = d.mres / d.se
    return d
print('\n-- binned residuals (z = mean residual / se); flagged |z|>3 --')
for col in ['p_cur_n', 'p_cur_s_rate', 'p_prior_n', 'p_prior_s_rate', 'b_cur_n', 'b_cur_s_rate', 'asof_p_prev1_game_success_rate', 'p_form5', 'li', 'inning', 'game_month', 'count_state', 'p_cur_mid_rate', 'p_cur_rev_rate', 'TM_fb_speed', 'ROLE_ppg']:
    if col not in Fv: continue
    d = binned(col); flag = d[np.abs(d.z) > 3]
    print(f'{col:32s} max|z|={np.abs(d.z).max():.1f}', ('FLAG: ' + ', '.join(f'x~{row.x:.3g}: {row.mres:+.4f}(z{row.z:+.1f})' for _, row in flag.iterrows())) if len(flag) else '')
# (3) within-pitcher residual autocorrelation (consecutive rows of the same pitcher in row order)
d = pd.DataFrame({'pid': pid, 'res': res, 'ord': va})
d = d.sort_values(['pid', 'ord']); prev = d.groupby('pid').res.shift(1); ok = prev.notna()
print(f'\nlag-1 residual autocorrelation within pitcher: {np.corrcoef(d.res[ok], prev[ok])[0,1]:.4f} (n={ok.sum()})')
# per-pitcher-season mean residual variance vs expected noise (is there unexplained pitcher-level signal left?)
g = d.groupby('pid').res.agg(['mean', 'size']); g = g[g['size'] >= 200]
exp_var = np.mean(0.25 / g['size']); obs_var = np.var(g['mean'])
print(f'pitcher-level residual: observed var of per-pitcher mean residual {obs_var:.6f} vs expected noise var {exp_var:.6f} -> unexplained pitcher-level var ~ {max(obs_var - exp_var, 0):.6f} (sd {np.sqrt(max(obs_var - exp_var, 0)):.4f}); potential BSS gain if captured ~ {1e5 * max(obs_var - exp_var, 0) / (r*(1-r)):.0f}')
gb = pd.DataFrame({'bid': Fv.batter_id.values, 'res': res}).groupby('bid').res.agg(['mean', 'size']); gb = gb[gb['size'] >= 200]
print(f'batter-level: observed {np.var(gb["mean"]):.6f} vs noise {np.mean(0.25/gb["size"]):.6f} -> unexplained ~ {max(np.var(gb["mean"]) - np.mean(0.25/gb["size"]), 0):.6f}; potential gain ~ {1e5*max(np.var(gb["mean"]) - np.mean(0.25/gb["size"]),0)/(r*(1-r)):.0f}')

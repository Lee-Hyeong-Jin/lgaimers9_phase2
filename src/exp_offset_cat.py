"""CatBoost MultiClass with baseline (class-wise raw scores) from the Bayes posterior mean of success."""
import sys, os, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.special import softmax
from exp2 import fold_data, feature_sets, oracle_shift
from features import _recover_indicators, RATE_COLS_P, CAT_FEATURES
from features_dev import bayes_features
from common import load_train, bss_score, EXP
vs = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
tr = load_train(); y = tr.control_success.values; seas = tr.season.values
F, stats = fold_data(tr, vs, 'prev'); fs = feature_sets(F); feats = fs['V1']
cat = [c for c in CAT_FEATURES + ['home_team'] if c in feats] + ['pitcher_id', 'batter_id']
ind = _recover_indicators(tr, 'pitcher_id', 'asof_pitcher_n', RATE_COLS_P)
far = (1 - ind['s'] - ind['rev'] - ind['mid']).clip(0, 1)
y4 = np.stack([ind['s'].values, ind['rev'].values, ind['mid'].values, far.values], 1).argmax(1)
bay = bayes_features(tr, F, stats)
mu = dict(zip(stats['league'].season, stats['league'].league_rate)); mu_row = pd.Series(seas).map(mu).fillna(mu[max(mu)]).values
pm = np.clip(mu_row + bay['bayes_dev'], 0.05, 0.95)
tr_idx = np.where(seas < vs)[0]; va = np.where(seas == vs)[0]; r = y[va].mean(); isR = tr.game_type.values[va] == 'R'
fshare = np.bincount(y4[tr_idx], minlength=4)[1:].astype(float); fshare /= fshare.sum()
init = np.stack([np.log(pm)] + [np.log((1 - pm) * fshare[k]) for k in range(3)], 1)
X = F[feats].copy()
for c in cat: X[c] = X[c].astype(int)
ptr = Pool(X.iloc[tr_idx], y4[tr_idx], cat_features=cat, baseline=init[tr_idx]); pva = Pool(X.iloc[va], cat_features=cat, baseline=init[va])
m = CatBoostClassifier(loss_function='MultiClass', iterations=2500, learning_rate=0.02, depth=7, l2_leaf_reg=10, random_seed=0, verbose=0, task_type='GPU', border_count=128, thread_count=4)
m.fit(ptr); raw = m.predict(pva, prediction_type='RawFormulaVal'); ps_ = softmax(raw, axis=1)[:, 0]
rr = y[va][isR].mean()
print(f'[fold {vs}] CAT-ID multiclass + bayes baseline: raw={bss_score(y[va], ps_):.1f} oracle={bss_score(y[va], oracle_shift(ps_, r)[0]):.1f} | R-only oracle={bss_score(y[va][isR], oracle_shift(ps_[isR], rr)[0]):.1f}', flush=True)
np.save(os.path.join(EXP, 'oof', f'cat_off_{vs}.npy'), ps_)

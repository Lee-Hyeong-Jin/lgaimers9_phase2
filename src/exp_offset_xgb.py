"""XGBoost multiclass with base_margin from the Bayes posterior (GPU)."""
import sys, os, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, xgboost as xgb
from scipy.special import softmax
from exp2 import fold_data, feature_sets, oracle_shift
from features import _recover_indicators, RATE_COLS_P
from features_dev import bayes_features, BAYES_HP
from common import load_train, bss_score, EXP
vs = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
tr = load_train(); y = tr.control_success.values; seas = tr.season.values
F, stats = fold_data(tr, vs, 'prev'); fs = feature_sets(F); feats = fs['V1']
ind = _recover_indicators(tr, 'pitcher_id', 'asof_pitcher_n', RATE_COLS_P)
far = (1 - ind['s'] - ind['rev'] - ind['mid']).clip(0, 1)
y4 = np.stack([ind['s'].values, ind['rev'].values, ind['mid'].values, far.values], 1).argmax(1)
bay = bayes_features(tr, F, stats, **BAYES_HP)
mu = dict(zip(stats['league'].season, stats['league'].league_rate)); mu_row = pd.Series(seas).map(mu).fillna(mu[max(mu)]).values
pm = np.clip(mu_row + bay['bayes_dev'], 0.05, 0.95)
tr_idx = np.where(seas < vs)[0]; va = np.where(seas == vs)[0]; r = y[va].mean(); isR = tr.game_type.values[va] == 'R'
fshare = np.bincount(y4[tr_idx], minlength=4)[1:].astype(float); fshare /= fshare.sum()
init = np.stack([np.log(pm)] + [np.log((1 - pm) * fshare[k]) for k in range(3)], 1)
wts = np.exp(-0.15 * (vs - 1 - seas[tr_idx])).astype(float)
ps_ = np.zeros(len(va))
for sd in (0, 1):
    p = {'objective': 'multi:softprob', 'num_class': 4, 'eta': 0.02, 'max_depth': 6, 'min_child_weight': 500, 'subsample': 0.8, 'colsample_bytree': 0.6, 'reg_lambda': 10.0, 'tree_method': 'hist', 'device': 'cuda', 'seed': sd, 'max_bin': 256}
    dtr = xgb.DMatrix(F[feats].iloc[tr_idx], y4[tr_idx], weight=wts, base_margin=init[tr_idx]); dva = xgb.DMatrix(F[feats].iloc[va], base_margin=init[va])
    m = xgb.train(p, dtr, num_boost_round=500); ps_ += m.predict(dva)[:, 0] / 2
rr = y[va][isR].mean()
print(f'[fold {vs}] XGB multiclass + bayes base_margin (2 seeds): raw={bss_score(y[va], ps_):.1f} oracle={bss_score(y[va], oracle_shift(ps_, r)[0]):.1f} | R-only oracle={bss_score(y[va][isR], oracle_shift(ps_[isR], rr)[0]):.1f}', flush=True)
np.save(os.path.join(EXP, 'oof', f'xgb_off_{vs}.npy'), ps_)

"""Binary LGB with init_score = logit(bayes posterior mean): trees model deviations from the explicit level estimate."""
import sys, os, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.special import logit, expit
from exp2 import fold_data, feature_sets, oracle_shift
from features import CAT_FEATURES
from features_dev import bayes_features
from common import load_train, bss_score, EXP
from exp import DEFAULT_PARAMS
vs = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
tr = load_train(); y = tr.control_success.values; seas = tr.season.values
F, stats = fold_data(tr, vs, 'prev'); fs = feature_sets(F); feats = fs['V1']
cat = [c for c in CAT_FEATURES + ['home_team'] if c in feats]
bay = bayes_features(tr, F, stats)
mu = dict(zip(stats['league'].season, stats['league'].league_rate)); mu_row = pd.Series(seas).map(mu).fillna(mu[max(mu)]).values
pm = np.clip(mu_row + bay['bayes_dev'], 0.05, 0.95)
tr_idx = np.where(seas < vs)[0]; va = np.where(seas == vs)[0]; r = y[va].mean()
wts = np.exp(-0.15 * (vs - 1 - seas[tr_idx])).astype(float)
p = {**DEFAULT_PARAMS, 'objective': 'binary', 'num_threads': 4}
for name, init in [('binary ref (no offset)', None), ('binary + bayes offset', logit(pm))]:
    ps_ = np.zeros(len(va))
    for sd in (0, 1):
        dtr = lgb.Dataset(F[feats].iloc[tr_idx], y[tr_idx], weight=wts, init_score=None if init is None else init[tr_idx], categorical_feature=cat)
        m = lgb.train({**p, 'seed': sd}, dtr, num_boost_round=180)
        raw = m.predict(F[feats].iloc[va], raw_score=True) + (0 if init is None else init[va]); ps_ += expit(raw) / 2
    print(f'[fold {vs}] {name:26s} raw={bss_score(y[va], ps_):.1f} oracle={bss_score(y[va], oracle_shift(ps_, r)[0]):.1f}', flush=True)
    np.save(os.path.join(EXP, 'oof', f'bin_{"off" if init is not None else "ref"}_{vs}.npy'), ps_)

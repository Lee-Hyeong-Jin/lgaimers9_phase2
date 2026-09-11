import sys, os, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, lightgbm as lgb
from exp2 import fold_data, feature_sets, oracle_shift
from features import _recover_indicators, RATE_COLS_P, CAT_FEATURES
from features_dev import bayes_features, batter_bayes_features
from common import load_train, bss_score, EXP
from exp import DEFAULT_PARAMS
vs = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
tr = load_train(); y = tr.control_success.values; seas = tr.season.values
F, stats = fold_data(tr, vs, 'prev'); fs = feature_sets(F); feats = fs['V1']
cat = [c for c in CAT_FEATURES + ['home_team'] if c in feats]
ind = _recover_indicators(tr, 'pitcher_id', 'asof_pitcher_n', RATE_COLS_P)
far = (1 - ind['s'] - ind['rev'] - ind['mid']).clip(0, 1)
y4 = np.stack([ind['s'].values, ind['rev'].values, ind['mid'].values, far.values], 1).argmax(1)
bay = bayes_features(tr, F, stats); bb = batter_bayes_features(tr, F, stats)
for k, v in {**bay, **bb}.items(): F[k] = v
tr_idx = np.where(seas < vs)[0]; va = np.where(seas == vs)[0]; r = y[va].mean(); isR = tr.game_type.values[va] == 'R'
wts = np.exp(-0.15 * (vs - 1 - seas[tr_idx])).astype(float)
p = {**DEFAULT_PARAMS, 'objective': 'multiclass', 'num_class': 4, 'num_threads': 6}
def run(fl, name):
    ps_ = np.zeros(len(va))
    for sd in (0, 1):
        m = lgb.train({**p, 'seed': sd, 'bagging_seed': sd, 'feature_fraction_seed': sd}, lgb.Dataset(F[fl].iloc[tr_idx], y4[tr_idx], weight=wts, categorical_feature=cat), num_boost_round=180)
        ps_ += m.predict(F[fl].iloc[va])[:, 0] / 2
    rr = y[va][isR].mean()
    print(f'[fold {vs}] {name:40s} all: raw={bss_score(y[va], ps_):.1f} oracle={bss_score(y[va], oracle_shift(ps_, r)[0]):.1f} | R-only oracle={bss_score(y[va][isR], oracle_shift(ps_[isR], rr)[0]):.1f}', flush=True)
    return ps_
p1 = run(feats + list(bay), 'absolute view + pitcher bayes')
p2 = run(feats + list(bay) + list(bb), 'absolute view + pitcher&batter bayes')
np.save(os.path.join(EXP, 'oof', f'mcs_absbayes_{vs}.npy'), p1); np.save(os.path.join(EXP, 'oof', f'mcs_absbayes2_{vs}.npy'), p2)

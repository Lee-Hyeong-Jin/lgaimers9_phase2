"""Factorized model: P(success|x) = sum_t P(t|x) * P(success|t,x), with pitch-type group t observed in training
(recovered from cumulative pitch-mix rates) but latent at test time."""
import sys, os, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, lightgbm as lgb
from exp2 import fold_data, feature_sets, oracle_shift
from features import _recover_indicators, RATE_COLS_P, CAT_FEATURES
from common import load_train, bss_score, EXP
from exp import DEFAULT_PARAMS
vs = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
tr = load_train(); y = tr.control_success.values; seas = tr.season.values
F, stats = fold_data(tr, vs, 'prev'); fs = feature_sets(F); feats = fs['V1']
cat = [c for c in CAT_FEATURES + ['home_team'] if c in feats]
ind = _recover_indicators(tr, 'pitcher_id', 'asof_pitcher_n', RATE_COLS_P)
far = (1 - ind['s'] - ind['rev'] - ind['mid']).clip(0, 1)
y4 = np.stack([ind['s'].values, ind['rev'].values, ind['mid'].values, far.values], 1).argmax(1)
ptype = np.stack([ind['fb'].values, ind['br'].values, ind['os'].values], 1).argmax(1)
tr_idx = np.where(seas < vs)[0]; va = np.where(seas == vs)[0]; r = y[va].mean(); isR = tr.game_type.values[va] == 'R'
wts = np.exp(-0.15 * (vs - 1 - seas[tr_idx])).astype(float)
base = {**DEFAULT_PARAMS, 'num_threads': 10}
# (1) pitch-type model P(t | x)
pt_params = {**base, 'objective': 'multiclass', 'num_class': 3}
Pt = np.zeros((len(va), 3))
for sd in (0, 1):
    m = lgb.train({**pt_params, 'seed': sd}, lgb.Dataset(F[feats].iloc[tr_idx], ptype[tr_idx], weight=wts, categorical_feature=cat), num_boost_round=150)
    Pt += m.predict(F[feats].iloc[va]) / 2
acc = (Pt.argmax(1) == ptype[va]).mean(); print(f'[fold {vs}] pitch-type model: val accuracy={acc:.3f}, mean P(fb)={Pt[:,0].mean():.3f} (true {np.mean(ptype[va]==0):.3f})', flush=True)
# (2) outcome model with TRUE type as input (train), marginalized at test
F2 = F.copy(); F2['ptype'] = ptype
oc_params = {**base, 'objective': 'multiclass', 'num_class': 4}
Ps = np.zeros(len(va))
for sd in (0, 1):
    m = lgb.train({**oc_params, 'seed': sd}, lgb.Dataset(F2[feats + ['ptype']].iloc[tr_idx], y4[tr_idx], weight=wts, categorical_feature=cat + ['ptype']), num_boost_round=180)
    Xv = F2[feats + ['ptype']].iloc[va].copy(); ps_t = np.zeros(len(va))
    for t in range(3):
        Xv['ptype'] = t; ps_t += Pt[:, t] * m.predict(Xv)[:, 0]
    Ps += ps_t / 2
rr = y[va][isR].mean()
print(f'[fold {vs}] factorized (type-conditional, marginalized): raw={bss_score(y[va], Ps):.1f} oracle={bss_score(y[va], oracle_shift(Ps, r)[0]):.1f} | R-only oracle={bss_score(y[va][isR], oracle_shift(Ps[isR], rr)[0]):.1f}', flush=True)
np.save(os.path.join(EXP, 'oof', f'mcs_factorized_{vs}.npy'), Ps)

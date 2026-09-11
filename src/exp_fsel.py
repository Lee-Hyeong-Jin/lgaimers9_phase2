import sys, os, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, lightgbm as lgb
from exp2 import fold_data, feature_sets, oracle_shift
from features import _recover_indicators, RATE_COLS_P, CAT_FEATURES
from common import load_train, bss_score
from exp import DEFAULT_PARAMS
tr = load_train(); y = tr.control_success.values
ind = _recover_indicators(tr, 'pitcher_id', 'asof_pitcher_n', RATE_COLS_P)
far = (1 - ind['s'] - ind['rev'] - ind['mid']).clip(0, 1)
y4 = np.stack([ind['s'].values, ind['rev'].values, ind['mid'].values, far.values], 1).argmax(1)
F, _ = fold_data(tr, 2024, 'prev'); fs = feature_sets(F); feats = [c for c in fs['V1'] if not c.startswith(('p_prior_ew', 'p_prior2', 'p_last_minus_prior2', 'p_ew05'))]
s = tr.season.values; tr_idx = np.where(s < 2024)[0]; va_idx = np.where(s == 2024)[0]; r = y[va_idx].mean()
wts = np.exp(-0.15 * (2023 - s[tr_idx])).astype(float)
p = {**DEFAULT_PARAMS, 'objective': 'multiclass', 'num_class': 4, 'num_threads': 10}
def run(fl, name):
    cat = [c for c in CAT_FEATURES + ['home_team'] if c in fl]; ps = np.zeros(len(va_idx)); imp = None
    for sd in (0, 1):
        m = lgb.train({**p, 'seed': sd, 'bagging_seed': sd, 'feature_fraction_seed': sd}, lgb.Dataset(F[fl].iloc[tr_idx], y4[tr_idx], weight=wts, categorical_feature=cat), num_boost_round=180)
        ps += m.predict(F[fl].iloc[va_idx])[:, 0] / 2
        imp = pd.Series(m.feature_importance('gain'), index=fl) if imp is None else imp + pd.Series(m.feature_importance('gain'), index=fl)
    print(f'{name:30s} nfeat={len(fl):3d} raw={bss_score(y[va_idx], ps):.1f} oracle={bss_score(y[va_idx], oracle_shift(ps, r)[0]):.1f}', flush=True)
    return imp
imp = run(feats, 'reference')
order = imp.sort_values()
for frac in (0.3, 0.5):
    drop = set(order.index[:int(len(order) * frac)])
    run([c for c in feats if c not in drop], f'drop bottom {int(frac*100)}% by gain')

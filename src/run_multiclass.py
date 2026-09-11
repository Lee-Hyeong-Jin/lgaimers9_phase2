import sys, os, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, lightgbm as lgb, time
from exp2 import fold_data, feature_sets, oracle_shift
from features import _recover_indicators, RATE_COLS_P, CAT_FEATURES
from common import load_train, bss_score, log_experiment
from exp import DEFAULT_PARAMS
tr = load_train()
ind = _recover_indicators(tr, 'pitcher_id', 'asof_pitcher_n', RATE_COLS_P)
far = (1 - ind['s'] - ind['rev'] - ind['mid']).clip(0, 1)
Y4 = np.stack([ind['s'].values, ind['rev'].values, ind['mid'].values, far.values], 1)
y4 = Y4.argmax(1)  # 0=success 1=rev 2=mid 3=far
print('class dist', np.bincount(y4) / len(y4))
y = tr.control_success.values
for vs in (2024, 2023):
    F, _ = fold_data(tr, vs, 'prev'); fs = feature_sets(F); feats = fs['V1']
    cat = [c for c in CAT_FEATURES if c in feats]
    tr_idx = np.where(tr.season.values < vs)[0]; va_idx = np.where(tr.season.values == vs)[0]
    params = {**DEFAULT_PARAMS, 'objective': 'multiclass', 'num_class': 4, 'num_threads': 5}
    dtr = lgb.Dataset(F[feats].iloc[tr_idx], y4[tr_idx], categorical_feature=cat)
    t = time.time()
    m = lgb.train(params, dtr, num_boost_round=400)
    r = y[va_idx].mean()
    for it in (100, 150, 200, 300, 400):
        P = m.predict(F[feats].iloc[va_idx], num_iteration=it)
        p = P[:, 0]
        po, _ = oracle_shift(p, r)
        print(f'multiclass fold {vs} iter {it}: score={bss_score(y[va_idx], p):.1f} oracle={bss_score(y[va_idx], po):.1f} pm={p.mean():.4f} ({time.time()-t:.0f}s)', flush=True)
        log_experiment(f'exp030_multiclass_{vs}_it{it}', {'score': bss_score(y[va_idx], p), 'oracle': bss_score(y[va_idx], po)})

import sys, os, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, lightgbm as lgb
from exp2 import fold_data, feature_sets, oracle_shift
from features import _recover_indicators, RATE_COLS_P, CAT_FEATURES
from common import load_train, bss_score
from exp import DEFAULT_PARAMS
tr = load_train(); y = tr.control_success.values
ind = _recover_indicators(tr, 'pitcher_id', 'asof_pitcher_n', RATE_COLS_P)
far = (1 - ind['s'] - ind['rev'] - ind['mid']).clip(0, 1)
y4 = np.stack([ind['s'].values, ind['rev'].values, ind['mid'].values, far.values], 1).argmax(1)
F, _ = fold_data(tr, 2024, 'prev'); fs = feature_sets(F); feats = fs['V1']
cat = [c for c in CAT_FEATURES + ['home_team'] if c in feats]
isF = (tr.game_type.values == 'F'); s = tr.season.values
va = np.where((s == 2024) & isF)[0]; yv = y[va]; r = yv.mean()
joint = np.load('experiments/oof/mcs_pb_2024.npy')[isF[s == 2024]]
print(f'F rows 2024: n={len(va)} joint-model local: raw={bss_score(yv, joint):.1f} oracle={bss_score(yv, oracle_shift(joint, r)[0]):.1f}')
for name, trmask, rounds in [('F-only all seasons', (s < 2024) & isF, 150), ('F-only 2023 (new regime)', (s == 2023) & isF, 80), ('F-only 2023 + 2020-22 wt0.3', (s < 2024) & isF, 150)]:
    tr_idx = np.where(trmask)[0]
    w = None
    if 'wt0.3' in name: w = np.where(s[tr_idx] >= 2023, 1.0, 0.3)
    p = {**DEFAULT_PARAMS, 'objective': 'multiclass', 'num_class': 4, 'num_threads': 8, 'min_data_in_leaf': 200, 'num_leaves': 31}
    ps = np.zeros(len(va))
    for sd in (0, 1):
        m = lgb.train({**p, 'seed': sd}, lgb.Dataset(F[feats].iloc[tr_idx], y4[tr_idx], weight=w, categorical_feature=cat), num_boost_round=rounds)
        ps += m.predict(F[feats].iloc[va])[:, 0] / 2
    print(f'{name:32s}: raw={bss_score(yv, ps):.1f} oracle={bss_score(yv, oracle_shift(ps, r)[0]):.1f}  | blend w joint 50/50 oracle={bss_score(yv, oracle_shift((ps+joint)/2, r)[0]):.1f}')

import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import lightgbm as lgb
from exp import run, RAW_FEATS, get_fold_data, DEFAULT_PARAMS
from features import add_change_features
from common import load_train, bss_score, brier
import exp as E
tr = load_train()
# patch cache with change features
for vs in (2024, 2023):
    F, stats = get_fold_data(tr, vs)
    E._CACHE[vs] = (add_change_features(F), stats)
F, _ = E._CACHE[2024]
ALL = list(F.columns)
CHG = [c for c in ALL if c.startswith(('p_chg','b_chg','drift_est','p_cur_logn','b_cur_logn'))]
BASE = [c for c in ALL if c not in CHG]
ABS = ['season','pitcher_id','batter_id']
V1 = [c for c in BASE if c!='season']
V2 = [c for c in BASE if c not in ABS]
V3 = V2 + CHG
V3s = V3 + ['season']
res = {}
res['v1_noseason'],_ = run('exp002_noseason', V1)
res['v2_noabs'],_ = run('exp003_noabs', V2)
res['v3_noabs_chg'], oofs = run('exp004_noabs_chg', V3, save_oof=True)
res['v3s_chg_season'],_ = run('exp005_chg_season', V3s)
# recency: train from 2021 only
res['v3_recent'],_ = run('exp006_noabs_chg_recent2021', V3, min_train_season=2021)
# residual by month for v3 fold 2024
p = oofs[2024]; va = tr[tr.season==2024].reset_index(drop=True)
df = pd.DataFrame({'y':va.control_success, 'p':p, 'm':va.game_month, 'cur_n':F.loc[tr.season.values==2024,'p_cur_n'].values})
print("fold2024 residual by month:\n", df.groupby('m').agg(y=('y','mean'), p=('p','mean'), n=('y','size')).round(4).to_string())
df['nb'] = pd.cut(df.cur_n, [-1,0,20,50,100,200,500,1000,1e9])
print("by cur_n bucket:\n", df.groupby('nb').agg(y=('y','mean'), p=('p','mean'), n=('y','size')).round(4).to_string())

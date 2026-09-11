import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, lightgbm as lgb
from exp import get_fold_data, DEFAULT_PARAMS, RAW_FEATS
from features import add_change_features, CAT_FEATURES
from common import load_train, bss_score, brier
tr = load_train(); y = tr.control_success.values
params = {**DEFAULT_PARAMS, 'num_threads': 7}
for vs in (2024, 2023):
    F, stats = get_fold_data(tr, vs); F = add_change_features(F)
    feats = list(F.columns)
    cat = [c for c in CAT_FEATURES if c in feats]
    tr_idx = np.where(tr.season.values < vs)[0]; va_idx = np.where(tr.season.values == vs)[0]
    dtr = lgb.Dataset(F.iloc[tr_idx], y[tr_idx], categorical_feature=cat)
    m = lgb.train(params, dtr, num_boost_round=1500)
    yv = y[va_idx]; r = yv.mean()
    rows = []
    for it in [50, 100, 150, 200, 300, 400, 600, 800, 1000, 1500]:
        p = m.predict(F.iloc[va_idx], num_iteration=it)
        shift = r - p.mean()
        # oracle: shift on logit scale to match mean
        from scipy.special import logit, expit
        lp = logit(np.clip(p, 1e-6, 1-1e-6))
        lo, hi = -1, 1
        for _ in range(40):
            mid = (lo+hi)/2
            if expit(lp+mid).mean() < r: lo = mid
            else: hi = mid
        po = expit(lp + (lo+hi)/2)
        rows.append({'iter': it, 'raw': bss_score(yv, p), 'oracle_shift': bss_score(yv, po), 'pred_mean': p.mean(), 'shift_logit': (lo+hi)/2})
    print(f"=== fold {vs} (y_mean={r:.4f}) ===")
    print(pd.DataFrame(rows).round(4).to_string())
    # calibration vs cur rate buckets for iteration 300
    p = m.predict(F.iloc[va_idx], num_iteration=300)
    Fv = F.iloc[va_idx]
    df = pd.DataFrame({'y': yv, 'p': p, 'cur': Fv.p_cur_s_rate.values, 'prior': Fv.p_prior_s_rate.values, 'cur_n': Fv.p_cur_n.values})
    df = df[(df.cur_n>=500)&(df.prior.notna())]
    df['cb'] = pd.cut(df.cur - df.prior, [-1,-0.08,-0.05,-0.03,-0.01,0.01,0.03,0.05,0.08,1])
    print("cur_n>=500: by (cur - prior) bucket:\n", df.groupby('cb').agg(n=('y','size'), y=('y','mean'), p=('p','mean'), cur=('cur','mean'), prior=('prior','mean')).round(4).to_string())

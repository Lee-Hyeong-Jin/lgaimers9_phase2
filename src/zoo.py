"""Model zoo: train a model type on temporal folds at fixed rounds, save OOF preds for blending.
usage: python src/zoo.py --model lgb_bin --name zoo_lgb_bin --rounds 170 --seeds 0,1,2 --feats V1
"""
import sys, os, json, time, argparse, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.special import expit, logit
from exp2 import fold_data, feature_sets, oracle_shift
from features import _recover_indicators, RATE_COLS_P, CAT_FEATURES
from common import load_train, bss_score, log_experiment, EXP
from exp import DEFAULT_PARAMS

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)  # lgb_bin | lgb_mc | xgb | cat
ap.add_argument('--name', required=True)
ap.add_argument('--feats', default='V1')
ap.add_argument('--rounds', type=int, default=170)
ap.add_argument('--seeds', default='0')
ap.add_argument('--params', default='{}')
ap.add_argument('--folds', default='2024,2023')
ap.add_argument('--threads', type=int, default=7)
ap.add_argument('--wexp', type=float, default=0.0)
ap.add_argument('--cat_ids', action='store_true')
ap.add_argument('--cat_ids_p', action='store_true')  # pitcher_id only as categorical
ap.add_argument('--cat_combo', action='store_true')  # add pitcher x {batter hand, count_state, inning bucket} keys as categorical
ap.add_argument('--h2', action='store_true')  # 2024-H2 fold: add first half of val season rows (row order) to training, validate on second half
ap.add_argument('--lgb_cat_ids', action='store_true')
ap.add_argument('--min_train_season', type=int, default=2019)
ap.add_argument('--drop', default='')  # comma list of feature-name prefixes to drop
ap.add_argument('--add', default='')   # comma list of extra feature names to add
ap.add_argument('--oldF', default='keep')  # keep | drop | w0.3 : handling of old-regime (season<2023) Futures rows in training  # treat pitcher_id/batter_id as categorical (CatBoost)  # recency weight exp(-wexp*(max_train_season-season))
a = ap.parse_args()
tr = load_train(); y = tr.control_success.values
OOF = os.path.join(EXP, 'oof'); os.makedirs(OOF, exist_ok=True)
y4 = None
NCLS = 4
if a.model in ('lgb_mc', 'lgb_mc12pt', 'lgb_mc12bs', 'xgb_mc', 'cat_mc', 'lgb_mc3a', 'lgb_mc3b', 'lgb_mc5'):
    ind = _recover_indicators(tr, 'pitcher_id', 'asof_pitcher_n', RATE_COLS_P)
    far = (1 - ind['s'] - ind['rev'] - ind['mid']).clip(0, 1)
    y4 = np.stack([ind['s'].values, ind['rev'].values, ind['mid'].values, far.values], 1).argmax(1)
    if a.model == 'lgb_mc5':    # success / rev-only / mid-only / rev&mid / far
        rv = np.rint(ind['rev'].values); md = np.rint(ind['mid'].values); sc = np.rint(ind['s'].values)
        y4 = np.where(sc == 1, 0, np.where((rv == 1) & (md == 1), 3, np.where(rv == 1, 1, np.where(md == 1, 2, 4)))); NCLS = 5
    if a.model == 'lgb_mc3a':   # success / mid / (rev+far)
        y4 = np.where(y4 == 3, 1, y4); NCLS = 3
    elif a.model == 'lgb_mc3b':  # success / rev / (mid+far)
        y4 = np.where(y4 == 3, 2, y4); NCLS = 3
    if a.model == 'lgb_mc12pt':
        ypt = np.stack([ind['fb'].values, ind['br'].values, ind['os'].values], 1).argmax(1)
        y4 = y4 * 3 + ypt; NCLS = 12
    elif a.model == 'lgb_mc12bs':
        ybs = np.stack([ind['ball'].values, ind['strike'].values, (1 - ind['ball'] - ind['strike']).clip(0, 1).values], 1).argmax(1)
        y4 = y4 * 3 + ybs; NCLS = 12
res = {'model': a.model, 'rounds': a.rounds, 'folds': {}}
for vs in [int(x) for x in a.folds.split(',')]:
    F, _ = fold_data(tr, vs, 'prev'); fs = feature_sets(F)
    if a.feats in fs: feats = fs[a.feats]
    elif a.feats == 'V1TM': feats = [c for c in fs['V1'] if not c.startswith('ROLE_')]
    elif a.feats == 'V1ONLY': feats = [c for c in fs['V1'] if not c.startswith(('ROLE_', 'TM_'))]
    elif a.feats == 'NOCUR': feats = [c for c in fs['V1'] if not (c.startswith(('p_cur', 'b_cur', 'p_chg', 'b_chg', 'drift_est', 'p_form')) or c in ('p_cur_s',))]
    elif a.feats == 'NOPRIOR': feats = [c for c in fs['V1'] if not (c.startswith(('p_prior', 'b_prior', 'p_last', 'b_last', 'p_n_seasons', 'b_n_seasons', 'TM_', 'ROLE_')) or c.endswith(('_shr100', '_shr500', '_shr300', '_shr200')))]
    else: feats = a.feats.split(',')
    if a.drop: feats = [c for c in feats if not c.startswith(tuple(a.drop.split(',')))]
    if a.add: feats = feats + [c for c in a.add.split(',') if c and c not in feats and c in F.columns]
    cat = [c for c in CAT_FEATURES + ['home_team'] if c in feats] + (['pitcher_id', 'batter_id'] if a.lgb_cat_ids else [])
    if a.cat_combo:
        pid = F['pitcher_id'].values.astype(np.int64)
        F['pk_hand'] = pid * 2 + (F['batter_hand'].values == 1).astype(np.int64)
        F['pk_cnt'] = pid * 16 + F['count_state'].astype('category').cat.codes.values.astype(np.int64)
        F['pk_inn'] = pid * 4 + np.clip((F['inning'].values - 1) // 3, 0, 3).astype(np.int64)
        feats = feats + ['pk_hand', 'pk_cnt', 'pk_inn']
    tr_idx = np.where((tr.season.values < vs) & (tr.season.values >= a.min_train_season))[0]; va_idx = np.where(tr.season.values == vs)[0]
    if a.h2: k = len(va_idx) // 2; tr_idx = np.concatenate([tr_idx, va_idx[:k]]); va_idx = va_idx[k:]
    ftag = f'{vs}H2' if a.h2 else f'{vs}'
    oldF = (tr.game_type.values[tr_idx] == 'F') & (tr.season.values[tr_idx] < 2023)
    if a.oldF == 'drop': tr_idx = tr_idx[~oldF]
    X = F[feats]; Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
    ps = np.zeros(len(va_idx)); t = time.time()
    for s in [int(x) for x in a.seeds.split(',')]:
        prm = json.loads(a.params)
        if a.model == 'lgb_bin':
            import lightgbm as lgb
            p = {**DEFAULT_PARAMS, 'num_threads': a.threads, 'seed': s, 'bagging_seed': s, 'feature_fraction_seed': s, **prm}
            m = lgb.train(p, lgb.Dataset(Xtr, y[tr_idx], categorical_feature=cat), num_boost_round=a.rounds)
            ps += m.predict(Xva)
        elif a.model in ('lgb_mc', 'lgb_mc12pt', 'lgb_mc12bs', 'lgb_mc3a', 'lgb_mc3b', 'lgb_mc5'):
            import lightgbm as lgb
            p = {**DEFAULT_PARAMS, 'objective': 'multiclass', 'num_class': NCLS, 'num_threads': a.threads, 'seed': s, 'bagging_seed': s, 'feature_fraction_seed': s, **prm}
            wts = np.ones(len(tr_idx)) if a.wexp == 0 else np.exp(-a.wexp * (tr.season.values[tr_idx].max() - tr.season.values[tr_idx])).astype(float)
            if a.oldF.startswith('w'): wts = wts * np.where((tr.game_type.values[tr_idx] == 'F') & (tr.season.values[tr_idx] < 2023), float(a.oldF[1:]), 1.0)
            m = lgb.train(p, lgb.Dataset(Xtr, y4[tr_idx], weight=wts, categorical_feature=cat), num_boost_round=a.rounds)
            pr = m.predict(Xva)
            ps += pr[:, 0] if NCLS in (3, 4, 5) else pr[:, :3].sum(1)
        elif a.model in ('xgb', 'xgb_mc'):
            import xgboost as xgb
            p = {'objective': 'binary:logistic', 'eta': 0.03, 'max_depth': 7, 'min_child_weight': 200, 'subsample': 0.8, 'colsample_bytree': 0.6, 'reg_lambda': 10.0, 'tree_method': 'hist', 'device': 'cuda', 'seed': s, 'max_bin': 256, **prm}
            if a.model == 'xgb_mc':
                p.update({'objective': 'multi:softprob', 'num_class': 4})
                dtr = xgb.DMatrix(Xtr, y4[tr_idx]); dva = xgb.DMatrix(Xva)
                m = xgb.train(p, dtr, num_boost_round=a.rounds)
                ps += m.predict(dva)[:, 0]
            else:
                dtr = xgb.DMatrix(Xtr, y[tr_idx]); dva = xgb.DMatrix(Xva)
                m = xgb.train(p, dtr, num_boost_round=a.rounds)
                ps += m.predict(dva)
        elif a.model in ('cat', 'cat_mc', 'cat_bin_ids'):
            from catboost import CatBoostClassifier
            cf = [c for c in cat if c in feats] + (['pitcher_id', 'batter_id'] if (a.cat_ids or a.model == 'cat_bin_ids') else []) + (['pitcher_id'] if a.cat_ids_p else []) + (['pk_hand', 'pk_cnt', 'pk_inn'] if a.cat_combo else [])
            Xtr2 = Xtr.copy(); Xva2 = Xva.copy()
            for c in cf: Xtr2[c] = Xtr2[c].astype(int); Xva2[c] = Xva2[c].astype(int)
            if prm.get('combos'):
                for X2 in (Xtr2, Xva2):
                    X2['pc_combo'] = X2['pitcher_id'].astype(int) * 100 + X2['count_state'].astype(int)
                    X2['ph_combo'] = X2['pitcher_id'].astype(int) * 10 + X2['batter_hand'].astype(int)
                    X2['pg_combo'] = X2['pitcher_id'].astype(int) * 10 + X2['game_type_F'].astype(int)
                    X2['bc_combo'] = X2['batter_id'].astype(int) * 100 + X2['count_state'].astype(int)
                cf = cf + ['pc_combo', 'ph_combo', 'pg_combo', 'bc_combo']
            kw = dict(iterations=a.rounds, learning_rate=prm.get('lr', 0.05), depth=prm.get('depth', 7), l2_leaf_reg=prm.get('l2', 10), random_seed=s, verbose=0, task_type=prm.get('task', 'GPU'), border_count=prm.get('border', 128), thread_count=a.threads)
            if prm.get('ctrc'): kw['max_ctr_complexity'] = prm['ctrc']
            if prm.get('ordered'): kw['boosting_type'] = 'Ordered'
            if prm.get('subsample'): kw['bootstrap_type'] = 'Bernoulli'; kw['subsample'] = prm['subsample']
            if prm.get('rs') is not None: kw['random_strength'] = prm['rs']
            if prm.get('grow'): kw['grow_policy'] = prm['grow']; kw['max_leaves'] = prm.get('max_leaves', 63) if prm['grow'] == 'Lossguide' else None; kw = {k: v for k, v in kw.items() if v is not None}
            swt = None if a.wexp == 0 else np.exp(-a.wexp * (tr.season.values[tr_idx].max() - tr.season.values[tr_idx])).astype(float)
            if os.environ.get('F_W'): swt = (np.ones(len(tr_idx)) if swt is None else swt) * np.where(tr.game_type.values[tr_idx] == 'F', float(os.environ['F_W']), 1.0)
            if a.model == 'cat_mc':
                m = CatBoostClassifier(loss_function='MultiClass', **kw); m.fit(Xtr2, y4[tr_idx], cat_features=cf, sample_weight=swt)
                ps += m.predict_proba(Xva2)[:, 0]
            else:
                m = CatBoostClassifier(**kw); m.fit(Xtr2, y[tr_idx], cat_features=cf)
                ps += m.predict_proba(Xva2)[:, 1]
        print(f'  seed {s} done ({time.time()-t:.0f}s)', flush=True)
    ps /= len(a.seeds.split(','))
    r = y[va_idx].mean(); po, sh = oracle_shift(ps, r)
    sc, so = bss_score(y[va_idx], ps), bss_score(y[va_idx], po)
    res['folds'][ftag] = {'score': sc, 'oracle': so, 'pm': float(ps.mean()), 'ym': float(r), 'time': time.time() - t}
    print(f'[{a.name}] fold {ftag}: score={sc:.1f} oracle={so:.1f} pm={ps.mean():.4f} ym={r:.4f} ({time.time()-t:.0f}s)', flush=True)
    np.save(os.path.join(OOF, f'{a.name}_{ftag}.npy'), ps)
log_experiment(a.name, res)

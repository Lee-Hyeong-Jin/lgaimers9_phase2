"""Experiment runner v2: v2 features, optional level offset (init_score), oracle-shift diagnostics."""
import sys, os, time, json, warnings
warnings.filterwarnings('ignore')
import os
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.special import logit, expit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_train, bss_score, brier, log_experiment, EXP
import features2 as f2
from features import CAT_FEATURES, add_change_features
from exp import DEFAULT_PARAMS

_CACHE = {}


def fold_data(tr, vs, mode='prev'):
    key = (vs, mode)
    if key not in _CACHE:
        stats = f2.build_stats(tr[tr.season < vs])
        F = f2.build_features(tr, stats, mode=mode)
        F = add_change_features(F)
        _CACHE[key] = (F, stats)
    return _CACHE[key]


def oracle_shift(p, r):
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6)); lo, hi = -2, 2
    for _ in range(50):
        mid = (lo + hi) / 2
        if expit(lp + mid).mean() < r: lo = mid
        else: hi = mid
    return expit(lp + (lo + hi) / 2), (lo + hi) / 2


def run2(name, feats, mode='prev', offset=False, params=None, folds=(2024, 2023), num_rounds=3000, early_stop=300,
         min_train_season=2019, weight_fn=None, iters_grid=(100, 200, 300, 500, 800, 1200), cat=None, save_oof=False,
         tr=None, verbose=True):
    tr = load_train() if tr is None else tr
    params = {**DEFAULT_PARAMS, **(params or {})}
    cat = [c for c in (CAT_FEATURES + ['home_team'] if cat is None else cat) if c in feats]
    y = tr.control_success.values
    res = {'feats': len(feats), 'mode': mode, 'offset': offset, 'params': params, 'folds': {}}
    oofs = {}
    for vs in folds:
        F, stats = fold_data(tr, vs, mode)
        tr_idx = np.where((tr.season.values < vs) & (tr.season.values >= min_train_season))[0]
        va_idx = np.where(tr.season.values == vs)[0]
        X = F[feats]
        init_tr = logit(F['L_ref'].values[tr_idx]) if offset else None
        init_va = logit(F['L_ref'].values[va_idx]) if offset else np.zeros(len(va_idx))
        w = None if weight_fn is None else weight_fn(tr.iloc[tr_idx], F.iloc[tr_idx])
        dtr = lgb.Dataset(X.iloc[tr_idx], y[tr_idx], weight=w, init_score=init_tr, categorical_feature=cat, free_raw_data=False)
        dva = lgb.Dataset(X.iloc[va_idx], y[va_idx], init_score=init_va if offset else None, reference=dtr, categorical_feature=cat)
        t = time.time()
        m = lgb.train(params, dtr, num_boost_round=num_rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(0)])
        yv = y[va_idx]; r = yv.mean()
        grid = {}
        for it in list(iters_grid) + [m.best_iteration]:
            if it > m.current_iteration(): continue
            raw = m.predict(X.iloc[va_idx], num_iteration=it, raw_score=True) + init_va
            p = expit(raw)
            po, sh = oracle_shift(p, r)
            grid[it] = {'raw': round(bss_score(yv, p), 1), 'oracle': round(bss_score(yv, po), 1), 'pm': round(float(p.mean()), 4), 'shift': round(sh, 4)}
        p = expit(m.predict(X.iloc[va_idx], num_iteration=m.best_iteration, raw_score=True) + init_va)
        po, sh = oracle_shift(p, r)
        res['folds'][vs] = {'score': bss_score(yv, p), 'oracle': bss_score(yv, po), 'best_iter': m.best_iteration,
                            'pred_mean': float(p.mean()), 'y_mean': float(r), 'grid': grid, 'time': time.time() - t}
        oofs[vs] = p
        if verbose:
            print(f"[{name}] fold {vs}: score={res['folds'][vs]['score']:.1f} oracle={res['folds'][vs]['oracle']:.1f} iter={m.best_iteration} "
                  f"pm={p.mean():.4f} ym={r:.4f} ({time.time()-t:.0f}s)  grid={ {k: (v['raw'], v['oracle']) for k, v in grid.items()} }", flush=True)
        if vs == folds[0]:
            imp = pd.Series(m.feature_importance('gain'), index=feats).sort_values(ascending=False)
            res['importance_top'] = imp.head(30).round(0).to_dict()
    res['mean_score'] = float(np.mean([f['score'] for f in res['folds'].values()]))
    res['mean_oracle'] = float(np.mean([f['oracle'] for f in res['folds'].values()]))
    print(f"[{name}] MEAN score={res['mean_score']:.1f} oracle={res['mean_oracle']:.1f}", flush=True)
    log_experiment(name, res)
    if save_oof:
        os.makedirs(os.path.join(EXP, 'oof'), exist_ok=True)
        for vs, p in oofs.items():
            np.save(os.path.join(EXP, 'oof', f'{name}_{vs}.npy'), p)
    return res, oofs


def feature_sets(F):
    ALL = list(F.columns)
    ABS_PRIOR = [c for c in ALL if c.startswith('p_prior_') and c.endswith('_rate')] + ['p_last_rate', 'b_last_rate'] + \
                [c for c in ALL if c.startswith('b_prior_') and c.endswith('_rate')] + ['p_prior_F_rate', 'p_prior_R_rate', 'p_prior_vsL_rate', 'p_prior_vsR_rate', 'p_prior_vsHand_rate']
    ABS_PRIOR = sorted(set(ABS_PRIOR))
    ABS_ASOF = ['asof_p_success_rate', 'asof_b_success_rate', 'p_all_s_shr200', 'b_all_s_shr300', 'p_cur_s_shr100', 'p_cur_s_shr500', 'b_cur_s_shr300']
    V1 = [c for c in ALL if not (c.startswith('P_') or c.startswith('B_') or (c.startswith('CTX_') and os.environ.get('CTX_FEATS') != '1') or c in ('L_ref', 'L_all_ref', 'p_cur_dev', 'b_cur_dev', 'p_cur_dev_shr200', 'b_cur_dev_shr300', 'p_cur_minus_skill', 'b_cur_minus_skill', 'p_est', 'b_est', 'home_team', 'p_prior_logn', 'b_prior_logn'))]
    return {'ALL': ALL, 'V1': V1, 'ABS_PRIOR': ABS_PRIOR, 'ABS_ASOF': ABS_ASOF}

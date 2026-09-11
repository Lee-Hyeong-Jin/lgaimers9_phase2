"""Experiment runner: temporal folds, LightGBM, logging."""
import sys, os, time, json, argparse
import numpy as np, pandas as pd
import lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_train, bss_score, brier, log_experiment, EXP
from features import build_stats, build_features, CAT_FEATURES
from cv import temporal_folds

RAW_FEATS = ['season', 'game_month', 'game_dow', 'inning', 'is_top', 'game_type_F', 'balls', 'strikes', 'outs',
             'run_top', 'run_bot', 'run_total', 'score_diff_home', 'score_diff_p', 'r1', 'r2', 'r3', 'n_runners',
             'base_state', 'home_we', 'li', 'pitcher_id', 'batter_id', 'pitcher_hand', 'batter_hand',
             'pitcher_team', 'batter_team', 'asof_p_n', 'asof_p_success_rate', 'asof_p_reverse_rate',
             'asof_p_middle_rate', 'asof_p_ball_rate', 'asof_p_strike_rate', 'asof_p_prev1_game_success_rate',
             'asof_p_prev3_game_success_rate', 'asof_p_prev5_game_success_rate', 'asof_p_prev1_game_middle_rate',
             'asof_p_prev3_game_middle_rate', 'asof_p_prev5_game_middle_rate', 'asof_b_n', 'asof_b_success_rate',
             'asof_b_middle_rate', 'asof_p_fastball_rate', 'asof_p_breaking_rate', 'asof_p_offspeed_rate']

DEFAULT_PARAMS = dict(objective='binary', learning_rate=0.03, num_leaves=63, min_data_in_leaf=500,
                      feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                      verbose=-1, num_threads=14, seed=42)

_CACHE = {}


def get_fold_data(tr, vs, feats_fn=build_features):
    """Build features for fold: stats from seasons < vs; features for all rows (train rows use their own prior seasons)."""
    key = vs
    if key in _CACHE:
        return _CACHE[key]
    stats = build_stats(tr[tr.season < vs])
    F = feats_fn(tr, stats)
    _CACHE[key] = (F, stats)
    return F, stats


def run(name, feats, params=None, folds=(2024, 2023), num_rounds=5000, early_stop=200, weight_fn=None,
        min_train_season=2019, tr=None, cat=None, verbose=True, save_oof=False):
    tr = load_train() if tr is None else tr
    params = {**DEFAULT_PARAMS, **(params or {})}
    cat = [c for c in (CAT_FEATURES if cat is None else cat) if c in feats]
    res = {'feats': len(feats), 'params': params, 'folds': {}}
    oofs = {}
    for vs in folds:
        F, stats = get_fold_data(tr, vs)
        tr_idx = np.where((tr.season.values < vs) & (tr.season.values >= min_train_season))[0]
        va_idx = np.where(tr.season.values == vs)[0]
        X = F[feats]
        y = tr.control_success.values
        w = None if weight_fn is None else weight_fn(tr.iloc[tr_idx])
        dtr = lgb.Dataset(X.iloc[tr_idx], y[tr_idx], weight=w, categorical_feature=cat, free_raw_data=False)
        dva = lgb.Dataset(X.iloc[va_idx], y[va_idx], reference=dtr, categorical_feature=cat)
        t = time.time()
        m = lgb.train(params, dtr, num_boost_round=num_rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(0)])
        p = m.predict(X.iloc[va_idx], num_iteration=m.best_iteration)
        sc = bss_score(y[va_idx], p)
        res['folds'][vs] = {'score': sc, 'brier': brier(y[va_idx], p), 'best_iter': m.best_iteration,
                            'pred_mean': float(p.mean()), 'y_mean': float(y[va_idx].mean()), 'time': time.time() - t}
        oofs[vs] = p
        if verbose:
            print(f"[{name}] fold {vs}: score={sc:.1f} brier={res['folds'][vs]['brier']:.5f} iter={m.best_iteration} "
                  f"pred_mean={p.mean():.4f} y_mean={y[va_idx].mean():.4f} ({time.time()-t:.0f}s)", flush=True)
        if vs == folds[0]:
            imp = pd.Series(m.feature_importance('gain'), index=feats).sort_values(ascending=False)
            res['importance_top'] = imp.head(25).round(0).to_dict()
    res['mean_score'] = float(np.mean([f['score'] for f in res['folds'].values()]))
    print(f"[{name}] MEAN score = {res['mean_score']:.1f}", flush=True)
    log_experiment(name, res)
    if save_oof:
        os.makedirs(os.path.join(EXP, 'oof'), exist_ok=True)
        for vs, p in oofs.items():
            np.save(os.path.join(EXP, 'oof', f'{name}_{vs}.npy'), p)
    return res, oofs

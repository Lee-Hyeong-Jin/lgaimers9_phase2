"""Stage-1 pitcher current-level estimator.
Snapshots: for each training pitch row (pitcher, season, cur_n) we already have per-row cur/prior/form features.
Target for a snapshot = success rate of the pitcher's REMAINING pitches in that season (excluding the current pitch).
We train a regressor on subsampled rows (to keep it fast), then use its prediction as a feature for the pitch model.
For test rows the same function applies (row's own features only)."""
import sys, os, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, lightgbm as lgb

LEVEL_FEATS = ['p_cur_n', 'p_cur_s_rate', 'p_cur_rev_rate', 'p_cur_mid_rate', 'p_cur_ball_rate', 'p_cur_strike_rate',
               'p_prior_n', 'p_prior_s_rate', 'p_prior_rev_rate', 'p_prior_mid_rate', 'p_last_rate', 'p_last_n', 'p_last_gap', 'p_n_seasons',
               'asof_p_prev1_game_success_rate', 'asof_p_prev3_game_success_rate', 'asof_p_prev5_game_success_rate',
               'asof_p_prev1_game_middle_rate', 'asof_p_prev5_game_middle_rate', 'p_form1', 'p_form5', 'p_form5_prior',
               'p_prior_F_share', 'p_prior_R_rate', 'p_prior_F23_rate', 'p_cur_fb_rate', 'p_cur_br_rate', 'p_cur_os_rate',
               'p_prior_srate_fb', 'p_prior_srate_br', 'p_prior_srate_os', 'p_mix_exp_rate', 'ROLE_start_share', 'ROLE_ppg',
               'TM_fb_speed', 'TM_rel_consistency_h', 'TM_rel_consistency_s', 'TM_minor_share', 'game_month', 'season', 'game_type_F']


def remaining_rate_target(tr):
    """y_rem = (season total successes - cumulative through this pitch) / (season total n - cumulative n), per pitcher-season."""
    d = pd.DataFrame({'p': tr.pitcher_id.values, 's': tr.season.values, 'y': tr.control_success.values.astype(float)})
    d['ord'] = np.arange(len(d))
    g = d.groupby(['p', 's'])
    tot_s = g.y.transform('sum'); tot_n = g.y.transform('size')
    cum_s = g.y.cumsum(); cum_n = g.cumcount() + 1  # inclusive of the current pitch
    rem_n = tot_n - cum_n; rem_s = tot_s - cum_s
    with np.errstate(invalid='ignore', divide='ignore'):
        y_rem = np.where(rem_n >= 30, rem_s / rem_n, np.nan)
    return y_rem, rem_n.values


def fit_level_model(F, tr, train_mask, seed=0, n_sub=400000, rounds=400):
    y_rem, rem_n = remaining_rate_target(tr)
    idx = np.where(train_mask & ~np.isnan(y_rem))[0]
    rng = np.random.RandomState(seed); idx = rng.choice(idx, min(n_sub, len(idx)), replace=False)
    feats = [c for c in LEVEL_FEATS if c in F.columns]
    params = dict(objective='regression', learning_rate=0.03, num_leaves=31, min_data_in_leaf=300, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=10, seed=seed)
    w = np.sqrt(rem_n[idx])  # weight snapshots by remaining sample size (label precision)
    m = lgb.train(params, lgb.Dataset(F[feats].iloc[idx], y_rem[idx], weight=w), num_boost_round=rounds)
    return m, feats


if __name__ == '__main__':
    from exp2 import fold_data, feature_sets, oracle_shift
    from features import _recover_indicators, RATE_COLS_P, CAT_FEATURES
    from common import load_train, bss_score
    from exp import DEFAULT_PARAMS
    tr = load_train(); y = tr.control_success.values
    F, _ = fold_data(tr, 2024, 'prev'); fs = feature_sets(F); feats = fs['V1']
    cat = [c for c in CAT_FEATURES + ['home_team'] if c in feats]
    ind = _recover_indicators(tr, 'pitcher_id', 'asof_pitcher_n', RATE_COLS_P)
    far = (1 - ind['s'] - ind['rev'] - ind['mid']).clip(0, 1)
    y4 = np.stack([ind['s'].values, ind['rev'].values, ind['mid'].values, far.values], 1).argmax(1)
    s = tr.season.values; tr_idx = np.where(s < 2024)[0]; va_idx = np.where(s == 2024)[0]; r = y[va_idx].mean()
    # stage 1: level model trained on seasons < 2024 (target uses only those seasons' future pitches)
    y_rem, rem_n = remaining_rate_target(tr)
    # season-wise out-of-fold stage-1 predictions for training rows; val rows use a model trained on all seasons < 2024
    F['p_level_est'] = np.nan
    for ss in sorted(set(s[s < 2024])):
        lm, lfeats = fit_level_model(F, tr, (s < 2024) & (s != ss))
        F.loc[s == ss, 'p_level_est'] = lm.predict(F.loc[s == ss, lfeats])
        print('  oof season', ss, 'done', flush=True)
    lm, lfeats = fit_level_model(F, tr, s < 2024)
    F.loc[s == 2024, 'p_level_est'] = lm.predict(F.loc[s == 2024, lfeats])
    F['p_level_minus_cur'] = F['p_level_est'] - F['p_cur_s_rate']
    F['p_level_minus_prior'] = F['p_level_est'] - F['p_prior_s_rate']
    vm = va_idx[~np.isnan(y_rem[va_idx])]
    print('stage1 level model: val corr(pred, remaining rate) =', round(np.corrcoef(F['p_level_est'].values[vm], y_rem[vm])[0, 1], 4),
          ' vs cur_rate corr =', round(np.corrcoef(np.nan_to_num(F['p_cur_s_rate'].values[vm], nan=0.5), y_rem[vm])[0, 1], 4), flush=True)
    # stage 2: pitch model with/without the level feature (2 seeds)
    p = {**DEFAULT_PARAMS, 'objective': 'multiclass', 'num_class': 4, 'num_threads': 10}
    wts = np.exp(-0.15 * (2023 - s[tr_idx])).astype(float)
    for name, fl in [('ref', feats), ('+level', feats + ['p_level_est', 'p_level_minus_cur', 'p_level_minus_prior'])]:
        ps = np.zeros(len(va_idx))
        for sd in (0, 1):
            m = lgb.train({**p, 'seed': sd, 'bagging_seed': sd, 'feature_fraction_seed': sd}, lgb.Dataset(F[fl].iloc[tr_idx], y4[tr_idx], weight=wts, categorical_feature=cat), num_boost_round=180)
            ps += m.predict(F[fl].iloc[va_idx])[:, 0] / 2
        print(f'{name:8s} raw={bss_score(y[va_idx], ps):.1f} oracle={bss_score(y[va_idx], oracle_shift(ps, r)[0]):.1f}', flush=True)
        np.save(f'experiments/oof/mcs_level_{name}_2024.npy', ps)

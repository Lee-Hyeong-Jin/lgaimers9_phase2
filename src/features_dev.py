"""Season-invariant 'deviation view': subtract the training-season league mean of each rate-type feature.
For rows of seasons beyond training (val/test), the last training season's mean is used (persistence)."""
import numpy as np, pandas as pd
from bayes_level import season_priors, row_posterior

BAYES_HP = dict(a=0.6, tau2=0.004, v_new=0.004)

SUCCESS_RATE_FEATS = ['p_cur_s_rate', 'p_prior_s_rate', 'p_last_rate', 'asof_p_success_rate', 'p_cur_s_shr100', 'p_cur_s_shr500',
                      'p_all_s_shr200', 'b_cur_s_rate', 'b_prior_s_rate', 'b_last_rate', 'asof_b_success_rate', 'b_cur_s_shr300',
                      'b_all_s_shr300', 'asof_p_prev1_game_success_rate', 'asof_p_prev3_game_success_rate', 'asof_p_prev5_game_success_rate',
                      'p_mix_exp_rate', 'p_prior_srate_fb', 'p_prior_srate_br', 'p_prior_srate_os', 'pb_prior_rate', 'pb_prior_shr30',
                      'p_prior_F_rate', 'p_prior_R_rate', 'p_prior_F23_rate', 'p_prior_vsL_rate', 'p_prior_vsR_rate', 'p_prior_vsHand_rate']
MODE_RATE_FEATS = {'rev': ['p_cur_rev_rate', 'p_prior_rev_rate', 'asof_p_reverse_rate'],
                   'mid': ['p_cur_mid_rate', 'p_prior_mid_rate', 'asof_p_middle_rate', 'b_cur_mid_rate', 'b_prior_mid_rate', 'asof_b_middle_rate',
                           'asof_p_prev1_game_middle_rate', 'asof_p_prev3_game_middle_rate', 'asof_p_prev5_game_middle_rate'],
                   'ball': ['p_cur_ball_rate', 'p_prior_ball_rate', 'asof_p_ball_rate'],
                   'strike': ['p_cur_strike_rate', 'p_prior_strike_rate', 'asof_p_strike_rate']}


def season_means(train, ind):
    """Per-season league means of success and failure-mode/ball/strike rates from training rows."""
    d = pd.DataFrame({'season': train.season.values, 's': train.control_success.values.astype(float), 'rev': ind['rev'].values,
                      'mid': ind['mid'].values, 'ball': ind['ball'].values, 'strike': ind['strike'].values})
    return d.groupby('season').mean()


def deviation_view(F, seasons, means, bayes=None):
    """Return a copy of F with rate features replaced by deviations from the season mean (+ optional bayes features)."""
    F = F.copy(); seasons = np.asarray(seasons)
    max_s = means.index.max()
    key = np.minimum(seasons, max_s)
    for col in SUCCESS_RATE_FEATS:
        if col in F: F[col] = F[col].values - means['s'].reindex(key).values
    for mode, cols in MODE_RATE_FEATS.items():
        for col in cols:
            if col in F: F[col] = F[col].values - means[mode].reindex(key).values
    if bayes is not None:
        for k, v in bayes.items(): F[k] = v
    return F


def bayes_features(df, F, stats, a=0.6, tau2=0.004, v_new=0.004, src=False):
    ps = stats['ps'][['pitcher_id', 'season', 'n', 's'] + (['n_LG', 's_LG', 'n_F', 's_F'] if src else [])].copy()
    league = dict(zip(stats['league'].season, stats['league'].league_rate))
    if src:
        # remove label-source effects from pitcher-season successes so the state-space level is source-free
        eff = stats['src_eff']; mx = int(eff.index.max())
        def _e(col, seasons): return eff[col].reindex(np.minimum(np.asarray(seasons, dtype=int), mx)).values
        nB = ps.n.values - ps.n_LG.values - ps.n_F.values
        ps['s'] = ps.s.values - ps.n_LG.values * _e('eff_LG', ps.season) - ps.n_F.values * _e('eff_F', ps.season) - nB * _e('eff_B', ps.season)
    tab, mu = season_priors(ps[['pitcher_id', 'season', 'n', 's']], league, a, tau2, v_new)
    if not src:
        pm, pv, m0, v0 = row_posterior(df, F, tab, mu, v_new)
        mu_row = pd.Series(np.asarray(df.season)).map(mu).values
        return {'bayes_dev': pm - mu_row, 'bayes_pv': pv, 'bayes_m0': m0, 'bayes_v0': v0}
    # season-to-date successes: subtract the pitcher's expected source effect (typical source mix), then add the row's source effect back
    seas = np.asarray(df.season); is_lgp = (np.asarray(df.pitcher_team_id) == 13)
    fsh = np.nan_to_num(F['p_prior_F_share'].values, nan=0.0) if 'p_prior_F_share' in F else np.zeros(len(df))
    share_F = np.clip(fsh, 0.0, 1.0); share_LG = np.where(is_lgp, 1.0 - share_F, 0.11 * (1.0 - share_F)); share_B = np.clip(1.0 - share_LG - share_F, 0.0, 1.0)
    e_LG, e_F, e_B = _e('eff_LG', seas), _e('eff_F', seas), _e('eff_B', seas)
    exp_eff = share_LG * e_LG + share_F * e_F + share_B * e_B
    F2 = F[['p_cur_n', 'p_cur_s']].copy(); F2['p_cur_s'] = F2['p_cur_s'].values - np.maximum(F2['p_cur_n'].values, 0.0) * exp_eff
    pm, pv, m0, v0 = row_posterior(df, F2, tab, mu, v_new)
    row_src = np.where(np.asarray(df.game_type) == 'F', e_F, np.where(is_lgp | (np.asarray(df.batter_team_id) == 13), e_LG, e_B))
    mu_row = pd.Series(seas).map(mu).values
    return {'bayes_dev': pm - mu_row + row_src, 'bayes_pv': pv, 'bayes_m0': m0, 'bayes_v0': v0}


def batter_bayes_features(df, F, stats, a=0.6, tau2=0.001, v_new=0.0015):
    """Dynamic-Bayesian posterior of the BATTER's current-season deviation (same machinery as pitchers, bs table)."""
    bs = stats['bs'][['batter_id', 'season', 'n', 's']].rename(columns={'batter_id': 'pitcher_id'})
    league = dict(zip(stats['league'].season, stats['league'].league_rate))
    tab, mu = season_priors(bs, league, a, tau2, v_new)
    d = df[['batter_id', 'season']].reset_index(drop=True).rename(columns={'batter_id': 'pitcher_id'}).merge(tab, on=['pitcher_id', 'season'], how='left')
    m0 = d.prior_m.fillna(0.0).values; v0 = d.prior_v.fillna(v_new).values
    mu_row = pd.Series(np.asarray(df.season)).map(mu).values
    n = F['b_cur_n'].values; s = np.nan_to_num(F['b_cur_s_rate'].values * n)
    prec = 1.0 / v0 + n / 0.25
    m1 = (m0 / v0 + (s - n * mu_row) / 0.25) / prec; v1 = 1.0 / prec
    return {'bbayes_dev': m1, 'bbayes_pv': v1, 'bbayes_m0': m0, 'bbayes_v0': v0}

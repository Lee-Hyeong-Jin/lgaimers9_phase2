"""Explicit dynamic-Bayesian estimator of each pitcher's current-season control level (probability scale, normal approx).
State: deviation d_{p,s} = theta_{p,s} - mu_s. Transition: d_s = a * d_{s-1} + eps, eps ~ N(0, tau2) (per gap year).
Observation within season: successes ~ Binomial(n, mu_s + d) -> normal approx with per-pitch variance sig2.
Per-row posterior uses ONLY the row's own season-to-date (n, s) and the pitcher's prior-season totals (training stats)."""
import sys, os, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd

SIG2 = 0.25


def season_priors(ps, league, a, tau2, v_new, max_season=2026):
    """Per (pitcher, season y): prior mean/var of the deviation BEFORE observing season y, from recursion over prior seasons."""
    seasons = np.arange(int(ps.season.min()), max_season + 1)
    pit = np.sort(ps.pitcher_id.unique())
    n_w = ps.pivot(index='pitcher_id', columns='season', values='n').reindex(index=pit, columns=seasons).fillna(0.0).values
    s_w = ps.pivot(index='pitcher_id', columns='season', values='s').reindex(index=pit, columns=seasons).fillna(0.0).values
    mu = np.array([league.get(int(y), np.nan) for y in seasons])
    # league mean for seasons beyond training: persistence of the last known
    last_mu = mu[~np.isnan(mu)][-1]; mu = np.where(np.isnan(mu), last_mu, mu)
    m = np.zeros(len(pit)); v = np.full(len(pit), v_new); started = np.zeros(len(pit), bool)
    prior_m = np.zeros_like(n_w); prior_v = np.zeros_like(n_w)
    for j, y in enumerate(seasons):
        prior_m[:, j] = m; prior_v[:, j] = v
        n = n_w[:, j]; s = s_w[:, j]; obs = n > 0
        # update with this season's totals (posterior at end of season)
        prec = 1.0 / v + np.where(obs, n / SIG2, 0.0)
        mean_obs = np.where(obs, (s - n * mu[j]) / np.maximum(n, 1), 0.0)
        m_post = (m / v + np.where(obs, mean_obs * n / SIG2, 0.0)) / prec; v_post = 1.0 / prec
        started |= obs
        # transition to next season (only for pitchers that have started; others keep the v_new prior)
        m = np.where(started, a * m_post, 0.0); v = np.where(started, a * a * v_post + tau2, v_new)
    tab = pd.DataFrame({'pitcher_id': np.repeat(pit, len(seasons)), 'season': np.tile(seasons, len(pit)),
                        'prior_m': prior_m.ravel(), 'prior_v': prior_v.ravel()})
    return tab, dict(zip(seasons.tolist(), mu.tolist()))


def row_posterior(df, F, tab, mu_by_season, v_new):
    d = df[['pitcher_id', 'season']].reset_index(drop=True).merge(tab, on=['pitcher_id', 'season'], how='left')
    m0 = d.prior_m.fillna(0.0).values; v0 = d.prior_v.fillna(v_new).values
    mu = pd.Series(df.season.values).map(mu_by_season).values
    n = np.maximum(F['p_cur_n'].values, 0.0); s = np.clip(np.nan_to_num(F['p_cur_s'].values), 0.0, np.maximum(n, 0.0))  # guard: s in [0, n]
    prec = 1.0 / v0 + n / SIG2
    m1 = (m0 / v0 + (s - n * mu) / SIG2) / prec; v1 = 1.0 / prec
    return mu + m1, v1, m0, v0


if __name__ == '__main__':
    import lightgbm as lgb
    from exp2 import fold_data, feature_sets, oracle_shift
    from features import _recover_indicators, RATE_COLS_P, CAT_FEATURES
    from common import load_train, bss_score
    from exp import DEFAULT_PARAMS
    tr = load_train(); y = tr.control_success.values; seas = tr.season.values
    F, stats = fold_data(tr, 2024, 'prev'); fs = feature_sets(F)
    ps = stats['ps'][['pitcher_id', 'season', 'n', 's']]
    league = dict(zip(stats['league'].season, stats['league'].league_rate))
    tr_idx = np.where((seas >= 2020) & (seas < 2024))[0]; va_idx = np.where(seas == 2024)[0]; r = y[va_idx].mean()
    # --- fit hyperparameters on training seasons 2020-2023 (Brier of the pure level predictor) ---
    best = None
    for a in (0.6, 0.75, 0.85, 0.95, 1.0):
        for tau2 in (0.0003, 0.001, 0.002, 0.004):
            for v_new in (0.002, 0.004, 0.008):
                tab, mu = season_priors(ps, league, a, tau2, v_new)
                pm, pv, _, _ = row_posterior(tr, F, tab, mu, v_new)
                b = np.mean((np.clip(pm[tr_idx], 0.01, 0.99) - y[tr_idx]) ** 2)
                if best is None or b < best[0]: best = (b, a, tau2, v_new)
    b, a, tau2, v_new = best
    print(f'fitted hyperparameters: a={a} tau2={tau2} v_new={v_new} (train Brier {b:.5f})', flush=True)
    tab, mu = season_priors(ps, league, a, tau2, v_new)
    pm, pv, m0, v0 = row_posterior(tr, F, tab, mu, v_new)
    pv_ = np.clip(pm[va_idx], 0.01, 0.99)
    print(f'Bayes level-only predictor, fold 2024: raw={bss_score(y[va_idx], pv_):.1f} oracle={bss_score(y[va_idx], oracle_shift(pv_, r)[0]):.1f}', flush=True)
    # --- LGB multiclass with pitcher-level features only (same information) ---
    ind = _recover_indicators(tr, 'pitcher_id', 'asof_pitcher_n', RATE_COLS_P)
    far = (1 - ind['s'] - ind['rev'] - ind['mid']).clip(0, 1)
    y4 = np.stack([ind['s'].values, ind['rev'].values, ind['mid'].values, far.values], 1).argmax(1)
    LEVEL = ['season', 'p_cur_n', 'p_cur_s_rate', 'p_cur_s', 'p_prior_n', 'p_prior_s_rate', 'p_last_rate', 'p_last_n', 'p_last_gap', 'p_n_seasons', 'p_cur_s_shr100', 'p_cur_s_shr500', 'p_all_s_shr200', 'asof_p_success_rate', 'asof_p_n']
    tr_all = np.where(seas < 2024)[0]
    p = {**DEFAULT_PARAMS, 'objective': 'multiclass', 'num_class': 4, 'num_threads': 6}
    ps_ = np.zeros(len(va_idx))
    for sd in (0, 1):
        m = lgb.train({**p, 'seed': sd}, lgb.Dataset(F[LEVEL].iloc[tr_all], y4[tr_all]), num_boost_round=150)
        ps_ += m.predict(F[LEVEL].iloc[va_idx])[:, 0] / 2
    print(f'LGB pitcher-level-only features, fold 2024: raw={bss_score(y[va_idx], ps_):.1f} oracle={bss_score(y[va_idx], oracle_shift(ps_, r)[0]):.1f}', flush=True)
    # --- LGB level-only + Bayes posterior features ---
    F['bayes_pm'] = pm; F['bayes_pv'] = pv; F['bayes_m0'] = m0; F['bayes_v0'] = v0
    ps2 = np.zeros(len(va_idx))
    for sd in (0, 1):
        m = lgb.train({**p, 'seed': sd}, lgb.Dataset(F[LEVEL + ['bayes_pm', 'bayes_pv', 'bayes_m0', 'bayes_v0']].iloc[tr_all], y4[tr_all]), num_boost_round=150)
        ps2 += m.predict(F[LEVEL + ['bayes_pm', 'bayes_pv', 'bayes_m0', 'bayes_v0']].iloc[va_idx])[:, 0] / 2
    print(f'LGB level-only + Bayes features, fold 2024: raw={bss_score(y[va_idx], ps2):.1f} oracle={bss_score(y[va_idx], oracle_shift(ps2, r)[0]):.1f}', flush=True)
    np.save('experiments/oof/bayes_level_2024.npy', pv_)
    # how well does each track the pitcher's realized 2024 rate? (per pitcher-season, n>=200)
    d = pd.DataFrame({'pid': tr.pitcher_id.values[va_idx], 'y': y[va_idx], 'bayes': pv_, 'lgb': ps_})
    g = d.groupby('pid').agg(n=('y', 'size'), y=('y', 'mean'), bayes=('bayes', 'mean'), lgb=('lgb', 'mean')); g = g[g.n >= 200]
    print(f'per-pitcher (n>=200, {len(g)} pitchers): corr(actual, bayes mean)={np.corrcoef(g.y, g.bayes)[0,1]:.3f}  corr(actual, lgb mean)={np.corrcoef(g.y, g.lgb)[0,1]:.3f}')

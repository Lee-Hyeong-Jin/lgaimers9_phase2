"""Feature engineering.
import warnings; warnings.filterwarnings("ignore")

Two stages so that validation and inference share identical code paths:
  1. build_stats(train_df): per-(pitcher, season) / per-(batter, season) totals computed
     from official training rows only (seasons < val season for validation, all seasons for test).
  2. build_features(df, stats): per-row features using ONLY the row's own columns + stats.
     No cross-row information from `df` is used (rule: each test row predicted independently).

Key insight: asof_* rates are exact cumulative means over the entity's whole history (2019-),
so  cur_count = asof_n - prior_seasons_count  and  cur_success = asof_n*asof_rate - prior_success
recover the entity's *current-season-to-date* record exactly from a single row.
"""
import os
import numpy as np
import pandas as pd

RATE_COLS_P = {  # asof rate column -> short name (all share denominator asof_pitcher_n)
    'asof_pitcher_success_rate': 's', 'asof_pitcher_reverse_rate': 'rev',
    'asof_pitcher_middle_rate': 'mid', 'asof_pitcher_ball_rate': 'ball',
    'asof_pitcher_strike_rate': 'strike', 'asof_pitcher_fastball_rate': 'fb',
    'asof_pitcher_breaking_rate': 'br', 'asof_pitcher_offspeed_rate': 'os',
}
RATE_COLS_B = {'asof_batter_success_rate': 's', 'asof_batter_middle_rate': 'mid'}
P_KEYS = list(RATE_COLS_P.values())
B_KEYS = list(RATE_COLS_B.values())
MAX_SEASON = 2026


def _recover_indicators(df, id_col, n_col, rate_cols, target='control_success'):
    """Recover per-pitch 0/1 indicators from consecutive cumulative rates (sorted by id, row order).
    The entity's final pitch has no successor -> imputed with expectations."""
    d = df[[id_col, n_col, target] + list(rate_cols)].copy()
    d['_ord'] = np.arange(len(d))
    d = d.sort_values([id_col, '_ord'])
    n = d[n_col].values.astype(float)
    out = {}
    nxt_id_same = (d[id_col].shift(-1) == d[id_col]).values
    for c, k in rate_cols.items():
        cum = np.nan_to_num(d[c].values * n)          # count through previous pitch
        cum_next = np.roll(cum, -1)
        ind = np.where(nxt_id_same, np.rint(cum_next - cum), np.nan)
        out[k] = ind
    res = pd.DataFrame(out, index=d.index)
    # impute final pitch of each entity
    y = d[target].values.astype(float)
    last = ~nxt_id_same
    for k in res.columns:
        col = res[k].values
        if k == 's':
            col[last] = y[last]
        elif k in ('rev', 'mid'):
            # failure modes: 0 if success; else conditional expectation given failure
            r = np.nan_to_num(d[[c for c, kk in rate_cols.items() if kk == k][0]].values[last])
            sr = np.nan_to_num(d['asof_pitcher_success_rate'].values[last]) if 'asof_pitcher_success_rate' in d else 0.52
            cond = np.clip(r / np.clip(1 - sr, 1e-3, None), 0, 1)
            col[last] = np.where(y[last] == 1, 0.0, cond)
        else:
            r = np.nan_to_num(d[[c for c, kk in rate_cols.items() if kk == k][0]].values[last])
            col[last] = r
        res[k] = col
    return res.sort_index()


def build_stats(train):
    """Per-(entity, season) totals from training rows."""
    tr = train.reset_index(drop=True)
    pind = _recover_indicators(tr, 'pitcher_id', 'asof_pitcher_n', RATE_COLS_P)
    bind = _recover_indicators(tr, 'batter_id', 'asof_batter_n', RATE_COLS_B)
    p = pd.DataFrame({'pitcher_id': tr.pitcher_id, 'season': tr.season, 'n': 1.0})
    for k in P_KEYS:
        p[k] = pind[k].values
    p['n_F'] = (tr.game_type == 'F').astype(float)
    p['n_LG'] = ((tr.game_type == 'R') & ((tr.pitcher_team_id == 13) | (tr.batter_team_id == 13))).astype(float)   # LG-involved R pitches (label-source regime)
    p['s_LG'] = p['n_LG'] * tr.control_success
    p['n_new'] = (tr.season >= 2023).astype(float); p['s_new'] = p['n_new'] * tr.control_success   # new label regime (2023+) pitches
    p['s_F'] = p['n_F'] * tr.control_success
    p['n_L'] = (tr.batter_hand == 1).astype(float)   # vs batter_hand==1
    p['s_L'] = p['n_L'] * tr.control_success
    p['n_2k'] = (tr.strikes_before == 2).astype(float)   # two-strike counts
    p['s_2k'] = p['n_2k'] * tr.control_success
    p['n_00'] = ((tr.balls_before == 0) & (tr.strikes_before == 0)).astype(float)   # first pitch of PA
    p['s_00'] = p['n_00'] * tr.control_success
    p['n_late'] = (tr.inning >= 7).astype(float); p['s_late'] = p['n_late'] * tr.control_success
    p['n_L2k'] = ((tr.batter_hand == 1) & (tr.strikes_before == 2)).astype(float); p['s_L2k'] = p['n_L2k'] * tr.control_success
    p['rev_L'] = p['n_L'] * p['rev']; p['mid_L'] = p['n_L'] * p['mid']   # class-specific platoon sums
    for k in ('fb', 'br', 'os'):  # pitch-type-specific successes (type recovered from cumulative mix rates)
        p['s_' + k] = p[k] * tr.control_success
    new_regime = (tr.season >= 2023).astype(float)  # Futures measurement regime changed in 2023
    p['n_F23'] = p['n_F'] * new_regime; p['s_F23'] = p['s_F'] * new_regime
    ps = p.groupby(['pitcher_id', 'season']).sum().reset_index()
    b = pd.DataFrame({'batter_id': tr.batter_id, 'season': tr.season, 'n': 1.0})
    for k in B_KEYS:
        b[k] = bind[k].values
    b['n_F'] = (tr.game_type == 'F').astype(float)
    b['s_F'] = b['n_F'] * tr.control_success
    bs = b.groupby(['batter_id', 'season']).sum().reset_index()
    # pitcher x batter matchup per season
    pb = pd.DataFrame({'pb_key': tr.pitcher_id.values.astype(np.int64) * 100000 + tr.batter_id.values.astype(np.int64), 'season': tr.season, 'n': 1.0, 's': tr.control_success.astype(float)})
    pbs = pb.groupby(['pb_key', 'season']).sum().reset_index()
    # league-level per-season rate (for drift-aware priors)
    league = tr.groupby('season').control_success.agg(['mean', 'size']).reset_index()
    league.columns = ['season', 'league_rate', 'league_n']
    # label-source effects per season: LG-involved R / non-LG R / F, relative to the pooled league mean
    lg_r = (tr.game_type == 'R') & ((tr.pitcher_team_id == 13) | (tr.batter_team_id == 13)); f_ = tr.game_type == 'F'
    src = pd.DataFrame({'season': tr.season.values, 'y': tr.control_success.values.astype(float), 'src': np.where(f_, 'F', np.where(lg_r, 'LG', 'B'))})
    lm = src.groupby('season').y.mean(); se = src.groupby(['season', 'src']).y.mean().unstack('src').reindex(columns=['LG', 'B', 'F'])
    src_eff = pd.DataFrame({'eff_LG': se['LG'] - lm, 'eff_B': se['B'] - lm, 'eff_F': se['F'] - lm}).fillna(0.0)
    return {'ps': ps, 'bs': bs, 'pbs': pbs, 'league': league, 'max_season': int(tr.season.max()), 'src_eff': src_eff}


def _prior_table(tab, id_col, cols, max_season=MAX_SEASON, last_col='s'):
    """For each (id, season y) in a dense grid: sums over seasons < y, plus last-active-season info."""
    ids = tab[id_col].unique()
    seasons = np.arange(tab.season.min(), max_season + 1)
    grid = pd.MultiIndex.from_product([ids, seasons], names=[id_col, 'season'])
    t = tab.set_index([id_col, 'season']).reindex(grid).fillna(0.0)
    wide = {c: t[c].unstack('season') for c in cols}
    prior = {}
    for c in cols:
        cs = wide[c].cumsum(axis=1).shift(1, axis=1).fillna(0.0)
        prior['prior_' + c] = cs
    # last active season stats (n, s) and gap
    n_w = wide['n']
    act = n_w.gt(0)
    seas = pd.DataFrame(np.tile(seasons, (len(ids), 1)), index=n_w.index, columns=n_w.columns)
    last_seas = seas.where(act).ffill(axis=1).shift(1, axis=1)
    last_n = n_w.where(act).ffill(axis=1).shift(1, axis=1)
    last_s = wide[last_col].where(act).ffill(axis=1).shift(1, axis=1)
    prior['last_season_gap'] = seas - last_seas
    prior['last_n'] = last_n
    prior['last_s'] = last_s
    prior['n_seasons'] = act.astype(float).cumsum(axis=1).shift(1, axis=1).fillna(0.0)
    out = pd.concat({k: v.stack(dropna=False) for k, v in prior.items()}, axis=1).reset_index()
    return out



def _weighted_prior(tab, id_col, cols, decay, max_season=MAX_SEASON):
    """Recency-weighted sums over seasons < y: sum_k w^(y-1-k) * x_k (w=decay). Returns frame keyed by (id, season)."""
    ids = tab[id_col].unique(); seasons = np.arange(tab.season.min(), max_season + 1)
    grid = pd.MultiIndex.from_product([ids, seasons], names=[id_col, 'season'])
    t = tab.set_index([id_col, 'season']).reindex(grid).fillna(0.0)
    out = {}
    for c in cols:
        w = t[c].unstack('season'); acc = np.zeros(len(w)); res = np.zeros((len(w), len(seasons)))
        for j, y in enumerate(seasons):
            res[:, j] = acc; acc = decay * acc + w[y].values
        out['wprior_' + c] = pd.DataFrame(res, index=w.index, columns=pd.Index(seasons, name='season')).stack()
    return pd.concat(out, axis=1).reset_index()


def build_features(df, stats, league_prior=None):
    """Row-wise features. df: train or test rows (any seasons). Only own columns + stats used."""
    d = df.copy()
    ps, bs = stats['ps'], stats['bs']
    P_COLS = ['n'] + P_KEYS + ['n_F', 's_F', 'n_L', 's_L', 's_fb', 's_br', 's_os', 'n_F23', 's_F23'] + (['n_2k', 's_2k'] if 'n_2k' in ps.columns else []) + (['n_00', 's_00'] if 'n_00' in ps.columns else []) + (['n_late', 's_late', 'n_L2k', 's_L2k'] if 'n_late' in ps.columns else []) + (['rev_L', 'mid_L'] if 'rev_L' in ps.columns else []) + (['n_new', 's_new', 'n_LG', 's_LG'] if 'n_new' in ps.columns else [])
    B_COLS = ['n'] + B_KEYS + ['n_F', 's_F']
    pp = _prior_table(ps, 'pitcher_id', P_COLS)
    bp = _prior_table(bs, 'batter_id', B_COLS)
    d = d.merge(pp, on=['pitcher_id', 'season'], how='left')
    d = d.merge(bp.rename(columns={c: 'b' + c for c in bp.columns if c not in ('batter_id', 'season')}),
                on=['batter_id', 'season'], how='left')
    for c in pp.columns:
        if c.startswith('prior_') or c == 'n_seasons':
            d[c] = d[c].fillna(0.0)
    for c in bp.columns:
        if c.startswith('prior_') or c == 'n_seasons':
            d['b' + c] = d['b' + c].fillna(0.0)

    F = pd.DataFrame(index=d.index)
    # ---------- raw situational ----------
    F['season'] = d.season
    F['game_month'] = d.game_month
    F['game_dow'] = d.game_dayofweek
    F['inning'] = d.inning
    F['is_top'] = (d.top_bottom == 'T').astype(int)
    F['game_type_F'] = (d.game_type == 'F').astype(int)
    F['balls'] = d.balls_before
    F['strikes'] = d.strikes_before
    F['count_state'] = d.balls_before * 3 + d.strikes_before
    F['outs'] = d.outs_before
    F['run_top'] = d.run_top_before
    F['run_bot'] = d.run_bot_before
    F['run_total'] = d.run_total_before
    F['score_diff_home'] = d.score_diff_home
    F['score_diff_p'] = d.score_diff_pitcher_team
    F['abs_score_diff'] = d.score_diff_pitcher_team.abs()
    F['r1'] = d.runner_on_1b; F['r2'] = d.runner_on_2b; F['r3'] = d.runner_on_3b
    F['n_runners'] = d.num_runners_on
    F['base_state'] = d.base_state.map({'___': 0, '1__': 1, '_2_': 2, '__3': 3, '12_': 4, '1_3': 5, '_23': 6, '123': 7}).astype(int)
    F['home_we'] = d.home_win_expectancy
    F['pitcher_we'] = np.where(d.top_bottom == 'T', d.home_win_expectancy, d.away_win_expectancy)
    F['li'] = d.li
    F['pitcher_hand'] = d.pitcher_hand
    F['batter_hand'] = d.batter_hand
    F['platoon_same'] = (d.pitcher_hand == d.batter_hand).astype(int)
    F['pitcher_team'] = d.pitcher_team_id
    F['batter_team'] = d.batter_team_id
    F['pitcher_id'] = d.pitcher_id
    F['batter_id'] = d.batter_id
    # ---------- raw asof ----------
    F['asof_p_n'] = d.asof_pitcher_n
    for c in RATE_COLS_P:
        F[c.replace('asof_pitcher_', 'asof_p_')] = d[c]
    for c in ['asof_pitcher_prev1_game_success_rate', 'asof_pitcher_prev3_game_success_rate',
              'asof_pitcher_prev5_game_success_rate', 'asof_pitcher_prev1_game_middle_rate',
              'asof_pitcher_prev3_game_middle_rate', 'asof_pitcher_prev5_game_middle_rate']:
        F[c.replace('asof_pitcher_', 'asof_p_')] = d[c]
    F['asof_b_n'] = d.asof_batter_n
    for c in RATE_COLS_B:
        F[c.replace('asof_batter_', 'asof_b_')] = d[c]
    # ---------- decomposition: current-season-to-date (cur) vs prior seasons ----------
    pn = d.asof_pitcher_n.values.astype(float)
    F['p_prior_n'] = d.prior_n
    F['p_cur_n'] = np.clip(pn - d.prior_n.values, 0, None)
    cur_n = F['p_cur_n'].values
    for c, k in RATE_COLS_P.items():
        tot = np.nan_to_num(d[c].values * pn)
        cur = tot - d['prior_' + k].values
        with np.errstate(invalid='ignore', divide='ignore'):
            F['p_cur_' + k + '_rate'] = np.where(cur_n > 0, cur / np.maximum(cur_n, 1), np.nan)
            F['p_prior_' + k + '_rate'] = np.where(d.prior_n.values > 0, d['prior_' + k].values / np.maximum(d.prior_n.values, 1), np.nan)
    F['p_cur_s'] = np.nan_to_num(d.asof_pitcher_success_rate.values * pn) - d.prior_s.values
    with np.errstate(invalid='ignore', divide='ignore'):
        F['p_prior_F_share'] = np.where(d.prior_n > 0, d.prior_n_F / d.prior_n, np.nan)
        F['p_prior_F_rate'] = np.where(d.prior_n_F > 0, d.prior_s_F / d.prior_n_F, np.nan)
        F['p_prior_F23_n'] = d.prior_n_F23
        F['p_prior_F23_rate'] = np.where(d.prior_n_F23 >= 20, d.prior_s_F23 / np.maximum(d.prior_n_F23, 1), np.nan)
        F['p_prior_F23_share'] = np.where(d.prior_n > 0, d.prior_n_F23 / np.maximum(d.prior_n, 1), np.nan)
        F['p_prior_R_rate'] = np.where(d.prior_n - d.prior_n_F > 0, (d.prior_s - d.prior_s_F) / (d.prior_n - d.prior_n_F), np.nan)
        F['p_prior_vsL_rate'] = np.where(d.prior_n_L > 0, d.prior_s_L / d.prior_n_L, np.nan)
        F['p_prior_vsR_rate'] = np.where(d.prior_n - d.prior_n_L > 0, (d.prior_s - d.prior_s_L) / (d.prior_n - d.prior_n_L), np.nan)
        F['p_prior_vsHand_rate'] = np.where(d.batter_hand == 1, F['p_prior_vsL_rate'], F['p_prior_vsR_rate'])
        # context-differential priors (shrunk, prior seasons): platoon (vs current batter hand) and two-strike trait
        def _shr_diff(s1, n1, s0, n0, K=1000.0):
            with np.errstate(invalid='ignore', divide='ignore'):
                m1 = s1 / np.maximum(n1, 1); m0 = s0 / np.maximum(n0, 1)
                ne = 1.0 / (1.0 / np.maximum(n1, 1) + 1.0 / np.maximum(n0, 1))
                return np.where((n1 > 0) & (n0 > 0), (m1 - m0) * ne / (ne + K), 0.0)
        pn_, ps_ = d.prior_n.values, d.prior_s.values
        isL = (d.batter_hand.values == 1)
        n1 = np.where(isL, d.prior_n_L.values, pn_ - d.prior_n_L.values); s1 = np.where(isL, d.prior_s_L.values, ps_ - d.prior_s_L.values)
        F['CTX_hand_diff'] = _shr_diff(s1, n1, ps_ - s1, pn_ - n1)
        F['CTX_hand_adj'] = F['CTX_hand_diff'].values * np.where(pn_ > 0, (pn_ - n1) / np.maximum(pn_, 1), 0.5)
        n2 = d.prior_n_2k.values if 'prior_n_2k' in d.columns else np.zeros(len(d)); s2 = d.prior_s_2k.values if 'prior_s_2k' in d.columns else np.zeros(len(d))
        d2 = _shr_diff(s2, n2, ps_ - s2, pn_ - n2); sh2 = np.where(pn_ > 0, n2 / np.maximum(pn_, 1), 0.28)
        F['CTX_2k_diff'] = d2
        F['CTX_2k_adj'] = np.where(d.strikes_before.values == 2, d2 * (1 - sh2), -d2 * sh2)
        if 'prior_n_late' in d.columns and os.environ.get('CTX_V3') == '1':
            K3 = float(os.environ.get('CTX_K', '1000'))
            # recency-weighted platoon differential
            wp = _weighted_prior(ps, 'pitcher_id', ['n', 's', 'n_L', 's_L'], float(os.environ.get('CTX_DECAY', '0.6')))
            dw = d[['pitcher_id', 'season']].merge(wp, on=['pitcher_id', 'season'], how='left').fillna(0.0)
            wn, ws_ = dw.wprior_n.values, dw.wprior_s.values; wnL, wsL = dw.wprior_n_L.values, dw.wprior_s_L.values
            n1w = np.where(isL, wnL, wn - wnL); s1w = np.where(isL, wsL, ws_ - wsL)
            F['CTX_hand_diff_w'] = _shr_diff(s1w, n1w, ws_ - s1w, wn - n1w, K=K3)
            F['CTX_hand_adj_w'] = F['CTX_hand_diff_w'].values * np.where(wn > 0, (wn - n1w) / np.maximum(wn, 1e-9), 0.5)
            # late-inning (>=7) differential
            nl, sl = d.prior_n_late.values, d.prior_s_late.values
            dl = _shr_diff(sl, nl, ps_ - sl, pn_ - nl, K=K3); shl = np.where(pn_ > 0, nl / np.maximum(pn_, 1), 0.33)
            F['CTX_late_diff'] = dl
            F['CTX_late_adj'] = np.where(d.inning.values >= 7, dl * (1 - shl), -dl * shl)
            # LHB & two-strike cell vs rest
            nc, sc_ = d.prior_n_L2k.values, d.prior_s_L2k.values
            dc = _shr_diff(sc_, nc, ps_ - sc_, pn_ - nc, K=K3); shc = np.where(pn_ > 0, nc / np.maximum(pn_, 1), 0.13)
            F['CTX_L2k_diff'] = dc
            F['CTX_L2k_adj'] = np.where(isL & (d.strikes_before.values == 2), dc * (1 - shc), -dc * shc)
        if 'prior_n_new' in d.columns and os.environ.get('LGNEW') == '1':
            # regime-consistent priors: pitcher's rate/share in the 2023+ label regime, and share of old-regime LG pitches in history
            nn_, sn_ = d.prior_n_new.values, d.prior_s_new.values
            F['p_prior_new_n'] = nn_
            F['p_prior_new_rate'] = np.where(nn_ >= 30, sn_ / np.maximum(nn_, 1), np.nan)
            F['p_prior_new_share'] = np.where(pn_ > 0, nn_ / np.maximum(pn_, 1), np.nan)
            old_lg = (d.prior_n_LG.values + d.prior_n_F.values) - (nn_ * 0)  # LG-involved (R) + F pitches, all regimes
            F['p_prior_LG_share'] = np.where(pn_ > 0, old_lg / np.maximum(pn_, 1), np.nan)
        if 'prior_rev_L' in d.columns and os.environ.get('CTX_CLS') == '1':
            # class-specific platoon differentials (reverse / middle rate vs current hand minus other hand)
            for cls_, tot_ in (('rev', 'prior_rev'), ('mid', 'prior_mid')):
                cL = d['prior_' + cls_ + '_L'].values; cT = d[tot_].values
                c1 = np.where(isL, cL, cT - cL)
                F['CTX_hand_' + cls_ + '_diff'] = _shr_diff(c1, n1, cT - c1, pn_ - n1, K=float(os.environ.get('CTX_K', '1000')))
        if 'prior_n_00' in d.columns and os.environ.get('CTX_V2') == '1':
            n0_, s0_ = d.prior_n_00.values, d.prior_s_00.values
            d0 = _shr_diff(s0_, n0_, ps_ - s0_, pn_ - n0_); sh0 = np.where(pn_ > 0, n0_ / np.maximum(pn_, 1), 0.26)
            F['CTX_00_diff'] = d0
            F['CTX_00_adj'] = np.where((d.balls_before.values == 0) & (d.strikes_before.values == 0), d0 * (1 - sh0), -d0 * sh0)
        F['p_last_n'] = d.last_n
        F['p_last_rate'] = np.where(d.last_n > 0, d.last_s / d.last_n, np.nan)
    F['p_last_gap'] = d.last_season_gap
    F['p_n_seasons'] = d.n_seasons
    # pitch-type-specific prior success rates and mix-weighted expected rate
    with np.errstate(invalid='ignore', divide='ignore'):
        base = F['p_prior_s_rate'].values
        mix_exp = np.zeros(len(d)); wsum = np.zeros(len(d))
        for k in ('fb', 'br', 'os'):
            nk = d['prior_' + k].values; sk = d['prior_s_' + k].values
            F['p_prior_srate_' + k] = np.where(nk >= 30, sk / np.maximum(nk, 1), np.nan)
            shr = (sk + 100 * np.nan_to_num(base, nan=0.52)) / (nk + 100)
            mix = F['p_cur_' + k + '_rate'].values
            mix = np.where(np.isnan(mix), np.where(d.prior_n.values > 0, nk / np.maximum(d.prior_n.values, 1), np.nan), mix)
            mix = np.clip(np.nan_to_num(mix, nan=0.0), 0, 1)
            mix_exp += mix * shr; wsum += mix
        F['p_mix_exp_rate'] = np.where(wsum > 0, mix_exp / np.maximum(wsum, 1e-9), np.nan)
        F['p_mix_exp_minus_prior'] = F['p_mix_exp_rate'] - base
    # batter
    bn = d.asof_batter_n.values.astype(float)
    F['b_prior_n'] = d.bprior_n
    F['b_cur_n'] = np.clip(bn - d.bprior_n.values, 0, None)
    bcur_n = F['b_cur_n'].values
    for c, k in RATE_COLS_B.items():
        tot = np.nan_to_num(d[c].values * bn)
        cur = tot - d['bprior_' + k].values
        with np.errstate(invalid='ignore', divide='ignore'):
            F['b_cur_' + k + '_rate'] = np.where(bcur_n > 0, cur / np.maximum(bcur_n, 1), np.nan)
            F['b_prior_' + k + '_rate'] = np.where(d.bprior_n.values > 0, d['bprior_' + k].values / np.maximum(d.bprior_n.values, 1), np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        F['b_last_rate'] = np.where(d.blast_n > 0, d.blast_s / d.blast_n, np.nan)
        F['b_last_n'] = d.blast_n
    F['b_last_gap'] = d.blast_season_gap
    F['b_n_seasons'] = d.bn_seasons
    # ---------- shrunken estimates ----------
    lp = 0.52 if league_prior is None else league_prior
    p_prior_rate = F['p_prior_s_rate'].fillna(lp).values
    for K in (100, 500):
        F[f'p_cur_s_shr{K}'] = (np.nan_to_num(F['p_cur_s'].values) + K * p_prior_rate) / (cur_n + K)
    F['p_all_s_shr200'] = (np.nan_to_num(d.asof_pitcher_success_rate.values * pn) + 200 * lp) / (pn + 200)
    b_prior_rate = F['b_prior_s_rate'].fillna(lp).values
    b_cur_s = np.nan_to_num(d.asof_batter_success_rate.values * bn) - d.bprior_s.values
    F['b_cur_s_shr300'] = (b_cur_s + 300 * b_prior_rate) / (bcur_n + 300)
    F['b_all_s_shr300'] = (np.nan_to_num(d.asof_batter_success_rate.values * bn) + 300 * lp) / (bn + 300)
    return F


CAT_FEATURES = ['pitcher_team', 'batter_team', 'base_state', 'count_state']


def add_change_features(F):
    """Relative (drift-aware) features: current-season-to-date vs prior seasons / last season."""
    F = F.copy()
    F['p_chg_s'] = F['p_cur_s_rate'] - F['p_prior_s_rate']
    F['p_chg_s_last'] = F['p_cur_s_rate'] - F['p_last_rate']
    F['b_chg_s'] = F['b_cur_s_rate'] - F['b_prior_s_rate']
    F['b_chg_s_last'] = F['b_cur_s_rate'] - F['b_last_rate']
    F['p_chg_rev'] = F['p_cur_rev_rate'] - F['p_prior_rev_rate']
    F['p_chg_mid'] = F['p_cur_mid_rate'] - F['p_prior_mid_rate']
    F['b_chg_mid'] = F['b_cur_mid_rate'] - F['b_prior_mid_rate']
    # precision-weighted combined drift estimate from the row's two entities
    wp = F['p_cur_n'] / (F['p_cur_n'] + 200.0)
    wb = F['b_cur_n'] / (F['b_cur_n'] + 200.0)
    cp = F['p_chg_s'].fillna(0.0) * wp
    cb = F['b_chg_s'].fillna(0.0) * wb
    F['drift_est'] = (cp + cb) / (wp + wb + 1e-6)
    F['p_cur_logn'] = np.log1p(F['p_cur_n'])
    F['b_cur_logn'] = np.log1p(F['b_cur_n'])
    # recent form vs season-to-date / prior
    F['p_form1'] = F['asof_p_prev1_game_success_rate'] - F['p_cur_s_rate']
    F['p_form3'] = F['asof_p_prev3_game_success_rate'] - F['p_cur_s_rate']
    F['p_form5'] = F['asof_p_prev5_game_success_rate'] - F['p_cur_s_rate']
    F['p_form5_prior'] = F['asof_p_prev5_game_success_rate'] - F['p_prior_s_rate']
    F['p_form_mid5'] = F['asof_p_prev5_game_middle_rate'] - F['p_cur_mid_rate']
    F['p_form_trend'] = F['asof_p_prev1_game_success_rate'] - F['asof_p_prev5_game_success_rate']
    return F


def add_matchup_features(F, df, stats):
    """Pitcher x batter matchup priors from training seasons < row season (row's own ids + train stats)."""
    pbs = stats.get('pbs')
    if pbs is None: return F
    d = pd.DataFrame({'pb_key': df.pitcher_id.values.astype(np.int64) * 100000 + df.batter_id.values.astype(np.int64), 'season': df.season.values})
    # restrict grid to keys present in df to keep it small
    keys = np.unique(d.pb_key.values)
    sub = pbs[pbs.pb_key.isin(keys)]
    if len(sub) == 0:
        # no training history for any pair in this frame: fall through with zero counts so that every formula below is evaluated
        # with exactly the same floating-point arithmetic as for n=0 rows in the general path (the NN's quantile transform is
        # sensitive to the ~1e-16 residue of (30*base)/30 - base, so the value must NOT be simplified to a literal 0.0)
        n = np.zeros(len(d)); sv = np.zeros(len(d))
    else:
        pt = _prior_table(sub, 'pb_key', ['n', 's'])[['pb_key', 'season', 'prior_n', 'prior_s']]
        d = d.merge(pt, on=['pb_key', 'season'], how='left')
        n = d.prior_n.fillna(0.0).values; sv = d.prior_s.fillna(0.0).values
    F['pb_prior_n'] = n
    with np.errstate(invalid='ignore', divide='ignore'):
        F['pb_prior_rate'] = np.where(n >= 10, sv / np.maximum(n, 1), np.nan)
        base = F['p_prior_s_rate'].fillna(0.52).values
        F['pb_prior_shr30'] = (sv + 30 * base) / (n + 30)
        F['pb_prior_dev'] = F['pb_prior_shr30'] - base
    return F

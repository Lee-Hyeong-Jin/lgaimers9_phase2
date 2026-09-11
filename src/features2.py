"""v2 features: drift/regime-invariant skill priors (deviation from season x game_type league mean),
import warnings; warnings.filterwarnings("ignore")
current-season deviations, and explicit level reference for offset modeling.
Built on top of features.build_features (v1)."""
import os
import numpy as np
import pandas as pd
from features import (build_stats as build_stats_v1, build_features as build_features_v1,
                      _recover_indicators, _prior_table, RATE_COLS_P, RATE_COLS_B, MAX_SEASON)

SKILL_P = ['n', 'res', 'n_R', 'res_R', 'n_F', 'res_F', 'n_L', 'res_L', 'rev_res', 'mid_res', 'far_res',
           'n_ahead', 'res_ahead', 'n_behind', 'res_behind', 'n_2k', 'res_2k', 'n_3b', 'res_3b', 'n_home', 'res_home']
SKILL_B = ['n', 'res', 'mid_res', 'n_vsL', 'res_vsL', 'n_2k', 'res_2k']


def league_table(tr):
    """season x game_type mean success + failure-mode means (from training rows)."""
    L = tr.groupby(['season', 'game_type']).control_success.mean().unstack()
    La = tr.groupby('season').control_success.mean()
    return {'L_gt': L, 'L_all': La}


def build_stats(train):
    st = build_stats_v1(train)
    tr = train.reset_index(drop=True)
    lt = league_table(tr)
    st.update(lt)
    # residuals vs season x game_type mean
    Lrow = tr.set_index(['season', 'game_type']).index.map(lambda k: lt['L_gt'].loc[k[0], k[1]]).values
    y = tr.control_success.values.astype(float)
    res = y - Lrow
    pind = _recover_indicators(tr, 'pitcher_id', 'asof_pitcher_n', RATE_COLS_P)
    bind = _recover_indicators(tr, 'batter_id', 'asof_batter_n', RATE_COLS_B)
    # failure-mode residuals vs season x game_type means
    fm = pd.DataFrame({'season': tr.season, 'game_type': tr.game_type, 'rev': pind['rev'].values,
                       'mid': pind['mid'].values})
    fm['far'] = 1 - y - fm.rev - fm.mid
    fm['far'] = fm['far'].clip(0, 1)
    fmm = fm.groupby(['season', 'game_type'])[['rev', 'mid', 'far']].transform('mean')
    st['FM_gt'] = fm.groupby(['season', 'game_type'])[['rev', 'mid', 'far']].mean()
    isF = (tr.game_type == 'F').values
    ahead = (tr.strikes_before > tr.balls_before).values
    behind = (tr.balls_before > tr.strikes_before).values
    two_k = (tr.strikes_before == 2).values
    three_b = (tr.balls_before == 3).values
    vsL = (tr.batter_hand == 1).values
    home = (tr.top_bottom == 'T').values
    p = pd.DataFrame({'pitcher_id': tr.pitcher_id, 'season': tr.season, 'n': 1.0, 'res': res,
                      'n_R': (~isF) * 1.0, 'res_R': (~isF) * res, 'n_F': isF * 1.0, 'res_F': isF * res,
                      'n_L': vsL * 1.0, 'res_L': vsL * res,
                      'rev_res': fm.rev.values - fmm.rev.values, 'mid_res': fm.mid.values - fmm.mid.values,
                      'far_res': fm.far.values - fmm.far.values,
                      'n_ahead': ahead * 1.0, 'res_ahead': ahead * res, 'n_behind': behind * 1.0, 'res_behind': behind * res,
                      'n_2k': two_k * 1.0, 'res_2k': two_k * res, 'n_3b': three_b * 1.0, 'res_3b': three_b * res,
                      'n_home': home * 1.0, 'res_home': home * res})
    st['ps2'] = p.groupby(['pitcher_id', 'season']).sum().reset_index()
    bmid = bind['mid'].values
    bmm = pd.DataFrame({'season': tr.season, 'game_type': tr.game_type, 'mid': bmid}).groupby(['season', 'game_type']).mid.transform('mean').values
    vsLHP = (tr.pitcher_hand == 1).values
    b = pd.DataFrame({'batter_id': tr.batter_id, 'season': tr.season, 'n': 1.0, 'res': res, 'mid_res': bmid - bmm,
                      'n_vsL': vsLHP * 1.0, 'res_vsL': vsLHP * res, 'n_2k': two_k * 1.0, 'res_2k': two_k * res})
    st['bs2'] = b.groupby(['batter_id', 'season']).sum().reset_index()
    # team x season levels (regime proxies): pitcher_team and batter_team deviations
    st['team_p'] = tr.groupby(['season', 'pitcher_team_id']).control_success.mean()
    st['team_b'] = tr.groupby(['season', 'batter_team_id']).control_success.mean()
    # ---- pitcher role stats per season (games, starts, pitches/game) from row-order game segmentation ----
    tA = np.where(tr.top_bottom == 'T', tr.pitcher_team_id, tr.batter_team_id)
    tB = np.where(tr.top_bottom == 'T', tr.batter_team_id, tr.pitcher_team_id)
    gk = pd.Series(tr.season.astype(str) + '_' + tr.game_month.astype(str) + '_' + tr.game_dayofweek.astype(str) + '_' + pd.Series(tA).astype(str) + '_' + pd.Series(tB).astype(str) + '_' + tr.game_type)
    gid = ((gk != gk.shift()) | (tr.inning < tr.inning.shift())).cumsum().values
    g = pd.DataFrame({'pitcher_id': tr.pitcher_id, 'season': tr.season, 'gid': gid, 'inning': tr.inning, 'tb': tr.top_bottom})
    first = g.groupby(['gid', 'tb']).pitcher_id.first().rename('starter').reset_index()
    g = g.merge(first, on=['gid', 'tb'], how='left')
    g['is_start'] = (g.pitcher_id == g.starter)
    pg = g.groupby(['pitcher_id', 'season', 'gid']).agg(n=('inning', 'size'), start=('is_start', 'max'), inn_first=('inning', 'min'), inn_last=('inning', 'max')).reset_index()
    role = pg.groupby(['pitcher_id', 'season']).agg(games=('n', 'size'), starts=('start', 'mean'), ppg=('n', 'mean'), inn_first=('inn_first', 'mean'), inn_last=('inn_last', 'mean')).reset_index()
    st['role'] = role
    # end-of-season form: last 5 games and second half of the pitcher's season (per pitcher, season)
    pg2 = g.groupby(['pitcher_id', 'season', 'gid']).agg(n=('inning', 'size')).reset_index()
    pg2['s'] = pd.DataFrame({'pitcher_id': tr.pitcher_id, 'season': tr.season, 'gid': gid, 'y': tr.control_success}).groupby(['pitcher_id', 'season', 'gid']).y.sum().values
    pg2['gord'] = pg2.groupby(['pitcher_id', 'season']).cumcount(ascending=False)  # 0 = last game
    last5 = pg2[pg2.gord < 5].groupby(['pitcher_id', 'season']).agg(n_last5g=('n', 'sum'), s_last5g=('s', 'sum')).reset_index()
    h = pd.DataFrame({'pitcher_id': tr.pitcher_id, 'season': tr.season, 'y': tr.control_success.astype(float)})
    h['k'] = h.groupby(['pitcher_id', 'season']).cumcount(); h['tot'] = h.groupby(['pitcher_id', 'season']).y.transform('size')
    h2 = h[h.k >= h.tot / 2].groupby(['pitcher_id', 'season']).agg(n_h2=('y', 'size'), s_h2=('y', 'sum')).reset_index()
    st['endform'] = last5.merge(h2, on=['pitcher_id', 'season'], how='outer').fillna(0.0)
    try:
        from trackman_feats import build_tm_table
        st['tm'] = build_tm_table()
    except Exception as e:
        print('trackman table unavailable:', e)
        st['tm'] = None
    return st


def level_ref(df, stats, mode='prev', guess=None):
    """Reference success level per row (used for offset / cur deviation).
    mode='true': own season x game_type mean (training rows only).
    mode='prev': previous season's mean for that game_type (persistence forecast); 2019 -> own.
    For rows whose season is not in stats (test), use `guess` dict {'R': p, 'F': p} or last season + 0."""
    L = stats['L_gt']
    seasons = df.season.values; gt = df.game_type.values
    out = np.full(len(df), np.nan)
    max_s = L.index.max()
    for g in ('R', 'F'):
        m = gt == g
        if mode == 'true':
            ref = pd.Series(seasons[m]).map(L[g]).values
        else:
            ref = pd.Series(seasons[m] - 1).map(L[g]).values
            first = seasons[m] == L.index.min()
            ref = np.where(first, pd.Series(seasons[m]).map(L[g]).values, ref)
        # unseen seasons (test): guess or last available
        unseen = seasons[m] > max_s
        gval = (guess or {}).get(g, L[g].loc[max_s])
        ref = np.where(unseen, gval, ref)
        out[m] = ref
    return out


def build_features(df, stats, mode='prev', guess=None):
    F = build_features_v1(df, stats)
    from features import add_matchup_features
    F = add_matchup_features(F, df.reset_index(drop=True), stats)
    d = df.reset_index(drop=True)
    pp = _prior_table(stats['ps2'], 'pitcher_id', SKILL_P, last_col='res')
    bp = _prior_table(stats['bs2'], 'batter_id', SKILL_B, last_col='res')
    pp = pp.rename(columns={c: 'P_' + c for c in pp.columns if c not in ('pitcher_id', 'season')})
    bp = bp.rename(columns={c: 'B_' + c for c in bp.columns if c not in ('batter_id', 'season')})
    d = d.merge(pp, on=['pitcher_id', 'season'], how='left').merge(bp, on=['batter_id', 'season'], how='left')
    for c in d.columns:
        if c.startswith(('P_prior_', 'B_prior_')):
            d[c] = d[c].fillna(0.0)

    def rate(num, den, minn=1):
        den = den.values if hasattr(den, 'values') else den
        num = num.values if hasattr(num, 'values') else num
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(den >= minn, num / np.maximum(den, 1e-9), np.nan)

    def shrink(num, den, k):
        return num / (den + k)
    # ---- pitcher skill (deviation) priors ----
    F['P_skill'] = rate(d.P_prior_res, d.P_prior_n)
    for k in (200, 1000):
        F[f'P_skill_shr{k}'] = shrink(d.P_prior_res, d.P_prior_n, k)
    F['P_skill_R'] = rate(d.P_prior_res_R, d.P_prior_n_R)
    F['P_skill_F'] = rate(d.P_prior_res_F, d.P_prior_n_F)
    F['P_skill_gt'] = np.where(d.game_type == 'F', F['P_skill_F'], F['P_skill_R'])
    F['P_skill_gt_shr300'] = np.where(d.game_type == 'F', shrink(d.P_prior_res_F, d.P_prior_n_F, 300), shrink(d.P_prior_res_R, d.P_prior_n_R, 300))
    F['P_skill_vsL'] = rate(d.P_prior_res_L, d.P_prior_n_L)
    F['P_skill_vsR'] = rate(d.P_prior_res - d.P_prior_res_L, d.P_prior_n - d.P_prior_n_L)
    F['P_skill_vsHand'] = np.where(d.batter_hand == 1, F['P_skill_vsL'], F['P_skill_vsR'])
    F['P_skill_vsHand_shr300'] = np.where(d.batter_hand == 1, shrink(d.P_prior_res_L, d.P_prior_n_L, 300),
                                          shrink(d.P_prior_res - d.P_prior_res_L, d.P_prior_n - d.P_prior_n_L, 300))
    F['P_rev_skill'] = rate(d.P_prior_rev_res, d.P_prior_n)
    F['P_mid_skill'] = rate(d.P_prior_mid_res, d.P_prior_n)
    F['P_far_skill'] = rate(d.P_prior_far_res, d.P_prior_n)
    F['P_skill_ahead'] = rate(d.P_prior_res_ahead, d.P_prior_n_ahead, 30)
    F['P_skill_behind'] = rate(d.P_prior_res_behind, d.P_prior_n_behind, 30)
    F['P_skill_2k'] = rate(d.P_prior_res_2k, d.P_prior_n_2k, 30)
    F['P_skill_3b'] = rate(d.P_prior_res_3b, d.P_prior_n_3b, 30)
    F['P_skill_home'] = rate(d.P_prior_res_home, d.P_prior_n_home, 30)
    F['P_skill_away'] = rate(d.P_prior_res - d.P_prior_res_home, d.P_prior_n - d.P_prior_n_home, 30)
    F['P_skill_ha'] = np.where(d.top_bottom == 'T', F['P_skill_home'], F['P_skill_away'])
    # count-context skill matching the row's count
    ah = (d.strikes_before > d.balls_before).values; bh = (d.balls_before > d.strikes_before).values
    F['P_skill_count'] = np.where(ah, F['P_skill_ahead'], np.where(bh, F['P_skill_behind'], np.nan))
    F['P_last_skill'] = rate(d.P_last_s, d.P_last_n) if 'P_last_s' in d else np.nan  # (last_s here is res of last season)
    # ---- batter skill priors ----
    F['B_skill'] = rate(d.B_prior_res, d.B_prior_n)
    F['B_skill_shr300'] = shrink(d.B_prior_res, d.B_prior_n, 300)
    F['B_mid_skill'] = rate(d.B_prior_mid_res, d.B_prior_n)
    F['B_skill_vsL'] = rate(d.B_prior_res_vsL, d.B_prior_n_vsL, 30)
    F['B_skill_vsR'] = rate(d.B_prior_res - d.B_prior_res_vsL, d.B_prior_n - d.B_prior_n_vsL, 30)
    F['B_skill_vsHand'] = np.where(d.pitcher_hand == 1, F['B_skill_vsL'], F['B_skill_vsR'])
    F['B_skill_2k'] = rate(d.B_prior_res_2k, d.B_prior_n_2k, 30)
    F['B_last_skill'] = rate(d.B_last_s, d.B_last_n) if 'B_last_s' in d else np.nan
    # ---- level reference & current-season deviations ----
    Lref = level_ref(d, stats, mode=mode, guess=guess)
    F['L_ref'] = Lref
    # all-game level reference for cur (cur mixes R and F): use overall season mean analogously
    La = stats['L_all']; max_s = La.index.max()
    if mode == 'true':
        La_row = pd.Series(d.season.values).map(La).values
    else:
        La_row = pd.Series(d.season.values - 1).map(La).values
        La_row = np.where(d.season.values == La.index.min(), pd.Series(d.season.values).map(La).values, La_row)
    ga = None if guess is None else guess.get('all', None)
    La_row = np.where(d.season.values > max_s, (La.loc[max_s] if ga is None else ga), La_row)
    F['L_all_ref'] = La_row
    F['p_cur_dev'] = F['p_cur_s_rate'] - La_row
    F['b_cur_dev'] = F['b_cur_s_rate'] - La_row
    F['p_cur_dev_shr200'] = (np.nan_to_num(F['p_cur_s'].values - F['p_cur_n'].values * La_row) + 200 * np.nan_to_num(F['P_skill'].values)) / (F['p_cur_n'].values + 200)
    bcs = np.nan_to_num(F['b_cur_s_rate'].values - La_row) * F['b_cur_n'].values
    F['b_cur_dev_shr300'] = (bcs + 300 * np.nan_to_num(F['B_skill'].values)) / (F['b_cur_n'].values + 300)
    F['p_cur_minus_skill'] = F['p_cur_dev'] - F['P_skill']
    F['b_cur_minus_skill'] = F['b_cur_dev'] - F['B_skill']
    # combined best-guess pitcher rate = Lref + shrunk skill
    F['p_est'] = Lref + F['p_cur_dev_shr200']
    F['b_est'] = Lref + F['b_cur_dev_shr300']
    F['home_team'] = np.where(d.top_bottom == 'T', d.pitcher_team_id, d.batter_team_id)
    if os.environ.get('LG_FEATS', '1') == '1':  # default ON (adopted 08-27): label-source regime of LG-involved games
        F['LG_game'] = ((d.pitcher_team_id.values == 13) | (d.batter_team_id.values == 13)).astype(float)
        if os.environ.get('LG_MAY', '1') == '1':  # default ON (adopted 08-27): regime switched in May 2023; F rows are new-regime from 2023 start
            F['LG_regime'] = F['LG_game'].values * ((d.season.values > 2023) | ((d.season.values == 2023) & ((d.game_month.values >= 5) | (d.game_type.values == 'F'))))
        else:
            F['LG_regime'] = F['LG_game'].values * (d.season.values >= 2023)
    if os.environ.get('LG30F') == '1':  # explicit LG-process 3-0 count indicator (row's own count + team ids)
        F['LG_30'] = F['LG_regime'].values * ((d.balls_before.values == 3) & (d.strikes_before.values == 0)) * (d.game_type.values == 'R')
    if os.environ.get('TEAM_LAST') == '1' and 'team_p' in stats:
        # last-season team effects (deviation from league-season mean), pitching team and home team; test season -> last training season
        tp = stats['team_p'].reset_index(); tp.columns = ['season', 'team', 'rate']
        lg = stats['league'].set_index('season')['league_rate'] if 'league' in stats else tp.groupby('season').rate.mean()
        tp['dev'] = tp.rate.values - tp.season.map(lg).values
        mx = int(tp.season.max()); dev_map = {(int(r.season), int(r.team)): r.dev for r in tp.itertuples()}
        prev_season = np.minimum(d.season.values - 1, mx)
        F['TEAM_last_dev_p'] = np.array([dev_map.get((int(s_), int(t)), 0.0) for s_, t in zip(prev_season, d.pitcher_team_id.values)])
        F['TEAM_last_dev_home'] = np.array([dev_map.get((int(s_), int(t)), 0.0) for s_, t in zip(prev_season, F['home_team'].values)])
    # ---- role priors ----
    from trackman_feats import prior_weighted, TM_METRICS, TYPES
    role = stats.get('role')
    if role is not None:
        rp = prior_weighted(role, 'pitcher_id', 'games', ['starts', 'ppg', 'inn_first', 'inn_last'])
        rp = rp.rename(columns={c: 'ROLE_' + c for c in rp.columns if c not in ('pitcher_id', 'season')})
        d = d.merge(rp, on=['pitcher_id', 'season'], how='left')
        F['ROLE_games_prior'] = d.ROLE_prior_n.fillna(0.0)
        F['ROLE_start_share'] = d.ROLE_prior_starts  # weighted mean of per-season start count?? -> use share
        F['ROLE_ppg'] = d.ROLE_prior_ppg
        F['ROLE_ppg_last'] = d.ROLE_last_ppg
        F['ROLE_inn_first'] = d.ROLE_prior_inn_first
        F['ROLE_inn_last'] = d.ROLE_prior_inn_last
        F['ROLE_cur_pace'] = F['p_cur_n'] / np.maximum(d.game_month.values - 2, 1)
        F['ROLE_inning_minus_first'] = d.inning.values - F['ROLE_inn_first']
    # ---- trackman priors ----
    tm = stats.get('tm')
    if tm is not None:
        mets = [c for c in tm.columns if c.startswith('tm_') and c != 'tm_n']
        tp = prior_weighted(tm, 'pitcher_id', 'tm_n', mets)
        tp = tp.rename(columns={c: 'TM_' + c for c in tp.columns if c not in ('pitcher_id', 'season')})
        d = d.merge(tp, on=['pitcher_id', 'season'], how='left')
        F['TM_n_prior'] = d.TM_prior_n.fillna(0.0)
        for m in mets:
            F['TM_' + m[3:]] = d['TM_prior_' + m]
        for m in ['tm_fb_speed', 'tm_rel_speed_mean', 'tm_spin_rate_mean', 'tm_fb_relh_std', 'tm_fb_rels_std', 'tm_rel_consistency_h', 'tm_rel_consistency_s', 'tm_minor_share']:
            F['TM_last_' + m[3:]] = d['TM_last_' + m]
        F['TM_fb_speed_trend'] = d['TM_last_tm_fb_speed'] - d['TM_prior_tm_fb_speed']
    F['p_cur_logn'] = np.log1p(F['p_cur_n']); F['b_cur_logn'] = np.log1p(F['b_cur_n'])
    F['p_prior_logn'] = np.log1p(F['p_prior_n']); F['b_prior_logn'] = np.log1p(F['b_prior_n'])
    return F


SKILL_FEATS = ['P_skill', 'P_skill_shr200', 'P_skill_shr1000', 'P_skill_R', 'P_skill_F', 'P_skill_gt', 'P_skill_gt_shr300',
               'P_skill_vsL', 'P_skill_vsR', 'P_skill_vsHand', 'P_skill_vsHand_shr300', 'P_rev_skill', 'P_mid_skill',
               'P_far_skill', 'P_skill_ahead', 'P_skill_behind', 'P_skill_2k', 'P_skill_3b', 'P_skill_home', 'P_skill_away',
               'P_skill_ha', 'P_skill_count', 'B_skill', 'B_skill_shr300', 'B_mid_skill', 'B_skill_vsL', 'B_skill_vsR',
               'B_skill_vsHand', 'B_skill_2k', 'L_ref', 'L_all_ref', 'p_cur_dev', 'b_cur_dev', 'p_cur_dev_shr200',
               'b_cur_dev_shr300', 'p_cur_minus_skill', 'b_cur_minus_skill', 'p_est', 'b_est', 'home_team',
               'p_cur_logn', 'b_cur_logn', 'p_prior_logn', 'b_prior_logn']

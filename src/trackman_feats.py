"""Trackman pitcher-season profile table (via matched IDs) + role stats helpers."""
import warnings; warnings.filterwarnings("ignore")
import os, numpy as np, pandas as pd
from common import load_trackman, ROOT

TM_METRICS = ['rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break', 'extension', 'rel_height', 'rel_side', 'zone_speed']
TYPES = ['Fastball', 'Slider', 'Curveball', 'ChangeUp', 'Splitter', 'Sinker', 'Cutter']


def build_tm_table(pmap_path=None):
    tm = load_trackman()
    pmap = pd.read_parquet(pmap_path or os.path.join(ROOT, 'data', 'derived', 'pitcher_map.parquet'))
    pmap = pmap[(pmap.purity > 0.9) & (pmap.votes >= 30)]
    tm = tm.merge(pmap[['pitcher_id', 'tm_id']].rename(columns={'tm_id': 'pitcher_trackman_id'}), on='pitcher_trackman_id')
    tm['ChangeUp'] = tm.tagged_pitch_type.isin(['ChangeUp', 'Changeup'])
    for t in TYPES:
        if t != 'ChangeUp': tm[t] = tm.tagged_pitch_type == t
    tm['minor'] = tm.trackman_game_id.str.contains('Minor|Futures|Test', regex=True)
    tm['is_fb'] = tm.pitch_type_group == 'fastball'
    g = tm.groupby(['pitcher_id', 'season'])
    out = g.size().rename('tm_n').to_frame()
    for m in TM_METRICS:
        out['tm_' + m + '_mean'] = g[m].mean()
        out['tm_' + m + '_std'] = g[m].std()
    for t in TYPES:
        out['tm_mix_' + t] = g[t].mean()
    out['tm_minor_share'] = g['minor'].mean()
    fb = tm[tm.is_fb].groupby(['pitcher_id', 'season'])
    out['tm_fb_speed'] = fb['rel_speed'].mean()
    out['tm_fb_spin'] = fb['spin_rate'].mean()
    out['tm_fb_ivb'] = fb['induced_vert_break'].mean()
    out['tm_fb_hb'] = fb['horz_break'].mean()
    out['tm_fb_relh_std'] = fb['rel_height'].std()
    out['tm_fb_rels_std'] = fb['rel_side'].std()
    out['tm_fb_speed_std'] = fb['rel_speed'].std()
    # release consistency within pitch type (mean of per-type std, weighted)
    pt = tm.groupby(['pitcher_id', 'season', 'pitch_type_group']).agg(n=('rel_height', 'size'), rh=('rel_height', 'std'), rs=('rel_side', 'std')).reset_index()
    pt['rh_w'] = pt.rh * pt.n; pt['rs_w'] = pt.rs * pt.n
    ptg = pt.groupby(['pitcher_id', 'season'])[['rh_w', 'rs_w', 'n']].sum()
    out['tm_rel_consistency_h'] = ptg.rh_w / ptg.n
    out['tm_rel_consistency_s'] = ptg.rs_w / ptg.n
    return out.reset_index()


def prior_weighted(tab, id_col, n_col, metrics, max_season=2026):
    """For each (id, season y): n-weighted mean of metrics over seasons < y, and last active season values."""
    ids = tab[id_col].unique(); seasons = np.arange(tab.season.min(), max_season + 1)
    grid = pd.MultiIndex.from_product([ids, seasons], names=[id_col, 'season'])
    t = tab.set_index([id_col, 'season']).reindex(grid)
    n = t[n_col].fillna(0.0).unstack('season')
    cn = n.cumsum(axis=1).shift(1, axis=1).fillna(0.0)
    act = n.gt(0)
    res = {'prior_n': cn}
    for m in metrics:
        v = t[m].unstack('season')
        wsum = (v.fillna(0.0) * n).cumsum(axis=1).shift(1, axis=1).fillna(0.0)
        nn = (n * v.notna()).cumsum(axis=1).shift(1, axis=1).fillna(0.0)
        res['prior_' + m] = wsum / nn.replace(0, np.nan)
        res['last_' + m] = v.where(act).ffill(axis=1).shift(1, axis=1)
    out = pd.concat({k: v.stack(dropna=False) for k, v in res.items()}, axis=1).reset_index()
    return out

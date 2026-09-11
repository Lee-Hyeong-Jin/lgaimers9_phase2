"""Explore matching trackman games/pitchers to main-data games/pitchers via pitch-sequence fingerprints."""
import sys, os, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, time
from collections import defaultdict, Counter
from common import load_train, load_trackman, ROOT
tr = load_train(); tm = load_trackman()
tr['idx'] = np.arange(len(tr))
# ---- main data game segments ----
tr['tA'] = np.where(tr.top_bottom == 'T', tr.pitcher_team_id, tr.batter_team_id)
tr['tB'] = np.where(tr.top_bottom == 'T', tr.batter_team_id, tr.pitcher_team_id)
gk = tr.season.astype(str) + '_' + tr.game_month.astype(str) + '_' + tr.game_dayofweek.astype(str) + '_' + tr.tA.astype(str) + '_' + tr.tB.astype(str) + '_' + tr.game_type
# also split when inning decreases (new game with same key)
newg = (gk != gk.shift()) | (tr.inning < tr.inning.shift())
tr['gid'] = newg.cumsum()
print('main games:', tr.gid.nunique())
# ---- trackman ----
tm = tm.sort_values(['trackman_game_id', 'pitch_no']).reset_index(drop=True)
tm['tb'] = np.where(tm.top_bottom == 'Top', 'T', 'B')
tm['ph'] = np.where(tm.pitcher_hand == 'Left', 1, 2); tm['bh'] = np.where(tm.batter_hand == 'Left', 1, 2)
# check hand code mapping guess via proportions
print('main pitcher_hand==1 frac', (tr.pitcher_hand == 1).mean(), 'tm Left frac', (tm.pitcher_hand == 'Left').mean())
print('main batter_hand==1 frac', (tr.batter_hand == 1).mean(), 'tm Left frac', (tm.batter_hand == 'Left').mean())


def fingerprint(df, inn, tb, b, s, o, ph, bh):
    return list(zip(df[inn].values, df[tb].values, df[b].values, df[s].values, df[o].values, df[ph].values, df[bh].values))

main_games = {}
for gid, g in tr.groupby('gid'):
    main_games[gid] = {'key': (g.season.iloc[0], g.game_month.iloc[0], g.game_dayofweek.iloc[0]), 'fp': fingerprint(g, 'inning', 'top_bottom', 'balls_before', 'strikes_before', 'outs_before', 'pitcher_hand', 'batter_hand'), 'n': len(g), 'gt': g.game_type.iloc[0]}
tm_games = {}
for gid, g in tm.groupby('trackman_game_id'):
    tm_games[gid] = {'key': (g.season.iloc[0], g.game_month.iloc[0], g.game_dayofweek.iloc[0]), 'fp': fingerprint(g, 'inning', 'tb', 'balls_before', 'strikes_before', 'outs_before', 'ph', 'bh'), 'n': len(g)}
by_key = defaultdict(list)
for gid, g in tm_games.items(): by_key[g['key']].append(gid)
import difflib
t = time.time(); matches = {}; scores = []
for gid, g in main_games.items():
    cands = by_key.get(g['key'], [])
    best, bs = None, 0
    fa = g['fp']
    ca = Counter(fa)
    for c in cands:
        fb = tm_games[c]['fp']
        # quick multiset similarity
        cb = Counter(fb)
        inter = sum((ca & cb).values())
        sim = inter / max(len(fa), len(fb))
        if sim > bs: best, bs = c, sim
    matches[gid] = (best, bs)
    scores.append(bs)
print('matching time', time.time() - t)
scores = np.array(scores)
print('similarity quantiles', np.quantile(scores, [0, .05, .1, .25, .5, .75, .9, 1]).round(3))
print('frac games with sim>0.9:', (scores > 0.9).mean(), ' >0.8:', (scores > 0.8).mean(), ' >0.6:', (scores > 0.6).mean())
# duplicates: same trackman game matched by multiple main games?
c = Counter([m[0] for m in matches.values() if m[1] > 0.8])
print('tm games matched by >1 main games:', sum(1 for v in c.values() if v > 1))
# ---- pitcher mapping via half-inning co-occurrence in well-matched games ----
votes = Counter(); bvotes = Counter()
tm_idx = tm.set_index('trackman_game_id')
for gid, (tg, sim) in matches.items():
    if sim < 0.85: continue
    a = tr[tr.gid == gid]; b = tm_idx.loc[[tg]]
    for (inn, tb), ga in a.groupby(['inning', 'top_bottom']):
        gb = b[(b.inning == inn) & (b.tb == tb)]
        pa = ga.pitcher_id.unique(); pb = gb.pitcher_trackman_id.unique()
        if len(pa) == 1 and len(pb) == 1:
            votes[(pa[0], pb[0])] += len(ga)
        # batters: sequence of batters within half inning
        ba = ga.batter_id.drop_duplicates().tolist(); bb = gb.batter_trackman_id.drop_duplicates().tolist()
        if len(ba) == len(bb):
            for x, y in zip(ba, bb): bvotes[(x, y)] += 1
# resolve pitcher mapping: for each main pitcher, best tm pitcher by votes; check purity
pm = defaultdict(Counter)
for (p, q), v in votes.items(): pm[p][q] += v
rows = []
for p, cnt in pm.items():
    q, v = cnt.most_common(1)[0]
    rows.append({'pitcher_id': p, 'tm_id': q, 'votes': v, 'purity': v / sum(cnt.values())})
pmap = pd.DataFrame(rows)
print('pitchers mapped:', len(pmap), ' of', tr.pitcher_id.nunique(), ' purity>0.9:', (pmap.purity > 0.9).mean(), ' votes>=100:', (pmap.votes >= 100).mean())
print(pmap.describe().to_string())
# uniqueness of tm ids
print('tm ids used by >1 main pitchers:', (pmap.groupby('tm_id').size() > 1).sum())
bm = defaultdict(Counter)
for (p, q), v in bvotes.items(): bm[p][q] += v
brows = []
for p, cnt in bm.items():
    q, v = cnt.most_common(1)[0]
    brows.append({'batter_id': p, 'tm_id': q, 'votes': v, 'purity': v / sum(cnt.values())})
bmap = pd.DataFrame(brows)
print('batters mapped:', len(bmap), ' of', tr.batter_id.nunique(), ' purity>0.9:', (bmap.purity > 0.9).mean())
# sanity: hand agreement for mapped pitchers
ph_main = tr.groupby('pitcher_id').pitcher_hand.agg(lambda x: x.mode()[0])
ph_tm = tm.groupby('pitcher_trackman_id').ph.agg(lambda x: x.mode()[0])
pmap['hand_ok'] = pmap.pitcher_id.map(ph_main).values == pmap.tm_id.map(ph_tm).values
print('hand agreement:', pmap.hand_ok.mean())
# pitch-count agreement per season for mapped pitchers
a = tr.groupby(['pitcher_id', 'season']).size().rename('n_main').reset_index().merge(pmap[['pitcher_id', 'tm_id']], on='pitcher_id')
b = tm.groupby(['pitcher_trackman_id', 'season']).size().rename('n_tm').reset_index().rename(columns={'pitcher_trackman_id': 'tm_id'})
ab = a.merge(b, on=['tm_id', 'season'], how='left')
print('count ratio n_tm/n_main quantiles:', (ab.n_tm / ab.n_main).quantile([.05, .25, .5, .75, .95]).round(3).to_dict())
os.makedirs(os.path.join(ROOT, 'data', 'derived'), exist_ok=True)
pmap.to_parquet(os.path.join(ROOT, 'data', 'derived', 'pitcher_map.parquet')); bmap.to_parquet(os.path.join(ROOT, 'data', 'derived', 'batter_map.parquet'))
pd.DataFrame([{'gid': k, 'tm_game': v[0], 'sim': v[1]} for k, v in matches.items()]).to_parquet(os.path.join(ROOT, 'data', 'derived', 'game_map.parquet'))
tr[['idx', 'gid']].to_parquet(os.path.join(ROOT, 'data', 'derived', 'train_gid.parquet'))

import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
pd.set_option('display.width', 250)
from common import load_train, bss_score, brier
from features import _recover_indicators, RATE_COLS_P
tr = load_train()
ind = _recover_indicators(tr, 'pitcher_id', 'asof_pitcher_n', RATE_COLS_P)
d = pd.concat([tr[['season','game_month','game_type','pitcher_id','batter_id','control_success','balls_before','strikes_before','pitcher_hand']], ind], axis=1)
d['far'] = 1 - d.s - d.rev - d.mid
print("=== failure-mode decomposition by season ===")
print(d.groupby('season')[['s','rev','mid','far','ball','strike','fb','br','os']].mean().round(4).to_string())
print("\n=== by season x game_type ===")
print(d.groupby(['season','game_type'])[['s','rev','mid','far']].mean().round(4).to_string())
print("\n=== F share by season ===", d.groupby('season').game_type.apply(lambda x:(x=='F').mean()).round(3).to_dict())
# common shift vs composition: pitchers with >=500 pitches in consecutive seasons
ps = d.groupby(['pitcher_id','season']).agg(n=('s','size'), r=('s','mean'), rev=('rev','mean'), mid=('mid','mean'), far=('far','mean')).reset_index()
a = ps.merge(ps.assign(season=ps.season-1), on=['pitcher_id','season'], suffixes=('','_next'))
a = a[(a.n>=300)&(a.n_next>=300)]
print("\n=== same-pitcher change (n>=300 both years), season -> season+1 ===")
print(a.groupby('season').apply(lambda g: pd.Series({'n_p':len(g), 'dr':np.average(g.r_next-g.r, weights=g.n_next), 'drev':np.average(g.rev_next-g.rev, weights=g.n_next), 'dmid':np.average(g.mid_next-g.mid, weights=g.n_next), 'dfar':np.average(g.far_next-g.far, weights=g.n_next)})).round(4).to_string())
# league change
lr = d.groupby('season').s.mean()
print("\nleague change:", (lr.shift(-1)-lr).round(4).to_dict())
# by count state per season (is the count effect stable?)
print("\n=== success by balls per season ===")
print(d.groupby(['season','balls_before']).s.mean().unstack().round(3).to_string())
print("\n=== success by strikes per season ===")
print(d.groupby(['season','strikes_before']).s.mean().unstack().round(3).to_string())
print("\n=== by pitcher_hand per season ===")
print(d.groupby(['season','pitcher_hand']).s.mean().unstack().round(3).to_string())
print("\n=== by pitch type per season ===")
for k in ['fb','br','os']:
    print(k, d[d[k]==1].groupby('season').s.mean().round(3).to_dict())

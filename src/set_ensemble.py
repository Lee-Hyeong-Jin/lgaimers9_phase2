"""Set ensemble.json blend weights / global shift. usage:
python src/set_ensemble.py --weights lgb_mc_meta.json:0.7,lgb_v1_meta.json:0.3 --prob_shift -0.006 --blend logit"""
import os, sys, json, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--dir', default=os.path.join(ROOT, 'submit', 'model'))
ap.add_argument('--weights', default='')
ap.add_argument('--prob_shift', type=float, default=None)
ap.add_argument('--blend', default=None)
ap.add_argument('--models', default=None)  # comma list to set the model list explicitly
ap.add_argument('--cold_slope', type=float, default=None)
ap.add_argument('--lg_shift', type=float, default=None)  # additive prob shift for LG-involved KBO rows
ap.add_argument('--f_shift', type=float, default=None)   # additive prob shift for Futures rows
ap.add_argument('--lg_shift_p', type=float, default=None)   # LG-involved rows, LG pitching
ap.add_argument('--lg_shift_opp', type=float, default=None) # LG-involved rows, opponent pitching
ap.add_argument('--lg30_shift', type=float, default=None)   # LG-involved KBO rows at 3-0 count
ap.add_argument('--lg_shift_home', type=float, default=None) # extra shift for LG home games
ap.add_argument('--lg_shift_away', type=float, default=None) # extra shift for LG away games
ap.add_argument('--cold_center', type=float, default=None)
ap.add_argument('--sharpen', default=None)  # JSON list of {col, lo, hi, slope}; '[]' clears
a = ap.parse_args()
p = os.path.join(a.dir, 'ensemble.json')
ens = json.load(open(p)) if os.path.exists(p) else {'models': [], 'blend': 'logit'}
if a.models: ens['models'] = a.models.split(',')
if a.blend: ens['blend'] = a.blend
if a.prob_shift is not None: ens['prob_shift'] = a.prob_shift
if a.cold_slope is not None: ens['cold_slope'] = a.cold_slope
if a.lg_shift is not None: ens['lg_shift'] = a.lg_shift
if a.f_shift is not None: ens['f_shift'] = a.f_shift
if a.lg_shift_p is not None: ens['lg_shift_p'] = a.lg_shift_p
if a.lg_shift_opp is not None: ens['lg_shift_opp'] = a.lg_shift_opp
if a.lg30_shift is not None: ens['lg30_shift'] = a.lg30_shift
if a.lg_shift_home is not None: ens['lg_shift_home'] = a.lg_shift_home
if a.lg_shift_away is not None: ens['lg_shift_away'] = a.lg_shift_away
if a.cold_center is not None: ens['cold_center'] = a.cold_center
if a.sharpen is not None: ens['sharpen'] = json.loads(a.sharpen)
for kv in [x for x in a.weights.split(',') if x]:
    k, v = kv.split(':')
    mp = os.path.join(a.dir, k); meta = json.load(open(mp)); meta['weight'] = float(v); json.dump(meta, open(mp, 'w'))
    if k not in ens['models']: ens['models'].append(k)
json.dump(ens, open(p, 'w'), indent=1)
print(json.dumps(ens))
for m in ens['models']:
    meta = json.load(open(os.path.join(a.dir, m))); print(m, 'type', meta.get('type'), 'weight', meta.get('weight'), 'boosters', len(meta['boosters']))

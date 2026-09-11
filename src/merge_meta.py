"""Merge booster lists of extra-seed metas into a base meta. usage: merge_meta.py base_meta.json extra_meta.json [...]"""
import json, os, sys
d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'submit', 'model')
base = json.load(open(os.path.join(d, sys.argv[1])))
for extra in sys.argv[2:]:
    e = json.load(open(os.path.join(d, extra)))
    assert e['feats'] == base['feats'] and e.get('cat') == base.get('cat') and e.get('type') == base.get('type')
    base['boosters'] = sorted(set(base['boosters'] + e['boosters']))
json.dump(base, open(os.path.join(d, sys.argv[1]), 'w'))
ens_p = os.path.join(d, 'ensemble.json'); ens = json.load(open(ens_p))
ens['models'] = [m for m in ens['models'] if m not in sys.argv[2:]]; json.dump(ens, open(ens_p, 'w'))
print(sys.argv[1], 'boosters:', len(base['boosters']))

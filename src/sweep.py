"""CLI sweep runner. Example:
python src/sweep.py --name hp1 --feats V1 --params '{"num_leaves":31}' --folds 2024
"""
import sys, os, json, argparse, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from exp2 import run2, fold_data, feature_sets
from common import load_train

ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True)
ap.add_argument('--feats', default='V1')
ap.add_argument('--drop', default='')  # comma list of features to drop
ap.add_argument('--add', default='')   # comma list of features to add
ap.add_argument('--params', default='{}')
ap.add_argument('--folds', default='2024,2023')
ap.add_argument('--mode', default='prev')
ap.add_argument('--offset', action='store_true')
ap.add_argument('--min_train_season', type=int, default=2019)
ap.add_argument('--weight', default='none')  # none | lin | exp:<rate>
ap.add_argument('--threads', type=int, default=5)
ap.add_argument('--num_rounds', type=int, default=3000)
ap.add_argument('--early_stop', type=int, default=300)
ap.add_argument('--save_oof', action='store_true')
a = ap.parse_args()
tr = load_train()
F, _ = fold_data(tr, int(a.folds.split(',')[0]), a.mode)
fs = feature_sets(F)
feats = fs[a.feats] if a.feats in fs else [c for c in F.columns if c not in fs['ABS_PRIOR']] if a.feats == 'NO_ABS' else a.feats.split(',')
feats = [c for c in feats if c not in a.drop.split(',')] + [c for c in a.add.split(',') if c and c not in feats]
params = json.loads(a.params); params['num_threads'] = a.threads
wf = None
if a.weight == 'lin':
    wf = lambda d, f: (d.season.values - d.season.min() + 1).astype(float)
elif a.weight.startswith('exp:'):
    rate = float(a.weight.split(':')[1])
    wf = lambda d, f: np.exp(-rate * (d.season.max() - d.season.values)).astype(float)
run2(a.name, feats, mode=a.mode, offset=a.offset, params=params, folds=tuple(int(x) for x in a.folds.split(',')),
     min_train_season=a.min_train_season, weight_fn=wf, num_rounds=a.num_rounds, early_stop=a.early_stop, save_oof=a.save_oof)

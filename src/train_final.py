"""Train final models on all training seasons and export artifacts to submit/model/."""
import sys, os, json, time, argparse, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, joblib, lightgbm as lgb
from common import load_train, ROOT
import features2 as f2
from features import add_change_features, CAT_FEATURES
from exp import DEFAULT_PARAMS
from exp2 import feature_sets

ap = argparse.ArgumentParser()
ap.add_argument('--feats', default='V1')
ap.add_argument('--params', default='{}')
ap.add_argument('--rounds', type=int, default=180)
ap.add_argument('--seeds', default='0,1,2,3,4')
ap.add_argument('--out', default=os.path.join(ROOT, 'submit', 'model'))
ap.add_argument('--tag', default='lgb_v1')
ap.add_argument('--model', default='lgb_bin')  # lgb_bin | lgb_mc
ap.add_argument('--weight', type=float, default=1.0)
ap.add_argument('--cat_ids', action='store_true')
ap.add_argument('--wexp', type=float, default=0.0)
ap.add_argument('--fw', type=float, default=1.0)  # sample-weight multiplier for Futures rows  # recency weight exp(-wexp*(max_season-season))
ap.add_argument('--offset', default='')  # 'bayes' -> init_score/baseline from the dynamic-Bayesian pitcher level
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
tr = load_train()
t = time.time()
stats = f2.build_stats(tr)
F = add_change_features(f2.build_features(tr, stats, mode='prev'))
fs = feature_sets(F)
feats = fs[a.feats] if a.feats in fs else a.feats.split(',')
cat = [c for c in CAT_FEATURES + ['home_team'] if c in feats]
init = None
if a.offset in ('bayes', 'bayes_src'):
    from features_dev import bayes_features, BAYES_HP
    from features import _recover_indicators as _ri, RATE_COLS_P as _rc
    bay = bayes_features(tr, F, stats, src=a.offset.endswith('_src'), **BAYES_HP)
    mu = dict(zip(stats['league'].season, stats['league'].league_rate)); mu_row = pd.Series(tr.season.values).map(mu).values
    pm = np.clip(mu_row + bay['bayes_dev'], 0.05, 0.95)
    _ind = _ri(tr, 'pitcher_id', 'asof_pitcher_n', _rc); _far = (1 - _ind['s'] - _ind['rev'] - _ind['mid']).clip(0, 1)
    _y4 = np.stack([_ind['s'].values, _ind['rev'].values, _ind['mid'].values, _far.values], 1).argmax(1)
    fshare = np.bincount(_y4, minlength=4)[1:].astype(float); fshare /= fshare.sum()
    if a.model in ('lgb_mc', 'cat_mc'):
        init = np.stack([np.log(pm)] + [np.log((1 - pm) * fshare[k]) for k in range(3)], 1)
    else:
        from scipy.special import logit as _logit; init = _logit(pm)
    print('offset=bayes: init computed', init.shape, 'fshare', fshare.round(4), flush=True)
print('features', len(feats), 'built in', round(time.time() - t, 1), 's')
y = tr.control_success.values
params = {**DEFAULT_PARAMS, **json.loads(a.params)}
if a.model == 'lgb_mc':
    from features import _recover_indicators, RATE_COLS_P
    ind = _recover_indicators(tr, 'pitcher_id', 'asof_pitcher_n', RATE_COLS_P)
    far = (1 - ind['s'] - ind['rev'] - ind['mid']).clip(0, 1)
    y = np.stack([ind['s'].values, ind['rev'].values, ind['mid'].values, far.values], 1).argmax(1)
    params.update({'objective': 'multiclass', 'num_class': 4})
boosters = []
if a.model in ('cat', 'cat_mc'):
    from catboost import CatBoostClassifier
    if a.cat_ids: cat = cat + [c for c in ('pitcher_id', 'batter_id') if c in feats]
    Xc = F[feats].copy()
    for c in cat: Xc[c] = Xc[c].astype(int)
    prm = json.loads(a.params)
    if a.model == 'cat_mc':
        from features import _recover_indicators, RATE_COLS_P
        ind = _recover_indicators(tr, 'pitcher_id', 'asof_pitcher_n', RATE_COLS_P)
        far = (1 - ind['s'] - ind['rev'] - ind['mid']).clip(0, 1)
        y = np.stack([ind['s'].values, ind['rev'].values, ind['mid'].values, far.values], 1).argmax(1)
    for s in [int(x) for x in a.seeds.split(',')]:
        kw = dict(iterations=a.rounds, learning_rate=prm.get('lr', 0.05), depth=prm.get('depth', 7), l2_leaf_reg=prm.get('l2', 10),
                  random_seed=s, verbose=0, task_type=prm.get('task_type', 'GPU'), border_count=128, thread_count=prm.get('threads', 8))
        if a.model == 'cat_mc': kw['loss_function'] = 'MultiClass'
        m = CatBoostClassifier(**kw)
        if init is not None:
            from catboost import Pool
            m.fit(Pool(Xc, y, cat_features=cat, baseline=init))
        else:
            m.fit(Xc, y, cat_features=cat)
        fn = f'{a.tag}_seed{s}.cbm'; m.save_model(os.path.join(a.out, fn)); boosters.append(fn)
        pr = m.predict_proba(Xc[:200000]); pr = pr[:, 0] if a.model == 'cat_mc' else pr[:, 1]
        print('trained', a.model, 'seed', s, 'pred mean', round(float(pr.mean()), 4), flush=True)
for s in ([] if a.model in ('cat', 'cat_mc') else [int(x) for x in a.seeds.split(',')]):
    p = {**params, 'seed': s, 'bagging_seed': s, 'feature_fraction_seed': s}
    wts = None if a.wexp == 0 else np.exp(-a.wexp * (tr.season.values.max() - tr.season.values)).astype(float)
    if a.fw != 1.0:
        wts = (np.ones(len(tr)) if wts is None else wts) * np.where(tr.game_type.values == 'F', a.fw, 1.0)
    dtr = lgb.Dataset(F[feats], y, weight=wts, init_score=init, categorical_feature=cat)
    m = lgb.train(p, dtr, num_boost_round=a.rounds)
    m.save_model(os.path.join(a.out, f'{a.tag}_seed{s}.txt'))
    boosters.append(f'{a.tag}_seed{s}.txt')
    pr = m.predict(F[feats][:200000]); pr = pr[:, 0] if pr.ndim == 2 else pr
    print('trained seed', s, 'pred mean on train', round(float(pr.mean()), 4), flush=True)
# strip stats to what inference needs
slim = {k: stats[k] for k in ['ps', 'bs', 'league', 'max_season', 'L_gt', 'L_all', 'ps2', 'bs2', 'FM_gt', 'team_p', 'team_b', 'role', 'tm', 'pbs', 'endform']}
pass  # stats.pkl is written only by src/save_stats.py (avoid stale overwrites)
meta = {'type': a.model, 'feats': feats, 'cat': cat, 'boosters': boosters, 'mode': 'prev', 'level_guess': None, 'logit_shift': 0.0, 'weight': a.weight, 'offset': a.offset, 'fshare': (fshare.tolist() if init is not None else None)}
json.dump(meta, open(os.path.join(a.out, f'{a.tag}_meta.json'), 'w'))
ens_path = os.path.join(a.out, 'ensemble.json')
ens = json.load(open(ens_path)) if os.path.exists(ens_path) else {'models': [], 'blend': 'logit'}
if f'{a.tag}_meta.json' not in ens['models']: ens['models'].append(f'{a.tag}_meta.json')
json.dump(ens, open(ens_path, 'w'))
print('saved to', a.out)

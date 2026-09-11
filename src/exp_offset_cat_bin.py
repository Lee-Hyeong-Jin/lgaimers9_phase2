"""CatBoost binary (Logloss) with IDs and baseline = logit(bayes posterior mean)."""
import sys, os, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.special import logit, expit
from exp2 import fold_data, feature_sets, oracle_shift
from features import CAT_FEATURES
from features_dev import bayes_features, BAYES_HP
from common import load_train, bss_score, EXP
vs = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
tr = load_train(); y = tr.control_success.values; seas = tr.season.values
F, stats = fold_data(tr, vs, 'prev'); fs = feature_sets(F); feats = fs['V1']
cat = [c for c in CAT_FEATURES + ['home_team'] if c in feats] + ['pitcher_id', 'batter_id']
bay = bayes_features(tr, F, stats, **BAYES_HP)
mu = dict(zip(stats['league'].season, stats['league'].league_rate)); mu_row = pd.Series(seas).map(mu).fillna(mu[max(mu)]).values
init = logit(np.clip(mu_row + bay['bayes_dev'], 0.05, 0.95))
tr_idx = np.where(seas < vs)[0]; va = np.where(seas == vs)[0]; r = y[va].mean(); isR = tr.game_type.values[va] == 'R'
X = F[feats].copy()
for c in cat: X[c] = X[c].astype(int)
m = CatBoostClassifier(loss_function='Logloss', iterations=2500, learning_rate=0.02, depth=7, l2_leaf_reg=10, random_seed=0, verbose=0, task_type='GPU', border_count=128, thread_count=4)
m.fit(Pool(X.iloc[tr_idx], y[tr_idx], cat_features=cat, baseline=init[tr_idx]))
raw = m.predict(Pool(X.iloc[va], cat_features=cat, baseline=init[va]), prediction_type='RawFormulaVal'); ps_ = expit(raw)
rr = y[va][isR].mean()
print(f'[fold {vs}] CAT-ID binary + bayes baseline: raw={bss_score(y[va], ps_):.1f} oracle={bss_score(y[va], oracle_shift(ps_, r)[0]):.1f} | R-only oracle={bss_score(y[va][isR], oracle_shift(ps_[isR], rr)[0]):.1f}', flush=True)
np.save(os.path.join(EXP, 'oof', f'cat_binoff_{vs}.npy'), ps_)

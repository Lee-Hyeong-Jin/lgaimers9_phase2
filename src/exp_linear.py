"""Diversity member: L2-regularized logistic regression on V1 features (+ Bayes level as a feature). Fold OOF saved as lin_<tag>_<fold>.npy"""
import sys, os, numpy as np, pandas as pd; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import load_train, bss_score
from exp2 import fold_data, feature_sets, oracle_shift
from features_dev import bayes_features, BAYES_HP
vs = int(sys.argv[1]) if len(sys.argv) > 1 else 2024; C = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05; tag = sys.argv[3] if len(sys.argv) > 3 else 'lin'
tr = load_train(); y = tr.control_success.values; seas = tr.season.values
F, stats = fold_data(tr, vs, 'prev'); fs = feature_sets(F); feats = [c for c in fs['V1'] if c not in ('pitcher_id', 'batter_id', 'pitcher_team', 'batter_team', 'home_team', 'count_state', 'base_state', 'season')]
bay = bayes_features(tr, F, stats, **BAYES_HP); F['bayes_dev'] = bay['bayes_dev']; feats = feats + ['bayes_dev']
X = F[feats].astype(np.float32).values; nanm = np.isnan(X); X = np.nan_to_num(X, nan=0.0); nan_cols = np.where(nanm.mean(0) > 0.01)[0]; X = np.concatenate([X, nanm[:, nan_cols].astype(np.float32)], 1)
tr_idx = np.where(seas < vs)[0]; va = np.where(seas == vs)[0]
sc = StandardScaler().fit(X[tr_idx]); Xs = sc.transform(X).astype(np.float32); Xs = np.clip(Xs, -6, 6)
w = np.exp(-0.15 * (vs - 1 - seas[tr_idx]))
m = LogisticRegression(C=C, max_iter=300, solver='lbfgs'); m.fit(Xs[tr_idx], y[tr_idx], sample_weight=w)
p = m.predict_proba(Xs[va])[:, 1]; r = y[va].mean(); isR = tr.game_type.values[va] == 'R'
print(f'[lin {tag} C={C}] fold {vs}: raw={bss_score(y[va], p):.1f} oracle={bss_score(y[va], oracle_shift(p, r)[0]):.1f} | R-only oracle={bss_score(y[va][isR], oracle_shift(p[isR], y[va][isR].mean())[0]):.1f} | n_feats {Xs.shape[1]}', flush=True)
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments', 'oof', f'lin_{tag}_{vs}.npy'), p)

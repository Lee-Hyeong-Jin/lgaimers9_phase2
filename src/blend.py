"""Evaluate blends of saved OOF predictions."""
import sys, os, itertools, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize
from common import load_train, bss_score, EXP
from exp2 import oracle_shift
tr = load_train(); y = tr.control_success.values
OOF = os.path.join(EXP, 'oof')
names = sys.argv[1:] if len(sys.argv) > 1 else sorted(set(f.rsplit('_', 1)[0] for f in os.listdir(OOF) if f.endswith('.npy')))
for vs in (2024, 2023):
    va = np.where(tr.season.values == vs)[0]; yv = y[va]; r = yv.mean()
    P = {}
    for n in names:
        f = os.path.join(OOF, f'{n}_{vs}.npy')
        if os.path.exists(f): P[n] = np.load(f)
    if not P: continue
    print(f'=== fold {vs} ===')
    for n, p in P.items():
        print(f'  {n:40s} raw={bss_score(yv, p):7.1f} oracle={bss_score(yv, oracle_shift(p, r)[0]):7.1f} pm={p.mean():.4f}')
    if len(P) > 1:
        M = np.stack([logit(np.clip(P[n], 1e-6, 1 - 1e-6)) for n in P], 1)
        # equal-weight logit blend
        pe = expit(M.mean(1))
        print(f'  {"EQUAL logit blend":40s} raw={bss_score(yv, pe):7.1f} oracle={bss_score(yv, oracle_shift(pe, r)[0]):7.1f}')
        # optimized weights (oracle-shifted objective) -- optimistic, for guidance only
        def obj(w):
            w = np.abs(w) / np.abs(w).sum(); p = expit(M @ w); return np.mean((oracle_shift(p, r)[0] - yv) ** 2)
        w0 = np.ones(M.shape[1]) / M.shape[1]
        o = minimize(obj, w0, method='Nelder-Mead', options={'maxiter': 300, 'xatol': 1e-3})
        w = np.abs(o.x) / np.abs(o.x).sum(); pw = expit(M @ w)
        print(f'  {"OPT weights":40s} raw={bss_score(yv, pw):7.1f} oracle={bss_score(yv, oracle_shift(pw, r)[0]):7.1f} w={dict(zip(P, w.round(3)))}')

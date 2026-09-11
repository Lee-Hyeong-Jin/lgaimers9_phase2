"""Evaluate members/compositions on the 2024-H2 rows under two training regimes:
   A = fold 2024 (train <2024) OOF sliced to H2 rows;  B = 2024H2 fold (train <2024 + 2024H1) OOF."""
import sys, os; sys.path.insert(0, 'src')
import numpy as np
from scipy.special import logit, expit
from common import load_train, bss_score
from exp2 import oracle_shift
tr = load_train(); y = tr.control_success.values; va = np.where(tr.season.values == 2024)[0]; k = len(va) // 2
h2 = va[k:]; yv = y[h2]; r = yv.mean()
members = sys.argv[1].split(',') if len(sys.argv) > 1 else ['mcs_pb', 'mcs_off', 'bin_off', 'zoo_catids_lr02_s3', 'zoo_cat_mc', 'nn_wide3_off', 'nn_ids_off', 'nn_ids_off_w15']
A, B = {}, {}
for n in members:
    fa, fb = f'experiments/oof/{n}_2024.npy', f'experiments/oof/{n}_2024H2.npy'
    if os.path.exists(fa): A[n] = np.load(fa)[k:]
    if os.path.exists(fb): B[n] = np.load(fb)
def sc(p): return bss_score(yv, oracle_shift(p, r)[0])
print(f"{'member':22s} {'A: train<2024':>14s} {'B: +2024H1':>12s} {'B-A':>7s}")
for n in members:
    a = sc(A[n]) if n in A else float('nan'); b = sc(B[n]) if n in B else float('nan')
    print(f"{n:22s} {a:14.1f} {b:12.1f} {b-a:+7.1f}")
comps = {
 'v15': {'mcs_pb':.47,'zoo_catids_lr02_s3':.41,'zoo_cat_mc':.12},
 'v17': {'zoo_catids_lr02_s3':.3,'zoo_cat_mc':.05,'bin_off':.15,'mcs_pb':.15,'mcs_off':.35},
 'v19': {'zoo_catids_lr02_s3':.21,'zoo_cat_mc':.035,'bin_off':.105,'mcs_pb':.105,'mcs_off':.245,'nn_ids_off':.2,'nn_wide3_off':.1},
 'v20a': {'zoo_catids_lr02_s3':.18,'zoo_cat_mc':.03,'bin_off':.09,'mcs_pb':.09,'mcs_off':.21,'nn_ids_off':.3,'nn_wide3_off':.1},
 'v20d': {'zoo_catids_lr02_s3':.35,'mcs_off':.25,'nn_ids_off':.3,'nn_wide3_off':.1},
 'v23': {'zoo_catids_lr02_s3':.35,'mcs_off':.25,'nn_ids_off_w15':.3,'nn_wide3_off':.1},
 'v24': {'zoo_catids_lr02_s3':.4,'mcs_off':.15,'nn_ids_off_w15':.35,'nn_wide3_off':.1},
 'noLGB': {'zoo_catids_lr02_s3':.45,'nn_ids_off_w15':.4,'nn_wide3_off':.15},
 'IDonly': {'zoo_catids_lr02_s3':.5,'nn_ids_off_w15':.5},
}
LB = {'v15':1109,'v17':1104,'v19':1124,'v20a':1126,'v20d':1133}
def blend(S, w):
    w = {kk:v for kk,v in w.items() if kk in S}
    if not w: return None
    return expit(sum(w[kk]*logit(np.clip(S[kk],1e-6,1-1e-6)) for kk in w)/sum(w.values()))
print(f"\n{'comp':8s} {'A':>8s} {'B':>8s} {'A-v15':>7s} {'B-v15':>7s} {'LB-1109':>8s}  missing")
ra = blend(A, comps['v15']); rb = blend(B, comps['v15'])
for c, w in comps.items():
    pa, pb = blend(A, w), blend(B, w); miss = [kk for kk in w if kk not in B]
    sa = sc(pa) if pa is not None else float('nan'); sb = sc(pb) if pb is not None else float('nan')
    print(f"{c:8s} {sa:8.1f} {sb:8.1f} {sa-sc(ra):+7.1f} {(sb-sc(rb)) if rb is not None else float('nan'):+7.1f} {('%+d'%(LB[c]-1109)) if c in LB else '?':>8s}  {miss}")

"""Given LB scores for the chain v20d -> v23 -> v25 -> v27a -> v27, estimate per-path transfer ratios (LB delta / fold-2024 delta).
Usage: python src/transfer_model.py v23=1140 v25=1145 v27a=1150 v27=1160   (any subset; unknown ones skipped)"""
import sys
FOLD = {'v20d': 971.6, 'v23': 976.0, 'v25': 979.1, 'v27a': 984.5, 'v27': 988.6, 'v26': 985.6, 'v24': 975.5}
LB = {'v20d': 1133.0}
for a in sys.argv[1:]:
    k, v = a.split('='); LB[k] = float(v)
chain = ['v20d', 'v23', 'v25', 'v27a', 'v27']
paths = {('v20d', 'v23'): 'NN-ID recency weight (ID path)', ('v23', 'v25'): 'LGB-offset CTX (offset path)', ('v25', 'v27a'): 'NN-ID CTX offset (ID+offset path)', ('v27a', 'v27'): 'CatBoost-ID CTX (ID path)'}
prev = None
for k in chain:
    if k in LB:
        if prev is not None:
            df, dl = FOLD[k] - FOLD[prev], LB[k] - LB[prev]
            print(f'{prev}->{k}: fold {df:+.1f}  LB {dl:+.1f}  ratio {dl/df if abs(df) > 0.3 else float("nan"):+.2f}   [{paths.get((prev, k), "combined")}]')
        prev = k
print('Rule of thumb: ratio >= 1 -> push that path harder (weights up); ratio <= 0 -> reduce dependence on that path.')

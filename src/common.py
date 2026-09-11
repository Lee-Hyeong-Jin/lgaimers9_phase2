"""Common utilities: paths, loading, metric."""
import os, json, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
EXP = os.path.join(ROOT, 'experiments')
MODELS = os.path.join(ROOT, 'models')
ID_COL, TARGET = 'row_id', 'control_success'


def load_train():
    p = os.path.join(DATA, 'train.parquet')
    if not os.path.exists(p):
        df = pd.read_csv(os.path.join(DATA, 'train.csv'), encoding='utf-8-sig')
        df.to_parquet(p)
    return pd.read_parquet(p)


def load_trackman():
    p = os.path.join(DATA, 'trackman.parquet')
    if not os.path.exists(p):
        df = pd.read_csv(os.path.join(DATA, 'trackman_history.csv'), encoding='utf-8-sig')
        df.to_parquet(p)
    return pd.read_parquet(p)


def brier(y, p):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def bss_score(y, p):
    """Competition score: 100000 * (1 - BS / (r(1-r)))."""
    y = np.asarray(y, dtype=float)
    r = y.mean()
    return 100000.0 * (1.0 - brier(y, p) / (r * (1 - r)))


def log_experiment(name, result: dict, path=None):
    path = path or os.path.join(EXP, 'results.jsonl')
    rec = {'name': name, 'time': time.strftime('%Y-%m-%d %H:%M:%S'), **result}
    with open(path, 'a') as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=float) + '\n')
    return rec

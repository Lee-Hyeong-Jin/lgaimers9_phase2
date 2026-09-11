import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp import run, RAW_FEATS, get_fold_data
from common import load_train
tr = load_train()
F, _ = get_fold_data(tr, 2024)
ALL = list(F.columns)
res0, _ = run('exp000_raw_lgb', RAW_FEATS)
res1, _ = run('exp001_allfeats_lgb', ALL)
import json
print(json.dumps(res1['importance_top'], indent=1))

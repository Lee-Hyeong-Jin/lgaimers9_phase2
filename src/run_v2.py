import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings('ignore')
from exp2 import run2, fold_data, feature_sets
from common import load_train
tr = load_train()
F, _ = fold_data(tr, 2024, 'prev')
fs = feature_sets(F)
ALL, V1 = fs['ALL'], fs['V1']
NO_ABS = [c for c in ALL if c not in fs['ABS_PRIOR']]
print("n ALL", len(ALL), "V1", len(V1), "NO_ABS", len(NO_ABS))
run2('exp010_v1_ref', V1)
run2('exp011_v2_all_prev_nooff', ALL, mode='prev', offset=False)
run2('exp012_v2_all_prev_off', ALL, mode='prev', offset=True)
run2('exp013_v2_all_true_off', ALL, mode='true', offset=True)
run2('exp014_v2_noabs_prev_off', NO_ABS, mode='prev', offset=True)
run2('exp015_v2_all_prev_off_noseason', [c for c in ALL if c != 'season'], mode='prev', offset=True)

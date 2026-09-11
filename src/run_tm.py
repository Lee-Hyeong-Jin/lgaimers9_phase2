import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings('ignore')
from exp2 import run2, fold_data, feature_sets
from common import load_train
tr = load_train()
F, _ = fold_data(tr, 2024, 'prev')
fs = feature_sets(F)
ALL = list(F.columns); V1 = fs['V1']
ROLE = [c for c in ALL if c.startswith('ROLE_')]; TM = [c for c in ALL if c.startswith('TM_')]
V1 = [c for c in V1 if c not in ROLE + TM]
SK = [c for c in ALL if c not in V1 + ROLE + TM]
print('V1', len(V1), 'ROLE', len(ROLE), 'TM', len(TM), 'SK', len(SK))
P = {'num_threads': 14}
run2('exp040_v1_role', V1 + ROLE, params=P)
run2('exp041_v1_tm', V1 + TM, params=P)
run2('exp042_v1_role_tm', V1 + ROLE + TM, params=P)
run2('exp043_all_role_tm', ALL, params=P)

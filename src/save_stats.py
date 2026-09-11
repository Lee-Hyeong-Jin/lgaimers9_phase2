import sys, os, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joblib
from common import load_train, ROOT
import features2 as f2
stats = f2.build_stats(load_train())
slim = {k: stats[k] for k in ['ps', 'bs', 'league', 'max_season', 'L_gt', 'L_all', 'ps2', 'bs2', 'FM_gt', 'team_p', 'team_b', 'role', 'tm', 'pbs', 'endform', 'src_eff']}
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'submit', 'model', 'stats.pkl')
joblib.dump(slim, out, compress=3); print('saved', out, {k: (v.shape if hasattr(v, 'shape') else v) for k, v in slim.items()})

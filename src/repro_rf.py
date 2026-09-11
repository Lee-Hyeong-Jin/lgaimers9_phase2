import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, time
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from common import load_train, bss_score
tr = load_train()
cat = ['top_bottom','game_type','base_state']
num = [c for c in tr.columns if c not in cat+['row_id','control_success']]
for vs in (2024, 2023):
    pre = ColumnTransformer([('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat), ('num', SimpleImputer(strategy='median'), num)])
    m = Pipeline([('pre', pre), ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=200, n_jobs=7, random_state=42))])
    a = tr[tr.season<vs]; b = tr[tr.season==vs]
    t=time.time(); m.fit(a.drop(columns=['row_id','control_success']), a.control_success)
    p = m.predict_proba(b.drop(columns=['row_id','control_success']))[:,1]
    print(f"RF baseline fold {vs}: score={bss_score(b.control_success.values, p):.1f} pred_mean={p.mean():.4f} y_mean={b.control_success.mean():.4f} ({time.time()-t:.0f}s)", flush=True)
    # also by month
    df = pd.DataFrame({'y':b.control_success.values,'p':p,'m':b.game_month.values})
    print(df.groupby('m').agg(y=('y','mean'),p=('p','mean')).round(4).T.to_string())

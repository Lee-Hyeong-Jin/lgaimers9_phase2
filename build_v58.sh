#!/bin/bash
# v58 = v57 constants (lg30 0.105) + robustness fixes (matchup early-return columns, string dtypes for code columns); v58b = v48 constants (lg30 0.09) + same fixes
cd /home/lhjin0j/Documents/lgaimers_phase2; PY=.venv/bin/python; SP=/tmp/claude-1003/-home-lhjin0j/00606e73-03e7-4212-9065-ceeeae595927/scratchpad
M=lgb_mc_off_ctx_lgm_fw2_meta.json,cat_mc_ids3_ctx_lgm_meta.json,nn_wide3_off_lgm_meta.json,nn_ids_off8w_ctx_lgm_meta.json
W=lgb_mc_off_ctx_lgm_fw2_meta.json:0.25,cat_mc_ids3_ctx_lgm_meta.json:0.35,nn_wide3_off_lgm_meta.json:0.10,nn_ids_off8w_ctx_lgm_meta.json:0.30
build() { $PY src/set_ensemble.py --models $M --weights $W --blend logit --prob_shift -0.004 --lg_shift 0.015 --lg_shift_p 0.015 --lg_shift_opp 0.015 --f_shift 0.0 --lg30_shift $2 --lg_shift_home 0.0 --lg_shift_away 0.0 > /dev/null; $PY src/build_submit.py | grep wrote; cp submit.zip $1; echo "### $1 md5 $(md5sum $1 | cut -c1-8) script sha1 $(unzip -p $1 script.py | sha1sum | cut -c1-10)"; }
build submit_v58_lg30_105_fix.zip 0.105
build submit_v58b_lg30_09_fix.zip 0.09
echo "=== strict audit v58 ==="; bash $SP/strict_audit.sh submit_v58_lg30_105_fix.zip 2>&1 | grep -E "^###|full run|format:|determinism|subset|reversed|single-row|official 5-row|Traceback|Error"
cp $SP/evalsim/output/full.csv $SP/evalsim/preds/submit_v58_lg30_105_fix.csv 2>/dev/null
echo "=== v58 full-file predictions vs v57 (expect 0) ==="; $PY - <<'PY'
import pandas as pd, numpy as np
SP='/tmp/claude-1003/-home-lhjin0j/00606e73-03e7-4212-9065-ceeeae595927/scratchpad/evalsim'
a=pd.read_csv(f'{SP}/preds/submit_v57_lg30_105.csv'); b=pd.read_csv(f'{SP}/preds/submit_v58_lg30_105_fix.csv'); assert (a.row_id.values==b.row_id.values).all()
print('max|v58 - v57| over 253,507 rows: %.2e'%np.abs(a.control_success.values-b.control_success.values).max())
PY
echo "=== crash scenarios on v58 (actual script, single-row files) ==="; cd $SP/evalsim && rm -rf model script.py requirements.txt output && unzip -q -o /home/lhjin0j/Documents/lgaimers_phase2/submit_v58_lg30_105_fix.zip -d . && $SP/evalsim/venv/bin/python - <<'PY'
import pandas as pd, numpy as np, joblib, subprocess, shutil
SP='/tmp/claude-1003/-home-lhjin0j/00606e73-03e7-4212-9065-ceeeae595927/scratchpad/evalsim'
full=pd.read_csv('data/test_full.csv', encoding='utf-8-sig'); st=joblib.load('model/stats.pkl'); pbs=set(st['pbs'].pb_key.values)
fullpred=pd.read_csv(f'{SP}/preds/submit_v58_lg30_105_fix.csv').set_index('row_id').control_success
def run(df, label, ref=None):
    df.to_csv('data/test.csv', index=False, encoding='utf-8-sig'); pd.DataFrame({'row_id': df.row_id.values, 'control_success': 0.5}).to_csv('data/sample_submission.csv', index=False, encoding='utf-8-sig'); shutil.rmtree('output', ignore_errors=True)
    r=subprocess.run([f'{SP}/venv/bin/python','script.py'], capture_output=True, text=True)
    if r.returncode!=0: print(f'{label}: CRASH ->', [l for l in r.stderr.strip().splitlines() if l.strip()][-1][:120]); return None
    o=pd.read_csv('output/submission.csv').set_index('row_id').control_success; print(f'{label}: ok', {k: round(float(v),9) for k,v in o.items()}, '' if ref is None else '| vs reference diff %.2e'%max(abs(float(o[k])-ref[k]) for k in o.index if k in ref)); return o
# (1) never-seen pair alone vs with companions
i=100000; row=full.iloc[[i]].copy(); pid=int(row.pitcher_id.iloc[0]); newb=[b for b in sorted(full.batter_id.unique()) if (pid*100000+b) not in pbs][0]; row['batter_id']=newb
a=run(row, '(1) new-pair row ALONE'); b=run(pd.concat([row, full.iloc[[200000]]]), '(1) new-pair row + 1 companion')
if a is not None and b is not None: print('    new-pair row: alone vs with companion diff %.2e'%abs(float(a.iloc[0])-float(b.loc[row.row_id.iloc[0]])))
# (2) base_state '123' row alone
j=int(np.where(full.row_id.values=='TEST_044931')[0][0]); run(full.iloc[[j]], '(2) bases-loaded row TEST_044931 ALONE', fullpred)
# (3) a 1-row file that is BOTH new-pair and bases-loaded
row2=full.iloc[[j]].copy(); pid2=int(row2.pitcher_id.iloc[0]); nb2=[b for b in sorted(full.batter_id.unique()) if (pid2*100000+b) not in pbs][0]; row2['batter_id']=nb2
c=run(row2, '(3) new-pair + bases-loaded row ALONE'); d=run(pd.concat([row2, full.iloc[[0]]]), '(3) same row + 1 companion')
if c is not None and d is not None: print('    diff alone vs companion %.2e'%abs(float(c.iloc[0])-float(d.loc[row2.row_id.iloc[0]])))
shutil.copy('data/test_full.csv','data/test.csv'); shutil.copy('data/sub_full.csv','data/sample_submission.csv')
PY
echo DONE_V58

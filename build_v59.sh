#!/bin/bash
# v59 = v57 constants (lg30 0.105) + corrected robustness fixes (matchup empty branch falls through with zero counts; string dtypes); v59b = v48 constants + same fixes
cd /home/lhjin0j/Documents/lgaimers_phase2; PY=.venv/bin/python; SP=/tmp/claude-1003/-home-lhjin0j/00606e73-03e7-4212-9065-ceeeae595927/scratchpad; J=/home/lhjin0j/.claude/jobs/00606e73/tmp; VP=$SP/evalsim/venv/bin/python
M=lgb_mc_off_ctx_lgm_fw2_meta.json,cat_mc_ids3_ctx_lgm_meta.json,nn_wide3_off_lgm_meta.json,nn_ids_off8w_ctx_lgm_meta.json
W=lgb_mc_off_ctx_lgm_fw2_meta.json:0.25,cat_mc_ids3_ctx_lgm_meta.json:0.35,nn_wide3_off_lgm_meta.json:0.10,nn_ids_off8w_ctx_lgm_meta.json:0.30
build() { $PY src/set_ensemble.py --models $M --weights $W --blend logit --prob_shift -0.004 --lg_shift 0.015 --lg_shift_p 0.015 --lg_shift_opp 0.015 --f_shift 0.0 --lg30_shift $2 --lg_shift_home 0.0 --lg_shift_away 0.0 > /dev/null; $PY src/build_submit.py | grep wrote; cp submit.zip $1; echo "### $1 md5 $(md5sum $1 | cut -c1-8) script sha1 $(unzip -p $1 script.py | sha1sum | cut -c1-10)"; }
build submit_v59_lg30_105_fix2.zip 0.105
build submit_v59b_lg30_09_fix2.zip 0.09
rm -f submit_v58_lg30_105_fix.zip submit_v58b_lg30_09_fix.zip
echo "=== strict audit v59 ==="; bash $SP/strict_audit.sh submit_v59_lg30_105_fix2.zip 2>&1 | grep -E "^###|full run|format:|determinism|subset|reversed|single-row|official 5-row|Traceback|Error"
cd $SP/evalsim && rm -rf model script.py requirements.txt output && unzip -q -o /home/lhjin0j/Documents/lgaimers_phase2/submit_v59_lg30_105_fix2.zip -d . && cp data/test_full.csv data/test.csv && cp data/sub_full.csv data/sample_submission.csv
echo "=== [1] v59 full run; equality vs v57/v48 ==="; $VP script.py 2>&1 | grep -E "saved|Traceback"; cp output/submission.csv preds/submit_v59_lg30_105_fix2.csv
$VP - <<'PY'
import pandas as pd, numpy as np
SP='/tmp/claude-1003/-home-lhjin0j/00606e73-03e7-4212-9065-ceeeae595927/scratchpad/evalsim'; b=pd.read_csv(f'{SP}/preds/submit_v59_lg30_105_fix2.csv')
for name in ['submit_v57_lg30_105','pred_submit_v48_lg30']:
    a=pd.read_csv(f'{SP}/preds/{name}.csv'); assert (a.row_id.values==b.row_id.values).all(); d=np.abs(a.control_success.values-b.control_success.values); print('max|v59 - %s| = %.2e ; rows >1e-9: %d'%(name, d.max(), int((d>1e-9).sum())))
PY
echo "=== [2] new-pair / bases-loaded scenarios on v59 (actual script) ==="; $VP - <<'PY'
import pandas as pd, numpy as np, joblib, subprocess, shutil
SP='/tmp/claude-1003/-home-lhjin0j/00606e73-03e7-4212-9065-ceeeae595927/scratchpad/evalsim'
full=pd.read_csv('data/test_full.csv', encoding='utf-8-sig'); st=joblib.load('model/stats.pkl'); pbs=set(st['pbs'].pb_key.values); bat=sorted(full.batter_id.unique())
fullpred=pd.read_csv(f'{SP}/preds/submit_v59_lg30_105_fix2.csv').set_index('row_id').control_success
def run(df, label):
    df.to_csv('data/test.csv', index=False, encoding='utf-8-sig'); pd.DataFrame({'row_id': df.row_id.values, 'control_success': 0.5}).to_csv('data/sample_submission.csv', index=False, encoding='utf-8-sig'); shutil.rmtree('output', ignore_errors=True)
    r=subprocess.run([f'{SP}/venv/bin/python','script.py'], capture_output=True, text=True)
    if r.returncode!=0: print(f'{label}: CRASH ->', [l for l in r.stderr.strip().splitlines() if l.strip()][-1][:120]); return None
    return pd.read_csv('output/submission.csv').set_index('row_id').control_success
def newpair(i):
    row=full.iloc[[i]].copy(); p=int(row.pitcher_id.iloc[0]); nb=[b for b in bat if (p*100000+b) not in pbs][0]; row['batter_id']=nb; return row
worst=0
for i in [100000, 5, 77777, 150001, 240000]:
    row=newpair(i); rid=row.row_id.iloc[0]
    a=run(row,'alone'); b=run(pd.concat([row, full.iloc[[200000]]]),'+1'); c=run(pd.concat([row, full.iloc[200000:200500]]),'+500'); d=run(pd.concat([full.iloc[100:300], row]),'+200 before')
    vals=[float(x[rid]) for x in (a,b,c,d)]; w=max(vals)-min(vals); worst=max(worst,w); print('new-pair row %s: alone %.9f | +1 %.9f | +500 %.9f | 200 before %.9f | spread %.2e'%(rid, *vals, w))
print('WORST spread over 5 new-pair rows: %.2e'%worst)
j=int(np.where(full.row_id.values=='TEST_044931')[0][0]); o=run(full.iloc[[j]],'bases-loaded alone'); print('bases-loaded row alone vs full: %.2e'%abs(float(o['TEST_044931'])-float(fullpred['TEST_044931'])))
hyp=full.iloc[300:500].copy()
for k in range(len(hyp)):
    p=int(hyp.pitcher_id.values[k]); nb=[b for b in bat if (p*100000+b) not in pbs][0]; hyp.iat[k, hyp.columns.get_loc('batter_id')]=int(nb)
o4=run(hyp,'only-new-pairs 200'); o5=run(pd.concat([hyp, full.iloc[[0,1,2]]]),'only-new-pairs 200 + 3 normal'); print('200 never-seen-pair rows: alone vs with 3 normal rows max|diff| %.2e'%max(abs(float(o4[r])-float(o5[r])) for r in hyp.row_id.values))
shutil.copy('data/test_full.csv','data/test.csv'); shutil.copy('data/sub_full.csv','data/sample_submission.csv')
PY
echo "=== [3] intermediates for a new-pair row: alone vs companion (must be bitwise equal) ==="; $VP - <<'PY' 2>&1 | grep -v Warning
import importlib.util, json, numpy as np, pandas as pd, joblib, torch
spec=importlib.util.spec_from_file_location('sub','script.py'); sub=importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
stats=joblib.load('model/stats.pkl'); ens=json.load(open('model/ensemble.json')); metas={m: json.load(open('model/'+m)) for m in ens['models']}
full=pd.read_csv('data/test_full.csv', encoding='utf-8-sig', dtype={'base_state': str}); pbs=set(stats['pbs'].pb_key.values)
row=full.iloc[[100000]].copy(); p=int(row.pitcher_id.iloc[0]); row['batter_id']=[b for b in sorted(full.batter_id.unique()) if (p*100000+b) not in pbs][0]
def feats(df):
    df=df.reset_index(drop=True); return df, sub.add_change_features(sub.build_features(df, stats, mode='prev', guess=None))
dA,FA=feats(row); dB,FB=feats(pd.concat([row, full.iloc[[200000]]])); fe=metas[ens['models'][0]]['feats']
a=FA.iloc[0][fe].astype(float).values; b=FB.iloc[0][fe].astype(float).values; d=np.abs(a-b); d[np.isnan(a)&np.isnan(b)]=0; print('178 features max|diff| (bitwise): %.2e'%d.max())
prep=joblib.load('model/nn_ids_off8w_ctx_lgm_prep.pkl'); X=[]
for df,F in ((dA,FA),(dB,FB)):
    Fn=F.copy(); Fn['season_rel']=np.clip(prep['max_season']-Fn['season'].values,0,10); X.append(sub.transform_numeric(Fn, prep['num_feats'], prep['qt'], prep['nan_cols'])[0])
print('NN input max|diff|: %.2e'%np.abs(X[0]-X[1]).max())
PY
echo DONE_V59

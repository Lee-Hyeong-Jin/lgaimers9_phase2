#!/bin/bash
# v53-v56: lg30 raised to 0.14 (LB inversion of v48=1224 -> e=0.142); probes rebuilt on top. Sequential audits (sandbox must not be shared).
cd /home/lhjin0j/Documents/lgaimers_phase2; PY=.venv/bin/python; AUD=/tmp/claude-1003/-home-lhjin0j/00606e73-03e7-4212-9065-ceeeae595927/scratchpad/audit.sh
build() { $PY src/set_ensemble.py --models lgb_mc_off_ctx_lgm_fw2_meta.json,cat_mc_ids3_ctx_lgm_meta.json,nn_wide3_off_lgm_meta.json,nn_ids_off8w_ctx_lgm_meta.json --weights lgb_mc_off_ctx_lgm_fw2_meta.json:0.25,cat_mc_ids3_ctx_lgm_meta.json:0.35,nn_wide3_off_lgm_meta.json:0.10,nn_ids_off8w_ctx_lgm_meta.json:0.30 --prob_shift $2 --blend logit --lg_shift $3 --lg_shift_p $4 --lg_shift_opp $5 --f_shift $6 --lg30_shift $7 --lg_shift_home 0.0 --lg_shift_away 0.0 > /dev/null; $PY src/build_submit.py | grep wrote; cp submit.zip $1; echo "### $1 $(md5sum $1 | cut -c1-8)"; bash $AUD /home/lhjin0j/Documents/lgaimers_phase2/$1 2>&1 | grep -v -i warning | grep -E 'ensemble:|full run|single-row|full:|Traceback|Error'; cp /tmp/claude-1003/-home-lhjin0j/00606e73-03e7-4212-9065-ceeeae595927/scratchpad/evalsim/output/full.csv /tmp/claude-1003/-home-lhjin0j/00606e73-03e7-4212-9065-ceeeae595927/scratchpad/evalsim/preds/${1%.zip}.csv; }
build submit_v53_lg30_14.zip          -0.004 0.015 0.015 0.015 0.0    0.14
build submit_v54_lg30_14_fplus10.zip  -0.004 0.015 0.015 0.015 0.010  0.14
build submit_v55_lg30_14_split.zip    -0.004 0.015 0.011 0.019 0.0    0.14
build submit_v56_lg30_14_nonlg.zip    -0.002 0.013 0.013 0.013 -0.002 0.14
echo DONE_v53_56

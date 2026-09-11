#!/bin/bash
# rebuild v48-v51 with the restructured (independent) shift blocks; sequential audits
cd /home/lhjin0j/Documents/lgaimers_phase2; PY=.venv/bin/python; AUD=/tmp/claude-1003/-home-lhjin0j/00606e73-03e7-4212-9065-ceeeae595927/scratchpad/audit.sh
build() { $PY src/set_ensemble.py --models lgb_mc_off_ctx_lgm_fw2_meta.json,cat_mc_ids3_ctx_lgm_meta.json,nn_wide3_off_lgm_meta.json,nn_ids_off8w_ctx_lgm_meta.json --weights lgb_mc_off_ctx_lgm_fw2_meta.json:0.25,cat_mc_ids3_ctx_lgm_meta.json:0.35,nn_wide3_off_lgm_meta.json:0.10,nn_ids_off8w_ctx_lgm_meta.json:0.30 --prob_shift $2 --blend logit --lg_shift $3 --lg_shift_p $4 --lg_shift_opp $5 --f_shift $6 --lg30_shift 0.09 --lg_shift_home 0.0 --lg_shift_away 0.0 > /dev/null; $PY src/build_submit.py | grep wrote; cp submit.zip $1; echo "### $1"; bash $AUD /home/lhjin0j/Documents/lgaimers_phase2/$1 2>&1 | grep -v -i warning | grep -E 'full run|single-row|full:|Traceback|Error'; }
build submit_v48_lg30.zip -0.004 0.015 0.015 0.015 0.0
build submit_v49_lg30_fplus10.zip -0.004 0.015 0.015 0.015 0.010
build submit_v50_lg30_nonlg.zip -0.002 0.013 0.013 0.013 -0.002
build submit_v51_lg30_split.zip -0.004 0.015 0.011 0.019 0.0

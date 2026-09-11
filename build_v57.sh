#!/bin/bash
# v57: v48 with lg30 = 0.105 (= fold-2024 training residual, unshrunk). Training-grounded alternative to LB-inverted 0.14.
cd /home/lhjin0j/Documents/lgaimers_phase2; PY=.venv/bin/python; AUD=/tmp/claude-1003/-home-lhjin0j/00606e73-03e7-4212-9065-ceeeae595927/scratchpad/audit.sh
$PY src/set_ensemble.py --models lgb_mc_off_ctx_lgm_fw2_meta.json,cat_mc_ids3_ctx_lgm_meta.json,nn_wide3_off_lgm_meta.json,nn_ids_off8w_ctx_lgm_meta.json --weights lgb_mc_off_ctx_lgm_fw2_meta.json:0.25,cat_mc_ids3_ctx_lgm_meta.json:0.35,nn_wide3_off_lgm_meta.json:0.10,nn_ids_off8w_ctx_lgm_meta.json:0.30 --prob_shift -0.004 --blend logit --lg_shift 0.015 --lg_shift_p 0.015 --lg_shift_opp 0.015 --f_shift 0.0 --lg30_shift 0.105 --lg_shift_home 0.0 --lg_shift_away 0.0 > /dev/null
$PY src/build_submit.py | grep wrote; cp submit.zip submit_v57_lg30_105.zip; echo "### submit_v57_lg30_105.zip $(md5sum submit_v57_lg30_105.zip | cut -c1-8)"
bash $AUD /home/lhjin0j/Documents/lgaimers_phase2/submit_v57_lg30_105.zip 2>&1 | grep -v -i warning | grep -E 'ensemble:|full run|single-row|full:|Traceback|Error'
cp /tmp/claude-1003/-home-lhjin0j/00606e73-03e7-4212-9065-ceeeae595927/scratchpad/evalsim/output/full.csv /tmp/claude-1003/-home-lhjin0j/00606e73-03e7-4212-9065-ceeeae595927/scratchpad/evalsim/preds/submit_v57_lg30_105.csv
echo DONE_v57

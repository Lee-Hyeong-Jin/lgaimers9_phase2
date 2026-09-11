#!/bin/bash
# v41 = v33 + LG rows +0.015 ; v43 = v37 weights (cat .40/lgb .25/nnid .35) + LG rows +0.010
set -e
cd /home/lhjin0j/Documents/lgaimers_phase2; PY=.venv/bin/python; AUD=/tmp/claude-1003/-home-lhjin0j/00606e73-03e7-4212-9065-ceeeae595927/scratchpad/audit.sh
$PY src/set_ensemble.py --models lgb_mc_off_ctx_lgm_fw2_meta.json,cat_mc_ids3_ctx_lgm_meta.json,nn_wide3_off_lgm_meta.json,nn_ids_off8w_ctx_lgm_meta.json \
  --weights lgb_mc_off_ctx_lgm_fw2_meta.json:0.25,cat_mc_ids3_ctx_lgm_meta.json:0.35,nn_wide3_off_lgm_meta.json:0.10,nn_ids_off8w_ctx_lgm_meta.json:0.30 --prob_shift -0.004 --blend logit --lg_shift 0.015 --f_shift 0.0
$PY src/build_submit.py; cp submit.zip submit_v41_lgplus15.zip; bash $AUD /home/lhjin0j/Documents/lgaimers_phase2/submit_v41_lgplus15.zip
$PY src/set_ensemble.py --models lgb_mc_off_ctx_lgm_fw2_meta.json,cat_mc_ids3_ctx_lgm_meta.json,nn_ids_off8w_ctx_lgm_meta.json \
  --weights lgb_mc_off_ctx_lgm_fw2_meta.json:0.25,cat_mc_ids3_ctx_lgm_meta.json:0.40,nn_ids_off8w_ctx_lgm_meta.json:0.35 --prob_shift -0.004 --blend logit --lg_shift 0.010 --f_shift 0.0
$PY src/build_submit.py; cp submit.zip submit_v43_weights_lgplus10.zip; bash $AUD /home/lhjin0j/Documents/lgaimers_phase2/submit_v43_weights_lgplus10.zip

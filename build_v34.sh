#!/bin/bash
# v34 = hedge: v33 members (full seeds) + v27 members (CatBoost 3 seeds, LGB 4 seeds) at half weights
set -e
cd /home/lhjin0j/Documents/lgaimers_phase2; PY=.venv/bin/python
$PY src/set_ensemble.py --models lgb_mc_off_ctx_lgm_fw2_meta.json,cat_mc_ids3_ctx_lgm_meta.json,nn_wide3_off_lgm_meta.json,nn_ids_off8w_ctx_lgm_meta.json,lgb_mc_off_ctx_s4_meta.json,cat_mc_ids3_ctx_s3_meta.json,nn_wide3_off_meta.json,nn_ids_off8w_ctx_meta.json \
  --weights lgb_mc_off_ctx_lgm_fw2_meta.json:0.125,cat_mc_ids3_ctx_lgm_meta.json:0.175,nn_wide3_off_lgm_meta.json:0.05,nn_ids_off8w_ctx_lgm_meta.json:0.15,lgb_mc_off_ctx_s4_meta.json:0.125,cat_mc_ids3_ctx_s3_meta.json:0.175,nn_wide3_off_meta.json:0.05,nn_ids_off8w_ctx_meta.json:0.15 --prob_shift -0.004 --blend logit
$PY src/build_submit.py
cp submit.zip submit_v34_hedge.zip
bash /tmp/claude-1003/-home-lhjin0j/00606e73-03e7-4212-9065-ceeeae595927/scratchpad/audit.sh /home/lhjin0j/Documents/lgaimers_phase2/submit_v34_hedge.zip

#!/bin/bash
# v27h = v27 with ID-heavier weights (CatBoost-CTX .45, LGB-CTX .15)
set -e
cd /home/lhjin0j/Documents/lgaimers_phase2; PY=.venv/bin/python
$PY src/set_ensemble.py --models lgb_mc_off_ctx_meta.json,cat_mc_ids3_ctx_meta.json,nn_wide3_off_meta.json,nn_ids_off8w_ctx_meta.json \
  --weights lgb_mc_off_ctx_meta.json:0.15,cat_mc_ids3_ctx_meta.json:0.45,nn_wide3_off_meta.json:0.10,nn_ids_off8w_ctx_meta.json:0.30 --prob_shift -0.004 --blend logit
$PY src/build_submit.py
cp submit.zip submit_v27h_idheavy.zip
bash /tmp/claude-1003/-home-lhjin0j/00606e73-03e7-4212-9065-ceeeae595927/scratchpad/audit.sh /home/lhjin0j/Documents/lgaimers_phase2/submit_v27h_idheavy.zip

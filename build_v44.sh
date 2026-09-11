#!/bin/bash
# tomorrow's probes on top of v41 (LG +0.015):
# v44 = + F rows -0.005 ; v45 = non-LG rows +0.002 (prob_shift -0.002, lg total kept +0.011 rel., F kept) ; v46 = split LG shift: LG pitching +0.011, opponent pitching +0.019
set -e
cd /home/lhjin0j/Documents/lgaimers_phase2; PY=.venv/bin/python; AUD=/tmp/claude-1003/-home-lhjin0j/00606e73-03e7-4212-9065-ceeeae595927/scratchpad/audit.sh
M=lgb_mc_off_ctx_lgm_fw2_meta.json,cat_mc_ids3_ctx_lgm_meta.json,nn_wide3_off_lgm_meta.json,nn_ids_off8w_ctx_lgm_meta.json
W=lgb_mc_off_ctx_lgm_fw2_meta.json:0.25,cat_mc_ids3_ctx_lgm_meta.json:0.35,nn_wide3_off_lgm_meta.json:0.10,nn_ids_off8w_ctx_lgm_meta.json:0.30
$PY src/set_ensemble.py --models $M --weights $W --prob_shift -0.004 --blend logit --lg_shift 0.015 --lg_shift_p 0.015 --lg_shift_opp 0.015 --f_shift -0.005
$PY src/build_submit.py; cp submit.zip submit_v44_fshift.zip; bash $AUD /home/lhjin0j/Documents/lgaimers_phase2/submit_v44_fshift.zip
$PY src/set_ensemble.py --models $M --weights $W --prob_shift -0.002 --blend logit --lg_shift 0.013 --lg_shift_p 0.013 --lg_shift_opp 0.013 --f_shift -0.002
$PY src/build_submit.py; cp submit.zip submit_v45_nonlg.zip; bash $AUD /home/lhjin0j/Documents/lgaimers_phase2/submit_v45_nonlg.zip
$PY src/set_ensemble.py --models $M --weights $W --prob_shift -0.004 --blend logit --lg_shift 0.015 --lg_shift_p 0.011 --lg_shift_opp 0.019 --f_shift 0.0
$PY src/build_submit.py; cp submit.zip submit_v46_lgsplit.zip; bash $AUD /home/lhjin0j/Documents/lgaimers_phase2/submit_v46_lgsplit.zip

#!/usr/bin/env bash
# End-to-end reproduction of the final submission (v15/v16 configuration). Run from the project root.
set -e
PY=.venv/bin/python
uv venv --python 3.11.15 .venv -q
uv pip install -q "numpy==1.26.4" "pandas==2.0.3" "scipy==1.15.3" "scikit-learn==1.8.0" "joblib==1.5.3" "lightgbm==4.7.0" "catboost==1.2.10" pyarrow
$PY src/trackman_match.py                         # Trackman <-> main ID crosswalk (data/derived)
$PY src/save_stats.py                              # training-data statistics -> submit/model/stats.pkl
$PY src/train_final.py --model lgb_mc --feats V1 --rounds 180 --seeds 0,1,2,3,4,5,6,7 --tag lgb_mc9 --weight 0.47 --wexp 0.15 --params '{"num_threads":8}'
$PY src/train_final.py --model cat_mc --cat_ids --feats V1 --rounds 2750 --seeds 0,1,2,3,4,5,6,7 --tag cat_mc_ids3 --weight 0.41 --params '{"lr":0.02,"depth":7,"l2":10}'
$PY src/train_final.py --model cat_mc --feats V1 --rounds 800 --seeds 0,1,2 --tag cat_mc --weight 0.05 --params '{"lr":0.05,"depth":7,"l2":10}'
$PY src/train_final.py --model lgb_mc --feats V1 --rounds 180 --seeds 0,1,2,3,4,5,6,7 --tag lgb_mc_off --weight 0.35 --wexp 0.15 --offset bayes --params '{"num_threads":8}'
$PY src/train_final.py --model lgb_bin --feats V1 --rounds 180 --seeds 0,1,2,3 --tag lgb_bin_off --weight 0.15 --wexp 0.15 --offset bayes --params '{"num_threads":6}'
$PY src/set_ensemble.py --models lgb_mc_off_meta.json,lgb_mc9_meta.json,lgb_bin_off_meta.json,cat_mc_ids3_meta.json,cat_mc_meta.json --weights lgb_mc_off_meta.json:0.35,lgb_mc9_meta.json:0.15,lgb_bin_off_meta.json:0.15,cat_mc_ids3_meta.json:0.30,cat_mc_meta.json:0.05 --prob_shift -0.005 --blend logit
$PY src/build_submit.py                            # -> submit.zip

# ===== v33 (08-28): LG label-regime features (LG_game / LG_regime with the May-2023 switch, default ON in features2.py),
#       CTX platoon/two-strike priors (CTX_FEATS=1), Futures rows x2 for the LGB member, CTX offsets for the NNs =====
# stats (must include n_2k/s_2k/n_LG/s_LG/src_eff): $PY src/save_stats.py
# CTX_FEATS=1 $PY src/train_final.py --model lgb_mc --feats V1 --rounds 180 --seeds 0,1,2,3,4,5,6,7 --tag lgb_mc_off_ctx_lgm_fw2 --weight 0.25 --wexp 0.15 --fw 2.0 --offset bayes --params '{"num_threads":8}'
# CTX_FEATS=1 $PY src/train_final.py --model cat_mc --cat_ids --feats V1 --rounds 2750 --seeds 0,1,2,3,4,5,6,7 --tag cat_mc_ids3_ctx_lgm --weight 0.35 --params '{"lr":0.02,"depth":7,"l2":10}'
# $PY src/nn_export.py --tag nn_ids_off8w_ctx_lgm --epochs 8 --lr 1e-3 --hidden 768,384,192 --drop 0.2 --id_drop 0.15 --seeds 0,1,2,3,4,5,6,7 --weight 0.3 --offset --ctx_offset --wexp 0.15
# $PY src/nn_export.py --tag nn_wide3_off_lgm --epochs 8 --lr 1e-3 --hidden 768,384,192 --drop 0.2 --id_drop 0.15 --seeds 0,1,2 --weight 0.1 --offset --no_ids
# bash build_v33.sh   # set_ensemble (.25/.35/.10/.30, shift -0.004, logit blend) -> build_submit -> submit_v33_lgmay.zip -> audit
# ===== v41 (08-28, LB 1207.8 = current best): v33 members + LG-involved KBO rows +0.015 (ensemble.json lg_shift; row's own team ids only) =====
# $PY src/set_ensemble.py --models lgb_mc_off_ctx_lgm_fw2_meta.json,cat_mc_ids3_ctx_lgm_meta.json,nn_wide3_off_lgm_meta.json,nn_ids_off8w_ctx_lgm_meta.json \
#   --weights lgb_mc_off_ctx_lgm_fw2_meta.json:0.25,cat_mc_ids3_ctx_lgm_meta.json:0.35,nn_wide3_off_lgm_meta.json:0.10,nn_ids_off8w_ctx_lgm_meta.json:0.30 --prob_shift -0.004 --blend logit --lg_shift 0.015 --f_shift 0.0
# $PY src/build_submit.py   # -> submit_v41_lgplus15.zip

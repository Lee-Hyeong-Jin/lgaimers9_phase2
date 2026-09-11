#!/usr/bin/env bash
# End-to-end reproduction of the final submission v59 (LB 1225; = v48 models + 3-0 constant 0.105 + robustness fixes) from the official data only.
# Run from the project root: bash reproduce.sh   (needs uv, an NVIDIA GPU for CatBoost-GPU + PyTorch; ~1 h on RTX 3060 / 16 CPU)
# Pipeline: train.csv + trackman_history.csv -> ID crosswalk -> training statistics (stats.pkl) -> 4 members -> ensemble.json -> submit.zip
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
mkdir -p submit/model experiments data/derived
[ -x .venv/bin/python ] || uv venv --python 3.11.15 .venv -q
uv pip install -q --python $PY "numpy==1.26.4" "pandas==2.0.3" "scipy==1.15.3" "scikit-learn==1.8.0" "joblib==1.5.3" "lightgbm==4.7.0" "catboost==1.2.10" "pyarrow==25.0.1"
uv pip install -q --python $PY "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu128
export LG_FEATS=1 LG_MAY=1 CTX_FEATS=1          # LG label-regime features (May-2023 switch) + CTX priors for the tree members; the NN members use CTX only through their Bayes+CTX offset (CTX_FEATS=0 below), as in the original training
echo "[1/6] trackman <-> main ID crosswalk (data/derived)";        $PY src/trackman_match.py
echo "[2/6] training statistics -> submit/model/stats.pkl";       $PY src/save_stats.py
echo "[3/6] LGB multiclass + Bayes offset, Futures x2 (CPU) || CatBoost-ID multiclass (GPU)"
$PY src/train_final.py --model lgb_mc --feats V1 --rounds 180 --seeds 0,1,2,3,4,5,6,7 --tag lgb_mc_off_ctx_lgm_fw2 --weight 0.25 --wexp 0.15 --fw 2.0 --offset bayes --params '{"num_threads":8}' > experiments/repro_lgb.log 2>&1 &
LGB_PID=$!
$PY src/train_final.py --model cat_mc --cat_ids --feats V1 --rounds 2750 --seeds 0,1,2,3,4,5,6,7 --tag cat_mc_ids3_ctx_lgm --weight 0.35 --params '{"lr":0.02,"depth":7,"l2":10}' > experiments/repro_cat.log 2>&1
wait $LGB_PID
echo "[4/6] NN-ID (8 seeds, Bayes+CTX offset) and NN no-ID (3 seeds, Bayes offset)"
CTX_FEATS=0 $PY src/nn_export.py --tag nn_ids_off8w_ctx_lgm --epochs 8 --lr 1e-3 --hidden 768,384,192 --drop 0.2 --id_drop 0.15 --seeds 0,1,2,3,4,5,6,7 --weight 0.3 --offset --ctx_offset --wexp 0.15
CTX_FEATS=0 $PY src/nn_export.py --tag nn_wide3_off_lgm --epochs 8 --lr 1e-3 --hidden 768,384,192 --drop 0.2 --id_drop 0.15 --seeds 0,1,2 --weight 0.1 --offset --no_ids
M=lgb_mc_off_ctx_lgm_fw2_meta.json,cat_mc_ids3_ctx_lgm_meta.json,nn_wide3_off_lgm_meta.json,nn_ids_off8w_ctx_lgm_meta.json
W=lgb_mc_off_ctx_lgm_fw2_meta.json:0.25,cat_mc_ids3_ctx_lgm_meta.json:0.35,nn_wide3_off_lgm_meta.json:0.10,nn_ids_off8w_ctx_lgm_meta.json:0.30
echo "[5/6] v59 (FINAL, LB 1225): logit blend, prob_shift -0.004, LG-involved KBO rows +0.015, LG-R 3-0 count +0.105 (constants: docs/CONSTANTS_RATIONALE.md)"
$PY src/set_ensemble.py --models $M --weights $W --blend logit --prob_shift -0.004 --lg_shift 0.015 --lg_shift_p 0.015 --lg_shift_opp 0.015 --f_shift 0.0 --lg30_shift 0.105 --lg_shift_home 0.0 --lg_shift_away 0.0
$PY src/build_submit.py && cp submit.zip submit_v59_repro.zip
echo "[6/6] v59b (backup): same with LG-R 3-0 count +0.09 (v48 constant)"
$PY src/set_ensemble.py --models $M --weights $W --blend logit --prob_shift -0.004 --lg_shift 0.015 --lg_shift_p 0.015 --lg_shift_opp 0.015 --f_shift 0.0 --lg30_shift 0.09 --lg_shift_home 0.0 --lg_shift_away 0.0
$PY src/build_submit.py && cp submit.zip submit_v59b_repro.zip
echo DONE_REPRO

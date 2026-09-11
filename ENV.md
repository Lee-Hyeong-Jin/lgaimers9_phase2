# 개발 환경 (학습 코드 재현용)
- OS: Linux-6.18.44-1-lts-x86_64-with-glibc2.44
- Python: 3.11.15 (uv-managed venv, 평가 서버 3.11.15와 동일)
- numpy 1.26.4, pandas 2.0.3, scipy 1.15.3, scikit-learn 1.8.0, joblib 1.5.3
- lightgbm 4.7.0, xgboost 3.2.0, catboost 1.2.10, torch 2.11.0+cu128
- HW: 16 vCPU, 31GB RAM, RTX 3060 12GB

## 최종 제출(v48/v57) 재현 환경 (08-29 확정)
- Python 3.11.15 (uv), numpy 1.26.4, pandas 2.0.3, scipy 1.15.3, scikit-learn 1.8.0, joblib 1.5.3, pyarrow 25.0.1, lightgbm 4.7.0, catboost 1.2.10 (GPU 학습), torch 2.11.0+cu128 (NN 학습·추론 스크립트는 CPU/GPU 무관)
- 학습 하드웨어: RTX 3060 12GB + 16 CPU, 약 1시간 (LGB 8시드 CPU ‖ CatBoost-ID 8시드 GPU → NN 8+3시드)
- 재현: `bash reproduce.sh` (공식 데이터 2종만 입력 → `submit_v59_repro.zip` = 최종본 v59(LB 1225) 구성, `submit_v59b_repro.zip` = v48 상수 예비). 구 v15 레시피는 `reproduce_v15_legacy.sh`.

# LG Aimers 9기 Phase 2 — 투구 제구 성공 확률 예측

**최종 결과: Private LB 1225.20622** — 최종 제출 `submit_v59_lg30_105_fix2.zip` (빌드 `build_v59.sh`, 규칙 검토 `docs/RULE_AUDIT_v59.md`, 솔루션 요약 `docs/SOLUTION.md`)

> 대회 데이터(`data/`, `open.zip`)와 모델 가중치(`submit/model/`), 개인정보 포함 제출물(`final_submission*`)은 저장소에서 제외되어 있습니다. 재현은 아래 절차 참고.

## 구조
- `data/` 원본 데이터 (+ `data/derived/` Trackman↔메인 ID 매핑, 학습 데이터로만 생성)
- `src/common.py` 로딩/지표, `src/cv.py` 시계열 폴드, `src/features.py`(v1)·`src/features2.py`(v2)·`src/trackman_feats.py` 피처
- `src/exp.py`, `src/exp2.py`, `src/sweep.py`, `src/zoo.py`, `src/nn.py`, `src/blend.py` 실험 러너
- `src/train_final.py` 최종 학습 → `submit/model/`, `src/build_submit.py` → `submit.zip` (script.py에 피처 코드 인라인)
- `experiments/LOG.md` 실험 기록, `experiments/results.jsonl` 자동 로그

## 규칙 준수 설계
- 추론 시 각 test 행은 **자기 행의 컬럼 + 학습 데이터로 미리 계산한 통계(stats.pkl)** 만 사용. test 행 간 집계/정렬/누적 없음.
- 검증: 단일 행 test / 순서 셔플 / 전체 test 예측이 완전히 동일(차이 0.0).
- Trackman은 2019~2024만 사용, 행의 시즌보다 이전 시즌 집계만 사용.

## 재현
```
uv venv --python 3.11.15 .venv && source .venv/bin/activate
uv pip install numpy==1.26.4 pandas==2.0.3 scipy==1.15.3 scikit-learn==1.8.0 joblib==1.5.3 lightgbm xgboost catboost pyarrow
python src/trackman_match.py            # ID 매핑 (data/derived)
python src/train_final.py --model lgb_mc --feats V1 --rounds 165 --seeds 0,1,2,3,4 --tag lgb_mc
python src/train_final.py --model lgb_bin --feats V1 --rounds 180 --seeds 0,1,2,3,4 --tag lgb_v1
python src/set_ensemble.py --weights lgb_mc_meta.json:0.7,lgb_v1_meta.json:0.3 --prob_shift 0.0
python src/build_submit.py
```

## 최종 제출 재현 (v15/v16 구성)
`./reproduce.sh` — Trackman 매칭 → 통계 저장 → LGB 다중클래스 8시드(최근가중) → CatBoost-ID 8시드(lr0.02×2750) → CatBoost MC 3시드 → 앙상블 설정(.47/.41/.12, shift −0.005) → submit.zip

## 08-28 현재 최선 (LB 실측 1207.8)
- `submit_v41_lgplus15.zip` = v33(fold 2024 오라클 1022.9; LB 1195) + LG 관여 KBO 행 +0.015 시프트(LB 프로브 역산 e≈+0.013). 핵심: LG 관여 경기의 라벨 체제(2023년 5월 전환)를 명시하는 `LG_game`/`LG_regime` 피처(features2.py 기본 ON) + CTX 플래툰/2K prior + LGB F×2 + NN CTX 오프셋. 재현 명령은 `reproduce.sh` 하단, 후보 순위·근거는 `docs/CANDIDATES_0829.md`.

## 재현 (2차 평가용)
- `bash reproduce.sh` → `submit_v59_repro.zip`(최종본 v59, LB 1225 구성) / `submit_v59b_repro.zip`(v48 상수 예비). 최종 제출 파일: `submit_v59_lg30_105_fix2.zip`. 환경·소요 시간은 `ENV.md`, 재현 검증 결과는 `docs/SOLUTION.md` "재현성 검증", 후처리 상수의 근거는 `docs/CONSTANTS_RATIONALE.md`.

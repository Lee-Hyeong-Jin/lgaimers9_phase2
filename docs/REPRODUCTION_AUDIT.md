# REPRODUCTION_AUDIT — 제출 학습 코드의 독립 클린룸 재현 검증 (2026-09-02)

증거 로그: `experiments/reproduction_audit.log` (인벤토리·명령·시각·해시·비교 수치 전부 기록). 원칙: 실행·재생성·수치 비교만 신뢰. 이전 대화·의도·기억은 증거로 사용하지 않음.

## 1. 파일 인벤토리 (검증 대상: `final_submission/02_학습재현코드`)
- 코드 19개 파일만 존재: `src/*.py` 17개 + `reproduce.sh` + `README_실행방법.md` (전 파일 sha256 기록, 로그 [INV]).
- 데이터·기존 모델·기존 예측·피처 캐시·중간 산출물: **폴더에 없음**. 재사용 금지 대상(프로젝트의 기존 zip·preds·repro 디렉터리)은 클린룸에 복사하지 않음.

## 2. 재현에 필요한 최소 파일
위 19개 + 공식 데이터 4개(`train.csv, trackman_history.csv, test.csv, sample_submission.csv`) + uv/인터넷(패키지 설치) + NVIDIA GPU. 그 외 없음.

## 3–4. 학습/추론 파이프라인 (코드에서 재구성)
`reproduce.sh`: mkdir → uv venv(3.11.15)·패키지 고정 설치 → env(LG_FEATS/LG_MAY/CTX_FEATS=1; NN은 CTX_FEATS=0) → ① `trackman_match.py`(train+trackman→투수/타자 대응표) → ② `save_stats.py`(train 전체→stats.pkl 16표) → ③ `train_final.py` LGB 8시드(CPU) ‖ CatBoost 8시드(GPU) → ④ `nn_export.py` 8+3시드 → ⑤ `set_ensemble.py`(상수) → ⑥ `build_submit.py`(추론 스크립트 인라인+zip). 추론: zip의 `script.py`(test+model만 읽음). dead code: `cv.py`의 폴드 유틸(검증용), 학습 전용 함수는 추론에서 도달 불가.

## 5. 의존성/환경
README 명시 = 코드 import와 일치(numpy/pandas/scipy/sklearn/joblib/pyarrow/lightgbm/catboost/torch, uv, GPU). 누락 없음.

## 6. 숨은 의존성 감사 (로그 [HID])
절대경로/홈/getcwd/expanduser 0건(모든 경로가 스크립트 위치 기준). 환경변수는 기본값 있는 플래그뿐이며 reproduce.sh가 명시 export. 인터넷 = 패키지 설치 1곳(문서화됨). locale/타임존 의존 없음. **발견된 요구사항**: uv 바이너리, NVIDIA GPU(CatBoost task_type=GPU) — 둘 다 README/스크립트에 문서화됨(DOCUMENTED).

## 7. Precomputed artifact 감사
학습이 로드하는 사전 산출물: 없음(대응표·stats·prep 전부 이 실행에서 생성 = REGENERATABLE, 실증: Run A/B에서 재생성됨). 추론이 로드하는 model/* 전부 이 실행의 산출물.

## 8. Randomness/Determinism (로그 [RNG])
LGB: seed 3종 고정 + num_threads 8 고정 → **비트 결정적(실증: 8/8 파일 원 제출·Run A·Run B 모두 동일)**. CatBoost(GPU)·PyTorch(GPU): 시드 고정에도 커널 비결정성 존재(실증: 파일 상이). QuantileTransformer rs=0. train/val 분할 없음(전량 학습).

## 9. 클린룸 구성
- 기준(O): 원 제출 zip(sha256 605695923513f51a)을 별도 디렉터리에서 재실행해 기준 예측 생성(pred sha256 12a5cabb207f1c73) — **캐시 예측 파일을 비교 기준으로 쓰지 않음**.
- Run A/B: 새 디렉터리에 02 폴더 사본 + 공식 4파일 심볼릭링크만(시작 전 인벤토리 기록). 기존 모델/예측/캐시 없음.

## 10–11. 실행 (수동 개입 0회, 단일 명령)
`bash reproduce.sh` — Run A 19:0x→19:36 exit 0, Run B 19:37→20:05 exit 0 (로그: 각 디렉터리 repro.log; 단계 [1/6]~[6/6] 정상, Traceback 0).

## 12. 새 아티팩트
Run A zip sha256 f11755e9adb54f4b · Run B e24d7c3431f719c5 (각 37파일, 91.6MB). 원 제출과 비트 비교: model/ 35개 중 14개 동일(LGB 8 + meta 4 + ensemble.json + stats.pkl), script.py sha1 동일. A vs B: 16개 동일(위 + 대응표 산출 동일에 따른 일치).

## 13. 새 모델 추론 (각각 새 zip만 사용, 추론 클린룸 내용 기록: data/model/script/requirements뿐)
Run A 예측: 253,507행, NaN/Inf 0, mean .497868, sha256 613b492203f4dc03. Run B: mean .497767, sha256 c7e356ce41db0c97. (기준 O: mean .497891, sha256 12a5cabb207f1c73 — 세 해시 모두 상이 → 자기 자신 비교 아님)

## 14–15. 원 제출과의 수치 비교 (동일 입력: 모의 평가 253,507행; 실제 평가 데이터는 비공개 → UNKNOWN)
| 비교 | max | mean | median | RMSE | corr / rank | >1e-6 | >0.005 | 분위수 90/99% |
|---|---|---|---|---|---|---|---|---|
| Run A vs 원 제출 | 0.002243 | 0.000329 | 0.000278 | 0.000413 | 0.999948 / 0.999941 | 253,024 | **0** | 0.00068 / 0.00108 |
| Run B vs 원 제출 | 0.002278 | 0.000366 | 0.000307 | 0.000460 | 0.999940 / 0.999932 | 253,064 | **0** | 0.00076 / 0.00119 |
| 최대 차이 행 | 특정 투수(24619)에 집중 — NN/CatBoost 표현의 미세 이동, 구조적 편차 아님 | | | | | | | |

## 16. Run A vs Run B (독립 재학습 간 산포)
max 0.001536 · mean 0.000236 · >0.005 0행 · corr 0.999977 — **원 제출과의 차이(평균 3.3~3.7e-4)가 재학습 간 산포(2.4e-4)와 같은 규모** → 차이는 누락 정보가 아니라 GPU 비결정성으로 설명됨.

## 17. README 문서 정확성
클린룸은 README의 절차(공식 4파일 배치 → `bash reproduce.sh`)만으로 완주 — DOCUMENTED. 문서에 없는 수동 작업 0. README의 사전 검증 수치(평균 0.0004·최대 0.0025)와 이번 실측(0.00033~0.00037·0.0023)이 부합.

## 18–19. 발견 문제 / 수동 개입
코드 수정 0, 수동 개입 0, REPRODUCTION REQUIRED MODIFICATION 없음. 경미한 관찰: reproduce.sh의 [6/6] 예비본(v59b)까지 포함해 총 45분×2회.

## 20. Red team
- stale cache/기존 모델 재사용: 클린룸 인벤토리로 배제(시작 시 코드+링크만; 추론 디렉터리 내용 기록). ✔
- 자기 자신 비교: 기준 O를 재생성했고 세 예측의 sha256이 모두 다름을 확인. ✔
- 잘못된 모델 로드: 추론은 각 Run의 새 zip을 새 디렉터리에 풀어 실행(경로상 다른 모델 부재). ✔
- 절대경로/환경 의존/버전 불일치: [HID]/[ENV] 0건; venv은 스크립트가 생성. ✔
- 못 찾은 것 ≠ 없음 증명: 위 항목은 인벤토리·해시·실행으로 **증명**; 단 GPU 비트 재현 불가는 한계로 명시.

## 21. UNKNOWN
실제(비공개) 평가 데이터에 대한 예측 동일성 · CatBoost/NN의 GPU 비트 단위 재현(불가능 성질) · 주최측 서버(L4)에서의 재실행 결과.

---
### 최종 판정: **FUNCTIONAL REPRODUCTION** (구성 요소 다수는 EXACT: LightGBM 8/8·대응표·통계·전처리기·추론 스크립트 비트 동일; 예측 차이는 GPU 비결정성 산포 이내)

EVIDENCE LEVEL: **LEVEL 3 — EXECUTED** (클린룸 2회 재학습 + 재추론 + 수치 비교 + 해시)
핵심 근거: Run A/B 모두 단일 명령 완주; LGB 8/8·stats·대응표 비트 동일; A/B vs 원 제출 평균 |Δ| 3.3e-4/3.7e-4·0.005 초과 0행; A vs B 산포 2.4e-4로 동일 규모
max_abs_prediction_diff: 0.002243 (Run A) / 0.002278 (Run B)
mean_abs_prediction_diff: 0.000329 (Run A) / 0.000366 (Run B)
prediction_hash_original: 12a5cabb207f1c73 (원 제출 zip 재실행 산출)
prediction_hash_reproduced: 613b492203f4dc03 (Run A) / c7e356ce41db0c97 (Run B)
manual_intervention_required: 없음 (단일 명령, 코드 수정 0)
hidden_dependency_found: 없음 (uv·인터넷(패키지 설치)·GPU는 문서화된 요구사항)

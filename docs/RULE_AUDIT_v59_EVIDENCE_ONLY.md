# RULE_AUDIT_v59_EVIDENCE_ONLY — 검증 가능한 증거만으로 판정한 v59 규칙 감사

작성: 2026-08-29. 증거 로그: `experiments/rule_audit_v59_evidence.log`(태그 [ART][ENTRY][IO][NET/ENV][KW][PRE][CALLSITES][TRAINCODE][LBLOG][CLEAN][SINGLE][OOF][LINES][TM]), 하네스 로그: `/home/lhjin0j/.claude/jobs/00606e73/tmp/audit2/invariance_audit2b.log`. 이전 감사 문서·대화·기억은 증거로 쓰지 않았다. 아래 모든 줄 번호는 **실제 제출 zip에서 꺼낸 `script.py`(sha1 fcb09a5246, 1,021줄)** 기준이다.

증거 강도: L3 = 실행/재계산/해시/불변성 테스트로 직접 증명, L2 = 코드·아티팩트·공식 문서로 직접 확인, L1 = 로그/정황, L0 = 없음.

---

## 1. 공식 규칙 evidence table (08-29 직접 조회)
| RULE_ID | 공식 문구(원문) | SOURCE | STATUS | ALLOW | PROHIBIT | UNKNOWN |
|---|---|---|---|---|---|---|
| R-PRETRAIN | "공식적으로 누구에게나 가중치가 공개되었으며, 최소한 비상업적 이용이 허용된 라이선스(MIT, Apache 2.0 등) 하에 배포된 모델 및 가중치만 사용 가능합니다." | rules | EXPLICIT | 공개·비상업 허용 가중치 | 그 외 사전학습 가중치 | — |
| R-API | "원격 서버 기반의 API 형태로만 접근 가능한 모델(OpenAI API, Gemini API 등)은 사용이 불가합니다." "모든 작업은 로컬 환경에서 직접 코드로 실행 및 재현 가능해야 하며, 외부 서버에 의존하는 방식은 제한됩니다." | rules | EXPLICIT | 로컬 실행 | 원격 API·외부 서버 의존 | — |
| R-EXT | "온라인 해커톤(Phase 2)에서 제공하는 공식 데이터 외의 외부 데이터는 사용할 수 없습니다." | rules; FAQ 08-21 17:24 "외부 데이터는 사용 불가능합니다." | EXPLICIT | 공식 4개 파일 | 그 외 모든 데이터 | "외부 데이터"에 도메인 지식이 포함되는 범위 |
| R-INDEP | "평가 데이터(test.csv)의 각 행은 하나의 독립적인 예측 대상입니다." "각 행에 포함된 입력 변수와 주최 측이 제공한 공식 학습 데이터만을 이용" "평가 데이터의 다른 행이나 전체 평가 데이터의 분포를 이용해 특정 행의 예측값을 보정하거나 생성하는 방식은 정상적인 추론 절차로 인정되지 않습니다." | rules 4) | EXPLICIT | 행 입력 + 공식 학습 데이터 | 다른 행/전체 분포 이용 | — |
| R-INDEP-2 | 재안내: 행 A 예측은 ① 행 A 입력 ② 행 A 입력만으로 만든 파생 ③ 공식 학습 데이터 ④ 학습 데이터만으로 만든 통계·모델·파생 으로만; 금지: test 다른 행 누적 통계, rolling/lag, 전체 평균·분포·빈도·순위 보정, 선수·팀·월·경기 단위 집계, 시점상 과거 행 이용; 판정 예시: "test.csv에 해당 행 1개만 있는 경우"와 "전체가 함께 있는 경우"의 예측값이 같아야 함 | talkboard/417123 (08-13) | EXPLICIT | ①~④ | 열거된 5가지 | — |
| R-INDEP-FAQ | "test.csv 내 다른 행의 asof_* 값을 이용해 특정 행의 정답을 복원·추정하여 예측에 활용하는 것은 규칙 위반입니다." / 과거 test 행 이용 "평가 데이터의 다른 행을 이용하는 것이므로 불가능합니다." / "추론 시 평가 데이터셋의 다른 행의 정보를 사용하지 않는 것에 반드시 유의" | FAQ 08-19 09:11, 09:08 | EXPLICIT | — | test 행 간 정보 사용 | — |
| R-TRAIN | "학습 데이터 내에서는 제약사항 없습니다." / "학습 데이터에서 도출된 값에는 제약 사항 없습니다." / privileged 정보 aux head·증류 "질문1~3 모두 가능합니다." / Trackman 측정값 학습 활용 "제공된 학습 데이터 범위 내에서는 활용에 별도 제약이 없습니다." | FAQ 08-10 08:49, 08-19 09:09, 08-20 21:07, 08-14 12:18 | EXPLICIT(학습 한정) | train 내부 미래 행·도출값 | — | test 단계로의 확장은 R-INDEP로 금지 |
| R-ASOF | "해당 방법(asof 누적 − 학습기간 누적 상수표)은 학습 데이터에서 추출한 정보를 바탕으로 평가 데이터셋의 추론 행에 독립적으로 적용하는 것이므로 가능합니다." | FAQ 08-13 16:40 | EXPLICIT | 학습 상수표 + 행 자신의 asof | — | — |
| R-TM | "1) 트랙맨 데이터는 학습데이터 기간에 대해서만 제공되므로 가능합니다. 2) 문제없습니다."(투수 ID 대응·이전 시즌 요약) / 타자 "네 가능한 방법입니다." | FAQ 08-07 09:22, 08-12 09:51 | EXPLICIT | train↔trackman ID 대응 추정, 이전 시즌 Trackman 요약 피처 | — | — |
| R-LB | "리더보드 점수를 참고하여 모델, 하이퍼파라미터 또는 앙상블 가중치 등을 선택·조정하는 것은 가능합니다. 따라서 문의주신 1, 2번 방식 모두 허용됩니다." | FAQ 08-12 12:48 | EXPLICIT | 후보 제출 비교·점수차 보간 | — | — |
| R-LB-2 | "모델은 동일하게 유지한 채 모든 예측값에 적용되는 보정 상수만을 변경하여 반복 제출하고, 리더보드 점수가 가장 높은 값을 선택하는 방식은 권장하지 않습니다." (…"반복 횟수와 탐색 방식에 따라 … 리더보드 프로빙으로 판단될 수 있습니다" "소명을 요청할 수 있으며 … 규칙 위반 여부를 별도로 검토") | FAQ 08-19 12:11 | INTERPRETIVE | — | — | 허용/위반의 경계(횟수·방식) |
| R-LB-3 | "말씀해주신 정도의 후보값 비교·보간은 허용되는 범위로 판단합니다." | FAQ 08-19 12:16 | INTERPRETIVE | 학습 추세 기반 상수의 후보 비교·보간 | — | "정도"의 수치 기준 |
| R-LB-4 | "'평가 데이터의 정답 분포나 특성을 역으로 추정하려는 수준의 탐색'은 과도한 리더보드 프로빙으로 판단될 수 있습니다." | FAQ 08-25 11:35 | INTERPRETIVE | — | — | 수준의 기준 |
| R-FINAL | "'최종 제출물'을 선택하는 절차가 없으며, 리더보드 제출 결과 중 최고점 제출물이 순위 산정의 기준" | FAQ 08-14 15:07 | EXPLICIT | — | — | — |
| R-REPRO | "학습 및 추론 재현 코드를 주최 측 서버 환경에서 실행하여 생성된 결과가 제출 결과와 동일하거나, 모델 및 실행 환경에 따른 일반적인 오차 범위 내에서 재현되는 경우 통과" | FAQ 08-28 09:41 | INTERPRETIVE | — | — | "일반적인 오차 범위"의 수치 |
| R-INFO | "참가자는 투구 이전 시점에서 활용 가능한 정보만을 바탕으로 예측 모델을 설계할 수 있어야합니다." / "전체 추론 실행 시간 ≤ 10분 (245,789개 샘플 추론)" "패키지 설치 시간 ≤ 10분" "제출 파일 용량 ≤ 10GB" "오프라인 환경 실행" | description | EXPLICIT | — | 사후 정보 | — |
공식 자료에 없는 사항(예: 동점 처리 세부, 소명 절차의 형식)은 이 문서에서 추정하지 않는다.

## 2. v59 실제 pipeline (아티팩트에서 역구성; [ART][ENTRY][IO])
- 아티팩트: `submit_v59_lg30_105_fix2.zip` sha256 605695923513f51a…, md5 084398f280, 91,632,331 B, 37파일, 숨김 항목 0. `requirements.txt` = `lightgbm==4.7.0 catboost==1.2.10`. (L2)
- 진입점: `script.py` L1020–1021 `if __name__ == '__main__': main()`; `main()` L909–1019. 정의 함수 25개 중 main에서 도달 17개; **도달 불가 8개**: `_recover_indicators, batter_bayes_features, build_stats, build_stats_v1, build_tm_table, deviation_view, league_table, season_means`. (L2)
- 읽기: L911 `./data/test.csv`(dtype: base_state/top_bottom/game_type/row_id 문자열), L912 `./data/sample_submission.csv`, L914 `./model/stats.pkl`, L915 `./model/ensemble.json`, L920 `*_meta.json`, L930 `*_prep.pkl`, L937 `.pt`, L954 `.cbm`, L967 `.txt`. 쓰기: L1016 `./output/submission.csv`. 네트워크·subprocess·getcwd·expanduser 참조 0; `os.environ` 참조는 기본값이 있는 플래그뿐(L258–291, L656–664). (L2)
- 흐름: L925 `F = add_change_features(build_features(test, stats))` → 멤버별: NN(L926–941: prep 변환 → `bayes_init` → `nn_predict`), CatBoost(L949–963: `predict_proba` 클래스0 logit 평균), LightGBM(L966–979: `softmax(raw+init)` 클래스0 logit 평균 + L978 `logit(L_ref)`) → L982–984 가중 logit 블렌드 → L987 `prob_shift` → L990–997 조건부 상수(팀 13 관여 R행, 3-0) → L1010 clip → L1011–1016 row_id 매핑 후 저장. (L2)
- 설정값(`ensemble.json`, [PRE]): blend logit, prob_shift −0.004, lg_shift 0.015 (=lg_shift_p=lg_shift_opp), lg30_shift 0.105, f_shift 0, lg_shift_home/away 0, cold_slope 0, sharpen []. 멤버 가중 .25/.35/.10/.30, 오프셋 LightGBM 'bayes', NN-ID 'bayes_ctx', NN-wide3 'bayes', CatBoost 없음. (L2)

## 3. Data lineage (실제 코드 기준; 줄 번호 = 아티팩트 script.py)
| NAME | SOURCE | CODE | ROWS USED | FIT SCOPE | TEST DEP | TEMPORAL | OFFICIAL | VERDICT | EVIDENCE |
|---|---|---|---|---|---|---|---|---|---|
| 원본 상황 31개(season…batter_id) | test 행 공식 컬럼 | L180–209 | 행 자신 | 없음 | 없음(§4) | 투구 직전 컬럼 | R-INDEP ① | SAFE | L2+L3 |
| 원본 asof 18개 | test 행 `asof_*` | L210–220 | 행 자신 | 없음 | 없음 | 공식 사전 계산 | R-ASOF | SAFE | L2+L3 |
| 시즌 분해 p_prior_*/p_cur_*/b_* | 행 asof × n − `stats.ps/bs`(시즌<y 합) | L221–241, L315–329; `_prior_table` L120–143 호출 L167–168(operand = stats 표) | 행 + train 표 | train(ps n합 1,475,092 = train 행수) | 없음 | 행 시즌 미만 시즌 합(cumsum.shift(1)) | R-ASOF, R-INDEP ④ | SAFE | L2+L3 |
| CTX 4개 | ps 이전 시즌 플래툰/2K 합 + 행 hand/strikes | L244–257 | 행 + train | train | 없음 | 이전 시즌 | R-INDEP ④ | SAFE | L2+L3 |
| 직전 시즌·구종별·축소 추정 | ps/bs + 행 | L296–339 | 행 + train | train | 없음 | 이전 시즌 | R-INDEP ④ | SAFE | L2+L3 |
| 매치업 4개 | `stats.pbs`(투수×타자×시즌<y) + 행 ID | L374–397; L380 `np.unique(배치 키)`는 학습 표 필터, L382–386 빈 경우 n=sv=0으로 일반 식 통과 | 행 + train | train | **값 없음**(§4 E/E2/G, 미학습 조합 행 비트 동일; 행별 조회 동치 0.0) | 이전 시즌 | R-INDEP ④ | SAFE | L3 |
| LG_game / LG_regime | 행 팀 ID·시즌·월·경기유형 | L657–661 | 행 자신 | 없음 | 없음 | — | R-INDEP ② | SAFE | L2+L3 |
| ROLE 8개 | `stats.role` 이전 시즌 가중 평균 + 행 월·이닝 | L673–686; `prior_weighted` L441–457 호출 L676 | 행 + train | train | 없음 | 이전 시즌 | R-INDEP ④ | SAFE | L2+L3 |
| TM 43개 | `stats.tm`(trackman 2019~24) 이전 시즌 가중 평균 | L687–699 호출 L691 | 행 pitcher_id + train/trackman | train | 없음 | ≤2024 | R-TM | SAFE(§8) | L2+L3 |
| 변화·폼 12개 | 위 피처 행 내 산술 | L346–371 | 행 자신 | 없음 | 없음 | — | R-INDEP ② | SAFE | L2+L3 |
| L_ref | `stats.L_gt[2024, game_type]` | L550–572, L978 | train 상수 | train | 없음 | 학습 시즌 | R-INDEP ④ | SAFE | L2 |
| NN 변환(분위수·결측 지시자·범주 코드·season_rel) | prep.pkl(train fit) | L864–877, L931–933 | 행 | train(quantiles_ (200,117); 코드표 ⊆ train ID) | 없음 | — | R-INDEP ④ | SAFE | L2+L3 |
| 베이즈 오프셋 | `stats.ps` 시즌 재귀 + `league` + 행 p_cur_n/p_cur_s | L721–756, L798–822, L896–906 호출 L807(operand = stats ps) | 행 + train | train | 없음 | 이전 시즌 + 행 자신 | R-INDEP ④ | SAFE | L2+L3 |
| 모델 4종 | train 2019~2024(train_final.py L26, nn_export.py L28) | L926–979 | — | train | 없음(트리 0, NN ≤6e-8 배치 커널) | — | R-INDEP ④ | SAFE | L2+L3 |
| 후처리 상수 | ensemble.json | L987–1010 | 행 조건 | §9 | 없음 | — | §9 | §9 | L2 |
| 제출 | row_id 매핑, 결측 0.4861(train) | L1011–1016 | — | train | 없음 | — | — | SAFE | L2 |
`stats.pkl`에서 추론이 실제로 읽는 표: ps, bs, pbs, league, L_gt, L_all, ps2, bs2, role, tm(코드 L164–168, L579–580, L676, L691, L807). FM_gt/team_p/team_b/endform/src_eff는 코드 경로에서 참조되지 않음(L664는 TEAM_LAST 플래그 OFF).

## 4. Test-row independence — 실행 결과 (L3)
**(a) 파일 단위, 실제 script.py, 새 디렉터리 `audit2/clean`, `env -i`, 소켓 차단** ([CLEAN][SINGLE]):
- 전체(모의 253,507행) 출력 = 기준 v59 예측: row_count 253,507 | max_abs_diff 0.0 | mean_abs_diff 0.0 | mismatch 0 | sha256(output) = sha256(reference) = 12a5cabb207f1c73.
- 공식 배포 5행 샘플: 0.4094138791, 0.3756602966, 0.4467664781, 0.4863868374, 0.5122441886.
- 단독 1행 파일 4개(새 무작위): |Δ| 1.12e-9 / 2.23e-9 / 2.31e-9 / 2.06e-10 (최대 2.31e-9).
**(b) 중간값 단위 하네스(아티팩트 script.py를 모듈로 import, 새 무작위 9행, 케이스 A 전체 / B 단독 / C1·C2 무작위 500행 / D 전체 순열 / E 같은 투수 제거 / E2 같은 타자 제거 / F 같은 팀·경기·월 제거 / G 다른 행 값 교란 / C1 반복)**:
- 하네스 전체 실행 = 실제 스크립트 출력: max|diff| 5.55e-17 (하네스가 실제 경로와 동일함을 확인).
| 케이스 (9행) | max_abs_prediction_diff | mean_abs_prediction_diff | mismatch(>1e-6) | 178 피처 / NN 입력 / 오프셋 / 트리 멤버 max_abs_feature_diff |
|---|---|---|---|---|
| B 단독 1행 | 1.24e-8 | 6.23e-9 | 0 | 0 / 0 / 0 / 0 (NN 멤버만 ≤7.9e-8) |
| C1 +무작위 500행 | 0.0 | 0.0 | 0 | 0 / 0 / 0 / 0 |
| C2 +다른 무작위 500행 | 0.0 | 0.0 | 0 | 0 / 0 / 0 / 0 |
| D 전체 순열 | 0.0 | 0.0 | 0 | 0 / 0 / 0 / 0 |
| E 같은 투수 행 제거 | 0.0 | 0.0 | 0 | 0 / 0 / 0 / 0 |
| E2 같은 타자 행 제거 | 0.0 | 0.0 | 0 | 0 / 0 / 0 / 0 |
| F 같은 팀·경기·월 행 제거 | 0.0 | 0.0 | 0 | 0 / 0 / 0 / 0 |
| G 다른 행 값 교란(ID·카운트·asof 뒤섞기) | 0.0 | 0.0 | 0 | 0 / 0 / 0 / 0 |
| C1 반복(결정성) | 0.0 | 0.0 | 0 | 0 / 0 / 0 / 0 |
| 전체 253,507행 순열 vs 원본 | 1.08e-8 | 1.25e-12 | 0 | 178 피처 0.0 |
- 부동소수점 분리 실험: 같은 배치 반복(C1_repeat) 0.0, 같은 크기·다른 내용(C1 vs C2) 0.0 → 차이는 **배치 크기가 다른 B(단독)에서 NN 멤버에만** 발생(≤7.9e-8; 다른 행의 내용이 아니라 torch float32 행렬곱의 배치 크기별 커널 차이). 트리 멤버·피처·오프셋은 모든 케이스에서 정확히 0.
- 원인 추적이 필요한 "다른 행에 의한 변화": 없음(0건).

## 5. Test-derived information 탐색 ([KW])
키워드(groupby/rolling/expanding/shift/rank/quantile/mean/median/std/count/value_counts/normalize/transform/ffill/bfill/sort/cumsum/cumcount/percentile/argsort) 전수 검색 결과 70여 건 중 **도달 가능한 함수 내 발생은 L129·135·136·137·141(`_prior_table`), L447·452·453·455(`prior_weighted`), L667(`stats['league']`), L724(`ps.pitcher_id.unique`), L867(`qt.transform`)** 뿐이며, 호출 지점 [CALLSITES]에서 첫 인자는 모두 `stats` 표(ps, bs, pbs 부분집합 `sub`, ps2, bs2, role, tm)이다. test 프레임(`test`, `d`, `F`)을 대상으로 한 집계는 없다. 나머지 발생은 dead code(`build_stats*`, `build_tm_table`, `_recover_indicators`, `league_table`, `season_means`). 직렬화 객체 검사: `stats.pkl` 16표 모두 시즌 2019~2024만, `prep.pkl`은 train fit 변환기·코드표만([PRE]). (L2; 실행 검증은 §4 L3)

## 6. Precomputed artifact 감사 ([PRE])
| 파일 | 생성 주체 | 생성 데이터 | test 정보 가능성 | 외부 데이터 | 아티팩트 포함 |
|---|---|---|---|---|---|
| model/stats.pkl | `src/save_stats.py` L5 `build_stats(load_train())` (동일 함수가 아티팩트 script.py L478–547에 dead code로 존재) | train.csv(+trackman_history via `build_tm_table` L402–438, `pitcher_map.parquet`) | 없음: 16표 시즌 ≤2024, ps.n 합 = train 행수 1,475,092, s 합 772,603 | 없음 | 포함 |
| model/ensemble.json | `src/set_ensemble.py` | 사람이 지정한 상수(§9) | 없음(상수) | 없음 | 포함 |
| model/*_meta.json ×4 | train_final.py / nn_export.py | 피처 목록·가중·오프셋 유형·fshare | 없음 | 없음 | 포함 |
| model/*_prep.pkl ×2 | nn_export.py L36–42 (QuantileTransformer fit, 코드표) | train 행 | 없음: 코드표 ⊆ train ID(792/830), quantiles_(200,117) | 없음 | 포함 |
| model/lgb_*.txt ×8 | train_final.py(LightGBM) | train | 텍스트에 트리·파라미터·feature_names만 | 없음 | 포함 |
| model/cat_*.cbm ×8 | train_final.py(CatBoost GPU) | train | 모델 파일 | 없음 | 포함 |
| model/nn_*.pt ×11 | nn_export.py | train | 34 텐서 = 층 가중치/BN 통계 | 없음 | 포함 |
| data/derived/*.parquet (학습 측) | `src/trackman_match.py` L105–107 | train.csv + trackman_history.csv | 없음 | 없음 | 미포함(추론 불필요) |
provenance를 확인하지 못한 파일: 없음.

## 7. TRAIN / TEST 경계
- **TRAIN RULE**(R-TRAIN, R-TM): 학습 코드는 `load_train()`·`load_trackman()`만 읽는다([TRAINCODE] src/train_final.py L26, nn_export.py L28, save_stats.py L5, trackman_match.py L6; common.py L16/L24 = train.csv/trackman_history.csv). 4클래스·보조 라벨은 train 내부의 다음 행 asof 차분(아티팩트 L38–70 `_recover_indicators`, 학습 스크립트에서 호출)으로 만든다 → FAQ 08-10 "학습 데이터 내에서는 제약사항 없습니다", 08-20 aux head 허용에 해당. 통계·변환기·모델 fit 범위 = train만(§6).
- **TEST RULE**(R-INDEP, R-INDEP-2, R-INDEP-FAQ): 추론은 행 + 학습 산출물만(§3), 실행 검증(§4). 학습 단계의 "미래 행 참조"가 test 행 간에 적용되는 코드는 없다(`_recover_indicators`는 추론 경로에서 도달 불가; test에 정답 컬럼도 없음).

## 8. TrackMan 감사 ([TM])
- 사용 여부: **사용함**. 트리 멤버 입력 178개 중 `TM_*` 43개(목록은 로그 [TM]); NN 입력에는 없음(prep num_feats 117에 TM_ 없음).
- 대응표: `src/trackman_match.py` — train 경기 지문 ↔ Trackman 경기 유사도(L48), 유사도 ≥0.85 경기에서 반이닝 공동 등장 투표(L63, L69), 투수별 최다 득표 + purity(L80); 타자 대응표도 생성(L91)되나 **추론 피처에서는 투수 대응표만 사용**(`build_tm_table` L402–438은 `pitcher_map.parquet`만 읽음). 대응 생성 데이터 = train.csv + trackman_history.csv(L6). test 관여: 없음(파일 입력 자체가 없음).
- 집계 기간: `stats.tm` 시즌 [2019..2024]([PRE]); 추론에서 `prior_weighted(tm, 'pitcher_id', 'tm_n', mets)`(L691)로 행 시즌 미만 시즌의 가중 평균·마지막 활동 시즌 값. 2025 포함: 없음(표에 2025 없음; 설명서·FAQ상 2025 Trackman 미제공).
- 현재 test 행의 Trackman 정보: 없음(test.csv에 Trackman 컬럼 없음; 코드도 참조 없음). 다른 test 행 관여: 없음(§4).
- FAQ 허용 범위(R-TM: "대응 관계 추정" + "투구 시점 이전 시즌까지의 트랙맨 통계치를 투수 단위 요약 피처로")와 구현(투수 대응표 + 이전 시즌 투수 단위 요약)이 일치한다. 판정: 43개 피처 모두 SAFE (L2 코드 + L3 §4).

## 9. Post-processing / shift 감사 (코드 L982–1010, `ensemble.json`)
| 항목 | VALUE | CODE | DERIVED FROM (증거) | TEST DATA USED | LB FEEDBACK USED | OFFICIAL BASIS | VERDICT / 강도 |
|---|---|---|---|---|---|---|---|
| 블렌드 가중 | .25/.35/.10/.30, logit | L982–984, meta weight | fold-2024 OOF 블렌드(재계산: oracle 1022.9, [OOF]); LOG L440 "가중 변형 평탄" | NO | LOG L451: v37 가중 변형 LB 1195 = v33 → 중립, 채택 안 함 | R-LB 허용 | SAFE / L3 |
| 시드 평균 | 트리 logit 평균·NN 확률 평균 | L939, L957–961, L969–976 | 설계 | NO | NO | — | SAFE / L2 |
| `+logit(L_ref)` (LightGBM) | R: logit(0.48971), F: logit(0.45928) | L978; L_gt from stats | train 상수 | NO | NO | R-INDEP ④ | SAFE / L2 |
| `prob_shift` | −0.004 | L987 | LOG L69/L122: v2→v3(0→−0.007: 1062→1071), v10→v10s(−0.005→−0.008: 1108→1102)로 "최적 −0.004"; LOG L212 폴드 레벨 점검(−0.003~−0.004) | NO | **YES**(2회) | R-LB 허용 / R-LB-2 비권장 | AMBIGUOUS(낮음) / L1 |
| `lg_shift` (p=opp) | +0.015, 팀 13 관여 R행 | L990–993 | LOG L446/L450/L451: v36 −0.005→1184, v40 +0.010→1207, v41 +0.015→1207.8, "역산 e≈+0.013" | NO | **YES**(3회, 상수-only) | R-LB(허용), R-LB-3(허용 범위), R-LB-2/4(정도에 따라 검토) | **AMBIGUOUS** / L1(LB 값은 사용자 보고·LOG) |
| `lg30_shift` | +0.105, 위 행 중 3볼 0스트라이크 | L996–997 | fold-2024 OOF 재계산 [OOF]: LG-R 3-0 잔차 +0.1047(590행), 규칙 +0.105 → +10.2; LOG L461 | NO | NO(v48 제출 전 결정; 0.09→0.105는 학습 잔차값) | R-INDEP ④(학습 통계) | SAFE / L3 |
| f_shift, home/away, cold_slope, sharpen | 0/0/0/[] | L992–1009 | 비활성 | NO | NO | — | SAFE / L2 |
| clip [0.001, 0.999] | 고정 | L1010 | — | NO | NO | — | SAFE / L2 |
| 결측 채움 | 0.4861 | L1013–1014 | `stats.L_all[2024]` | NO | NO | — | SAFE / L2 |
"왜 이 group인가"의 근거(재계산, `experiments/fact_audit_v59.log` [D3][D4][D5][D6]): F 행 100% 팀 13 관여; LG 관여 R 편차 2023 +.041/2024 +.039(R 평균 대비), 2023년 4월 −.022 → 5월 +.047; LG-R 3-0 성공률 .598 vs LG-R .543, 비LG 3-0 .468 — 모두 train.csv에서 계산된 값이다(L3). 
출처 불명 magic constant: 없음(모든 값이 위 표로 추적됨).

## 10. Leaderboard feedback 감사 (근거: `experiments/LOG.md` 기록 L54–L122, L311, L438–L451, L517, L541; LB 점수 자체는 사용자 보고값이며 공개 LB에는 최고점 1225.20622만 표시됨)
| 결정 | LB 사용 | 근거 줄 | 공식 근거 | 판정 |
|---|---|---|---|---|
| 모델/멤버 선택(v15~v20d, v33) | YES | L311, L438 | R-LB 허용 | SAFE(L1) |
| 피처 선택(LG 체제 피처 v33) | 결과 확인 | L438 | R-LB 허용 | SAFE(L1) |
| 하이퍼파라미터 | 폴드 기반 | — | — | SAFE(L1) |
| group 선택(팀 13, 3-0) | NO(학습 통계) | fact_audit [D3]–[D6], [OOF] | R-INDEP ④ | SAFE(L3) |
| shift 크기 prob_shift | YES(2회) | L69, L122 | R-LB / R-LB-2 | AMBIGUOUS(낮음) |
| shift 크기 lg_shift | YES(3회 상수-only) | L446–451 | R-LB-2/3/4 | AMBIGUOUS |
| ensemble weight | 중립 확인만 | L451 | R-LB | SAFE(L1) |
| calibration | 없음 | — | — | — |
v59에서 새로 LB로 정한 값: 없음(v48→v59 변경은 lg30 0.105 = 학습 잔차값, 코드 강건성 수정).

## 11. External data 감사
prediction에 영향을 주는 raw source: `train.csv`(OFFICIAL), `trackman_history.csv`(OFFICIAL), `test.csv`·`sample_submission.csv`(OFFICIAL, 평가 서버), 이상 4개. 코드에 다른 파일 읽기 없음([IO]); 네트워크 참조 없음([NET/ENV]); 소켓을 차단한 실행에서 출력 동일([CLEAN] sha256 일치). 외부 통계·선수 정보·웹 데이터·API 응답이 입력에 들어가는 경로: 코드상 없음(L2) + 실행(L3). 외부 지식이 "설계 아이디어"에 쓰였는지는 코드로 검증 불가(UNKNOWN)이나, 모델 입력·상수의 도출은 §9·§3의 train 계산으로 추적된다.

## 12. Clean-room 재현 ([CLEAN])
새 디렉터리 `audit2/clean` = zip 내용물 + `data/test.csv`(모의 253,507행) + `data/sample_submission.csv` + 소켓 차단 sitecustomize만. `env -i HOME=/nonexistent PATH=/usr/bin:/bin`로 실행(개발 환경변수 없음), 인터프리터 = 서버 기본 패키지 버전의 clean venv(Python 3.11.15, torch 2.7.1+cpu, numpy 1.26.4, pandas 2.0.3, scipy 1.15.3, joblib 1.5.3, lightgbm 4.7.0, catboost 1.2.10; pyarrow·GPU 없음). 결과: 32.2초, row_count 253,507, max_abs_diff 0.0, mean_abs_diff 0.0, mismatch 0, **sha256 12a5cabb207f1c73 = 기준 파일 sha256**. 캐시·이전 예측·실험 산출물·숨김 파일·환경변수 의존: 없음(디렉터리 내용 [CLEAN] "data model nonet requirements.txt script.py").
학습 측 재현(별도 실행, `lgaimers_phase2_repro2`, 08-29 14:18~15:00): `fact_audit_v59.log` [R1][R2][P3] — 대응표 3종 동일, LightGBM 8/8 비트 동일, 재현 zip 예측 vs 제출 v59 평균 |Δ| 0.000402·최대 0.00248·상관 0.999924·0.005 초과 0행(GPU 학습 비결정성). R-REPRO "일반적인 오차 범위"의 수치 기준은 UNKNOWN이므로 통과 여부는 운영진 판단.

## 13. Artifact ↔ 문서 ↔ 과거 설명 불일치 (실제 artifact > 실행 결과 > 코드 > 문서 > 과거 설명)
| 항목 | 아티팩트/실행 | 문서/과거 설명 | 조치 |
|---|---|---|---|
| 실제 평가 데이터 행 수 | 설명 페이지 245,789 | 문서·보고 일부 "253,507행"(모의 파일 값) | REPRO_SPEC 정정 완료; 본 문서는 모의 파일임을 명시 |
| "모델 파일 35개 v48과 비트 동일" | 34개 동일 + ensemble.json(lg30 변경) | 과거 설명 "35개" | 본 문서에서 정정 |
| 현재 순위 | 공개 LB 2위(1225.20622), 1위 1295.68 | 과거 보고 "1위 추월" | 정정 |
| ensemble.json 실험용 키(cold_slope 0, cold_center, sharpen []) | 아티팩트에 존재(비활성) | 클린 재현본에는 없음 | 동작 동일(`.get` 기본값); 문서 기재 |
| 그 외 lineage·상수·경로 | 아티팩트 = 코드 = 문서(§3, §9) | 일치 | — |

## 14. Red team — 반증 시도 결과
| 가설 | 시도 | 결과 | 구분 |
|---|---|---|---|
| hidden test dependency | §4 A~G + 파일 단위 단독/전체 + 행별 조회 동치 | 피처·오프셋·트리 0, NN ≤6e-8(배치 크기 커널), 최종 ≤1e-8 | 없음을 증명(L3) |
| accidental transductive feature | [KW] 전수 + [CALLSITES] operand 확인 | test 프레임 집계 0 | 없음을 증명(L2) |
| test-wide statistics / calibration | L982–1016 검토 + §4 D/G | 상수만; 순열·교란 불변 | 없음을 증명(L3) |
| temporal leakage | `_prior_table` cumsum.shift(1), `prior_weighted`, `season_priors` 시즌<y; stats 시즌 ≤2024 | 미래 정보 경로 없음 | 없음을 증명(L2) |
| precomputed test artifact | §6 | 모든 표 시즌 ≤2024, ps 합 = train | 없음을 증명(L2) |
| external data contamination | §11 | 입력 4개 파일; 소켓 차단 동일 | 없음을 증명(L2/L3) |
| LB-derived correction | §9 | lg_shift/prob_shift 크기 LB 비교로 채택 | **찾음 → AMBIGUOUS**(R-LB-2/3/4) |
| train/test fit contamination | §6 prep/stats 검사 | train만 | 없음을 증명(L2) |
| undocumented manual file | [ART] 목록 37파일 전수, 숨김 0 | 없음 | 없음을 증명(L2) |
| stale artifact / wrong version | zip sha256·script sha1 = 재현본 script sha1 fcb09a5246; 클린 재현 zip과 script 동일 | 일치 | 없음을 증명(L3) |
| cached prediction reuse | 클린룸 새 디렉터리 실행 sha256 일치 | 재생성됨 | 없음을 증명(L3) |
| regeneration failure | 클린 학습 재현 평균 |Δ| 0.0004 | 재생성 성공(GPU 오차) | 증명(L3), 통과 기준은 UNKNOWN |
| 단독 1행 실행 실패 | §4(a) 4행 + 이전 5 신생 조합·만루 행(build_v59.log) | 모두 실행·일치 | 없음을 증명(L3) |

## 15–16. 판정
- **CRITICAL**: 없음.
- **HIGH RISK**: 없음.
- **AMBIGUOUS**: (1) `lg_shift +0.015` — 같은 모델에 상수만 다른 제출 3회의 LB 비교로 채택. 공식: R-LB(허용), R-LB-3(허용 범위), R-LB-2·R-LB-4(정도에 따라 검토·소명). 증거 강도 L1(LOG 기록·사용자 보고 LB). (2) `prob_shift −0.004` — 2회 비교, 폴드 근거 있음, 같은 성격(낮음). (3) 재현 통과 기준 "일반적인 오차 범위"의 수치 — UNKNOWN에 가까움.
- **UNKNOWN**: 과거 개별 제출의 LB 점수(사용자 보고 외 검증 수단 없음); 외부 지식이 설계 아이디어에 쓰였는지(코드로 검증 불가); 실제 2025 평가 파일의 구성(비공개).
- **SAFE(L2 이상)**: 행 독립성(L3), lineage 전 계열(L2/L3), TrackMan(L2/L3), TRAIN/TEST 경계(L2), 후처리 중 lg30·블렌드·클리핑·L_ref(L2/L3), 외부 데이터·API 없음(L2/L3), 사전 계산 아티팩트(L2), 클린룸 추론 재현(L3), 아티팩트 버전 일치(L3).

### 최종 판정 근거
핵심 test-row independence는 실제 아티팩트에 대해 파일 단위·중간값 단위(A~G)·클린룸에서 L3로 증명되었고, 클린룸 추론은 sha256까지 동일하게 재현되었다. CRITICAL·HIGH RISK는 없다. 남은 AMBIGUOUS는 v59 코드가 아니라 이미 리더보드에 존재하는 제출 이력(v36/v40/v41)에서 상수 크기를 정한 방식에 대한 운영진의 "정도" 판단이며, 운영진은 같은 유형을 "허용되는 범위"(R-LB-3)로 답한 기록이 있다. 이 항목은 참가자가 코드로 해소할 수 없고 증거 강도가 L1이므로, 지시된 기준("해결되지 않은 핵심 AMBIGUOUS가 존재하면 SAFE TO SUBMIT으로 판정하지 마라")에 따라 SAFE TO SUBMIT을 쓰지 않는다. 제출 금지 사유(CRITICAL/HIGH RISK)도 없다.

OPERATOR CONFIRMATION REQUIRED

---

## 부록 — 최종 결정 (2026-08-29, 참가자 판단 반영)
- AMBIGUOUS로 남겼던 `lg_shift +0.015` / `prob_shift −0.004`의 결정 방식(리더보드 후보 비교)에 대해 참가자가 **허용으로 판단**함. 공식 근거: FAQ 08-12 12:48("후보값 제출 비교·점수차 보간 모두 허용"), FAQ 08-19 12:16("학습 데이터의 시즌별 추세를 바탕으로 보정 필요성을 판단한 뒤 … 말씀해주신 정도의 후보값 비교·보간은 허용되는 범위"). 우리 상수는 학습 라벨 통계로 방향·범위를 정한 뒤 후보 3회(lg_shift)·2회(prob_shift)를 비교한 것으로, 위 답변이 기술한 유형에 해당한다.
- 유지하는 조건: FAQ 08-19 12:11의 단서("권장하지 않음… 소명 요청 가능")에 대비해 `docs/CONSTANTS_RATIONALE.md`를 2차 평가 제출물에 첨부하고, 상수만 바꾼 추가 제출은 하지 않는다.
- 이 결정으로 §15–16의 미해결 핵심 AMBIGUOUS가 해소되며(CRITICAL·HIGH RISK 0, 행 독립성·클린룸 재현 L3), 최종 판정을 아래와 같이 갱신한다.

SAFE TO SUBMIT

## 부록 2 — 운영진 공식 확인 (2026-08-31, talkboard 417082)
- 참가자(이형진) 08-29 질문 "학습 데이터의 라벨 통계에서 확인한 패턴을 바탕으로, 평가할 때 각 행의 팀 ID만으로 계산한 변수를 사용해도 되는지" → **DACON.GM: "문의주신 내용 모두 가능합니다."** (LG_game/LG_regime 명시 허용)
- 타 참가자 08-28 질문 "calibration 계수와 같은 하이퍼파라미터를 이전 리더보드 제출 결과를 참고해 변경하는 것이 허용되는지" → **DACON.GM: "가능합니다."** (§15–16의 AMBIGUOUS(lg_shift·prob_shift 결정 방식)에 대한 공식 근거 보강)
이로써 본 감사의 AMBIGUOUS 항목은 모두 운영진 답변으로 해소되었다.

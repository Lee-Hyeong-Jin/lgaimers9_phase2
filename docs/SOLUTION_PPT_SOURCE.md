# SOLUTION_PPT_SOURCE — 새 솔루션 PPT 내부 검증용 소스 문서 (2026-09-02)

원칙: 기존 PPT는 source가 아님. 아래 모든 사실(FACT)은 실제 코드·아티팩트·실행 로그에서 확인된 것만 사용.
증거 약어: EV1=`experiments/fact_audit_v59.log`(태그 [D*][O*][P*][A*][R*][I*]), EV2=`experiments/rule_audit_v59_evidence.log`([ART][ENTRY][LINES][PRE][CALLSITES][CLEAN][SINGLE][OOF][INV2]), EV3=`experiments/reproduction_audit.log`, EV4=제출 아티팩트 `submit_v59_lg30_105_fix2.zip`의 script.py/meta/prep/stats(직접 열람), EV5=`experiments/results.jsonl`, EV6=`docs/FACT_AUDIT_v59.md`(공개 LB 조회 기록), EV7=학습 코드 `final_submission/02_학습재현코드/src/*`.

## FACT TABLE (PPT에 쓰는 모든 기술 주장; STATUS=VERIFIED만 사용)
| # | CLAIM | STATUS | EVIDENCE | CODE | 최종 모델 사용 |
|---|---|---|---|---|---|
| F1 | train 1,475,092행(2019~2024)·시즌 성공률 .56467→.48610 하락, trackman 1,793,078행, 실제 평가 245,789행(비공개) | VERIFIED | EV1 [D1][D2][D10], 설명 페이지 | — | 배경 |
| F2 | asof_*는 2019년부터의 통산 누적(시즌 리셋 없음): 같은 투수 연속 행에서 asof_n이 정확히 1씩 증가 100% | VERIFIED | EV1 사전 검증(파일순 step==1 1.0; 08-29 재계산 로그) | — | 핵심 전제 |
| F3 | 시즌 분해: p_prior_n=Σ(시즌<y)n, p_cur_n=asof_n−prior_n, cur_k=rate_k·asof_n−prior_k (무작위 train 행 6개 실측 일치, 2019 행 prior=0) | VERIFIED | 09-02 스팟체크 출력(대화 로그) + EV4 L221–241 | script.py L221–241 | 예 |
| F4 | 라벨 복원: 연속 행 cum 차분→4클래스, y4=[772,603/337,950/170,320/194,219], fshare=[.4810751/.2424522/.2764727], 복원 s=정답 100% | VERIFIED | EV1 [D8] | features.py L30–70(EV7) | 예(학습) |
| F5 | 라벨 소스 관찰: F행 100% 팀13 관여; LG관여 R−R전체 편차 2019 −.0776→2024 +.0392; 2023년 4월 −.022→5월 +.0469; LG-R 3-0 성공률 .598 vs LG-R .543, 비LG 3-0 .468 | VERIFIED | EV1 [D3][D4]([O8] R평균 대비 정의)[D5][D6] | — | 피처·상수 근거 |
| F6 | 피처 수: 트리 178(범주형: 팀2·base_state·count_state(+CatBoost는 ID 2)), NN 수치 117+결측지시자 58, TM_ 43개 | VERIFIED | EV2 [PRE], meta/prep(EV4) | meta.json/prep.pkl | 예 |
| F7 | Trackman 매칭: 경기 4,868, 지문 유사도≥0.85, 투수 760/792(purity>0.9 99.6%), 피처는 purity>0.9&votes≥30(701명) 투수 대응만 | VERIFIED | EV3 [INV]·재현 로그(760/792), EV2 [TM] | trackman_match.py; script L402–438 | 예 |
| F8 | 베이즈 오프셋: a=.6, τ²=.004, v_new=.004, σ²=.25, pm=clip(μ+m1(+CTX, NN-ID만),.05,.95), init=[log pm, log((1−pm)fshare_k)] | VERIFIED | EV4 L719–756, L896–906 | script.py | 예 |
| F9 | 멤버·HP: LGB mc4 8시드(lr.03, 잎63, min500, ff.7, bag.8/1, L2 10, 180라운드, 스레드8, wexp .15, F×2, offset bayes) / CatBoost mc 8시드(2750, lr.02, d7, L2 10, border128, GPU, ID범주형, 무오프셋) / NN-ID 8시드(768-384-192, drop.2, idDrop.15, AdamW 1e-3/1e-5, OneCycle, 8ep, bs4096, offset bayes_ctx, wexp.15) / NN무ID 3시드(offset bayes) | VERIFIED | EV4 meta·모델 파라미터 블록(EV2 [PRE]), reproduce.sh(EV7) | train_final.py, nn_export.py | 예 |
| F10 | 앙상블: logit 가중 .25/.35/.10/.30 (시드: 트리=로짓 평균, NN=확률 평균); LGB 경로에 +logit(L_ref) 항 존재 | VERIFIED | EV4 L966–984, ensemble.json | script.py | 예 |
| F11 | 후처리 적용값: 전역 −0.004 → 팀13 관여 R행 +0.015 → 그중 3-0 카운트 +0.105 → clip[.001,.999]; 결측 채움 .4861 | VERIFIED | EV4 L987–1014, ensemble.json | script.py | 예 |
| F12 | 상수 선택 근거(구분 명시): +0.105 = fold-2024 잔차 +0.1047(590행)·규칙 이득 +10.2(재계산); −0.004 = 폴드 레벨 점검 + 초기 LB 후보 비교; +0.015 = 학습 편차 범위 내 LB 후보 3개 비교 | VERIFIED(적용값·+0.105 근거는 재계산; LB 비교 사실은 실험 로그 기록) | EV1 [O3][O5], EV2 [OOF], LOG L446–451 | — | 예 |
| F13 | 검증 체계: 시간 분할(2023년까지 학습→2024 검증, 통계도 2023년까지) fold-2024; 멤버 oracle 972.9/954.8(1시드)·977.4(3시드)/963.1/916.6, 블렌드 1022.9(전역 최적 shift 적용 기준) | VERIFIED | EV1 [O1][O2], EV5 results.jsonl | exp_offset_mc/zoo/nn(개발용) | 예(선택 기준) |
| F14 | 추론: test.csv+학습 산출물만, 32초/253,507행(모의), RAM 3.3GB, CPU only, zip 37파일 91.6MB | VERIFIED | EV2 [CLEAN], EV1 [A1] | script.py | 예 |
| F15 | 행 독립성: 단독 1행 = 전체(≤2.3e-9; 케이스 A~G 피처·트리 0, NN≤8e-8) | VERIFIED | EV2 [SINGLE][INV2] | — | 예(설계 원칙) |
| F16 | 최종 점수: Public/Private 1225.20622 (Private=종료 시점 Public; 1위) | VERIFIED(공개 LB 조회 기록; 1위=순위표 1건 실격 후 사용자 확인) | EV6, 평가 페이지 | — | — |
| F17 | 재현: 제출된 02 폴더 그대로 새 환경 재실행(공식 4파일만, ~45분) → LGB 8/8 비트 동일·ID표/통계 동일·예측 평균 |Δ| ≈3~4e-4·최대 ≤2.5e-3·0.005 초과 0행 → FUNCTIONAL~NEAR-EXACT (GPU 비결정성) | VERIFIED | EV3(Run A, 진행 중→완료 수치로 확정), EV1 [P1][P3][R2] | reproduce.sh | — |
| 제외 | 이진 대비 +43, ROLE/TM +15 등 개발 중 비교 수치 | UNVERIFIED(재계산 안 함) | — | — | PPT 미사용 |

## 슬라이드 설계 (13장)
| # | TITLE | KEY MESSAGE | CONTENT/VISUAL | FACTS |
|---|---|---|---|---|
| 1 | 표지 | 솔루션 정체 | 제목·이름·점수·최종 파일명 | F16 |
| 2 | Solution Overview | 전체 그림 한 장 | 파이프라인 플로(데이터→통계→피처→4멤버→블렌드→보정) + 핵심 3축 요약 | F3,F5,F9,F10,F11 |
| 3 | 문제 이해 → 설계 원칙 | 행 독립 제약을 설계 원리로 | 규칙 요약, "행+학습 통계"만으로 계산, 평가식 | F1,F15 |
| 4 | 데이터 관찰 ① asof의 성질 | 통산 누적 → 정확 분해 가능 | step==1 관찰, 분해 식 유도 그림 | F2,F3 |
| 5 | 데이터 관찰 ② 라벨 소스 구조 | 라벨 과정이 소스별로 다름 | 검증 수치 표(연도 편차·월별 반전·3-0) | F5 |
| 6 | Feature ① 시즌 분해 | 의미→계산→모델 정보 | 수식 + 축소 추정 + 폼/변화 | F3 |
| 7 | Feature ② 사전(prior) 계열 | 이전 시즌만 사용 | CTX 수식, 매치업 shr30, ROLE, TM 43(매칭 요약), LG_regime | F6,F7 |
| 8 | Labels | 4클래스 + 보조 | 복원 알고리즘, 분포·fshare | F4 |
| 9 | 베이즈 레벨 오프셋 | 리그 하락 흡수 | 상태공간 수식·상수, init 구성 | F8, F1 |
| 10 | Models & Ensemble | 최종 4멤버만 | HP·시드 표, 블렌드 식(+logit(L_ref) 명시) | F9,F10 |
| 11 | Validation & Post-processing | 폴드 기준 채택 + 보정 명세 | fold-2024 점수 표; 보정 3단계와 "적용값 vs 선택 근거" 구분 | F13,F11,F12 |
| 12 | Final Inference Pipeline | 제출물 그대로 | 추론 흐름·자원·아티팩트 구성 | F14,F15 |
| 13 | Reproducibility | 실측 재현 결과 | reproduce.sh 절차 + Run A 수치 + 판정(FUNCTIONAL~NEAR-EXACT) | F17 |

## 사전 할루시네이션 감사
- 미사용 피처/모델(P_/B_ 잔차, CTX 확장, 소스 전용 모델, DART 등): 슬라이드에 없음 ✓
- 존재하지 않는 점수/실험: F13의 수치만 사용(재계산 로그) ✓ / 개발 중 비교 수치(+43 등) 제외 ✓
- 후처리 근거: "적용값"(코드)과 "선택 근거"(F12) 구분 서술 ✓
- 재현성: Run A 실측치로만 서술(완료 후 수치 삽입) ✓

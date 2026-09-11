# SOLUTION_PPT_AUDIT (2026-09-02)
- 기존 PPT: final_submission/03_솔루션_PPT.pptx sha256 d619d236d07f186e — 내용 미참조(파일명·위치 확인 용도만), 완전 폐기.
- 새 PPT: 13장, `docs/SOLUTION_PPT_SOURCE.md`의 FACT TABLE(F1~F17, 전부 VERIFIED)만 사용해 처음부터 생성.
- technical_claims_checked: 17 (F1~F17) + 제외 목록 1식
- verified_claims: 17 / unverified_claims **used**: 0 (미검증 개발 수치 — 이진 대비 +43, ROLE/TM +15 등 — 은 전부 미사용)
- numerical_claims_checked: 31 (데이터 사실 9, 분해·라벨 6, 폴드 점수 6, 후처리 상수 4, 재현 수치 6) — 근거: experiments/fact_audit_v59.log, rule_audit_v59_evidence.log, reproduction_audit.log(Run A), ppt_fact_check.log, results.jsonl
- incorrect_claims_found: 0 (생성 전 소스 문서 감사에서 미검증 후보 사전 제외)
- visual_issues_found: 10 (플로 화살표 텍스트박스 높이 휴리스틱 경고) / visual_issues_fixed: 10 (높이 0.42in로 보정) → 재검사 0
- 검증 방법: python-pptx 재열람(13장), 도형 경계(슬라이드 밖 0), 최소 폰트(≥10pt), 텍스트 오버플로 휴리스틱(0), 표 크기 확인. 렌더링 도구(soffice) 부재로 픽셀 렌더 검사는 수행하지 못함(UNKNOWN) — 대신 기하 검사로 대체.
- 재현성 슬라이드 수치 = Run A 실측(reproduction_audit.log): 평균|Δ| 0.000329, 최대 0.002243, corr 0.999948, >0.005 0행, LGB 8/8 비트 동일 → FUNCTIONAL~NEAR-EXACT 표기.

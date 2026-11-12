# 학습 시간을 고려한 LightGBM 최종 보고서 수정

사용자 요청에 따라 동일한 피처·분할에서 관측 정확도가 근접한 HGB 400회 두 단계와 LightGBM 두 단계 중, 학습 시간이 짧은 LightGBM을 최종 모델로 선택했습니다.

| 후보 | 학습 내부 5-fold 평균 로그 RMSE | 평균 모델 적합 시간 |
|---|---:|---:|
| HGB 두 단계, 400회 고정 | 0.095340 | 13.44초 |
| **선택: LightGBM 두 단계, 400회 고정** | **0.095367** | **6.76초** |
| 이전 제출 HGB 두 단계, 조기 종료 | 0.096619 | 6.75초 |

HGB 400회와 LightGBM의 관측 오차 차이는 약 0.000027 dex이며, 시간은 약 49.7% 짧았습니다. 이전 제출 HGB보다 빠르다는 주장은 아닙니다. 시간·정확도 절충은 결과 확인 후 정한 사용자 선택이고 정확도 동등성 검정을 수행한 것은 아닙니다. 시간은 동일 컴퓨터·CPU 6스레드의 관측값이며 배경 작업 등을 완전히 통제한 성능 시험은 아닙니다.

## 수정 범위

- 저장소 루트의 `DataHub_701104_final.ipynb`: 최종 모델 코드, 선택 근거, 비교표, 오차 분석, 물리 해석, 결론, 모델 카드와 RESULTS를 LightGBM에 맞췄습니다.
- 원래 Part I와 DO NOT MODIFY 코드는 보존합니다. HGB에서 수행한 피처·문턱·제거 실험은 HGB의 선택 이력으로 구분합니다.
- 기존 비교 폴더의 `report.md`, `make_report.py`, `README.md`도 최종 선택과 일치시킵니다. CSV/OOF 원수치는 바꾸지 않습니다.
- 두 단계 모두 34개 입력, LightGBM 400회, 학습률 0.08, num_leaves=31, max_bin=255, seed=42, 6스레드, 조기 종료 없음입니다. 0 문턱 0.9는 기존 HGB에서 이전했으며 재튜닝하지 않습니다.

## 실행과 검증 자료

- `selection_decision.json`: 선택 근거와 원래 35개 분할 기록·7개 구성의 요약.
- `execution_evidence.json`: 전체 노트북 순차 실행의 RESULTS와 새 LightGBM 예측에 따른 진단·중요도·실행 정보.
- `independent_prediction.py` / `independent_prediction.json`: 이미 검토한 비교 코드로 최종 학습 분할을 별도로 적합하고 검증 점수를 재현한 결과.
- `validation.json`: 순차 실행·오류 부재, Part I와 고정 코드 보존, 입력 계약, 독립 예측과 지표 일치, 선택 설명을 확인하는 최신 검증 결과.
- `requirements.txt`: 이번 실행에 사용한 환경.
- `reference/`: 수정 전 HGB 노트북 및 변경한 비교 문서의 원본. 이전 비교의 artifact manifest와 검토 기록은 이 시점의 소스·문서에 적용되는 역사 기록입니다.

저장소 `.venv`에서 다음과 같이 실행합니다.

```powershell
.venv\Scripts\python.exe "work/조장/lightgbm_report_revision_2026-11-12/run_notebook.py"
.venv\Scripts\python.exe "work/조장/lightgbm_report_revision_2026-11-12/finalize_report.py"
.venv\Scripts\python.exe "work/조장/lightgbm_report_revision_2026-11-12/independent_prediction.py"
.venv\Scripts\python.exe "work/조장/lightgbm_report_revision_2026-11-12/validate_report.py"
```

최종 검증셋 점수는 노트북 RESULTS와 `execution_evidence.json`을 따르며 위 교차검증 평균과 구분합니다. 기존 검증 표본은 이전 분석에서도 사용됐으므로 새로운 시험 일반화 점수로 해석하지 않습니다. +1 로그 offset 역시 기존 팀 가정입니다.

## 최종 확인

전체 61개 코드 셀의 순차 실행과 그래프 10개를 확인했습니다. 최종 LightGBM의 공통 검증 로그 RMSE는 **0.0957048430**이며 별도 비교 코드의 재현 결과와 일치했습니다. Part I, 고정 템플릿 코드, 원자료와 중간 노트북의 보존 검사도 통과했습니다.

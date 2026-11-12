# HGB · LightGBM · CatBoost 고정 예산 비교 (2026-11-12)

현재 최종 모델(HGB 두 단계, 34개 피처, 0 문턱 0.9)과 LightGBM, CatBoost 대안을 **같은 피처, 같은 외부 학습 분할, 같은 내부 5-fold**에서 비교하는 실험입니다. 전체 7개 구성의 5-fold 실행과 독립 OOF 검증을 완료했습니다. 결과와 보고서용 설명은 [report.md](report.md)에, 원수치는 `results/`에 있습니다. HGB 400회 두 단계의 평균 로그 RMSE는 0.095340, LightGBM 두 단계는 0.095367, 기존 제출 모델 재현은 0.096619입니다.

> 이 비교 결과로 최종 모델이 자동으로 바뀌지 않습니다. `DataHub_701104_final.ipynb`의 최종 모델·제출 점수는 그대로이며, 교체 여부는 팀이 따로 판단합니다.

## 파일

| 파일 | 내용 |
| --- | --- |
| `protocol.json` | 점수 계산 전에 고정한 설정 전체 (피처 순서, 분할, 모델 파라미터, 선택 규칙, 재현 기준값, 주의사항). 실행 뒤 수정하지 않습니다. |
| `benchmark.py` | 실행 스크립트. `protocol.json`을 읽고 SHA256만 기록합니다. import만으로는 아무것도 실행되지 않습니다. |
| `results/` | 전체 실행 출력 (실행 후 생성) |
| `smoke_results/` | smoke 점검 출력 (실행 후 생성, 비교 근거로 쓰지 않음) |

## 실행 방법

저장소 루트의 `.venv`(Python 3.13, scikit-learn 1.9.1, lightgbm, catboost 설치)를 씁니다. 저장소 루트에서:

```powershell
# 1) 코드 경로 점검: 학습 3000행, 2-fold, 20회 -> smoke_results/
.venv\Scripts\python.exe "work/조장/model_comparison_2026-11-12/benchmark.py" --smoke

# 2) 전체 실행: 7개 모델 x 5-fold, 400회 -> results/
.venv\Scripts\python.exe "work/조장/model_comparison_2026-11-12/benchmark.py" --threads 6
```

- `--models a,b`: 일부 모델만 실행합니다(진단용). 완료 기록에 나머지 모델은 미완료로 남습니다.
- `--resume`: 기존 출력 폴더의 지문(protocol·데이터·노트북·스크립트 해시, 피처, 분할 해시, 라이브러리 버전, 스레드 수)이 같을 때만, 끝나지 않은 모델을 다시 계산합니다. 이미 완료된 모델은 건너뜁니다.
- `--folds`: 전체 실행은 5, smoke는 2만 허용합니다. 다른 값은 거부합니다.
- 기존 결과가 있는 폴더에 `--resume` 없이 쓰면 거부합니다. smoke는 `results/`에, 전체 실행은 `smoke_results/`에 쓸 수 없고, manifest의 smoke 표시가 다른 폴더에도 쓰지 않습니다.
- 모델과 fold는 순서대로 하나씩 실행합니다(벽시계 시간 비교용). HGB와 BLAS는 `threadpoolctl.threadpool_limits(threads)`, LightGBM은 `n_jobs`, CatBoost는 `thread_count`로 스레드 수를 맞춥니다.

## 비교 규약 요약

1. **데이터와 피처.** 제공 CSV만 씁니다(ID `subhalo_id`, 타깃 `sfr_linear`, 입력 13개). 최종 노트북에서 `build_environment_features`, `build_mass_history_features` 두 함수만 `ast`로 꺼내 정의하고, 노트북의 다른 코드는 실행하지 않습니다. 환경 피처 17개는 노트북 9.2와 같이 13개 입력으로 사전식 정렬(`mergesort`, `na_position='last'`)한 순서로 계산한 뒤 원래 index로 되돌립니다. 타깃을 쓰지 않는 기하 피처이므로 CSV 전체 행의 입력으로 계산합니다(조교 FAQ 허용). 피처 순서는 원래 13개 + 환경 17개 + 질량 성장·비율 4개 = 34개입니다. NaN은 `fof_dv_group`(고립 은하)에만 허용하고, 대체나 스케일링 없이 각 라이브러리의 기본 결측 처리에 맡깁니다.
2. **분할.** 외부 분할은 노트북과 같은 `train_test_split(test_size=0.2, random_state=42, stratify=(y > 0))`입니다. 반환된 학습 인덱스 순서 그대로 166,749행을 쓰고, 그 안에서 `KFold(n_splits=5, shuffle=True, random_state=42)`로 나눕니다. **전체 타깃의 0 여부는 공통 층화 분할에만 사용하고, 외부 val 타깃은 모델 학습·후보 채점에 쓰지 않습니다.**
3. **타깃과 지표.** z = log10(SFR + 1)로 학습하고, 예측은 yhat = max(10^z − 1, 0)으로 되돌린 뒤 log10(yhat + 1)과 log10(y + 1)의 RMSE(`rmse_log`)로 평가합니다. SFR = 0, SFR > 0, SFR > 100 부분집합 RMSE와 0 재현율·정밀도(yhat == 0 기준)를 함께 남깁니다. 부분집합이 비면 0이 아니라 NaN입니다.
4. **두 가지 구조.**
   - single: 회귀기 하나가 모든 행의 z를 학습합니다.
   - two_stage: 분류기가 P(SFR > 0)를, 양수 회귀기가 SFR > 0 행의 z를 학습하고 z = p · m으로 합칩니다. P(SFR = 0) = 1 − p ≥ 0.9이면 z = 0(SFR = 0)을 냅니다.
5. **모델 7개** (자세한 파라미터는 `protocol.json`, 실제 값은 `manifest.json`의 `get_params`):

   | 이름 | 설정 |
   | --- | --- |
   | `hgb_single_fixed400`, `hgb_two_stage_fixed400` | HistGradientBoosting, learning_rate 0.08, 정확히 400회, max_leaf_nodes 31, max_bins 255, early stopping 끔 |
   | `lightgbm_single_fixed400`, `lightgbm_two_stage_fixed400` | LGBM, learning_rate 0.08, 400회, num_leaves 31, max_bin 255, deterministic, force_col_wise, 검증 세트·콜백 없음 |
   | `catboost_single_fixed400`, `catboost_two_stage_fixed400` | CatBoost, learning_rate 0.08, 400회, depth 5(대칭 트리, 잎 32개), border_count 254, CPU, eval_set 없음 |
   | `hgb_two_stage_reference` | 최종 노트북과 같은 HGB(learning_rate 0.08, max_iter 400, max_leaf_nodes 31). early stopping은 sklearn 기본값(`'auto'`, 학습 fold 안에서 10%를 떼어 냄)이라 실제 트리 수가 400보다 적습니다. |

6. **튜닝 없음.** 하이퍼파라미터 격자, 결과를 보고 바꾸는 설정, 채점 fold를 쓰는 early stopping, 문턱 재선택이 모두 없습니다. 0.9 문턱은 HGB 최종 모델에서 고른 값이며 LightGBM/CatBoost에 맞게 다시 고르지 않았습니다.
7. **선택 규칙(고정).** 내부 5-fold 평균 `rmse_log`로 순위를 매깁니다. 후보가 `hgb_two_stage_reference`보다 평균이 낮고 **5개 fold 중 4개 이상**에서 낮을 때만 `promising = true`로 표시합니다. 이 표시는 탐색적 신호이며 증명이 아니고, 최종 모델을 자동으로 바꾸지 않습니다.

## 출력 파일 (`results/`)

| 파일 | 내용 |
| --- | --- |
| `manifest.json` | 학습 전에 저장. 시작 시각(UTC), smoke 여부, git 커밋, CSV·노트북·스크립트·protocol SHA256, 추출 함수 소스 해시, 피처 목록, 환경 정렬 해시, 외부 학습 인덱스(순서 포함)와 fold별 학습/채점 ID 해시, 라이브러리·Python 버전, 스레드 정보, 모델별 `get_params`, 행 수, 주의사항 |
| `protocol_snapshot.json` | 실행에 쓴 `protocol.json`의 바이트 복사본 |
| `fold_results.csv` | 모델·fold가 끝날 때마다 저장. `rmse_log`, 부분집합 RMSE, 0 재현율·정밀도, 양수인데 0으로 낸 수, 학습·예측 시간, 분류기·회귀기 트리 수 |
| `oof_predictions.npz` | 모델·fold가 끝날 때마다 저장. `train_indices`(CSV 행 번호), `subhalo_id`, `y_true_linear`, `y_true_log`, `fold_id`, 모델별 `yhat_linear_<모델>`(미완료는 NaN) |
| `summary.csv` | 모델별 fold 평균·표준편차, 평균 학습 시간, OOF 지표, 순위, `promising_vs_reference` |
| `paired_differences.csv` | 같은 fold끼리의 차이(후보 − 기준). 기준은 `hgb_two_stage_reference`와, LightGBM/CatBoost는 같은 구조의 통제 HGB(`hgb_single_fixed400` 또는 `hgb_two_stage_fixed400`) |
| `paired_summary.json` | 비교별 평균 차이, 표준편차, 후보가 이긴/진 fold 수 |
| `summary.json` | 기계 판독용 요약: 완료 기록, 순위, 짝 비교, OOF 일관성 점검, reference 재현 점검, 주장하지 않는 것 목록 |

## 재현 점검

`hgb_two_stage_reference`는 기존 진단(`work/조장/review_revision_2026-11-11/diagnostic_cv.csv`의 `E1H1_34`, 노트북 9.2 보조)과 같은 설정·분할입니다. 그 기록의 fold 평균은 0.096619이고 fold별 값과 트리 수는 `protocol.json`의 `reference_expected`에 있습니다. 전체 실행 뒤 `summary.json`의 `reference_reproduction.within_tolerance`(fold당 절대 차이 1e-5 이하)를 먼저 확인해 주세요. 라이브러리 버전이나 스레드 수에 따라 부동소수점 차이가 날 수 있으며, 허용 범위를 벗어나면 다른 모델 결과를 쓰기 전에 원인을 확인해야 합니다.

## 보고서에 쓸 때 (§9 · §10)

- §9(최종 모델)에는 "같은 34개 피처와 같은 학습 내부 5-fold에서 LightGBM·CatBoost를 고정 예산으로 비교했다"는 사실과 `summary.csv`의 평균 ± 표준편차, 짝 비교의 fold 승패 수를 옮깁니다.
- §10(비교·한계)에는 아래 주의사항을 함께 적습니다. 결과가 `promising`이어도 "더 좋은 모델임을 보였다"가 아니라 "추가 검토할 가치가 있다"로 써 주세요.
- 이 결과는 **외부 학습 분할 내부의 교차검증**입니다. 외부 val은 이전 단계에서 여러 번 쓰였으므로 이 비교는 새 시험 데이터에서의 일반화 성능이 아닙니다.

## 한계와 주의사항

- 400회 · learning_rate 0.08은 비교를 위한 **고정 예산**이며 어느 알고리즘에도 전역 최적 설정이라고 주장하지 않습니다.
- 계산 예산이 같다고 말할 수 없습니다. two_stage는 분류기와 양수 회귀기 두 모델을 학습하고, HGB·LightGBM은 잎 31개의 leaf-wise 트리, CatBoost는 깊이 5의 대칭 트리(잎 최대 32개)를 씁니다. 정규화·분할 규칙·결측 처리 등 나머지 기본값도 라이브러리마다 다르며, 실제 값은 `manifest.json`의 `get_params`와 각 라이브러리 문서로 확인합니다.
- 0.9 문턱은 HGB에서 고른 값이라 HGB 계열에 유리할 수 있습니다.
- reference는 early stopping으로 트리 수가 fold마다 달라 통제 모델과 예산이 다릅니다.
- fold들은 데이터를 공유하므로 독립 표본이 아닙니다. p값이나 신뢰구간을 내지 않습니다.
- 시드는 42 하나이고 무작위 KFold만 씁니다. 시드 반복과 공간 블록 일반화는 범위 밖입니다.
- 학습·예측 시간은 이 컴퓨터, 이 스레드 수에서 순차 실행한 값이라 다른 환경에서는 달라집니다.

## 출처

- 최종 노트북: `DataHub_701104_final.ipynb` 9.2(환경·질량 피처, `TwoStageLogSFR`, 정렬 순서), 9.2 보조 진단(학습 내부 5-fold, eps 1, 문턱 0.9, 트리 상한 400)
- 기준 commit: `6bc3988dd7effb7694192ce059f515fa496fa499`
- 기존 진단 기록: `work/조장/review_revision_2026-11-11/diagnostic_cv.csv`
- 조교 FAQ: `조교_FAQ_2026-11-11.md` (로그 지표, 환경 피처의 입력 합집합 사용)


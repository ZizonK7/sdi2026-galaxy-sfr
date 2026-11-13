# 계층형 46피처 모델: 꼬리 회귀기 교체 실험

전체 로그 RMSE 기준으로 seed42에서 고정한 후보는 **LightGBM 400 / 31 leaves**다. 같은 시드에서 극단값(SFR>100) 오차가 가장 작은 모델은 **기존 HGB400**다.

이번 기준 모델은 직전의 계층형 46피처 모델이다. 현재 채택된 A34나 초기 3분류 46피처 모델과의 직접 비교가 아니다. 학습한 후보를 별도 저장했으며 현재 모델 지정과 GitHub는 변경하지 않았다.

## 이번 결과의 판단

LightGBM 400 / 31 leaves가 두 시드 모두 전체 점수 1위였다. HGB 대비 전체 개선 폭은 seed42 약 0.019%, seed7 약 0.099%로 작다. 실제 0 구간 RMSE도 두 시드에서 감소했지만 seed7 차이는 매우 작다. 극단값 RMSE는 seed42에서 약 0.85% 악화, seed7에서 약 0.47% 개선으로 일관되지 않았다.

따라서 LightGBM은 전체 점수를 위한 후보로 보관하되, 극단값 개선이 확인됐다고 판단하지 않는다. CatBoost는 전체 점수 개선이 시드에 따라 달랐고, 두 설정 모두 극단값에서는 HGB보다 나빴다. ExtraTrees는 두 설정·두 시드 모두 전체와 극단값 오차가 증가해 이번 구조에서는 제외하는 편이 타당하다. 이번 결과만으로 현재 채택 모델을 바꾸지는 않는다.

## 비교 조건

- 0/양수 HGB400 분류기와 양수 내 중간/꼬리 HGB400 분류기는 고정했다.
- 일반 회귀기 HGB400, 46개 피처, 잔차 피처 생성, P(SFR=0)≥0.9이면 0 처리 규칙도 고정했다.
- 꼬리 회귀기는 각 학습 fold의 **실제 SFR>10** 행으로만 학습했다. 분류기가 꼬리로 예측한 행을 학습 표본으로 쓰지 않았다.
- 회귀 목표는 log10(1+SFR)이며 분류 확률과 일반·꼬리 회귀 예측을 로그 공간에서 혼합했다.
- LightGBM·CatBoost·ExtraTrees 각 2개 설정을 비교했다. 상세 고정 파라미터는 tail_models.py에 있다.
- 기존 train 80%에서 seed42와 seed7 각각 5-fold OOF를 계산했다. 원래 holdout 20%의 정답은 평가하지 않았다.
- 전체 점수는 **5개 fold 로그 RMSE의 평균**, 구간별 점수는 **합친 OOF의 로그 RMSE**다. 모두 낮을수록 좋다.
- train 166,749개 중 꼬리 학습 대상 SFR>10은 11,696개, 극단값 평가 대상 SFR>100은 209개다.
- seed42의 최저 전체 RMSE 모델을 먼저 고정한 뒤 seed7을 계산했다. seed7 결과를 보고 다시 선정하지 않았다.

## 전체 후보 결과

### Seed 42

| 꼬리 회귀기 | 전체 | 실제 0 | SFR>10 | SFR>100 |
|---|---:|---:|---:|---:|
| 기존 HGB400 | 0.09340135 | 0.102411 | 0.148325 | 0.515734 |
| LightGBM 400 / 31 leaves | 0.09338403 | 0.102333 | 0.148485 | 0.520110 |
| LightGBM 600 / 63 leaves | 0.09346678 | 0.102444 | 0.148821 | 0.522116 |
| CatBoost 400 / depth 6 | 0.09346629 | 0.102415 | 0.148744 | 0.534243 |
| CatBoost 600 / depth 8 | 0.09356444 | 0.102485 | 0.149355 | 0.542903 |
| ExtraTrees 300 / leaf min 3 | 0.09394337 | 0.102676 | 0.151250 | 0.566539 |
| ExtraTrees 300 / leaf min 10 | 0.09422384 | 0.102594 | 0.152988 | 0.584232 |

### Seed 7

| 꼬리 회귀기 | 전체 | 실제 0 | SFR>10 | SFR>100 |
|---|---:|---:|---:|---:|
| 기존 HGB400 | 0.09393696 | 0.105236 | 0.148811 | 0.525803 |
| LightGBM 400 / 31 leaves | 0.09384352 | 0.105228 | 0.148384 | 0.523347 |
| LightGBM 600 / 63 leaves | 0.09393489 | 0.105220 | 0.149059 | 0.529749 |
| CatBoost 400 / depth 6 | 0.09391207 | 0.105034 | 0.148790 | 0.533979 |
| CatBoost 600 / depth 8 | 0.09387650 | 0.105049 | 0.148514 | 0.533587 |
| ExtraTrees 300 / leaf min 3 | 0.09432336 | 0.105098 | 0.151096 | 0.566456 |
| ExtraTrees 300 / leaf min 10 | 0.09456700 | 0.105130 | 0.152471 | 0.580674 |

## 사전에 고정한 전체 점수 후보의 변화

후보: **LightGBM 400 / 31 leaves**. 아래 변화율은 기존 HGB 대비이며 음수가 개선이다.

| 시드 | 전체 변화율 | 실제 0 변화율 | SFR>100 변화율 | 전체 개선 fold |
|---|---:|---:|---:|---:|
| 42 | -0.0185% | -0.0756% | +0.8484% | 3/5 |
| 7 | -0.0995% | -0.0071% | -0.4670% | 3/5 |

분류기와 0 처리 규칙이 같으므로 0으로 출력되는 행과 0 판정 precision/recall은 동일하다. 실제 0 행의 RMSE가 달라지는 것은 0으로 고정하지 않은 행에서 꼬리 회귀값이 혼합 예측에 영향을 주기 때문이다.

## 해석과 한계

꼬리 회귀기 교체가 전체 점수를 조금 개선하더라도 극단값까지 개선했다는 뜻은 아니다. SFR>10 전문가의 학습 범위와 SFR>100 극단값 평가 범위를 구분해야 한다. 분류기의 꼬리 확률이 낮으면 좋은 꼬리 회귀 예측도 최종 혼합에서 작아질 수 있다.

summary.csv의 extreme_expert_only_diagnostic_rmse는 실제 극단값 행에 꼬리 전문가를 직접 적용한 진단 수치다. 실제 배포 예측은 정답을 알 수 없으므로 이 수치를 모델 성능이나 선정 기준으로 쓰지 않았다.

두 시드는 같은 관측치를 다시 분할한 탐색적 비교다. 이전 실험에서도 사용한 train 데이터이므로 새 독립 테스트가 아니며, 작은 차이를 확정적인 일반화 성능 향상으로 해석할 수 없다. 극단값도 209개로 적다. 알고리즘별 계산량과 트리 구조가 달라 동등 계산 예산의 비교는 아니다.

## 저장한 모델

- `selected_model.joblib`: LightGBM 400 / 31 leaves. SHA256 `6f722bcac69e3f033265b97591bdea36bc0959e95a86acb471b363b2b2abeb78`.
- `extreme_tail_option.joblib`: 기존 HGB400. SHA256 `9f07599647a43d2b243ed2b3842ee7694077e70bb4a2b1c35e69454421be2985`.

`selected_model.joblib`는 seed42 전체 점수로 고정한 실험 후보이며 자동 채택을 뜻하지 않는다. `extreme_tail_option.joblib`가 있으면 seed42의 극단값 점수가 가장 낮은 별도 선택지다. 그 선택지가 HGB인 경우 기존 모델을 보존한 것이며 새로 개선한 꼬리 모델이 아니다.

## 불러오기

패키지를 풀고 requirements.txt의 버전을 사용한다. 파이썬 모듈 파일도 모델과 함께 보관한다. reference는 환경 피처 계산에 쓰는 고정 원본 카탈로그이며 sfr_linear는 제거한다. ID는 입력과 reference의 중복 행 식별에 사용하고 모델 피처에는 넣지 않는다.

```python
import sys
from pathlib import Path
import joblib
import pandas as pd
bundle = Path("/absolute/path/to/hierarchical_tail_models")
sys.path.insert(0, str(bundle))
from feature_pipeline import predict_sfr
model = joblib.load(bundle / "selected_model.joblib")
reference = pd.read_csv("/absolute/path/to/reference.csv").drop(columns="sfr_linear", errors="ignore")
new_table = pd.read_csv("/absolute/path/to/input.csv").drop(columns="sfr_linear", errors="ignore")
prediction = predict_sfr(new_table, model, reference)
```

## 재현과 검증

실험 재실행은 직전 hierarchical_gate46/의 metadata·모델·OOF 확률 파일과 원본 CSV를 필요로 한다. --source-dir로 그 경로를 지정한다. 캐시가 없으면 고정된 기존 회귀기를 fold 안에서 재학습한다. 예측만 할 때는 이전 실험 폴더가 필요하지 않다.

```bash
python run_experiment.py --data /absolute/path/to/data.csv \
  --source-dir /absolute/path/to/hierarchical_gate46 \
  --cache /absolute/path/to/feature_search_cache
```

저장 OOF에서 요약·fold 점수 1,526개를 재계산해 확인했다. 기존 예측 재구성, OOF 행 일치, 0 처리 동일, 모델 해시, 원본 모델 보존을 검증했다. 학습 실행에서는 저장/재로딩 예측 일치, 고정한 분류기·일반 회귀 예측 일치, 원시 입력 예측 인터페이스도 확인했다.

원본 데이터는 ZIP에 넣지 않았다. source SHA와 라이브러리 버전은 metadata.json, 선정 규칙은 protocol.json, 검증 결과는 validation.json에 기록했다.

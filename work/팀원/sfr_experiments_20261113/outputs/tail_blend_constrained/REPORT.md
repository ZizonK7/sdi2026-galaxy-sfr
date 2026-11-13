# 전체 오차 증가를 제한한 꼬리 전문가 혼합

**기존 무가중치 LightGBM과 3배 가중치 LightGBM을 50:50으로 섞은 모델**이 사용자가 요청한 전체·극단값 오차 동시 감소를 세 시드에서 모두 달성했다. 이 후보를 **balanced_model.joblib**로 저장했다. 현재 채택 모델은 자동으로 교체하지 않았다.

**0 구간까지 모든 시드에서 좋아진 것은 아니다.** 실제 0 구간은 seed42·7에서 개선됐지만 seed99에서 약 0.0583% 악화됐다. 사용자의 전체·꼬리 목표와 별도로 이번 실험에 추가한 “0 구간 증가 없음”의 엄격한 조건은 통과하지 못했다. 이 실패를 숨기거나 조건 통과로 바꾸지 않았다.

원래의 엄격한 수용 절차는 **기존 무가중치 LightGBM** 유지로 기록되어 있다. recommendation.json과 recommended_model.joblib는 이 엄격 조건의 결과를 보존한 감사용 기록이며, 새 혼합 후보를 사용하려면 balanced_model.joblib를 불러온다.

## 목적과 비교 기준

- 개선 목표는 실제 **SFR>100** 극단값의 로그 RMSE다.
- 꼬리 회귀기의 학습 구간은 기존처럼 **SFR>10**이다. 분류 경계를 100으로 바꾸는 실험이 아니다.
- 전체 로그 RMSE 증가 허용치를 기존 대비 **상대 0.1% 이내**로 두었다. 사용자의 “크게 늘어나지 않는” 조건에 대한 작업 가정이다. 절대 RMSE +0.1이 아니다.
- 실제 SFR=0 구간 RMSE는 증가시키지 않는 조건을 추가했다.
- 기준은 최근 계층형 46피처·무가중치 LightGBM400/31 꼬리 모델이다. 현재 채택 상태인 A34와의 직접 비교는 아니다.
- 두 HGB 분류기, 일반 HGB 회귀기, 46개 피처, P(0)≥0.9의 0 처리 규칙은 그대로다.
- 회귀기의 로그 예측을 섞었다. 예측할 때 실제 SFR이나 정답 구간을 사용하지 않는다.

## 선택과 검증 절차

기존 LightGBM과 HGB 무가중치·HGB/LightGBM 가중치 2·3·5 모델을 25·50·75% 비중으로 섞은 21개 고정 후보와 기존 모델을 seed42 OOF에서 비교했다. 전체 증가 ≤0.1%·0 구간 증가 없음 조건에서 극단값 RMSE가 가장 낮은 “꼬리 개선 우선” 후보와, 전체 증가 자체를 허용하지 않는 “전체 증가 없는 후보”를 각각 고정했다.

이 두 후보만 기존 seed7과 새로 학습한 seed99의 5-fold에서 확인했다. 그 뒤 비율이나 가중치를 다시 조정하지 않았다. 최종 수용은 세 시드 모두 조건을 만족하는지로 판단한다. 꼬리 개선 우선 후보를 먼저 수용하고, 실패하면 전체 증가 없는 후보, 둘 다 실패하면 기존 모델을 유지한다.

모든 시드는 **같은 원래 train80% 관측치**의 재분할이다. seed99도 독립 테스트셋은 아니다. 기존 피처와 모델은 이전 실험에서 선택된 것이므로 이 결과는 탐색적 반복 교차검증이다. 원래 holdout20%의 정답은 이번 실험에서 평가하지 않았다.

## 고정한 두 후보

- **꼬리 개선 우선**: 기존 25% + lgb_w3 75%. 세 시드 수용 조건 통과: 아니오.
- **전체 증가 없는 후보**: 기존 50% + lgb_w3 50%. 세 시드 수용 조건 통과: 아니오.

전체는 5개 fold 로그 RMSE의 평균, 구간별 점수는 합친 OOF의 로그 RMSE다. 모두 낮을수록 좋다.

| 시드 | 모델 | 전체 | 실제 0 | SFR>100 |
|---:|---|---:|---:|---:|
| 42 | 기존 무가중치 LightGBM | 0.09338403 | 0.102333 | 0.520110 |
| 42 | 꼬리 개선 우선 | 0.09342456 | 0.102303 | 0.510864 |
| 42 | 전체 증가 없는 후보 | 0.09336728 | 0.102312 | 0.513193 |
| 7 | 기존 무가중치 LightGBM | 0.09384352 | 0.105228 | 0.523347 |
| 7 | 꼬리 개선 우선 | 0.09390176 | 0.105151 | 0.517281 |
| 7 | 전체 증가 없는 후보 | 0.09383616 | 0.105175 | 0.518486 |
| 99 | 기존 무가중치 LightGBM | 0.09341216 | 0.104190 | 0.492741 |
| 99 | 꼬리 개선 우선 | 0.09338591 | 0.104284 | 0.487549 |
| 99 | 전체 증가 없는 후보 | 0.09334787 | 0.104251 | 0.488371 |

## 기준 모델 대비 변화율

| 시드 | 후보 | 전체 | 실제 0 | SFR>100 | 수용 조건 |
|---:|---|---:|---:|---:|---|
| 42 | 꼬리 개선 우선 | +0.0434% | -0.0301% | -1.7776% | 통과 |
| 7 | 꼬리 개선 우선 | +0.0621% | -0.0731% | -1.1591% | 통과 |
| 99 | 꼬리 개선 우선 | -0.0281% | +0.0898% | -1.0535% | 미통과 |
| 42 | 전체 증가 없는 후보 | -0.0179% | -0.0212% | -1.3299% | 통과 |
| 7 | 전체 증가 없는 후보 | -0.0078% | -0.0500% | -0.9288% | 통과 |
| 99 | 전체 증가 없는 후보 | -0.0688% | +0.0583% | -0.8868% | 미통과 |

음수는 개선이다. P(0)≥0.9인 행은 모든 후보에서 그대로 0이다. 꼬리 전문가의 원래 로그 예측을 혼합한 다음 SFR≥0 보정을 한 번 적용한다. 회귀의 음수 예측을 보정해 추가로 0이 되는 행은 분류기의 0 판정과 구분한다. 기준 대비 0 출력 행 차이는 metadata.json의 zero_output_differences에 기록했다.

최종 추천은 정해 둔 조건을 통과한 후보라는 뜻이다. 모집단 성능이 보장된다는 의미는 아니며, SFR>100 평가 표본은 209개로 적다. 특정 시드의 작은 차이를 확정적 우위로 해석하지 않는다.

## 0 구간을 보호하려고 추가로 확인한 방법

seed99 결과를 본 뒤, 기존 예측이 SFR>10인 행에만 50:50 혼합을 적용하고 나머지는 기존 예측을 유지하는 고정 규칙을 한 번 더 확인했다. 실제 정답이 아닌 기존 모델의 예측으로 적용 여부를 결정했다. 이는 세 시드를 다시 사용한 **사후 탐색**이며 새로운 독립 검증으로 해석하지 않는다.

| 시드 | 전체 변화율 | 실제 0 변화율 | SFR>100 변화율 |
|---:|---:|---:|---:|
| 42 | -0.0295% | -0.0716% | -1.3962% |
| 7 | -0.0121% | -0.0333% | -0.9942% |
| 99 | -0.0686% | +0.0562% | -0.8393% |

이 규칙도 seed99에서 실제 0 구간 오차 증가를 없애지 못했다. 따라서 추가 규칙을 최종 후보에 넣지 않고 처음 고정한 50:50 혼합을 보관했다.

## 모델과 실행

`balanced_model.joblib`가 사용자의 전체·극단값 목표를 충족한 50:50 혼합 후보다. `balanced_candidate.json`에 선택 이유와 0 구간의 제한을 명시했다.

`tail_priority.joblib`와 `strict_balance.joblib`는 고정한 두 후보다. `strict_balance.joblib`는 balanced_model.joblib와 동일하다. `recommended_model.joblib`는 추가 엄격 조건 때문에 유지한 기존 기준 모델이다. 분류기·일반 회귀기·피처 전처리까지 포함한다. 기존 전체 train80%로 학습된 전문가들을 그대로 재사용해 혼합했으며, 정답 기반 사후 라우팅은 없다.

원본 모델 파일과 current_model.json을 변경하지 않았고 GitHub에 푸시하지 않았다.

- `tail_priority.joblib` SHA256 `a5f6f0c5282c721596414d25316004fde90a4a82e1090f8c7a4cf352dfb82602`
- `strict_balance.joblib` SHA256 `c8878b1af31ed5845bd2967cefe5ee850c7f7d253d2ad9ad44bbdd9d84e54fb7`
- `recommended_model.joblib` SHA256 `2a04721ecab37b8ad8b0bd10c1328f4121c98cf492ad42fc8b53d9679bce508a`
- `balanced_model.joblib` SHA256 `c8878b1af31ed5845bd2967cefe5ee850c7f7d253d2ad9ad44bbdd9d84e54fb7`

requirements.txt의 환경에서 패키지의 파이썬 모듈을 함께 사용한다. reference는 기존 환경 피처의 고정 기준 카탈로그다.

```python
import sys
from pathlib import Path
import joblib
import pandas as pd
bundle = Path("/absolute/path/to/tail_blend_constrained")
sys.path.insert(0, str(bundle))
from feature_pipeline import predict_sfr
model = joblib.load(bundle / "balanced_model.joblib")
reference = pd.read_csv("/absolute/path/to/reference.csv").drop(columns="sfr_linear", errors="ignore")
new_table = pd.read_csv("/absolute/path/to/input.csv").drop(columns="sfr_linear", errors="ignore")
predictions = predict_sfr(new_table, model, reference)
```

실험 재실행에는 원본 CSV와 이전 hierarchical_tail_weighting·hierarchical_gate46·hierarchical_tail_models 폴더의 모델·메타데이터·OOF 파일이 필요하다. 예측만 실행할 때는 이전 폴더가 필요하지 않다.

```bash
python run_experiment.py --data /absolute/path/to/data.csv \
  --weighted-source /absolute/path/to/hierarchical_tail_weighting \
  --gate-source /absolute/path/to/hierarchical_gate46 \
  --tail-source /absolute/path/to/hierarchical_tail_models
```

## 검증

OOF·선택·fold 점수 1,453개를 재계산하고 후보 선정·수용 조건을 재현했다. 원본 OOF 혼합, 새 시드의 정확한 fold 행, 0 확률 규칙 보존과 0 보정 경계 처리, 모델 해시와 현재 모델 보존을 확인했다. 학습 실행에서는 로그 공간 혼합의 동치성, 분류기·일반 회귀기 고정, 저장/재로딩 일치, 원시 입력 인터페이스를 검증했다. 단위 테스트 5개로 로그 공간 혼합·0 규칙·잘못된 비율 거부·음수 로그의 혼합 후 보정·추가 보호 규칙의 예측 기반 경계를 확인했다. balanced_model의 원시 입력 인터페이스도 별도로 검증했다.

전체 후보 점수는 screen_seed42.csv, 최종 비교는 comparison.csv, 변화율은 acceptance.csv, 선정 규칙은 protocol.json, 검증 결과는 validation.json에 있다. 원본 CSV는 ZIP에 포함하지 않았다.

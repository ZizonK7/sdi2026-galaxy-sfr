# 은하 SFR 모델 실험 기록 — 2026-11-13

지금까지 진행한 팀원 모델 실험의 코드, 비교표, 검증 기록, 학습 모델을 모았다. 목표는 **전체 로그 RMSE를 크게 훼손하지 않으면서 실제 0과 극단값(SFR>100)을 잘 맞추는 것**이다.

**이번 커밋은 실험 기록 공유이며 새 모델의 자동 채택이 아니다.** 기존 운용 모델은 [`../current_model.json`](../current_model.json)이 가리키는 A34다. 조장 폴더, 제출 노트북, 루트 README와 데이터는 변경하지 않는다.

## 먼저 볼 결과

| 후보 | 구성 | 판단 |
|---|---|---|
| [50:50 혼합](outputs/tail_blend_constrained/REPORT.md) | 계층형 HGB 분류기 + 일반 HGB + 무가중치/극단값3배 LightGBM 꼬리 예측을 로그 공간에서 50:50 혼합 | 혼합 전 모델보다 세 시드 모두 전체·SFR>100 RMSE 감소. 0 구간은 두 시드 개선, 한 시드 소폭 악화 |
| [0/양수 sigmoid 보정](outputs/gate_calibration46/REPORT.md) | 위 혼합 모델의 회귀값은 고정하고 0/양수 확률만 보정 | 직전 혼합 모델보다 0 RMSE 약2.1~2.6% 감소, 극단값 RMSE 약0.43~0.46% 증가. 혼합 전 모델과 비교하면 세 시드 모두0·극단값 개선 |

두 결과의 **비교 기준을 구분해야 한다.** 보정 모델이 직전50:50 혼합 모델보다 꼬리에서 좋아진 것은 아니다.

### 주요 모델의 절대 점수

전체는5개 fold의 log10(1+SFR) RMSE 평균, 실제0·SFR>100은 합친 OOF의 해당 구간 RMSE다. 모두 낮을수록 좋다.

| 시드 | 모델 | 전체 | 실제0 | SFR>100 |
|---:|---|---:|---:|---:|
| 42 | 무가중치 LightGBM | 0.09338403 | 0.102333 | 0.520110 |
| 42 | 50:50 혼합 | 0.09336728 | 0.102312 | 0.513193 |
| 42 | 50:50 + 0/양수 sigmoid | 0.09342094 | 0.100162 | 0.515535 |
| 7 | 무가중치 LightGBM | 0.09384352 | 0.105228 | 0.523347 |
| 7 | 50:50 혼합 | 0.09383616 | 0.105175 | 0.518486 |
| 7 | 50:50 + 0/양수 sigmoid | 0.09380571 | 0.102485 | 0.520705 |
| 99 | 무가중치 LightGBM | 0.09341216 | 0.104190 | 0.492741 |
| 99 | 50:50 혼합 | 0.09334787 | 0.104251 | 0.488371 |
| 99 | 50:50 + 0/양수 sigmoid | 0.09331735 | 0.101654 | 0.490598 |

원래 train80%의 동일한 관측치를 시드42·7·99로 재분할했다. **세 시드는 독립 테스트셋이 아니다.** 최근 실험에서 원래 holdout20% 정답은 평가하지 않았다. 극단값은209개로 적다.

## 가져왔던 조장 원본과의 동일 분할 비교

조장 커밋 `21d02b5d0c829aff67cafd89da62c3386ac7864f`의 HGB400 0/양수2단계 원본을 재현한 결과와 비교했다. 두 모델의 관측치·정답·fold 배치가 같은지 확인했다. 시드99의 조장 원본 결과는 없어 제외했다.

| 시드 | 구간 | 조장 원본 | 현재50:50 | 변화율 |
|---:|---|---:|---:|---:|
| 42 | overall | 0.095340 | 0.093367 | -2.07% |
| 42 | zero | 0.103163 | 0.102312 | -0.82% |
| 42 | tail100 | 0.567045 | 0.513193 | -9.50% |
| 7 | overall | 0.095709 | 0.093836 | -1.96% |
| 7 | zero | 0.104994 | 0.105175 | +0.17% |
| 7 | tail100 | 0.561686 | 0.518486 | -7.69% |

전체 RMSE는 두 시드 모두 약2%, SFR>100 RMSE는 약7.7~9.5% 감소했다. 실제0 개선은 시드별로 달랐다. 이 차이는46피처·계층형 분류·회귀기 교체·혼합을 모두 반영한 결과이며50:50 혼합 단독 효과가 아니다. 전체 점수는10개 fold 모두 원본보다 낮았다.

## 현재 구조와 경계

1. 첫 HGB 분류기: SFR=0 / SFR>0.
2. 두 번째 HGB 분류기: 실제 양수에서 중간(0<SFR≤10) / 꼬리(SFR>10).
3. 각 회귀기는 분류기가 예측한 구간이 아니라 **실제 정답 구간**의 학습 행으로 학습한다.
4. 회귀값은 log10(1+SFR) 공간에서 분류 확률로 혼합한다. P(SFR=0)≥0.9이면0으로 고정하고 마지막에 음수를0으로 보정한다.
5. **10은 전문가 학습·분류 경계, 100은 극단값 평가와 학습 가중치 기준**이다.

46개 입력 피처는 원본·환경34개와 성장/블랙홀 잔차·상호작용·이웃 은하 피처12개다. 잔차 피처의 기준 곡선은 각 학습 fold에서만 학습했다. 제공된 기준 카탈로그의 위치·속도 등 입력으로 환경 피처를 계산하며 SFR 정답을 환경 피처에 쓰지 않는다.

## 실험 순서

| 폴더 | 내용 |
|---|---|
| [log_sfr_reevaluation](outputs/log_sfr_reevaluation/) | 로그 공간 기준 재평가 |
| [a_log_development](outputs/a_log_development/), [a_component_comparison](outputs/a_component_comparison/), [a_feature_comparison](outputs/a_feature_comparison/) | 초기 A 개선·부품·피처 비교 |
| [a34_model_comparison](outputs/a34_model_comparison/), [a34_random_forest](outputs/a34_random_forest/), [a34_feature_weighting](outputs/a34_feature_weighting/) | 회귀기/분류기 교체, RandomForest, 피처 강조 비교 |
| [a_zero_threshold](outputs/a_zero_threshold/), [a34_zero_gate_adopted](outputs/a34_zero_gate_adopted/) | 0.9 규칙과 기존 채택 모델의 기록 |
| [b_tail_diagnosis](outputs/b_tail_diagnosis/) | 꼬리 오차 진단 |
| [team_lead_model_21d02b5](outputs/team_lead_model_21d02b5/) | 조장 커밋21d02b5 모델의 원본·추출 코드·재현 |
| [leader_three_regime_experiment](outputs/leader_three_regime_experiment/) | 0/중간/꼬리 및 경계10·30·100 비교 |
| [mass_growth_residual_experiment](outputs/mass_growth_residual_experiment/) | 같은 질량대 대비 성장량 잔차 |
| [feature_search](outputs/feature_search/) | 피처 추가·제거와46피처 선택 |
| [hierarchical_gate46](outputs/hierarchical_gate46/) | 0/양수와 양수 내 중간/꼬리의 두 이진 분류기 |
| [hierarchical_tail_models](outputs/hierarchical_tail_models/) | HGB·LightGBM·CatBoost·ExtraTrees 꼬리 회귀 비교 |
| [hierarchical_tail_weighting](outputs/hierarchical_tail_weighting/) | 실제 SFR>100 학습 가중치2·3·5배 비교 |
| [tail_blend_constrained](outputs/tail_blend_constrained/) | 전체 오차 증가를 제한한 꼬리 회귀 혼합 |
| [gate_calibration46](outputs/gate_calibration46/) | 내부3-fold로 sigmoid/isotonic 보정,0/양수·중간/꼬리·둘 다 비교 |

## 후보 불러오기

후보별 `requirements.txt`를 사용한다. 최근 후보는 Python3.12, scikit-learn1.9.1, LightGBM4.7.0 환경에서 검증했다. **신뢰할 수 있는 이 저장소의 joblib만 불러온다.**

- 전체·꼬리 균형 후보: [`outputs/tail_blend_constrained/balanced_model.joblib`](outputs/tail_blend_constrained/balanced_model.joblib)
- 0 개선 절충 후보: [`outputs/gate_calibration46/sigmoid_positive.joblib`](outputs/gate_calibration46/sigmoid_positive.joblib)

`tail_blend_constrained/recommended_model.joblib`는 추가한 엄격한 “모든 시드에서0 구간 증가 없음” 조건이 실패했을 때 보존한 기존 모델이다. **새50:50 혼합 후보의 파일 이름은 balanced_model.joblib**다.

아래는 이 폴더를 작업 디렉터리로 삼은 별도 Python 프로세스에서0/양수 보정 후보를 사용하는 예시다. 모델 폴더의 Python 모듈을 함께 보관해야 한다.

```python
from pathlib import Path
import sys
import joblib
import pandas as pd

bundle = Path('outputs/gate_calibration46').resolve()
sys.path.insert(0, str(bundle))
from feature_pipeline import predict_sfr

model = joblib.load(bundle / 'sigmoid_positive.joblib')
reference = pd.read_csv('/path/to/reference.csv').drop(columns='sfr_linear', errors='ignore')
query = pd.read_csv('/path/to/input.csv').drop(columns='sfr_linear', errors='ignore')
prediction = predict_sfr(query, model, reference)
```

## 저장 범위와 재현

- 코드·노트북·결과표·설정·메타데이터·검증 기록·학습 모델을 포함했다. `work/`에는 실험 준비·패키징 스크립트도 보관했다.
- 원본 CSV를 중복 업로드하지 않았다. 저장소 루트의 원본 데이터 이름을 그대로 사용한다.
- 약446MB의 행별 OOF/회귀 성분 `.npz`, 중복 ZIP, 실행 환경, 캐시는 Git에서 제외했다. **요약·fold별 점수는 포함했다.** 제외한49개 `.npz`의 경로·크기·SHA256은 [OMITTED_ARTIFACTS.json](OMITTED_ARTIFACTS.json)에 있다.
- 따라서 이전 OOF를 재사용하는 실험 스크립트를 실행하려면 해당 `.npz`와 캐시를 원 실험 작업공간에서 복원하거나 선행 실험부터 재생성해야 한다. 이 저장소 묶음만으로 과거 OOF 검증을 즉시 재실행할 수 있다고 주장하지 않는다. **저장 모델로 새 데이터를 예측하는 데는 OOF 파일이 필요하지 않다.**
- 각 실험의 `REPORT.md`에 실행 조건과 의존성이 있다. 일부 보조 스크립트·원본 메타데이터에는 원 작업공간의 절대 경로가 감사 기록으로 남아 있다. 재실행할 때 `--data`·캐시 경로를 현재 환경에 맞춘다.
- `outputs/current_model.json`은 원 실험의 로컬 포인터 스냅샷이다. 공유 저장소의 운용 포인터는 [`../current_model.json`](../current_model.json)이며 이번 푸시에서는 변경하지 않았다.
- `LOCAL_SHA256SUMS`는 OOF까지 포함했던 원 로컬 패키지의 체크섬이다. 이 Git 묶음에 실제로 포함된 파일은 새 `SHA256SUMS`로 확인한다. 원본 복사 목록은 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)에 기록했다.

검증 기록은 당시 실행한 실험의 결과다. 최근 보정 실험은 내부/외부 분리45개와 점수2,457개 재계산, 저장/재로딩·원시 입력 예측 테스트를 통과했다. 이번 푸시 단계에서는 파일 무결성과 업로드 범위를 추가 확인했다.

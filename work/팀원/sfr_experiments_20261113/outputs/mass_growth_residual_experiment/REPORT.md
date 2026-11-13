# 같은 질량대 대비 성장량 잔차 추가 실험

## 결과

| 모델 | 전체 로그 RMSE 전 → 후 | SFR>100 로그 RMSE 전 → 후 | 실제 0 로그 RMSE 전 → 후 | 전체 개선 fold | 꼬리 개선 fold |
|---|---:|---:|---:|---:|---:|
| 조장 0/양수 | 0.095340 → 0.095313 | 0.567045 → 0.563491 | 0.103163 → 0.104357 | 3/5 | 2/5 |
| 3분류 T=10 | 0.095962 → 0.095998 | 0.587391 → 0.581419 | 0.107060 → 0.105781 | 1/5 | 2/5 |
| 기존 0 확률 방식 + 3분류 T=10 | 0.095483 → 0.095560 | 0.589265 → 0.577249 | 0.103023 → 0.103763 | 2/5 | 3/5 |


전체는 5-fold RMSE 평균이며, 실제 0·SFR>100 구간은 모든 OOF 행을 합쳐 계산한 로그 RMSE다.
모든 지표는 log10(1+SFR) 공간이다. 작은 값이 좋다. 극단 구간 SFR>100은 209개다.
전체 pooled OOF RMSE와 SFR>10·>30·>100 구간, 로그 편향 및 선형 RMSE는 `summary.csv`에 있다.

추가 후 전체 점수가 가장 낮은 구성은 **조장 0/양수**(0.095313351)다.
피처 추가 전 조장 원본(0.095340038)보다 낮았다.
이번 비교는 이전에도 사용한 CV에서 수행한 파생 변수 실험이며 독립 시험 성능이나 통계적 유의성을 의미하지 않는다.
결과를 보고 bin 수나 꼬리 경계를 추가 탐색하지 않았다. 현재 채택 모델은 바꾸지 않았다.

## 변수의 정확한 정의

`growth = log10_stellar_mass - log10_main_prog_stellar_mass`

`stellar_growth_residual_at_mass = growth - expected_growth(current_stellar_mass)`

각 CV 학습 fold의 현재 항성 질량을 20개 분위 구간으로 나눈다.
구간별 대표 질량은 중앙값, 기대 성장량은 평균이다. 대표 질량 사이를 선형 보간하며 양 끝 밖에서는 끝값을 유지한다.
검증 행은 이 고정된 기준선을 적용받는다. 같은 질량대의 기대 성장량보다 크면 잔차가 양수다.
분위 경계 중복은 제거하고 빈 구간은 사용하지 않는다.

여기서 성장량은 두 시점 항성 질량의 **로그 비율**이며 시간으로 나눈 성장률이 아니다.
원래 존재하는 질량·질량 차이 입력을 재표현한 피처이며 새로운 관측 정보가 추가된 것은 아니다.
전구체 질량의 센티널 후보도 기존 모델과 동일하게 유지했으므로 물리적 성장으로 단정하지 않는다.

## 비교와 누출 방지

- 입력 34→35개로 잔차 변수 하나만 추가했다. 원래 질량·성장 피처도 유지했다.
- 조장 HGB400 이진 두 단계, 3분류 T=10, 기존 이진 0 확률을 사용하는 3분류 T=10에 동일하게 적용했다.
- 각 구성의 분류기와 회귀기에 모두 35개 피처를 제공했다. 기존 0 확률 방식의 이진 분류기도 새 피처로 학습한다.
- HGB 학습률 .08, 400회, 잎 31개, 조기 종료 없음, 0 출력 문턱 .9를 고정했다.
- 기존 80:20 분할에서 학습 166,749행만 사용했다. KFold(5, shuffle=True, random_state=42).
- 기대 성장 곡선은 **각 fold 학습 입력만**으로 적합한다. 검증 입력과 SFR 타깃은 곡선 적합에 사용하지 않는다.
- 환경 피처만 기존 FAQ 계약대로 같은 전체 참고 은하 집합의 입력 기하로 계산했다.
- 피처 추가 전 OOF는 직전 실험의 캐시를 사용했고, 데이터 해시·행 번호·실제값·fold 및 모델 모듈이 일치하는지 확인했다.
- 원래 20% 검증셋의 성능은 이번에 평가하지 않았다.

| 모델 | 0 재현율 전 → 후 | 0 정밀도 전 → 후 | 실제 양수인데 0 출력 전 → 후 |
|---|---:|---:|---:|
| 조장 0/양수 | 80.48% → 80.29% | 98.18% → 98.20% | 215 → 212 |
| 3분류 T=10 | 82.73% → 82.92% | 97.98% → 97.94% | 246 → 252 |
| 기존 0 확률 방식 + 3분류 T=10 | 80.48% → 80.28% | 98.18% → 98.20% | 215 → 212 |


## 꼬리 분류 진단

직접 3분류 T=10에서 SFR>100 은하의 평균 꼬리 확률은
0.877437 → 0.882995다.
꼬리 회귀기만 정답 구간을 알고 적용했을 때의 진단 RMSE는
0.357422 → 0.363151다.
이 진단값은 실제 배포 가능한 점수가 아니며 최종 혼합 점수와 구분한다.
자세한 분류 진단은 `tail_gate_diagnostics.csv`에 있다.

## 저장 모델과 실행

세 구성의 새 모델을 원래 학습 166,749행으로 다시 학습해 저장했다.
기대 성장량 변환기도 같은 학습 집합에 적합한 상태로 모델 파일에 포함돼 있다.
예측할 때 새 배치로 기준선을 다시 적합하면 안 된다.

이 폴더를 작업 디렉터리로 하고 Python 3.12와 `requirements.txt` 환경에서 사용한다.

```python
import joblib
import pandas as pd
from leader_model import predict_sfr

model = joblib.load('leader_two_stage_residual.joblib')
reference = pd.read_csv('/path/to/Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv')
reference = reference.drop(columns='sfr_linear')
query = reference.iloc[:100].copy()
prediction = predict_sfr(query, model, reference)
```

`model.predict(X34)`도 지원한다. 원래 34개 피처를 주면 저장된 변환기가 35번째 피처를 자동 생성한다.
`leader_model.py`, `three_regime.py`, `zero_preserved.py`, `growth_residual.py`가 모듈 경로에 있어야 한다.
환경 기하용 참고 은하 집합은 기존 모델처럼 전체 고유 은하의 고정 합집합을 사용한다.

재현 명령:

```bash
python test_growth_residual.py
python run_experiment.py --data '/path/to/data.csv' --baseline-dir '/path/to/leader_three_regime_experiment'
```

baseline-dir에는 직전 실험의 모델 모듈과 OOF 캐시가 필요하다.
곡선의 배치 독립성, 범위 밖 처리, 동률 질량과 SFR 미사용 테스트를 통과했다.
모든 OOF 행의 1회 커버리지, 모델 저장/재로딩 일치, 원본 입력 표 예측 인터페이스도 확인했다.

## 파일

- `summary.csv`, `changes.csv`, `paired_differences.csv`, `fold_results.csv`: 비교 결과
- `oof_predictions.npz`: 전/후 예측, 실제값, fold, 원본 행 번호, fold 밖에서 계산한 잔차
- `fold_feature_curves.json`: fold별 질량-기대성장 기준선과 학습 행 수
- `*_residual.joblib`: 변환기를 포함한 학습 완료 모델
- `protocol.json`, `metadata.json`: 실행 전 조건과 결과/검증 기록

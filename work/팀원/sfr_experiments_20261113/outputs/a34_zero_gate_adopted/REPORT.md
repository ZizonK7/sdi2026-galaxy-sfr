# 채택 모델: P(SFR=0) ≥ 0.9이면 정확히 0 출력

사용자 요청에 따라 이 규칙을 현재 모델의 기본 출력 정책으로 채택했다.

- 분류: LightGBM / 일반 회귀: LightGBM / 꼬리 회귀: HGB
- 입력 34개, T=10, 로그 혼합 유지. 밀도/BH 추가 우대 없음.
- P(0)≥0.9인 경우 정확히 0. P(0)<0.9이면 기존 예측과 동일.
- 기존 학습 모델을 감싸서 저장했다. 재학습하지 않았다.

## 변경 전후 같은 5-fold OOF 비교

| 지표 | 적용 전 | 적용 후 |
|---|---:|---:|
| log10(1+SFR) RMSE | 0.095191101 | 0.095231152 |

총 12,124개에 0을 출력했다. 이 중 실제 0은 11,888개,
실제 양수는 236개다. 강제0 정밀도는 98.05%,
실제0 재현율은 82.34%다. 개선 fold는 0/5다.
구간별·fold별 수치는 `scores.csv`, `fold_scores.csv`에 있다. holdout은 평가하지 않았다.
이전의 0.095152는 fold마다 부품을 선택한 절차의 점수이며, 여기의 고정 구성 점수와 다르다.

## 사용

이 폴더를 Python 모듈 검색 경로에 두고 다음처럼 불러온다.

```python
import joblib
from features import build_feature_sets
model = joblib.load('A34_zero_gate_090.joblib')
# 환경 계산에는 같은 상자의 고유 참고 은하 합집합을 고정해서 사용한다.
X34 = build_feature_sets(catalogue)['A34_both']
predictions = model.predict(X34.to_numpy())
# 필요하면 기존 예측과 확률도 확인할 수 있다.
predictions, soft_predictions, p_zero = model.predict_with_details(X34.to_numpy())
```

`model.predict()`에 0 규칙이 기본 포함되어 있으므로 별도 후처리는 필요 없다.
작은 배치별로 환경 변수를 다시 계산하지 말고 고정 참고 집합에서 계산한 행을 선택한다.
`models.py`, `zero_gate.py` 및 `requirements.txt`의 환경이 필요하다.
macOS LightGBM의 OpenMP 라이브러리 경로는 기존 실행과 동일하게 설정한다.

## 확인

경계값 0.9 포함 여부와 미적용 행 보존 테스트를 통과했고, 실제 입력 3,000행에서 저장/재로딩 후
기준 이상은 정확히0, 기준 미만은 이전 예측과 완전히 같음을 확인했다. 기존 모델 파일의 SHA256도 변하지 않았다.
이 폴더의 모델을 현재 채택본으로 지정했으며 `outputs/current_model.json`에 경로를 기록했다.
이전 실험 모델·결과는 비교 이력으로 보존했다. GitHub는 변경하지 않았다.

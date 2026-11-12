# 현재 채택 모델: A34 + 0 확률 임계값 0.9

분류 LightGBM, 일반 회귀 LightGBM, 꼬리 회귀 HGB를 사용한다. 입력은 원본 13개 + 환경 17개 + 질량 차이 4개다.
`P(SFR=0) >= 0.9`이면 정확히 0을 출력하고, 나머지는 기존 로그 공간 혼합 예측을 유지한다.

5-fold OOF의 `log10(1+SFR)` RMSE는 적용 전 **0.095191101**, 적용 후 **0.095231152**다.
점수 개선에 의한 선택이 아니라 사용자 요청으로 채택했다. 원래 20% holdout은 평가하지 않았다.
세부 결과는 [REPORT.md](REPORT.md), 구간별 결과는 [scores.csv](scores.csv)에 있다.

## 실행

Python 3.12에서 이 폴더를 작업 디렉터리로 사용한다.

```bash
python -m pip install -r requirements.txt
python test_zero_gate.py
python evaluate_oof.py
```

```python
from pathlib import Path
import joblib
import pandas as pd
from features import build_feature_sets

repo = Path.cwd().parents[2]  # 작업 디렉터리: work/팀원/a34_zero_gate_adopted
catalogue = pd.read_csv(repo / 'Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv')
X34 = build_feature_sets(catalogue)['A34_both']
model = joblib.load('A34_zero_gate_090.joblib')
prediction, original_prediction, p_zero = model.predict_with_details(X34.to_numpy())
```

환경 피처는 학습 때와 같은 고정된 전체 참고 집합에서 계산한 뒤 필요한 행을 선택한다.
작은 예측 배치만으로 이웃 밀도 등을 다시 계산하면 입력이 달라진다.
위 예시는 모델 호출 방법이며, 학습 행의 예측을 검증 점수로 사용하지 않는다.
macOS에서 LightGBM import가 실패하면 OpenMP 런타임을 설치하거나 해당 라이브러리 경로를 설정해야 한다.

`models.py`와 `zero_gate.py`는 joblib 역직렬화에 필요하다. 모델 파일의 SHA256과 버전은 `metadata.json`에 기록했다.
원래 학습된 회귀기와 분류기를 재학습하지 않고 출력 규칙만 추가했다.
`leader_environment.py`는 조장 브랜치의 커밋 `6bc3988dd7effb7694192ce059f515fa496fa499`에서 사용한 환경 피처를 기반으로 한다.

`oof_predictions.npz`에는 원본 데이터 행 번호, fold, 실제값, 기존 예측, 적용 후 예측, 0 확률이 들어 있다.
`evaluate_oof.py`로 저장된 OOF 결과의 규칙과 점수를 재확인할 수 있다.

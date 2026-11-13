# 피처 탐색 결과: best_joint_add_all

학습 완료 모델: `best_feature_model.joblib`.
구조는 0 / 0<SFR≤10 / SFR>10의 HGB 다중분류기와 실제 일반·꼬리 구간별 HGB 로그 회귀기다.
모든 학습기는 학습률 .08, 400회, 잎 최대31, 조기 종료 없음이다.
P0≥.9이면 0, 나머지는 두 회귀기의 로그 예측을 확률로 섞어 선형 SFR로 반환한다.

- 시드42 전체5-fold 평균 로그 RMSE: 기존3분류 0.095962 → 선택 0.093973
- 고정 후 시드7: 기존3분류 0.095791 → 선택 0.094544
- 시드7의 현재 채택A: 0.095710
- 모델별 피처 수: 3분류기46개, 일반 회귀기46개, 꼬리 회귀기46개

이번 후보 범위의 최저 CV 구성이다. 같은 관측값의 분할을 바꿔 확인했으며 독립 시험이 아니다.
현재 채택 파일이나 GitHub는 변경하지 않았다.

[상세 보고서](REPORT.md), [최종 피처 목록](selected_features.json), [비교 결과](confirmation_summary.csv).

이 폴더를 작업 디렉터리로 하고 Python3.12와 requirements.txt 환경에서 사용한다.

```python
import joblib
import pandas as pd
from feature_pipeline import predict_sfr

model = joblib.load('best_feature_model.joblib')
reference = pd.read_csv('/path/to/Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv')
reference = reference.drop(columns='sfr_linear')
prediction = predict_sfr(reference.iloc[:100].copy(), model, reference)
```

전체 고유 은하 참고 집합을 고정해 환경·동반 은하 변수를 계산한다.
질량별 평균 곡선은 저장 모델에 포함되며 예측 배치에서 다시 학습하지 않는다.
원본 열이 학습기 입력에서 제외돼도 파생 변수 계산에는 필요할 수 있으므로 원본13개를 모두 전달한다.

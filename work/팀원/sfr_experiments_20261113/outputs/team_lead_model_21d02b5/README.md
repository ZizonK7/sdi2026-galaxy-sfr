# 조장 브랜치 최신 모델 가져오기

출처: `ZizonK7/sdi2026-galaxy-sfr`, `codex/team-lead-work`, 커밋
[`21d02b5d0c829aff67cafd89da62c3386ac7864f`](https://github.com/ZizonK7/sdi2026-galaxy-sfr/commit/21d02b5d0c829aff67cafd89da62c3386ac7864f).

최신 최종 모델은 **TwoStageLogSFR: HGB 이진 분류기 + HGB 양수 회귀기**다.
각각 학습률 0.08, 최대 잎 31개, 400회 학습이며 조기 종료를 사용하지 않는다.
원본 13개 + 환경 17개 + 질량 차이 4개, 총 34개 피처를 두 모델 모두 사용한다.

- 분류기: 전체 학습 행의 실제 `SFR > 0` 여부를 학습한다.
- 회귀기: 실제 `SFR > 0`인 학습 행만으로 `log10(SFR+1)`을 학습한다. 일반/꼬리를 분리하지 않는다.
- 예측: 양수 확률 × 양수 회귀기의 로그 예측을 계산하고 `10**z - 1`로 되돌려 0 이상으로 자른다.
- `P(SFR=0) >= 0.9`이면 정확히 0을 반환한다.

## 가져온 파일의 의미

`DataHub_701104_final.ipynb`는 해당 커밋의 원본 노트북이며 Git blob SHA를 확인했다.
`leader_model.py`는 원본의 환경/질량 피처, 모델 정의와 예측 인터페이스를 추출한 모듈이다.
노트북 전역 변수에 의존하던 `predict_sfr`의 모델과 참고 표는 명시적인 인자로 바꿨다.
환경 계산 전 원본과 같은 사전식 행 정렬을 유지한다.

최신 모델의 학습 완료 파일은 원격 브랜치에 없었다. `team_lead_hgb400.joblib`는 가져온 코드를
같은 데이터, 같은 80:20 층화 분할, 시드 42로 **로컬에서 다시 학습한 모델**이다.
학습 행은 166,749개이며 검증 행 41,688개의 타깃은 모델 학습에 사용하지 않았다.
로컬 재현 로그 RMSE는 **0.0953135093**으로 원본 보고값 **0.095314**와 소수점 6자리까지 일치했다.
정확한 0 출력 2,893개, 그 중 실제 양수 51개, 0 정밀도 98.24%, 실제 0 재현율 78.73%도 재현됐다.
설정과 실제 재현 점수는 `metadata.json`, 검증 예측은 `validation_predictions.npz`에 있다.

`source_README.md`는 조장의 최신 변경 설명이다. `comparison_report_historical.md`는 이전
LightGBM 선택 당시 문서이므로 최종 모델의 판단 기준으로 사용하지 않는다.
최신 노트북과 커밋에서 HGB 400회로 되돌렸음을 확인했다.

## 사용 방법

Python 3.12에서 이 폴더를 작업 디렉터리로 사용한다.

```bash
python -m pip install -r requirements.txt
```

```python
import joblib
import pandas as pd
from leader_model import predict_sfr

model = joblib.load('team_lead_hgb400.joblib')
reference = pd.read_csv('/path/to/Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv')
reference = reference.drop(columns='sfr_linear')
query = reference.iloc[:100].copy()
predictions = predict_sfr(query, model, reference)
```

참고 은하 집합은 학습 때와 동일한 전체 은하 집합으로 고정한다. 새 은하를 예측할 때도
전체 질의 은하를 포함한 고정 합집합을 사용한다. 같은 은하가 겹칠 수 있으므로 `subhalo_id`는
중복 확인에 필요하지만 모델 피처로 사용하지 않는다. ID 충돌·중복·누락은 원본 규칙대로 거부한다.
34개 피처를 이미 계산했다면 올바른 열 이름을 가진 DataFrame으로 `model.predict(X34)`를 호출한다.

재학습:

```bash
python train_imported.py --data '/path/to/Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv'
```

원본 노트북 보고값 0.095314는 **20% 공통 검증셋 점수**다.
우리 A의 0.095231은 **학습 내부 5-fold OOF 점수**이므로 숫자만으로 우열을 비교할 수 없다.
현재 채택한 A 모델과 `outputs/current_model.json`은 변경하지 않았다.

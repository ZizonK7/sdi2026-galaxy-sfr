# A34의 HGB 부품을 다른 모델로 교체한 결과

가장 낮은 OOF 로그 RMSE는 **내부 검증으로 고른 부품 조합: 0.095152**, 기존 대비 **1.78%** 개선이다.
각 fold의 내부 검증만으로 고른 부품 조합의 OOF 점수는 **0.095152**, 개선율은 **1.78%**다.

## 전체 결과

| 교체 대상 | 로그 RMSE ↓ | 개선율 (%) | 개선 fold / 5 |
|---|---:|---:|---:|
| 내부 검증으로 고른 부품 조합 | 0.095152 | 1.775556 | 5.000000 |
| 분류 → LightGBM | 0.095591 | 1.321949 | 5.000000 |
| 일반 회귀 → CatBoost | 0.096437 | 0.448101 | 5.000000 |
| 일반 회귀 → LightGBM | 0.096442 | 0.443357 | 5.000000 |
| 꼬리 회귀 → LightGBM | 0.096828 | 0.044790 | 3.000000 |
| 기존 HGB / HGB / HGB | 0.096872 | 0.000000 | 0.000000 |
| 꼬리 회귀 → CatBoost | 0.096989 | -0.121729 | 1.000000 |
| 꼬리 회귀 → ExtraTrees | 0.097466 | -0.613312 | 0.000000 |
| 분류 → CatBoost | 0.098643 | -1.828480 | 0.000000 |

모든 점수는 동일한 166,749개 학습 은하의 5-fold OOF 예측에서 계산한 log10(1+SFR) RMSE다.
가장 좋은 개별 후보의 이름은 이 표를 본 뒤 판단한 것이므로, 전체 탐색에 대한 최종 독립 검증을 뜻하지 않는다.
조합 행은 각 외부 fold를 열기 전에 내부 검증에서 구성까지 정한 선택 절차의 평가다.
기존 기준의 HGB 설정은 고정하고 대체 모델만 작은 범위로 튜닝했으므로 알고리즘 자체의 절대 우열을 입증하는 비교는 아니다.

## 유지한 구조와 튜닝

- 입력 34개: 원래 13개 + 환경 17개 + 질량 관계 4개. 환경 계산은 이전 실험과 같은 전체 라벨 없는 참고 은하 집합을 사용.
- 부류 0 / 0<SFR≤10 / SFR>10. 분류기 하나와 실제 부류로 학습하는 회귀기 둘 유지.
- 회귀 목표 ln(1+SFR), 로그 공간 확률 혼합, 음수 로그 예측의 하한 0 유지. P(0)≥.9 강제 0 규칙은 사용하지 않았다.
- 외부 검증: 기존 KFold(5, shuffle=True, random_state=42). 각 외부 학습 fold를 다시 80/20으로 나누고 실제 3부류로 층화(seed=1000+fold).
- 내부 검증에서 **최종 혼합 로그 RMSE**로 설정을 선택했다. 선택 뒤 외부 학습 fold 전체로 재학습하고 외부 검증 fold를 평가했다.
- HGB: learning_rate=.08, max_iter=150, max_leaf_nodes=31, random_state=42. 기존 조기 종료 유지.
- CatBoost: (300회, depth6) / (600회, depth8), learning_rate=.05, l2_leaf_reg=5. 분류 MultiClass, 회귀 RMSE.
- LightGBM: (300회, 31 leaves) / (600회, 63 leaves), learning_rate=.05, reg_lambda=5, min_child_samples=20. 분류 multiclass, 회귀 squared error.
- ExtraTrees 꼬리: 300 trees, min_samples_leaf 3/5/10, max_features=1.0.
- 각 부품에서 기존 HGB 대비 내부 점수가 좋아지는 대안만 조합했다. 조합의 내부 점수가 기준보다 나빠지면 가장 좋은 단일 교체로 되돌리는 규칙을 미리 고정했다.
- 외부 fold의 타깃은 설정·조합 선택에 사용하지 않았다. 남겨둔 20% holdout은 평가하지 않았다.

## fold별 결과

| 교체 대상 | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| 기존 HGB / HGB / HGB | 0.097830 | 0.096479 | 0.096882 | 0.095315 | 0.097829 |
| 내부 검증으로 고른 부품 조합 | 0.095817 | 0.094461 | 0.095611 | 0.093838 | 0.096012 |
| 분류 → CatBoost | 0.099181 | 0.098042 | 0.098450 | 0.097533 | 0.099989 |
| 일반 회귀 → CatBoost | 0.097381 | 0.096028 | 0.096579 | 0.094832 | 0.097344 |
| 꼬리 회귀 → CatBoost | 0.097922 | 0.096913 | 0.096936 | 0.095338 | 0.097816 |
| 꼬리 회귀 → ExtraTrees | 0.098655 | 0.097275 | 0.097149 | 0.095738 | 0.098483 |
| 분류 → LightGBM | 0.096144 | 0.095033 | 0.095782 | 0.094336 | 0.096642 |
| 일반 회귀 → LightGBM | 0.097344 | 0.095933 | 0.096633 | 0.094797 | 0.097477 |
| 꼬리 회귀 → LightGBM | 0.097695 | 0.096414 | 0.096889 | 0.095426 | 0.097697 |

## 내부 검증이 선택한 조합

| 외부 fold | gate | mid | tail | inner_baseline | inner_combo | fallback |
|---|---:|---:|---:|---:|---:|---:|
| 1 | lgb | cat | cat | 0.097961 | 0.096643 | False |
| 2 | lgb | lgb | lgb | 0.096604 | 0.095833 | False |
| 3 | lgb | lgb | hgb | 0.098746 | 0.096617 | False |
| 4 | lgb | cat | hgb | 0.097582 | 0.095236 | False |
| 5 | lgb | cat | lgb | 0.098369 | 0.097135 | False |

fold마다 알고리즘이나 설정이 다를 수 있다. 조합 행을 하나의 고정된 설정이 5회 검증된 점수라고 해석하지 않는다.
세부 후보 설정과 내부 점수는 `selections.json`, 학습 기록은 `fit_records.json`에 있다.

## SFR 구간별 결과

| 교체 대상 | 0<SFR<=3 | 10<SFR<=30 | 30<SFR<=100 | 3<SFR<=10 | SFR=0 | SFR>100 |
|---|---:|---:|---:|---:|---:|---:|
| 기존 HGB / HGB / HGB | 0.081963 | 0.128401 | 0.185919 | 0.104274 | 0.111915 | 0.590043 |
| 내부 검증으로 고른 부품 조합 | 0.081342 | 0.125365 | 0.178804 | 0.102926 | 0.106734 | 0.565380 |
| 분류 → CatBoost | 0.082371 | 0.129865 | 0.189585 | 0.105027 | 0.123961 | 0.581550 |
| 일반 회귀 → CatBoost | 0.081603 | 0.127214 | 0.185815 | 0.103708 | 0.111634 | 0.590720 |
| 꼬리 회귀 → CatBoost | 0.081950 | 0.127889 | 0.184071 | 0.104340 | 0.111907 | 0.614650 |
| 꼬리 회귀 → ExtraTrees | 0.082071 | 0.127440 | 0.189156 | 0.104695 | 0.112133 | 0.642929 |
| 분류 → LightGBM | 0.081690 | 0.126985 | 0.179210 | 0.103516 | 0.107213 | 0.557344 |
| 일반 회귀 → LightGBM | 0.081601 | 0.126830 | 0.185829 | 0.103840 | 0.111630 | 0.591481 |
| 꼬리 회귀 → LightGBM | 0.081975 | 0.128164 | 0.185412 | 0.104160 | 0.111932 | 0.590464 |

전체 1위 후보에서 기존보다 악화된 구간: 없음.
극단 꼬리는 표본 209개이므로 전체 개선과 별개로 해석한다.

## 학습 완료 모델

`A34_selected.joblib`은 원래 학습 데이터 166,749개 안에서 seed2026의 내부 분할로 조합을 선택하고,
그 사양으로 학습 데이터 전체에 재학습한 모델이다. holdout 타깃은 사용하지 않았다.
이 파일은 내부 선택 절차를 실행한 결과이며, OOF 표의 가장 좋은 단일 행을 그대로 저장한 파일이라는 뜻은 아니다.

선택 사양:

```json
{
  "gate": {
    "family": "lgb",
    "params": {
      "n_estimators": 600,
      "num_leaves": 63
    }
  },
  "mid": {
    "family": "lgb",
    "params": {
      "n_estimators": 600,
      "num_leaves": 63
    }
  },
  "tail": {
    "family": "hgb",
    "params": {}
  }
}
```

예측 예시(이 폴더에서 실행):

```python
import joblib
from features import build_feature_sets
model = joblib.load('A34_selected.joblib')
# catalogue: 같은 상자의 고유 참고 은하 합집합. 입력 열만 필요하다.
X34 = build_feature_sets(catalogue)['A34_both']
predictions = model.predict(X34.to_numpy())
```

환경을 작은 예측 배치마다 다시 계산하면 이웃 수와 FoF가 달라진다. 참고 은하 집합을 고정한 뒤 필요한 행의 예측을 골라야 한다.
직렬화 모델은 이 폴더의 `models.py`와 같은 라이브러리 버전으로 불러온다.

## 검증 및 재현

- 세 테스트 통과: 로그 혼합식, 대체 모델의 결측값/실제 부류 학습, 내부 검증 행의 학습 제외.
- 기존 A34 OOF를 1e-9 허용 오차 안에서 재현했다.
- 저장 예측의 전체·구간·fold 점수 및 저장 부품으로 만든 조합을 별도 재계산해 일치 확인.
- 재학습 모델 저장/불러오기 전후 예측 일치. 학습 경고 기록 0개.
- 총 실행 1178.5초. 데이터 Git blob SHA `06479cedd323253f08af9983c73f395cc02d3774`.
- 실행 환경은 `requirements.txt`와 `metadata.json`에 기록했다. GitHub는 변경하지 않았다.

```bash
python -m pip install -r requirements.txt
python test_models.py
python run_experiment.py --data '/path/to/Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv' --reference baseline_reference.npz --out rerun --fit-final
```

macOS에서 libomp 로딩 오류가 나면 OpenMP 런타임 경로가 필요하다. 이 실행에서는 설치된 scikit-learn의 `.dylibs` 디렉터리를 `DYLD_LIBRARY_PATH`로 지정했다.
세부 API는 [LightGBM 공식 문서](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html)와 [CatBoost 공식 문서](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor)를 참고했다.

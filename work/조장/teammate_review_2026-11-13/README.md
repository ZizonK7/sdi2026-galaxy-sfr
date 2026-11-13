# 팀원 50:50 꼬리 모델 검토 (2026-11-13)

팀원 실험(`work/팀원/sfr_experiments_20261113/`, 커밋 `39326e7`·`c52c5e4`)의 최종 후보가 무엇 때문에 좋아졌는지 확인했다.
팀원 코드(`outputs/tail_blend_constrained/`의 `feature_pipeline.py` 등)를 그대로 불러 같은 fold에서 세 모델을 다시 학습했다.

| 모델 | CV 시드 42 | CV 시드 7 | holdout 20% | holdout SFR > 100 |
|---|---:|---:|---:|---:|
| A. 조장 2단계, 34개 (`21d02b5`) | 0.095340 | 0.095709 | 0.095314 | 0.614 |
| B. 같은 2단계 + 팀원 피처 46개 | 0.093387 | 0.093771 | **0.094063** | **0.586** |
| C. 팀원 최종 (계층형 분류 + 꼬리 LightGBM 50:50) | 0.093367 | 0.093836 | 0.094205 | 0.638 |

CV는 학습 80% 안의 5-fold 평균 로그 RMSE(log10(SFR + 1)), holdout은 공통 검증 20%다. 팀원은 holdout을 쓰지 않았다.

- 팀원 숫자(C)와 조장 숫자(A)가 그대로 재현됐다.
- **B − A: −0.0019 dex, 10/10 fold 개선.** holdout −0.00125, 부트스트랩 95% 구간 [−0.0022, −0.0003].
- **C − B: +0.00002 dex로 사실상 같다.** 꼬리(SFR > 100)는 CV 두 시드와 holdout 모두 B가 낫다.
- 결론: 개선은 **추가 피처 12개**에서 오고, 꼬리 단계 분리·혼합은 같은 피처의 2단계보다 낫지 않다. README의 기준(전체를 훼손하지 않으며 꼬리를 더 맞힌다)은 단계를 늘리지 않고 피처만으로 충족됐다. 최종 모델을 B로 바꿨다.

재현(저장소 루트에서, 팀원 코드 폴더와 데이터 경로를 넘긴다):

```bash
python work/조장/teammate_review_2026-11-13/review_compare.py "work/팀원/sfr_experiments_20261113/outputs/tail_blend_constrained" "Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv" work/조장/teammate_review_2026-11-13
```

`review_boot.py`는 같은 인자로 holdout의 A·B 차이와 구간별 기여를 계산한다. lightgbm 4.7.0이 필요하다(C 재현용). 4코어에서 약 5분.

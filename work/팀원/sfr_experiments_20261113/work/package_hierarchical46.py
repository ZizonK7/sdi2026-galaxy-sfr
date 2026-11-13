from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[1]/'outputs/hierarchical_gate46'
summary=pd.read_csv(root/'summary.csv')
pairs=pd.read_csv(root/'paired_differences.csv')
meta=json.loads((root/'metadata.json').read_text())
names={'baseline':'기존46개 직접3분류','zero_gate_only':'0 확률만 이진 분류기로 교체','hierarchical':'0/양수 → 일반/꼬리 이진 분류'}
table='| 시드 | 모델 | 전체 로그 RMSE | 실제0 로그 RMSE | SFR>100 로그 RMSE | 0 재현율 | 0 정밀도 |\n|---|---|---:|---:|---:|---:|---:|\n'
for _,r in summary.iterrows():
    table+=f'| {r.seed} | {names[r.model]} | {r.mean_fold_rmse:.6f} | {r.zero_log_rmse:.6f} | {r.tail100_log_rmse:.6f} | {r.zero_recall:.2%} | {r.zero_precision:.2%} |\n'
effects=[]
for seed in [42,7]:
    b=summary[(summary.seed==seed)&(summary.model=='baseline')].iloc[0]
    for name in ['zero_gate_only','hierarchical']:
        a=summary[(summary.seed==seed)&(summary.model==name)].iloc[0]
        p=pairs[(pairs.seed==seed)&(pairs.model==name)]
        effects.append(dict(seed=seed,model=name,
            overall_improvement_percent=100*(b.mean_fold_rmse-a.mean_fold_rmse)/b.mean_fold_rmse,
            zero_improvement_percent=100*(b.zero_log_rmse-a.zero_log_rmse)/b.zero_log_rmse,
            tail100_improvement_percent=100*(b.tail100_log_rmse-a.tail100_log_rmse)/b.tail100_log_rmse,
            overall_improved_folds=int((p.log_rmse_delta<0).sum()),
            zero_improved_folds=int((p.zero_log_rmse_delta<0).sum()),
            tail100_improved_folds=int((p.tail100_log_rmse_delta<0).sum())))
pd.DataFrame(effects).to_csv(root/'changes.csv',index=False)
diag='| 시드 | 모델 | 실제0인데 예측>1 | 실제양수인데 예측0 | 실제>100인데 예측<10 |\n|---|---|---:|---:|---:|\n'
for _,r in summary.iterrows():
    diag+=f'| {r.seed} | {names[r.model]} | {int(r.zero_pred_above1)} | {int(r.false_zeros)} | {int(r.tail_pred_below10)} |\n'
effect_table='| 시드 | 전체 개선율 | 0 개선율 | 꼬리 개선율 | 전체 개선 fold |\n|---|---:|---:|---:|---:|\n'
for r in effects:
    if r['model']=='hierarchical':
        effect_table+=f"| {r['seed']} | {r['overall_improvement_percent']:+.3f}% | {r['zero_improvement_percent']:+.3f}% | {r['tail100_improvement_percent']:+.3f}% | {r['overall_improved_folds']}/5 |\n"
# Recompute key scores from saved OOF arrays independently of the training loop.
checks=0
for seed in [42,7]:
    data=np.load(root/f'oof_seed{seed}.npz')
    y=data['actual'];fold=data['fold']
    assert len(np.unique(data['source_row_index']))==166749 and set(fold)=={1,2,3,4,5}
    np.testing.assert_allclose(data['hierarchical_probs'].sum(axis=1),1.)
    np.testing.assert_array_equal(data['hierarchical'][data['hierarchical_probs'][:,0]>=.9],0.)
    for name in names:
        d=np.log10(1+data[name])-np.log10(1+y)
        row=summary[(summary.seed==seed)&(summary.model==name)].iloc[0]
        np.testing.assert_allclose(np.mean([np.sqrt(np.mean(d[fold==f]**2)) for f in range(1,6)]),row.mean_fold_rmse,rtol=1e-12)
        np.testing.assert_allclose(np.sqrt(np.mean(d[y==0]**2)),row.zero_log_rmse,rtol=1e-12)
        np.testing.assert_allclose(np.sqrt(np.mean(d[y>100]**2)),row.tail100_log_rmse,rtol=1e-12)
        checks+=3
(root/'validation.json').write_text(json.dumps({'metric_recomputations':checks,'oof_coverage':True,'probability_sums':True,'zero_gate':True},indent=2)+'\n')
report=f'''# 46개 피처: 0 전용 분류기와 양수 전용 일반/꼬리 분류기

## 결과

{table}

기존 대비 전체 계층형 분류 모델의 개선율은 다음과 같다. 양수는 개선, 음수는 악화다.

{effect_table}

전체는 각 시드5-fold RMSE 평균, 실제0·SFR>100은 해당 시드의 모든 OOF 행을 모은 RMSE다.
평가 로그는 log10(1+SFR)이다. `zero_gate_only`는 원인 분리를 위한 진단값이며,
0/양수 확률만 바꾸고 기존3분류기의 양수 내 일반/꼬리 비율은 유지한다.
`hierarchical`은 양수 내 일반/꼬리 분류기도 실제 양수 학습 행으로 새로 학습한 요청 모델이다.
이전34개 피처의 단순 확률 재조합 실험과 달리 두 번째 전용 이진 분류기를 학습했다.
이번에 피처·문턱·가중치·학습 횟수는 탐색하지 않았다. 현재 채택 모델이나 GitHub는 변경하지 않았다.

## 어떻게 학습하고 예측하는가

1. 0/양수 분류기: 전체 학습 행에서 정답 `SFR>0`을 학습한다.
2. 일반/꼬리 분류기: **실제 SFR>0**인 학습 행만 사용해 정답 `SFR>10`을 학습한다.
3. 일반·꼬리 회귀기: 기존46개 모델의 회귀기를 그대로 사용한다. 두 회귀기의 학습 대상은 각각 실제 0<SFR≤10과 SFR>10이다.

`p=P(SFR>0)`이고 `q=P(SFR>10 | SFR>0)`이면

```
P0    = 1-p
P일반 = p*(1-q)
P꼬리 = p*q
예측로그 = P일반*일반회귀로그 + P꼬리*꼬리회귀로그
```

P0≥.9이면 예측로그를0으로 바꾸고, `max(10**예측로그-1,0)`으로 선형 SFR을 반환한다.
가장 큰 확률의 구간으로 강제 라우팅하지 않는다. 두 번째 분류기의 학습 행을 첫 분류기의 예측으로 정하지도 않는다.
학습기는 모두 HGB, 학습률.08, 400회, 잎최대31, 조기종료없음, random_state42다.
46개 피처도 동일하다. 질량별 잔차 기준선은 각 fold의 학습 입력으로만 적합한다.

## 큰 실수의 변화

{diag}

0 회귀 오차와 0 재현율은 다른 지표다. 0을 더 많이 찾아도 놓친 소수의 0 은하에서 큰 양수를 예측하면 RMSE가 커질 수 있다.
꼬리 회귀기는 고정했으므로 꼬리 성능 변화는 분류 확률이 달라진 결과다.
`summary.csv`에는 SFR>10·>30·>100의 로그/선형 오차와 편향, 큰0 오차의 제곱합도 있다.

## 공정한 비교를 위한 확인

- 원래80:20 분할의 학습166,749행에서 KFold5 시드42와7을 사용했다. 기존46개 결과와 같은 fold다.
- 기존 회귀기의 OOF 출력 캐시를 그대로 재사용했다. 데이터·코드 지문과 원본 행 번호를 확인하고,
  각 fold에서 기존3분류 예측을 캐시로 다시 만들어 원본 저장 예측과 일치하는지 확인했다.
- 원래20% 검증셋은 이번 실험에서 채점하거나 선택에 사용하지 않았다.
- 같은 관측값을 반복 사용한 비교이며 독립 데이터 검증은 아니다. 시드별 결과를 별도로 제시한다.
- 단위 테스트에서 실제 양수 행만 두 번째 분류기 학습에 들어가는지, 확률 곱과 .9 경계값을 확인했다.
- 저장 OOF에서 {checks}개 지표를 재계산했고 확률 합, 0 문턱, 모든 행의 커버리지를 확인했다.

## 저장 모델과 사용

`hierarchical46.joblib`는 원래 학습166,749행에 두 이진 분류기를 학습하고,
기존46개 모델의 학습 완료 회귀기와 잔차 변환기를 결합한 파일이다.
기존 회귀기의 출력은 저장·재로딩 후에도 정확히 동일함을 확인했다.
원래 모델 파일의 SHA256도 유지된다.

이 폴더를 작업 디렉터리로 하고 Python3.12, requirements.txt 환경에서 사용한다.

```python
import joblib
import pandas as pd
from feature_pipeline import predict_sfr

model = joblib.load('hierarchical46.joblib')
reference = pd.read_csv('/path/to/Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv')
reference = reference.drop(columns='sfr_linear')
prediction = predict_sfr(reference.iloc[:100].copy(), model, reference)
```

환경·동반 은하 계산에는 같은 전체 고유 은하 합집합을 고정해서 사용한다.
잔차 기준선은 모델에 저장돼 있으며 예측 배치에서 다시 학습하지 않는다.
재실행에는 원본 데이터, `feature_search` 산출물 폴더와 해당 캐시가 필요하다.
캐시가 없으면 스크립트가 같은 설정의 원래 부품을 다시 학습하고 원본 예측과의 일치를 검증한다.

```bash
python test_hierarchical.py
python run_experiment.py --data '/path/to/data.csv' --source-dir '/path/to/feature_search' --cache '/path/to/cache'
```

`oof_seed42.npz`, `oof_seed7.npz`의 hierarchical_probs 열 순서는 `[P0, P일반, P꼬리]`,
binary_probabilities는 `[P양수, P꼬리_given_양수]`다.
fit_counts.csv에는 두 번째 분류기가 실제 양수 행만 학습한 행 수를 기록했다.
'''
(root/'REPORT.md').write_text(report)
(root/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n'
    for p in sorted(root.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
archive=root.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.iterdir()):
        if p.is_file():z.write(p,arcname=root.name+'/'+p.name)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None
print(table)
print(pd.DataFrame(effects).to_string(index=False))
print(diag)

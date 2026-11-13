from pathlib import Path
import json,zipfile,collections
import numpy as np
import pandas as pd
import scipy,sklearn,catboost,lightgbm,joblib,threadpoolctl

out=Path('outputs/a34_model_comparison')
meta=json.loads((out/'metadata.json').read_text())
selections=json.loads((out/'selections.json').read_text())
final=json.loads((out/'final_selection.json').read_text())
records=json.loads((out/'fit_records.json').read_text())
overall=pd.read_csv(out/'overall_comparison.csv');scores=pd.read_csv(out/'scores.csv');folds=pd.read_csv(out/'fold_scores.csv')
d=np.load(out/'oof_predictions.npz');parts=np.load(out/'oof_components.npz');y=d['actual'];f=d['fold']
names={'A34_HGB':'기존 HGB / HGB / HGB','cat_gate':'분류 → CatBoost','lgb_gate':'분류 → LightGBM',
       'cat_mid':'일반 회귀 → CatBoost','lgb_mid':'일반 회귀 → LightGBM',
       'cat_tail':'꼬리 회귀 → CatBoost','lgb_tail':'꼬리 회귀 → LightGBM',
       'extra_tail':'꼬리 회귀 → ExtraTrees','C_inner_selected':'내부 검증으로 고른 부품 조합'}

def md(frame,index_label):
    lines=['| '+index_label+' | '+' | '.join(map(str,frame.columns))+' |','|---|'+'---:|'*len(frame.columns)]
    for idx,row in frame.iterrows():
        vals=[f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x) for x in row]
        lines.append('| '+names.get(idx,str(idx))+' | '+' | '.join(vals)+' |')
    return '\n'.join(lines)

segs={'All':np.ones(len(y),bool),'SFR=0':y==0,'0<SFR<=3':(y>0)&(y<=3),'3<SFR<=10':(y>3)&(y<=10),
      '10<SFR<=30':(y>10)&(y<=30),'30<SFR<=100':(y>30)&(y<=100),'SFR>100':y>100}
for row in scores.itertuples():
    mask=segs[row.segment];p=d[row.model]
    actual=np.sqrt(np.mean((np.log10(1+p[mask])-np.log10(1+y[mask]))**2))
    assert abs(actual-row.rmse)<1e-12 and int(mask.sum())==row.n
    assert np.isfinite(p).all() and (p>=0).all()
for row in folds.itertuples():
    mask=f==row.fold
    actual=np.sqrt(np.mean((np.log10(1+d[row.model][mask])-np.log10(1+y[mask]))**2))
    assert abs(actual-row.rmse)<1e-12
ref=np.load(out/'baseline_reference.npz')
np.testing.assert_allclose(d['A34_HGB'],ref['A34_both'],rtol=1e-9,atol=1e-9)
# Reconstruct the selected mixture independently from saved parts and fold selections.
for k,s in enumerate(selections,1):
    mask=f==k
    keys={slot:s['combo_keys'].get(slot,'hgb_'+slot) for slot in ['gate','mid','tail']}
    p=parts[keys['gate']][mask]
    pred=np.exp(p[:,1]*parts[keys['mid']][mask]+p[:,2]*parts[keys['tail']][mask])-1
    np.testing.assert_allclose(pred,d['C_inner_selected'][mask],rtol=1e-12,atol=1e-12)

choices=[]
for k,s in enumerate(selections,1):
    choices.append(dict(fold=k,**{slot:s['combo_specs'][slot]['family'] for slot in ['gate','mid','tail']},
        inner_baseline=s['baseline_rmse'],inner_combo=s['combo_rmse'],fallback=s['combo_fallback']))
pd.DataFrame(choices).to_csv(out/'combination_choices.csv',index=False)
warning_records=[r for r in records if r.get('warnings')]
tab=overall.set_index('model')[['rmse','improvement_pct','fold_wins_vs_baseline']]
tab.columns=['로그 RMSE ↓','개선율 (%)','개선 fold / 5']
best=overall.iloc[0];combo=overall.set_index('model').loc['C_inner_selected']
bestgroups=scores.query('model==@best.model').set_index('segment')
basegroups=scores.query("model=='A34_HGB'").set_index('segment')
worse=[seg for seg in segs if seg!='All' and bestgroups.loc[seg,'rmse']>basegroups.loc[seg,'rmse']]
versions={'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,
          'scikit-learn':sklearn.__version__,'catboost':catboost.__version__,'lightgbm':lightgbm.__version__,
          'joblib':joblib.__version__,'threadpoolctl':threadpoolctl.__version__}
(out/'requirements.txt').write_text(''.join(f'{k}=={v}\n' for k,v in versions.items()))
meta['validation']='3 tests passed; OOF overall/group/fold scores independently recomputed; baseline reproduced; saved component reconstruction matches selected combination; final serialized model prediction roundtrip passed.'
meta['runtime_note']='macOS LightGBM used existing sklearn/.dylibs/libomp.dylib via DYLD_LIBRARY_PATH; no system library changes.'
(out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
report=f'''# A34의 HGB 부품을 다른 모델로 교체한 결과

가장 낮은 OOF 로그 RMSE는 **{names[best.model]}: {best.rmse:.6f}**, 기존 대비 **{best.improvement_pct:.2f}%** 개선이다.
각 fold의 내부 검증만으로 고른 부품 조합의 OOF 점수는 **{combo.rmse:.6f}**, 개선율은 **{combo.improvement_pct:.2f}%**다.

## 전체 결과

{md(tab,'교체 대상')}

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

{md(folds.pivot(index='model',columns='fold',values='rmse'),'교체 대상')}

## 내부 검증이 선택한 조합

{md(pd.DataFrame(choices).set_index('fold'),'외부 fold')}

fold마다 알고리즘이나 설정이 다를 수 있다. 조합 행을 하나의 고정된 설정이 5회 검증된 점수라고 해석하지 않는다.
세부 후보 설정과 내부 점수는 `selections.json`, 학습 기록은 `fit_records.json`에 있다.

## SFR 구간별 결과

{md(scores.pivot(index='model',columns='segment',values='rmse').drop(columns='All'),'교체 대상')}

전체 1위 후보에서 기존보다 악화된 구간: {', '.join(worse) or '없음'}.
극단 꼬리는 표본 209개이므로 전체 개선과 별개로 해석한다.

## 학습 완료 모델

`A34_selected.joblib`은 원래 학습 데이터 166,749개 안에서 seed2026의 내부 분할로 조합을 선택하고,
그 사양으로 학습 데이터 전체에 재학습한 모델이다. holdout 타깃은 사용하지 않았다.
이 파일은 내부 선택 절차를 실행한 결과이며, OOF 표의 가장 좋은 단일 행을 그대로 저장한 파일이라는 뜻은 아니다.

선택 사양:

```json
{json.dumps(final['combo_specs'],ensure_ascii=False,indent=2)}
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
- 재학습 모델 저장/불러오기 전후 예측 일치. 학습 경고 기록 {len(warning_records)}개.
- 총 실행 {meta['elapsed_seconds']:.1f}초. 데이터 Git blob SHA `{meta['data_git_blob_sha']}`.
- 실행 환경은 `requirements.txt`와 `metadata.json`에 기록했다. GitHub는 변경하지 않았다.

```bash
python -m pip install -r requirements.txt
python test_models.py
python run_experiment.py --data '/path/to/Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv' --reference baseline_reference.npz --out rerun --fit-final
```

macOS에서 libomp 로딩 오류가 나면 OpenMP 런타임 경로가 필요하다. 이 실행에서는 설치된 scikit-learn의 `.dylibs` 디렉터리를 `DYLD_LIBRARY_PATH`로 지정했다.
세부 API는 [LightGBM 공식 문서](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html)와 [CatBoost 공식 문서](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor)를 참고했다.
'''
(out/'REPORT.md').write_text(report)
archive=out.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(out.iterdir()):
        if p.is_file():z.write(p,arcname=f'{out.name}/{p.name}')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None;print('ZIP verified:',len(z.namelist()),'files')
print(overall.to_string(index=False))
print('CHOICES:',json.dumps(choices))
print('FINAL SPECS:',json.dumps(final['combo_specs']))
print('WARNINGS:',warning_records)

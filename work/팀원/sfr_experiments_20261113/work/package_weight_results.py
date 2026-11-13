from pathlib import Path
import json,zipfile
import numpy as np
import pandas as pd
import scipy,sklearn,lightgbm,catboost,joblib,threadpoolctl

out=Path('outputs/a34_feature_weighting')
meta=json.loads((out/'metadata.json').read_text());records=json.loads((out/'fit_records.json').read_text())
overall=pd.read_csv(out/'overall_comparison.csv');scores=pd.read_csv(out/'scores.csv');folds=pd.read_csv(out/'fold_scores.csv')
importance=pd.read_csv(out/'current_gate_importance.csv');perms=pd.read_csv(out/'permutation_summary.csv')
gate=pd.read_csv(out/'gate_metrics.csv');d=np.load(out/'oof_predictions.npz');p=np.load(out/'oof_probabilities.npz');ex=np.load(out/'fixed_experts.npz')
y=d['actual'];fold=d['fold'];names={'baseline':'기존 (1, 1)','density_x2':'밀도만 2배 (2, 1)','BH_x2':'BH만 2배 (1, 2)',
    'both_x1_5':'둘 다 1.5배 (1.5, 1.5)','both_x2':'둘 다 2배 (2, 2)'}
segs={'All':np.ones(len(y),bool),'SFR=0':y==0,'0<SFR<=3':(y>0)&(y<=3),'3<SFR<=10':(y>3)&(y<=10),
    '10<SFR<=30':(y>10)&(y<=30),'30<SFR<=100':(y>30)&(y<=100),'SFR>100':y>100}
for row in scores.itertuples():
    mask=segs[row.segment]
    actual=np.sqrt(np.mean((np.log10(1+y[mask])-np.log10(1+d[row.model][mask]))**2))
    assert abs(actual-row.rmse)<1e-12 and row.n==mask.sum()
for row in folds.itertuples():
    mask=fold==row.fold
    actual=np.sqrt(np.mean((np.log10(1+y[mask])-np.log10(1+d[row.model][mask]))**2))
    assert abs(actual-row.rmse)<1e-12
for name in names:
    prob=p[name];pred=np.exp(prob[:,1]*ex['mid']+prob[:,2]*ex['tail'])-1.
    np.testing.assert_allclose(pred,d[name],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(prob.sum(axis=1),1.)
    assert np.isfinite(d[name]).all() and (d[name]>=0).all()
old=np.load('outputs/a34_model_comparison/oof_components.npz')
np.testing.assert_allclose(p['baseline'],old['lgb_gate'],rtol=1e-9,atol=1e-9)
np.testing.assert_array_equal(d['source_row_index'],old['source_row_index'])

def md(frame,label):
    lines=['| '+label+' | '+' | '.join(map(str,frame.columns))+' |','|---|'+'---:|'*len(frame.columns)]
    for idx,row in frame.iterrows():
        vals=[f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in row]
        lines.append('| '+names.get(idx,str(idx))+' | '+' | '.join(vals)+' |')
    return '\n'.join(lines)

tab=overall.set_index('model')[['rmse','improvement_pct','fold_wins']]
tab.columns=['로그 RMSE ↓','개선율 (%)','개선 fold / 5']
best=overall.iloc[0]
warnings=[r for r in records if r['warnings']]
usage=pd.DataFrame(records).groupby('model')[['density_splits','bh_splits','total_splits']].mean()
usage['density_split_pct']=100*usage.density_splits/usage.total_splits
usage['bh_split_pct']=100*usage.bh_splits/usage.total_splits
usage.to_csv(out/'split_usage.csv')
foldgain=pd.read_csv(out/'baseline_fold_importance.csv').groupby('feature').gain_pct.agg(['mean','std']).sort_values('mean',ascending=False)
foldgain.to_csv(out/'baseline_cv_gain_summary.csv')
diag=perms[~perms.feature.str.startswith('GROUP_')].set_index('feature')
display_importance=importance.head(10)
if 'log10_BH_mass' not in display_importance.feature.to_list():
    display_importance=pd.concat([display_importance,importance.query("feature=='log10_BH_mass'")])
versions={'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__,
          'lightgbm':lightgbm.__version__,'catboost':catboost.__version__,'joblib':joblib.__version__,'threadpoolctl':threadpoolctl.__version__}
(out/'requirements.txt').write_text(''.join(f'{k}=={v}\n' for k,v in versions.items()))
meta['validation']='2 tests passed; named feature weighting and LightGBM parameter uptake checked; baseline fold gate probabilities reproduced; saved all/group/fold scores and fixed-expert mixture independently recomputed.'
(out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
conclusion=('이번 비교에서는 기존의 균등 분할 우대를 유지하는 것이 적절하다.' if best.model=='baseline'
    else f'이번 비교의 최저 점수는 {names[best.model]}이며 기존 대비 {best.improvement_pct:.3f}% 개선됐다. 같은 CV를 보고 고른 후보이므로 채택 전 추가 검증이 필요하다.')
report=f'''# 분류기 중요도 및 밀도·BH 우대 실험

{conclusion}

## 현재 분류기가 사용한 변수

원래 학습 데이터 전체로 학습해 저장한 `A34_selected.joblib`의 LightGBM 분류기를 읽었다.
분류 대상은 0 / 0<SFR≤10 / SFR>10의 세 부류다. 0 판정만의 중요도와 같지 않다.
아래 gain 비율은 학습 중 분할에서 얻은 손실 감소량의 비중이다. 원인·독립적 기여율이나 예측 오차 감소율은 아니다.

{md(display_importance.set_index('feature')[['rank','gain_pct','splits']],'변수')}

밀도는 이미 학습 gain의 약 36.3%로 1위다. BH 질량 자체는 22위지만 BH/항성 로그 질량비는 6위다.
BH 정보를 모델이 무시한다는 뜻은 아니다. 원본과 파생 변수가 관련되어 중요도가 분산될 수 있다.
전체 34개 순위는 `current_gate_importance.csv`, fold별 기준 분류기의 평균 gain은 `baseline_cv_gain_summary.csv`에 있다.

## 검증 데이터에서 확인한 중요도

각 fold의 외부 검증에서 3,000개를 3부류 비율에 맞춰 뽑고, 각 입력을 2회 섞었다.
그 입력은 분류기에만 섞어 넣었으며 일반·꼬리 회귀기의 예측은 고정했다.
값이 클수록 해당 분류기가 그 입력에 의존했다는 의미다. 표는 3부류 log loss 증가 평균으로 정렬했다.

{md(diag[['delta_log_loss_mean','delta_zero_log_loss_mean','delta_mixed_rmse_mean']].head(10),'변수')}

BH 원본 진단:

{md(diag.loc[['log10_BH_mass','bh_to_star'],['delta_log_loss_mean','delta_zero_log_loss_mean','delta_mixed_rmse_mean']],'변수')}

상관된 변수는 서로 대체할 수 있고, 파생 변수 하나만 섞으면 원본과의 수학적 관계가 깨질 수 있다.
따라서 이 순위를 독립적/인과적 중요도라고 해석하지 않는다. 5개 fold×2회 값의 표준편차도 독립 실험 10개의 신뢰구간이 아니다.
보조로 밀도 묶음(원본 밀도·5/10NN·반경1/2/5 이웃 수)과 BH 묶음(BH 질량·BH/항성비)을 함께 섞은 결과도 `permutation_summary.csv`에 있다.

## 밀도와 BH를 더 중요하게 보는 방법

[LightGBM 공식 문서의 feature_contri](https://lightgbm.readthedocs.io/en/stable/Parameters.html#feature_contri)에 따라 후보 분할 이득에 변수별 배수를 곱했다.
입력 숫자를 단순히 확대하지 않고, 학습 시 해당 변수로 분할하는 선택을 우대했다.
이번 우대 대상은 정확히 `log10_overdensity_plus_1`과 `log10_BH_mass` 두 열이다.
나머지 32개는 배수 1을 유지했다. BH/항성비나 모든 환경 변수를 함께 우대한 실험은 아니다.
값이 클수록 SFR을 높이거나 낮추라는 단조 제약, 강제 0 문턱, 클래스/표본 가중치는 적용하지 않았다.

## 예측 성능 비교

{md(tab,'밀도·BH 분할 우대 배수')}

기준은 현재 저장된 구성과 같은 LightGBM 분류 / LightGBM 일반 회귀 / HGB 꼬리 회귀다.
입력34개, T=10, log_offset=1, 로그 혼합을 유지하고 분류기만 다시 학습했다.
모든 분류기 설정은 600회·63 leaves·learning_rate=.05·reg_lambda=5·min_child_samples=20이며, 우대 배수만 바꿨다.
지표는 같은 166,749개 학습 은하의 KFold(5, shuffle=True, random_state=42) OOF log10(1+SFR) RMSE다.
배수는 결과를 보기 전에 고정했다. 비교 후 가장 좋은 행을 선택하는 것은 반복 모델 개발에 해당하며 독립 최종 성능 검증은 아니다.
holdout 타깃은 평가하지 않았다. 이 실행에서는 기존 학습 완료 모델을 덮어쓰지 않았다.

## fold 및 SFR 구간별 결과

{md(folds.pivot(index='model',columns='fold',values='rmse'),'우대 설정')}

{md(scores.pivot(index='model',columns='segment',values='rmse').drop(columns='All'),'우대 설정')}

## 분류 품질과 사용 빈도

{md(gate.set_index('model'),'우대 설정')}

{md(usage[['density_split_pct','bh_split_pct']],'우대 설정')}

분할 횟수 비중은 실제 사용 빈도를 보여주며 성능 개선 자체를 뜻하지 않는다.
특히 가중 모델의 gain은 우대 배수가 개입하므로, 가중 후 gain 증가를 예측력이 좋아졌다는 근거로 삼지 않았다.
최종 판단은 외부 fold의 혼합 RMSE로 한다.

## 검증과 재현

- 변수 이름에 따른 배수 배치와 LightGBM 파라미터 전달 테스트 2개 통과.
- 각 fold에서 기준 분류 확률이 이전 OOF와 1e-9 안으로 일치했다.
- 저장한 전체·구간·fold 점수 및 고정 회귀기와의 혼합을 독립 재계산해 확인했다.
- 학습 경고 기록 {len(warnings)}개. 총 실행 시간 {meta['elapsed_seconds']:.1f}초.
- 환경 계산은 이전 실험과 같은 고유 참고 은하 전체의 입력만 사용했다. 기하 계산에 SFR을 넣지 않았다.

```bash
python -m pip install -r requirements.txt
python test_weighting.py
python run_weighting.py --data '/path/to/Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv' --previous ../a34_model_comparison --out rerun
```

재실행에는 이전 `a34_model_comparison`의 OOF 부품·예측·선택 기록이 필요하다.
저장 모델의 gain 표는 `read_importance.py`로 읽는다. macOS LightGBM OpenMP 경로는 앞 실행과 동일하게 설정한다.
`oof_predictions.npz`, `oof_probabilities.npz`, `fixed_experts.npz`에 점수 재계산에 필요한 값을 저장했다.
'''
(out/'REPORT.md').write_text(report)
archive=out.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for path in sorted(out.iterdir()):
        if path.is_file():z.write(path,arcname=f'{out.name}/{path.name}')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None;print('ZIP verified',len(z.namelist()),'files')
print(overall.to_string(index=False));print('PERMUTATION\n'+perms.head(12).to_string(index=False))
print('BH\n'+diag.loc[['log10_BH_mass','bh_to_star']].to_string())
print('USAGE\n'+usage.to_string());print('Warnings:',warnings)

from pathlib import Path
import json,zipfile
import numpy as np
import pandas as pd
import scipy,sklearn,lightgbm,catboost,joblib,threadpoolctl

out=Path('outputs/a34_random_forest')
meta=json.loads((out/'metadata.json').read_text());choices=json.loads((out/'choices.json').read_text())
records=json.loads((out/'fit_records.json').read_text())
scores=pd.read_csv(out/'scores.csv');folds=pd.read_csv(out/'fold_scores.csv');overall=pd.read_csv(out/'overall_comparison.csv')
d=np.load(out/'oof_predictions.npz');c=np.load(out/'oof_components.npz');y=d['actual'];f=d['fold']
old=np.load('outputs/a34_model_comparison/oof_components.npz')
np.testing.assert_array_equal(old['fold'],f)
names={'HGB_base':'전부 HGB 기준','LGB_base':'LightGBM / LightGBM / HGB 기준'}
for ctx in ['HGB','LGB']:
    for slot,lab in [('gate','분류 → RF'),('mid','일반 회귀 → RF'),('tail','꼬리 회귀 → RF'),('combo','개선 RF 부품 조합')]:
        names[ctx+'_rf_'+slot]=('HGB 기준: ' if ctx=='HGB' else '현재 조합 기준: ')+lab

def md(frame,label):
    lines=['| '+label+' | '+' | '.join(map(str,frame.columns))+' |','|---|'+'---:|'*len(frame.columns)]
    for idx,row in frame.iterrows():
        v=[f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x) for x in row]
        lines.append('| '+names.get(idx,str(idx))+' | '+' | '.join(v)+' |')
    return '\n'.join(lines)

segs={'All':np.ones(len(y),bool),'SFR=0':y==0,'0<SFR<=3':(y>0)&(y<=3),'3<SFR<=10':(y>3)&(y<=10),
      '10<SFR<=30':(y>10)&(y<=30),'30<SFR<=100':(y>30)&(y<=100),'SFR>100':y>100}
for row in scores.itertuples():
    mask=segs[row.segment];p=d[row.model]
    assert np.isfinite(p).all() and (p>=0).all()
    actual=np.sqrt(np.mean((np.log10(1+y[mask])-np.log10(1+p[mask]))**2))
    assert abs(actual-row.rmse)<1e-12 and row.n==mask.sum()
for row in folds.itertuples():
    mask=f==row.fold
    actual=np.sqrt(np.mean((np.log10(1+y[mask])-np.log10(1+d[row.model][mask]))**2))
    assert abs(actual-row.rmse)<1e-12
for sel in choices:
    mask=f==sel['fold'];ctx=sel['context']
    comp={s:c['reference_'+('lgb_' if ctx=='LGB' and s!='tail' else 'hgb_')+s][mask] for s in ['gate','mid','tail']}
    for slot in sel['accepted_slots']:comp[slot]=c[ctx+'_'+slot][mask]
    p=comp['gate'];pred=np.exp(p[:,1]*comp['mid']+p[:,2]*comp['tail'])-1
    np.testing.assert_allclose(pred,d[ctx+'_rf_combo'][mask],rtol=1e-12,atol=1e-12)
warnings=[r for r in records if r['warnings']]
compact=[dict(fold=s['fold'],context=s['context'],accepted_slots=', '.join(s['accepted_slots']) or 'none',
    gate_leaf=s['selected_specs']['gate']['params']['min_samples_leaf'],
    mid_leaf=s['selected_specs']['mid']['params']['min_samples_leaf'],
    tail_leaf=s['selected_specs']['tail']['params']['min_samples_leaf']) for s in choices]
pd.DataFrame(compact).to_csv(out/'selection_summary.csv',index=False)
best=overall.query("model.str.startswith('LGB_')",engine='python').sort_values('rmse').iloc[0]
tab=overall.set_index('model')[['rmse','improvement_pct','fold_wins_vs_context']]
tab.columns=['로그 RMSE ↓','각 기준 대비 개선율 (%)','개선 fold / 5']
meta['validation']='Two RF tests passed; cached folds/rows/parameters verified; inner HGB scores reproduced; saved overall/group/fold scores recomputed; selected combinations reconstructed from component predictions.'
(out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
versions={'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__,
    'lightgbm':lightgbm.__version__,'catboost':catboost.__version__,'joblib':joblib.__version__,'threadpoolctl':threadpoolctl.__version__}
(out/'requirements.txt').write_text(''.join(f'{k}=={v}\n' for k,v in versions.items()))
# A concise statement of adoption based on the current architecture's OOF comparison.
decision=('현재 조합을 유지한다. 이번 설정에서 RF 추가 조합의 개선이 확인되지 않았다.'
    if best.model=='LGB_base' or best.rmse>=overall.set_index('model').loc['LGB_base','rmse']
    else f'추가 검증 후보는 {names[best.model]}이며 현재 기준 대비 {best.improvement_pct:.2f}% 개선됐다. 이 실행에서는 기존 학습 완료 모델을 교체하지 않았다.')
report=f'''# A34에 Random Forest를 적용한 실험

{decision}

## 결과

{md(tab,'구성')}

지표는 log10(1+SFR) RMSE이고 낮을수록 좋다. 개선율은 각 행이 속한 HGB 또는 현재 조합 기준으로 계산했다.
모든 점수는 같은 166,749개 학습 은하의 5-fold OOF 예측에서 계산했다.

기존 보고서의 0.095152는 fold마다 알고리즘까지 선택한 절차의 점수다.
이번 `LGB_base`는 저장된 최종 모델과 같은 LightGBM 분류 / LightGBM 일반 회귀 / HGB 꼬리 회귀를 고정한 구성의 OOF 점수다.
두 기준의 의미가 다르므로 수치를 섞지 않았다. 반복된 모델 선택에 쓰인 CV이므로 최종 독립 검증 점수는 아니다.

## 설정과 평가 절차

- 입력 34개, T=10, ln(1+SFR) 회귀 목표, 로그 공간 확률 혼합을 유지했다. 강제 0 규칙은 사용하지 않았다.
- RandomForestClassifier / RandomForestRegressor: 나무 200개, bootstrap=True, random_state=42, n_jobs=4.
- 분류 max_features='sqrt', gini 손실. 회귀 max_features=.7, squared_error 손실.
- min_samples_leaf=3과 10을 각 부품별로 비교했다. 환경 변수의 NaN은 sklearn RF의 기본 결측값 처리에 맡겼다.
- 기존과 같은 외부 KFold(5, shuffle=True, random_state=42). 내부 20% 검증을 3부류로 층화(seed1000+fold)했다.
- 각 기준에서 전체 혼합 로그 RMSE로 RF 설정을 따로 선택했다. 선택 후 외부 학습 fold 전체로 재학습했다.
- 개별 교체가 내부 기준보다 나아진 부품만 조합했다. 공동 조합이 내부 기준보다 나쁘면 가장 좋은 단일 교체로 되돌렸다.
- 기준 모델의 외부 예측은 이전 실행에서 재사용했다. 데이터 해시·행 번호·fold·LightGBM 설정을 확인했고, 내부 HGB 점수도 1e-12 안에서 재현됐다.
- 참고 은하 집합의 입력만으로 환경을 계산했다. holdout 타깃을 평가하거나 튜닝에 사용하지 않았다.

## fold별 결과

{md(folds.pivot(index='model',columns='fold',values='rmse'),'구성')}

## 구간별 결과

{md(scores.pivot(index='model',columns='segment',values='rmse').drop(columns='All'),'구성')}

## 해석과 한계

이번 결과는 위의 200개 나무 및 두 가지 잎 크기 설정에 대한 평가다. RF 알고리즘 전체의 가능성을 배제하는 결과는 아니다.
두 기준을 함께 사용해 HGB만 있을 때의 교체 효과와 현재 LightGBM 조합에서의 추가 효과를 구분했다.
특정 구간의 개선만으로 채택하지 않고 전체 혼합 로그 RMSE를 우선했다.
기존 `A34_selected.joblib`은 변경하지 않았다. 이번 요청은 후보 시험으로, 새 전체 학습 모델은 저장하지 않았다.

## 검증 및 파일

- RF 분류·회귀의 NaN 처리, bootstrap, 실제 부류 목표, 로그 혼합에 관한 테스트 2개 통과.
- 저장한 전체·구간·fold 점수와 조합 예측을 독립 재계산해 일치 확인.
- 학습 경고 기록 {len(warnings)}개. 총 실행 시간 {meta['elapsed_seconds']:.1f}초.
- `choices.json`, `selection_summary.csv`: 내부 선택 설정과 채택 부품
- `fit_records.json`: 설정별 내부 점수·학습 시간·경고
- `oof_predictions.npz`, `oof_components.npz`: OOF 예측과 부품 출력
- `scores.csv`, `fold_scores.csv`, `overall_comparison.csv`: 점수

```bash
python -m pip install -r requirements.txt
python test_rf.py
python run_rf.py --data '/path/to/Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv' --previous ../a34_model_comparison --out rerun
```

재실행에는 이전 `a34_model_comparison` 폴더의 OOF 예측·부품·선택 기록이 필요하다.
macOS LightGBM의 OpenMP 연결은 앞 실험과 같이 scikit-learn의 `.dylibs`를 `DYLD_LIBRARY_PATH`로 지정했다.
'''
(out/'REPORT.md').write_text(report)
archive=out.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(out.iterdir()):
        if p.is_file():z.write(p,arcname=f'{out.name}/{p.name}')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None;print('ZIP verified',len(z.namelist()),'files')
print(overall.to_string(index=False));print('Choices:',compact);print('Warnings:',warnings)

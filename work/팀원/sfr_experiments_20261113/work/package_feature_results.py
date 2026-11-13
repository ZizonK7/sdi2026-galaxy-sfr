from pathlib import Path
import json, zipfile
import numpy as np
import pandas as pd
import scipy, sklearn, joblib, threadpoolctl

out=Path('outputs/a_feature_comparison')
meta=json.loads((out/'metadata.json').read_text())
overall=pd.read_csv(out/'overall_comparison.csv')
scores=pd.read_csv(out/'scores.csv')
folds=pd.read_csv(out/'fold_scores.csv')
gates=pd.read_csv(out/'gate_diagnostics.csv')
details=json.loads((out/'fit_details.json').read_text())
pred=np.load(out/'oof_predictions.npz');prob=np.load(out/'oof_probabilities.npz')
y=pred['actual'];f=pred['fold']
labels={'A13_baseline':'기존 A (13개)','A17_mass':'질량 4개 추가 (17개)',
        'A30_environment':'환경 17개 추가 (30개)','A34_both':'질량+환경 추가 (34개)'}

def md_table(frame,label):
    lines=['| '+label+' | '+' | '.join(map(str,frame.columns))+' |','|---|'+'---:|'*len(frame.columns)]
    for idx,row in frame.iterrows():
        vals=[f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in row]
        lines.append('| '+labels.get(idx,str(idx))+' | '+' | '.join(vals)+' |')
    return '\n'.join(lines)

for row in scores.itertuples():
    masks={'All':np.ones(len(y),bool),'SFR=0':y==0,'0<SFR<=3':(y>0)&(y<=3),
        '3<SFR<=10':(y>3)&(y<=10),'10<SFR<=30':(y>10)&(y<=30),
        '30<SFR<=100':(y>30)&(y<=100),'SFR>100':y>100}
    mask=masks[row.segment]
    actual=np.sqrt(np.mean((np.log10(1+pred[row.model][mask])-np.log10(1+y[mask]))**2))
    assert abs(actual-row.rmse)<1e-12 and int(mask.sum())==row.n
for row in folds.itertuples():
    mask=f==row.fold
    actual=np.sqrt(np.mean((np.log10(1+pred[row.model][mask])-np.log10(1+y[mask]))**2))
    assert abs(actual-row.rmse)<1e-12
for name in labels:
    assert np.isfinite(pred[name]).all() and (pred[name]>=0).all()
    np.testing.assert_allclose(prob[name].sum(axis=1),1.,rtol=1e-12,atol=1e-12)
ref=np.load(out/'baseline_reference.npz')
np.testing.assert_allclose(pred['A13_baseline'],ref['A_log1p_T10_soft'],rtol=1e-9,atol=1e-9)
np.testing.assert_array_equal(pred['source_row_index'],ref['source_row_index'])
best=overall.iloc[0]
foldwide=folds.pivot(index='fold',columns='model',values='rmse')
pairrows=[]
for cand,base in [('A17_mass','A13_baseline'),('A30_environment','A13_baseline'),
                  ('A34_both','A13_baseline'),('A34_both','A17_mass'),('A34_both','A30_environment')]:
    diff=foldwide[cand]-foldwide[base]
    pairrows.append(dict(candidate=cand,reference=base,mean_delta=diff.mean(),min_delta=diff.min(),
        max_delta=diff.max(),improved_folds=int((diff<0).sum())))
pairs=pd.DataFrame(pairrows);pairs.to_csv(out/'paired_comparisons.csv',index=False)
tab=overall.set_index('model')[['rmse','improvement_pct','fold_wins_vs_baseline']]
tab.columns=['로그 RMSE ↓','개선율 (%)','개선 fold 수 / 5']
groupwide=scores.pivot(index='segment',columns='model',values='rmse')
warnings=[d for d in details if d['warnings']]
assert not warnings, warnings
inter=pd.read_csv(out/'feature_interaction.csv')
directions=[]
for name in ['A17_mass','A30_environment','A34_both']:
    better=[s for s in groupwide.index if s!='All' and groupwide.loc[s,name]<groupwide.loc[s,'A13_baseline']]
    worse=[s for s in groupwide.index if s!='All' and groupwide.loc[s,name]>=groupwide.loc[s,'A13_baseline']]
    directions.append(f'- {labels[name]}: 개선 구간 {", ".join(better) or "없음"}; 미개선 구간 {", ".join(worse) or "없음"}.')
meta['validation']='Three tests passed; saved overall/group/fold scores independently recomputed; baseline OOF reproduced within 1e-9; all predictions finite/nonnegative; no fit warnings.'
(out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
versions={'numpy':np.__version__,'scipy':scipy.__version__,'scikit-learn':sklearn.__version__,
    'pandas':pd.__version__,'joblib':joblib.__version__,'threadpoolctl':threadpoolctl.__version__}
(out/'requirements.txt').write_text(''.join(f'{k}=={v}\n' for k,v in versions.items()))
leader='https://github.com/ZizonK7/sdi2026-galaxy-sfr/blob/'+meta['leader_commit']
report=f'''# A에 조장 파생 변수를 적용한 비교

가장 낮은 점수는 **{labels[best.model]}: {best.rmse:.6f}**이다.
기존 A 대비 로그 RMSE 개선율은 **{best.improvement_pct:.2f}%**, 개선 fold 수는 **{int(best.fold_wins_vs_baseline)}/5**다.
이 값들은 우리 A를 실제로 다시 학습해 얻은 OOF 결과이며, 조장 모델의 점수를 옮긴 것이 아니다.

## 통제한 조건

- A의 3분류(0 / 0<SFR≤10 / SFR>10), 실제 부류별 두 HGB 회귀기, 로그 공간 확률 혼합 유지.
- 회귀 목표 ln(1+SFR), 평가는 log10(1+SFR) RMSE. 이전 실험과 같은 정의로 고정했다.
- 모든 부품 HGB: learning_rate=.08, max_iter=150, max_leaf_nodes=31, random_state=42. 조기 종료 기본값 유지.
- 입력 묶음만 변경했다. 분류기·일반 회귀기·꼬리 회귀기 모두 해당 묶음을 사용한다.
- 기존 166,749개 학습 표본의 행 순서 및 KFold(5, shuffle=True, random_state=42) 유지.
- 남겨둔 41,688개 holdout의 타깃으로 점수를 계산하거나 모델을 선택하지 않았다. 이 프로젝트 전체에서 미열람인 검증셋이라고 주장하는 것은 아니다.

## 전체 결과

{md_table(tab,'입력')}

OOF 전체 RMSE는 모든 학습 표본의 검증 예측을 모아서 계산했다. fold RMSE의 산술평균과는 구별한다.
이번 비교로 선택한 후보의 개선 폭은 같은 CV에서 평가한 값이다. 반복된 모델 선택의 불확실성과 공간 일반화까지 입증한 결과는 아니다.

## 구간별 결과

{md_table(scores.pivot(index='model',columns='segment',values='rmse').drop(columns='All'),'입력')}

{chr(10).join(directions)}

## fold와 분류 진단

{md_table(folds.pivot(index='model',columns='fold',values='rmse'),'입력')}

{md_table(gates.set_index('model'),'입력')}

분류 log loss는 보조 진단이며 최종 선택은 전체 혼합 예측의 로그 RMSE 기준이다.
질량과 환경의 MSE 상호작용 `L(둘 다)-L(환경)-L(질량)+L(기존)`의 fold 평균은 {inter.mse_interaction.mean():.9f}다.
양수면 두 묶음의 단독 개선 효과가 일부 겹친다는 방향이고, 음수면 결합 개선이 단독 개선의 합보다 크다는 방향이다.
물리적 인과나 독립된 증거 수를 뜻하지 않는다.

## 추가 입력과 적용 방식

질량 4개는 현재 항성/전구체 항성, 현재 헤일로/전구체 헤일로, BH/항성, 전구체 항성/전구체 헤일로의 로그 질량 차이다.
원본의 특수값도 조장 구현과 동일하게 유지했다. 누적 질량비를 시간당 성장률이라고 해석하지 않는다.

환경 17개는 5/10번째 이웃 밀도, 반경 1/2/5의 이웃 수, 거대 헤일로 거리·상대속도,
FoF 그룹 크기·최대 질량·중심 여부·그룹 상대속도, 헤일로−항성 로그 질량 차이다.
주기 경계 box=200, FoF b=.2, 거대 헤일로 문턱 13.5/14.0을 고정했다.
단위·호스트의 물리적 의미까지 이번 예측 비교로 확정하지 않는다.

기하 계산에는 전체 208,437개 은하의 **입력만** 사용했다. SFR 열을 환경 함수에 넘기지 않았다.
이는 [FAQ Q2]({leader}/조교_FAQ_2026-11-11.md)의 라벨 없는 평가 입력 활용 허용에 맞춘 방식이다.
새 예측에서도 같은 고유 참고 은하 합집합을 고정해야 한다. 작은 배치별로 따로 환경을 계산하면 이웃 수·FoF 연결 길이가 바뀌므로 같은 절차가 아니다.
다른 상자 또는 다른 표본 밀도로의 일반화는 별도 검증 대상이다.

FoF 고립 은하의 상대속도 NaN은 그대로 유지하고 HGB의 결측값 처리를 사용했다.
`a_log_mixture.py`는 이전 모델에서 입력 검증의 NaN 허용만 변경했다. infinity와 잘못된 타깃은 계속 거부한다.
원래 13개 입력에서 이전 A 예측이 1e-9 오차 안으로 재현됨을 확인했다.

## 검증과 다음 단계

- 세 테스트 통과: SFR 열 무관성 및 행 순서 불변성, 주기 이동 및 경계 이웃 처리, NaN 허용/infinity 거부.
- 학습 경고 0개. 저장한 전체·구간·fold 점수를 별도 재계산해 일치 확인.
- 데이터 Git blob SHA `{meta['data_git_blob_sha']}`; 조장 코드 기준 `{meta['leader_commit']}`.
- 총 실행 시간 {meta['elapsed_seconds']:.1f}초. 실제 HGB 반복 수는 `fit_details.json`에 저장했다.
- 이번 실험은 입력 묶음 비교까지 수행했다. 학습 상한 400과 ExtraTrees 꼬리 교체는 다음 단계이며 이번 점수에 섞지 않았다.
- 다음 비교에서는 우세한 입력 묶음을 고정한 뒤, 부품별 학습 상한과 ExtraTrees 꼬리 교체를 별도로 검증한다.

## 재현

```bash
python -m pip install -r requirements.txt
python test_features.py
python run_experiment.py --data '/path/to/Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv' --references baseline_reference.npz --out rerun
```

모델 재학습에는 `LogTailMixtureSFR(tail_threshold=10.0, log_offset=1.0)`을 사용한다.
`features.build_feature_sets`에는 고정한 참고 은하 전체를 넣고, 반환된 행에서 학습/예측 대상을 선택한다.
OOF 예측과 코드를 제공하며 전체 학습 데이터로 재학습한 배포 모델은 포함하지 않는다. GitHub는 변경하지 않았다.

## 근거 및 파일

- [조장 환경 변수 함수]({leader}/work/조장/log_model_cv_2026-11-11/log_model_cv_env.py)
- [조장 질량 변수 정의]({leader}/work/조장/log_model_cv_2026-11-11/q5_mass_history.py)
- `overall_comparison.csv`, `scores.csv`, `fold_scores.csv`: 전체·구간·fold 점수
- `paired_comparisons.csv`, `feature_interaction.csv`, `gate_diagnostics.csv`: 짝 비교·입력 상호작용·분류 진단
- `oof_predictions.npz`, `oof_probabilities.npz`: 실제값·원본 행 번호·fold 및 후보별 검증 예측/확률
- `metadata.json`, `fit_details.json`: 입력 열·참고 집합·환경·학습 기록
'''
(out/'REPORT.md').write_text(report)
archive=out.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(out.iterdir()):
        if p.is_file():z.write(p,arcname=f'{out.name}/{p.name}')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    print('Verified zip:',len(z.namelist()),'files')
print(overall.to_string(index=False))
print(groupwide.round(6).to_string())
print(pairs.to_string(index=False))
print('MSE interaction:',inter.mse_interaction.mean())

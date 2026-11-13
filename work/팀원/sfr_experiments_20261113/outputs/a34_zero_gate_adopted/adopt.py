from pathlib import Path
import json,hashlib,zipfile
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from zero_gate import ZeroGatedMixture,apply_zero_threshold
from features import RAW,build_feature_sets

def main():
    old=Path('outputs/a34_model_comparison');out=Path(__file__).resolve().parent
    original_hash=hashlib.sha256((old/'A34_selected.joblib').read_bytes()).hexdigest()
    base=joblib.load(old/'A34_selected.joblib')
    adopted=ZeroGatedMixture(base,zero_threshold=.9)
    model_path=out/'A34_zero_gate_090.joblib'
    joblib.dump(adopted,model_path,compress=3)
    reloaded=joblib.load(model_path)
    # Real-input prediction validation; no performance metric on these fitted rows.
    data=Path('/Users/ljh/Downloads/Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv')
    raw=data.read_bytes();sha=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest();del raw
    assert sha=='06479cedd323253f08af9983c73f395cc02d3774'
    df=pd.read_csv(data)
    with threadpool_limits(limits=4):
        full=build_feature_sets(df[RAW])['A34_both']
        cached=np.load(old/'oof_components.npz');ix=cached['source_row_index']
        X=full.iloc[ix[:3000]].to_numpy()
        pred,soft,p0=reloaded.predict_with_details(X)
        np.testing.assert_array_equal(soft,base.predict(X))
        assert (p0>=.9).any() and (p0<.9).any()
        np.testing.assert_array_equal(pred[p0>=.9],0.)
        np.testing.assert_array_equal(pred[p0<.9],soft[p0<.9])
        np.testing.assert_array_equal(pred,adopted.predict(X))
    # Performance comparison uses previously saved OOF components, never fit-row predictions.
    y=cached['actual'];fold=cached['fold'];prob=cached['lgb_gate'];p0=prob[:,0]
    soft=np.maximum(np.exp(prob[:,1]*cached['lgb_mid']+prob[:,2]*cached['hgb_tail'])-1.,0.)
    pred=apply_zero_threshold(soft,p0,.9);forced=p0>=.9
    se0=(np.log10(1+soft)-np.log10(1+y))**2;se1=(np.log10(1+pred)-np.log10(1+y))**2
    segments={'All':np.ones(len(y),bool),'SFR=0':y==0,'0<SFR<=3':(y>0)&(y<=3),'3<SFR<=10':(y>3)&(y<=10),
        '10<SFR<=30':(y>10)&(y<=30),'30<SFR<=100':(y>30)&(y<=100),'SFR>100':y>100}
    rows=[dict(segment=s,n=int(m.sum()),before_rmse=float(np.sqrt(se0[m].mean())),after_rmse=float(np.sqrt(se1[m].mean())),forced_zero_rows=int((forced&m).sum())) for s,m in segments.items()]
    fr=[dict(fold=k,before_rmse=float(np.sqrt(se0[fold==k].mean())),after_rmse=float(np.sqrt(se1[fold==k].mean()))) for k in range(1,6)]
    pd.DataFrame(rows).to_csv(out/'scores.csv',index=False);pd.DataFrame(fr).to_csv(out/'fold_scores.csv',index=False)
    np.savez_compressed(out/'oof_predictions.npz',actual=y,fold=fold,source_row_index=ix,soft=soft,adopted=pred,p_zero=p0)
    meta_old=json.loads((old/'metadata.json').read_text())
    metadata=dict(adopted=True,policy='P(SFR=0)>=0.9 => exactly0; otherwise original log-space mixture prediction',
        threshold=.9,inclusive=True,model_file=model_path.name,base_specs=base.specs,features=meta_old['features'],
        base_model_sha256=original_hash,model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
        retrained=False,feature_weighting=False,data_git_blob_sha=sha,holdout_evaluated=False,
        metric='RMSE log10(1+SFR)',before_rmse=rows[0]['before_rmse'],after_rmse=rows[0]['after_rmse'],
        forced_zero_rows=int(forced.sum()),true_zero_forced=int((forced&(y==0)).sum()),positive_forced=int((forced&(y>0)).sum()),
        forced_zero_precision=float((y[forced]==0).mean()),zero_recall=float(forced[y==0].mean()),
        positive_forced_max_actual=float(y[forced&(y>0)].max()),
        improved_folds=sum(r['after_rmse']<r['before_rmse'] for r in fr),versions=meta_old['versions'],
        validation='Threshold boundary tests; real 3000-row predictions checked after serialization: gated outputs exactly0, ungated outputs identical to original, original model hash unchanged.',
        evaluation_note='OOF comparison for fixed LightGBM/LightGBM/HGB architecture; original20% holdout not evaluated. Adopted at explicit user request, not by threshold search.')
    assert hashlib.sha256((old/'A34_selected.joblib').read_bytes()).hexdigest()==original_hash
    (out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    report=f'''# 채택 모델: P(SFR=0) ≥ 0.9이면 정확히 0 출력

사용자 요청에 따라 이 규칙을 현재 모델의 기본 출력 정책으로 채택했다.

- 분류: LightGBM / 일반 회귀: LightGBM / 꼬리 회귀: HGB
- 입력 34개, T=10, 로그 혼합 유지. 밀도/BH 추가 우대 없음.
- P(0)≥0.9인 경우 정확히 0. P(0)<0.9이면 기존 예측과 동일.
- 기존 학습 모델을 감싸서 저장했다. 재학습하지 않았다.

## 변경 전후 같은 5-fold OOF 비교

| 지표 | 적용 전 | 적용 후 |
|---|---:|---:|
| log10(1+SFR) RMSE | {metadata['before_rmse']:.9f} | {metadata['after_rmse']:.9f} |

총 {metadata['forced_zero_rows']:,}개에 0을 출력했다. 이 중 실제 0은 {metadata['true_zero_forced']:,}개,
실제 양수는 {metadata['positive_forced']:,}개다. 강제0 정밀도는 {100*metadata['forced_zero_precision']:.2f}%,
실제0 재현율은 {100*metadata['zero_recall']:.2f}%다. 개선 fold는 {metadata['improved_folds']}/5다.
구간별·fold별 수치는 `scores.csv`, `fold_scores.csv`에 있다. holdout은 평가하지 않았다.
이전의 0.095152는 fold마다 부품을 선택한 절차의 점수이며, 여기의 고정 구성 점수와 다르다.

## 사용

이 폴더를 Python 모듈 검색 경로에 두고 다음처럼 불러온다.

```python
import joblib
from features import build_feature_sets
model = joblib.load('A34_zero_gate_090.joblib')
# 환경 계산에는 같은 상자의 고유 참고 은하 합집합을 고정해서 사용한다.
X34 = build_feature_sets(catalogue)['A34_both']
predictions = model.predict(X34.to_numpy())
# 필요하면 기존 예측과 확률도 확인할 수 있다.
predictions, soft_predictions, p_zero = model.predict_with_details(X34.to_numpy())
```

`model.predict()`에 0 규칙이 기본 포함되어 있으므로 별도 후처리는 필요 없다.
작은 배치별로 환경 변수를 다시 계산하지 말고 고정 참고 집합에서 계산한 행을 선택한다.
`models.py`, `zero_gate.py` 및 `requirements.txt`의 환경이 필요하다.
macOS LightGBM의 OpenMP 라이브러리 경로는 기존 실행과 동일하게 설정한다.

## 확인

경계값 0.9 포함 여부와 미적용 행 보존 테스트를 통과했고, 실제 입력 3,000행에서 저장/재로딩 후
기준 이상은 정확히0, 기준 미만은 이전 예측과 완전히 같음을 확인했다. 기존 모델 파일의 SHA256도 변하지 않았다.
이 폴더의 모델을 현재 채택본으로 지정했으며 `outputs/current_model.json`에 경로를 기록했다.
이전 실험 모델·결과는 비교 이력으로 보존했다. GitHub는 변경하지 않았다.
'''
    (out/'REPORT.md').write_text(report)
    current=dict(model_path=str(model_path),module_directory=str(out),metadata_path=str(out/'metadata.json'),
        prediction_policy=metadata['policy'],zero_threshold=.9,architecture=['LightGBM','LightGBM','HGB'],n_features=34)
    (Path('outputs')/'current_model.json').write_text(json.dumps(current,ensure_ascii=False,indent=2))
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.iterdir()):
            if path.is_file():z.write(path,arcname=f'{out.name}/{path.name}')
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps(metadata,ensure_ascii=False,indent=2),flush=True)
    print('ADOPTED AND VERIFIED',flush=True)

if __name__=='__main__':main()

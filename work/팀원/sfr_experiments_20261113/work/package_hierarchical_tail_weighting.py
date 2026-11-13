"""Verify saved predictions/selection and produce a self-contained experiment report."""
from pathlib import Path
import hashlib
import json
import sys
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/hierarchical_tail_weighting'
sys.path.insert(0,str(OUT))
from metrics import metrics
from three_regime import combine

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    meta=json.loads((OUT/'metadata.json').read_text())
    summary=pd.read_csv(OUT/'summary.csv')
    folds=pd.read_csv(OUT/'fold_results.csv')
    inner=pd.read_csv(OUT/'inner_results.csv')
    choices=pd.read_csv(OUT/'inner_selections.csv')
    fits=pd.read_csv(OUT/'fit_records.csv')
    paired=pd.read_csv(OUT/'paired_differences.csv')
    configs=list(meta['configs'])
    names=summary[summary.seed==42].model.tolist()
    metric_checks=0
    for seed in (42,7):
        saved=np.load(OUT/f'oof_seed{seed}.npz')
        source=np.load(ROOT/f'outputs/hierarchical_gate46/oof_seed{seed}.npz')
        previous=np.load(ROOT/f'outputs/hierarchical_tail_models/oof_seed{seed}.npz')
        logs=np.load(OUT/f'tail_components_seed{seed}.npz')
        y=saved['actual'];p=source['hierarchical_probs']
        np.testing.assert_array_equal(saved['source_row_index'],source['source_row_index'])
        np.testing.assert_array_equal(y,source['actual'])
        np.testing.assert_array_equal(saved['fold'],source['fold'])
        np.testing.assert_allclose(saved['hgb_w1'],previous['hgb400_baseline'],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(saved['lgb_w1'],previous['lgb_400_31'],rtol=1e-10,atol=1e-10)
        for fold in range(1,6):
            va=np.flatnonzero(saved['fold']==fold);tr=np.flatnonzero(saved['fold']!=fold)
            label=f'seed{seed}_fold{fold}'
            selected=choices[choices.split==label].iloc[0]
            strata=np.select([y[tr]==0,y[tr]<=10,y[tr]<=100],[0,1,2],default=3)
            it,iv=train_test_split(tr,test_size=.2,random_state=142,stratify=strata)
            assert not np.intersect1d(iv,va).size and not np.intersect1d(it,va).size
            assert hashlib.sha256(it.tobytes()).hexdigest()==selected.train_rows_sha256
            assert hashlib.sha256(iv.tobytes()).hexdigest()==selected.validation_rows_sha256
            for name,prefix in [('nested_overall',''),('nested_hgb','hgb_'),('nested_lgb','lgb_')]:
                sub=inner[(inner.split==label)&inner.model.str.startswith(prefix)]
                expected=sub.sort_values(['log_rmse','model']).iloc[0].model
                assert selected[name]==expected
                np.testing.assert_array_equal(saved[name][va],saved[expected][va])
            key=hashlib.sha256(json.dumps([meta['source_fingerprint'],seed,fold,'mid',meta['features']]).encode()).hexdigest()
            cache=np.load(ROOT/'work/feature_search_cache'/(key+'.npz'))
            np.testing.assert_array_equal(cache['validation_rows'],va)
            for name in configs:
                reconstructed=np.maximum(10.**combine(p[va],cache['prediction'],logs[name][va])-1.,0.)
                np.testing.assert_allclose(saved[name][va],reconstructed,rtol=1e-10,atol=1e-10)
                row=fits[(fits.seed==seed)&(fits.fold==fold)&(fits.model==name)].iloc[0]
                weight=meta['configs'][name][1]
                ntail=int((y[tr]>10).sum());nextreme=int((y[tr]>100).sum())
                assert row.tail_training_rows==ntail and row.extreme_training_rows==nextreme
                assert row.sum_weights==ntail+(weight-1)*nextreme
        for name in names:
            pred=saved[name]
            assert np.isfinite(pred).all() and (pred>=0).all()
            np.testing.assert_array_equal(pred==0,saved['hgb_w1']==0)
            result=metrics(y,pred)
            row=summary[(summary.seed==seed)&(summary.model==name)].iloc[0]
            for key,value in result.items():
                np.testing.assert_allclose(value,row[key],rtol=1e-10,atol=1e-12);metric_checks+=1
            fold_scores=[]
            for fold in range(1,6):
                mask=saved['fold']==fold;result=metrics(y[mask],pred[mask])
                rowfold=folds[(folds.seed==seed)&(folds.fold==fold)&(folds.model==name)].iloc[0]
                for key,value in result.items():
                    np.testing.assert_allclose(value,rowfold[key],rtol=1e-10,atol=1e-12);metric_checks+=1
                fold_scores.append(result['log_rmse'])
            np.testing.assert_allclose(np.mean(fold_scores),row.mean_fold_rmse,rtol=1e-10,atol=1e-12);metric_checks+=1
    for name,prefix in [('nested_overall',''),('nested_hgb','hgb_'),('nested_lgb','lgb_')]:
        sub=inner[(inner.split=='final_train80')&inner.model.str.startswith(prefix)]
        assert meta['final_selection'][name]==sub.sort_values(['log_rmse','model']).iloc[0].model
    for record in meta['saved_models'].values():
        assert sha(OUT/record['file'])==record['sha256']
    assert sha(ROOT/'outputs/hierarchical_gate46/hierarchical46.joblib')==meta['source_model_sha256']
    assert sha(ROOT/'outputs/hierarchical_tail_models/selected_model.joblib')==meta['previous_model_sha256']
    assert sha(ROOT/'outputs/current_model.json')==meta['current_pointer_sha256']
    unit_log=(ROOT/'work/hierarchical_tail_weighting_tests.log').read_text()
    assert 'Ran 3 tests' in unit_log and '\nOK\n' in unit_log
    (OUT/'test_results.txt').write_text(unit_log)
    validation=dict(passed=True,metric_checks=metric_checks,
        baseline_predictions_reproduced=True,weighted_mixtures_reconstructed=True,
        internal_selection_disjoint_and_reproduced=True,training_weight_counts_verified=True,
        zero_outputs_unchanged=True,artifact_hashes_verified=True,current_model_unchanged=True,
        analytic_and_boundary_tests_passed=3, training_run_checks=meta['checks'])
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')

    def label(name):
        if name=='nested_overall':return '내부검증 선택: 전체 후보'
        if name=='nested_hgb':return '내부검증 선택: HGB 가중치'
        if name=='nested_lgb':return '내부검증 선택: LightGBM 가중치'
        family,weight=meta['configs'][name]
        return f'{"HGB" if family=="hgb" else "LightGBM"} ×{weight}'
    def row(seed,name):return summary[(summary.seed==seed)&(summary.model==name)].iloc[0]
    changes=[]
    for seed in (42,7):
        for name in names:
            if name.endswith('_w1'):continue
            baseline='hgb_w1' if name.startswith('hgb') or name=='nested_hgb' else 'lgb_w1'
            a,b=row(seed,name),row(seed,baseline)
            d=dict(seed=seed,model=name,baseline=baseline)
            for k in ('mean_fold_rmse','zero_log_rmse','tail100_log_rmse'):
                d[k+'_pct']=(float(a[k])/float(b[k])-1)*100
            sub=paired[(paired.seed==seed)&(paired.model==name)]
            d['overall_fold_wins']=int((sub.log_rmse_delta<0).sum())
            d['extreme_fold_wins']=int((sub.tail100_log_rmse_delta<0).sum())
            changes.append(d)
    delta=pd.DataFrame(changes);delta.to_csv(OUT/'changes.csv',index=False)
    weighted=[n for n in configs if not n.endswith('_w1')]
    consistent_tail=[n for n in weighted if (delta[delta.model==n].tail100_log_rmse_pct<0).all()]
    consistent_both=[n for n in consistent_tail if (delta[delta.model==n].mean_fold_rmse_pct<0).all()]
    consistent_zero_tail=[n for n in consistent_tail if (delta[delta.model==n].zero_log_rmse_pct<0).all()]
    all_overall_worse=bool((delta[delta.model.isin(weighted)].mean_fold_rmse_pct>0).all())
    overall_selected=meta['final_selection']['nested_overall']
    lines=['# 극단값 학습 가중치 실험','',
        '기존 계층형 46피처 모델에서 꼬리 회귀기의 학습 가중치만 바꿨다. '
        'HGB400과 LightGBM400/31 각각에 대해 극단값 가중치 1·2·3·5배를 비교했다. '
        '가중치 1은 각 회귀기 자체의 기존 결과다.', '',
        '## 결과 판단','',
        ('모든 고정 가중치 설정에서 두 시드 모두 전체 RMSE가 증가했다. 전체 로그 RMSE가 최종 평가 기준이라면 '
         '이번 가중치 설정을 채택할 근거는 부족하며, 무가중치 기준 모델을 유지하는 편이 타당하다.'
         if all_overall_worse else '전체와 구간별 결과를 함께 비교해야 한다.'),'',
        '두 시드에서 같은 계열의 무가중치 모델보다 극단값 RMSE가 모두 감소한 설정: '
        +(', '.join(label(n) for n in consistent_tail) if consistent_tail else '없음')+'.','',
        '그중 전체 RMSE도 두 시드에서 모두 감소한 설정: '
        +(', '.join(label(n) for n in consistent_both) if consistent_both else '없음')+'.','',
        '실제 0과 극단값 RMSE가 두 시드에서 함께 감소한 설정: '
        +(', '.join(label(n) for n in consistent_zero_tail) if consistent_zero_tail else '없음')+'. '
        '구간별 개선이 전체 점수 개선과 동일한 것은 아니다.','',
        '위 비교는 고정된 후보들의 관측 결과다. 같은 관측치로 분할을 바꾼 반복 실험이므로 '
        '작은 차이가 독립 테스트에서 재현된다고 단정할 수 없다. '
        '0 구간까지 포함한 변화율과 내부 선택 절차의 성능을 아래에 함께 기록했다.', '',
        '## 고정한 조건','',
        '- 두 HGB 분류기, 일반 HGB 회귀기, 46개 피처, 꼬리 경계 SFR=10, P(0)≥0.9의 0 처리 규칙을 고정했다.',
        '- 꼬리 전문가는 **실제 SFR>10인 학습 행**에만 학습했다. 분류기가 예측한 꼬리 행으로 학습하지 않았다.',
        '- 목표는 log10(1+SFR). 10<SFR≤100의 학습 가중치는 1, SFR>100은 1·2·3·5다. 가중치를 별도로 정규화하지 않았다.',
        '- 가중치는 꼬리 회귀기의 학습 손실에만 적용했다. **평가 RMSE에는 가중치를 적용하지 않았다.**',
        '- HGB는 400회·31잎·학습률0.08·early_stopping=False, LightGBM은 400트리·31잎·학습률0.08·reg_lambda=0이다. 기타 설정은 weighted_tail.py 참조.',
        '- 비교는 원래 train80%의 동일한 seed42/7 5-fold OOF다. 원래 holdout20% 정답은 사용하지 않았다.',
        f'- 전체 학습 행 {meta["train_rows"]:,}개, SFR>10은 {meta["tail_train_rows"]:,}개, SFR>100은 {meta["extreme_train_rows"]:,}개다.',
        '- 전체 RMSE는 5개 fold RMSE의 평균이다. 구간별 RMSE는 합친 OOF의 해당 실제 정답 구간에서 계산했다. 모두 로그 공간의 점수이며 낮을수록 좋다.', '',
        '## 고정 가중치별 결과','']
    for seed in (42,7):
        lines += [f'### Seed {seed}','','| 꼬리 회귀기·극단값 가중치 | 전체 | 실제 0 | SFR>10 | SFR>100 |',
                  '|---|---:|---:|---:|---:|']
        for name in configs:
            r=row(seed,name)
            lines.append(f'| {label(name)} | {r.mean_fold_rmse:.8f} | {r.zero_log_rmse:.6f} | {r.tail10_log_rmse:.6f} | {r.tail100_log_rmse:.6f} |')
        lines.append('')
    lines += ['## 같은 계열의 무가중치 모델 대비 변화율','','음수가 개선이다. HGB는 HGB ×1, LightGBM은 LightGBM ×1과 비교했다.','',
              '| 설정 | 시드 | 전체 변화율 | 실제 0 변화율 | SFR>100 변화율 | 극단값 개선 fold |',
              '|---|---:|---:|---:|---:|---:|']
    for name in weighted:
        for seed in (42,7):
            d=delta[(delta.seed==seed)&(delta.model==name)].iloc[0]
            lines.append(f'| {label(name)} | {seed} | {d.mean_fold_rmse_pct:+.3f}% | {d.zero_log_rmse_pct:+.3f}% | {d.tail100_log_rmse_pct:+.3f}% | {d.extreme_fold_wins}/5 |')
    lines += ['', '## 내부 검증으로 가중치를 고르는 절차의 성능','',
        '각 바깥 fold의 학습 데이터만 다시 80:20으로 나누고, 그 안에서 46피처의 잔차 계산·두 분류기·일반 회귀기·꼬리 회귀기를 학습했다. '
        '내부 검증의 **최종 혼합 전체 로그 RMSE**가 가장 낮은 설정을 골라 바깥 학습 데이터 전체로 재학습한 예측을 평가했다. '
        '바깥 검증 정답은 그 fold의 가중치 선정에 사용하지 않았다. '
        '이는 한 번의 내부 holdout을 둔 중첩 평가이며, 내부 전체 교차검증은 아니다.', '',
        '| 선택 절차 | 시드 | 전체 | 실제 0 | SFR>100 |', '|---|---:|---:|---:|---:|']
    for name in ('nested_overall','nested_hgb','nested_lgb'):
        for seed in (42,7):
            r=row(seed,name)
            lines.append(f'| {label(name)} | {seed} | {r.mean_fold_rmse:.8f} | {r.zero_log_rmse:.6f} | {r.tail100_log_rmse:.6f} |')
    lines += ['', '각 fold에서 선택한 설정은 inner_selections.csv, 선택에 사용한 점수는 inner_results.csv에 있다. '
        '이 절차의 점수는 fold별로 선택이 달라질 수 있는 절차의 점수이며, 특정 고정 가중치 모델 하나의 점수와 다르다.', '',
        '## 0과 극단값 해석','',
        '모든 후보의 0 출력 행과 0 판정 precision/recall은 동일했다. 실제 0의 RMSE 변화는 '
        '0으로 고정하지 않은 행에서 꼬리 전문가의 예측이 혼합값에 영향을 준 결과다.', '',
        '극단값 가중치는 큰 값의 오차를 학습 과정에서 더 중요하게 만든다. '
        '그러나 극단값 과대예측이나 다른 구간 오차 증가가 생길 수 있으므로 전체 점수와 함께 판단해야 한다. '
        'SFR>100 평가는 209개 관측치이므로 fold별 결과도 확인할 필요가 있다.', '',
        '## 모델 저장과 사용','',
        f'최종 전체 train80% 안에서 별도의 내부 검증 절차로 선택한 설정은 **{label(overall_selected)}**다. '
        '이 설정을 train80% 전체로 재학습해 selected_model.joblib로 저장했다. '
        '외부 OOF에서 가장 좋아 보이는 설정을 골라 덮어쓴 것이 아니며, 현재 모델로 자동 채택한 것도 아니다.', '',
        '가중치 2·3·5의 여섯 모델도 각각 저장했다. 모든 모델은 학습된 분류기·일반 회귀기·전처리를 포함한다. '
        'current_model.json과 기존 모델, GitHub는 변경하지 않았다.', '']
    for key,record in meta['saved_models'].items():
        lines.append(f'- `{record["file"]}`: {label(record["candidate"])}. SHA256 `{record["sha256"]}`.')
    lines += ['', '패키지 내 requirements.txt와 파이썬 모듈을 함께 사용한다. reference는 환경 피처 계산에 쓰는 원본 고정 카탈로그다.', '',
        '```python','import sys','from pathlib import Path','import joblib','import pandas as pd',
        'bundle = Path("/absolute/path/to/hierarchical_tail_weighting")','sys.path.insert(0, str(bundle))',
        'from feature_pipeline import predict_sfr','model = joblib.load(bundle / "selected_model.joblib")',
        'reference = pd.read_csv("/absolute/path/to/reference.csv").drop(columns="sfr_linear", errors="ignore")',
        'new_table = pd.read_csv("/absolute/path/to/input.csv").drop(columns="sfr_linear", errors="ignore")',
        'predictions = predict_sfr(new_table, model, reference)','```','',
        '실험 재실행은 원본 데이터와 이전 hierarchical_gate46 및 hierarchical_tail_models 폴더의 모델·메타데이터·OOF 파일을 필요로 한다. '
        '예측 실행에는 이전 폴더가 필요하지 않다.', '',
        '```bash','python run_experiment.py --data /absolute/path/to/data.csv \\',
        '  --source-dir /absolute/path/to/hierarchical_gate46 \\',
        '  --tail-source-dir /absolute/path/to/hierarchical_tail_models \\',
        '  --cache /absolute/path/to/feature_search_cache','```','',
        '## 검증','',
        f'저장한 OOF로 점수 {metric_checks:,}개를 재계산했다. 기존 HGB·LightGBM 기준 예측 재현, '
        '고정 확률과 일반 회귀값을 사용한 최종 혼합 예측 재구성, 내부 선택 재현, 내부/외부 검증 행 분리, '
        '학습 가중치 합과 극단값 개수, 0 출력 동일, 모델 해시를 확인했다. '
        '학습 실행에서는 저장/재로딩 일치, 분류기·일반 회귀 예측 고정, 원시 입력 인터페이스도 검증했다. '
        '별도 테스트 3개로 SFR=100 경계, 비꼬리 입력 거부, 동일 피처일 때 가중 로그 평균을 예측하는지 확인했다.', '',
        '선정 규칙은 protocol.json, 데이터·버전·모델 해시는 metadata.json, 검증 결과는 validation.json에 있다. '
        '원본 데이터는 배포 ZIP에 포함하지 않았다.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    paths=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='SHA256SUMS')
    (OUT/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in paths))
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file():z.write(p,arcname=OUT.name+'/'+p.name)
    print(json.dumps(validation,indent=2))
    print(delta.to_string(index=False))
    print('Both-seed extreme improvements:',consistent_tail)
    print('Both-seed overall and extreme improvements:',consistent_both)
    print('Final internally selected model:',overall_selected)

if __name__=='__main__':main()

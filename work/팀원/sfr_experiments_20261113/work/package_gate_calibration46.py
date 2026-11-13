from pathlib import Path
import hashlib
import json
import sys
import zipfile
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import brier_score_loss,log_loss

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/gate_calibration46'
sys.path.insert(0,str(OUT))
from calibration_model import VARIANTS,apply_maps
from hierarchical_model import hierarchical_probabilities
from three_regime import combine
from metrics import metrics

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    meta=json.loads((OUT/'metadata.json').read_text())
    summary=pd.read_csv(OUT/'summary.csv');folds=pd.read_csv(OUT/'fold_results.csv')
    diagnostic=pd.read_csv(OUT/'probability_diagnostics.csv');audit=pd.read_csv(OUT/'calibration_audit.csv')
    checks=0;inner_splits=0;paired=[];gate_details=[];reference_changes=[]
    for seed in meta['seeds']:
        saved=np.load(OUT/f'oof_seed{seed}.npz');parts=np.load(OUT/f'components_seed{seed}.npz')
        source=np.load(ROOT/f'outputs/tail_blend_constrained/oof_seed{seed}.npz')
        maps=joblib.load(OUT/f'fold_calibrators_seed{seed}.joblib')
        y=saved['actual'];fold=saved['fold'];n=len(y)
        np.testing.assert_array_equal(y,source['actual'])
        np.testing.assert_array_equal(saved['source_row_index'],source['source_row_index'])
        np.testing.assert_allclose(saved['baseline'],source['strict_balance'],rtol=1e-9,atol=1e-9)
        unweighted=metrics(y,source['baseline'])
        unweighted['mean_fold_rmse']=float(np.mean([metrics(y[fold==f],source['baseline'][fold==f])['log_rmse'] for f in range(1,6)]))
        calibrated={v:np.empty((n,2)) for v in VARIANTS}
        for f,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=seed).split(y),1):
            assert (fold[va]==f).all()
            log=audit[audit.split==f'seed{seed}_fold{f}'].iloc[0]
            assert hashlib.sha256(tr.tobytes()).hexdigest()==log.train_rows_sha256
            assert hashlib.sha256(va.tobytes()).hexdigest()==log.validation_rows_sha256
            cache=np.load(ROOT/'work/gate_calibration_cache'/(log.cache_key+'.npz'))
            np.testing.assert_array_equal(cache['training_rows'],tr)
            strata=np.where(y[tr]==0,0,np.where(y[tr]<=10,1,2))
            for inner,(it,iv) in enumerate(StratifiedKFold(3,shuffle=True,random_state=314).split(tr,strata),1):
                assert (cache['inner_fold'][iv]==inner).all()
                assert not np.intersect1d(tr[it],va).size and not np.intersect1d(tr[iv],va).size
                assert not np.intersect1d(tr[it],tr[iv]).size
                inner_splits+=1
            for variant in VARIANTS:
                prob=apply_maps(parts['raw_probabilities'][va],maps[f],variant)
                calibrated[variant][va]=prob
                p=hierarchical_probabilities(prob[:,0],prob[:,1])
                reconstructed=np.maximum(10.**combine(p,parts['mid_log'][va],parts['tail_log'][va])-1.,0.)
                np.testing.assert_allclose(saved[variant][va],reconstructed,rtol=1e-10,atol=1e-10)
                np.testing.assert_array_equal(saved[variant][va][p[:,0]>=.9],0.)
        for variant in VARIANTS:
            pred=saved[variant];result=metrics(y,pred)
            assert np.isfinite(pred).all() and (pred>=0).all()
            row=summary[(summary.seed==seed)&(summary.model==variant)].iloc[0]
            for key,value in result.items():np.testing.assert_allclose(value,row[key],rtol=1e-10,atol=1e-12);checks+=1
            scores=[]
            for f in range(1,6):
                mask=fold==f;r=metrics(y[mask],pred[mask]);scores.append(r['log_rmse'])
                stored=folds[(folds.seed==seed)&(folds.fold==f)&(folds.model==variant)].iloc[0]
                for key,value in r.items():np.testing.assert_allclose(value,stored[key],rtol=1e-10,atol=1e-12);checks+=1
                if variant!='baseline':
                    base=metrics(y[mask],saved['baseline'][mask])
                    paired.append(dict(seed=seed,fold=f,model=variant,**{k+'_delta':r[k]-base[k] for k in ('log_rmse','zero_log_rmse','tail100_log_rmse')}))
            np.testing.assert_allclose(np.mean(scores),row.mean_fold_rmse,rtol=1e-10,atol=1e-12);checks+=1
            reference_changes.append(dict(seed=seed,model=variant,reference='pre_blend_unweighted_lgb',
                **{key+'_pct':(row[key]/unweighted[key]-1)*100 for key in ('mean_fold_rmse','zero_log_rmse','tail100_log_rmse')}))
            prob=calibrated[variant];positive=y>0
            calculated=dict(positive_brier=brier_score_loss(positive,prob[:,0]),positive_log_loss=log_loss(positive,prob[:,0],labels=[0,1]),
                conditional_tail_brier=brier_score_loss(y[positive]>10,prob[positive,1]),conditional_tail_log_loss=log_loss(y[positive]>10,prob[positive,1],labels=[0,1]),
                forced_zero_rows=int((1-prob[:,0]>=.9).sum()),forced_zero_positives=int(((1-prob[:,0]>=.9)&positive).sum()),
                forced_zero_extremes=int(((1-prob[:,0]>=.9)&(y>100)).sum()),zero_outputs_extremes=int(((pred==0)&(y>100)).sum()))
            stored=diagnostic[(diagnostic.seed==seed)&(diagnostic.model==variant)].iloc[0]
            for key,value in calculated.items():np.testing.assert_allclose(value,stored[key],rtol=1e-10,atol=1e-12);checks+=1
            extreme=y>100
            gate_details.append(dict(seed=seed,model=variant,
                mean_p0_on_actual_zero=float((1-prob[y==0,0]).mean()),
                mean_positive_probability_on_actual_extreme=float(prob[extreme,0].mean()),
                mean_conditional_tail_probability_on_actual_extreme=float(prob[extreme,1].mean()),
                mean_joint_tail_probability_on_actual_extreme=float((prob[extreme,0]*prob[extreme,1]).mean())))
    for entry in meta['saved_models'].values():assert sha(OUT/entry['file'])==entry['sha256']
    assert sha(ROOT/'outputs/tail_blend_constrained/balanced_model.joblib')==meta['source_model_sha256']
    assert sha(ROOT/'outputs/current_model.json')==meta['current_pointer_sha256']
    testlog=(ROOT/'work/gate_calibration46_tests.log').read_text()
    assert 'Ran 4 tests' in testlog and '\nOK\n' in testlog
    (OUT/'test_results.txt').write_text(testlog)
    pd.DataFrame(paired).to_csv(OUT/'paired_differences.csv',index=False)
    pd.DataFrame(gate_details).to_csv(OUT/'gate_extreme_diagnostics.csv',index=False)
    reference_changes=pd.DataFrame(reference_changes)
    reference_changes.to_csv(OUT/'comparison_to_preblend_unweighted.csv',index=False)
    validation=dict(passed=True,recomputed_metric_checks=checks,inner_outer_splits_checked=inner_splits,
        all_predictions_reconstructed=True,baseline_reproduced=True,calibrated_zero_rule_verified=True,
        diagnostics_reproduced=True,model_hashes_verified=True,current_model_unchanged=True,
        unit_tests=4,training_run_checks=meta['checks'])
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    changes=[];assessment=[]
    for variant in VARIANTS:
        if variant=='baseline':continue
        for seed in meta['seeds']:
            b=summary[(summary.seed==seed)&(summary.model=='baseline')].iloc[0]
            a=summary[(summary.seed==seed)&(summary.model==variant)].iloc[0]
            changes.append(dict(seed=seed,model=variant,**{key+'_pct':(a[key]/b[key]-1)*100 for key in ('mean_fold_rmse','zero_log_rmse','tail100_log_rmse')}))
    changes=pd.DataFrame(changes);changes.to_csv(OUT/'changes.csv',index=False)
    for variant in VARIANTS:
        if variant=='baseline':continue
        rows=changes[changes.model==variant]
        assessment.append(dict(model=variant,extreme_improves_all=bool((rows.tail100_log_rmse_pct<0).all()),
            zero_nonworse_all=bool((rows.zero_log_rmse_pct<=1e-10).all()),
            overall_nonworse_all=bool((rows.mean_fold_rmse_pct<=1e-10).all()),
            overall_cost_within_01pct_all=bool((rows.mean_fold_rmse_pct<=.1+1e-10).all())))
    assessment=pd.DataFrame(assessment);assessment.to_csv(OUT/'assessment.csv',index=False)
    whole_tail=assessment[assessment.extreme_improves_all & assessment.overall_nonworse_all].model.tolist()
    cost_tail=assessment[assessment.extreme_improves_all & assessment.overall_cost_within_01pct_all].model.tolist()
    all_three=assessment[assessment.extreme_improves_all & assessment.overall_cost_within_01pct_all & assessment.zero_nonworse_all].model.tolist()
    zero_tradeoffs=assessment[assessment.overall_cost_within_01pct_all & assessment.zero_nonworse_all].model.tolist()
    labels={'baseline':'보정 없음(50:50 혼합 기준)'}
    for method in ('sigmoid','isotonic'):
        for scope,korean in [('positive','0/양수만'),('tail','중간/꼬리만'),('both','두 분류기 모두')]:labels[f'{method}_{scope}']=f'{method} — {korean}'
    show=lambda names:', '.join(labels[n] for n in names) if names else '없음'
    lines=['# 계층형 46피처 모델의 분류 확률 보정','',
        '기준은 기존 무가중치 LightGBM과 극단값 가중치3배 LightGBM을 50:50으로 섞은 모델이다. '
        '그 회귀 예측을 고정한 채 두 HGB 분류기의 확률을 보정했다. 현재 채택 모델이나 GitHub는 변경하지 않았다.', '',
        '## 세 시드에서 확인된 후보','',
        f'- 전체·SFR>100 RMSE가 함께 개선된 방법: **{show(whole_tail)}**.',
        f'- 전체 증가를 상대0.1% 이내로 제한하면서 SFR>100이 개선된 방법: **{show(cost_tail)}**.',
        f'- 위 조건에 실제0 RMSE 증가 없음까지 만족한 방법: **{show(all_three)}**.', '',
        f'별도로, 전체 증가0.1% 이내에서 실제0 RMSE가 세 시드 모두 증가하지 않은 방법은 **{show(zero_tradeoffs)}**이다. '
        '이들은0 구간을 위한 절충 후보이며, 꼬리 오차까지 개선한 것으로 해석하지 않는다.', '',
        '이는 미리 정한 여섯 방법을 반복 검증에서 관측한 결과다. 보정 방법을 바깥 검증 정답에 맞춰 학습하지 않았으며, '
        '모든 방법을 보고·저장했다. 위 목록이나 결과 순위는 탐색적 평가이며 독립 테스트에서의 우위를 보장하지 않는다.', '',
        '## 실험 조건','',
        '- 0/양수 확률만 보정, 양수 내 중간/꼬리 확률만 보정, 둘 다 보정하는 세 경우를 sigmoid와 isotonic 각각에서 시험했다.',
        '- 46피처, 두 HGB400 분류기의 설정, 일반 HGB 회귀기, 50:50 꼬리 회귀기, 학습 경계SFR>10, 극단값 평가SFR>100을 유지했다.',
        '- P(SFR=0)≥0.9이면0으로 예측하는 규칙은 유지했다. 보정 때문에 그 조건을 만족하는 행은 달라질 수 있다.',
        '- 평가 지표는 log10(1+SFR)의 무가중치 RMSE다. 전체는5개 fold 평균, 실제0과 극단값은 합친 OOF 점수다.',
        '- 원래 train80%의 동일한 세 시드42·7·99로5-fold를 평가했다. 원래 holdout20% 정답은 이번 실험에서 사용하지 않았다.',
        f'- 평가 행{meta["train_rows"]:,}개 중 SFR>100은{meta["extreme_rows"]:,}개다.', '',
        '## 보정 데이터 누수 방지','',
        '각 바깥 학습 fold를 다시3-fold로 나눴다. 내부 학습 행으로 피처 잔차 계산기와 두 분류기를 학습하고, '
        '내부 검증 행의 확률을 모아 보정기를 학습했다. 0/양수 보정에는 모든 행, '
        '중간/꼬리 보정에는 **실제 양수 행만** 사용했다. 바깥 검증 행은 보정기 학습에 들어가지 않았다.', '',
        '보정기는 내부 교차검증 확률로 학습한 뒤 바깥 학습 전체로 학습된 분류기에 적용했다. '
        '내부 모델과 전체 모델은 학습 표본 수가 다르므로 확률 분포 차이가 생길 수 있으며, '
        '바깥 검증 점수는 이 적용 방식까지 포함한 결과다. 예측 시에는 정답 구간을 사용하지 않는다.', '',
        'sigmoid는 확률의 logit을 입력으로 하는 LogisticRegression(C=1e6)이다. '
        'isotonic은 증가 방향의 비모수 보정이다. '
        '세 시드 모두 같은 관측치를 재분할했으므로 새 독립 테스트가 아니다.', '',
        '## 최종 혼합 로그 RMSE','']
    for seed in meta['seeds']:
        lines += [f'### Seed {seed}','','| 방법 | 전체 | 실제0 | SFR>100 |','|---|---:|---:|---:|']
        for variant in VARIANTS:
            r=summary[(summary.seed==seed)&(summary.model==variant)].iloc[0]
            lines.append(f'| {labels[variant]} | {r.mean_fold_rmse:.8f} | {r.zero_log_rmse:.6f} | {r.tail100_log_rmse:.6f} |')
        lines.append('')
    lines += ['## 기준 대비 변화율','','음수가 개선이다.','',
              '| 방법 | 시드 | 전체 | 실제0 | SFR>100 |','|---|---:|---:|---:|---:|']
    for _,r in changes.iterrows():lines.append(f'| {labels[r.model]} | {int(r.seed)} | {r.mean_fold_rmse_pct:+.3f}% | {r.zero_log_rmse_pct:+.3f}% | {r.tail100_log_rmse_pct:+.3f}% |')
    lines += ['', '## 더 이전의 무가중치 모델을 기준으로 본 누적 변화','',
        '위의 주 비교는 직전 **50:50 혼합 모델 대비 보정 효과**다. '
        '아래는 별도로, 혼합하기 전 **무가중치 LightGBM 꼬리 모델**에서 지금까지의 누적 변화를 확인한 것이다. '
        '이 참조 비교로 주 비교의 꼬리 악화를 개선이라고 바꿔 말하지 않는다.', '',
        '0/양수만 sigmoid 보정한 후보는 직전 혼합 모델보다 꼬리 오차가 조금 높지만, '
        '혼합 전 모델에 비해서는 세 시드 모두 실제0·극단값 오차가 낮고 전체 증가는0.1% 이내였다.', '',
        '| 시드 | 0/양수 sigmoid의 전체 변화 | 실제0 변화 | SFR>100 변화 |',
        '|---:|---:|---:|---:|']
    for _,r in reference_changes[reference_changes.model=='sigmoid_positive'].iterrows():
        lines.append(f'| {int(r.seed)} | {r.mean_fold_rmse_pct:+.3f}% | {r.zero_log_rmse_pct:+.3f}% | {r.tail100_log_rmse_pct:+.3f}% |')
    lines += ['', '전체 변형의 참조 비교는 comparison_to_preblend_unweighted.csv에 있다. '
        '이는 같은 OOF 관측치에 대한 추가 해석이며 독립 검증이나 새 모델 선정 절차가 아니다.', '',
        '## 확률 점수와0 처리 진단','',
        'probability_diagnostics.csv의 Brier와log-loss는 확률 자체의 품질을 보는 보조 지표다. '
        '이 값이 좋아져도 최종 혼합 RMSE가 좋아진다는 뜻은 아니다. '
        '중간/꼬리 확률 점수는 실제 양수에서만 계산했다. '
        'gate_extreme_diagnostics.csv에는 실제 극단값에서 평균 양수·꼬리 확률이 어떻게 바뀌는지 기록했다. '
        '이 정답별 분석은 진단에만 쓰며 예측 라우팅에는 사용하지 않는다.', '',
        '| 시드 | 방법 | 0으로 강제한 실제 양수 | 그중 SFR>100 | 실제0 재현율 |',
        '|---:|---|---:|---:|---:|']
    for _,r in diagnostic.iterrows():
        s=summary[(summary.seed==r.seed)&(summary.model==r.model)].iloc[0]
        lines.append(f'| {int(r.seed)} | {labels[r.model]} | {int(r.forced_zero_positives)} | {int(r.forced_zero_extremes)} | {100*s.zero_recall:.2f}% |')
    lines += ['', '## 저장한 모델','',
        '여섯 보정 모델을 모두 저장했다. 각 파일은 기존 회귀기와 피처 처리, 두 분류기, '
        '전체 train80% 내부3-fold로 학습한 보정기를 포함한다. 실험 후보이며 자동 채택하지 않았다.', '']
    for name,entry in meta['saved_models'].items():lines.append(f'- `{entry["file"]}`: {labels[name]}. SHA256 `{entry["sha256"]}`')
    lines += ['', 'requirements.txt와 패키지의 파이썬 모듈을 함께 사용한다. 다음은 sigmoid_both를 불러오는 사용 예시이며 추천을 뜻하지 않는다.', '',
        '```python','import sys','from pathlib import Path','import joblib','import pandas as pd',
        'bundle = Path("/absolute/path/to/gate_calibration46")','sys.path.insert(0, str(bundle))',
        'from feature_pipeline import predict_sfr','model = joblib.load(bundle / "sigmoid_both.joblib")',
        'reference = pd.read_csv("/absolute/path/to/reference.csv").drop(columns="sfr_linear", errors="ignore")',
        'new_table = pd.read_csv("/absolute/path/to/input.csv").drop(columns="sfr_linear", errors="ignore")',
        'predictions = predict_sfr(new_table, model, reference)','```','',
        '실험을 재실행할 때는 기존 tail_blend_constrained·hierarchical_gate46·hierarchical_tail_weighting 폴더와 '
        'work/feature_search_cache의 원래 일반 회귀기 OOF 캐시, 원본 CSV가 필요하다. '
        '예측 실행에는 이전 폴더나 원본 학습 정답이 필요하지 않다.', '',
        '```bash','python run_experiment.py --data /absolute/path/to/data.csv \\',
        '  --cache /absolute/path/to/gate_calibration_cache \\',
        '  --component-cache /absolute/path/to/feature_search_cache','```','',
        '## 검증','',
        f'저장 예측에서 점수{checks:,}개를 재계산하고 내부/외부 분리{inner_splits}개를 확인했다. '
        '모든 보정 예측을 원래 확률·회귀값·fold별 보정기에서 재구성했다. '
        '기존50:50 모델 예측 재현, 실제 양수에서의 조건부 보정, 확률합과0 규칙, 고정 회귀값, 모델 저장/재로딩, '
        '원시 입력 예측 인터페이스와 원본 모델 보존을 확인했다. 단위 테스트4개도 통과했다.', '',
        '상세 점수는 summary.csv, 변화율은 changes.csv, 세 시드 조건 비교는 assessment.csv, '
        '검증은 validation.json에 있다. 원본 데이터는 ZIP에 포함하지 않았다.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    files=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='SHA256SUMS')
    (OUT/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in files))
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(OUT.iterdir()):
            if p.is_file():archive.write(p,arcname=OUT.name+'/'+p.name)
    print(json.dumps(validation,indent=2))
    print(changes.to_string(index=False))
    print(assessment.to_string(index=False))

if __name__=='__main__':main()

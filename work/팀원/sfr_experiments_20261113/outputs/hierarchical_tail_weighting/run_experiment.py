import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from threadpoolctl import threadpool_limits
from feature_pipeline import UNIVERSE, FeatureEngineer, build_reference_features, fit_component, predict_sfr
from hierarchical_model import HierarchicalSFR, fit_binary_gates, positive_probability, hierarchical_probabilities
from three_regime import combine
from weighted_tail import CONFIGS, fit_tail, sample_weights
from metrics import metrics

ROOT = Path(__file__).resolve().parent
NESTED = ('nested_overall', 'nested_hgb', 'nested_lgb')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def predict_mixture(p, mid, tail):
    result = np.maximum(10.**combine(p, mid, tail)-1., 0.)
    assert np.isfinite(result).all() and (result >= 0).all()
    np.testing.assert_array_equal(result[p[:,0] >= .9], 0.)
    return result

def select_inside_training(X, y, training_rows, label, inner_records, selections):
    """An internal holdout; neither fitting nor selection uses outer-validation rows."""
    strata = np.select([y[training_rows] == 0, y[training_rows] <= 10, y[training_rows] <= 100], [0,1,2], default=3)
    it, iv = train_test_split(training_rows, test_size=.2, random_state=142, stratify=strata)
    assert not np.intersect1d(it,iv).size
    np.testing.assert_array_equal(np.sort(np.concatenate([it,iv])), np.sort(training_rows))
    engineer = FeatureEngineer().fit(X.iloc[it])
    A, B = engineer.transform(X.iloc[it]), engineer.transform(X.iloc[iv])
    start = time.monotonic()
    g0, gt = fit_binary_gates(A, y[it])
    p = hierarchical_probabilities(positive_probability(g0, B), positive_probability(gt, B))
    mid = fit_component('mid', UNIVERSE, A, y[it]).predict(B[UNIVERSE])
    mask = y[it] > 10
    local = []
    for name in CONFIGS:
        model = fit_tail(name, A.loc[mask,UNIVERSE], y[it][mask])
        pred = predict_mixture(p, mid, model.predict(B[UNIVERSE]))
        row = dict(split=label, model=name, **metrics(y[iv],pred))
        local.append(row); inner_records.append(row)
    chosen = {}
    for key, prefix in [('nested_overall',''), ('nested_hgb','hgb_'), ('nested_lgb','lgb_')]:
        rows = [r for r in local if r['model'].startswith(prefix)]
        chosen[key] = min(rows, key=lambda r:(r['log_rmse'], r['model']))['model']
    selections.append(dict(split=label, inner_train_rows=len(it), inner_validation_rows=len(iv),
        train_rows_sha256=hashlib.sha256(it.tobytes()).hexdigest(),
        validation_rows_sha256=hashlib.sha256(iv.tobytes()).hexdigest(), **chosen))
    pd.DataFrame(inner_records).to_csv(ROOT/'inner_results.csv',index=False)
    pd.DataFrame(selections).to_csv(ROOT/'inner_selections.csv',index=False)
    print(f'{label} internal selection {chosen} ({time.monotonic()-start:.1f}s)',flush=True)
    return chosen

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--source-dir',type=Path,default=ROOT.parent/'hierarchical_gate46')
    parser.add_argument('--tail-source-dir',type=Path,default=ROOT.parent/'hierarchical_tail_models')
    parser.add_argument('--cache',type=Path,required=True)
    args=parser.parse_args()
    source, tails = args.source_dir, args.tail_source_dir
    original_meta=json.loads((source/'metadata.json').read_text())
    previous_meta=json.loads((tails/'metadata.json').read_text())
    source_sha=sha(source/'hierarchical46.joblib')
    assert source_sha==original_meta['model_sha256']==previous_meta['source_model_sha256']
    assert sha(tails/'selected_model.joblib')==previous_meta['saved_models']['selected_model']['sha256']
    current_path=ROOT.parent/'current_model.json';current_sha=sha(current_path)
    raw=args.data.read_bytes()
    data_sha=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
    assert data_sha==original_meta['data_git_blob_sha']=='06479cedd323253f08af9983c73f395cc02d3774'
    fingerprint=hashlib.sha256((ROOT/'feature_pipeline.py').read_bytes()+(ROOT/'leader_model.py').read_bytes()+(ROOT/'three_regime.py').read_bytes()+data_sha.encode()).hexdigest()
    assert fingerprint==original_meta['source_fingerprint']
    df=pd.read_csv(args.data)
    ids,_=train_test_split(np.arange(len(df)),test_size=.2,random_state=42,stratify=(df.sfr_linear>0).astype(int))
    y=df.sfr_linear.to_numpy()[ids]
    records=[]; summaries=[]; fit_records=[]; inner_records=[]; selections=[]
    with threadpool_limits(limits=4):
        print('Building fixed reference features...',flush=True)
        X=build_reference_features(df).iloc[ids].reset_index(drop=True)
        for seed in (42,7):
            original=np.load(source/f'oof_seed{seed}.npz')
            previous=np.load(tails/f'oof_seed{seed}.npz')
            previous_tail=np.load(tails/f'tail_components_seed{seed}.npz')
            for saved in (original,previous):
                np.testing.assert_array_equal(ids,saved['source_row_index'])
                np.testing.assert_array_equal(y,saved['actual'])
            p=original['hierarchical_probs']
            predicted={name:np.full(len(y),np.nan) for name in (*CONFIGS,*NESTED)}
            tail_logs={name:np.full(len(y),np.nan) for name in CONFIGS}
            coverage=np.zeros(len(y),int)
            for fold,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=seed).split(X),1):
                assert (original['fold'][va]==fold).all() and not np.intersect1d(tr,va).size
                coverage[va]+=1
                label=f'seed{seed}_fold{fold}'
                print(f'{label} fitting internal selection...',flush=True)
                choices=select_inside_training(X,y,tr,label,inner_records,selections)
                engineer=FeatureEngineer().fit(X.iloc[tr])
                A,B=engineer.transform(X.iloc[tr]),engineer.transform(X.iloc[va])
                key=hashlib.sha256(json.dumps([fingerprint,seed,fold,'mid',UNIVERSE]).encode()).hexdigest()
                cache=args.cache/(key+'.npz')
                if cache.exists():
                    saved=np.load(cache);np.testing.assert_array_equal(saved['validation_rows'],va)
                    mid=saved['prediction']
                else:
                    mid=fit_component('mid',UNIVERSE,A,y[tr]).predict(B[UNIVERSE])
                mask=y[tr]>10
                for name,(family,weight) in CONFIGS.items():
                    if weight==1:
                        old_name='hgb400_baseline' if family=='hgb' else 'lgb_400_31'
                        logs=previous_tail[old_name][va];seconds=0.
                    else:
                        start=time.monotonic()
                        model=fit_tail(name,A.loc[mask,UNIVERSE],y[tr][mask])
                        logs=model.predict(B[UNIVERSE]);seconds=time.monotonic()-start
                        del model
                    pred=predict_mixture(p[va],mid,logs)
                    if weight==1:
                        np.testing.assert_allclose(pred,previous[old_name][va],rtol=1e-10,atol=1e-10)
                    predicted[name][va]=pred;tail_logs[name][va]=logs
                    row=dict(seed=seed,fold=fold,model=name,**metrics(y[va],pred))
                    records.append(row)
                    weights=sample_weights(y[tr][mask],weight)
                    fit_records.append(dict(seed=seed,fold=fold,model=name,tail_training_rows=int(mask.sum()),
                        extreme_training_rows=int((y[tr]>100).sum()),sum_weights=float(weights.sum()),seconds=seconds,
                        used_cached_baseline=weight==1))
                    print(f"{label} {name}: overall={row['log_rmse']:.8f} zero={row['zero_log_rmse']:.6f} tail100={row['tail100_log_rmse']:.6f}",flush=True)
                for name in NESTED:
                    pred=predicted[choices[name]][va]
                    predicted[name][va]=pred
                    records.append(dict(seed=seed,fold=fold,model=name,**metrics(y[va],pred)))
                pd.DataFrame(records).to_csv(ROOT/'fold_results.csv',index=False)
                pd.DataFrame(fit_records).to_csv(ROOT/'fit_records.csv',index=False)
            assert (coverage==1).all()
            table=pd.DataFrame(records)
            for name,pred in predicted.items():
                assert np.isfinite(pred).all()
                np.testing.assert_array_equal(pred==0,predicted['hgb_w1']==0)
                rows=table[(table.seed==seed)&(table.model==name)]
                summaries.append(dict(seed=seed,model=name,mean_fold_rmse=float(rows.log_rmse.mean()),**metrics(y,pred)))
            pd.DataFrame(summaries).to_csv(ROOT/'summary.csv',index=False)
            np.savez_compressed(ROOT/f'oof_seed{seed}.npz',actual=y,source_row_index=ids,fold=original['fold'],**predicted)
            np.savez_compressed(ROOT/f'tail_components_seed{seed}.npz',**tail_logs)
            print(f'Seed {seed} completed.',flush=True)
        print('Selecting final artifact inside original training80%...',flush=True)
        choices=select_inside_training(X,y,np.arange(len(y)),'final_train80',inner_records,selections)
        (ROOT/'final_selection.json').write_text(json.dumps(choices,indent=2)+'\n')
        source_model=joblib.load(source/'hierarchical46.joblib')
        lgb_source_model=joblib.load(tails/'selected_model.joblib')
        assert previous_meta['selected_model']=='lgb_400_31'
        Z=source_model.engineer_.transform(X)
        np.testing.assert_array_equal(Z.to_numpy(),FeatureEngineer().fit(X).transform(X).to_numpy())
        requested={name:name for name,(_,w) in CONFIGS.items() if w>1}
        requested['selected_model']=choices['nested_overall']
        saved_models={};full_tail={}
        for filename,name in requested.items():
            if name not in full_tail:
                if name=='hgb_w1':tail=source_model.tail_regressor_
                elif name=='lgb_w1':tail=lgb_source_model.tail_regressor_
                else:tail=fit_tail(name,Z.loc[y>10,UNIVERSE],y[y>10])
                full_tail[name]=tail
            model=HierarchicalSFR(source_model.engineer_,source_model.positive_gate_,source_model.conditional_tail_gate_,
                                  source_model.mid_regressor_,full_tail[name])
            path=ROOT/(filename+'.joblib');joblib.dump(model,path,compress=3)
            loaded=joblib.load(path);small=X.iloc[:1000]
            np.testing.assert_array_equal(model.predict(small),loaded.predict(small))
            old_p,old_mid,_=source_model.predict_components(small)
            new_p,new_mid,_=loaded.predict_components(small)
            np.testing.assert_array_equal(old_p,new_p);np.testing.assert_array_equal(old_mid,new_mid)
            saved_models[filename]=dict(candidate=name,file=path.name,sha256=sha(path))
            print('Saved:',filename,name,flush=True)
        loaded=joblib.load(ROOT/'selected_model.joblib')
        reference=df.drop(columns='sfr_linear')
        raw_pred=predict_sfr(reference.iloc[ids[:1000]],loaded,reference)
        np.testing.assert_allclose(raw_pred,loaded.predict(X.iloc[:1000]),rtol=1e-6,atol=1e-9)
    assert sha(source/'hierarchical46.joblib')==source_sha
    assert sha(current_path)==current_sha
    pd.DataFrame(records).to_csv(ROOT/'fold_results.csv',index=False)
    paired=[]
    table=pd.DataFrame(records)
    for seed in (42,7):
        for name in (*CONFIGS,*NESTED):
            baseline='hgb_w1' if name.startswith('hgb') or name=='nested_hgb' else 'lgb_w1'
            if name==baseline:continue
            base=table[(table.seed==seed)&(table.model==baseline)].set_index('fold')
            candidate=table[(table.seed==seed)&(table.model==name)].set_index('fold')
            for fold in range(1,6):
                paired.append(dict(seed=seed,fold=fold,model=name,baseline=baseline,
                    **{k+'_delta':float(candidate.loc[fold,k]-base.loc[fold,k]) for k in ('log_rmse','zero_log_rmse','tail100_log_rmse')}))
    pd.DataFrame(paired).to_csv(ROOT/'paired_differences.csv',index=False)
    metadata=dict(data_git_blob_sha=data_sha,source_fingerprint=fingerprint,source_model_sha256=source_sha,
        previous_model_sha256=sha(tails/'selected_model.joblib'),current_pointer_sha256=current_sha,
        seeds=[42,7],train_rows=len(y),tail_train_rows=int((y>10).sum()),extreme_train_rows=int((y>100).sum()),
        features=UNIVERSE,configs=CONFIGS,tail_threshold=10,weight_threshold=100,zero_threshold=.9,
        target='log10(1+SFR)',final_selection=choices,saved_models=saved_models,
        holdout_evaluated=False,current_model_replaced=False,
        checks=dict(baselines_reproduced=True,outer_coverage_once=True,internal_selection_disjoint=True,
            unweighted_evaluation=True,zero_routing_identical=True,gates_and_mid_fixed=True,
            saved_reload=True,raw_interface=True,source_model_unchanged=True),
        versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','lightgbm','joblib']})
    (ROOT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('WEIGHT EXPERIMENT COMPLETE',flush=True)
    print(pd.DataFrame(summaries)[['seed','model','mean_fold_rmse','zero_log_rmse','tail100_log_rmse']].to_string(index=False),flush=True)

if __name__=='__main__':main()

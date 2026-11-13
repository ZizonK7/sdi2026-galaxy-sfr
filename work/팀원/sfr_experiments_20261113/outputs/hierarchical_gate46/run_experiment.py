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
from feature_pipeline import (UNIVERSE, FeatureEngineer, build_reference_features,
                              fit_component, component_prediction, predict_sfr)
from three_regime import combine
from hierarchical_model import (HierarchicalSFR, hierarchical_probabilities,
                                fit_binary_gates, positive_probability)
from metrics import metrics


def diagnostics(y, pred):
    z=np.log10(1+pred)-np.log10(1+y)
    zero=y==0;tail=y>100
    return dict(zero_pred_above1=int((zero&(pred>1)).sum()),
        severe_zero_sse=float((z[zero&(pred>1)]**2).sum()),
        total_zero_sse=float((z[zero]**2).sum()),
        tail_pred_below10=int((tail&(pred<10)).sum()),
        tail_pred_zero=int((tail&(pred==0)).sum()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--source-dir',type=Path,default=Path(__file__).resolve().parent.parent/'feature_search')
    parser.add_argument('--cache',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    source=args.source_dir
    args.cache.mkdir(parents=True,exist_ok=True)
    old_meta=json.loads((source/'metadata.json').read_text())
    old_model_hash=hashlib.sha256((source/'best_feature_model.joblib').read_bytes()).hexdigest()
    assert old_model_hash==old_meta['model_sha256']
    raw=args.data.read_bytes()
    data_sha=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
    assert data_sha==old_meta['data_git_blob_sha']=='06479cedd323253f08af9983c73f395cc02d3774'
    fingerprint=hashlib.sha256((root/'feature_pipeline.py').read_bytes()+(root/'leader_model.py').read_bytes()+(root/'three_regime.py').read_bytes()+data_sha.encode()).hexdigest()
    assert fingerprint==old_meta['source_fingerprint']
    plan=old_meta['frozen_selection']['plan']
    assert all(cols==UNIVERSE for cols in plan.values())
    df=pd.read_csv(args.data)
    ids,_=train_test_split(np.arange(len(df)),test_size=.2,random_state=42,stratify=(df.sfr_linear>0).astype(int))
    y=df.sfr_linear.to_numpy()[ids]
    records=[];summary=[];paired=[];counts=[]
    with threadpool_limits(limits=4):
        print('Building unchanged reference features...',flush=True)
        X=build_reference_features(df).iloc[ids].reset_index(drop=True)
        for seed in [42,7]:
            old=np.load(source/('stage2_oof.npz' if seed==42 else 'confirmation_oof.npz'))
            before_name=old_meta['frozen_selection']['name'] if seed==42 else 'selected'
            np.testing.assert_array_equal(old['source_row_index'],ids)
            np.testing.assert_array_equal(old['actual'],y)
            preds={name:np.full(len(y),np.nan) for name in ['baseline','zero_gate_only','hierarchical']}
            probs=np.full((len(y),3),np.nan)
            binary_prob=np.full((len(y),2),np.nan)
            coverage=np.zeros(len(y),int)
            for fold,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=seed).split(X),1):
                assert (old['fold'][va]==fold).all()
                coverage[va]+=1
                engineer=FeatureEngineer().fit(X.iloc[tr])
                A,B=engineer.transform(X.iloc[tr]),engineer.transform(X.iloc[va])
                parts={}
                for slot in ['gate','mid','tail']:
                    key=hashlib.sha256(json.dumps([fingerprint,seed,fold,slot,UNIVERSE]).encode()).hexdigest()
                    path=args.cache/(key+'.npz')
                    if not path.exists():
                        component=fit_component(slot,UNIVERSE,A,y[tr])
                        result=component_prediction(component,slot,UNIVERSE,B)
                        np.savez_compressed(path,prediction=result,validation_rows=va)
                    saved=np.load(path)
                    np.testing.assert_array_equal(saved['validation_rows'],va)
                    parts[slot]=saved['prediction']
                baseline=np.maximum(10.**combine(parts['gate'],parts['mid'],parts['tail'])-1.,0.)
                np.testing.assert_allclose(baseline,old[before_name][va],rtol=1e-10,atol=1e-10)
                start=time.monotonic()
                zero_gate,tail_gate=fit_binary_gates(A,y[tr])
                seconds=time.monotonic()-start
                assert zero_gate.n_iter_==tail_gate.n_iter_==400
                p=positive_probability(zero_gate,B)
                q=positive_probability(tail_gate,B)
                joint=hierarchical_probabilities(p,q)
                np.testing.assert_allclose(joint.sum(axis=1),1.)
                old_positive=parts['gate'][:,1]+parts['gate'][:,2]
                q_old=np.divide(parts['gate'][:,2],old_positive,out=np.full(len(va),.5),where=old_positive>0)
                joint_zero_only=hierarchical_probabilities(p,q_old)
                current={
                    'baseline':baseline,
                    'zero_gate_only':np.maximum(10.**combine(joint_zero_only,parts['mid'],parts['tail'])-1.,0.),
                    'hierarchical':np.maximum(10.**combine(joint,parts['mid'],parts['tail'])-1.,0.),
                }
                probs[va]=joint
                binary_prob[va]=np.column_stack([p,q])
                for name,pred in current.items():
                    assert np.isfinite(pred).all() and (pred>=0).all()
                    if name!='baseline':np.testing.assert_array_equal(pred[joint[:,0]>=.9],0.)
                    preds[name][va]=pred
                    records.append(dict(seed=seed,fold=fold,model=name,**metrics(y[va],pred),**diagnostics(y[va],pred)))
                counts.append(dict(seed=seed,fold=fold,zero_gate_train_rows=len(tr),
                    conditional_tail_train_rows=int((y[tr]>0).sum()),classifier_fit_seconds=seconds))
                pd.DataFrame(records).to_csv(root/'fold_results.csv',index=False)
                print(f"seed{seed} fold{fold}: all {records[-3]['log_rmse']:.9f} -> {records[-1]['log_rmse']:.9f}; zero {records[-3]['zero_log_rmse']:.6f} -> {records[-1]['zero_log_rmse']:.6f}; tail100 {records[-3]['tail100_log_rmse']:.6f} -> {records[-1]['tail100_log_rmse']:.6f}; gates fit {seconds:.1f}s",flush=True)
            assert (coverage==1).all()
            table=pd.DataFrame(records)
            for name,pred in preds.items():
                rows=table[(table.seed==seed)&(table.model==name)]
                summary.append(dict(seed=seed,model=name,mean_fold_rmse=float(rows.log_rmse.mean()),
                    **metrics(y,pred),**diagnostics(y,pred)))
            base=table[(table.seed==seed)&(table.model=='baseline')].set_index('fold')
            for name in ['zero_gate_only','hierarchical']:
                rows=table[(table.seed==seed)&(table.model==name)].set_index('fold')
                for fold in range(1,6):
                    paired.append(dict(seed=seed,model=name,fold=fold,**{k+'_delta':float(rows.loc[fold,k]-base.loc[fold,k])
                       for k in ['log_rmse','zero_log_rmse','tail100_log_rmse','zero_recall','zero_precision']}))
            np.savez_compressed(root/f'oof_seed{seed}.npz',actual=y,source_row_index=ids,fold=old['fold'],
                                hierarchical_probs=probs,binary_probabilities=binary_prob,**preds)
            pd.DataFrame(summary).to_csv(root/'summary.csv',index=False)
        print('Fitting two final classifiers; reusing original fitted regressors...',flush=True)
        base=joblib.load(source/'best_feature_model.joblib')
        engineer=FeatureEngineer().fit(X)
        np.testing.assert_array_equal(engineer.transform(X).to_numpy(),base.engineer_.transform(X).to_numpy())
        Z=base.engineer_.transform(X)
        positive_gate,tail_gate=fit_binary_gates(Z,y)
        model=HierarchicalSFR(base.engineer_,positive_gate,tail_gate,base.models_['mid'],base.models_['tail'])
        joblib.dump(model,root/'hierarchical46.joblib',compress=3)
        loaded=joblib.load(root/'hierarchical46.joblib')
        sample=X.iloc[:1000]
        np.testing.assert_array_equal(loaded.predict(sample),model.predict(sample))
        _,mid,tail=loaded.predict_components(sample)
        _,old_mid,old_tail=base.predict_components(sample)
        np.testing.assert_array_equal(mid,old_mid);np.testing.assert_array_equal(tail,old_tail)
        reference=df.drop(columns='sfr_linear')
        raw_pred=predict_sfr(reference.iloc[ids[:1000]],loaded,reference)
        np.testing.assert_allclose(raw_pred,loaded.predict(sample),rtol=1e-6,atol=1e-9)
    assert hashlib.sha256((source/'best_feature_model.joblib').read_bytes()).hexdigest()==old_model_hash
    pd.DataFrame(paired).to_csv(root/'paired_differences.csv',index=False)
    pd.DataFrame(counts).to_csv(root/'fit_counts.csv',index=False)
    metadata=dict(source_model_sha256=old_model_hash,source_fingerprint=fingerprint,data_git_blob_sha=data_sha,
        source_commit=old_meta['source_commit'],features=UNIVERSE,train_rows=len(y),
        tail_threshold=10.,zero_threshold=.9,seeds=[42,7],holdout_evaluated=False,
        current_model_replaced=False,regression_experts_reused_unchanged=True,
        model_sha256=hashlib.sha256((root/'hierarchical46.joblib').read_bytes()).hexdigest(),
        checks=dict(oof_coverage_once_per_seed=True,original_predictions_reconstructed=True,
                    conditional_gate_true_positive_training=True,saved_reload=True,experts_exact=True,raw_interface=True),
        versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','joblib']})
    (root/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print('EXPERIMENT COMPLETE',flush=True)


if __name__=='__main__':
    main()

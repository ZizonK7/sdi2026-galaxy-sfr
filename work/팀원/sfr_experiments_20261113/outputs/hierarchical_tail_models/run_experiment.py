import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,train_test_split
from threadpoolctl import threadpool_limits
from feature_pipeline import UNIVERSE,FeatureEngineer,build_reference_features,fit_component,component_prediction,predict_sfr
from three_regime import combine
from hierarchical_model import HierarchicalSFR
from tail_models import CANDIDATES,make_tail
from metrics import metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--source-dir',type=Path,default=Path(__file__).resolve().parent.parent/'hierarchical_gate46')
    parser.add_argument('--cache',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent;source=args.source_dir
    args.cache.mkdir(parents=True,exist_ok=True)
    meta_source=json.loads((source/'metadata.json').read_text())
    old_hash=hashlib.sha256((source/'hierarchical46.joblib').read_bytes()).hexdigest()
    assert old_hash==meta_source['model_sha256']
    raw=args.data.read_bytes();data_sha=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
    assert data_sha==meta_source['data_git_blob_sha']=='06479cedd323253f08af9983c73f395cc02d3774'
    fingerprint=hashlib.sha256((root/'feature_pipeline.py').read_bytes()+(root/'leader_model.py').read_bytes()+(root/'three_regime.py').read_bytes()+data_sha.encode()).hexdigest()
    assert fingerprint==meta_source['source_fingerprint']
    df=pd.read_csv(args.data)
    ids,_=train_test_split(np.arange(len(df)),test_size=.2,random_state=42,stratify=(df.sfr_linear>0).astype(int))
    y=df.sfr_linear.to_numpy()[ids]
    records=[];summaries=[];fits=[];frozen=None;selected_tail=None
    names=['hgb400_baseline']+list(CANDIDATES)
    with threadpool_limits(limits=4):
        print('Building unchanged input-only reference features...',flush=True)
        X=build_reference_features(df).iloc[ids].reset_index(drop=True)
        for seed in [42,7]:
            old=np.load(source/f'oof_seed{seed}.npz')
            np.testing.assert_array_equal(ids,old['source_row_index']);np.testing.assert_array_equal(y,old['actual'])
            p=old['hierarchical_probs']
            np.testing.assert_allclose(p.sum(axis=1),1.)
            predicted={name:np.full(len(y),np.nan) for name in names}
            tail_logs={name:np.full(len(y),np.nan) for name in names}
            coverage=np.zeros(len(y),int)
            for fold,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=seed).split(X),1):
                assert (old['fold'][va]==fold).all()
                coverage[va]+=1
                engineer=FeatureEngineer().fit(X.iloc[tr])
                A,B=engineer.transform(X.iloc[tr]),engineer.transform(X.iloc[va])
                parts={}
                for slot in ['mid','tail']:
                    key=hashlib.sha256(json.dumps([fingerprint,seed,fold,slot,UNIVERSE]).encode()).hexdigest()
                    path=args.cache/(key+'.npz')
                    if not path.exists():
                        m=fit_component(slot,UNIVERSE,A,y[tr])
                        np.savez_compressed(path,prediction=component_prediction(m,slot,UNIVERSE,B),validation_rows=va)
                    saved=np.load(path);np.testing.assert_array_equal(saved['validation_rows'],va)
                    parts[slot]=saved['prediction']
                baseline=np.maximum(10.**combine(p[va],parts['mid'],parts['tail'])-1.,0.)
                np.testing.assert_allclose(baseline,old['hierarchical'][va],rtol=1e-10,atol=1e-10)
                mask=y[tr]>10
                for name in names:
                    if name=='hgb400_baseline':
                        log_tail=parts['tail'];pred=baseline;seconds=0.
                    else:
                        start=time.monotonic()
                        model=make_tail(name)
                        model.fit(A.loc[mask,UNIVERSE],np.log10(1+y[tr][mask]))
                        seconds=time.monotonic()-start
                        log_tail=model.predict(B[UNIVERSE])
                        pred=np.maximum(10.**combine(p[va],parts['mid'],log_tail)-1.,0.)
                        del model
                        fits.append(dict(seed=seed,fold=fold,candidate=name,training_rows=int(mask.sum()),seconds=seconds))
                    assert np.isfinite(pred).all() and (pred>=0).all()
                    np.testing.assert_array_equal(pred[p[va,0]>=.9],0.)
                    predicted[name][va]=pred;tail_logs[name][va]=log_tail
                    row=dict(seed=seed,fold=fold,model=name,**metrics(y[va],pred))
                    records.append(row)
                    print(f"seed{seed} fold{fold} {name}: all={row['log_rmse']:.9f} tail100={row['tail100_log_rmse']:.6f} zero={row['zero_log_rmse']:.6f} fit={seconds:.1f}s",flush=True)
                    pd.DataFrame(records).to_csv(root/'fold_results.csv',index=False)
                    pd.DataFrame(fits).to_csv(root/'fit_records.csv',index=False)
            assert (coverage==1).all()
            table=pd.DataFrame(records)
            for name in names:
                pred=predicted[name];assert np.isfinite(pred).all()
                rows=table[(table.seed==seed)&(table.model==name)]
                extreme=y>100
                expert_rmse=float(np.sqrt(np.mean((np.maximum(tail_logs[name][extreme],0)-np.log10(1+y[extreme]))**2)))
                summaries.append(dict(seed=seed,model=name,mean_fold_rmse=float(rows.log_rmse.mean()),
                    extreme_expert_only_diagnostic_rmse=expert_rmse,**metrics(y,pred)))
            summary=pd.DataFrame(summaries)
            summary.to_csv(root/'summary.csv',index=False)
            np.savez_compressed(root/f'oof_seed{seed}.npz',actual=y,source_row_index=ids,fold=old['fold'],**predicted)
            np.savez_compressed(root/f'tail_components_seed{seed}.npz',**tail_logs)
            if seed==42:
                rank=summary[summary.seed==42].sort_values(['mean_fold_rmse','model'])
                frozen=rank.iloc[0].model
                selected_tail=summary[summary.seed==42].sort_values(['tail100_log_rmse','mean_fold_rmse','model']).iloc[0].model
                choice=dict(overall_winner=frozen,extreme_tail_winner=selected_tail,
                    selected_on_seed42_only=True,reselection_on_seed7=False,candidates=CANDIDATES)
                (root/'frozen_selection.json').write_text(json.dumps(choice,indent=2)+'\n')
                print('FROZEN SELECTION:',json.dumps(choice),flush=True)
        source_model=joblib.load(source/'hierarchical46.joblib')
        engineer=FeatureEngineer().fit(X)
        np.testing.assert_array_equal(engineer.transform(X).to_numpy(),source_model.engineer_.transform(X).to_numpy())
        Z=source_model.engineer_.transform(X)
        saved_models={}
        requested={'selected_model':frozen}
        if selected_tail!=frozen:requested['extreme_tail_option']=selected_tail
        for filename,name in requested.items():
            print('Final fitted artifact:',filename,name,flush=True)
            if name=='hgb400_baseline':new_tail=source_model.tail_regressor_
            else:
                new_tail=make_tail(name).fit(Z.loc[y>10,UNIVERSE],np.log10(1+y[y>10]))
            model=HierarchicalSFR(source_model.engineer_,source_model.positive_gate_,source_model.conditional_tail_gate_,
                                  source_model.mid_regressor_,new_tail)
            path=root/(filename+'.joblib')
            joblib.dump(model,path,compress=3)
            loaded=joblib.load(path)
            small=X.iloc[:1000]
            np.testing.assert_array_equal(model.predict(small),loaded.predict(small))
            old_p,old_mid,_=source_model.predict_components(small)
            new_p,new_mid,_=loaded.predict_components(small)
            np.testing.assert_array_equal(old_p,new_p);np.testing.assert_array_equal(old_mid,new_mid)
            saved_models[filename]=dict(candidate=name,file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        loaded=joblib.load(root/'selected_model.joblib')
        reference=df.drop(columns='sfr_linear')
        raw_pred=predict_sfr(reference.iloc[ids[:1000]],loaded,reference)
        np.testing.assert_allclose(raw_pred,loaded.predict(X.iloc[:1000]),rtol=1e-6,atol=1e-9)
    assert hashlib.sha256((source/'hierarchical46.joblib').read_bytes()).hexdigest()==old_hash
    paired=[];table=pd.DataFrame(records)
    for seed in [42,7]:
        base=table[(table.seed==seed)&(table.model=='hgb400_baseline')].set_index('fold')
        for name in CANDIDATES:
            rows=table[(table.seed==seed)&(table.model==name)].set_index('fold')
            for fold in range(1,6):
                paired.append(dict(seed=seed,model=name,fold=fold,**{k+'_delta':float(rows.loc[fold,k]-base.loc[fold,k])
                    for k in ['log_rmse','zero_log_rmse','tail100_log_rmse']}))
    pd.DataFrame(paired).to_csv(root/'paired_differences.csv',index=False)
    meta=dict(source_model_sha256=old_hash,source_fingerprint=fingerprint,data_git_blob_sha=data_sha,
        seeds=[42,7],train_rows=len(y),tail_train_rows=int((y>10).sum()),features=UNIVERSE,
        tail_threshold=10.,zero_threshold=.9,selected_model=frozen,extreme_tail_option=selected_tail,
        saved_models=saved_models,holdout_evaluated=False,current_model_replaced=False,
        checks=dict(baseline_predictions_reconstructed=True,oof_coverage_once=True,
            gate_probabilities_fixed=True,mid_regressor_fixed=True,source_model_unchanged=True,
            saved_reload=True,raw_interface=True),
        versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','lightgbm','catboost','joblib']})
    (root/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print('TAIL EXPERIMENT COMPLETE',flush=True)
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)


if __name__=='__main__':main()

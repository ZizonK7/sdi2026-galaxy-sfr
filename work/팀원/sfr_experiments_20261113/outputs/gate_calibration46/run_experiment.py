import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import brier_score_loss, log_loss
from threadpoolctl import threadpool_limits
from feature_pipeline import UNIVERSE, FeatureEngineer, build_reference_features, fit_component, predict_sfr
from hierarchical_model import fit_binary_gates, positive_probability, hierarchical_probabilities
from weighted_tail import fit_tail
from calibration_model import VARIANTS,fit_maps,apply_maps,CalibratedSFR
from three_regime import combine
from metrics import metrics

ROOT=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def crossfit_probabilities(X,y,rows,label,cache,identity,fit_records):
    key=hashlib.sha256(identity.encode()+rows.tobytes()).hexdigest()
    path=cache/(key+'.npz')
    if path.exists():
        saved=np.load(path);np.testing.assert_array_equal(saved['training_rows'],rows)
        print(label,'inner probabilities restored from cache',flush=True)
        return saved['probabilities'],saved['inner_fold'],key
    result=np.full((len(rows),2),np.nan);coverage=np.zeros(len(rows),int);inner_fold=np.zeros(len(rows),int)
    strata=np.where(y[rows]==0,0,np.where(y[rows]<=10,1,2))
    for f,(it,iv) in enumerate(StratifiedKFold(3,shuffle=True,random_state=314).split(rows,strata),1):
        start=time.monotonic();tr,va=rows[it],rows[iv]
        assert not np.intersect1d(tr,va).size
        eng=FeatureEngineer().fit(X.iloc[tr])
        A,B=eng.transform(X.iloc[tr]),eng.transform(X.iloc[va])
        g0,gt=fit_binary_gates(A,y[tr])
        result[iv,0]=positive_probability(g0,B)
        result[iv,1]=positive_probability(gt,B)
        coverage[iv]+=1;inner_fold[iv]=f
        fit_records.append(dict(split=label,inner_fold=f,train_rows=len(tr),validation_rows=len(va),
            positive_train_rows=int((y[tr]>0).sum()),seconds=time.monotonic()-start,
            training_rows_sha256=hashlib.sha256(tr.tobytes()).hexdigest(),
            validation_rows_sha256=hashlib.sha256(va.tobytes()).hexdigest()))
        pd.DataFrame(fit_records).to_csv(ROOT/'inner_fit_records.csv',index=False)
        print(f'{label} inner{f}/3 complete ({time.monotonic()-start:.1f}s)',flush=True)
        del g0,gt,A,B
    assert (coverage==1).all() and np.isfinite(result).all() and (result>=0).all() and (result<=1).all()
    np.savez_compressed(path,training_rows=rows,probabilities=result,inner_fold=inner_fold)
    return result,inner_fold,key

def diagnostics(y,prob,pred):
    positive=y>0
    return dict(positive_brier=float(brier_score_loss(positive,prob[:,0])),
        positive_log_loss=float(log_loss(positive,prob[:,0],labels=[0,1])),
        conditional_tail_brier=float(brier_score_loss(y[positive]>10,prob[positive,1])),
        conditional_tail_log_loss=float(log_loss(y[positive]>10,prob[positive,1],labels=[0,1])),
        forced_zero_rows=int((1-prob[:,0]>=.9).sum()),
        forced_zero_positives=int(((1-prob[:,0]>=.9)&positive).sum()),
        forced_zero_extremes=int(((1-prob[:,0]>=.9)&(y>100)).sum()),
        zero_outputs_extremes=int(((pred==0)&(y>100)).sum()))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=ROOT.parents[1]/'work/gate_calibration_cache')
    parser.add_argument('--component-cache',type=Path,default=ROOT.parents[1]/'work/feature_search_cache')
    args=parser.parse_args();args.cache.mkdir(parents=True,exist_ok=True)
    blend_source=ROOT.parent/'tail_blend_constrained'
    gate_source=ROOT.parent/'hierarchical_gate46'
    weight_source=ROOT.parent/'hierarchical_tail_weighting'
    meta_blend=json.loads((blend_source/'balanced_candidate.json').read_text())
    meta_gate=json.loads((gate_source/'metadata.json').read_text())
    baseline_path=blend_source/'balanced_model.joblib';base_sha=sha(baseline_path)
    assert base_sha==meta_blend['sha256']
    pointer_sha=sha(ROOT.parent/'current_model.json')
    raw=args.data.read_bytes();data_sha=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
    assert data_sha==meta_gate['data_git_blob_sha']
    fingerprint=hashlib.sha256((ROOT/'feature_pipeline.py').read_bytes()+(ROOT/'leader_model.py').read_bytes()+(ROOT/'three_regime.py').read_bytes()+data_sha.encode()).hexdigest()
    assert fingerprint==meta_gate['source_fingerprint']
    identity=hashlib.sha256((ROOT/'run_experiment.py').read_bytes()+(ROOT/'hierarchical_model.py').read_bytes()+fingerprint.encode()).hexdigest()
    df=pd.read_csv(args.data)
    ids,_=train_test_split(np.arange(len(df)),test_size=.2,random_state=42,stratify=(df.sfr_linear>0).astype(int))
    y=df.sfr_linear.to_numpy()[ids]
    fit_records=[];audit=[];records=[];summaries=[];diagnostic_rows=[]
    with threadpool_limits(limits=4):
        print('Building unchanged reference features...',flush=True)
        X=build_reference_features(df).iloc[ids].reset_index(drop=True)
        for seed in (42,7,99):
            old=np.load(blend_source/f'oof_seed{seed}.npz')
            np.testing.assert_array_equal(ids,old['source_row_index']);np.testing.assert_array_equal(y,old['actual'])
            if seed in (42,7):
                raw_gates=np.load(gate_source/f'oof_seed{seed}.npz')
                tails=np.load(weight_source/f'tail_components_seed{seed}.npz')
            predictions={v:np.full(len(y),np.nan) for v in VARIANTS}
            probabilities={v:np.full((len(y),2),np.nan) for v in VARIANTS}
            mid_logs=np.full(len(y),np.nan);tail_logs=np.full(len(y),np.nan)
            coverage=np.zeros(len(y),int);maps_by_fold={}
            for fold,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=seed).split(X),1):
                label=f'seed{seed}_fold{fold}';assert (old['fold'][va]==fold).all()
                coverage[va]+=1
                inner,inner_fold,key=crossfit_probabilities(X,y,tr,label,args.cache,identity,fit_records)
                maps=fit_maps(inner,y[tr]);maps_by_fold[fold]=maps
                audit.append(dict(split=label,cache_key=key,training_rows=len(tr),
                    positive_calibration_rows=int((y[tr]>0).sum()),
                    train_rows_sha256=hashlib.sha256(tr.tobytes()).hexdigest(),
                    validation_rows_sha256=hashlib.sha256(va.tobytes()).hexdigest()))
                if seed in (42,7):
                    raw_prob=raw_gates['binary_probabilities'][va]
                    mid_key=hashlib.sha256(json.dumps([fingerprint,seed,fold,'mid',UNIVERSE]).encode()).hexdigest()
                    cached=np.load(args.component_cache/(mid_key+'.npz'))
                    np.testing.assert_array_equal(cached['validation_rows'],va)
                    mid=cached['prediction'];tail=.5*(tails['lgb_w1'][va]+tails['lgb_w3'][va])
                else:
                    component_path=args.cache/f'outer99_{fold}_{identity}.npz'
                    if component_path.exists():
                        cached=np.load(component_path);np.testing.assert_array_equal(cached['validation_rows'],va)
                        raw_prob,mid,tail=cached['probabilities'],cached['mid'],cached['tail']
                    else:
                        eng=FeatureEngineer().fit(X.iloc[tr])
                        A,B=eng.transform(X.iloc[tr]),eng.transform(X.iloc[va])
                        g0,gt=fit_binary_gates(A,y[tr])
                        raw_prob=np.column_stack([positive_probability(g0,B),positive_probability(gt,B)])
                        mid=fit_component('mid',UNIVERSE,A,y[tr]).predict(B[UNIVERSE])
                        mask=y[tr]>10
                        t1=fit_tail('lgb_w1',A.loc[mask,UNIVERSE],y[tr][mask]).predict(B[UNIVERSE])
                        t3=fit_tail('lgb_w3',A.loc[mask,UNIVERSE],y[tr][mask]).predict(B[UNIVERSE])
                        tail=.5*(t1+t3)
                        np.savez_compressed(component_path,validation_rows=va,probabilities=raw_prob,mid=mid,tail=tail)
                mid_logs[va]=mid;tail_logs[va]=tail
                for variant in VARIANTS:
                    prob=apply_maps(raw_prob,maps,variant)
                    p=hierarchical_probabilities(prob[:,0],prob[:,1])
                    pred=np.maximum(10.**combine(p,mid,tail)-1.,0.)
                    assert np.isfinite(pred).all() and (pred>=0).all()
                    np.testing.assert_array_equal(pred[p[:,0]>=.9],0.)
                    if variant=='baseline':np.testing.assert_allclose(pred,old['strict_balance'][va],rtol=1e-9,atol=1e-9)
                    predictions[variant][va]=pred;probabilities[variant][va]=prob
                    result=metrics(y[va],pred)
                    records.append(dict(seed=seed,fold=fold,model=variant,**result))
                pd.DataFrame(records).to_csv(ROOT/'fold_results.csv',index=False)
                pd.DataFrame(audit).to_csv(ROOT/'calibration_audit.csv',index=False)
                print(label,'outer evaluation complete',flush=True)
            assert (coverage==1).all()
            frame=pd.DataFrame(records)
            for variant in VARIANTS:
                scores=frame[(frame.seed==seed)&(frame.model==variant)]
                result=dict(seed=seed,model=variant,mean_fold_rmse=float(scores.log_rmse.mean()),**metrics(y,predictions[variant]))
                summaries.append(result)
                diagnostic_rows.append(dict(seed=seed,model=variant,**diagnostics(y,probabilities[variant],predictions[variant])))
            pd.DataFrame(summaries).to_csv(ROOT/'summary.csv',index=False)
            pd.DataFrame(diagnostic_rows).to_csv(ROOT/'probability_diagnostics.csv',index=False)
            np.savez_compressed(ROOT/f'oof_seed{seed}.npz',actual=y,source_row_index=ids,fold=old['fold'],**predictions)
            np.savez_compressed(ROOT/f'components_seed{seed}.npz',raw_probabilities=probabilities['baseline'],mid_log=mid_logs,tail_log=tail_logs)
            joblib.dump(maps_by_fold,ROOT/f'fold_calibrators_seed{seed}.joblib',compress=3)
            print('SEED COMPLETE',seed,flush=True)
            print(pd.DataFrame(summaries).query('seed==@seed')[['model','mean_fold_rmse','zero_log_rmse','tail100_log_rmse']].to_string(index=False),flush=True)
        print('Fitting full-train cross-fitted calibration maps...',flush=True)
        full_rows=np.arange(len(y))
        inner,inner_fold,key=crossfit_probabilities(X,y,full_rows,'full_train80',args.cache,identity,fit_records)
        maps=fit_maps(inner,y);joblib.dump(maps,ROOT/'full_calibration_maps.joblib',compress=3)
        audit.append(dict(split='full_train80',cache_key=key,training_rows=len(y),
            positive_calibration_rows=int((y>0).sum()),train_rows_sha256=hashlib.sha256(full_rows.tobytes()).hexdigest(),validation_rows_sha256=None))
        pd.DataFrame(audit).to_csv(ROOT/'calibration_audit.csv',index=False)
        base_model=joblib.load(baseline_path);small=X.iloc[:1000]
        _,base_mid,base_tail=base_model.predict_components(small)
        models={}
        for variant in VARIANTS:
            if variant=='baseline':continue
            model=CalibratedSFR(base_model,maps,variant)
            p,mid,tail=model.predict_components(small)
            np.testing.assert_array_equal(mid,base_mid);np.testing.assert_array_equal(tail,base_tail)
            np.testing.assert_allclose(p.sum(axis=1),1.)
            path=ROOT/(variant+'.joblib');joblib.dump(model,path,compress=3)
            restored=joblib.load(path);np.testing.assert_array_equal(model.predict(small),restored.predict(small))
            models[variant]=dict(file=path.name,sha256=sha(path))
        model=joblib.load(ROOT/'sigmoid_both.joblib')
        reference=df.drop(columns='sfr_linear')
        result=predict_sfr(reference.iloc[ids[:1000]],model,reference)
        np.testing.assert_allclose(result,model.predict(small),rtol=1e-6,atol=1e-9)
    assert sha(baseline_path)==base_sha and sha(ROOT.parent/'current_model.json')==pointer_sha
    metadata=dict(source_model_sha256=base_sha,data_git_blob_sha=data_sha,source_fingerprint=fingerprint,
        cache_identity=identity,current_pointer_sha256=pointer_sha,features=UNIVERSE,
        train_rows=len(y),tail_rows=int((y>10).sum()),extreme_rows=int((y>100).sum()),seeds=[42,7,99],
        variants=VARIANTS,inner_folds=3,inner_seed=314,saved_models=models,
        holdout_scored=False,current_model_replaced=False,variant_selected_on_outer_labels=False,
        checks=dict(baseline_reconstructed=True,inner_coverage_once=True,inner_outer_disjoint=True,
            true_positive_conditional_calibration=True,fixed_expert_predictions=True,
            calibrated_zero_rule=True,saved_reload=True,raw_interface=True,source_model_unchanged=True),
        versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','lightgbm','joblib']})
    (ROOT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('CALIBRATION EXPERIMENT COMPLETE',flush=True)

if __name__=='__main__':main()

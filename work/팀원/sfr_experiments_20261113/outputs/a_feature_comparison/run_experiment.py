from pathlib import Path
import os, json, argparse, hashlib, time, warnings
os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
import numpy as np
import pandas as pd
import sklearn, scipy
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from a_log_mixture import LogTailMixtureSFR
from features import RAW, build_feature_sets


def rmse(y,p):
    return float(np.sqrt(np.mean((np.log10(1+p)-np.log10(1+y))**2)))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True)
    ap.add_argument('--references',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    start=time.time();b=Path(args.data).read_bytes()
    sha=hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()
    sha256=hashlib.sha256(b).hexdigest();del b
    assert sha=='06479cedd323253f08af9983c73f395cc02d3774'
    df=pd.read_csv(args.data)
    assert df.subhalo_id.notna().all() and df.subhalo_id.is_unique
    # Only geometry/observables from the full reference catalogue; never targets.
    with threadpool_limits(limits=4):
        sets=build_feature_sets(df[RAW])
    print(f'Built feature sets from {len(df):,} unlabeled rows in {time.time()-start:.1f}s',flush=True)
    train_indices,holdout_indices=train_test_split(np.arange(len(df)),test_size=.2,
        random_state=42,stratify=(df.sfr_linear>0).astype(int))
    y=df.sfr_linear.to_numpy()[train_indices]
    ref=np.load(args.references)
    assert np.array_equal(train_indices,ref['source_row_index'])
    assert np.array_equal(y,ref['actual'])
    folds=ref['fold'].copy()
    inputs={name:frame.iloc[train_indices].to_numpy() for name,frame in sets.items()}
    preds={name:np.zeros(len(y)) for name in sets}
    probs={name:np.zeros((len(y),3)) for name in sets}
    details=[]
    with threadpool_limits(limits=4):
        for k,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=42).split(y),1):
            assert np.all(folds[te]==k)
            for name,X in inputs.items():
                t=time.time()
                m=LogTailMixtureSFR(tail_threshold=10.,log_offset=1.)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    m.fit(X[tr],y[tr])
                    p,zm,zt=m.parts(X[te])
                pred=m.combine(p,zm,zt,1.)
                assert np.isfinite(pred).all() and (pred>=0).all()
                assert np.allclose(p.sum(axis=1),1.)
                if name=='A13_baseline':
                    assert np.allclose(pred,ref['A_log1p_T10_soft'][te],rtol=1e-9,atol=1e-9)
                preds[name][te]=pred;probs[name][te]=p
                details.append(dict(fold=k,model=name,n_features=X.shape[1],seconds=time.time()-t,
                    classifier_trees=m.classifier_.n_iter_,mid_trees=m.mid_regressor_.n_iter_,
                    tail_trees=m.tail_regressor_.n_iter_,class_counts=m.class_counts_.tolist(),
                    warnings=[str(w.message) for w in caught]))
                print(f'fold {k}/5 {name}: {rmse(y[te],pred):.6f}, {time.time()-t:.1f}s',flush=True)
            (out/'fit_details.json').write_text(json.dumps(details,indent=2))
    np.savez_compressed(out/'oof_predictions.npz',actual=y,fold=folds,source_row_index=train_indices,**preds)
    np.savez_compressed(out/'oof_probabilities.npz',actual=y,fold=folds,source_row_index=train_indices,**probs)
    np.savez_compressed(out/'baseline_reference.npz',actual=y,fold=folds,
        source_row_index=train_indices,A_log1p_T10_soft=ref['A_log1p_T10_soft'])
    segments={'All':np.ones(len(y),bool),'SFR=0':y==0,'0<SFR<=3':(y>0)&(y<=3),
        '3<SFR<=10':(y>3)&(y<=10),'10<SFR<=30':(y>10)&(y<=30),
        '30<SFR<=100':(y>30)&(y<=100),'SFR>100':y>100}
    rows=[];fr=[];gates=[]
    truthclasses=np.where(y==0,0,np.where(y<=10,1,2))
    for name,pred in preds.items():
        for seg,mask in segments.items():
            rows.append(dict(model=name,segment=seg,n=int(mask.sum()),rmse=rmse(y[mask],pred[mask])))
        for k in range(1,6):
            mask=folds==k
            fr.append(dict(model=name,fold=k,rmse=rmse(y[mask],pred[mask])))
        gates.append(dict(model=name,log_loss=float(log_loss(truthclasses,probs[name],labels=[0,1,2])),
            accuracy=float((probs[name].argmax(axis=1)==truthclasses).mean())))
    scores=pd.DataFrame(rows);fs=pd.DataFrame(fr)
    tab=scores.query("segment=='All'").copy().sort_values('rmse')
    base=rmse(y,preds['A13_baseline'])
    tab['improvement_pct']=100*(base-tab.rmse)/base
    pivot=fs.pivot(index='fold',columns='model',values='rmse')
    tab['fold_wins_vs_baseline']=[int((pivot[n]<pivot.A13_baseline).sum()) for n in tab.model]
    scores.to_csv(out/'scores.csv',index=False);fs.to_csv(out/'fold_scores.csv',index=False)
    tab.to_csv(out/'overall_comparison.csv',index=False)
    pd.DataFrame(gates).to_csv(out/'gate_diagnostics.csv',index=False)
    interaction=[]
    for k in range(1,6):
        mask=folds==k
        losses={n:rmse(y[mask],p[mask])**2 for n,p in preds.items()}
        interaction.append(dict(fold=k,mse_interaction=losses['A34_both']-losses['A30_environment']-losses['A17_mass']+losses['A13_baseline']))
    pd.DataFrame(interaction).to_csv(out/'feature_interaction.csv',index=False)
    meta=dict(data_git_blob_sha=sha,data_sha256=sha256,
        leader_commit='6bc3988dd7effb7694192ce059f515fa496fa499',
        leader_source='work/조장/log_model_cv_2026-11-11/log_model_cv_env.py',
        catalogue_rows=len(df),train_rows=len(y),holdout_rows=len(holdout_indices),
        holdout_targets_evaluated=False,baseline_reproduced=True,
        reference_policy='Fixed complete 208437-row unique catalogue; geometry uses unlabeled validation/holdout features as allowed by FAQ Q2. Fold labels are never used in feature construction.',
        metric='RMSE log10(1+SFR); same convention as previous A experiments; not asserted as confirmed official zero transform',
        model='A: 3-class HGB gate + two log-target HGB regressors; soft mixture in log space',
        tail_threshold=10.,log_offset=1.,hgb_params=dict(learning_rate=.08,max_iter=150,max_leaf_nodes=31,random_state=42),
        preprocessing='FoF singleton velocity remains NaN; HGB native missing-value routing; no imputation. Mass sentinel values preserved as in leader source.',
        cv='Original train split; KFold(5, shuffle=True, random_state=42)',
        feature_sets={n:list(s.columns) for n,s in sets.items()},
        catalogue_nan_counts={n:{c:int(v) for c,v in s.isna().sum().items() if v} for n,s in sets.items()},
        versions=dict(numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        elapsed_seconds=time.time()-start)
    (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print(tab.to_string(index=False),flush=True)


if __name__=='__main__':
    main()

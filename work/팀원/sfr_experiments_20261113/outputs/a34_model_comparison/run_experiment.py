from pathlib import Path
import os, json, argparse, hashlib, time, warnings, copy
os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
import numpy as np
import pandas as pd
import sklearn, scipy, catboost, lightgbm, joblib
from sklearn.model_selection import train_test_split,KFold
from threadpoolctl import threadpool_limits
from features import RAW,build_feature_sets
from models import BASE,GRID,FAMILIES,labels,fit_part,part,mix,score,FixedMixture,HGB


def tune(X,y,seed,tag,records):
    """Only this inner split selects configurations, families, and the combination."""
    a,b=train_test_split(np.arange(len(y)),test_size=.2,random_state=seed,stratify=labels(y))
    assert not np.intersect1d(a,b).size
    baseparts={}
    for s in BASE:
        m=fit_part(s,BASE[s],X[a],y[a]);baseparts[s]=part(m,s,X[b])
    baseline=score(y[b],mix(baseparts))
    best={};innerparts={};inner_scores={}
    for slot,families in FAMILIES.items():
        for family in families:
            key=f'{family}_{slot}';options=[]
            for params in GRID[family]:
                spec=dict(family=family,params=params);t=time.time()
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter('always');m=fit_part(slot,spec,X[a],y[a])
                    p=part(m,slot,X[b])
                sc=score(y[b],mix({**baseparts,slot:p}))
                records.append(dict(stage=tag,slot=slot,spec=spec,inner_rmse=sc,
                    inner_baseline=baseline,seconds=time.time()-t,
                    warnings=[str(w.message) for w in ws]))
                options.append((sc,spec,p))
                print(f'  {tag} {key} {params}: inner={sc:.6f}, {time.time()-t:.1f}s',flush=True)
            sc,spec,p=min(options,key=lambda z:z[0])
            best[key]=spec;innerparts[key]=p;inner_scores[key]=sc
    combo=copy.deepcopy(BASE);combined_parts=baseparts.copy();selected_keys={}
    for slot,families in FAMILIES.items():
        key=min([f'{family}_{slot}' for family in families],key=inner_scores.get)
        if inner_scores[key]<baseline:
            combo[slot]=best[key];combined_parts[slot]=innerparts[key];selected_keys[slot]=key
    combo_score=score(y[b],mix(combined_parts))
    # Combining individually improving components can interact adversely.
    # Predeclared safeguard: if inner mixture worsens, keep the best single swap.
    fallback=False
    if combo_score>baseline:
        fallback=True;combo=copy.deepcopy(BASE);selected_keys={}
        key=min(inner_scores,key=inner_scores.get)
        if inner_scores[key]<baseline:
            slot=key.split('_')[1];combo[slot]=best[key];selected_keys[slot]=key
            combo_score=inner_scores[key]
        else:combo_score=baseline
    selection=dict(stage=tag,inner_split_seed=seed,inner_train_n=len(a),inner_valid_n=len(b),
        baseline_rmse=baseline,best_specs=best,inner_scores=inner_scores,combo_specs=combo,
        combo_keys=selected_keys,combo_rmse=combo_score,combo_fallback=fallback)
    return best,selection


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True)
    ap.add_argument('--reference',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--fit-final',action='store_true');args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    started=time.time();raw=Path(args.data).read_bytes()
    sha=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest();del raw
    assert sha=='06479cedd323253f08af9983c73f395cc02d3774'
    df=pd.read_csv(args.data)
    with threadpool_limits(limits=4):Xfull=build_feature_sets(df[RAW])['A34_both']
    ix,_=train_test_split(np.arange(len(df)),test_size=.2,random_state=42,stratify=(df.sfr_linear>0).astype(int))
    X=Xfull.iloc[ix].to_numpy();y=df.sfr_linear.to_numpy()[ix]
    ref=np.load(args.reference)
    assert np.array_equal(ix,ref['source_row_index']) and np.array_equal(y,ref['actual'])
    fold=ref['fold'];names=['A34_HGB']+[f'{fam}_{s}' for s,fs in FAMILIES.items() for fam in fs]+['C_inner_selected']
    predictions={n:np.zeros(len(y)) for n in names};records=[];selections=[]
    oofparts={f'hgb_{s}':np.zeros((len(y),3)) if s=='gate' else np.zeros(len(y)) for s in BASE}
    for s,fs in FAMILIES.items():
        for fam in fs:oofparts[f'{fam}_{s}']=np.zeros((len(y),3)) if s=='gate' else np.zeros(len(y))
    with threadpool_limits(limits=4):
        for k,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=42).split(X),1):
            assert np.all(fold[te]==k)
            print(f'OUTER FOLD {k}/5: tuning only inside {len(tr):,} training rows',flush=True)
            best,sel=tune(X[tr],y[tr],1000+k,f'fold{k}',records);selections.append(sel)
            baseparts={};models={}
            for slot,spec in BASE.items():
                model=fit_part(slot,spec,X[tr],y[tr]);p=part(model,slot,X[te])
                baseparts[slot]=p;oofparts[f'hgb_{slot}'][te]=p
            predictions['A34_HGB'][te]=mix(baseparts)
            np.testing.assert_allclose(predictions['A34_HGB'][te],ref['A34_both'][te],rtol=1e-9,atol=1e-9)
            candidate_parts={}
            for key,spec in best.items():
                slot=key.split('_')[1];t=time.time()
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter('always');model=fit_part(slot,spec,X[tr],y[tr])
                    p=part(model,slot,X[te])
                candidate_parts[key]=p;oofparts[key][te]=p
                predictions[key][te]=mix({**baseparts,slot:p})
                records.append(dict(stage=f'fold{k}_refit',slot=slot,spec=spec,seconds=time.time()-t,
                    warnings=[str(w.message) for w in ws]))
            combo=baseparts.copy()
            for slot,key in sel['combo_keys'].items():combo[slot]=candidate_parts[key]
            predictions['C_inner_selected'][te]=mix(combo)
            print('OUTER scores '+', '.join(f'{n}={score(y[te],p[te]):.6f}' for n,p in predictions.items()),flush=True)
            print('COMBO '+json.dumps(sel['combo_specs']),flush=True)
            (out/'fit_records.json').write_text(json.dumps(records,indent=2))
            (out/'selections.json').write_text(json.dumps(selections,indent=2))
            np.savez_compressed(out/'checkpoint.npz',completed_folds=k,**predictions)
    np.savez_compressed(out/'oof_predictions.npz',actual=y,fold=fold,source_row_index=ix,**predictions)
    np.savez_compressed(out/'oof_components.npz',actual=y,fold=fold,source_row_index=ix,**oofparts)
    np.savez_compressed(out/'baseline_reference.npz',actual=y,fold=fold,source_row_index=ix,A34_both=ref['A34_both'])
    segs={'All':np.ones(len(y),bool),'SFR=0':y==0,'0<SFR<=3':(y>0)&(y<=3),
        '3<SFR<=10':(y>3)&(y<=10),'10<SFR<=30':(y>10)&(y<=30),
        '30<SFR<=100':(y>30)&(y<=100),'SFR>100':y>100}
    rows=[];fr=[]
    for name,p in predictions.items():
        assert np.isfinite(p).all() and (p>=0).all()
        for seg,mask in segs.items():rows.append(dict(model=name,segment=seg,n=int(mask.sum()),rmse=score(y[mask],p[mask])))
        for k in range(1,6):
            mask=fold==k;fr.append(dict(model=name,fold=k,rmse=score(y[mask],p[mask])))
    scores=pd.DataFrame(rows);folds=pd.DataFrame(fr)
    tab=scores.query("segment=='All'").copy().sort_values('rmse');base=score(y,predictions['A34_HGB'])
    tab['improvement_pct']=100*(base-tab.rmse)/base
    pivot=folds.pivot(index='fold',columns='model',values='rmse')
    tab['fold_wins_vs_baseline']=[int((pivot[n]<pivot.A34_HGB).sum()) for n in tab.model]
    scores.to_csv(out/'scores.csv',index=False);folds.to_csv(out/'fold_scores.csv',index=False);tab.to_csv(out/'overall_comparison.csv',index=False)
    print('FINAL OOF\n'+tab.to_string(index=False),flush=True)
    if args.fit_final:
        print('Selecting final specifications using an internal split of the original training data',flush=True)
        with threadpool_limits(limits=4):
            _,final_sel=tune(X,y,2026,'final_training',records)
            final=FixedMixture(specs=final_sel['combo_specs']).fit(X,y)
            check=final.predict(X[:100])
            assert np.isfinite(check).all() and (check>=0).all()
            joblib.dump(final,out/'A34_selected.joblib',compress=3)
            restored=joblib.load(out/'A34_selected.joblib')
            np.testing.assert_allclose(restored.predict(X[:100]),check,rtol=1e-12,atol=1e-12)
        (out/'final_selection.json').write_text(json.dumps(final_sel,indent=2))
        (out/'fit_records.json').write_text(json.dumps(records,indent=2))
    meta=dict(data_git_blob_sha=sha,train_rows=len(y),catalogue_rows=len(df),holdout_evaluated=False,
        features=list(Xfull.columns),feature_source_commit='6bc3988dd7effb7694192ce059f515fa496fa499',
        reference_catalogue='Fixed full unlabeled catalogue; same geometry as A34; target excluded',
        metric='RMSE log10(1+SFR)',tail_threshold=10,log_offset=1,zero_gate=False,
        cv='Original train split; outer KFold(5,shuffle=True,random_state=42); inner 20% split stratified by 3 regimes; seed=1000+fold',
        tuning='Two configs each CatBoost/LightGBM, three ExtraTrees; choose by full mixture inner log RMSE, refit on entire outer training fold.',
        combination='Per role choose only a single swap that improves inner score; if joint inner score worsens baseline, fall back to best improving single swap. No outer-fold labels used for combination.',
        hgb_params=HGB,grid=GRID,versions=dict(numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,catboost=catboost.__version__,lightgbm=lightgbm.__version__),
        final_model_fitted=args.fit_final,elapsed_seconds=time.time()-started)
    (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2));(out/'checkpoint.npz').unlink()
    print('COMPLETE',flush=True)

if __name__=='__main__':main()

from pathlib import Path
import os,json,hashlib,time,argparse,warnings,copy
os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import KFold,train_test_split
from threadpoolctl import threadpool_limits
from features import RAW,build_feature_sets
from models import BASE,fit_part,part,mix,score,labels

CONTEXTS={'HGB':BASE,'LGB':{
    'gate':dict(family='lgb',params=dict(n_estimators=600,num_leaves=63)),
    'mid':dict(family='lgb',params=dict(n_estimators=600,num_leaves=63)),
    'tail':BASE['tail']}}
RF_GRID=[dict(min_samples_leaf=3),dict(min_samples_leaf=10)]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True)
    ap.add_argument('--previous',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    started=time.time();raw=Path(args.data).read_bytes()
    sha=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest();del raw
    assert sha=='06479cedd323253f08af9983c73f395cc02d3774'
    df=pd.read_csv(args.data)
    with threadpool_limits(limits=4):features=build_feature_sets(df[RAW])['A34_both']
    ix,_=train_test_split(np.arange(len(df)),test_size=.2,random_state=42,stratify=(df.sfr_linear>0).astype(int))
    X=features.iloc[ix].to_numpy();y=df.sfr_linear.to_numpy()[ix]
    old=Path(args.previous);cached=np.load(old/'oof_components.npz');ref=np.load(old/'oof_predictions.npz')
    prior_specs=json.loads((old/'selections.json').read_text())
    for source in [cached,ref]:
        np.testing.assert_array_equal(ix,source['source_row_index']);np.testing.assert_array_equal(y,source['actual'])
    for sel in prior_specs:
        for s in ['gate','mid']:assert sel['best_specs']['lgb_'+s]==CONTEXTS['LGB'][s]
    fold=ref['fold'];outer_base={
        'HGB':{s:cached['hgb_'+s] for s in BASE},
        'LGB':{'gate':cached['lgb_gate'],'mid':cached['lgb_mid'],'tail':cached['hgb_tail']}}
    np.testing.assert_allclose(mix(outer_base['HGB']),ref['A34_HGB'],rtol=1e-9,atol=1e-9)
    predictions={};rf_parts={}
    for ctx in CONTEXTS:
        predictions[ctx+'_base']=mix(outer_base[ctx])
        for slot in BASE:
            predictions[ctx+'_rf_'+slot]=np.zeros(len(y))
            rf_parts[ctx+'_'+slot]=np.zeros((len(y),3)) if slot=='gate' else np.zeros(len(y))
        predictions[ctx+'_rf_combo']=np.zeros(len(y))
    records=[];choices=[]
    with threadpool_limits(limits=4):
        for k,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=42).split(X),1):
            assert np.all(fold[te]==k)
            a,b=train_test_split(tr,test_size=.2,random_state=1000+k,stratify=labels(y[tr]))
            assert not np.intersect1d(a,b).size and not np.intersect1d(tr,te).size
            print(f'FOLD {k}/5: inner train={len(a):,}, validation={len(b):,}',flush=True)
            inner_base={};cache={}
            for ctx,specs in CONTEXTS.items():
                inner_base[ctx]={}
                for slot,spec in specs.items():
                    key=(slot,json.dumps(spec,sort_keys=True))
                    if key not in cache:
                        m=fit_part(slot,spec,X[a],y[a]);cache[key]=part(m,slot,X[b])
                    inner_base[ctx][slot]=cache[key]
            baseline={c:score(y[b],mix(p)) for c,p in inner_base.items()}
            # Sanity-check the recreated HGB inner baseline against the previous run.
            assert abs(baseline['HGB']-prior_specs[k-1]['baseline_rmse'])<1e-12
            candidates={};selected={c:{} for c in CONTEXTS}
            for slot in BASE:
                candidates[slot]=[]
                for params in RF_GRID:
                    spec=dict(family='rf',params=params);t=time.time()
                    with warnings.catch_warnings(record=True) as ws:
                        warnings.simplefilter('always');m=fit_part(slot,spec,X[a],y[a]);p=part(m,slot,X[b])
                    sc={c:score(y[b],mix({**base,slot:p})) for c,base in inner_base.items()}
                    records.append(dict(fold=k,phase='inner',slot=slot,spec=spec,scores=sc,seconds=time.time()-t,
                        warnings=[str(w.message) for w in ws]))
                    candidates[slot].append(dict(spec=spec,pred=p,scores=sc))
                    print(f'  RF {slot} leaf={params["min_samples_leaf"]}: inner {sc}, {time.time()-t:.1f}s',flush=True)
                for ctx in CONTEXTS:
                    selected[ctx][slot]=min(candidates[slot],key=lambda q:q['scores'][ctx])
            # The same RF fit is reused when both contexts choose the same setting.
            outer_candidates={}
            for slot in BASE:
                for ctx in CONTEXTS:
                    item=selected[ctx][slot];key=(slot,json.dumps(item['spec'],sort_keys=True))
                    if key not in outer_candidates:
                        t=time.time()
                        with warnings.catch_warnings(record=True) as ws:
                            warnings.simplefilter('always');m=fit_part(slot,item['spec'],X[tr],y[tr]);p=part(m,slot,X[te])
                        outer_candidates[key]=p
                        records.append(dict(fold=k,phase='outer_refit',slot=slot,spec=item['spec'],seconds=time.time()-t,
                            warnings=[str(w.message) for w in ws]))
                        print(f'  refit RF {slot} {item["spec"]["params"]}: {time.time()-t:.1f}s',flush=True)
                    p=outer_candidates[key];rf_parts[ctx+'_'+slot][te]=p
                    predictions[ctx+'_rf_'+slot][te]=mix({**{s:v[te] for s,v in outer_base[ctx].items()},slot:p})
            for ctx in CONTEXTS:
                accepted=[s for s in BASE if selected[ctx][s]['scores'][ctx]<baseline[ctx]]
                inner_mix={**inner_base[ctx],**{s:selected[ctx][s]['pred'] for s in accepted}}
                combo_score=score(y[b],mix(inner_mix));fallback=False
                if combo_score>baseline[ctx]:
                    fallback=True;slot=min(BASE,key=lambda s:selected[ctx][s]['scores'][ctx])
                    accepted=[slot] if selected[ctx][slot]['scores'][ctx]<baseline[ctx] else []
                    combo_score=selected[ctx][slot]['scores'][ctx] if accepted else baseline[ctx]
                components={s:v[te] for s,v in outer_base[ctx].items()}
                for slot in accepted:components[slot]=rf_parts[ctx+'_'+slot][te]
                predictions[ctx+'_rf_combo'][te]=mix(components)
                choices.append(dict(fold=k,context=ctx,baseline_inner=baseline[ctx],combo_inner=combo_score,
                    selected_specs={s:selected[ctx][s]['spec'] for s in BASE},
                    selected_inner_scores={s:selected[ctx][s]['scores'][ctx] for s in BASE},
                    accepted_slots=accepted,fallback=fallback))
            print('OUTER '+', '.join(f'{n}={score(y[te],p[te]):.6f}' for n,p in predictions.items()),flush=True)
            (out/'fit_records.json').write_text(json.dumps(records,indent=2));(out/'choices.json').write_text(json.dumps(choices,indent=2))
            np.savez_compressed(out/'checkpoint.npz',completed_folds=k,**predictions)
    np.savez_compressed(out/'oof_predictions.npz',actual=y,fold=fold,source_row_index=ix,**predictions)
    np.savez_compressed(out/'oof_components.npz',actual=y,fold=fold,source_row_index=ix,**rf_parts,
        **{'reference_'+key:cached[key] for key in ['hgb_gate','hgb_mid','hgb_tail','lgb_gate','lgb_mid']})
    segs={'All':np.ones(len(y),bool),'SFR=0':y==0,'0<SFR<=3':(y>0)&(y<=3),'3<SFR<=10':(y>3)&(y<=10),
        '10<SFR<=30':(y>10)&(y<=30),'30<SFR<=100':(y>30)&(y<=100),'SFR>100':y>100}
    rows=[];fr=[]
    for name,p in predictions.items():
        assert np.isfinite(p).all() and (p>=0).all()
        for seg,mask in segs.items():rows.append(dict(model=name,segment=seg,n=int(mask.sum()),rmse=score(y[mask],p[mask])))
        for k in range(1,6):
            mask=fold==k;fr.append(dict(model=name,fold=k,rmse=score(y[mask],p[mask])))
    scores=pd.DataFrame(rows);fs=pd.DataFrame(fr);tab=scores.query("segment=='All'").copy()
    piv=fs.pivot(index='fold',columns='model',values='rmse')
    tab['improvement_pct']=[100*(score(y,predictions[n.split('_')[0]+'_base'])-r)/score(y,predictions[n.split('_')[0]+'_base']) for n,r in zip(tab.model,tab.rmse)]
    tab['fold_wins_vs_context']=[int((piv[n]<piv[n.split('_')[0]+'_base']).sum()) for n in tab.model]
    scores.to_csv(out/'scores.csv',index=False);fs.to_csv(out/'fold_scores.csv',index=False);tab.to_csv(out/'overall_comparison.csv',index=False)
    meta=dict(data_git_blob_sha=sha,features=list(features.columns),train_rows=len(y),catalogue_rows=len(df),
        metric='RMSE log10(1+SFR)',tail_threshold=10,log_offset=1,zero_gate=False,holdout_evaluated=False,
        reference_policy='Same fixed full unlabeled catalogue as previous A34 runs; target excluded from geometry.',
        contexts=CONTEXTS,rf_grid=RF_GRID,rf_fixed=dict(n_estimators=200,bootstrap=True,gate_max_features='sqrt',regression_max_features=.7,n_jobs=4,random_state=42),
        cv='Same outer KFold(5, shuffle=True, random_state=42); inner 20% stratified by target regime, seed1000+fold; select separately by full mixture RMSE in each context.',
        baseline_reuse='Outer component predictions reused only after data, row order, parameters and fold checks. Recreated inner HGB scores match previous run within 1e-12.',
        selection_note='LGB context fixes the previously chosen LightGBM/LightGBM/HGB architecture. Its new OOF score is distinct from the prior fold-varying combination-selection score.',
        sklearn=sklearn.__version__,elapsed_seconds=time.time()-started)
    (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2));(out/'checkpoint.npz').unlink()
    print(tab.to_string(index=False),flush=True);print('COMPLETE',flush=True)

if __name__=='__main__':main()

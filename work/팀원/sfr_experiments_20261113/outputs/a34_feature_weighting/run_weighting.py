from pathlib import Path
import os,json,time,hashlib,argparse,warnings
os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
import numpy as np
import pandas as pd
import sklearn,lightgbm
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from features import RAW,build_feature_sets
from models import fit_part,part,mix,score,labels

DENSITY='log10_overdensity_plus_1';BH='log10_BH_mass'
VARIANTS={'baseline':(1.,1.),'density_x2':(2.,1.),'BH_x2':(1.,2.),'both_x1_5':(1.5,1.5),'both_x2':(2.,2.)}

def feature_weights(names,density,bh):
    if density<=0 or bh<=0:raise ValueError('Weights must be positive')
    w=np.ones(len(names));w[names.index(DENSITY)]=density;w[names.index(BH)]=bh
    return w.tolist()

def gate_spec(names,weights):
    p=dict(n_estimators=600,num_leaves=63)
    if weights!=(1.,1.):p['feature_contri']=feature_weights(names,*weights)
    return dict(family='lgb',params=p)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--previous',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True);start=time.time()
    raw=Path(args.data).read_bytes();sha=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest();del raw
    assert sha=='06479cedd323253f08af9983c73f395cc02d3774'
    df=pd.read_csv(args.data)
    with threadpool_limits(limits=4):full=build_feature_sets(df[RAW])['A34_both']
    names=list(full.columns)
    ix,_=train_test_split(np.arange(len(df)),test_size=.2,random_state=42,stratify=(df.sfr_linear>0).astype(int))
    X=full.iloc[ix].to_numpy();y=df.sfr_linear.to_numpy()[ix];lab=labels(y)
    old=Path(args.previous);cache=np.load(old/'oof_components.npz');oldpred=np.load(old/'oof_predictions.npz')
    for data in [cache,oldpred]:
        np.testing.assert_array_equal(data['source_row_index'],ix);np.testing.assert_array_equal(data['actual'],y)
        np.testing.assert_array_equal(data['fold'],cache['fold'])
    old_selections=json.loads((old/'selections.json').read_text())
    for s in old_selections:
        assert s['best_specs']['lgb_gate']==gate_spec(names,(1.,1.))
        assert s['best_specs']['lgb_mid']==gate_spec(names,(1.,1.))
    fold=cache['fold'];mid=cache['lgb_mid'];tail=cache['hgb_tail']
    preds={n:np.zeros(len(y)) for n in VARIANTS};probs={n:np.zeros((len(y),3)) for n in VARIANTS}
    records=[];perms=[];baseline_fold_importances=[]
    groups={n:[i] for i,n in enumerate(names)}
    groups['GROUP_density']=[names.index(n) for n in [DENSITY,'log_rho_5nn','log_rho_10nn','log1p_N_r1','log1p_N_r2','log1p_N_r5']]
    groups['GROUP_BH']=[names.index(n) for n in [BH,'bh_to_star']]
    with threadpool_limits(limits=4):
        for k,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=42).split(X),1):
            assert np.all(fold[te]==k)
            print(f'FOLD {k}/5',flush=True)
            for name,weights in VARIANTS.items():
                t=time.time();spec=gate_spec(names,weights)
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter('always');m=fit_part('gate',spec,X[tr],y[tr]);p=part(m,'gate',X[te])
                pred=mix(dict(gate=p,mid=mid[te],tail=tail[te]))
                if name=='baseline':
                    np.testing.assert_allclose(p,cache['lgb_gate'][te],rtol=1e-9,atol=1e-9)
                    baseline_model=m
                    gain=m.booster_.feature_importance(importance_type='gain')
                    split=m.booster_.feature_importance(importance_type='split')
                    baseline_fold_importances.extend(dict(fold=k,feature=n,gain_pct=float(100*g/gain.sum()),splits=int(s)) for n,g,s in zip(names,gain,split))
                else:
                    # Ensure LightGBM actually received the intended split-gain multipliers.
                    np.testing.assert_array_equal(m.booster_.params['feature_contri'],feature_weights(names,*weights))
                probs[name][te]=p;preds[name][te]=pred
                splits=m.booster_.feature_importance(importance_type='split')
                records.append(dict(fold=k,model=name,weights=weights,seconds=time.time()-t,
                    classifier_log_loss=float(log_loss(lab[te],p,labels=[0,1,2])),
                    zero_log_loss=float(log_loss((y[te]==0).astype(int),p[:,0],labels=[0,1])),
                    density_splits=int(splits[names.index(DENSITY)]),bh_splits=int(splits[names.index(BH)]),
                    total_splits=int(splits.sum()),warnings=[str(w.message) for w in ws]))
                print(f'  {name}: mixed RMSE={score(y[te],pred):.6f}, {time.time()-t:.1f}s',flush=True)
            # Model reliance on held-out inputs. Experts stay fixed to isolate the gate.
            sampled,_=train_test_split(te,train_size=3000,random_state=7000+k,stratify=lab[te])
            Xsample=X[sampled];basep=part(baseline_model,'gate',Xsample)
            basemix=score(y[sampled],mix(dict(gate=basep,mid=mid[sampled],tail=tail[sampled])))
            basell=log_loss(lab[sampled],basep,labels=[0,1,2])
            basezero=log_loss((y[sampled]==0).astype(int),basep[:,0],labels=[0,1])
            for repeat in range(2):
                order=np.random.default_rng(9000+10*k+repeat).permutation(len(sampled))
                for key,columns in groups.items():
                    shuffled=Xsample.copy();shuffled[:,columns]=Xsample[order][:,columns]
                    p=part(baseline_model,'gate',shuffled)
                    perms.append(dict(fold=k,repeat=repeat,feature=key,n=len(sampled),
                        delta_log_loss=float(log_loss(lab[sampled],p,labels=[0,1,2])-basell),
                        delta_zero_log_loss=float(log_loss((y[sampled]==0).astype(int),p[:,0],labels=[0,1])-basezero),
                        delta_mixed_rmse=score(y[sampled],mix(dict(gate=p,mid=mid[sampled],tail=tail[sampled])))-basemix))
            print(f'  held-out permutation complete; elapsed {time.time()-start:.1f}s',flush=True)
            (out/'fit_records.json').write_text(json.dumps(records,indent=2));pd.DataFrame(perms).to_csv(out/'permutation_details.csv',index=False)
            np.savez_compressed(out/'checkpoint.npz',completed_folds=k,**preds)
    np.savez_compressed(out/'oof_predictions.npz',actual=y,fold=fold,source_row_index=ix,**preds)
    np.savez_compressed(out/'oof_probabilities.npz',actual=y,fold=fold,source_row_index=ix,**probs)
    np.savez_compressed(out/'fixed_experts.npz',mid=mid,tail=tail)
    segs={'All':np.ones(len(y),bool),'SFR=0':y==0,'0<SFR<=3':(y>0)&(y<=3),'3<SFR<=10':(y>3)&(y<=10),
          '10<SFR<=30':(y>10)&(y<=30),'30<SFR<=100':(y>30)&(y<=100),'SFR>100':y>100}
    rows=[];fr=[];gate_metrics=[]
    for name,p in preds.items():
        assert np.isfinite(p).all() and (p>=0).all()
        for seg,mask in segs.items():rows.append(dict(model=name,segment=seg,n=int(mask.sum()),rmse=score(y[mask],p[mask])))
        for k in range(1,6):
            mask=fold==k;fr.append(dict(model=name,fold=k,rmse=score(y[mask],p[mask])))
        gate_metrics.append(dict(model=name,log_loss=float(log_loss(lab,probs[name],labels=[0,1,2])),
            zero_log_loss=float(log_loss((y==0).astype(int),probs[name][:,0],labels=[0,1])),accuracy=float((probs[name].argmax(axis=1)==lab).mean())))
    scores=pd.DataFrame(rows);fs=pd.DataFrame(fr);tab=scores.query("segment=='All'").copy().sort_values('rmse')
    baseline=score(y,preds['baseline']);piv=fs.pivot(index='fold',columns='model',values='rmse')
    tab['improvement_pct']=100*(baseline-tab.rmse)/baseline
    tab['fold_wins']=[int((piv[n]<piv.baseline).sum()) for n in tab.model]
    scores.to_csv(out/'scores.csv',index=False);fs.to_csv(out/'fold_scores.csv',index=False);tab.to_csv(out/'overall_comparison.csv',index=False)
    pd.DataFrame(gate_metrics).to_csv(out/'gate_metrics.csv',index=False)
    pi=pd.DataFrame(perms).groupby('feature')[['delta_log_loss','delta_zero_log_loss','delta_mixed_rmse']].agg(['mean','std'])
    pi.columns=['_'.join(c) for c in pi.columns];pi.sort_values('delta_log_loss_mean',ascending=False).to_csv(out/'permutation_summary.csv')
    pd.DataFrame(baseline_fold_importances).to_csv(out/'baseline_fold_importance.csv',index=False)
    meta=dict(data_git_blob_sha=sha,features=names,metric='RMSE log10(1+SFR)',train_rows=len(y),holdout_evaluated=False,
        contexts='Fixed LightGBM gate / LightGBM mid / HGB tail, 34 inputs, T10, log offset1, soft mixture; only gate retrained.',
        weighted_columns=[DENSITY,BH],variants=VARIANTS,
        mechanism='LightGBM feature_contri: multiply each candidate feature split gain by its multiplier; all other features remain1. No input scaling, class/sample weighting or forced direction.',
        evaluation='Same KFold(5,shuffle=True,random_state=42). Fixed prespecified multipliers, no outer-fold-driven tuning. Comparing and selecting after CV remains exploratory model development.',
        permutation='Baseline gate only; stratified 3000 validation rows per fold, two permutations each for all34 features plus two groups. Experts unchanged. Correlated/derived inputs limit independent or causal interpretation.',
        reference_catalogue='Same fixed full unlabeled galaxy catalogue, target excluded from geometry.',
        source_versions=dict(sklearn=sklearn.__version__,lightgbm=lightgbm.__version__),elapsed_seconds=time.time()-start)
    (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2));(out/'checkpoint.npz').unlink()
    print(tab.to_string(index=False),flush=True);print(pi.sort_values('delta_log_loss_mean',ascending=False).head(12).to_string(),flush=True)
    print('COMPLETE',flush=True)

if __name__=='__main__':main()

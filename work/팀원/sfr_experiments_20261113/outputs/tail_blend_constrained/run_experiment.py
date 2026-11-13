import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from threadpoolctl import threadpool_limits
from feature_pipeline import UNIVERSE, FeatureEngineer, build_reference_features, fit_component, predict_sfr
from hierarchical_model import HierarchicalSFR, fit_binary_gates, positive_probability, hierarchical_probabilities
from weighted_tail import fit_tail
from blend_model import TailBlend, blend_sfr_predictions, blend_log_predictions
from three_regime import combine
from metrics import metrics

ROOT=Path(__file__).resolve().parent
PARTNERS=('hgb_w1','hgb_w2','hgb_w3','hgb_w5','lgb_w2','lgb_w3','lgb_w5')
GRID={'baseline':dict(partner='lgb_w1',alpha=0.)}
GRID.update({f'{p}_a{int(a*100)}':dict(partner=p,alpha=a) for p in PARTNERS for a in (.25,.5,.75)})

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def evaluate(seed, name, y, pred, fold, rows):
    assert np.isfinite(pred).all() and (pred>=0).all()
    fold_scores=[]
    for f in range(1,6):
        mask=fold==f
        result=metrics(y[mask],pred[mask]);fold_scores.append(result['log_rmse'])
        rows.append(dict(seed=seed,fold=f,model=name,**result))
    return dict(seed=seed,model=name,mean_fold_rmse=float(np.mean(fold_scores)),**metrics(y,pred))

def feasible(row,base,limit):
    return row['mean_fold_rmse']<=base['mean_fold_rmse']*limit+1e-14 and row['zero_log_rmse']<=base['zero_log_rmse']+1e-14

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--weighted-source',type=Path,default=ROOT.parent/'hierarchical_tail_weighting')
    ap.add_argument('--gate-source',type=Path,default=ROOT.parent/'hierarchical_gate46')
    ap.add_argument('--tail-source',type=Path,default=ROOT.parent/'hierarchical_tail_models')
    ap.add_argument('--cache',type=Path,default=ROOT.parents[1]/'work/feature_search_cache')
    args=ap.parse_args()
    weight_meta=json.loads((args.weighted_source/'metadata.json').read_text())
    gate_meta=json.loads((args.gate_source/'metadata.json').read_text())
    tail_meta=json.loads((args.tail_source/'metadata.json').read_text())
    pointer_sha=sha(ROOT.parent/'current_model.json')
    source_paths={'baseline':args.tail_source/'selected_model.joblib','hgb_w1':args.gate_source/'hierarchical46.joblib'}
    source_paths.update({p:args.weighted_source/(p+'.joblib') for p in PARTNERS if p!='hgb_w1'})
    source_hashes={n:sha(p) for n,p in source_paths.items()}
    assert source_hashes['baseline']==tail_meta['saved_models']['selected_model']['sha256']
    assert source_hashes['hgb_w1']==gate_meta['model_sha256']
    for n in PARTNERS:
        if n!='hgb_w1':assert source_hashes[n]==weight_meta['saved_models'][n]['sha256']
    raw=args.data.read_bytes();data_sha=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
    assert data_sha==weight_meta['data_git_blob_sha']
    records=[];summaries=[];frozen={};saved_predictions={}
    for seed in (42,7):
        old=np.load(args.weighted_source/f'oof_seed{seed}.npz')
        y,ids,fold=old['actual'],old['source_row_index'],old['fold']
        gate_source=np.load(args.gate_source/f'oof_seed{seed}.npz')
        zero_gate_mask=gate_source['hierarchical_probs'][:,0]>=.9
        if seed==42:
            predictions={name:blend_sfr_predictions(old['lgb_w1'],old[c['partner']],c['alpha']) for name,c in GRID.items()}
            for name,pred in predictions.items():
                summaries.append(evaluate(seed,name,y,pred,fold,records))
            base=summaries[0]
            for role,limit in [('tail_priority',1.001),('strict_balance',1.)]:
                valid=[r for r in summaries if feasible(r,base,limit)]
                chosen=min(valid,key=lambda r:(r['tail100_log_rmse'],r['mean_fold_rmse'],r['model']))
                frozen[role]=dict(name=chosen['model'],**GRID[chosen['model']])
            (ROOT/'frozen_selection.json').write_text(json.dumps(frozen,indent=2)+'\n')
            pd.DataFrame(summaries).to_csv(ROOT/'screen_seed42.csv',index=False)
            print('Frozen on seed42:',json.dumps(frozen),flush=True)
        else:
            predictions={'baseline':old['lgb_w1']}
            for choice in frozen.values():
                predictions[choice['name']]=blend_sfr_predictions(old['lgb_w1'],old[choice['partner']],choice['alpha'])
            for name,pred in predictions.items():
                summaries.append(evaluate(seed,name,y,pred,fold,records))
        keep={'baseline':old['lgb_w1']}
        keep.update({role:predictions[c['name']] for role,c in frozen.items()})
        for pred in keep.values():np.testing.assert_array_equal(pred==0,old['lgb_w1']==0)
        # Prove the saved-SFR screen is identical to the actual tail-expert blend,
        # including the zero floor, using the original unclipped components.
        tail_source=np.load(args.weighted_source/f'tail_components_seed{seed}.npz')
        for f in range(1,6):
            va=np.flatnonzero(fold==f)
            key=hashlib.sha256(json.dumps([weight_meta['source_fingerprint'],seed,f,'mid',UNIVERSE]).encode()).hexdigest()
            cached=np.load(args.cache/(key+'.npz'));np.testing.assert_array_equal(cached['validation_rows'],va)
            p=gate_source['hierarchical_probs'][va];mid=cached['prediction']
            base_log=combine(p,mid,tail_source['lgb_w1'][va])
            configs_to_check=GRID if seed==42 else {c['name']:c for c in frozen.values()}
            for name,c in configs_to_check.items():
                partner_log=combine(p,mid,tail_source[c['partner']][va])
                np.testing.assert_allclose(predictions[name][va],blend_log_predictions(base_log,partner_log,c['alpha']),rtol=1e-10,atol=1e-10)
        saved_predictions[seed]=keep
        np.savez_compressed(ROOT/f'oof_seed{seed}.npz',actual=y,source_row_index=ids,fold=fold,zero_gate_mask=zero_gate_mask,**keep)
    pd.DataFrame(summaries).to_csv(ROOT/'summary.csv',index=False)
    pd.DataFrame(records).to_csv(ROOT/'fold_results.csv',index=False)
    needed={'lgb_w1'}|{c['partner'] for c in frozen.values()}
    with threadpool_limits(limits=4):
        print('Building unchanged features; fitting new seed99 confirmation...',flush=True)
        df=pd.read_csv(args.data)
        expected_ids,_=train_test_split(np.arange(len(df)),test_size=.2,random_state=42,stratify=(df.sfr_linear>0).astype(int))
        np.testing.assert_array_equal(ids,expected_ids)
        np.testing.assert_array_equal(y,df.sfr_linear.to_numpy()[ids])
        X=build_reference_features(df).iloc[ids].reset_index(drop=True)
        pred99={name:np.full(len(y),np.nan) for name in {'baseline'}|{c['name'] for c in frozen.values()}}
        fold99=np.zeros(len(y),int);coverage=np.zeros(len(y),int);fit_records=[];zero_gate_mask99=np.zeros(len(y),bool)
        for f,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=99).split(X),1):
            start=time.monotonic();fold99[va]=f;coverage[va]+=1
            engineer=FeatureEngineer().fit(X.iloc[tr])
            A,B=engineer.transform(X.iloc[tr]),engineer.transform(X.iloc[va])
            g0,gt=fit_binary_gates(A,y[tr])
            p=hierarchical_probabilities(positive_probability(g0,B),positive_probability(gt,B))
            zero_gate_mask99[va]=p[:,0]>=.9
            mid=fit_component('mid',UNIVERSE,A,y[tr]).predict(B[UNIVERSE])
            logs={}
            mask=y[tr]>10
            for name in sorted(needed):
                logs[name]=fit_tail(name,A.loc[mask,UNIVERSE],y[tr][mask]).predict(B[UNIVERSE])
            base_log=combine(p,mid,logs['lgb_w1'])
            base=np.maximum(10.**base_log-1,0.)
            pred99['baseline'][va]=base
            for choice in frozen.values():
                a=choice['alpha'];partner=choice['partner']
                partner_log=combine(p,mid,logs[partner])
                final_log=combine(p,mid,(1-a)*logs['lgb_w1']+a*logs[partner])
                pred=np.maximum(10.**final_log-1,0.)
                np.testing.assert_allclose(pred,blend_log_predictions(base_log,partner_log,a),rtol=1e-10,atol=1e-10)
                np.testing.assert_array_equal(pred[p[:,0]>=.9],0.)
                pred99[choice['name']][va]=pred
            fit_records.append(dict(seed=99,fold=f,train_rows=len(tr),tail_rows=int(mask.sum()),
                extreme_rows=int((y[tr]>100).sum()),seconds=time.monotonic()-start))
            print(f'seed99 fold{f} complete ({time.monotonic()-start:.1f}s)',flush=True)
            pd.DataFrame(fit_records).to_csv(ROOT/'fit_records_seed99.csv',index=False)
        assert (coverage==1).all()
        for name,pred in pred99.items():summaries.append(evaluate(99,name,y,pred,fold99,records))
        keep={'baseline':pred99['baseline'],**{role:pred99[c['name']] for role,c in frozen.items()}}
        saved_predictions[99]=keep
        np.savez_compressed(ROOT/'oof_seed99.npz',actual=y,source_row_index=ids,fold=fold99,zero_gate_mask=zero_gate_mask99,**keep)
        table=pd.DataFrame(summaries)
        table.to_csv(ROOT/'summary.csv',index=False);pd.DataFrame(records).to_csv(ROOT/'fold_results.csv',index=False)
        acceptance=[];approved={}
        for role,choice in frozen.items():
            limit=1.001 if role=='tail_priority' else 1.
            checks=[]
            for seed in (42,7,99):
                base=table[(table.seed==seed)&(table.model=='baseline')].iloc[0]
                candidate=table[(table.seed==seed)&(table.model==choice['name'])].iloc[0]
                passed=feasible(candidate,base,limit) and candidate.tail100_log_rmse<base.tail100_log_rmse
                checks.append(bool(passed))
                acceptance.append(dict(seed=seed,role=role,model=choice['name'],accepted=bool(passed),
                    overall_pct=(candidate.mean_fold_rmse/base.mean_fold_rmse-1)*100,
                    zero_pct=(candidate.zero_log_rmse/base.zero_log_rmse-1)*100,
                    extreme_pct=(candidate.tail100_log_rmse/base.tail100_log_rmse-1)*100))
            approved[role]=all(checks)
        recommendation=next((r for r in ('tail_priority','strict_balance') if approved[r]),'baseline')
        acceptance_df=pd.DataFrame(acceptance);acceptance_df.to_csv(ROOT/'acceptance.csv',index=False)
        (ROOT/'recommendation.json').write_text(json.dumps(dict(recommended=recommendation,approved=approved),indent=2)+'\n')
        print('Acceptance:',acceptance_df.to_string(index=False),flush=True)
        base_model=joblib.load(source_paths['baseline']);models={'baseline':base_model}
        saved_models={}
        for role,choice in frozen.items():
            if choice['name']=='baseline':model=base_model
            else:
                partner_model=joblib.load(source_paths[choice['partner']])
                # Source experts were trained with exactly the same full-train features/gates/mid.
                small=X.iloc[:1000]
                np.testing.assert_array_equal(base_model.engineer_.transform(small).to_numpy(),partner_model.engineer_.transform(small).to_numpy())
                bp,bm,_=base_model.predict_components(small);pp,pm,_=partner_model.predict_components(small)
                np.testing.assert_array_equal(bp,pp);np.testing.assert_array_equal(bm,pm)
                model=HierarchicalSFR(base_model.engineer_,base_model.positive_gate_,base_model.conditional_tail_gate_,
                    base_model.mid_regressor_,TailBlend(base_model.tail_regressor_,partner_model.tail_regressor_,choice['alpha']))
                np.testing.assert_allclose(model.predict(small),blend_log_predictions(base_model.predict_log(small),partner_model.predict_log(small),choice['alpha']),rtol=1e-10,atol=1e-10)
            models[role]=model
        to_save={role:models[role] for role in frozen}
        to_save['recommended_model']=models[recommendation]
        for filename,model in to_save.items():
            path=ROOT/(filename+'.joblib');joblib.dump(model,path,compress=3)
            restored=joblib.load(path)
            np.testing.assert_array_equal(model.predict(X.iloc[:1000]),restored.predict(X.iloc[:1000]))
            saved_models[filename]=dict(file=path.name,sha256=sha(path))
        reference=df.drop(columns='sfr_linear')
        chosen=models[recommendation]
        raw_pred=predict_sfr(reference.iloc[ids[:1000]],chosen,reference)
        np.testing.assert_allclose(raw_pred,chosen.predict(X.iloc[:1000]),rtol=1e-6,atol=1e-9)
    for name,path in source_paths.items():assert sha(path)==source_hashes[name]
    assert sha(ROOT.parent/'current_model.json')==pointer_sha
    metadata=dict(data_git_blob_sha=data_sha,source_hashes=source_hashes,current_pointer_sha256=pointer_sha,
        features=UNIVERSE,train_rows=len(y),tail_train_rows=int((y>10).sum()),extreme_rows=int((y>100).sum()),
        seeds=[42,7,99],selection_seed=42,frozen=frozen,recommendation=recommendation,
        allowed_relative_overall_increase=.001,zero_increase_allowed=False,grid=GRID,
        saved_models=saved_models,holdout_scored=False,current_model_replaced=False,
        zero_output_differences={str(seed):{role:int(np.sum((pred==0)!=(values['baseline']==0))) for role,pred in values.items()} for seed,values in saved_predictions.items()},
        checks=dict(zero_gate_rule_preserved=True,screen_clipping_equivalence=True,new_seed99_fit=True,fold_feature_engineering=True,
            log_blend_equivalence=True,fixed_gates_and_mid=True,saved_reload=True,raw_interface=True,source_models_unchanged=True),
        versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','lightgbm','joblib']})
    (ROOT/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('CONSTRAINED BLEND COMPLETE; recommended:',recommendation,flush=True)

if __name__=='__main__':main()

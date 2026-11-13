import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from threadpoolctl import threadpool_limits
from feature_pipeline import (BASE, ADDITIONS, REMOVALS, UNIVERSE, FeatureEngineer,
    FeatureSelectedThreeRegime, build_reference_features, fit_component, component_prediction,
    predict_sfr)
from leader_model import TwoStageLogSFR, HGB_FIXED400
from three_regime import combine
from metrics import metrics

SLOTS = ['gate','mid','tail']


def columns(add=(), remove=()):
    selected = set(BASE) | set(add)
    selected -= set(remove)
    return [c for c in UNIVERSE if c in selected]


def plan_id(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:16]


def predictions(parts):
    return np.maximum(10.**combine(parts['gate'], parts['mid'], parts['tail'], .9)-1.,0.)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--old-dir', type=Path, default=Path(__file__).resolve().parent.parent/'leader_three_regime_experiment')
    parser.add_argument('--current-a-dir', type=Path, default=Path(__file__).resolve().parent.parent/'a34_zero_gate_adopted')
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    args.cache.mkdir(parents=True,exist_ok=True)
    raw = args.data.read_bytes()
    data_sha = hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
    assert data_sha == '06479cedd323253f08af9983c73f395cc02d3774'
    fingerprint = hashlib.sha256((root/'feature_pipeline.py').read_bytes()+(root/'leader_model.py').read_bytes()+(root/'three_regime.py').read_bytes()+data_sha.encode()).hexdigest()
    for filename in ['leader_model.py','three_regime.py']:
        assert (root/filename).read_bytes() == (args.old_dir/filename).read_bytes()
    df = pd.read_csv(args.data)
    ids,_ = train_test_split(np.arange(len(df)),test_size=.2,random_state=42,stratify=(df.sfr_linear>0).astype(int))
    y = df.sfr_linear.to_numpy()[ids]
    cached = np.load(args.old_dir/'oof_predictions.npz')
    original_components = np.load(args.old_dir/'oof_components.npz')['three_T10']
    np.testing.assert_array_equal(ids,cached['source_row_index'])
    np.testing.assert_array_equal(y,cached['actual'])
    for _,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=42).split(ids),1):
        assert (cached['fold'][va]==_).all()
    baseline_plan = {s:BASE.copy() for s in SLOTS}
    model_plan_cache = {}
    fit_records = []
    with threadpool_limits(limits=4):
        print('Building full reference geometry and companion features...',flush=True)
        X = build_reference_features(df).iloc[ids].reset_index(drop=True)
        fold_data_cache = {}

        def fold_data(seed,fold):
            key=(seed,fold)
            if key not in fold_data_cache:
                tr,va=list(KFold(5,shuffle=True,random_state=seed).split(X))[fold-1]
                engineer=FeatureEngineer().fit(X.iloc[tr])
                assert engineer.fit_rows_==len(tr) and not np.intersect1d(tr,va).size
                fold_data_cache[key]=(tr,va,engineer.transform(X.iloc[tr]),engineer.transform(X.iloc[va]))
            return fold_data_cache[key]

        def part(seed,fold,slot,cols):
            tr,va,A,B=fold_data(seed,fold)
            if seed==42 and cols==BASE:
                if slot=='gate': return original_components[va,:3]
                return original_components[va,3 if slot=='mid' else 4]
            identity=json.dumps([fingerprint,seed,fold,slot,cols])
            digest=hashlib.sha256(identity.encode()).hexdigest()
            path=args.cache/(digest+'.npz')
            if path.exists():
                saved=np.load(path)
                np.testing.assert_array_equal(saved['validation_rows'],va)
                return saved['prediction']
            start=time.monotonic()
            model=fit_component(slot,cols,A,y[tr])
            pred=component_prediction(model,slot,cols,B)
            assert np.isfinite(pred).all() and model.n_iter_==400
            elapsed=time.monotonic()-start
            np.savez_compressed(path,prediction=pred,validation_rows=va)
            fit_records.append(dict(seed=seed,fold=fold,slot=slot,feature_count=len(cols),features=cols,seconds=elapsed,cache_key=digest))
            (root/'component_fits.json').write_text(json.dumps(fit_records,indent=2)+'\n')
            return pred

        def plan_parts(seed,fold,plan):
            return {s:part(seed,fold,s,plan[s]) for s in SLOTS}

        sets={'baseline':BASE.copy()}
        sets.update({'add_'+name:columns(add=cols) for name,cols in ADDITIONS.items()})
        sets['add_all']=UNIVERSE.copy()
        sets.update({'drop_'+name:columns(remove=cols) for name,cols in REMOVALS.items()})
        screening=[]
        for name,cols in sets.items():
            for fold in (1,2):
                _,va,_,_=fold_data(42,fold)
                p=plan_parts(42,fold,{s:cols for s in SLOTS})
                base=plan_parts(42,fold,baseline_plan)
                for mode in ['joint']+SLOTS:
                    used=p if mode=='joint' else {**base,mode:p[mode]}
                    screening.append(dict(candidate=name,mode=mode,fold=fold,n_features=len(cols),**metrics(y[va],predictions(used))))
            pd.DataFrame(screening).to_csv(root/'screening_folds.csv',index=False)
            avg=np.mean([r['log_rmse'] for r in screening if r['candidate']==name and r['mode']=='joint'])
            print(f'SCREEN {name}: mean={avg:.9f}, n={len(cols)}',flush=True)
        screen=pd.DataFrame(screening)
        ranking=screen.groupby(['candidate','mode']).log_rmse.mean().reset_index(name='mean_fold_rmse')
        ranking.to_csv(root/'screening_summary.csv',index=False)
        joint=ranking[ranking['mode']=='joint'].sort_values(['mean_fold_rmse','candidate']).iloc[0].candidate
        mix={}
        stacked={}
        slot_choices={}
        for slot in SLOTS:
            sub=ranking[ranking['mode']==slot].set_index('candidate').mean_fold_rmse
            best=sub.sort_values().index[0]
            mix[slot]=sets[best]
            add=[];remove=[];used_add=[];used_remove=[]
            for name,cols in ADDITIONS.items():
                if sub.loc['baseline']-sub.loc['add_'+name] >= .00002:
                    add.extend(cols);used_add.append(name)
            for name,cols in REMOVALS.items():
                if sub.loc['baseline']-sub.loc['drop_'+name] >= .00002:
                    remove.extend(cols);used_remove.append(name)
            stacked[slot]=columns(add,remove)
            slot_choices[slot]=dict(best_single=best,stacked_additions=used_add,stacked_removals=used_remove)
        proposed={'baseline':baseline_plan,'best_joint_'+joint:{s:sets[joint] for s in SLOTS},
                  'best_component_mix':mix,'stacked_component_changes':stacked}
        plans={};seen={}
        for name,plan in proposed.items():
            key=plan_id(plan)
            if key not in seen:
                seen[key]=name;plans[name]=plan
        (root/'stage2_plans.json').write_text(json.dumps(dict(plans=plans,screen_choices=slot_choices),indent=2)+'\n')
        print('STAGE2 plans:',json.dumps({n:{s:len(c) for s,c in p.items()} for n,p in plans.items()}),flush=True)
        stage2=[];oof={}
        for name,plan in plans.items():
            pred=np.full(len(y),np.nan)
            for fold in range(1,6):
                _,va,_,_=fold_data(42,fold)
                pred[va]=predictions(plan_parts(42,fold,plan))
                row=dict(model=name,fold=fold,**metrics(y[va],pred[va]))
                stage2.append(row)
                print(f'STAGE2 {name} fold{fold}: {row["log_rmse"]:.9f}',flush=True)
            assert np.isfinite(pred).all()
            oof[name]=pred
            pd.DataFrame(stage2).to_csv(root/'stage2_folds.csv',index=False)
        stage2_df=pd.DataFrame(stage2)
        summaries=[]
        for name in plans:
            rows=stage2_df[stage2_df.model==name]
            summaries.append(dict(model=name,mean_fold_rmse=float(rows.log_rmse.mean()),
                total_component_features=sum(len(v) for v in plans[name].values()),**metrics(y,oof[name])))
        summary=pd.DataFrame(summaries).sort_values(['mean_fold_rmse','total_component_features','model'])
        summary.to_csv(root/'stage2_summary.csv',index=False)
        winner=summary.iloc[0].model
        selected=plans[winner]
        # Freeze before observing seed7 confirmation. Never reselect based on that score.
        frozen=dict(name=winner,plan=selected,seed42_mean_rmse=float(summary.iloc[0].mean_fold_rmse),
                    selection='Minimum full5fold seed42 mean log RMSE; seed7 unseen at selection time')
        (root/'frozen_selection.json').write_text(json.dumps(frozen,indent=2)+'\n')
        np.savez_compressed(root/'stage2_oof.npz',actual=y,source_row_index=ids,fold=cached['fold'],**oof)
        print('FROZEN WINNER:',winner,frozen['seed42_mean_rmse'],flush=True)
        # Current A has an original unsorted geometry contract. Keep it for a fair reproduction.
        sys.path.insert(0,str(args.current_a_dir.resolve()))
        from models import FixedMixture
        from zero_gate import ZeroGatedMixture
        import features as current_a_features
        current_meta=json.loads((args.current_a_dir/'metadata.json').read_text())
        XA=current_a_features.build_feature_sets(df[current_a_features.RAW])['A34_both'].iloc[ids].to_numpy()
        confirmation=[];confirmation_oof={n:np.full(len(y),np.nan) for n in ['three_baseline','selected','leader_two_stage','current_A']}
        fold7=np.zeros(len(y),int)
        for fold in range(1,6):
            tr,va,A,B=fold_data(7,fold)
            fold7[va]=fold
            for name,plan in [('three_baseline',baseline_plan),('selected',selected)]:
                pred=predictions(plan_parts(7,fold,plan))
                confirmation_oof[name][va]=pred
                confirmation.append(dict(model=name,fold=fold,**metrics(y[va],pred)))
                print(f'CONFIRM seed7 {name} fold{fold}: {confirmation[-1]["log_rmse"]:.9f}',flush=True)
            leader=TwoStageLogSFR(HGB_FIXED400,BASE,BASE,zero_threshold=.9).fit(A[BASE],y[tr])
            pa=leader.predict(B[BASE])
            confirmation_oof['leader_two_stage'][va]=pa
            confirmation.append(dict(model='leader_two_stage',fold=fold,**metrics(y[va],pa)))
            current=ZeroGatedMixture(FixedMixture(current_meta['base_specs']).fit(XA[tr],y[tr]),.9)
            pa=current.predict(XA[va])
            confirmation_oof['current_A'][va]=pa
            confirmation.append(dict(model='current_A',fold=fold,**metrics(y[va],pa)))
            pd.DataFrame(confirmation).to_csv(root/'confirmation_folds.csv',index=False)
            print(f'CONFIRM benchmarks fold{fold}: leader={confirmation[-2]["log_rmse"]:.9f}, currentA={confirmation[-1]["log_rmse"]:.9f}',flush=True)
        confirm_df=pd.DataFrame(confirmation)
        confirm_summary=[]
        for name,pred in confirmation_oof.items():
            assert np.isfinite(pred).all()
            rows=confirm_df[confirm_df.model==name]
            confirm_summary.append(dict(model=name,mean_fold_rmse=float(rows.log_rmse.mean()),**metrics(y,pred)))
        pd.DataFrame(confirm_summary).to_csv(root/'confirmation_summary.csv',index=False)
        np.savez_compressed(root/'confirmation_oof.npz',actual=y,source_row_index=ids,fold=fold7,**confirmation_oof)
        print('Refitting frozen winner...',flush=True)
        model=FeatureSelectedThreeRegime(selected).fit(X,y)
        joblib.dump(model,root/'best_feature_model.joblib',compress=3)
        loaded=joblib.load(root/'best_feature_model.joblib')
        np.testing.assert_array_equal(loaded.predict(X.iloc[:1000]),model.predict(X.iloc[:1000]))
        reference=df.drop(columns='sfr_linear')
        check=predict_sfr(reference.iloc[ids[:1000]],loaded,reference)
        np.testing.assert_allclose(check,loaded.predict(X.iloc[:1000]),rtol=1e-6,atol=1e-9)
        meta=dict(data_git_blob_sha=data_sha,source_fingerprint=fingerprint,train_rows=len(y),
                  source_commit='21d02b5d0c829aff67cafd89da62c3386ac7864f',frozen_selection=frozen,
                  zero_threshold=.9,tail_threshold=10.,model_params=HGB_FIXED400,
                  holdout_evaluated=False,current_model_replaced=False,
                  model_sha256=hashlib.sha256((root/'best_feature_model.joblib').read_bytes()).hexdigest(),
                  checks=dict(cached_baseline_alignment=True,all_oof_finite=True,transform_fold_fitted=True,
                              saved_model_reload=True,raw_interface=True),
                  interpretation='Feature-search winner within stated scope. Seed7 tests split sensitivity on reused observations; no independent test claim.')
        (root/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
        print('SEARCH COMPLETED',flush=True)
        print(summary.to_string(index=False),flush=True)
        print(pd.DataFrame(confirm_summary).to_string(index=False),flush=True)


if __name__=='__main__':
    main()

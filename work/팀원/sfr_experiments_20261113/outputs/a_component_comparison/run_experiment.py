from pathlib import Path
import os, argparse, hashlib, json, time, warnings
os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
import numpy as np
import pandas as pd
import sklearn, catboost
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from component_models import factories, mix, HGB_PARAMS, CAT_PARAMS

FEATURES = ['log10_halo_mass','log10_stellar_mass','log10_BH_mass',
    'log10_main_prog_mass','log10_main_prog_stellar_mass','log10_mass_ratio',
    'log10_overdensity_plus_1','pos_x','pos_y','pos_z','vel_x','vel_y','vel_z']
VARIANTS = {
    'A_baseline': ('hgb_gate','hgb_mid','hgb_tail'),
    'E1_cat_gate': ('cat_gate','hgb_mid','hgb_tail'),
    'E2_spline_gate': ('spline_gate','hgb_mid','hgb_tail'),
    'E3_cat_mid': ('hgb_gate','cat_mid','hgb_tail'),
    'E4_spline_mid': ('hgb_gate','spline_mid','hgb_tail'),
    'E5_cat_tail': ('hgb_gate','hgb_mid','cat_tail'),
    'E6_extra_tail': ('hgb_gate','hgb_mid','extra_tail'),
}

def rmse(y, p):
    return float(np.sqrt(np.mean((np.log10(1+p)-np.log10(1+y))**2)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--references', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    raw = Path(args.data).read_bytes()
    sha = hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
    assert sha == '06479cedd323253f08af9983c73f395cc02d3774'
    del raw
    df = pd.read_csv(args.data)
    Xt, _, yt, _ = train_test_split(df[FEATURES], df.sfr_linear, test_size=.2,
        random_state=42, stratify=(df.sfr_linear > 0).astype(int))
    indices = Xt.index.to_numpy(); X = Xt.to_numpy(); y = yt.to_numpy()
    del df, Xt, yt, _
    ref = np.load(args.references)
    assert np.array_equal(indices, ref['source_row_index'])
    assert np.array_equal(y, ref['actual'])
    folds = ref['fold'].copy()
    fs = factories()
    parts = {k: np.zeros((len(y), 3)) if k.endswith('gate') else np.zeros(len(y)) for k in fs}
    predictions = {k: np.zeros(len(y)) for k in VARIANTS}
    details = []; start = time.time()
    with threadpool_limits(limits=4):
        for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=42).split(X), 1):
            assert np.all(folds[te] == k)
            classes = np.where(y[tr] == 0, 0, np.where(y[tr] <= 10, 1, 2))
            print(f'Fold {k}/5; class counts {np.bincount(classes).tolist()}', flush=True)
            for key, factory in fs.items():
                t = time.time()
                if key.endswith('gate'):
                    rows = tr; target = classes
                else:
                    rows = tr[classes == (1 if key.endswith('mid') else 2)]
                    target = np.log1p(y[rows])
                model = factory()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model.fit(X[rows], target)
                fit_secs = time.time()-t
                if key.endswith('gate'):
                    order = [list(model.classes_).index(c) for c in (0,1,2)]
                    p = model.predict_proba(X[te])[:, order]
                    assert np.allclose(p.sum(axis=1), 1)
                else:
                    p = np.maximum(model.predict(X[te]), 0)
                assert np.isfinite(p).all()
                parts[key][te] = p
                detail = dict(fold=k, component=key, training_rows=len(rows), fit_seconds=fit_secs,
                    warnings=[str(w.message) for w in caught])
                if hasattr(model, 'named_steps') and 'logisticregression' in model.named_steps:
                    detail['n_iter'] = model.named_steps['logisticregression'].n_iter_.tolist()
                details.append(detail)
                print(f'  {key}: {fit_secs:.1f}s; warnings={len(caught)}', flush=True)
                del model
            for name, (g,m,t) in VARIANTS.items():
                predictions[name][te] = mix(parts[g][te], parts[m][te], parts[t][te])
            # Reproduce the previously scored model before interpreting replacements.
            assert np.allclose(predictions['A_baseline'][te], ref['A_log1p_T10_soft'][te], rtol=1e-9, atol=1e-9)
            print('Fold scores: '+', '.join(f'{n}={rmse(y[te],p[te]):.6f}' for n,p in predictions.items()), flush=True)
            np.savez_compressed(out/'checkpoint.npz', completed_folds=k, **parts)
            (out/'fit_details.json').write_text(json.dumps(details, indent=2))

    single_scores = {n: rmse(y,p) for n,p in predictions.items()}
    selected = list(VARIANTS['A_baseline'])
    for slot, names in [(0,['E1_cat_gate','E2_spline_gate']), (1,['E3_cat_mid','E4_spline_mid']), (2,['E5_cat_tail','E6_extra_tail'])]:
        best = min(names, key=single_scores.get)
        if single_scores[best] < single_scores['A_baseline']:
            selected[slot] = VARIANTS[best][slot]
    selected_tuple = tuple(selected)
    combo_note = 'No new combination: selected components equal an existing candidate.'
    if selected_tuple not in VARIANTS.values():
        predictions['C_selected_combo_exploratory'] = mix(*(parts[s] for s in selected))
        combo_note = 'Components selected using these same OOF scores; exploratory, not independent validation.'

    segments = {'All':np.ones(len(y),bool),'SFR=0':y==0,'0<SFR<=3':(y>0)&(y<=3),
        '3<SFR<=10':(y>3)&(y<=10),'10<SFR<=30':(y>10)&(y<=30),
        '30<SFR<=100':(y>30)&(y<=100),'SFR>100':y>100}
    rows=[]; foldrows=[]
    for name,p in predictions.items():
        assert np.isfinite(p).all() and (p>=0).all()
        for seg,mask in segments.items():
            rows.append(dict(model=name, segment=seg, n=int(mask.sum()), rmse=rmse(y[mask],p[mask])))
        for k in range(1,6):
            mask=folds==k
            foldrows.append(dict(model=name, fold=k, rmse=rmse(y[mask],p[mask])))
    scores=pd.DataFrame(rows); fold_scores=pd.DataFrame(foldrows)
    overall=scores.query("segment=='All'").copy().sort_values('rmse')
    base=single_scores['A_baseline']
    overall['improvement_pct']=100*(base-overall.rmse)/base
    overall['fold_wins_vs_baseline']=[sum(rmse(y[folds==k],predictions[n][folds==k]) < rmse(y[folds==k],predictions['A_baseline'][folds==k]) for k in range(1,6)) for n in overall.model]
    scores.to_csv(out/'scores.csv',index=False)
    fold_scores.to_csv(out/'fold_scores.csv',index=False)
    overall.to_csv(out/'overall_comparison.csv',index=False)
    trueclasses=np.where(y==0,0,np.where(y<=10,1,2))
    gates=[dict(component=k,log_loss=float(log_loss(trueclasses,parts[k],labels=[0,1,2])),accuracy=float((parts[k].argmax(axis=1)==trueclasses).mean())) for k in fs if k.endswith('gate')]
    pd.DataFrame(gates).to_csv(out/'gate_diagnostics.csv',index=False)
    np.savez_compressed(out/'oof_predictions.npz',actual=y,fold=folds,source_row_index=indices,**predictions)
    np.savez_compressed(out/'oof_components.npz',actual=y,fold=folds,source_row_index=indices,**parts)
    metadata=dict(commit='fef00ffaf2992104c8d6dc0e87ab0b30275d8c9d', data_git_blob_sha=sha,
        features=FEATURES,n_train=len(y),holdout_evaluated=False,baseline_reproduced=True,
        cv='Original 80% train split; KFold(5,shuffle=True,random_state=42)',
        metric='RMSE in log10(1+SFR); fixed comparison convention, official zero handling not established',
        target='ln(1+SFR)',tail_threshold=10,log_offset=1,mixture='probability-weighted mean in log space',
        versions=dict(sklearn=sklearn.__version__,catboost=catboost.__version__,numpy=np.__version__,pandas=pd.__version__),
        hgb_params=HGB_PARAMS,catboost_params=CAT_PARAMS,variants=VARIANTS,selected_components=selected,
        combination_note=combo_note,elapsed_seconds=time.time()-start,
        tuning='One prespecified configuration per replacement; different capacities/training time; no hyperparameter search.',
        features_note='Spline gate adds density x stellar mass basis interaction. Spline mid adds stellar x BH and stellar x progenitor-stellar interactions. All transforms fit on fold training subset.',
        leader_hypotheses='Density helps zero classification; stellar/progenitor/BH mass helps positive regression. Previous observations were different model/metric; hypotheses only.')
    (out/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    (out/'checkpoint.npz').unlink()
    print(overall.to_string(index=False),flush=True)
    print(combo_note,flush=True)

if __name__ == '__main__':
    main()

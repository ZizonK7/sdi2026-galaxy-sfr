"""Exploratory +/-2 fitted-normal density segmentation; no notebook changes.

Fit each cut only on zero-SFR galaxies in the current training partition.
Prediction routes ALL galaxies using only their observed density.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
from pathlib import Path
import sys
import csv
import json
import time
OUT = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[3]
if os.environ.get('SFR_ANALYSIS_DEPENDENCIES'):
    sys.path.insert(0, os.environ['SFR_ANALYSIS_DEPENDENCIES'])
os.environ['MPLCONFIGDIR'] = str(OUT / '.mplconfig')
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from scipy import stats

with (ROOT / 'Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv').open(newline='', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    names = reader.fieldnames
    data = np.array([[float(row[c]) for c in names] for row in reader])
features = [c for c in names if c not in ['subhalo_id', 'sfr_linear']]
X = data[:, [names.index(c) for c in features]]
y = data[:, names.index('sfr_linear')]
d = data[:, names.index('log10_overdensity_plus_1')]
tr, va = train_test_split(np.arange(len(y)), test_size=.2, random_state=42,
                          stratify=(y > 0).astype(int))
HGB = dict(learning_rate=.08, max_iter=150, max_leaf_nodes=31, random_state=42)

def cuts_for(idx):
    v = d[idx][y[idx] == 0]
    return np.array([v.mean() - 2*v.std(), v.mean() + 2*v.std()])

def groups(idx, cuts):
    return np.searchsorted(cuts, d[idx], side='left')

def rmse(yt, p):
    return float(np.sqrt(np.mean((yt-p)**2)))

def fit_predict(train, test):
    cuts = cuts_for(train)
    tg, vg = groups(train, cuts), groups(test, cuts)
    single = HistGradientBoostingRegressor(**HGB).fit(X[train], y[train])
    base = single.predict(X[test])
    divided = np.full(len(test), np.nan)
    iterations = []
    for g in range(3):
        gi = train[tg == g]
        model = HistGradientBoostingRegressor(**HGB).fit(X[gi], y[gi])
        divided[vg == g] = model.predict(X[test[vg == g]])
        iterations.append(int(model.n_iter_))
    assert np.isfinite(divided).all()
    return base, divided, cuts, {'pooled': int(single.n_iter_), 'segments': iterations}

def paired_ci(yt, a, b, B=1000, seed=42):
    """Percentile paired iid-row bootstrap: RMSE(b) - RMSE(a)."""
    ea, eb = (a-yt)**2, (b-yt)**2
    rng = np.random.default_rng(seed)
    diffs = []
    for start in range(0, B, 25):
        ix = rng.integers(0, len(yt), size=(min(25, B-start), len(yt)), dtype=np.int32)
        diffs.extend(np.sqrt(eb[ix].mean(axis=1))-np.sqrt(ea[ix].mean(axis=1)))
    return [float(np.sqrt(eb.mean())-np.sqrt(ea.mean())),
            *[float(v) for v in np.quantile(diffs, [.025, .975])]]

start = time.time()
base, divided, cuts, iterations = fit_predict(tr, va)
# Existing adopted two-stage model is included as a reference, with identical settings.
classifier = HistGradientBoostingClassifier(**HGB).fit(X[tr], (y[tr]>0).astype(int))
positive_idx = tr[y[tr]>0]
positive_reg = HistGradientBoostingRegressor(**HGB).fit(X[positive_idx], y[positive_idx])
hurdle = classifier.predict_proba(X[va])[:, 1] * np.maximum(positive_reg.predict(X[va]), 0)
predictions = {'pooled_HGB':base, 'three_segment_HGB':divided, 'existing_two_stage':hurdle}
print(json.dumps({'stage':'validation fit complete','seconds':time.time()-start,
                  'cuts':cuts.tolist(),'RMSE':{k:rmse(y[va],p) for k,p in predictions.items()} }), flush=True)

metrics = []
vg = groups(va, cuts)
masks = {'all':np.ones(len(va),dtype=bool), 'density_low':vg==0, 'density_middle':vg==1,
         'density_high':vg==2, 'SFR_zero':y[va]==0, 'SFR_positive':y[va]>0,
         'SFR_top10pct':y[va]>=np.quantile(y[va], .9)}
for name, mask in masks.items():
    for model_name, p in predictions.items():
        metrics.append({'population':name,'model':model_name,'n':int(mask.sum()),
                        'RMSE':rmse(y[va][mask],p[mask]),
                        'MAE':float(np.mean(np.abs(y[va][mask]-p[mask])))})
cis = {name:paired_ci(y[va][mask],base[mask],divided[mask]) for name,mask in masks.items()}
vs_hurdle = paired_ci(y[va],hurdle,divided)
group_stats = []
for partition, idx in [('train',tr),('val',va)]:
    grp = groups(idx, cuts)
    for g in range(3):
        yt = y[idx][grp==g]
        group_stats.append({'partition':partition,'group':g,'n':len(yt),
                            'zeros':int((yt==0).sum()),'positive':int((yt>0).sum()),
                            'zero_fraction':float(np.mean(yt==0)), 'mean_SFR':float(yt.mean()),
                            'max_SFR':float(yt.max())})

# Cuts are re-estimated on each fold's training data, never its held-out targets.
folds = []
oof_base, oof_divided = np.full(len(tr),np.nan), np.full(len(tr),np.nan)
for i,(ft,fe) in enumerate(KFold(n_splits=5,shuffle=True,random_state=42).split(tr),1):
    p0,p1,fc,ni = fit_predict(tr[ft],tr[fe])
    oof_base[fe], oof_divided[fe] = p0,p1
    r0,r1 = rmse(y[tr[fe]],p0),rmse(y[tr[fe]],p1)
    row = {'fold':i,'cut_low':float(fc[0]),'cut_high':float(fc[1]),
           'pooled_RMSE':r0,'segmented_RMSE':r1,'delta_RMSE':r1-r0,
           'heldout_max_SFR':float(y[tr[fe]].max()),'iterations':ni}
    folds.append(row)
    print(json.dumps({'stage':'CV','result':row,'seconds':time.time()-start}),flush=True)

report = {'features':features,'parameters':HGB,'cuts':cuts.tolist(),
          'cut_definition':'mean +/- 2 population standard deviations of log-density among training SFR=0 galaxies',
          'boundary_rule':'low: density <= lower; middle: lower < density <= upper; high: density > upper',
          'training_rows':len(tr),'validation_rows':len(va),
          'group_statistics':group_stats,'validation_metrics':metrics,
          'validation_delta_RMSE_and_paired_bootstrap_95CI_vs_pooled':cis,
          'validation_delta_RMSE_and_paired_bootstrap_95CI_vs_existing_two_stage':vs_hurdle,
          'validation_iterations':iterations,'cv_folds':folds,
          'cv_pooled_OOF_RMSE':rmse(y[tr],oof_base),
          'cv_segmented_OOF_RMSE':rmse(y[tr],oof_divided),
          'cv_fold_mean_delta':float(np.mean([r['delta_RMSE'] for r in folds])),
          'cv_fold_delta_sd':float(np.std([r['delta_RMSE'] for r in folds],ddof=1)),
          'cv_pooled_OOF_MAE':float(np.mean(np.abs(y[tr]-oof_base))),
          'cv_segmented_OOF_MAE':float(np.mean(np.abs(y[tr]-oof_divided))),
          'limitations':[
              'Exploratory cut rule suggested after inspecting training EDA; CV is not an independent confirmation of the rule selection.',
              'Single existing validation split; row-wise bootstrap assumes independent galaxies and omits model-fit uncertainty.',
              'No hyperparameter tuning; default early_stopping=auto may differ with segment sample size.',
              'Separate regression significance is not tested; predictive error is compared directly.',
              'SFR=0 density distribution does not directly identify P(SFR=0 | density).'],
          'elapsed_seconds':time.time()-start}
(OUT/'density_segment_regression_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
for filename,rows in [('density_segment_metrics.csv',metrics),('density_segment_group_sizes.csv',group_stats)]:
    with (OUT/filename).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
np.savez_compressed(OUT/'density_segment_predictions.npz',validation_row_indices=va,y_true=y[va],
                    density=d[va],group=vg,**predictions)
print(json.dumps(report,ensure_ascii=True,indent=2),flush=True)

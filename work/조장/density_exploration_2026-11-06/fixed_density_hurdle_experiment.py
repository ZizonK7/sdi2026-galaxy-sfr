"""Middle-density hurdle regression with a frozen density-only Gaussian Bayes gate.

Class priors and class-conditional densities use the training partition only.
q0(d) uses a truncated normal for zero SFR and a truncated KDE for positive SFR.
All hurdle comparisons share exactly the same positive-only regression.
Low/high predictions remain the original segment regressions for full-model metrics.
"""
import os
os.environ['OMP_NUM_THREADS']='4'
os.environ['OPENBLAS_NUM_THREADS']='4'
from pathlib import Path
import sys, csv, json, time
OUT=Path(__file__).parent
ROOT = Path(__file__).resolve().parents[3]
if os.environ.get('SFR_ANALYSIS_DEPENDENCIES'):
    sys.path.insert(0, os.environ['SFR_ANALYSIS_DEPENDENCIES'])
import numpy as np
from scipy import stats,optimize,special
from sklearn.model_selection import train_test_split,KFold
from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
from sklearn.metrics import log_loss,brier_score_loss,roc_auc_score

with (ROOT/'Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv').open(newline='',encoding='utf-8-sig') as f:
    reader=csv.DictReader(f); names=reader.fieldnames
    data=np.array([[float(row[c]) for c in names] for row in reader])
features=[c for c in names if c not in ['subhalo_id','sfr_linear']]
X=data[:,[names.index(c) for c in features]]
y=data[:,names.index('sfr_linear')]
d=data[:,names.index('log10_overdensity_plus_1')]
tr,va=train_test_split(np.arange(len(y)),test_size=.2,random_state=42,stratify=(y>0).astype(int))
HGB=dict(learning_rate=.08,max_iter=150,max_leaf_nodes=31,random_state=42)
METHODS=['direct_segment_regression','fixed_gaussian_density_gate','learned_density_only_gate','learned_all_features_gate']

def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
def cuts_for(ix):
    z=d[ix][y[ix]==0]
    return np.array([z.mean()-2*z.std(),z.mean()+2*z.std()])

class GaussianDensityGate:
    def fit(self,dd,yy,lo,hi):
        self.lo,self.hi=lo,hi
        z=dd[yy==0]; positive=dd[yy>0]
        self.prior=float(len(z)/len(dd))
        def nll(theta):
            mu,logsd=theta; sd=np.exp(logsd)
            return -np.sum(stats.truncnorm.logpdf(z,(lo-mu)/sd,(hi-mu)/sd,loc=mu,scale=sd))
        result=optimize.minimize(nll,[z.mean(),np.log(z.std())],method='L-BFGS-B',
                                 bounds=[(None,None),(-10,5)])
        assert result.success, result.message
        self.mu=float(result.x[0]);self.sd=float(np.exp(result.x[1]))
        self.kde=stats.gaussian_kde(positive,bw_method='scott')
        self.positive_norm=float(self.kde.integrate_box_1d(lo,hi))
        self.meta={'zero_prior':self.prior,'zero_truncated_normal_loc':self.mu,
                   'zero_truncated_normal_scale':self.sd,'positive_kde_bandwidth':float(np.sqrt(self.kde.covariance[0,0])),
                   'positive_kde_support_normalization':self.positive_norm,
                   'training_zeros':len(z),'training_positive':len(positive)}
        return self
    def predict(self,dd):
        log0=stats.truncnorm.logpdf(dd,(self.lo-self.mu)/self.sd,(self.hi-self.mu)/self.sd,
                                   loc=self.mu,scale=self.sd)
        fp=np.maximum(self.kde(dd)/self.positive_norm,np.finfo(float).tiny)
        logodds=np.log(self.prior/(1-self.prior))+log0-np.log(fp)
        q=special.expit(logodds)
        assert np.isfinite(q).all() and np.all((q>=0)&(q<=1))
        return q

def fit_partition(train,test,save_grid=False):
    cuts=cuts_for(train)
    tg=np.searchsorted(cuts,d[train],side='left')
    vg=np.searchsorted(cuts,d[test],side='left')
    full_direct=np.full(len(test),np.nan)
    for g in range(3):
        m=HistGradientBoostingRegressor(**HGB).fit(X[train[tg==g]],y[train[tg==g]])
        full_direct[vg==g]=m.predict(X[test[vg==g]])
    mt=train[tg==1];mv=test[vg==1];pos=mt[y[mt]>0]
    reg=HistGradientBoostingRegressor(**HGB).fit(X[pos],y[pos])
    raw=reg.predict(X[mv]);positive_mean=np.maximum(raw,0)
    gate=GaussianDensityGate().fit(d[mt],y[mt],*cuts)
    q=gate.predict(d[mv])
    labels=(y[mt]==0).astype(int)
    density_clf=HistGradientBoostingClassifier(**HGB).fit(d[mt,None],labels)
    all_clf=HistGradientBoostingClassifier(**HGB).fit(X[mt],labels)
    density_q=density_clf.predict_proba(d[mv,None])[:,1]
    all_q=all_clf.predict_proba(X[mv])[:,1]
    mid_pred={'direct_segment_regression':full_direct[vg==1],
              'fixed_gaussian_density_gate':(1-q)*positive_mean,
              'learned_density_only_gate':(1-density_q)*positive_mean,
              'learned_all_features_gate':(1-all_q)*positive_mean}
    full_pred={}
    for key,p in mid_pred.items():
        full_pred[key]=full_direct.copy();full_pred[key][vg==1]=p
    ztrue=(y[mv]==0).astype(int)
    gate_metrics={}
    for key,qp in [('fixed_gaussian_density_gate',q),('learned_density_only_gate',density_q),('learned_all_features_gate',all_q)]:
        gate_metrics[key]={'log_loss':float(log_loss(ztrue,qp)),
                           'Brier':float(brier_score_loss(ztrue,qp)),
                           'ROC_AUC':float(roc_auc_score(ztrue,qp)),
                           'mean_predicted_zero_probability':float(qp.mean()),
                           'observed_zero_fraction':float(ztrue.mean())}
    meta={'cuts':cuts.tolist(),'gate_fit':gate.meta,'positive_regression_iterations':int(reg.n_iter_),
          'positive_predictions_clipped':int(np.sum(raw<0)), 'middle_train_n':len(mt),
          'middle_test_n':len(mv),'gate_classification_metrics':gate_metrics}
    if save_grid:
        grid=np.linspace(cuts[0]+1e-8,cuts[1]-1e-8,250)
        np.savez_compressed(OUT/'fixed_density_gate_curves.npz',density_grid=grid,
                            gaussian_q=gate.predict(grid),density_HGB_q=density_clf.predict_proba(grid[:,None])[:,1],
                            validation_middle_density=d[mv],validation_middle_iszero=ztrue,
                            validation_middle_q=q,positive_mean=positive_mean,
                            validation_middle_y=y[mv],validation_middle_indices=mv)
    return full_pred,vg,meta

def paired_ci(yt,a,b,B=1000):
    ea,eb=(a-yt)**2,(b-yt)**2;rng=np.random.default_rng(42);diff=[]
    for start in range(0,B,25):
        ii=rng.integers(0,len(yt),(min(25,B-start),len(yt)),dtype=np.int32)
        diff.extend(np.sqrt(eb[ii].mean(axis=1))-np.sqrt(ea[ii].mean(axis=1)))
    return [float(np.sqrt(eb.mean())-np.sqrt(ea.mean())),*[float(v) for v in np.quantile(diff,[.025,.975])]]

start=time.time()
pred,vg,meta=fit_partition(tr,va,save_grid=True)
previous=np.load(OUT/'density_segment_predictions.npz')
assert np.array_equal(previous['validation_row_indices'],va)
assert np.allclose(previous['three_segment_HGB'],pred['direct_segment_regression'],rtol=1e-10,atol=1e-10)
assert np.array_equal(pred['fixed_gaussian_density_gate'][vg!=1],pred['direct_segment_regression'][vg!=1])
print(json.dumps({'stage':'validation','meta':meta,'RMSE':{k:rmse(y[va],p) for k,p in pred.items()},'seconds':time.time()-start}),flush=True)
masks={'all':np.ones(len(va),dtype=bool),'middle':vg==1,
       'middle_zero':(vg==1)&(y[va]==0),'middle_positive':(vg==1)&(y[va]>0),
       'middle_SFR_gt100':(vg==1)&(y[va]>100)}
metrics=[];cis={}
for pop,mask in masks.items():
    cis[pop]={}
    for key,p in pred.items():
        metrics.append({'population':pop,'method':key,'n':int(mask.sum()),
                        'RMSE':rmse(y[va][mask],p[mask]),'MAE':float(np.mean(np.abs(y[va][mask]-p[mask]))),
                        'mean_prediction':float(p[mask].mean()),'mean_true_SFR':float(y[va][mask].mean())})
        if key!='direct_segment_regression':
            cis[pop][key]=paired_ci(y[va][mask],pred['direct_segment_regression'][mask],p[mask])

oof={k:np.full(len(tr),np.nan) for k in METHODS};mid_oof=np.zeros(len(tr),dtype=bool);folds=[]
for i,(ft,fe) in enumerate(KFold(n_splits=5,shuffle=True,random_state=42).split(tr),1):
    fp,fg,fmeta=fit_partition(tr[ft],tr[fe])
    mid_oof[fe]=(fg==1)
    for key in METHODS: oof[key][fe]=fp[key]
    fr={'fold':i,'cuts':fmeta['cuts'],'all_RMSE':{k:rmse(y[tr[fe]],p) for k,p in fp.items()},
        'middle_RMSE':{k:rmse(y[tr[fe]][fg==1],p[fg==1]) for k,p in fp.items()},
        'gate_classification_metrics':fmeta['gate_classification_metrics']}
    folds.append(fr)
    print(json.dumps({'stage':'CV','fold':fr,'seconds':time.time()-start}),flush=True)
cv={}
for pop,mask in [('all',np.ones(len(tr),dtype=bool)),('middle',mid_oof)]:
    cv[pop]={key:{'RMSE':rmse(y[tr][mask],p[mask]),'MAE':float(np.mean(np.abs(y[tr][mask]-p[mask])))} for key,p in oof.items()}
report={'experiment':'Frozen density-only zero probability times shared positive-only regression, middle segment only',
        'validation':meta,'features':features,'parameters':HGB,'validation_metrics':metrics,
        'validation_delta_RMSE_vs_direct_and_paired_bootstrap_95CI':cis,
        'CV':cv,'CV_folds':folds,'elapsed_seconds':time.time()-start,
        'limitations':['Exploratory experiment after prior inspection of this validation set; no fresh final test set.',
                      'Normality of density conditional on zero SFR is approximate; posterior also depends on positive density estimate.',
                      'Positive-class KDE uses Scott bandwidth and is renormalized on the middle interval; no bandwidth tuning.',
                      'All density fits, priors and segment cutoffs refitted within each CV training fold.',
                      'Bootstrap assumes iid validation galaxies and conditions on the fitted models.',
                      'Default early_stopping=auto retained; no hyperparameter optimization.']}
(OUT/'fixed_density_hurdle_results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'fixed_density_hurdle_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
np.savez_compressed(OUT/'fixed_density_hurdle_predictions.npz',validation_row_indices=va,group=vg,y_true=y[va],**pred)
print(json.dumps({'validation_metrics':metrics,'CIs':cis,'CV':cv,'seconds':time.time()-start},indent=2),flush=True)

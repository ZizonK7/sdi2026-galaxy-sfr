from pathlib import Path
import os, json, time, argparse, hashlib
os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from a_log_mixture import LogTailMixtureSFR

ap=argparse.ArgumentParser()
ap.add_argument('--data',required=True)
ap.add_argument('--references',required=True,help='oof_predictions.npz from log_sfr_reevaluation')
ap.add_argument('--out',required=True)
args=ap.parse_args()
out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
data=Path(args.data); b=data.read_bytes()
sha=hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest();del b
assert sha=='06479cedd323253f08af9983c73f395cc02d3774'
df=pd.read_csv(data); features=[c for c in df if c not in ['subhalo_id','sfr_linear']]
Xa,_,ya,_=train_test_split(df[features],df.sfr_linear,test_size=.2,random_state=42,stratify=(df.sfr_linear>0).astype(int))
indices=Xa.index.to_numpy();X=Xa.to_numpy();y=ya.to_numpy();del df,Xa,ya,_
ref=np.load(args.references)
assert np.array_equal(indices,ref['source_row_index']) and np.array_equal(y,ref['actual'])
preds={k:ref[k].copy() for k in ['A_soft_T30','A_hard_T30','B_T30','HGB_log1p']}
preds['HGB_log_eps001']=np.zeros(len(y))
for tag in ['log1p','log_eps001']:
 for suffix in ['T30_soft','T30_linear_mix','T30_hard','T30_bounded','T10_soft']:
  preds[f'A_{tag}_{suffix}']=np.zeros(len(y))
folds=list(KFold(5,shuffle=True,random_state=42).split(X)); fold_id=ref['fold']
hgb=dict(learning_rate=.08,max_iter=150,max_leaf_nodes=31,random_state=42)
details=[];start=time.time()
with threadpool_limits(limits=4):
 for k,(tr,te) in enumerate(folds,1):
  assert np.all(fold_id[te]==k)
  print(f'Fold {k}/5: log-space A, fixed expert settings',flush=True)
  for offset,tag in [(1.,'log1p'),(.001,'log_eps001')]:
   for T in [30,10]:
    m=LogTailMixtureSFR(tail_threshold=T,log_offset=offset)
    assert clone(m).get_params()==m.get_params()
    m.fit(X[tr],y[tr])
    prob,zm,zt=m.parts(X[te])
    assert np.allclose(prob.sum(axis=1),1)
    soft=m.predict(X[te]);preds[f'A_{tag}_T{T}_soft'][te]=soft
    assert np.isfinite(soft).all() and (soft>=0).all()
    assert np.allclose(np.log(soft+offset),prob[:,0]*np.log(offset)+prob[:,1]*zm+prob[:,2]*zt)
    if T==30:
     for route,space,suffix in [('soft','linear','linear_mix'),('hard','log','hard')]:
      preds[f'A_{tag}_T30_{suffix}'][te]=m.combine(prob,zm,zt,offset,route,space)
     m.clip_to_regime=True
     preds[f'A_{tag}_T30_bounded'][te]=m.predict(X[te])
    details.append(dict(fold=k,offset=offset,T=T,class_counts=m.class_counts_.tolist(),
        classifier_trees=m.classifier_.n_iter_,mid_trees=m.mid_regressor_.n_iter_,tail_trees=m.tail_regressor_.n_iter_))
   if offset!=1:
    zhat=HistGradientBoostingRegressor(**hgb).fit(X[tr],np.log(y[tr]+offset)).predict(X[te])
    preds['HGB_log_eps001'][te]=np.maximum(np.exp(zhat)-offset,0)
  print(f'Fold {k} complete, elapsed {time.time()-start:.1f}s',flush=True)

np.savez_compressed(out/'oof_predictions.npz',actual=y,fold=fold_id,source_row_index=indices,**preds)
rows=[];foldrows=[]
segments={'All':np.ones(len(y),bool),'SFR=0':y==0,'0<SFR<=3':(y>0)&(y<=3),'3<SFR<=30':(y>3)&(y<=30),'30<SFR<=100':(y>30)&(y<=100),'SFR>100':y>100}
for name,p in preds.items():
 assert np.isfinite(p).all()
 for metric,offset in [('log10_1plus',1.),('log10_plus_1e-3',.001),('log10_plus_1e-6',.000001)]:
  err=np.log10(np.maximum(p,0)+offset)-np.log10(y+offset)
  for seg,mask in segments.items():
   rows.append(dict(model=name,metric=metric,segment=seg,n=int(mask.sum()),rmse=float(np.sqrt(np.mean(err[mask]**2))),zero_predictions_pct=float((p[mask]==0).mean()*100)))
  for k in range(1,6):
   foldrows.append(dict(model=name,metric=metric,fold=k,rmse=float(np.sqrt(np.mean(err[fold_id==k]**2)))))
scores=pd.DataFrame(rows);scores.to_csv(out/'scores.csv',index=False)
pd.DataFrame(foldrows).to_csv(out/'fold_scores.csv',index=False)
tab=scores.query("segment=='All'").pivot(index='model',columns='metric',values='rmse').sort_values('log10_1plus')
tab.to_csv(out/'overall_comparison.csv')
meta=dict(commit='fef00ffaf2992104c8d6dc0e87ab0b30275d8c9d',data_git_blob_sha=sha,
    n_train=len(y),holdout_evaluated=False,sklearn=sklearn.__version__,cv='Original train split; KFold(5, shuffle=True, random_state=42)',
    hgb_params=hgb,features=features,elapsed_seconds=time.time()-start,
    scenario_note='Offsets define separate hypothetical evaluation scenarios; not tuned hyperparameters or confirmed official formulas.',
    experiment_note='A retains 3-class gate; experts learn log(y+offset); soft mode mixes in log space before inverse transform. Ablations compare linear-space mixture, hard routing, support clipping, and T=10.',
    learning_details=details)
(out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
print(tab.round(6).to_string(),flush=True)

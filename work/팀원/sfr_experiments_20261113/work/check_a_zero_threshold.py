from pathlib import Path
import json
import numpy as np
import pandas as pd

source=Path('outputs/a_feature_comparison')
out=Path('outputs/a_zero_threshold');out.mkdir(exist_ok=True,parents=True)
d=np.load(source/'oof_predictions.npz');q=np.load(source/'oof_probabilities.npz')
for key in ['actual','fold','source_row_index']:
    np.testing.assert_array_equal(d[key],q[key])
y=d['actual'];fold=d['fold'];soft=d['A34_both'];p0=q['A34_both'][:,0]
mask=p0>=.9
gated=np.where(mask,0.,soft)
err_soft=(np.log10(1+soft)-np.log10(1+y))**2
err_gate=(np.log10(1+gated)-np.log10(1+y))**2
segments={'All':np.ones(len(y),bool),'SFR=0':y==0,'0<SFR<=3':(y>0)&(y<=3),
    '3<SFR<=10':(y>3)&(y<=10),'10<SFR<=30':(y>10)&(y<=30),
    '30<SFR<=100':(y>30)&(y<=100),'SFR>100':y>100}
rows=[]
for name,m in segments.items():
    a=float(np.sqrt(err_soft[m].mean()));b=float(np.sqrt(err_gate[m].mean()))
    rows.append(dict(segment=name,n=int(m.sum()),soft_rmse=a,gate_rmse=b,delta=b-a,
        gated_rows=int((mask&m).sum())))
fr=[]
for k in range(1,6):
    m=fold==k
    a=float(np.sqrt(err_soft[m].mean()));b=float(np.sqrt(err_gate[m].mean()))
    fr.append(dict(fold=k,soft_rmse=a,gate_rmse=b,delta=b-a))
false_zero=mask&(y>0);true_zero=mask&(y==0)
details=dict(model='A34_both',threshold=.9,metric='RMSE log10(1+SFR)',
    n=len(y),gated_rows=int(mask.sum()),true_zero_gated=int(true_zero.sum()),
    positive_gated=int(false_zero.sum()),
    zero_output_precision=float((y[mask]==0).mean()),
    true_zero_recall=float(mask[y==0].mean()),
    positive_gated_actual_median=float(np.median(y[false_zero])),
    positive_gated_actual_max=float(y[false_zero].max()),
    gated_soft_prediction_median=float(np.median(soft[mask])),
    zero_mse_change_total_weight=float((err_gate[y==0]-err_soft[y==0]).sum()/len(y)),
    positive_mse_change_total_weight=float((err_gate[y>0]-err_soft[y>0]).sum()/len(y)),
    improved_folds=sum(r['delta']<0 for r in fr),
    interpretation='Threshold fixed by user; post-processing existing OOF predictions, no retraining or threshold optimization; holdout not evaluated.')
pd.DataFrame(rows).to_csv(out/'scores.csv',index=False)
pd.DataFrame(fr).to_csv(out/'fold_scores.csv',index=False)
(out/'summary.json').write_text(json.dumps(details,ensure_ascii=False,indent=2))
print(pd.DataFrame(rows).to_string(index=False))
print(pd.DataFrame(fr).to_string(index=False))
print(json.dumps(details,ensure_ascii=False,indent=2))

from pathlib import Path
import argparse, hashlib, json, os, platform, time
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).parent / 'mpl-cache'))
import numpy as np
import pandas as pd
import scipy, sklearn, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
data = Path(args.data)
raw = data.read_bytes()
blob_sha = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
assert blob_sha == '06479cedd323253f08af9983c73f395cc02d3774', blob_sha
del raw
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
exec((Path(__file__).parent / 'exact_models.py').read_text(), globals())
df = pd.read_csv(data)
features = [c for c in df.columns if c not in ['subhalo_id', 'sfr_linear']]
# Same split and row ordering as james_tail.ipynb. Holdout is never evaluated.
X_train, _, y_train, _ = train_test_split(
    df[features], df.sfr_linear, test_size=.20, random_state=42,
    stratify=(df.sfr_linear > 0).astype(int))
ids = df.loc[X_train.index, 'subhalo_id'].to_numpy()
source_index = X_train.index.to_numpy()
X = X_train.reset_index(drop=True); y = y_train.to_numpy()
assert len(y) == 166749
del df, X_train, y_train, _
folds = list(KFold(5, shuffle=True, random_state=42).split(X))
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
names = ['p_positive', 'p_tail_given_positive', 'e_mid', 'e_tail', 'prediction', 'two_stage']
arrays = {k: np.zeros(len(y)) for k in names}
fold_id = np.zeros(len(y), dtype=int)
fold_rows = []
start = time.time()
with threadpool_limits(limits=4):
    for k, (tr, te) in enumerate(folds, 1):
        print(f'Fold {k}/5: fitting unchanged B(T=30) and TwoStageSFR', flush=True)
        b = NestedTailSFR(HGB, 30).fit(X.iloc[tr], y[tr])
        two = TwoStageSFR(HGB).fit(X.iloc[tr], y[tr])
        parts = b.parts(X.iloc[te])
        for name, a in zip(names[:4], parts): arrays[name][te] = a
        arrays['prediction'][te] = b.predict(X.iloc[te])
        arrays['two_stage'][te] = two.predict(X.iloc[te])
        fold_id[te] = k
        row = dict(fold=k, n=len(te), n_gt100=int((y[te]>100).sum()),
                   n_gt300=int((y[te]>300).sum()),
                   B_rmse=rmse(y[te], arrays['prediction'][te]),
                   two_stage_rmse=rmse(y[te], arrays['two_stage'][te]),
                   positive_classifier_trees=b.clf_zero_.n_iter_,
                   tail_classifier_trees=b.clf_tail_.n_iter_,
                   mid_regressor_trees=b.reg_mid_.n_iter_,
                   tail_regressor_trees=b.reg_tail_.n_iter_)
        fold_rows.append(row)
        print(json.dumps(row), flush=True)

p, q, em, et, pred, old = [arrays[n] for n in names]
mix = q*et + (1-q)*em
# Exact signed residual identity, interpreted only for actual tail y>30:
# y - prediction = (y-e_tail) + (1-p)*e_tail + p*(1-q)*(e_tail-e_mid).
regression_gap = y-et
positive_gate_drop = (1-p)*et
tail_gate_drop = p*(1-q)*(et-em)
assert np.allclose(y-pred, regression_gap+positive_gate_drop+tail_gate_drop)
assert np.allclose(pred, p*mix)
assert np.all(np.isfinite(pred)) and pred.min() >= 0
assert np.array_equal(np.unique(fold_id), np.arange(1, 6))
segments = {
    'SFR=0': y==0,
    '0<SFR<=30': (y>0)&(y<=30),
    '30<SFR<=100': (y>30)&(y<=100),
    '100<SFR<=300': (y>100)&(y<=300),
    'SFR>300': y>300,
    'SFR>100': y>100,
    'SFR>30': y>30,
}
rows=[]
for name, m in segments.items():
    r = dict(segment=name, n=int(m.sum()), actual_mean=float(y[m].mean()),
        final_mean=float(pred[m].mean()), final_rmse=rmse(y[m], pred[m]),
        old_rmse=rmse(y[m], old[m]), p_positive_mean=float(p[m].mean()),
        p_tail_mean=float(q[m].mean()), p_tail_median=float(np.median(q[m])),
        p_tail_p10=float(np.quantile(q[m], .1)),
        p_tail_below_05_pct=float((q[m]<.5).mean()*100),
        p_tail_below_09_pct=float((q[m]<.9).mean()*100),
        e_mid_mean=float(em[m].mean()), e_tail_mean=float(et[m].mean()),
        e_tail_rmse=rmse(y[m], et[m]),
        mean_y_minus_e_tail=float(regression_gap[m].mean()),
        mean_positive_gate_drop=float(positive_gate_drop[m].mean()),
        mean_tail_gate_drop=float(tail_gate_drop[m].mean()),
        final_mean_underprediction=float((y[m]-pred[m]).mean()),
        sse_share_pct=float(((y[m]-pred[m])**2).sum()/((y-pred)**2).sum()*100))
    rows.append(r)
summary = pd.DataFrame(rows)
summary.to_csv(out/'segment_diagnosis.csv', index=False)
pd.DataFrame(fold_rows).to_csv(out/'fold_metrics.csv', index=False)

# Labels below are used ONLY for an unattainable diagnostic counterfactual.
# Do not use oracle predictions for model selection, submission, or claim them as bounds.
oracle_tail_mix = np.where(y>30, et, em)
counterfactuals = {
    'actual_B': pred,
    'oracle_tail_membership_only': p*oracle_tail_mix,
    'oracle_positive_membership_only': (y>0)*mix,
    'oracle_both_memberships': (y>0)*oracle_tail_mix,
}
oracle_rows=[]
for name, a in counterfactuals.items():
    for seg, m in {'All':np.ones(len(y),bool), **segments}.items():
        oracle_rows.append(dict(diagnostic=name,segment=seg,n=int(m.sum()),rmse=rmse(y[m],a[m])))
pd.DataFrame(oracle_rows).to_csv(out/'oracle_diagnostics_NOT_achievable.csv',index=False)

row_table = pd.DataFrame({'source_row_index':source_index, 'subhalo_id':ids,
    'fold':fold_id, 'actual_sfr':y, **arrays,
    'regression_gap':regression_gap,'positive_gate_drop':positive_gate_drop,
    'tail_gate_drop':tail_gate_drop,'squared_error':(y-pred)**2})
row_table.loc[y>100].sort_values('squared_error',ascending=False).to_csv(out/'high_sfr_209_rows.csv',index=False)
np.savez_compressed(out/'oof_components.npz', actual=y, source_row_index=source_index,
    subhalo_id=ids, fold=fold_id, **arrays)

meta = dict(repository='ZizonK7/sdi2026-galaxy-sfr',
    commit='fef00ffaf2992104c8d6dc0e87ab0b30275d8c9d',data_git_blob_sha=blob_sha,
    train_n=len(y),holdout_evaluated=False,threshold=30,seed=42,
    cv='KFold(5, shuffle=True, random_state=42) on original training split',
    features=features, model_source='Exact class definitions from james_tail.ipynb cells 3 and 14',
    python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
    scipy=scipy.__version__,sklearn=sklearn.__version__,matplotlib=matplotlib.__version__,
    threads=4, elapsed_seconds=time.time()-start,
    B_oof_rmse=rmse(y,pred),two_stage_oof_rmse=rmse(y,old),
    expected_saved_B_rounded=8.798,expected_saved_two_stage_rounded=9.126,
    reproduced_to_3dp=round(rmse(y,pred),3)==8.798 and round(rmse(y,old),3)==9.126,
    identity_max_error=float(np.max(np.abs((y-pred)-(regression_gap+positive_gate_drop+tail_gate_drop)))),
    limitations=['Signed residual decomposition, not causal attribution or SSE decomposition.',
        'Oracle membership uses true targets only for diagnosis and is not deployable or a guaranteed bound.',
        'No new model selected and no holdout evaluation.',
        'Existing model was selected using these folds; these are diagnostic estimates.'])
(out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
print('METADATA', json.dumps(meta,ensure_ascii=False), flush=True)
print(summary.to_string(index=False), flush=True)
print(pd.DataFrame(oracle_rows).query("segment in ['All','SFR>100','SFR>300']").to_string(index=False),flush=True)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig, ax=plt.subplots(1,2,figsize=(12.5,4.8))
tail_summary=summary.set_index('segment').loc[['30<SFR<=100','100<SFR<=300','SFR>300']]
xx=np.arange(3); bottom=np.zeros(3)
for key,label,color in [('mean_y_minus_e_tail','Tail regressor underprediction','#3159a7'),
    ('mean_tail_gate_drop','Tail probability mixing','#e79a38'),
    ('mean_positive_gate_drop','Positive probability','#8297b1')]:
    v=tail_summary[key].to_numpy()
    ax[0].bar(xx,v,bottom=bottom,label=label,color=color)
    bottom+=v
ax[0].set_xticks(xx,['30 < SFR ≤ 100','100 < SFR ≤ 300','SFR > 300'])
ax[0].set_ylabel('Mean true SFR − final prediction')
ax[0].set_title('Where the mean underprediction comes from')
ax[0].legend(fontsize=8)
hi=y>100
ax[1].scatter(y[hi],et[hi],s=16,alpha=.65,label='Tail regressor alone',color='#3159a7')
ax[1].scatter(y[hi],pred[hi],s=16,alpha=.65,label='Final B prediction',color='#e79a38')
ax[1].plot([30,2500],[30,2500],'--',color='gray',lw=1)
ax[1].set_xscale('log');ax[1].set_yscale('log')
ax[1].set_xlabel('True SFR');ax[1].set_ylabel('OOF predicted SFR')
ax[1].set_title('High-SFR galaxies (n=209)');ax[1].legend(fontsize=8)
fig.suptitle('B(T=30): unchanged model, training-only 5-fold OOF',fontsize=13)
fig.tight_layout();fig.savefig(out/'diagnosis.png',dpi=170,bbox_inches='tight')
plt.close(fig)

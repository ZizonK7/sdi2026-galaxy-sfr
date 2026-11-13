from pathlib import Path
import os, json, hashlib, time
os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, KFold
from threadpoolctl import threadpool_limits

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--data',required=True)
parser.add_argument('--components',required=True,help='oof_components.npz from prior B diagnosis')
parser.add_argument('--out',default='log_sfr_reevaluation')
args=parser.parse_args()
OUT=Path(args.out); OUT.mkdir(parents=True,exist_ok=True)
RANDOM_SEED=42
np.random.seed(42)
# 제출본 7절의 모델을 그대로 (설정 변경 없음)
HGB = dict(learning_rate=0.08, max_iter=150, max_leaf_nodes=31, random_state=RANDOM_SEED)

class TwoStageSFR:
    """분류기 P(y>0|x)와 양수 은하만으로 학습한 회귀기 E[y|y>0,x]를 곱하는 모델 (제출본 7절과 같음)."""
    def __init__(self, hgb_params, reg_loss="squared_error", clip_zero=True):
        self.hgb_params, self.reg_loss, self.clip_zero = hgb_params, reg_loss, clip_zero
    def fit(self, X, y):
        y = np.asarray(y, dtype=float); pos = y > 0
        self.clf_ = HistGradientBoostingClassifier(**self.hgb_params).fit(X, pos.astype(int))
        self.reg_ = HistGradientBoostingRegressor(loss=self.reg_loss, **self.hgb_params).fit(X[pos], y[pos])
        return self
    def predict_proba_positive(self, X): return self.clf_.predict_proba(X)[:, 1]
    def predict_positive_mean(self, X):
        e = self.reg_.predict(X); return np.clip(e, 0, None) if self.clip_zero else e
    def predict(self, X): return self.predict_proba_positive(X) * self.predict_positive_mean(X)


# 실험 A. 꼬리 전문 모델 정의
class TailMixtureSFR:
    """3부류 분류기 P(0/보통/꼬리 | x) × 부류별 회귀기. mode='soft'는 확률로 섞고, 'hard'는 가장 큰 확률의 부류로 보낸다."""
    def __init__(self, hgb_params, tail_threshold, mode="soft"):
        self.hgb_params, self.T, self.mode = hgb_params, tail_threshold, mode
    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        cls = np.where(y == 0, 0, np.where(y <= self.T, 1, 2))
        self.clf_ = HistGradientBoostingClassifier(**self.hgb_params).fit(X, cls)
        self.reg_mid_ = HistGradientBoostingRegressor(**self.hgb_params).fit(X[cls == 1], y[cls == 1])
        self.reg_tail_ = HistGradientBoostingRegressor(**self.hgb_params).fit(X[cls == 2], y[cls == 2])
        self.n_tail_ = int((cls == 2).sum())
        return self
    def parts(self, X):
        P = self.clf_.predict_proba(X)
        e_mid = np.clip(self.reg_mid_.predict(X), 0, None)
        e_tail = np.clip(self.reg_tail_.predict(X), 0, None)
        return P, e_mid, e_tail
    def predict(self, X):
        P, e_mid, e_tail = self.parts(X)
        if self.mode == "soft":
            return P[:, 1] * e_mid + P[:, 2] * e_tail
        k = P.argmax(axis=1)
        return np.select([k == 1, k == 2], [e_mid, e_tail], 0.0)

# 실험 B. 모델 정의
class NestedTailSFR:
    """P(>0) × [P(꼬리|>0)·E꼬리 + (1−P(꼬리|>0))·E보통]. 0 판단은 현재 2단계 모델과 같은 2부류 분류기."""
    def __init__(self, hgb_params, tail_threshold):
        self.hgb_params, self.T = hgb_params, tail_threshold
    def fit(self, X, y):
        y = np.asarray(y, dtype=float); pos = y > 0
        Xp, yp = X[pos], y[pos]; tail = yp > self.T
        self.clf_zero_ = HistGradientBoostingClassifier(**self.hgb_params).fit(X, pos.astype(int))
        self.clf_tail_ = HistGradientBoostingClassifier(**self.hgb_params).fit(Xp, tail.astype(int))
        self.reg_mid_ = HistGradientBoostingRegressor(**self.hgb_params).fit(Xp[~tail], yp[~tail])
        self.reg_tail_ = HistGradientBoostingRegressor(**self.hgb_params).fit(Xp[tail], yp[tail])
        return self
    def parts(self, X):
        p_pos = self.clf_zero_.predict_proba(X)[:, 1]
        p_tail = self.clf_tail_.predict_proba(X)[:, 1]
        e_mid = np.clip(self.reg_mid_.predict(X), 0, None)
        e_tail = np.clip(self.reg_tail_.predict(X), 0, None)
        return p_pos, p_tail, e_mid, e_tail
    def predict(self, X):
        p_pos, p_tail, e_mid, e_tail = self.parts(X)
        return p_pos * (p_tail * e_tail + (1 - p_tail) * e_mid)

data=Path(args.data)
b=data.read_bytes(); sha=hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest();del b
assert sha=='06479cedd323253f08af9983c73f395cc02d3774'
df=pd.read_csv(data)
features=[c for c in df if c not in ['subhalo_id','sfr_linear']]
Xtrain,_,ytrain,_=train_test_split(df[features],df.sfr_linear,test_size=.2,random_state=42,stratify=(df.sfr_linear>0).astype(int))
source_index=Xtrain.index.to_numpy();X=Xtrain.reset_index(drop=True);y=ytrain.to_numpy();del df,Xtrain,ytrain,_
z=np.load(args.components)
assert np.array_equal(z['source_row_index'],source_index) and np.array_equal(z['actual'],y)
preds={'TwoStage':z['two_stage'].copy(),'B_T30':z['prediction'].copy()}
names=['HGB_linear','HGB_log1p','HGB_log1p_smearing','A_soft_T10','A_soft_T30','A_soft_T100','A_hard_T30','B_T10','C_leaf10_300','C_leaf5_300','C_weight_sqrt','C_weight_linear','C_leaf5_sqrt']
for n in names:preds[n]=np.zeros(len(y))
variants={
 'C_leaf10_300':(dict(min_samples_leaf=10,max_iter=300),None),
 'C_leaf5_300':(dict(min_samples_leaf=5,max_iter=300),None),
 'C_weight_sqrt':({},.5),'C_weight_linear':({},1.),
 'C_leaf5_sqrt':(dict(min_samples_leaf=5,max_iter=300),.5)}
folds=list(KFold(5,shuffle=True,random_state=42).split(X))
fold_id=z['fold'];start=time.time()
with threadpool_limits(limits=4):
 for k,(tr,te) in enumerate(folds,1):
  assert np.all(fold_id[te]==k)
  print(f'Fold {k}/5: original A/B/C and existing log1p reference',flush=True)
  Xa,ya,Xb=X.iloc[tr],y[tr],X.iloc[te]
  preds['HGB_linear'][te]=HistGradientBoostingRegressor(**HGB).fit(Xa,ya).predict(Xb)
  logmodel=HistGradientBoostingRegressor(**HGB).fit(Xa,np.log1p(ya))
  logs=logmodel.predict(Xb)
  preds['HGB_log1p'][te]=np.expm1(logs)
  smear=np.exp(np.log1p(ya)-logmodel.predict(Xa)).mean()
  preds['HGB_log1p_smearing'][te]=np.exp(logs)*smear-1
  for T in [10,30,100]:
   m=TailMixtureSFR(HGB,T,'soft').fit(Xa,ya)
   preds[f'A_soft_T{T}'][te]=m.predict(Xb)
   if T==30:
    m.mode='hard';preds['A_hard_T30'][te]=m.predict(Xb)
  preds['B_T10'][te]=NestedTailSFR(HGB,10).fit(Xa,ya).predict(Xb)
  tail=ya>30
  p=z['p_positive'][te];q=z['p_tail_given_positive'][te];em=z['e_mid'][te]
  for name,(extra,alpha) in variants.items():
   w=None if alpha is None else ya[tail]**alpha
   m=HistGradientBoostingRegressor(**{**HGB,**extra}).fit(Xa[tail],ya[tail],sample_weight=w)
   preds[name][te]=p*(q*np.maximum(m.predict(Xb),0)+(1-q)*em)
  print(f'Fold {k} complete, elapsed {time.time()-start:.1f}s',flush=True)
  np.savez_compressed(OUT/'oof_predictions.npz',actual=y,fold=fold_id,source_row_index=source_index,**preds)

expected={'TwoStage':9.126,'B_T30':8.798,'HGB_linear':9.057,'A_soft_T10':8.856,'A_soft_T30':8.932,'A_soft_T100':11.136,'A_hard_T30':9.768,'B_T10':8.818,'C_leaf10_300':8.953,'C_leaf5_300':8.943,'C_weight_sqrt':9.030,'C_weight_linear':9.176,'C_leaf5_sqrt':8.821}
checks={n:dict(observed=float(np.sqrt(np.mean((y-preds[n])**2))),expected_rounded=e) for n,e in expected.items()}
assert all(round(v['observed'],3)==v['expected_rounded'] for v in checks.values()),checks
segments={'All':np.ones(len(y),bool),'SFR=0':y==0,'0<SFR<=3':(y>0)&(y<=3),'3<SFR<=30':(y>3)&(y<=30),'30<SFR<=100':(y>30)&(y<=100),'SFR>100':y>100,'SFR>300':y>300}
rows=[]; foldrows=[]
for n,raw in preds.items():
 assert np.isfinite(raw).all()
 clipped=np.maximum(raw,0)
 for metric,err in [('linear_raw',raw-y),('log10_1plus',np.log10(clipped+1)-np.log10(y+1)),('log10_plus_1e-3',np.log10(clipped+1e-3)-np.log10(y+1e-3)),('log10_plus_1e-6',np.log10(clipped+1e-6)-np.log10(y+1e-6))]:
  for seg,mask in segments.items():
   rows.append(dict(model=n,metric=metric,segment=seg,n=int(mask.sum()),rmse=float(np.sqrt(np.mean(err[mask]**2))),sse_share_pct=float(np.sum(err[mask]**2)/np.sum(err**2)*100),negative_predictions=int((raw<0).sum())))
  for k in range(1,6):
   foldrows.append(dict(model=n,metric=metric,fold=k,rmse=float(np.sqrt(np.mean(err[fold_id==k]**2)))))
scores=pd.DataFrame(rows);scores.to_csv(OUT/'scores.csv',index=False)
pd.DataFrame(foldrows).to_csv(OUT/'fold_scores.csv',index=False)
alltab=scores.query("segment=='All'").pivot(index='model',columns='metric',values='rmse').sort_values('log10_1plus')
alltab.to_csv(OUT/'overall_comparison.csv')
meta=dict(commit='fef00ffaf2992104c8d6dc0e87ab0b30275d8c9d',data_git_blob_sha=sha,sklearn=sklearn.__version__,n_train=len(y),holdout_evaluated=False,
 metrics_note='Scenario scores, not official scores: exact zero handling remains unspecified. Negative predictions clipped to 0 for all log scores; raw predictions for original linear RMSE.',
 model_note='Original model classes/settings. A_hard_T30 reuses A_soft_T30 fitted model; C reuses unchanged B_T30 OOF gates and mid regressor. Log1p and smearing reference reproduce definitions in midterm notebook.',
 reproduction_checks=checks,elapsed_seconds=time.time()-start)
(OUT/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
print(alltab.round(6).to_string(),flush=True)
print('B error contributions',scores.query("model=='B_T30'").to_string(index=False),flush=True)

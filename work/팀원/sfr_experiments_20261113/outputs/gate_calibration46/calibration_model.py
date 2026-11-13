"""Cross-fit sigmoid/isotonic maps for the two hierarchical binary gates."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from hierarchical_model import positive_probability, hierarchical_probabilities
from feature_pipeline import UNIVERSE
from three_regime import combine

VARIANTS={'baseline':(None,'none')}
VARIANTS.update({f'{method}_{scope}':(method,scope)
                for method in ('sigmoid','isotonic') for scope in ('positive','tail','both')})

def logit(p):
    p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7)
    return np.log(p)-np.log1p(-p)

class ProbabilityMap:
    def __init__(self,method):
        if method not in ('sigmoid','isotonic'):raise ValueError(method)
        self.method=method

    def fit(self,p,labels):
        p=np.asarray(p,dtype=float);labels=np.asarray(labels)
        if p.ndim!=1 or p.shape!=labels.shape or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
            raise ValueError('Expected finite probability and label vectors')
        if set(np.unique(labels))!={0,1}:raise ValueError('Need both binary classes')
        if self.method=='sigmoid':
            self.model_=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000,tol=1e-8).fit(logit(p).reshape(-1,1),labels)
        else:
            self.model_=IsotonicRegression(y_min=0.,y_max=1.,increasing=True,out_of_bounds='clip').fit(p,labels)
        return self

    def predict(self,p):
        p=np.asarray(p,dtype=float)
        if not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('Invalid probability')
        if self.method=='sigmoid':return self.model_.predict_proba(logit(p).reshape(-1,1))[:,1]
        return self.model_.predict(p)

def fit_maps(probabilities,actual):
    y=np.asarray(actual);positive=y>0
    result={}
    for method in ('sigmoid','isotonic'):
        result[method]=dict(
            positive=ProbabilityMap(method).fit(probabilities[:,0],positive.astype(int)),
            tail=ProbabilityMap(method).fit(probabilities[positive,1],(y[positive]>10).astype(int)))
    return result

def apply_maps(probabilities,maps,variant):
    method,scope=VARIANTS[variant]
    p=np.asarray(probabilities)[:,0].copy();q=np.asarray(probabilities)[:,1].copy()
    if scope in ('positive','both'):p=maps[method]['positive'].predict(p)
    if scope in ('tail','both'):q=maps[method]['tail'].predict(q)
    return np.column_stack([p,q])

class CalibratedSFR:
    def __init__(self,base_model,maps,variant):
        if variant not in VARIANTS:raise ValueError(variant)
        self.base_model=base_model
        self.maps=maps
        self.variant=variant
        self.zero_threshold=.9
        self.tail_threshold=10.

    def predict_components(self,X):
        base=self.base_model
        Z=base.engineer_.transform(X)[UNIVERSE]
        raw=np.column_stack([positive_probability(base.positive_gate_,Z),positive_probability(base.conditional_tail_gate_,Z)])
        calibrated=apply_maps(raw,self.maps,self.variant)
        p=hierarchical_probabilities(calibrated[:,0],calibrated[:,1])
        return p,base.mid_regressor_.predict(Z),base.tail_regressor_.predict(Z)

    def predict_log(self,X):return combine(*self.predict_components(X),self.zero_threshold)

    def predict(self,X):return np.maximum(10.**self.predict_log(X)-1.,0.)

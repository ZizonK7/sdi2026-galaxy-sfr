"""Adopted output policy: force SFR=0 when the three-class gate's P(0)>=.9."""
import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted
from models import part,mix

def apply_zero_threshold(soft_prediction,p_zero,threshold=.9):
    if not 0<=threshold<=1:raise ValueError('threshold must be in [0,1]')
    soft=np.asarray(soft_prediction);p=np.asarray(p_zero)
    if soft.ndim!=1 or soft.shape!=p.shape:raise ValueError('Expected equal one-dimensional arrays')
    if not np.isfinite(soft).all() or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError('Predictions must be finite and probabilities in [0,1]')
    return np.where(p>=threshold,0.,soft)

class ZeroGatedMixture:
    """Prediction wrapper around an already fitted A34 FixedMixture.

    Does not retrain or change the underlying classifier/regressors.
    Inputs must be the original 34 features in the recorded order.
    """
    def __init__(self,base_model,zero_threshold=.9):
        if not 0<=zero_threshold<=1:raise ValueError('threshold must be in [0,1]')
        check_is_fitted(base_model,'models_')
        self.base_model=base_model
        self.zero_threshold=zero_threshold
        self.n_features_in_=base_model.n_features_in_

    def predict_with_details(self,X):
        X=check_array(X,ensure_all_finite='allow-nan')
        if X.shape[1]!=self.n_features_in_:raise ValueError('Feature count mismatch')
        parts={s:part(m,s,X) for s,m in self.base_model.models_.items()}
        soft=mix(parts);p0=parts['gate'][:,0]
        return apply_zero_threshold(soft,p0,self.zero_threshold),soft,p0

    def predict(self,X):
        return self.predict_with_details(X)[0]

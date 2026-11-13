"""A34 component factories and an estimator for a fixed selected specification."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

HGB=dict(learning_rate=.08,max_iter=150,max_leaf_nodes=31,random_state=42)
GRID={
 'cat':[dict(iterations=300,depth=6),dict(iterations=600,depth=8)],
 'lgb':[dict(n_estimators=300,num_leaves=31),dict(n_estimators=600,num_leaves=63)],
 'extra':[dict(n_estimators=300,min_samples_leaf=3),dict(n_estimators=300,min_samples_leaf=5),dict(n_estimators=300,min_samples_leaf=10)],
}
FAMILIES={'gate':['cat','lgb'],'mid':['cat','lgb'],'tail':['cat','lgb','extra']}
BASE={s:dict(family='hgb',params={}) for s in FAMILIES}

def make_model(slot,spec):
    family=spec['family'];p=spec['params']
    if family=='hgb':
        return (HistGradientBoostingClassifier if slot=='gate' else HistGradientBoostingRegressor)(**HGB)
    if family=='cat':
        common=dict(learning_rate=.05,l2_leaf_reg=5,random_seed=42,thread_count=4,
                    allow_writing_files=False,verbose=False,nan_mode='Min',**p)
        return (CatBoostClassifier(loss_function='MultiClass',**common) if slot=='gate'
                else CatBoostRegressor(loss_function='RMSE',**common))
    if family=='lgb':
        common=dict(learning_rate=.05,reg_lambda=5.,min_child_samples=20,
                    random_state=42,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True,**p)
        return (LGBMClassifier(objective='multiclass',**common) if slot=='gate'
                else LGBMRegressor(objective='regression',**common))
    if family=='extra' and slot=='tail':
        return ExtraTreesRegressor(criterion='squared_error',max_features=1.,n_jobs=4,random_state=42,**p)
    raise ValueError((slot,spec))

def as_input(model,X):
    # Consistent feature names avoid sklearn warnings in LightGBM's wrapper.
    return pd.DataFrame(X,columns=[f'f{i}' for i in range(X.shape[1])]) if isinstance(model,(LGBMClassifier,LGBMRegressor)) else X

def labels(y):
    return np.where(y==0,0,np.where(y<=10,1,2))

def fit_part(slot,spec,X,y):
    lab=labels(y)
    mask=np.ones(len(y),bool) if slot=='gate' else lab==(1 if slot=='mid' else 2)
    target=lab if slot=='gate' else np.log(y[mask]+1.)
    model=make_model(slot,spec)
    model.fit(as_input(model,X[mask]),target)
    return model

def part(model,slot,X):
    if slot=='gate':
        order=[list(model.classes_).index(c) for c in (0,1,2)]
        p=model.predict_proba(as_input(model,X))[:,order]
        assert np.allclose(p.sum(axis=1),1.)
        return p
    return np.maximum(model.predict(as_input(model,X)),0.)

def mix(parts):
    p=parts['gate']
    return np.maximum(np.exp(p[:,1]*parts['mid']+p[:,2]*parts['tail'])-1.,0.)

def score(y,pred):
    return float(np.sqrt(np.mean((np.log10(1+y)-np.log10(1+pred))**2)))

class FixedMixture(RegressorMixin,BaseEstimator):
    def __init__(self,specs=None):
        self.specs=specs
    def fit(self,X,y):
        X,y=check_X_y(X,y,ensure_all_finite='allow-nan',y_numeric=True)
        if (y<0).any() or not np.array_equal(np.unique(labels(y)),[0,1,2]):
            raise ValueError('Need nonnegative targets and all three regimes')
        self.n_features_in_=X.shape[1]
        self.models_={s:fit_part(s,p,X,y) for s,p in (self.specs or BASE).items()}
        return self
    def predict(self,X):
        check_is_fitted(self,'models_')
        X=check_array(X,ensure_all_finite='allow-nan')
        if X.shape[1]!=self.n_features_in_:raise ValueError('Feature count mismatch')
        return mix({s:part(m,s,X) for s,m in self.models_.items()})

"""Fixed candidates for a tail-only ablation of the hierarchical46 model."""
from sklearn.ensemble import ExtraTreesRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

CANDIDATES={
 'lgb_400_31':dict(family='LightGBM',params=dict(n_estimators=400,num_leaves=31,learning_rate=.08,reg_lambda=0.)),
 'lgb_600_63':dict(family='LightGBM',params=dict(n_estimators=600,num_leaves=63,learning_rate=.05,reg_lambda=5.)),
 'cat_400_depth6':dict(family='CatBoost',params=dict(iterations=400,depth=6,learning_rate=.08,l2_leaf_reg=3.)),
 'cat_600_depth8':dict(family='CatBoost',params=dict(iterations=600,depth=8,learning_rate=.05,l2_leaf_reg=5.)),
 'extra_300_leaf3':dict(family='ExtraTrees',params=dict(n_estimators=300,min_samples_leaf=3)),
 'extra_300_leaf10':dict(family='ExtraTrees',params=dict(n_estimators=300,min_samples_leaf=10)),
}


def make_tail(name):
    config=CANDIDATES[name]
    p=config['params']
    if config['family']=='LightGBM':
        return LGBMRegressor(objective='regression',min_child_samples=20,random_state=42,
                            n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True,**p)
    if config['family']=='CatBoost':
        return CatBoostRegressor(loss_function='RMSE',random_seed=42,thread_count=4,
                                 allow_writing_files=False,verbose=False,nan_mode='Min',**p)
    if config['family']=='ExtraTrees':
        return ExtraTreesRegressor(criterion='squared_error',max_features=1.,random_state=42,n_jobs=4,**p)
    raise ValueError(name)

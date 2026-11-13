"""Extreme-target sample weights apply only to the tail expert's training loss."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from lightgbm import LGBMRegressor
from leader_model import HGB_FIXED400

WEIGHTS = (1, 2, 3, 5)
CONFIGS = {f'{family}_w{weight}': (family, weight)
           for family in ('hgb', 'lgb') for weight in WEIGHTS}

def sample_weights(actual_tail_sfr, multiplier):
    y = np.asarray(actual_tail_sfr, dtype=float)
    if y.ndim != 1 or not np.isfinite(y).all() or (y <= 10).any():
        raise ValueError('Tail training rows must have finite true SFR>10')
    if multiplier not in WEIGHTS:
        raise ValueError('Use one of the prespecified multipliers: 1,2,3,5')
    return np.where(y > 100, float(multiplier), 1.)

def make_tail(family):
    if family == 'hgb':
        return HistGradientBoostingRegressor(**HGB_FIXED400)
    if family == 'lgb':
        return LGBMRegressor(objective='regression', n_estimators=400,
            num_leaves=31, learning_rate=.08, reg_lambda=0.,
            min_child_samples=20, random_state=42, n_jobs=4, verbosity=-1,
            deterministic=True, force_col_wise=True)
    raise ValueError(family)

def fit_tail(name, features, actual_tail_sfr):
    family, multiplier = CONFIGS[name]
    y = np.asarray(actual_tail_sfr, dtype=float)
    weights = sample_weights(y, multiplier)
    model = make_tail(family)
    # Preserve the exact unweighted baseline fitting path for multiplier=1.
    kwargs = {} if multiplier == 1 else {'sample_weight': weights}
    return model.fit(features, np.log10(1+y), **kwargs)

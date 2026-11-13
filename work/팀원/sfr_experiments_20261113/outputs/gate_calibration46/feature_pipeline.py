"""Target-free feature engineering and per-component feature selection."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from leader_model import (FINAL_FEATURES, HGB_FIXED400, build_features,
                          prepare_prediction_union, RHO_CRIT)
from three_regime import combine

BASE = FINAL_FEATURES + ['log_rho_5nn','log_rho_10nn','log1p_N_r1','log1p_N_r2','log1p_N_r5',
 'dist_host13.5','r_over_R200_13.5','v_over_V200_13.5','host_logm_13.5',
 'dist_host14','r_over_R200_14','v_over_V200_14','fof_log_richness','fof_is_central',
 'fof_group_max_logm','fof_dv_group','halo_minus_stellar','dlogMs_growth','dlogMh_growth','bh_to_star','prog_star_to_halo']
ADDITIONS = {
 'mass_growth': ['mass_growth_product'],
 'merger_growth': ['merger_flag','merger_growth_product','merger_strength_growth_product'],
 'growth_residual': ['stellar_growth_residual_at_mass'],
 'bh_residual': ['bh_residual_at_stellar_mass'],
 'companions': ['nn_log_distance','nn_log_stellar_mass_ratio','nn_relative_speed',
                'nn_r_over_primary_R200','nn3_max_log_stellar_mass_ratio','nn3_min_relative_speed'],
}
REMOVALS = {
 'positions': ['pos_x','pos_y','pos_z'],
 'velocities': ['vel_x','vel_y','vel_z'],
 'phase_space': ['pos_x','pos_y','pos_z','vel_x','vel_y','vel_z'],
 'progenitor_masses': ['log10_main_prog_mass','log10_main_prog_stellar_mass'],
 'raw_merger': ['log10_mass_ratio'],
 'raw_bh': ['log10_BH_mass'],
 'host14': ['dist_host14','r_over_R200_14','v_over_V200_14'],
 'group_velocity': ['fof_dv_group'],
}
UNIVERSE = BASE + [c for cols in ADDITIONS.values() for c in cols]


def companion_features(frame, box=200.):
    """Nearest three OTHER galaxies in the periodic box; no SFR or ID input."""
    raw = frame[FINAL_FEATURES].copy()
    order = raw.sort_values(FINAL_FEATURES, kind='mergesort', na_position='last').index
    ordered = raw.loc[order]
    pos = ordered[['pos_x','pos_y','pos_z']].to_numpy() % box
    if len(pos) < 4 or len(np.unique(pos, axis=0)) != len(pos):
        raise ValueError('Need at least four distinct periodic positions')
    distance, neighbors = cKDTree(pos, boxsize=box).query(pos, k=4)
    assert np.array_equal(neighbors[:,0], np.arange(len(pos)))
    neighbors, distance = neighbors[:,1:], distance[:,1:]
    vel = ordered[['vel_x','vel_y','vel_z']].to_numpy()
    stellar = ordered.log10_stellar_mass.to_numpy()
    speed = np.linalg.norm(vel[neighbors] - vel[:,None,:], axis=2)
    ratios = stellar[neighbors] - stellar[:,None]
    radius = (3 * 10.**ordered.log10_halo_mass.to_numpy() / (4*np.pi*200*RHO_CRIT))**(1/3)
    out = pd.DataFrame({
        'nn_log_distance':np.log10(distance[:,0]),
        'nn_log_stellar_mass_ratio':ratios[:,0],
        'nn_relative_speed':speed[:,0],
        'nn_r_over_primary_R200':distance[:,0]/radius,
        'nn3_max_log_stellar_mass_ratio':ratios.max(axis=1),
        'nn3_min_relative_speed':speed.min(axis=1),
    }, index=ordered.index)
    return out.reindex(frame.index)


def build_reference_features(catalogue):
    X = build_features(catalogue)
    assert list(X.columns) == BASE
    return X.join(companion_features(catalogue))


class ConditionalMean:
    def fit(self, mass, value):
        edges = np.unique(np.quantile(mass, np.linspace(0,1,21)[1:-1]))
        groups = np.searchsorted(edges, mass, side='right')
        self.mass_ = np.array([np.median(mass[groups==i]) for i in np.unique(groups)])
        self.mean_ = np.array([value[groups==i].mean() for i in np.unique(groups)])
        return self

    def predict(self, mass):
        return np.interp(mass, self.mass_, self.mean_)


class FeatureEngineer:
    def fit(self, X):
        m = X.log10_stellar_mass.to_numpy()
        self.growth_ = ConditionalMean().fit(m, X.dlogMs_growth.to_numpy())
        self.bh_ = ConditionalMean().fit(m, X.log10_BH_mass.to_numpy())
        self.fit_rows_ = len(X)
        return self

    def transform(self, X):
        out = X.copy()
        m, g, ratio = X.log10_stellar_mass.to_numpy(), X.dlogMs_growth.to_numpy(), X.log10_mass_ratio.to_numpy()
        out['mass_growth_product'] = (m - 10.) * g
        out['merger_flag'] = (ratio > -6.).astype(float)
        out['merger_growth_product'] = (ratio > -6.) * g
        out['merger_strength_growth_product'] = np.maximum(ratio + 6., 0.) * g
        out['stellar_growth_residual_at_mass'] = g - self.growth_.predict(m)
        out['bh_residual_at_stellar_mass'] = X.log10_BH_mass.to_numpy() - self.bh_.predict(m)
        assert set(out.columns) == set(UNIVERSE)
        return out[UNIVERSE]


def fit_component(slot, columns, X, y):
    labels = np.where(y == 0, 0, np.where(y <= 10., 1, 2))
    if slot == 'gate':
        return HistGradientBoostingClassifier(**HGB_FIXED400).fit(X[columns], labels)
    mask = labels == (1 if slot == 'mid' else 2)
    return HistGradientBoostingRegressor(**HGB_FIXED400).fit(X.loc[mask,columns], np.log10(1+y[mask]))


def component_prediction(model, slot, columns, X):
    if slot == 'gate':
        order = [list(model.classes_).index(i) for i in (0,1,2)]
        return model.predict_proba(X[columns])[:, order]
    return model.predict(X[columns])


class FeatureSelectedThreeRegime:
    def __init__(self, plan):
        self.plan = {k:list(v) for k,v in plan.items()}

    def fit(self, X, y):
        y = np.asarray(y, float)
        self.engineer_ = FeatureEngineer().fit(X)
        Z = self.engineer_.transform(X)
        self.models_ = {s:fit_component(s, cols, Z, y) for s,cols in self.plan.items()}
        return self

    def predict_components(self, X):
        Z = self.engineer_.transform(X)
        return tuple(component_prediction(self.models_[s], s, self.plan[s], Z) for s in ['gate','mid','tail'])

    def predict_log(self, X):
        return combine(*self.predict_components(X), .9)

    def predict(self, X):
        return np.maximum(10.**self.predict_log(X)-1.,0.)


def predict_sfr(new_table, model, reference):
    both, locations = prepare_prediction_union(new_table, reference)
    X = build_reference_features(both).iloc[locations]
    return model.predict(X)

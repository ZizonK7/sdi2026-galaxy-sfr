"""Feature ablation using one fixed, unlabeled reference galaxy catalogue.

Environment coordinates use periodic box=200, FoF b=.2, host cutoffs 13.5/14.
Do not run separately on prediction mini-batches: this changes the environment.
Keep the same unique catalogue union, then select model rows by original index.
"""
import numpy as np
import pandas as pd
from leader_environment import build_environment_features

RAW = ['log10_halo_mass','log10_stellar_mass','log10_BH_mass',
       'log10_main_prog_mass','log10_main_prog_stellar_mass','log10_mass_ratio',
       'log10_overdensity_plus_1','pos_x','pos_y','pos_z','vel_x','vel_y','vel_z']


def build_feature_sets(frame):
    raw = frame[RAW].copy()  # Pass neither IDs nor SFR to feature functions.
    if not np.isfinite(raw.to_numpy()).all():
        raise ValueError('Raw feature catalogue must contain finite values')
    if len(raw) < 11:
        raise ValueError('At least 11 catalogue galaxies required for 10-NN')
    pos = raw[['pos_x','pos_y','pos_z']].to_numpy() % 200.
    if len(np.unique(pos,axis=0)) != len(pos):
        raise ValueError('Duplicate periodic coordinates require explicit handling')
    if not (raw.log10_halo_mass >= 14.).any():
        raise ValueError('Reference catalogue has no halo above host threshold 14')
    mass = pd.DataFrame({
        'dlogMs_growth':raw.log10_stellar_mass-raw.log10_main_prog_stellar_mass,
        'dlogMh_growth':raw.log10_halo_mass-raw.log10_main_prog_mass,
        'bh_to_star':raw.log10_BH_mass-raw.log10_stellar_mass,
        'prog_star_to_halo':raw.log10_main_prog_stellar_mass-raw.log10_main_prog_mass,
    },index=raw.index)
    env = build_environment_features(raw,box=200.,fof_b=.2,host_thresholds=(13.5,14.))
    assert len(env.columns)==17 and not np.isinf(env.to_numpy()).any()
    assert not env.drop(columns='fof_dv_group').isna().any().any()
    return {
        'A13_baseline':raw,
        'A17_mass':pd.concat([raw,mass],axis=1),
        'A30_environment':pd.concat([raw,env],axis=1),
        'A34_both':pd.concat([raw,env,mass],axis=1),
    }

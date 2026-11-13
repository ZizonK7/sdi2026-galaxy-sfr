import unittest
import numpy as np
import pandas as pd
from feature_pipeline import companion_features, FINAL_FEATURES, FeatureEngineer, BASE, ADDITIONS, UNIVERSE


class Tests(unittest.TestCase):
    def test_periodic_companion_excludes_self_and_target(self):
        X = pd.DataFrame({c:np.ones(4) for c in FINAL_FEATURES})
        X[['pos_x','pos_y','pos_z']] = [[.1,0,0],[199.9,0,0],[20,0,0],[80,0,0]]
        X[['vel_x','vel_y','vel_z']] = [[0,0,0],[3,4,0],[0,0,0],[0,0,0]]
        X.log10_stellar_mass = [9.,10.,8.,7.]
        X.log10_halo_mass = 12.
        result = companion_features(X)
        self.assertAlmostEqual(result.loc[0,'nn_log_distance'], np.log10(.2))
        self.assertAlmostEqual(result.loc[0,'nn_log_stellar_mass_ratio'], 1.)
        self.assertAlmostEqual(result.loc[0,'nn_relative_speed'], 5.)
        pd.testing.assert_frame_equal(result, companion_features(X.assign(sfr_linear=[1.,2.,3.,999.])))
        pd.testing.assert_frame_equal(result, companion_features(X.iloc[::-1]).reindex(X.index))

    def test_engineer_keeps_fitted_reference_and_batch_invariance(self):
        columns = BASE + ADDITIONS['companions']
        X = pd.DataFrame({c:np.zeros(100) for c in columns})
        X.log10_stellar_mass = np.linspace(8,12,100)
        X.dlogMs_growth = np.linspace(0,1,100)
        X.log10_BH_mass = np.linspace(5,8,100)
        model = FeatureEngineer().fit(X)
        nodes = model.growth_.mean_.copy()
        result = model.transform(X)
        self.assertEqual(list(result.columns), UNIVERSE)
        pd.testing.assert_frame_equal(result.iloc[:3], model.transform(X.iloc[:3]))
        np.testing.assert_array_equal(model.growth_.mean_, nodes)


if __name__ == '__main__':
    unittest.main()

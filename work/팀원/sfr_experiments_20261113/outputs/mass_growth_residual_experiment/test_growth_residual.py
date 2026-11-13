import unittest
import numpy as np
import pandas as pd
from growth_residual import MassGrowthResidual, FEATURE


class Tests(unittest.TestCase):
    def test_training_reference_interpolation_and_frozen_transform(self):
        train = pd.DataFrame({'log10_stellar_mass': [1., 2., 3., 4.], 'dlogMs_growth': [1., 3., 5., 7.]})
        model = MassGrowthResidual(n_bins=2).fit(train)
        # Medians 1.5,3.5; average growth 2,6. At mass 2.5, expected growth=4.
        query = pd.DataFrame({'log10_stellar_mass': [2.5, 99.], 'dlogMs_growth': [5., 9.]})
        result = model.transform(query)
        np.testing.assert_allclose(result[FEATURE], [1., 3.])
        np.testing.assert_allclose(model.growth_nodes_, [2., 6.])
        pd.testing.assert_frame_equal(model.transform(query).iloc[:1], model.transform(query.iloc[:1]))
        self.assertNotIn(FEATURE, query)

    def test_tied_mass_and_unrelated_target_do_not_change_feature(self):
        X = pd.DataFrame({'log10_stellar_mass': [1., 1., 1.], 'dlogMs_growth': [0., 1., 2.], 'sfr_linear': [0., 1., 999.]})
        a = MassGrowthResidual().fit(X)
        b = MassGrowthResidual().fit(X.assign(sfr_linear=[999., 0., 1.]))
        np.testing.assert_array_equal(a.transform(X)[FEATURE], b.transform(X)[FEATURE])
        np.testing.assert_allclose(a.transform(X)[FEATURE], [-1., 0., 1.])


if __name__ == '__main__':
    unittest.main()

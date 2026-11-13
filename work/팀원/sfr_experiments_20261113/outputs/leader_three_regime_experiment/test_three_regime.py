import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from three_regime import ThreeRegimeLogSFR, combine
from zero_preserved import redistribute


class Tests(unittest.TestCase):
    def test_preserved_binary_zero_probability(self):
        p = np.array([[.99, .002, .008], [1., 0., 0.]])
        new = redistribute(p, np.array([.8, .1]))
        np.testing.assert_allclose(new, [[.2, .16, .64], [.9, .05, .05]])
        np.testing.assert_allclose(new.sum(axis=1), 1.)

    def test_log_mixture_and_inclusive_gate(self):
        p = np.array([[.9, .05, .05], [.899, .05, .051], [.1, .6, .3]])
        z = combine(p, np.ones(3), np.full(3, 2.))
        np.testing.assert_allclose(z, [0., .152, 1.2])

    def test_experts_use_true_regimes(self):
        # A classifier that cannot predict at all must not affect expert training membership.
        with patch('three_regime.HistGradientBoostingClassifier') as clf, patch('three_regime.HistGradientBoostingRegressor') as reg:
            X = pd.DataFrame({'x': [10., 20., 30., 40.]}, index=[9, 2, 8, 3])
            y = np.array([0., 1., 10., 11.])
            model = ThreeRegimeLogSFR(tail_threshold=10).fit(X, y)
            np.testing.assert_array_equal(clf.return_value.fit.call_args.args[1], [0, 1, 1, 2])
            calls = reg.return_value.fit.call_args_list
            self.assertEqual(calls[0].args[0].index.tolist(), [2, 8])
            self.assertEqual(calls[1].args[0].index.tolist(), [3])
            np.testing.assert_allclose(calls[1].args[1], np.log10([12.]))
            self.assertEqual(model.training_counts_, [1, 2, 1])
            clf.return_value.fit.return_value.predict_proba.assert_not_called()


if __name__ == '__main__':
    unittest.main()

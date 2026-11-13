import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from weighted_tail import fit_tail, sample_weights


class WeightedTailTests(unittest.TestCase):
    def test_strict_extreme_boundary(self):
        np.testing.assert_array_equal(sample_weights([10.001, 100., 100.001], 5), [1., 1., 5.])

    def test_reject_non_tail_training_rows(self):
        for actual in ([0., 200.], [10., 200.], [np.nan, 200.]):
            with self.assertRaises(ValueError):
                sample_weights(actual, 2)

    def test_constant_inputs_predict_weighted_log_mean(self):
        # No feature can split these rows. The optimal squared-loss prediction is
        # analytically known; this checks weighting is applied to log targets.
        X = np.zeros((40, 2))
        y = np.array([20.] * 20 + [200.] * 20)
        with threadpool_limits(limits=1):
            for family in ('hgb', 'lgb'):
                for multiplier in (1, 2, 3, 5):
                    weights = np.where(y > 100, multiplier, 1.)
                    expected = np.average(np.log10(1+y), weights=weights)
                    model = fit_tail(f'{family}_w{multiplier}', X, y)
                    np.testing.assert_allclose(model.predict(X[:2]), expected, atol=1e-6, rtol=0)


if __name__ == '__main__':
    unittest.main()

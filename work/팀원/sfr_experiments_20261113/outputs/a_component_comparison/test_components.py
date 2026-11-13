import unittest
import numpy as np
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from component_models import PhysicalSplineFeatures, ComponentLogMixture, mix


class ComponentTests(unittest.TestCase):
    def test_log_space_mixture_and_zero_floor(self):
        p = np.array([[1.,0.,0.],[0.,.5,.5],[0.,1.,0.]])
        result = mix(p, np.array([2.,np.log(4.),-1.]), np.array([3.,np.log(16.),3.]))
        np.testing.assert_allclose(result, [0.,7.,0.], atol=1e-14)

    def test_spline_heldout_values_do_not_refit_knots(self):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(60,13))
        # Concentrated/sentinel-like column: uniform knots remain valid.
        X[:50,5] = -6.
        f = PhysicalSplineFeatures().fit(X)
        knots = [b.t.copy() for b in f.spline_.bsplines_]
        extreme = X[:5].copy(); extreme[:,:7] += 100
        result = f.transform(extreme)
        self.assertEqual(result.shape, (5,84))
        self.assertTrue(np.isfinite(result).all())
        for old, b in zip(knots, f.spline_.bsplines_):
            np.testing.assert_array_equal(old,b.t)
        np.testing.assert_array_equal(result[:,42:48], extreme[:,7:])

    def test_reusable_estimator_clone_and_probability_mixture(self):
        rng = np.random.default_rng(20)
        X = rng.normal(size=(300,13))
        y = np.where(X[:,0] < -.4, 0., np.where(X[:,0] > .4, 20., 3.))
        original = ComponentLogMixture()
        m = clone(original)
        with threadpool_limits(limits=2):
            m.fit(X,y)
            pred = m.predict(X[:20])
            expected = mix(m.gate_.predict_proba(X[:20]), m.mid_.predict(X[:20]), m.tail_.predict(X[:20]))
        np.testing.assert_allclose(pred,expected)
        self.assertFalse(hasattr(original,'gate_'))
        self.assertTrue(np.isfinite(pred).all() and (pred >= 0).all())

if __name__ == '__main__':
    unittest.main()

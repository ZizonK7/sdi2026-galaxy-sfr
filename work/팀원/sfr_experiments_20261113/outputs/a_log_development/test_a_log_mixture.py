import unittest
import numpy as np
from sklearn.base import clone
from threadpoolctl import threadpool_limits
from a_log_mixture import LogTailMixtureSFR

class LogMixtureTests(unittest.TestCase):
    def test_zero_class_term_for_nonunit_offset(self):
        # Half zero, half y=3*offset: inverse expected log is exactly offset.
        for offset in [1., .001]:
            p=np.array([[.5,.5,0.]])
            z=np.array([np.log(4*offset)])
            got=LogTailMixtureSFR.combine(p,z,z,offset)
            np.testing.assert_allclose(got,[offset],rtol=1e-12)

    def test_hard_routing_outputs_exact_zero(self):
        p=np.array([[.8,.1,.1]])
        z=np.array([np.log(31.001)])
        self.assertEqual(LogTailMixtureSFR.combine(p,z,z,.001,'hard')[0],0.)

    def test_fit_clone_and_prediction_contract(self):
        rng=np.random.default_rng(42)
        X=rng.normal(size=(180,3));y=np.tile([0.,2.,100.],60)
        m=LogTailMixtureSFR(log_offset=.001,hgb_params={'max_iter':5,'min_samples_leaf':5})
        with threadpool_limits(limits=2):
            fitted=clone(m).fit(X,y)
            pred=fitted.predict(X[:11])
            self.assertEqual(pred.shape,(11,))
            self.assertTrue(np.isfinite(pred).all() and (pred>=0).all())
            with self.assertRaises(ValueError): fitted.predict(np.ones((3,4)))
            with self.assertRaises(ValueError): clone(m).fit(X,np.zeros(len(y)))
            with self.assertRaises(ValueError): clone(m).fit(X,-np.ones(len(y)))

if __name__=='__main__':unittest.main()

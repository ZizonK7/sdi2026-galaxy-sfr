import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from models import make_model,part,labels,mix

class RFTests(unittest.TestCase):
    def test_random_forest_bootstrap_missing_values_and_class_order(self):
        rng=np.random.default_rng(11);X=rng.normal(size=(150,34));X[:40,30]=np.nan
        y=np.tile([0.,3.,20.],50);cl=labels(y)
        with threadpool_limits(limits=2):
            model=make_model('gate',{'family':'rf','params':{'min_samples_leaf':3}})
            model.set_params(n_estimators=5);model.fit(X,cl);p=part(model,'gate',X)
        self.assertTrue(model.bootstrap);self.assertEqual(p.shape,(150,3))
        np.testing.assert_allclose(p.sum(axis=1),1.)
        self.assertTrue(np.isfinite(p).all())
    def test_regression_targets_and_log_mixture(self):
        rng=np.random.default_rng(12);X=rng.normal(size=(150,34));X[:40,30]=np.nan
        y=np.tile([0.,3.,20.],50);cl=labels(y);parts={}
        with threadpool_limits(limits=2):
            for slot,label in [('mid',1),('tail',2)]:
                m=make_model(slot,{'family':'rf','params':{'min_samples_leaf':3}})
                m.set_params(n_estimators=5);m.fit(X[cl==label],np.log1p(y[cl==label]))
                parts[slot]=part(m,slot,X[:2])
        parts['gate']=np.array([[1.,0.,0.],[0.,.5,.5]])
        np.testing.assert_allclose(mix(parts),[0.,np.sqrt(4*21)-1],atol=1e-12)

if __name__=='__main__':unittest.main()

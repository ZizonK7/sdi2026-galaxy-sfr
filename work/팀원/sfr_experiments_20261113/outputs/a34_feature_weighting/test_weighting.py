import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from run_weighting import feature_weights,gate_spec,DENSITY,BH
from models import fit_part,part

class WeightTests(unittest.TestCase):
    def test_weights_follow_column_names_not_positions(self):
        names=['x',BH,'z',DENSITY]
        np.testing.assert_array_equal(feature_weights(names,2.,1.5),[1,1.5,1,2])
        with self.assertRaises(ValueError):feature_weights(names,-1,2)
    def test_gain_weights_are_consumed_by_lightgbm(self):
        rng=np.random.default_rng(7);X=rng.normal(size=(300,4));X[:30,2]=np.nan
        y=np.where(X[:,0]<-.4,0.,np.where(X[:,0]>.4,20.,3.))
        spec=gate_spec([DENSITY,'x',BH,'z'],(2.,1.5));spec['params'].update(n_estimators=5,num_leaves=7)
        with threadpool_limits(limits=2):
            m=fit_part('gate',spec,X,y);p=part(m,'gate',X[:20])
        np.testing.assert_array_equal(m.booster_.params['feature_contri'],[2.,1.,1.5,1.])
        np.testing.assert_allclose(p.sum(axis=1),1.)
        self.assertTrue(np.isfinite(p).all())

if __name__=='__main__':unittest.main()

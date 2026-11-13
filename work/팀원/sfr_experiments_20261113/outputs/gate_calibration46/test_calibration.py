import unittest
import numpy as np
from calibration_model import ProbabilityMap,fit_maps,apply_maps
from hierarchical_model import hierarchical_probabilities
from three_regime import combine

class ConstantMap:
    def __init__(self,value):self.value=value
    def predict(self,p):return np.full_like(p,self.value,dtype=float)

class CalibrationTests(unittest.TestCase):
    def test_finite_monotonic_maps_including_endpoints(self):
        probabilities=np.repeat([.1,.3,.5,.7,.9],10)
        labels=np.concatenate([np.r_[np.ones(n),np.zeros(10-n)] for n in (2,4,5,6,8)])
        grid=np.linspace(0,1,101)
        for method in ('sigmoid','isotonic'):
            pred=ProbabilityMap(method).fit(probabilities,labels).predict(grid)
            self.assertTrue(np.isfinite(pred).all())
            self.assertTrue(((pred>=0)&(pred<=1)).all())
            self.assertTrue((np.diff(pred)>=-1e-12).all())
        np.testing.assert_allclose(ProbabilityMap('isotonic').fit(probabilities,labels).predict([.1,.3,.5,.7,.9]),[.2,.4,.5,.6,.8])

    def test_conditional_map_uses_only_true_positive_labels(self):
        y=np.array([0.,0.,1.,2.,20.,30.])
        raw=np.column_stack([[.1,.2,.6,.8,.7,.9],[1.,1.,.1,.2,.8,.9]])
        maps=fit_maps(raw,y)
        expected=ProbabilityMap('isotonic').fit(raw[y>0,1],(y[y>0]>10).astype(int))
        grid=np.linspace(0,1,21)
        np.testing.assert_array_equal(maps['isotonic']['tail'].predict(grid),expected.predict(grid))

    def test_scope_and_zero_rule(self):
        raw=np.array([[.8,.3],[.99,.9]])
        maps={'sigmoid':{'positive':ConstantMap(.05),'tail':ConstantMap(.8)}}
        p_only=apply_maps(raw,maps,'sigmoid_positive')
        q_only=apply_maps(raw,maps,'sigmoid_tail')
        np.testing.assert_array_equal(p_only[:,1],raw[:,1])
        np.testing.assert_array_equal(q_only[:,0],raw[:,0])
        np.testing.assert_array_equal(apply_maps(raw,maps,'baseline'),raw)
        both=apply_maps(raw,maps,'sigmoid_both')
        p=hierarchical_probabilities(both[:,0],both[:,1])
        np.testing.assert_allclose(p.sum(axis=1),1.)
        np.testing.assert_array_equal(combine(p,[1.,1.],[3.,3.]),0.)

    def test_invalid_probabilities_and_single_class_rejected(self):
        for probabilities,labels in [([-.1,.5],[0,1]),([0.,np.nan],[0,1]),([.1,.2],[0,0])]:
            with self.assertRaises(ValueError):ProbabilityMap('sigmoid').fit(probabilities,labels)

if __name__=='__main__':unittest.main()

import unittest
import numpy as np
from blend_model import TailBlend, ProtectedTailBlend, blend_sfr_predictions, blend_log_predictions
from three_regime import combine

class FixedRegressor:
    def __init__(self, value): self.value=np.asarray(value)
    def predict(self, X): return self.value[:len(X)]

class BlendTests(unittest.TestCase):
    def test_log_space_and_endpoints(self):
        a=np.array([0.,3.,99.]);b=np.array([0.,15.,24.])
        np.testing.assert_allclose(blend_sfr_predictions(a,b,.5),np.sqrt((1+a)*(1+b))-1.)
        np.testing.assert_allclose(blend_sfr_predictions(a,b,0.),a)
        np.testing.assert_allclose(blend_sfr_predictions(a,b,1.),b)

    def test_shared_gate_tail_blend_equivalence_and_zero_rule(self):
        p=np.array([[.95,.04,.01],[.2,.5,.3],[.1,.1,.8]])
        mid=np.array([.5,1.,1.5]);a=np.array([2.,2.,2.]);b=np.array([2.5,3.,4.])
        for alpha in (.25,.5,.75):
            tail=TailBlend(FixedRegressor(a),FixedRegressor(b),alpha).predict(mid)
            pred=10.**combine(p,mid,tail)-1.
            expected=blend_sfr_predictions(10.**combine(p,mid,a)-1.,10.**combine(p,mid,b)-1.,alpha)
            np.testing.assert_allclose(pred,expected,rtol=1e-12,atol=1e-12)
            self.assertEqual(pred[0],0.)

    def test_reject_invalid_inputs(self):
        for alpha in (-.1,1.1,np.nan):
            with self.assertRaises(ValueError):blend_sfr_predictions([0],[1],alpha)
            with self.assertRaises(ValueError):TailBlend(None,None,alpha)
        with self.assertRaises(ValueError):blend_sfr_predictions([-1],[1],.5)

    def test_zero_floor_applied_after_log_mixture(self):
        a=np.array([-.1,.2,0.]);b=np.array([.1,-.2,0.])
        pred=blend_log_predictions(a,b,.25)
        np.testing.assert_allclose(pred,[0.,10.**.1-1.,0.],rtol=1e-12,atol=1e-12)
        # Clipping each expert's final output before mixing is a different rule.
        wrong=blend_sfr_predictions(np.maximum(10.**a-1,0),np.maximum(10.**b-1,0),.25)
        self.assertGreater(wrong[0],pred[0])

    def test_protection_routes_using_prediction_and_strict_threshold(self):
        base=FixedRegressor([0.,9.,10.,11.,100.])
        mixed=FixedRegressor([4.,9.5,20.,12.,90.])
        model=ProtectedTailBlend(base,mixed,10.)
        np.testing.assert_array_equal(model.predict(np.arange(5)),[0.,9.,10.,12.,90.])

if __name__=='__main__':unittest.main()

import unittest
import numpy as np
from zero_gate import apply_zero_threshold

class ThresholdTests(unittest.TestCase):
    def test_inclusive_boundary_and_unchanged_other_predictions(self):
        soft=np.array([.3,.4,8.,2.]);p=np.array([.899999,.9,.99,.01])
        result=apply_zero_threshold(soft,p)
        np.testing.assert_array_equal(result,[.3,0.,0.,2.])
        np.testing.assert_array_equal(soft,[.3,.4,8.,2.])
    def test_invalid_threshold_and_shape_rejected(self):
        with self.assertRaises(ValueError):apply_zero_threshold([1.],[.9],1.1)
        with self.assertRaises(ValueError):apply_zero_threshold([1.,2.],[.9])

if __name__=='__main__':unittest.main()

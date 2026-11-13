import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from hierarchical_model import hierarchical_probabilities, fit_binary_gates
from three_regime import combine


class Tests(unittest.TestCase):
    def test_joint_probabilities_and_zero_boundary(self):
        p=hierarchical_probabilities([.8,.1,0.,1.],[.3,.5,1.,0.])
        np.testing.assert_allclose(p,[[.2,.56,.24],[.9,.05,.05],[1.,0.,0.],[0.,1.,0.]])
        np.testing.assert_allclose(p.sum(axis=1),1.)
        np.testing.assert_allclose(combine(p,np.ones(4),np.full(4,2.)),[1.04,0.,0.,1.])

    def test_conditional_gate_uses_true_positive_subset(self):
        X=pd.DataFrame({'x':[1.,2.,3.,4.]},index=[8,3,7,9])
        with patch('hierarchical_model.HistGradientBoostingClassifier') as factory:
            fit_binary_gates(X,np.array([0.,10.,11.,2.]))
            calls=factory.return_value.fit.call_args_list
            self.assertEqual(calls[0].args[0].index.tolist(),[8,3,7,9])
            np.testing.assert_array_equal(calls[0].args[1],[0,1,1,1])
            self.assertEqual(calls[1].args[0].index.tolist(),[3,7,9])
            np.testing.assert_array_equal(calls[1].args[1],[0,1,0])
            factory.return_value.fit.return_value.predict_proba.assert_not_called()


if __name__=='__main__':
    unittest.main()

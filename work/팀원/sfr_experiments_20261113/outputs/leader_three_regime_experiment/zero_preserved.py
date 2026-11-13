"""Preserve the original binary zero probability while splitting positive mass."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from leader_model import HGB_FIXED400
from three_regime import combine


def redistribute(p_three, p_positive):
    positive_mass = p_three[:, 1] + p_three[:, 2]
    tail_fraction = np.divide(p_three[:, 2], positive_mass,
                              out=np.full(len(positive_mass), .5), where=positive_mass > 0)
    return np.column_stack([1 - p_positive, p_positive * (1 - tail_fraction), p_positive * tail_fraction])


class ZeroPreservedThreeRegime:
    """Wrap a fitted three-regime model with a separately fitted binary gate."""
    def __init__(self, three_regime, binary_gate):
        self.three_regime = three_regime
        self.binary_gate = binary_gate
        self.zero_threshold = three_regime.zero_threshold

    def predict_components(self, X):
        p, mid, tail = self.three_regime.predict_components(X)
        p_positive = self.binary_gate.predict_proba(X[self.three_regime.feature_names_])[:, 1]
        return redistribute(p, p_positive), mid, tail

    def predict_log(self, X):
        return combine(*self.predict_components(X), self.zero_threshold)

    def predict(self, X):
        return np.maximum(10.**self.predict_log(X) - 1., 0.)

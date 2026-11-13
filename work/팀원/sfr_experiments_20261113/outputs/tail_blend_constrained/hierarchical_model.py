"""Independent zero/positive gate followed by conditional mid/tail classification."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from leader_model import HGB_FIXED400
from feature_pipeline import UNIVERSE
from three_regime import combine


def hierarchical_probabilities(p_positive, p_tail_given_positive):
    p = np.asarray(p_positive, dtype=float)
    q = np.asarray(p_tail_given_positive, dtype=float)
    if p.ndim != 1 or p.shape != q.shape or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Expected matched finite one-dimensional probability arrays')
    if (p < 0).any() or (p > 1).any() or (q < 0).any() or (q > 1).any():
        raise ValueError('Probabilities must be between zero and one')
    return np.column_stack([1-p, p*(1-q), p*q])


def fit_binary_gates(X46, y):
    y = np.asarray(y, dtype=float)
    positive = y > 0
    if not np.isfinite(y).all() or (y < 0).any() or not (y == 0).any() or not (y > 10).any() or not ((y > 0) & (y <= 10)).any():
        raise ValueError('Need all three nonnegative target regimes')
    # Both classifiers use true training labels, never predicted routing.
    positive_gate = HistGradientBoostingClassifier(**HGB_FIXED400).fit(X46, positive.astype(int))
    conditional_tail_gate = HistGradientBoostingClassifier(**HGB_FIXED400).fit(X46.loc[positive], (y[positive] > 10).astype(int))
    return positive_gate, conditional_tail_gate


def positive_probability(model, X46):
    return model.predict_proba(X46)[:, list(model.classes_).index(1)]


class HierarchicalSFR:
    """Two binary classifiers and two unchanged, true-regime-trained regressors."""
    def __init__(self, engineer, positive_gate, conditional_tail_gate, mid_regressor, tail_regressor):
        self.engineer_ = engineer
        self.positive_gate_ = positive_gate
        self.conditional_tail_gate_ = conditional_tail_gate
        self.mid_regressor_ = mid_regressor
        self.tail_regressor_ = tail_regressor
        self.zero_threshold = .9
        self.tail_threshold = 10.

    def predict_components(self, X):
        Z = self.engineer_.transform(X)[UNIVERSE]
        p = positive_probability(self.positive_gate_, Z)
        q = positive_probability(self.conditional_tail_gate_, Z)
        return hierarchical_probabilities(p, q), self.mid_regressor_.predict(Z), self.tail_regressor_.predict(Z)

    def predict_log(self, X):
        return combine(*self.predict_components(X), self.zero_threshold)

    def predict(self, X):
        return np.maximum(10.**self.predict_log(X)-1., 0.)

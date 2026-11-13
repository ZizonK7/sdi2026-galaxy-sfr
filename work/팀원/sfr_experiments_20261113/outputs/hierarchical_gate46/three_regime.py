"""Three-class probability gate with regressors trained on true SFR regimes."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from leader_model import HGB_FIXED400


def combine(probabilities, mid_log, tail_log, zero_threshold=.9):
    p = np.asarray(probabilities)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError('Expected probabilities in class order zero, mid, tail')
    if not np.allclose(p.sum(axis=1), 1.) or (p < 0).any():
        raise ValueError('Invalid class probabilities')
    z = p[:, 1] * np.asarray(mid_log) + p[:, 2] * np.asarray(tail_log)
    return np.where(p[:, 0] >= zero_threshold, 0., z)


class ThreeRegimeLogSFR:
    def __init__(self, tail_threshold=30., zero_threshold=.9, params=None):
        self.tail_threshold = tail_threshold
        self.zero_threshold = zero_threshold
        self.params = HGB_FIXED400.copy() if params is None else params.copy()

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if not np.isfinite(y).all() or (y < 0).any() or self.tail_threshold <= 0:
            raise ValueError('Need nonnegative finite targets and a positive tail threshold')
        if not 0 <= self.zero_threshold <= 1:
            raise ValueError('Invalid zero threshold')
        self.feature_names_ = list(X.columns)
        labels = np.where(y == 0, 0, np.where(y <= self.tail_threshold, 1, 2))
        self.training_counts_ = np.bincount(labels, minlength=3).tolist()
        if min(self.training_counts_) == 0:
            raise ValueError('All three true target regimes must be present')
        self.clf_ = HistGradientBoostingClassifier(**self.params).fit(X, labels)
        # Each expert uses TRUE training labels, never the classifier's fitted predictions.
        self.mid_ = HistGradientBoostingRegressor(**self.params).fit(X.loc[labels == 1], np.log10(1 + y[labels == 1]))
        self.tail_ = HistGradientBoostingRegressor(**self.params).fit(X.loc[labels == 2], np.log10(1 + y[labels == 2]))
        return self

    def predict_components(self, X):
        X = X[self.feature_names_]
        order = [list(self.clf_.classes_).index(i) for i in (0, 1, 2)]
        return self.clf_.predict_proba(X)[:, order], self.mid_.predict(X), self.tail_.predict(X)

    def predict_log(self, X):
        return combine(*self.predict_components(X), self.zero_threshold)

    def predict(self, X):
        return np.maximum(10. ** self.predict_log(X) - 1., 0.)

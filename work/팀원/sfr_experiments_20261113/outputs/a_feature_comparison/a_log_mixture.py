"""A's zero / ordinary / tail mixture, with experts trained in log-SFR space.

log_offset belongs to the evaluation definition; do not tune it to improve a score.
For g(y)=ln(y+offset), soft prediction estimates E[g(Y)|X] and then inverts g.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted


class LogTailMixtureSFR(RegressorMixin, BaseEstimator):
    def __init__(self, tail_threshold=30.0, log_offset=1.0,
                 hgb_params=None, routing='soft', mix_space='log',
                 clip_to_regime=False):
        self.tail_threshold = tail_threshold
        self.log_offset = log_offset
        self.hgb_params = hgb_params
        self.routing = routing
        self.mix_space = mix_space
        self.clip_to_regime = clip_to_regime

    def fit(self, X, y):
        # Singleton FoF groups have undefined relative group velocity (NaN).
        # Preserve missingness for HGB's native missing-value handling; reject inf.
        X, y = check_X_y(X, y, dtype=float, y_numeric=True, ensure_all_finite='allow-nan')
        if self.tail_threshold <= 0 or self.log_offset <= 0:
            raise ValueError('tail_threshold and log_offset must be positive')
        if np.any(y < 0):
            raise ValueError('SFR targets must be nonnegative')
        if self.routing not in ('soft', 'hard') or self.mix_space not in ('log', 'linear'):
            raise ValueError('Unsupported routing or mix_space')
        classes = np.where(y == 0, 0, np.where(y <= self.tail_threshold, 1, 2))
        if not np.array_equal(np.unique(classes), np.arange(3)):
            raise ValueError('Training data must contain zero, ordinary, and tail examples')
        self.n_features_in_ = X.shape[1]
        self.params_ = dict(learning_rate=.08, max_iter=150,
                            max_leaf_nodes=31, random_state=42)
        if self.hgb_params is not None:
            self.params_.update(self.hgb_params)
        self.classifier_ = HistGradientBoostingClassifier(**self.params_).fit(X, classes)
        z = np.log(y + self.log_offset)
        self.mid_regressor_ = HistGradientBoostingRegressor(**self.params_).fit(X[classes == 1], z[classes == 1])
        self.tail_regressor_ = HistGradientBoostingRegressor(**self.params_).fit(X[classes == 2], z[classes == 2])
        self.class_counts_ = np.bincount(classes, minlength=3)
        return self

    def parts(self, X):
        check_is_fitted(self, ['classifier_', 'mid_regressor_', 'tail_regressor_'])
        X = check_array(X, dtype=float, ensure_all_finite='allow-nan')
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Input feature count differs from training')
        prob = self.classifier_.predict_proba(X)
        z0 = np.log(self.log_offset)
        zm = np.maximum(self.mid_regressor_.predict(X), z0)
        zt = np.maximum(self.tail_regressor_.predict(X), z0)
        if self.clip_to_regime:
            boundary = np.log(self.tail_threshold + self.log_offset)
            zm = np.minimum(zm, boundary)
            zt = np.maximum(zt, boundary)
        return prob, zm, zt

    def predict(self, X):
        prob, zm, zt = self.parts(X)
        return self.combine(prob, zm, zt, self.log_offset,
                            self.routing, self.mix_space)

    @staticmethod
    def combine(prob, zm, zt, log_offset, routing='soft', mix_space='log'):
        z0 = np.log(log_offset)
        if routing == 'hard':
            labels = prob.argmax(axis=1)
            z = np.select([labels == 1, labels == 2], [zm, zt], z0)
        elif mix_space == 'log':
            # The zero class contributes log(offset), which is nonzero unless offset=1.
            z = prob[:, 0]*z0 + prob[:, 1]*zm + prob[:, 2]*zt
        else:
            # Ablation: log-trained experts, but mixing after inverse transform.
            return np.maximum(prob[:, 1]*(np.exp(zm)-log_offset)
                              + prob[:, 2]*(np.exp(zt)-log_offset), 0)
        prediction = np.maximum(np.exp(z)-log_offset, 0)
        # Avoid a tiny positive rounding residue for the hard-routed zero class.
        if routing == 'hard':
            prediction = np.where(labels == 0, 0.0, prediction)
        return prediction

"""Fixed candidates for replacing individual components of A-log (T=10).

Raw feature order is recorded in metadata.json. Spline knots and scaling are
learned on the component's training subset only. No validation-set fitting.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, ExtraTreesRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

HGB_PARAMS = dict(learning_rate=.08, max_iter=150, max_leaf_nodes=31, random_state=42)
CAT_PARAMS = dict(iterations=300, depth=6, learning_rate=.05, l2_leaf_reg=5,
                  random_seed=42, thread_count=4, allow_writing_files=False, verbose=False)


class PhysicalSplineFeatures(TransformerMixin, BaseEstimator):
    """Cubic splines for seven physical features; six linear position/velocity features.

    Uniform knots avoid duplicate quantile knots in columns with sentinel/seed masses.
    Constant extrapolation avoids unstable cubic extrapolation outside train support.
    Interactions are tensor products of the specified pairs of spline bases.
    """
    def __init__(self, interactions=((6, 1),), n_knots=5):
        self.interactions = interactions
        self.n_knots = n_knots

    def fit(self, X, y=None):
        X = check_array(X)
        if X.shape[1] != 13:
            raise ValueError('Expected the original 13 features in recorded order')
        self.n_features_in_ = X.shape[1]
        self.spline_ = SplineTransformer(n_knots=self.n_knots, degree=3,
                                         knots='uniform', include_bias=False,
                                         extrapolation='constant').fit(X[:, :7])
        self.basis_width_ = self.spline_.n_features_out_ // 7
        return self

    def transform(self, X):
        check_is_fitted(self, 'spline_')
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Unexpected feature count')
        s = self.spline_.transform(X[:, :7])
        w = self.basis_width_
        parts = [s, X[:, 7:]]
        for a, b in self.interactions:
            ab = s[:, a*w:(a+1)*w, None] * s[:, None, b*w:(b+1)*w]
            parts.append(ab.reshape(len(X), -1))
        return np.concatenate(parts, axis=1)


def factories():
    return {
        'hgb_gate': lambda: HistGradientBoostingClassifier(**HGB_PARAMS),
        'hgb_mid': lambda: HistGradientBoostingRegressor(**HGB_PARAMS),
        'hgb_tail': lambda: HistGradientBoostingRegressor(**HGB_PARAMS),
        'cat_gate': lambda: CatBoostClassifier(loss_function='MultiClass', **CAT_PARAMS),
        'spline_gate': lambda: make_pipeline(PhysicalSplineFeatures(interactions=((6, 1),)),
            StandardScaler(), LogisticRegression(C=1., solver='lbfgs', max_iter=1500, tol=1e-5)),
        'cat_mid': lambda: CatBoostRegressor(loss_function='RMSE', **CAT_PARAMS),
        'spline_mid': lambda: make_pipeline(PhysicalSplineFeatures(interactions=((1, 2), (1, 4))),
            StandardScaler(), Ridge(alpha=10.)),
        'cat_tail': lambda: CatBoostRegressor(loss_function='RMSE', **CAT_PARAMS),
        'extra_tail': lambda: ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5,
            max_features=1., criterion='squared_error', n_jobs=4, random_state=42),
    }


def mix(prob, mid, tail):
    """ln(1+SFR) mixture; zero expert is exactly zero in transformed space."""
    return np.expm1(prob[:, 1]*np.maximum(mid, 0) + prob[:, 2]*np.maximum(tail, 0))


class ComponentLogMixture(RegressorMixin, BaseEstimator):
    """Reusable estimator. Fit experts on true classes, mix probabilities at inference."""
    def __init__(self, gate='hgb_gate', mid='hgb_mid', tail='hgb_tail', tail_threshold=10.):
        self.gate = gate
        self.mid = mid
        self.tail = tail
        self.tail_threshold = tail_threshold

    def fit(self, X, y):
        X, y = check_X_y(X, y, y_numeric=True)
        if self.tail_threshold <= 0 or np.any(y < 0):
            raise ValueError('Positive threshold and nonnegative targets required')
        labels = np.where(y == 0, 0, np.where(y <= self.tail_threshold, 1, 2))
        if not np.array_equal(np.unique(labels), [0, 1, 2]):
            raise ValueError('All three true classes must be present')
        fs = factories()
        self.n_features_in_ = X.shape[1]
        self.gate_ = fs[self.gate]().fit(X, labels)
        self.mid_ = fs[self.mid]().fit(X[labels == 1], np.log1p(y[labels == 1]))
        self.tail_ = fs[self.tail]().fit(X[labels == 2], np.log1p(y[labels == 2]))
        return self

    def predict(self, X):
        check_is_fitted(self, ['gate_', 'mid_', 'tail_'])
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Unexpected feature count')
        order = [list(self.gate_.classes_).index(c) for c in (0, 1, 2)]
        return mix(self.gate_.predict_proba(X)[:, order], self.mid_.predict(X), self.tail_.predict(X))

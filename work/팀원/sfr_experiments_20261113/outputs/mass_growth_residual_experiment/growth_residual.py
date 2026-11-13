"""Input-only, fold-fitted growth residual relative to stellar mass peers."""
import numpy as np

FEATURE = 'stellar_growth_residual_at_mass'
MASS = 'log10_stellar_mass'
GROWTH = 'dlogMs_growth'


class MassGrowthResidual:
    def __init__(self, n_bins=20):
        self.n_bins = n_bins

    def fit(self, X):
        mass, growth = X[MASS].to_numpy(), X[GROWTH].to_numpy()
        if not np.isfinite(mass).all() or not np.isfinite(growth).all():
            raise ValueError('Mass and growth must be finite')
        self.edges_ = np.unique(np.quantile(mass, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
        groups = np.searchsorted(self.edges_, mass, side='right')
        rows = [(float(np.median(mass[groups == b])), float(growth[groups == b].mean()), int((groups == b).sum()))
                for b in np.unique(groups)]
        self.mass_nodes_ = np.array([r[0] for r in rows])
        self.growth_nodes_ = np.array([r[1] for r in rows])
        self.counts_ = np.array([r[2] for r in rows])
        if len(self.mass_nodes_) > 1 and not (np.diff(self.mass_nodes_) > 0).all():
            raise ValueError('Mass nodes must be strictly increasing')
        self.fit_rows_ = len(X)
        return self

    def transform(self, X):
        mass, growth = X[MASS].to_numpy(), X[GROWTH].to_numpy()
        if not np.isfinite(mass).all() or not np.isfinite(growth).all():
            raise ValueError('Mass and growth must be finite')
        result = X.copy()
        # Clamp beyond the fitted mass range rather than extrapolate a slope.
        expectation = np.interp(mass, self.mass_nodes_, self.growth_nodes_)
        result[FEATURE] = growth - expectation
        return result


class ResidualFeatureModel:
    """Persist the training-fitted feature map with its fitted regression model."""
    def __init__(self, transformer, model):
        self.transformer = transformer
        self.model = model

    def predict(self, X):
        return self.model.predict(self.transformer.transform(X))

    def predict_log(self, X):
        return self.model.predict_log(self.transformer.transform(X))

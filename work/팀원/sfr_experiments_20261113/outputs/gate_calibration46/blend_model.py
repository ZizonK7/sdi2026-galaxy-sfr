"""Blend log-SFR tail experts; all gates and the mid expert remain fixed."""
import numpy as np


class TailBlend:
    def __init__(self, baseline, candidate, alpha):
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('alpha must be in [0,1]')
        self.baseline = baseline
        self.candidate = candidate
        self.alpha = float(alpha)

    def predict(self, X):
        return ((1-self.alpha)*self.baseline.predict(X)
                + self.alpha*self.candidate.predict(X))


def blend_sfr_predictions(baseline, candidate, alpha):
    """Cached-OOF convenience; exact if both outputs are positive or both zero.

    When pre-clipping signs differ, use blend_log_predictions instead.
    """
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('alpha must be in [0,1]')
    a, b = np.asarray(baseline), np.asarray(candidate)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or (a<0).any() or (b<0).any():
        raise ValueError('Expected matched finite nonnegative SFR predictions')
    return np.maximum(10.**((1-alpha)*np.log10(1+a)+alpha*np.log10(1+b))-1.,0.)


def blend_log_predictions(baseline_log, candidate_log, alpha):
    """Mix the unclipped log predictions and apply the SFR>=0 floor once."""
    a,b=np.asarray(baseline_log),np.asarray(candidate_log)
    if not np.isfinite(alpha) or not 0<=alpha<=1:
        raise ValueError('alpha must be in [0,1]')
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected matched finite log predictions')
    return np.maximum(10.**((1-alpha)*a+alpha*b)-1.,0.)


class ProtectedTailBlend:
    """Use the frozen blend only when the baseline predicts SFR>10.

    Routing depends on a prediction available at inference, never on true SFR.
    """
    def __init__(self, baseline, blended, threshold=10.):
        if not np.isfinite(threshold) or threshold<0:
            raise ValueError('threshold must be finite and nonnegative')
        self.baseline=baseline
        self.blended=blended
        self.threshold=float(threshold)

    def predict(self,X):
        baseline=self.baseline.predict(X)
        blended=self.blended.predict(X)
        return np.where(baseline>self.threshold,blended,baseline)

    def predict_log(self,X):
        return np.log10(1+self.predict(X))

# 제출본 7절의 모델을 그대로 (설정 변경 없음)
HGB = dict(learning_rate=0.08, max_iter=150, max_leaf_nodes=31, random_state=RANDOM_SEED)

class TwoStageSFR:
    """분류기 P(y>0|x)와 양수 은하만으로 학습한 회귀기 E[y|y>0,x]를 곱하는 모델 (제출본 7절과 같음)."""
    def __init__(self, hgb_params, reg_loss="squared_error", clip_zero=True):
        self.hgb_params, self.reg_loss, self.clip_zero = hgb_params, reg_loss, clip_zero
    def fit(self, X, y):
        y = np.asarray(y, dtype=float); pos = y > 0
        self.clf_ = HistGradientBoostingClassifier(**self.hgb_params).fit(X, pos.astype(int))
        self.reg_ = HistGradientBoostingRegressor(loss=self.reg_loss, **self.hgb_params).fit(X[pos], y[pos])
        return self
    def predict_proba_positive(self, X): return self.clf_.predict_proba(X)[:, 1]
    def predict_positive_mean(self, X):
        e = self.reg_.predict(X); return np.clip(e, 0, None) if self.clip_zero else e
    def predict(self, X): return self.predict_proba_positive(X) * self.predict_positive_mean(X)


# 실험 B. 모델 정의
class NestedTailSFR:
    """P(>0) × [P(꼬리|>0)·E꼬리 + (1−P(꼬리|>0))·E보통]. 0 판단은 현재 2단계 모델과 같은 2부류 분류기."""
    def __init__(self, hgb_params, tail_threshold):
        self.hgb_params, self.T = hgb_params, tail_threshold
    def fit(self, X, y):
        y = np.asarray(y, dtype=float); pos = y > 0
        Xp, yp = X[pos], y[pos]; tail = yp > self.T
        self.clf_zero_ = HistGradientBoostingClassifier(**self.hgb_params).fit(X, pos.astype(int))
        self.clf_tail_ = HistGradientBoostingClassifier(**self.hgb_params).fit(Xp, tail.astype(int))
        self.reg_mid_ = HistGradientBoostingRegressor(**self.hgb_params).fit(Xp[~tail], yp[~tail])
        self.reg_tail_ = HistGradientBoostingRegressor(**self.hgb_params).fit(Xp[tail], yp[tail])
        return self
    def parts(self, X):
        p_pos = self.clf_zero_.predict_proba(X)[:, 1]
        p_tail = self.clf_tail_.predict_proba(X)[:, 1]
        e_mid = np.clip(self.reg_mid_.predict(X), 0, None)
        e_tail = np.clip(self.reg_tail_.predict(X), 0, None)
        return p_pos, p_tail, e_mid, e_tail
    def predict(self, X):
        p_pos, p_tail, e_mid, e_tail = self.parts(X)
        return p_pos * (p_tail * e_tail + (1 - p_tail) * e_mid)


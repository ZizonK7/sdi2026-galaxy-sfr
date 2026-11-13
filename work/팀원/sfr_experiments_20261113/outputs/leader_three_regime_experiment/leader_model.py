"""Model definitions extracted from team-lead commit 21d02b5.

Original functions/classes are preserved; predict_sfr requires explicit model/reference.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

FINAL_FEATURES = ['log10_halo_mass', 'log10_stellar_mass', 'log10_BH_mass',
 'log10_main_prog_mass', 'log10_main_prog_stellar_mass', 'log10_mass_ratio',
 'log10_overdensity_plus_1', 'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z']
ID_COLUMN = 'subhalo_id'
BOX_SIZE = 200.0
RHO_CRIT = 2.775e11
G_KMS = 4.30091e-9
LOG_EPS = 1.0
ZERO_THRESHOLD = 0.9
HGB_FIXED400 = dict(learning_rate=.08, max_iter=400, max_leaf_nodes=31,
                    random_state=42, early_stopping=False)


def build_environment_features(frame, box=BOX_SIZE, fof_b=0.2, host_thresholds=(13.5, 14.0)):
    """pos_*, vel_*, log10_halo_mass, log10_stellar_mass만 사용한다. frame과 같은 index의 DataFrame을 돌려준다."""
    pos = frame[["pos_x", "pos_y", "pos_z"]].to_numpy() % box          # 주기 경계 상자 [0, box)
    vel = frame[["vel_x", "vel_y", "vel_z"]].to_numpy()
    mh = frame["log10_halo_mass"].to_numpy()
    n = len(frame)
    tree = cKDTree(pos, boxsize=box)
    out = {}
    # (a) 이웃 밀도: k번째 이웃 거리 밀도(헤일로 안쪽 척도), 고정 반경 이웃 수(더 큰 척도)
    d, _ = tree.query(pos, k=11)
    out["log_rho_5nn"] = np.log10(5 / (4 / 3 * np.pi * d[:, 5] ** 3))
    out["log_rho_10nn"] = np.log10(10 / (4 / 3 * np.pi * d[:, 10] ** 3))
    for r in (1.0, 2.0, 5.0):
        out[f"log1p_N_r{int(r)}"] = np.log1p(tree.query_ball_point(pos, r, return_length=True) - 1)
    # (b) 가장 가까운 거대 헤일로: 거리, r/R200, |Δv|/V200, 호스트 질량
    r200 = lambda logm: (3 * 10 ** logm / (4 * np.pi * 200 * RHO_CRIT)) ** (1 / 3)   # h^-1 Mpc
    v200 = lambda logm: np.sqrt(G_KMS * 10 ** logm / r200(logm))                      # km/s
    for thr in host_thresholds:
        sel = np.where(mh >= thr)[0]
        dist, j = cKDTree(pos[sel], boxsize=box).query(pos, k=1)
        hj = sel[j]
        dv = np.sqrt(((vel - vel[hj]) ** 2).sum(axis=1))
        tag = f"{thr:g}"
        out[f"dist_host{tag}"] = dist
        out[f"r_over_R200_{tag}"] = dist / r200(mh[hj])
        out[f"v_over_V200_{tag}"] = dv / v200(mh[hj])
        if thr == host_thresholds[0]:
            out[f"host_logm_{tag}"] = mh[hj]
    # (c) friends-of-friends 그룹 (Davis et al. 1985): 연결 길이 = b × 평균 간격
    link = fof_b * box / n ** (1 / 3)
    pairs = tree.query_pairs(link, output_type="ndarray")
    _, lab = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)), directed=False)
    rich = np.bincount(lab)[lab]
    gmax = pd.Series(mh).groupby(lab).transform("max").to_numpy()
    vmean = pd.DataFrame(vel).groupby(lab).transform("mean").to_numpy()
    dv_grp = np.sqrt(((vel - vmean) ** 2).sum(axis=1))
    dv_grp[rich == 1] = np.nan                                         # 고립 은하는 정의 불가 (HGB는 NaN 처리 가능)
    out["fof_log_richness"] = np.log(rich)
    out["fof_is_central"] = (mh >= gmax - 1e-9).astype(float)          # 그룹 질량 최대를 대리 중심으로 정의; 동률 여러 개 가능
    out["fof_group_max_logm"] = gmax                                   # 호스트 헤일로 질량의 대리
    out["fof_dv_group"] = dv_grp
    # (d) 조석 박탈의 대리
    out["halo_minus_stellar"] = mh - frame["log10_stellar_mass"].to_numpy()
    return pd.DataFrame(out, index=frame.index)

def build_mass_history_features(frame):
    """주어진 질량 열의 차이 4개 (트리는 한 번에 열 하나로만 나눠 차이를 스스로 만들기 어렵다). 타깃 미사용.
    센티널 후보(6.0 등; 공식 의미 미확정)은 그대로 둔다 — 원래 열도 입력에 남아 트리가 따로 가를 수 있다."""
    return pd.DataFrame({
        "dlogMs_growth": frame["log10_stellar_mass"] - frame["log10_main_prog_stellar_mass"],    # 무차원 누적 항성 질량비; 센티널 후보 별도 진단
        "dlogMh_growth": frame["log10_halo_mass"] - frame["log10_main_prog_mass"],               # 무차원 누적 헤일로 질량비
        "bh_to_star": frame["log10_BH_mass"] - frame["log10_stellar_mass"],                      # 블랙홀 상대 질량; 피드백 인과를 직접 측정하지 않음
        "prog_star_to_halo": frame["log10_main_prog_stellar_mass"] - frame["log10_main_prog_mass"],  # 전구체의 항성/헤일로 질량비
    }, index=frame.index)

def to_log(sfr, eps=LOG_EPS):
    return np.log10(np.asarray(sfr, dtype=float) + eps)

def from_log(z, eps=LOG_EPS):
    return np.clip(10 ** np.asarray(z, dtype=float) - eps, 0, None)

def log_rmse(y_true, pred, eps):
    """선형 SFR 예측을 0 이상 절단 후 같은 offset으로 평가한다. eps는 반드시 명시한다."""
    import math
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    truth, prediction = list(y_true), list(pred)
    if not truth or len(truth) != len(prediction):
        raise ValueError("nonempty matched arrays required")
    if any(not math.isfinite(float(v)) or v < 0 for v in truth):
        raise ValueError("true SFR must be finite and nonnegative")
    if any(not math.isfinite(float(v)) for v in prediction):
        raise ValueError("predictions must be finite")
    return math.sqrt(math.fsum((math.log10(float(t) + eps) - math.log10(max(float(p), 0.0) + eps)) ** 2
                              for t, p in zip(truth, prediction)) / len(truth))

class TwoStageLogSFR:
    """ẑ(x) = P(SFR>0|x)·E[z|SFR>0,x] + (1 − P)·z(0). 두 단계가 서로 다른 피처 열을 쓸 수 있다. predict는 선형 SFR.
    zero_threshold가 있으면 P(SFR=0|x) ≥ zero_threshold인 은하는 z(0), 즉 SFR = 0을 그대로 낸다."""

    def __init__(self, hgb_params, clf_cols, reg_cols, eps=LOG_EPS, clf_params=None, zero_threshold=None):
        self.hgb_params, self.clf_cols, self.reg_cols, self.eps = hgb_params, list(clf_cols), list(reg_cols), eps
        self.zero_threshold = zero_threshold
        self.clf_params = hgb_params if clf_params is None else clf_params

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        pos = y > 0
        self.clf_ = HistGradientBoostingClassifier(**self.clf_params).fit(X[self.clf_cols], pos.astype(int))
        self.reg_ = HistGradientBoostingRegressor(**self.hgb_params).fit(X.loc[pos, self.reg_cols], to_log(y[pos], self.eps))
        return self

    def predict_proba_positive(self, X):
        return self.clf_.predict_proba(X[self.clf_cols])[:, 1]

    def predict_log_positive(self, X):
        return self.reg_.predict(X[self.reg_cols])

    def predict_log(self, X):
        p = self.predict_proba_positive(X)
        z0 = float(to_log(0.0, self.eps))
        z = p * self.predict_log_positive(X) + (1 - p) * z0
        return z if self.zero_threshold is None else np.where(1 - p >= self.zero_threshold, z0, z)

    def predict(self, X):
        return from_log(self.predict_log(X), self.eps)

def prepare_prediction_union(new_table, reference, *, assume_disjoint=False):
    """겹칠 수 있는 표에는 고유 ID 필수. 같은 ID의 입력이 다르면 조용히 대체하지 않고 거부한다.
    assume_disjoint=True는 ID 없는 표가 완전히 새 은하라는 호출자의 명시적 계약이다.
    배치를 나눌 때도 reference에 전체 질의 은하를 포함해 동일한 합집합을 유지해야 한다."""
    for label, table in (("new_table", new_table), ("reference", reference)):
        missing = set(FINAL_FEATURES) - set(table.columns)
        if missing:
            raise ValueError(f"{label}: missing features {sorted(missing)}")
        if ID_COLUMN in table.columns:
            if table[ID_COLUMN].isna().any() or table[ID_COLUMN].duplicated().any():
                raise ValueError(f"{label}: null/duplicate ID")
        elif len(table) and not assume_disjoint:
            raise ValueError(f"{label}: ID required; disjoint ID-less tables need explicit assume_disjoint=True")
    new = new_table[FINAL_FEATURES].reset_index(drop=True)
    ref = reference[FINAL_FEATURES].copy()
    if ID_COLUMN in new_table.columns and ID_COLUMN in reference.columns:
        ni = new_table.set_index(ID_COLUMN)[FINAL_FEATURES]
        ri = reference.set_index(ID_COLUMN)[FINAL_FEATURES]
        common = ni.index.intersection(ri.index)
        left, right = ni.loc[common], ri.loc[common]
        same = left.eq(right) | (left.isna() & right.isna())
        if not same.to_numpy().all():
            raise ValueError("same ID has different feature values")
        ref = ref.loc[~reference[ID_COLUMN].isin(new_table[ID_COLUMN]).to_numpy()]
    # ID 확인 후 피처의 사전순으로 정렬해 동일 합집합의 행 순서를 고정한다.
    # 최근접 거리가 동률인 경우에도 같은 기준 순서를 사용한다.
    both = pd.concat([ref, new], ignore_index=True)
    order = both.sort_values(FINAL_FEATURES, kind="mergesort", na_position="last").index.to_numpy()
    locations = np.empty(len(both), dtype=int)
    locations[order] = np.arange(len(both))
    return both.iloc[order].reset_index(drop=True), locations[len(ref):]

def predict_sfr(new_table, model, reference, return_log=False, *, assume_disjoint=False):
    """타깃 없는 입력에서 new_table 원래 순서로 반환. ID는 식별·충돌 확인에만 사용한다.
    참고 표는 명시적으로 전달한다. 평가표 전체를 한 번에 전달하거나 동일 합집합을 reference로 고정한다.
    ID 없는 기존 행을 disjoint로 선언하면 중복을 판별할 수 없으므로 호출자 책임이다."""
    if reference is None:
        raise ValueError("An explicit fixed reference catalogue is required")
    both, locations = prepare_prediction_union(new_table, reference, assume_disjoint=assume_disjoint)
    env = build_environment_features(both)
    mass = build_mass_history_features(both)
    X_new = both.join(env).join(mass).iloc[locations]
    return model.predict_log(X_new) if return_log else model.predict(X_new)

def build_features(catalogue):
    """Match the notebook's lexicographic ordering before environment calculation."""
    raw = catalogue[FINAL_FEATURES].copy()
    if not raw.index.is_unique:
        raise ValueError('Catalogue index must be unique')
    order = raw.sort_values(FINAL_FEATURES, kind='mergesort', na_position='last').index
    env = build_environment_features(raw.loc[order]).reindex(raw.index)
    return raw.join(env).join(build_mass_history_features(raw))


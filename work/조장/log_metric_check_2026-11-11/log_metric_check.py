"""2026-11-11 평가 기준이 log-SFR RMSE로 바뀌었을 때(조교 FAQ Q6) 지금 후보 모델이 어떻게 되는지 빠르게 본다.

- 공통 분할의 학습 데이터(166,749개) 안 5-fold CV(seed 42) 한 번. 검증 데이터는 쓰지 않는다.
- "log-SFR"에서 SFR = 0을 어떻게 다루는지 공지에 없어 세 가지 정의로 잰다.
    D1 = log10(1 + SFR), D2 = log10(SFR + 0.01), D3 = log10(SFR) (양수 은하만 평가)
- 모든 모델은 기준선과 같은 HGB 설정. 환경 피처 17개는 제출 노트북 9.2의 build_environment_features와 같다.
- 빠른 확인용이다. 반복 CV와 구간 추정은 하지 않았으므로 작은 차이는 해석하지 않는다.

실행: python log_metric_check.py   (약 2분)
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from sklearn.model_selection import KFold, train_test_split

OUT = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 42
HGB = dict(learning_rate=0.08, max_iter=150, max_leaf_nodes=31, random_state=RANDOM_SEED)
BOX_SIZE, RHO_CRIT, G_KMS = 200.0, 2.775e11, 4.30091e-9
EPS = 0.01          # D2에서 더하는 값 (임의로 고른 예시. 공식 정의가 아니다)
T_TAIL = 30.0       # 팀원 실험 B의 꼬리 경계


def build_environment_features(frame, box=BOX_SIZE, fof_b=0.2, host_thresholds=(13.5, 14.0)):
    """제출 노트북 9.2와 같은 함수 (pos_*, vel_*, log10_halo_mass, log10_stellar_mass만 사용, 타깃 미사용)."""
    pos = frame[["pos_x", "pos_y", "pos_z"]].to_numpy() % box
    vel = frame[["vel_x", "vel_y", "vel_z"]].to_numpy()
    mh = frame["log10_halo_mass"].to_numpy()
    n = len(frame)
    tree = cKDTree(pos, boxsize=box)
    out = {}
    d, _ = tree.query(pos, k=11)
    out["log_rho_5nn"] = np.log10(5 / (4 / 3 * np.pi * d[:, 5] ** 3))
    out["log_rho_10nn"] = np.log10(10 / (4 / 3 * np.pi * d[:, 10] ** 3))
    for r in (1.0, 2.0, 5.0):
        out[f"log1p_N_r{int(r)}"] = np.log1p(tree.query_ball_point(pos, r, return_length=True) - 1)
    r200 = lambda logm: (3 * 10 ** logm / (4 * np.pi * 200 * RHO_CRIT)) ** (1 / 3)
    v200 = lambda logm: np.sqrt(G_KMS * 10 ** logm / r200(logm))
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
    link = fof_b * box / n ** (1 / 3)
    pairs = tree.query_pairs(link, output_type="ndarray")
    _, lab = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)), directed=False)
    rich = np.bincount(lab)[lab]
    gmax = pd.Series(mh).groupby(lab).transform("max").to_numpy()
    vmean = pd.DataFrame(vel).groupby(lab).transform("mean").to_numpy()
    dv_grp = np.sqrt(((vel - vmean) ** 2).sum(axis=1))
    dv_grp[rich == 1] = np.nan
    out["fof_log_richness"] = np.log(rich)
    out["fof_is_central"] = (mh >= gmax - 1e-9).astype(float)
    out["fof_group_max_logm"] = gmax
    out["fof_dv_group"] = dv_grp
    out["halo_minus_stellar"] = mh - frame["log10_stellar_mass"].to_numpy()
    return pd.DataFrame(out, index=frame.index)


t_start = time.time()
df = pd.read_csv(ROOT / "Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv")
F13 = [c for c in df.columns if c not in ["subhalo_id", "sfr_linear"]]
ENV = build_environment_features(df)
F30 = F13 + list(ENV.columns)
X, y = df[F13], df["sfr_linear"]
Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=(y > 0).astype(int))
Xtr = Xtr.join(ENV).reset_index(drop=True)
yv = ytr.to_numpy(float)
n = len(yv)
YMIN = float(yv[yv > 0].min())
print(f"학습 {n:,}개 | SFR = 0 {np.mean(yv == 0):.2%} | 최소 양수 SFR {YMIN:.4f}", flush=True)

L_A, L_E, L_H = "선형 기준선 (13)", "Part II 초안: 환경 분류기 × 선형 회귀 13", "Part II 초안 + 팀원 꼬리 분리 T=30"
G13, G30 = "로그 타깃 직접 회귀 (13)", "로그 타깃 직접 회귀 (13+환경)"
H13, H30 = "환경 분류기 × 로그 회귀 13", "환경 분류기 × 로그 회귀 13+환경"
NAMES = [L_A, L_E, L_H, G13, G30, H13, H30]
Z = {"D1": lambda v: np.log10(1 + v), "D2": lambda v: np.log10(v + EPS), "D3": lambda v: np.log10(np.clip(v, YMIN, None))}
clip0 = lambda a: np.clip(a, 0, None)
oof = {d: {k: np.zeros(n) for k in NAMES} for d in Z}

for tr, te in KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED).split(Xtr):
    Xa, Xb, ya = Xtr.iloc[tr], Xtr.iloc[te], yv[tr]
    A13, B13, A30, B30 = Xa[F13], Xb[F13], Xa[F30], Xb[F30]
    pos = ya > 0
    Xp, yp = A13[pos], ya[pos]
    tl = yp > T_TAIL
    p30 = HGBC(**HGB).fit(A30, pos.astype(int)).predict_proba(B30)[:, 1]          # 1단계: 환경 분류기
    pt = HGBC(**HGB).fit(Xp, tl.astype(int)).predict_proba(B13)[:, 1]
    lin = {L_A: clip0(HGBR(**HGB).fit(A13, ya).predict(B13)),
           L_E: p30 * clip0(HGBR(**HGB).fit(Xp, yp).predict(B13)),
           L_H: p30 * (pt * clip0(HGBR(**HGB).fit(Xp[tl], yp[tl]).predict(B13))
                       + (1 - pt) * clip0(HGBR(**HGB).fit(Xp[~tl], yp[~tl]).predict(B13)))}
    for d, z in Z.items():
        for k, p in lin.items():                       # 선형 SFR로 학습한 모델: 예측을 로그로 옮겨 평가
            oof[d][k][te] = z(p)
        if d == "D3":                                  # 양수 은하만 평가하므로 양수로 학습한 log10 회귀만 쓴다
            oof[d][G13][te] = oof[d][H13][te] = HGBR(**HGB).fit(Xp, np.log10(yp)).predict(B13)
            oof[d][G30][te] = oof[d][H30][te] = HGBR(**HGB).fit(A30[pos], np.log10(yp)).predict(B30)
            continue
        z0 = float(z(0.0))                             # SFR = 0의 로그 값 (D1: 0, D2: -2)
        oof[d][G13][te] = HGBR(**HGB).fit(A13, z(ya)).predict(B13)
        oof[d][G30][te] = HGBR(**HGB).fit(A30, z(ya)).predict(B30)
        # 로그 공간의 2단계: E[z|x] = P(SFR>0|x)·E[z|SFR>0,x] + (1 − P)·z(0)
        oof[d][H13][te] = p30 * HGBR(**HGB).fit(Xp, z(yp)).predict(B13) + (1 - p30) * z0
        oof[d][H30][te] = p30 * HGBR(**HGB).fit(A30[pos], z(yp)).predict(B30) + (1 - p30) * z0

truth = {d: z(yv) for d, z in Z.items()}
mask = {"D1": np.ones(n, bool), "D2": np.ones(n, bool), "D3": yv > 0}
rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
label = {"D1": "log10(1+SFR)", "D2": f"log10(SFR+{EPS:g})", "D3": "log10(SFR), 양수만"}
tab = pd.DataFrame({f"{label[d]} 전체": {k: rm(oof[d][k][mask[d]], truth[d][mask[d]]) for k in NAMES} for d in Z})
for d in ("D1", "D2"):
    tab[f"{label[d]} | SFR=0"] = {k: rm(oof[d][k][yv == 0], truth[d][yv == 0]) for k in NAMES}
    tab[f"{label[d]} | SFR>0"] = {k: rm(oof[d][k][yv > 0], truth[d][yv > 0]) for k in NAMES}
tab.index.name = "model"
pd.set_option("display.width", 250)
print("학습 데이터 5-fold OOF의 RMSE (dex)")
print(tab.round(4).to_string())

share = {}
for d in ("D1", "D2"):
    sse = (oof[d][L_E] - truth[d]) ** 2
    share[label[d]] = {"SFR = 0": float(sse[yv == 0].sum() / sse.sum()), "0 < SFR ≤ 1": float(sse[(yv > 0) & (yv <= 1)].sum() / sse.sum()),
                       "SFR > 100": float(sse[yv > 100].sum() / sse.sum())}
    print(f"{label[d]}: Part II 초안 모델의 오차제곱합 비중 — " + ", ".join(f"{k} {v:.1%}" for k, v in share[label[d]].items()))

tab.round(4).to_csv(OUT / "log_metric_check_rmse.csv", encoding="utf-8-sig")
(OUT / "log_metric_check_results.json").write_text(json.dumps(
    {"n_train": n, "cv": "5-fold, seed 42, train split only", "eps_D2": EPS, "min_positive_sfr": YMIN, "hgb": HGB,
     "rmse_dex": json.loads(tab.to_json(force_ascii=False)), "sse_share_of_part2_draft_model": share,
     "seconds": time.time() - t_start}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"저장 완료, {time.time() - t_start:.0f}초")

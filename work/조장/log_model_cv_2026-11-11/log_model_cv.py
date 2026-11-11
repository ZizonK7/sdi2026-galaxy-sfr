"""2026-11-11 로그 기준(조교 FAQ Q6) 최종 모델 고르기 — 학습 분할 안 반복 CV (5-fold × seed 5 = 25 fold).

계획: ../2026-11-11_평가기준_변경과_계획.md 3절 "실험 순서" 1~3. 검증 데이터는 쓰지 않는다(최종 후보를 고정한 뒤 노트북에서 한 번).

로그 정의 (공지에 없음 — 셋 다 잰다)
  D1 = log10(1 + SFR)              ← 주 지표 (답이 오기 전 기본)
  D2 = log10(SFR + 0.01)           (작은 값을 더하는 방식의 예. 0.01은 임의)
  D3 = log10(SFR), 양수 은하만 평가
  회귀기의 타깃은 각 정의의 z로 따로 학습한다(D3은 양수 은하의 log10 SFR).

모델 (HGB 설정은 모두 기준선과 같다: learning_rate 0.08, max_leaf_nodes 31, max_iter 150, seed 42)
  M0  선형 기준선: HGB 회귀(13) → z(max(ŷ, 0))
  M1  Part II 초안: 환경 분류기(30) × 선형 양수 회귀(13) → z(ŷ)
  M2  환경 분류기(30) × 로그 양수 회귀(13):  E[z|x] = p·ẑ₊ + (1 − p)·z(0)
  M3  환경 분류기(30) × 로그 양수 회귀(30 = 13 + 환경 17)
  M4  M2/M3 중 고른 쪽 + 팀원 꼬리 분리(실험 B, T = 30): ẑ₊ = q·ẑ_꼬리 + (1 − q)·ẑ_보통, q = P(SFR > 30 | x, SFR > 0)
  M5  M2/M3 중 고른 쪽에서 분류기 트리 상한만 150 → 400 (중간 보고서에서 분류기가 상한 150에 닿았다)
  M2h 참고: M2를 딱 자르기(p > 0.5면 ẑ₊, 아니면 z(0)). 보고용, 고르기에 쓰지 않는다.

판단 규칙 (돌리기 전에 적음, 2026-11-11)
  - 비교는 같은 fold의 짝 차이 Δ = RMSE(후보) − RMSE(기준), D1로 판단한다.
  - 문턱은 'Δ < 0인 fold 20개(80%)'. 평균 Δ < 0 이고 22개 이상이면 '낫다'. 문턱 ±1인 19~21개는 '경계'로 적고
    단순한 쪽(기준)을 남긴다. 18개 이하이거나 평균 Δ ≥ 0이면 '낫지 않다'.
    (fold끼리 독립이 아니어서 구간은 내지 않고 평균·SD·최소·최대·음수 fold 수를 적는다.)
  - 질문 1 (회귀기에 환경 피처): M3 vs M2. 낫지 않으면 M2.
  - 질문 2 (꼬리 분리): M4 vs 질문 1에서 고른 쪽.
  - 질문 3 (분류기 트리 수): M5 vs 질문 1에서 고른 쪽.
  - 질문 2·3이 둘 다 '낫다'면 둘을 합친 것은 이 실행에서 재지 않았으므로 노트북에서 val 한 번으로만 확인하고, 판단은 각각의 결과로 적는다.
  - D2·D3에서 평균 Δ의 부호가 D1과 반대면 '정의에 따라 갈림'으로 적는다(판단은 D1).

실행: python log_model_cv.py   (4코어 CPU에서 약 10분)
출력: fold_rmse.csv (fold × 모델 × 정의), summary.json, pair_diffs.csv
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from sklearn.model_selection import KFold, train_test_split

OUT = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[3]
RANDOM_SEED = 42
HGB = dict(learning_rate=0.08, max_iter=150, max_leaf_nodes=31, random_state=RANDOM_SEED)
EPS = 0.01
T_TAIL = 30.0
CV_SEEDS = (42, 7, 11, 23, 101)


def build_environment_features(frame, box=200.0, fof_b=0.2, host_thresholds=(13.5, 14.0)):
    """제출 노트북 9.2와 같은 함수 (log_metric_check.py와 같은 정의, 타깃 미사용)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    RHO_CRIT, G_KMS = 2.775e11, 4.30091e-9
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
print(f"학습 {n:,}개 | SFR = 0 {np.mean(yv == 0):.2%} | 피처 계산 포함 {time.time() - t_start:.0f}초", flush=True)

Z = {"D1": lambda v: np.log10(1 + v), "D2": lambda v: np.log10(v + EPS), "D3": lambda v: np.log10(np.clip(v, YMIN, None))}
MASK = {"D1": np.ones(n, bool), "D2": np.ones(n, bool), "D3": yv > 0}
TRUTH = {d: z(yv) for d, z in Z.items()}
MODELS = ["M0", "M1", "M2", "M3", "M4_13", "M4_30", "M5_13", "M5_30", "M2h"]
rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))

rows = []
for seed in CV_SEEDS:
    for k, (tr, te) in enumerate(KFold(n_splits=5, shuffle=True, random_state=seed).split(Xtr)):
        t0 = time.time()
        Xa, Xb, ya = Xtr.iloc[tr], Xtr.iloc[te], yv[tr]
        pos = ya > 0
        tail = ya[pos] > T_TAIL
        p150 = HGBC(**HGB).fit(Xa[F30], pos.astype(int)).predict_proba(Xb[F30])[:, 1]
        c400 = HGBC(**{**HGB, "max_iter": 400}).fit(Xa[F30], pos.astype(int))
        p400 = c400.predict_proba(Xb[F30])[:, 1]
        lin_base = np.clip(HGBR(**HGB).fit(Xa[F13], ya).predict(Xb[F13]), 0, None)
        lin_pos = np.clip(HGBR(**HGB).fit(Xa.loc[pos, F13], ya[pos]).predict(Xb[F13]), 0, None)
        qt = {fs: HGBC(**HGB).fit(Xa.loc[pos, cols], tail.astype(int)).predict_proba(Xb[cols])[:, 1]
              for fs, cols in (("13", F13), ("30", F30))}
        yt = yv[te]
        for d, z in Z.items():
            z0 = float(z(0.0))
            zp = z(ya[pos]) if d != "D3" else np.log10(ya[pos])
            e = {fs: HGBR(**HGB).fit(Xa.loc[pos, cols], zp).predict(Xb[cols]) for fs, cols in (("13", F13), ("30", F30))}
            et = {}
            for fs, cols in (("13", F13), ("30", F30)):
                Xp = Xa.loc[pos, cols]
                e_tail = HGBR(**HGB).fit(Xp[tail], zp[tail]).predict(Xb[cols])
                e_norm = HGBR(**HGB).fit(Xp[~tail], zp[~tail]).predict(Xb[cols])
                et[fs] = qt[fs] * e_tail + (1 - qt[fs]) * e_norm
            if d == "D3":       # 양수 은하만 평가: 분류기는 끼지 않고 양수 회귀 ẑ₊만 본다
                pred = {"M0": z(lin_base), "M1": z(lin_pos), "M2": e["13"], "M3": e["30"], "M4_13": et["13"], "M4_30": et["30"],
                        "M5_13": e["13"], "M5_30": e["30"], "M2h": e["13"]}
            else:
                mix = lambda p, ez: p * ez + (1 - p) * z0
                pred = {"M0": z(lin_base), "M1": z(p150 * lin_pos),
                        "M2": mix(p150, e["13"]), "M3": mix(p150, e["30"]),
                        "M4_13": mix(p150, et["13"]), "M4_30": mix(p150, et["30"]),
                        "M5_13": mix(p400, e["13"]), "M5_30": mix(p400, e["30"]),
                        "M2h": np.where(p150 > 0.5, e["13"], z0)}
            m = MASK[d][te]
            for name, p in pred.items():
                r = {"seed": seed, "fold": k, "def": d, "model": name, "rmse": rm(p[m], TRUTH[d][te][m])}
                if d != "D3":
                    r["rmse_zero"] = rm(p[yt == 0], TRUTH[d][te][yt == 0])
                    r["rmse_pos"] = rm(p[yt > 0], TRUTH[d][te][yt > 0])
                rows.append(r)
        rows.append({"seed": seed, "fold": k, "def": "-", "model": "clf400_n_iter", "rmse": float(c400.n_iter_)})
        print(f"seed {seed} fold {k}: {time.time() - t0:.0f}초 (분류기 400 상한 실제 트리 {c400.n_iter_})", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "fold_rmse.csv", index=False, encoding="utf-8-sig")
res = res[res["def"] != "-"]


def pair(d, cand, ref):
    a = res[(res["def"] == d) & (res.model == cand)].set_index(["seed", "fold"])["rmse"]
    b = res[(res["def"] == d) & (res.model == ref)].set_index(["seed", "fold"])["rmse"]
    dd = (a - b).to_numpy()
    return {"def": d, "cand": cand, "ref": ref, "mean": float(dd.mean()), "sd": float(dd.std(ddof=1)), "min": float(dd.min()),
            "max": float(dd.max()), "n_neg": int((dd < 0).sum()), "n": len(dd)}


def verdict(p):
    if p["mean"] < 0 and p["n_neg"] >= 22:
        return "낫다"
    if p["mean"] < 0 and p["n_neg"] >= 19:
        return "경계 (단순한 쪽 유지)"
    return "낫지 않다"


def ask(cand, ref):
    ps = {d: pair(d, cand, ref) for d in Z}
    v = verdict(ps["D1"])
    split = [d for d in ("D2", "D3") if np.sign(ps[d]["mean"]) != np.sign(ps["D1"]["mean"])]
    return ps, v, split


summary = {"n_train": n, "cv": f"5-fold x seeds {list(CV_SEEDS)} on train split", "eps_D2": EPS, "T_tail": T_TAIL, "hgb": HGB}
summary["mean_rmse"] = {d: res[res["def"] == d].groupby("model")["rmse"].mean().round(5).to_dict() for d in Z}
summary["mean_rmse_zero"] = {d: res[res["def"] == d].groupby("model")["rmse_zero"].mean().round(5).to_dict() for d in ("D1", "D2")}
summary["mean_rmse_pos"] = {d: res[res["def"] == d].groupby("model")["rmse_pos"].mean().round(5).to_dict() for d in ("D1", "D2")}
pairs = []
q1, v1, s1 = ask("M3", "M2")
reg = "30" if v1 == "낫다" else "13"
base = "M3" if reg == "30" else "M2"
q2, v2, s2 = ask(f"M4_{reg}", base)
q3, v3, s3 = ask(f"M5_{reg}", base)
qh, vh, sh = ask("M2h", "M2")
q0, v0, s0 = ask("M2", "M1")
summary["decisions"] = {
    "Q1 회귀기 환경 피처 (M3 vs M2)": {"verdict": v1, "split_defs": s1},
    f"Q2 꼬리 분리 (M4_{reg} vs {base})": {"verdict": v2, "split_defs": s2},
    f"Q3 분류기 트리 400 (M5_{reg} vs {base})": {"verdict": v3, "split_defs": s3},
    "참고: 딱 자르기 (M2h vs M2)": {"verdict": vh, "split_defs": sh},
    "참고: 로그 회귀 (M2 vs M1)": {"verdict": v0, "split_defs": s0},
    "chosen_regressor_features": reg,
}
for ps in (q1, q2, q3, qh, q0):
    pairs.extend(ps.values())
pd.DataFrame(pairs).to_csv(OUT / "pair_diffs.csv", index=False, encoding="utf-8-sig")
summary["pairs"] = pairs
n_iter = pd.read_csv(OUT / "fold_rmse.csv").query("model == 'clf400_n_iter'")["rmse"]
summary["clf400_n_iter"] = {"min": float(n_iter.min()), "median": float(n_iter.median()), "max": float(n_iter.max())}
summary["seconds"] = time.time() - t_start
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

pd.set_option("display.width", 250)
print("\n25 fold 평균 RMSE (dex)")
print(pd.DataFrame(summary["mean_rmse"]).round(4).to_string())
print("\n짝 차이 (후보 − 기준)")
print(pd.DataFrame(pairs).round(4).to_string())
print("\n판단:", json.dumps(summary["decisions"], ensure_ascii=False, indent=1))
print(f"끝, {summary['seconds']:.0f}초")

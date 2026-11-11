"""질문 4 (첫 실행 뒤에 더함, 2026-11-11): 로그 회귀기도 트리 상한 150에 닿았다(전체 학습 n_iter_ = 150).
분류기와 같은 질문 — 회귀기 트리 상한 150 → 400 — 을 log_model_cv.py와 같은 25 fold, 같은 판단 규칙으로 본다.
기준 = M5_30 (분류기 400 × 로그 회귀 30, 회귀기 150). 후보 = 회귀기만 max_iter 400. 지표 D1 = log10(1 + SFR).
판단: 평균 Δ < 0 이고 Δ < 0 fold 22개 이상이면 '낫다', 19~21개는 '경계'(단순한 쪽 = 150 유지).
D2(log10(SFR + 0.01), 회귀기 타깃도 D2로 학습)는 함께 적되 판단에 쓰지 않는다. 실행 약 6분.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from sklearn.model_selection import KFold, train_test_split

from log_model_cv_env import build_environment_features  # noqa: E402  (같은 폴더, 아래에서 만든다)

OUT = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[3]
HGB = dict(learning_rate=0.08, max_iter=150, max_leaf_nodes=31, random_state=42)
CV_SEEDS = (42, 7, 11, 23, 101)
t_start = time.time()
df = pd.read_csv(ROOT / "Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv")
F13 = [c for c in df.columns if c not in ["subhalo_id", "sfr_linear"]]
ENV = build_environment_features(df)
F30 = F13 + list(ENV.columns)
Xtr, _, ytr, _ = train_test_split(df[F13], df["sfr_linear"], test_size=0.20, random_state=42, stratify=(df["sfr_linear"] > 0).astype(int))
Xtr = Xtr.join(ENV).reset_index(drop=True)
yv = ytr.to_numpy(float)
Z = {"D1": lambda v: np.log10(1 + v), "D2": lambda v: np.log10(v + 0.01)}
rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
rows = []
for seed in CV_SEEDS:
    for k, (tr, te) in enumerate(KFold(n_splits=5, shuffle=True, random_state=seed).split(Xtr)):
        t0 = time.time()
        Xa, Xb, ya = Xtr.iloc[tr], Xtr.iloc[te], yv[tr]
        pos = ya > 0
        p = HGBC(**{**HGB, "max_iter": 400}).fit(Xa[F30], pos.astype(int)).predict_proba(Xb[F30])[:, 1]
        for d, z in Z.items():
            z0 = float(z(0.0))
            for it in (150, 400):
                r = HGBR(**{**HGB, "max_iter": it}).fit(Xa.loc[pos, F30], z(ya[pos]))
                pred = p * r.predict(Xb[F30]) + (1 - p) * z0
                rows.append({"seed": seed, "fold": k, "def": d, "reg_max_iter": it, "rmse": rm(pred, z(yv[te])), "n_iter": r.n_iter_})
        print(f"seed {seed} fold {k}: {time.time() - t0:.0f}초", flush=True)
res = pd.DataFrame(rows)
res.to_csv(OUT / "q4_fold_rmse.csv", index=False, encoding="utf-8-sig")
out = {}
for d in Z:
    a = res[(res["def"] == d) & (res.reg_max_iter == 400)].set_index(["seed", "fold"])["rmse"]
    b = res[(res["def"] == d) & (res.reg_max_iter == 150)].set_index(["seed", "fold"])["rmse"]
    dd = (a - b).to_numpy()
    n_neg = int((dd < 0).sum())
    out[d] = {"mean_150": float(b.mean()), "mean_400": float(a.mean()), "mean_diff": float(dd.mean()), "sd": float(dd.std(ddof=1)),
              "min": float(dd.min()), "max": float(dd.max()), "n_neg": n_neg,
              "n_iter_400": res[(res["def"] == d) & (res.reg_max_iter == 400)]["n_iter"].describe()[["min", "50%", "max"]].tolist()}
v = out["D1"]
out["verdict_D1"] = "낫다" if v["mean_diff"] < 0 and v["n_neg"] >= 22 else ("경계 (150 유지)" if v["mean_diff"] < 0 and v["n_neg"] >= 19 else "낫지 않다")
out["seconds"] = time.time() - t_start
(OUT / "q4_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))

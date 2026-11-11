"""[질량 성장·비율 피처 4개를 더한 모델(M8)로 다시 고르기 — 질문 5 채택 뒤 M7 문턱 0.8이 규칙상 '나빠짐'(22/25)이 되어, 같은 규칙을 그대로 다시 적용한다]

0일 확률이 높으면 정확히 0을 내는 규칙 (2026-11-11, 조장 요청 — 11-06 실험정리의 '하드 분기 RMSE는 평가하지 않았다'를 마무리).

최종 모델(환경 분류기 × 로그 회귀기, 두 단계 모두 30개, 트리 상한 400)에서 출력만 바꾼다:
    P(SFR = 0 | x) = 1 − p ≥ τ  이면 ẑ = z(0) = 0 (SFR = 0을 그대로 출력),  아니면 ẑ = p·m + (1 − p)·0 (지금 방식).
τ 후보: 0.5, 0.7, 0.8, 0.9, 0.95, 0.99. 같은 25 fold(5-fold × seed 42, 7, 11, 23, 101), 지표 D1 = log10(1 + SFR).

판단 규칙 (돌리기 전에 적음)
  - 짝 차이 Δ = RMSE(규칙) − RMSE(지금 방식). '나빠짐' = 평균 Δ > 0 이고 Δ > 0인 fold가 22개 이상.
    '좋아짐' = 평균 Δ < 0 이고 Δ < 0인 fold가 22개 이상. 그 밖은 '같음'(19~21개면 '경계'로 표시).
  - 목적은 설명(SFR = 0을 그대로 출력)이므로, '나빠짐'이 아닌 τ 중 가장 낮은 τ(0을 가장 많이 내는 것)를 고른다.
    모든 τ가 '나빠짐'이면 규칙을 쓰지 않는다.
  - D2 = log10(SFR + 0.01)도 같은 예측으로 재서 함께 적는다(판단에 쓰지 않음).
실행 약 4분. 출력: zero_gate34_folds.csv, zero_gate34_summary.json
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from sklearn.model_selection import KFold, train_test_split

from log_model_cv_env import build_environment_features

OUT = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[3]
H = dict(learning_rate=0.08, max_leaf_nodes=31, random_state=42, max_iter=400)
TAUS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
t_start = time.time()
df = pd.read_csv(ROOT / "Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv")
F13 = [c for c in df.columns if c not in ["subhalo_id", "sfr_linear"]]
ENV = build_environment_features(df)
A = pd.DataFrame({"dlogMs_growth": df["log10_stellar_mass"] - df["log10_main_prog_stellar_mass"],
                  "dlogMh_growth": df["log10_halo_mass"] - df["log10_main_prog_mass"],
                  "bh_to_star": df["log10_BH_mass"] - df["log10_stellar_mass"],
                  "prog_star_to_halo": df["log10_main_prog_stellar_mass"] - df["log10_main_prog_mass"]}, index=df.index)
F = F13 + list(ENV.columns) + list(A.columns)
Xtr, _, ytr, _ = train_test_split(df[F13].join(A), df["sfr_linear"], test_size=0.20, random_state=42, stratify=(df["sfr_linear"] > 0).astype(int))
X = Xtr.join(ENV).reset_index(drop=True)[F]
y = ytr.to_numpy(float)
z = np.log10(1 + y)
rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
rows = []
for seed in (42, 7, 11, 23, 101):
    for k, (tr, te) in enumerate(KFold(n_splits=5, shuffle=True, random_state=seed).split(X)):
        pos = y[tr] > 0
        p = HGBC(**H).fit(X.iloc[tr], pos.astype(int)).predict_proba(X.iloc[te])[:, 1]
        m = HGBR(**H).fit(X.iloc[tr][pos], z[tr][pos]).predict(X.iloc[te])
        soft = p * m
        yt, zt = y[te], z[te]
        zero = yt == 0
        variants = {"soft": soft, **{f"tau{t}": np.where(1 - p >= t, 0.0, soft) for t in TAUS}}
        for name, zp in variants.items():
            sfr = np.clip(10 ** zp - 1, 0, None)          # 노트북의 from_log와 같이 0 아래를 자른다 (D2·선형 평가용)
            rows.append({"seed": seed, "fold": k, "rule": name, "d1": rm(zp, zt), "d1_zero": rm(zp[zero], zt[zero]), "d1_pos": rm(zp[~zero], zt[~zero]),
                         "d2": rm(np.log10(sfr + 0.01), np.log10(yt + 0.01)), "lin": rm(sfr, yt),
                         "zero_exact_share": float(np.mean(zp[zero] == 0)), "pos_set_zero": int(np.sum(zp[~zero] == 0)),
                         "pos_set_zero_max_sfr": float(yt[~zero][zp[~zero] == 0].max()) if np.any(zp[~zero] == 0) else 0.0})
        print(f"seed {seed} fold {k}: {time.time() - t_start:.0f}s", flush=True)
res = pd.DataFrame(rows)
res.to_csv(OUT / "zero_gate34_folds.csv", index=False, encoding="utf-8-sig")
mean = res.groupby("rule", sort=False).agg(d1=("d1", "mean"), d1_zero=("d1_zero", "mean"), d1_pos=("d1_pos", "mean"), d2=("d2", "mean"),
                                           lin=("lin", "mean"), zero_exact_share=("zero_exact_share", "mean"),
                                           pos_set_zero_per_fold=("pos_set_zero", "mean"), pos_set_zero_max_sfr=("pos_set_zero_max_sfr", "max"))
base = res[res.rule == "soft"].set_index(["seed", "fold"])["d1"]
verdicts = {}
for t in TAUS:
    d = (res[res.rule == f"tau{t}"].set_index(["seed", "fold"])["d1"] - base).to_numpy()
    up, dn = int((d > 0).sum()), int((d < 0).sum())
    v = "나빠짐" if d.mean() > 0 and up >= 22 else ("좋아짐" if d.mean() < 0 and dn >= 22 else
                                                  ("경계" if max(up, dn) >= 19 else "같음"))
    verdicts[f"tau{t}"] = {"mean_diff": float(d.mean()), "sd": float(d.std(ddof=1)), "min": float(d.min()), "max": float(d.max()),
                           "n_worse": up, "n_better": dn, "verdict": v}
ok = [t for t in TAUS if verdicts[f"tau{t}"]["verdict"] != "나빠짐"]
chosen = min(ok) if ok else None
summary = {"cv": "5-fold x seeds 42,7,11,23,101 on train split", "metric": "D1 log10(1+SFR) RMSE", "taus": TAUS,
           "mean": json.loads(mean.round(5).to_json(orient="index", force_ascii=False)), "verdicts": verdicts,
           "chosen_tau": chosen, "seconds": time.time() - t_start}
(OUT / "zero_gate34_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
pd.set_option("display.width", 250)
print(mean.round(4).to_string())
print(pd.DataFrame(verdicts).T.round(5).to_string())
print("고른 τ:", chosen)

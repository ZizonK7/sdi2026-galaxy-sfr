"""질문 5 (2026-11-11): 질량 성장·비율 피처 4개를 두 단계에 더하면 나아지나.

후보 피처 (주어진 열의 차이 — 트리는 한 번에 열 하나로만 나눠 차이를 스스로 만들기 어렵다):
  dlogMs_growth     = log10_stellar_mass − log10_main_prog_stellar_mass   (전구체 시점 대비 항성 질량 성장 = 최근 별 생성의 기록)
  dlogMh_growth     = log10_halo_mass − log10_main_prog_mass               (헤일로 성장 = 가스 유입의 대리)
  bh_to_star        = log10_BH_mass − log10_stellar_mass                   (블랙홀 상대 크기 = 피드백 세기의 대리)
  prog_star_to_halo = log10_main_prog_stellar_mass − log10_main_prog_mass  (전구체 시점의 별 생성 효율)
센티널(전구체 없음 등)은 그대로 둔다 — 원래 열도 입력에 남아 트리가 따로 가를 수 있다.

기준 = 지금 최종 M7 (환경 분류기 × 로그 회귀기, 30개, 트리 상한 400, P(0) ≥ 0.8이면 0).
후보 = M8: M7과 같고 두 단계 피처만 30 + 4 = 34개.
같은 25 fold(5-fold × seed 42, 7, 11, 23, 101), 지표 D1 = log10(1 + SFR).
판단 규칙(돌리기 전에 적음): 평균 Δ < 0 이고 Δ < 0인 fold 22개 이상이면 '낫다' → 채택. 19~21개는 '경계'(채택 안 함). 그 밖은 '낫지 않다'.
참고(판단에 안 씀): 문턱 없는 두 모델끼리(M6 vs M6+A)의 Δ, D2 = log10(SFR + 0.01), SFR = 0·> 0 나눠 보기.
이 질문은 같은 날 5-fold 한 번의 탐색(0.1029 → 0.0967)을 본 뒤에 세웠다. 실행 약 5분.
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
TAU = 0.8
t_start = time.time()
df = pd.read_csv(ROOT / "Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv")
F13 = [c for c in df.columns if c not in ["subhalo_id", "sfr_linear"]]
ENV = build_environment_features(df)
A = pd.DataFrame({"dlogMs_growth": df["log10_stellar_mass"] - df["log10_main_prog_stellar_mass"],
                  "dlogMh_growth": df["log10_halo_mass"] - df["log10_main_prog_mass"],
                  "bh_to_star": df["log10_BH_mass"] - df["log10_stellar_mass"],
                  "prog_star_to_halo": df["log10_main_prog_stellar_mass"] - df["log10_main_prog_mass"]}, index=df.index)
F30 = F13 + list(ENV.columns)
F34 = F30 + list(A.columns)
Xtr, _, ytr, _ = train_test_split(df[F13].join(ENV).join(A), df["sfr_linear"], test_size=0.20, random_state=42, stratify=(df["sfr_linear"] > 0).astype(int))
X = Xtr.reset_index(drop=True)
y = ytr.to_numpy(float)
rm = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
Z = {"D1": (lambda v: np.log10(1 + v)), "D2": (lambda v: np.log10(v + 0.01))}
rows = []
for seed in (42, 7, 11, 23, 101):
    for k, (tr, te) in enumerate(KFold(n_splits=5, shuffle=True, random_state=seed).split(X)):
        pos = y[tr] > 0
        z1 = np.log10(1 + y)
        out = {}
        for tag, cols in (("30", F30), ("34", F34)):
            p = HGBC(**H).fit(X.iloc[tr][cols], pos.astype(int)).predict_proba(X.iloc[te][cols])[:, 1]
            m = HGBR(**H).fit(X.iloc[tr][cols][pos], z1[tr][pos]).predict(X.iloc[te][cols])
            soft = p * m
            out[f"M6_{tag}"] = soft
            out[f"M7_{tag}"] = np.where(1 - p >= TAU, 0.0, soft)
        yt = y[te]
        zero = yt == 0
        for name, zp in out.items():
            sfr = np.clip(10 ** zp - 1, 0, None)
            rows.append({"seed": seed, "fold": k, "model": name, "d1": rm(zp, z1[te]), "d1_zero": rm(zp[zero], z1[te][zero]),
                         "d1_pos": rm(zp[~zero], z1[te][~zero]), "d2": rm(np.log10(sfr + 0.01), np.log10(yt + 0.01)), "lin": rm(sfr, yt)})
        print(f"seed {seed} fold {k}: {time.time() - t_start:.0f}s", flush=True)
res = pd.DataFrame(rows)
res.to_csv(OUT / "q5_folds.csv", index=False, encoding="utf-8-sig")
mean = res.groupby("model", sort=False)[["d1", "d1_zero", "d1_pos", "d2", "lin"]].mean()


def pair(cand, ref, col="d1"):
    d = (res[res.model == cand].set_index(["seed", "fold"])[col] - res[res.model == ref].set_index(["seed", "fold"])[col]).to_numpy()
    n = int((d < 0).sum())
    v = "낫다" if d.mean() < 0 and n >= 22 else ("경계" if d.mean() < 0 and n >= 19 else "낫지 않다")
    return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "min": float(d.min()), "max": float(d.max()), "n_better": n, "verdict": v}


summary = {"question": "Q5 mass-history features (M8 = M7 + 4) vs M7", "features": list(A.columns),
           "mean": json.loads(mean.round(5).to_json(orient="index")),
           "Q5 M7_34 vs M7_30 (D1, 판단)": pair("M7_34", "M7_30"),
           "참고 M6_34 vs M6_30 (D1)": pair("M6_34", "M6_30"), "참고 M7_34 vs M7_30 (D2)": pair("M7_34", "M7_30", "d2"),
           "참고 M7_34 vs M6_34 (D1, 문턱이 여전히 나빠지지 않나)": pair("M7_34", "M6_34"), "seconds": time.time() - t_start}
(OUT / "q5_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
pd.set_option("display.width", 200)
print(mean.round(4).to_string())
for k, v in summary.items():
    if isinstance(v, dict) and "verdict" in v:
        print(k, {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()})

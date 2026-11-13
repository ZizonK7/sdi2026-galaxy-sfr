import sys, numpy as np, pandas as pd
CODE, DATA = sys.argv[1], sys.argv[2]  # argv[3]는 쓰지 않음
sys.path.insert(0, CODE)
from sklearn.ensemble import HistGradientBoostingClassifier as HC, HistGradientBoostingRegressor as HR
from leader_model import HGB_FIXED400
from feature_pipeline import build_reference_features, FeatureEngineer, UNIVERSE, BASE
def two_stage(A, B, ytr, cols):
    pos = ytr > 0
    p = HC(**HGB_FIXED400).fit(A[cols], pos.astype(int)).predict_proba(B[cols])[:, 1]
    m = HR(**HGB_FIXED400).fit(A.loc[pos, cols], np.log10(1 + ytr[pos])).predict(B[cols])
    return np.maximum(10 ** np.where(1 - p >= .9, 0., p * m) - 1, 0.)
df = pd.read_csv(DATA)
from sklearn.model_selection import train_test_split
ids, vids = train_test_split(np.arange(len(df)), test_size=.2, random_state=42, stratify=(df.sfr_linear > 0).astype(int))
y = df.sfr_linear.to_numpy(); Xall = build_reference_features(df)
from threadpoolctl import threadpool_limits
with threadpool_limits(4):
    eng = FeatureEngineer().fit(Xall.iloc[ids]); A, B = eng.transform(Xall.iloc[ids]), eng.transform(Xall.iloc[vids])
    pa = two_stage(A, B, y[ids], BASE); pb = two_stage(A, B, y[ids], UNIVERSE)
yv = y[vids]; za, zb, zt = np.log10(1+pa), np.log10(1+pb), np.log10(1+yv)
ea, eb = (za-zt)**2, (zb-zt)**2
rng = np.random.default_rng(42); n = len(yv); d = []
for _ in range(2000):
    i = rng.integers(0, n, n); d.append(np.sqrt(eb[i].mean()) - np.sqrt(ea[i].mean()))
d = np.array(d)
print(f"holdout A {np.sqrt(ea.mean()):.6f}  B {np.sqrt(eb.mean()):.6f}  diff {np.sqrt(eb.mean())-np.sqrt(ea.mean()):+.6f}  95% [{np.quantile(d,.025):+.6f}, {np.quantile(d,.975):+.6f}]  P(diff<0) {np.mean(d<0):.3f}")
print(f"better rows: B closer on {np.mean(np.abs(zb-zt) < np.abs(za-zt)):.1%} of galaxies; median |err| A {np.median(np.abs(za-zt)):.4f} B {np.median(np.abs(zb-zt)):.4f}")
for name, m in {"SFR=0": yv==0, "0<SFR<=1": (yv>0)&(yv<=1), "1<SFR<=10": (yv>1)&(yv<=10), "10<SFR<=100": (yv>10)&(yv<=100), "SFR>100": yv>100}.items():
    print(f"{name:12s} n={m.sum():6d}  A {np.sqrt(ea[m].mean()):.4f}  B {np.sqrt(eb[m].mean()):.4f}  SSE share of improvement {(ea[m].sum()-eb[m].sum())/(ea.sum()-eb.sum()):+.1%}")

"""팀원 50:50 모델 검토: 같은 fold에서 (A) 조장 2단계 34개, (B) 2단계 46개, (C) 팀원 최종(계층형 46개 + 50:50 꼬리)을 비교.
그리고 각 모델을 train80 전체로 학습해 holdout 20%(val)에서 한 번 평가한다. 선택에는 쓰지 않는다."""
import sys, json, time
import numpy as np, pandas as pd
CODE = sys.argv[1]; DATA = sys.argv[2]; OUT = sys.argv[3]
sys.path.insert(0, CODE)
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import HistGradientBoostingClassifier as HC, HistGradientBoostingRegressor as HR
from threadpoolctl import threadpool_limits
from leader_model import HGB_FIXED400
from feature_pipeline import build_reference_features, FeatureEngineer, UNIVERSE, BASE, fit_component
from hierarchical_model import fit_binary_gates, hierarchical_probabilities, positive_probability
from three_regime import combine
from weighted_tail import fit_tail

df = pd.read_csv(DATA)
ids, vids = train_test_split(np.arange(len(df)), test_size=.2, random_state=42, stratify=(df.sfr_linear > 0).astype(int))
yall = df.sfr_linear.to_numpy()
Xall = build_reference_features(df)
assert list(Xall.columns[:34]) == BASE

def two_stage(A, B, ytr, cols):
    pos = ytr > 0
    p = HC(**HGB_FIXED400).fit(A[cols], pos.astype(int)).predict_proba(B[cols])[:, 1]
    m = HR(**HGB_FIXED400).fit(A.loc[pos, cols], np.log10(1 + ytr[pos])).predict(B[cols])
    z = np.where(1 - p >= .9, 0., p * m)
    return np.maximum(10 ** z - 1, 0.)

def teammate(A, B, ytr):
    g0, gt = fit_binary_gates(A[UNIVERSE], ytr)
    p = hierarchical_probabilities(positive_probability(g0, B[UNIVERSE]), positive_probability(gt, B[UNIVERSE]))
    mid = fit_component('mid', UNIVERSE, A, ytr).predict(B[UNIVERSE])
    mk = ytr > 10
    t1 = fit_tail('lgb_w1', A.loc[mk, UNIVERSE], ytr[mk]).predict(B[UNIVERSE])
    t3 = fit_tail('lgb_w3', A.loc[mk, UNIVERSE], ytr[mk]).predict(B[UNIVERSE])
    out = {}
    for name, tail in {'C0_teammate_unblended': t1, 'C_teammate_5050': .5 * t1 + .5 * t3}.items():
        out[name] = np.maximum(10 ** combine(p, mid, tail, .9) - 1, 0.)
    return out

def run(Xtr, ytr, Xte):
    eng = FeatureEngineer().fit(Xtr)
    A, B = eng.transform(Xtr), eng.transform(Xte)
    preds = {'A_two_stage_34': two_stage(A, B, ytr, BASE), 'B_two_stage_46': two_stage(A, B, ytr, UNIVERSE)}
    preds.update(teammate(A, B, ytr))
    return preds

def lr(y, p): return float(np.sqrt(np.mean((np.log10(1 + y) - np.log10(1 + p)) ** 2)))
def scores(y, p, fold=None):
    r = {'log_rmse': lr(y, p), 'zero': lr(y[y == 0], p[y == 0]), 'pos': lr(y[y > 0], p[y > 0]), 'tail100': lr(y[y > 100], p[y > 100]),
         'mid_0_10': lr(y[(y > 0) & (y <= 10)], p[(y > 0) & (y <= 10)]), 'tail10_100': lr(y[(y > 10) & (y <= 100)], p[(y > 10) & (y <= 100)])}
    if fold is not None: r['mean_fold'] = float(np.mean([lr(y[fold == f], p[fold == f]) for f in np.unique(fold)]))
    return r

X = Xall.iloc[ids].reset_index(drop=True); y = yall[ids]
res = {}; fold_rows = []
with threadpool_limits(limits=4):
    for seed in (42, 7):
        oof = {}; fold = np.zeros(len(y), int)
        for f, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=seed).split(X), 1):
            t = time.time(); fold[va] = f
            for k, v in run(X.iloc[tr], y[tr], X.iloc[va]).items():
                oof.setdefault(k, np.zeros(len(y)))[va] = v
                fold_rows.append({'seed': seed, 'fold': f, 'model': k, 'log_rmse': lr(y[va], v)})
            print(f'seed {seed} fold {f} {time.time()-t:.0f}s', flush=True)
        res[f'cv_seed{seed}'] = {k: scores(y, v, fold) for k, v in oof.items()}
    Xv = Xall.iloc[vids]; yv = yall[vids]
    vp = run(X, y, Xv.reset_index(drop=True))
    res['holdout_val'] = {k: scores(yv, v) for k, v in vp.items()}
pd.DataFrame(fold_rows).to_csv(OUT + '/review_folds.csv', index=False)
json.dump(res, open(OUT + '/review_results.json', 'w'), indent=1)
print(json.dumps(res, indent=1))

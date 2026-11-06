"""Compare learned zero-SFR classifiers with the fixed Gaussian-density gate.

Same middle-density population, original seed=42 split, and threshold=0.5.
Two controlled changes are evaluated separately:
1. Learn P(SFR=0 | density) freely with a density-only HGB classifier.
2. Learn P(SFR=0 | all 13 features), with density included.
No regression, threshold tuning, or hyperparameter search is performed.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
import sys

if os.environ.get("SFR_ANALYSIS_DEPENDENCIES"):
    sys.path.insert(0, os.environ["SFR_ANALYSIS_DEPENDENCIES"])
import joblib
import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold, train_test_split
from fixed_density_zero_classifier import FixedDensityZeroClassifier, classification_metrics

PARAMETERS = dict(learning_rate=.08, max_iter=150, max_leaf_nodes=31, random_state=42)
METHODS = ["fixed_gaussian_density", "learned_density_only", "learned_all_features"]
THRESHOLD = .5


def fit_learned_models(X, density, sfr, train, test):
    labels = (sfr[train] == 0).astype(int)
    d_model = HistGradientBoostingClassifier(**PARAMETERS).fit(density[train, None], labels)
    x_model = HistGradientBoostingClassifier(**PARAMETERS).fit(X[train], labels)
    assert d_model.classes_.tolist() == x_model.classes_.tolist() == [0, 1]
    return {
        "learned_density_only": d_model.predict_proba(density[test, None])[:, 1],
        "learned_all_features": x_model.predict_proba(X[test])[:, 1],
    }, {"learned_density_only": d_model, "learned_all_features": x_model}


def paired_accuracy_comparison(sfr, reference, candidate):
    actual = sfr == 0
    a = (reference >= THRESHOLD) == actual
    b = (candidate >= THRESHOLD) == actual
    improved, worsened = int(np.sum(~a & b)), int(np.sum(a & ~b))
    n = improved + worsened
    diff = b.astype(float) - a.astype(float)
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(40):
        ix = rng.integers(0, len(diff), (25, len(diff)))
        boot.extend(diff[ix].mean(axis=1))
    return {
        "accuracy_delta": float(diff.mean()),
        "paired_row_bootstrap_95CI": np.quantile(boot, [.025, .975]).tolist(),
        "corrected_reference_errors": improved,
        "introduced_new_errors": worsened,
        "exact_McNemar_pvalue": float(stats.binomtest(improved, n, .5).pvalue) if n else 1.,
    }


def make_plot(fixed, models, X, density, sfr, test, validation, cv, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output/".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("C:/Windows/Fonts/malgun.ttf")
    if font.exists(): font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": "Malgun Gothic" if font.exists() else "DejaVu Sans",
                         "font.size": 11, "axes.unicode_minus": False, "svg.fonttype": "none"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 7), facecolor="#FAFBFD",
                              gridspec_kw={"width_ratios": [1.2, 1]})
    fig.subplots_adjust(left=.075, right=.975, top=.73, bottom=.25, wspace=.28)
    fig.text(.075, .92, "0/양수 판정: 확률 함수를 학습하면 달라질까?", fontsize=23,
             weight="bold", color="#18273B")
    fig.text(.075, .85, "같은 중간 밀도 구간 · 같은 검증 은하 6,948개 · 모든 모델의 0 판정 문턱값 0.5", fontsize=12, color="#526175")
    grid = np.linspace(fixed.lower+1e-8, fixed.upper, 400)
    a.plot(grid, fixed.predict_zero_probability(grid), color="#CC6439", lw=2.4, label="기존 정규분포 기반 함수")
    a.plot(grid, models["learned_density_only"].predict_proba(grid[:, None])[:, 1],
           color="#286BB5", lw=1.7, label="밀도만 사용해 학습한 함수")
    edges = np.linspace(fixed.lower, fixed.upper, 19)
    centers, rates = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (density[test] > lo) & (density[test] <= hi)
        if m.any(): centers.append((lo+hi)/2); rates.append(float(np.mean(sfr[test][m] == 0)))
    a.scatter(centers, rates, color="#526175", s=20, zorder=4, label="검증의 실제 0 비율")
    a.axhline(THRESHOLD, color="#A1AAB6", ls="--", lw=1)
    a.set(xlim=(fixed.lower-.025, fixed.upper+.025), ylim=(-.025, 1.04),
          xlabel="주변 밀도 log10(1 + δ)", ylabel="SFR=0 확률 / 비율")
    a.set_title("밀도만 사용하는 두 함수 비교", loc="left", pad=15, fontsize=14, weight="bold", color="#24415F")
    a.legend(frameon=False, fontsize=9, loc="lower right")
    positions = np.arange(3); width = .32
    ve = [100*(1-validation[k]["accuracy"]) for k in METHODS]
    ce = [100*(1-cv[k]["accuracy"]) for k in METHODS]
    b.bar(positions-width/2, ve, width=width, color="#4480BE", label="기존 검증 분할", zorder=3)
    b.bar(positions+width/2, ce, width=width, color="#91B6DF", label="학습 내 5-fold", zorder=3)
    for i in range(3):
        b.text(i-width/2, ve[i]+.35, f"{ve[i]:.2f}", ha="center", fontsize=10, color="#24415F")
        b.text(i+width/2, ce[i]+.35, f"{ce[i]:.2f}", ha="center", fontsize=10, color="#24415F")
    b.set_xticks(positions, ["기존 정규분포\n밀도 함수", "밀도만 사용\n함수 학습", "전체 13개 피처\n확률 학습"], fontsize=10)
    b.set_ylim(0, max(ve+ce)+3.6); b.set_ylabel("오판정 비율 (%)", labelpad=10)
    b.set_title("오류율 비교 (낮을수록 좋음)", loc="left", pad=15, fontsize=14, weight="bold", color="#24415F")
    b.legend(frameon=False, fontsize=9, loc="upper right")
    for p in (a, b):
        p.set_facecolor("#FAFBFD"); p.grid(axis="y", color="#DCE3ED", lw=.7, zorder=0)
        p.spines[["top", "right", "left"]].set_visible(False)
        p.spines["bottom"].set_color("#CDD7E3"); p.tick_params(length=0, pad=8, colors="#526175")
    v = validation["learned_all_features"]
    fig.text(.075, .14, f"전체 피처 모델: 0 판정 정밀도 {v['zero_precision']:.1%} · 0 재현율 {v['zero_recall']:.1%} · ROC-AUC {v['ROC_AUC']:.3f}",
             fontsize=12, weight="bold", color="#24415F")
    fig.text(.075, .08, "밀도만 학습하는 비교는 함수 형태의 효과, 전체 피처 비교는 추가 정보의 효과를 확인합니다.", fontsize=11, color="#526175")
    fig.text(.075, .03, "정규분포 가정·문턱값 튜닝 없이 HGB 분류를 사용했습니다. CV에서는 각 학습 fold 안에서 구간 경계를 다시 계산했습니다.",
             fontsize=9, color="#768398")
    for ext in ["png", "svg"]:
        fig.savefig(output/f"classifier_comparison.{ext}", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[2]/
                        "Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("learned_zero_classifier_results"))
    args = parser.parse_args(); out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    with args.data.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f); columns = reader.fieldnames; rows = list(reader)
    feature_names = [c for c in columns if c not in ["subhalo_id", "sfr_linear"]]
    X = np.array([[float(r[c]) for c in feature_names] for r in rows])
    density = X[:, feature_names.index("log10_overdensity_plus_1")]
    sfr = np.array([float(r["sfr_linear"]) for r in rows])
    tr, va = train_test_split(np.arange(len(sfr)), test_size=.2, random_state=42,
                              stratify=(sfr > 0).astype(int))
    prior_dir = Path(__file__).with_name("fixed_density_zero_classifier_results")
    fixed = FixedDensityZeroClassifier.load(prior_dir/"model.npz")
    train = tr[fixed.middle_mask(density[tr])]; test = va[fixed.middle_mask(density[va])]
    assert len(train) == 27517 and len(test) == 6948
    probabilities, models = fit_learned_models(X, density, sfr, train, test)
    probabilities["fixed_gaussian_density"] = fixed.predict_zero_probability(density[test])
    with (prior_dir/"validation_predictions.csv").open(newline="", encoding="utf-8-sig") as f:
        old_rows = list(csv.DictReader(f))
    assert [int(r["csv_row_index"]) for r in old_rows] == test.tolist()
    assert np.allclose(probabilities["fixed_gaussian_density"], [float(r["probability_zero"]) for r in old_rows])
    validation = {key: classification_metrics(sfr[test], probabilities[key], THRESHOLD) for key in METHODS}
    for key, model in models.items():
        features = ["log10_overdensity_plus_1"] if key == "learned_density_only" else feature_names
        bundle = {"model": model, "features": features, "lower_exclusive": fixed.lower,
                  "upper_inclusive": fixed.upper, "threshold": THRESHOLD,
                  "class_labels": {"0": "SFR>0", "1": "SFR=0"}}
        joblib.dump(bundle, out/f"{key}.joblib")
        restored = joblib.load(out/f"{key}.joblib")
        inputs = density[test[:100], None] if key == "learned_density_only" else X[test[:100]]
        assert np.allclose(restored["model"].predict_proba(inputs)[:, 1], probabilities[key][:100])
    paired = {key: paired_accuracy_comparison(sfr[test], probabilities["fixed_gaussian_density"], probabilities[key])
              for key in METHODS[1:]}
    print(json.dumps({"stage": "validation", "metrics": validation, "paired": paired}), flush=True)
    folds = []; oof_indices = []; oof = {key: [] for key in METHODS}
    for fold, (ft, fe) in enumerate(KFold(n_splits=5, shuffle=True, random_state=42).split(tr), 1):
        fm = FixedDensityZeroClassifier.fit(density[tr[ft]], sfr[tr[ft]])
        train_fold = tr[ft][fm.middle_mask(density[tr[ft]])]
        test_fold = tr[fe][fm.middle_mask(density[tr[fe]])]
        fp, _ = fit_learned_models(X, density, sfr, train_fold, test_fold)
        fp["fixed_gaussian_density"] = fm.predict_zero_probability(density[test_fold])
        oof_indices.extend(test_fold.tolist())
        for key in METHODS: oof[key].extend(fp[key].tolist())
        row = {"fold": fold, "lower": fm.lower, "upper": fm.upper,
               "metrics": {key: classification_metrics(sfr[test_fold], fp[key], THRESHOLD) for key in METHODS}}
        folds.append(row)
        print(json.dumps({"fold": fold, "accuracy": {key: row["metrics"][key]["accuracy"] for key in METHODS}}), flush=True)
    oof_indices = np.asarray(oof_indices)
    cv = {key: classification_metrics(sfr[oof_indices], np.asarray(oof[key]), THRESHOLD) for key in METHODS}
    previous = json.loads((prior_dir/"metrics.json").read_text(encoding="utf-8"))
    assert np.isclose(previous["train_CV_pooled"]["accuracy"], cv["fixed_gaussian_density"]["accuracy"])
    report = {
        "scope": "Middle-density SFR=0 versus SFR>0 classification only; no regression",
        "threshold": THRESHOLD, "training_middle": len(train), "validation_middle": len(test),
        "lower_exclusive": fixed.lower, "upper_inclusive": fixed.upper,
        "parameters": PARAMETERS, "all_feature_names": feature_names,
        "validation": validation, "train_CV_pooled": cv, "train_CV_folds": folds,
        "validation_paired_comparison_against_fixed": paired,
        "iterations": {key: int(model.n_iter_) for key, model in models.items()},
        "controls": ["Same validation rows, threshold, and CV partitions for all methods",
                     "Density-only learner changes function form without adding features",
                     "All-feature learner additionally uses the 12 non-density features",
                     "Existing saved fixed probability function reused for validation",
                     "All CV cutoffs and models fit only within training fold",
                     "Saved/reloaded learned models reproduce probabilities"],
        "limitations": ["Validation set reused after earlier exploration; this is not a fresh final test",
                        "No tuning or additional probability calibration",
                        "Row-wise paired inference assumes independent galaxies",
                        "Paired p-values are exploratory; two comparisons were examined"],
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "scikit-learn", "joblib"]},
    }
    (out/"metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out/"comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        keys = ["method", "accuracy", "zero_precision", "zero_recall", "zero_F1", "ROC_AUC", "log_loss", "SFR_gt100_pred_zero", "CV_accuracy"]
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); writer.writeheader()
        for key in METHODS: writer.writerow({"method": key, **validation[key], "CV_accuracy": cv[key]["accuracy"]})
    with (out/"validation_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["csv_row_index", "density", "sfr", "actual_is_zero"]+
                        [item for key in METHODS for item in [f"{key}_q_zero", f"{key}_predicted_is_zero"]])
        for j, idx in enumerate(test):
            writer.writerow([int(idx), density[idx], sfr[idx], int(sfr[idx] == 0)]+
                            [item for key in METHODS for item in [probabilities[key][j], int(probabilities[key][j] >= THRESHOLD)]])
    make_plot(fixed, models, X, density, sfr, test, validation, cv, out)
    print(json.dumps({"validation": validation, "CV": cv, "paired": paired}, indent=2), flush=True)


if __name__ == "__main__":
    main()

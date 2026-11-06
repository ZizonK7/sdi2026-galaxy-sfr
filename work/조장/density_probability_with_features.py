"""Frozen density probability + full features versus learned density probability + full features.

For both primary models, the final classifier G receives the original 13 features
AND one density-derived probability. Only the source of that probability changes:
fixed_q_plus_X: truncated-Gaussian Bayes q_G(d), with no later adjustment to q_G;
learned_q_plus_X: density-only HGB q_L(d), without a Gaussian shape assumption.
G learns the final P(SFR=0 | X), so its output need not equal the density probability.

Five-fold cross-fitting supplies both probability features for G's training rows.
Every outer CV fold refits boundaries, inner probability models, and G using only
outer training data. The final classification threshold is always 0.5.
The frozen function is an auxiliary feature; its weight/influence is not forced.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse
import csv
from dataclasses import replace
import importlib.metadata
import json
from pathlib import Path
import sys
import time

if os.environ.get("SFR_ANALYSIS_DEPENDENCIES"):
    sys.path.insert(0, os.environ["SFR_ANALYSIS_DEPENDENCIES"])
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold, train_test_split
from fixed_density_zero_classifier import FixedDensityZeroClassifier, classification_metrics
from learned_zero_classifier import paired_accuracy_comparison

HGB = dict(learning_rate=.08, max_iter=150, max_leaf_nodes=31, random_state=42)
THRESHOLD = .5
METHODS = ["fixed_q_plus_X", "learned_q_plus_X", "X_only_reference"]


def fit_density_learner(density, sfr):
    return HistGradientBoostingClassifier(**HGB).fit(density[:, None], (sfr == 0).astype(int))


def fit_partition(X, d, sfr, train, test):
    # Cohort boundaries belong to this outer training partition only.
    frozen = FixedDensityZeroClassifier.fit(d[train], sfr[train])
    bounds = (frozen.lower, frozen.upper)
    mt = train[frozen.middle_mask(d[train])]
    mv = test[frozen.middle_mask(d[test])]
    fixed_oof = np.full(len(mt), np.nan)
    learned_oof = np.full(len(mt), np.nan)
    visits = np.zeros(len(mt), dtype=int)
    for inner_train, inner_test in KFold(n_splits=5, shuffle=True, random_state=42).split(mt):
        ti, vi = mt[inner_train], mt[inner_test]
        assert np.intersect1d(ti, vi).size == 0
        inner_fixed = FixedDensityZeroClassifier.fit(d[ti], sfr[ti], bounds=bounds)
        inner_learned = fit_density_learner(d[ti], sfr[ti])
        fixed_oof[inner_test] = inner_fixed.predict_zero_probability(d[vi])
        learned_oof[inner_test] = inner_learned.predict_proba(d[vi, None])[:, 1]
        visits[inner_test] += 1
    assert np.all(visits == 1)
    assert np.isfinite(fixed_oof).all() and np.isfinite(learned_oof).all()
    learned = fit_density_learner(d[mt], sfr[mt])
    fixed_q = frozen.predict_zero_probability(d[mv])
    learned_q = learned.predict_proba(d[mv, None])[:, 1]
    train_inputs = {
        "fixed_q_plus_X": np.column_stack([X[mt], fixed_oof]),
        "learned_q_plus_X": np.column_stack([X[mt], learned_oof]),
        "X_only_reference": X[mt],
    }
    test_inputs = {
        "fixed_q_plus_X": np.column_stack([X[mv], fixed_q]),
        "learned_q_plus_X": np.column_stack([X[mv], learned_q]),
        "X_only_reference": X[mv],
    }
    labels = (sfr[mt] == 0).astype(int)
    predictions, models = {}, {}
    for key in METHODS:
        models[key] = HistGradientBoostingClassifier(**HGB).fit(train_inputs[key], labels)
        assert models[key].classes_.tolist() == [0, 1]
        predictions[key] = models[key].predict_proba(test_inputs[key])[:, 1]
    return {
        "train_indices": mt, "test_indices": mv, "predictions": predictions,
        "frozen_density_function": frozen, "learned_density_function": learned,
        "final_classifiers": models, "fixed_q": fixed_q, "learned_q": learned_q,
        "training_probability_features": {"fixed": fixed_oof, "learned": learned_oof},
    }


def predict_bundle(bundle, X):
    """Return final P(SFR=0 | X) for the bundle's middle-density interval."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(bundle["features"]):
        raise ValueError("X must contain the bundle's features in saved order")
    density = X[:, bundle["density_column"]]
    if not np.all((density > bundle["lower_exclusive"]) & (density <= bundle["upper_inclusive"])):
        raise ValueError("This model is defined only inside the middle-density interval")
    if bundle["method"] == "fixed_q_plus_X":
        q = bundle["density_function"].predict_zero_probability(density)
    elif bundle["method"] == "learned_q_plus_X":
        q = bundle["density_function"].predict_proba(density[:, None])[:, 1]
    else:
        return bundle["classifier"].predict_proba(X)[:, 1]
    return bundle["classifier"].predict_proba(np.column_stack([X, q]))[:, 1]


def make_plot(validation, cv, paired, output):
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
                              gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(left=.075, right=.97, top=.73, bottom=.25, wspace=.3)
    fig.text(.075, .93, "밀도 확률 + 다른 피처로 최종 0 확률 학습", fontsize=23, weight="bold", color="#18273B")
    fig.text(.075, .86, "고정 확률과 학습 확률을 각각 같은 13개 피처에 추가 · 같은 최종 분류기 · 문턱값 0.5", fontsize=12, color="#526175")
    names = ["고정 qG(d)\n+ 전체 피처", "학습 qL(d)\n+ 전체 피처", "전체 피처만\n참고 모델"]
    pos = np.arange(3); width = .32
    v = [100*validation[k]["accuracy"] for k in METHODS]
    c = [100*cv[k]["accuracy"] for k in METHODS]
    a.bar(pos-width/2, v, width=width, color="#4480BE", label="기존 검증 분할", zorder=3)
    a.bar(pos+width/2, c, width=width, color="#91B6DF", label="학습 내 5-fold", zorder=3)
    for i in range(3):
        a.text(i-width/2, v[i]+.6, f"{v[i]:.2f}", ha="center", fontsize=10, color="#24415F")
        a.text(i+width/2, c[i]+.6, f"{c[i]:.2f}", ha="center", fontsize=10, color="#24415F")
    a.set_xticks(pos, names, fontsize=10); a.set_ylim(0, 103)
    a.set_ylabel("0/양수 판정 정확도 (%)", labelpad=10)
    a.set_title("정확도 비교", loc="left", pad=15, fontsize=14, weight="bold", color="#24415F")
    fig.legend(*a.get_legend_handles_labels(), frameon=False, fontsize=9,
               ncol=2, loc="upper left", bbox_to_anchor=(.075, .82))
    a.set_facecolor("#FAFBFD"); a.grid(axis="y", color="#DCE3ED", lw=.7, zorder=0)
    a.spines[["top", "right", "left"]].set_visible(False)
    a.spines["bottom"].set_color("#CDD7E3"); a.tick_params(length=0, pad=8, colors="#526175")
    m = validation["fixed_q_plus_X"]["confusion"]
    cm = np.array([[m["true_zero_pred_zero"], m["true_zero_pred_positive"]],
                   [m["true_positive_pred_zero"], m["true_positive_pred_positive"]]])
    b.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max()*1.1)
    for i in range(2):
        for j in range(2):
            b.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=19,
                   weight="bold", color="white" if cm[i,j] > cm.max()*.6 else "#24415F")
    b.set_xticks([0, 1], ["0으로 판정", "양수로 판정"])
    b.set_yticks([0, 1], ["실제 SFR=0", "실제 SFR>0"])
    b.set_title("고정 qG(d) + 전체 피처의 혼동행렬", loc="left", pad=17, fontsize=13, weight="bold", color="#24415F")
    b.tick_params(length=0, pad=9)
    for spine in b.spines.values(): spine.set_visible(False)
    delta = paired["learned_vs_fixed"]["accuracy_delta"]*100
    lo, hi = [100*x for x in paired["learned_vs_fixed"]["paired_row_bootstrap_95CI"]]
    fig.text(.075, .14, f"학습 확률 모델 - 고정 확률 모델: 검증 정확도 차이 {delta:+.2f}%p · 쌍 부트스트랩 95% 구간 [{lo:+.2f}, {hi:+.2f}]%p", fontsize=11, color="#24415F")
    fig.text(.075, .085, "qG 함수는 고정되지만, 최종 확률은 다른 피처로 보완됩니다. qG의 영향력을 일정하게 강제하는 방식은 아닙니다.", fontsize=10, color="#526175")
    fig.text(.075, .035, "학습용 보조 확률은 내부 5-fold로 자신을 제외하고 계산했습니다. 기존 검증 자료를 재사용한 탐색 결과입니다.", fontsize=9, color="#768398")
    for ext in ["png", "svg"]:
        fig.savefig(output/f"probability_plus_features_comparison.{ext}", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[2]/
                        "Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("density_probability_with_features_results"))
    args = parser.parse_args(); out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with args.data.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f); columns = reader.fieldnames; rows = list(reader)
    features = [c for c in columns if c not in ["subhalo_id", "sfr_linear"]]
    X = np.array([[float(r[c]) for c in features] for r in rows])
    d_index = features.index("log10_overdensity_plus_1"); d = X[:, d_index]
    sfr = np.array([float(r["sfr_linear"]) for r in rows])
    tr, va = train_test_split(np.arange(len(sfr)), test_size=.2, random_state=42,
                              stratify=(sfr > 0).astype(int))
    result = fit_partition(X, d, sfr, tr, va)
    mt, mv = result["train_indices"], result["test_indices"]
    assert len(mt) == 27517 and len(mv) == 6948
    # Check that adding fixed bounds preserved the earlier frozen-function model.
    old_dir = Path(__file__).with_name("fixed_density_zero_classifier_results")
    prior = FixedDensityZeroClassifier.load(old_dir/"model.npz")
    assert np.allclose(prior.predict_zero_probability(d[mv]), result["fixed_q"], rtol=1e-9)
    frozen = result["frozen_density_function"]
    refit_in_middle = FixedDensityZeroClassifier.fit(d[mt], sfr[mt], bounds=(frozen.lower, frozen.upper))
    assert np.allclose(refit_in_middle.predict_zero_probability(d[mv[:100]]), result["fixed_q"][:100], rtol=1e-9)
    validation = {key: classification_metrics(sfr[mv], result["predictions"][key], THRESHOLD) for key in METHODS}
    paired = {
        "learned_vs_fixed": paired_accuracy_comparison(sfr[mv], result["predictions"]["fixed_q_plus_X"], result["predictions"]["learned_q_plus_X"]),
        "fixed_vs_X_only": paired_accuracy_comparison(sfr[mv], result["predictions"]["X_only_reference"], result["predictions"]["fixed_q_plus_X"]),
    }
    for key in METHODS:
        bundle = {"method": key, "features": features, "density_column": d_index,
                  "lower_exclusive": frozen.lower, "upper_inclusive": frozen.upper,
                  "threshold": THRESHOLD, "classifier": result["final_classifiers"][key],
                  "density_function": (replace(frozen) if key == "fixed_q_plus_X" else
                                       result["learned_density_function"] if key == "learned_q_plus_X" else None)}
        joblib.dump(bundle, out/f"{key}.joblib")
        restored = joblib.load(out/f"{key}.joblib")
        assert np.allclose(predict_bundle(restored, X[mv[:100]]), result["predictions"][key][:100])
    print(json.dumps({"stage": "validation", "metrics": validation, "paired": paired,
                      "elapsed_seconds": time.time()-start}), flush=True)
    oof_indices = []; oof = {key: [] for key in METHODS}; folds = []
    for fold, (ft, fe) in enumerate(KFold(n_splits=5, shuffle=True, random_state=42).split(tr), 1):
        fr = fit_partition(X, d, sfr, tr[ft], tr[fe])
        ix = fr["test_indices"]; oof_indices.extend(ix.tolist())
        for key in METHODS: oof[key].extend(fr["predictions"][key].tolist())
        fm = {key: classification_metrics(sfr[ix], fr["predictions"][key], THRESHOLD) for key in METHODS}
        folds.append({"fold": fold, "metrics": fm})
        print(json.dumps({"stage": "CV", "fold": fold,
                          "accuracy": {key: fm[key]["accuracy"] for key in METHODS},
                          "elapsed_seconds": time.time()-start}), flush=True)
    oof_indices = np.asarray(oof_indices)
    cv = {key: classification_metrics(sfr[oof_indices], np.asarray(oof[key]), THRESHOLD) for key in METHODS}
    report = {
        "scope": "Middle-density 0/positive classification; no regression",
        "combination": "Final classifier G receives [original 13 features, density-derived probability]",
        "fixed_model": "p0(X) = G_fixed(X, q_G(d)); q_G is frozen Gaussian Bayes",
        "learned_model": "p0(X) = G_learned(X, q_L(d)); q_L is a density-only HGB without Gaussian shape constraint",
        "training_probability_features": "5-fold cross-fitted within the outer training cohort for both models",
        "features": features, "parameters_for_density_learner_and_G": HGB,
        "threshold": THRESHOLD, "lower_exclusive": frozen.lower, "upper_inclusive": frozen.upper,
        "training_middle": len(mt), "validation_middle": len(mv),
        "validation": validation, "train_CV_pooled": cv, "train_CV_folds": folds,
        "paired_validation_comparisons": paired,
        "iterations": {key: int(model.n_iter_) for key, model in result["final_classifiers"].items()},
        "checks": ["Frozen Gaussian probability reproduces previously saved function",
                   "Fixed-bound fitting matches default fitting in the same cohort",
                   "Each final-classifier training probability is predicted without that row's label in the density fit",
                   "Both main final classifiers receive the same 13 raw features plus one probability feature",
                   "Saved model bundles reproduce final probabilities"],
        "limitations": ["Fixed here means the q_G function is frozen; G can change its influence or ignore it",
                        "q_G(d) is deterministic in density, which is also a raw input, so it adds an inductive representation but no new observation",
                        "Density learners are fit before final G; this is not joint end-to-end optimization",
                        "Outer cohort boundaries depend only on outer training data and are shared by inner probability models",
                        "Original validation set reused after exploration; not a fresh final test",
                        "Paired row bootstrap conditions on fitted models and assumes iid galaxies"],
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "scikit-learn", "joblib"]},
        "elapsed_seconds": time.time()-start,
    }
    (out/"metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out/"comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["method", "accuracy", "zero_precision", "zero_recall", "zero_F1", "ROC_AUC", "log_loss", "SFR_gt100_pred_zero", "CV_accuracy"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); writer.writeheader()
        for key in METHODS: writer.writerow({"method": key, **validation[key], "CV_accuracy": cv[key]["accuracy"]})
    with (out/"validation_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["csv_row_index", "density", "sfr", "actual_is_zero", "fixed_density_q", "learned_density_q"]+
                        [f"{key}_final_q_zero" for key in METHODS])
        for j, idx in enumerate(mv):
            writer.writerow([int(idx), d[idx], sfr[idx], int(sfr[idx] == 0), result["fixed_q"][j], result["learned_q"][j]]+
                            [result["predictions"][key][j] for key in METHODS])
    np.savez_compressed(out/"training_probability_features.npz", row_indices=mt,
                        fixed_q_crossfit=result["training_probability_features"]["fixed"],
                        learned_q_crossfit=result["training_probability_features"]["learned"])
    make_plot(validation, cv, paired, out)
    print(json.dumps({"validation": validation, "CV": cv, "paired": paired}, indent=2), flush=True)


if __name__ == "__main__":
    main()

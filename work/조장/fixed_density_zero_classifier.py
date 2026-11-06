"""SFR=0 classifier with a frozen, density-only Gaussian Bayes probability.

Scope: the middle-density interval derived from training SFR=0 galaxies.
f(d | SFR=0) is a truncated Gaussian; f(d | SFR>0) is a normalized KDE.
q(d) = pi0*f0(d) / (pi0*f0(d) + (1-pi0)*f_positive(d)).
The default decision is q(d) >= 0.5. No regression or other features are used.

Dependencies: numpy, scipy, scikit-learn, matplotlib. Optional dependency path:
SFR_ANALYSIS_DEPENDENCIES. Run this file to fit, export, and evaluate the model.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse
import csv
from dataclasses import dataclass
from functools import cached_property
import importlib.metadata
import json
from pathlib import Path
import sys

if os.environ.get("SFR_ANALYSIS_DEPENDENCIES"):
    sys.path.insert(0, os.environ["SFR_ANALYSIS_DEPENDENCIES"])

import numpy as np
from scipy import optimize, special, stats


@dataclass(frozen=True)
class FixedDensityZeroClassifier:
    """Predict only on lower < density <= upper; True means predicted SFR=0."""

    lower: float
    upper: float
    zero_prior: float
    normal_loc: float
    normal_scale: float
    positive_densities: np.ndarray
    kde_factor: float
    threshold: float = 0.5

    @classmethod
    def fit(cls, density, sfr, threshold=0.5, *, bounds=None):
        density = np.asarray(density, dtype=float)
        sfr = np.asarray(sfr, dtype=float)
        if density.ndim != 1 or density.shape != sfr.shape:
            raise ValueError("density and sfr must be matching one-dimensional arrays")
        if not np.isfinite(density).all() or not np.isfinite(sfr).all() or np.any(sfr < 0):
            raise ValueError("Training values must be finite; SFR must be nonnegative")
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between zero and one")
        zero_density = density[sfr == 0]
        if len(zero_density) < 3:
            raise ValueError("At least three zero-SFR training galaxies are required")
        if bounds is None:
            lower, upper = zero_density.mean() + np.array([-2, 2]) * zero_density.std(ddof=0)
        else:
            lower, upper = map(float, bounds)
            if not np.isfinite([lower, upper]).all() or lower >= upper:
                raise ValueError("bounds must be finite and ordered lower < upper")
        middle = (density > lower) & (density <= upper)
        z = density[middle & (sfr == 0)]
        p = density[middle & (sfr > 0)]
        if len(z) < 3 or len(p) < 3 or z.std() == 0 or p.std() == 0:
            raise ValueError("Both middle-segment classes need varying density samples")

        def nll(theta):
            mu, log_sd = theta
            sd = np.exp(log_sd)
            return -stats.truncnorm.logpdf(
                z, (lower-mu)/sd, (upper-mu)/sd, loc=mu, scale=sd
            ).sum()

        fitted = optimize.minimize(
            nll, [z.mean(), np.log(z.std())], method="L-BFGS-B",
            bounds=[(None, None), (-10, 5)],
        )
        if not fitted.success:
            raise RuntimeError(f"Truncated-normal fitting failed: {fitted.message}")
        kde = stats.gaussian_kde(p, bw_method="scott")
        return cls(float(lower), float(upper), float(len(z)/(len(z)+len(p))),
                   float(fitted.x[0]), float(np.exp(fitted.x[1])),
                   p.copy(), float(kde.factor), float(threshold))

    def middle_mask(self, density):
        d = np.asarray(density, dtype=float)
        return (d > self.lower) & (d <= self.upper)

    @cached_property
    def _positive_kde(self):
        return stats.gaussian_kde(self.positive_densities, bw_method=self.kde_factor)

    @cached_property
    def _positive_normalization(self):
        return self._positive_kde.integrate_box_1d(self.lower, self.upper)

    def predict_zero_probability(self, density):
        d = np.atleast_1d(np.asarray(density, dtype=float))
        if d.ndim != 1 or not np.isfinite(d).all() or not self.middle_mask(d).all():
            raise ValueError("This model accepts finite densities in its middle interval only")
        if len(d) == 0:
            return np.empty(0)
        log_f0 = stats.truncnorm.logpdf(
            d, (self.lower-self.normal_loc)/self.normal_scale,
            (self.upper-self.normal_loc)/self.normal_scale,
            loc=self.normal_loc, scale=self.normal_scale,
        )
        fp = np.maximum(self._positive_kde(d)/self._positive_normalization,
                        np.finfo(float).tiny)
        q = special.expit(np.log(self.zero_prior/(1-self.zero_prior)) + log_f0 - np.log(fp))
        if not np.isfinite(q).all():
            raise RuntimeError("Nonfinite probabilities")
        return q

    def predict_is_zero(self, density, threshold=None):
        t = self.threshold if threshold is None else threshold
        if not 0 < t < 1:
            raise ValueError("threshold must be between zero and one")
        return self.predict_zero_probability(density) >= t

    def save(self, path):
        np.savez_compressed(
            path, lower=self.lower, upper=self.upper, zero_prior=self.zero_prior,
            normal_loc=self.normal_loc, normal_scale=self.normal_scale,
            positive_densities=self.positive_densities,
            kde_factor=self.kde_factor, threshold=self.threshold,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as stored:
            return cls(**{name: stored[name].copy() if name == "positive_densities"
                          else float(stored[name]) for name in cls.__dataclass_fields__})


def classification_metrics(sfr, q, threshold=0.5):
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 balanced_accuracy_score, brier_score_loss,
                                 f1_score, log_loss, precision_score,
                                 recall_score, roc_auc_score)
    is_zero = np.asarray(sfr) == 0
    pred = np.asarray(q) >= threshold
    tp = int(np.sum(is_zero & pred))
    fn = int(np.sum(is_zero & ~pred))
    fp = int(np.sum(~is_zero & pred))
    tn = int(np.sum(~is_zero & ~pred))
    return {
        "threshold": threshold, "n": len(sfr), "actual_zero": int(is_zero.sum()),
        "predicted_zero": int(pred.sum()), "accuracy": float(accuracy_score(is_zero, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(is_zero, pred)),
        "zero_precision": float(precision_score(is_zero, pred, zero_division=0)),
        "zero_recall": float(recall_score(is_zero, pred, zero_division=0)),
        "zero_F1": float(f1_score(is_zero, pred, zero_division=0)),
        "positive_recall": float(recall_score(~is_zero, ~pred, zero_division=0)),
        "ROC_AUC": float(roc_auc_score(is_zero, q)),
        "average_precision_zero": float(average_precision_score(is_zero, q)),
        "Brier": float(brier_score_loss(is_zero, q)), "log_loss": float(log_loss(is_zero, q)),
        "confusion": {"true_zero_pred_zero": tp, "true_zero_pred_positive": fn,
                      "true_positive_pred_zero": fp, "true_positive_pred_positive": tn},
        "SFR_gt100_actual": int(np.sum(sfr > 100)),
        "SFR_gt100_pred_zero": int(np.sum((sfr > 100) & pred)),
        "max_SFR_in_false_zero": float(np.max(sfr[~is_zero & pred])) if fp else None,
    }


def make_plot(model, sfr, density, q, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output/".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font = Path("C:/Windows/Fonts/malgun.ttf")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": "Malgun Gothic" if font.exists() else "DejaVu Sans",
                         "axes.unicode_minus": False, "svg.fonttype": "none", "font.size": 11})
    metrics = classification_metrics(sfr, q, model.threshold)
    cf = metrics["confusion"]
    cm = np.array([[cf["true_zero_pred_zero"], cf["true_zero_pred_positive"]],
                   [cf["true_positive_pred_zero"], cf["true_positive_pred_positive"]]])
    fig, (ax, matrix) = plt.subplots(1, 2, figsize=(13, 6.5), facecolor="#FAFBFD",
                                    gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(left=.075, right=.97, top=.73, bottom=.23, wspace=.35)
    fig.text(.075, .92, "밀도 기반 SFR=0 판정 모델", fontsize=23, weight="bold", color="#18273B")
    fig.text(.075, .85, f"중간 밀도 구간 ({model.lower:.3f}, {model.upper:.3f}] · 검증 {len(sfr):,}개 · 판정 기준 q(d) ≥ {model.threshold:g}",
             fontsize=12, color="#526175")
    grid = np.linspace(model.lower+1e-8, model.upper, 400)
    ax.plot(grid, model.predict_zero_probability(grid), color="#CC6439", lw=2.5, label="고정한 정규분포 기반 확률")
    edges = np.linspace(model.lower, model.upper, 19)
    centers, rates, errors = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (density > lo) & (density <= hi)
        n = int(mask.sum())
        if not n:
            continue
        p = float(np.mean(sfr[mask] == 0)); z = 1.96
        center = (p+z*z/(2*n))/(1+z*z/n)
        half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        centers.append((lo+hi)/2); rates.append(p)
        errors.append([p-center+half, center+half-p])
    ax.errorbar(centers, rates, yerr=np.array(errors).T, fmt="o", ms=4, capsize=2,
                color="#286BB5", label="검증의 실제 0 비율 (95% 구간)")
    ax.axhline(model.threshold, color="#64758A", ls="--", lw=1.2, label="0으로 판정하는 확률 문턱")
    ax.set(xlim=(model.lower-.025, model.upper+.025), ylim=(-.03, 1.04),
           xlabel="주변 밀도 log10(1 + δ)", ylabel="SFR=0 확률 / 비율")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_facecolor("#FAFBFD"); ax.grid(axis="y", color="#DCE3ED", lw=.7)
    ax.spines[["top", "right"]].set_visible(False)
    matrix.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max()*1.1)
    for i in range(2):
        for j in range(2):
            matrix.text(j, i, f"{cm[i,j]:,}", ha="center", va="center", fontsize=19,
                        weight="bold", color="white" if cm[i,j] > cm.max()*.6 else "#24415F")
    matrix.set_xticks([0, 1], ["0으로 판정", "양수로 판정"])
    matrix.set_yticks([0, 1], ["실제 SFR=0", "실제 SFR>0"])
    matrix.set_title("혼동행렬", loc="left", pad=17, fontsize=14, weight="bold", color="#24415F")
    matrix.tick_params(length=0, pad=9)
    for spine in matrix.spines.values(): spine.set_visible(False)
    fig.text(.075, .125, f"정확도 {metrics['accuracy']:.1%}  |  0 판정 정밀도 {metrics['zero_precision']:.1%}  |  0 재현율 {metrics['zero_recall']:.1%}  |  ROC-AUC {metrics['ROC_AUC']:.3f}",
             fontsize=12, weight="bold", color="#24415F")
    fig.text(.075, .065, "학습 데이터로 분포와 확률을 추정한 뒤 고정했습니다. 이번 모델은 밀도만 사용하며, SFR의 크기는 예측하지 않습니다.",
             fontsize=10, color="#526175")
    for ext in ["png", "svg"]:
        fig.savefig(output/f"zero_classifier.{ext}", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    from sklearn.model_selection import KFold, train_test_split
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[2]/
                        "Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("fixed_density_zero_classifier_results"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with args.data.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    density = np.array([float(r["log10_overdensity_plus_1"]) for r in rows])
    sfr = np.array([float(r["sfr_linear"]) for r in rows])
    tr, va = train_test_split(np.arange(len(sfr)), test_size=.2, random_state=42,
                              stratify=(sfr > 0).astype(int))
    model = FixedDensityZeroClassifier.fit(density[tr], sfr[tr], args.threshold)
    middle_train = tr[model.middle_mask(density[tr])]
    middle_val = va[model.middle_mask(density[va])]
    q = model.predict_zero_probability(density[middle_val])
    model.save(args.output_dir/"model.npz")
    loaded = FixedDensityZeroClassifier.load(args.output_dir/"model.npz")
    assert np.allclose(loaded.predict_zero_probability(density[middle_val[:100]]), q[:100], rtol=1e-12)
    assert len(tr) == 166749 and len(middle_train) == 27517 and len(middle_val) == 6948
    assert int(np.sum(sfr[middle_val] == 0)) == 3420
    with (args.output_dir/"validation_predictions.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["csv_row_index", "density", "sfr", "actual_is_zero", "probability_zero", "predicted_state"])
        for idx, qi in zip(middle_val, q):
            w.writerow([int(idx), density[idx], sfr[idx], int(sfr[idx] == 0), qi,
                        "SFR=0" if qi >= model.threshold else "SFR>0"])

    thresholds = sorted(set([args.threshold, .5, .8, .9, .95]))
    diagnostics = [classification_metrics(sfr[middle_val], q, t) for t in thresholds]
    folds = []; oof_indices = []; oof_q = []
    for fold, (ft, fe) in enumerate(KFold(n_splits=5, shuffle=True, random_state=42).split(tr), 1):
        fm = FixedDensityZeroClassifier.fit(density[tr[ft]], sfr[tr[ft]], args.threshold)
        test = tr[fe][fm.middle_mask(density[tr[fe]])]
        fq = fm.predict_zero_probability(density[test])
        folds.append({"fold": fold, "lower": fm.lower, "upper": fm.upper,
                      "metrics": classification_metrics(sfr[test], fq, args.threshold)})
        oof_indices.extend(test.tolist()); oof_q.extend(fq.tolist())
        print(json.dumps({"fold": fold, "zero_F1": folds[-1]["metrics"]["zero_F1"]}), flush=True)
    oof_indices = np.asarray(oof_indices); oof_q = np.asarray(oof_q)
    report = {
        "scope": "Middle-density galaxies only; no regression or other features",
        "zero_definition": "Provided sfr_linear equals exactly zero; class 1 means SFR=0",
        "probability": "Bayes posterior: truncated Gaussian zero density + normalized positive KDE + observed class prior",
        "normal_loc": model.normal_loc, "normal_scale": model.normal_scale,
        "positive_kde_factor": model.kde_factor, "zero_prior": model.zero_prior,
        "lower_exclusive": model.lower, "upper_inclusive": model.upper,
        "train_total": len(tr), "train_middle": len(middle_train),
        "validation_default": classification_metrics(sfr[middle_val], q, args.threshold),
        "validation_threshold_diagnostics": diagnostics,
        "train_CV_pooled": classification_metrics(sfr[oof_indices], oof_q, args.threshold),
        "train_CV_folds": folds,
        "majority_positive_baseline_validation_accuracy": float(np.mean(sfr[middle_val] > 0)),
        "checks": ["Existing seed=42 stratified split and middle segment counts reproduced",
                   "Saved/reloaded model probabilities match", "No held-out target used for fitting"],
        "threshold_policy": "0.5 default is set before evaluation; 0.8/0.9/0.95 are descriptive diagnostics, not selected on validation",
        "limitations": ["This validation set has already been inspected in earlier exploratory experiments",
                        "Within each CV training fold both cutoffs and probability function are refitted",
                        "Gaussian fit and positive KDE are approximations; no probability calibration or threshold optimization",
                        "The model does not classify outside its middle-density interval"],
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "scikit-learn", "matplotlib"]},
    }
    (args.output_dir/"metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output_dir/"threshold_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        keys = ["threshold", "accuracy", "zero_precision", "zero_recall", "zero_F1", "predicted_zero", "SFR_gt100_pred_zero"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(diagnostics)
    make_plot(model, sfr[middle_val], density[middle_val], q, args.output_dir)
    print(json.dumps({k: report[k] for k in ["validation_default", "validation_threshold_diagnostics", "train_CV_pooled"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

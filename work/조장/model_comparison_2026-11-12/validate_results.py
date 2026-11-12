"""Recompute benchmark scores independently from saved out-of-fold predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(folder):
    folder = Path(folder)
    root = Path(__file__).resolve().parents[3]
    data = root / "Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv"
    df = pd.read_csv(data)
    saved = np.load(folder / "oof_predictions.npz", allow_pickle=False)
    rows = pd.read_csv(folder / "fold_results.csv")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert not manifest.get("smoke", False), "Smoke results cannot validate the full experiment"
    expected_train, expected_val = train_test_split(
        df.index.to_numpy(), test_size=0.2, random_state=42,
        stratify=(df.sfr_linear > 0).astype(int),
    )
    train = saved["train_indices"]
    np.testing.assert_array_equal(train, expected_train)
    assert not np.intersect1d(train, expected_val).size
    y = saved["y_true_linear"]
    np.testing.assert_array_equal(y, df.sfr_linear.to_numpy()[train])
    z = np.log10(y + 1)
    np.testing.assert_allclose(saved["y_true_log"], z, rtol=0, atol=1e-14)
    folds = saved["fold_id"]
    for fold, (fit, test) in enumerate(KFold(5, shuffle=True, random_state=42).split(train)):
        assert not np.intersect1d(fit, test).size
        np.testing.assert_array_equal(np.flatnonzero(folds == fold), test)
    expected_models = {
        "hgb_single_fixed400", "hgb_two_stage_fixed400",
        "lightgbm_single_fixed400", "lightgbm_two_stage_fixed400",
        "catboost_single_fixed400", "catboost_two_stage_fixed400",
        "hgb_two_stage_reference",
    }
    assert set(rows.model) == expected_models
    assert len(rows) == 35 and not rows.duplicated(["model", "fold"]).any()
    checks = 0
    max_error = 0.0
    means = {}
    aggregated = pd.read_csv(folder / "summary.csv").set_index("model")
    for model in sorted(expected_models):
        pred = saved[f"yhat_linear_{model}"]
        assert pred.shape == y.shape and np.isfinite(pred).all() and (pred >= 0).all()
        residual2 = (np.log10(pred + 1) - z) ** 2
        values = []
        for fold in range(5):
            mask = folds == fold
            record = rows.loc[(rows.model == model) & (rows.fold == fold)].iloc[0]
            assert int(record.n_eval) == int(mask.sum())
            metrics = {
                "rmse_log": np.sqrt(residual2[mask].mean()),
                "rmse_log_zero": np.sqrt(residual2[mask & (y == 0)].mean()),
                "rmse_log_positive": np.sqrt(residual2[mask & (y > 0)].mean()),
                "rmse_log_high_gt100": np.sqrt(residual2[mask & (y > 100)].mean()),
                "zero_recall": np.mean(pred[mask & (y == 0)] == 0),
                "zero_precision": np.mean(y[mask & (pred == 0)] == 0),
                "positive_predicted_zero_count": np.sum(mask & (y > 0) & (pred == 0)),
            }
            for name, value in metrics.items():
                observed = float(record[name])
                assert np.isclose(value, observed, rtol=1e-11, atol=1e-12, equal_nan=True), (model, fold, name, value, observed)
                if np.isfinite(value):
                    max_error = max(max_error, abs(float(value) - observed))
                checks += 1
            assert record.training_seconds > 0 and record.prediction_seconds >= 0
            if "fixed400" in model:
                assert int(record.regressor_iterations) == 400
                if "two_stage" in model:
                    assert int(record.classifier_iterations) == 400
            values.append(float(metrics["rmse_log"]))
        means[model] = float(np.mean(values))
        for name, value in {
            "mean_rmse_log": means[model],
            "sd_rmse_log": float(np.std(values, ddof=1)),
            "oof_rmse_log": float(np.sqrt(residual2.mean())),
        }.items():
            assert np.isclose(float(aggregated.loc[model, name]), value, rtol=1e-11, atol=1e-12), (model, name)
            checks += 1
    pairs = pd.read_csv(folder / "paired_differences.csv")
    for record in pairs.itertuples(index=False):
        candidate = rows.loc[(rows.model == record.candidate) & (rows.fold == record.fold), "rmse_log"].iloc[0]
        baseline = rows.loc[(rows.model == record.baseline) & (rows.fold == record.fold), "rmse_log"].iloc[0]
        assert np.isclose(record.delta_rmse_log, candidate - baseline, rtol=1e-11, atol=1e-12)
        checks += 1
    original = pd.read_csv(root / "work/조장/review_revision_2026-11-11/diagnostic_cv.csv")
    original_rmse = original.loc[original.model == "E1H1_34", "RMSE"].mean()
    reference_error = abs(means["hgb_two_stage_reference"] - original_rmse)
    assert reference_error < 1e-8, ("Reference reproduction drift", reference_error)
    fingerprint = manifest["fingerprint"]
    assert fingerprint["data_sha256"] == digest(data)
    assert fingerprint["script_sha256"] == digest(Path(__file__).with_name("benchmark.py"))
    assert fingerprint["protocol_sha256"] == digest(Path(__file__).with_name("protocol.json"))
    assert fingerprint["notebook_sha256"] == digest(root / "DataHub_701104_final.ipynb")
    audit = json.loads(Path(__file__).with_name("feature_audit.json").read_text(encoding="utf-8"))
    assert fingerprint["feature_frame_hash"] == audit["feature_frame_hash"]
    protected = json.loads(Path(__file__).with_name("source_baseline.json").read_text(encoding="utf-8"))
    for path, expected in protected["preserved_source_sha256"].items():
        assert digest(root / path) == expected, ("Protected source changed", path)
    result = {
        "passed": True, "n_training": len(train), "n_untouched_validation": len(expected_val),
        "models": len(expected_models), "fold_records": len(rows), "independent_metric_checks": checks,
        "maximum_metric_absolute_error": max_error, "reference_mean_absolute_error": reference_error,
        "preserved_sources": True,
        "mean_fold_log_rmse": means,
        "artifact_sha256": {p.name: digest(p) for p in sorted(folder.iterdir()) if p.suffix in {".csv", ".npz", ".json"} and p.name != "validation.json"},
    }
    (folder / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    validate(parser.parse_args().folder)

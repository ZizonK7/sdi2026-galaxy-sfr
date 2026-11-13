import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from threadpoolctl import threadpool_limits
from leader_model import TwoStageLogSFR, HGB_FIXED400, build_features, predict_sfr
from three_regime import ThreeRegimeLogSFR, combine


def metrics(y, pred):
    actual_zero = y == 0
    zero = pred == 0
    delta = np.log10(1 + pred) - np.log10(1 + y)
    result = dict(log_rmse=float(np.sqrt(np.mean(delta**2))),
                  zero_log_rmse=float(np.sqrt(np.mean(delta[actual_zero]**2))),
                  zero_recall=float(zero[actual_zero].mean()),
                  zero_precision=float(actual_zero[zero].mean()) if zero.any() else None,
                  zero_outputs=int(zero.sum()), false_zeros=int((zero & ~actual_zero).sum()))
    for threshold in (10, 30, 100):
        mask = y > threshold
        result.update({f'tail{threshold}_n': int(mask.sum()),
                       f'tail{threshold}_log_rmse': float(np.sqrt(np.mean(delta[mask]**2))),
                       f'tail{threshold}_log_bias': float(delta[mask].mean()),
                       f'tail{threshold}_linear_rmse': float(np.sqrt(np.mean((pred[mask] - y[mask])**2)))})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    payload = args.data.read_bytes()
    data_sha = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
    assert data_sha == '06479cedd323253f08af9983c73f395cc02d3774'
    df = pd.read_csv(args.data)
    ids, _ = train_test_split(np.arange(len(df)), test_size=.2, random_state=42,
                             stratify=(df.sfr_linear > 0).astype(int))
    with threadpool_limits(limits=4):
        print('Building fixed sorted catalogue features...', flush=True)
        all_features = build_features(df)
        X = all_features.iloc[ids].reset_index(drop=True)
        y = df.sfr_linear.to_numpy()[ids]
        names = ['leader_two_stage', 'three_T10', 'three_T30', 'three_T100']
        predictions = {name: np.full(len(y), np.nan) for name in names}
        fold_ids = np.zeros(len(y), dtype=int)
        components = {}
        folds = []
        coverage = np.zeros(len(y), dtype=int)
        for fold, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=42).split(X), 1):
            coverage[va] += 1
            fold_ids[va] = fold
            for name in names:
                model = (TwoStageLogSFR(HGB_FIXED400, X.columns, X.columns, zero_threshold=.9)
                         if name == 'leader_two_stage' else ThreeRegimeLogSFR(int(name.split('T')[1])))
                t = time.monotonic()
                model.fit(X.iloc[tr], y[tr])
                seconds = time.monotonic() - t
                pred = model.predict(X.iloc[va])
                assert np.isfinite(pred).all() and (pred >= 0).all()
                predictions[name][va] = pred
                if name.startswith('three_'):
                    p, mid, tail = model.predict_components(X.iloc[va])
                    values = np.column_stack([p, mid, tail])
                    components.setdefault(name, np.full((len(y), 5), np.nan))[va] = values
                    np.testing.assert_array_equal(pred[p[:, 0] >= .9], 0.)
                    assert model.clf_.n_iter_ == model.mid_.n_iter_ == model.tail_.n_iter_ == 400
                    counts = model.training_counts_
                else:
                    counts = [int((y[tr] == 0).sum()), int((y[tr] > 0).sum())]
                row = dict(model=name, fold=fold, fit_seconds=seconds, training_counts=json.dumps(counts), **metrics(y[va], pred))
                folds.append(row)
                pd.DataFrame(folds).to_csv(root / 'fold_results.csv', index=False)
                print(f"fold {fold}/5 {name}: RMSE={row['log_rmse']:.9f}, zero={row['zero_log_rmse']:.6f}, tail>100={row['tail100_log_rmse']:.6f}, fit={seconds:.1f}s", flush=True)
        assert (coverage == 1).all()
        table = pd.DataFrame(folds)
        summaries = []
        for name in names:
            assert np.isfinite(predictions[name]).all()
            rows = table[table.model == name]
            summaries.append(dict(model=name, mean_fold_rmse=float(rows.log_rmse.mean()),
                                  sd_fold_rmse=float(rows.log_rmse.std(ddof=1)),
                                  mean_fit_seconds=float(rows.fit_seconds.mean()),
                                  **metrics(y, predictions[name])))
        summary = pd.DataFrame(summaries)
        summary.to_csv(root / 'summary.csv', index=False)
        candidates = summary[summary.model != 'leader_two_stage']
        selected = candidates.sort_values(['mean_fold_rmse', 'model']).iloc[0].model
        threshold = int(selected.split('T')[1])
        np.savez_compressed(root / 'oof_predictions.npz', actual=y, source_row_index=ids,
                            fold=fold_ids, **predictions)
        np.savez_compressed(root / 'oof_components.npz', **components)
        # Audit prediction reconstruction from saved probabilities and expert log outputs.
        for name, values in components.items():
            rebuilt = np.maximum(10.**combine(values[:, :3], values[:, 3], values[:, 4]) - 1., 0.)
            np.testing.assert_array_equal(rebuilt, predictions[name])
        baseline = table[table.model == names[0]].set_index('fold')
        paired = []
        for name in names[1:]:
            rows = table[table.model == name].set_index('fold')
            for fold in range(1, 6):
                paired.append(dict(model=name, fold=fold, **{
                    metric + '_delta': float(rows.loc[fold, metric] - baseline.loc[fold, metric])
                    for metric in ['log_rmse', 'zero_log_rmse', 'tail10_log_rmse', 'tail30_log_rmse', 'tail100_log_rmse', 'zero_recall', 'zero_precision']}))
        pd.DataFrame(paired).to_csv(root / 'paired_differences.csv', index=False)
        print(f'Refitting best three-regime candidate: {selected}', flush=True)
        fitted = ThreeRegimeLogSFR(threshold).fit(X, y)
        joblib.dump(fitted, root / 'best_three_regime.joblib', compress=3)
        loaded = joblib.load(root / 'best_three_regime.joblib')
        test = X.iloc[:1000]
        np.testing.assert_array_equal(loaded.predict(test), fitted.predict(test))
        reference = df.drop(columns='sfr_linear')
        from_raw = predict_sfr(reference.iloc[ids[:1000]], loaded, reference)
        np.testing.assert_allclose(from_raw, loaded.predict(test), rtol=1e-6, atol=1e-9)
        base = summary.iloc[0]
        best = summary[summary.model == selected].iloc[0]
        metadata = dict(selected_candidate=selected, tail_threshold=threshold,
                        zero_threshold=.9, params=HGB_FIXED400, features=list(X.columns),
                        source_commit='21d02b5d0c829aff67cafd89da62c3386ac7864f',
                        data_git_blob_sha=data_sha, training_counts=fitted.training_counts_,
                        train_rows=len(y), holdout_evaluated=False, production_model_changed=False,
                        baseline_mean_fold_rmse=float(base.mean_fold_rmse), candidate_mean_fold_rmse=float(best.mean_fold_rmse),
                        baseline_pooled_oof_rmse=float(base.log_rmse), candidate_pooled_oof_rmse=float(best.log_rmse),
                        overall_improved_folds=sum(r['model'] == selected and r['log_rmse_delta'] < 0 for r in paired),
                        checks={'oof_coverage_exactly_once': True, 'oof_reconstructed': True,
                                'saved_model_reload': True, 'raw_prediction_interface': True},
                        versions={k: importlib.metadata.version(k) for k in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib']},
                        selection_note='Best among three candidates on this CV; exploratory, not an independent test.')
        (root / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
        print(summary.to_string(index=False), flush=True)
        print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Rebuild the team-lead final model on the original 80% training split."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from leader_model import (TwoStageLogSFR, HGB_FIXED400, FINAL_FEATURES,
                          build_features, predict_sfr, log_rmse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    payload = args.data.read_bytes()
    data_sha = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
    assert data_sha == '06479cedd323253f08af9983c73f395cc02d3774', 'Unexpected source data'
    df = pd.read_csv(args.data)
    y = df.sfr_linear
    tr, va = train_test_split(np.arange(len(df)), test_size=.2, random_state=42,
                             stratify=(y > 0).astype(int))
    with threadpool_limits(limits=4):
        print('Building original sorted catalogue features...', flush=True)
        X = build_features(df)
        assert X.shape == (len(df), 34)
        model = TwoStageLogSFR(HGB_FIXED400, X.columns, X.columns, zero_threshold=.9)
        t = time.monotonic()
        model.fit(X.iloc[tr], y.iloc[tr])
        elapsed = time.monotonic() - t
        print(f'Trained in {elapsed:.1f}s; evaluating original validation split...', flush=True)
        pred = model.predict(X.iloc[va])
        assert model.clf_.n_iter_ == model.reg_.n_iter_ == 400
        assert np.isfinite(pred).all() and (pred >= 0).all()
        p0 = 1 - model.predict_proba_positive(X.iloc[va])
        np.testing.assert_array_equal(pred[p0 >= .9], 0.)
        model_path = root / 'team_lead_hgb400.joblib'
        joblib.dump(model, model_path, compress=3)
        loaded = joblib.load(model_path)
        np.testing.assert_array_equal(loaded.predict(X.iloc[va]), pred)
        # Verify the standalone raw-table interface and row order on a fixed reference.
        reference = df.drop(columns='sfr_linear')
        sample = reference.iloc[va[:1000]]
        direct = predict_sfr(sample, loaded, reference)
        np.testing.assert_allclose(direct, pred[:1000], rtol=1e-6, atol=1e-9)
        reverse = predict_sfr(sample.iloc[::-1], loaded, reference)
        np.testing.assert_allclose(reverse[::-1], direct, rtol=1e-6, atol=1e-9)
    actual = y.iloc[va].to_numpy()
    zeros = pred == 0
    score = log_rmse(actual, pred, 1.)
    metadata = {
        'source_commit': '21d02b5d0c829aff67cafd89da62c3386ac7864f',
        'artifact_origin': 'Locally refitted from upstream source; no upstream fitted model was available',
        'model': 'TwoStageLogSFR', 'params': HGB_FIXED400, 'features': list(X.columns),
        'zero_threshold': .9, 'log_eps': 1., 'train_rows': len(tr), 'validation_rows': len(va),
        'training_seconds': elapsed, 'data_git_blob_sha': data_sha,
        'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'validation_log10_1p_rmse': score,
        'upstream_reported_validation_rmse_rounded': .095314,
        'matches_upstream_reported_6_decimals': round(score, 6) == .095314,
        'validation_linear_rmse': float(np.sqrt(np.mean((actual - pred)**2))),
        'validation_zero_outputs': int(zeros.sum()),
        'validation_false_zero_outputs': int((zeros & (actual > 0)).sum()),
        'validation_zero_precision': float((actual[zeros] == 0).mean()),
        'validation_zero_recall': float(zeros[actual == 0].mean()),
        'checks': {'model_reload_exact': True, 'both_estimators_400_iterations': True,
                   'zero_gate': True, 'raw_interface_matches': True, 'row_order': True},
        'versions': {name: importlib.metadata.version(name) for name in
                     ['numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl']},
        'evaluation_note': 'Original 20% validation score; not internal 5-fold OOF. No current-A evaluation or replacement.',
    }
    (root / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    np.savez_compressed(root / 'validation_predictions.npz', source_row_index=va,
                        actual=actual, prediction=pred, p_zero=p0)
    (root / 'requirements.txt').write_text(''.join(f'{k}=={v}\n' for k, v in metadata['versions'].items()))
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

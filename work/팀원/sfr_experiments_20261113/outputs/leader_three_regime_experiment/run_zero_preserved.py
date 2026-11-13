"""Exploratory follow-up: reuse OOF experts and restore the baseline binary gate."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from leader_model import HGB_FIXED400, build_features
from three_regime import combine
from zero_preserved import ZeroPreservedThreeRegime, redistribute
from run_experiment import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    meta = json.loads((root / 'metadata.json').read_text())
    name = meta['selected_candidate']
    data = np.load(root / 'oof_predictions.npz')
    y = data['actual']
    components = np.load(root / 'oof_components.npz')[name]
    df = pd.read_csv(args.data)
    ids = data['source_row_index']
    np.testing.assert_array_equal(y, df.sfr_linear.to_numpy()[ids])
    rows = []
    pred = np.full(len(y), np.nan)
    p0 = np.full(len(y), np.nan)
    with threadpool_limits(limits=4):
        X = build_features(df).iloc[ids].reset_index(drop=True)
        for fold, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=42).split(X), 1):
            assert (data['fold'][va] == fold).all()
            gate = HistGradientBoostingClassifier(**HGB_FIXED400).fit(X.iloc[tr], (y[tr] > 0).astype(int))
            positive = gate.predict_proba(X.iloc[va])[:, 1]
            p = redistribute(components[va, :3], positive)
            np.testing.assert_allclose(p.sum(axis=1), 1.)
            p0[va] = p[:, 0]
            pred[va] = np.maximum(10.**combine(p, components[va, 3], components[va, 4]) - 1., 0.)
            np.testing.assert_array_equal(pred[va][p[:, 0] >= .9], 0.)
            row = dict(model='zero_preserved_' + name, fold=fold, **metrics(y[va], pred[va]))
            rows.append(row)
            print(f"fold {fold}/5: {row['log_rmse']:.9f}, zero={row['zero_log_rmse']:.6f}, tail>100={row['tail100_log_rmse']:.6f}", flush=True)
        assert np.isfinite(pred).all()
        result = dict(model='zero_preserved_' + name, mean_fold_rmse=float(np.mean([r['log_rmse'] for r in rows])),
                      sd_fold_rmse=float(np.std([r['log_rmse'] for r in rows], ddof=1)), **metrics(y, pred))
        pd.DataFrame([result]).to_csv(root / 'zero_preserved_summary.csv', index=False)
        pd.DataFrame(rows).to_csv(root / 'zero_preserved_folds.csv', index=False)
        np.savez_compressed(root / 'zero_preserved_oof.npz', actual=y, fold=data['fold'],
                            source_row_index=ids, prediction=pred, p_zero=p0)
        three = joblib.load(root / 'best_three_regime.joblib')
        gate = HistGradientBoostingClassifier(**HGB_FIXED400).fit(X, (y > 0).astype(int))
        model = ZeroPreservedThreeRegime(three, gate)
        joblib.dump(model, root / 'zero_preserved_three_regime.joblib', compress=3)
        loaded = joblib.load(root / 'zero_preserved_three_regime.joblib')
        np.testing.assert_array_equal(loaded.predict(X.iloc[:1000]), model.predict(X.iloc[:1000]))
    result['exploratory_followup'] = True
    result['three_regime_threshold_selected_on_same_cv'] = meta['tail_threshold']
    result['production_model_changed'] = False
    result['holdout_evaluated'] = False
    (root / 'zero_preserved_metadata.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

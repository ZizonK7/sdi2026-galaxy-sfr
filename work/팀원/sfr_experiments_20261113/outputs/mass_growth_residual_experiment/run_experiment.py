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
from three_regime import ThreeRegimeLogSFR
from zero_preserved import ZeroPreservedThreeRegime
from growth_residual import MassGrowthResidual, ResidualFeatureModel, FEATURE
from metrics import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, default=Path(__file__).resolve().parent.parent / 'leader_three_regime_experiment')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    old = args.baseline_dir
    payload = args.data.read_bytes()
    sha = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
    assert sha == '06479cedd323253f08af9983c73f395cc02d3774'
    for name in ['leader_model.py', 'three_regime.py', 'zero_preserved.py']:
        assert (root / name).read_bytes() == (old / name).read_bytes()
    df = pd.read_csv(args.data)
    ids, _ = train_test_split(np.arange(len(df)), test_size=.2, random_state=42,
                             stratify=(df.sfr_linear > 0).astype(int))
    y = df.sfr_linear.to_numpy()[ids]
    before = np.load(old / 'oof_predictions.npz')
    protected = np.load(old / 'zero_preserved_oof.npz')
    for d in [before, protected]:
        np.testing.assert_array_equal(d['source_row_index'], ids)
        np.testing.assert_array_equal(d['actual'], y)
    names = ['leader_two_stage', 'three_T10', 'zero_preserved_three_T10']
    baseline = {names[0]: before[names[0]], names[1]: before[names[1]], names[2]: protected['prediction']}
    new = {name: np.full(len(y), np.nan) for name in names}
    curves, rows, diagnostics = [], [], []
    new_components = np.full((len(y), 5), np.nan)
    new_residual = np.full(len(y), np.nan)
    coverage = np.zeros(len(y), dtype=int)
    with threadpool_limits(limits=4):
        print('Building original 34 features...', flush=True)
        X = build_features(df).iloc[ids].reset_index(drop=True)
        for fold, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=42).split(X), 1):
            assert (before['fold'][va] == fold).all() and (protected['fold'][va] == fold).all()
            assert not np.intersect1d(tr, va).size
            coverage[va] += 1
            transformer = MassGrowthResidual(20).fit(X.iloc[tr])
            A, B = transformer.transform(X.iloc[tr]), transformer.transform(X.iloc[va])
            assert A.shape[1] == B.shape[1] == 35 and transformer.fit_rows_ == len(tr)
            new_residual[va] = B[FEATURE]
            curves.append(dict(fold=fold, fitted_rows=len(tr), nodes_mass=transformer.mass_nodes_.tolist(),
                               nodes_mean_growth=transformer.growth_nodes_.tolist(), bin_counts=transformer.counts_.tolist()))
            start = time.monotonic()
            leader = TwoStageLogSFR(HGB_FIXED400, A.columns, A.columns, zero_threshold=.9).fit(A, y[tr])
            leader_time = time.monotonic() - start
            start = time.monotonic()
            three = ThreeRegimeLogSFR(10).fit(A, y[tr])
            three_time = time.monotonic() - start
            models = {names[0]: leader, names[1]: three,
                      names[2]: ZeroPreservedThreeRegime(three, leader.clf_)}
            p, mid, tail = three.predict_components(B)
            new_components[va] = np.column_stack([p, mid, tail])
            for name, model in models.items():
                pred = model.predict(B)
                assert np.isfinite(pred).all() and (pred >= 0).all()
                new[name][va] = pred
                for phase, pr in [('before', baseline[name][va]), ('after', pred)]:
                    row = dict(model=name, phase=phase, fold=fold, **metrics(y[va], pr))
                    rows.append(row)
                print(f"fold {fold}/5 {name}: {rows[-2]['log_rmse']:.9f} -> {rows[-1]['log_rmse']:.9f}; tail100 {rows[-2]['tail100_log_rmse']:.6f} -> {rows[-1]['tail100_log_rmse']:.6f}", flush=True)
            pd.DataFrame(rows).to_csv(root / 'fold_results.csv', index=False)
            print(f'fit seconds: leader={leader_time:.1f}, three={three_time:.1f}', flush=True)
        assert (coverage == 1).all()
        table = pd.DataFrame(rows)
        summary = []
        for name in names:
            for phase, pr in [('before', baseline[name]), ('after', new[name])]:
                assert np.isfinite(pr).all()
                fold_rows = table[(table.model == name) & (table.phase == phase)]
                summary.append(dict(model=name, phase=phase, mean_fold_rmse=float(fold_rows.log_rmse.mean()),
                                    sd_fold_rmse=float(fold_rows.log_rmse.std(ddof=1)), **metrics(y, pr)))
        pd.DataFrame(summary).to_csv(root / 'summary.csv', index=False)
        paired = []
        for name in names:
            a = table[(table.model == name) & (table.phase == 'after')].set_index('fold')
            b = table[(table.model == name) & (table.phase == 'before')].set_index('fold')
            for fold in range(1, 6):
                paired.append(dict(model=name, fold=fold, **{metric + '_delta':float(a.loc[fold, metric] - b.loc[fold, metric])
                  for metric in ['log_rmse', 'zero_log_rmse', 'tail10_log_rmse', 'tail30_log_rmse', 'tail100_log_rmse', 'zero_recall', 'zero_precision']}))
        pd.DataFrame(paired).to_csv(root / 'paired_differences.csv', index=False)
        (root / 'fold_feature_curves.json').write_text(json.dumps(curves, indent=2) + '\n')
        np.savez_compressed(root / 'oof_predictions.npz', actual=y, source_row_index=ids, fold=before['fold'],
          growth_residual=new_residual, **{f'before_{k}':v for k,v in baseline.items()}, **{f'after_{k}':v for k,v in new.items()})
        np.savez_compressed(root / 'oof_three_components.npz', after=new_components,
                            before=np.load(old / 'oof_components.npz')['three_T10'])
        print('Refitting three candidate artifacts on original training 80%...', flush=True)
        transformer = MassGrowthResidual(20).fit(X)
        full = transformer.transform(X)
        leader = TwoStageLogSFR(HGB_FIXED400, full.columns, full.columns, zero_threshold=.9).fit(full, y)
        three = ThreeRegimeLogSFR(10).fit(full, y)
        models = {names[0]:leader, names[1]:three, names[2]:ZeroPreservedThreeRegime(three, leader.clf_)}
        artifacts = {}
        for name, model in models.items():
            wrapper = ResidualFeatureModel(transformer, model)
            path = root / (name + '_residual.joblib')
            joblib.dump(wrapper, path, compress=3)
            loaded = joblib.load(path)
            np.testing.assert_array_equal(loaded.predict(X.iloc[:1000]), model.predict(full.iloc[:1000]))
            artifacts[name] = {'file':path.name, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        # Verify integration on raw, unlabeled data; reference transform remains frozen.
        reference = df.drop(columns='sfr_linear')
        loaded = joblib.load(root / artifacts[names[1]]['file'])
        raw_prediction = predict_sfr(reference.iloc[ids[:1000]], loaded, reference)
        np.testing.assert_allclose(raw_prediction, loaded.predict(X.iloc[:1000]), rtol=1e-6, atol=1e-9)
    meta = dict(feature=FEATURE, n_features_before=34, n_features_after=35, n_bins=20,
                fitted_reference='Per-fold training input only; final artifacts fit reference on original training 80%',
                data_git_blob_sha=sha, original_rows=len(df), training_rows=len(y),
                holdout_evaluated=False, production_model_changed=False,
                checks={'source_modules_match_baseline':True, 'fold_indices_and_labels_match':True,
                        'oof_coverage_once':True, 'saved_model_reload':True, 'raw_interface':True},
                artifacts=artifacts,
                versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','joblib']},
                caution='Exploratory ablation on previously used CV, not an independent test. No claim of time-normalized growth.')
    (root / 'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()

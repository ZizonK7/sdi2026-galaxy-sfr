"""Verify the adopted rule and reported metrics from saved out-of-fold predictions."""
from pathlib import Path
import json
import numpy as np
from zero_gate import apply_zero_threshold


def main():
    root = Path(__file__).resolve().parent
    metadata = json.loads((root / 'metadata.json').read_text())
    with np.load(root / 'oof_predictions.npz') as data:
        adopted = apply_zero_threshold(data['soft'], data['p_zero'], .9)
        np.testing.assert_array_equal(adopted, data['adopted'])
        actual = np.log10(1 + data['actual'])
        for name, key in [('soft', 'before_rmse'), ('adopted', 'after_rmse')]:
            rmse = float(np.sqrt(np.mean((np.log10(1 + data[name]) - actual) ** 2)))
            np.testing.assert_allclose(rmse, metadata[key], rtol=1e-12)
            print(f'{key}: {rmse:.12f}')
    print('Saved OOF policy and metrics verified.')


if __name__ == '__main__':
    main()

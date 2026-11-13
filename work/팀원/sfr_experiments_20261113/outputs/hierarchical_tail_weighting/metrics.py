import numpy as np

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

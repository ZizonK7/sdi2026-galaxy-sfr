"""Reproduce final LightGBM predictions with the independently reviewed benchmark implementation."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BENCH = ROOT / 'work/조장/model_comparison_2026-11-12'


def main():
    spec = importlib.util.spec_from_file_location('reviewed_benchmark',BENCH/'benchmark.py')
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    protocol = json.loads((BENCH/'protocol.json').read_text(encoding='utf-8'))
    _, df = benchmark.load_data(ROOT,protocol)
    namespace = {'np':np,'pd':pd,'cKDTree':cKDTree,'coo_matrix':coo_matrix,'connected_components':connected_components,
                 **protocol['source']['environment_globals']}
    funcs,_ = benchmark.extract_notebook_functions(ROOT/'DataHub_701104_final.ipynb',protocol['source']['extracted_functions'],namespace)
    with threadpool_limits(6):
        X,_,_,_ = benchmark.build_features(df,protocol,funcs)
        y=df.sfr_linear.to_numpy(float)
        tr,va=train_test_split(df.index.to_numpy(),test_size=.2,random_state=42,stratify=(y>0).astype(int))
        pred,seconds,_,clf_n,reg_n=benchmark.fit_predict(protocol['models']['lightgbm_two_stage_fixed400'],X.iloc[tr],y[tr],X.iloc[va],None,6,.9)
    ztrue,zpred=np.log10(y[va]+1),np.log10(pred+1)
    out = {
        'algorithm':'LightGBM','n_train':len(tr),'n_val':len(va),'classifier_iterations':clf_n,'regressor_iterations':reg_n,
        'validation_log_rmse':float(np.sqrt(np.mean((ztrue-zpred)**2))),
        'validation_mae':float(mean_absolute_error(y[va],pred)),
        'validation_rmse':float(mean_squared_error(y[va],pred)**.5),
        'validation_r2':float(r2_score(y[va],pred)),
        'validation_log_rmse_eps0.01':float(np.sqrt(np.mean((np.log10(y[va]+.01)-np.log10(pred+.01))**2))),
        'zero_sfr_log_rmse':float(np.sqrt(np.mean((ztrue[y[va]==0]-zpred[y[va]==0])**2))),
        'positive_sfr_log_rmse':float(np.sqrt(np.mean((ztrue[y[va]>0]-zpred[y[va]>0])**2))),
        'independent_refit_seconds':seconds,
    }
    np.savez_compressed(HERE/'independent_predictions.npz',validation_indices=va,y_true=y[va],predictions=pred)
    (HERE/'independent_prediction.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()

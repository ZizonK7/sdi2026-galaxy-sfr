"""Execute the whole final notebook in the repository venv and export revision evidence."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PATH = ROOT / "DataHub_701104_final.ipynb"

EXPORT = r'''
def _revision_table(frame):
    frame = frame.copy()
    if not isinstance(frame.index, pd.MultiIndex):
        frame.index = frame.index.map(str)
    return json.loads(frame.reset_index().to_json(orient="records"))

_rp = FINAL_MODEL.predict_proba_positive(X_val_env)
_rz = FINAL_MODEL.predict_log_positive(X_val_env)
_ry = y_val.to_numpy(float)
_rztrue = to_log(_ry)
_rzpred = to_log(VAL_PREDICTIONS)
_rsoft = to_log(from_log(_rp * _rz + (1-_rp) * to_log(0.0)))
_rgated = (1-_rp) >= ZERO_THRESHOLD
_rdiff = (_rzpred-_rztrue)**2 - (_rsoft-_rztrue)**2
_rzero = _ry==0
_routzero = VAL_PREDICTIONS==0
_rsq = (_rzpred-_rztrue)**2
_rband = ((1-_rp)>=.88)&((1-_rp)<=.92)
REPORT_REVISION_EVIDENCE = {
    "results": RESULTS,
    "log_table": _revision_table(log_table),
    "log_bins": _revision_table(log_bins),
    "gate_cost": _revision_table(GATE_COST),
    "importance_classifier": _revision_table(imp_c),
    "importance_regressor": _revision_table(imp_r),
    "environment_errors": _revision_table(env_err),
    "satellite_table": _revision_table(sat_tab),
    "diagnostics": {
        "n_val": len(_ry), "true_zero_count": int(_rzero.sum()),
        "zero_true_positive": int((_routzero&_rzero).sum()),
        "positive_predicted_zero": int((_routzero&~_rzero).sum()),
        "zero_recall": float(_routzero[_rzero].mean()),
        "zero_precision": float(_rzero[_routzero].mean()),
        "zero_sse_share": float(_rsq[_rzero].sum()/_rsq.sum()),
        "zero_missed_by_p05_count": int((_rzero&(_rp>.5)).sum()),
        "zero_missed_by_p05_sse_share": float(_rsq[_rzero&(_rp>.5)].sum()/_rsq[_rzero].sum()),
        "high_sfr_count": int((_ry>100).sum()),
        "high_sfr_mean_log_residual": float((_rzpred-_rztrue)[_ry>100].mean()),
        "high_sfr_sse_share": float(_rsq[_ry>100].sum()/_rsq.sum()),
        "gate_mse_delta_true_zero": float(_rdiff[_rzero].sum()/len(_ry)),
        "gate_mse_delta_positive": float(_rdiff[~_rzero].sum()/len(_ry)),
        "gate_mse_delta_all": float(_rdiff.mean()),
        "p0_band_count": int(_rband.sum()),
        "p0_band_actual_zero": int((_rband&_rzero).sum()),
        "p0_band_mean_p0": float((1-_rp)[_rband].mean()),
        "classifier_iterations": int(FINAL_MODEL.clf_.n_iter_),
        "regressor_iterations": int(FINAL_MODEL.reg_.n_iter_),
    },
}
print("REVISION_EVIDENCE_JSON=" + json.dumps(REPORT_REVISION_EVIDENCE,ensure_ascii=False,default=str))
'''


def main():
    start = time.perf_counter()
    notebook = nbformat.read(PATH, as_version=4)
    original_sources = [(c.cell_type, c.source) for c in notebook.cells]
    notebook.cells.append(nbformat.v4.new_code_cell(EXPORT))
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS":"6", "OPENBLAS_NUM_THREADS":"6", "MKL_NUM_THREADS":"6",
                "NUMEXPR_NUM_THREADS":"6", "LOKY_MAX_CPU_COUNT":"1", "PYTHONIOENCODING":"utf-8",
                "MPLBACKEND":"module://matplotlib_inline.backend_inline"})
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    km.kernel_spec.env.update({k:env[k] for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","LOKY_MAX_CPU_COUNT","MPLBACKEND")})

    def progress(cell, cell_index, **kwargs):
        print(f"Executed cell {cell_index+1}/{len(notebook.cells)}: {cell.source.splitlines()[0][:110]}", flush=True)

    try:
        km.start_kernel(cwd=str(ROOT), env=env)
        client = NotebookClient(notebook, km=km, timeout=3600, resources={"metadata":{"path":str(ROOT)}},
                                allow_errors=False, on_cell_executed=progress)
        client.execute()
    except Exception:
        nbformat.write(notebook, HERE / "execution_failed.ipynb")
        raise
    finally:
        if km.has_kernel:
            km.shutdown_kernel(now=True)

    evidence = None
    for output in notebook.cells[-1].outputs:
        text = output.get("text", "")
        if isinstance(text, list):
            text = "".join(text)
        marker = "REVISION_EVIDENCE_JSON="
        if marker in text:
            evidence = json.loads(text.split(marker,1)[1].strip())
    assert evidence is not None
    notebook.cells.pop()
    current = nbformat.read(PATH, as_version=4)
    assert [(c.cell_type,c.source) for c in current.cells] == original_sources, "Notebook source changed during execution"
    nbformat.write(notebook, PATH)
    evidence["execution"] = {
        "seconds": time.perf_counter()-start,
        "code_cells": sum(c.cell_type=="code" for c in notebook.cells),
        "png_outputs": sum("image/png" in o.get("data",{}) for c in notebook.cells for o in c.get("outputs",[])),
        "notebook_sha256_before_prose_refresh": hashlib.sha256(PATH.read_bytes()).hexdigest(),
        "kernel_python": sys.executable,
        "threads": 6,
        "permutation_joblib_workers": 1,
    }
    (HERE / "execution_evidence.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"completed":True,"execution":evidence["execution"],"model":evidence["results"]["model_name"],
                      "validation_log_rmse":evidence["results"]["validation_log_rmse"]},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()

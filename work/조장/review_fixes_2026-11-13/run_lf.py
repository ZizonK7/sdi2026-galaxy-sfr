"""Execute a notebook copy top to bottom inside a run folder that holds the LF (as-distributed) CSV.

Usage: python run_lf.py <run_dir>
The run folder must contain DataHub_701104_final.ipynb and the CSV. Outputs are checkpointed after every cell.
"""
import os
import sys
from pathlib import Path

run_dir = Path(sys.argv[1]).resolve()
os.environ.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', LOKY_MAX_CPU_COUNT='4',
                  MPLBACKEND='module://matplotlib_inline.backend_inline')
os.environ['IPYTHONDIR'] = str(run_dir / '_ipython')
os.environ['JUPYTER_RUNTIME_DIR'] = str(run_dir / '_jupyter-runtime')
os.environ['JOBLIB_TEMP_FOLDER'] = str(run_dir / '_joblib')
for name in ('_ipython', '_jupyter-runtime', '_joblib'):
    (run_dir / name).mkdir(exist_ok=True)

import hashlib
import json
import time

import nbformat
from nbclient import NotebookClient

sys.stdout.reconfigure(encoding='utf-8')
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

csv = run_dir / 'Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv'
raw = csv.read_bytes()
print('CSV sha256', hashlib.sha256(raw).hexdigest(), 'CR bytes', raw.count(b'\r'), flush=True)
del raw

path = run_dir / 'DataHub_701104_final.ipynb'
nb = nbformat.read(path, as_version=4)
client = NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata': {'path': str(run_dir)}},
                        record_timing=True)
times = []
started = time.monotonic()
status = 'error'
try:
    with client.setup_kernel():
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != 'code':
                continue
            begin = time.monotonic()
            first = cell.source.splitlines()[0][:90] if cell.source else 'empty'
            print(f'RUN {i}/{len(nb.cells)} {first}', flush=True)
            try:
                client.execute_cell(cell, i)
            finally:
                nbformat.write(nb, path)
            times.append({'cell': i, 'seconds': round(time.monotonic() - begin, 3)})
            print(f'OK  {i} {times[-1]["seconds"]:.2f}s', flush=True)
    status = 'complete'
finally:
    nbformat.write(nb, path)
    (run_dir / 'execution_times.json').write_text(
        json.dumps({'status': status, 'seconds': time.monotonic() - started, 'cells': times}, indent=2),
        encoding='utf-8')
print(f'COMPLETE {len(times)} code cells in {time.monotonic() - started:.1f}s', flush=True)

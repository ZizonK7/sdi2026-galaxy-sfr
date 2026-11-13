"""Extract the final model from the pinned notebook without executing its experiments."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'outputs/team_lead_model_21d02b5'
raw = (ROOT / 'DataHub_701104_final.ipynb').read_bytes()
assert hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest() == '2139283816711f2d116e0136b6f00598591c3a90'
nb = json.loads(raw)
header = '''"""Model definitions extracted from team-lead commit 21d02b5.

Original functions/classes are preserved; predict_sfr requires explicit model/reference.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

FINAL_FEATURES = ['log10_halo_mass', 'log10_stellar_mass', 'log10_BH_mass',
 'log10_main_prog_mass', 'log10_main_prog_stellar_mass', 'log10_mass_ratio',
 'log10_overdensity_plus_1', 'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z']
ID_COLUMN = 'subhalo_id'
BOX_SIZE = 200.0
RHO_CRIT = 2.775e11
G_KMS = 4.30091e-9
LOG_EPS = 1.0
ZERO_THRESHOLD = 0.9
HGB_FIXED400 = dict(learning_rate=.08, max_iter=400, max_leaf_nodes=31,
                    random_state=42, early_stopping=False)
'''
parts = [header]
extracted = []
for i in (90, 91, 92):
    source = ''.join(nb['cells'][i]['source'])
    for node in ast.parse(source).body:
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            continue
        if node.name == 'expect_input_error':
            continue
        code = ast.get_source_segment(source, node)
        if node.name == 'predict_sfr':
            code = code.replace('model=FINAL_MODEL, reference=None', 'model, reference')
            code = code.replace('reference = df if reference is None else reference',
                                'if reference is None:\n        raise ValueError("An explicit fixed reference catalogue is required")')
            code = code.replace('참고 표 기본값은 df.', '참고 표는 명시적으로 전달한다.')
        parts.append(code)
        extracted.append({'cell_index': i, 'name': node.name})
parts.append('''def build_features(catalogue):
    """Match the notebook's lexicographic ordering before environment calculation."""
    raw = catalogue[FINAL_FEATURES].copy()
    if not raw.index.is_unique:
        raise ValueError('Catalogue index must be unique')
    order = raw.sort_values(FINAL_FEATURES, kind='mergesort', na_position='last').index
    env = build_environment_features(raw.loc[order]).reindex(raw.index)
    return raw.join(env).join(build_mass_history_features(raw))
''')
(ROOT / 'leader_model.py').write_text('\n\n'.join(parts) + '\n')
(ROOT / 'source_manifest.json').write_text(json.dumps({
    'repository': 'ZizonK7/sdi2026-galaxy-sfr', 'branch': 'codex/team-lead-work',
    'commit': '21d02b5d0c829aff67cafd89da62c3386ac7864f',
    'notebook_git_blob_sha': '2139283816711f2d116e0136b6f00598591c3a90',
    'extracted_definitions': extracted,
    'adaptations': ['Explicit model/reference arguments for standalone predict_sfr',
                    'build_features wraps original functions with original catalogue sort'],
    'upstream_fitted_artifact_available': False,
}, ensure_ascii=False, indent=2) + '\n')
print('Verified notebook SHA and extracted', len(extracted), 'definitions.')

"""Check the executed final report, independent predictions, and historical-source preservation."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import nbformat
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BENCH = ROOT / 'work/조장/model_comparison_2026-11-12'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path=ROOT/'DataHub_701104_final.ipynb'
    nb=nbformat.read(path,as_version=4)
    original=nbformat.read(HERE/'reference/DataHub_701104_final_hgb.ipynb',as_version=4)
    nbformat.validate(nb)
    assert [(c.cell_type,c.source) for c in nb.cells[1:85]] == [(c.cell_type,c.source) for c in original.cells[1:85]], 'Part I source changed'
    def feature_functions(book):
        return {node.name:ast.dump(node,include_attributes=False) for cell in book.cells if cell.cell_type=='code'
                for node in ast.parse(cell.source).body if isinstance(node,ast.FunctionDef)
                and node.name in {'build_environment_features','build_mass_history_features'}}
    assert feature_functions(nb)==feature_functions(original), 'Feature construction changed'
    fixed=[c.source for c in original.cells if c.cell_type=='code' and 'DO NOT MODIFY' in c.source]
    for source in fixed:
        assert any(c.cell_type=='code' and c.source==source for c in nb.cells), 'Fixed template code changed'
    counts=[]
    for i,c in enumerate(nb.cells):
        if c.cell_type!='code': continue
        ast.parse(c.source,filename=f'cell {i}')
        assert c.execution_count is not None
        counts.append(c.execution_count)
        assert all(o.output_type!='error' for o in c.outputs)
    assert counts==list(range(1,len(counts)+1)), 'Non-sequential execution'
    evidence=json.loads((HERE/'execution_evidence.json').read_text(encoding='utf-8'))
    result=evidence['results']
    output=[]
    cell=next(c for c in nb.cells if c.cell_type=='code' and c.source.startswith('# STUDENT TODO — update model_name'))
    for o in cell.outputs:
        if o.output_type=='stream':
            output.append(o.text)
    printed=json.loads(''.join(output))
    assert printed==result
    assert result['model_name']=='TwoStageLightGBMLogSFR' and result['algorithm']=='LightGBM'
    assert result['n_features']==34 and result['features']==result['stage2_regressor_features']
    assert len(set(result['features']))==34
    assert result['diagnostic_cv']['algorithm']=='HGB historical diagnostic'
    assert all(result['prediction_contract_checks'].values())
    params=result['model_params']
    assert params==dict(n_estimators=400,num_leaves=31,max_bin=255,learning_rate=.08,random_state=42,n_jobs=6,verbosity=-1,deterministic=True,force_col_wise=True)
    assert evidence['diagnostics']['classifier_iterations']==evidence['diagnostics']['regressor_iterations']==400
    other=json.loads((HERE/'independent_prediction.json').read_text(encoding='utf-8'))
    differences={}
    for key in ['validation_log_rmse','validation_mae','validation_rmse','validation_r2',
                'validation_log_rmse_eps0.01','zero_sfr_log_rmse','positive_sfr_log_rmse']:
        differences[key]=abs(result[key]-other[key])
        assert np.isclose(result[key],other[key],rtol=1e-12,atol=1e-12), (key,result[key],other[key])
    old_manifest=json.loads((BENCH/'results/manifest.json').read_text(encoding='utf-8'))
    assert digest(HERE/'reference/DataHub_701104_final_hgb.ipynb')==old_manifest['fingerprint']['notebook_sha256']
    assert digest(ROOT/old_manifest['paths']['data'])==old_manifest['fingerprint']['data_sha256']
    protected=json.loads((BENCH/'source_baseline.json').read_text(encoding='utf-8'))
    for name,value in protected['preserved_source_sha256'].items():
        if name=='DataHub_701104_final.ipynb': continue
        assert digest(ROOT/name)==value, name
    comparison=result['algorithm_comparison']
    assert len(comparison['records'])==35
    assert comparison['decision']['selected_model']=='lightgbm_two_stage_fixed400'
    assert comparison['decision']['decision_after_benchmark'] is True
    assert comparison['decision']['accuracy_equivalence_established'] is False
    assert comparison['csv_sha256']==old_manifest['fingerprint']['data_sha256']
    md='\n'.join(c.source for c in nb.cells if c.cell_type=='markdown')
    for heading in ['## 최종 보고서 읽기 안내','### 9.3 Final Method Description','## 13. Limitations & Conclusion','## 14. Final Model Card']:
        section=next(c.source for c in nb.cells if c.cell_type=='markdown' and c.source.startswith(heading))
        assert 'LightGBM' in section, heading
    for heading in ['## 최종 보고서 읽기 안내','### 10.1 Required Comparison Table','## 13. Limitations & Conclusion','## 14. Final Model Card']:
        section=next(c.source for c in nb.cells if c.cell_type=='markdown' and c.source.startswith(heading))
        assert f"{result['validation_log_rmse']:.6f}" in section or f"{result['validation_log_rmse']:.7f}" in section, heading
    assert '13.44' in md and '6.76' in md and '6.75' in md
    assert '동등성' in md and '결과 확인 후' in md
    # Reuse existing meaningful metric/offset/negative-prediction regression checks.
    spec=importlib.util.spec_from_file_location('prior_report_checks',ROOT/'work/조장/review_revision_2026-11-11/validate_revision.py')
    validation=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation)
    functions=validation.static_checks(nb,validation.read_notebook(ROOT/'DataHub_701104_mid.ipynb'))
    validation.numerical_checks(functions)
    validation.plot_checks(nb)
    images=0
    for c in nb.cells:
        for o in c.get('outputs',[]):
            if 'image/png' in o.get('data',{}): images+=1
    checked={
        'passed':True,'model_name':result['model_name'],'algorithm':result['algorithm'],
        'code_cells':len(counts),'png_outputs':images,'fixed_source_cells':len(fixed),
        'part_i_source_preserved':True,'protected_data_template_mid_preserved':True,
        'prediction_contract_checks':result['prediction_contract_checks'],
        'independent_metric_differences':differences,'validation_log_rmse':result['validation_log_rmse'],
        'decision_after_benchmark':True,'notebook_sha256':digest(path),
        'historical_notebook_sha256':digest(HERE/'reference/DataHub_701104_final_hgb.ipynb'),
    }
    (HERE/'validation.json').write_text(json.dumps(checked,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(checked,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

"""Check saved OOF coverage and reported scores without fitting any estimator."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd


def main():
    root=Path(__file__).resolve().parent
    checks=0
    for prefix, archive in [('stage2','stage2_oof.npz'),('confirmation','confirmation_oof.npz')]:
        data=np.load(root/archive)
        assert len(np.unique(data['source_row_index']))==len(data['actual'])==166749
        assert set(data['fold'])=={1,2,3,4,5}
        table=pd.read_csv(root/(prefix+'_summary.csv'))
        y=data['actual'];actual=np.log10(1+y)
        for _,row in table.iterrows():
            pred=data[row.model]
            assert np.isfinite(pred).all() and (pred>=0).all()
            se=(np.log10(1+pred)-actual)**2
            per_fold=[float(np.sqrt(se[data['fold']==fold].mean())) for fold in range(1,6)]
            np.testing.assert_allclose(np.mean(per_fold),row.mean_fold_rmse,rtol=1e-12)
            np.testing.assert_allclose(np.sqrt(se.mean()),row.log_rmse,rtol=1e-12)
            np.testing.assert_allclose(np.sqrt(se[y==0].mean()),row.zero_log_rmse,rtol=1e-12)
            np.testing.assert_allclose(np.sqrt(se[y>100].mean()),row.tail100_log_rmse,rtol=1e-12)
            checks+=4
    plan=json.loads((root/'frozen_selection.json').read_text())['plan']
    for cols in plan.values():
        assert len(cols)==len(set(cols))
        assert not {'sfr_linear','subhalo_id'} & set(cols)
    meta=json.loads((root/'metadata.json').read_text())
    assert hashlib.sha256((root/'best_feature_model.joblib').read_bytes()).hexdigest()==meta['model_sha256']
    result=dict(recomputed_metric_checks=checks,oof_coverage=True,nonnegative_finite=True,
                no_target_or_identifier_features=True,model_hash=True)
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

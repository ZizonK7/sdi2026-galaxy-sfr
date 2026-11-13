import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from feature_pipeline import build_reference_features,predict_sfr
from blend_model import ProtectedTailBlend
from metrics import metrics

ROOT=Path(__file__).resolve().parent

def main():
    meta=json.loads((ROOT/'metadata.json').read_text())
    assert meta['frozen']['strict_balance']==dict(name='lgb_w3_a50',partner='lgb_w3',alpha=.5)
    rows=[];fold_rows=[];accepted=[]
    for seed in (42,7,99):
        source=np.load(ROOT/f'oof_seed{seed}.npz')
        y=source['actual'];base=source['baseline'];mixed=source['strict_balance'];fold=source['fold']
        pred=np.where(base>10.,mixed,base)
        np.testing.assert_array_equal(pred[base<=10.],base[base<=10.])
        np.testing.assert_array_equal(pred[source['zero_gate_mask']],0.)
        base_mean=np.mean([metrics(y[fold==f],base[fold==f])['log_rmse'] for f in range(1,6)])
        scores=[]
        for f in range(1,6):
            mask=fold==f;result=metrics(y[mask],pred[mask]);scores.append(result['log_rmse'])
            fold_rows.append(dict(seed=seed,fold=f,**result))
        result=metrics(y,pred);mean=float(np.mean(scores));b=metrics(y,base)
        passed=(mean<=base_mean*1.001+1e-14 and result['zero_log_rmse']<=b['zero_log_rmse']+1e-14
                and result['tail100_log_rmse']<b['tail100_log_rmse'])
        accepted.append(bool(passed))
        rows.append(dict(seed=seed,mean_fold_rmse=mean,**result,
            overall_pct=(mean/base_mean-1)*100,zero_pct=(result['zero_log_rmse']/b['zero_log_rmse']-1)*100,
            extreme_pct=(result['tail100_log_rmse']/b['tail100_log_rmse']-1)*100,
            routed_rows=int((base>10).sum()),routed_actual_zero=int(((base>10)&(y==0)).sum()),
            routed_actual_extreme=int(((base>10)&(y>100)).sum()),accepted=bool(passed)))
        np.savez_compressed(ROOT/f'protected_oof_seed{seed}.npz',actual=y,source_row_index=source['source_row_index'],fold=fold,prediction=pred)
    table=pd.DataFrame(rows);table.to_csv(ROOT/'protected_summary.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(ROOT/'protected_folds.csv',index=False)
    status=dict(accepted_all=all(accepted),original_recommendation_unchanged=meta['recommendation'],
                inference_threshold=10.,fraction=.5,post_hoc_followup=True,independent_test=False)
    print(table[['seed','mean_fold_rmse','zero_log_rmse','tail100_log_rmse','overall_pct','zero_pct','extreme_pct','routed_actual_zero','accepted']].to_string(index=False),flush=True)
    if all(accepted):
        with threadpool_limits(limits=4):
            baseline=joblib.load(ROOT.parent/'hierarchical_tail_models/selected_model.joblib')
            blend=joblib.load(ROOT/'strict_balance.joblib')
            model=ProtectedTailBlend(baseline,blend,10.)
            path=ROOT/'protected_model.joblib';joblib.dump(model,path,compress=3)
            restored=joblib.load(path)
            data_path=Path('/Users/ljh/Downloads/Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv')
            raw=data_path.read_bytes()
            assert hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()==meta['data_git_blob_sha']
            df=pd.read_csv(data_path);reference=df.drop(columns='sfr_linear')
            X=build_reference_features(df).iloc[:1000]
            b=baseline.predict(X);m=blend.predict(X)
            np.testing.assert_array_equal(model.predict(X),np.where(b>10,m,b))
            np.testing.assert_array_equal(model.predict(X),restored.predict(X))
            raw_predictions=predict_sfr(reference.iloc[:1000],restored,reference)
            np.testing.assert_allclose(raw_predictions,model.predict(X),rtol=1e-6,atol=1e-9)
            status.update(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),saved_reload=True,raw_interface=True)
    (ROOT/'protected_metadata.json').write_text(json.dumps(status,indent=2)+'\n')
    print('PROTECTED FOLLOW-UP COMPLETE',flush=True)

if __name__=='__main__':main()

"""Export the frozen50:50 candidate that meets the user's overall/tail objective.

Do not hide the failure of the additional strict-zero guardrail on seed99.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from feature_pipeline import build_reference_features,predict_sfr

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True)
    args=parser.parse_args()
    meta=json.loads((ROOT/'metadata.json').read_text())
    table=pd.read_csv(ROOT/'acceptance.csv')
    rows=table[table.role=='strict_balance']
    assert set(rows.seed)=={42,7,99}
    assert (rows.overall_pct<0).all() and (rows.extreme_pct<0).all()
    assert meta['frozen']['strict_balance']==dict(name='lgb_w3_a50',partner='lgb_w3',alpha=.5)
    source=ROOT/'strict_balance.joblib';dest=ROOT/'balanced_model.joblib'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['saved_models']['strict_balance']['sha256']
    shutil.copy2(source,dest)
    with threadpool_limits(limits=4):
        raw=args.data.read_bytes()
        assert hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()==meta['data_git_blob_sha']
        df=pd.read_csv(args.data);reference=df.drop(columns='sfr_linear')
        model=joblib.load(dest)
        X=build_reference_features(df).iloc[:1000]
        result=predict_sfr(reference.iloc[:1000],model,reference)
        np.testing.assert_allclose(result,model.predict(X),rtol=1e-6,atol=1e-9)
    result=dict(file=dest.name,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
        config=meta['frozen']['strict_balance'],objective='overall log RMSE and actual SFR>100 log RMSE decrease in all three seeds',
        objective_met=True,additional_strict_zero_guardrail_met=False,
        zero_rmse_pct_changes=dict(zip(rows.seed.astype(str),rows.zero_pct)),
        original_acceptance_and_recommendation_preserved=True,raw_interface_verified=True,
        current_model_replaced=False,independent_test=False)
    (ROOT/'balanced_candidate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()

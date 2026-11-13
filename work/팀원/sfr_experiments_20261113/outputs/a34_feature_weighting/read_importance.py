from pathlib import Path
import json
import joblib
import pandas as pd

def read_importance(previous,out):
    previous=Path(previous);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((previous/'metadata.json').read_text())
    model=joblib.load(previous/'A34_selected.joblib')
    gate=model.models_['gate'];names=meta['features']
    assert gate.n_features_in_==len(names)==34
    gain=gate.booster_.feature_importance(importance_type='gain')
    split=gate.booster_.feature_importance(importance_type='split')
    table=pd.DataFrame(dict(feature=names,gain=gain,gain_pct=100*gain/gain.sum(),splits=split))
    table=table.sort_values('gain',ascending=False).reset_index(drop=True)
    table.insert(0,'rank',range(1,len(table)+1))
    table.to_csv(out/'current_gate_importance.csv',index=False)
    print(table.to_string(index=False),flush=True)
    return table

if __name__=='__main__':
    read_importance('outputs/a34_model_comparison','outputs/a34_feature_weighting')

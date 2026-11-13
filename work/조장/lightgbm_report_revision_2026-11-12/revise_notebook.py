"""Apply the user's LightGBM accuracy/time choice to the final report notebook."""
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
NB_PATH = ROOT / "DataHub_701104_final.ipynb"
BENCH = ROOT / "work/조장/model_comparison_2026-11-12"


def src(cell):
    return "".join(cell.get("source", []))


def put(cell, text):
    cell["source"] = text.splitlines(keepends=True)
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def main():
    original = json.loads((HERE / "reference/DataHub_701104_final_hgb.ipynb").read_text(encoding="utf-8"))
    nb = json.loads(json.dumps(original))
    cells = nb["cells"]
    summary = pd.read_csv(BENCH / "results/summary.csv")
    fold_records = pd.read_csv(BENCH / "results/fold_results.csv").to_dict("records")
    manifest = json.loads((BENCH / "results/manifest.json").read_text(encoding="utf-8"))
    oldhash = hashlib.sha256((HERE / "reference/DataHub_701104_final_hgb.ipynb").read_bytes()).hexdigest()
    assert oldhash == manifest["fingerprint"]["notebook_sha256"]
    by_model = summary.set_index("model")
    lgb, hgb = by_model.loc["lightgbm_two_stage_fixed400"], by_model.loc["hgb_two_stage_fixed400"]
    decision = {
        "selected_model": "lightgbm_two_stage_fixed400",
        "reason": "관측 오차가 근접한 후보 중 학습 시간을 고려해 LightGBM을 선택한다는 사용자 결정.",
        "user_instruction": "학습 시간을 근거로 LIGHTGBM쪽으로 보고서 수정해줘.",
        "decision_after_benchmark": True,
        "mean_rmse_delta_vs_hgb_fixed400": float(lgb.mean_rmse_log-hgb.mean_rmse_log),
        "observed_mean_training_seconds": float(lgb.mean_training_seconds),
        "hgb_fixed400_mean_training_seconds": float(hgb.mean_training_seconds),
        "observed_time_reduction_fraction": float(1-lgb.mean_training_seconds/hgb.mean_training_seconds),
        "accuracy_equivalence_established": False,
    }
    decision["reason"] = "관측 오차가 근접한 후보 중 학습 시간을 고려해 LightGBM을 선택한다는 사용자 결정."
    comparison = {
        "records": fold_records,
        "summary": summary.to_dict("records"),
        "decision": decision,
        "source_commit": manifest["git"]["commit"],
        "notebook_sha256_before_revision": oldhash,
        "csv_sha256": manifest["fingerprint"]["data_sha256"],
        "protocol_sha256": manifest["fingerprint"]["protocol_sha256"],
        "benchmark_script_sha256": manifest["fingerprint"]["script_sha256"],
        "comparison_scope": "같은 학습 내부 5-fold, 34개 피처, log10(SFR+1), 알고리즘·구조당 고정 설정 하나",
    }
    (HERE / "selection_decision.json").write_text(json.dumps(comparison,ensure_ascii=False,indent=2),encoding="utf-8")
    # Keep HGB's historical 25-fold feature/threshold experiments as explicitly labeled history.
    text = src(cells[87]).replace("최종 0 출력 문턱 0.9", "이전 HGB의 0 출력 문턱 0.9").replace("최종 모델과 직전 후보 비교", "이전 HGB와 직전 후보 비교")
    text = text.replace("이후의 대조 실험과 오차 진단은 이 모델을 고정한 상태에서 수행했다.",
                        "이 표는 HGB에서 구조·피처·문턱을 고른 이력이다. 이후 별도의 알고리즘 비교에서 정확도와 학습 시간을 함께 고려해 최종 학습기를 LightGBM으로 변경했다. 0.9는 HGB에서 고른 값을 이전했으며 LightGBM에서 재튜닝하지 않았다.")
    text += "\n**최종 알고리즘 선택.** 같은 34개 피처와 학습 내부 5-fold에서 단일·두 단계 HGB, LightGBM, CatBoost를 비교했다. HGB 400회 두 단계의 평균 로그 RMSE는 0.095340, LightGBM은 0.095367로 차이가 0.000027 dex였고, 평균 학습 시간은 각각 13.44초와 6.76초였다. 관측 정확도의 작은 차이를 수용하고 학습 시간이 약 49.7% 짧은 LightGBM 두 단계를 선택했다. 기존 조기 종료 HGB는 0.096619와 6.75초로, LightGBM의 시간 이점은 최고 정확도 후보인 HGB 400회 모델과 비교한 것이다. 시간·정확도 절충은 비교 결과를 확인한 뒤 정한 선택 근거이며 동등성·비열등성 검정은 수행하지 않았다.\n"
    put(cells[87], text)
    text = src(cells[89]).replace('("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")', '("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "matplotlib")')
    text = text.replace('"selection_policy": "최종 모델 고정 후 대조 실험과 오차 진단 수행"',
                        '"selection_policy": "기존 HGB 선택 이력을 보존하고, 7구성 5-fold의 관측 정확도·학습 시간과 사용자 결정으로 LightGBM 선택",\n    "source_commit_before_revision": "6bc3988dd7effb7694192ce059f515fa496fa499"')
    put(cells[89], text)
    text = src(cells[90]).replace('FINAL_HGB = {**HGB, "max_iter": 400}', 'FINAL_HGB = {**HGB, "max_iter": 400}')
    put(cells[90], text)
    text = src(cells[91])
    text = text.replace("# 최종 0.9는 평균 +0.0000193 dex, 19/25 악화의 경계 후보이며 동일성/비열등성 증거가 아니다.",
                        "# HGB에서의 0.9는 평균 +0.0000193 dex, 19/25 악화의 경계 후보다. LightGBM에서는 재선택하지 않았다.")
    text = text.replace("# 30개 피처 모델에서는 0.8이었고, 질량 성장 피처를 더한 최종 모델에서 같은 규칙을 다시 적용해 0.9가 됐다.",
                        "# HGB 30개 피처 모델에서는 0.8, HGB 34개에서는 0.9였으며 최종 LightGBM도 통제 비교의 0.9를 유지한다.")
    marker = 'print("실행 설정:", {"LOG_EPS": LOG_EPS'
    before, after = text.split(marker, 1)
    new_model = '''from lightgbm import LGBMClassifier, LGBMRegressor


class TwoStageLightGBMLogSFR(TwoStageLogSFR):
    """동일한 로그 결합·0 문턱에 LightGBM 분류기와 양수 회귀기를 사용하는 최종 모델."""

    def __init__(self, lgbm_params, clf_cols, reg_cols, eps=LOG_EPS, zero_threshold=None):
        super().__init__(lgbm_params, clf_cols, reg_cols, eps=eps, zero_threshold=zero_threshold)
        self.lgbm_params = dict(lgbm_params)

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        pos = y > 0  # 예측값이 아닌 실제 학습 타깃으로 양수 집단을 정의한다.
        self.clf_ = LGBMClassifier(objective="binary", **self.lgbm_params).fit(X[self.clf_cols], pos.astype(int))
        self.reg_ = LGBMRegressor(objective="regression", **self.lgbm_params).fit(
            X.loc[pos, self.reg_cols], to_log(y[pos], self.eps))
        return self


FINAL_LGBM = dict(n_estimators=400, num_leaves=31, max_bin=255, learning_rate=0.08,
                  random_state=42, n_jobs=6, verbosity=-1, deterministic=True, force_col_wise=True)
print("실행 설정:", {"algorithm": "LightGBM", "LOG_EPS": LOG_EPS,
                  "ZERO_THRESHOLD": ZERO_THRESHOLD, "LightGBM": FINAL_LGBM,
                  "classifier_features": CLF_FEATURES, "regressor_features": REG_FEATURES})
FINAL_MODEL = TwoStageLightGBMLogSFR(FINAL_LGBM, CLF_FEATURES, REG_FEATURES,
                                   zero_threshold=ZERO_THRESHOLD)
'''
    train_block = after[after.index('_t0 = time.time()'):]
    train_block += '\nassert FINAL_MODEL.clf_.n_iter_ == FINAL_MODEL.reg_.n_iter_ == 400\n'
    put(cells[91], before + new_model + '\n' + train_block)
    put(cells[93], src(cells[93]).replace("최종 모델 (34개, 문턱 0.9)", "이전 HGB 모델 (34개, 문턱 0.9)"))
    put(cells[94], "**이전 HGB의 반복 교차검증 선택 이력.**\n\n" + src(cells[94]).replace("최종 34개 피처", "이전 HGB의 34개 피처").replace("최종 모델(34개", "이전 HGB 모델(34개"))
    put(cells[95], src(cells[95]).replace("단일 로그 회귀와 환경×성장 피처", "이전 HGB의 단일 로그 회귀와 환경×성장 피처"))
    put(cells[97], src(cells[97]).replace("**실행 결과", "**이전 HGB의 실행 결과"))
    text = src(cells[98]).replace("를 HGB 두 단계로 추정한다.", "를 LightGBM 분류기와 양수 로그 회귀기로 추정한다.")
    start = text.index("**학습.**")
    end = text.index("**적용 계약.**")
    text = text[:start] + "**학습과 선택.** 두 단계 모두 LightGBM(learning_rate=0.08, n_estimators=400, num_leaves=31, max_bin=255, seed=42, CPU 6스레드)을 학습한다. 조기 종료와 평가셋 콜백은 쓰지 않는다. 분류기는 모든 학습 은하, 회귀기는 실제 SFR>0인 은하를 사용한다. 분류 확률로 학습 행을 제거하지 않는다. 같은 피처·분할의 HGB 400회와 오차가 근접하면서 관측 학습 시간이 짧아 LightGBM을 선택했다. 두 알고리즘의 정확도 동등성을 입증한 것은 아니다.\n\n" + text[end:]
    put(cells[98], text)
    text = src(cells[101])
    text = text.replace('_p_fin = FINAL_MODEL.predict_proba_positive(X_val_env)',
                        'PREVIOUS_HGB_MODEL = TwoStageLogSFR(FINAL_HGB, CLF_FEATURES, REG_FEATURES, zero_threshold=ZERO_THRESHOLD).fit(X_train_env, y_train)\n_PREVIOUS_HGB_PREDICTIONS = PREVIOUS_HGB_MODEL.predict(X_val_env)\n_p_previous_hgb = PREVIOUS_HGB_MODEL.predict_proba_positive(X_val_env)\n_p_fin = FINAL_MODEL.predict_proba_positive(X_val_env)')
    text = text.replace('"최종 모델 (34개, 성장·비율 포함, 문턱 0.9)": VAL_PREDICTIONS,',
                        '"이전 HGB 모델 (34개, 조기 종료, 문턱 0.9)": _PREVIOUS_HGB_PREDICTIONS,\n    "최종 LightGBM 모델 (34개, 400회, 문턱 0.9)": VAL_PREDICTIONS,')
    text = text.replace('"ref: FINAL + tail split T=30": from_log(_p_fin * _e_tail + (1 - _p_fin) * to_log(0.0)),',
                        '"ref: previous HGB + tail split T=30": from_log(_p_previous_hgb * _e_tail + (1 - _p_previous_hgb) * to_log(0.0)),')
    text = text.replace('"ref: FINAL retrained for D2 (LOG_EPS = 0.01)": TwoStageLogSFR(FINAL_HGB,',
                        '"ref: FINAL retrained for D2 (LOG_EPS = 0.01)": TwoStageLightGBMLogSFR(FINAL_LGBM,')
    put(cells[101], text)
    text = src(cells[117])
    text = text.replace('"model_name": type(FINAL_MODEL).__name__,',
                        '"model_name": type(FINAL_MODEL).__name__,\n    "algorithm": "LightGBM",\n    "model_params": FINAL_LGBM,\n    "algorithm_comparison": ALGORITHM_COMPARISON,')
    text = text.replace('"used_for_model_selection": False}', '"used_for_model_selection": False, "algorithm": "HGB historical diagnostic"}')
    text = text.replace('"notes": "로그 변환은 log10(SFR + 1)을 가정했다. 같은 검증 표본을 Part I부터 재사용했다. 문턱 0.9는 작은 교차검증 비용을 수용한 0 출력 정책이다. 최종 모델을 고정한 뒤 학습 내부 5개 분할 대조와 오차 진단을 수행했다. 환경 피처는 같은 상자의 고유 은하 합집합을 사용한다. 적용 범위는 13.1절에 정리했다.",',
                        '"notes": "로그 변환 log10(SFR + 1)은 가정이다. 기존 HGB의 피처·문턱 선택 이력을 보존하고, 같은 34개 피처의 7구성 5-fold 비교와 정확도·관측 학습 시간의 절충으로 LightGBM을 선택했다. 시간 고려 결정은 비교 결과 확인 후 정했으며 정확도 동등성을 주장하지 않는다. 0.9 문턱은 HGB에서 이전했고 재튜닝하지 않았다. 같은 검증 표본을 Part I부터 재사용했으며 학습내 CV와 구분한다. 환경 피처는 같은 상자의 고유 은하 합집합을 사용한다.",')
    put(cells[117], text)
    # Embed the already measured CV evidence so the final notebook needs no extra result files.
    public_comparison = json.loads(json.dumps(comparison,ensure_ascii=False))
    public_comparison['decision'].pop('user_instruction',None)
    public_comparison['decision']['reason']='관측 오차 차이와 학습 시간을 절충해 LightGBM을 선택했다.'
    payload = json.dumps(public_comparison, ensure_ascii=False, allow_nan=False)
    comparison_code = "# 9.2 알고리즘 비교: 이미 측정한 7구성×5-fold의 기록. 새 학습이나 외부 val 채점은 하지 않는다.\n" + \
        "ALGORITHM_COMPARISON = json.loads(r'''" + payload + "''')\n" + \
        "assert ALGORITHM_COMPARISON['csv_sha256'] == REVISION_RUNTIME['csv_sha256']\n" + \
        "ALGORITHM_CV = pd.DataFrame(ALGORITHM_COMPARISON['records'])\n" + \
        "ALGORITHM_SUMMARY = pd.DataFrame(ALGORITHM_COMPARISON['summary']).set_index('model')\n" + \
        "assert len(ALGORITHM_CV) == 35 and not ALGORITHM_CV.duplicated(['model','fold']).any()\n" + \
        "display(ALGORITHM_SUMMARY[['mean_rmse_log','sd_rmse_log','mean_training_seconds']].sort_values('mean_rmse_log').round(6))\n" + \
        "print('선택:', ALGORITHM_COMPARISON['decision'])\n"
    nb['cells'][98:98] = [
        {"cell_type":"markdown","metadata":{},"id":"lightgbm-selection","source":["#### 9.2 알고리즘 비교와 학습 시간에 따른 최종 선택\n\n", "같은 34개 피처와 학습 내부 5-fold에서 측정한 기록이다. HGB 400회 두 단계의 평균 로그 RMSE 0.095340과 LightGBM의 0.095367은 관측 차이가 0.000027 dex였고, 평균 학습 시간은 13.44초와 6.76초였다. 최고 정확도 후보 대비 약 49.7% 짧은 관측 학습 시간을 선택 근거로 LightGBM 두 단계를 채택했다. 기존 조기 종료 HGB(6.75초)보다 빠르다는 주장은 하지 않는다. 시간·정확도 절충은 결과 확인 후 정했고, 동등성·비열등성 검정은 하지 않았다. 시간은 같은 컴퓨터·6스레드의 순차 실행 관측값이며 배경 작업을 완전히 통제하지 않았다. 알고리즘별 고정 설정 하나, 서로 다른 트리 구조와 총 학습량의 비교라는 제한이 있다.\n"]},
        {"cell_type":"code","metadata":{},"id":"lightgbm-comparison-evidence","source":comparison_code.splitlines(keepends=True),"outputs":[],"execution_count":None},
    ]
    nb['metadata'].setdefault('lightgbm_revision',{}).update({"date":"2026-11-12","original_notebook_sha256":oldhash,"selection":"LightGBM by observed accuracy/training-time tradeoff"})
    NB_PATH.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    print('Revised final notebook code and selection rationale; numerical discussion will be refreshed after execution.')


if __name__ == '__main__':
    main()

"""원본 보존·선택 로그 지표·순차 실행을 검사한다. 전체 노트북은 실행하지 않는다."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
FUNCTIONS = {"to_log", "from_log", "log_rmse", "log_scores", "selected_log_result_fields"}
BASE_KEYS = {"model_name", "features", "n_features", "random_seed", "validation_mae",
             "validation_rmse", "validation_r2", "zero_sfr_mae", "positive_sfr_mae", "training_seconds", "notes"}
LOG_KEYS = {"log_metric_definition", "validation_log_rmse", "zero_sfr_log_rmse",
            "positive_sfr_log_rmse", "validation_log_rmse_eps1", "validation_log_rmse_eps0.01"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source(cell):
    value = cell.get("source", "")
    require(isinstance(value, str) or isinstance(value, list) and all(isinstance(x, str) for x in value), "셀 source 형식 오류")
    return value if isinstance(value, str) else "".join(value)


def read_notebook(path):
    path = path.resolve()
    require(path.is_relative_to(ROOT), "작업 폴더 밖 노트북은 읽지 않는다")
    nb = json.loads(path.read_text(encoding="utf-8-sig"))
    require(nb.get("nbformat") == 4 and isinstance(nb.get("nbformat_minor"), int), "nbformat 4 형식 필요")
    require(isinstance(nb.get("metadata"), dict) and isinstance(nb.get("cells"), list), "노트북 기본 구조 오류")
    for c in nb["cells"]:
        require(c.get("cell_type") in {"code", "markdown", "raw"} and isinstance(c.get("metadata"), dict), "셀 구조 오류")
        source(c)
        if c["cell_type"] == "code":
            require(isinstance(c.get("outputs"), list) and (c.get("execution_count") is None or type(c["execution_count"]) is int), "코드 셀 구조 오류")
    return nb


def static_checks(nb, original):
    cells = nb["cells"]
    cursor = 0
    for old in original["cells"][:84]:
        while cursor < len(cells) and (cells[cursor]["cell_type"], source(cells[cursor])) != (old["cell_type"], source(old)):
            cursor += 1
        require(cursor < len(cells), "원본 Part I 84셀의 source/순서가 보존되지 않았다")
        cursor += 1
    fixed = [source(c) for c in original["cells"] if c["cell_type"] == "code" and "DO NOT MODIFY" in source(c)]
    remaining = [source(c) for c in cells if c["cell_type"] == "code"]
    for s in fixed:
        require(s in remaining, "DO NOT MODIFY 원본 코드가 누락/변경되었다")
        remaining.remove(s)
    nodes = [node for i, c in enumerate(cells) if c["cell_type"] == "code"
             for node in ast.parse(source(c), filename=f"cell_{i}").body]
    functions = {}
    results = []
    for node in nodes:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            require(node.name not in functions, f"평가 함수 중복: {node.name}")
            functions[node.name] = node
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RESULTS" for t in node.targets):
            results.append(node.value)
    require(set(functions) == FUNCTIONS, "필수 평가 함수 누락")
    require(len(results) == 1 and isinstance(results[0], ast.Dict), "RESULTS 단일 dict 정의 필요")
    keys = {k.value for k in results[0].keys if isinstance(k, ast.Constant)}
    require(BASE_KEYS <= keys, f"RESULTS 필수 키 누락: {sorted(BASE_KEYS - keys)}")
    spreads = [v for k, v in zip(results[0].keys, results[0].values) if k is None]
    require(len(spreads) == 1 and ast.unparse(spreads[0]) == "selected_log_result_fields(_ls, LOG_EPS)", "RESULTS 선택 지표 연결 불일치")
    require(not keys & LOG_KEYS, "RESULTS 선택 지표 중복 덮어쓰기")
    returns = [x for x in ast.walk(functions["selected_log_result_fields"]) if isinstance(x, ast.Return)]
    require(len(returns) == 1 and isinstance(returns[0].value, ast.Dict), "선택 결과 필드 반환 구조 오류")
    require(LOG_KEYS <= {k.value for k in returns[0].value.keys if isinstance(k, ast.Constant)}, "RESULTS 로그 필수 키 누락")
    template_fixed = sum(bool(re.match(r"# DO NOT MODIFY\b", s)) for s in fixed)
    print(f"정적 통과: JSON 기본 구조/코드 AST, Part I 84셀, DO NOT MODIFY 문구 {len(fixed)}셀 "
          f"(템플릿 고정 {template_fixed}셀+인용 {len(fixed)-template_fixed}셀), RESULTS 키/연결")
    return functions


def numerical_checks(functions):
    import numpy as np
    from sklearn.metrics import mean_absolute_error, r2_score
    y, pred = np.array([0, .005, 2, 30]), np.array([-.3, .1, 4, 20])
    def direct(truth, prediction, eps):
        return math.sqrt(math.fsum((math.log10(float(t) + eps) - math.log10(max(float(p), 0) + eps)) ** 2
                                  for t, p in zip(truth, prediction)) / len(truth))
    def close(a, b):
        require(math.isfinite(float(a)) and math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12), f"수치 불일치: {a} != {b}")
    for eps in (1.0, .01):
        ns = {"np": np, "LOG_EPS": eps, "y_val": y, "mean_absolute_error": mean_absolute_error,
              "r2_score": r2_score, "rmse": lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))}
        # 의존 순서로 지정한 다섯 함수만 실행하며 학습·입출력 코드는 추출하지 않는다.
        module = ast.Module(body=[functions[k] for k in ("to_log", "from_log", "log_rmse", "log_scores", "selected_log_result_fields")], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), "<평가 함수>", "exec"), ns)
        require(np.allclose(ns["from_log"](ns["to_log"](y)), y, rtol=1e-12, atol=1e-12), "로그 역변환 불일치")
        for scores in (ns["log_scores"](pred), ns["log_scores"](pred, y, eps)):
            fields = ns["selected_log_result_fields"](scores, eps)
            expected = {"selected log RMSE": direct(y, pred, eps), "D1 log RMSE": direct(y, pred, 1), "D2 log RMSE": direct(y, pred, .01),
                        "selected | SFR=0": direct(y[:1], pred[:1], eps), "selected | SFR>0": direct(y[1:], pred[1:], eps),
                        "linear MAE": mean_absolute_error(y, pred), "linear RMSE": ns["rmse"](y, pred), "linear R2": r2_score(y, pred)}
            for key, value in expected.items():
                close(scores[key], value)
            require(fields["log_metric_definition"] == f"log10(SFR + {eps:g})", "RESULTS 정의 문자열 불일치")
            for field, key in zip(("validation_log_rmse", "zero_sfr_log_rmse", "positive_sfr_log_rmse", "validation_log_rmse_eps1", "validation_log_rmse_eps0.01"), ("selected log RMSE", "selected | SFR=0", "selected | SFR>0", "D1 log RMSE", "D2 log RMSE")):
                close(fields[field], expected[key])
            require(not math.isclose(scores["linear MAE"], mean_absolute_error(y, np.clip(pred, 0, None))), "음수 예측의 선형 MAE 회귀 검출 실패")
            if eps == .01:
                close(fields["validation_log_rmse"], scores["D2 log RMSE"])
                require(not math.isclose(fields["validation_log_rmse"], scores["D1 log RMSE"]), "eps.01 주지표가 D1에 고정되었다")
    print("수치 통과: eps1/eps.01, 선택·D1·D2·부분집합·RESULTS·원래 선형 예측")


def plot_checks(nb):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    cell = next(c for c in nb["cells"] if source(c).startswith("# Starter diagnostic plots"))
    y = pd.Series([0.0, .005, 10.0])
    prediction = np.log10(np.array([0.0, 30.0, 0.0]) + .01)
    ns = {"np": np, "plt": plt, "LOG_EPS": .01, "y_val": y,
          "VAL_LOG_PREDICTIONS": prediction, "to_log": lambda a: np.log10(np.asarray(a) + .01)}
    show = plt.show
    try:
        plt.show = lambda: None
        exec(compile(source(cell), "<offset plot regression>", "exec"), ns)
        left, right = ns["axes"]
        require(left.get_xlim()[0] < -2 and left.get_ylim()[0] < -2, "eps.01의 0 SFR 로그 좌표가 잘린다")
        require("0.01" in left.get_xlabel() and "0.01" in left.get_ylabel() and "0.01" in right.get_xlabel(), "그림 레이블이 선택 eps와 다르다")
        residual = prediction - np.log10(y.to_numpy() + .01)
        require(right.get_xlim()[0] <= residual.min() and right.get_xlim()[1] >= residual.max(), "잔차 꼬리가 그림 범위에서 잘린다")
    finally:
        plt.show = show
        plt.close("all")
    print("그림 통과: eps.01 레이블·음수 로그 좌표·잔차 전체 범위")


def executed_checks(nb):
    counts, candidates = [], []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        if source(c).strip():
            counts.append(c.get("execution_count"))
        for output in c["outputs"]:
            require(output.get("output_type") != "error", "실행 오류 출력 존재")
            for raw in (output.get("text", ""), output.get("data", {}).get("text/plain", "")):
                raw = "".join(raw) if isinstance(raw, list) else raw
                for match in re.finditer(r"\{", raw):
                    try:
                        value, _ = json.JSONDecoder().raw_decode(raw[match.start():])
                    except ValueError:
                        continue
                    if isinstance(value, dict) and BASE_KEYS <= value.keys():
                        candidates.append(value)
    require(counts and counts == list(range(1, len(counts) + 1)), "코드 실행 횟수가 1부터 연속·비중복이어야 한다")
    require(len(candidates) == 1, "RESULTS 출력 JSON을 하나 찾을 수 있어야 한다")
    r = candidates[0]
    require(BASE_KEYS | LOG_KEYS <= r.keys(), "실행 RESULTS 필수 키 누락")
    match = re.fullmatch(r"log10\(SFR \+ ([0-9.eE+-]+)\)", r["log_metric_definition"])
    require(match is not None, "선택 로그 정의 형식 오류")
    eps = float(match[1])
    require(math.isfinite(eps) and eps > 0 and r.get("primary_metric_is_assumption") is True, "offset 또는 가정 표시 오류")
    audit = r["metric_audit"]
    for key, value in (("validation_log_rmse", audit["selected"]), ("validation_log_rmse_eps1", audit["D1"]), ("validation_log_rmse_eps0.01", audit["D2"])):
        require(math.isfinite(r[key]) and math.isclose(r[key], value, rel_tol=1e-12, abs_tol=1e-12), "RESULTS metric_audit 불일치")
    if eps in (1, .01):
        require(math.isclose(audit["selected"], audit["D1" if eps == 1 else "D2"], rel_tol=1e-12, abs_tol=1e-12), "선택 offset 값 불일치")
    runtime = r["reproducibility"]
    require(r.get("source_commit") == runtime.get("source_commit") == "c8ae0de", "원본 커밋 표기 불일치")
    csvs = list(ROOT.glob("*.csv"))
    require(len(csvs) == 1 and runtime.get("csv_name") == csvs[0].name, "CSV 식별 불일치")
    with csvs[0].open("rb") as stream:
        require(runtime.get("csv_sha256") == hashlib.file_digest(stream, "sha256").hexdigest(), "CSV SHA256 불일치")
    require(r["n_features"] == 34 and len(r["features"]) == len(set(r["features"])) == 34 and r["stage2_regressor_features"] == r["features"], "34/34 피처 계약 불일치")
    checks = r["prediction_contract_checks"]
    require(all(checks.get(k) is True for k in ("validation_reproduction", "row_order", "fixed_union_batches", "duplicate_null_missing_conflicting_ID_rejected")), "입력 계약 검사 미통과")
    print(f"실행 통과: 연속 {len(counts)}셀, 오류 없음, RESULTS/offset/CSV SHA256/34피처/입력계약")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=ROOT / "DataHub_701104_final.ipynb")
    parser.add_argument("--static", action="store_true", help="표준 라이브러리 정적 검사만 수행")
    parser.add_argument("--executed", action="store_true", help="저장된 순차 실행 출력도 확인")
    args = parser.parse_args()
    nb = read_notebook(args.notebook)
    functions = static_checks(nb, read_notebook(ROOT / "DataHub_701104_mid.ipynb"))
    if not args.static:
        numerical_checks(functions)
        plot_checks(nb)
    if args.executed:
        executed_checks(nb)
    print("검증 완료: " + ("정적" if args.static else "정적+수치") + ("+실행 출력" if args.executed else " (실행 출력 미검사)"))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"검증 실패: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

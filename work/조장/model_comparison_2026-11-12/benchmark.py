"""HGB / LightGBM / CatBoost 고정 예산 비교 (학습 분할 내부 5-fold만, 외부 val 미채점).

설정은 같은 폴더의 protocol.json에 고정되어 있다. 이 스크립트는 protocol.json을 읽고 해시만 남기며 다시 쓰지 않는다.
import만으로는 아무 실험도 실행하지 않는다.

    python benchmark.py                 # 7개 모델 전체 실행 -> results/
    python benchmark.py --smoke         # 3000행, 2-fold, 20회 점검 -> smoke_results/
    python benchmark.py --models a,b    # 일부 모델만 (진단용, 완료 기록에 미완료로 남음)
    python benchmark.py --resume        # 같은 지문(fingerprint)의 기존 출력에서 미완료 모델만 다시 계산
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
PROTOCOL_PATH = SCRIPT_DIR / "protocol.json"

MODEL_NAMES = (
    "hgb_single_fixed400",
    "hgb_two_stage_fixed400",
    "lightgbm_single_fixed400",
    "lightgbm_two_stage_fixed400",
    "catboost_single_fixed400",
    "catboost_two_stage_fixed400",
    "hgb_two_stage_reference",
)
REFERENCE = "hgb_two_stage_reference"
CONTROLLED_HGB = {"single": "hgb_single_fixed400", "two_stage": "hgb_two_stage_fixed400"}
LOG_OFFSET = 1.0  # z = log10(SFR + 1)

LIBRARY_MODULES = {"sklearn": "sklearn.ensemble", "lightgbm": "lightgbm", "catboost": "catboost"}
ITER_PARAM = {"sklearn": "max_iter", "lightgbm": "n_estimators", "catboost": "iterations"}
THREAD_PARAM = {"lightgbm": "n_jobs", "catboost": "thread_count"}

FOLD_COLUMNS = [
    "model", "fold", "n_train", "n_eval", "rmse_log", "rmse_log_zero", "rmse_log_positive", "rmse_log_high_gt100",
    "zero_recall", "zero_precision", "positive_predicted_zero_count", "training_seconds", "prediction_seconds",
    "classifier_iterations", "regressor_iterations",
]


def log(message):
    print(message, flush=True)


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_ids(ids):
    return sha256_bytes(np.ascontiguousarray(np.asarray(ids, dtype=np.int64)).tobytes())


def write_atomic_text(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path, obj):
    write_atomic_text(path, json.dumps(obj, ensure_ascii=False, indent=2, default=str, allow_nan=True) + "\n")


def write_csv(path, frame):
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="고정 예산 HGB/LightGBM/CatBoost 비교 (학습 내부 CV)")
    parser.add_argument("--root", type=Path, default=None, help="저장소 루트 (기본: 스크립트 기준 parents[3])")
    parser.add_argument("--output", type=Path, default=None, help="출력 폴더 (기본: results/, smoke는 smoke_results/)")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--smoke", action="store_true", help="3000행 · 2-fold · 20회 코드 경로 점검")
    parser.add_argument("--models", type=str, default=None, help="쉼표로 구분한 모델 이름 (기본: 7개 전체)")
    parser.add_argument("--folds", type=int, default=None, help="전체 실행은 5만, smoke는 2만 허용")
    parser.add_argument("--resume", action="store_true", help="기존 출력이 같은 지문일 때 미완료 모델만 계산")
    return parser.parse_args(argv)


# ---------------------------------------------------------------- 원본 함수 추출 / 데이터

def extract_notebook_functions(notebook_path, names, namespace):
    """노트북 코드 셀에서 지정한 최상위 def만 ast로 꺼내 namespace 안에서 정의한다. 다른 셀 코드는 실행하지 않는다."""
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = {}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        text = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        try:
            tree = ast.parse(text)
        except SyntaxError:  # IPython 매직 등
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                if node.name in sources:
                    raise RuntimeError(f"노트북에 {node.name} 정의가 둘 이상 있다")
                sources[node.name] = ast.get_source_segment(text, node)
    missing = [n for n in names if n not in sources]
    if missing:
        raise RuntimeError(f"노트북에서 함수를 찾지 못했다: {missing}")
    for name in names:
        exec(compile(sources[name], f"<{notebook_path.name}:{name}>", "exec"), namespace)
    return ({n: namespace[n] for n in names},
            {n: {"sha256": sha256_bytes(sources[n].encode("utf-8")), "n_lines": sources[n].count("\n") + 1} for n in names})


def load_data(root, protocol):
    spec = protocol["data"]
    data_path = root / protocol["source"]["data_file"]
    df = pd.read_csv(data_path)
    id_col, target = spec["id_column"], spec["target_column"]
    inputs = [c for c in df.columns if c not in (id_col, target)]
    if inputs != spec["original_features"]:
        raise ValueError(f"CSV 입력 열이 protocol과 다르다: {inputs}")
    if len(df) != spec["expected_rows"]:
        raise ValueError(f"행 수 {len(df)} != {spec['expected_rows']}")
    if not isinstance(df.index, pd.RangeIndex):
        raise ValueError("CSV index가 RangeIndex가 아니다")
    if not df[id_col].is_unique:
        raise ValueError("subhalo_id가 고유하지 않다")
    y = df[target].astype(float)
    if not (np.isfinite(y).all() and (y >= 0).all()):
        raise ValueError("타깃은 유한하고 0 이상이어야 한다")
    return data_path, df


def build_features(df, protocol, funcs):
    spec = protocol["data"]
    base_cols = spec["original_features"]
    base = df[base_cols]
    # 최종 노트북 9.2와 같은 순서: 13개 입력으로 사전식 정렬한 뒤 계산하고 원래 index로 되돌린다 (동률 이웃 안정화).
    env_order = base.sort_values(base_cols, kind="mergesort", na_position="last").index
    env = funcs["build_environment_features"](base.loc[env_order]).reindex(df.index)
    mass = funcs["build_mass_history_features"](base)
    if list(env.columns) != spec["environment_features"]:
        raise ValueError(f"환경 피처 이름/순서가 protocol과 다르다: {list(env.columns)}")
    if list(mass.columns) != spec["mass_history_features"]:
        raise ValueError(f"질량 피처 이름/순서가 protocol과 다르다: {list(mass.columns)}")
    X = pd.concat([base, env, mass], axis=1)
    features = base_cols + spec["environment_features"] + spec["mass_history_features"]
    assert list(X.columns) == features and len(features) == 34
    values = X.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError(f"무한대 값이 있는 열: {list(X.columns[np.isinf(values).any(axis=0)])}")
    nan_counts = {c: int(v) for c, v in X.isna().sum().items() if v}
    bad = sorted(set(nan_counts) - set(spec["allowed_nan_features"]))
    if bad:
        raise ValueError(f"허용되지 않은 NaN 열: {bad}")
    return X, features, nan_counts, sha256_ids(env_order.to_numpy())


# ---------------------------------------------------------------- 모델

def estimator_class(library, class_name):
    return getattr(importlib.import_module(LIBRARY_MODULES[library]), class_name)


def make_estimator(library, part, iterations, threads):
    params = dict(part["params"])
    if iterations is not None:
        params[ITER_PARAM[library]] = iterations
    if library in THREAD_PARAM:
        params[THREAD_PARAM[library]] = threads
    return estimator_class(library, part["class"])(**params)


def fitted_iterations(estimator, library):
    if library == "sklearn":
        return int(estimator.n_iter_)
    if library == "lightgbm":
        return int(estimator.booster_.current_iteration())
    return int(estimator.tree_count_)


def fit_predict(spec, Xa, ya, Xb, iterations, threads, tau):
    """선형 SFR 예측(0 이상 절단)과 시간·트리 수를 돌려준다. 채점 fold(Xb)는 학습에 쓰지 않는다."""
    library = spec["library"]
    t0 = time.perf_counter()
    if spec["kind"] == "single":
        reg = make_estimator(library, spec["regressor"], iterations, threads).fit(Xa, np.log10(ya + LOG_OFFSET))
        train_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        z = np.asarray(reg.predict(Xb), dtype=float)
        clf_iter = 0
    else:
        pos = ya > 0
        clf = make_estimator(library, spec["classifier"], iterations, threads).fit(Xa, pos.astype(int))
        if [int(c) for c in clf.classes_] != [0, 1]:
            raise RuntimeError(f"분류기 classes_ 이상: {clf.classes_}")
        rows = np.flatnonzero(pos)
        reg = make_estimator(library, spec["regressor"], iterations, threads).fit(
            Xa.iloc[rows], np.log10(ya[rows] + LOG_OFFSET))
        train_seconds = time.perf_counter() - t0
        t1 = time.perf_counter()
        p = np.asarray(clf.predict_proba(Xb), dtype=float)[:, 1]
        m = np.asarray(reg.predict(Xb), dtype=float)
        z = p * m + (1 - p) * 0.0  # z(0) = log10(0 + 1) = 0
        z = np.where(1 - p >= tau, 0.0, z)
        clf_iter = fitted_iterations(clf, library)
    yhat = np.maximum(10 ** z - LOG_OFFSET, 0.0)
    return yhat, train_seconds, time.perf_counter() - t1, clf_iter, fitted_iterations(reg, library)


def template_params(spec, iterations, threads):
    """manifest 기록용: 학습 전 같은 설정으로 만든 추정기의 get_params()."""
    out = {}
    for part in ("classifier", "regressor"):
        if part in spec:
            est = make_estimator(spec["library"], spec[part], iterations, threads)
            out[part] = {"class": spec[part]["class"], "protocol_params": spec[part]["params"],
                         "get_params": est.get_params()}
    return out


# ---------------------------------------------------------------- 지표

def score(y, yhat, high_threshold):
    lt, lp = np.log10(y + LOG_OFFSET), np.log10(yhat + LOG_OFFSET)

    def rmse(mask):
        return float(np.sqrt(np.mean((lt[mask] - lp[mask]) ** 2))) if mask.any() else float("nan")

    zero, out_zero = y == 0, yhat == 0
    return {
        "rmse_log": rmse(np.ones(len(y), dtype=bool)),
        "rmse_log_zero": rmse(zero),
        "rmse_log_positive": rmse(~zero),
        "rmse_log_high_gt100": rmse(y > high_threshold),
        "zero_recall": float(np.mean(out_zero[zero])) if zero.any() else float("nan"),
        "zero_precision": float(np.mean(zero[out_zero])) if out_zero.any() else float("nan"),
        "positive_predicted_zero_count": int((out_zero & ~zero).sum()),
    }


def mean_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def sd_or_nan(values):
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1)) if len(values) > 1 else float("nan")


# ---------------------------------------------------------------- 출력 상태

def load_existing_state(output, fingerprint, n_rows):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    diff = sorted(k for k in fingerprint if manifest.get("fingerprint", {}).get(k) != fingerprint[k])
    if diff:
        raise SystemExit(f"--resume 거부: 기존 manifest와 지문이 다르다 {diff}")
    folds = pd.read_csv(output / "fold_results.csv") if (output / "fold_results.csv").exists() else pd.DataFrame(columns=FOLD_COLUMNS)
    oof = {}
    if (output / "oof_predictions.npz").exists():
        with np.load(output / "oof_predictions.npz") as data:
            for key in data.files:
                if key.startswith("yhat_linear_"):
                    arr = data[key].astype(float)
                    if arr.shape != (n_rows,):
                        raise SystemExit(f"--resume 거부: {key} 길이가 다르다")
                    oof[key[len("yhat_linear_"):]] = arr
    return manifest, folds, oof


def model_complete(name, folds, oof, n_splits):
    return (name in oof and np.isfinite(oof[name]).all()
            and sorted(folds.loc[folds.model == name, "fold"].astype(int)) == list(range(n_splits)))


def save_oof(output, base_arrays, oof):
    tmp = output / "oof_predictions.tmp.npz"
    arrays = dict(base_arrays)
    arrays["models"] = np.array([m for m in MODEL_NAMES if m in oof], dtype=str)
    for name in MODEL_NAMES:
        if name in oof:
            arrays[f"yhat_linear_{name}"] = oof[name]
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, output / "oof_predictions.npz")


# ---------------------------------------------------------------- 요약

def summarize(protocol, folds, oof, y_train, fold_id, test_sets, n_splits, smoke):
    high = protocol["target"]["high_sfr_threshold_linear"]
    complete = [m for m in MODEL_NAMES if model_complete(m, folds, oof, n_splits)]
    rows = []
    for name in MODEL_NAMES:
        f = folds[folds.model == name].sort_values("fold")
        if f.empty:
            continue
        spec = protocol["models"][name]
        row = {"model": name, "library": spec["library"], "kind": spec["kind"], "n_folds": len(f),
               "complete": name in complete,
               "mean_rmse_log": float(f.rmse_log.mean()), "sd_rmse_log": sd_or_nan(f.rmse_log),
               "min_rmse_log": float(f.rmse_log.min()), "max_rmse_log": float(f.rmse_log.max())}
        for col in ("rmse_log_zero", "rmse_log_positive", "rmse_log_high_gt100", "zero_recall", "zero_precision"):
            row[f"mean_{col}"] = mean_or_nan(f[col])
        row.update({"total_positive_predicted_zero": int(f.positive_predicted_zero_count.sum()),
                    "mean_training_seconds": float(f.training_seconds.mean()),
                    "total_training_seconds": float(f.training_seconds.sum()),
                    "mean_prediction_seconds": float(f.prediction_seconds.mean()),
                    "mean_classifier_iterations": float(f.classifier_iterations.mean()),
                    "mean_regressor_iterations": float(f.regressor_iterations.mean())})
        if name in complete:
            row.update({f"oof_{k}": v for k, v in score(y_train, oof[name], high).items()})
        rows.append(row)
    summary = pd.DataFrame(rows)

    # 짝지은 fold 차이: 후보 - 기준 (같은 fold 번호끼리)
    pairs = []
    for name in MODEL_NAMES:
        if name == REFERENCE:
            continue
        pairs.append((name, REFERENCE, "reference_final_notebook_hgb"))
        spec = protocol["models"][name]
        if spec["library"] != "sklearn":
            pairs.append((name, CONTROLLED_HGB[spec["kind"]], "controlled_hgb_same_kind"))
    paired_rows, paired_summary = [], {}
    min_wins = protocol["selection_rule"]["min_fold_wins"] if n_splits == 5 else n_splits
    for cand, base, role in pairs:
        a = folds[folds.model == cand].set_index("fold").rmse_log
        b = folds[folds.model == base].set_index("fold").rmse_log
        common = sorted(set(a.index) & set(b.index))
        if not common:
            continue
        label = f"{cand} - {base}"
        deltas = [float(a[k] - b[k]) for k in common]
        for k, d in zip(common, deltas):
            paired_rows.append({"comparison": label, "candidate": cand, "baseline": base, "baseline_role": role,
                                "fold": int(k), "candidate_rmse_log": float(a[k]), "baseline_rmse_log": float(b[k]),
                                "delta_rmse_log": d, "candidate_better": d < 0})
        both_complete = cand in complete and base in complete
        mean_delta = float(np.mean(deltas))
        wins = int(sum(d < 0 for d in deltas))
        entry = {"candidate": cand, "baseline": base, "baseline_role": role, "n_folds": len(common),
                 "both_complete": both_complete, "mean_delta_rmse_log": mean_delta, "sd_delta_rmse_log": sd_or_nan(deltas),
                 "candidate_better_folds": wins, "baseline_better_folds": int(sum(d > 0 for d in deltas)),
                 "tied_folds": int(sum(d == 0 for d in deltas))}
        if role == "reference_final_notebook_hgb":
            entry["promising"] = bool(both_complete and mean_delta < 0 and wins >= min_wins)
            entry["promising_rule"] = f"mean delta < 0 and candidate_better_folds >= {min_wins}/{n_splits}"
        paired_summary[label] = entry
    paired = pd.DataFrame(paired_rows)

    if not summary.empty:
        summary["rank_mean_rmse_log"] = summary.mean_rmse_log.where(summary.complete).rank(method="min")
        summary["promising_vs_reference"] = [
            paired_summary.get(f"{m} - {REFERENCE}", {}).get("promising", False) if m != REFERENCE else False
            for m in summary.model]

    # OOF 일관성: 모든 행이 정확히 한 번 채점되고, fold 가중 RMSE가 OOF RMSE와 같아야 한다.
    coverage = np.bincount(np.concatenate(test_sets), minlength=len(y_train))
    checks = {"every_row_scored_once": bool((coverage == 1).all()),
              "fold_id_matches_test_sets": bool(all((fold_id[t] == k).all() for k, t in enumerate(test_sets)))}
    for name in complete:
        f = folds[folds.model == name].sort_values("fold")
        pooled = math.sqrt(float((f.n_eval * f.rmse_log ** 2).sum()) / float(f.n_eval.sum()))
        oof_rmse = float(summary.loc[summary.model == name, "oof_rmse_log"].iloc[0])
        checks[name] = {"finite_nonnegative": bool(np.isfinite(oof[name]).all() and (oof[name] >= 0).all()),
                        "shape": list(oof[name].shape), "oof_rmse_log": oof_rmse, "pooled_fold_rmse_log": pooled,
                        "pooled_matches_oof": bool(abs(pooled - oof_rmse) <= 1e-10)}

    reproduction = None
    expected = protocol["reference_expected"]
    if not smoke and REFERENCE in complete:
        f = folds[folds.model == REFERENCE].sort_values("fold")
        diffs = [abs(float(o) - e) for o, e in zip(f.rmse_log, expected["fold_rmse_log"])]
        reproduction = {"expected_mean_rmse_log": expected["mean_rmse_log"], "observed_mean_rmse_log": float(f.rmse_log.mean()),
                        "expected_fold_rmse_log": expected["fold_rmse_log"], "observed_fold_rmse_log": [float(v) for v in f.rmse_log],
                        "max_abs_fold_diff": max(diffs), "abs_tolerance_per_fold": expected["abs_tolerance_per_fold"],
                        "within_tolerance": bool(max(diffs) <= expected["abs_tolerance_per_fold"]),
                        "expected_classifier_iterations": expected["fold_classifier_iterations"],
                        "observed_classifier_iterations": [int(v) for v in f.classifier_iterations],
                        "expected_regressor_iterations": expected["fold_regressor_iterations"],
                        "observed_regressor_iterations": [int(v) for v in f.regressor_iterations]}
    return summary, paired, paired_summary, complete, checks, reproduction


# ---------------------------------------------------------------- 실행

def git_info(root):
    def run(*cmd):
        try:
            return subprocess.run(["git", "-C", str(root), *cmd], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception as exc:  # git 없음 등
            return f"unavailable: {exc}"
    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_entries": len([s for s in status.splitlines() if s.strip()]) if not status.startswith("unavailable") else None}


def library_versions(libraries):
    import scipy
    import sklearn
    import threadpoolctl
    out = {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
           "sklearn": sklearn.__version__, "threadpoolctl": threadpoolctl.__version__}
    for lib in ("lightgbm", "catboost"):
        if lib in libraries:
            out[lib] = importlib.import_module(lib).__version__
    return out


def main(argv=None):
    args = parse_args(argv)
    started = utc_now()
    wall0 = time.perf_counter()
    root = (args.root or SCRIPT_PATH.parents[3]).resolve()
    protocol_bytes = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    if set(protocol["models"]) != set(MODEL_NAMES):
        raise SystemExit("protocol.json 모델 목록이 스크립트와 다르다")
    tau = float(protocol["two_stage"]["zero_threshold"])
    if tau != 0.9:
        raise SystemExit("protocol zero_threshold는 0.9로 고정되어 있어야 한다")
    if args.threads < 1:
        raise SystemExit("--threads는 1 이상")

    if args.models:
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = sorted(set(requested) - set(MODEL_NAMES))
        if unknown:
            raise SystemExit(f"알 수 없는 모델: {unknown}")
        requested = [m for m in MODEL_NAMES if m in requested]
    else:
        requested = list(MODEL_NAMES)

    smoke = bool(args.smoke)
    n_splits = protocol["smoke"]["n_splits"] if smoke else protocol["internal_cv"]["n_splits"]
    if args.folds is not None and args.folds != n_splits:
        raise SystemExit(f"--folds는 {'smoke' if smoke else '전체'} 실행에서 {n_splits}만 허용된다")
    iterations = protocol["smoke"]["iterations"] if smoke else None  # None이면 protocol 값(400) 그대로

    full_default = (SCRIPT_DIR / protocol["full_default_output_subdir"]).resolve()
    smoke_default = (SCRIPT_DIR / protocol["smoke"]["default_output_subdir"]).resolve()
    output = (args.output.resolve() if args.output else (smoke_default if smoke else full_default))
    if smoke and output == full_default:
        raise SystemExit("smoke 실행은 전체 결과 폴더에 쓸 수 없다")
    if not smoke and output == smoke_default:
        raise SystemExit("전체 실행은 smoke_results 폴더에 쓸 수 없다")
    if (output / "manifest.json").exists():
        old = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if bool(old.get("smoke")) != smoke:
            raise SystemExit(f"출력 폴더의 기존 manifest smoke={old.get('smoke')}와 이번 실행 smoke={smoke}가 다르다")
        if not args.resume:
            raise SystemExit(f"{output}에 기존 결과가 있다. 이어서 하려면 --resume, 아니면 다른 --output을 쓴다")
    output.mkdir(parents=True, exist_ok=True)

    libraries = sorted({protocol["models"][m]["library"] for m in requested})
    for lib in libraries:  # 학습 전에 import 실패를 드러낸다
        importlib.import_module(LIBRARY_MODULES[lib])

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    from sklearn.model_selection import KFold, train_test_split
    from threadpoolctl import threadpool_info, threadpool_limits

    with threadpool_limits(limits=args.threads):
        log(f"[setup] root={root} output={output} smoke={smoke} threads={args.threads} models={requested}")
        data_path, df = load_data(root, protocol)
        notebook_path = root / protocol["source"]["notebook_file"]
        namespace = {"np": np, "pd": pd, "cKDTree": cKDTree, "coo_matrix": coo_matrix,
                     "connected_components": connected_components, **protocol["source"]["environment_globals"]}
        funcs, function_hashes = extract_notebook_functions(notebook_path, protocol["source"]["extracted_functions"], namespace)
        t0 = time.perf_counter()
        X, features, nan_counts, env_order_hash = build_features(df, protocol, funcs)
        log(f"[setup] 피처 {len(features)}개 계산 {time.perf_counter() - t0:.1f} s (CSV 전체 {len(df):,}행 입력 기하)")

        id_col, target = protocol["data"]["id_column"], protocol["data"]["target_column"]
        y = df[target].astype(float)
        split = protocol["outer_split"]
        train_ids, val_ids = train_test_split(df.index.to_numpy(), test_size=split["test_size"],
                                              random_state=split["random_state"], stratify=(y > 0).astype(int))
        if not smoke and len(train_ids) != split["expected_train_rows"]:
            raise ValueError(f"외부 학습 행 수 {len(train_ids)} != {split['expected_train_rows']}")
        n_outer_train = len(train_ids)
        if smoke:
            sub, _ = train_test_split(np.arange(len(train_ids)), train_size=protocol["smoke"]["max_train_rows"],
                                      random_state=42, stratify=(y.iloc[train_ids].to_numpy() > 0).astype(int))
            train_ids = train_ids[sub]
        # 외부 val 행(val_ids)의 타깃은 이후 어디에서도 읽지 않는다.
        X_train = X.iloc[train_ids].reset_index(drop=True)
        y_train = y.iloc[train_ids].to_numpy(dtype=float)
        subhalo_ids = df[id_col].iloc[train_ids].to_numpy()
        assert not set(train_ids.tolist()) & set(val_ids.tolist())

        cv = protocol["internal_cv"]
        folds_idx = list(KFold(n_splits=n_splits, shuffle=cv["shuffle"], random_state=cv["random_state"]).split(X_train))
        fold_id = np.full(len(y_train), -1, dtype=np.int64)
        for k, (tr, te) in enumerate(folds_idx):
            assert not set(tr.tolist()) & set(te.tolist())
            fold_id[te] = k
        test_sets = [te for _, te in folds_idx]
        high = protocol["target"]["high_sfr_threshold_linear"]

        fingerprint = {
            "smoke": smoke, "protocol_sha256": sha256_bytes(protocol_bytes),
            "library_versions": library_versions(["sklearn", "lightgbm", "catboost"]),
            "data_sha256": sha256_bytes(data_path.read_bytes()), "notebook_sha256": sha256_bytes(notebook_path.read_bytes()),
            "script_sha256": sha256_bytes(SCRIPT_PATH.read_bytes()), "extracted_functions": function_hashes,
            "features": features, "environment_order_sha256": env_order_hash,
            "feature_frame_hash": sha256_bytes(pd.util.hash_pandas_object(X, index=True).to_numpy().tobytes()),
            "train_ids_sha256_ordered": sha256_ids(train_ids), "outer_val_ids_sha256_ordered": sha256_ids(val_ids),
            "folds": [{"fold": k, "n_train": len(tr), "n_test": len(te),
                       "train_original_ids_sha256": sha256_ids(train_ids[tr]),
                       "test_original_ids_sha256": sha256_ids(train_ids[te])} for k, (tr, te) in enumerate(folds_idx)],
            "n_splits": n_splits, "iterations_override": iterations, "threads": args.threads,
        }

        resumed = False
        folds_df = pd.DataFrame(columns=FOLD_COLUMNS)
        oof = {}
        old_manifest = {}
        if args.resume and (output / "manifest.json").exists():
            old_manifest, folds_df, oof = load_existing_state(output, fingerprint, len(y_train))
            resumed = True
            for name in requested:
                if not model_complete(name, folds_df, oof, n_splits):
                    folds_df = folds_df[folds_df.model != name]
                    oof.pop(name, None)
        fold_rows = folds_df[FOLD_COLUMNS].to_dict(orient="records")

        manifest = {
            "protocol_id": protocol["protocol_id"], "smoke": smoke, "started_utc": old_manifest.get("started_utc", started),
            # 버전은 지문에도 포함해 서로 다른 환경의 결과가 섞이는 이어 실행을 거부한다.
            "runs": old_manifest.get("runs", []) + [{
                "started_utc": started, "requested_models": requested, "resume": resumed,
                "argv": sys.argv[1:] if argv is None else list(argv), "library_versions": library_versions(libraries)}],
            "fingerprint": fingerprint, "git": git_info(root), "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "threadpool_info": threadpool_info(),
            "paths": {"root": str(root), "data": data_path.name, "notebook": notebook_path.name,
                      "script": str(SCRIPT_PATH.relative_to(root)) if SCRIPT_PATH.is_relative_to(root) else str(SCRIPT_PATH),
                      "protocol": PROTOCOL_PATH.name, "output": str(output)},
            "rows": {"csv": len(df), "outer_train": n_outer_train, "outer_val_unscored": len(val_ids),
                     "used_for_cv": len(y_train), "used_zero": int((y_train == 0).sum()),
                     "used_positive": int((y_train > 0).sum()), "used_high_gt100": int((y_train > high).sum()),
                     "fold_positive_train": [int((y_train[tr] > 0).sum()) for tr, _ in folds_idx]},
            "feature_nan_counts_all_rows": nan_counts,
            "target": protocol["target"], "zero_threshold": tau,
            "model_params": {**old_manifest.get("model_params", {}),
                             **{m: {"library": protocol["models"][m]["library"], "kind": protocol["models"][m]["kind"],
                                    **template_params(protocol["models"][m], iterations, args.threads)} for m in requested}},
            "caveats": protocol["caveats"] + ([protocol["smoke"]["note"]] if smoke else []),
        }
        write_json(output / "manifest.json", manifest)
        (output / "protocol_snapshot.json").write_bytes(protocol_bytes)
        base_arrays = {"train_indices": train_ids.astype(np.int64), "subhalo_id": subhalo_ids,
                       "y_true_linear": y_train, "y_true_log": np.log10(y_train + LOG_OFFSET), "fold_id": fold_id}
        for name in requested:
            if name not in oof:
                oof[name] = np.full(len(y_train), np.nan)
        log(f"[setup] manifest 저장. 학습 {len(y_train):,}행, {n_splits}-fold, 반복 {iterations or 400}")

        for name in requested:
            if resumed and model_complete(name, folds_df, oof, n_splits):
                log(f"[{name}] 기존 결과 완료 — 건너뜀")
                continue
            spec = protocol["models"][name]
            pred = np.full(len(y_train), np.nan)
            for k, (tr, te) in enumerate(folds_idx):
                yhat, train_s, pred_s, clf_iter, reg_iter = fit_predict(
                    spec, X_train.iloc[tr], y_train[tr], X_train.iloc[te], iterations, args.threads, tau)
                if not (np.isfinite(yhat).all() and (yhat >= 0).all()):
                    raise RuntimeError(f"{name} fold {k}: 예측이 유한한 0 이상 값이 아니다")
                if not smoke and "fixed400" in name:
                    assert reg_iter == 400, (name, "regressor", reg_iter)
                    if spec["kind"] == "two_stage":
                        assert clf_iter == 400, (name, "classifier", clf_iter)
                pred[te] = yhat
                row = {"model": name, "fold": k, "n_train": len(tr), "n_eval": len(te),
                       **score(y_train[te], yhat, high), "training_seconds": train_s, "prediction_seconds": pred_s,
                       "classifier_iterations": clf_iter, "regressor_iterations": reg_iter}
                fold_rows.append(row)
                folds_df = pd.DataFrame(fold_rows, columns=FOLD_COLUMNS)
                write_csv(output / "fold_results.csv", folds_df)
                oof[name] = pred
                save_oof(output, base_arrays, oof)
                log(f"[{name}] fold {k + 1}/{n_splits}: rmse_log={row['rmse_log']:.6f} "
                    f"train={train_s:.1f}s pred={pred_s:.2f}s trees(clf/reg)={clf_iter}/{reg_iter}")
            oof[name] = pred
            save_oof(output, base_arrays, oof)
            log(f"[{name}] 완료, OOF 저장")

        summary, paired, paired_summary, complete, checks, reproduction = summarize(
            protocol, folds_df, oof, y_train, fold_id, test_sets, n_splits, smoke)

    write_csv(output / "summary.csv", summary)
    write_csv(output / "paired_differences.csv", paired)
    write_json(output / "paired_summary.json", paired_summary)
    ranking = (summary[summary.complete].sort_values("mean_rmse_log")[["model", "mean_rmse_log", "sd_rmse_log"]]
               .to_dict(orient="records") if not summary.empty else [])
    result = {
        "protocol_id": protocol["protocol_id"], "smoke": smoke,
        "protocol_sha256": fingerprint["protocol_sha256"], "script_sha256": fingerprint["script_sha256"],
        "completion": {"started_utc": started, "finished_utc": utc_now(), "wall_seconds": time.perf_counter() - wall0,
                       "requested_models": requested, "completed_models": complete,
                       "missing_models": [m for m in MODEL_NAMES if m not in complete],
                       "all_models_complete": len(complete) == len(MODEL_NAMES), "resumed": resumed},
        "evaluation_scope": "외부 학습 분할 내부 CV의 OOF만 채점. 전체 타깃의 0 여부는 공통 층화 분할에만 사용하며 외부 val 타깃은 모델 학습·후보 채점에 쓰지 않았다.",
        "selection_rule": protocol["selection_rule"],
        "ranking_by_mean_rmse_log": ranking,
        "models": summary.to_dict(orient="records"),
        "paired": paired_summary,
        "oof_consistency": checks,
        "reference_reproduction": reproduction,
        "claims": {"hyperparameter_tuning": "없음 (protocol 고정값만 사용)",
                   "global_optimality": "주장하지 않음",
                   "significance": "fold가 독립이 아니므로 p값·신뢰구간을 계산하지 않음",
                   "equal_compute_budget": "주장하지 않음 (single/two_stage 트리 수와 라이브러리 트리 구조가 다름)",
                   "final_model_replaced": False},
    }
    write_json(output / "summary.json", result)
    log(f"[done] 완료 모델 {len(complete)}/{len(MODEL_NAMES)} -> {output}")
    if reproduction is not None and not reproduction["within_tolerance"]:
        log(f"[warn] reference 재현 차이 {reproduction['max_abs_fold_diff']:.2e} > 허용 {reproduction['abs_tolerance_per_fold']:g}")
    return result


if __name__ == "__main__":
    main()

"""Build a report-ready comparison from independently validated full results."""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
LABELS = {
    "hgb_single_fixed400": "HGB 단일 회귀 · 400회",
    "hgb_two_stage_fixed400": "HGB 두 단계 · 400회",
    "lightgbm_single_fixed400": "LightGBM 단일 회귀 · 400회",
    "lightgbm_two_stage_fixed400": "LightGBM 두 단계 · 400회",
    "catboost_single_fixed400": "CatBoost 단일 회귀 · 400회",
    "catboost_two_stage_fixed400": "CatBoost 두 단계 · 400회",
    "hgb_two_stage_reference": "기존 제출 HGB 두 단계 · 조기 종료",
}


def main():
    results = HERE / "results"
    validation = json.loads((results / "validation.json").read_text(encoding="utf-8"))
    assert validation["passed"]
    summary = pd.read_csv(results / "summary.csv").set_index("model")
    folds = pd.read_csv(results / "fold_results.csv")
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    reference = "hgb_two_stage_reference"
    baseline = float(summary.loc[reference, "mean_rmse_log"])
    ref_folds = folds[folds.model == reference].set_index("fold").rmse_log
    best = summary.mean_rmse_log.idxmin()
    best_score = float(summary.loc[best, "mean_rmse_log"])
    improvement = 100 * (baseline - best_score) / baseline
    revision = HERE.parent / "lightgbm_report_revision_2026-11-12"
    chosen = summary.loc["lightgbm_two_stage_fixed400"] if (revision / "selection_decision.json").exists() else None
    if chosen is not None:
        fixed_hgb = summary.loc["hgb_two_stage_fixed400"]
        selection_lead = (
            f"**최종 선택은 LightGBM 두 단계 모델입니다.** HGB 400회 두 단계의 평균 로그 RMSE는 {fixed_hgb.mean_rmse_log:.6f}, "
            f"LightGBM은 {chosen.mean_rmse_log:.6f}로 관측 차이가 작았습니다. 평균 학습 시간은 "
            f"{fixed_hgb.mean_training_seconds:.2f}초와 {chosen.mean_training_seconds:.2f}초였으며, "
            f"학습 시간이 약 {100*(1-chosen.mean_training_seconds/fixed_hgb.mean_training_seconds):.1f}% 짧은 LightGBM을 선택했습니다. "
            "이는 비교 결과를 확인한 뒤 정한 정확도·시간 절충이며 정확도 동등성을 입증한 것은 아닙니다. "
            "기존 조기 종료 HGB와는 시간이 비슷하므로, 시간 이점의 비교 대상은 HGB 400회 후보입니다."
        )
    else:
        selection_lead = (f"이번 고정 설정 비교에서 평균 로그 RMSE가 가장 낮은 구성은 **{LABELS[best]}**였습니다. "
                          f"평균 {best_score:.6f}로 기존 제출 모델의 {baseline:.6f}보다 {improvement:.2f}% 낮았습니다.")
    lines = [
        "# HGB·LightGBM·CatBoost 비교 결과 (2026-11-12)", "",
        selection_lead, "",
        "## 비교 조건", "",
        "- 제공 CSV 208,437행 중 기존 80:20 층화 분할의 학습 166,749행만 후보 평가에 사용했습니다. "
        "외부 검증 41,688행의 타깃은 학습·후보 채점에 쓰지 않았습니다. 전체 타깃의 0 여부는 공통 층화 분할에만 사용했습니다.",
        "- 동일한 34개 피처(원래 13 + 환경 17 + 성장·비율 4), 동일한 KFold(5, shuffle=True, seed=42)를 사용했습니다. "
        "환경 피처는 FAQ에 따라 전체 은하의 입력 기하로 만들고, 최종 노트북과 같은 사전식 정렬을 적용했습니다.",
        "- 학습·평가 로그 정의는 **log10(SFR + 1)**입니다. +1은 기존 팀 가정을 유지한 값이며, 공식 offset이 다르면 이 순위를 그대로 적용할 수 없습니다. "
        "음수 역변환 예측은 SFR=0으로 절단한 뒤 평가했습니다.",
        "- 통제 비교 6개는 학습률 0.08, 400회 학습, 조기 종료 없이 설정 하나씩을 실행했습니다. "
        "기존 제출 모델은 원래의 조기 종료를 유지해 별도로 재현했습니다. 모든 두 단계 모델의 0 판정 문턱은 0.9로 고정했습니다.",
        "- HGB·LightGBM은 최대 잎 31개, CatBoost는 대칭 트리 깊이 5(최대 잎 32개)입니다. "
        "세부 기본값과 실제 계산량까지 같은 비교는 아닙니다. CPU 6개 스레드로 순차 실행했습니다.",
        "- 결과를 보기 전에 protocol.json에 순위와 탐색 후보 규칙을 고정했습니다. "
        "기존 제출 모델보다 평균 RMSE가 낮고 4/5개 이상 분할에서 낮을 때만 추가 검토 후보로 표시했습니다.", "",
        "## 전체 결과", "",
        "평균±표준편차는 5개 분할 RMSE의 기술 통계입니다. 신뢰구간이 아닙니다. 낮은 RMSE가 좋습니다.", "",
        "| 구성 | 로그 RMSE 평균 ± SD | 기존 대비 Δ | 기존보다 낮은 fold | 0 재현율 | 0 정밀도 | 평균 학습 시간 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model, row in summary.sort_values("mean_rmse_log").iterrows():
        scores = folds[folds.model == model].set_index("fold").rmse_log
        wins = int(((scores - ref_folds) < 0).sum())
        comparison = "기준" if model == reference else f"{wins}/5"
        lines.append(
            f"| {LABELS[model]} | {row.mean_rmse_log:.6f} ± {row.sd_rmse_log:.6f} | "
            f"{row.mean_rmse_log - baseline:+.6f} | {comparison} | "
            f"{100 * row.mean_zero_recall:.2f}% | {100 * row.mean_zero_precision:.2f}% | "
            f"{row.mean_training_seconds:.2f}초 |"
        )
    lines += ["", "학습 시간은 모델 적합만 잰 벽시계 시간이며 피처 계산·예측·결과 저장은 제외했습니다. "
              "백그라운드 작업을 완전히 통제하지 않았으므로 시간 비교는 참고값입니다.", "",
              "## 알고리즘과 구조를 나누어 볼 때", "",
              "| 알고리즘 (400회) | 단일 회귀 | 두 단계 | 두 단계 − 단일 | 두 단계가 낮은 fold |",
              "|---|---:|---:|---:|---:|"]
    for family in ("hgb", "lightgbm", "catboost"):
        single, hurdle = f"{family}_single_fixed400", f"{family}_two_stage_fixed400"
        s = float(summary.loc[single, "mean_rmse_log"])
        h = float(summary.loc[hurdle, "mean_rmse_log"])
        a = folds[folds.model == single].set_index("fold").rmse_log
        b = folds[folds.model == hurdle].set_index("fold").rmse_log
        lines.append(f"| {family} | {s:.6f} | {h:.6f} | {h-s:+.6f} | {int((b<a).sum())}/5 |")
    params = manifest["model_params"]
    controlled = [m for m in summary.index if "fixed400" in m]
    assert all(params[m]["regressor"] for m in controlled)
    lines += ["", "## 해석과 보고서에 옮길 문장", "",
              f"동일한 34개 입력과 학습 내부 5-fold에서 HGB, LightGBM, CatBoost의 단일 로그 회귀 및 두 단계 구조를 비교했다. "
              f"이번 설정에서 가장 낮은 평균 로그 RMSE는 {LABELS[best]}의 {best_score:.6f}였으며, "
              f"기존 제출 모델({baseline:.6f}) 대비 {improvement:.2f}% 낮았다. "
              "통제 비교에는 각 알고리즘·구조 조합의 고정 설정 하나만 사용했으므로 탐색적 비교로 해석한다. "
              ("관측 정확도의 작은 차이와 학습 시간의 절충으로 LightGBM을 최종 선택했다. 정확도 동등성을 입증한 것은 아니다."
               if chosen is not None else "최종 교체 판단에는 추가 검증과 과학적 해석의 재점검이 필요하다."), "",
              "HGB와 LightGBM 두 단계의 평균 차이는 약 0.000027 dex로 작습니다. "
              "이번 기록의 순위만으로 두 알고리즘의 일반적인 우열을 판단할 수 없습니다. "
              "두 단계 구조는 세 알고리즘 모두에서 단일 회귀보다 5/5개 분할의 오차가 낮았습니다.", "",
              "보고서 9절에는 비교 동기와 고정 설정을, 10절에는 위 결과표와 분할별 차이를 넣을 수 있습니다. "
              "이 수치를 기존 보고서의 외부 검증 점수와 같은 종류의 결과로 섞지 않아야 합니다.", "",
              "## 확인과 적용 범위", "",
              f"- 독립 검증 스크립트가 저장된 OOF 예측에서 {validation['independent_metric_checks']}개 지표를 다시 계산했습니다. "
              f"전체 {validation['fold_records']}개 모델·분할 기록, 행 커버리지와 원래 분할, 소스·데이터 해시가 일치했습니다.",
              f"- 알고리즘 비교 당시 기존 제출 HGB의 평균 RMSE 재현 차이는 {validation['reference_mean_absolute_error']:.3g}였습니다. "
              "당시 원본 노트북은 ../lightgbm_report_revision_2026-11-12/reference에 보존했습니다. "
              "재현 값은 새 비교의 기준점이며 외부 성능 검증이 아닙니다.",
              "- 0.9 문턱은 HGB에서 선택한 값으로 다른 알고리즘에 최적화하지 않았습니다. "
              "여러 라이브러리의 기본 정규화와 대칭/비대칭 트리 구조도 다릅니다.",
              "- 단일 모델과 두 단계 모델은 총 트리 수와 학습량이 다릅니다. "
              "이번 기록은 동일 계산 예산의 경쟁이나 각 알고리즘의 최선 성능을 입증하지 않습니다.",
              "- HGB 400회 고정 구성은 기존 조기 종료 구성보다 트리를 더 학습하고, "
              "조기 종료용 내부 10%를 떼지 않아 각 fold의 학습 행도 더 사용합니다. 개선을 트리 수 하나의 효과로 분리할 수 없습니다.",
              "- 무작위 5-fold 한 번만 실행했습니다. 분할끼리 독립이 아니므로 p값·독립 표본 가정의 신뢰구간을 내지 않았습니다. "
              "시드 반복·공간 블록 검증·새 데이터 일반화는 확인하지 않았습니다.",
              "- 후보 비교용 고정 설정이며 하이퍼파라미터 탐색은 하지 않았습니다. "
              "기존 검증 표본·학습 데이터는 이전 분석에서도 사용됐으므로 완전히 새로운 시험 결과가 아닙니다.", "",
              "## 재현 자료", "",
              "- 고정 설정: [protocol.json](protocol.json), 실행 환경: [requirements.txt](requirements.txt)",
              "- 평균 결과: [results/summary.csv](results/summary.csv), 분할별 결과: [results/fold_results.csv](results/fold_results.csv)",
              "- 동일 분할 차이: [results/paired_differences.csv](results/paired_differences.csv)",
              "- 실행·소스 지문: [results/manifest.json](results/manifest.json), 독립 검증: [results/validation.json](results/validation.json)",
              ("- 최신 보고서 검증: .venv\\Scripts\\python.exe work/조장/lightgbm_report_revision_2026-11-12/validate_report.py"
               if chosen is not None else "- 독립 재계산: .venv\\Scripts\\python.exe work/조장/model_comparison_2026-11-12/validate_results.py work/조장/model_comparison_2026-11-12/results"), ""]
    if chosen is not None:
        lines += ["## 최신 반영과 과거 검증 범위", "",
                  "학습 시간에 따른 최종 선택과 새 모델의 검증은 ../lightgbm_report_revision_2026-11-12/README.md를 따릅니다. "
                  "기존 benchmark artifact_manifest/company_review는 LightGBM 선택 이전의 코드·문서와 원본 HGB 노트북에 대한 기록입니다. "
                  "이후 승인된 노트북 변경 때문에 과거 검증 스크립트의 원본 파일 불변 조건은 현재 체크아웃에 적용할 수 없습니다. "
                  "최신 수정 검증은 보존된 원본 스냅샷과 원래 CV 증거를 대조합니다.", ""]
    (HERE / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {HERE / 'report.md'}; best={best}, mean={best_score:.6f}")


if __name__ == "__main__":
    main()

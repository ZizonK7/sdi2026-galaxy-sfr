"""지정된 검토 수정만 반영한다. 노트북 코드 실행 및 외부 패키지 사용 없음."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = ROOT / "DataHub_701104_final.ipynb"


def serialize(notebook):
    # 101~103번 칸에는 나중에 추가된 키가 있어 sort_keys=True는 원본과 다르다.
    return (json.dumps(notebook, indent=1, ensure_ascii=False, sort_keys=False) + "\n").replace("\n", "\r\n").encode("utf-8")


def main():
    original = NOTEBOOK.read_bytes()
    notebook = json.loads(original)
    assert serialize(notebook) == original, "읽기 → 그대로 쓰기 바이트 불일치"
    cells = notebook["cells"]
    assert len(cells) == 126

    def replace(index, old, new):
        source = "".join(cells[index]["source"])
        assert source.count(old) == 1, (index, old, source.count(old))
        cells[index]["source"] = source.replace(old, new).splitlines(keepends=True)

    # 검토 1: 같은 데이터의 LF/CRLF 차이는 허용하고, 다른 내용은 경고한다.
    replace(99, "assert ALGORITHM_COMPARISON['csv_sha256'] == REVISION_RUNTIME['csv_sha256']", r'''# 기록 당시 CSV는 줄바꿈이 CRLF인 사본이었다. 줄바꿈만 다른 같은 데이터(배포 원본은 LF)는 같은 데이터로 본다. 이 칸은 저장된 기록만 보여 준다.
_csv_crlf_sha256 = hashlib.sha256(DATA_PATH.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest()
if ALGORITHM_COMPARISON['csv_sha256'] != _csv_crlf_sha256:
    print("참고: 지금 CSV의 내용이 알고리즘 비교를 기록할 때와 다르다. 아래 표는 저장된 기록이며 새로 계산한 값이 아니다.")''')

    # 검토 2: 92번 칸의 기본 reference=df, ID/입력 검사, 피처 생성과 일치한다.
    preprocessing = next(line for line in cells[122]["source"] if line.startswith("**Preprocessing / feature engineering:**"))
    assert preprocessing.endswith("\n")
    replace(122, preprocessing, preprocessing + "**Predicting on new evaluation data:** `predict_sfr(eval_table)`를 쓴다(9.2절 코드, 9.3절 설명). 원래 입력 13개 열과 `subhalo_id`만 있으면 되고, 같은 상자의 기존 은하(`df`)와 합쳐 환경·이웃 피처를 계산한다. `FINAL_MODEL.predict`는 가공한 46개 열을 받으므로 원래 표에는 바로 쓸 수 없다.\n")

    # 검토 3
    replace(113, "**Zero / quiescent galaxies:**", "**첫 그림(예측 대 실제, 잔차 분포):** 왼쪽 그림에서 예측은 대각선을 따라가지만 log10(SFR + 1) > 2 근처에서 대각선 아래로 처진다(극단값 과소예측). 오른쪽 잔차 분포는 0 근처에 몰려 있고, 실제 SFR = 0인 은하(주황)는 0의 오른쪽(과대예측)에만 있다.\n\n**Zero / quiescent galaxies:**")

    # 검토 4: 원문 중 생략된 부분은 그대로 둔다.
    replacements = [
        (98, "아래 기록은 `work/조장/model_comparison_2026-11-12/`의 실행 결과이며, 이 셀에서는 새로 학습하지 않는다.", "아래 기록은 같은 조건으로 따로 실행한 결과(제출물에 미포함)이며, 이 셀에서는 새로 학습하지 않는다."),
        (98, "결과를 보기 전에 `protocol.json`에 고정했다.", "결과를 보기 전에 고정했다."),
        (101, "팀원은 34개 입력에 피처 묶음 5개를 더하거나 빼는 탐색을 했다(`work/팀원/sfr_experiments_20261113/outputs/feature_search/`).", "추가 피처 탐색에서는 34개 입력에 피처 묶음 5개를 더하거나 빼며 비교했다(별도 실행 기록)."),
        (103, "팀원 코드로 같은 fold에서 다시 학습해 비교했다(`work/조장/teammate_review_2026-11-13/`).", "그 코드로 같은 fold에서 다시 학습해 비교했다(별도 실행 기록, 이 노트북에서 재현하지 않음)."),
        (90, " 자세한 EDA는 work/조장/2026-11-07_위치속도_환경EDA.md.", ""),
        (90, "# (3) 팀원 피처 탐색에서 고른 추가 피처 12개 (work/팀원/sfr_experiments_20261113/outputs/feature_search). 모두", "# (3) 추가 피처 탐색에서 고른 12개 (별도 실행 기록). 모두"),
        (93, "# 같은 계산을 D2·D3 정의까지 넓힌 원본: work/조장/log_model_cv_2026-11-11/(log_model_cv.py, q4_reg_trees.py). 약 5~6분.", "# 같은 계산을 D2·D3 정의까지 넓힌 원본은 별도 실행 기록에 있다. 약 5~6분."),
    ]
    for args in replacements:
        replace(*args)
    # 99번의 긴 JSON 문자열에도 내부 파일명이 있다. 완료 조건 6에 따라 설명만 정리한다.
    replace(99, "protocol.json에 결과 전에 고정한 순위 기준", "결과 전에 고정한 순위 기준")

    # 검토 5: 저장 출력에서 확인한 출처만 표시한다. 숫자는 바꾸지 않는다.
    replace(103, "| 0/보통/꼬리 + 꼬리 50:50 혼합 | 0.093367 | 0.093836 | 0.513 / 0.518 | 0.094205 | 0.638 |\n", "| 0/보통/꼬리 + 꼬리 50:50 혼합 | 0.093367 | 0.093836 | 0.513 / 0.518 | 0.094205 | 0.638 |\n\n위 표에서 두 단계 행은 이 노트북의 출력이고, 0/보통/꼬리 행은 별도 실행 기록이다.\n")
    replace(103, "−0.00125 dex(부트스트랩 95% 구간 [−0.0022, −0.0003])", "−0.00125 dex(부트스트랩 95% 구간 [−0.0022, −0.0003])(별도 실행 기록)")
    replace(120, "34개 입력 모델의 0.155에서", "34개 입력 모델(별도 실행 기록)의 0.155에서")

    # 검토 6~8
    replace(120, "이웃 은하 피처는 작게(0.001~0.002) 쓰였다", "이웃 은하 피처는 0.002 이하로 작게 쓰였다")
    replace(120, "중심 은하는 어느 그룹에서든 약 0.4%다", "중심 은하는 어느 그룹에서든 0.4% 이하다")
    replace(87, "+0.00002 dex, 19/25 악화. 미리 정한 악화 기준(22/25)에 못 미쳐 채택했다", "+0.00002 dex, 19/25 악화. 점수 차이가 무시할 수준이고(검증셋에서도 +0.000002 dex) 미리 정한 기각 기준(22/25 악화)에 해당하지 않아, 꺼진 은하에 정확히 0을 내는 출력 정책으로 채택했다")

    # 검토 9: 중요도 계산·표·print는 보존하고 그리는 부분만 고친다.
    replace(119, 'fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))', '''from matplotlib.patches import Patch

def _feature_color(c):
    return "#1baf7a" if c in ENV_FEATURES else ("#6250d6" if c in MASS_FEATURES else ("#e87ba4" if c in EXTRA_FEATURES else "#eb6834"))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))''')
    replace(119, 'color="#2a78d6")', 'color=[_feature_color(c) for c in order_c])')
    replace(119, 'color=["#1baf7a" if c in ENV_FEATURES else ("#6250d6" if c in MASS_FEATURES else ("#e87ba4" if c in EXTRA_FEATURES else "#eb6834")) for c in order_r]', 'color=[_feature_color(c) for c in order_r]')
    replace(119, r'(top 12)\ngreen = environment, purple = mass history, pink = added 12, orange = original', r'\npermutation importance (top 12)')
    # 제목 앞에 있던 공백까지 함께 제거한다.
    replace(119, r' | SFR>0] \npermutation importance', r' | SFR>0]\npermutation importance')
    replace(119, 'plt.tight_layout(); plt.show()', '''fig.legend(handles=[Patch(facecolor="#1baf7a", label="environment"),
                    Patch(facecolor="#6250d6", label="mass history"),
                    Patch(facecolor="#e87ba4", label="added 12"),
                    Patch(facecolor="#eb6834", label="original")],
           loc="lower center", ncol=4, frameon=False)
plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.show()''')

    # 검토 10
    replace(116, "#### 11.3 입력 진단 — FoF 그룹 연결의 민감도", "### 11.3 입력 진단 — FoF 그룹 연결의 민감도")

    # 검토 12: 지정된 중립 표현. 대상 범위에서 추가 발견된 역할 표현은 없다.
    replacements = [
        (0, "그리고 팀원의 피처 탐색에서 고른 이웃 은하·병합×성장·질량 대비 잔차 12개다.", "그리고 추가 피처 탐색에서 고른 이웃 은하·병합×성장·질량 대비 잔차 12개다."),
        (87, "② 팀원이 46개 입력에서 0/보통/꼬리 구조,", "② 별도 꼬리 실험에서 46개 입력으로 0/보통/꼬리 구조,"),
        (87, "팀원 피처 탐색에서 위치·속도 열을 빼는 경우도 비교했지만", "추가 피처 탐색에서 위치·속도 열을 빼는 경우도 비교했지만"),
        (87, "| 추가 피처 12개 (팀원 피처 탐색) |", "| 추가 피처 12개 |"),
        (101, "#### 9.2 보조 (4) — 추가 피처 12개 (팀원 피처 탐색)", "#### 9.2 보조 (4) — 추가 피처 12개"),
        (103, "팀원의 최종 후보는 같은 46개 입력에서", "별도 꼬리 실험의 최종 후보는 같은 46개 입력에서"),
        (103, "(0.095962 대 0.095340, 팀원 기록)", "(0.095962 대 0.095340, 별도 실행 기록)"),
        (103, "46개 구성은 팀원이 시드 42의 fold에서 여러 후보를 보고 골랐으므로", "46개 구성은 시드 42의 fold에서 여러 후보를 보고 골랐으므로"),
        (103, "팀원의 탐색은 공통 검증셋을 쓰지 않았고", "그 탐색은 공통 검증셋을 쓰지 않았고"),
        (104, "- 추가 12개(팀원 피처 탐색):", "- 추가 12개(추가 피처 탐색):"),
        (107, "# 팀원 꼬리 분리(T = 30)를", "# 꼬리 분리(T = 30)를"),
        (121, "특히 추가 피처 12개는 팀원이 시드 42의 fold에서 여러 후보를 비교해 골랐으므로", "특히 추가 피처 12개는 시드 42의 fold에서 여러 후보를 비교해 골랐으므로"),
        (124, "추가 피처는 팀원 탐색 후 학습 내부 10-fold로 확인했다", "추가 피처는 별도 탐색 후 학습 내부 10-fold로 확인했다"),
    ]
    for args in replacements:
        replace(*args)
    for index in [0, *range(85, 126)]:
        assert not any(term in "".join(cells[index]["source"]) for term in ("팀원", "조장", "work/", "protocol.json")), index

    # 검토 13: 필수 키 유지 여부는 static_checks.py가 템플릿 AST와 대조한다.
    replace(124, 'print(json.dumps(RESULTS, indent=2, ensure_ascii=False))', '''_bulky = ("algorithm_comparison", "diagnostic_cv_records", "diagnostic_interaction_records")
print(json.dumps({k: v for k, v in RESULTS.items() if k not in _bulky}, indent=2, ensure_ascii=False))
print(f"(표시 생략: {', '.join(_bulky)} — 원자료 기록이며 RESULTS 변수에는 그대로 들어 있다)")''')

    # 검토 14~15
    assert "lightgbm_revision" in notebook["metadata"]
    del notebook["metadata"]["lightgbm_revision"]
    replace(122, "노트북 전체 실행 약 11분(25-fold 반복 교차검증 포함)", "노트북 전체 실행 약 11~15분(하드웨어에 따라 다름, 25-fold 반복 교차검증 포함)")

    # 검토 11: 위치 교환은 마지막에 한다. 칸 내용·메타데이터·출력은 이동만 한다.
    cells[0], cells[1] = cells[1], cells[0]
    NOTEBOOK.write_bytes(serialize(notebook))
    print("검토 15건 반영 완료. 노트북 실행 및 저장 출력 변경 없음.")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    main()

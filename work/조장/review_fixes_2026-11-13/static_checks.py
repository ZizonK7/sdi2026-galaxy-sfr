"""HEAD/템플릿/저장 출력만 읽는 정적 확인. 노트북 실행은 하지 않는다."""
import ast
import copy
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name("static_checks.json")
NAME = "DataHub_701104_final.ipynb"
CSV = "Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv"
ALLOWED = {0, 87, 90, 93, 98, 99, 101, 103, 104, 107, 113, 116, 119, 120, 121, 122, 124}
CODE_CELLS = {90, 93, 99, 107, 119, 124}
BULKY = ("algorithm_comparison", "diagnostic_cv_records", "diagnostic_interaction_records")


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def source(cell):
    return "".join(cell["source"])


def text_outputs(cell):
    for output in cell.get("outputs", []):
        if "text" in output:
            yield "text", "".join(output["text"])
        for kind, value in output.get("data", {}).items():
            if kind in ("text/plain", "text/html"):
                yield kind, "".join(value)


def assignment(tree, name):
    return next(node.value for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))


def output_audit(cells):
    """숫자 후보는 전 출력에서 찾되 의미 판단은 worklog의 출처 표에서 한다."""
    values = ["0.093387", "0.093771", "0.512", "0.503", "0.094063", "0.586",
              "0.093367", "0.093836", "0.513", "0.518", "0.094205", "0.638",
              "0.095962", "0.095340", "-0.00125", "-0.0022", "-0.0003", "0.155"]
    findings = {}
    for value in values:
        variants = {value, f"{float(value):.3f}", f"{float(value):.4f}", f"{float(value):.6f}"}
        variants |= {v.replace("-", "−") for v in variants}
        patterns = [re.compile(r"(?<![\d.])" + re.escape(v) + r"(?![\d.])") for v in variants]
        hits = []
        for index, cell in enumerate(cells):
            for kind, text in text_outputs(cell):
                # HTML에서는 한 행의 모델 이름과 값을 함께 보관한다.
                chunks = re.findall(r"<tr\b[^>]*>.*?</tr>", text, re.S) if kind == "text/html" else text.splitlines()
                for chunk in chunks:
                    if any(pattern.search(chunk) for pattern in patterns):
                        plain = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", chunk))).strip()
                        hits.append({"cell": index, "kind": kind, "context": plain[:420]})
        findings[value] = {"variants": sorted(variants), "candidate_hits": hits,
                           "note": "숫자열 후보이며 동일한 의미의 출처인지는 worklog 표를 본다."}
    return findings


def main():
    baseline_bytes = git("show", "HEAD:" + NAME)
    baseline = json.loads(baseline_bytes)
    current_bytes = (ROOT / NAME).read_bytes()
    current = json.loads(current_bytes)
    template = json.loads((ROOT / "DataHub_Midterm_Final_Report_Template.ipynb").read_bytes())
    order = [1, 0, *range(2, 126)]
    # aligned[원래 칸 번호]로 비교한다.
    aligned = [None] * 126
    if len(current["cells"]) == 126:
        for new, old in enumerate(order):
            aligned[old] = current["cells"][new]
    checks = []

    def check(number, label, function):
        try:
            detail = function()
            checks.append({"number": number, "label": label, "passed": True, "detail": detail})
        except Exception as error:
            checks.append({"number": number, "label": label, "passed": False,
                           "error": f"{type(error).__name__}: {error}"})

    def shape():
        assert len(current["cells"]) == len(baseline["cells"]) == 126
        # 소스 이외 전체 속성으로 이동 순서와 칸 메타데이터 보존도 확인한다.
        for index, cell in enumerate(aligned):
            assert {k: v for k, v in cell.items() if k != "source"} == {
                k: v for k, v in baseline["cells"][index].items() if k != "source"}, index
        assert source(aligned[1]).startswith("# DataHub")
        assert source(aligned[0]).startswith("## 최종 보고서 읽기 안내")
        return {"cell_count": 126, "original_indices_in_new_order": order}

    def protected():
        original = [(i, c["source"]) for i, c in enumerate(baseline["cells"])
                    if c["cell_type"] == "code" and source(c).startswith("# DO NOT MODIFY")]
        required = [c["source"] for c in template["cells"]
                    if c["cell_type"] == "code" and source(c).startswith("# DO NOT MODIFY")]
        assert len(original) == len(required) == 7, (len(original), len(required))
        assert [s for _, s in original] == required
        for i, s in original:
            assert aligned[i]["source"] == s, i
        return {"original_cell_indices": [i for i, _ in original], "template_equal": True}

    def part_one():
        assert all(aligned[i]["source"] == baseline["cells"][i]["source"] for i in range(1, 85))
        return {"preserved_original_indices": [1, 84], "cell_count": 84}

    def changes():
        changed = [i for i, cell in enumerate(aligned) if cell["source"] != baseline["cells"][i]["source"]]
        assert set(changed) == ALLOWED, changed
        for i, cell in enumerate(aligned):
            for key in ("outputs", "execution_count"):
                assert (key in cell) == (key in baseline["cells"][i]), (i, key)
                assert cell.get(key) == baseline["cells"][i].get(key), (i, key)
        expected_meta = {k: v for k, v in baseline["metadata"].items() if k != "lightgbm_revision"}
        assert current["metadata"] == expected_meta
        assert {k: v for k, v in current.items() if k not in ("cells", "metadata")} == {
            k: v for k, v in baseline.items() if k not in ("cells", "metadata")}
        return {"changed_original_indices": changed, "outputs_and_execution_count_preserved": True,
                "only_deleted_metadata_key": "lightgbm_revision"}

    def syntax():
        changed_code = {i for i in ALLOWED if aligned[i]["cell_type"] == "code"}
        assert changed_code == CODE_CELLS
        for i in sorted(changed_code):
            ast.parse(source(aligned[i]), filename=f"original_cell_{i}")
        for i in (90, 93, 107):
            assert ast.dump(ast.parse(source(aligned[i]))) == ast.dump(ast.parse(source(baseline["cells"][i]))), i
        # RESULTS 딕셔너리의 유일한 수정은 notes의 역할 표현이다.
        old_results = copy.deepcopy(assignment(ast.parse(source(baseline["cells"][124])), "RESULTS"))
        new_results = assignment(ast.parse(source(aligned[124])), "RESULTS")
        for key, val in zip(old_results.keys, old_results.values):
            if isinstance(key, ast.Constant) and key.value == "notes":
                val.value = val.value.replace("추가 피처는 팀원 탐색 후 학습 내부 10-fold로 확인했다", "추가 피처는 별도 탐색 후 학습 내부 10-fold로 확인했다")
        assert ast.dump(old_results) == ast.dump(new_results)
        before99 = assignment(ast.parse(source(baseline["cells"][99])), "ALGORITHM_COMPARISON")
        after99 = assignment(ast.parse(source(aligned[99])), "ALGORITHM_COMPARISON")
        old_record = json.loads(ast.literal_eval(before99.args[0]))
        new_record = json.loads(ast.literal_eval(after99.args[0]))
        old_record["decision"]["rule"] = old_record["decision"]["rule"].replace(
            "protocol.json에 결과 전에 고정한 순위 기준", "결과 전에 고정한 순위 기준")
        assert old_record == new_record
        marker = "# 위성/중심 × 호스트 질량별"
        old119, new119 = source(baseline["cells"][119]), source(aligned[119])
        assert old119.split("fig, axes =")[0] == new119.split("from matplotlib.patches import Patch")[0]
        assert old119.split(marker)[1] == new119.split(marker)[1]
        return {"parsed_original_code_indices": sorted(changed_code), "results_dictionary_only_notes_changed": True,
                "comment_only_code_ast_equal": [90, 93, 107], "algorithm_record_only_rule_file_reference_removed": True,
                "importance_calculation_tables_prints_preserved": True}

    def terms():
        counts = {term: sum(source(aligned[i]).count(term) for i in [0, *range(85, 126)])
                  for term in ("팀원", "조장", "work/", "protocol.json")}
        assert not any(counts.values()), counts
        assert "lightgbm_revision" not in current["metadata"]
        return {"source_term_counts": counts, "lightgbm_revision_absent": True}

    def hashes():
        record_call = assignment(ast.parse(source(aligned[99])), "ALGORITHM_COMPARISON")
        recorded = json.loads(ast.literal_eval(record_call.args[0]))["csv_sha256"]
        work_bytes = (ROOT / CSV).read_bytes()
        blob_bytes = git("cat-file", "blob", "HEAD:" + CSV)
        normalize = lambda raw: raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        sha = lambda raw: hashlib.sha256(raw).hexdigest()
        assert b"\r\n" in work_bytes and b"\r\n" not in blob_bytes
        assert sha(normalize(work_bytes)) == sha(normalize(blob_bytes)) == recorded
        assert sha(blob_bytes).startswith("0710e7b2")
        # 실행 없이 AST의 bytes 상수로 이스케이프가 올바른지 확인한다.
        expression = assignment(ast.parse(source(aligned[99])), "_csv_crlf_sha256")
        expected = ast.parse('hashlib.sha256(DATA_PATH.read_bytes().replace(b"\\r\\n", b"\\n").replace(b"\\n", b"\\r\\n")).hexdigest()', mode="eval").body
        assert ast.dump(expression) == ast.dump(expected)
        assert any(isinstance(node, ast.Import) and any(alias.name == "hashlib" for alias in node.names)
                   for node in ast.parse(source(aligned[89])).body)
        assert "DATA_PATH" in source(aligned[6])
        return {"recorded_sha256": recorded, "working_csv_raw_sha256": sha(work_bytes),
                "head_blob_lf_raw_sha256": sha(blob_bytes), "working_csv_crlf_normalized_sha256": sha(normalize(work_bytes)),
                "head_blob_crlf_normalized_sha256": sha(normalize(blob_bytes)), "byte_literals_verified_by_ast": True}

    def scope():
        status = git("status", "--short").decode("utf-8")
        entries = git("status", "--porcelain=v1", "-z", "--untracked-files=all").decode("utf-8").split("\0")
        paths = [entry[3:] for entry in entries if entry]
        allowed_files = {f"work/조장/review_fixes_2026-11-13/{name}" for name in
                         ("apply_fixes.py", "static_checks.py", "static_checks.json", "worklog.md")} | {NAME}
        assert paths and set(paths) <= allowed_files, paths
        assert current_bytes.count(b"\r\n") == current_bytes.count(b"\n")
        assert current_bytes.endswith(b"\r\n") and not current_bytes.endswith(b"\r\n\r\n")
        baseline_crlf = baseline_bytes.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        serialized = (json.dumps(baseline, indent=1, ensure_ascii=False, sort_keys=False) + "\n").replace("\n", "\r\n").encode("utf-8")
        assert serialized == baseline_crlf
        return {"git_diff_stat": git("diff", "--stat").decode("utf-8"), "git_status_short": status,
                "changed_paths": paths, "crlf_preserved": True, "baseline_roundtrip_exact": True,
                "serialization": {"indent": 1, "sort_keys": False, "ensure_ascii": False}}

    for num, label, func in [(1, "JSON・칸 수・순서", shape), (2, "DO NOT MODIFY 7칸 보존", protected),
                             (3, "Part I 보존", part_one), (4, "변경 범위・출력 보존", changes),
                             (5, "변경 코드 문법・계산 보존", syntax), (6, "내부 표현・메타데이터", terms),
                             (7, "CRLF/LF 해시와 bytes 리터럴", hashes)]:
        check(num, label, func)

    required_dict = assignment(ast.parse(next(source(c) for c in template["cells"]
                                              if c["cell_type"] == "code" and "RESULTS = {" in source(c))), "RESULTS")
    required_keys = [ast.literal_eval(key) for key in required_dict.keys]
    current_tree = ast.parse(source(aligned[124]))
    current_keys = [ast.literal_eval(k) for k in assignment(current_tree, "RESULTS").keys if k is not None]
    bulky = ast.literal_eval(assignment(current_tree, "_bulky"))
    assert bulky == BULKY
    required_visible = set(required_keys) <= set(current_keys) - set(bulky)
    checks.append({"number": "13-extra", "label": "RESULTS 필수 키 표시 유지", "passed": required_visible,
                   "detail": {"required_keys": required_keys, "bulky_keys": list(bulky)}})
    original_results_text = next(t for _, t in text_outputs(baseline["cells"][124]) if t.lstrip().startswith("{"))
    results, _ = json.JSONDecoder().raw_decode(original_results_text.lstrip())
    sizes = {key: len(json.dumps(value, indent=2, ensure_ascii=False)) for key, value in results.items()}
    output_size = {"measure": "각 값을 json.dumps(indent=2, ensure_ascii=False)로 나타낸 글자 수(키와 바깥쪽 들여쓰기 제외)",
                   "characters_by_key": dict(sorted(sizes.items(), key=lambda p: -p[1])),
                   "bulky_value_characters": sum(sizes[k] for k in BULKY),
                   "all_value_characters": sum(sizes.values()),
                   "bulky_fraction": sum(sizes[k] for k in BULKY) / sum(sizes.values())}
    # 검토 항목은 1~15 전체 번호를 유지한다. 실행·시각 확인은 지휘 모델 담당이다.
    review_markers = {
        1: (99, "_csv_crlf_sha256 ="), 2: (122, "**Predicting on new evaluation data:**"),
        3: (113, "**첫 그림(예측 대 실제, 잔차 분포):**"), 4: (98, "결과를 보기 전에 고정했다.**"),
        5: (103, "위 표에서 두 단계 행은 이 노트북의 출력이고, 0/보통/꼬리 행은 별도 실행 기록이다."),
        6: (120, "이웃 은하 피처는 0.002 이하로 작게 쓰였다"), 7: (120, "중심 은하는 어느 그룹에서든 0.4% 이하다"),
        8: (87, "출력 정책으로 채택했다"), 9: (119, 'loc="lower center", ncol=4, frameon=False'),
        10: (116, "### 11.3 입력 진단 — FoF 그룹 연결의 민감도"), 12: (124, "추가 피처는 별도 탐색 후"),
        13: (124, "if k not in _bulky"), 15: (122, "노트북 전체 실행 약 11~15분")}
    review_items = {str(i): {"original_cell": cell, "source_marker_present": marker in source(aligned[cell])}
                    for i, (cell, marker) in review_markers.items()}
    review_items["11"] = {"source_marker_present": source(current["cells"][0]) == source(baseline["cells"][1])}
    review_items["14"] = {"source_marker_present": "lightgbm_revision" not in current["metadata"]}
    checks.append({"number": "1-15-extra", "label": "검토 15건 반영 표시", "passed": all(v["source_marker_present"] for v in review_items.values())})
    result = {"notebook_executed": False, "basis": "git show HEAD:DataHub_701104_final.ipynb",
              "checks": checks, "review_items": {str(i): review_items[str(i)] for i in range(1, 16)},
              "number_output_search": output_audit(baseline["cells"]), "results_output_sizes": output_size}
    # 결과 파일도 범위 확인 대상에 들어가도록 먼저 만든 뒤 최종 결과로 덮어쓴다.
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    check(8, "파일 변경 범위・모양・git 상태", scope)
    result["all_passed"] = all(item["passed"] for item in checks)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for item in checks:
        print(f"{item['number']}: {'PASS' if item['passed'] else 'FAIL'} {item['label']}")
        if not item["passed"]:
            print(item.get("error", item.get("detail")))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())

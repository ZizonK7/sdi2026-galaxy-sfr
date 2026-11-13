from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1] / 'outputs/leader_three_regime_experiment'
meta = json.loads((root / 'metadata.json').read_text())
summary = pd.read_csv(root / 'summary.csv').set_index('model')
oof = np.load(root / 'oof_predictions.npz')
components = np.load(root / 'oof_components.npz')
paired = pd.read_csv(root / 'paired_differences.csv')
y = oof['actual']
labels = {'leader_two_stage': '조장 원본: 0/양수', 'three_T10': '3분류: T=10',
          'three_T30': '3분류: T=30', 'three_T100': '3분류: T=100'}
diagnostics = []
for name in components.files:
    threshold = int(name.split('T')[1])
    values = components[name]
    probs = values[:, :3]
    true_class = np.where(y == 0, 0, np.where(y <= threshold, 1, 2))
    predicted_class = probs.argmax(axis=1)
    extreme = y > 100
    # Expert-only predictions on true tails are a diagnostic using known true classes,
    # not a deployable score: the real model must infer the gate probabilities.
    expert = np.maximum(values[extreme, 4], 0.)
    diagnostics.append(dict(model=name, threshold=threshold,
        true_tail_rows=int((true_class == 2).sum()),
        true_extreme_rows=int(extreme.sum()),
        mean_tail_probability_on_extreme=float(probs[extreme, 2].mean()),
        tail_argmax_recall=float((predicted_class[true_class == 2] == 2).mean()),
        zero_argmax_recall=float((predicted_class[true_class == 0] == 0).mean()),
        extreme_tail_expert_only_rmse=float(np.sqrt(np.mean((expert - np.log10(1 + y[extreme]))**2))),
        extreme_final_rmse=float(summary.loc[name, 'tail100_log_rmse'])))
pd.DataFrame(diagnostics).to_csv(root / 'gate_diagnostics.csv', index=False)

table = '| 모델 | 전체 로그 RMSE (fold 평균) | 실제 0 로그 RMSE | SFR>100 로그 RMSE | 0 재현율 | 0 정밀도 |\n|---|---:|---:|---:|---:|---:|\n'
for name, row in summary.iterrows():
    table += f"| {labels[name]} | {row.mean_fold_rmse:.6f} | {row.zero_log_rmse:.6f} | {row.tail100_log_rmse:.6f} | {row.zero_recall:.2%} | {row.zero_precision:.2%} |\n"
base = summary.loc['leader_two_stage']
best = summary.loc[meta['selected_candidate']]
tail_best = summary.loc[summary.index != 'leader_two_stage'].tail100_log_rmse.idxmin()
improvement = 100 * (base.mean_fold_rmse - best.mean_fold_rmse) / base.mean_fold_rmse
outcome = f'전체 로그 RMSE는 조장 원본 대비 {abs(improvement):.3f}% ' + ('개선' if improvement > 0 else '악화')
tail_delta = best.tail100_log_rmse - base.tail100_log_rmse
zero_delta = best.zero_log_rmse - base.zero_log_rmse
status = ('이번 후보는 전체·실제 0·SFR>100의 로그 RMSE가 모두 낮았다.'
          if improvement > 0 and tail_delta < 0 and zero_delta < 0
          else '전체·0·꼬리 중 개선되지 않은 지표가 있으므로, 세 구간을 동시에 개선한 모델이라고 결론내릴 수 없다.')
report = f'''# 조장 HGB400 모델에서 0 / 일반 / 꼬리 분리 실험

## 결과

{outcome}됐다. 3분류 후보 중 전체 fold 평균이 가장 낮은 구성은 **T={meta['tail_threshold']}**다.
조장 원본보다 전체 점수가 낮은 fold는 **{meta['overall_improved_folds']}/5**다.
{status}

{table}

표의 전체 RMSE는 조장 비교 기준과 맞춘 5개 fold RMSE의 평균이다.
나머지 구간 지표는 같은 OOF 행을 모아 계산했다. 전체 pooled OOF RMSE는 원본
{base.log_rmse:.9f}, 선택 후보 {best.log_rmse:.9f}이며 `summary.csv`의 `log_rmse` 열이다.
0 재현율은 실제 0 중 정확히 0을 출력한 비율, 정밀도는 0 출력 중 실제 0인 비율이다.
실제 0 은하는 {(y == 0).sum():,}개, SFR>100은 {(y > 100).sum():,}개다.

선택 후보의 실제 0 구간 로그 RMSE 변화는 {zero_delta:+.9f}, SFR>100 변화는 {tail_delta:+.9f}다.
음수이면 개선이다. 꼬리 로그 편향은 원본 {base.tail100_log_bias:.6f}, 선택 후보 {best.tail100_log_bias:.6f}다.
음수 편향은 과소예측을 뜻한다. SFR>100 점수만으로 가장 나은 3분류 후보는 {labels[tail_best]}이다.

## 구현한 구조

1. 실제 SFR로 0 / 0<SFR≤T / SFR>T의 학습 정답을 만든다.
2. HGB 다중분류기가 전체 학습 데이터로 세 확률을 학습한다.
3. 일반 HGB 회귀기는 실제 일반 구간만, 꼬리 HGB 회귀기는 실제 꼬리 구간만 학습한다.
4. 회귀 타깃은 log10(1+SFR)이다. 예측 로그는 `p_normal*z_normal + p_tail*z_tail`이다.
5. p_zero≥0.9이면 로그 예측을 0으로 바꾸고, `max(10**z-1, 0)`으로 선형 SFR을 반환한다.

회귀기 학습 대상을 분류기의 예측으로 정하지 않는다. 예측할 때도 argmax로 한 회귀기만 고르지 않는다.
'일반 구간'은 중앙값 하나를 뜻하지 않고 0보다 크고 T 이하인 모든 학습 행을 뜻한다.
모든 학습기는 학습률 .08, 400회, 최대 잎 31개, 조기 종료 없음이다. 다중분류는 클래스별 트리를 만들므로
원본 이진분류와 계산량이 같지 않다. `mean_fit_seconds`에 실측 시간을 기록했다.

## 비교 조건과 해석 범위

조장 최신 커밋 `21d02b5d0c829aff67cafd89da62c3386ac7864f`의 피처 코드와 사전식 참고 은하 정렬을 유지했다.
전체 은하의 입력 기하로 환경 피처를 계산하며 타깃은 피처 계산에 쓰지 않는다.
원래 80:20 분할에서 학습 166,749행만 사용하고, 같은 KFold(5, shuffle=True, seed=42)로 원본과 세 후보를 모두 재학습했다.
기존 20% 검증셋은 이번 실험에서 채점하거나 후보 선택에 사용하지 않았다.
주 지표는 log10(1+SFR) RMSE이며, 후보 T=10/30/100과 선택 기준은 실행 전에 `protocol.json`에 기록했다.
같은 CV에서 후보를 고른 탐색 결과이므로 최종 독립 성능 검증은 아니다. 모든 구성에서 0 문턱은 .9로 고정했다.

꼬리 회귀기가 좋아도 꼬리 확률이 낮으면 최종 예측은 일반 회귀기 쪽으로 내려간다.
`gate_diagnostics.csv`는 실제 극단 은하의 꼬리 확률과 꼬리 회귀기 단독 오차를 나눠 보여준다.
단독 오차는 정답 구간을 알고 골랐을 때의 진단값이며 실제 사용할 수 있는 모델 성능이 아니다.
0 RMSE에는 놓친 0의 과대예측이, 꼬리 RMSE에는 구간 분류와 회귀 양쪽의 오차가 반영된다.

## 저장 모델과 사용

`best_three_regime.joblib`는 T={meta['tail_threshold']} 후보를 원래 학습 166,749행에 다시 적합한 모델이다.
현재 채택 A와 조장 원본 모델은 별도로 유지하며 자동 교체나 GitHub 푸시는 하지 않았다.

Python 모듈을 찾을 수 있도록 이 폴더에서 실행한다.

```python
import joblib
import pandas as pd
from leader_model import predict_sfr

model = joblib.load('best_three_regime.joblib')
reference = pd.read_csv('/path/to/Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv')
reference = reference.drop(columns='sfr_linear')
query = reference.iloc[:100].copy()
prediction = predict_sfr(query, model, reference)
```

참고 은하 집합을 고정해 사용한다. 배치마다 환경 피처를 다시 정의하지 않는다.
재실행: `python run_experiment.py --data '/path/to/data.csv'`.
경계값·실제 정답별 학습 테스트: `python test_three_regime.py`.

OOF 모든 행이 정확히 한 번 예측됐고, 저장된 확률/회귀 출력으로 최종 예측을 재구성했다.
최종 모델 저장·재로딩 일치와 원본 표 입력 인터페이스도 확인했다.

## 파일

- `summary.csv`, `fold_results.csv`, `paired_differences.csv`: 전체·0·꼬리 점수와 fold별 변화
- `oof_predictions.npz`: 실제값, 원본 행 번호, fold, 구성별 OOF 최종 예측
- `oof_components.npz`: 각 후보별 열 순서 `[p_zero, p_mid, p_tail, mid_log, tail_log]`
- `gate_diagnostics.csv`: 구간 분류와 꼬리 회귀 진단
- `metadata.json`, `protocol.json`: 선택 결과·실행 전 조건
- `three_regime.py`, `leader_model.py`, `run_experiment.py`: 모델과 재현 코드
'''
if (root / 'zero_preserved_metadata.json').exists():
    extra = json.loads((root / 'zero_preserved_metadata.json').read_text())
    extra_folds = pd.read_csv(root / 'zero_preserved_folds.csv')
    base_folds = pd.read_csv(root / 'fold_results.csv').query("model == 'leader_two_stage'").sort_values('fold')
    wins = int((extra_folds.sort_values('fold').log_rmse.to_numpy() < base_folds.log_rmse.to_numpy()).sum())
    extra_report = f'''## 추가 확인: 기존 0 확률 보존

직접 3분류의 0 구간 악화를 관찰한 뒤 추가한 탐색 실험이다. 후보 설정을 추가하기 전에
`zero_preserved_protocol.json`에 조건을 기록했다. 독립 검증으로 해석하지 않는다.

조장 원본의 이진 분류기를 같은 학습 fold 안에서 학습해 P(양수)를 구하고,
기본 3분류 T=10의 일반/꼬리 확률 비율로 양수 확률만 나눈다. 일반·꼬리 회귀기는 동일하다.
즉 `Ptail = P양수_original × Ptail_three/(Pmid_three+Ptail_three)`이며,
`Pmid = P양수_original - Ptail`, `P0 = 1-P양수_original`이다.
P0≥.9 규칙은 원본과 같은 이진 분류기의 확률에 적용한다.

| 지표 | 조장 원본 | 0 확률 보존 + T=10 |
|---|---:|---:|
| 전체 로그 RMSE (fold 평균) | {base.mean_fold_rmse:.6f} | {extra['mean_fold_rmse']:.6f} |
| 실제 0 로그 RMSE (pooled OOF) | {base.zero_log_rmse:.6f} | {extra['zero_log_rmse']:.6f} |
| SFR>100 로그 RMSE (pooled OOF) | {base.tail100_log_rmse:.6f} | {extra['tail100_log_rmse']:.6f} |
| 0 재현율 | {base.zero_recall:.2%} | {extra['zero_recall']:.2%} |
| 0 정밀도 | {base.zero_precision:.2%} | {extra['zero_precision']:.2%} |

원본보다 전체 점수가 좋은 fold는 {wins}/5다. 저장 파일은 `zero_preserved_three_regime.joblib`이다.
동일한 `predict_sfr` 인터페이스를 사용하며 `zero_preserved.py`가 추가로 필요하다.
기본 3분류보다 전체 점수는 회복했지만 원본보다 낮아지지 않았고 극단 꼬리 점수도 개선되지 않았다.
따라서 이번 실험에서 원본을 대체할 성능 근거는 얻지 못했다.

꼬리 분리를 계속 발전시킬 때는 회귀기 교체만큼 꼬리 확률을 얼마나 잘 추정하는지 확인해야 한다.
T=100은 학습 표본 전체에서도 209개뿐이고 실제 극단 은하의 평균 꼬리 확률이 0.306이었다.
정답 구간을 알고 꼬리 회귀기만 쓴 진단 RMSE는 0.190이지만 실제 최종 예측은 0.659다.
이 차이는 정답 구간을 알 수 없는 실제 예측에서 확률 추정과 혼합이 중요한 제약임을 보여준다.
다음 실험 후보는 0 분류를 유지한 상태의 양수 전용 일반/꼬리 이진 분류기와 확률 보정이다.
현재 추가 변형은 양수 전용 이진 분류기를 새로 학습한 것이 아니라 기존 3분류 확률을 조건부로 환산한 것이다.

'''
    report = report.replace('## 구현한 구조', extra_report + '## 구현한 구조')
    print('Zero-preserved follow-up:', json.dumps(extra, ensure_ascii=False))
(root / 'REPORT.md').write_text(report)
(root / 'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n'
    for p in sorted(root.iterdir()) if p.is_file() and p.name != 'SHA256SUMS'))
with zipfile.ZipFile(root.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
    for path in sorted(root.iterdir()):
        if path.is_file():
            z.write(path, arcname=f'{root.name}/{path.name}')
with zipfile.ZipFile(root.with_suffix('.zip')) as z:
    assert z.testzip() is None
print(table)
print(outcome, status)
print(pd.DataFrame(diagnostics).to_string(index=False))

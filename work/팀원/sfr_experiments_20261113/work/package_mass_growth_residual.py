from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1] / 'outputs/mass_growth_residual_experiment'
meta = json.loads((root / 'metadata.json').read_text())
summary = pd.read_csv(root / 'summary.csv')
pairs = pd.read_csv(root / 'paired_differences.csv')
oof = np.load(root / 'oof_predictions.npz')
comp = np.load(root / 'oof_three_components.npz')
y = oof['actual']
extreme = y > 100
labels = {'leader_two_stage':'조장 0/양수', 'three_T10':'3분류 T=10',
          'zero_preserved_three_T10':'기존 0 확률 방식 + 3분류 T=10'}
table = '| 모델 | 전체 로그 RMSE 전 → 후 | SFR>100 로그 RMSE 전 → 후 | 실제 0 로그 RMSE 전 → 후 | 전체 개선 fold | 꼬리 개선 fold |\n|---|---:|---:|---:|---:|---:|\n'
changes = []
for name, label in labels.items():
    b = summary[(summary.model == name) & (summary.phase == 'before')].iloc[0]
    a = summary[(summary.model == name) & (summary.phase == 'after')].iloc[0]
    p = pairs[pairs.model == name]
    wins = int((p.log_rmse_delta < 0).sum())
    tail_wins = int((p.tail100_log_rmse_delta < 0).sum())
    table += f'| {label} | {b.mean_fold_rmse:.6f} → {a.mean_fold_rmse:.6f} | {b.tail100_log_rmse:.6f} → {a.tail100_log_rmse:.6f} | {b.zero_log_rmse:.6f} → {a.zero_log_rmse:.6f} | {wins}/5 | {tail_wins}/5 |\n'
    changes.append(dict(model=name, overall_improvement_percent=100*(b.mean_fold_rmse-a.mean_fold_rmse)/b.mean_fold_rmse,
                        tail100_improvement_percent=100*(b.tail100_log_rmse-a.tail100_log_rmse)/b.tail100_log_rmse,
                        zero_improvement_percent=100*(b.zero_log_rmse-a.zero_log_rmse)/b.zero_log_rmse,
                        overall_improved_folds=wins, tail100_improved_folds=tail_wins))
pd.DataFrame(changes).to_csv(root / 'changes.csv', index=False)
gate = []
for phase in ('before', 'after'):
    values = comp[phase]
    truth_class = np.where(y == 0, 0, np.where(y <= 10, 1, 2))
    gate.append(dict(phase=phase, mean_tail_probability_on_sfr_gt100=float(values[extreme,2].mean()),
                     tail_class_argmax_recall=float((values[truth_class==2,:3].argmax(axis=1)==2).mean()),
                     extreme_expert_only_log_rmse=float(np.sqrt(np.mean((np.maximum(values[extreme,4],0)-np.log10(1+y[extreme]))**2)))))
pd.DataFrame(gate).to_csv(root / 'tail_gate_diagnostics.csv', index=False)
ztable = '| 모델 | 0 재현율 전 → 후 | 0 정밀도 전 → 후 | 실제 양수인데 0 출력 전 → 후 |\n|---|---:|---:|---:|\n'
for name, label in labels.items():
    b = summary[(summary.model==name)&(summary.phase=='before')].iloc[0]
    a = summary[(summary.model==name)&(summary.phase=='after')].iloc[0]
    ztable += f'| {label} | {b.zero_recall:.2%} → {a.zero_recall:.2%} | {b.zero_precision:.2%} → {a.zero_precision:.2%} | {int(b.false_zeros)} → {int(a.false_zeros)} |\n'
best = summary[summary.phase == 'after'].sort_values('mean_fold_rmse').iloc[0]
base = summary[(summary.phase == 'before') & (summary.model == 'leader_two_stage')].iloc[0]
overall_best_statement = ('낮았다' if best.mean_fold_rmse < base.mean_fold_rmse else '높았다')
model_file = meta['artifacts'][best.model]['file']
report = f'''# 같은 질량대 대비 성장량 잔차 추가 실험

## 결과

{table}

전체는 5-fold RMSE 평균이며, 실제 0·SFR>100 구간은 모든 OOF 행을 합쳐 계산한 로그 RMSE다.
모든 지표는 log10(1+SFR) 공간이다. 작은 값이 좋다. 극단 구간 SFR>100은 {extreme.sum():,}개다.
전체 pooled OOF RMSE와 SFR>10·>30·>100 구간, 로그 편향 및 선형 RMSE는 `summary.csv`에 있다.

추가 후 전체 점수가 가장 낮은 구성은 **{labels[best.model]}**({best.mean_fold_rmse:.9f})다.
피처 추가 전 조장 원본({base.mean_fold_rmse:.9f})보다 {overall_best_statement}.
이번 비교는 이전에도 사용한 CV에서 수행한 파생 변수 실험이며 독립 시험 성능이나 통계적 유의성을 의미하지 않는다.
결과를 보고 bin 수나 꼬리 경계를 추가 탐색하지 않았다. 현재 채택 모델은 바꾸지 않았다.

## 변수의 정확한 정의

`growth = log10_stellar_mass - log10_main_prog_stellar_mass`

`stellar_growth_residual_at_mass = growth - expected_growth(current_stellar_mass)`

각 CV 학습 fold의 현재 항성 질량을 20개 분위 구간으로 나눈다.
구간별 대표 질량은 중앙값, 기대 성장량은 평균이다. 대표 질량 사이를 선형 보간하며 양 끝 밖에서는 끝값을 유지한다.
검증 행은 이 고정된 기준선을 적용받는다. 같은 질량대의 기대 성장량보다 크면 잔차가 양수다.
분위 경계 중복은 제거하고 빈 구간은 사용하지 않는다.

여기서 성장량은 두 시점 항성 질량의 **로그 비율**이며 시간으로 나눈 성장률이 아니다.
원래 존재하는 질량·질량 차이 입력을 재표현한 피처이며 새로운 관측 정보가 추가된 것은 아니다.
전구체 질량의 센티널 후보도 기존 모델과 동일하게 유지했으므로 물리적 성장으로 단정하지 않는다.

## 비교와 누출 방지

- 입력 34→35개로 잔차 변수 하나만 추가했다. 원래 질량·성장 피처도 유지했다.
- 조장 HGB400 이진 두 단계, 3분류 T=10, 기존 이진 0 확률을 사용하는 3분류 T=10에 동일하게 적용했다.
- 각 구성의 분류기와 회귀기에 모두 35개 피처를 제공했다. 기존 0 확률 방식의 이진 분류기도 새 피처로 학습한다.
- HGB 학습률 .08, 400회, 잎 31개, 조기 종료 없음, 0 출력 문턱 .9를 고정했다.
- 기존 80:20 분할에서 학습 166,749행만 사용했다. KFold(5, shuffle=True, random_state=42).
- 기대 성장 곡선은 **각 fold 학습 입력만**으로 적합한다. 검증 입력과 SFR 타깃은 곡선 적합에 사용하지 않는다.
- 환경 피처만 기존 FAQ 계약대로 같은 전체 참고 은하 집합의 입력 기하로 계산했다.
- 피처 추가 전 OOF는 직전 실험의 캐시를 사용했고, 데이터 해시·행 번호·실제값·fold 및 모델 모듈이 일치하는지 확인했다.
- 원래 20% 검증셋의 성능은 이번에 평가하지 않았다.

{ztable}

## 꼬리 분류 진단

직접 3분류 T=10에서 SFR>100 은하의 평균 꼬리 확률은
{gate[0]['mean_tail_probability_on_sfr_gt100']:.6f} → {gate[1]['mean_tail_probability_on_sfr_gt100']:.6f}다.
꼬리 회귀기만 정답 구간을 알고 적용했을 때의 진단 RMSE는
{gate[0]['extreme_expert_only_log_rmse']:.6f} → {gate[1]['extreme_expert_only_log_rmse']:.6f}다.
이 진단값은 실제 배포 가능한 점수가 아니며 최종 혼합 점수와 구분한다.
자세한 분류 진단은 `tail_gate_diagnostics.csv`에 있다.

## 저장 모델과 실행

세 구성의 새 모델을 원래 학습 166,749행으로 다시 학습해 저장했다.
기대 성장량 변환기도 같은 학습 집합에 적합한 상태로 모델 파일에 포함돼 있다.
예측할 때 새 배치로 기준선을 다시 적합하면 안 된다.

이 폴더를 작업 디렉터리로 하고 Python 3.12와 `requirements.txt` 환경에서 사용한다.

```python
import joblib
import pandas as pd
from leader_model import predict_sfr

model = joblib.load('{model_file}')
reference = pd.read_csv('/path/to/Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv')
reference = reference.drop(columns='sfr_linear')
query = reference.iloc[:100].copy()
prediction = predict_sfr(query, model, reference)
```

`model.predict(X34)`도 지원한다. 원래 34개 피처를 주면 저장된 변환기가 35번째 피처를 자동 생성한다.
`leader_model.py`, `three_regime.py`, `zero_preserved.py`, `growth_residual.py`가 모듈 경로에 있어야 한다.
환경 기하용 참고 은하 집합은 기존 모델처럼 전체 고유 은하의 고정 합집합을 사용한다.

재현 명령:

```bash
python test_growth_residual.py
python run_experiment.py --data '/path/to/data.csv' --baseline-dir '/path/to/leader_three_regime_experiment'
```

baseline-dir에는 직전 실험의 모델 모듈과 OOF 캐시가 필요하다.
곡선의 배치 독립성, 범위 밖 처리, 동률 질량과 SFR 미사용 테스트를 통과했다.
모든 OOF 행의 1회 커버리지, 모델 저장/재로딩 일치, 원본 입력 표 예측 인터페이스도 확인했다.

## 파일

- `summary.csv`, `changes.csv`, `paired_differences.csv`, `fold_results.csv`: 비교 결과
- `oof_predictions.npz`: 전/후 예측, 실제값, fold, 원본 행 번호, fold 밖에서 계산한 잔차
- `fold_feature_curves.json`: fold별 질량-기대성장 기준선과 학습 행 수
- `*_residual.joblib`: 변환기를 포함한 학습 완료 모델
- `protocol.json`, `metadata.json`: 실행 전 조건과 결과/검증 기록
'''
(root / 'REPORT.md').write_text(report)
(root / 'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n'
    for p in sorted(root.iterdir()) if p.is_file() and p.name != 'SHA256SUMS'))
archive = root.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.iterdir()):
        if p.is_file(): z.write(p, arcname=f'{root.name}/{p.name}')
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
print(table)
print(pd.DataFrame(changes).to_string(index=False))
print(pd.DataFrame(gate).to_string(index=False))

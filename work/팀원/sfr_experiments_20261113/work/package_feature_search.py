from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[1]/'outputs/feature_search'
meta=json.loads((root/'metadata.json').read_text())
frozen=json.loads((root/'frozen_selection.json').read_text())
stage=pd.read_csv(root/'stage2_summary.csv')
confirm=pd.read_csv(root/'confirmation_summary.csv').set_index('model')
folds=pd.read_csv(root/'confirmation_folds.csv')
screen=pd.read_csv(root/'screening_summary.csv')
plan=frozen['plan']
baseline=stage[stage.model=='baseline'].iloc[0]
winner=stage[stage.model==frozen['name']].iloc[0]
baseline_features=json.loads((root/'stage2_plans.json').read_text())['plans']['baseline']['gate']
feature_rows=[]
for slot,cols in plan.items():
    feature_rows.append(dict(component=slot,count=len(cols),
                             added=[c for c in cols if c not in baseline_features],
                             removed=[c for c in baseline_features if c not in cols],features=cols))
(root/'selected_features.json').write_text(json.dumps(feature_rows,ensure_ascii=False,indent=2)+'\n')

def table(frame):
    lines=['| 구성 | 전체 로그 RMSE | 실제 0 로그 RMSE | SFR>100 로그 RMSE | 0 재현율 | 0 정밀도 |',
           '|---|---:|---:|---:|---:|---:|']
    for _,r in frame.iterrows():
        lines.append(f'| {r.model} | {r.mean_fold_rmse:.6f} | {r.zero_log_rmse:.6f} | {r.tail100_log_rmse:.6f} | {r.zero_recall:.2%} | {r.zero_precision:.2%} |')
    return '\n'.join(lines)

paired=[]
sel=folds[folds.model=='selected'].set_index('fold')
for reference in ['three_baseline','leader_two_stage','current_A']:
    ref=folds[folds.model==reference].set_index('fold')
    for fold in range(1,6):
        paired.append(dict(reference=reference,fold=fold,**{c+'_delta':float(sel.loc[fold,c]-ref.loc[fold,c])
            for c in ['log_rmse','zero_log_rmse','tail100_log_rmse']}))
pd.DataFrame(paired).to_csv(root/'confirmation_paired_differences.csv',index=False)
wins=sum(r['reference']=='three_baseline' and r['log_rmse_delta']<0 for r in paired)
improvement42=100*(baseline.mean_fold_rmse-winner.mean_fold_rmse)/baseline.mean_fold_rmse
improvement7=100*(confirm.loc['three_baseline','mean_fold_rmse']-confirm.loc['selected','mean_fold_rmse'])/confirm.loc['three_baseline','mean_fold_rmse']
current42=json.loads((root.parent/'a34_zero_gate_adopted/metadata.json').read_text())['after_rmse']
best_vs_current7=confirm.loc['selected','mean_fold_rmse']-confirm.loc['current_A','mean_fold_rmse']
recommendation=dict(frozen_candidate=frozen['name'],
    improvement_percent_vs_three_baseline_seed42=improvement42,
    improvement_percent_vs_three_baseline_seed7=improvement7,
    confirmation_overall_improved_folds_vs_three_baseline=wins,
    confirmation_mean_rmse_delta_vs_current_A=float(best_vs_current7),
    current_A_seed42_pooled_oof_rmse=current42,selected_seed42_pooled_oof_rmse=float(winner.log_rmse),
    confirmation_zero_rmse_delta_vs_current_A=float(confirm.loc['selected','zero_log_rmse']-confirm.loc['current_A','zero_log_rmse']),
    confirmation_tail100_rmse_delta_vs_current_A=float(confirm.loc['selected','tail100_log_rmse']-confirm.loc['current_A','tail100_log_rmse']),
    confirmation_overall_improved_folds_vs_current_A=sum(r['reference']=='current_A' and r['log_rmse_delta']<0 for r in paired),
    replaces_current_model=False,
    scope='Best frozen plan in HGB400 three-regime feature search; not a global optimum or independent-test claim')
(root/'recommendation.json').write_text(json.dumps(recommendation,ensure_ascii=False,indent=2)+'\n')
feature_table='| 부품 | 피처 수 | 추가 | 제거 |\n|---|---:|---|---|\n'
slot_names={'gate':'3분류기','mid':'일반 회귀기','tail':'꼬리 회귀기'}
for r in feature_rows:
    feature_table+=f"| {slot_names[r['component']]} | {r['count']} | {', '.join(r['added']) or '없음'} | {', '.join(r['removed']) or '없음'} |\n"
report=f'''# 새 변수 추가·기존 피처 제거 탐색

## 선택과 재확인

이번 탐색 범위의 선택은 **{frozen['name']}**다. 선택 단계(seed42)의 기존 HGB400 3분류 대비
평균 전체 로그 RMSE 개선율은 {improvement42:+.3f}%, 별도 분할 시드7에서는 {improvement7:+.3f}%다.
개선율이 음수면 악화다. 시드7에서 3분류 원본보다 전체 점수가 낮은 fold는 {wins}/5다.
시드7 결과를 보기 전에 조합을 고정했고, 그 결과를 보고 조합을 바꾸지 않았다.

### 선택 단계: 같은 5-fold, 시드42

{table(stage)}

### 고정 후 분할 민감도 확인: 5-fold, 시드7

{table(confirm.reset_index())}

`selected`는 시드42에서 고른 피처 구성, `three_baseline`은 조장 코드에서 만든 34개 피처의
0/일반/꼬리 HGB400 T=10 모델, `leader_two_stage`는 조장 원본의 0/양수 HGB400 모델이다.
`current_A`는 현재 채택된 LightGBM/LightGBM/HGB + P0≥.9 모델을 동일한 시드7 fold에 재학습한 결과다.
현재 A는 원래의 환경 피처 계산 순서와 학습 설정을 유지했다.

위 전체 점수는 fold RMSE 평균이며, 0·SFR>100 지표는 전체 OOF 행을 모아 계산했다.
모든 점수는 log10(1+SFR) 공간이다. SFR>10·>30·>100, 선형 RMSE와 편향도 CSV에 있다.
시드42에서 선택 후보의 pooled OOF RMSE는 {winner.log_rmse:.9f}, 기존 A의 저장된 pooled OOF는 {current42:.9f}다.
시드7에서 선택 후보−현재 A의 평균 RMSE 차이는 {best_vs_current7:+.9f}다. 음수면 선택 후보가 낮다.
현재 채택 모델은 자동 교체하지 않았다.

## 선택한 피처 구성

{feature_table}

전체 열 목록은 `selected_features.json`, 기계가 읽을 설정은 `frozen_selection.json`에 있다.
**피처 제거는 해당 학습기의 입력 열에서 제외한다는 뜻이다. 원자료 열을 지우는 것이 아니다.**
예를 들어 pos/vel을 모델 입력에서 빼더라도 환경·동반 은하 특성 계산에는 원본 위치·속도가 필요하다.
어떤 변수를 제외해 점수가 좋아졌다고 해서 그 물리량이 천문학적으로 무의미하다는 결론은 낼 수 없다.

## 탐색한 후보

추가 묶음 5개와 전체 추가 조합:

- 질량×성장: `(log10_stellar_mass-10) * dlogMs_growth`.
- 병합×성장: `log10_mass_ratio > -6` 표시, 그 표시×성장, `max(log10_mass_ratio+6,0)`×성장.
  -6의 물리적 의미를 단정하지 않으며 주어진 수치 구간을 표현하는 변수다.
- 같은 질량대 대비 항성 성장량 잔차: 현재 항성 질량별 평균 `dlogMs_growth`에서 벗어난 양.
- 같은 질량대 대비 BH 잔차: 현재 항성 질량별 평균 `log10_BH_mass`에서 벗어난 양.
- 동반 은하 6개: 최근접 거리의 로그, 최근접 은하/자신의 항성 질량 로그 비율, 상대속력,
  거리/자신의 R200, 가까운 3개 중 최대 항성 질량 로그 비율, 가까운 3개 중 최소 상대속력.

제거 묶음 8개: 원본 위치3개, 속도3개, 위치·속도6개, 전구체 질량2개, 원본 병합 지표,
원본 BH 질량, 10^14 이상 호스트 관련 거리·속도3개, FoF 그룹 상대속력.

BH/성장 잔차의 질량 조건부 평균은 각 학습 fold의 20개 질량 분위 구간에서 평균을 구하고,
구간 대표 질량(중앙값) 사이를 선형 보간했다. 검증 입력이나 SFR 타깃으로 기준선을 적합하지 않는다.
성장은 시간으로 나눈 성장률이 아니라 `log M현재 − log M전구체`다. 센티널 후보는 원본처럼 유지했다.
동반 은하는 제공된 전체 입력 표의 주기 경계 상자200 안 최근접 다른 은하다.
최근접이 실제로 중력적으로 묶였거나 병합 중이라는 가정을 하지 않는다. 자기 자신은 이웃에서 제외했다.

## 탐색 절차

1. 기존 학습80% 안의 KFold5 시드42 중 첫 두 fold에서 후보14개와 기준선을 계산했다.
2. 각 후보의 세 부품 출력을 저장해 전체 교체 및 한 부품만 교체했을 때의 점수를 비교했다.
3. 전체 피처 구성 중 가장 좋은 것, 부품별 가장 좋은 것을 결합한 것, 개별로 도움이 된 추가·제거를
   합친 것을 5개 fold 전체에서 평가했다. 중복 조합은 제거했다.
4. 합칠 때는 한 부품 교체의 두 fold 평균 개선이 최소0.00002인 개별 묶음만 포함했다.
5. 시드42 전체 fold 평균 RMSE 최소를 선택하고 동점이면 부품 피처 수 합이 적은 것을 택했다.
6. 조합을 고정한 후 시드7에서 원본3분류·조장2단계·현재A와 비교했다.

학습률 .08, HGB400, 최대 잎31, 조기 종료 없음, T=10, 0 문턱 .9는 고정했다.
일반/꼬리 회귀기는 실제 SFR 구간으로 학습하고 예측 때는 확률로 로그 공간에서 섞는다.
원래20% 검증셋은 이번 탐색에서 채점하지 않았다. 여러 번 사용한 관측값의 다른 분할로 확인한 것이므로
독립 데이터 시험은 아니다. 후보 선택에 따른 낙관 편향이 남으며 모든 변수 조합의 전역 최적을 찾은 것은 아니다.
두 fold 선별에서 탈락한 후보를 모두5fold로 평가한 것은 아니다.

## 사용

`best_feature_model.joblib`는 선택한 구성을 원래 학습166,749행에 적합한 파일이다.
질량 조건부 평균 변환도 모델에 포함돼 있으므로 예측 배치에서 다시 학습하지 않는다.
이 폴더에서 Python3.12, requirements.txt 환경으로 실행한다.

```python
import joblib
import pandas as pd
from feature_pipeline import predict_sfr

model = joblib.load('best_feature_model.joblib')
reference = pd.read_csv('/path/to/Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv')
reference = reference.drop(columns='sfr_linear')
query = reference.iloc[:100].copy()
prediction = predict_sfr(query, model, reference)
```

환경·동반 은하 계산에는 같은 고유 은하 합집합을 고정해 사용한다.
작은 배치만으로 이웃을 다시 정의하면 입력이 달라진다.
피처 모델은 34개 기본 피처+6개 동반 은하 피처를 받은 뒤 내부에서 파생 변수를 생성하고 각 부품의 열을 선택한다.

재현에는 원본 데이터와 이전 `leader_three_regime_experiment`, `a34_zero_gate_adopted` 폴더가 필요하다.
`run_search.py --help`의 data, old-dir, current-a-dir, cache 인자를 지정한다.
캐시는 코드·데이터·분할·열 목록 지문으로 구분되며 정확히 같은 부품 결과만 재사용한다.
단위 검증: `python test_features.py`.
macOS에서 LightGBM 비교를 재실행하려면 OpenMP 런타임 경로도 설정해야 할 수 있다.

## 파일

- screening_folds.csv / screening_summary.csv: 첫 두 fold의 전체·부품별 효과
- stage2_plans.json / stage2_summary.csv / stage2_folds.csv: 조합 비교
- frozen_selection.json / selected_features.json: 고정된 최종 구성
- confirmation_summary.csv / confirmation_folds.csv / confirmation_paired_differences.csv: 시드7 재확인
- stage2_oof.npz / confirmation_oof.npz: 원본 행 번호·fold·실제값·예측
- protocol.json / component_fits.json / metadata.json / recommendation.json: 조건·실행·검증·판단 기록

주기 경계 이웃, 자기 제외, 행 순서, SFR 미사용, 잔차 변환의 배치 독립성 검증을 통과했다.
저장/재로딩 일치와 원본 입력 표를 통한 최종 예측도 확인했다.
'''
(root/'REPORT.md').write_text(report)
readme=f'''# 피처 탐색 결과: {frozen['name']}

학습 완료 모델: `best_feature_model.joblib`.
구조는 0 / 0<SFR≤10 / SFR>10의 HGB 다중분류기와 실제 일반·꼬리 구간별 HGB 로그 회귀기다.
모든 학습기는 학습률 .08, 400회, 잎 최대31, 조기 종료 없음이다.
P0≥.9이면 0, 나머지는 두 회귀기의 로그 예측을 확률로 섞어 선형 SFR로 반환한다.

- 시드42 전체5-fold 평균 로그 RMSE: 기존3분류 {baseline.mean_fold_rmse:.6f} → 선택 {winner.mean_fold_rmse:.6f}
- 고정 후 시드7: 기존3분류 {confirm.loc['three_baseline','mean_fold_rmse']:.6f} → 선택 {confirm.loc['selected','mean_fold_rmse']:.6f}
- 시드7의 현재 채택A: {confirm.loc['current_A','mean_fold_rmse']:.6f}
- 모델별 피처 수: {', '.join(slot_names[r['component']]+str(r['count'])+'개' for r in feature_rows)}

이번 후보 범위의 최저 CV 구성이다. 같은 관측값의 분할을 바꿔 확인했으며 독립 시험이 아니다.
현재 채택 파일이나 GitHub는 변경하지 않았다.

[상세 보고서](REPORT.md), [최종 피처 목록](selected_features.json), [비교 결과](confirmation_summary.csv).

이 폴더를 작업 디렉터리로 하고 Python3.12와 requirements.txt 환경에서 사용한다.

```python
import joblib
import pandas as pd
from feature_pipeline import predict_sfr

model = joblib.load('best_feature_model.joblib')
reference = pd.read_csv('/path/to/Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv')
reference = reference.drop(columns='sfr_linear')
prediction = predict_sfr(reference.iloc[:100].copy(), model, reference)
```

전체 고유 은하 참고 집합을 고정해 환경·동반 은하 변수를 계산한다.
질량별 평균 곡선은 저장 모델에 포함되며 예측 배치에서 다시 학습하지 않는다.
원본 열이 학습기 입력에서 제외돼도 파생 변수 계산에는 필요할 수 있으므로 원본13개를 모두 전달한다.
'''
(root/'README.md').write_text(readme)
(root/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n'
    for p in sorted(root.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
archive=root.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.iterdir()):
        if p.is_file(): z.write(p,arcname=root.name+'/'+p.name)
with zipfile.ZipFile(archive) as z: assert z.testzip() is None
print(json.dumps(recommendation,ensure_ascii=False,indent=2))
print(feature_table)
print(table(confirm.reset_index()))

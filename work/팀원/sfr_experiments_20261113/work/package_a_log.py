from pathlib import Path
import json, io, contextlib, zipfile, shutil
import numpy as np
root=Path('/Users/ljh/Documents/Codex/2026-11-11/x20')
out=root/'outputs/a_log_development'
old=np.load(root/'outputs/log_sfr_reevaluation/oof_predictions.npz')
np.savez_compressed(out/'reference_predictions.npz',**{k:old[k] for k in ['actual','fold','source_row_index','A_soft_T30','A_hard_T30','B_T30','HGB_log1p']})
shutil.copy(root/'outputs/b_tail_diagnosis/requirements.txt',out/'requirements.txt')
report='''# A를 로그 SFR 평가에 맞게 발전시킨 결과

2026-11-11. 원래 main 커밋 fef00ffaf2992104c8d6dc0e87ab0b30275d8c9d 기준. 모델은 새 파일 `a_log_mixture.py`의 `LogTailMixtureSFR`로 구현했다.

## 핵심 변경

0 / 보통 / 꼬리 3부류 분류기는 유지하고, 두 회귀기가 원래 SFR 대신 `z = ln(SFR+c)`를 학습하도록 했다. 회귀 결과는 **로그 공간에서 확률로 섞은 다음** 원래 SFR로 되돌린다.

`z_hat = P(0|X)*ln(c) + P(보통|X)*E[z|보통,X] + P(꼬리|X)*E[z|꼬리,X]`

`SFR_hat = max(exp(z_hat)-c, 0)`

c=1일 때만 0 부류의 로그값이 0이다. c가 0.001이면 0 부류 항 ln(0.001)을 반드시 포함한다. 작은 상수에서 이 항을 빠뜨리면 로그 공간 혼합이 아니다. smearing 보정은 적용하지 않았다.

## 실험 범위

- 제공 데이터와 해시가 일치하는 CSV, 기존 학습 166,749개, 같은 5-fold OOF.
- 검증셋 미사용. 기존 A의 HGB 설정(learning_rate=0.08, max_iter=150, max_leaf_nodes=31, random_state=42) 유지.
- 로그 학습+로그 혼합, 로그 학습+선형 혼합, 하드 분기, 회귀 출력의 부류 범위 제한, T=10을 비교.
- 공식 0 처리 식이 미확정이라 c=1과 c=0.001을 별도 평가 시나리오로 실험했다. c는 점수를 보고 고르는 하이퍼파라미터가 아니라 공식 평가식에 맞춰야 하는 값이다.

## log10(1+SFR) 기준

| 모델 | 전체 OOF RMSE |
|---|---:|
| 기존 A soft T=30 | 0.115995 |
| 기존 B T=30 | 0.117033 |
| 로그 타깃 단일 HGB | 0.111775 |
| 새 A: 로그 학습, 선형 혼합, T=30 | 0.111157 |
| 새 A: 로그 학습, 로그 혼합, T=30 | 0.108273 |
| **새 A: 로그 학습, 로그 혼합, T=10** | **0.107447** |

새 A T=10은 기존 A보다 **7.4%**, 로그 타깃 HGB보다 **3.9%** 낮다. 두 비교 모두 5개 fold에서 전부 개선됐다. T=30을 유지해도 개선되므로 효과가 임계값 변경만으로 생긴 것은 아니다. T=10은 새 T=30보다 5개 fold 중 4개에서 좋았으나, 차이가 작아 보편적인 최적 임계값이라고 단정하지 않는다.

회귀 출력을 해당 부류의 범위로 제한한 버전은 이번 예측에서 실질적인 변화가 없었다. 하드 분기는 같은 지표에서 0.113776으로 부드러운 로그 혼합보다 나빴다.

### 구간별 장단점

| 실제 구간 | 기존 A T=30 | 로그 타깃 HGB | 새 A-log T=10 |
|---|---:|---:|---:|
| SFR=0 | 0.187195 | 0.145245 | **0.131733** |
| 0<SFR≤3 | 0.093867 | 0.089088 | **0.087555** |
| 3<SFR≤30 | 0.124497 | 0.128806 | **0.123909** |
| 30<SFR≤100 | **0.190321** | 0.229979 | 0.218490 |
| SFR>100 | **0.498365** | 0.711481 | 0.672682 |

낮은 SFR을 더 잘 맞춰 전체 로그 점수가 좋아졌다. 기존 A 대비 높은 SFR의 로그 오차는 나빠졌으며 모든 구간에서 좋아진 모델은 아니다. 반면 단일 로그 타깃 HGB보다는 표시한 모든 구간에서 개선됐다.

## log10(SFR+0.001) 기준의 별도 실험

| 이 식에 맞춰 평가한 모델 | 전체 OOF RMSE |
|---|---:|
| 기존 A hard T=30 | 0.445956 |
| ln(SFR+0.001) 타깃 단일 HGB | 0.406791 |
| 새 A-log(c=0.001), T=30 | 0.391420 |
| **새 A-log(c=0.001), T=10** | **0.389652** |

같은 변환으로 학습한 단일 HGB보다 약 **4.2%** 낮고, 5개 fold 모두 개선됐다. 이 표는 공식 상수를 0.001로 확정한다는 뜻이 아니다. c=1로 학습한 모델과 c=0.001로 학습한 모델은 서로 다른 모델이다.

## 현재 판단

`LogTailMixtureSFR(tail_threshold=10, log_offset=1)`을 log1p 기준의 유력 후보로 남긴다. 공식 평가가 log(SFR+0.001)이면 offset=0.001 후보를 사용해야 한다. 실제 0 처리 규칙이 다른 경우 이에 맞춰 학습·평가를 다시 정의해야 한다.

이 결과는 같은 학습 CV에서 후보를 개발·비교한 것이며 독립 최종 테스트 성능이 아니다. 기존 검증셋은 이번에 평가하지 않았다. GitHub 저장소도 수정하지 않았다.

## 사용법

```python
from a_log_mixture import LogTailMixtureSFR

# X_train: 원래 순서의 입력 피처 13개, y_train: 원래 선형 SFR
model = LogTailMixtureSFR(tail_threshold=10, log_offset=1.0)
model.fit(X_train, y_train)
predictions = model.predict(X_new)  # 출력은 원래 SFR 단위, 비음수
```

`log_offset`은 공식 평가의 변환에 맞춘다. 자연로그로 학습하고 같은 오프셋의 log10으로 평가해도 상수배 차이만 있어 목적과 순위는 같다. 3부류가 모두 있는 학습 데이터가 필요하다. 피처의 열 순서는 학습·예측에서 동일하게 유지해야 한다.

## 전체 실험 재현

```sh
python -m pip install -r requirements.txt
python run_experiment.py --data "Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv" --references reference_predictions.npz --out rerun_results
python test_a_log_mixture.py
```

묶음에 비교 기준의 OOF 예측을 포함했다. 원본 CSV는 포함하지 않았다. 단위 테스트 3개가 통과했으며, 비단위 오프셋의 0 항, 하드 분기의 정확한 0 출력, clone/fit/predict와 잘못된 입력 처리 등을 확인했다. 전체 OOF 예측의 유한성·비음수성·로그 혼합 항등식도 실행 중 검사했다.

`scores.csv`, `fold_scores.csv`, `overall_comparison.csv`, `metadata.json`, `oof_predictions.npz`에 상세 결과와 재현 정보가 있다.
'''
(out/'REPORT.md').write_text(report)
meta=json.loads((out/'metadata.json').read_text())
meta['tests']='3 unit tests passed; all OOF predictions finite/nonnegative; log-space mixture identity verified.'
meta['numerical_note']='Hard-routed zero class returns exact zero; inverse-transform floating roundoff (~2e-19 at offset=.001) corrected without changing routes or training.'
(out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))

cells=[];env={};count=0
def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
def code(s):
 global count
 count+=1;buf=io.StringIO()
 with contextlib.redirect_stdout(buf):exec(s,env)
 cells.append(dict(cell_type='code',metadata={},execution_count=count,source=s.splitlines(True),outputs=[dict(output_type='stream',name='stdout',text=buf.getvalue().splitlines(True))]))
md(report.split('## 使用法')[0] if '## 使用法' in report else report.split('## 사용법')[0])
import os
os.chdir(out)
code("import pandas as pd\nfrom pathlib import Path\nroot=Path('.')\nscores=pd.read_csv(root/'overall_comparison.csv',index_col=0)\nprint(scores.round(6).to_string())\n")
code("folds=pd.read_csv(root/'fold_scores.csv')\nt=folds[folds.metric=='log10_1plus'].pivot(index='fold',columns='model',values='rmse')\nprint(t[['A_soft_T30','HGB_log1p','A_log1p_T30_soft','A_log1p_T10_soft']].round(6).to_string())\nprint('New A beats log HGB in folds:',int((t.A_log1p_T10_soft<t.HGB_log1p).sum()))\n")
md('## 사용과 재현\n'+report.split('## 사용법')[1])
nb=dict(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12.14'}},nbformat=4,nbformat_minor=5)
for i,c in enumerate(cells):c['id']=f'a-log-{i:02d}'
(out/'A_log_results.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1))
with zipfile.ZipFile(root/'outputs/a_log_development.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(out.iterdir()):
  if p.is_file():z.write(p,arcname='a_log_development/'+p.name)
with zipfile.ZipFile(root/'outputs/a_log_development.zip') as z:
 assert z.testzip() is None
 print('Archive verified:',len(z.namelist()),'files')
print('Package complete')

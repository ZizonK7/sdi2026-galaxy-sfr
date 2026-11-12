"""Refresh report prose from the completed LightGBM notebook execution."""
import ast
import copy
import json
from pathlib import Path

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BENCH = ROOT / "work/조장/model_comparison_2026-11-12"


def main():
    evidence = json.loads((HERE / 'execution_evidence.json').read_text(encoding='utf-8'))
    raw_evidence = HERE / 'execution_evidence_raw.json'
    if not raw_evidence.exists():
        raw_evidence.write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    r, d, execution = evidence['results'], evidence['diagnostics'], evidence['execution']
    gate_counts = {True: 0, False: 0}
    for entry in evidence['gate_cost']:
        if entry['gated']:
            gate_counts[bool(entry['true_zero'])] += int(entry['n'])
    assert r['model_name'] == 'TwoStageLightGBMLogSFR' and r['algorithm'] == 'LightGBM'
    nb = nbformat.read(ROOT / 'DataHub_701104_final.ipynb', as_version=4)
    old = nbformat.read(HERE / 'reference/DataHub_701104_final_hgb.ipynb', as_version=4)
    old_md = {c.source.splitlines()[0]: c.source for c in old.cells if c.cell_type=='markdown' and c.source}

    def update(prefix, text):
        found = [c for c in nb.cells if c.cell_type=='markdown' and c.source.startswith(prefix)]
        assert len(found)==1, prefix
        found[0].source = text.strip()+'\n'

    table = pd.DataFrame(evidence['log_table']).set_index('model')
    final_key = '최종 LightGBM 모델 (34개, 400회, 문턱 0.9)'
    previous_key = '이전 HGB 모델 (34개, 조기 종료, 문턱 0.9)'
    final, previous = table.loc[final_key], table.loc[previous_key]
    cv = pd.DataFrame(r['algorithm_comparison']['summary']).set_index('model')
    lgb, hgb, ref = (cv.loc[m] for m in ['lightgbm_two_stage_fixed400','hgb_two_stage_fixed400','hgb_two_stage_reference'])
    delta = float(lgb.mean_rmse_log-hgb.mean_rmse_log)
    saving = 100*(1-lgb.mean_training_seconds/hgb.mean_training_seconds)
    choice = (f"HGB 400회 두 단계의 교차검증 평균 로그 RMSE는 {hgb.mean_rmse_log:.6f}, LightGBM은 {lgb.mean_rmse_log:.6f}로 관측 차이가 {delta:.6f} dex였다. "
              f"평균 학습 시간은 각각 {hgb.mean_training_seconds:.2f}초와 {lgb.mean_training_seconds:.2f}초로, "
              f"관측 정확도의 작은 차이를 수용하고 학습 시간이 약 {saving:.1f}% 짧은 LightGBM 두 단계를 최종 모델로 선택했다.")
    update('## 최종 보고서 읽기 안내', f'''## 최종 보고서 읽기 안내

Part I는 중간 분석과 고정 템플릿 코드를 보존한 기록이다. 당시의 선형 평가 설명과 모델 선택은 Part II의 로그 평가 정의 및 최종 결론과 함께 읽어야 한다.

최종 모델은 **LightGBM 분류기와 양수 로그 회귀기를 결합한 두 단계 모델**이다. {choice}

FAQ Q6에 따라 로그 SFR RMSE를 주지표로 삼았으며 구체적 변환은 **log10(SFR + 1)**로 가정했다. 같은 공통 검증 표본의 최종 LightGBM RMSE는 **{r['validation_log_rmse']:.7f} dex**다. 위 학습 내부 교차검증 평균과 이 검증셋 점수는 별개의 값이다. 검증 표본은 Part I부터 재사용했다.

변경·선택 근거는 9.1–9.3절, 비교 결과는 10절, 오차·물리 해석은 11–12절, 한계·결론은 13절, 모델 카드와 기계 판독 결과는 14–15절에 정리했다.''')
    rows = [
        ('Baseline','Baseline (HGB 13, linear target)','HGB · 원래 13개','선형 SFR'),
        ('중간 단일 로그 회귀','Midterm 7(b): log1p direct regression (13)','HGB · 원래 13개','ln(1+SFR), 역변환'),
        ('중간 두 단계','Midterm 7(d): two-stage, linear (13 / 13)','HGB · 13/13개','선형 SFR, 확률 곱'),
        ('Part II 초안','Part II draft (11-07): env clf x linear reg (30 / 13)','HGB · 환경 분류 30 / 회귀 13','선형 SFR, 확률 곱'),
        ('성장 피처 이전 후보','로그 두 단계 (30개, 트리 상한 400, 문턱 0.8)','HGB · 30/30개','로그 결합, 0 문턱 0.8'),
        ('이전 제출 HGB',previous_key,'HGB · 34/34개, 조기 종료','로그 결합, 0 문턱 0.9'),
        ('**최종 LightGBM**',final_key,'**LightGBM · 34/34개, 각 400회**','로그 결합, 0 문턱 0.9'),
    ]
    val_lines = ['| Model | 입력·알고리즘 | 타깃 처리 | 검증 MAE | 검증 RMSE | R² | 검증 로그 RMSE (D1) |',
                 '|---|---|---|---:|---:|---:|---:|']
    for name,key,features,target in rows:
        a=table.loc[key]
        val_lines.append(f"| {name} | {features} | {target} | {a['linear MAE']:.4f} | {a['linear RMSE']:.4f} | {a['linear R2']:.4f} | {a['D1 log RMSE']:.6f} |")
    retrained=table.loc['ref: FINAL retrained for D2 (LOG_EPS = 0.01)']
    update('### 10.1 Required Comparison Table', '''### 10.1 Required Comparison Table — **STUDENT RESPONSE**

**공통 검증 표본의 성능.** 동일한 검증 은하 41,688개에서 계산했다. 선형 지표에는 각 모델의 원래 예측을 사용하고 로그 지표에서만 음수를 0으로 절단했다. 모델 선택의 근거인 학습 내부 교차검증·시간 표와 구분한다. LightGBM 선택 후 이 표를 계산했으며 이 검증셋으로 알고리즘을 다시 고르지 않았다.

'''+'\n'.join(val_lines)+f'''

### 10.2 Results Discussion

**정확도와 학습 시간의 절충.** {choice} 기존 조기 종료 HGB는 CV 평균 {ref.mean_rmse_log:.6f}, 학습 시간 {ref.mean_training_seconds:.2f}초였다. 따라서 시간 이점은 최고 정확도 후보인 HGB 400회 모델을 기준으로 한 관측값이며, 이전 제출 HGB보다 빠르다는 근거는 아니다. 시간·정확도 절충은 비교 결과 확인 후 정한 선택 기준이며 동등성이나 비열등성을 입증하지 않았다.

최종 LightGBM의 공통 검증 로그 RMSE는 **{r['validation_log_rmse']:.6f}**, 선형 RMSE는 **{r['validation_rmse']:.4f}**, MAE는 **{r['validation_mae']:.4f}**, R²는 **{r['validation_r2']:.4f}**다. 실제 SFR=0과 SFR>0의 로그 RMSE는 각각 {r['zero_sfr_log_rmse']:.6f}와 {r['positive_sfr_log_rmse']:.6f}다. 구간별 오차와 정확한 0 출력의 손익은 11절에서 확인한다.

과거 HGB 실험에서 로그 타깃과 성장·비율 요약의 효과를 확인했다. 그 선택 이력과 환경×성장 대조는 HGB에서 수행한 결과이며, LightGBM에서 동일한 전체 제거 실험을 반복한 것으로 해석하지 않는다. 세 알고리즘의 400회 비교에서는 두 단계가 단일 회귀보다 5/5개 분할의 오차가 낮았지만, 두 단계는 모델 두 개를 학습하므로 계산량이 더 크다.

**로그 정의 민감도.** 같은 최종 예측을 D2=log10(SFR+0.01)로 평가한 값은 {r['validation_log_rmse_eps0.01']:.6f}다. LightGBM을 eps=0.01로 다시 학습한 참고 모델의 D2 RMSE는 {retrained['D2 log RMSE']:.6f}, D1 RMSE는 {retrained['D1 log RMSE']:.6f}였다. 문턱은 0.9로 고정했고 이 참고 결과로 모델을 재선택하지 않았다. 공식 로그 정의가 확인되면 학습·평가·문턱을 같은 정의로 다시 검토해야 한다.

위 계산표의 ref 행은 문턱·로그 정의 민감도와 과거 HGB 변형의 참고 결과다. 행 단위 쌍 부트스트랩 구간은 적합된 모델과 현재 표본에 조건부인 값이며 공간 의존과 선택 변동을 반영하지 않는다.''')
    update('### 11.1 Error-Analysis Discussion', f'''### 11.1 Error-Analysis Discussion — **STUDENT RESPONSE**

최종 LightGBM은 실제 SFR=0인 은하 {d['true_zero_count']:,}개 중 {d['zero_true_positive']:,}개를 정확히 0으로 예측했고, 양수인 은하 {d['positive_predicted_zero']:,}개에도 0을 출력했다. 정확한 0 출력의 정밀도는 {100*d['zero_precision']:.2f}%, 실제 0 재현율은 {100*d['zero_recall']:.2f}%였다.

**0인 은하.** 로그 RMSE는 {r['zero_sfr_log_rmse']:.6f} dex이며 전체 로그 오차제곱합(SSE)의 {100*d['zero_sse_share']:.2f}%를 차지했다. 실제 0인데 양수 확률이 0.5를 넘은 {d['zero_missed_by_p05_count']}개가 이 집단 SSE의 {100*d['zero_missed_by_p05_sse_share']:.2f}%를 차지했다. 분류 확률의 오류가 최종 예측의 오차로 이어질 수 있다.

**양수인 은하.** 로그 RMSE는 {r['positive_sfr_log_rmse']:.6f} dex였다. 작은 값의 과대예측과 큰 값의 과소예측은 위 구간별 잔차표와 그림에서 평가한다. 회귀기는 실제 양수만으로 학습하지만, 최종 성능은 0을 포함한 검증 은하 전체에서 계산했다.

**큰 꼬리.** SFR>100인 은하는 {d['high_sfr_count']}개였고 평균 로그 잔차는 {d['high_sfr_mean_log_residual']:+.4f} dex였다. 이 구간은 전체 로그 SSE의 {100*d['high_sfr_sse_share']:.2f}%를 차지했다. 로그 예측 목적 아래에서도 큰 SFR의 과소예측이 남았다.

0 출력 문턱의 손익과 입력 특수값·환경 구조에 관한 진단은 다음 절에 정리했다.''')
    old_gate = old_md['#### 11.2 최종 모델의 오차 진단']
    prefix = old_gate.split('**0 출력 문턱의 손익.**')[0]
    tail = '**전구체 특수값'+old_gate.split('**전구체 특수값',1)[1]
    clipping_note = (f" 문턱과 별개로 역변환 후 음수를 0으로 절단한 실제 0 은하가 {d['zero_true_positive']-gate_counts[True]}개 있어, 11.1절의 전체 0 출력 수와 문턱 적용 수는 구분했다."
                     if d['zero_true_positive'] != gate_counts[True] else '')
    update('#### 11.2 최종 모델의 오차 진단', prefix + f'''**0 출력 문턱의 손익.** 최종 LightGBM에서 실제 0인 은하 {gate_counts[True]:,}개와 양수인 은하 {gate_counts[False]}개가 문턱에 의해 0으로 출력됐다. 전체 MSE에 대한 기여 변화는 실제 0에서 **{d['gate_mse_delta_true_zero']:+.10f}**, 양수에서 **{d['gate_mse_delta_positive']:+.10f}**, 합계 **{d['gate_mse_delta_all']:+.10f}**였다. 이는 고정한 문턱의 현재 검증셋 진단이며, LightGBM의 최적 문턱을 입증하거나 문턱을 재선택한 결과가 아니다.{clipping_note}

''' + tail)
    def feature_names(records):
        return ', '.join(str(x.get('index',next(iter(x.values())))) for x in records[:4])
    interpretation = old_md['### 12.1 Scientific Interpretation — **STUDENT RESPONSE**']
    start = interpretation.index('분류기는')
    end = interpretation.index('순열 중요도는')
    interpretation = interpretation[:start] + f"최종 LightGBM의 분류기 순열 중요도 상위 입력은 {feature_names(evidence['importance_classifier'])}였고, 양수 로그 회귀기의 상위 입력은 {feature_names(evidence['importance_regressor'])}였다. " + interpretation[end:]
    calibration = (f"P(0)=0.88–0.92인 검증 은하 {d['p0_band_count']}개 중 실제 0은 {d['p0_band_actual_zero']}개"
                   f"({100*d['p0_band_actual_zero']/d['p0_band_count']:.1f}%), 평균 예측 확률은 {100*d['p0_band_mean_p0']:.1f}%였다") if d['p0_band_count'] else '이 구간에 검증 표본이 없어 보정을 판단하지 않았다'
    interpretation = interpretation.replace('P(0) = 0.88–0.92인 검증 은하 177개 중 실제 0은 164개(92.7%), 평균 예측 확률은 약 90.0%였다',calibration)
    interpretation += '\n환경×성장 제거 대조는 이전 HGB에서 수행한 이력이다. 최종 LightGBM의 순열 중요도와 해당 HGB 대조를 같은 알고리즘의 효과 추정으로 합치지 않는다.\n'
    update('### 12.1 Scientific Interpretation',interpretation)
    limitations = old_md['## 13. Limitations & Conclusion — **REQUIRED**'].split('### 13.2 Conclusion')[0]
    old5 = next(line for line in limitations.splitlines() if line.startswith('5. **선택 과정'))
    limitations = limitations.replace(old5, '5. **선택 과정과 대조 조건.** 과거 HGB의 피처·문턱 실험과, 이후 7구성 5-fold 알고리즘 비교를 구분한다. 0.9 문턱은 HGB에서 고른 값을 LightGBM에 이전했으며 재튜닝하지 않았다. 시간·정확도 절충 기준은 결과 확인 후 정했다. 전체 데이터를 이미 이전 분석에서 관찰했으므로 새로운 시험 일반화의 증거는 아니다. 두 단계와 단일 회귀는 총 학습량이 다르고, HGB의 환경×성장 대조를 LightGBM에서 반복하지 않았다.')
    limitations += '\n7. **학습 시간의 적용 범위.** 같은 컴퓨터·CPU 6스레드에서 순차 실행한 모델 적합 시간이다. 배경 작업·실행 순서·예측 지연·피처 계산 비용을 완전히 통제한 속도 시험은 아니다. LightGBM의 관측 시간 이점은 HGB 400회와 비교한 값이며 기존 조기 종료 HGB와는 시간이 비슷했다. HGB 고정 400회는 조기 종료 모델보다 학습 행도 더 사용하므로 개선을 트리 수만의 효과로 분리할 수 없다.\n\n'
    limitations += f'''### 13.2 Conclusion

최종 모델로 **LightGBM 두 단계 모델**을 선택했다. 0 여부의 확률과 양수일 때의 조건부 로그 평균을 결합하며, 기본 13개·환경 17개·성장 및 비율 4개의 입력과 P(0)≥0.9의 0 출력 정책을 사용한다. 공통 검증셋의 D1=log10(SFR+1) RMSE는 **{r['validation_log_rmse']:.6f} dex**, 선형 RMSE는 **{r['validation_rmse']:.4f}**, MAE는 **{r['validation_mae']:.4f}**였다.

{choice} 이는 고정 설정의 관측 성능과 시간을 함께 고려한 선택이며, 두 알고리즘의 정확도 동등성이나 일반적인 속도 우위를 주장하지 않는다. 같은 5개 분할에서 세 알고리즘 모두 두 단계가 단일 회귀보다 낮은 오차를 보였다.

최종 LightGBM의 오차·순열 중요도는 새 예측으로 다시 계산했다. 과거 HGB의 물리적 피처 선택 근거를 보존하되 LightGBM에서 같은 제거 실험을 반복한 것으로 확대하지 않았다. 다음 검증 과제는 공식 평가 정의와 입력 조건 확인, 공간 의존 평가, 문턱·확률 보정과 물리적 해석의 보완이다.
'''
    update('## 13. Limitations & Conclusion',limitations)
    update('## 14. Final Model Card & Reproducibility', f'''## 14. Final Model Card & Reproducibility — **REQUIRED**

**모델:** `TwoStageLightGBMLogSFR`. LightGBM 분류기와 실제 SFR>0 은하의 로그 회귀기를 사용한다. 두 단계 모두 34개 입력, 학습률 0.08, 트리 400개, num_leaves=31, max_bin=255, seed=42, CPU 6스레드이며 조기 종료는 쓰지 않는다. eps=1과 P(0)≥0.9의 0 출력 문턱을 적용하고 선형 SFR을 반환한다. ID는 입력 충돌 확인에만 쓴다.

**선택 근거:** 동일한 학습 내부 5-fold의 정확도와 관측 학습 시간을 함께 고려했다. HGB 400회와 LightGBM의 평균 로그 RMSE는 {hgb.mean_rmse_log:.6f}/{lgb.mean_rmse_log:.6f}, 평균 적합 시간은 {hgb.mean_training_seconds:.2f}/{lgb.mean_training_seconds:.2f}초였다. 관측 정확도의 작은 차이를 수용하고 LightGBM을 채택했다. 시간 고려는 결과 확인 후 정한 절충이며 동등성·비열등성 검정은 하지 않았다.

**검증 설계:** 학습 80%·검증 20%, 0 여부 층화, 시드 42다. HGB의 기존 25-fold 선택 이력, HGB의 고정 구조 대조, 최종 알고리즘 선택에 사용한 7구성 5-fold 기록을 구분해 제공한다. LightGBM 문턱은 별도로 재튜닝하지 않았다.

**재현 환경:** 시작 코드에서 Python·LightGBM 등 패키지 버전과 CSV SHA256을 출력한다. 모델 설정과 실제 트리 수를 기록하고, 함께 제공한 수정본 requirements.txt에 환경을 고정했다. 최종 공통 학습 분할 전체의 적합 시간은 {r['training_seconds']:.3f}초로, 위 CV fold 평균 시간과 다른 측정이다.

**성능과 실행 확인:** 검증 D1 RMSE {r['validation_log_rmse']:.6f}, 같은 예측의 D2 민감도 값 {r['validation_log_rmse_eps0.01']:.6f}다. 전체 {execution['code_cells']}개 코드 셀을 깨끗한 커널에서 순서대로 실행했고, 검증 재현·행 순서·동일 합집합 배치·잘못된 ID 거부 검사를 통과했다. 그래프 출력은 {execution['png_outputs']}개다. 값과 정의는 15절 RESULTS에 저장했다.

**적용 범위:** 같은 상자의 참고 집합을 사용하는 예측을 대상으로 한다. 공식 로그 정의, 공간 의존, 물리 단위, 외부 일반화, 시간 측정과 FoF 구조 민감도에 관한 제한은 13.1절을 따른다.''')
    for c in nb.cells:
        if c.cell_type=='code' and c.source.startswith('# 11절 보조:'):
            c.source=c.source.replace('SFR = 0을 그대로 낸 은하 (P(0) ≥ {ZERO_THRESHOLD}):',
                                      '최종 SFR = 0 출력 (0 문턱과 역변환 절단 포함):')
            for output in c.outputs:
                if output.output_type=='stream':
                    output.text=output.text.replace(f'SFR = 0을 그대로 낸 은하 (P(0) ≥ {r["zero_threshold_P0"]}):',
                                                    '최종 SFR = 0 출력 (0 문턱과 역변환 절단 포함):')
        if c.get('id')=='lightgbm-comparison-evidence':
            c.metadata.setdefault('jupyter',{})['source_hidden']=True
            # The scientific report records the research decision, not the chat instruction.
            public=copy.deepcopy(r['algorithm_comparison'])
            public['decision'].pop('user_instruction',None)
            public['decision']['reason']='관측 오차 차이와 학습 시간을 절충해 LightGBM을 선택했다.'
            old_payload=json.dumps(r['algorithm_comparison'],ensure_ascii=False,allow_nan=False)
            new_payload=json.dumps(public,ensure_ascii=False,allow_nan=False)
            assert old_payload in c.source or new_payload in c.source
            c.source=c.source.replace(old_payload,new_payload)
            for output in c.outputs:
                if output.output_type=='stream' and '선택: ' in output.text:
                    lines=output.text.splitlines()
                    for i,line in enumerate(lines):
                        if line.startswith('선택: '):
                            old_decision=ast.literal_eval(line[len('선택: '):])
                            assert old_decision['selected_model']==public['decision']['selected_model']
                            lines[i]='선택: '+repr(public['decision'])
                    output.text='\n'.join(lines)+'\n'
            r['algorithm_comparison']=public
    result_cell=next(c for c in nb.cells if c.cell_type=='code' and c.source.startswith('# STUDENT TODO — update model_name'))
    for output in result_cell.outputs:
        if output.output_type=='stream' and output.text.lstrip().startswith('{'):
            printed=json.loads(output.text)
            assert printed['validation_log_rmse']==r['validation_log_rmse']
            printed['algorithm_comparison']=r['algorithm_comparison']
            output.text=json.dumps(printed,indent=2,ensure_ascii=False)+'\n'
    evidence['editorial_metadata_refresh']='실행 후 결정 설명과 출력 표제를 정리했다. 학습 설정·예측·지표는 바꾸지 않았다.'
    (HERE/'execution_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    nbformat.write(nb,ROOT/'DataHub_701104_final.ipynb')
    # The comparison report now records the user's final choice as a subsequent decision.
    comparative = (HERE/'reference/comparison_before_selection.md').read_text(encoding='utf-8')
    oldlead = comparative.split('\n\n')[1]
    newlead = (f"**최종 선택은 LightGBM 두 단계 모델입니다.** {choice} "
               f"기존 조기 종료 HGB의 시간은 {ref.mean_training_seconds:.2f}초였으므로, 시간 이점의 비교 대상은 HGB 400회 후보입니다. "
               f"최종 노트북의 모델 코드·검증 점수·오차 분석·모델 카드도 LightGBM으로 갱신했습니다.")
    comparative = comparative.replace(oldlead,newlead,1)
    comparative = comparative.replace('최종 교체 판단에는 동일 프로토콜의 추가 검증과 기존 과학적 해석의 재점검이 필요하다.',
                                      '비교 결과 확인 후 관측 정확도 차이와 학습 시간을 절충해 LightGBM을 선택했다. 정확도 동등성을 입증한 것은 아니다.')
    comparative = comparative.replace('기존 제출 HGB의 평균 RMSE 재현 차이는 0입니다.',
                                      '알고리즘 비교 당시 기존 제출 HGB의 평균 RMSE 재현 차이는 0이었습니다. 당시 원본 노트북은 lightgbm_report_revision_2026-11-12/reference에 보존했습니다.')
    comparative = comparative.replace('독립 재계산: .venv\\Scripts\\python.exe work/조장/model_comparison_2026-11-12/validate_results.py work/조장/model_comparison_2026-11-12/results',
                                      '최신 보고서 검증: .venv\\Scripts\\python.exe work/조장/lightgbm_report_revision_2026-11-12/validate_report.py')
    comparative += f'''\n## LightGBM 최종 반영 (사용자 선택 이후)\n\n최종 모델은 `TwoStageLightGBMLogSFR`이며, 재실행한 공통 검증셋의 로그 RMSE는 {r['validation_log_rmse']:.6f}입니다. 이 값은 위 내부 CV 평균과 구분합니다. 시간·정확도 선택은 비교 후 정한 정책이며, 수정 전 실험 자료와 해당 시점의 검토 기록은 역사 기록으로 보존했습니다. 최신 노트북과 진단 검증은 [수정 기록](../lightgbm_report_revision_2026-11-12/README.md)을 따릅니다.\n'''
    (BENCH/'report.md').write_text(comparative,encoding='utf-8')
    (HERE/'report.md').write_text(newlead+f"\n\n공통 검증셋 로그 RMSE: **{r['validation_log_rmse']:.6f}**.\n\n"+
                                  '새 예측에 맞춰 최종 노트북의 10–15절 수치와 해석을 갱신했습니다. 상세 표는 최종 노트북과 ../model_comparison_2026-11-12/report.md에 있습니다.\n',encoding='utf-8')
    print('Refreshed final notebook prose, model card, and comparison report from actual LightGBM evidence.')


if __name__=='__main__':
    main()

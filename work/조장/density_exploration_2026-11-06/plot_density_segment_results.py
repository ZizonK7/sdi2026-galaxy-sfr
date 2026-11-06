import os
from pathlib import Path
import sys
import json
OUT=Path(__file__).parent
if os.environ.get('SFR_ANALYSIS_DEPENDENCIES'):
    sys.path.insert(0, os.environ['SFR_ANALYSIS_DEPENDENCIES'])
os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False,
                     'font.size':11,'svg.fonttype':'none','pdf.fonttype':42})
r=json.loads((OUT/'density_segment_regression_results.json').read_text(encoding='utf-8'))
fig,(a,b)=plt.subplots(1,2,figsize=(13,6.6),facecolor='#FAFBFD',gridspec_kw={'width_ratios':[1.3,1]})
fig.subplots_adjust(left=.17,right=.97,top=.72,bottom=.26,wspace=.42)
fig.text(.065,.92,'밀도 세 구간으로 나누면 SFR 회귀가 좋아질까?',fontsize=22,weight='bold',color='#18273B')
fig.text(.065,.85,'Q–Q -2, +2에 대응하는 밀도 1.365, 2.409로 분할 · 동일한 13개 피처와 HGB 설정',fontsize=12,color='#526175')
names=['all','density_low','density_middle','density_high']
labels=['전체 검증 은하','밀도 ≤ 1.365','1.365 < 밀도 ≤ 2.409','밀도 > 2.409']
ci=r['validation_delta_RMSE_and_paired_bootstrap_95CI_vs_pooled']
for i,k in enumerate(names):
    point,lo,hi=ci[k]
    a.errorbar(point,i,xerr=[[point-lo],[hi-point]],fmt='o',ms=7,capsize=4,
               color='#286BB5' if i==0 else '#637F9E',elinewidth=2)
    a.text(hi+.2,i,f'{point:+.2f}',va='center',fontsize=10,color='#34465C')
a.set_yticks(range(4),labels)
a.set_ylim(3.65,-.7)
a.set_xlim(-4.4,8.2)
a.set_title('기존 검증 분할: 차이와 95% 구간',loc='left',fontsize=13,pad=19,weight='bold',color='#24415F')
a.set_xlabel('RMSE 차이 (세 구간 모델 - 단일 모델)',labelpad=13)
delta=[f['delta_RMSE'] for f in r['cv_folds']]
b.bar(np.arange(1,6),delta,width=.62,color=['#4480BE' if v<0 else '#CC784A' for v in delta],zorder=3)
for i,v in enumerate(delta,1):
    b.text(i,v+(.06 if v>=0 else -.06),f'{v:+.2f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=11)
b.set_ylim(-.65,1.5)
b.set_xticks(np.arange(1,6),[f'{i}번' for i in range(1,6)])
b.set_title('학습 내 5-fold CV: 분할마다 다른 결과',loc='left',fontsize=13,pad=19,weight='bold',color='#24415F')
b.set_xlabel('교차검증 분할',labelpad=13)
b.set_ylabel('RMSE 차이',labelpad=10)
for p in (a,b):
    p.set_facecolor('#FAFBFD')
    p.axvline(0,color='#526175',lw=1,zorder=0) if p is a else p.axhline(0,color='#526175',lw=1,zorder=0)
    p.grid(axis='x' if p is a else 'y',color='#DCE3ED',lw=.7,zorder=0)
    p.spines[['top','right','left']].set_visible(False)
    p.spines['bottom'].set_color('#CDD7E3')
    p.tick_params(length=0,pad=8,colors='#526175')
fig.text(.065,.14,'음수 = 구간별 모델이 개선됨   |   양수 = 구간별 모델이 악화됨',fontsize=11,color='#34465C')
fig.text(.065,.09,'기존 검증 RMSE 9.513 → 8.612  ·  5-fold 전체 예측 RMSE 9.057 → 9.081',fontsize=12,weight='bold',color='#24415F')
fig.text(.065,.04,'검증 구간은 동일 은하를 재표집한 쌍 부트스트랩 1,000회. CV에서는 각 학습 fold 안에서만 경계를 다시 계산했습니다.',fontsize=9,color='#768398')
for ext in ['png','svg','pdf']:
    fig.savefig(OUT/f'density_segment_regression_comparison.{ext}',dpi=180,facecolor=fig.get_facecolor())

print('Saved density segmentation comparison chart.')

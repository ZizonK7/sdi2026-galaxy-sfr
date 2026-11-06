import os,sys,json
from pathlib import Path
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
r=json.loads((OUT/'fixed_density_hurdle_results.json').read_text(encoding='utf-8'))
c=np.load(OUT/'fixed_density_gate_curves.npz')
fig,(a,b)=plt.subplots(1,2,figsize=(14,7.6),facecolor='#FAFBFD',gridspec_kw={'width_ratios':[1.2,1]})
fig.subplots_adjust(left=.075,right=.975,top=.74,bottom=.29,wspace=.27)
fig.text(.075,.93,'고정한 밀도 확률 × 양수 회귀: 실험 결과',fontsize=23,weight='bold',color='#18273B')
fig.text(.075,.865,'중간 밀도 구간 1.365 < d ≤ 2.409 · 학습 27,517개 / 검증 6,948개 · 양수 회귀기는 동일하게 사용',fontsize=12,color='#526175')
grid=c['density_grid'];a.plot(grid,c['gaussian_q'],color='#CC6439',lw=2.5,label='정규분포 기반 고정 확률',zorder=4)
dd=c['validation_middle_density'];zz=c['validation_middle_iszero'];edges=np.linspace(grid[0],grid[-1],19)
centres=[];proportions=[];lower=[];upper=[]
for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
    m=(dd>=lo)&((dd<hi) if i<17 else (dd<=hi))
    n=int(m.sum());k=int(zz[m].sum());p=k/n;z=1.96
    centre=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    centres.append((lo+hi)/2);proportions.append(p);lower.append(p-centre+half);upper.append(centre+half-p)
a.errorbar(centres,proportions,yerr=[lower,upper],fmt='o',ms=4,color='#286BB5',elinewidth=1,capsize=2,
           label='검증 데이터의 실제 0 비율 (95% 구간)',zorder=3)
a.set_xlim(grid[0]-.025,grid[-1]+.025);a.set_ylim(-.025,1.04)
a.set_title('밀도로 추정한 SFR=0 확률',loc='left',fontsize=14,weight='bold',pad=18,color='#24415F')
a.set_xlabel('주변 밀도  log10(1 + δ)',labelpad=12);a.set_ylabel('SFR=0 확률 / 비율',labelpad=12)
a.legend(frameon=False,fontsize=9,loc='lower right')
keys=['direct_segment_regression','fixed_gaussian_density_gate','learned_density_only_gate','learned_all_features_gate']
labels=['기존\n직접 회귀','밀도 확률 고정\n(정규 기반)','밀도만 학습\n(정규 가정 없음)','전체 피처로\n확률 학습']
vals=[next(row['RMSE'] for row in r['validation_metrics'] if row['population']=='middle' and row['method']==k) for k in keys]
b.bar(range(4),vals,color=['#99AABF','#CC784A','#D4A075','#4480BE'],width=.63,zorder=3)
for i,v in enumerate(vals): b.text(i,v+.32,f'{v:.3f}',ha='center',va='bottom',fontsize=11,color='#24415F',weight='bold')
b.set_ylim(0,25.8);b.set_xticks(range(4),labels,fontsize=9)
b.set_title('중간 구간 검증 RMSE (낮을수록 좋음)',loc='left',fontsize=14,weight='bold',pad=18,color='#24415F')
b.set_ylabel('SFR 원래 척도의 RMSE',labelpad=10)
for p in (a,b):
    p.set_facecolor('#FAFBFD');p.grid(axis='y',color='#DCE3ED',lw=.7,zorder=0)
    p.spines[['top','right','left']].set_visible(False);p.spines['bottom'].set_color('#CDD7E3')
    p.tick_params(length=0,pad=8,colors='#526175')
fig.text(.075,.17,'고정 확률 모델: 중간 구간의 0 은하 RMSE 2.256 → 1.579, 양수 은하 RMSE 28.260 → 30.438',fontsize=12,weight='bold',color='#24415F')
fig.text(.075,.12,'중간 구간 SFR > 100인 36개: 고정 모델의 평균 0 확률 33.3% / 전체 피처 모델 0.7%',fontsize=11,color='#34465C')
fig.text(.075,.075,'5-fold 전체 예측 RMSE: 기존 세 구간 9.081 / 밀도 확률 고정 9.245 / 전체 피처 확률 8.986',fontsize=11,color='#34465C')
fig.text(.075,.03,'모든 확률·분포·회귀는 학습 자료로만 적합. 낮은/높은 밀도 구간은 기존 회귀를 유지한 탐색 실험입니다.',fontsize=9,color='#768398')
for ext in ['png','svg','pdf']:
    fig.savefig(OUT/f'fixed_density_hurdle_comparison.{ext}',dpi=180,facecolor=fig.get_facecolor())
print('Saved fixed_density_hurdle_comparison.png/svg/pdf')

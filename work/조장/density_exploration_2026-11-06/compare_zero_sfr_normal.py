"""Compare training zero-SFR log-density values with a fitted normal reference."""
import csv
import json
import os
from pathlib import Path
import sys

OUT = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[3]
if os.environ.get('SFR_ANALYSIS_DEPENDENCIES'):
    sys.path.insert(0, os.environ['SFR_ANALYSIS_DEPENDENCIES'])
os.environ['MPLCONFIGDIR'] = str(OUT / '.mplconfig')
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

with (ROOT / 'Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv').open(newline='', encoding='utf-8-sig') as f:
    records = list(csv.DictReader(f))
density = np.array([float(r['log10_overdensity_plus_1']) for r in records])
sfr = np.array([float(r['sfr_linear']) for r in records])
tr, _ = train_test_split(np.arange(len(sfr)), test_size=.20, random_state=42,
                         stratify=(sfr > 0).astype(int))
z = density[tr][sfr[tr] == 0]
edges = np.arange(-.75, 2.75 + 1e-9, .25)
counts, _ = np.histogram(z, bins=edges)
with (OUT / 'zero_sfr_counts_by_density.csv').open(newline='', encoding='utf-8-sig') as f:
    saved = list(csv.DictReader(f))
assert len(tr) == 166749 and len(z) == 14438
assert counts.tolist() == [int(r['zero_sfr_galaxies']) for r in saved]
assert np.isfinite(z).all()
mu, sd = stats.norm.fit(z)
quantile_levels = [.001, .01, .05, .25, .5, .75, .95, .99, .999]
report = {
    'population': 'SFR=0 galaxies in the existing EDA training split',
    'variable': 'log10_overdensity_plus_1',
    'n': len(z), 'mean': float(mu), 'standard_deviation_mle_ddof0': float(sd),
    'skewness': float(stats.skew(z, bias=False)),
    'excess_kurtosis': float(stats.kurtosis(z, bias=False)),
    'median': float(np.median(z)), 'minimum': float(z.min()), 'maximum': float(z.max()),
    'quantile_levels': quantile_levels, 'observed_quantiles': np.quantile(z, quantile_levels).tolist(),
    'normal_quantiles': stats.norm.ppf(quantile_levels, loc=mu, scale=sd).tolist(),
    'observed_fraction_at_or_below_1_5': float(np.mean(z <= 1.5)),
    'normal_fraction_at_or_below_1_5': float(stats.norm.cdf(1.5, loc=mu, scale=sd)),
    'split_validation': 'All 14 histogram-bin counts match saved notebook EDA 1 exactly.'
}
(OUT / 'zero_sfr_normal_statistics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
with (OUT / 'zero_sfr_normal_bin_comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f)
    w.writerow(['left_exclusive', 'right_inclusive', 'observed_count', 'normal_expected_count'])
    expected = len(z) * np.diff(stats.norm.cdf(edges, loc=mu, scale=sd))
    for lo, hi, count, exp in zip(edges[:-1], edges[1:], counts, expected):
        w.writerow([lo, hi, count, exp])

font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                     'font.size': 11, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig, (ax, qq) = plt.subplots(1, 2, figsize=(14, 7.4), facecolor='#FAFBFD',
                             gridspec_kw={'width_ratios': [1.35, 1]})
fig.subplots_adjust(left=.07, right=.97, top=.73, bottom=.23, wspace=.24)
fig.text(.07, .92, 'SFR=0 은하의 주변 밀도, 정규분포와 비교', fontsize=23, weight='bold', color='#18273B')
fig.text(.07, .852, f'같은 학습 데이터 14,438개  |  평균 {mu:.3f} · 표준편차 {sd:.3f}인 정규분포를 겹쳤습니다.',
         fontsize=12, color='#526175')
ax.hist(z, bins=edges, density=True, color='#7EADE0', edgecolor='#FAFBFD', linewidth=1.4,
        label='실제 분포 (0.25 dex 구간)', zorder=2)
xs = np.linspace(-.75, 2.9, 900)
ax.plot(xs, stats.norm.pdf(xs, loc=mu, scale=sd), color='#CC6439', lw=2.6,
        label='같은 평균·표준편차의 정규분포', zorder=4)
ax.axvline(1.5, color='#637387', ls='--', lw=1.2, zorder=3, label='앞서 살펴본 기준 1.5')
ax.set_xlim(-.75, 2.85)
ax.set_ylim(bottom=0)
ax.set_xticks(np.arange(-.5, 2.76, .5))
ax.set_title('분포 모양 비교', loc='left', pad=15, fontsize=14, weight='bold', color='#24415F')
ax.set_xlabel('주변 밀도  log10(1 + δ)', labelpad=13)
ax.set_ylabel('확률밀도', labelpad=12)
ax.legend(frameon=False, fontsize=9, loc='upper left')

# All observations enter the Q-Q plot; the line is the fitted-normal reference.
normal_order_quantiles, ordered_values = stats.probplot(z, dist='norm', fit=False)
qq.scatter(normal_order_quantiles, ordered_values, s=7, alpha=.5, linewidths=0,
           color='#286BB5', rasterized=True, zorder=3)
line_x = np.array([min(normal_order_quantiles), max(normal_order_quantiles)])
qq.plot(line_x, mu + sd * line_x, color='#CC6439', lw=2, label='정규분포라면 따르는 직선')
qq.set_title('정규 Q–Q 그래프', loc='left', pad=15, fontsize=14, weight='bold', color='#24415F')
qq.set_xlabel('표준정규분포의 이론적 분위수', labelpad=13)
qq.set_ylabel('실제 주변 밀도의 분위수', labelpad=10)
qq.legend(frameon=False, fontsize=9, loc='upper left')
for panel in [ax, qq]:
    panel.set_facecolor('#FAFBFD')
    panel.grid(axis='y', color='#DCE3ED', lw=.8, zorder=0)
    panel.tick_params(length=0, pad=7, colors='#526175')
    panel.spines[['top', 'right', 'left']].set_visible(False)
    panel.spines['bottom'].set_color('#CDD7E3')
fig.text(.07, .105, f'왜도 {report["skewness"]:.2f} (정규분포: 0)   |   초과첨도 {report["excess_kurtosis"]:.2f} (정규분포: 0)',
         fontsize=11, color='#34465C')
fig.text(.07, .063, '왼쪽은 모양을 비교하기 위해 확률밀도로 표시했습니다. 오른쪽은 점들이 직선에 가까울수록 정규분포에 가깝습니다.',
         fontsize=10, color='#526175')
fig.text(.07, .025, '제공 CSV에서 학습 분할(seed=42, SFR 양수 여부로 층화)을 재현했으며, 기존 EDA의 14개 구간별 개수와 모두 일치합니다.',
         fontsize=9, color='#768398')
for ext in ['png', 'svg', 'pdf']:
    fig.savefig(OUT / f'zero_sfr_density_vs_normal.{ext}', dpi=180, facecolor=fig.get_facecolor())
print(json.dumps(report, ensure_ascii=True, indent=2))

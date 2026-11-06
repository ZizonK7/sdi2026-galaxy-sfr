"""Plot the saved EDA 1 training-set counts, without changing the notebook."""
import csv
import json
import os
from pathlib import Path
import re
import sys

if os.environ.get('SFR_ANALYSIS_DEPENDENCIES'):
    sys.path.insert(0, os.environ['SFR_ANALYSIS_DEPENDENCIES'])
os.environ['MPLCONFIGDIR'] = str(Path(__file__).parent / '.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator, StrMethodFormatter

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent
nb = json.loads((ROOT / 'DataHub_701104_mid.ipynb').read_text(encoding='utf-8'))
cell = next(c for c in nb['cells'] if 'edges = np.arange(-0.75' in ''.join(c['source']))
table = next(''.join(o['data']['text/plain']) for o in cell['outputs']
             if 'text/plain' in o.get('data', {}))
rows = []
for line in table.splitlines():
    m = re.match(r'^\(([-\d.]+), ([-\d.]+)\]\s+(\d+)\s+(\d+)\s+([\d.]+)', line)
    if m:
        lo, hi, total, zero, rate = m.groups()
        rows.append((float(lo), float(hi), int(total), int(zero), float(rate)))
assert len(rows) == 14
assert sum(r[2] for r in rows) == 166749
assert sum(r[3] for r in rows) == 14438
assert all(abs(hi - lo - .25) < 1e-9 for lo, hi, *_ in rows)
with (OUT / 'zero_sfr_counts_by_density.csv').open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['density_left_exclusive', 'density_right_inclusive', 'all_galaxies', 'zero_sfr_galaxies', 'zero_sfr_percent'])
    writer.writerows(rows)

font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                     'font.size': 12, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(13, 7.2), facecolor='#FAFBFD')
ax.set_facecolor('#FAFBFD')
fig.subplots_adjust(left=.085, right=.975, top=.77, bottom=.22)
fig.text(.085, .92, '주변 밀도에 따른 SFR=0 은하의 개수', fontsize=23, weight='bold', color='#18273B')
fig.text(.085, .856, '학습 데이터 166,749개 중 SFR=0 은하 14,438개  |  밀도 구간 폭 0.25 dex',
         fontsize=12, color='#526175')
centres = [(lo + hi) / 2 for lo, hi, *_ in rows]
counts = [r[3] for r in rows]
bars = ax.bar(centres, counts, width=.215,
              color=['#1D62B6' if c == max(counts) else '#7EADE0' for c in counts],
              edgecolor='none', zorder=3)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, count + 105, f'{count:,}',
            ha='center', va='bottom', fontsize=11, color='#24415F',
            weight='bold' if count == max(counts) else 'normal')
ax.set_xlim(-.75, 2.75)
ax.set_ylim(0, 6250)
edges = [-.75 + .25 * i for i in range(15)]
ax.set_xticks(edges, [f'{v:g}' for v in edges], fontsize=10)
ax.set_xlabel('주변 밀도  log10(1 + δ)', labelpad=16, color='#34465C')
ax.set_ylabel('SFR=0 은하 수 (개)', labelpad=15, color='#34465C')
ax.yaxis.set_major_locator(MultipleLocator(1000))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.grid(axis='y', color='#DCE3ED', linewidth=.8, zorder=0)
ax.tick_params(axis='both', length=0, pad=8, colors='#526175')
for side in ['top', 'right', 'left']:
    ax.spines[side].set_visible(False)
ax.spines['bottom'].set_color('#CDD7E3')
fig.text(.085, .072, '구간은 (왼쪽 경계, 오른쪽 경계]이며, 막대 위 숫자는 실제 개수입니다.',
         fontsize=10, color='#526175')
fig.text(.085, .035, '출처: DataHub_701104_mid.ipynb · EDA 1 저장 집계 (학습 데이터, seed=42)',
         fontsize=9, color='#768398')
for ext in ['png', 'svg', 'pdf']:
    fig.savefig(OUT / f'zero_sfr_counts_by_density.{ext}', dpi=180, facecolor=fig.get_facecolor())
print(json.dumps({'bins':len(rows), 'training_total':sum(r[2] for r in rows),
                  'zero_total':sum(counts), 'peak_bin':'(1.75, 2.00]', 'peak_count':max(counts)}))

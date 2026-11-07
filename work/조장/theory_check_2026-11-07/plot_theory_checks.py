"""2026-11-07 이론 검토 그림 두 장. 한글 글꼴이 없는 환경에서도 그려지도록 그림 안 글자는 영어로 쓴다.
theory_checks.py가 만든 theory_check_results.json을 읽고, sSFR·SFR 바닥 그림만 CSV에서 다시 계산한다."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, special
from sklearn.model_selection import train_test_split

OUT = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[3]
R = json.loads((OUT / 'theory_check_results.json').read_text(encoding='utf-8'))
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'            # 범주 슬롯 1~3 (검증된 기본 팔레트)
RAMP = ['#86b6ef', '#3987e5', '#1c5cab', '#0d366b']              # 순차 램프 250/400/550/700 (서열 4단계)
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold', 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.edgecolor': '#cdd7e3',
                     'svg.fonttype': 'none', 'figure.facecolor': SURF, 'axes.facecolor': SURF, 'legend.fontsize': 8.8})

def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color=GRID, lw=.8, zorder=0)
    ax.tick_params(length=0, pad=5)

edges = np.array(R['bin_edges']); mids = (edges[:-1] + edges[1:]) / 2; w = 0.25
A, B, D = R['A_all_galaxy_density'], R['B_quenched_fraction_models'], R['D_peng_separability']
all_obs = np.array([b['all_observed'] for b in A['bins']]); zero_obs = np.array([b['zero_observed'] for b in A['bins']])
n_all, n_zero = A['moments']['n'], R['train_zero']
xs = np.linspace(-0.9, 2.9, 700)

# ───────── 그림 1: 치우친 전체 분포 × 시그모이드 퀜칭 비율 = 가운데가 종 모양 ─────────
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
ax = axes[0]
ax.bar(mids, all_obs / (n_all * w), width=w - 0.02, color=BLUE, zorder=2, label='All training galaxies (n = 166,749)')
sn, nf = A['skewnormal_fit'], A['normal_fit']
ax.plot(xs, stats.skewnorm.pdf(xs, sn['shape_alpha'], sn['loc'], sn['scale']), color=ORANGE, lw=2, zorder=4,
        label=f"Skew-normal fit (shape = {sn['shape_alpha']:.1f})")
ax.plot(xs, stats.norm.pdf(xs, nf['mean'], nf['sd']), color=INK2, lw=1.5, ls='--', zorder=3, label='Normal fit (same mean and sd)')
ax.set_xlim(-0.85, 2.85); ax.set_ylim(bottom=0)
ax.set_title('A. Density of all galaxies is right-skewed, not normal', loc='left')
ax.set_xlabel('log10(1 + overdensity)'); ax.set_ylabel('Probability density'); ax.legend(frameon=False, loc='upper right')

ax = axes[1]
frac = zero_obs / all_obs; m = zero_obs > 0
ax.scatter(mids[m], frac[m], s=52, color=BLUE, zorder=4, edgecolor=SURF, linewidth=1.5, label='Observed, 0.25 dex bins (bins with 0 omitted)')
lg = B['models']['logistic']; fl = B['models']['logistic_floor']
ax.plot(xs, special.expit(lg['params'][0] + lg['params'][1] * xs), color=ORANGE, lw=2, zorder=3,
        label=f"Logistic in density, 2 params (half point {lg['d_at_half']:.2f})")
e0 = fl['floor_eps0']
ax.plot(xs, e0 + (1 - e0) * special.expit(fl['params'][0] + fl['params'][1] * xs), color=AQUA, lw=2, ls='--', zorder=3,
        label=f"Logistic + constant floor {e0 * 100:.2f}%, 3 params")
ax.set_yscale('log'); ax.set_ylim(4e-5, 1.6); ax.set_xlim(-0.85, 2.85)
ax.set_title('B. SFR = 0 fraction rises as a sigmoid of density', loc='left')
ax.set_xlabel('log10(1 + overdensity)'); ax.set_ylabel('Fraction with SFR = 0 (log scale)'); ax.legend(frameon=False, loc='upper left')

ax = axes[2]
ax.bar(mids, zero_obs, width=w - 0.02, color=BLUE, zorder=2, label='Observed SFR = 0 galaxies (n = 14,438)')
pred = np.array(lg['predicted_zero_by_bin'])
ax.plot(mids, pred, color=ORANGE, lw=2, marker='o', ms=6, markeredgecolor=SURF, markeredgewidth=1.2, zorder=4,
        label='All-galaxy histogram × logistic (no normal assumed)')
mu, sd = B['observed_zero_moments']['mean'], B['observed_zero_moments']['sd']
ax.plot(xs, n_zero * w * stats.norm.pdf(xs, mu, sd), color=INK2, lw=1.5, ls='--', zorder=3,
        label=f'Normal fit from 2026-11-06 (mean {mu:.3f}, sd {sd:.3f})')
ax.set_xlim(0.4, 2.85); ax.set_ylim(0, 7400)
ax.set_title('C. The product reproduces the bell shape in the middle', loc='left')
ax.set_xlabel('log10(1 + overdensity)'); ax.set_ylabel('Number of SFR = 0 galaxies per bin'); ax.legend(frameon=False, loc='upper left')
for a in axes: style(a)
fig.suptitle('Why the density of SFR = 0 galaxies looks normal: a skewed parent distribution × a sigmoid quenched fraction',
             x=0.012, ha='left', fontsize=13, fontweight='bold')
fig.text(0.012, 0.012, 'Training split (seed 42, stratified on SFR > 0): 166,749 galaxies, 14,438 with SFR = 0. Bins are 0.25 dex wide. '
         'Panel C: orange = sum over galaxies in each bin of the fitted logistic probability.', fontsize=8.6, color=INK2)
fig.tight_layout(rect=(0, 0.035, 1, 0.94))
for ext in ['png', 'svg']:
    fig.savefig(OUT / f'theory_fig1_product_explains_bump.{ext}', dpi=160, facecolor=SURF)
plt.close(fig)

# ───────── 그림 2: 질량의 역할과 0의 정체 ─────────
df = pd.read_csv(ROOT / 'Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv')
sfr_all = df['sfr_linear'].to_numpy()
tr, _ = train_test_split(np.arange(len(df)), test_size=.20, random_state=42, stratify=(sfr_all > 0).astype(int))
t = df.iloc[tr].reset_index(drop=True)
pos = t.loc[t['sfr_linear'] > 0, 'sfr_linear'].to_numpy()

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
ax = axes[0]
cells = pd.DataFrame(D['cells_logit'])
qe = D['mass_quartile_edges']
for i, name in enumerate(['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']):
    c = cells[(cells['mass'] == name) & (cells['n'] >= 100)].sort_values('bin_left')
    ax.plot(c['bin_left'] + 0.125, c['observed'], color=RAMP[i], lw=2, marker='o', ms=6, markeredgecolor=SURF,
            markeredgewidth=1.2, zorder=3, label=f'Stellar mass {name}: log M* {qe[i]:.1f}–{qe[i + 1]:.1f}')
ax.set_xlim(0.95, 2.75); ax.set_ylim(0, 1.03)
ax.set_title('A. At the same density, lower-mass galaxies are SFR = 0 more often', loc='left')
ax.set_xlabel('log10(1 + overdensity), bin centre'); ax.set_ylabel('Fraction with SFR = 0'); ax.legend(frameon=False, loc='lower right')

ax = axes[1]
E2 = R['E2_mass_quenching_as_low_ssfr']['by_log10_BH_mass']
xc = np.array([5.5, 6.5, 7.25, 7.75, 8.25, 8.75, 9.5])
med = np.array([r['median'] for r in E2]); q10 = np.array([r['q10'] for r in E2]); q90 = np.array([r['q90'] for r in E2])
ax.fill_between(xc, q10, q90, color=BLUE, alpha=0.16, lw=0, zorder=2, label='10th–90th percentile')
ax.plot(xc, med, color=BLUE, lw=2, marker='o', ms=6, markeredgecolor=SURF, markeredgewidth=1.2, zorder=4, label='Median log10 sSFR')
ax.axhline(-11, color=ORANGE, lw=1.5, ls='--', zorder=3, label='sSFR = 1e-11 /yr (usual "quenched" line)')
for x, r in zip(xc, E2):
    ax.annotate(f"{r['frac_below_m11'] * 100:.0f}% below", (x, r['q10']), textcoords='offset points', xytext=(0, -13), ha='center', fontsize=8.3, color=INK2)
ax.set_xlim(5.1, 9.9); ax.set_ylim(-12.4, -9.0)
ax.set_title('B. Big black holes lower sSFR, but rarely to exactly 0 (d ≤ 1.365)', loc='left')
ax.set_xlabel('log10 black-hole mass, bin centre (last bin 9–11)'); ax.set_ylabel('log10 sSFR [1/yr], galaxies with SFR > 0'); ax.legend(frameon=False, loc='lower left')

ax = axes[2]
bins = np.arange(-3.0, 3.61, 0.1)
ax.hist(np.log10(pos), bins=bins, color=BLUE, zorder=2, label=f'SFR > 0 (n = {len(pos):,})')
ax.bar(-3.35, n_zero, width=0.28, color=ORANGE, zorder=3, label=f'Exactly 0 (n = {n_zero:,})')
fl_sfr = R['F_sfr_near_zero_structure']['min_positive']
ax.axvline(np.log10(fl_sfr), color=INK2, ls=':', lw=1.2, zorder=3)
ax.text(np.log10(fl_sfr) + 0.06, n_zero * 0.62, f'smallest positive\nSFR = {fl_sfr:.4f}', fontsize=8.5, color=INK2)
ax.set_xticks([-3.35, -2, -1, 0, 1, 2, 3]); ax.set_xticklabels(['= 0', '-2', '-1', '0', '1', '2', '3'])
ax.set_xlim(-3.7, 3.6)
ax.set_title('C. SFR has a hard floor near 0.01 and a spike at exactly 0', loc='left')
ax.set_xlabel('log10 SFR [Msun/yr], 0.1 dex bins'); ax.set_ylabel('Number of galaxies'); ax.legend(frameon=False, loc='upper right')
for a in axes: style(a)
fig.suptitle('What the zeros are: environment quenching plus a discrete star-formation floor; mass quenching shows up as low sSFR, not as 0',
             x=0.012, ha='left', fontsize=13, fontweight='bold')
fig.text(0.012, 0.012, 'Same training split. A: only bins with at least 100 galaxies in that quartile. B: galaxies with log10(1 + overdensity) ≤ 1.365 and SFR > 0; '
         'labels give the share below sSFR = 1e-11 /yr. C: the gap between 0 and the smallest positive SFR is empty.', fontsize=8.6, color=INK2)
fig.tight_layout(rect=(0, 0.035, 1, 0.94))
for ext in ['png', 'svg']:
    fig.savefig(OUT / f'theory_fig2_mass_and_sfr_floor.{ext}', dpi=160, facecolor=SURF)
print('saved figures')

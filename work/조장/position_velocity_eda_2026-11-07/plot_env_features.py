"""위치·속도 환경 EDA 그림 3장. 그림 글자는 영어(한글 글꼴 없는 환경)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

OUT = Path(__file__).parent; ROOT = Path(__file__).resolve().parents[3]
R = json.loads((OUT / 'env_features_results.json').read_text(encoding='utf-8'))
BLUE, ORANGE, AQUA, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#c3c2b7'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
SEQ = LinearSegmentedColormap.from_list('seq', ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold', 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'axes.edgecolor': '#cdd7e3', 'svg.fonttype': 'none', 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'legend.fontsize': 8.8})
def style(ax, grid='y'):
    ax.spines[['top', 'right']].set_visible(False); ax.grid(axis=grid, color=GRID, lw=.8, zorder=0); ax.tick_params(length=0, pad=5)

BOX = 200.0
df = pd.read_csv(ROOT / 'Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv')
n = len(df); pos = df[['pos_x', 'pos_y', 'pos_z']].to_numpy() % BOX
sfr = df['sfr_linear'].to_numpy(); zero = sfr == 0; mh = df['log10_halo_mass'].to_numpy()
tr, _ = train_test_split(np.arange(n), test_size=.2, random_state=42, stratify=(sfr > 0).astype(int))
is_tr = np.zeros(n, bool); is_tr[tr] = True
tree = cKDTree(pos, boxsize=BOX)
link = 0.2 * BOX / n ** (1 / 3)
pairs = tree.query_pairs(link, output_type='ndarray')
_, lab = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)), directed=False)
rich = np.bincount(lab)[lab]
km = KMeans(n_clusters=13, n_init=3, random_state=0).fit(pos).labels_
r200 = lambda logm: (3 * 10 ** logm / (4 * np.pi * 200 * 2.775e11)) ** (1 / 3)

# ───────── 그림 1: 상자 단면 ─────────
imax = int(np.argmax(mh)); z0 = pos[imax, 2]; half = 5.0
dz = (pos[:, 2] - z0 + BOX / 2) % BOX - BOX / 2
slab = np.abs(dz) < half
big = (mh >= 14.0) & slab
fig, axes = plt.subplots(1, 3, figsize=(17, 6.0))
ax = axes[0]
ax.scatter(pos[slab, 0], pos[slab, 1], s=1.2, color=GRAY, linewidths=0, zorder=2, label=f'All galaxies in slab (n = {slab.sum():,})')
m = slab & is_tr & zero
ax.scatter(pos[m, 0], pos[m, 1], s=3, color=ORANGE, linewidths=0, zorder=3, label=f'SFR = 0, training set (n = {m.sum():,})')
for i in np.where(big)[0]:
    ax.add_patch(plt.Circle((pos[i, 0], pos[i, 1]), 2 * r200(mh[i]), fill=False, color=BLUE, lw=1.2, zorder=4))
ax.scatter([], [], s=60, facecolor='none', edgecolor=BLUE, label=f'Halos ≥ 10^14 h⁻¹ M☉, circle = 2 R200 (n = {big.sum()})')
ax.set_title(f'A. A {2*half:.0f} h⁻¹ Mpc slab: SFR = 0 galaxies sit in the nodes of the web', loc='left')
ax = axes[1]
cols = plt.cm.tab20(np.linspace(0, 1, 13))
ax.scatter(pos[slab, 0], pos[slab, 1], s=1.5, c=cols[km[slab]], linewidths=0, zorder=2)
ax.set_title('B. K-means (k = 13) on positions: geometric blobs, not structures', loc='left')
ax = axes[2]
ax.scatter(pos[slab, 0], pos[slab, 1], s=1.2, color=GRAY, linewidths=0, zorder=2, label='Isolated or small groups (N < 31)')
m2 = slab & (rich >= 31)
ax.scatter(pos[m2, 0], pos[m2, 1], s=3, color=BLUE, linewidths=0, zorder=3, label=f'Members of FoF groups with N ≥ 31 (n = {m2.sum():,})')
ax.set_title('C. Friends-of-friends groups (b = 0.2): compact, density-defined', loc='left')
for a in axes:
    a.set_xlim(0, BOX); a.set_ylim(0, BOX); a.set_aspect('equal'); a.set_xlabel('x [h⁻¹ Mpc]'); a.set_ylabel('y [h⁻¹ Mpc]'); style(a, grid='both')
    if a.get_legend_handles_labels()[0]: a.legend(frameon=True, facecolor=SURF, edgecolor=GRID, loc='upper left', markerscale=3)
fig.suptitle('Positions carry environment only after interpretation: web nodes, groups and cluster-centric distance', x=0.012, ha='left', fontsize=13, fontweight='bold')
fig.text(0.012, 0.012, f'Slab centred on the most massive halo (log M = {mh[imax]:.2f}), z within ±{half:.0f} h⁻¹ Mpc. Periodic box of 200 h⁻¹ Mpc. '
         f'K-means zero-fraction range across its 13 clusters: {R["kmeans"]["13"]["zero_fraction_min"]:.3f}–{R["kmeans"]["13"]["zero_fraction_max"]:.3f}; FoF richness bins: 0.004–0.924.', fontsize=8.6, color=INK2)
fig.tight_layout(rect=(0, 0.06, 1, 0.94)); [fig.savefig(OUT / f'env_fig1_box_slab.{e}', dpi=150, facecolor=SURF) for e in ['png', 'svg']]; plt.close(fig)

# ───────── 그림 2: 은하단 중심 거리·위상공간·FoF ─────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax = axes[0, 0]
for thr, col, lab_ in [('14.0', BLUE, 'nearest halo ≥ 10^14'), ('13.5', ORANGE, 'nearest halo ≥ 10^13.5')]:
    rows = R['zero_fraction_vs_r_over_R200'][thr]; x = np.arange(len(rows)); y = [r['fraction'] for r in rows]
    ax.plot(x, y, color=col, lw=2, marker='o', ms=6, markeredgecolor=SURF, markeredgewidth=1.2, zorder=3, label=lab_)
    ax.fill_between(x, [r['ci_lo'] for r in rows], [r['ci_hi'] for r in rows], color=col, alpha=.15, lw=0, zorder=2)
ax.set_xticks(x, [r['bin'] for r in rows], rotation=35, ha='right'); ax.set_ylim(0, 1.03)
ax.set_title('A. SFR = 0 fraction vs cluster-centric distance r / R200', loc='left'); ax.set_xlabel('r / R200 of the nearest massive halo'); ax.set_ylabel('Fraction with SFR = 0 (train)')
ax.legend(frameon=False, loc='upper right')
ax = axes[0, 1]
P = R['phase_space_zero_fraction']; N_ = np.array(P['n']); K_ = np.array(P['zero']); F_ = np.where(N_ > 0, K_ / np.maximum(N_, 1), np.nan)
im = ax.imshow(F_, origin='lower', cmap=SEQ, vmin=0, vmax=1, extent=[0, 3, 0, 3], aspect='auto', zorder=2)
for i in range(6):
    for j in range(6):
        if N_[i, j] > 0: ax.text(j * .5 + .25, i * .5 + .25, f'{F_[i, j]:.2f}\nn={N_[i, j]}', ha='center', va='center', fontsize=7.2, color='white' if F_[i, j] > .55 else INK)
cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.02); cb.set_label('Fraction with SFR = 0'); cb.outline.set_visible(False)
ax.set_title('B. Phase space around halos ≥ 10^13.5: distance and velocity both matter', loc='left'); ax.set_xlabel('r / R200'); ax.set_ylabel('|Δv| / V200 (relative to the halo)')
ax = axes[1, 0]
for key, col, lab_ in [('zero_fraction_vs_richness_central', BLUE, 'Central (most massive in its FoF group)'), ('zero_fraction_vs_richness_satellite', ORANGE, 'Satellite')]:
    rows = R['fof_b0.2'][key]; x = np.arange(len(rows)); y = [r['fraction'] if r['fraction'] is not None else np.nan for r in rows]
    ax.plot(x, y, color=col, lw=2, marker='o', ms=6, markeredgecolor=SURF, markeredgewidth=1.2, zorder=3, label=lab_)
ax.set_xticks(x, [r['bin'] for r in rows]); ax.set_ylim(0, 1.03)
ax.set_title('C. FoF group richness: satellites quench, centrals never reach SFR = 0', loc='left'); ax.set_xlabel('Number of galaxies in the FoF group (b = 0.2)'); ax.set_ylabel('Fraction with SFR = 0 (train)')
ax.legend(frameon=False, loc='upper left')
ax = axes[1, 1]
rows = R['fof_b0.2']['zero_fraction_vs_group_max_logm_satellites']; x = np.arange(len(rows))
ax.bar(x, [r['fraction'] for r in rows], width=.7, color=BLUE, zorder=2)
for xi, r in zip(x, rows): ax.text(xi, r['fraction'] + .02, f"n={r['n']:,}", ha='center', fontsize=8, color=INK2)
ax.set_xticks(x, [r['bin'] for r in rows]); ax.set_ylim(0, 1.08)
ax.set_title('D. Satellites: SFR = 0 fraction vs host mass (largest halo in the group)', loc='left'); ax.set_xlabel('log10 host halo mass [h⁻¹ M☉]'); ax.set_ylabel('Fraction with SFR = 0 (train satellites)')
for a in axes.flat: style(a)
fig.suptitle('The textbook environment variables, measured from pos_* and vel_*: cluster-centric radius, phase space, central/satellite, host mass', x=0.012, ha='left', fontsize=12.5, fontweight='bold')
fig.text(0.012, 0.012, 'Training split only for targets; positions and velocities of all 208,437 galaxies used for the geometry. R200 and V200 from the halo mass in the data. Shaded bands in A are 95% Wilson intervals.', fontsize=8.6, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 0.95)); [fig.savefig(OUT / f'env_fig2_cluster_centric.{e}', dpi=150, facecolor=SURF) for e in ['png', 'svg']]; plt.close(fig)

# ───────── 그림 3: 제공 밀도의 척도, 질량별 중심/위성, 신호 확인 ─────────
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
ax = axes[0]
S = R['provided_density_vs_candidates_spearman']; names = ['log_rho_5nn', 'log_rho_10nn', 'log1p_N_r1', 'log1p_N_r2', 'log1p_N_r5']
labels = ['5th-NN density', '10th-NN density', 'count r < 1', 'count r < 2', 'count r < 5']
ax.bar(np.arange(5), [S[k] for k in names], width=.7, color=BLUE, zorder=2)
for i, k in enumerate(names): ax.text(i, S[k] + .01, f'{S[k]:.3f}', ha='center', fontsize=8.5, color=INK2)
ax.set_xticks(np.arange(5), labels, rotation=20, ha='right'); ax.set_ylim(0.7, 1.0)
ax.set_title('A. The provided overdensity behaves like a count within ≈ 2 h⁻¹ Mpc', loc='left'); ax.set_ylabel('Spearman correlation with log10(1 + overdensity)')
ax = axes[1]
C = R['fof_b0.2']['zero_fraction_central_vs_satellite_by_stellar_mass']; keys = list(C.keys()); x = np.arange(len(keys))
ax.plot(x, [C[k]['satellite']['fraction'] for k in keys], color=ORANGE, lw=2, marker='o', ms=6, markeredgecolor=SURF, markeredgewidth=1.2, zorder=3, label='Satellites')
ax.plot(x, [C[k]['central']['fraction'] for k in keys], color=BLUE, lw=2, marker='o', ms=6, markeredgecolor=SURF, markeredgewidth=1.2, zorder=3, label='Centrals')
ax.set_xticks(x, keys); ax.set_ylim(0, .5)
ax.set_title('B. At every stellar mass, only satellites reach SFR = 0', loc='left'); ax.set_xlabel('log10 stellar mass bin'); ax.set_ylabel('Fraction with SFR = 0 (train)'); ax.legend(frameon=False, loc='upper left')
ax = axes[2]
Z = R['signal_check_zero_classifier']; Rg = R['signal_check_positive_regressor']
vals = [('Middle-segment\nzero accuracy', Z['middle_segment_base13']['accuracy'], Z['middle_segment_base13_plus_env']['accuracy']),
        ('Middle-segment\nzero AUC', Z['middle_segment_base13']['AUC'], Z['middle_segment_base13_plus_env']['AUC']),
        ('Positive-SFR RMSE\n(log1p scale)', Rg['base13']['rmse_log1p_scale']['mean'], Rg['base13_plus_env']['rmse_log1p_scale']['mean'])]
x = np.arange(3); w = .36
ax.bar(x - w / 2, [v[1] for v in vals], w, color=GRAY, zorder=2, label='13 original features')
ax.bar(x + w / 2, [v[2] for v in vals], w, color=BLUE, zorder=2, label='+ 17 environment features')
for xi, v in zip(x, vals):
    ax.text(xi - w / 2, v[1] + .01, f'{v[1]:.3f}', ha='center', fontsize=8.3, color=INK2); ax.text(xi + w / 2, v[2] + .01, f'{v[2]:.3f}', ha='center', fontsize=8.3, color=INK2)
ax.set_xticks(x, [v[0] for v in vals]); ax.set_ylim(0, 1.12)
ax.set_title('C. Preliminary 5-fold check: classification gains, regression does not', loc='left'); ax.legend(frameon=False, loc='upper right')
for a in axes: style(a)
fig.suptitle('What the new environment features add', x=0.012, ha='left', fontsize=13, fontweight='bold')
fig.text(0.012, 0.012, 'A: all 208,437 galaxies. B: training split, FoF b = 0.2. C: HistGradientBoosting, 5-fold CV inside the training split; raw-scale RMSE for positives is 9.03 vs 9.09 (unchanged within noise).', fontsize=8.6, color=INK2)
fig.tight_layout(rect=(0, 0.035, 1, 0.94)); [fig.savefig(OUT / f'env_fig3_scale_and_signal.{e}', dpi=160, facecolor=SURF) for e in ['png', 'svg']]
print('figures saved')

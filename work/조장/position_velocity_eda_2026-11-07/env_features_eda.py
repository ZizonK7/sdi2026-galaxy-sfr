"""2026-11-07 위치·속도에서 환경 피처를 만드는 EDA.
천문학의 표준 환경 정의(이웃 밀도, 은하단 중심 거리, 위상공간 속도, FoF 그룹과 중심/위성)를
제공 데이터의 pos_*, vel_*에 적용하고, SFR=0 비율과의 관계를 학습 분할에서 본다.
피처는 208,437개 전체의 위치·속도로 계산하되, 타깃은 학습 분할에서만 본다. K-means와 FoF를 비교한다.
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

t0 = time.time()
OUT = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[3]
BOX = 200.0                                        # h^-1 cMpc (Kuwayama 외 2026의 L200N1024)
G = 4.30091e-9                                     # Mpc km^2 s^-2 Msun^-1
RHO_CRIT = 2.775e11                                # h^2 Msun Mpc^-3
df = pd.read_csv(ROOT / 'Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv')
n = len(df)
pos = df[['pos_x', 'pos_y', 'pos_z']].to_numpy() % BOX
vel = df[['vel_x', 'vel_y', 'vel_z']].to_numpy()
sfr = df['sfr_linear'].to_numpy(); zero = sfr == 0
mh = df['log10_halo_mass'].to_numpy(); ms = df['log10_stellar_mass'].to_numpy()
dens = df['log10_overdensity_plus_1'].to_numpy()
tr, va = train_test_split(np.arange(n), test_size=.20, random_state=42, stratify=(sfr > 0).astype(int))
is_tr = np.zeros(n, bool); is_tr[tr] = True
L, U = 1.3647506592477927, 2.408565397956945
R = {'n_all': int(n), 'n_train': int(len(tr)), 'box': BOX}

def wilson(k, nn, z=1.96):
    k, nn = np.asarray(k, float), np.asarray(nn, float)
    p = np.where(nn > 0, k / np.maximum(nn, 1), np.nan)
    c = (p + z * z / (2 * nn)) / (1 + z * z / nn); h = z * np.sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn)) / (1 + z * z / nn)
    return c - h, c + h

def zero_frac_table(x, edges, mask=None, labels=None):
    m = is_tr if mask is None else (is_tr & mask)
    b = np.digitize(x, edges) - 1
    rows = []
    for i in range(len(edges) - 1):
        sel = m & (b == i)
        k, nn = int(zero[sel].sum()), int(sel.sum())
        lo, hi = wilson(k, nn) if nn else (np.nan, np.nan)
        rows.append({'bin': labels[i] if labels else f'[{edges[i]:g}, {edges[i+1]:g})', 'n': nn, 'zero': k,
                     'fraction': k / nn if nn else None, 'ci_lo': float(lo), 'ci_hi': float(hi)})
    return rows

# ───────────── 1. 이웃 밀도 (k번째 이웃 거리, 고정 반경 개수) ─────────────
tree = cKDTree(pos, boxsize=BOX)
dk, _ = tree.query(pos, k=11)
d5, d10 = dk[:, 5], dk[:, 10]
log_rho5 = np.log10(5 / (4 / 3 * np.pi * d5 ** 3)); log_rho10 = np.log10(10 / (4 / 3 * np.pi * d10 ** 3))
N1 = tree.query_ball_point(pos, 1.0, return_length=True) - 1
N2 = tree.query_ball_point(pos, 2.0, return_length=True) - 1
N5 = tree.query_ball_point(pos, 5.0, return_length=True) - 1
print(f'[1] 이웃 밀도 계산 {time.time()-t0:.0f}s  | 평균 이웃 수 r<1: {N1.mean():.2f}, r<2: {N2.mean():.2f}, r<5: {N5.mean():.2f}')
cands = {'log_rho_5nn': log_rho5, 'log_rho_10nn': log_rho10, 'log1p_N_r1': np.log1p(N1), 'log1p_N_r2': np.log1p(N2), 'log1p_N_r5': np.log1p(N5)}
R['provided_density_vs_candidates_spearman'] = {k: float(stats.spearmanr(dens, v).statistic) for k, v in cands.items()}
R['provided_density_vs_candidates_spearman_train_zero_AUC'] = {k: float(roc_auc_score(zero[tr], v[tr])) for k, v in cands.items()}
R['provided_density_zero_AUC_train'] = float(roc_auc_score(zero[tr], dens[tr]))

# ───────────── 2. 가장 가까운 거대 헤일로: 중심 거리 r/R200, 상대 속도 |Δv|/V200 ─────────────
def r200(logm):                                     # h^-1 Mpc, M in h^-1 Msun
    return (3 * 10 ** logm / (4 * np.pi * 200 * RHO_CRIT)) ** (1 / 3)
def v200(logm):                                     # km/s (h 상쇄)
    return np.sqrt(G * 10 ** logm / r200(logm))
host = {}
for thr in [13.5, 14.0]:
    sel = np.where(mh >= thr)[0]
    th = cKDTree(pos[sel], boxsize=BOX)
    dist, j = th.query(pos, k=1)
    hj = sel[j]
    dv = vel - vel[hj]; dv_abs = np.sqrt((dv ** 2).sum(1))
    rn = dist / r200(mh[hj]); vn = dv_abs / v200(mh[hj])
    host[thr] = {'n_hosts': int(len(sel)), 'dist': dist, 'host_logm': mh[hj], 'r_norm': rn, 'v_norm': vn, 'dv': dv_abs}
    print(f'[2] 헤일로 ≥ 10^{thr}: {len(sel)}개, R200 범위 {r200(mh[sel]).min():.2f}~{r200(mh[sel]).max():.2f} h^-1 Mpc, V200 범위 {v200(mh[sel]).min():.0f}~{v200(mh[sel]).max():.0f} km/s')
R['massive_halos'] = {str(k): {'n_hosts': v['n_hosts'], 'R200_max': float(r200(mh[mh >= k]).max()), 'V200_max': float(v200(mh[mh >= k]).max())} for k, v in host.items()}
r_edges = [0, 0.5, 1, 1.5, 2, 3, 5, 8, 12, 20, 1e9]
r_labels = ['0–0.5', '0.5–1', '1–1.5', '1.5–2', '2–3', '3–5', '5–8', '8–12', '12–20', '>20']
R['zero_fraction_vs_r_over_R200'] = {}
for thr in [13.5, 14.0]:
    h = host[thr]
    R['zero_fraction_vs_r_over_R200'][str(thr)] = zero_frac_table(h['r_norm'], r_edges, labels=r_labels)
    # 같은 r/R200에서 속도가 가르는가: r<2 안에서 v_norm 상·하위 반
    near = h['r_norm'] < 2
    R['zero_fraction_vs_r_over_R200'][f'{thr}_within2_by_vnorm'] = zero_frac_table(h['v_norm'], [0, 0.5, 1, 1.5, 2, 3, 1e9], mask=near, labels=['0–0.5', '0.5–1', '1–1.5', '1.5–2', '2–3', '>3'])
# 위상공간 격자 (호스트 ≥ 13.5, r/R200 < 3)
h = host[13.5]; near = is_tr & (h['r_norm'] < 3)
re_, ve_ = np.linspace(0, 3, 7), np.linspace(0, 3, 7)
ri, vi = np.clip(np.digitize(h['r_norm'], re_) - 1, 0, 5), np.clip(np.digitize(h['v_norm'], ve_) - 1, 0, 5)
grid_n = np.zeros((6, 6), int); grid_k = np.zeros((6, 6), int)
np.add.at(grid_n, (vi[near], ri[near]), 1); np.add.at(grid_k, (vi[near], ri[near]), zero[near].astype(int))
R['phase_space_zero_fraction'] = {'r_edges': re_.tolist(), 'v_edges': ve_.tolist(), 'n': grid_n.tolist(), 'zero': grid_k.tolist(),
                                  'note': 'rows = |Δv|/V200 bins, cols = r/R200 bins; hosts log M >= 13.5; train only',
                                  'median_v_norm_within_1R200': float(np.median(h['v_norm'][is_tr & (h['r_norm'] < 1)]))}
print(f'[2] r/R200<1 위성의 |Δv|/V200 중앙값 {R["phase_space_zero_fraction"]["median_v_norm_within_1R200"]:.2f} (단위가 km/s면 1 근처여야 함)')

# ───────────── 3. FoF 그룹 (Davis 외 1985: 연결 길이 = 0.2 × 평균 간격) ─────────────
mean_sep = BOX / n ** (1 / 3)
fof = {}
for b in [0.2, 0.3]:
    link = b * mean_sep
    pairs = tree.query_pairs(link, output_type='ndarray')
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    ncomp, lab = connected_components(g, directed=False)
    rich = np.bincount(lab)[lab]
    gmax = pd.Series(mh).groupby(lab).transform('max').to_numpy()
    central = (mh >= gmax - 1e-9)
    vmean = pd.DataFrame(vel).groupby(lab).transform('mean').to_numpy()
    dv_grp = np.sqrt(((vel - vmean) ** 2).sum(1)); dv_grp[rich == 1] = np.nan
    fof[b] = {'link': link, 'label': lab, 'richness': rich, 'central': central, 'dv_group': dv_grp, 'group_max_logm': gmax}
    print(f'[3] FoF b={b}: 연결 길이 {link:.3f} h^-1 Mpc, 그룹 {ncomp}개, 고립(N=1) {np.mean(rich==1):.1%}, N≥10 그룹 소속 {np.mean(rich>=10):.1%}, 위성 {np.mean(~central):.1%}')
F = fof[0.2]
rich_edges = [1, 2, 3, 6, 11, 31, 101, 1e9]; rich_labels = ['1', '2', '3–5', '6–10', '11–30', '31–100', '>100']
R['fof_b0.2'] = {'link_length': F['link'], 'mean_separation': mean_sep,
                 'zero_fraction_vs_richness_all': zero_frac_table(F['richness'], rich_edges, labels=rich_labels),
                 'zero_fraction_vs_richness_central': zero_frac_table(F['richness'], rich_edges, mask=F['central'], labels=rich_labels),
                 'zero_fraction_vs_richness_satellite': zero_frac_table(F['richness'], rich_edges, mask=~F['central'], labels=rich_labels),
                 'zero_fraction_central_vs_satellite_by_stellar_mass': {}}
for lo, hi in [(9.3, 9.5), (9.5, 9.8), (9.8, 10.3), (10.3, 11), (11, 13)]:
    mm = (ms > lo) & (ms <= hi)
    R['fof_b0.2']['zero_fraction_central_vs_satellite_by_stellar_mass'][f'{lo}–{hi}'] = {
        'central': zero_frac_table(np.zeros(n), [-1, 1], mask=mm & F['central'], labels=['all'])[0],
        'satellite': zero_frac_table(np.zeros(n), [-1, 1], mask=mm & ~F['central'], labels=['all'])[0]}
# 어제의 '낮은 밀도 0 은하 435개'는 무엇인가
lowz = is_tr & zero & (dens <= L)
R['low_density_zeros_435'] = {'n': int(lowz.sum()), 'satellite_fraction_fof': float((~F['central'])[lowz].mean()),
                              'satellite_fraction_all_low_density_train': float((~F['central'])[is_tr & (dens <= L)].mean()),
                              'median_dist_to_host14': float(np.median(host[14.0]['dist'][lowz])), 'median_dist_to_host14_low_density_positive': float(np.median(host[14.0]['dist'][is_tr & ~zero & (dens <= L)])),
                              'median_richness': float(np.median(F['richness'][lowz]))}
# FoF 그룹 최대 질량(호스트 질량 대리)별 0 비율
R['fof_b0.2']['zero_fraction_vs_group_max_logm_satellites'] = zero_frac_table(F['group_max_logm'], [8, 11, 12, 13, 13.5, 14, 16], mask=~F['central'], labels=['<11', '11–12', '12–13', '13–13.5', '13.5–14', '>14'])

# ───────────── 4. K-means와의 비교 ─────────────
R['kmeans'] = {}
for k in [13, 100]:
    km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(pos)
    lab = km.labels_
    sizes = np.bincount(lab); zf = np.array([zero[is_tr & (lab == i)].mean() for i in range(k)])
    R['kmeans'][str(k)] = {'cluster_size_min': int(sizes.min()), 'cluster_size_max': int(sizes.max()),
                           'zero_fraction_min': float(zf.min()), 'zero_fraction_max': float(zf.max()), 'zero_fraction_sd': float(zf.std()),
                           'zero_fractions': zf.tolist(), 'zero_AUC_train_using_cluster_mean': float(roc_auc_score(zero[tr], zf[lab[tr]]))}
    print(f'[4] K-means k={k}: 크기 {sizes.min()}~{sizes.max()}, 0 비율 범위 {zf.min():.3f}~{zf.max():.3f}')
R['fof_richness_zero_AUC_train'] = float(roc_auc_score(zero[tr], np.log(F['richness'][tr])))
R['fof_group_max_logm_zero_AUC_train'] = float(roc_auc_score(zero[tr], F['group_max_logm'][tr]))

# ───────────── 5. 새 피처가 신호를 더하는가 (예비, 학습 분할 안 5-fold) ─────────────
base_cols = [c for c in df.columns if c not in ['subhalo_id', 'sfr_linear']]
new = pd.DataFrame({'log_rho_5nn': log_rho5, 'log_rho_10nn': log_rho10, 'log1p_N_r1': np.log1p(N1), 'log1p_N_r2': np.log1p(N2), 'log1p_N_r5': np.log1p(N5),
                    'dist_host13.5': host[13.5]['dist'], 'r_over_R200_13.5': host[13.5]['r_norm'], 'v_over_V200_13.5': host[13.5]['v_norm'], 'host_logm_13.5': host[13.5]['host_logm'],
                    'dist_host14': host[14.0]['dist'], 'r_over_R200_14': host[14.0]['r_norm'], 'v_over_V200_14': host[14.0]['v_norm'],
                    'fof_log_richness': np.log(F['richness']), 'fof_is_central': F['central'].astype(float), 'fof_dv_group': F['dv_group'], 'fof_group_max_logm': F['group_max_logm'],
                    'halo_minus_stellar': mh - ms})
Xb = df[base_cols].to_numpy(); Xn = np.column_stack([Xb, new.to_numpy()])
R['new_feature_names'] = list(new.columns)
mid_tr = tr[(dens[tr] > L) & (dens[tr] <= U)]
def cv_clf(X, idx):
    aucs, accs = [], []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(idx, zero[idx]):
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0).fit(X[idx[a]], zero[idx[a]])
        p = m.predict_proba(X[idx[b]])[:, 1]; aucs.append(roc_auc_score(zero[idx[b]], p)); accs.append(np.mean((p >= .5) == zero[idx[b]]))
    return {'AUC': float(np.mean(aucs)), 'accuracy': float(np.mean(accs)), 'AUC_folds': [float(x) for x in aucs]}
R['signal_check_zero_classifier'] = {'middle_segment_base13': cv_clf(Xb, mid_tr), 'middle_segment_base13_plus_env': cv_clf(Xn, mid_tr),
                                     'all_train_base13': cv_clf(Xb, tr), 'all_train_base13_plus_env': cv_clf(Xn, tr)}
print(f'[5] 분류 (중간 구간 {len(mid_tr)}개) AUC base {R["signal_check_zero_classifier"]["middle_segment_base13"]["AUC"]:.4f} → +env {R["signal_check_zero_classifier"]["middle_segment_base13_plus_env"]["AUC"]:.4f}; '
      f'정확도 {R["signal_check_zero_classifier"]["middle_segment_base13"]["accuracy"]:.4f} → {R["signal_check_zero_classifier"]["middle_segment_base13_plus_env"]["accuracy"]:.4f}  ({time.time()-t0:.0f}s)')
pos_tr = tr[sfr[tr] > 0]
def cv_reg(X, idx):
    out = {'rmse_raw': [], 'rmse_log1p_scale': []}
    for a, b in KFold(5, shuffle=True, random_state=0).split(idx):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=0).fit(X[idx[a]], sfr[idx[a]])
        p = m.predict(X[idx[b]]); out['rmse_raw'].append(float(np.sqrt(np.mean((sfr[idx[b]] - p) ** 2))))
        ml = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=0).fit(X[idx[a]], np.log1p(sfr[idx[a]]))
        pl = ml.predict(X[idx[b]]); out['rmse_log1p_scale'].append(float(np.sqrt(np.mean((np.log1p(sfr[idx[b]]) - pl) ** 2))))
    return {k: {'mean': float(np.mean(v)), 'folds': v} for k, v in out.items()}
R['signal_check_positive_regressor'] = {'base13': cv_reg(Xb, pos_tr), 'base13_plus_env': cv_reg(Xn, pos_tr)}
s = R['signal_check_positive_regressor']
print(f'[5] 양수 회귀 ({len(pos_tr)}개) CV RMSE 원척도 base {s["base13"]["rmse_raw"]["mean"]:.3f} → +env {s["base13_plus_env"]["rmse_raw"]["mean"]:.3f}; '
      f'log1p 척도 {s["base13"]["rmse_log1p_scale"]["mean"]:.4f} → {s["base13_plus_env"]["rmse_log1p_scale"]["mean"]:.4f}  ({time.time()-t0:.0f}s)')
# 새 피처 중요도 (전체 학습 분류기, 중간 구간): permutation 대신 단일 피처 AUC
R['single_feature_zero_AUC_middle_train'] = {c: float(roc_auc_score(zero[mid_tr], np.nan_to_num(new[c].to_numpy()[mid_tr], nan=np.nanmedian(new[c])))) for c in new.columns}
R['single_feature_zero_AUC_middle_train']['provided_density'] = float(roc_auc_score(zero[mid_tr], dens[mid_tr]))

# 저장
(OUT / 'env_features_results.json').write_text(json.dumps(R, ensure_ascii=False, indent=2, default=float), encoding='utf-8')
np.savez_compressed(OUT / 'env_features_all_rows.npz', subhalo_id=df['subhalo_id'].to_numpy(), **{c: new[c].to_numpy() for c in new.columns})
print('saved', time.time() - t0)

# ───────────── 요약 출력 ─────────────
print('\n=== 제공 밀도와 후보 환경 측정의 Spearman / 학습 0 AUC ===')
for k in cands: print(f'  {k:14s} rho={R["provided_density_vs_candidates_spearman"][k]:.3f}  AUC={R["provided_density_vs_candidates_spearman_train_zero_AUC"][k]:.3f}')
print(f'  제공 밀도 AUC={R["provided_density_zero_AUC_train"]:.3f}, FoF log richness AUC={R["fof_richness_zero_AUC_train"]:.3f}, 그룹 최대 질량 AUC={R["fof_group_max_logm_zero_AUC_train"]:.3f}')
print('\n=== 0 비율 vs 가장 가까운 ≥10^14 헤일로까지 r/R200 (학습) ===')
for r in R['zero_fraction_vs_r_over_R200']['14.0']: print(f'  {r["bin"]:7s} n={r["n"]:7d} 0비율={r["fraction"]:.3f}')
print('=== r/R200<2 안에서 |Δv|/V200별 (≥10^14) ===')
for r in R['zero_fraction_vs_r_over_R200']['14.0_within2_by_vnorm']: print(f'  {r["bin"]:7s} n={r["n"]:6d} 0비율={r["fraction"]:.3f}')
print('\n=== FoF 그룹 richness별 0 비율: 전체 / 중심 / 위성 ===')
for a, b_, c in zip(R['fof_b0.2']['zero_fraction_vs_richness_all'], R['fof_b0.2']['zero_fraction_vs_richness_central'], R['fof_b0.2']['zero_fraction_vs_richness_satellite']):
    print(f'  N={a["bin"]:7s} 전체 n={a["n"]:7d} {a["fraction"]:.3f} | 중심 n={b_["n"]:7d} {b_["fraction"] if b_["fraction"] is not None else float("nan"):.3f} | 위성 n={c["n"]:6d} {c["fraction"] if c["fraction"] is not None else float("nan"):.3f}')
print('=== 항성 질량별 중심/위성 0 비율 ===')
for k, v in R['fof_b0.2']['zero_fraction_central_vs_satellite_by_stellar_mass'].items():
    print(f'  M*={k:9s} 중심 n={v["central"]["n"]:6d} {v["central"]["fraction"]:.4f} | 위성 n={v["satellite"]["n"]:6d} {v["satellite"]["fraction"]:.3f}')
print('=== 위성의 0 비율 vs 그룹 최대 헤일로 질량 ===')
for r in R['fof_b0.2']['zero_fraction_vs_group_max_logm_satellites']: print(f'  {r["bin"]:8s} n={r["n"]:6d} 0비율={r["fraction"] if r["fraction"] is not None else float("nan"):.3f}')
print('\n=== 낮은 밀도 0 은하 435개 ===', R['low_density_zeros_435'])
print('\n=== K-means vs FoF ===')
for k, v in R['kmeans'].items(): print(f'  K-means k={k}: 0 비율 범위 {v["zero_fraction_min"]:.3f}~{v["zero_fraction_max"]:.3f}, 군집 평균으로 0을 가르는 AUC {v["zero_AUC_train_using_cluster_mean"]:.3f}')
print('\n=== 중간 구간 단일 피처 AUC ===')
for k, v in sorted(R['single_feature_zero_AUC_middle_train'].items(), key=lambda kv: -abs(kv[1] - .5)): print(f'  {k:20s} {v:.3f}')

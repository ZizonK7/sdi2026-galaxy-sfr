"""2026-11-07 이론 검토.
SFR=0 은하의 주변 밀도 분포가 '중간 구간에서 정규분포처럼' 보이는 이유를,
밀도장 이론(로그정규)과 퀜칭 이론(환경·질량 퀜칭, Peng+2010 분리 가설)으로 나누어
학습 분할(2026-11-06과 같은 분할)에서 수치로 확인한다. 그림은 그리지 않는다.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats, optimize, special
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.model_selection import train_test_split

OUT = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[3]
DENS, TARGET = 'log10_overdensity_plus_1', 'sfr_linear'
df = pd.read_csv(ROOT / 'Galaxy Features with Position and Velocity for Star Formation Rate Prediction.csv')
sfr_all = df[TARGET].to_numpy()
tr, va = train_test_split(np.arange(len(df)), test_size=.20, random_state=42, stratify=(sfr_all > 0).astype(int))
t = df.iloc[tr].reset_index(drop=True)
d = t[DENS].to_numpy()
q = (t[TARGET].to_numpy() == 0)
edges = np.arange(-.75, 2.75 + 1e-9, .25)
nb = len(edges) - 1
bin_idx = np.clip(np.searchsorted(edges, d, side='left') - 1, 0, nb - 1)   # (left, right] 구간
assert len(t) == 166749 and q.sum() == 14438
L, U = 1.3647506592477927, 2.408565397956945                                   # 어제 경계 (평균 ± 2σ)
R = {'train_n': int(len(t)), 'train_zero': int(q.sum()), 'bin_edges': edges.tolist()}

def moments(x):
    x = np.asarray(x, float)
    return {'n': int(len(x)), 'mean': float(x.mean()), 'sd': float(x.std()), 'median': float(np.median(x)),
            'skew': float(stats.skew(x, bias=False)), 'excess_kurtosis': float(stats.kurtosis(x, bias=False)),
            'min': float(x.min()), 'max': float(x.max())}

def wmoments(x, w):
    w = w / w.sum(); m = (w * x).sum(); c = x - m; v = (w * c ** 2).sum()
    return {'mean': float(m), 'sd': float(np.sqrt(v)), 'skew': float((w * c ** 3).sum() / v ** 1.5),
            'excess_kurtosis': float((w * c ** 4).sum() / v ** 2 - 3)}

def aic(ll, k): return float(2 * k - 2 * ll)

# ───────────────────────── A. 전체 은하의 밀도 분포: 로그정규 밀도장 이론 ─────────────────────────
mu_a, sd_a = stats.norm.fit(d)
ll_norm_all = float(stats.norm.logpdf(d, mu_a, sd_a).sum())
g1 = stats.skew(d)
delta0 = np.sign(g1) * np.sqrt(np.pi / 2 * (abs(g1) ** (2 / 3)) / (abs(g1) ** (2 / 3) + ((4 - np.pi) / 2) ** (2 / 3)))
delta0 = np.clip(delta0, -.99, .99); a0 = delta0 / np.sqrt(1 - delta0 ** 2)
sc0 = d.std() / np.sqrt(1 - 2 * delta0 ** 2 / np.pi); loc0 = d.mean() - sc0 * delta0 * np.sqrt(2 / np.pi)
a_sn, loc_sn, sc_sn = stats.skewnorm.fit(d, a0, loc=loc0, scale=sc0)
ll_sn_all = float(stats.skewnorm.logpdf(d, a_sn, loc_sn, sc_sn).sum())
counts_all = np.bincount(bin_idx, minlength=nb)
counts_zero = np.bincount(bin_idx[q], minlength=nb)
R['A_all_galaxy_density'] = {
    'moments': moments(d),
    'normal_fit': {'mean': float(mu_a), 'sd': float(sd_a), 'loglik': ll_norm_all, 'AIC': aic(ll_norm_all, 2)},
    'skewnormal_fit': {'shape_alpha': float(a_sn), 'loc': float(loc_sn), 'scale': float(sc_sn),
                       'loglik': ll_sn_all, 'AIC': aic(ll_sn_all, 3)},
    'bins': [{'left': float(edges[i]), 'right': float(edges[i + 1]), 'all_observed': int(counts_all[i]),
              'all_normal_expected': float(len(d) * (stats.norm.cdf(edges[i + 1], mu_a, sd_a) - stats.norm.cdf(edges[i], mu_a, sd_a))),
              'all_skewnormal_expected': float(len(d) * (stats.skewnorm.cdf(edges[i + 1], a_sn, loc_sn, sc_sn) - stats.skewnorm.cdf(edges[i], a_sn, loc_sn, sc_sn))),
              'zero_observed': int(counts_zero[i])} for i in range(nb)],
}

# ───────────────────────── B. 퀜칭 비율 f_q(d)의 모양: 환경 퀜칭 이론 ─────────────────────────
def fit(nll, x0, k, name):
    r = optimize.minimize(nll, x0, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 40000, 'maxfev': 40000})
    r = optimize.minimize(nll, r.x, method='BFGS')
    return {'name': name, 'params': r.x.tolist(), 'loglik': float(-r.fun), 'AIC': aic(-r.fun, k)}

def p_logit(th, x): return special.expit(th[0] + th[1] * x)
def p_probit(th, x): return stats.norm.cdf(th[0] + th[1] * x)
def p_floor(th, x):
    e0 = special.expit(th[2]); return e0 + (1 - e0) * special.expit(th[0] + th[1] * x)
def p_floor_ceil(th, x):
    e0, e1 = special.expit(th[2]), special.expit(th[3]); return e0 + (e1 - e0) * special.expit(th[0] + th[1] * x)
def bern_nll(pfun):
    def nll(th):
        p = np.clip(pfun(th, d), 1e-12, 1 - 1e-12)
        return -(np.log(p[q]).sum() + np.log1p(-p[~q]).sum())
    return nll
models = {
    'logistic': (p_logit, fit(bern_nll(p_logit), [-10, 6], 2, 'logistic')),
    'probit': (p_probit, fit(bern_nll(p_probit), [-6, 3.5], 2, 'probit')),
    'logistic_floor': (p_floor, fit(bern_nll(p_floor), [-10, 6, -6], 3, 'logistic + 질량 퀜칭 바닥')),
    'logistic_floor_ceiling': (p_floor_ceil, fit(bern_nll(p_floor_ceil), [-10, 6, -6, 5], 4, 'logistic + 바닥 + 천장')),
}
# 포화 모형(14 구간 경험 비율)의 로그우도: 매끈한 모형이 구간 비율을 얼마나 놓치는지
p_bin = counts_zero / counts_all
ll_sat_bins = float(np.sum(counts_zero[counts_zero > 0] * np.log(p_bin[counts_zero > 0]))
                    + np.sum((counts_all - counts_zero)[p_bin < 1] * np.log1p(-p_bin[p_bin < 1])))
B = {'binned_saturated_loglik': ll_sat_bins, 'binned_saturated_AIC': aic(ll_sat_bins, nb), 'models': {}}
for key, (pf, res) in models.items():
    p_hat = pf(np.array(res['params']), d)
    exp_zero = np.array([p_hat[bin_idx == i].sum() for i in range(nb)])
    keep = exp_zero >= 5
    chi2 = float((((counts_zero - exp_zero) ** 2) / exp_zero)[keep].sum())
    info = dict(res)
    info.update({
        'floor_eps0': float(special.expit(res['params'][2])) if len(res['params']) >= 3 else 0.0,
        'ceiling_eps1': float(special.expit(res['params'][3])) if len(res['params']) >= 4 else 1.0,
        'd_at_half': float(-res['params'][0] / res['params'][1]),
        'predicted_zero_total': float(exp_zero.sum()),
        'predicted_zero_by_bin': exp_zero.tolist(),
        'chi2_bins_exp_ge5': chi2, 'n_bins_used': int(keep.sum()),
        'implied_zero_density_moments': wmoments(d, p_hat),         # 모형이 말하는 SFR=0 은하의 밀도 분포 모멘트
        'implied_moments_middle_only': wmoments(d[(d > L) & (d <= U)], p_hat[(d > L) & (d <= U)]),
    })
    B['models'][key] = info
B['observed_zero_by_bin'] = counts_zero.tolist()
B['observed_zero_moments'] = moments(d[q])
B['observed_zero_moments_middle_only'] = moments(d[q & (d > L) & (d <= U)])
B['binned_fraction'] = [{'left': float(edges[i]), 'right': float(edges[i + 1]), 'n': int(counts_all[i]),
                         'zero': int(counts_zero[i]), 'fraction': float(p_bin[i])} for i in range(nb)]
# 완전 모수 곱: 전체 밀도를 정규로 두고 probit을 곱한 '확장 왜정규' — 격자에서 모멘트
th_p = np.array(models['probit'][1]['params'])
grid = np.linspace(-1.5, 3.5, 20001)
dens = stats.norm.pdf(grid, mu_a, sd_a) * stats.norm.cdf(th_p[0] + th_p[1] * grid)
B['parametric_normal_times_probit'] = {'moments': wmoments(grid, dens),
                                       'moments_middle_only': wmoments(grid[(grid > L) & (grid <= U)], dens[(grid > L) & (grid <= U)])}
R['B_quenched_fraction_models'] = B

# ───────────────────────── C. SFR=0 은하 밀도: 정규 vs 왜정규 직접 적합 ─────────────────────────
z = d[q]
mu_z, sd_z = stats.norm.fit(z); ll_n = float(stats.norm.logpdf(z, mu_z, sd_z).sum())
a_z, loc_z, sc_z = stats.skewnorm.fit(z, -1.5, loc=2.05, scale=0.33)
ll_s = float(stats.skewnorm.logpdf(z, a_z, loc_z, sc_z).sum())
zm = z[(z > L) & (z <= U)]
mu_m, sd_m = stats.norm.fit(zm); ll_nm = float(stats.norm.logpdf(zm, mu_m, sd_m).sum())
a_m_, loc_m_, sc_m_ = stats.skewnorm.fit(zm, 0.5, loc=1.8, scale=0.27)
ll_sm = float(stats.skewnorm.logpdf(zm, a_m_, loc_m_, sc_m_).sum())
R['C_zero_density_normal_vs_skewnormal'] = {
    'all_zero': {'normal': {'mean': float(mu_z), 'sd': float(sd_z), 'AIC': aic(ll_n, 2)},
                 'skewnormal': {'alpha': float(a_z), 'loc': float(loc_z), 'scale': float(sc_z), 'AIC': aic(ll_s, 3)},
                 'delta_AIC_skew_minus_normal': aic(ll_s, 3) - aic(ll_n, 2)},
    'middle_only': {'n': int(len(zm)),
                    'normal': {'mean': float(mu_m), 'sd': float(sd_m), 'AIC': aic(ll_nm, 2)},
                    'skewnormal': {'alpha': float(a_m_), 'loc': float(loc_m_), 'scale': float(sc_m_), 'AIC': aic(ll_sm, 3)},
                    'delta_AIC_skew_minus_normal': aic(ll_sm, 3) - aic(ll_nm, 2)},
}

# ───────────────────────── D. Peng+2010 분리 가설: 질량 × 환경 ─────────────────────────
mass_q = pd.qcut(t['log10_stellar_mass'], 4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])
cell = pd.DataFrame({'m': mass_q.to_numpy(), 'b': bin_idx, 'q': q}).groupby(['m', 'b'], observed=True)['q'].agg(['size', 'sum']).reset_index()
cell.columns = ['m', 'b', 'n', 'k']
mi = {m: i for i, m in enumerate(['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])}
m_i = cell['m'].map(mi).to_numpy(); b_i = cell['b'].to_numpy(); n_c = cell['n'].to_numpy(float); k_c = cell['k'].to_numpy(float)
def cell_nll(p):
    p = np.clip(p, 1e-12, 1 - 1e-12); return -(k_c * np.log(p) + (n_c - k_c) * np.log1p(-p)).sum()
p_sat = np.where(n_c > 0, k_c / n_c, 0); ll_sat = -cell_nll(p_sat)
def nll_logit_add(th): return cell_nll(special.expit(th[:4][m_i] + th[4:][b_i]))
def nll_peng(th): return cell_nll(1 - np.exp(-(np.exp(th[:4][m_i]) + np.exp(th[4:][b_i]))))   # (1-f)=(1-ε_m)(1-ε_ρ)
def fit_cells(nll, x0):
    r = optimize.minimize(nll, x0, method='BFGS', options={'maxiter': 20000, 'gtol': 1e-8})
    r = optimize.minimize(nll, r.x, method='Nelder-Mead', options={'maxiter': 60000, 'maxfev': 60000, 'xatol': 1e-9, 'fatol': 1e-9})
    r = optimize.minimize(nll, r.x, method='BFGS', options={'maxiter': 20000, 'gtol': 1e-9})
    return r
x0_logit = np.concatenate([np.zeros(4), special.logit(np.clip(p_bin, 1e-4, 1 - 1e-4))])
r_logit = fit_cells(nll_logit_add, x0_logit)
x0_peng = np.concatenate([np.full(4, np.log(0.002)), np.log(np.clip(-np.log1p(-np.clip(p_bin, 0, 0.995)), 1e-4, None))])
r_peng = fit_cells(nll_peng, x0_peng)
k_add = 4 + nb - 1
n_cells = int(len(cell))
def cell_table(pfit):
    out = []
    for i in range(len(cell)):
        out.append({'mass': str(cell['m'][i]), 'bin_left': float(edges[b_i[i]]), 'n': int(n_c[i]), 'zero': int(k_c[i]),
                    'observed': float(p_sat[i]), 'fitted': float(pfit[i])})
    return out
p_fit_logit = special.expit(r_logit.x[:4][m_i] + r_logit.x[4:][b_i])
p_fit_peng = 1 - np.exp(-(np.exp(r_peng.x[:4][m_i]) + np.exp(r_peng.x[4:][b_i])))
# Peng식 환경 퀜칭 효율 ε_ρ(d | m) = [f(m,d) - f(m,d_ref)] / [1 - f(m,d_ref)], d_ref = 밀도 ≤ 1.0 구간을 합침
eps_rows = []
for m in mi:
    sub = cell[cell['m'] == m]
    ref = sub[sub['b'] <= 6]; f_ref = ref['k'].sum() / ref['n'].sum()
    for _, r in sub.iterrows():
        f = r['k'] / r['n'] if r['n'] > 0 else np.nan
        eps_rows.append({'mass': m, 'bin_left': float(edges[int(r['b'])]), 'n': int(r['n']), 'f_q': float(f),
                         'eps_rho': float((f - f_ref) / (1 - f_ref)) if r['n'] > 0 else None, 'f_ref': float(f_ref)})
R['D_peng_separability'] = {
    'cells': n_cells, 'saturated_loglik': float(ll_sat),
    'logit_additive': {'loglik': float(-r_logit.fun), 'k': k_add, 'AIC': aic(-r_logit.fun, k_add),
                       'G2_vs_saturated': float(2 * (r_logit.fun + ll_sat)), 'df': n_cells - k_add,
                       'p_value': float(stats.chi2.sf(2 * (r_logit.fun + ll_sat), n_cells - k_add)),
                       'mass_effects_logit': r_logit.x[:4].tolist()},
    'peng_product_additive': {'loglik': float(-r_peng.fun), 'k': k_add, 'AIC': aic(-r_peng.fun, k_add),
                              'G2_vs_saturated': float(2 * (r_peng.fun + ll_sat)), 'df': n_cells - k_add,
                              'p_value': float(stats.chi2.sf(2 * (r_peng.fun + ll_sat), n_cells - k_add)),
                              'mass_quenching_eps_m': (1 - np.exp(-np.exp(r_peng.x[:4]))).tolist()},
    'saturated_AIC': aic(ll_sat, n_cells),
    'cells_logit': cell_table(p_fit_logit), 'cells_peng': cell_table(p_fit_peng),
    'eps_rho_by_mass': eps_rows,
    'mass_quartile_edges': [float(x) for x in np.quantile(t['log10_stellar_mass'], [0, .25, .5, .75, 1])],
}

# ───────────────────────── E. 낮은 밀도의 SFR=0 은하 435개: 질량 퀜칭(AGN) 이론 ─────────────────────────
groups = {
    'low_zero (d<=L, SFR=0)': q & (d <= L), 'low_positive (d<=L, SFR>0)': ~q & (d <= L),
    'middle_zero': q & (d > L) & (d <= U), 'middle_positive': ~q & (d > L) & (d <= U),
    'high_zero (d>U)': q & (d > U),
}
cols = ['log10_stellar_mass', 'log10_BH_mass', 'log10_halo_mass', 'log10_main_prog_mass', 'log10_main_prog_stellar_mass', 'log10_mass_ratio', DENS]
bh_min = float(t['log10_BH_mass'].min())
E = {'BH_mass_min_value': bh_min, 'groups': {}}
for name, m in groups.items():
    g = t[m]
    E['groups'][name] = {
        'n': int(m.sum()),
        'median': {c: float(g[c].median()) for c in cols},
        'q10_q90_stellar': [float(g['log10_stellar_mass'].quantile(.1)), float(g['log10_stellar_mass'].quantile(.9))],
        'q10_q90_BH': [float(g['log10_BH_mass'].quantile(.1)), float(g['log10_BH_mass'].quantile(.9))],
        'frac_BH_at_min': float((g['log10_BH_mass'] <= bh_min + 1e-9).mean()),
        'frac_halo_lt_stellar': float((g['log10_halo_mass'] < g['log10_stellar_mass']).mean()),
        'frac_recent_merger_ratio_gt_m6': float((g['log10_mass_ratio'] > -6).mean()),
        'frac_stellar_ge_11': float((g['log10_stellar_mass'] >= 11).mean()),
        'frac_BH_ge_8': float((g['log10_BH_mass'] >= 8).mean()),
    }
# 낮은 밀도에서 SFR=0 비율이 항성 질량·BH 질량에 따라 어떻게 변하는가
low = t[d <= L]
for col, cuts in [('log10_stellar_mass', [8, 9, 10, 10.5, 11, 11.5, 12, 14]), ('log10_BH_mass', [5, 6, 7, 7.5, 8, 8.5, 9, 11])]:
    bins = pd.cut(low[col], cuts)
    gb = low.groupby(bins, observed=True)[TARGET].agg(n='size', zero=lambda s: int((s == 0).sum()))
    E[f'low_density_zero_fraction_by_{col}'] = [{'bin': str(i), 'n': int(r.n), 'zero': int(r.zero), 'fraction': float(r.zero / r.n)} for i, r in gb.iterrows()]
R['E_low_density_zero_vs_mass_quenching'] = E

# ───────────────────────── F. SFR=0의 정의: 시뮬레이션 별 생성 문턱 이론 ─────────────────────────
pos = t.loc[~q, TARGET].to_numpy()
R['F_sfr_near_zero_structure'] = {
    'positive_n': int(len(pos)), 'positive_unique_values': int(len(np.unique(pos))),
    'min_positive': float(pos.min()),
    'positive_quantiles': {str(p): float(np.quantile(pos, p)) for p in [.001, .01, .05, .1, .25, .5]},
    'count_below': {str(x): int((pos < x).sum()) for x in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]},
    'log10_positive_moments': moments(np.log10(pos)),
}

# ───────────────────────── G. 항성 질량별 SFR=0 밀도 분포 (어제 질문 3) ─────────────────────────
G = []
for m in mi:
    zz = d[q & (mass_q.to_numpy() == m)]
    G.append({'mass': m, **moments(zz)})
R['G_zero_density_by_mass_quartile'] = G

# ───────────────────────── H. 상자 크기·최고 밀도 상한·속도 (유한 상자, 램 압력) ─────────────────────────
H = {'position_min': {c: float(df[c].min()) for c in ['pos_x', 'pos_y', 'pos_z']},
     'position_max': {c: float(df[c].max()) for c in ['pos_x', 'pos_y', 'pos_z']},
     'density_max_all_data': float(df[DENS].max()), 'density_min_all_data': float(df[DENS].min())}
top = t[d > 2.25]
Z = linkage(top[['pos_x', 'pos_y', 'pos_z']].to_numpy(), method='single')
for thr in [1.0, 2.0, 5.0]:
    H[f'top_density_gt_2.25_n{len(top)}_clumps_within_{thr}'] = int(len(np.unique(fcluster(Z, thr, criterion='distance'))))
top_zero_frac = [{'left': float(edges[i]), 'zero_over_all': float(counts_zero[i] / counts_all[i])} for i in range(nb) if counts_all[i] > 0]
H['zero_share_of_all_by_bin'] = top_zero_frac
speed = np.sqrt((t[['vel_x', 'vel_y', 'vel_z']] ** 2).sum(axis=1)).to_numpy()
mid = (d > L) & (d <= U)
H['speed_median'] = {'all': float(np.median(speed)), 'low_density': float(np.median(speed[d <= L])),
                     'middle_zero': float(np.median(speed[mid & q])), 'middle_positive': float(np.median(speed[mid & ~q])),
                     'high_density': float(np.median(speed[d > U]))}
H['speed_AUC_for_zero_within_middle'] = float(stats.mannwhitneyu(speed[mid & q], speed[mid & ~q]).statistic / ((mid & q).sum() * (mid & ~q).sum()))
# 중간 구간에서 밀도만 vs 밀도+log 속도 logistic (AIC)
dm, qm, sm = d[mid], q[mid], np.log10(speed[mid] + 1e-9)
def nll2(th, X):
    eta = th[0] + X @ th[1:]; return -((-np.logaddexp(0, -eta))[qm].sum() + (-np.logaddexp(0, eta))[~qm].sum())
X1, X2 = dm[:, None], np.column_stack([dm, sm])
r1 = optimize.minimize(nll2, [0, 1], args=(X1,), method='BFGS'); r2 = optimize.minimize(nll2, [0, 1, 0], args=(X2,), method='BFGS')
H['middle_logistic'] = {'density_only_AIC': aic(-r1.fun, 2), 'density_plus_logspeed_AIC': aic(-r2.fun, 3),
                        'logspeed_coef': float(r2.x[2]), 'density_coef_with_speed': float(r2.x[1])}
R['H_box_and_velocity'] = H


# ───────────────────────── E2. 질량 퀜칭(AGN)은 SFR=0이 아니라 낮은 sSFR로 나타나는가 ─────────────────────────
lowpos = t[(d <= L) & ~q].copy()
lowpos['log_ssfr'] = np.log10(lowpos[TARGET]) - lowpos['log10_stellar_mass']
E2 = {'note': 'd<=L, SFR>0 은하. log10 sSFR = log10(SFR) - log10(M*)', 'n': int(len(lowpos)),
      'median_log_ssfr_all': float(lowpos['log_ssfr'].median())}
for col, cuts in [('log10_BH_mass', [5, 6, 7, 7.5, 8, 8.5, 9, 11]), ('log10_stellar_mass', [9, 10, 10.5, 11, 11.5, 12, 14])]:
    g = lowpos.groupby(pd.cut(lowpos[col], cuts), observed=True)['log_ssfr']
    E2[f'by_{col}'] = [{'bin': str(i), 'n': int(s.size), 'median': float(s.median()), 'q10': float(s.quantile(.1)),
                        'q90': float(s.quantile(.9)), 'frac_below_m11': float((s < -11).mean()),
                        'frac_below_m10_5': float((s < -10.5).mean())} for i, s in g]
R['E2_mass_quenching_as_low_ssfr'] = E2
allpos = t[~q]
g = allpos.groupby(pd.cut(allpos['log10_stellar_mass'], [9, 9.5, 10, 10.5, 11, 12, 14]), observed=True)[TARGET]
R['F_sfr_near_zero_structure']['min_positive_by_stellar_mass'] = [
    {'bin': str(i), 'n': int(s.size), 'min': float(s.min()), 'q0.1pct': float(s.quantile(.001)), 'q1pct': float(s.quantile(.01))} for i, s in g]
R['F_sfr_near_zero_structure']['stellar_mass_min'] = float(t['log10_stellar_mass'].min())
R['F_sfr_near_zero_structure']['BH_mass_floor_value'] = bh_min

(OUT / 'theory_check_results.json').write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding='utf-8')

# ───────────────────────── 요약 출력 ─────────────────────────
A = R['A_all_galaxy_density']
print('=== A. 전체 학습 은하의 밀도 d (로그정규 밀도장 이론) ===')
print(f"n={A['moments']['n']}, 평균={A['moments']['mean']:.3f}, sd={A['moments']['sd']:.3f}, 왜도={A['moments']['skew']:.3f}, 초과첨도={A['moments']['excess_kurtosis']:.3f}")
print(f"정규 AIC={A['normal_fit']['AIC']:.0f}  왜정규 AIC={A['skewnormal_fit']['AIC']:.0f} (alpha={A['skewnormal_fit']['shape_alpha']:.2f})")
print('구간   전체관측  정규기대  왜정규기대 | SFR=0 관측')
for b in A['bins']:
    print(f"({b['left']:5.2f},{b['right']:5.2f}] {b['all_observed']:7d} {b['all_normal_expected']:9.0f} {b['all_skewnormal_expected']:9.0f} | {b['zero_observed']:6d}")
print('\n=== B. 퀜칭 비율 f_q(d) 모형 (환경 퀜칭) ===')
print(f"구간 포화모형 AIC={B['binned_saturated_AIC']:.1f}")
for k, m in B['models'].items():
    im, imm = m['implied_zero_density_moments'], m['implied_moments_middle_only']
    print(f"{k:24s} AIC={m['AIC']:.1f} d_half={m['d_at_half']:.3f} 바닥={m['floor_eps0']:.4f} 천장={m['ceiling_eps1']:.4f} "
          f"예측0합={m['predicted_zero_total']:.0f} chi2={m['chi2_bins_exp_ge5']:.1f}({m['n_bins_used']}구간) "
          f"| 모형이 말하는 0의 d: 왜도={im['skew']:.3f} 첨도={im['excess_kurtosis']:.3f} | 중간만: 왜도={imm['skew']:.3f} 첨도={imm['excess_kurtosis']:.3f}")
om, omm = B['observed_zero_moments'], B['observed_zero_moments_middle_only']
print(f"관측 0의 d: 왜도={om['skew']:.3f} 첨도={om['excess_kurtosis']:.3f} | 중간만: 왜도={omm['skew']:.3f} 첨도={omm['excess_kurtosis']:.3f}")
pp = B['parametric_normal_times_probit']
print(f"완전 모수 곱(정규×probit): 왜도={pp['moments']['skew']:.3f} 첨도={pp['moments']['excess_kurtosis']:.3f} | 중간만: 왜도={pp['moments_middle_only']['skew']:.3f} 첨도={pp['moments_middle_only']['excess_kurtosis']:.3f}")
print('구간    관측0   logistic  +바닥  +바닥천장')
for i in range(nb):
    print(f"({edges[i]:5.2f}] {counts_zero[i]:6d} {B['models']['logistic']['predicted_zero_by_bin'][i]:8.1f} {B['models']['logistic_floor']['predicted_zero_by_bin'][i]:8.1f} {B['models']['logistic_floor_ceiling']['predicted_zero_by_bin'][i]:8.1f}")
C = R['C_zero_density_normal_vs_skewnormal']
print('\n=== C. SFR=0 밀도: 정규 vs 왜정규 ===')
print(f"전체 0: ΔAIC(왜정규−정규)={C['all_zero']['delta_AIC_skew_minus_normal']:.1f}, alpha={C['all_zero']['skewnormal']['alpha']:.2f}")
print(f"중간만: ΔAIC={C['middle_only']['delta_AIC_skew_minus_normal']:.1f}, alpha={C['middle_only']['skewnormal']['alpha']:.2f}")
D = R['D_peng_separability']
print('\n=== D. Peng+2010 분리 가설 (질량 4분위 × 밀도 14구간) ===')
print(f"셀 {D['cells']}개, 포화 AIC={D['saturated_AIC']:.1f}")
print(f"logit 가법: AIC={D['logit_additive']['AIC']:.1f} G2={D['logit_additive']['G2_vs_saturated']:.1f} df={D['logit_additive']['df']} p={D['logit_additive']['p_value']:.3g}")
print(f"Peng 곱 가법: AIC={D['peng_product_additive']['AIC']:.1f} G2={D['peng_product_additive']['G2_vs_saturated']:.1f} df={D['peng_product_additive']['df']} p={D['peng_product_additive']['p_value']:.3g}  ε_m={np.round(D['peng_product_additive']['mass_quenching_eps_m'], 4)}")
print('ε_ρ(d|m) 표 (행=밀도 구간 왼쪽, 열=질량 분위):')
er = pd.DataFrame(D['eps_rho_by_mass']).pivot(index='bin_left', columns='mass', values='eps_rho')
print(er.round(3).to_string())
print('관측 f_q (행=밀도, 열=질량):')
fq = pd.DataFrame(D['eps_rho_by_mass']).pivot(index='bin_left', columns='mass', values='f_q')
print(fq.round(3).to_string())
E = R['E_low_density_zero_vs_mass_quenching']
print('\n=== E. 낮은 밀도 SFR=0 435개 vs 다른 집단 (질량 퀜칭) ===')
print(f"BH 질량 최소값 {E['BH_mass_min_value']:.3f}")
for name, g in E['groups'].items():
    md = g['median']
    print(f"{name:30s} n={g['n']:6d} 항성={md['log10_stellar_mass']:.2f} BH={md['log10_BH_mass']:.2f} 헤일로={md['log10_halo_mass']:.2f} 병합비={md['log10_mass_ratio']:.2f} "
          f"| BH최소비율={g['frac_BH_at_min']:.2%} 헤일로<항성={g['frac_halo_lt_stellar']:.2%} 항성≥11={g['frac_stellar_ge_11']:.1%} BH≥8={g['frac_BH_ge_8']:.1%}")
for key in ['low_density_zero_fraction_by_log10_stellar_mass', 'low_density_zero_fraction_by_log10_BH_mass']:
    print(key); [print(f"   {r['bin']:14s} n={r['n']:7d} 0={r['zero']:4d} 비율={r['fraction']:.3%}") for r in E[key]]
F = R['F_sfr_near_zero_structure']
print('\n=== F. SFR 0 근처 구조 ===')
print(f"양수 {F['positive_n']}개, 고유값 {F['positive_unique_values']}개, 최소 양수={F['min_positive']:.3e}, 분위수={ {k: round(v, 5) for k, v in F['positive_quantiles'].items()} }")
print(f"작은 값 개수 {F['count_below']}, log10 양수 SFR 왜도={F['log10_positive_moments']['skew']:.3f} 첨도={F['log10_positive_moments']['excess_kurtosis']:.3f}")
print('\n=== E2. 낮은 밀도 양수 은하의 sSFR (질량 퀜칭은 0이 아니라 낮은 sSFR로) ===')
for r in R['E2_mass_quenching_as_low_ssfr']['by_log10_BH_mass']:
    print(f"   BH {r['bin']:12s} n={r['n']:7d} 중앙값={r['median']:.2f} 10%={r['q10']:.2f} sSFR<1e-11 비율={r['frac_below_m11']:.1%}")
print('\n=== G. 질량 분위별 SFR=0 밀도 ===')
for g in R['G_zero_density_by_mass_quartile']:
    print(f"{g['mass']:10s} n={g['n']:5d} 평균={g['mean']:.3f} sd={g['sd']:.3f} 왜도={g['skew']:.3f} 첨도={g['excess_kurtosis']:.3f}")
H = R['H_box_and_velocity']
print('\n=== H. 상자·최고 밀도·속도 ===')
print('위치 범위', H['position_min'], H['position_max'], '| 밀도 범위', H['density_min_all_data'], H['density_max_all_data'])
print({k: v for k, v in H.items() if 'clumps' in k})
print('속도 중앙값', {k: round(v, 1) for k, v in H['speed_median'].items()}, '| 중간 구간 |v| AUC(0 기준)', round(H['speed_AUC_for_zero_within_middle'], 3))
print('중간 logistic', H['middle_logistic'])

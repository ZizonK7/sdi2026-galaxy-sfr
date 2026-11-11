"""log_model_cv.py와 제출 노트북 9.2의 환경 피처 함수 (q4_reg_trees.py가 import한다)."""
import numpy as np
import pandas as pd


def build_environment_features(frame, box=200.0, fof_b=0.2, host_thresholds=(13.5, 14.0)):
    """제출 노트북 9.2와 같은 함수 (log_metric_check.py와 같은 정의, 타깃 미사용)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    RHO_CRIT, G_KMS = 2.775e11, 4.30091e-9
    pos = frame[["pos_x", "pos_y", "pos_z"]].to_numpy() % box
    vel = frame[["vel_x", "vel_y", "vel_z"]].to_numpy()
    mh = frame["log10_halo_mass"].to_numpy()
    n = len(frame)
    tree = cKDTree(pos, boxsize=box)
    out = {}
    d, _ = tree.query(pos, k=11)
    out["log_rho_5nn"] = np.log10(5 / (4 / 3 * np.pi * d[:, 5] ** 3))
    out["log_rho_10nn"] = np.log10(10 / (4 / 3 * np.pi * d[:, 10] ** 3))
    for r in (1.0, 2.0, 5.0):
        out[f"log1p_N_r{int(r)}"] = np.log1p(tree.query_ball_point(pos, r, return_length=True) - 1)
    r200 = lambda logm: (3 * 10 ** logm / (4 * np.pi * 200 * RHO_CRIT)) ** (1 / 3)
    v200 = lambda logm: np.sqrt(G_KMS * 10 ** logm / r200(logm))
    for thr in host_thresholds:
        sel = np.where(mh >= thr)[0]
        dist, j = cKDTree(pos[sel], boxsize=box).query(pos, k=1)
        hj = sel[j]
        dv = np.sqrt(((vel - vel[hj]) ** 2).sum(axis=1))
        tag = f"{thr:g}"
        out[f"dist_host{tag}"] = dist
        out[f"r_over_R200_{tag}"] = dist / r200(mh[hj])
        out[f"v_over_V200_{tag}"] = dv / v200(mh[hj])
        if thr == host_thresholds[0]:
            out[f"host_logm_{tag}"] = mh[hj]
    link = fof_b * box / n ** (1 / 3)
    pairs = tree.query_pairs(link, output_type="ndarray")
    _, lab = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)), directed=False)
    rich = np.bincount(lab)[lab]
    gmax = pd.Series(mh).groupby(lab).transform("max").to_numpy()
    vmean = pd.DataFrame(vel).groupby(lab).transform("mean").to_numpy()
    dv_grp = np.sqrt(((vel - vmean) ** 2).sum(axis=1))
    dv_grp[rich == 1] = np.nan
    out["fof_log_richness"] = np.log(rich)
    out["fof_is_central"] = (mh >= gmax - 1e-9).astype(float)
    out["fof_group_max_logm"] = gmax
    out["fof_dv_group"] = dv_grp
    out["halo_minus_stellar"] = mh - frame["log10_stellar_mass"].to_numpy()
    return pd.DataFrame(out, index=frame.index)

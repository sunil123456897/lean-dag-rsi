"""
SLEEP-DAG-RSI v4/v6 Gauss-Newton epistatic pruning, dual-Gram spectral
diagnostics, and autonomous FrameTransforms (`SpectralFrameLifter`).
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple


def _jacobi_eigh_symmetric(mat: List[List[float]], max_sweeps: int = 240) -> Tuple[List[float], List[List[float]]]:
    """
    Deterministic symmetric eigenvalue solver for real symmetric matrices G in R^{d x d}.
    Uses fast C-level `numpy.linalg.eigh` when NumPy is available and d > 12, with
    pure-Python deterministic Jacobi rotations as a zero-dependency fallback.
    Returns (eigenvalues_ascending, eigenvectors_as_columns_list[mode][coord]).
    """
    d = len(mat)
    if d == 0:
        return [], []
    if d > 12:
        try:
            import numpy as _np
            arr = _np.asarray(mat, dtype=_np.float64)
            arr = 0.5 * (arr + arr.T)
            w, v_np = _np.linalg.eigh(arr)
            evals_np = [float(x) for x in w.tolist()]
            evecs_np = [[float(v_np[coord, mode]) for coord in range(d)] for mode in range(d)]
            return evals_np, evecs_np
        except Exception:
            pass
    a = [[float(mat[r][c]) for c in range(d)] for r in range(d)]
    v = [[1.0 if r == c else 0.0 for c in range(d)] for r in range(d)]

    for _ in range(max_sweeps):
        max_off = 0.0
        p, q = 0, 1
        for r in range(d):
            for c in range(r + 1, d):
                if abs(a[r][c]) > max_off:
                    max_off = abs(a[r][c])
                    p, q = r, c
        if max_off < 1e-14:
            break
        diff = a[q][q] - a[p][p]
        if abs(a[p][q]) < 1e-30:
            t = 0.0
        else:
            phi = diff / (2.0 * a[p][q])
            t = 1.0 / (abs(phi) + math.sqrt(phi * phi + 1.0))
            if phi < 0.0:
                t = -t
        c_rot = 1.0 / math.sqrt(1.0 + t * t)
        s_rot = t * c_rot
        tau = s_rot / (1.0 + c_rot)
        h = t * a[p][q]
        a[p][p] -= h
        a[q][q] += h
        a[p][q] = 0.0
        a[q][p] = 0.0
        for r in range(d):
            if r != p and r != q:
                g_rp = a[r][p]
                g_rq = a[r][q]
                a[r][p] = g_rp - s_rot * (g_rq + tau * g_rp)
                a[p][r] = a[r][p]
                a[r][q] = g_rq + s_rot * (g_rp - tau * g_rq)
                a[q][r] = a[r][q]
        for r in range(d):
            v_rp = v[r][p]
            v_rq = v[r][q]
            v[r][p] = v_rp - s_rot * (v_rq + tau * v_rp)
            v[r][q] = v_rq + s_rot * (v_rp - tau * v_rq)

    evals = [a[i][i] for i in range(d)]
    evecs = [[v[coord][mode] for coord in range(d)] for mode in range(d)]
    order = sorted(range(d), key=lambda idx: evals[idx])
    return [evals[i] for i in order], [evecs[i] for i in order]


def prune_epistatic_gauss_newton(
    eval_buckets_fn: Callable[[Sequence[float]], Sequence[float]],
    theta_opt: Sequence[float],
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    theta_default: Optional[Sequence[float]] = None,
    param_names: Optional[Sequence[str]] = None,
    macro_step: float = 0.04,
    sloppy_rel_tol: float = 1e-3,
    score_drop_tol: float = 1e-3,
) -> Dict[str, Any]:
    """
    Lean-DAG-RSI v4 Scale-Invariant Empirical Bucket Covariance (G = (1/M) J^T J)
    Sloppy-Mode Pruner & Epistatic Invariant Collapser:
      1. Normalizes coordinates to dimensionless unit hypercube tilde_theta in [0, 1]^d
         so eigenvectors are invariant to parameter units (e.g. [0, 100] vs [0, 0.001]).
      2. Computes macro-step bucket Jacobian J_{j,i} = (phi_j(th + delta*e_i) - phi_j(th - delta*e_i)) / (2*delta)
         and positive semi-definite sensitivity matrix G = (1/M) J^T J.
      3. Detects coupled epistatic null modes (G q_m ~ 0 where >=2 coordinates have large weights,
         e.g. when objective depends only on ratio theta_i / theta_j or sum theta_i + theta_j)
         that single-coordinate `prune_tunable_params` cannot prune!
      4. Collapses redundant parameter(s) along the sloppy eigenvector while holding the
         invariant constant, and probes the sharpest curvature eigenvector for plateau safety.
    """
    d = len(theta_opt)
    if d == 0:
        return {"active_count": 0, "sloppy_modes": [], "coupled_invariants": []}

    names = [str(x) for x in param_names] if param_names is not None else [f"theta_{i}" for i in range(d)]
    defaults = [float(x) for x in theta_default] if theta_default is not None else [0.0] * d
    if bounds is not None:
        lows = [float(b[0]) for b in bounds]
        highs = [float(b[1]) for b in bounds]
    else:
        lows = [min(0.0, float(theta_opt[i]) - 1.0) for i in range(d)]
        highs = [max(1.0, abs(float(theta_opt[i])) * 2.0) for i in range(d)]
    spans = [max(1e-9, highs[i] - lows[i]) for i in range(d)]

    base_theta = [float(x) for x in theta_opt]
    base_buckets = [float(v) for v in eval_buckets_fn(base_theta)]
    m_buckets = max(1, len(base_buckets))
    base_score = sum(base_buckets) / float(m_buckets)

    # Compute Scale-Invariant Gauss-Newton Sensitivity + Squared Macro-Hessian at theta_opt:
    #   G = (1/M) J(theta*)^T J(theta*) + (1/M) sum_{j=1}^M (H^(j)(theta*))^T H^(j)(theta*) >= 0
    # Why this exact combination works on both step functions and smooth nonlinear ridges:
    #   - When grad(phi_j) != 0, J^T J captures the first-order bucket sensitivity.
    #   - At an interior optimum theta* where g(theta*) = g* (so grad(phi_j) = 0),
    #     nabla^2 [(g(theta) - g*)^2] = 2 nabla g(theta*) nabla g(theta*)^T + 0,
    #     so H^(j)(theta*) has the EXACT nullspace of the nonlinear invariant g(theta) = g*
    #     without stencil-rotation bias!
    j_mat: List[List[float]] = [[0.0] * d for _ in range(m_buckets)]
    step_phys_vec = [macro_step * spans[i] for i in range(d)]
    for i in range(d):
        th_plus = list(base_theta)
        th_minus = list(base_theta)
        th_plus[i] = min(highs[i], base_theta[i] + step_phys_vec[i])
        th_minus[i] = max(lows[i], base_theta[i] - step_phys_vec[i])
        actual_d_norm = max(1e-9, (th_plus[i] - th_minus[i]) / spans[i])
        b_plus = [float(v) for v in eval_buckets_fn(th_plus)]
        b_minus = [float(v) for v in eval_buckets_fn(th_minus)]
        for j in range(m_buckets):
            j_mat[j][i] = (b_plus[j] - b_minus[j]) / actual_d_norm

    # Macro-step bucket Hessians H^(j)_{r,c} in normalized coordinates
    h_buckets: List[List[List[float]]] = [
        [[0.0] * d for _ in range(d)] for _ in range(m_buckets)
    ]
    for r in range(d):
        # Diagonal second difference H^(j)_{r,r}
        th_p = list(base_theta)
        th_m = list(base_theta)
        th_p[r] = min(highs[r], base_theta[r] + step_phys_vec[r])
        th_m[r] = max(lows[r], base_theta[r] - step_phys_vec[r])
        d_r = max(1e-9, 0.5 * (th_p[r] - th_m[r]) / spans[r])
        bp = [float(v) for v in eval_buckets_fn(th_p)]
        bm = [float(v) for v in eval_buckets_fn(th_m)]
        for j in range(m_buckets):
            h_buckets[j][r][r] = (bp[j] - 2.0 * base_buckets[j] + bm[j]) / (d_r * d_r)

        for c in range(r + 1, d):
            th_pp = list(base_theta)
            th_pm = list(base_theta)
            th_mp = list(base_theta)
            th_mm = list(base_theta)
            th_pp[r] = min(highs[r], base_theta[r] + step_phys_vec[r])
            th_pp[c] = min(highs[c], base_theta[c] + step_phys_vec[c])
            th_pm[r] = min(highs[r], base_theta[r] + step_phys_vec[r])
            th_pm[c] = max(lows[c], base_theta[c] - step_phys_vec[c])
            th_mp[r] = max(lows[r], base_theta[r] - step_phys_vec[r])
            th_mp[c] = min(highs[c], base_theta[c] + step_phys_vec[c])
            th_mm[r] = max(lows[r], base_theta[r] - step_phys_vec[r])
            th_mm[c] = max(lows[c], base_theta[c] - step_phys_vec[c])
            d_rc = max(
                1e-9,
                0.25
                * ((th_pp[r] - th_mp[r]) / spans[r])
                * ((th_pp[c] - th_pm[c]) / spans[c]),
            )
            bpp = [float(v) for v in eval_buckets_fn(th_pp)]
            bpm = [float(v) for v in eval_buckets_fn(th_pm)]
            bmp = [float(v) for v in eval_buckets_fn(th_mp)]
            bmm = [float(v) for v in eval_buckets_fn(th_mm)]
            for j in range(m_buckets):
                val_rc = (bpp[j] - bpm[j] - bmp[j] + bmm[j]) / (4.0 * d_rc)
                h_buckets[j][r][c] = val_rc
                h_buckets[j][c][r] = val_rc

    g_mat: List[List[float]] = [[0.0] * d for _ in range(d)]
    for r in range(d):
        for c in range(r, d):
            val_j = sum(j_mat[j][r] * j_mat[j][c] for j in range(m_buckets)) / float(m_buckets)
            val_h = (
                sum(
                    sum(h_buckets[j][r][k] * h_buckets[j][c][k] for k in range(d))
                    for j in range(m_buckets)
                )
                / float(m_buckets)
            )
            g_mat[r][c] = val_j + val_h
            g_mat[c][r] = val_j + val_h

    evals, evecs = _jacobi_eigh_symmetric(g_mat)
    max_eval = max(max(evals), 1e-12)
    sloppy_thresh = sloppy_rel_tol * max_eval

    sloppy_modes: List[Dict[str, Any]] = []
    coupled_invariants: List[str] = []
    collapsed_theta = list(base_theta)
    frozen_indices: Set[int] = set()

    for m_idx, (lam, vec) in enumerate(zip(evals, evecs)):
        if lam <= sloppy_thresh:
            active_coords = [i for i in range(d) if abs(vec[i]) >= 0.25]
            sloppy_modes.append(
                {
                    "mode_index": m_idx,
                    "eigenvalue": round(lam, 10),
                    "eigenvector": [round(x, 6) for x in vec],
                    "active_params": [names[i] for i in active_coords],
                    "is_coupled_epistatic": len(active_coords) >= 2,
                }
            )
            if len(active_coords) == 1:
                idx0 = active_coords[0]
                trial = list(collapsed_theta)
                trial[idx0] = defaults[idx0]
                s_try = sum(float(v) for v in eval_buckets_fn(trial)) / float(m_buckets)
                if base_score - s_try <= score_drop_tol:
                    collapsed_theta[idx0] = defaults[idx0]
                    frozen_indices.add(idx0)
            elif len(active_coords) >= 2:
                # Coupled epistatic ridge: move along null eigenvector `vec` in normalized space
                # so that one parameter hits its default/anchor while the coupled partner compensates!
                i_primary, i_secondary = active_coords[0], active_coords[1]
                coupled_invariants.append(
                    f"invariant({names[i_primary]}, {names[i_secondary]}): "
                    f"{vec[i_primary]:+.3f}*d_norm({names[i_primary]}) {vec[i_secondary]:+.3f}*d_norm({names[i_secondary]}) = 0"
                )
                # Try collapsing i_secondary to defaults[i_secondary] while sliding i_primary along `vec`
                if abs(vec[i_secondary]) > 1e-6:
                    delta_norm_sec = (defaults[i_secondary] - collapsed_theta[i_secondary]) / spans[i_secondary]
                    alpha = delta_norm_sec / vec[i_secondary]
                    trial = list(collapsed_theta)
                    for k_coord in active_coords:
                        trial[k_coord] = collapsed_theta[k_coord] + alpha * vec[k_coord] * spans[k_coord]
                    # Also test multiplicative ratio invariance theta_primary / theta_secondary = const
                    trial_ratio = list(collapsed_theta)
                    if abs(collapsed_theta[i_secondary]) > 1e-9 and abs(defaults[i_secondary]) > 1e-9:
                        ratio = collapsed_theta[i_primary] / collapsed_theta[i_secondary]
                        trial_ratio[i_secondary] = defaults[i_secondary]
                        trial_ratio[i_primary] = ratio * defaults[i_secondary]
                    for cand in (trial_ratio, trial):
                        s_cand = sum(float(v) for v in eval_buckets_fn(cand)) / float(m_buckets)
                        if base_score - s_cand <= score_drop_tol:
                            collapsed_theta = cand
                            frozen_indices.add(i_secondary)
                            break

    # Probe along sharpest eigenvector (largest eigenvalue = steepest curvature direction)
    sharpest_vec = evecs[-1]
    sharp_probes: List[float] = []
    for sign in (-1.0, 1.0):
        probe_th = [
            min(highs[i], max(lows[i], collapsed_theta[i] + sign * macro_step * sharpest_vec[i] * spans[i]))
            for i in range(d)
        ]
        b_probe = [float(v) for v in eval_buckets_fn(probe_th)]
        sharp_probes.append(sum(b_probe) / float(m_buckets))

    collapsed_score = sum(float(v) for v in eval_buckets_fn(collapsed_theta)) / float(m_buckets)
    return {
        "base_score": round(base_score, 6),
        "collapsed_score": round(collapsed_score, 6),
        "collapsed_theta": [round(x, 8) for x in collapsed_theta],
        "eigenvalues": [round(ev, 8) for ev in evals],
        "sloppy_modes": sloppy_modes,
        "coupled_invariants": coupled_invariants,
        "frozen_params": [names[i] for i in sorted(frozen_indices)],
        "effective_dof": d - len(frozen_indices),
        "sharpest_eigenvector_mean": round(sum(sharp_probes) / len(sharp_probes), 6),
    }


def _dual_gram_eigh_from_jacobian(
    j_mat: Sequence[Sequence[float]],
) -> Tuple[List[float], List[List[float]]]:
    """
    Fast Dual Gram Eigendecomposition for Rectangular Jacobians `J in R^{M x d}` when `M < d`:
      - Instead of forming and diagonalizing `G = (1/M) J^T J in R^{d x d}` in O(d^3) time,
        forms the dual Gram matrix `K = (1/M) J J^T in R^{M x M}` in O(M^2 d) time,
        diagonalizes `K u_r = lambda_r u_r` in O(M^3) time, and recovers the exact
        orthonormal primal eigenvectors in R^d via:
            v_r = (J^T u_r) / ||J^T u_r||_2 = (1 / sqrt(M * lambda_r)) J^T u_r.
    Returns `(eigenvalues_ascending_M, primal_eigenvectors_M_x_d)`.
    """
    m_rows = len(j_mat)
    if m_rows == 0:
        return [], []
    d_cols = len(j_mat[0])
    if d_cols == 0:
        return [], []

    try:
        import numpy as _np
        j_np = _np.asarray(j_mat, dtype=_np.float64)
        k_np = (j_np @ j_np.T) / float(m_rows)
        k_np = 0.5 * (k_np + k_np.T)
        w_np, u_np = _np.linalg.eigh(k_np)
        evals_k = [max(0.0, float(x)) for x in w_np.tolist()]
        # Lift dual eigenvectors u_r in R^M to primal eigenvectors v_r in R^d: W = J^T @ U
        w_primal = j_np.T @ u_np  # shape (d, M)
        evecs_d: List[List[float]] = []
        for r in range(m_rows):
            col = w_primal[:, r]
            norm_c = float(_np.linalg.norm(col))
            if norm_c > 1e-15:
                evecs_d.append((col / norm_c).tolist())
            else:
                evecs_d.append([0.0] * d_cols)
        return evals_k, evecs_d
    except Exception:
        pass

    k_mat: List[List[float]] = [[0.0] * m_rows for _ in range(m_rows)]
    for r in range(m_rows):
        jr = j_mat[r]
        for c in range(r, m_rows):
            jc = j_mat[c]
            dot_rc = sum(float(jr[i]) * float(jc[i]) for i in range(d_cols)) / float(m_rows)
            k_mat[r][c] = dot_rc
            k_mat[c][r] = dot_rc

    evals_k, evecs_k = _jacobi_eigh_symmetric(k_mat)
    pos_evals_k = [max(0.0, float(ev)) for ev in evals_k]
    evecs_d = []
    for r in range(m_rows):
        ur = evecs_k[r]
        wr = [sum(float(ur[b]) * float(j_mat[b][i]) for b in range(m_rows)) for i in range(d_cols)]
        norm_w = math.sqrt(sum(val * val for val in wr))
        if norm_w > 1e-15:
            evecs_d.append([val / norm_w for val in wr])
        else:
            evecs_d.append([0.0] * d_cols)
    return pos_evals_k, evecs_d


def compute_spectral_diagnostics(
    g_mat: Optional[Sequence[Sequence[float]]] = None,
    m_buckets: Optional[int] = None,
    train_score: float = 0.0,
    oos_score: float = 0.0,
    pass_mask: Optional[Sequence[int]] = None,
    use_hutchinson_above_d: int = 30,
    seed: int = 42,
    j_mat: Optional[Sequence[Sequence[float]]] = None,
    null_rel_tol: float = 1e-12,
) -> Dict[str, Any]:
    """
    SLEEP-DAG-RSI v6/v7/v8 Regularized Spectral Diagnostics on G = (1/M) J^T J:
      1. Dual Gram Fast Path (`M < d`): When `j_mat` in R^{M x d} is provided with `M < d`
         (e.g. `d = 2560`, `M = 16`), diagonalizes the Dual Gram matrix `K = (1/M) J J^T in R^{M x M}`
         in O(M^2 d + M^3) time instead of O(d^3), recovering the exact non-zero spectrum
         and lifted R^d eigenvectors without ever allocating a d x d matrix.
      2. Computes Relative-Threshold Numerical Rank `r_sup = |{r : lambda_r > tau_null * lambda_max}|`
         (`tau_null = 1e-12`) to index `lambda_{r_sup}` for `kappa_eff(G) = lambda_max / max(lambda_{r_sup}, 1e-12)`.
         Unlike participation-ratio indexing `r_eff = (Tr G)^2 / Tr(G^2)` (which is blinded to
         small stiff eigenvalues like `(1.0, 0.8, 1e-8)` where `r_eff < 2`), `r_sup` detects
         every non-zero multi-scale valley while excluding structural rank-deficient zeros (`M < d`).
      3. Retains `r_eff` as an auxiliary trace diagnostic and outputs the 4D Stall Signature:
         `sigma = (log10(kappa_eff), r_eff / d, train_oos_gap, fingerprint_entropy)`.
    """
    use_dual = False
    if j_mat is not None and len(j_mat) > 0 and len(j_mat[0]) > 0:
        m_j = len(j_mat)
        d_j = len(j_mat[0])
        if g_mat is None or len(g_mat) == 0 or m_j < d_j:
            use_dual = True
            d = d_j
            if m_buckets is None:
                m_buckets = m_j
    if not use_dual:
        d = len(g_mat) if g_mat is not None else 0

    if d == 0:
        return {
            "d": 0,
            "r_sup": 0,
            "r_sup_ratio": 0.0,
            "r_eff": 0.0,
            "r_eff_ratio": 0.0,
            "kappa_eff": 1.0,
            "log10_kappa_eff": 0.0,
            "stall_signature": (0.0, 0.0, 0.0, 0.0),
            "is_ill_conditioned": False,
            "used_dual_gram": False,
        }

    if use_dual and j_mat is not None:
        evals, evecs = _dual_gram_eigh_from_jacobian(j_mat)
        pos_evals = [max(0.0, float(ev)) for ev in evals]
        lam_max = max(pos_evals[-1] if pos_evals else 0.0, 1e-12)
        tr_g = sum(pos_evals)
        tr_g2 = sum(ev * ev for ev in pos_evals)
        r_eff = (tr_g * tr_g) / max(tr_g2, 1e-24)
        observable_rank = min(d, int(m_buckets)) if (m_buckets is not None and m_buckets > 0) else len(pos_evals)
    else:
        assert g_mat is not None
        evals, evecs = _jacobi_eigh_symmetric([list(row) for row in g_mat])
        pos_evals = [max(0.0, float(ev)) for ev in evals]
        lam_max = max(pos_evals[-1] if pos_evals else 0.0, 1e-12)

        if d > use_hutchinson_above_d:
            # Hutchinson stochastic Rademacher trace estimation O(m * d^2) with m=16
            rng_h = random.Random(seed)
            m_probes = 16
            tr_est = 0.0
            tr_sq_est = 0.0
            for _ in range(m_probes):
                v = [1.0 if rng_h.random() >= 0.5 else -1.0 for _ in range(d)]
                gv = [sum(float(g_mat[r][c]) * v[c] for c in range(d)) for r in range(d)]
                v_gv = sum(v[r] * gv[r] for r in range(d))
                gv_norm_sq = sum(x * x for x in gv)
                tr_est += v_gv / float(m_probes)
                tr_sq_est += gv_norm_sq / float(m_probes)
            r_eff = (tr_est * tr_est) / max(tr_sq_est, 1e-24)
        else:
            tr_g = sum(pos_evals)
            tr_g2 = sum(ev * ev for ev in pos_evals)
            r_eff = (tr_g * tr_g) / max(tr_g2, 1e-24)

        observable_rank = min(d, int(m_buckets)) if (m_buckets is not None and m_buckets > 0) else d

    candidate_evals = pos_evals[-observable_rank:] if observable_rank > 0 else pos_evals
    active_support_evals = [ev for ev in candidate_evals if ev > float(null_rel_tol) * lam_max]
    r_sup = max(1, len(active_support_evals))
    lam_support = max(min(active_support_evals) if active_support_evals else 0.0, 1e-12)

    r_eff = max(1.0, min(float(d), r_eff))
    kappa_eff = lam_max / lam_support
    log10_kappa = math.log10(max(1.0, kappa_eff))

    # Shannon entropy of pass_mask bit distribution in [0, 1]
    fp_entropy = 0.0
    if pass_mask and len(pass_mask) > 0:
        p1 = sum(1 for b in pass_mask if b) / float(len(pass_mask))
        p0 = 1.0 - p1
        if 0.0 < p1 < 1.0:
            fp_entropy = -(p1 * math.log2(p1) + p0 * math.log2(p0))

    train_oos_gap = float(train_score) - float(oos_score)
    sig = (
        round(log10_kappa, 4),
        round(r_eff / float(d), 4),
        round(train_oos_gap, 4),
        round(fp_entropy, 4),
    )
    return {
        "d": d,
        "observable_rank": observable_rank,
        "r_sup": r_sup,
        "r_sup_ratio": round(r_sup / float(d), 6),
        "eigenvalues": [round(ev, 10) for ev in pos_evals],
        "eigenvectors": evecs,
        "r_eff": round(r_eff, 6),
        "r_eff_ratio": round(r_eff / float(d), 6),
        "kappa_eff": round(kappa_eff, 4),
        "log10_kappa_eff": round(log10_kappa, 4),
        "train_oos_gap": round(train_oos_gap, 6),
        "fingerprint_entropy": round(fp_entropy, 6),
        "stall_signature": sig,
        "is_ill_conditioned": bool(kappa_eff >= 1e8),
        "used_dual_gram": use_dual,
    }


class FrameTransform:
    """
    Base class for Stage 2 Autonomous Coordinate / Representation Transformations (Phi -> Phi').
    Enforces two hard safety invariants:
      1. `inverse_projection(u_lifted) -> x_orig`: Maps any point in the lifted/rotated
         frame back to the immutable original task coordinates for outer & Blind Vault scoring.
      2. `condense_active_set(...)`: Collapses inactive auxiliary/slack coordinates after
         warmup to prevent monotonic dimension bloat.
    """

    name: str = "identity"
    warmup_grace_steps: int = 1

    def forward_project(self, x_orig: Sequence[float]) -> List[float]:
        return [float(v) for v in x_orig]

    def inverse_projection(self, u_lifted: Sequence[float]) -> List[float]:
        return [float(v) for v in u_lifted]


class EigenRotationTransform(FrameTransform):
    """
    Preconditioned Eigen-Basis Rotation (`Phi -> Phi_eigen`):
    Diagonalizes the ill-conditioned Gauss-Newton matrix `G = V Lambda V^T` (or the
    Dual Gram matrix `K = (1/M) J J^T in R^{M x M}` in O(M^2 d + M^3) when `j_mat` is
    provided with `M < d`), freezes sloppy/nullspace directions (`lambda_i < null_rel_tol * lambda_max`),
    and scales active stiff eigenvectors by `1 / sqrt(lambda_i)`.
    Turns a narrow valley with `kappa_eff > 10^8` into an isotropic sphere (`kappa = 1`),
    while `inverse_projection(y)` maps back to exact original coordinates `theta`.
    """

    name: str = "eigen_rotation"

    def __init__(
        self,
        theta_anchor: Sequence[float],
        g_mat: Optional[Sequence[Sequence[float]]] = None,
        spans: Optional[Sequence[float]] = None,
        null_rel_tol: float = 1e-12,
        j_mat: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        self.theta_anchor = [float(x) for x in theta_anchor]
        self.d = len(self.theta_anchor)
        self.spans = [float(s) for s in spans] if spans is not None else [1.0] * self.d
        self.used_dual_gram = False
        if j_mat is not None and len(j_mat) > 0 and (g_mat is None or len(g_mat) == 0 or len(j_mat) < self.d):
            evals, evecs = _dual_gram_eigh_from_jacobian(j_mat)
            self.used_dual_gram = True
        else:
            assert g_mat is not None
            evals, evecs = _jacobi_eigh_symmetric([list(r) for r in g_mat])
        lam_max = max((max(0.0, float(ev)) for ev in evals), default=1.0)
        lam_max = max(lam_max, 1e-12)

        self.active_modes: List[int] = [
            m for m, ev in enumerate(evals) if max(0.0, float(ev)) >= null_rel_tol * lam_max
        ]
        if not self.active_modes:
            self.active_modes = list(range(len(evals)))

        self.eigenvalues = [max(1e-12, float(evals[m])) for m in self.active_modes]
        self.eigenvectors = [[float(evecs[m][coord]) for coord in range(self.d)] for m in self.active_modes]
        # Metric preconditioner scales step by 1 / sqrt(lambda_m) normalized to median scale
        inv_sqrt = [1.0 / math.sqrt(ev) for ev in self.eigenvalues]
        max_scale = max(inv_sqrt) if inv_sqrt else 1.0
        self.scales = [s / max_scale for s in inv_sqrt]
        self.warmup_grace_steps = 1

    @property
    def active_dim(self) -> int:
        return len(self.active_modes)

    @property
    def preconditioned_condition_number(self) -> float:
        """Condition number of the preconditioned Gauss-Newton metric in lifted coordinates u."""
        if not self.eigenvalues:
            return 1.0
        lifted_evals = [
            ev * (sc * sc) for ev, sc in zip(self.eigenvalues, self.scales)
        ]
        return max(lifted_evals) / max(min(lifted_evals), 1e-24)

    def forward_project(self, x_orig: Sequence[float]) -> List[float]:
        dx_norm = [
            (float(x_orig[i]) - self.theta_anchor[i]) / max(1e-12, self.spans[i])
            for i in range(self.d)
        ]
        y: List[float] = []
        for m_idx, vec in enumerate(self.eigenvectors):
            proj = sum(dx_norm[i] * vec[i] for i in range(self.d))
            y.append(proj / max(1e-12, self.scales[m_idx]))
        return y

    def inverse_projection(self, u_lifted: Sequence[float]) -> List[float]:
        dx_norm = [0.0] * self.d
        for m_idx, vec in enumerate(self.eigenvectors):
            if m_idx < len(u_lifted):
                coeff = float(u_lifted[m_idx]) * self.scales[m_idx]
                for i in range(self.d):
                    dx_norm[i] += coeff * vec[i]
        return [
            self.theta_anchor[i] + dx_norm[i] * self.spans[i]
            for i in range(self.d)
        ]


class KktSlackTransform(FrameTransform):
    """
    Smooth KKT Slack-Variable Lifter with Active-Set Condensation (`Phi -> Phi_kkt`):
    Resolves non-smooth max-min tie traps `max_x min_{k in 1..K} f_k(x)` (such as
    equiareal triangle ties in Heilbronn) by lifting `x in R^d` to `u = (x, z) in R^{d + |A|}`
    where `z` are auxiliary slack targets for active constraints `A`.
    Includes `condense_active_set()` to prune inactive slack variables after 3 steps,
    preventing monotonic dimension bloat (`d -> d + K -> d + |A_tight|`).
    """

    name: str = "kkt_slack_lift"

    def __init__(
        self,
        base_dim: int,
        num_constraints: int,
        mu_penalty: float = 10.0,
        active_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.base_dim = int(base_dim)
        self.num_constraints = int(num_constraints)
        self.mu_penalty = float(mu_penalty)
        self.active_indices: List[int] = (
            [int(i) for i in active_indices]
            if active_indices is not None
            else list(range(num_constraints))
        )
        self.warmup_grace_steps = 1
        self.step_count = 0

    @property
    def lifted_dim(self) -> int:
        return self.base_dim + len(self.active_indices)

    def forward_project(
        self,
        x_orig: Sequence[float],
        initial_constraint_vals: Optional[Sequence[float]] = None,
    ) -> List[float]:
        base = [float(x_orig[i]) for i in range(self.base_dim)]
        if initial_constraint_vals is not None:
            slacks = [float(initial_constraint_vals[k]) for k in self.active_indices]
        else:
            slacks = [0.0 for _ in self.active_indices]
        return base + slacks

    def inverse_projection(self, u_lifted: Sequence[float]) -> List[float]:
        """Strictly strips auxiliary KKT slack variables and returns original x in R^d."""
        return [float(u_lifted[i]) for i in range(self.base_dim)]

    def evaluate_lifted_surrogate(
        self,
        u_lifted: Sequence[float],
        constraint_eval_fn: Callable[[Sequence[float]], Sequence[float]],
    ) -> float:
        x_orig = self.inverse_projection(u_lifted)
        f_vals = [float(v) for v in constraint_eval_fn(x_orig)]
        if not self.active_indices:
            return min(f_vals) if f_vals else 0.0
        slacks = [float(u_lifted[self.base_dim + idx]) for idx in range(len(self.active_indices))]
        z_mean = sum(slacks) / float(len(slacks))
        penalty = 0.0
        for idx, k_con in enumerate(self.active_indices):
            viol = max(0.0, slacks[idx] - f_vals[k_con])
            tie_dev = slacks[idx] - z_mean
            penalty += self.mu_penalty * (viol * viol) + 0.5 * self.mu_penalty * (tie_dev * tie_dev)
        return z_mean - penalty

    def condense_active_set(
        self,
        u_lifted: Sequence[float],
        constraint_vals: Sequence[float],
        sigma_mult: float = 1.5,
    ) -> Dict[str, Any]:
        """
        Active-Set Condensation (Unlifting): drops slack variables for constraints
        that sit comfortably above the minimum active tie manifold (`f_k - f_min > sigma_mult * std`),
        shrinking the state dimension back to the tight active tie set.
        """
        self.step_count += 1
        x_orig = self.inverse_projection(u_lifted)
        f_vals = [float(v) for v in constraint_vals]
        if not self.active_indices or len(self.active_indices) <= 1:
            return {
                "condensed_u": list(u_lifted),
                "active_slack_count": len(self.active_indices),
                "pruned_slack_count": 0,
            }

        active_f = [f_vals[k] for k in self.active_indices]
        f_min = min(active_f)
        f_mean = sum(active_f) / float(len(active_f))
        f_std = math.sqrt(
            max(1e-12, sum((v - f_mean) ** 2 for v in active_f) / float(len(active_f)))
        )
        thresh = f_min + sigma_mult * f_std

        kept_indices: List[int] = []
        kept_slacks: List[float] = []
        for pos, k_con in enumerate(self.active_indices):
            if f_vals[k_con] <= thresh + 1e-9:
                kept_indices.append(k_con)
                slack_val = (
                    float(u_lifted[self.base_dim + pos])
                    if (self.base_dim + pos) < len(u_lifted)
                    else f_vals[k_con]
                )
                kept_slacks.append(slack_val)

        if not kept_indices:
            min_k = min(self.active_indices, key=lambda k: f_vals[k])
            kept_indices = [min_k]
            kept_slacks = [f_vals[min_k]]

        pruned_count = len(self.active_indices) - len(kept_indices)
        self.active_indices = kept_indices
        condensed_u = x_orig + kept_slacks
        return {
            "condensed_u": condensed_u,
            "active_indices": list(self.active_indices),
            "active_slack_count": len(self.active_indices),
            "pruned_slack_count": pruned_count,
            "lifted_dim": self.lifted_dim,
        }


class DyadicHomotopyTransform(FrameTransform):
    """
    Multi-Scale Dyadic Coarsening & Prolongation (`Phi -> Phi_dyadic`):
    Maps a fine discretization grid of dimension `P` to a coarse grid of dimension
    `P_coarse = max(4, P // 2^level)` via block averaging, and prolongates back to
    exact dimension `P` via piecewise-linear interpolation in `inverse_projection`.
    """

    name: str = "dyadic_homotopy"

    def __init__(self, fine_dim: int, level: int = 2) -> None:
        self.fine_dim = int(fine_dim)
        self.level = max(1, int(level))
        self.coarse_dim = max(4, self.fine_dim // (2 ** self.level))
        self.warmup_grace_steps = 1

    def forward_project(self, x_orig: Sequence[float]) -> List[float]:
        n = len(x_orig)
        if n <= self.coarse_dim:
            return [float(v) for v in x_orig]
        coarse: List[float] = []
        for i in range(self.coarse_dim):
            start = int(round(i * n / float(self.coarse_dim)))
            end = max(start + 1, int(round((i + 1) * n / float(self.coarse_dim))))
            chunk = [float(x_orig[j]) for j in range(start, min(n, end))]
            coarse.append(sum(chunk) / float(len(chunk)))
        return coarse

    def inverse_projection(self, u_lifted: Sequence[float]) -> List[float]:
        c_len = len(u_lifted)
        if c_len == 0:
            return [0.0] * self.fine_dim
        if c_len == 1 or self.fine_dim == 1:
            return [float(u_lifted[0])] * self.fine_dim
        fine: List[float] = []
        for i in range(self.fine_dim):
            pos = i * (c_len - 1) / float(self.fine_dim - 1)
            idx_lo = int(math.floor(pos))
            idx_hi = min(c_len - 1, idx_lo + 1)
            frac = pos - idx_lo
            val = (1.0 - frac) * float(u_lifted[idx_lo]) + frac * float(u_lifted[idx_hi])
            fine.append(val)
        return fine


class OrientationFlipTransform(FrameTransform):
    """
    Order-Type Barrier Escape (`Phi -> Phi_flip`) for non-smooth signed minimax
    problems `max_{x in D} min_{k=1..K} |g_k(x)|` (e.g., Heilbronn triangle/square/convex
    packings and signed margin arrangements).

    Smooth optimizers and local KKT/SLP stall inside a single combinatorial sign cell
    (`order type`) because crossing `g_k(x) = 0` creates a sharp zero-value barrier.
    This operator explicitly flips the required sign `s'_k = -sign(g_k(x))` on selected
    tight constraints and solves a lazy cutting-plane Trust-Region SLP (HiGHS) with an
    UNBOUNDED slack variable `delta in (-inf, +inf)` and hard `time_limit` so the solver
    pushes `x` cleanly across `g_k(x) = 0` into the opposite orientation before free
    TR-SLP polishing.
    """

    name: str = "orientation_flip"

    def __init__(
        self,
        base_dim: int,
        initial_signs: Optional[Sequence[float]] = None,
        flipped_indices: Optional[Sequence[int]] = None,
        time_limit: float = 2.0,
    ) -> None:
        self.base_dim = int(base_dim)
        self.signs: List[float] = (
            [1.0 if float(s) >= 0.0 else -1.0 for s in initial_signs]
            if initial_signs is not None
            else []
        )
        self.flipped_indices: List[int] = (
            [int(i) for i in flipped_indices] if flipped_indices is not None else []
        )
        for idx in self.flipped_indices:
            if 0 <= idx < len(self.signs):
                self.signs[idx] = -self.signs[idx]
        self.time_limit = max(0.1, float(time_limit))
        self.warmup_grace_steps = 2

    def forward_project(self, x_orig: Sequence[float]) -> List[float]:
        return [float(x_orig[i]) for i in range(self.base_dim)]

    def inverse_projection(self, u_lifted: Sequence[float]) -> List[float]:
        return [float(u_lifted[i]) for i in range(self.base_dim)]

    @staticmethod
    def select_flip_candidates(signed_vals: Sequence[float], top_k: int = 20) -> List[int]:
        """Return indices of the `top_k` tightest `|g_k(x)|` constraints (active order-type walls)."""
        indexed = [(abs(float(v)), i) for i, v in enumerate(signed_vals)]
        indexed.sort(key=lambda t: t[0])
        return [i for _, i in indexed[: max(1, min(int(top_k), len(indexed)))]]

    def with_flipped(
        self,
        signed_vals: Sequence[float],
        flip_indices: Sequence[int],
    ) -> "OrientationFlipTransform":
        """Create a transform instance whose signs match `signed_vals` except inverted on `flip_indices`."""
        base_signs = [1.0 if float(v) >= 0.0 else -1.0 for v in signed_vals]
        return OrientationFlipTransform(
            base_dim=self.base_dim,
            initial_signs=base_signs,
            flipped_indices=flip_indices,
            time_limit=self.time_limit,
        )

    def evaluate_forced_sign_margin(self, signed_vals: Sequence[float]) -> float:
        """Signed margin `min_k (s'_k * g_k(x))`: negative before crossing, positive once crossed."""
        if len(signed_vals) == 0:
            return 0.0
        if not self.signs or len(self.signs) != len(signed_vals):
            return min(abs(float(v)) for v in signed_vals)
        return min(self.signs[i] * float(signed_vals[i]) for i in range(len(signed_vals)))

    def solve_forced_flip_slp(
        self,
        x_init: Sequence[float],
        eval_signed_fn: Callable[[Any], Any],
        grad_pool_fn: Callable[[Any, Any], Any],
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
        flip_indices: Sequence[int],
        linear_ub: Optional[Tuple[Any, Any]] = None,
        project_fn: Optional[Callable[[Any], Any]] = None,
        r0: float = 0.06,
        max_flip_steps: int = 32,
        max_polish_steps: int = 60,
        tol: float = 1e-11,
        pool_cap: int = 2200,
    ) -> Dict[str, Any]:
        """
        Lazy cutting-plane Trust-Region SLP (HiGHS) with unbounded slack `delta in (-inf, +inf)`:
          Phase 1: Forces `s_all[k] = -s_all[k]` for `k in flip_indices` and pushes `x` across `g_k(x) = 0`.
          Phase 2: Releases signs to current orientation and polishes the new cell to convergence.

        `eval_signed_fn(x)` -> 1D array of signed constraint values `g(x)` of shape `(K,)`.
        `grad_pool_fn(x, pool)` -> 2D array (or sparse matrix) of shape `(len(pool), d)` with rows `grad g_k(x)`.
        `linear_ub` -> optional `(A_lin, b_lin)` enforcing `A_lin @ x <= b_lin` (e.g., simplex `x_i + y_i <= 1`).
        """
        import numpy as np
        from scipy.optimize import linprog
        from scipy.sparse import csc_matrix, csr_matrix, hstack, issparse, vstack

        x = np.asarray(x_init, dtype=np.float64).copy()
        d = len(x)
        lb_box = np.asarray(lower_bounds, dtype=np.float64)
        ub_box = np.asarray(upper_bounds, dtype=np.float64)
        if project_fn is not None:
            x = np.asarray(project_fn(x), dtype=np.float64)
        else:
            x = np.clip(x, lb_box, ub_box)

        g0 = np.asarray(eval_signed_fn(x), dtype=np.float64)
        init_min_abs = float(np.abs(g0).min())
        s_forced = np.where(g0 >= 0.0, 1.0, -1.0)
        for idx in flip_indices:
            s_forced[int(idx)] = -s_forced[int(idx)]

        c_obj = np.zeros(d + 1, dtype=np.float64)
        c_obj[d] = -1.0

        def _solve_step(x_cur: np.ndarray, pool: np.ndarray, s_vec: np.ndarray, sc_pool: np.ndarray, r: float):
            G_pool = grad_pool_fn(x_cur, pool)
            m_act = len(pool)
            neg_s = -s_vec[pool]
            if issparse(G_pool):
                # Multiply each row by -s_k and append +1.0 column for slack delta
                G_scaled = csr_matrix(G_pool).multiply(neg_s[:, None])
                slack_col = csr_matrix(np.ones((m_act, 1), dtype=np.float64))
                A_ub = hstack([G_scaled, slack_col], format="csr")
            else:
                A_dense = np.empty((m_act, d + 1), dtype=np.float64)
                A_dense[:, :d] = neg_s[:, None] * np.asarray(G_pool, dtype=np.float64)
                A_dense[:, d] = 1.0
                A_ub = A_dense
            b_ub = sc_pool

            if linear_ub is not None:
                A_lin, b_lin = linear_ub
                b_rem = np.asarray(b_lin, dtype=np.float64) - np.asarray(A_lin @ x_cur, dtype=np.float64)
                n_lin = len(b_rem)
                if issparse(A_ub) or issparse(A_lin):
                    A_lin_s = hstack([csr_matrix(A_lin), csr_matrix((n_lin, 1))], format="csr")
                    A_ub = vstack([csr_matrix(A_ub), A_lin_s], format="csr")
                else:
                    A_lin_d = np.zeros((n_lin, d + 1), dtype=np.float64)
                    A_lin_d[:, :d] = np.asarray(A_lin, dtype=np.float64)
                    A_ub = np.vstack([A_ub, A_lin_d])
                b_ub = np.concatenate([b_ub, b_rem])

            lb = np.empty(d + 1, dtype=np.float64)
            ub = np.empty(d + 1, dtype=np.float64)
            lb[:d] = np.maximum(lb_box - x_cur, -r)
            ub[:d] = np.minimum(ub_box - x_cur, r)
            lb[d] = -np.inf
            ub[d] = np.inf
            res = linprog(
                c_obj,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=list(zip(lb, ub)),
                method="highs",
                options={"presolve": False, "time_limit": self.time_limit},
            )
            if not res.success or res.x is None:
                return None
            return res.x[:d]

        def _run_tr_loop(x_start: np.ndarray, s_vec: np.ndarray, r_init: float, max_it: int, stop_on_pos: bool):
            xc = x_start.copy()
            r = float(r_init)
            steps = 0
            K = len(s_vec)
            min_pool = min(max(40, 2 * d), K)
            for step in range(max_it):
                steps += 1
                sc_all = s_vec * np.asarray(eval_signed_fn(xc), dtype=np.float64)
                mn = float(sc_all.min())
                thresh = max(mn * 3.0, 3.0 * abs(mn), 0.002) if mn < 0.0 else 3.0 * max(mn, 1e-12)
                pool = np.flatnonzero(sc_all <= thresh)
                if len(pool) > pool_cap:
                    pool = np.argpartition(sc_all, pool_cap)[:pool_cap]
                elif len(pool) < min_pool:
                    pool = np.argpartition(sc_all, min_pool - 1)[:min_pool] if min_pool < K else np.arange(K, dtype=np.int32)

                xn = None
                for _exp in range(4):
                    dx = _solve_step(xc, pool, s_vec, sc_all[pool], r)
                    if dx is None:
                        break
                    cand = xc + dx
                    cand = np.asarray(project_fn(cand), dtype=np.float64) if project_fn is not None else np.clip(cand, lb_box, ub_box)
                    sc_new = s_vec * np.asarray(eval_signed_fn(cand), dtype=np.float64)
                    mn_new = float(sc_new.min())
                    t_new = max(mn_new * 2.0, 2.0 * abs(mn_new), 0.001) if mn_new < 0.0 else 2.0 * max(mn_new, 1e-12)
                    newpool = np.flatnonzero(sc_new <= t_new)
                    if len(newpool) > pool_cap // 2:
                        newpool = np.argpartition(sc_new, pool_cap // 2)[: pool_cap // 2]
                    xn = cand
                    if len(newpool) == 0 or np.isin(newpool, pool).all():
                        break
                    pool = np.union1d(pool, newpool)
                    if len(pool) > int(pool_cap * 1.3):
                        break

                if xn is None:
                    r *= 0.5
                    if r < tol:
                        break
                    continue

                v_new = float((s_vec * np.asarray(eval_signed_fn(xn), dtype=np.float64)).min())
                if v_new > mn + 1e-14:
                    xc = xn
                    r = min(r * 1.5, 0.25)
                    if stop_on_pos and v_new > 0.0 and step >= 8:
                        break
                else:
                    r *= 0.5
                    if r < tol:
                        break
            return xc, steps

        # Phase 1: push flipped constraints across 0 into opposite orientation
        x_p1, steps1 = _run_tr_loop(x, s_forced, r0, max_flip_steps, stop_on_pos=True)
        g_p1 = np.asarray(eval_signed_fn(x_p1), dtype=np.float64)
        crossed = bool(all(s_forced[int(idx)] * g_p1[int(idx)] > 0.0 for idx in flip_indices))

        # Phase 2: release signs to current cell and polish freely
        s_free = np.where(g_p1 >= 0.0, 1.0, -1.0)
        x_final, steps2 = _run_tr_loop(x_p1, s_free, min(r0, 0.04), max_polish_steps, stop_on_pos=False)
        g_final = np.asarray(eval_signed_fn(x_final), dtype=np.float64)
        final_min_abs = float(np.abs(g_final).min())

        return {
            "x": x_final.tolist(),
            "min_abs": final_min_abs,
            "initial_min_abs": init_min_abs,
            "crossed": crossed,
            "improved": bool(final_min_abs > init_min_abs + 1e-12),
            "steps": steps1 + steps2,
        }


class SpectralMetaAtlas:
    """
    Cross-Task Spectral Operator Atlas (`meta_operator_atlas`):
    Indexes verified structural transformations (`eigen_rotation`, `kkt_slack_lift`,
    `orientation_flip`, `dyadic_homotopy`) by dimensionless stall signatures and task
    geometry, with automatic provenance quarantine if a transferred operator fails to
    produce positive gain within `max_trial_probes = 2` evaluations.
    """

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = [
            {
                "operator": "kkt_slack_lift",
                "task_family": "geometric_extremal",
                "min_log10_kappa": 6.0,
                "requires_min_tie": True,
                "description": "Lifts non-smooth min-tie bottleneck into smooth KKT slack variables with active-set condensation.",
            },
            {
                "operator": "orientation_flip",
                "task_family": "geometric_extremal",
                "requires_signed_barrier": True,
                "description": "Flips sign of tightest near-collinear constraints and solves unbounded-slack TR-SLP to escape order-type walls.",
            },
            {
                "operator": "dyadic_homotopy",
                "task_family": "fourier_spectral",
                "min_dim": 64,
                "description": "Coarsens high-resolution grid by 2^k and prolongates back via linear spline homotopy.",
            },
            {
                "operator": "eigen_rotation",
                "task_family": "general",
                "min_log10_kappa": 6.0,
                "description": "Preconditioned rotation into Gauss-Newton eigenvectors V with nullspace freezing.",
            },
        ]
        self.trial_counts: Dict[Tuple[str, str], int] = {}
        self.positive_credits: Dict[Tuple[str, str], int] = {}
        self.quarantined: Set[Tuple[str, str]] = set()

    def record_outcome(self, task_name: str, operator_name: str, gained_score: bool) -> bool:
        key = (task_name, operator_name)
        self.trial_counts[key] = self.trial_counts.get(key, 0) + 1
        if gained_score:
            self.positive_credits[key] = self.positive_credits.get(key, 0) + 1
        elif self.trial_counts[key] >= 2 and self.positive_credits.get(key, 0) == 0:
            self.quarantined.add(key)
        return key not in self.quarantined

    def select_operator(
        self,
        task_name: str,
        diagnostics: Dict[str, Any],
        has_min_tie_structure: bool = False,
        consecutive_oos_failures: int = 0,
        has_signed_barrier: bool = False,
    ) -> Optional[str]:
        log10_k = float(diagnostics.get("log10_kappa_eff", 0.0))
        d = int(diagnostics.get("d", 0))
        stalled = bool(log10_k >= 6.0 or consecutive_oos_failures >= 4)
        if not stalled:
            return None

        candidates: List[str] = []
        if has_min_tie_structure:
            candidates.append("kkt_slack_lift")
        if has_signed_barrier:
            candidates.append("orientation_flip")
        if d >= 64:
            candidates.append("dyadic_homotopy")
        candidates.append("eigen_rotation")

        for op in candidates:
            if (task_name, op) not in self.quarantined:
                return op
        return None


class SpectralFrameLifter:
    """
    Stage 2 Coordinator: inspects Gauss-Newton matrix `G = (1/M) J^T J` and DAG OOS
    trajectory, queries `SpectralMetaAtlas`, and instantiates the matching `FrameTransform`
    with a mandatory inverse projection `Phi^-1` and warmup grace budget.
    """

    def __init__(self, atlas: Optional[SpectralMetaAtlas] = None) -> None:
        self.atlas = atlas or SpectralMetaAtlas()

    def detect_and_lift(
        self,
        task_name: str,
        theta_anchor: Sequence[float],
        g_mat: Optional[Sequence[Sequence[float]]] = None,
        m_buckets: Optional[int] = None,
        spans: Optional[Sequence[float]] = None,
        train_score: float = 0.0,
        oos_score: float = 0.0,
        pass_mask: Optional[Sequence[int]] = None,
        has_min_tie_structure: bool = False,
        num_tie_constraints: int = 0,
        consecutive_oos_failures: int = 0,
        j_mat: Optional[Sequence[Sequence[float]]] = None,
        has_signed_barrier: bool = False,
    ) -> Dict[str, Any]:
        diag = compute_spectral_diagnostics(
            g_mat=g_mat,
            m_buckets=m_buckets,
            train_score=train_score,
            oos_score=oos_score,
            pass_mask=pass_mask,
            j_mat=j_mat,
        )
        op_name = self.atlas.select_operator(
            task_name=task_name,
            diagnostics=diag,
            has_min_tie_structure=has_min_tie_structure,
            consecutive_oos_failures=consecutive_oos_failures,
            has_signed_barrier=has_signed_barrier,
        )
        if op_name is None:
            return {
                "triggered": False,
                "operator": None,
                "diagnostics": diag,
                "transform": FrameTransform(),
                "warmup_grace_steps": 0,
            }

        if op_name == "kkt_slack_lift" and num_tie_constraints > 0:
            transform: FrameTransform = KktSlackTransform(
                base_dim=len(theta_anchor),
                num_constraints=num_tie_constraints,
            )
        elif op_name == "orientation_flip":
            transform = OrientationFlipTransform(base_dim=len(theta_anchor))
        elif op_name == "dyadic_homotopy":
            transform = DyadicHomotopyTransform(fine_dim=len(theta_anchor), level=2)
        else:
            transform = EigenRotationTransform(
                theta_anchor=theta_anchor,
                g_mat=g_mat,
                spans=spans,
                j_mat=j_mat,
            )

        return {
            "triggered": True,
            "operator": op_name,
            "diagnostics": diag,
            "transform": transform,
            "warmup_grace_steps": transform.warmup_grace_steps,
        }

__all__ = [
    "prune_epistatic_gauss_newton",
    "compute_spectral_diagnostics",
    "FrameTransform",
    "EigenRotationTransform",
    "KktSlackTransform",
    "DyadicHomotopyTransform",
    "OrientationFlipTransform",
    "SpectralMetaAtlas",
    "SpectralFrameLifter",
]

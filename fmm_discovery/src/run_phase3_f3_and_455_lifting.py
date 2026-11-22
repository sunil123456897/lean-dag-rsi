"""
Phase 3 (Final ~14-min CPU-Only Assault — ZERO GPU usage)
Two Coordinated High-Leverage Tracks:
  Track A (4x5x5 : R = 75 Knockout Push):
    - Start from Phase 2's best 4x5x5:R=75 basin structure (`supp=7, unres=7` -> only 7 parity
      contradictions out of 10,000 Brent equations, -1 below AlphaTensor char-0 SOTA 76!).
    - Run 12T Numba deep multi-`plus` tunneling (`75->76->75` AND `75->77->75` + disjoint splits)
      + GL(F2) shears + direct `unres_coker_4` SA strictly preserving `R == 75`.
    - Run violated-equation diagnosis (exact cokernel rows P @ e1 == 1) + targeted violated-term
      multi-`plus` sniper + exact Z3 5.1.0 freeze-and-repair around the ~7 violated equations.
  Track B (3-adic Lifting & F_3 `coker_viol_9`-Guided Repair of GPU `f3_pool_phase3.json`):
    - Flexibly ingest `@fmm-architect`'s `artifacts/f3_pool_phase3.json` (2-bitplane u64 OR dense U/V/W).
    - Screen EVERY F_3 scheme through Stage L0-p3 (`mod9_oracle_batch_u64`, support + unrestricted).
    - Run 12T Numba F_3 `coker_viol_9`-guided flip-graph SA (`F_3` flips + `plus`/`reduce` + `GL(F_3)`)
      on obstructed low-rank / record-beating hits.
    - Fire Stage L1-p3 (`linear_hensel_lift_p3`, up to 3^40 + Wang rational reconstruction -> Z[1/2]/Q/Z/Z_T)
      on ALL `coker_viol_9 == 0` hits.
    - Certify every lifted char-0 scheme via `brent_verifier.verify_scheme()` + 100x int + 20x M_2(Z)
      non-commutative check, saving any 2026 world record to `artifacts/lifted_record_<m>x<n>x<p>_R<R>_<ring>.json`.

Hardware: Pure CPU 12T (Numba LLVM + NumPy + Z3 5.1.0). ZERO torch / ZERO CUDA / ZERO GPU usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numba as nb
import numpy as np
import z3

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_FMM_ROOT = os.path.dirname(_SRC_DIR)
_REPO_ROOT = os.path.dirname(_FMM_ROOT)
for _p in (_SRC_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from brent_verifier import BilinearScheme, verify_scheme  # noqa: E402
from hensel_lifter import (  # noqa: E402
    _build_jacobian_and_rref_nb,
    _f3_add_u64,
    _mod4_oracle_support_nb,
    _mod4_oracle_unrestricted_nb,
    _mod9_oracle_nb,
    _popcount64,
    fast_verify_f2_u64,
    linear_hensel_lift_p3,
    mod9_oracle_u64,
    scheme_to_u64_f2,
    scheme_to_u64_f3,
)
from run_phase1_lifting import (  # noqa: E402
    _apply_plus_nb,
    _check_local_reduce_nb,
    _group_lin_reduce_nb,
    _rebuild_edges_nb,
    _rng_next,
    _scheme_nnz_nb,
    _scheme_orbit_hash_nb,
    _snipe_1step_fast_nb,
    _targeted_expand_nb,
    _update_edges_after_flip_nb,
)
from run_phase2_lifting_assault import (  # noqa: E402
    _apply_gl_shear_nb,
    _verify_noncommutative_and_random_int,
    attempt_certified_lift_and_save,
    gl_polish_strict,
    make_plus_relaxed_seeds,
)


# Verified November 2026 characteristic-0 world records (from artifacts/sota_targets.json)
SOTA_CHAR0_2026: Dict[Tuple[int, int, int], Dict[str, int]] = {
    (2, 2, 2): {"F2": 7, "ZT": 7, "Z": 7, "Q": 7},
    (3, 3, 3): {"F2": 23, "ZT": 23, "Z": 23, "Q": 23},
    (3, 3, 6): {"F2": 42, "ZT": 42, "Z": 42, "Q": 40},
    (4, 4, 4): {"F2": 47, "ZT": 49, "Z": 49, "Q": 48},
    (4, 4, 5): {"F2": 60, "ZT": 61, "Z": 61, "Q": 61},
    (4, 5, 5): {"F2": 73, "ZT": 76, "Z": 76, "Q": 76},
    (4, 5, 6): {"F2": 89, "ZT": 90, "Z": 90, "Q": 90},
    (4, 4, 8): {"F2": 94, "ZT": 96, "Z": 96, "Q": 95},
    (4, 4, 9): {"F2": 107, "ZT": 107, "Z": 107, "Q": 104},
    (5, 5, 5): {"F2": 93, "ZT": 96, "Z": 96, "Q": 94},
    (5, 5, 6): {"F2": 112, "ZT": 116, "Z": 116, "Q": 116},
    (6, 6, 6): {"F2": 153, "ZT": 153, "Z": 153, "Q": 153},
    (6, 6, 7): {"F2": 183, "ZT": 183, "Z": 183, "Q": 183},
    (5, 7, 8): {"F2": 204, "ZT": 204, "Z": 204, "Q": 204},
    (5, 8, 8): {"F2": 230, "ZT": 230, "Z": 230, "Q": 230},
    (7, 7, 7): {"F2": 245, "ZT": 250, "Z": 250, "Q": 249},
    (7, 7, 8): {"F2": 275, "ZT": 278, "Z": 278, "Q": 277},
}


# =============================================================================
# 1. Track A — Deep Multi-Plus Tunneling & Violated-Equation Sniper for 4x5x5:75
# =============================================================================


@nb.njit(cache=True)
def _disjoint_split_nb(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    R: int,
    state: np.uint64,
) -> Tuple[int, np.uint64]:
    """Split a high-weight row along a random factor with popcount >= 2 into two disjoint halves."""
    best_r = -1
    best_ax = -1
    best_w = 1
    for _ in range(10):
        state = _rng_next(state)
        r = np.int64(state % np.uint64(R))
        state = _rng_next(state)
        ax = np.int64(state % np.uint64(3))
        mask = u[r] if ax == 0 else (v[r] if ax == 1 else w[r])
        pc = np.int64(_popcount64(mask))
        if pc > best_w:
            best_w = pc
            best_r = r
            best_ax = ax
    if best_r < 0:
        return _targeted_expand_nb(u, v, w, R, state)

    mask = u[best_r] if best_ax == 0 else (v[best_r] if best_ax == 1 else w[best_r])
    half = np.uint64(0)
    taken = 0
    target_take = best_w // 2
    tmp = mask
    while tmp != 0 and taken < target_take:
        lsb = tmp & (~tmp + np.uint64(1))
        state = _rng_next(state)
        if (state & np.uint64(1)) != 0 or np.int64(_popcount64(tmp)) <= (target_take - taken):
            half |= lsb
            taken += 1
        tmp ^= lsb
    if half == 0 or half == mask:
        half = mask & (~mask + np.uint64(1))

    u[R] = u[best_r]
    v[R] = v[best_r]
    w[R] = w[best_r]
    if best_ax == 0:
        u[best_r] = half
        u[R] = mask ^ half
    elif best_ax == 1:
        v[best_r] = half
        v[R] = mask ^ half
    else:
        w[best_r] = half
        w[R] = mask ^ half
    return R + 1, state


@nb.njit(parallel=True, cache=True)
def _tunnel_455_sa_nb(
    m: int,
    n: int,
    p: int,
    U_in: np.ndarray,
    V_in: np.ndarray,
    W_in: np.ndarray,
    R_in: np.ndarray,
    R_target: int,
    n_iters: int,
    use_unres: bool,
    hot_terms: np.ndarray,
    seed_base: np.uint64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    12-thread multi-plus tunneling SA for 4x5x5:R=75 (strictly preserving R == R_target):
      - Walkers 0..7  (Tier 0): Cold exploiters (1-2 flips, GL shears, 1-plus bridges 75->76->75)
      - Walkers 8..15 (Tier 1): Balanced 1-plus & 2-plus tunnelers (75->77->75 + disjoint splits)
      - Walkers 16..23(Tier 2): Hot non-local 2-plus & 3-plus tunnelers + violated-term focused plus!
    """
    nw = U_in.shape[0]
    R_cap = U_in.shape[1]
    n_hot = hot_terms.shape[0]

    U_b = U_in.copy()
    V_b = V_in.copy()
    W_b = W_in.copy()
    R_b = R_in.copy()
    ck_b = np.full(nw, 999999, dtype=np.int32)
    evals = np.zeros(nw, dtype=np.int64)

    for wid in nb.prange(nw):
        u_cur = U_in[wid].copy()
        v_cur = V_in[wid].copy()
        w_cur = W_in[wid].copy()
        R_cur = int(R_in[wid])

        if use_unres:
            c0 = _mod4_oracle_unrestricted_nb(m, n, p, u_cur[:R_cur], v_cur[:R_cur], w_cur[:R_cur])[0]
        else:
            c0 = _mod4_oracle_support_nb(m, n, p, u_cur[:R_cur], v_cur[:R_cur], w_cur[:R_cur])[0]
        cur_ck = int(c0) if c0 >= 0 else 999999
        cur_nnz = _scheme_nnz_nb(u_cur, v_cur, w_cur, R_cur)

        best_ck = cur_ck
        best_R = R_cur
        best_nnz = cur_nnz
        ck_b[wid] = best_ck
        R_b[wid] = best_R

        state = seed_base ^ (np.uint64(wid + 53) * np.uint64(0x9E3779B97F4A7C15))
        if state == 0:
            state = np.uint64(19)

        u_try = np.empty(R_cap, dtype=np.uint64)
        v_try = np.empty(R_cap, dtype=np.uint64)
        w_try = np.empty(R_cap, dtype=np.uint64)
        ej_i = np.empty(4096, dtype=np.int32)
        ej_j = np.empty(4096, dtype=np.int32)
        ej_m = np.empty(4096, dtype=np.int8)
        deg = np.zeros((3, R_cap), dtype=np.int32)
        nbr = np.empty((3, R_cap, 16), dtype=np.int32)

        tier = 0 if wid < 8 else (1 if wid < 16 else 2)
        t_hi = 1.0 if tier == 0 else (4.0 if tier == 1 else 10.0)
        t_lo = 0.08 if tier == 0 else (0.22 if tier == 1 else 0.55)

        ev = 1
        for it in range(n_iters):
            if best_ck == 0:
                break

            for k in range(R_cur):
                u_try[k] = u_cur[k]
                v_try[k] = v_cur[k]
                w_try[k] = w_cur[k]
            R_try = R_cur
            ne = _rebuild_edges_nb(u_try, v_try, w_try, R_try, ej_i, ej_j, ej_m)

            # Choose move family: GL shear (supp only), 2-plus tunnel, 1-plus tunnel, or local flips
            did_gl = False
            if (not use_unres) and (it % 8 == 7):
                state = _rng_next(state)
                g_id = np.int64(state % np.uint64(3))
                dim_g = m if g_id == 0 else (n if g_id == 1 else p)
                state = _rng_next(state)
                r1 = np.int64(state % np.uint64(dim_g))
                state = _rng_next(state)
                r2 = np.int64(state % np.uint64(dim_g - 1))
                if r2 >= r1:
                    r2 += np.int64(1)
                ug, vg, wg = _apply_gl_shear_nb(m, n, p, u_try, v_try, w_try, R_try, g_id, r1, r2)
                for k in range(R_try):
                    u_try[k] = ug[k]
                    v_try[k] = vg[k]
                    w_try[k] = wg[k]
                did_gl = True

            if not did_gl:
                n_expand = 0
                if tier == 0:
                    if ne == 0 or (it % 4 == 3):
                        n_expand = 1
                elif tier == 1:
                    if it % 5 == 4:
                        n_expand = 2
                    elif ne == 0 or (it % 3 == 2):
                        n_expand = 1
                else:
                    if it % 3 == 2:
                        n_expand = 2
                    elif it % 7 == 6:
                        n_expand = 3
                    else:
                        n_expand = 1

                for ex_i in range(n_expand):
                    if R_try + 1 >= R_cap:
                        break
                    state = _rng_next(state)
                    mode_ex = np.int64(state % np.uint64(5))
                    if mode_ex == 0 and n_hot >= 2 and R_try >= R_target:
                        # Violated-term focused plus(i, j)!
                        state = _rng_next(state)
                        hi = int(hot_terms[np.int64(state % np.uint64(n_hot))])
                        state = _rng_next(state)
                        if (state & np.uint64(1)) == 0:
                            hj = int(hot_terms[np.int64((state >> np.uint64(1)) % np.uint64(n_hot))])
                        else:
                            hj = int(np.int64((state >> np.uint64(1)) % np.uint64(R_try)))
                        if hi < R_try and hj < R_try and hi != hj:
                            state = _rng_next(state)
                            R_try = _apply_plus_nb(u_try, v_try, w_try, R_try, hi, hj, np.int64(state % np.uint64(6)))
                        else:
                            R_try, state = _targeted_expand_nb(u_try, v_try, w_try, R_try, state)
                    elif mode_ex == 1:
                        R_try, state = _disjoint_split_nb(u_try, v_try, w_try, R_try, state)
                    else:
                        R_try, state = _targeted_expand_nb(u_try, v_try, w_try, R_try, state)

                if n_expand > 0:
                    ne = _rebuild_edges_nb(u_try, v_try, w_try, R_try, ej_i, ej_j, ej_m)

                state = _rng_next(state)
                if n_expand == 0:
                    nfl = np.int64(1) if (tier == 0 and (state & np.uint64(3)) != 0) else (1 + np.int64(state % np.uint64(6)))
                elif n_expand == 1:
                    nfl = 10 + np.int64(state % np.uint64(36))
                else:
                    nfl = 24 + np.int64(state % np.uint64(56))

                for fl in range(nfl):
                    if ne == 0:
                        break
                    found = False
                    si = np.int64(0)
                    sj = np.int64(0)
                    sm = np.int64(0)
                    if n_expand > 0 and R_try > R_target and fl >= 4 and (fl & 1) == 0:
                        found, si, sj, sm = _snipe_1step_fast_nb(
                            u_try, v_try, w_try, R_try, ej_i, ej_j, ej_m, ne, deg, nbr
                        )
                    if not found:
                        state = _rng_next(state)
                        ek = np.int64(state % np.uint64(ne))
                        si = np.int64(ej_i[ek])
                        sj = np.int64(ej_j[ek])
                        sm = np.int64(ej_m[ek])
                        state = _rng_next(state)
                        if (state & np.uint64(1)) != 0:
                            tmp = si
                            si = sj
                            sj = tmp
                    if sm == 0:
                        v_try[si] ^= v_try[sj]
                        w_try[sj] ^= w_try[si]
                    elif sm == 1:
                        u_try[si] ^= u_try[sj]
                        w_try[sj] ^= w_try[si]
                    else:
                        u_try[si] ^= u_try[sj]
                        v_try[sj] ^= v_try[si]
                    old_Rt = R_try
                    R_try = _check_local_reduce_nb(u_try, v_try, w_try, R_try, si, sj)
                    if R_try < old_Rt:
                        R_try = _group_lin_reduce_nb(u_try, v_try, w_try, R_try)
                        ne = _rebuild_edges_nb(u_try, v_try, w_try, R_try, ej_i, ej_j, ej_m)
                        if n_expand > 0 and R_try <= R_target:
                            break
                    else:
                        ne = _update_edges_after_flip_nb(u_try, v_try, w_try, R_try, sm, si, sj, ej_i, ej_j, ej_m, ne)

            if R_try > R_target:
                continue
            # Strictly preserve R == R_target (never collapse 75 -> 73!)
            while R_try < R_target and R_try + 1 < R_cap:
                R_try, state = _targeted_expand_nb(u_try, v_try, w_try, R_try, state)
                if R_try > R_target:
                    break
            if R_try != R_target:
                continue

            if use_unres:
                ck = _mod4_oracle_unrestricted_nb(m, n, p, u_try[:R_try], v_try[:R_try], w_try[:R_try])[0]
            else:
                ck = _mod4_oracle_support_nb(m, n, p, u_try[:R_try], v_try[:R_try], w_try[:R_try])[0]
            ev += 1
            if ck < 0:
                continue

            nnz_t = _scheme_nnz_nb(u_try, v_try, w_try, R_try)
            frac = float(it) / float(max(1, n_iters))
            temp = t_lo + (t_hi - t_lo) * (1.0 - frac)
            accept = False
            if ck < cur_ck:
                accept = True
            elif ck == cur_ck:
                if nnz_t <= cur_nnz + 24:
                    accept = True
            else:
                diff = float(ck - cur_ck)
                if diff < 3.5 * temp:
                    state = _rng_next(state)
                    u01 = float(state & np.uint64(0xFFFFFF)) / float(0x1000000)
                    if u01 < np.exp(-diff / temp):
                        accept = True

            if accept:
                R_cur = R_try
                cur_ck = int(ck)
                cur_nnz = nnz_t
                for k in range(R_cur):
                    u_cur[k] = u_try[k]
                    v_cur[k] = v_try[k]
                    w_cur[k] = w_try[k]

                if (cur_ck < best_ck) or (cur_ck == best_ck and cur_nnz < best_nnz):
                    best_ck = cur_ck
                    best_R = R_cur
                    best_nnz = cur_nnz
                    for k in range(best_R):
                        U_b[wid, k] = u_cur[k]
                        V_b[wid, k] = v_cur[k]
                        W_b[wid, k] = w_cur[k]
                    R_b[wid] = best_R
                    ck_b[wid] = best_ck

        evals[wid] = ev

    return U_b, V_b, W_b, R_b, ck_b, evals


def diagnose_violated_equations_and_terms(
    m: int,
    n: int,
    p: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
) -> Dict[str, Any]:
    """
    Build exact GF(2) unrestricted Jacobian RREF (`_build_f2_jacobian_rref_nb`),
    extract EVERY violated cokernel row (`P_mat[r_i] @ e1 == 1`), identify the exact
    Brent triples participating in those violated cokernel combinations, and rank all R terms
    by violation-touch frequency. Also constructs a partial ternary sign assignment satisfying
    the `rank_J1` consistent equations and counts exact integer Brent violations out of m^2 n^2 p^2.
    """
    res = _build_jacobian_and_rref_nb(m, n, p, u, v, w, False)
    if res[0] < 0:
        raise ValueError("Invalid F2 scheme in diagnosis!")
    (
        _,
        num_vars,
        num_active,
        rank_J1,
        nnz_total,
        gauge_fixed,
        eq_triples,
        var_u,
        var_v,
        var_w,
        pivot_cols,
        free_cols,
        rref_mat,
        P_mat,
        e1_vec,
    ) = res
    R = len(u)
    mn, np_dim, mp = m * n, n * p, m * p
    n_eq_words = P_mat.shape[1]

    e1_words = np.zeros(n_eq_words, dtype=np.uint64)
    for idx in range(num_active):
        if e1_vec[idx] != 0:
            e1_words[idx >> 6] |= np.uint64(1) << np.uint64(idx & 63)

    viol_coker_rows: List[int] = []
    viol_eq_counts: Dict[int, int] = {}
    term_touch = np.zeros(R, dtype=np.int32)

    for r_i in range(rank_J1, num_active):
        parity = 0
        for wi in range(n_eq_words):
            parity ^= int(_popcount64(P_mat[r_i, wi] & e1_words[wi])) & 1
        if parity == 1:
            viol_coker_rows.append(r_i)
            # Extract equations participating in this violated cokernel dependency (and nonzero in e1)
            for wi in range(n_eq_words):
                bits = int(P_mat[r_i, wi] & e1_words[wi])
                while bits:
                    lsb = bits & -bits
                    eq_idx = (wi << 6) + (lsb.bit_length() - 1)
                    if eq_idx < num_active:
                        viol_eq_counts[eq_idx] = viol_eq_counts.get(eq_idx, 0) + 1
                    bits ^= lsb

    # Map violated active equations back to (a, b, c) and touching terms
    viol_triples: List[Tuple[int, int, int]] = []
    for eq_idx in sorted(viol_eq_counts.keys()):
        flat = int(eq_triples[eq_idx])
        c = flat % mp
        tmp = flat // mp
        b = tmp % np_dim
        a = tmp // np_dim
        viol_triples.append((a, b, c))
        am = np.uint64(1) << np.uint64(a)
        bm = np.uint64(1) << np.uint64(b)
        cm = np.uint64(1) << np.uint64(c)
        wt = viol_eq_counts[eq_idx]
        for r in range(R):
            if ((u[r] & am) != 0 and (v[r] & bm) != 0) or ((u[r] & am) != 0 and (w[r] & cm) != 0) or (
                (v[r] & bm) != 0 and (w[r] & cm) != 0
            ):
                term_touch[r] += wt

    # Also build a partial support-restricted ternary sign assignment via pivot rows of support RREF
    res_s = _build_jacobian_and_rref_nb(m, n, p, u, v, w, True)
    (_, nv_s, na_s, rk_s, _, _, _, vu_s, vv_s, vw_s, piv_s, free_s, rref_s, P_s, e1_s) = res_s
    # Compute x_s ignoring cokernel rows (exact solution on all rank_J1 pivot equations!)
    n_eq_ws = P_s.shape[1]
    e1_ws = np.zeros(n_eq_ws, dtype=np.uint64)
    for idx in range(na_s):
        if e1_s[idx] != 0:
            e1_ws[idx >> 6] |= np.uint64(1) << np.uint64(idx & 63)
    x_s = np.zeros(nv_s, dtype=np.uint8)
    for r_i in range(rk_s):
        rhs_b = 0
        for wi in range(n_eq_ws):
            rhs_b ^= int(_popcount64(P_s[r_i, wi] & e1_ws[wi])) & 1
        x_s[piv_s[r_i]] = rhs_b

    # Convert to ternary U_z, V_z, W_z (bit=1 -> -1, bit=0 -> +1 on support)
    U_z = np.zeros((R, mn), dtype=np.int64)
    V_z = np.zeros((R, np_dim), dtype=np.int64)
    W_z = np.zeros((R, mp), dtype=np.int64)
    for r in range(R):
        for a in range(mn):
            if (int(u[r]) >> a) & 1:
                vid = int(vu_s[r, a])
                U_z[r, a] = -1 if (vid >= 0 and x_s[vid] == 1) else 1
        for b in range(np_dim):
            if (int(v[r]) >> b) & 1:
                vid = int(vv_s[r, b])
                V_z[r, b] = -1 if (vid >= 0 and x_s[vid] == 1) else 1
        for c in range(mp):
            if (int(w[r]) >> c) & 1:
                vid = int(vw_s[r, c])
                W_z[r, c] = -1 if (vid >= 0 and x_s[vid] == 1) else 1

    # Compute exact integer Brent residual tensor E[a, b, c]
    T_got = np.einsum("ra,rb,rc->abc", U_z, V_z, W_z)
    T_ex = np.zeros((mn, np_dim, mp), dtype=np.int64)
    for i1 in range(m):
        for i2 in range(n):
            for j2 in range(p):
                T_ex[i1 * n + i2, i2 * p + j2, i1 * p + j2] = 1
    E_res = T_ex - T_got
    nz_coords = np.argwhere(E_res != 0)
    int_viol_count = int(len(nz_coords))

    # Score terms by participation in integer residual equations too
    int_term_touch = np.zeros(R, dtype=np.int32)
    for a, b, c in nz_coords:
        ai, bi, ci = int(a), int(b), int(c)
        for r in range(R):
            if U_z[r, ai] != 0 and V_z[r, bi] != 0 and W_z[r, ci] != 0:
                int_term_touch[r] += 3
            elif (U_z[r, ai] != 0 and V_z[r, bi] != 0) or (U_z[r, ai] != 0 and W_z[r, ci] != 0) or (
                V_z[r, bi] != 0 and W_z[r, ci] != 0
            ):
                int_term_touch[r] += 1

    hot_order = np.argsort(-(term_touch * 2 + int_term_touch)).astype(np.int32)
    return {
        "unres_coker_4": len(viol_coker_rows),
        "supp_coker_4": int(_mod4_oracle_support_nb(m, n, p, u, v, w)[0]),
        "num_viol_e1_triples": len(viol_triples),
        "partial_sign_int_violations": int_viol_count,
        "partial_sign_max_abs_err": int(np.max(np.abs(E_res))) if int_viol_count > 0 else 0,
        "hot_terms": hot_order,
        "U_z": U_z,
        "V_z": V_z,
        "W_z": W_z,
        "E_res": E_res,
    }


def attempt_z3_local_freeze_repair(
    m: int,
    n: int,
    p: int,
    U_z: np.ndarray,
    V_z: np.ndarray,
    W_z: np.ndarray,
    hot_terms: np.ndarray,
    k_free: int = 8,
    timeout_s: float = 20.0,
) -> Dict[str, Any]:
    """
    Z3 5.1.0 Local Freeze-and-Repair around the ~7 violated Brent equations:
      - Freeze the (R - k_free) cleanest ternary terms in U_z, V_z, W_z.
      - Let the `k_free` hottest terms vary freely over {-1, 0, +1} (allowing BOTH sign changes
        AND sparse off-support edits on those `k_free` terms!) ONLY on the active coordinates
        (a, b, c) where the frozen residual tensor is non-zero or where those k_free terms sit!
      - Check exact integer Brent equations for ALL (a, b, c) in active slice!
    """
    t0 = time.perf_counter()
    R = U_z.shape[0]
    mn, np_dim, mp = m * n, n * p, m * p
    free_set = [int(x) for x in hot_terms[:k_free]]
    fixed_set = [r for r in range(R) if r not in set(free_set)]

    # Residual tensor that the k_free terms MUST match identically over Z!
    T_ex = np.zeros((mn, np_dim, mp), dtype=np.int64)
    for i1 in range(m):
        for i2 in range(n):
            for j2 in range(p):
                T_ex[i1 * n + i2, i2 * p + j2, i1 * p + j2] = 1
    T_fixed = np.einsum("ra,rb,rc->abc", U_z[fixed_set], V_z[fixed_set], W_z[fixed_set])
    T_need = T_ex - T_fixed

    # Restrict each free term's allowed coordinates to its current support union the residual's active axes
    nz_need = np.argwhere(T_need != 0)
    if len(nz_need) == 0:
        return {"status": "already_zero", "elapsed_ms": 0.0, "scheme": None}

    need_a = set(int(x) for x in nz_need[:, 0])
    need_b = set(int(x) for x in nz_need[:, 1])
    need_c = set(int(x) for x in nz_need[:, 2])

    # Check if residual is compact enough for Z3 (avoid blowing up on 10,000 dense cubic constraints)
    if len(nz_need) > 650:
        return {
            "status": "skipped_residual_too_dense",
            "k_free": k_free,
            "residual_nnz": int(len(nz_need)),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "scheme": None,
        }

    # Build Z3 ternary variables (restricted to each term's support + small off-support halo on residual coords)
    solver = z3.Solver()
    solver.set("timeout", int(timeout_s * 1000))

    u_vars: Dict[Tuple[int, int], Any] = {}
    v_vars: Dict[Tuple[int, int], Any] = {}
    w_vars: Dict[Tuple[int, int], Any] = {}
    term_as: List[List[int]] = []
    term_bs: List[List[int]] = []
    term_cs: List[List[int]] = []

    for idx_f, r in enumerate(free_set):
        cur_a = {a for a in range(mn) if U_z[r, a] != 0}
        cur_b = {b for b in range(np_dim) if V_z[r, b] != 0}
        cur_c = {c for c in range(mp) if W_z[r, c] != 0}
        # Allow existing support + up to 2 residual coords if small
        al = sorted(cur_a | (need_a if len(need_a) <= 10 else set()))
        bl = sorted(cur_b | (need_b if len(need_b) <= 10 else set()))
        cl = sorted(cur_c | (need_c if len(need_c) <= 10 else set()))
        term_as.append(al)
        term_bs.append(bl)
        term_cs.append(cl)
        for a in al:
            var = z3.Int(f"fu_{idx_f}_{a}")
            solver.add(var >= -1, var <= 1)
            u_vars[(idx_f, a)] = var
        for b in bl:
            var = z3.Int(f"fv_{idx_f}_{b}")
            solver.add(var >= -1, var <= 1)
            v_vars[(idx_f, b)] = var
        for c in cl:
            var = z3.Int(f"fw_{idx_f}_{c}")
            solver.add(var >= -1, var <= 1)
            w_vars[(idx_f, c)] = var

    # Collect all triples touched by T_need OR by any free term's allowed box
    active_triples: set[Tuple[int, int, int]] = set((int(x[0]), int(x[1]), int(x[2])) for x in nz_need)
    for idx_f in range(k_free):
        for a in term_as[idx_f]:
            for b in term_bs[idx_f]:
                for c in term_cs[idx_f]:
                    active_triples.add((a, b, c))

    if len(active_triples) > 3500:
        return {
            "status": "skipped_box_too_large",
            "k_free": k_free,
            "residual_nnz": int(len(nz_need)),
            "active_triples": len(active_triples),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "scheme": None,
        }

    for a, b, c in active_triples:
        tgt = int(T_need[a, b, c])
        terms_abc = []
        for idx_f in range(k_free):
            uv = u_vars.get((idx_f, a))
            vv = v_vars.get((idx_f, b))
            wv = w_vars.get((idx_f, c))
            if uv is not None and vv is not None and wv is not None:
                terms_abc.append(uv * vv * wv)
        if not terms_abc:
            if tgt != 0:
                return {
                    "status": "unsat_uncovered_residual",
                    "k_free": k_free,
                    "residual_nnz": int(len(nz_need)),
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                    "scheme": None,
                }
        else:
            solver.add(z3.Sum(terms_abc) == tgt)

    chk = solver.check()
    dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    if chk != z3.sat:
        return {
            "status": str(chk),
            "k_free": k_free,
            "residual_nnz": int(len(nz_need)),
            "active_triples": len(active_triples),
            "elapsed_ms": dt_ms,
            "scheme": None,
        }

    model = solver.model()
    U_new = U_z.copy()
    V_new = V_z.copy()
    W_new = W_z.copy()
    for idx_f, r in enumerate(free_set):
        U_new[r, :] = 0
        V_new[r, :] = 0
        W_new[r, :] = 0
        for a in term_as[idx_f]:
            U_new[r, a] = model.eval(u_vars[(idx_f, a)], model_completion=True).as_long()
        for b in term_bs[idx_f]:
            V_new[r, b] = model.eval(v_vars[(idx_f, b)], model_completion=True).as_long()
        for c in term_cs[idx_f]:
            W_new[r, c] = model.eval(w_vars[(idx_f, c)], model_completion=True).as_long()

    cand = BilinearScheme(
        m=m, n=n, p=p, U=U_new.tolist(), V=V_new.tolist(), W=W_new.tolist(), ring="Z", name=f"{m}x{n}x{p}_R{R}_z3_repair"
    )
    rep = verify_scheme(cand)
    return {
        "status": "sat_verified" if rep.valid else "sat_failed_verify",
        "k_free": k_free,
        "residual_nnz": int(len(nz_need)),
        "active_triples": len(active_triples),
        "elapsed_ms": dt_ms,
        "scheme": cand if rep.valid else None,
        "report": rep,
    }


def run_track_a_455_knockout(
    manifest_path: str,
    artifacts_dir: str,
    records_saved: List[Dict[str, Any]],
    budget_s: float = 285.0,
) -> Dict[str, Any]:
    """
    Targeted Knockout Push on 4x5x5 : R = 75 (`supp=7, unres=7 -> 0`):
      Stage A1: Recapture diverse `coker <= 10` basins with strict `R == 75` + 2-plus tunneling.
      Stage A2: Direct `unres_coker_4` SA from the `coker <= 10` basins (preserving `R == 75`)!
      Stage A3: Violated-equation diagnosis + violated-term targeted multi-plus sniper + Z3 freeze-repair.
      Stop-Crank early if plateaued with zero progress after deep tunneling.
    """
    t0 = time.perf_counter()
    deadline = t0 + budget_s
    print(f"\n=== Track A: Targeted Knockout Push on 4x5x5 : R = 75 (budget={budget_s:.0f}s) ===", flush=True)

    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    entry = next(e for e in manifest["seeds"] if e["seed_id"] == "4x5x5_R73_arai")
    sc73 = BilinearScheme.load_json(entry["file"])
    u73, v73, w73 = scheme_to_u64_f2(sc73)
    assert fast_verify_f2_u64(4, 5, 5, u73, v73, w73)
    prov = entry["provenance"]

    # GL pre-polish base 4x5x5:73
    pu73, pv73, pw73, ck_base = gl_polish_strict(4, 5, 5, u73, v73, w73, max_passes=2, sparsify_first=True)
    print(f"  [Base 4x5x5:R73 GL-polished] supp_coker_4={ck_base}", flush=True)

    # Archive: orbit_hash -> (unres_ck_or_999999, supp_ck, nnz, u, v, w)
    archive: Dict[int, Tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]] = {}
    best_supp = 999999
    best_unres = 999999
    total_evals = 0
    hit_zero = False
    lift_logs: List[Dict[str, Any]] = []

    def _register(uc: np.ndarray, vc: np.ndarray, wc: np.ndarray, check_unres: bool = False) -> Tuple[int, int]:
        nonlocal best_supp, best_unres, hit_zero
        if len(uc) != 75:
            return 999999, 999999
        h = int(_scheme_orbit_hash_nb(uc, vc, wc, 75))
        if h in archive and not check_unres:
            return archive[h][0], archive[h][1]
        s_ck = int(_mod4_oracle_support_nb(4, 5, 5, uc, vc, wc)[0])
        if s_ck < 0:
            return 999999, 999999
        u_ck = 999999
        if check_unres or s_ck <= 14:
            u_ck = int(_mod4_oracle_unrestricted_nb(4, 5, 5, uc, vc, wc)[0])
        nnz = int(_scheme_nnz_nb(uc, vc, wc, 75))
        if h not in archive or (u_ck, s_ck, nnz) < (archive[h][0], archive[h][1], archive[h][2]):
            archive[h] = (u_ck, s_ck, nnz, uc.copy(), vc.copy(), wc.copy())
        if s_ck < best_supp:
            best_supp = s_ck
        if u_ck < best_unres:
            best_unres = u_ck
        if s_ck == 0 or u_ck == 0:
            print(f"  *** ZERO OBSTRUCTION HIT on 4x5x5:R=75 (supp={s_ck}, unres={u_ck}) -> FIRING L1+L2! ***", flush=True)
            lres = attempt_certified_lift_and_save(
                4,
                5,
                5,
                uc,
                vc,
                wc,
                provenance=prov,
                artifacts_dir=artifacts_dir,
                records_saved=records_saved,
                z3_timeout_s=25.0,
                hensel_trials=64,
                label=f"phase3_4x5x5_R75_s{s_ck}_u{u_ck}",
            )
            lift_logs.append(lres)
            if lres["lifted"]:
                hit_zero = True
        return u_ck, s_ck

    # Stage A1: Deep SUPP + 2-plus/3-plus tunneling push on 4x5x5:R=75 (~72% of budget; NO early cutoff at 7!)
    n_walkers = 24
    R_cap = 80
    hot_terms = np.arange(75, dtype=np.int32)
    init_seeds = make_plus_relaxed_seeds(4, 5, 5, [(pu73, pv73, pw73)], extra_ranks=2, n_out=24, seed=3001)
    for su, sv, sw in init_seeds:
        if len(su) == 75:
            _register(su, sv, sw, check_unres=False)

    stage1_deadline = min(deadline, t0 + budget_s * 0.72)
    rd = 0
    prev_supp_a1 = best_supp
    stall_a1 = 0
    while time.perf_counter() < stage1_deadline and not hit_zero:
        rd += 1
        ranked = sorted(archive.values(), key=lambda x: (x[1], x[0], x[2]))
        elites = ranked[: min(16, len(ranked))]
        U_in = np.zeros((n_walkers, R_cap), dtype=np.uint64)
        V_in = np.zeros((n_walkers, R_cap), dtype=np.uint64)
        W_in = np.zeros((n_walkers, R_cap), dtype=np.uint64)
        R_in = np.full(n_walkers, 75, dtype=np.int32)

        fresh = (
            make_plus_relaxed_seeds(4, 5, 5, [(pu73, pv73, pw73)], extra_ranks=2, n_out=6, seed=3100 + rd * 97)
            if rd % 4 == 0
            else []
        )
        for k in range(n_walkers):
            if fresh and k >= n_walkers - len(fresh) and len(fresh[k - (n_walkers - len(fresh))][0]) == 75:
                fu, fv, fw = fresh[k - (n_walkers - len(fresh))]
                U_in[k, :75], V_in[k, :75], W_in[k, :75] = fu, fv, fw
            elif k < 6:
                el = elites[k % min(3, len(elites))]
                U_in[k, :75], V_in[k, :75], W_in[k, :75] = el[3], el[4], el[5]
            else:
                el = elites[k % len(elites)]
                U_in[k, :75], V_in[k, :75], W_in[k, :75] = el[3], el[4], el[5]

        Ub, Vb, Wb, Rb, ckb, evb = _tunnel_455_sa_nb(
            4, 5, 5, U_in, V_in, W_in, R_in, 75, 260, False, hot_terms, np.uint64(300001 + rd * 7919)
        )
        total_evals += int(np.sum(evb))
        order = np.argsort(ckb)
        for ri, wi in enumerate(order[:8]):
            wi_i = int(wi)
            if int(Rb[wi_i]) != 75:
                continue
            uc, vc, wc = Ub[wi_i, :75].copy(), Vb[wi_i, :75].copy(), Wb[wi_i, :75].copy()
            if ri < 2:
                uc, vc, wc, _ = gl_polish_strict(4, 5, 5, uc, vc, wc, max_passes=2, sparsify_first=False)
            _register(uc, vc, wc, check_unres=(ri < 3 or int(ckb[wi_i]) <= 10))
            if hit_zero:
                break

        if best_supp < prev_supp_a1:
            prev_supp_a1 = best_supp
            stall_a1 = 0
            # Dynamically refresh hot violated terms from the new champion basin!
            best_cur = min(archive.values(), key=lambda x: (x[1], x[0], x[2]))
            d_tmp = diagnose_violated_equations_and_terms(4, 5, 5, best_cur[3], best_cur[4], best_cur[5])
            hot_terms = d_tmp["hot_terms"][:20].copy()
            print(
                f"  [A1 NEW BEST supp={best_supp} unres={best_unres}] r{rd:2d} t={time.perf_counter()-t0:5.1f}s "
                f"int_viols={d_tmp['partial_sign_int_violations']}/10000 (max_err={d_tmp['partial_sign_max_abs_err']})",
                flush=True,
            )
        else:
            stall_a1 += 1
            if rd % 3 == 0 or rd == 1:
                print(
                    f"  [A1 SUPP+2plus r{rd:2d}] t={time.perf_counter()-t0:5.1f}s evals={total_evals:6d} "
                    f"best_supp={best_supp} best_unres={best_unres} archive={len(archive)} stall={stall_a1}",
                    flush=True,
                )
        if stall_a1 >= 14:
            print("  [Stop-Crank A1] 14 consecutive rounds without supp improvement -> advancing to A2 UNRES!", flush=True)
            break

    # Diagnose the #1 basin to extract hot violated terms before Stage A2!
    diag_info: Dict[str, Any] = {}
    if archive and not hit_zero:
        best_entry = min(archive.values(), key=lambda x: (x[0], x[1], x[2]))
        diag_info = diagnose_violated_equations_and_terms(4, 5, 5, best_entry[3], best_entry[4], best_entry[5])
        hot_terms = diag_info["hot_terms"][:20].copy()
        print(
            f"  [Diagnosis on best 4x5x5:R=75] unres_coker={diag_info['unres_coker_4']}, "
            f"supp_coker={diag_info['supp_coker_4']}, viol_e1_triples={diag_info['num_viol_e1_triples']}, "
            f"partial_sign_int_viols={diag_info['partial_sign_int_violations']}/10000 "
            f"(max_err={diag_info['partial_sign_max_abs_err']})",
            flush=True,
        )

    # Stage A2: Direct `unres_coker_4` multi-plus tunneling SA strictly preserving R == 75 (~22% of budget)
    stage2_deadline = min(deadline, t0 + budget_s * 0.94)
    rd2 = 0
    stall_rounds = 0
    prev_best_unres = best_unres
    while time.perf_counter() < stage2_deadline and not hit_zero and archive:
        rd2 += 1
        ranked_u = sorted(archive.values(), key=lambda x: (x[0], x[1], x[2]))
        elites_u = ranked_u[: min(12, len(ranked_u))]
        U_in = np.zeros((n_walkers, R_cap), dtype=np.uint64)
        V_in = np.zeros((n_walkers, R_cap), dtype=np.uint64)
        W_in = np.zeros((n_walkers, R_cap), dtype=np.uint64)
        R_in = np.full(n_walkers, 75, dtype=np.int32)
        for k in range(n_walkers):
            el = elites_u[k % len(elites_u)]
            U_in[k, :75], V_in[k, :75], W_in[k, :75] = el[3], el[4], el[5]

        Ub, Vb, Wb, Rb, ckb, evb = _tunnel_455_sa_nb(
            4, 5, 5, U_in, V_in, W_in, R_in, 75, 45, True, hot_terms, np.uint64(600001 + rd2 * 104729)
        )
        total_evals += int(np.sum(evb))
        order = np.argsort(ckb)
        for ri, wi in enumerate(order[:8]):
            wi_i = int(wi)
            if int(Rb[wi_i]) != 75:
                continue
            uc, vc, wc = Ub[wi_i, :75].copy(), Vb[wi_i, :75].copy(), Wb[wi_i, :75].copy()
            _register(uc, vc, wc, check_unres=True)
            if hit_zero:
                break

        if best_unres < prev_best_unres:
            prev_best_unres = best_unres
            stall_rounds = 0
            best_entry = min(archive.values(), key=lambda x: (x[0], x[1], x[2]))
            diag_info = diagnose_violated_equations_and_terms(4, 5, 5, best_entry[3], best_entry[4], best_entry[5])
            hot_terms = diag_info["hot_terms"][:20].copy()
        else:
            stall_rounds += 1

        print(
            f"  [A2 UNRES+2plus r{rd2:2d}] t={time.perf_counter()-t0:5.1f}s evals={total_evals:6d} "
            f"best_unres={best_unres} best_supp={best_supp} stall={stall_rounds}/4",
            flush=True,
        )
        if stall_rounds >= 4:
            print("  [Stop-Crank A2] Plateaued for 4 consecutive UNRES+2plus rounds -> advancing to A3!", flush=True)
            break

    # Stage A3: Exact Z3 Local Freeze-and-Repair around the violated equations
    z3_repair_logs: List[Dict[str, Any]] = []
    if archive and not hit_zero and time.perf_counter() < deadline:
        best_entry = min(archive.values(), key=lambda x: (x[0], x[1], x[2]))
        diag_info = diagnose_violated_equations_and_terms(4, 5, 5, best_entry[3], best_entry[4], best_entry[5])
        print(
            f"  [A3 Final Diagnosis] unres_coker={diag_info['unres_coker_4']}, supp_coker={diag_info['supp_coker_4']}, "
            f"partial_sign_int_viols={diag_info['partial_sign_int_violations']}/10000",
            flush=True,
        )
        for k_free in (6, 10, 14):
            rem = deadline - time.perf_counter()
            if rem < 12.0 or hit_zero:
                break
            zres = attempt_z3_local_freeze_repair(
                4,
                5,
                5,
                diag_info["U_z"],
                diag_info["V_z"],
                diag_info["W_z"],
                diag_info["hot_terms"],
                k_free=k_free,
                timeout_s=min(22.0, rem - 2.0),
            )
            log_item = {k: v for k, v in zres.items() if k not in ("scheme", "report")}
            z3_repair_logs.append(log_item)
            print(f"  [A3 Z3 freeze-repair k_free={k_free}] {log_item}", flush=True)
            if zres.get("scheme") is not None:
                cand: BilinearScheme = zres["scheme"]
                nc_ok = _verify_noncommutative_and_random_int(cand, 100, 20)
                if nc_ok:
                    ring_tag = "ZT" if zres["report"].is_ternary else "Z"
                    fname = f"lifted_record_4x5x5_R75_{ring_tag}.json"
                    out_path = os.path.join(artifacts_dir, fname)
                    cand.save_json(
                        out_path,
                        extra_metadata={
                            "status": "CERTIFIED_WORLD_RECORD_2026",
                            "lifted_ring": ring_tag,
                            "lift_method": f"z3_local_freeze_repair_k{k_free}",
                            "sota_2026_beaten": SOTA_CHAR0_2026[(4, 5, 5)],
                            "provenance": prov,
                            "brent_verifier_report": zres["report"].to_dict(),
                            "noncommutative_M2Z_20x_pass": True,
                            "random_int_100x_pass": True,
                        },
                    )
                    records_saved.append({"file": out_path, "format": [4, 5, 5], "rank": 75, "ring": ring_tag})
                    hit_zero = True
                    print(f"  !!! CERTIFIED WORLD RECORD SAVED: {out_path} !!!", flush=True)

    top_basins = sorted(archive.values(), key=lambda x: (x[0], x[1], x[2]))[:5]
    elapsed_a = time.perf_counter() - t0
    return {
        "track": "Track A: 4x5x5 : R = 75 Knockout Push",
        "format": [4, 5, 5],
        "R_target": 75,
        "sota_char0_2026": 76,
        "total_oracle_evals": total_evals,
        "unique_basins_archived": len(archive),
        "min_unres_coker_4": best_unres if best_unres < 999999 else None,
        "min_supp_coker_4": best_supp if best_supp < 999999 else None,
        "hit_zero_and_lifted": hit_zero,
        "diagnosis": {
            "unres_coker_4": diag_info.get("unres_coker_4"),
            "supp_coker_4": diag_info.get("supp_coker_4"),
            "num_viol_e1_triples": diag_info.get("num_viol_e1_triples"),
            "partial_sign_int_violations": diag_info.get("partial_sign_int_violations"),
            "partial_sign_max_abs_err": diag_info.get("partial_sign_max_abs_err"),
        }
        if diag_info
        else {},
        "z3_freeze_repair_attempts": z3_repair_logs,
        "lift_attempts": lift_logs,
        "top_basins": [
            {"rank": 75, "unres_coker_4": e[0] if e[0] < 999999 else None, "supp_coker_4": e[1], "nnz": e[2]}
            for e in top_basins
        ],
        "elapsed_s": round(elapsed_a, 2),
        "provenance": prov,
    }


# =============================================================================
# 2. Track B — Exact F_3 Flip-Graph + GL(F_3) + coker_viol_9 SA & 3-adic Lifter
# =============================================================================


@nb.njit(cache=True)
def _fast_verify_f3_u64_nb(
    m: int,
    n: int,
    p: int,
    up: np.ndarray,
    un: np.ndarray,
    vp: np.ndarray,
    vn: np.ndarray,
    wp: np.ndarray,
    wn: np.ndarray,
) -> bool:
    """Exact O(R * nnz) verification that 2-bitplane uint64[R] scheme satisfies Brent equations mod 3."""
    R = up.shape[0]
    mn = m * n
    np_dim = n * p
    mp = m * p
    tot = mn * np_dim * mp
    counts = np.zeros(tot, dtype=np.int16)
    for r in range(R):
        # Disjoint bitplane invariant check
        if (up[r] & un[r]) != 0 or (vp[r] & vn[r]) != 0 or (wp[r] & wn[r]) != 0:
            return False
        um = up[r] | un[r]
        vm = vp[r] | vn[r]
        wm = wp[r] | wn[r]
        tu = um
        while tu != 0:
            lu = tu & (~tu + np.uint64(1))
            a = np.int64(_popcount64(lu - np.uint64(1)))
            su = 1 if (up[r] & lu) != 0 else -1
            base_a = a * np_dim * mp
            tv = vm
            while tv != 0:
                lv = tv & (~tv + np.uint64(1))
                b = np.int64(_popcount64(lv - np.uint64(1)))
                suv = su if (vp[r] & lv) != 0 else -su
                base_ab = base_a + b * mp
                tw = wm
                while tw != 0:
                    lw = tw & (~tw + np.uint64(1))
                    c = np.int64(_popcount64(lw - np.uint64(1)))
                    counts[base_ab + c] += suv if (wp[r] & lw) != 0 else -suv
                    tw ^= lw
                tv ^= lv
            tu ^= lu

    for i1 in range(m):
        for i2 in range(n):
            a = i1 * n + i2
            for j2 in range(p):
                b = i2 * p + j2
                c = i1 * p + j2
                counts[(a * np_dim + b) * mp + c] -= 1

    for idx in range(tot):
        if counts[idx] % 3 != 0:
            return False
    return True


@nb.njit(cache=True)
def _f3_scheme_nnz_nb(
    up: np.ndarray, un: np.ndarray, vp: np.ndarray, vn: np.ndarray, wp: np.ndarray, wn: np.ndarray, R: int
) -> int:
    tot = 0
    for r in range(R):
        tot += int(_popcount64(up[r] | un[r])) + int(_popcount64(vp[r] | vn[r])) + int(_popcount64(wp[r] | wn[r]))
    return tot


@nb.njit(cache=True)
def _f3_apply_plus_nb(
    up: np.ndarray,
    un: np.ndarray,
    vp: np.ndarray,
    vn: np.ndarray,
    wp: np.ndarray,
    wn: np.ndarray,
    R: int,
    i: int,
    j: int,
    variant: int,
) -> int:
    """
    Exact ring-universal plus move over F_3 (valid identically in Z and F_3!):
      u_i v_i w_i + u_j v_j w_j
        = u_i (v_i - v_j) w_i + (u_i + u_j) v_j w_i + u_j v_j (w_j - w_i)
    (variant in 0..2 permutes which axis is split).
    """
    ui_p, ui_n = up[i], un[i]
    vi_p, vi_n = vp[i], vn[i]
    wi_p, wi_n = wp[i], wn[i]
    uj_p, uj_n = up[j], un[j]
    vj_p, vj_n = vp[j], vn[j]
    wj_p, wj_n = wp[j], wn[j]

    if variant == 0:
        # term i: u_i, (v_i - v_j), w_i
        vp[i], vn[i] = _f3_add_u64(vi_p, vi_n, vj_n, vj_p)
        # term j: u_j, v_j, (w_j - w_i)
        wp[j], wn[j] = _f3_add_u64(wj_p, wj_n, wi_n, wi_p)
        # term R: (u_i + u_j), v_j, w_i
        up[R], un[R] = _f3_add_u64(ui_p, ui_n, uj_p, uj_n)
        vp[R], vn[R] = vj_p, vj_n
        wp[R], wn[R] = wi_p, wi_n
    elif variant == 1:
        # u_i v_i w_i + u_j v_j w_j = (u_i - u_j) v_i w_i + u_j (v_i + v_j) w_i + u_j v_j (w_j - w_i)
        up[i], un[i] = _f3_add_u64(ui_p, ui_n, uj_n, uj_p)
        wp[j], wn[j] = _f3_add_u64(wj_p, wj_n, wi_n, wi_p)
        up[R], un[R] = uj_p, uj_n
        vp[R], vn[R] = _f3_add_u64(vi_p, vi_n, vj_p, vj_n)
        wp[R], wn[R] = wi_p, wi_n
    else:
        # u_i v_i w_i + u_j v_j w_j = u_i v_i (w_i - w_j) + u_i (v_i - v_j) w_j + (u_i + u_j) v_j w_j
        wp[i], wn[i] = _f3_add_u64(wi_p, wi_n, wj_n, wj_p)
        up[j], un[j] = _f3_add_u64(ui_p, ui_n, uj_p, uj_n)
        up[R], un[R] = ui_p, ui_n
        vp[R], vn[R] = _f3_add_u64(vi_p, vi_n, vj_n, vj_p)
        wp[R], wn[R] = wj_p, wj_n
    return R + 1


@nb.njit(cache=True)
def _f3_reduce_nb(
    up: np.ndarray,
    un: np.ndarray,
    vp: np.ndarray,
    vn: np.ndarray,
    wp: np.ndarray,
    wn: np.ndarray,
    R: int,
) -> int:
    """Exact F_3 rank reduction: drop zero rows & merge any two rows collinear on 2 of 3 axes."""
    changed = True
    while changed:
        changed = False
        # 1. Remove zero rows
        w_out = 0
        for r in range(R):
            if (up[r] | un[r]) == 0 or (vp[r] | vn[r]) == 0 or (wp[r] | wn[r]) == 0:
                changed = True
                continue
            if w_out != r:
                up[w_out], un[w_out] = up[r], un[r]
                vp[w_out], vn[w_out] = vp[r], vn[r]
                wp[w_out], wn[w_out] = wp[r], wn[r]
            w_out += 1
        R = w_out

        # 2. Merge pairs collinear on 2 axes
        for i in range(R):
            for j in range(i + 1, R):
                # Check u collinear
                su = 1 if (up[i] == up[j] and un[i] == un[j]) else (-1 if (up[i] == un[j] and un[i] == up[j]) else 0)
                sv = 1 if (vp[i] == vp[j] and vn[i] == vn[j]) else (-1 if (vp[i] == vn[j] and vn[i] == vp[j]) else 0)
                sw = 1 if (wp[i] == wp[j] and wn[i] == wn[j]) else (-1 if (wp[i] == wn[j] and wn[i] == wp[j]) else 0)
                if su != 0 and sv != 0:
                    if su * sv > 0:
                        wp[i], wn[i] = _f3_add_u64(wp[i], wn[i], wp[j], wn[j])
                    else:
                        wp[i], wn[i] = _f3_add_u64(wp[i], wn[i], wn[j], wp[j])
                    up[j], un[j] = np.uint64(0), np.uint64(0)
                    changed = True
                    break
                if su != 0 and sw != 0:
                    if su * sw > 0:
                        vp[i], vn[i] = _f3_add_u64(vp[i], vn[i], vp[j], vn[j])
                    else:
                        vp[i], vn[i] = _f3_add_u64(vp[i], vn[i], vn[j], vp[j])
                    up[j], un[j] = np.uint64(0), np.uint64(0)
                    changed = True
                    break
                if sv != 0 and sw != 0:
                    if sv * sw > 0:
                        up[i], un[i] = _f3_add_u64(up[i], un[i], up[j], un[j])
                    else:
                        up[i], un[i] = _f3_add_u64(up[i], un[i], un[j], up[j])
                    up[j], un[j] = np.uint64(0), np.uint64(0)
                    changed = True
                    break
            if changed:
                break
    return R


@nb.njit(cache=True)
def _f3_gl_shear_nb(
    m: int,
    n: int,
    p: int,
    up: np.ndarray,
    un: np.ndarray,
    vp: np.ndarray,
    vn: np.ndarray,
    wp: np.ndarray,
    wn: np.ndarray,
    R: int,
    group_id: int,
    r1: int,
    r2: int,
    sgn: int,
) -> None:
    """
    In-place exact elementary SL(Z) / GL(F_3) basis change:
      group 0 (GL_m): U[r2, :] += sgn * U[r1, :], W[r1, :] -= sgn * W[r2, :]
      group 1 (GL_n): U[:, r1] -= sgn * U[:, r2], V[r2, :] += sgn * V[r1, :]
      group 2 (GL_p): V[:, r1] -= sgn * V[:, r2], W[:, r2] += sgn * W[:, r1]
    Note: Exact over Z and F_3! (Check: <U', A> = <(P^-1)^T U, A> = <U, P^-1 A>, W' = P W!)
    """
    if group_id == 0:
        # P = I + sgn * E_{r1, r2}, P^-1 = I - sgn * E_{r1, r2}, (P^-1)^T = I - sgn * E_{r2, r1}
        # => U'[r2, j] = U[r2, j] - sgn * U[r1, j]; W'[r1, k] = W[r1, k] + sgn * W[r2, k]
        for r in range(R):
            a_p, a_n = up[r], un[r]
            for j in range(n):
                b1 = np.uint64(1) << np.uint64(r1 * n + j)
                b2 = np.uint64(1) << np.uint64(r2 * n + j)
                val1 = 1 if (a_p & b1) != 0 else (-1 if (a_n & b1) != 0 else 0)
                if val1 != 0:
                    val2 = 1 if (a_p & b2) != 0 else (-1 if (a_n & b2) != 0 else 0)
                    nv = (val2 - sgn * val1 + 3) % 3
                    a_p &= ~b2
                    a_n &= ~b2
                    if nv == 1:
                        a_p |= b2
                    elif nv == 2:
                        a_n |= b2
            up[r], un[r] = a_p, a_n

            c_p, c_n = wp[r], wn[r]
            for k in range(p):
                b1 = np.uint64(1) << np.uint64(r1 * p + k)
                b2 = np.uint64(1) << np.uint64(r2 * p + k)
                val2 = 1 if (c_p & b2) != 0 else (-1 if (c_n & b2) != 0 else 0)
                if val2 != 0:
                    val1 = 1 if (c_p & b1) != 0 else (-1 if (c_n & b1) != 0 else 0)
                    nv = (val1 + sgn * val2 + 3) % 3
                    c_p &= ~b1
                    c_n &= ~b1
                    if nv == 1:
                        c_p |= b1
                    elif nv == 2:
                        c_n |= b1
            wp[r], wn[r] = c_p, c_n
    elif group_id == 1:
        # A B = (A Q)(Q^-1 B), Q = I + sgn * E_{r1, r2}, Q^-1 = I - sgn * E_{r1, r2}
        # <U', A> = <U, A Q> = <U Q^T, A> => U'[:, r1] = U[:, r1] + sgn * U[:, r2]
        # <V', B> = <V, Q^-1 B> = <(Q^-1)^T V, B> => V'[r2, :] = V[r2, :] - sgn * V[r1, :]
        for r in range(R):
            a_p, a_n = up[r], un[r]
            for i in range(m):
                b1 = np.uint64(1) << np.uint64(i * n + r1)
                b2 = np.uint64(1) << np.uint64(i * n + r2)
                val2 = 1 if (a_p & b2) != 0 else (-1 if (a_n & b2) != 0 else 0)
                if val2 != 0:
                    val1 = 1 if (a_p & b1) != 0 else (-1 if (a_n & b1) != 0 else 0)
                    nv = (val1 + sgn * val2 + 3) % 3
                    a_p &= ~b1
                    a_n &= ~b1
                    if nv == 1:
                        a_p |= b1
                    elif nv == 2:
                        a_n |= b1
            up[r], un[r] = a_p, a_n

            b_p, b_n = vp[r], vn[r]
            for k in range(p):
                b1 = np.uint64(1) << np.uint64(r1 * p + k)
                b2 = np.uint64(1) << np.uint64(r2 * p + k)
                val1 = 1 if (b_p & b1) != 0 else (-1 if (b_n & b1) != 0 else 0)
                if val1 != 0:
                    val2 = 1 if (b_p & b2) != 0 else (-1 if (b_n & b2) != 0 else 0)
                    nv = (val2 - sgn * val1 + 3) % 3
                    b_p &= ~b2
                    b_n &= ~b2
                    if nv == 1:
                        b_p |= b2
                    elif nv == 2:
                        b_n |= b2
            vp[r], vn[r] = b_p, b_n
    else:
        # C' = C S, B' = B S => <V', B> = <V, B S^-1> = <V (S^-1)^T, B>, W' = W S
        # S = I + sgn * E_{r1, r2}, S^-1 = I - sgn * E_{r1, r2}
        # => V'[:, r1] = V[:, r1] - sgn * V[:, r2]; W'[:, r2] = W[:, r2] + sgn * W[:, r1]
        for r in range(R):
            b_p, b_n = vp[r], vn[r]
            for j in range(n):
                b1 = np.uint64(1) << np.uint64(j * p + r1)
                b2 = np.uint64(1) << np.uint64(j * p + r2)
                val2 = 1 if (b_p & b2) != 0 else (-1 if (b_n & b2) != 0 else 0)
                if val2 != 0:
                    val1 = 1 if (b_p & b1) != 0 else (-1 if (b_n & b1) != 0 else 0)
                    nv = (val1 - sgn * val2 + 3) % 3
                    b_p &= ~b1
                    b_n &= ~b1
                    if nv == 1:
                        b_p |= b1
                    elif nv == 2:
                        b_n |= b1
            vp[r], vn[r] = b_p, b_n

            c_p, c_n = wp[r], wn[r]
            for i in range(m):
                b1 = np.uint64(1) << np.uint64(i * p + r1)
                b2 = np.uint64(1) << np.uint64(i * p + r2)
                val1 = 1 if (c_p & b1) != 0 else (-1 if (c_n & b1) != 0 else 0)
                if val1 != 0:
                    val2 = 1 if (c_p & b2) != 0 else (-1 if (c_n & b2) != 0 else 0)
                    nv = (val2 + sgn * val1 + 3) % 3
                    c_p &= ~b2
                    c_n &= ~b2
                    if nv == 1:
                        c_p |= b2
                    elif nv == 2:
                        c_n |= b2
            wp[r], wn[r] = c_p, c_n


@nb.njit(parallel=True, cache=True)
def _f3_coker9_sa_batch_nb(
    m: int,
    n: int,
    p: int,
    Up_in: np.ndarray,
    Un_in: np.ndarray,
    Vp_in: np.ndarray,
    Vn_in: np.ndarray,
    Wp_in: np.ndarray,
    Wn_in: np.ndarray,
    R_in: np.ndarray,
    R_target: int,
    n_iters: int,
    support_restricted: bool,
    seed_base: np.uint64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    12-thread Numba F_3 flip-graph + GL(F_3) + plus/reduce SA directly minimizing `coker_viol_9`!
    Stops any walker immediately if `coker_viol_9 == 0`!
    """
    nw = Up_in.shape[0]
    R_cap = Up_in.shape[1]

    Up_b = Up_in.copy()
    Un_b = Un_in.copy()
    Vp_b = Vp_in.copy()
    Vn_b = Vn_in.copy()
    Wp_b = Wp_in.copy()
    Wn_b = Wn_in.copy()
    R_b = R_in.copy()
    ck_b = np.full(nw, 999999, dtype=np.int32)
    evals = np.zeros(nw, dtype=np.int64)

    for wid in nb.prange(nw):
        up_c = Up_in[wid].copy()
        un_c = Un_in[wid].copy()
        vp_c = Vp_in[wid].copy()
        vn_c = Vn_in[wid].copy()
        wp_c = Wp_in[wid].copy()
        wn_c = Wn_in[wid].copy()
        R_c = int(R_in[wid])

        c0 = _mod9_oracle_nb(m, n, p, up_c[:R_c], un_c[:R_c], vp_c[:R_c], vn_c[:R_c], wp_c[:R_c], wn_c[:R_c], support_restricted)[0]
        cur_ck = int(c0) if c0 >= 0 else 999999
        cur_nnz = _f3_scheme_nnz_nb(up_c, un_c, vp_c, vn_c, wp_c, wn_c, R_c)
        best_ck = cur_ck
        best_R = R_c
        best_nnz = cur_nnz
        ck_b[wid] = best_ck
        R_b[wid] = best_R

        state = seed_base ^ (np.uint64(wid + 71) * np.uint64(0x9E3779B97F4A7C15))
        if state == 0:
            state = np.uint64(29)

        up_t = np.empty(R_cap, dtype=np.uint64)
        un_t = np.empty(R_cap, dtype=np.uint64)
        vp_t = np.empty(R_cap, dtype=np.uint64)
        vn_t = np.empty(R_cap, dtype=np.uint64)
        wp_t = np.empty(R_cap, dtype=np.uint64)
        wn_t = np.empty(R_cap, dtype=np.uint64)
        ei = np.empty(4096, dtype=np.int32)
        ej = np.empty(4096, dtype=np.int32)
        ea = np.empty(4096, dtype=np.int8)
        es = np.empty(4096, dtype=np.int8)

        tier = 0 if wid < 8 else (1 if wid < 16 else 2)
        t_hi = 1.2 if tier == 0 else (4.5 if tier == 1 else 10.0)
        t_lo = 0.10 if tier == 0 else (0.25 if tier == 1 else 0.60)
        ev = 1

        for it in range(n_iters):
            if best_ck == 0:
                break
            for k in range(R_c):
                up_t[k], un_t[k] = up_c[k], un_c[k]
                vp_t[k], vn_t[k] = vp_c[k], vn_c[k]
                wp_t[k], wn_t[k] = wp_c[k], wn_c[k]
            R_t = R_c

            did_gl = False
            if support_restricted and (it % 6 == 5):
                state = _rng_next(state)
                gid = np.int64(state % np.uint64(3))
                dim_g = m if gid == 0 else (n if gid == 1 else p)
                if dim_g >= 2:
                    state = _rng_next(state)
                    r1 = np.int64(state % np.uint64(dim_g))
                    state = _rng_next(state)
                    r2 = np.int64(state % np.uint64(dim_g - 1))
                    if r2 >= r1:
                        r2 += np.int64(1)
                    state = _rng_next(state)
                    sgn_gl = 1 if (state & np.uint64(1)) == 0 else -1
                    _f3_gl_shear_nb(m, n, p, up_t, un_t, vp_t, vn_t, wp_t, wn_t, R_t, gid, r1, r2, sgn_gl)
                    did_gl = True

            if not did_gl:
                # Build F_3 flip edges
                ne = 0
                for i in range(R_t):
                    for j in range(i + 1, R_t):
                        if ne >= 4092:
                            break
                        if up_t[i] == up_t[j] and un_t[i] == un_t[j]:
                            ei[ne], ej[ne], ea[ne], es[ne] = i, j, 0, 1
                            ne += 1
                        elif up_t[i] == un_t[j] and un_t[i] == up_t[j]:
                            ei[ne], ej[ne], ea[ne], es[ne] = i, j, 0, -1
                            ne += 1
                        if vp_t[i] == vp_t[j] and vn_t[i] == vn_t[j]:
                            ei[ne], ej[ne], ea[ne], es[ne] = i, j, 1, 1
                            ne += 1
                        elif vp_t[i] == vn_t[j] and vn_t[i] == vp_t[j]:
                            ei[ne], ej[ne], ea[ne], es[ne] = i, j, 1, -1
                            ne += 1
                        if wp_t[i] == wp_t[j] and wn_t[i] == wn_t[j]:
                            ei[ne], ej[ne], ea[ne], es[ne] = i, j, 2, 1
                            ne += 1
                        elif wp_t[i] == wn_t[j] and wn_t[i] == wp_t[j]:
                            ei[ne], ej[ne], ea[ne], es[ne] = i, j, 2, -1
                            ne += 1

                do_plus = (ne == 0 or (tier >= 1 and it % 4 == 3)) and (R_t + 1 < R_cap)
                if do_plus:
                    state = _rng_next(state)
                    pi = np.int64(state % np.uint64(R_t))
                    state = _rng_next(state)
                    pj = np.int64(state % np.uint64(R_t - 1))
                    if pj >= pi:
                        pj += 1
                    state = _rng_next(state)
                    R_t = _f3_apply_plus_nb(up_t, un_t, vp_t, vn_t, wp_t, wn_t, R_t, pi, pj, np.int64(state % np.uint64(3)))

                nfl = 12 if do_plus else (1 + int(it % (2 if tier == 0 else 5)))
                for _ in range(nfl):
                    # Fast random edge scan if ne == 0 or after update
                    ne = 0
                    for i in range(R_t):
                        for j in range(i + 1, R_t):
                            if ne >= 4092:
                                break
                            if up_t[i] == up_t[j] and un_t[i] == un_t[j]:
                                ei[ne], ej[ne], ea[ne], es[ne] = i, j, 0, 1
                                ne += 1
                            elif up_t[i] == un_t[j] and un_t[i] == up_t[j]:
                                ei[ne], ej[ne], ea[ne], es[ne] = i, j, 0, -1
                                ne += 1
                            if vp_t[i] == vp_t[j] and vn_t[i] == vn_t[j]:
                                ei[ne], ej[ne], ea[ne], es[ne] = i, j, 1, 1
                                ne += 1
                            elif vp_t[i] == vn_t[j] and vn_t[i] == vp_t[j]:
                                ei[ne], ej[ne], ea[ne], es[ne] = i, j, 1, -1
                                ne += 1
                            if wp_t[i] == wp_t[j] and wn_t[i] == wn_t[j]:
                                ei[ne], ej[ne], ea[ne], es[ne] = i, j, 2, 1
                                ne += 1
                            elif wp_t[i] == wn_t[j] and wn_t[i] == wp_t[j]:
                                ei[ne], ej[ne], ea[ne], es[ne] = i, j, 2, -1
                                ne += 1
                    if ne == 0:
                        break
                    state = _rng_next(state)
                    ek = np.int64(state % np.uint64(ne))
                    si = int(ei[ek])
                    sj = int(ej[ek])
                    ax = int(ea[ek])
                    sgn = int(es[ek])
                    state = _rng_next(state)
                    if (state & np.uint64(1)) != 0:
                        si, sj = sj, si
                    state = _rng_next(state)
                    alpha = 1 if (state & np.uint64(1)) == 0 else -1
                    if ax == 0:
                        vp_t[si], vn_t[si] = _f3_add_u64(
                            vp_t[si], vn_t[si], vn_t[sj] if alpha > 0 else vp_t[sj], vp_t[sj] if alpha > 0 else vn_t[sj]
                        )
                        ap = (alpha * sgn) > 0
                        wp_t[sj], wn_t[sj] = _f3_add_u64(
                            wp_t[sj], wn_t[sj], wp_t[si] if ap else wn_t[si], wn_t[si] if ap else wp_t[si]
                        )
                    elif ax == 1:
                        up_t[si], un_t[si] = _f3_add_u64(
                            up_t[si], un_t[si], un_t[sj] if alpha > 0 else up_t[sj], up_t[sj] if alpha > 0 else un_t[sj]
                        )
                        ap = (alpha * sgn) > 0
                        wp_t[sj], wn_t[sj] = _f3_add_u64(
                            wp_t[sj], wn_t[sj], wp_t[si] if ap else wn_t[si], wn_t[si] if ap else wp_t[si]
                        )
                    else:
                        up_t[si], un_t[si] = _f3_add_u64(
                            up_t[si], un_t[si], un_t[sj] if alpha > 0 else up_t[sj], up_t[sj] if alpha > 0 else un_t[sj]
                        )
                        ap = (alpha * sgn) > 0
                        vp_t[sj], vn_t[sj] = _f3_add_u64(
                            vp_t[sj], vn_t[sj], vp_t[si] if ap else vn_t[si], vn_t[si] if ap else vp_t[si]
                        )
                    R_t = _f3_reduce_nb(up_t, un_t, vp_t, vn_t, wp_t, wn_t, R_t)
                    if do_plus and R_t <= R_target:
                        break

            if R_t > R_target:
                continue

            ck = _mod9_oracle_nb(m, n, p, up_t[:R_t], un_t[:R_t], vp_t[:R_t], vn_t[:R_t], wp_t[:R_t], wn_t[:R_t], support_restricted)[0]
            ev += 1
            if ck < 0:
                continue

            nnz_t = _f3_scheme_nnz_nb(up_t, un_t, vp_t, vn_t, wp_t, wn_t, R_t)
            frac = float(it) / float(max(1, n_iters))
            temp = t_lo + (t_hi - t_lo) * (1.0 - frac)
            accept = False
            if ck < cur_ck or (ck == cur_ck and R_t < R_c):
                accept = True
            elif ck == cur_ck and R_t == R_c:
                if nnz_t <= cur_nnz + 24:
                    accept = True
            else:
                diff = float(ck - cur_ck)
                if diff < 3.5 * temp:
                    state = _rng_next(state)
                    u01 = float(state & np.uint64(0xFFFFFF)) / float(0x1000000)
                    if u01 < np.exp(-diff / temp):
                        accept = True

            if accept:
                R_c = R_t
                cur_ck = int(ck)
                cur_nnz = nnz_t
                for k in range(R_c):
                    up_c[k], un_c[k] = up_t[k], un_t[k]
                    vp_c[k], vn_c[k] = vp_t[k], vn_t[k]
                    wp_c[k], wn_c[k] = wp_t[k], wn_t[k]
                if (
                    (cur_ck < best_ck)
                    or (cur_ck == best_ck and R_c < best_R)
                    or (cur_ck == best_ck and R_c == best_R and cur_nnz < best_nnz)
                ):
                    best_ck = cur_ck
                    best_R = R_c
                    best_nnz = cur_nnz
                    for k in range(best_R):
                        Up_b[wid, k], Un_b[wid, k] = up_c[k], un_c[k]
                        Vp_b[wid, k], Vn_b[wid, k] = vp_c[k], vn_c[k]
                        Wp_b[wid, k], Wn_b[wid, k] = wp_c[k], wn_c[k]
                    R_b[wid] = best_R
                    ck_b[wid] = best_ck

        evals[wid] = ev

    return Up_b, Un_b, Vp_b, Vn_b, Wp_b, Wn_b, R_b, ck_b, evals


def _parse_f3_pool_entries(pool_path: str) -> List[Dict[str, Any]]:
    """
    Flexibly load ALL F_3 schemes from `f3_pool_phase3.json`, supporting:
      - list OR dict with keys ('schemes', 'pool', 'records', 'best_schemes', 'hits', 'targets')
      - 2-bitplane uint64 arrays: ('u_pos','u_neg','v_pos','v_neg','w_pos','w_neg') OR ('up','un','vp','vn','wp','wn')
      - dense integer matrices ('U', 'V', 'W') in {-1, 0, 1} or {0, 1, 2}
    """
    with open(pool_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    raw_items: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        raw_items = [x for x in raw if isinstance(x, dict)]
    elif isinstance(raw, dict):
        # Support `f3_pool_v1` schema: {"formats": {"4x4x4": {"schemes": [...]}, ...}}
        if isinstance(raw.get("formats"), dict):
            for fmt_k, fmt_v in raw["formats"].items():
                if isinstance(fmt_v, dict) and isinstance(fmt_v.get("schemes"), list):
                    for sc_item in fmt_v["schemes"]:
                        if isinstance(sc_item, dict):
                            cp = dict(sc_item)
                            cp.setdefault("format_key", fmt_k)
                            raw_items.append(cp)
                elif isinstance(fmt_v, list):
                    raw_items.extend([x for x in fmt_v if isinstance(x, dict)])
        for key in ("schemes", "pool", "records", "best_schemes", "hits", "candidates", "results"):
            val = raw.get(key)
            if isinstance(val, list):
                raw_items.extend([x for x in val if isinstance(x, dict)])
            elif isinstance(val, dict):
                for subk, subv in val.items():
                    if isinstance(subv, dict):
                        cp = dict(subv)
                        cp.setdefault("_key", subk)
                        raw_items.append(cp)
                    elif isinstance(subv, list):
                        raw_items.extend([x for x in subv if isinstance(x, dict)])
        # Also handle dict mapping target -> {best_scheme: ...}
        if isinstance(raw.get("targets"), list):
            for tgt in raw["targets"]:
                if isinstance(tgt, dict):
                    if any(k in tgt for k in ("U", "u_pos", "up", "pu")):
                        raw_items.append(tgt)
                    for subk in ("best_scheme", "scheme", "schemes", "pool"):
                        sv = tgt.get(subk)
                        if isinstance(sv, dict):
                            cp = dict(sv)
                            cp.setdefault("target_label", tgt.get("target_label", tgt.get("format_str", "")))
                            raw_items.append(cp)
                        elif isinstance(sv, list):
                            for it_s in sv:
                                if isinstance(it_s, dict):
                                    raw_items.append(it_s)
        if not raw_items and any(k in raw for k in ("U", "u_pos", "up", "pu")):
            raw_items = [raw]

    parsed: List[Dict[str, Any]] = []
    seen_hashes: set[str] = set()

    for idx, item in enumerate(raw_items):
        # Extract (m, n, p)
        fmt = item.get("format") or item.get("dims") or item.get("shape")
        if isinstance(fmt, list) and len(fmt) == 3:
            m, n, p = int(fmt[0]), int(fmt[1]), int(fmt[2])
        elif all(k in item for k in ("m", "n", "p")):
            m, n, p = int(item["m"]), int(item["n"]), int(item["p"])
        else:
            continue

        up = un = vp = vn = wp = wn = None
        if all(k in item for k in ("pu", "nu", "pv", "nv", "pw", "nw")):
            up = np.array([int(x) for x in item["pu"]], dtype=np.uint64)
            un = np.array([int(x) for x in item["nu"]], dtype=np.uint64)
            vp = np.array([int(x) for x in item["pv"]], dtype=np.uint64)
            vn = np.array([int(x) for x in item["nv"]], dtype=np.uint64)
            wp = np.array([int(x) for x in item["pw"]], dtype=np.uint64)
            wn = np.array([int(x) for x in item["nw"]], dtype=np.uint64)
        elif all(k in item for k in ("u_pos", "u_neg", "v_pos", "v_neg", "w_pos", "w_neg")):
            up = np.array([int(x) for x in item["u_pos"]], dtype=np.uint64)
            un = np.array([int(x) for x in item["u_neg"]], dtype=np.uint64)
            vp = np.array([int(x) for x in item["v_pos"]], dtype=np.uint64)
            vn = np.array([int(x) for x in item["v_neg"]], dtype=np.uint64)
            wp = np.array([int(x) for x in item["w_pos"]], dtype=np.uint64)
            wn = np.array([int(x) for x in item["w_neg"]], dtype=np.uint64)
        elif all(k in item for k in ("up", "un", "vp", "vn", "wp", "wn")):
            up = np.array([int(x) for x in item["up"]], dtype=np.uint64)
            un = np.array([int(x) for x in item["un"]], dtype=np.uint64)
            vp = np.array([int(x) for x in item["vp"]], dtype=np.uint64)
            vn = np.array([int(x) for x in item["vn"]], dtype=np.uint64)
            wp = np.array([int(x) for x in item["wp"]], dtype=np.uint64)
            wn = np.array([int(x) for x in item["wn"]], dtype=np.uint64)
        elif all(k in item for k in ("U", "V", "W")):
            sc_tmp = BilinearScheme(
                m=m, n=n, p=p, U=item["U"], V=item["V"], W=item["W"], ring="Z3", modulus=3, name=str(item.get("name", f"f3_{idx}"))
            )
            up, un, vp, vn, wp, wn = scheme_to_u64_f3(sc_tmp)
        else:
            continue

        R = len(up)
        if R == 0 or len(un) != R or len(vp) != R or len(vn) != R or len(wp) != R or len(wn) != R:
            continue

        # Canonical row-sorted SHA-256 deduplication
        rows_sorted = sorted(
            (int(up[r]), int(un[r]), int(vp[r]), int(vn[r]), int(wp[r]), int(wn[r])) for r in range(R)
        )
        h_hex = hashlib.sha256(f"{m}x{n}x{p}:{rows_sorted}".encode("ascii")).hexdigest()[:24]
        if h_hex in seen_hashes:
            continue
        seen_hashes.add(h_hex)

        parsed.append(
            {
                "id": item.get("name") or item.get("id") or f"gpu_f3_{m}x{n}x{p}_R{R}_{idx}",
                "format": (m, n, p),
                "rank": R,
                "up": up,
                "un": un,
                "vp": vp,
                "vn": vn,
                "wp": wp,
                "wn": wn,
                "hash": h_hex,
                "source_meta": {
                    k: v
                    for k, v in item.items()
                    if k
                    not in (
                        "U",
                        "V",
                        "W",
                        "pu",
                        "nu",
                        "pv",
                        "nv",
                        "pw",
                        "nw",
                        "u_pos",
                        "u_neg",
                        "v_pos",
                        "v_neg",
                        "w_pos",
                        "w_neg",
                        "up",
                        "un",
                        "vp",
                        "vn",
                        "wp",
                        "wn",
                    )
                },
            }
        )

    return parsed


def _is_char0_world_record(fmt: Tuple[int, int, int], rank: int, ring_tag: str) -> Tuple[bool, Optional[Dict[str, int]]]:
    sota = SOTA_CHAR0_2026.get(fmt)
    if sota is None:
        return False, None
    if ring_tag == "ZT":
        return rank < sota["ZT"], sota
    if ring_tag == "Z":
        return rank < sota["Z"], sota
    return rank < sota["Q"], sota


def attempt_certified_p3_lift_and_save(
    m: int,
    n: int,
    p: int,
    up: np.ndarray,
    un: np.ndarray,
    vp: np.ndarray,
    vn: np.ndarray,
    wp: np.ndarray,
    wn: np.ndarray,
    source_meta: Dict[str, Any],
    artifacts_dir: str,
    records_saved: List[Dict[str, Any]],
    hensel_trials: int = 64,
    label: str = "",
) -> Dict[str, Any]:
    """
    Fire Stage L1-p3 (3-adic Linear Hensel up to 3^40 + Wang rational reconstruction -> Z_T / Z / Z[1/2] Q)
    on an F_3 scheme with coker_viol_9 == 0, verify via `brent_verifier.verify_scheme()` + 100x int + 20x M_2(Z),
    and save `artifacts/lifted_record_<m>x<n>x<p>_R<R>_<ring>.json` if it beats November 2026 char-0 SOTA!
    """
    R = len(up)
    fmt = (m, n, p)
    res_log: Dict[str, Any] = {
        "label": label,
        "format": [m, n, p],
        "rank": R,
        "lifted": False,
        "lifted_ring": None,
        "world_record": False,
        "saved_file": None,
        "attempts": [],
    }

    u64_tuple = (m, n, p, up, un, vp, vn, wp, wn)
    modes_to_try = [True]
    if m * n * p <= 128:
        modes_to_try.append(False)

    best_cand: Optional[BilinearScheme] = None
    best_rep = None
    best_ring_tag = ""
    best_method = ""
    best_details: Dict[str, Any] = {}

    for supp_mode in modes_to_try:
        lres = linear_hensel_lift_p3(
            u64_tuple,
            support_restricted=supp_mode,
            target_trits=40,
            max_trials=hensel_trials,
            seed=307 + R,
            name=f"{m}x{n}x{p}_R{R}_p3lift",
        )
        res_log["attempts"].append(
            {
                "method": lres.method,
                "status": lres.status,
                "success": lres.success,
                "elapsed_ms": lres.elapsed_ms,
                "details": lres.details,
            }
        )
        if lres.success and lres.scheme is not None and lres.report is not None and lres.report.valid:
            ring_tag = "ZT" if lres.report.is_ternary else ("Z" if lres.scheme.ring == "Z" else "Q")
            best_cand = lres.scheme
            best_rep = lres.report
            best_ring_tag = ring_tag
            best_method = lres.method
            best_details = lres.details or {}
            # Prefer ZT/Z over Q; if we already got ZT or Z, stop immediately
            if ring_tag in ("ZT", "Z"):
                break

    if best_cand is None or best_rep is None:
        return res_log

    # Extra defense-in-depth: 100x random integer + 20x non-commutative M_2(Z) block matmul!
    nc_ok = _verify_noncommutative_and_random_int(best_cand, n_int=100, n_blk=20)
    res_log["noncommutative_M2Z_20x_pass"] = nc_ok
    if not nc_ok:
        res_log["error"] = "FAILED_NONCOMMUTATIVE_CHECK"
        return res_log

    res_log["lifted"] = True
    res_log["lifted_ring"] = best_ring_tag
    res_log["max_abs_coeff"] = str(best_rep.max_abs_coeff)
    res_log["max_denom"] = int(best_details.get("max_denom", 1))
    res_log["naive_additions_total"] = best_rep.additive_complexity["naive_additions_total"]

    is_wr, sota_ref = _is_char0_world_record(fmt, R, best_ring_tag)
    res_log["world_record"] = is_wr
    if is_wr:
        fname = f"lifted_record_{m}x{n}x{p}_R{R}_{best_ring_tag}.json"
        out_path = os.path.join(artifacts_dir, fname)
        best_cand.save_json(
            out_path,
            extra_metadata={
                "status": "CERTIFIED_WORLD_RECORD_2026",
                "lifted_ring": best_ring_tag,
                "lift_method": best_method,
                "lift_details": best_details,
                "sota_2026_beaten": sota_ref,
                "source_f3_provenance": source_meta,
                "brent_verifier_report": best_rep.to_dict(),
                "noncommutative_M2Z_20x_pass": True,
                "random_int_100x_pass": True,
            },
        )
        res_log["saved_file"] = out_path
        records_saved.append(
            {
                "file": out_path,
                "format": [m, n, p],
                "rank": R,
                "ring": best_ring_tag,
                "sota_2026_beaten": sota_ref,
                "max_denom": res_log["max_denom"],
            }
        )
        print(
            f"  !!! CERTIFIED 3-ADIC CHAR-0 WORLD RECORD SAVED: {out_path} "
            f"(R={R}, ring={best_ring_tag}, max_denom={res_log['max_denom']}, beats SOTA={sota_ref}) !!!",
            flush=True,
        )
    return res_log


def run_track_b_f3_pool_lifting(
    pool_path: str,
    artifacts_dir: str,
    records_saved: List[Dict[str, Any]],
    budget_s: float = 240.0,
) -> Dict[str, Any]:
    """
    Ingest GPU `f3_pool_phase3.json`:
      1. Parse & deduplicate all F_3 schemes, verify mod-3 exactness.
      2. Run Stage L0-p3 (`mod9_oracle_u64` / `mod9_oracle_batch_u64`) on EVERY scheme.
      3. Fire Stage L1-p3 (`linear_hensel_lift_p3` -> Z_T / Z / Z[1/2] Q) on ALL `coker_viol_9 == 0` hits.
      4. Run 12T Numba F_3 `coker_viol_9`-guided flip-graph SA on obstructed promising hits (prioritizing
         record-beating or SOTA-matching ranks!) and lift any `coker_viol_9 == 0` breakthrough!
    """
    t0 = time.perf_counter()
    deadline = t0 + budget_s
    print(f"\n=== Track B: 3-adic Lifting & Repair of {os.path.basename(pool_path)} (budget={budget_s:.0f}s) ===", flush=True)

    if not os.path.exists(pool_path):
        print(f"  [SKIP] Pool file not found yet: {pool_path}", flush=True)
        return {"status": "missing_pool_file", "pool_path": pool_path, "schemes_loaded": 0}

    entries = _parse_f3_pool_entries(pool_path)
    print(f"  Loaded {len(entries)} unique F_3 schemes from {pool_path}", flush=True)

    # Sort entries by priority: record-beating gap over SOTA_Q first, then lowest rank per format
    def _priority_key(e: Dict[str, Any]) -> Tuple[int, int, int]:
        sota = SOTA_CHAR0_2026.get(e["format"])
        if sota is not None:
            gap_q = e["rank"] - sota["Q"]
            gap_zt = e["rank"] - sota["ZT"]
            return (0 if gap_q < 0 else (1 if gap_zt < 0 else 2), min(gap_q, gap_zt), e["rank"])
        return (3, 0, e["rank"])

    entries.sort(key=_priority_key)

    verified_f3 = 0
    invalid_f3 = 0
    zero_mod9_hits = 0
    lifted_count = 0
    lifted_q_count = 0
    lifted_zt_count = 0
    lifted_z_count = 0
    per_format_summary: Dict[str, Dict[str, Any]] = {}
    lift_logs: List[Dict[str, Any]] = []
    obstructed_promising: List[Tuple[int, Dict[str, Any]]] = []

    for e in entries:
        m, n, p = e["format"]
        R = e["rank"]
        fmt_str = f"{m}x{n}x{p}"
        ok_f3 = bool(_fast_verify_f3_u64_nb(m, n, p, e["up"], e["un"], e["vp"], e["vn"], e["wp"], e["wn"]))
        if not ok_f3:
            invalid_f3 += 1
            continue
        verified_f3 += 1

        o_supp = mod9_oracle_u64(m, n, p, e["up"], e["un"], e["vp"], e["vn"], e["wp"], e["wn"], support_restricted=True)
        o_unres = None
        if m * n * p <= 128 and not o_supp.is_mod9_liftable:
            o_unres = mod9_oracle_u64(m, n, p, e["up"], e["un"], e["vp"], e["vn"], e["wp"], e["wn"], support_restricted=False)

        best_ck9 = o_supp.coker_viol_9
        if o_unres is not None and o_unres.coker_viol_9 < best_ck9:
            best_ck9 = o_unres.coker_viol_9

        pf = per_format_summary.setdefault(
            fmt_str,
            {
                "format": [m, n, p],
                "sota_2026": SOTA_CHAR0_2026.get((m, n, p)),
                "schemes_verified_f3": 0,
                "min_rank_f3": 999999,
                "min_coker_viol_9": 999999,
                "zero_mod9_count": 0,
                "lifted_char0_count": 0,
                "min_lifted_rank_char0": None,
                "min_lifted_ring": None,
            },
        )
        pf["schemes_verified_f3"] += 1
        if R < pf["min_rank_f3"]:
            pf["min_rank_f3"] = R
        if best_ck9 < pf["min_coker_viol_9"]:
            pf["min_coker_viol_9"] = best_ck9

        sota_e = SOTA_CHAR0_2026.get((m, n, p))
        is_record_rank = sota_e is not None and (R < sota_e["Q"] or R < sota_e["ZT"])

        if best_ck9 == 0:
            zero_mod9_hits += 1
            pf["zero_mod9_count"] += 1
            # Lift immediately! (Use 64 trials on record ranks, 10 on non-record ranks to stay fast)
            trials_use = 64 if is_record_rank else 10
            if time.perf_counter() < deadline:
                lres = attempt_certified_p3_lift_and_save(
                    m,
                    n,
                    p,
                    e["up"],
                    e["un"],
                    e["vp"],
                    e["vn"],
                    e["wp"],
                    e["wn"],
                    source_meta={"pool_id": e["id"], "hash": e["hash"], **e["source_meta"]},
                    artifacts_dir=artifacts_dir,
                    records_saved=records_saved,
                    hensel_trials=trials_use,
                    label=e["id"],
                )
                lift_logs.append(lres)
                if lres["lifted"]:
                    lifted_count += 1
                    pf["lifted_char0_count"] += 1
                    if lres["lifted_ring"] == "Q":
                        lifted_q_count += 1
                    elif lres["lifted_ring"] == "ZT":
                        lifted_zt_count += 1
                    else:
                        lifted_z_count += 1
                    if pf["min_lifted_rank_char0"] is None or R < pf["min_lifted_rank_char0"]:
                        pf["min_lifted_rank_char0"] = R
                        pf["min_lifted_ring"] = lres["lifted_ring"]
        else:
            if is_record_rank or (sota_e is not None and R <= sota_e["ZT"] + 2):
                obstructed_promising.append((best_ck9, e))

    print(
        f"  [Track B Screen Done] verified_F3={verified_f3} invalid={invalid_f3} "
        f"zero_mod9={zero_mod9_hits} lifted_char0={lifted_count} (ZT={lifted_zt_count}, Z={lifted_z_count}, Q={lifted_q_count})",
        flush=True,
    )

    # Stage B2: Run 12T Numba F_3 `coker_viol_9`-guided flip-graph SA on top obstructed promising hits!
    repair_logs: List[Dict[str, Any]] = []
    obstructed_promising.sort(key=lambda x: (_priority_key(x[1]), x[0]))
    # Take up to 6 distinct (format, rank) groups to repair within remaining budget
    seen_fr: set[Tuple[Tuple[int, int, int], int]] = set()
    repair_queue: List[Tuple[int, Dict[str, Any]]] = []
    for ck9, e in obstructed_promising:
        key_fr = (e["format"], e["rank"])
        if key_fr not in seen_fr:
            seen_fr.add(key_fr)
            repair_queue.append((ck9, e))
        if len(repair_queue) >= 6:
            break

    if repair_queue and time.perf_counter() < deadline - 10.0:
        per_item_s = max(12.0, (deadline - time.perf_counter() - 5.0) / float(len(repair_queue)))
        for idx_r, (init_ck9, e) in enumerate(repair_queue):
            rem = deadline - time.perf_counter()
            if rem < 8.0:
                break
            m, n, p = e["format"]
            R = e["rank"]
            supp_only = (m * n * p) > 100
            nw = 24
            R_cap = R + 3
            Up_in = np.zeros((nw, R_cap), dtype=np.uint64)
            Un_in = np.zeros((nw, R_cap), dtype=np.uint64)
            Vp_in = np.zeros((nw, R_cap), dtype=np.uint64)
            Vn_in = np.zeros((nw, R_cap), dtype=np.uint64)
            Wp_in = np.zeros((nw, R_cap), dtype=np.uint64)
            Wn_in = np.zeros((nw, R_cap), dtype=np.uint64)
            R_in = np.full(nw, R, dtype=np.int32)
            for k in range(nw):
                Up_in[k, :R], Un_in[k, :R] = e["up"], e["un"]
                Vp_in[k, :R], Vn_in[k, :R] = e["vp"], e["vn"]
                Wp_in[k, :R], Wn_in[k, :R] = e["wp"], e["wn"]

            t_rep0 = time.perf_counter()
            rep_deadline = min(deadline - 3.0, t_rep0 + per_item_s)
            best_ck_rep = init_ck9
            rep_evals = 0
            rep_lifted = False
            iters_rd = 80 if m * n * p <= 128 else 16
            rd_r = 0
            while time.perf_counter() < rep_deadline and not rep_lifted:
                rd_r += 1
                Up_b, Un_b, Vp_b, Vn_b, Wp_b, Wn_b, Rb, ckb, evb = _f3_coker9_sa_batch_nb(
                    m, n, p, Up_in, Un_in, Vp_in, Vn_in, Wp_in, Wn_in, R_in, R, iters_rd, supp_only, np.uint64(800001 + idx_r * 997 + rd_r * 131)
                )
                rep_evals += int(np.sum(evb))
                order = np.argsort(ckb)
                top_w = int(order[0])
                if int(ckb[top_w]) < best_ck_rep:
                    best_ck_rep = int(ckb[top_w])
                # Feed top 6 walkers back into next round
                for k in range(nw):
                    src_w = int(order[k % 6])
                    rk = int(Rb[src_w])
                    Up_in[k, :rk], Un_in[k, :rk] = Up_b[src_w, :rk], Un_b[src_w, :rk]
                    Vp_in[k, :rk], Vn_in[k, :rk] = Vp_b[src_w, :rk], Vn_b[src_w, :rk]
                    Wp_in[k, :rk], Wn_in[k, :rk] = Wp_b[src_w, :rk], Wn_b[src_w, :rk]
                    R_in[k] = rk
                if int(ckb[top_w]) == 0:
                    rk = int(Rb[top_w])
                    print(f"  *** F3 REPAIR HIT coker_viol_9 == 0 on {m}x{n}x{p}:R={rk} -> FIRING L1-p3! ***", flush=True)
                    lres = attempt_certified_p3_lift_and_save(
                        m,
                        n,
                        p,
                        Up_b[top_w, :rk].copy(),
                        Un_b[top_w, :rk].copy(),
                        Vp_b[top_w, :rk].copy(),
                        Vn_b[top_w, :rk].copy(),
                        Wp_b[top_w, :rk].copy(),
                        Wn_b[top_w, :rk].copy(),
                        source_meta={"pool_id": e["id"], "repaired_f3_sa": True, **e["source_meta"]},
                        artifacts_dir=artifacts_dir,
                        records_saved=records_saved,
                        hensel_trials=64,
                        label=f"{e['id']}_f3_repaired",
                    )
                    lift_logs.append(lres)
                    if lres["lifted"]:
                        rep_lifted = True
                        lifted_count += 1
                        fmt_str = f"{m}x{n}x{p}"
                        per_format_summary[fmt_str]["lifted_char0_count"] += 1
                        if per_format_summary[fmt_str]["min_lifted_rank_char0"] is None or rk < per_format_summary[fmt_str]["min_lifted_rank_char0"]:
                            per_format_summary[fmt_str]["min_lifted_rank_char0"] = rk
                            per_format_summary[fmt_str]["min_lifted_ring"] = lres["lifted_ring"]
                        break

            rlog = {
                "id": e["id"],
                "format": [m, n, p],
                "rank": R,
                "init_coker_viol_9": init_ck9,
                "min_coker_viol_9": best_ck_rep,
                "evals": rep_evals,
                "lifted": rep_lifted,
                "elapsed_s": round(time.perf_counter() - t_rep0, 2),
            }
            repair_logs.append(rlog)
            print(f"  [B2 F3 repair {e['id']}] coker9: {init_ck9} -> {best_ck_rep}, evals={rep_evals}, lifted={rep_lifted} ({rlog['elapsed_s']}s)", flush=True)

    return {
        "status": "completed",
        "pool_path": pool_path,
        "schemes_loaded": len(entries),
        "schemes_verified_f3": verified_f3,
        "schemes_invalid_f3": invalid_f3,
        "zero_mod9_hits": zero_mod9_hits,
        "lifted_char0_total": lifted_count,
        "lifted_breakdown": {"ZT": lifted_zt_count, "Z": lifted_z_count, "Q": lifted_q_count},
        "per_format": per_format_summary,
        "f3_coker9_repair_runs": repair_logs,
        "lift_attempts_sample": lift_logs[:50],
        "total_lift_attempts": len(lift_logs),
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }


# =============================================================================
# 3. CLI Entry Point (Supports --mode push455 | ingest-f3 | all)
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 CPU-Only: 4x5x5:75 Knockout Push + 3-adic F3 Pool Lifting")
    parser.add_argument("--mode", choices=("all", "push455", "ingest-f3"), default="all")
    parser.add_argument("--push455-minutes", type=float, default=4.8, help="Budget (minutes) for Track A 4x5x5:75 push")
    parser.add_argument("--f3-minutes", type=float, default=4.5, help="Budget (minutes) for Track B F3 pool lifting")
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=os.path.join(_FMM_ROOT, "artifacts", "public_f2_seeds", "manifest.json"),
    )
    parser.add_argument(
        "--f3-pool-path",
        type=str,
        default=os.path.join(_FMM_ROOT, "artifacts", "f3_pool_phase3.json"),
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=os.path.join(_FMM_ROOT, "artifacts", "phase3_lifting_summary.json"),
    )
    args = parser.parse_args()

    artifacts_dir = os.path.join(_FMM_ROOT, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    t_start = time.perf_counter()

    summary: Dict[str, Any] = {}
    if os.path.exists(args.summary_path) and args.mode == "ingest-f3":
        with open(args.summary_path, "r", encoding="utf-8") as fh:
            summary = json.load(fh)

    records_saved: List[Dict[str, Any]] = list(summary.get("lifted_world_records", []))
    track_a_res = summary.get("track_a_4x5x5_R75_knockout")
    track_b_res = summary.get("track_b_f3_pool_3adic_lifting")

    if args.mode in ("all", "push455"):
        track_a_res = run_track_a_455_knockout(
            manifest_path=args.manifest_path,
            artifacts_dir=artifacts_dir,
            records_saved=records_saved,
            budget_s=max(30.0, args.push455_minutes * 60.0),
        )

    if args.mode in ("all", "ingest-f3"):
        track_b_res = run_track_b_f3_pool_lifting(
            pool_path=args.f3_pool_path,
            artifacts_dir=artifacts_dir,
            records_saved=records_saved,
            budget_s=max(30.0, args.f3_minutes * 60.0),
        )

    out_summary = {
        "phase": "Phase 3 (Final Shot): 3-adic Hensel Engine + 4x5x5:R=75 Knockout Push + GPU F_3 Pool Lifting",
        "hardware": "CPU-only (Numba 12T + 3-adic Hensel mod 3^40 + 64-bit 2-adic Hensel + Z3 5.1.0, ZERO GPU usage)",
        "track_a_4x5x5_R75_knockout": track_a_res,
        "track_b_f3_pool_3adic_lifting": track_b_res,
        "lifted_world_records": records_saved,
        "world_records_broken_count": len(records_saved),
        "elapsed_this_run_s": round(time.perf_counter() - t_start, 2),
    }
    with open(args.summary_path, "w", encoding="utf-8") as fh:
        json.dump(out_summary, fh, ensure_ascii=False, indent=2)

    print(
        f"\nPhase 3 ({args.mode}) COMPLETE -> {args.summary_path} "
        f"(world_records={len(records_saved)}, t={time.perf_counter() - t_start:.1f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

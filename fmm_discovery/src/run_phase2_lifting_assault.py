"""
Phase 2 (Option B — Track 2, CPU-Only — ZERO GPU usage)
Direct + Plus-Relaxed F2 -> Z_T / Z / Q Lifting Assault on Published 2023-2026 F2 World Records.

Hardware: 12-thread CPU (Numba LLVM + NumPy + Z3 5.1.0), ZERO torch/CUDA/GPU usage.
Targets (from `artifacts/public_f2_seeds/manifest.json` with exact URL + commit SHA + SHA-256 provenance):
  - Priority 1: 4x5x5 (base F2 R=73 vs char-0 SOTA 76; gap=3 -> direct R=73 + plus-relaxed R=74, 75)
  - Priority 2: 4x5x6 (base F2 R=89 x2 vs char-0 SOTA 90; gap=1 -> direct R=89 + 89->90->89 tunneling)
  - Priority 3: 4x4x8 (base F2 R=94 vs Q=95 / Z_T=96 -> direct R=94 + plus-relaxed R=95 -> Z_T)
  - Priority 4: 7x7x7 (base F2 R=245 vs Q=249 / Z_T=250; gap=4-5 -> direct R=245 + plus-relaxed R=246, 247, 248)
  - Priority 5: 4x4x5 (base F2 R=60 vs char-0 SOTA 61; gap=1 -> deep unres_coker_4 SA + F3/Z[1/2] diagnostic)
  - Bonus Calibration: 4x4x4 (base F2 R=47 -> plus-relaxed R=48 vs Q/C=48, Z_T=49)

Every candidate that reaches `unres_coker_4 == 0` or `supp_coker_4 == 0` immediately fires:
  - Stage L2: Z3 5.1.0 exact ternary sign lifter (`Z_T`),
  - Stage L1: 64-bit Linear Hensel lifting with 64+ nullspace kicks in ker(J1) (`steer_mode in {1,2,0}`)
    + Wang rational reconstruction (`Z` / `Q`),
  - Strict dual-path `brent_verifier.verify_scheme()` + 100x random integer + 2x2 non-commutative block matmul gate.
"""

from __future__ import annotations

import argparse
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
from dream_rsi_scripts.engine import LeanDAGPolicy, compute_max_tilted_objective  # noqa: E402
from dream_rsi_scripts.skill_state import BackwardSubGoalDecomposer, MutableSkillState  # noqa: E402
from dream_rsi_scripts.tree import DiscoveryTree, Node, Problem  # noqa: E402
from hensel_lifter import (  # noqa: E402
    _mod4_oracle_support_nb,
    _mod4_oracle_unrestricted_nb,
    _popcount64,
    fast_verify_f2_u64,
    linear_hensel_lift,
    mod4_oracle_batch_u64,
    mod4_oracle_batch_unrestricted_u64,
    mod4_oracle_u64,
    scheme_to_u64_f2,
    z3_ternary_lift,
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


# Verified November 2026 characteristic-0 world records (from sota_targets.json / 01_lille_sota_and_targets.md)
SOTA_CHAR0_2026: Dict[Tuple[int, int, int], Dict[str, int]] = {
    (4, 5, 5): {"F2": 73, "ZT": 76, "Z": 76, "Q": 76},
    (4, 5, 6): {"F2": 89, "ZT": 90, "Z": 90, "Q": 90},
    (4, 4, 8): {"F2": 94, "ZT": 96, "Z": 96, "Q": 95},
    (7, 7, 7): {"F2": 245, "ZT": 250, "Z": 250, "Q": 249},
    (4, 4, 5): {"F2": 60, "ZT": 61, "Z": 61, "Q": 61},
    (4, 4, 4): {"F2": 47, "ZT": 49, "Z": 49, "Q": 48},
}


# =============================================================================
# 1. Exact GL(F2) Shearing & Non-Regressing Support-Coker Polishing
# =============================================================================


@nb.njit(cache=True)
def _apply_gl_shear_nb(
    m: int,
    n: int,
    p: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    R: int,
    group_id: int,
    r1: int,
    r2: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact elementary GL(F2) basis change preserving the matrix multiplication tensor T_{m,n,p}:
      group 0 (GL_m): row r2 ^= row r1 in U, row r1 ^= row r2 in W
      group 1 (GL_n): col r1 ^= col r2 in U, row r2 ^= row r1 in V
      group 2 (GL_p): col r1 ^= col r2 in V, col r2 ^= col r1 in W
    Every such shear lifts to SL(Z), so unres_coker_4 == 0 is invariant while supp_coker_4 & nnz drop!
    """
    u2 = u[:R].copy()
    v2 = v[:R].copy()
    w2 = w[:R].copy()
    if group_id == 0:
        for r in range(R):
            ur = u2[r]
            for j in range(n):
                if ((ur >> np.uint64(r1 * n + j)) & np.uint64(1)) != 0:
                    ur ^= np.uint64(1) << np.uint64(r2 * n + j)
            u2[r] = ur
            wr = w2[r]
            for k in range(p):
                if ((wr >> np.uint64(r2 * p + k)) & np.uint64(1)) != 0:
                    wr ^= np.uint64(1) << np.uint64(r1 * p + k)
            w2[r] = wr
    elif group_id == 1:
        for r in range(R):
            ur = u2[r]
            for i in range(m):
                if ((ur >> np.uint64(i * n + r2)) & np.uint64(1)) != 0:
                    ur ^= np.uint64(1) << np.uint64(i * n + r1)
            u2[r] = ur
            vr = v2[r]
            for k in range(p):
                if ((vr >> np.uint64(r1 * p + k)) & np.uint64(1)) != 0:
                    vr ^= np.uint64(1) << np.uint64(r2 * p + k)
            v2[r] = vr
    else:
        for r in range(R):
            vr = v2[r]
            for j in range(n):
                if ((vr >> np.uint64(j * p + r2)) & np.uint64(1)) != 0:
                    vr ^= np.uint64(1) << np.uint64(j * p + r1)
            v2[r] = vr
            wr = w2[r]
            for i in range(m):
                if ((wr >> np.uint64(i * p + r1)) & np.uint64(1)) != 0:
                    wr ^= np.uint64(1) << np.uint64(i * p + r2)
            w2[r] = wr
    return u2, v2, w2


def gl_polish_strict(
    m: int,
    n: int,
    p: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    max_passes: int = 3,
    sparsify_first: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Strictly non-regressing GL(F2) shear polisher minimizing `(supp_coker_4, nnz)`.
    If `sparsify_first=True`, runs an initial nnz-only sparsification pass and keeps
    whichever of (original, sparsified) has lower `supp_coker_4` before greedy coker polishing!
    """
    R = len(u)
    u_cur, v_cur, w_cur = u.copy(), v.copy(), w.copy()
    best_ck = int(_mod4_oracle_support_nb(m, n, p, u_cur, v_cur, w_cur)[0])
    best_nnz = int(_scheme_nnz_nb(u_cur, v_cur, w_cur, R))

    if sparsify_first:
        us, vs, ws = u_cur.copy(), v_cur.copy(), w_cur.copy()
        nz_s = best_nnz
        for _ in range(5):
            imp = False
            for g, dim in ((0, m), (1, n), (2, p)):
                for r1 in range(dim):
                    for r2 in range(dim):
                        if r1 == r2:
                            continue
                        ut, vt, wt = _apply_gl_shear_nb(m, n, p, us, vs, ws, R, g, r1, r2)
                        nz = int(_scheme_nnz_nb(ut, vt, wt, R))
                        if nz < nz_s:
                            us, vs, ws = ut, vt, wt
                            nz_s = nz
                            imp = True
            if not imp:
                break
        ck_s = int(_mod4_oracle_support_nb(m, n, p, us, vs, ws)[0])
        if ck_s < best_ck or (ck_s == best_ck and nz_s < best_nnz):
            u_cur, v_cur, w_cur = us, vs, ws
            best_ck, best_nnz = ck_s, nz_s

    for _ in range(max_passes):
        imp = False
        for g, dim in ((0, m), (1, n), (2, p)):
            for r1 in range(dim):
                for r2 in range(dim):
                    if r1 == r2:
                        continue
                    ut, vt, wt = _apply_gl_shear_nb(m, n, p, u_cur, v_cur, w_cur, R, g, r1, r2)
                    ck = int(_mod4_oracle_support_nb(m, n, p, ut, vt, wt)[0])
                    if ck < best_ck or (ck == best_ck and int(_scheme_nnz_nb(ut, vt, wt, R)) < best_nnz):
                        u_cur, v_cur, w_cur = ut, vt, wt
                        best_ck = ck
                        best_nnz = int(_scheme_nnz_nb(u_cur, v_cur, w_cur, R))
                        imp = True
                        if best_ck == 0:
                            return u_cur, v_cur, w_cur, 0
        if not imp:
            break
    return u_cur, v_cur, w_cur, best_ck


# =============================================================================
# 2. 12-Thread Numba Simulated Annealing on `unres_coker_4` / `supp_coker_4`
# =============================================================================


@nb.njit(parallel=True, cache=True)
def _coker_sa_batch_nb(
    m: int,
    n: int,
    p: int,
    U_in: np.ndarray,
    V_in: np.ndarray,
    W_in: np.ndarray,
    R_in: np.ndarray,
    R_target: int,
    n_iters: int,
    max_burst: int,
    use_unres: bool,
    seed_base: np.uint64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    12-thread multi-temperature Simulated Annealing in the F2 flip-graph directly minimizing:
      - `unres_coker_4` (if `use_unres=True`) -> exact invariant mod-4 liftability into Z/Q, OR
      - `supp_coker_4` (if `use_unres=False`) -> ultra-fast support-restricted mod-4 liftability into Z_T!
    Multi-strategy walker portfolio (24 walkers):
      - Walkers 0..7: Cold greedy exploiters (T: 1.5 -> 0.12, short bursts 1..max_burst//2)
      - Walkers 8..15: Balanced SA explorers (T: 6.0 -> 0.35, variable bursts + 1/3 bridge rate)
      - Walkers 16..23: Hot tunneling diversifiers (T: 16.0 -> 0.8, 1/2 plus->flip->reduce bridges + occasional GL shear!)
    """
    nw = U_in.shape[0]
    R_cap = U_in.shape[1]

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

        state = seed_base ^ (np.uint64(wid + 31) * np.uint64(0x9E3779B97F4A7C15))
        if state == 0:
            state = np.uint64(17)

        u_try = np.empty(R_cap, dtype=np.uint64)
        v_try = np.empty(R_cap, dtype=np.uint64)
        w_try = np.empty(R_cap, dtype=np.uint64)
        ej_i = np.empty(4096, dtype=np.int32)
        ej_j = np.empty(4096, dtype=np.int32)
        ej_m = np.empty(4096, dtype=np.int8)
        deg = np.zeros((3, R_cap), dtype=np.int32)
        nbr = np.empty((3, R_cap, 16), dtype=np.int32)

        tier = 0 if wid < 8 else (1 if wid < 16 else 2)
        t_hi = 1.5 if tier == 0 else (6.0 if tier == 1 else 16.0)
        t_lo = 0.12 if tier == 0 else (0.35 if tier == 1 else 0.80)
        bridge_mod = 4 if tier == 0 else (3 if tier == 1 else 2)
        burst_lim = max(2, max_burst // 2) if tier == 0 else max_burst

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

            # Option A: If optimizing supp_coker_4, occasionally try a random GL(F2) shear (preserves R & ne!)
            did_gl = False
            if (not use_unres) and (it % 7 == 6):
                state = _rng_next(state)
                g_id = np.int64(state % np.uint64(3))
                dim_g = m if g_id == 0 else (n if g_id == 1 else p)
                if dim_g >= 2:
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
                do_bridge = (ne == 0 or (it % bridge_mod == bridge_mod - 1)) and (R_try + 1 < R_cap)
                if do_bridge:
                    R_try, state = _targeted_expand_nb(u_try, v_try, w_try, R_try, state)
                    ne = _rebuild_edges_nb(u_try, v_try, w_try, R_try, ej_i, ej_j, ej_m)

                state = _rng_next(state)
                if tier == 0 and (not do_bridge) and cur_ck <= 40:
                    # Fine-grained 1-2 flip local descent when close to coker=0!
                    nfl = np.int64(1) if ((state & np.uint64(3)) != 0) else np.int64(2)
                else:
                    nfl = (1 + np.int64(state % np.uint64(burst_lim))) * (3 if do_bridge else 1)
                for fl in range(nfl):
                    if ne == 0:
                        break
                    found = False
                    si = np.int64(0)
                    sj = np.int64(0)
                    sm = np.int64(0)
                    if do_bridge and fl >= 6 and (fl & 3) == 0:
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
                        if do_bridge and R_try <= R_target:
                            break
                    else:
                        ne = _update_edges_after_flip_nb(u_try, v_try, w_try, R_try, sm, si, sj, ej_i, ej_j, ej_m, ne)

            if R_try > R_target:
                continue
            # Preserve plus-relaxation budget: if flips collapsed R below R_target, re-expand back to R_target!
            while R_try < R_target and R_try + 1 < R_cap:
                R_try, state = _targeted_expand_nb(u_try, v_try, w_try, R_try, state)
                if R_try > R_target:
                    break
            if R_try > R_target:
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

                if (
                    (cur_ck < best_ck)
                    or (cur_ck == best_ck and R_cur < best_R)
                    or (cur_ck == best_ck and R_cur == best_R and cur_nnz < best_nnz)
                ):
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


# =============================================================================
# 3. Plus-Relaxation Seed Generator (R -> R + k via Plus & Disjoint Split)
# =============================================================================


def make_plus_relaxed_seeds(
    m: int,
    n: int,
    p: int,
    base_seeds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    extra_ranks: int,
    n_out: int = 24,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generate `n_out` diverse rank <= (R_base + extra_ranks) F2 schemes from `base_seeds`:
      - If extra_ranks == 0: returns base seeds + random flip-walked & GL-polished variants!
      - If extra_ranks > 0: applies exact F2 `plus` moves AND disjoint-support `split` moves
        (+ short mixing flips) so walkers explore genuinely distinct relaxed flip-graph basins!
    """
    rng = np.random.default_rng(seed)
    out: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    seen: set[int] = set()

    for idx in range(n_out * 3):
        if len(out) >= n_out:
            break
        bu, bv, bw = base_seeds[idx % len(base_seeds)]
        R0 = len(bu)
        R_cap = R0 + extra_ranks + 4
        u = np.zeros(R_cap, dtype=np.uint64)
        v = np.zeros(R_cap, dtype=np.uint64)
        w = np.zeros(R_cap, dtype=np.uint64)
        u[:R0], v[:R0], w[:R0] = bu, bv, bw
        R = R0

        if extra_ranks > 0:
            for ex in range(extra_ranks):
                # Alternate between sparse-pair plus (65%) and balanced disjoint-support split (35%)
                if idx % 3 == 2:
                    # Disjoint half-split of a high-weight term
                    weights = [
                        int(_popcount64(u[r])) + int(_popcount64(v[r])) + int(_popcount64(w[r])) for r in range(R)
                    ]
                    top_r = int(np.argsort(weights)[-1 - (int(rng.integers(0, min(8, R))))])
                    ax = int(rng.integers(0, 3))
                    arr = u if ax == 0 else (v if ax == 1 else w)
                    mask = int(arr[top_r])
                    bits = [b for b in range(64) if (mask >> b) & 1]
                    if len(bits) >= 2:
                        rng.shuffle(bits)
                        half = 0
                        for b in bits[: len(bits) // 2]:
                            half |= 1 << b
                        u[R], v[R], w[R] = u[top_r], v[top_r], w[top_r]
                        if ax == 0:
                            u[top_r], u[R] = np.uint64(half), np.uint64(mask ^ half)
                        elif ax == 1:
                            v[top_r], v[R] = np.uint64(half), np.uint64(mask ^ half)
                        else:
                            w[top_r], w[R] = np.uint64(half), np.uint64(mask ^ half)
                        R += 1
                    else:
                        i = int(rng.integers(0, R))
                        j = int(rng.integers(0, R - 1))
                        if j >= i:
                            j += 1
                        R = _apply_plus_nb(u, v, w, R, i, j, int(rng.integers(0, 6)))
                else:
                    # Plus move (sample 6 pairs, pick lowest XOR weight on half of seeds for sparsity)
                    best_i, best_j, best_w = 0, 1, 999999
                    trials = 6 if (idx & 1) == 0 else 1
                    for _ in range(trials):
                        i = int(rng.integers(0, R))
                        j = int(rng.integers(0, R - 1))
                        if j >= i:
                            j += 1
                        pw = (
                            int(_popcount64(u[i] ^ u[j]))
                            + int(_popcount64(v[i] ^ v[j]))
                            + int(_popcount64(w[i] ^ w[j]))
                        )
                        if pw < best_w:
                            best_w, best_i, best_j = pw, i, j
                    R = _apply_plus_nb(u, v, w, R, best_i, best_j, int(rng.integers(0, 6)))

        # Apply short mixing flips on seeds > len(base_seeds)
        if idx >= len(base_seeds):
            ei = np.empty(4096, dtype=np.int32)
            ej = np.empty(4096, dtype=np.int32)
            em = np.empty(4096, dtype=np.int8)
            ne = _rebuild_edges_nb(u, v, w, R, ei, ej, em)
            n_mix = int(rng.integers(3, 25))
            for _ in range(n_mix):
                if ne == 0:
                    break
                ek = int(rng.integers(0, ne))
                si, sj, sm = int(ei[ek]), int(ej[ek]), int(em[ek])
                if rng.integers(0, 2):
                    si, sj = sj, si
                if sm == 0:
                    v[si] ^= v[sj]
                    w[sj] ^= w[si]
                elif sm == 1:
                    u[si] ^= u[sj]
                    w[sj] ^= w[si]
                else:
                    u[si] ^= u[sj]
                    v[sj] ^= v[si]
                old_R = R
                R = _check_local_reduce_nb(u, v, w, R, si, sj)
                if R < old_R:
                    R = _group_lin_reduce_nb(u, v, w, R)
                    ne = _rebuild_edges_nb(u, v, w, R, ei, ej, em)
                else:
                    ne = _update_edges_after_flip_nb(u, v, w, R, sm, si, sj, ei, ej, em, ne)

        uc, vc, wc = u[:R].copy(), v[:R].copy(), w[:R].copy()
        if fast_verify_f2_u64(m, n, p, uc, vc, wc):
            h = int(_scheme_orbit_hash_nb(uc, vc, wc, R))
            if h not in seen:
                seen.add(h)
                out.append((uc, vc, wc))

    return out


# =============================================================================
# 4. Airtight Lifting Cascade & Non-Commutative World-Record Certification Gate
# =============================================================================


def _verify_noncommutative_and_random_int(scheme: BilinearScheme, n_int: int = 100, n_blk: int = 20) -> bool:
    """
    Extra defense-in-depth verification beyond `brent_verifier.verify_scheme()`:
      1. 100x random large integer matrices in [-1000, 1000]
      2. 20x random non-commuting 2x2 integer block matrices M_2(Z) (proves genuine non-commutative FMM!)
    """
    rng = np.random.default_rng(20261122)
    m, n, p, R = scheme.m, scheme.n, scheme.p, scheme.rank

    if scheme.ring == "Z":
        U = np.array(scheme.U, dtype=object)
        V = np.array(scheme.V, dtype=object)
        W = np.array(scheme.W, dtype=object)
    else:
        U = np.array(scheme.U, dtype=object)
        V = np.array(scheme.V, dtype=object)
        W = np.array(scheme.W, dtype=object)

    for _ in range(n_int):
        A = rng.integers(-1000, 1001, size=(m, n), dtype=np.int64).astype(object)
        B = rng.integers(-1000, 1001, size=(n, p), dtype=np.int64).astype(object)
        C_ref = A @ B
        a_flat = A.reshape(-1)
        b_flat = B.reshape(-1)
        m_vals = (U @ a_flat) * (V @ b_flat)
        C_got = (W.T @ m_vals).reshape(m, p)
        if not np.array_equal(C_got, C_ref):
            return False

    # Non-commutative 2x2 integer matrix entries!
    for _ in range(n_blk):
        A = rng.integers(-20, 21, size=(m, n, 2, 2), dtype=np.int64).astype(object)
        B = rng.integers(-20, 21, size=(n, p, 2, 2), dtype=np.int64).astype(object)
        # Reference block matmul: C[i,k] = sum_j A[i,j] @ B[j,k]
        C_ref = np.zeros((m, p, 2, 2), dtype=object)
        for i in range(m):
            for k in range(p):
                acc = np.zeros((2, 2), dtype=object)
                for j in range(n):
                    acc = acc + (A[i, j] @ B[j, k])
                C_ref[i, k] = acc

        A_flat = A.reshape(m * n, 2, 2)
        B_flat = B.reshape(n * p, 2, 2)
        C_got = np.zeros((m * p, 2, 2), dtype=object)
        for r in range(R):
            ar = np.zeros((2, 2), dtype=object)
            for idx in range(m * n):
                if U[r, idx] != 0:
                    ar = ar + U[r, idx] * A_flat[idx]
            br = np.zeros((2, 2), dtype=object)
            for idx in range(n * p):
                if V[r, idx] != 0:
                    br = br + V[r, idx] * B_flat[idx]
            pr = ar @ br  # NON-COMMUTATIVE product!
            for idx in range(m * p):
                if W[r, idx] != 0:
                    C_got[idx] = C_got[idx] + W[r, idx] * pr
        if not np.array_equal(C_got.reshape(m, p, 2, 2), C_ref):
            return False

    return True


def attempt_certified_lift_and_save(
    m: int,
    n: int,
    p: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    provenance: Dict[str, Any],
    artifacts_dir: str,
    records_saved: List[Dict[str, Any]],
    z3_timeout_s: float = 15.0,
    hensel_trials: int = 64,
    label: str = "",
) -> Dict[str, Any]:
    """
    Fire FULL Stage L2 (Z3 Z_T) + Stage L1 (64-bit Linear Hensel with 64 nullspace kicks in ker(J1)
    + Wang rational reconstruction) on any `supp_coker_4 == 0` or `unres_coker_4 == 0` hit!
    Certifies via `brent_verifier.verify_scheme()` + 100x int + 20x non-commutative M_2(Z) matmul!
    """
    R = len(u)
    o_s = mod4_oracle_u64(m, n, p, u, v, w, support_restricted=True)
    u_ck: Optional[int] = None
    if m * n * p <= 128:
        o_u = mod4_oracle_u64(m, n, p, u, v, w, support_restricted=False)
        u_ck = o_u.coker_viol_4

    info: Dict[str, Any] = {
        "label": label,
        "format": [m, n, p],
        "rank": R,
        "supp_coker_4": o_s.coker_viol_4,
        "unres_coker_4": u_ck,
        "lifted": False,
        "lifted_ring": None,
        "status": "obstructed_mod4",
        "attempts": [],
    }

    # If unres_coker_4 == 0 but supp_coker_4 > 0, first try GL(F2) polishing to see if support lifts!
    u_work, v_work, w_work = u.copy(), v.copy(), w.copy()
    supp_ck = o_s.coker_viol_4
    if supp_ck > 0 and u_ck == 0:
        u_work, v_work, w_work, supp_ck = gl_polish_strict(m, n, p, u_work, v_work, w_work, max_passes=3)
        info["supp_coker_4_after_gl"] = supp_ck

    lifted_scheme: Optional[BilinearScheme] = None
    lifted_method = ""

    # 1. Stage L2: Z3 exact ternary Z_T (when supp_coker_4 == 0)
    if supp_ck == 0:
        z3_res = z3_ternary_lift((m, n, p, u_work, v_work, w_work), timeout_s=z3_timeout_s, force_z3=False, name=label)
        info["attempts"].append(
            {"stage": "L2_Z3_ZT", "method": z3_res.method, "status": z3_res.status, "ms": z3_res.elapsed_ms}
        )
        if z3_res.success and z3_res.scheme is not None:
            lifted_scheme = z3_res.scheme
            lifted_method = z3_res.method
        else:
            h_s = linear_hensel_lift(
                (m, n, p, u_work, v_work, w_work),
                support_restricted=True,
                target_bits=64,
                max_trials=hensel_trials,
                name=label,
            )
            info["attempts"].append(
                {
                    "stage": "L1_Hensel_supp",
                    "status": h_s.status,
                    "max_level": h_s.max_level_reached,
                    "ms": h_s.elapsed_ms,
                }
            )
            if h_s.success and h_s.scheme is not None:
                lifted_scheme = h_s.scheme
                lifted_method = h_s.method

    # 2. Stage L1: Unrestricted 64-bit Linear Hensel + 64 nullspace kicks in ker(J1) + Wang!
    if lifted_scheme is None and (u_ck == 0 or supp_ck == 0) and (m * n * p <= 128):
        h_u = linear_hensel_lift(
            (m, n, p, u_work, v_work, w_work),
            support_restricted=False,
            target_bits=64,
            max_trials=hensel_trials,
            name=label,
        )
        info["attempts"].append(
            {
                "stage": "L1_Hensel_unres",
                "status": h_u.status,
                "max_level": h_u.max_level_reached,
                "ms": h_u.elapsed_ms,
                "details": h_u.details,
            }
        )
        if h_u.success and h_u.scheme is not None:
            lifted_scheme = h_u.scheme
            lifted_method = h_u.method

    if lifted_scheme is None:
        return info

    # AIRTIGHT CERTIFICATION GATE!
    rep = verify_scheme(lifted_scheme)
    if not rep.valid:
        raise RuntimeError(f"CRITICAL: lifted scheme {label} failed brent_verifier!")
    if not _verify_noncommutative_and_random_int(lifted_scheme, n_int=100, n_blk=20):
        raise RuntimeError(f"CRITICAL: lifted scheme {label} failed 100x int / M_2(Z) non-commutativity check!")

    ring_tag = "ZT" if (lifted_scheme.ring == "Z" and rep.is_ternary) else lifted_scheme.ring
    info["lifted"] = True
    info["lifted_ring"] = ring_tag
    info["status"] = f"verified_{ring_tag}"
    info["max_abs_coeff"] = str(rep.max_abs_coeff)
    info["is_ternary"] = rep.is_ternary
    info["nnz_total"] = rep.additive_complexity["nnz_total"]
    info["naive_additions"] = rep.additive_complexity["naive_additions_total"]

    # Check if 2026 char-0 SOTA is strictly beaten!
    sota = SOTA_CHAR0_2026.get(tuple(sorted((m, n, p))))
    beaten: List[str] = []
    if sota is not None:
        if ring_tag == "ZT" and R < sota["ZT"]:
            beaten.append(f"ZT ({R} < {sota['ZT']})")
        if lifted_scheme.ring == "Z" and R < sota["Z"]:
            beaten.append(f"Z ({R} < {sota['Z']})")
        if lifted_scheme.ring in ("Z", "Q") and R < sota["Q"]:
            beaten.append(f"Q ({R} < {sota['Q']})")

    if beaten:
        fname = f"lifted_record_{m}x{n}x{p}_R{R}_{ring_tag}.json"
        out_path = os.path.join(artifacts_dir, fname)
        rec_prov = {
            "base_f2_source": provenance.get("source_repo"),
            "paper": provenance.get("paper"),
            "url": provenance.get("url"),
            "commit_sha": provenance.get("commit_sha"),
            "base_f2_sha256": provenance.get("canonical_sha256"),
            "lifting_method": lifted_method,
            "ring": ring_tag,
            "max_abs_coeff": str(rep.max_abs_coeff),
            "is_ternary": rep.is_ternary,
            "nnz": rep.additive_complexity["nnz_total"],
            "adds": rep.additive_complexity["naive_additions_total"],
            "beaten_2026_sota": beaten,
            "verified_brent": True,
            "verified_100x_int_and_M2Z_noncommutative": True,
        }
        lifted_scheme.metadata.update({"provenance": rec_prov})
        lifted_scheme.save_json(out_path)
        rec_entry = {"file": out_path, "format": [m, n, p], "rank": R, "canonical_sha256": rep.canonical_sha256, **rec_prov}
        records_saved.append(rec_entry)
        info["world_record_saved"] = out_path
        print(f"  *** CERTIFIED 2026 WORLD RECORD SAVED: {fname} ({', '.join(beaten)}) ***", flush=True)

    return info


# =============================================================================
# 5. Per-Target SLEEP-DAG-RSI Lifting Assault (Direct + Plus-Relaxed)
# =============================================================================


def run_target_lifting_assault(
    target_label: str,
    m: int,
    n: int,
    p: int,
    R_target: int,
    extra_ranks: int,
    polished_bases: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    provenance: Dict[str, Any],
    budget_s: float,
    use_unres_sa: bool,
    artifacts_dir: str,
    records_saved: List[Dict[str, Any]],
    seed_offset: int,
) -> Dict[str, Any]:
    """
    Run multi-round SLEEP-DAG-RSI v8 (`DiscoveryTree` + `LeanDAGPolicy(tau=4.0)` +
    `BackwardSubGoalDecomposer` + `MutableSkillState`) Simulated Annealing minimizing
    `unres_coker_4` (or `supp_coker_4`) at exact rank <= R_target!
    """
    t0 = time.perf_counter()
    deadline = t0 + budget_s
    fmt_key = f"{m}x{n}x{p}"
    mode_tag = "UNRES" if use_unres_sa else "SUPP"
    print(
        f"\n=== [{target_label}] {fmt_key} R_target={R_target} (+{extra_ranks} relax) "
        f"mode={mode_tag} budget={budget_s:.1f}s ===",
        flush=True,
    )

    init_seeds = make_plus_relaxed_seeds(
        m, n, p, polished_bases, extra_ranks=extra_ranks, n_out=24, seed=seed_offset
    )
    if not init_seeds:
        init_seeds = polished_bases

    # SLEEP-DAG-RSI v8 structures
    tree = DiscoveryTree(task=f"p2_lift_{target_label}")
    policy = LeanDAGPolicy(entropic_tau=4.0)
    decomposer = BackwardSubGoalDecomposer()
    skill_state = MutableSkillState(task=f"p2_lift_{target_label}")
    problem = Problem()

    # Archive: hash -> (primary_ck, supp_ck, unres_ck, R, nnz, u, v, w)
    archive: Dict[int, Tuple[int, int, Optional[int], int, int, np.ndarray, np.ndarray, np.ndarray]] = {}
    lift_logs: List[Dict[str, Any]] = []
    total_evals = 0
    total_screened = 0
    best_supp_ck = 999999
    best_unres_ck: Optional[int] = None
    hit_zero = False

    def _register(uc: np.ndarray, vc: np.ndarray, wc: np.ndarray, br: int, att: int, known_ck: Optional[int] = None) -> int:
        nonlocal best_supp_ck, best_unres_ck, total_screened, hit_zero
        rr = len(uc)
        h = int(_scheme_orbit_hash_nb(uc, vc, wc, rr))
        if h in archive:
            return archive[h][0]
        if not fast_verify_f2_u64(m, n, p, uc, vc, wc):
            return 999999
        o_s = mod4_oracle_u64(m, n, p, uc, vc, wc, support_restricted=True)
        s_ck = o_s.coker_viol_4
        u_ck: Optional[int] = None
        if use_unres_sa:
            if known_ck is not None:
                u_ck = int(known_ck)
            else:
                u_ck = mod4_oracle_u64(m, n, p, uc, vc, wc, support_restricted=False).coker_viol_4
        elif s_ck == 0 or (m * n * p <= 128 and s_ck <= max(25, best_supp_ck)):
            u_ck = mod4_oracle_u64(m, n, p, uc, vc, wc, support_restricted=False).coker_viol_4

        total_screened += 1
        prim = u_ck if (use_unres_sa and u_ck is not None) else s_ck
        if s_ck < best_supp_ck:
            best_supp_ck = s_ck
        if u_ck is not None and (best_unres_ck is None or u_ck < best_unres_ck):
            best_unres_ck = u_ck

        archive[h] = (prim, s_ck, u_ck, rr, o_s.nnz_total, uc.copy(), vc.copy(), wc.copy())

        sc_val = -float(prim) - 0.1 * float(s_ck) - 0.001 * float(o_s.nnz_total)
        b_scores = [
            float(50.5 - prim),
            float(15.5 - prim),
            float(0.5 - (u_ck if u_ck is not None else 999)),
            float(0.5 - s_ck),
        ]
        tree.add(
            Node(
                id=f"{target_label}_n{len(tree.nodes)}" if len(tree.nodes) > 0 else "root",
                parent_id=None,
                branch=br,
                attempt=att,
                score=sc_val,
                bucket_scores=b_scores,
                pass_mask=[1 if x > 0 else 0 for x in b_scores],
                meta={"rank": rr, "supp_coker_4": s_ck, "unres_coker_4": u_ck, "nnz": o_s.nnz_total},
            )
        )

        # IMMEDIATE LIFT TRIGGER on ANY zero obstruction!
        if s_ck == 0 or u_ck == 0:
            print(
                f"  *** ZERO OBSTRUCTION HIT on {target_label} (R={rr}, supp_ck={s_ck}, unres_ck={u_ck}) -> FIRING L1+L2! ***",
                flush=True,
            )
            lres = attempt_certified_lift_and_save(
                m,
                n,
                p,
                uc,
                vc,
                wc,
                provenance=provenance,
                artifacts_dir=artifacts_dir,
                records_saved=records_saved,
                z3_timeout_s=15.0,
                hensel_trials=64,
                label=f"{target_label}_R{rr}_b{br}",
            )
            lift_logs.append(lres)
            if lres["lifted"]:
                hit_zero = True
        return prim

    for idx_s, (su, sv, sw) in enumerate(init_seeds):
        _register(su, sv, sw, 0, idx_s)
        if hit_zero:
            break

    n_walkers = 24
    R_cap = R_target + 4
    rng_py = np.random.default_rng(seed_offset + 777)

    # Calibrate per-round iterations so each round takes ~5-12 seconds across all 12 threads
    if use_unres_sa:
        iters_per_round = 140 if m * n * p <= 80 else (42 if m * n * p <= 100 else 18)
        max_burst = 14
    else:
        iters_per_round = 220 if m * n * p <= 128 else 35
        max_burst = 20

    rd = 0
    while time.perf_counter() < deadline and not hit_zero:
        rd += 1
        ranked = sorted(archive.values(), key=lambda x: (x[0], x[1], x[4]))
        elites = ranked[: min(16, len(ranked))]
        e_scores = [-float(e[0]) - 0.1 * float(e[1]) for e in elites]
        tilt = compute_max_tilted_objective(e_scores, tau=policy.entropic_tau, normalize_by_scale=True)
        probs = np.array(tilt["tilted_weights_normalized"], dtype=np.float64)
        probs = probs / np.sum(probs)

        U_in = np.zeros((n_walkers, R_cap), dtype=np.uint64)
        V_in = np.zeros((n_walkers, R_cap), dtype=np.uint64)
        W_in = np.zeros((n_walkers, R_cap), dtype=np.uint64)
        R_in = np.zeros(n_walkers, dtype=np.int32)

        # Every 4th round inject 6 freshly relaxed/perturbed restarts to avoid getting trapped in one valley!
        fresh_restarts = (
            make_plus_relaxed_seeds(m, n, p, polished_bases, extra_ranks=extra_ranks, n_out=6, seed=seed_offset + rd * 131)
            if (rd % 4 == 0)
            else []
        )
        for k in range(n_walkers):
            if fresh_restarts and k >= n_walkers - len(fresh_restarts):
                fu, fv, fw = fresh_restarts[k - (n_walkers - len(fresh_restarts))]
                rr = len(fu)
                U_in[k, :rr], V_in[k, :rr], W_in[k, :rr] = fu, fv, fw
                R_in[k] = rr
            elif k < 6:
                el = elites[k % min(4, len(elites))]
                rr = el[3]
                U_in[k, :rr], V_in[k, :rr], W_in[k, :rr] = el[5], el[6], el[7]
                R_in[k] = rr
            else:
                el = elites[int(rng_py.choice(len(elites), p=probs))]
                rr = el[3]
                U_in[k, :rr], V_in[k, :rr], W_in[k, :rr] = el[5], el[6], el[7]
                R_in[k] = rr

        Ub, Vb, Wb, Rb, ckb, evb = _coker_sa_batch_nb(
            m,
            n,
            p,
            U_in,
            V_in,
            W_in,
            R_in,
            R_target=R_target,
            n_iters=iters_per_round,
            max_burst=max_burst,
            use_unres=use_unres_sa,
            seed_base=np.uint64(seed_offset + rd * 99991),
        )
        total_evals += int(np.sum(evb))

        # Register top 8 diverse walker outputs + GL-polish the round winner!
        order_w = np.argsort(ckb)
        for rank_i, wi in enumerate(order_w[:8]):
            wi_i = int(wi)
            rw = int(Rb[wi_i])
            uc, vc, wc = Ub[wi_i, :rw].copy(), Vb[wi_i, :rw].copy(), Wb[wi_i, :rw].copy()
            if rank_i == 0 and not use_unres_sa:
                uc, vc, wc, _ = gl_polish_strict(m, n, p, uc, vc, wc, max_passes=2, sparsify_first=False)
                _register(uc, vc, wc, rd, wi_i, known_ck=None)
            else:
                _register(uc, vc, wc, rd, wi_i, known_ck=int(ckb[wi_i]) if use_unres_sa else None)
            if hit_zero:
                break

        if rd % 2 == 0 or rd == 1 or hit_zero:
            print(
                f"  [{target_label} r{rd:2d}] t={time.perf_counter()-t0:5.1f}s evals={total_evals:6d} "
                f"best_unres_ck={best_unres_ck} best_supp_ck={best_supp_ck} archive={len(archive)}",
                flush=True,
            )

    # Final unrestricted audit on top 6 lowest-supp_coker schemes if we ran in SUPP mode & mnp <= 128!
    if (not use_unres_sa) and (not hit_zero) and m * n * p <= 128:
        top_supp = sorted(archive.values(), key=lambda x: (x[1], x[4]))[:6]
        for idx_t, el in enumerate(top_supp):
            ou = mod4_oracle_u64(m, n, p, el[5], el[6], el[7], support_restricted=False)
            if best_unres_ck is None or ou.coker_viol_4 < best_unres_ck:
                best_unres_ck = ou.coker_viol_4
            if ou.coker_viol_4 == 0:
                lres = attempt_certified_lift_and_save(
                    m,
                    n,
                    p,
                    el[5],
                    el[6],
                    el[7],
                    provenance=provenance,
                    artifacts_dir=artifacts_dir,
                    records_saved=records_saved,
                    z3_timeout_s=15.0,
                    hensel_trials=64,
                    label=f"{target_label}_final_unres_{idx_t}",
                )
                lift_logs.append(lres)
                if lres["lifted"]:
                    hit_zero = True
                    break

    # SLEEP-DAG-RSI cycle commit
    decomp = decomposer.decompose([tree], problem=problem)
    dead_b = decomp.get("dead_buckets", [])
    node_scores = [n_obj.score for nid, n_obj in tree.nodes.items() if nid != tree.root_id]
    tilt_res = compute_max_tilted_objective(node_scores, tau=policy.entropic_tau, normalize_by_scale=True) if node_scores else {}
    skill_state.commit_cycle(
        tree=tree,
        problem=problem,
        coupled_invariants=[
            f"target={target_label}",
            f"best_unres_coker_4={best_unres_ck}",
            f"best_supp_coker_4={best_supp_ck}",
            f"dead_buckets={dead_b}",
        ],
    )

    # Top-5 lowest-obstruction summary
    top_entries = sorted(archive.values(), key=lambda x: (x[0], x[1], x[4]))[:5]
    top_summary = [
        {"rank": e[3], "supp_coker_4": e[1], "unres_coker_4": e[2], "nnz": e[4]} for e in top_entries
    ]
    elapsed_tot = time.perf_counter() - t0
    print(
        f"  [{target_label} DONE] t={elapsed_tot:.1f}s evals={total_evals} screened={total_screened} "
        f"min_unres_coker={best_unres_ck} min_supp_coker={best_supp_ck} lifted={hit_zero}",
        flush=True,
    )
    return {
        "target_label": target_label,
        "format": [m, n, p],
        "R_target": R_target,
        "extra_ranks_relaxed": extra_ranks,
        "sa_mode": mode_tag,
        "total_oracle_evals": total_evals,
        "unique_schemes_archived": total_screened,
        "min_unres_coker_4": best_unres_ck,
        "min_supp_coker_4": best_supp_ck if best_supp_ck < 999999 else None,
        "hit_zero_and_lifted": hit_zero,
        "lift_attempts": lift_logs,
        "top_basins": top_summary,
        "dead_buckets": dead_b,
        "policy_j_tau": round(float(tilt_res.get("j_tau", 0.0)), 4),
        "elapsed_s": round(elapsed_tot, 2),
        "provenance": provenance,
    }


# =============================================================================
# 6. Priority 5 Diagnostic: Z[1/2] / Odd-Prime F_3 Sign-Liftability Check
# =============================================================================


def check_f3_support_lift_z3(
    m: int,
    n: int,
    p: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    timeout_s: float = 12.0,
) -> Dict[str, Any]:
    """
    Test whether a support-restricted F2 scheme admits a valid scheme over F_3 = {-1, 0, +1} (mod 3)!
    Mathematical motivation (Priority 5 in #73):
      - Any scheme over Z[1/2] with denominators powers of 2 (like AlphaEvolve / rational 1/2 schemes)
        has a well-defined reduction mod 3 (since 2^-1 = 2 mod 3) even if it is mod-2 obstructed!
      - Solving the exact Brent equations mod 3 on the F2 support via Z3 tests whether the same
        support structure lifts over F_3 (first step of 3-adic Hensel -> Z[1/2] / Q!).
    """
    t0 = time.perf_counter()
    R = len(u)
    mn, np_dim, mp = m * n, n * p, m * p

    solver = z3.Solver()
    solver.set("timeout", int(timeout_s * 1000))

    # Each nonzero entry is s in {-1, +1} represented via Bool b (s = +1 if b else -1; mod 3: +1 or 2)
    # 2R gauge fixing: first nonzero in u[r] and v[r] fixed to +1!
    u_vars: Dict[Tuple[int, int], Any] = {}
    v_vars: Dict[Tuple[int, int], Any] = {}
    w_vars: Dict[Tuple[int, int], Any] = {}

    for r in range(R):
        first_u = True
        for a in range(mn):
            if (int(u[r]) >> a) & 1:
                if first_u:
                    u_vars[(r, a)] = None  # fixed +1
                    first_u = False
                else:
                    u_vars[(r, a)] = z3.Bool(f"u_{r}_{a}")
        first_v = True
        for b in range(np_dim):
            if (int(v[r]) >> b) & 1:
                if first_v:
                    v_vars[(r, b)] = None  # fixed +1
                    first_v = False
                else:
                    v_vars[(r, b)] = z3.Bool(f"v_{r}_{b}")
        for c in range(mp):
            if (int(w[r]) >> c) & 1:
                w_vars[(r, c)] = z3.Bool(f"w_{r}_{c}")

    # Collect active support triples (a, b, c)
    triples: Dict[Tuple[int, int, int], List[int]] = {}
    for r in range(R):
        a_list = [a for a in range(mn) if (int(u[r]) >> a) & 1]
        b_list = [b for b in range(np_dim) if (int(v[r]) >> b) & 1]
        c_list = [c for c in range(mp) if (int(w[r]) >> c) & 1]
        for a in a_list:
            for b in b_list:
                for c in c_list:
                    triples.setdefault((a, b, c), []).append(r)

    t_exact = set()
    for i1 in range(m):
        for i2 in range(n):
            a = i1 * n + i2
            for j2 in range(p):
                t_exact.add((a, i2 * p + j2, i1 * p + j2))

    # For each active triple (a,b,c) covered by k terms:
    # sum_{r} s_r == delta (mod 3), where s_r in {-1, +1}.
    # If x_r in {0, 1} is the parity of negative signs (s_r = 1 - 2*x_r), then
    # sum s_r = k - 2 * sum x_r == delta (mod 3) <=> sum x_r == (k - delta) * 2^-1 == 2*(k - delta) (mod 3)!
    for (a, b, c), r_list in triples.items():
        k = len(r_list)
        delta = 1 if (a, b, c) in t_exact else 0
        # Target sum of negative-parity bits mod 3:
        target_mod3 = (2 * (k - delta)) % 3
        neg_exprs = []
        for r in r_list:
            bits = [x for x in (u_vars[(r, a)], v_vars[(r, b)], w_vars[(r, c)]) if x is not None]
            if not bits:
                neg_exprs.append(z3.IntVal(0))
            elif len(bits) == 1:
                neg_exprs.append(z3.If(bits[0], 1, 0))
            else:
                xor_e = bits[0]
                for bt in bits[1:]:
                    xor_e = z3.Xor(xor_e, bt)
                neg_exprs.append(z3.If(xor_e, 1, 0))
        if k == 1:
            # Must be s_r == delta (mod 3): impossible if delta == 0 (since s_r in {+1,-1})!
            if delta == 0:
                return {
                    "format": [m, n, p],
                    "rank": R,
                    "ring": "F3_support",
                    "status": "unsat_trivial_k1",
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }
            solver.add(neg_exprs[0] == 0)
        else:
            s_sum = z3.Sum(neg_exprs)
            allowed = [v_int for v_int in range(k + 1) if v_int % 3 == target_mod3]
            if not allowed:
                return {
                    "format": [m, n, p],
                    "rank": R,
                    "ring": "F3_support",
                    "status": "unsat_trivial",
                    "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                }
            solver.add(z3.Or([s_sum == v_int for v_int in allowed]))

    chk = solver.check()
    dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "format": [m, n, p],
        "rank": R,
        "ring": "F3_support",
        "active_triples": len(triples),
        "status": str(chk),
        "elapsed_ms": dt_ms,
    }


# =============================================================================
# 7. Main Phase 2 Assault Campaign Runner
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 CPU-Only Direct + Plus-Relaxed F2 Lifting Assault")
    parser.add_argument("--total-minutes", type=float, default=12.5, help="Total CPU wall-clock budget in minutes")
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=os.path.join(_FMM_ROOT, "artifacts", "public_f2_seeds", "manifest.json"),
        help="Path to public F2 seeds manifest JSON",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=os.path.join(_FMM_ROOT, "artifacts", "phase2_lifting_summary.json"),
        help="Output Phase 2 lifting summary JSON path",
    )
    parser.add_argument(
        "--surgical-minutes",
        type=float,
        default=0.0,
        help="If > 0, append a focused plus-preserved surgical sniper pass on top targets onto existing summary",
    )
    args = parser.parse_args()

    artifacts_dir = os.path.join(_FMM_ROOT, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    t_start = time.perf_counter()

    if not os.path.exists(args.manifest_path):
        raise FileNotFoundError(f"Run fetch_public_f2_seeds.py first! Missing: {args.manifest_path}")

    with open(args.manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    by_id = {e["seed_id"]: e for e in manifest["seeds"]}

    # Load all certified F2 seeds into u64 arrays
    loaded: Dict[str, Tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]] = {}
    for sid, e in by_id.items():
        sc = BilinearScheme.load_json(e["file"])
        u, v, w = scheme_to_u64_f2(sc)
        assert fast_verify_f2_u64(sc.m, sc.n, sc.p, u, v, w), f"Seed {sid} failed F2 check!"
        loaded[sid] = (sc.m, sc.n, sc.p, u, v, w, e["provenance"])

    # Warm up Numba JIT kernels on tiny 4x4x4 before allocating campaign timers!
    t_jit = time.perf_counter()
    _, _, _, u47, v47, w47, _ = loaded["4x4x4_R47_km"]
    Uw = np.zeros((2, 51), dtype=np.uint64)
    Vw = np.zeros((2, 51), dtype=np.uint64)
    Ww = np.zeros((2, 51), dtype=np.uint64)
    Uw[:, :47], Vw[:, :47], Ww[:, :47] = u47, v47, w47
    _coker_sa_batch_nb(4, 4, 4, Uw, Vw, Ww, np.array([47, 47], dtype=np.int32), 47, 2, 4, True, np.uint64(1))
    _coker_sa_batch_nb(4, 4, 4, Uw, Vw, Ww, np.array([47, 47], dtype=np.int32), 47, 2, 4, False, np.uint64(2))
    mod4_oracle_batch_u64(4, 4, 4, Uw[:, :47], Vw[:, :47], Ww[:, :47])
    mod4_oracle_batch_unrestricted_u64(4, 4, 4, Uw[:, :47], Vw[:, :47], Ww[:, :47])
    print(f"[Phase 2 JIT warmup done in {time.perf_counter() - t_jit:.2f}s]", flush=True)

    total_s = max(60.0, args.total_minutes * 60.0)
    records_saved: List[Dict[str, Any]] = []
    assault_results: List[Dict[str, Any]] = []
    f3_diags: List[Dict[str, Any]] = []
    prior_elapsed = 0.0

    if args.surgical_minutes > 0.0 and os.path.exists(args.summary_path):
        with open(args.summary_path, "r", encoding="utf-8") as fh:
            prev = json.load(fh)
        assault_results = prev.get("targets", [])
        records_saved = prev.get("lifted_world_records", [])
        f3_diags = prev.get("f3_odd_prime_diagnostics", [])
        prior_elapsed = float(prev.get("elapsed_total_s", 0.0))

    def _save_checkpoint() -> None:
        summary = {
            "phase": "Phase 2 (Option B - Track 2) Direct + Plus-Relaxed F2 -> Z_T/Z/Q Lifting Assault",
            "hardware": "CPU-only (Numba 12T + 64-bit Linear Hensel + Z3 5.1.0, ZERO GPU usage)",
            "seed_manifest": args.manifest_path,
            "public_seeds_loaded": len(loaded),
            "total_oracle_evals": sum(int(r.get("total_oracle_evals", 0)) for r in assault_results),
            "total_unique_basins_screened": sum(int(r.get("unique_schemes_archived", 0)) for r in assault_results),
            "targets": assault_results,
            "f3_odd_prime_diagnostics": f3_diags,
            "lifted_world_records": records_saved,
            "elapsed_total_s": round(prior_elapsed + (time.perf_counter() - t_start), 2),
        }
        with open(args.summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)

    # Pre-polish each distinct public seed ONCE with non-regressing GL(F2) shears!
    t_gl = time.perf_counter()
    polished_cache: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    needed_sids = (
        {"4x5x5_R73_arai", "4x4x8_R94_km_ds", "7x7x7_R245_medley", "4x5x6_R89_kauers_a", "4x5x6_R89_kauers_b"}
        if args.surgical_minutes > 0.0
        else set(loaded.keys())
    )
    for sid, (m, n, p, u0, v0, w0, _) in loaded.items():
        if sid not in needed_sids:
            continue
        pu, pv, pw, ck_p = gl_polish_strict(m, n, p, u0, v0, w0, max_passes=2, sparsify_first=True)
        polished_cache[sid] = (pu, pv, pw)
        print(f"  [GL pre-polish] {sid:22s}: supp_coker_4 -> {ck_p}", flush=True)
    print(f"[GL pre-polishing done in {time.perf_counter() - t_gl:.1f}s]", flush=True)

    if args.surgical_minutes > 0.0:
        surg_s = args.surgical_minutes * 60.0
        schedule = [
            ("S1_4x5x5_R75_keepR_supp", ["4x5x5_R73_arai"], 75, 2, 0.48, False, 9101),
            ("S2_4x4x8_R95_keepR_supp", ["4x4x8_R94_km_ds"], 95, 1, 0.22, False, 9201),
            ("S3_4x5x6_R89_deep_supp", ["4x5x6_R89_kauers_a", "4x5x6_R89_kauers_b"], 89, 0, 0.16, False, 9301),
            ("S4_7x7x7_R248_keepR_supp", ["7x7x7_R245_medley"], 248, 3, 0.14, False, 9401),
        ]
        total_s = surg_s
    else:
        schedule = [
            ("P5_4x4x5_R60_unres", ["4x4x5_R60_km"], 60, 0, 0.18, True, 5001),
            ("P1_4x5x5_R75_plus2_unres", ["4x5x5_R73_arai"], 75, 2, 0.12, True, 1001),
            ("P1_4x5x5_R75_plus2_supp", ["4x5x5_R73_arai"], 75, 2, 0.07, False, 1004),
            ("P1_4x5x5_R74_plus1_unres", ["4x5x5_R73_arai"], 74, 1, 0.08, True, 1002),
            ("P1_4x5x5_R73_direct_supp", ["4x5x5_R73_arai"], 73, 0, 0.05, False, 1003),
            ("P2_4x5x6_R89_unres", ["4x5x6_R89_kauers_a", "4x5x6_R89_kauers_b"], 89, 0, 0.10, True, 2001),
            ("P2_4x5x6_R89_supp", ["4x5x6_R89_kauers_a", "4x5x6_R89_kauers_b"], 89, 0, 0.06, False, 2002),
            ("P3_4x4x8_R95_plus1_unres", ["4x4x8_R94_km_ds"], 95, 1, 0.08, True, 3001),
            ("P3_4x4x8_R94_direct_unres", ["4x4x8_R94_km_ds"], 94, 0, 0.06, True, 3002),
            ("P4_7x7x7_R248_plus3_supp", ["7x7x7_R245_medley"], 248, 3, 0.07, False, 4001),
            ("P4_7x7x7_R246_plus1_supp", ["7x7x7_R245_medley"], 246, 1, 0.04, False, 4002),
            ("P4_7x7x7_R245_direct_supp", ["7x7x7_R245_medley"], 245, 0, 0.04, False, 4003),
            ("B6_4x4x4_R48_plus1_unres", ["4x4x4_R47_km"], 48, 1, 0.05, True, 6001),
        ]

    for label, sids, r_tgt, ex_r, wt, use_unres, s_off in schedule:
        m, n, p = loaded[sids[0]][0], loaded[sids[0]][1], loaded[sids[0]][2]
        bases = [polished_cache[s] for s in sids]
        prov = loaded[sids[0]][6]
        res = run_target_lifting_assault(
            target_label=label,
            m=m,
            n=n,
            p=p,
            R_target=r_tgt,
            extra_ranks=ex_r,
            polished_bases=bases,
            provenance=prov,
            budget_s=total_s * wt,
            use_unres_sa=use_unres,
            artifacts_dir=artifacts_dir,
            records_saved=records_saved,
            seed_offset=s_off,
        )
        assault_results.append(res)
        _save_checkpoint()

    if args.surgical_minutes <= 0.0:
        print("\n=== Priority 5 Odd-Prime (F_3 / Z[1/2] Compatibility) Support-Lift Diagnostics ===", flush=True)
        for sid in ("4x4x5_R60_km", "4x4x4_R47_km"):
            m, n, p = loaded[sid][0], loaded[sid][1], loaded[sid][2]
            ug, vg, wg = polished_cache[sid]
            d_f3 = check_f3_support_lift_z3(m, n, p, ug, vg, wg, timeout_s=10.0)
            d_f3["seed_id"] = sid
            f3_diags.append(d_f3)
            print(f"  [F3 support check {sid}] status={d_f3['status']} active_triples={d_f3.get('active_triples')} ({d_f3['elapsed_ms']}ms)", flush=True)

    _save_checkpoint()
    print(
        f"\nPhase 2 Lifting Assault COMPLETE -> {args.summary_path} "
        f"(evals={sum(r['total_oracle_evals'] for r in assault_results)}, "
        f"world_records={len(records_saved)}, t={time.perf_counter() - t_start:.1f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

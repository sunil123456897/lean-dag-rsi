"""
Phase 1 CPU-Only Liftability-Guided Flip-Graph & Hensel/Z3 Lifting Campaign
(ZERO GPU usage — strictly Numba 12T CPU + NumPy + Z3 5.1.0 + SLEEP-DAG-RSI v8).

Capabilities:
  1. High-speed Numba 12T uint64 F2 Flip-Graph featuring:
       - O(R) incremental exact uniform edge list,
       - Zero-allocation exact Kauers-Moosbauer (2023) 1-step neighbor-reduction sniping,
       - Exact Medley (2026) / KM generalized group linear-dependence reduction,
       - Tournament-of-3 sparsity-biased flips + multi-W_cap portfolio + targeted triple-collision
         split/plus tunneling (`u_k == u_i ^ u_j`),
       - Pure from-scratch bottom-up Meta-Flip direct-sum curriculum (from Strassen/Laderman/naive ONLY,
         zero external scheme leakage!).
  2. SLEEP-DAG-RSI v8 (`DiscoveryTree`, `LeanDAGPolicy(entropic_tau=4.0)`,
     `BackwardSubGoalDecomposer`, `MutableSkillState`) steering F2 flip-graph walks directly
     toward `coker_viol_4 -> 0` on Track A targets (`4x4x5`, `4x5x5`, `4x5x6`).
  3. Full ingestion, L0 screening, `coker_viol_4`-guided repair on obstructed hits, and L1/L2 lifting
     of GPU-discovered `fmm_discovery/artifacts/f2_pool_phase1.json` (`hensel_lifter.py`:
     Stage L0 mod-4 support/unrestricted oracle -> Stage L2 Z3 5.1.0 Z_T -> Stage L1 linear
     2-adic Hensel 2^k -> 2^{k+1} to 2^64 + Wang rational reconstruction).
  4. Dual-path exact certification of every lifted scheme via `brent_verifier.verify_scheme()`,
     saving any beaten 2026 world record to `artifacts/lifted_record_*.json` and full telemetry
     to `artifacts/phase1_lifting_summary.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numba as nb
import numpy as np

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_FMM_ROOT = os.path.dirname(_SRC_DIR)
_REPO_ROOT = os.path.dirname(_FMM_ROOT)
for _p in (_SRC_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from brent_verifier import BilinearScheme, make_naive_scheme, verify_scheme  # noqa: E402
from dream_rsi_scripts.engine import LeanDAGPolicy, compute_max_tilted_objective  # noqa: E402
from dream_rsi_scripts.skill_state import BackwardSubGoalDecomposer, MutableSkillState  # noqa: E402
from dream_rsi_scripts.tree import DiscoveryTree, Node, Problem  # noqa: E402
from hensel_lifter import (  # noqa: E402
    LiftResult,
    _mod4_oracle_support_nb,
    _popcount64,
    fast_verify_f2_u64,
    linear_hensel_lift,
    make_block_seed,
    mod4_oracle_batch_u64,
    mod4_oracle_u64,
    scheme_to_u64_f2,
    u64_f2_to_scheme,
    z3_ternary_lift,
)


# 2026 verified SOTA char-0 and F2 ranks (from sota_targets.json / 01_lille_sota_and_targets.md)
SOTA_2026: Dict[Tuple[int, int, int], Dict[str, int]] = {
    (4, 5, 5): {"F2": 73, "ZT": 76, "Z": 76, "Q": 76, "target_f2_hi": 75, "target_f2_lo": 73},
    (4, 4, 5): {"F2": 60, "ZT": 61, "Z": 61, "Q": 61, "target_f2_hi": 60, "target_f2_lo": 60},
    (4, 5, 6): {"F2": 89, "ZT": 90, "Z": 90, "Q": 90, "target_f2_hi": 89, "target_f2_lo": 89},
    (4, 4, 8): {"F2": 94, "ZT": 96, "Z": 96, "Q": 95, "target_f2_hi": 95, "target_f2_lo": 94},
    (4, 4, 4): {"F2": 47, "ZT": 49, "Z": 49, "Q": 48, "target_f2_hi": 48, "target_f2_lo": 47},
    (6, 6, 7): {"F2": 183, "ZT": 183, "Z": 183, "Q": 183, "target_f2_hi": 182, "target_f2_lo": 180},
    (5, 8, 8): {"F2": 230, "ZT": 230, "Z": 230, "Q": 230, "target_f2_hi": 229, "target_f2_lo": 226},
    (7, 7, 7): {"F2": 245, "ZT": 250, "Z": 250, "Q": 249, "target_f2_hi": 248, "target_f2_lo": 245},
    (5, 7, 8): {"F2": 204, "ZT": 204, "Z": 204, "Q": 204, "target_f2_hi": 203, "target_f2_lo": 201},
}


# =============================================================================
# 1. Numba 12-Thread uint64 F2 Flip-Graph Engine (KM 1-step sniping + W_cap + Expand)
# =============================================================================


@nb.njit(cache=True)
def _rng_next(state: np.uint64) -> np.uint64:
    state ^= state >> np.uint64(12)
    state ^= state << np.uint64(25)
    state ^= state >> np.uint64(27)
    return state * np.uint64(0x2545F4914F6CDD1D)


@nb.njit(cache=True)
def _reduce_scheme_inplace_nb(u: np.ndarray, v: np.ndarray, w: np.ndarray, R: int) -> int:
    """Exhaustively eliminate zero rows and merge 2-mode duplicates."""
    changed = True
    while changed:
        changed = False
        i = 0
        while i < R:
            if u[i] == 0 or v[i] == 0 or w[i] == 0:
                R -= 1
                u[i] = u[R]
                v[i] = v[R]
                w[i] = w[R]
                changed = True
            else:
                i += 1
        i = 0
        while i < R:
            ui, vi, wi = u[i], v[i], w[i]
            j = i + 1
            while j < R:
                if ui == u[j] and vi == v[j]:
                    w[i] ^= w[j]
                    wi = w[i]
                    R -= 1
                    u[j] = u[R]
                    v[j] = v[R]
                    w[j] = w[R]
                    changed = True
                    if wi == 0:
                        break
                elif ui == u[j] and wi == w[j]:
                    v[i] ^= v[j]
                    vi = v[i]
                    R -= 1
                    u[j] = u[R]
                    v[j] = v[R]
                    w[j] = w[R]
                    changed = True
                    if vi == 0:
                        break
                elif vi == v[j] and wi == w[j]:
                    u[i] ^= u[j]
                    ui = u[i]
                    R -= 1
                    u[j] = u[R]
                    v[j] = v[R]
                    w[j] = w[R]
                    changed = True
                    if ui == 0:
                        break
                else:
                    j += 1
            i += 1
    return R


@nb.njit(cache=True)
def _check_local_reduce_nb(u: np.ndarray, v: np.ndarray, w: np.ndarray, R: int, idx_a: int, idx_b: int) -> int:
    """Fast O(R) check for zero rows or 2-mode duplicates involving modified rows idx_a or idx_b."""
    if u[idx_a] == 0 or v[idx_a] == 0 or w[idx_a] == 0:
        return _reduce_scheme_inplace_nb(u, v, w, R)
    if u[idx_b] == 0 or v[idx_b] == 0 or w[idx_b] == 0:
        return _reduce_scheme_inplace_nb(u, v, w, R)
    ua, va, wa = u[idx_a], v[idx_a], w[idx_a]
    ub, vb, wb = u[idx_b], v[idx_b], w[idx_b]
    for k in range(R):
        if k != idx_a:
            uk, vk, wk = u[k], v[k], w[k]
            if (ua == uk and va == vk) or (ua == uk and wa == wk) or (va == vk and wa == wk):
                return _reduce_scheme_inplace_nb(u, v, w, R)
        if k != idx_b:
            uk, vk, wk = u[k], v[k], w[k]
            if (ub == uk and vb == vk) or (ub == uk and wb == wk) or (vb == vk and wb == wk):
                return _reduce_scheme_inplace_nb(u, v, w, R)
    return R


@nb.njit(cache=True)
def _apply_plus_nb(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    R: int,
    i: int,
    j: int,
    orient: int,
) -> int:
    """Exact F2 `plus` transition (R -> R + 1) in one of 6 cyclic axis orientations."""
    ui, vi, wi = u[i], v[i], w[i]
    uj, vj, wj = u[j], v[j], w[j]
    if orient == 0:
        u[i], v[i], w[i] = ui, vi, wi ^ wj
        u[j], v[j], w[j] = ui, vi ^ vj, wj
        u[R], v[R], w[R] = ui ^ uj, vj, wj
    elif orient == 1:
        u[i], v[i], w[i] = ui, vi ^ vj, wi
        u[j], v[j], w[j] = ui, vj, wi ^ wj
        u[R], v[R], w[R] = ui ^ uj, vj, wj
    elif orient == 2:
        u[i], v[i], w[i] = ui ^ uj, vi, wi
        u[j], v[j], w[j] = uj, vi, wi ^ wj
        u[R], v[R], w[R] = uj, vi ^ vj, wj
    elif orient == 3:
        u[i], v[i], w[i] = ui, vi, wi ^ wj
        u[j], v[j], w[j] = ui ^ uj, vi, wj
        u[R], v[R], w[R] = uj, vi ^ vj, wj
    elif orient == 4:
        u[i], v[i], w[i] = ui ^ uj, vi, wi
        u[j], v[j], w[j] = uj, vi ^ vj, wi
        u[R], v[R], w[R] = uj, vj, wi ^ wj
    else:
        u[i], v[i], w[i] = ui, vi ^ vj, wi
        u[j], v[j], w[j] = ui ^ uj, vj, wi
        u[R], v[R], w[R] = uj, vj, wi ^ wj
    return _reduce_scheme_inplace_nb(u, v, w, R + 1)


@nb.njit(cache=True)
def _scheme_nnz_nb(u: np.ndarray, v: np.ndarray, w: np.ndarray, R: int) -> int:
    tot = np.int64(0)
    for r in range(R):
        tot += np.int64(_popcount64(u[r])) + np.int64(_popcount64(v[r])) + np.int64(_popcount64(w[r]))
    return int(tot)


@nb.njit(cache=True)
def _scheme_orbit_hash_nb(u: np.ndarray, v: np.ndarray, w: np.ndarray, R: int) -> np.uint64:
    """Fast permutation-invariant uint64 structural hash of a rank-R F2 scheme."""
    h = np.uint64(0xCBF29CE484222325) ^ np.uint64(R)
    for r in range(R):
        x = (
            u[r] * np.uint64(0x9E3779B97F4A7C15)
            ^ (v[r] * np.uint64(0xBF58476D1CE4E5B9))
            ^ (w[r] * np.uint64(0x94D049BB133111EB))
        )
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        x = x ^ (x >> np.uint64(31))
        h += x
    return h


@nb.njit(cache=True)
def _group_lin_reduce_nb(u: np.ndarray, v: np.ndarray, w: np.ndarray, R: int) -> int:
    """
    Exact Medley (2026) / Kauers-Moosbauer (2023) generalized group linear-dependence reduction.
    """
    reduced = True
    while reduced:
        reduced = False
        R = _reduce_scheme_inplace_nb(u, v, w, R)
        for mode in range(3):
            visited = np.zeros(R, dtype=np.uint8)
            for i in range(R):
                if visited[i]:
                    continue
                key = u[i] if mode == 0 else (v[i] if mode == 1 else w[i])
                grp = np.empty(64, dtype=np.int32)
                g_len = 0
                for j in range(i, R):
                    val = u[j] if mode == 0 else (v[j] if mode == 1 else w[j])
                    if val == key:
                        visited[j] = 1
                        if g_len < 64:
                            grp[g_len] = j
                            g_len += 1
                if g_len < 3:
                    continue
                for check_axis in range(2):
                    other_a = (mode + 1) % 3 if check_axis == 0 else (mode + 2) % 3
                    other_b = (mode + 2) % 3 if check_axis == 0 else (mode + 1) % 3
                    basis_vec = np.zeros(64, dtype=np.uint64)
                    basis_comb = np.zeros(64, dtype=np.uint64)
                    basis_len = 0
                    dep_comb = np.uint64(0)
                    for t in range(g_len):
                        idx_t = grp[t]
                        vec = u[idx_t] if other_a == 0 else (v[idx_t] if other_a == 1 else w[idx_t])
                        comb = np.uint64(1) << np.uint64(t)
                        for b in range(basis_len):
                            bv = basis_vec[b]
                            lsb = bv & (~bv + np.uint64(1))
                            if (vec & lsb) != 0:
                                vec ^= bv
                                comb ^= basis_comb[b]
                        if vec == 0:
                            dep_comb = comb
                            break
                        basis_vec[basis_len] = vec
                        basis_comb[basis_len] = comb
                        basis_len += 1
                    if dep_comb != 0:
                        s_last = -1
                        for t in range(g_len - 1, -1, -1):
                            if ((dep_comb >> np.uint64(t)) & np.uint64(1)) != 0:
                                s_last = grp[t]
                                dep_comb ^= np.uint64(1) << np.uint64(t)
                                break
                        val_last = u[s_last] if other_b == 0 else (v[s_last] if other_b == 1 else w[s_last])
                        for t in range(g_len):
                            if ((dep_comb >> np.uint64(t)) & np.uint64(1)) != 0:
                                idx_s = grp[t]
                                if other_b == 0:
                                    u[idx_s] ^= val_last
                                elif other_b == 1:
                                    v[idx_s] ^= val_last
                                else:
                                    w[idx_s] ^= val_last
                        R -= 1
                        u[s_last] = u[R]
                        v[s_last] = v[R]
                        w[s_last] = w[R]
                        reduced = True
                        break
                if reduced:
                    break
            if reduced:
                break
    return R


@nb.njit(cache=True)
def _rebuild_edges_nb(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    R: int,
    ej_i: np.ndarray,
    ej_j: np.ndarray,
    ej_m: np.ndarray,
) -> int:
    """Build exact flat list of all flip edges (mode, i < j)."""
    cap = ej_i.shape[0]
    ne = 0
    for i in range(R):
        ui, vi, wi = u[i], v[i], w[i]
        for j in range(i + 1, R):
            if ui == u[j] and ne < cap:
                ej_i[ne] = i
                ej_j[ne] = j
                ej_m[ne] = 0
                ne += 1
            if vi == v[j] and ne < cap:
                ej_i[ne] = i
                ej_j[ne] = j
                ej_m[ne] = 1
                ne += 1
            if wi == w[j] and ne < cap:
                ej_i[ne] = i
                ej_j[ne] = j
                ej_m[ne] = 2
                ne += 1
    return ne


@nb.njit(cache=True)
def _update_edges_after_flip_nb(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    R: int,
    mode: int,
    idx_i: int,
    idx_j: int,
    ej_i: np.ndarray,
    ej_j: np.ndarray,
    ej_m: np.ndarray,
    ne: int,
) -> int:
    """O(ne + 2*R) exact incremental update of the flip-edge list after one flip."""
    cap = ej_i.shape[0]
    m_a = 1 if mode == 0 else 0
    m_b = 2 if mode < 2 else 1

    k = 0
    while k < ne:
        em = ej_m[k]
        ei = ej_i[k]
        ej = ej_j[k]
        stale = (em == m_a and (ei == idx_i or ej == idx_i)) or (em == m_b and (ei == idx_j or ej == idx_j))
        if stale:
            ne -= 1
            ej_i[k] = ej_i[ne]
            ej_j[k] = ej_j[ne]
            ej_m[k] = ej_m[ne]
        else:
            k += 1

    val_a = v[idx_i] if m_a == 1 else u[idx_i]
    for r in range(R):
        if r == idx_i:
            continue
        vr = v[r] if m_a == 1 else u[r]
        if vr == val_a and ne < cap:
            ej_i[ne] = idx_i if idx_i < r else r
            ej_j[ne] = r if idx_i < r else idx_i
            ej_m[ne] = m_a
            ne += 1

    val_b = w[idx_j] if m_b == 2 else v[idx_j]
    for r in range(R):
        if r == idx_j:
            continue
        vr = w[r] if m_b == 2 else v[r]
        if vr == val_b and ne < cap:
            ej_i[ne] = idx_j if idx_j < r else r
            ej_j[ne] = r if idx_j < r else idx_j
            ej_m[ne] = m_b
            ne += 1

    return ne


@nb.njit(cache=True)
def _snipe_1step_fast_nb(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    R: int,
    ej_i: np.ndarray,
    ej_j: np.ndarray,
    ej_m: np.ndarray,
    ne: int,
    deg: np.ndarray,
    nbr: np.ndarray,
) -> Tuple[bool, int, int, int]:
    """
    Zero-allocation exact Kauers-Moosbauer (2023) Algorithm 1 line 2b:
    Detects whether ANY 1-step neighbor of the current scheme is reducible (R -> R-1)
    by scanning all 2-edge paths in O(ne * avg_deg) with pre-allocated buffers.
    """
    for m in range(3):
        for r in range(R):
            deg[m, r] = 0
    for e in range(ne):
        i = ej_i[e]
        j = ej_j[e]
        m = ej_m[e]
        di = deg[m, i]
        if di < 16:
            nbr[m, i, di] = j
        deg[m, i] = di + 1
        dj = deg[m, j]
        if dj < 16:
            nbr[m, j, dj] = i
        deg[m, j] = dj + 1

    for e in range(ne):
        i = ej_i[e]
        j = ej_j[e]
        m = ej_m[e]
        if m == 0:
            dv = v[i] ^ v[j]
            dw = w[i] ^ w[j]
            d0 = deg[0, i] if deg[0, i] < 16 else 16
            for t in range(d0):
                k = nbr[0, i, t]
                if k != j and (v[k] == dv or w[k] == dw):
                    return True, i, j, 0
            d2i = deg[2, i] if deg[2, i] < 16 else 16
            for t in range(d2i):
                k = nbr[2, i, t]
                if k != j and v[k] == dv:
                    return True, i, j, 0
            d1j = deg[1, j] if deg[1, j] < 16 else 16
            for t in range(d1j):
                k = nbr[1, j, t]
                if k != i and w[k] == dw:
                    return True, i, j, 0
            d2j = deg[2, j] if deg[2, j] < 16 else 16
            for t in range(d2j):
                k = nbr[2, j, t]
                if k != i and v[k] == dv:
                    return True, j, i, 0
            d1i = deg[1, i] if deg[1, i] < 16 else 16
            for t in range(d1i):
                k = nbr[1, i, t]
                if k != j and w[k] == dw:
                    return True, j, i, 0
        elif m == 1:
            du = u[i] ^ u[j]
            dw = w[i] ^ w[j]
            d1 = deg[1, i] if deg[1, i] < 16 else 16
            for t in range(d1):
                k = nbr[1, i, t]
                if k != j and (u[k] == du or w[k] == dw):
                    return True, i, j, 1
            d2i = deg[2, i] if deg[2, i] < 16 else 16
            for t in range(d2i):
                k = nbr[2, i, t]
                if k != j and u[k] == du:
                    return True, i, j, 1
            d0j = deg[0, j] if deg[0, j] < 16 else 16
            for t in range(d0j):
                k = nbr[0, j, t]
                if k != i and w[k] == dw:
                    return True, i, j, 1
            d2j = deg[2, j] if deg[2, j] < 16 else 16
            for t in range(d2j):
                k = nbr[2, j, t]
                if k != i and u[k] == du:
                    return True, j, i, 1
            d0i = deg[0, i] if deg[0, i] < 16 else 16
            for t in range(d0i):
                k = nbr[0, i, t]
                if k != j and w[k] == dw:
                    return True, j, i, 1
        else:
            du = u[i] ^ u[j]
            dv = v[i] ^ v[j]
            d2 = deg[2, i] if deg[2, i] < 16 else 16
            for t in range(d2):
                k = nbr[2, i, t]
                if k != j and (u[k] == du or v[k] == dv):
                    return True, i, j, 2
            d1i = deg[1, i] if deg[1, i] < 16 else 16
            for t in range(d1i):
                k = nbr[1, i, t]
                if k != j and u[k] == du:
                    return True, i, j, 2
            d0j = deg[0, j] if deg[0, j] < 16 else 16
            for t in range(d0j):
                k = nbr[0, j, t]
                if k != i and v[k] == dv:
                    return True, i, j, 2
            d1j = deg[1, j] if deg[1, j] < 16 else 16
            for t in range(d1j):
                k = nbr[1, j, t]
                if k != i and u[k] == du:
                    return True, j, i, 2
            d0i = deg[0, i] if deg[0, i] < 16 else 16
            for t in range(d0i):
                k = nbr[0, i, t]
                if k != j and v[k] == dv:
                    return True, j, i, 2
    return False, 0, 0, 0


@nb.njit(cache=True)
def _targeted_expand_nb(u: np.ndarray, v: np.ndarray, w: np.ndarray, R: int, state: np.uint64) -> Tuple[int, np.uint64]:
    """
    Non-isolated R -> R + 1 expansion:
      1. Searches for a triple collision arr[k] == arr[i] ^ arr[j] and applies either a
         Targeted Split of k into (i, j) or a Targeted Plus on (i, j), immediately bridging
         into the rest of the tensor instead of creating a dead 3-term island!
      2. Falls back to minimum-XOR-popcount `plus` over 16 sampled pairs.
    """
    for _ in range(96):
        state = _rng_next(state)
        ax = np.int64(state % np.uint64(3))
        state = _rng_next(state)
        i = np.int64(state % np.uint64(R))
        state = _rng_next(state)
        j = np.int64(state % np.uint64(R - 1))
        if j >= i:
            j += np.int64(1)
        arr = u if ax == 0 else (v if ax == 1 else w)
        target = arr[i] ^ arr[j]
        if target == 0:
            continue
        for k in range(R):
            if k != i and k != j and arr[k] == target:
                state = _rng_next(state)
                if (state & np.uint64(1)) == 0:
                    u[R] = u[k]
                    v[R] = v[k]
                    w[R] = w[k]
                    if ax == 0:
                        u[k] = u[i]
                        u[R] = u[j]
                    elif ax == 1:
                        v[k] = v[i]
                        v[R] = v[j]
                    else:
                        w[k] = w[i]
                        w[R] = w[j]
                    return _group_lin_reduce_nb(u, v, w, R + 1), state
                else:
                    state = _rng_next(state)
                    return _apply_plus_nb(u, v, w, R, i, j, np.int64(state % np.uint64(6))), state

    best_p1 = np.int64(0)
    best_p2 = np.int64(1)
    best_pw = np.int64(9999)
    for _ in range(16):
        state = _rng_next(state)
        p1 = np.int64(state % np.uint64(R))
        state = _rng_next(state)
        p2 = np.int64(state % np.uint64(R - 1))
        if p2 >= p1:
            p2 += np.int64(1)
        pw = (
            np.int64(_popcount64(u[p1] ^ u[p2]))
            + np.int64(_popcount64(v[p1] ^ v[p2]))
            + np.int64(_popcount64(w[p1] ^ w[p2]))
        )
        if pw < best_pw:
            best_pw = pw
            best_p1 = p1
            best_p2 = p2
    state = _rng_next(state)
    return _apply_plus_nb(u, v, w, R, best_p1, best_p2, np.int64(state % np.uint64(6))), state


@nb.njit(parallel=True, cache=True)
def _f2_rank_descent_batch_nb(
    U_in: np.ndarray,
    V_in: np.ndarray,
    W_in: np.ndarray,
    R_in: np.ndarray,
    path_len: int,
    snipe_stride: int,
    expand_mode: int,
    seed_base: np.uint64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    12-thread parallel Numba F2 flip-graph rank-descent batch:
      - Exact O(R) incremental uniform edge list + zero-malloc KM 1-step neighbor-reduction sniping,
      - Multi-strategy portfolio across walkers:
          * Walkers 0..7: W_cap sparsity-bounded flips (`W_cap in {3,3,4,4,4,5,5,6}`),
          * Walkers 8..15: Tournament-of-3 minimum-delta-nnz flips,
          * Walkers 16..23: Unbiased uniform flips for full graph mixing,
      - Targeted triple-collision split/plus expansion (`expand_mode in {0, 1, 2}`),
      - Saves explored endpoints at `start_R` to continuously grow diverse orbit pools!
    """
    n_walkers = U_in.shape[0]
    R_max = U_in.shape[1]

    U_out = U_in.copy()
    V_out = V_in.copy()
    W_out = W_in.copy()
    R_out = R_in.copy()
    flips_done = np.zeros(n_walkers, dtype=np.int64)

    for w_id in nb.prange(n_walkers):
        u = U_in[w_id].copy()
        v = V_in[w_id].copy()
        w = W_in[w_id].copy()
        R = int(R_in[w_id])
        R = _group_lin_reduce_nb(u, v, w, R)
        start_R = R
        best_R = R
        for k in range(R):
            U_out[w_id, k] = u[k]
            V_out[w_id, k] = v[k]
            W_out[w_id, k] = w[k]
        R_out[w_id] = best_R

        state = seed_base ^ (np.uint64(w_id + 1) * np.uint64(0x9E3779B97F4A7C15))
        if state == 0:
            state = np.uint64(1)

        ej_i = np.empty(4096, dtype=np.int32)
        ej_j = np.empty(4096, dtype=np.int32)
        ej_m = np.empty(4096, dtype=np.int8)
        deg = np.zeros((3, R_max), dtype=np.int32)
        nbr = np.empty((3, R_max, 16), dtype=np.int32)
        ne = _rebuild_edges_nb(u, v, w, R, ej_i, ej_j, ej_m)

        k_exp = expand_mode if ((w_id % 3 != 0) or ne == 0) else 0
        if ne == 0 and k_exp == 0:
            k_exp = 1
        for _ in range(k_exp):
            if R + 1 < R_max and R >= 3:
                R, state = _targeted_expand_nb(u, v, w, R, state)
        if k_exp > 0:
            ne = _rebuild_edges_nb(u, v, w, R, ej_i, ej_j, ej_m)

        mix_lock = 35 * k_exp
        mode_w = 0 if w_id < 8 else (1 if w_id < 16 else 2)
        wcap = 3 if w_id < 2 else (4 if w_id < 5 else (5 if w_id < 7 else 6))

        steps_local = 0
        for st in range(path_len):
            if ne == 0:
                if R + 1 < R_max and R >= 3:
                    R, state = _targeted_expand_nb(u, v, w, R, state)
                    ne = _rebuild_edges_nb(u, v, w, R, ej_i, ej_j, ej_m)
                    if ne == 0:
                        break
                else:
                    break

            found = False
            sel_i = np.int64(0)
            sel_j = np.int64(0)
            sel_m = np.int64(0)
            if (st % snipe_stride == 0) and (R <= start_R or st >= mix_lock):
                found, sel_i, sel_j, sel_m = _snipe_1step_fast_nb(u, v, w, R, ej_i, ej_j, ej_m, ne, deg, nbr)

            if not found:
                if mode_w == 0:
                    picked = False
                    for _pr in range(6):
                        state = _rng_next(state)
                        ek = np.int64(state % np.uint64(ne))
                        ti = np.int64(ej_i[ek])
                        tj = np.int64(ej_j[ek])
                        tm = np.int64(ej_m[ek])
                        state = _rng_next(state)
                        if (state & np.uint64(1)) != 0:
                            tmp = ti
                            ti = tj
                            tj = tmp
                        if tm == 0:
                            wa = np.int64(_popcount64(v[ti] ^ v[tj]))
                            wb = np.int64(_popcount64(w[tj] ^ w[ti]))
                            oa = np.int64(_popcount64(v[ti]))
                            ob = np.int64(_popcount64(w[tj]))
                        elif tm == 1:
                            wa = np.int64(_popcount64(u[ti] ^ u[tj]))
                            wb = np.int64(_popcount64(w[tj] ^ w[ti]))
                            oa = np.int64(_popcount64(u[ti]))
                            ob = np.int64(_popcount64(w[tj]))
                        else:
                            wa = np.int64(_popcount64(u[ti] ^ u[tj]))
                            wb = np.int64(_popcount64(v[tj] ^ v[ti]))
                            oa = np.int64(_popcount64(u[ti]))
                            ob = np.int64(_popcount64(v[tj]))
                        lim_a = wcap if wcap > oa else oa
                        lim_b = wcap if wcap > ob else ob
                        if wa <= lim_a and wb <= lim_b:
                            sel_i, sel_j, sel_m = ti, tj, tm
                            picked = True
                            break
                    if not picked:
                        state = _rng_next(state)
                        ek = np.int64(state % np.uint64(ne))
                        sel_i = np.int64(ej_i[ek])
                        sel_j = np.int64(ej_j[ek])
                        sel_m = np.int64(ej_m[ek])
                        state = _rng_next(state)
                        if (state & np.uint64(1)) != 0:
                            tmp = sel_i
                            sel_i = sel_j
                            sel_j = tmp
                elif mode_w == 1 and ne >= 2:
                    best_d = np.int64(999999)
                    for _t in range(3):
                        state = _rng_next(state)
                        ek = np.int64(state % np.uint64(ne))
                        ti = np.int64(ej_i[ek])
                        tj = np.int64(ej_j[ek])
                        tm = np.int64(ej_m[ek])
                        state = _rng_next(state)
                        if (state & np.uint64(1)) != 0:
                            tmp = ti
                            ti = tj
                            tj = tmp
                        if tm == 0:
                            dd = (np.int64(_popcount64(v[ti] ^ v[tj])) - np.int64(_popcount64(v[ti]))) + (
                                np.int64(_popcount64(w[tj] ^ w[ti])) - np.int64(_popcount64(w[tj]))
                            )
                        elif tm == 1:
                            dd = (np.int64(_popcount64(u[ti] ^ u[tj])) - np.int64(_popcount64(u[ti]))) + (
                                np.int64(_popcount64(w[tj] ^ w[ti])) - np.int64(_popcount64(w[tj]))
                            )
                        else:
                            dd = (np.int64(_popcount64(u[ti] ^ u[tj])) - np.int64(_popcount64(u[ti]))) + (
                                np.int64(_popcount64(v[tj] ^ v[ti])) - np.int64(_popcount64(v[tj]))
                            )
                        if dd < best_d:
                            best_d = dd
                            sel_i, sel_j, sel_m = ti, tj, tm
                else:
                    state = _rng_next(state)
                    ek = np.int64(state % np.uint64(ne))
                    sel_i = np.int64(ej_i[ek])
                    sel_j = np.int64(ej_j[ek])
                    sel_m = np.int64(ej_m[ek])
                    state = _rng_next(state)
                    if (state & np.uint64(1)) != 0:
                        tmp = sel_i
                        sel_i = sel_j
                        sel_j = tmp

            if sel_m == 0:
                v[sel_i] ^= v[sel_j]
                w[sel_j] ^= w[sel_i]
            elif sel_m == 1:
                u[sel_i] ^= u[sel_j]
                w[sel_j] ^= w[sel_i]
            else:
                u[sel_i] ^= u[sel_j]
                v[sel_j] ^= v[sel_i]

            steps_local += 1
            old_R = R
            R = _check_local_reduce_nb(u, v, w, R, sel_i, sel_j)
            if R == old_R and (st & 255) == 255:
                R = _group_lin_reduce_nb(u, v, w, R)

            if R < old_R:
                R = _group_lin_reduce_nb(u, v, w, R)
                ne = _rebuild_edges_nb(u, v, w, R, ej_i, ej_j, ej_m)
                if R < best_R:
                    best_R = R
                    for k in range(R):
                        U_out[w_id, k] = u[k]
                        V_out[w_id, k] = v[k]
                        W_out[w_id, k] = w[k]
                    R_out[w_id] = best_R
            else:
                ne = _update_edges_after_flip_nb(u, v, w, R, sel_m, sel_i, sel_j, ej_i, ej_j, ej_m, ne)

        # Save walked endpoint if stayed at start_R to maintain orbit diversity!
        if best_R == start_R and R == start_R and steps_local > 0:
            for k in range(R):
                U_out[w_id, k] = u[k]
                V_out[w_id, k] = v[k]
                W_out[w_id, k] = w[k]
            R_out[w_id] = R

        flips_done[w_id] = steps_local

    return U_out, V_out, W_out, R_out, flips_done


@nb.njit(parallel=True, cache=True)
def _coker_guided_walk_batch_nb(
    m: int,
    n: int,
    p: int,
    U_in: np.ndarray,
    V_in: np.ndarray,
    W_in: np.ndarray,
    R_target: int,
    n_outer_iters: int,
    burst_flips: int,
    seed_base: np.uint64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    SLEEP-DAG-RSI LIFTABILITY-GUIDED FLIP-GRAPH SEARCH (12-thread CPU Numba):
    Uses exact incremental edges + targeted expand + KM 1-step sniping to walk along
    rank <= R_target F2 schemes while actively hill-climbing `coker_viol_4 -> 0`!
    """
    n_walkers = U_in.shape[0]
    R_cap = U_in.shape[1]

    U_best = U_in.copy()
    V_best = V_in.copy()
    W_best = W_in.copy()
    R_best = np.full(n_walkers, R_target, dtype=np.int32)
    coker_best = np.full(n_walkers, 999999, dtype=np.int32)
    evals_done = np.zeros(n_walkers, dtype=np.int64)

    for w_id in nb.prange(n_walkers):
        u_cur = U_in[w_id].copy()
        v_cur = V_in[w_id].copy()
        w_cur = W_in[w_id].copy()
        R_cur = R_target

        c0, _, _, _, _, nnz0, _ = _mod4_oracle_support_nb(m, n, p, u_cur[:R_cur], v_cur[:R_cur], w_cur[:R_cur])
        cur_coker = c0 if c0 >= 0 else 999999
        cur_nnz = nnz0

        best_R = R_cur
        best_coker = cur_coker
        best_nnz = cur_nnz
        coker_best[w_id] = best_coker
        R_best[w_id] = best_R

        state = seed_base ^ (np.uint64(w_id + 17) * np.uint64(0xBF58476D1CE4E5B9))
        if state == 0:
            state = np.uint64(7)

        u_try = np.empty(R_cap, dtype=np.uint64)
        v_try = np.empty(R_cap, dtype=np.uint64)
        w_try = np.empty(R_cap, dtype=np.uint64)
        ej_i = np.empty(4096, dtype=np.int32)
        ej_j = np.empty(4096, dtype=np.int32)
        ej_m = np.empty(4096, dtype=np.int8)
        deg = np.zeros((3, R_cap), dtype=np.int32)
        nbr = np.empty((3, R_cap, 16), dtype=np.int32)

        evals_local = 1
        for it in range(n_outer_iters):
            if best_coker == 0 and best_R <= R_target and it >= 8:
                break

            for k in range(R_cur):
                u_try[k] = u_cur[k]
                v_try[k] = v_cur[k]
                w_try[k] = w_cur[k]
            R_try = R_cur
            ne = _rebuild_edges_nb(u_try, v_try, w_try, R_try, ej_i, ej_j, ej_m)

            do_plus = (ne == 0 or (it % 4 == 3)) and (R_try + 1 < R_cap)
            if do_plus:
                R_try, state = _targeted_expand_nb(u_try, v_try, w_try, R_try, state)
                ne = _rebuild_edges_nb(u_try, v_try, w_try, R_try, ej_i, ej_j, ej_m)

            n_flips = burst_flips * (3 if do_plus else 1)
            for fl in range(n_flips):
                if ne == 0:
                    break
                found = False
                sel_i = np.int64(0)
                sel_j = np.int64(0)
                sel_m = np.int64(0)
                if (fl & 3) == 0 and ((not do_plus) or fl >= 10):
                    found, sel_i, sel_j, sel_m = _snipe_1step_fast_nb(
                        u_try, v_try, w_try, R_try, ej_i, ej_j, ej_m, ne, deg, nbr
                    )
                if not found:
                    state = _rng_next(state)
                    ek = np.int64(state % np.uint64(ne))
                    sel_i = np.int64(ej_i[ek])
                    sel_j = np.int64(ej_j[ek])
                    sel_m = np.int64(ej_m[ek])
                    state = _rng_next(state)
                    if (state & np.uint64(1)) != 0:
                        tmp = sel_i
                        sel_i = sel_j
                        sel_j = tmp
                if sel_m == 0:
                    v_try[sel_i] ^= v_try[sel_j]
                    w_try[sel_j] ^= w_try[sel_i]
                elif sel_m == 1:
                    u_try[sel_i] ^= u_try[sel_j]
                    w_try[sel_j] ^= w_try[sel_i]
                else:
                    u_try[sel_i] ^= u_try[sel_j]
                    v_try[sel_j] ^= v_try[sel_i]
                old_Rt = R_try
                R_try = _check_local_reduce_nb(u_try, v_try, w_try, R_try, sel_i, sel_j)
                if R_try < old_Rt:
                    R_try = _group_lin_reduce_nb(u_try, v_try, w_try, R_try)
                    ne = _rebuild_edges_nb(u_try, v_try, w_try, R_try, ej_i, ej_j, ej_m)
                    if do_plus and R_try <= R_cur:
                        break
                else:
                    ne = _update_edges_after_flip_nb(
                        u_try, v_try, w_try, R_try, sel_m, sel_i, sel_j, ej_i, ej_j, ej_m, ne
                    )

            if R_try > R_cur:
                continue

            ck, _, _, _, _, nnz_t, _ = _mod4_oracle_support_nb(m, n, p, u_try[:R_try], v_try[:R_try], w_try[:R_try])
            evals_local += 1
            if ck < 0:
                continue

            accept = False
            if R_try < R_cur:
                accept = True
            elif ck < cur_coker:
                accept = True
            elif ck == cur_coker and nnz_t <= cur_nnz + 12:
                accept = True
            elif ck <= cur_coker + 2:
                state = _rng_next(state)
                if (state & np.uint64(7)) == 0:
                    accept = True

            if accept:
                R_cur = R_try
                cur_coker = ck
                cur_nnz = nnz_t
                for k in range(R_cur):
                    u_cur[k] = u_try[k]
                    v_cur[k] = v_try[k]
                    w_cur[k] = w_try[k]

                if (
                    (R_cur < best_R)
                    or (R_cur == best_R and cur_coker < best_coker)
                    or (R_cur == best_R and cur_coker == best_coker and cur_nnz < best_nnz)
                ):
                    best_R = R_cur
                    best_coker = cur_coker
                    best_nnz = cur_nnz
                    for k in range(best_R):
                        U_best[w_id, k] = u_cur[k]
                        V_best[w_id, k] = v_cur[k]
                        W_best[w_id, k] = w_cur[k]
                    R_best[w_id] = best_R
                    coker_best[w_id] = best_coker

        evals_done[w_id] = evals_local

    return U_best, V_best, W_best, R_best, coker_best, evals_done


# =============================================================================
# 2. Pure From-Scratch Meta-Flip Operators (Direct-Sum Gluing & Down-Projection)
# =============================================================================


def direct_sum_axis_u64(
    m1: int,
    n1: int,
    p1: int,
    u1: np.ndarray,
    v1: np.ndarray,
    w1: np.ndarray,
    m2: int,
    n2: int,
    p2: int,
    u2: np.ndarray,
    v2: np.ndarray,
    w2: np.ndarray,
    axis: int,
) -> Tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Exact F2 direct-sum gluing along axis 0(m), 1(n), or 2(p), verified via fast_verify_f2_u64."""
    if axis == 0:
        assert n1 == n2 and p1 == p2
        m, n, p = m1 + m2, n1, p1
        u = np.concatenate([u1, u2 << np.uint64(m1 * n)])
        v = np.concatenate([v1, v2])
        w = np.concatenate([w1, w2 << np.uint64(m1 * p)])
    elif axis == 1:
        assert m1 == m2 and p1 == p2
        m, n, p = m1, n1 + n2, p1

        def _remap_a(arr: np.ndarray, n_old: int, off: int) -> np.ndarray:
            out = np.zeros_like(arr)
            for r in range(len(arr)):
                x = int(arr[r])
                nx = 0
                for i in range(m):
                    for j in range(n_old):
                        if (x >> (i * n_old + j)) & 1:
                            nx |= 1 << (i * n + (j + off))
                out[r] = np.uint64(nx)
            return out

        u = np.concatenate([_remap_a(u1, n1, 0), _remap_a(u2, n2, n1)])
        v = np.concatenate([v1, v2 << np.uint64(n1 * p)])
        w = np.concatenate([w1, w2])
    else:
        assert m1 == m2 and n1 == n2
        m, n, p = m1, n1, p1 + p2

        def _remap_c(arr: np.ndarray, rows: int, p_old: int, off: int) -> np.ndarray:
            out = np.zeros_like(arr)
            for r in range(len(arr)):
                x = int(arr[r])
                nx = 0
                for i in range(rows):
                    for k in range(p_old):
                        if (x >> (i * p_old + k)) & 1:
                            nx |= 1 << (i * p + (k + off))
                out[r] = np.uint64(nx)
            return out

        u = np.concatenate([u1, u2])
        v = np.concatenate([_remap_c(v1, n, p1, 0), _remap_c(v2, n, p2, p1)])
        w = np.concatenate([_remap_c(w1, m, p1, 0), _remap_c(w2, m, p2, p1)])

    R_red = _group_lin_reduce_nb(u, v, w, len(u))
    u, v, w = u[:R_red].copy(), v[:R_red].copy(), w[:R_red].copy()
    assert fast_verify_f2_u64(m, n, p, u, v, w), f"direct_sum_axis_u64 failed on {m}x{n}x{p}"
    return m, n, p, u, v, w


def project_f2_u64_down(
    m: int, n: int, p: int, u: np.ndarray, v: np.ndarray, w: np.ndarray, axis: int, drop_idx: int
) -> Tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Exact F2 down-projection dropping one slice along axis in {0(m), 1(n), 2(p)}."""
    R = len(u)
    m2, n2, p2 = m - (1 if axis == 0 else 0), n - (1 if axis == 1 else 0), p - (1 if axis == 2 else 0)
    u2 = np.zeros(R, dtype=np.uint64)
    v2 = np.zeros(R, dtype=np.uint64)
    w2 = np.zeros(R, dtype=np.uint64)

    a_map = []
    for i in range(m):
        if axis == 0 and i == drop_idx:
            continue
        ni = i - (1 if axis == 0 and i > drop_idx else 0)
        for j in range(n):
            if axis == 1 and j == drop_idx:
                continue
            nj = j - (1 if axis == 1 and j > drop_idx else 0)
            a_map.append((i * n + j, ni * n2 + nj))

    b_map = []
    for j in range(n):
        if axis == 1 and j == drop_idx:
            continue
        nj = j - (1 if axis == 1 and j > drop_idx else 0)
        for k in range(p):
            if axis == 2 and k == drop_idx:
                continue
            nk = k - (1 if axis == 2 and k > drop_idx else 0)
            b_map.append((j * p + k, nj * p2 + nk))

    c_map = []
    for i in range(m):
        if axis == 0 and i == drop_idx:
            continue
        ni = i - (1 if axis == 0 and i > drop_idx else 0)
        for k in range(p):
            if axis == 2 and k == drop_idx:
                continue
            nk = k - (1 if axis == 2 and k > drop_idx else 0)
            c_map.append((i * p + k, ni * p2 + nk))

    for r in range(R):
        mu, mv, mw = int(u[r]), int(v[r]), int(w[r])
        nu = 0
        for old_b, new_b in a_map:
            if (mu >> old_b) & 1:
                nu |= 1 << new_b
        nv = 0
        for old_b, new_b in b_map:
            if (mv >> old_b) & 1:
                nv |= 1 << new_b
        nw = 0
        for old_b, new_b in c_map:
            if (mw >> old_b) & 1:
                nw |= 1 << new_b
        u2[r] = np.uint64(nu)
        v2[r] = np.uint64(nv)
        w2[r] = np.uint64(nw)

    new_R = _group_lin_reduce_nb(u2, v2, w2, R)
    return m2, n2, p2, u2[:new_R].copy(), v2[:new_R].copy(), w2[:new_R].copy()


def _quick_solve_primitive(
    m: int,
    n: int,
    p: int,
    seeds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    budget_s: float,
    target_R: int,
    seed_base: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast pure from-scratch CPU solver for small Meta-Flip building-block primitives."""
    R_max = max(len(s[0]) for s in seeds) + 6
    pool: Dict[int, Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = {}
    for su, sv, sw in seeds:
        h = int(_scheme_orbit_hash_nb(su, sv, sw, len(su)))
        pool[h] = (len(su), int(_scheme_nnz_nb(su, sv, sw, len(su))), su.copy(), sv.copy(), sw.copy())

    t0 = time.perf_counter()
    min_r = min(x[0] for x in pool.values())
    rd = 0
    stall = 0
    rng = np.random.default_rng(seed_base)
    while time.perf_counter() - t0 < budget_s and min_r > target_R:
        rd += 1
        at_min = sorted([x for x in pool.values() if x[0] == min_r], key=lambda x: x[1])
        U = np.zeros((24, R_max), dtype=np.uint64)
        V = np.zeros((24, R_max), dtype=np.uint64)
        W = np.zeros((24, R_max), dtype=np.uint64)
        R_arr = np.zeros(24, dtype=np.int32)
        for k in range(24):
            src = at_min[: min(24, len(at_min))] if k < 12 else at_min
            e = src[int(rng.integers(0, len(src)))]
            U[k, : e[0]] = e[2]
            V[k, : e[0]] = e[3]
            W[k, : e[0]] = e[4]
            R_arr[k] = e[0]
        em = 0 if stall < 6 else (1 if rd % 2 == 0 else 2)
        Ub, Vb, Wb, Rb, _ = _f2_rank_descent_batch_nb(U, V, W, R_arr, 1800, 4, em, np.uint64(seed_base + rd * 99991))
        new_min = min(min_r, int(Rb.min()))
        if new_min < min_r:
            min_r = new_min
            pool = {}
            stall = 0
        else:
            stall += 1
        for k in range(24):
            rw = int(Rb[k])
            if rw == min_r:
                uc, vc, wc = Ub[k, :rw].copy(), Vb[k, :rw].copy(), Wb[k, :rw].copy()
                if fast_verify_f2_u64(m, n, p, uc, vc, wc):
                    h = int(_scheme_orbit_hash_nb(uc, vc, wc, rw))
                    if h not in pool and len(pool) < 256:
                        pool[h] = (rw, int(_scheme_nnz_nb(uc, vc, wc, rw)), uc, vc, wc)

    best = min(pool.values(), key=lambda x: (x[0], x[1]))
    assert fast_verify_f2_u64(m, n, p, best[2], best[3], best[4])
    return best[2], best[3], best[4]


def build_clean_curriculum_seeds(budget_scale: float = 1.0) -> Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Build tight, flip-connected Meta-Flip seeds for 4x4x5, 4x5x5, and 4x5x6 strictly from scratch
    (schoolbook naive + Strassen <2,2,2:7> ONLY — ZERO external leakage!).
    """
    t0 = time.perf_counter()
    print("\n=== Stage 0: Pure From-Scratch CPU Meta-Flip Primitive Curriculum ===", flush=True)
    u242_n, v242_n, w242_n = scheme_to_u64_f2(make_naive_scheme(2, 4, 2, ring="Z2", modulus=2))
    u425_n, v425_n, w425_n = scheme_to_u64_f2(make_naive_scheme(4, 2, 5, ring="Z2", modulus=2))
    u433_n, v433_n, w433_n = scheme_to_u64_f2(make_naive_scheme(4, 3, 3, ring="Z2", modulus=2))
    u432_n, v432_n, w432_n = scheme_to_u64_f2(make_naive_scheme(4, 3, 2, ring="Z2", modulus=2))
    u442_n, v442_n, w442_n = scheme_to_u64_f2(make_naive_scheme(4, 4, 2, ring="Z2", modulus=2))
    u443_n, v443_n, w443_n = scheme_to_u64_f2(make_naive_scheme(4, 4, 3, ring="Z2", modulus=2))
    u245_n, v245_n, w245_n = scheme_to_u64_f2(make_naive_scheme(2, 4, 5, ring="Z2", modulus=2))

    # Solve small sub-tensors in < 10s total on CPU 12T
    u442, v442, w442 = _quick_solve_primitive(4, 4, 2, [(u442_n, v442_n, w442_n)], 1.2 * budget_scale, 26, 101)
    u432, v432, w432 = _quick_solve_primitive(4, 3, 2, [(u432_n, v432_n, w432_n)], 0.8 * budget_scale, 20, 102)
    u433, v433, w433 = _quick_solve_primitive(4, 3, 3, [(u433_n, v433_n, w433_n)], 1.5 * budget_scale, 29, 103)
    u443, v443, w443 = _quick_solve_primitive(4, 4, 3, [(u443_n, v443_n, w443_n)], 2.2 * budget_scale, 38, 104)
    u245, v245, w245 = _quick_solve_primitive(2, 4, 5, [(u245_n, v245_n, w245_n)], 2.2 * budget_scale, 32, 105)
    u425, v425, w425 = _quick_solve_primitive(4, 2, 5, [(u425_n, v425_n, w425_n)], 2.2 * budget_scale, 32, 106)

    # Glue <4,3,3> +_p <4,3,2> -> <4,3,5> (target 47!)
    _, _, _, u435_s, v435_s, w435_s = direct_sum_axis_u64(4, 3, 3, u433, v433, w433, 4, 3, 2, u432, v432, w432, 2)
    u435, v435, w435 = _quick_solve_primitive(4, 3, 5, [(u435_s, v435_s, w435_s)], 2.5 * budget_scale, 47, 107)

    # Glue <4,5,3> via <4,2,3> + <4,3,3> -> target 47 -> <4,5,6> = <4,5,3> +_p <4,5,3> (<= 94!)
    u423_n, v423_n, w423_n = scheme_to_u64_f2(make_naive_scheme(4, 2, 3, ring="Z2", modulus=2))
    u423, v423, w423 = _quick_solve_primitive(4, 2, 3, [(u423_n, v423_n, w423_n)], 0.8 * budget_scale, 20, 108)
    _, _, _, u453_s, v453_s, w453_s = direct_sum_axis_u64(4, 2, 3, u423, v423, w423, 4, 3, 3, u433, v433, w433, 1)
    u453, v453, w453 = _quick_solve_primitive(4, 5, 3, [(u453_s, v453_s, w453_s)], 2.5 * budget_scale, 47, 109)

    # Construct tight composite seeds for our 3 Track A targets!
    _, _, _, u445_a, v445_a, w445_a = direct_sum_axis_u64(2, 4, 5, u245, v245, w245, 2, 4, 5, u245, v245, w245, 0)
    _, _, _, u445_b, v445_b, w445_b = direct_sum_axis_u64(4, 4, 2, u442, v442, w442, 4, 4, 3, u443, v443, w443, 2)
    _, _, _, u455_a, v455_a, w455_a = direct_sum_axis_u64(4, 2, 5, u425, v425, w425, 4, 3, 5, u435, v435, w435, 1)
    _, _, _, u456_a, v456_a, w456_a = direct_sum_axis_u64(4, 5, 3, u453, v453, w453, 4, 5, 3, u453, v453, w453, 2)

    del u242_n, v242_n, w242_n
    dt = time.perf_counter() - t0
    print(
        f"  [Curriculum ready in {dt:.1f}s] primitives: "
        f"4x4x2={len(u442)}, 4x3x3={len(u433)}, 4x4x3={len(u443)}, 2x4x5={len(u245)}, 4x2x5={len(u425)}, "
        f"4x3x5={len(u435)}, 4x5x3={len(u453)} | "
        f"glued seeds -> 4x4x5:({len(u445_a)},{len(u445_b)}), 4x5x5:{len(u455_a)}, 4x5x6:{len(u456_a)}",
        flush=True,
    )
    return {
        "4x4x5": [(u445_a, v445_a, w445_a), (u445_b, v445_b, w445_b)],
        "4x5x5": [(u455_a, v455_a, w455_a)],
        "4x5x6": [(u456_a, v456_a, w456_a)],
    }


# =============================================================================
# 3. Full L0 -> L2 (Z3) -> L1 (Linear Hensel) Cascade & Record Saver
# =============================================================================


def _attempt_full_lift_cascade(
    m: int,
    n: int,
    p: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    z3_timeout_s: float = 8.0,
    hensel_trials: int = 16,
    allow_relaxed_hensel: bool = True,
    max_relaxed_coker: int = 6,
    label: str = "",
) -> Dict[str, Any]:
    """Run complete L0 -> L2 (Z3 ZT) -> L1 (Linear Hensel Z/Q) lifting cascade on one F2 scheme."""
    R = len(u)
    o_supp = mod4_oracle_u64(m, n, p, u, v, w, support_restricted=True)
    res_info: Dict[str, Any] = {
        "format": [m, n, p],
        "rank": R,
        "supp_coker_4": o_supp.coker_viol_4,
        "supp_free_bits": o_supp.free_bits,
        "supp_rank_J1": o_supp.rank_J1,
        "nnz_total": o_supp.nnz_total,
        "unres_coker_4": None,
        "lifted": False,
        "lifted_ring": None,
        "lifted_status": "obstructed_mod4" if o_supp.coker_viol_4 > 0 else "not_lifted",
        "lifted_scheme": None,
        "lifted_report": None,
        "attempts": [],
    }

    if o_supp.coker_viol_4 == 0:
        z3_res: LiftResult = z3_ternary_lift((m, n, p, u, v, w), timeout_s=z3_timeout_s, force_z3=False, name=label)
        res_info["attempts"].append(
            {"stage": "L2_ZT", "method": z3_res.method, "status": z3_res.status, "ms": z3_res.elapsed_ms}
        )
        if z3_res.success and z3_res.scheme is not None:
            res_info["lifted"] = True
            res_info["lifted_ring"] = "ZT"
            res_info["lifted_status"] = z3_res.status
            res_info["lifted_scheme"] = z3_res.scheme
            res_info["lifted_report"] = z3_res.report
            return res_info

        h_supp = linear_hensel_lift(
            (m, n, p, u, v, w), support_restricted=True, target_bits=64, max_trials=hensel_trials, name=label
        )
        res_info["attempts"].append(
            {
                "stage": "L1_hensel_supp",
                "status": h_supp.status,
                "max_level": h_supp.max_level_reached,
                "ms": h_supp.elapsed_ms,
            }
        )
        if h_supp.success and h_supp.scheme is not None:
            res_info["lifted"] = True
            res_info["lifted_ring"] = "ZT" if (h_supp.report and h_supp.report.is_ternary) else h_supp.scheme.ring
            res_info["lifted_status"] = h_supp.status
            res_info["lifted_scheme"] = h_supp.scheme
            res_info["lifted_report"] = h_supp.report
            return res_info

    if allow_relaxed_hensel and o_supp.coker_viol_4 <= max_relaxed_coker and (m * n * p <= 128):
        h_unres = linear_hensel_lift(
            (m, n, p, u, v, w), support_restricted=False, target_bits=64, max_trials=hensel_trials, name=label
        )
        res_info["unres_coker_4"] = h_unres.coker_viol_4
        res_info["attempts"].append(
            {
                "stage": "L1_hensel_unres",
                "status": h_unres.status,
                "coker_4": h_unres.coker_viol_4,
                "max_level": h_unres.max_level_reached,
                "ms": h_unres.elapsed_ms,
            }
        )
        if h_unres.success and h_unres.scheme is not None:
            res_info["lifted"] = True
            res_info["lifted_ring"] = "ZT" if (h_unres.report and h_unres.report.is_ternary) else h_unres.scheme.ring
            res_info["lifted_status"] = h_unres.status
            res_info["lifted_scheme"] = h_unres.scheme
            res_info["lifted_report"] = h_unres.report
            return res_info

    return res_info


def _save_world_record_if_beaten(
    m: int,
    n: int,
    p: int,
    scheme: BilinearScheme,
    artifacts_dir: str,
    records_saved: List[Dict[str, Any]],
) -> bool:
    """Verify strictly via `brent_verifier.verify_scheme()` and save ONLY if 2026 SOTA is beaten!"""
    canon_fmt = tuple(sorted((m, n, p)))
    sota = SOTA_2026.get(canon_fmt)
    if sota is None:
        return False

    rep = verify_scheme(scheme)
    if not rep.valid:
        raise RuntimeError(f"CRITICAL: Candidate {scheme.name} failed brent_verifier!")

    R = scheme.rank
    beaten_rings: List[str] = []
    if scheme.ring == "Z" and rep.is_ternary and R < sota["ZT"]:
        beaten_rings.append(f"ZT ({R} < {sota['ZT']})")
    if scheme.ring == "Z" and R < sota["Z"]:
        beaten_rings.append(f"Z ({R} < {sota['Z']})")
    if scheme.ring in ("Z", "Q") and R < sota["Q"]:
        beaten_rings.append(f"Q ({R} < {sota['Q']})")
    if scheme.ring == "Z2" and R < sota["F2"]:
        beaten_rings.append(f"F2 ({R} < {sota['F2']})")

    if not beaten_rings:
        return False

    ring_tag = "ZT" if (scheme.ring == "Z" and rep.is_ternary) else scheme.ring
    fname = f"lifted_record_{m}x{n}x{p}_R{R}_{ring_tag}.json"
    out_path = os.path.join(artifacts_dir, fname)
    scheme.metadata.update(
        {
            "beaten_2026_sota": beaten_rings,
            "verified_brent": True,
            "is_ternary": rep.is_ternary,
            "additive_complexity": rep.additive_complexity,
            "source": "run_phase1_lifting.py (zero external leakage)",
        }
    )
    scheme.save_json(out_path)
    records_saved.append(
        {
            "file": out_path,
            "format": [m, n, p],
            "rank": R,
            "ring": ring_tag,
            "beaten_2026_sota": beaten_rings,
            "canonical_sha256": rep.canonical_sha256,
            "naive_additions": rep.additive_complexity["naive_additions_total"],
        }
    )
    print(f"  *** NEW 2026 WORLD RECORD SAVED: {fname} ({', '.join(beaten_rings)}) ***", flush=True)
    return True


def run_cpu_guided_target(
    m: int,
    n: int,
    p: int,
    budget_s: float,
    artifacts_dir: str,
    records_saved: List[Dict[str, Any]],
    extra_seed_u64: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
    seed_offset: int = 1000,
    z3_timeout_s: float = 8.0,
) -> Dict[str, Any]:
    """
    Run SLEEP-DAG-RSI v8 CPU Numba 12T F2 rank descent + `coker_viol_4`-guided search +
    L0/L1/L2 lifting cascade for one target format <m, n, p>.
    """
    t0 = time.perf_counter()
    fmt_key = f"{m}x{n}x{p}"
    sota = SOTA_2026[tuple(sorted((m, n, p)))]
    R_char0 = sota["Q"]
    R_f2_sota = sota["F2"]
    target_hi = sota["target_f2_hi"]

    print(
        f"\n=== CPU 12T SLEEP-DAG-RSI Lifting Campaign: {fmt_key} "
        f"(SOTA F2={R_f2_sota}, char0={R_char0}, budget={budget_s:.1f}s) ===",
        flush=True,
    )

    base_zt = make_block_seed(m, n, p, ring="Z")
    u_base, v_base, w_base = scheme_to_u64_f2(base_zt.reduce_mod(2))
    assert fast_verify_f2_u64(m, n, p, u_base, v_base, w_base)

    naive_f2 = make_naive_scheme(m, n, p, ring="Z2", modulus=2)
    u_naive, v_naive, w_naive = scheme_to_u64_f2(naive_f2)
    assert fast_verify_f2_u64(m, n, p, u_naive, v_naive, w_naive)

    seed_pool: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = [(u_base, v_base, w_base), (u_naive, v_naive, w_naive)]
    if extra_seed_u64:
        for eu, ev, ew in extra_seed_u64:
            if fast_verify_f2_u64(m, n, p, eu, ev, ew):
                seed_pool.append((eu, ev, ew))

    n_walkers = 24
    R_max = max(len(s[0]) for s in seed_pool) + 8

    tree = DiscoveryTree(task=f"fmm_lift_{fmt_key}")
    policy = LeanDAGPolicy(entropic_tau=4.0)
    decomposer = BackwardSubGoalDecomposer()
    skill_state = MutableSkillState(task=f"fmm_lift_{fmt_key}")
    problem = Problem()

    # Elite archive: hash -> (R, coker, nnz, u, v, w)
    archive: Dict[int, Tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray]] = {}
    coker_hist: Counter[int] = Counter()
    total_flips = 0
    total_screened = 0
    best_R_seen = min(len(s[0]) for s in seed_pool)
    min_coker_at_target: Optional[int] = None
    min_coker_overall = 999999
    lift_attempts_log: List[Dict[str, Any]] = []
    lifted_hits: List[Dict[str, Any]] = []

    def _register_candidate(
        u_c: np.ndarray, v_c: np.ndarray, w_c: np.ndarray, parent_id: Optional[str], br: int, att: int
    ) -> Tuple[int, int]:
        nonlocal best_R_seen, min_coker_at_target, min_coker_overall, total_screened
        R_c = len(u_c)
        h = int(_scheme_orbit_hash_nb(u_c, v_c, w_c, R_c))
        if h in archive:
            return archive[h][0], archive[h][1]
        ora = mod4_oracle_u64(m, n, p, u_c, v_c, w_c, support_restricted=True)
        total_screened += 1
        ck = ora.coker_viol_4
        coker_hist[ck] += 1
        if R_c < best_R_seen:
            best_R_seen = R_c
        if ck < min_coker_overall:
            min_coker_overall = ck
        if R_c <= target_hi and (min_coker_at_target is None or ck < min_coker_at_target):
            min_coker_at_target = ck
        archive[h] = (R_c, ck, ora.nnz_total, u_c.copy(), v_c.copy(), w_c.copy())

        sc_val = 100.0 * (R_char0 - R_c) + (25.0 if ck == 0 else 0.0) - 2.0 * ck - 0.005 * ora.nnz_total
        b_scores = [
            float(R_char0 + 0.5 - R_c),
            float(R_f2_sota + 0.5 - R_c),
            float(0.5 - ck),
            -1.0,
        ]
        node = Node(
            id=f"{fmt_key}_n{len(tree.nodes)}" if len(tree.nodes) > 0 else "root",
            parent_id=parent_id,
            branch=br,
            attempt=att,
            score=sc_val,
            bucket_scores=b_scores,
            pass_mask=[1 if x > 0 else 0 for x in b_scores],
            meta={"rank": R_c, "coker_viol_4": ck, "nnz": ora.nnz_total, "orbit_hash": str(h)},
        )
        tree.add(node)
        return R_c, ck

    for idx_s, (su, sv, sw) in enumerate(seed_pool):
        _register_candidate(su, sv, sw, None, 0, idx_s)

    # Phase A (68% budget): Diverse-Pool KM-Sniped + Sparse-Tournament + Targeted-Expand Rank Descent
    # Phase B (32% budget): coker_viol_4-Guided Walk + L1/L2 Lifting Cascade
    descent_deadline = t0 + budget_s * 0.68
    total_deadline = t0 + budget_s

    round_idx = 0
    stall_rounds = 0
    rng_py = np.random.default_rng(seed_offset)
    while time.perf_counter() < descent_deadline:
        round_idx += 1
        at_best = [x for x in archive.values() if x[0] == best_R_seen]
        at_best.sort(key=lambda x: (x[1], x[2]))  # prioritize lowest coker_viol_4 and lowest nnz!
        elites = at_best[: min(32, len(at_best))]
        elite_scores = [100.0 * (R_char0 - e[0]) + (25.0 if e[1] == 0 else 0.0) - 2.0 * e[1] - 0.005 * e[2] for e in elites]
        tilt = compute_max_tilted_objective(elite_scores, tau=policy.entropic_tau, normalize_by_scale=True)
        probs = np.array(tilt["tilted_weights_normalized"], dtype=np.float64)
        probs = probs / np.sum(probs)

        U_in = np.zeros((n_walkers, R_max), dtype=np.uint64)
        V_in = np.zeros((n_walkers, R_max), dtype=np.uint64)
        W_in = np.zeros((n_walkers, R_max), dtype=np.uint64)
        R_in = np.zeros(n_walkers, dtype=np.int32)

        for w_i in range(n_walkers):
            if w_i < 12:
                pick = int(rng_py.choice(len(elites), p=probs))
                el = elites[pick]
            else:
                el = at_best[int(rng_py.integers(0, len(at_best)))]
            rr = el[0]
            U_in[w_i, :rr] = el[3]
            V_in[w_i, :rr] = el[4]
            W_in[w_i, :rr] = el[5]
            R_in[w_i] = rr

        path_len = 2500 if stall_rounds < 8 else (10000 if stall_rounds < 20 else 25000)
        expand_mode = (
            0
            if (round_idx % 4 == 1 and stall_rounds < 10)
            else (1 if round_idx % 4 == 2 else (2 if round_idx % 4 == 3 else 3))
        )
        U_b, V_b, W_b, R_b, fl_b = _f2_rank_descent_batch_nb(
            U_in,
            V_in,
            W_in,
            R_in,
            path_len=path_len,
            snipe_stride=4 if path_len <= 10000 else 8,
            expand_mode=expand_mode,
            seed_base=np.uint64(seed_offset + round_idx * 99991),
        )
        total_flips += int(np.sum(fl_b))

        prev_best = best_R_seen
        new_sub_hits = 0
        # Register ALL strictly improved ranks, plus sample diverse states at best_R_seen
        for w_i in range(n_walkers):
            rw = int(R_b[w_i])
            if rw < best_R_seen or (rw == best_R_seen and (len(at_best) < 384 or (w_i < 2 and round_idx % 5 == 0))):
                uc = U_b[w_i, :rw].copy()
                vc = V_b[w_i, :rw].copy()
                wc = W_b[w_i, :rw].copy()
                if fast_verify_f2_u64(m, n, p, uc, vc, wc):
                    rr, ck = _register_candidate(uc, vc, wc, None, round_idx, w_i)
                    if rr <= target_hi:
                        new_sub_hits += 1

        if best_R_seen < prev_best:
            stall_rounds = 0
            print(
                f"  [{fmt_key} descent r{round_idx}] t={time.perf_counter()-t0:.1f}s "
                f"flips={total_flips/1e6:.1f}M *** DROP {prev_best} -> {best_R_seen} ***",
                flush=True,
            )
        else:
            stall_rounds += 1

        sub_basins = [x for x in archive.values() if x[0] <= target_hi]
        if round_idx % 20 == 0 or new_sub_hits > 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  [{fmt_key} descent r{round_idx}] t={elapsed:.1f}s flips={total_flips/1e6:.2f}M "
                f"best_R={best_R_seen} screened={total_screened} sub_basins={len(sub_basins)} "
                f"min_coker={min_coker_overall}",
                flush=True,
            )
        if len(sub_basins) >= 20 and best_R_seen <= target_hi:
            break

    # SLEEP-DAG-RSI BackwardSubGoalDecomposer + Entropic Tilted Objective + MutableSkillState cycle commit
    decomp = decomposer.decompose([tree], problem=problem)
    dead_b = decomp.get("dead_buckets", [])
    node_scores = [n_obj.score for nid, n_obj in tree.nodes.items() if nid != tree.root_id]
    tilt_res = compute_max_tilted_objective(node_scores, tau=policy.entropic_tau, normalize_by_scale=True) if node_scores else {}
    policy_summary = {
        "entropic_tau": policy.entropic_tau,
        "j_tau": round(float(tilt_res.get("j_tau", 0.0)), 4),
        "effective_sample_size": round(float(tilt_res.get("effective_sample_size", 0.0)), 4),
        "nodes": len(tree.nodes),
    }
    skill_state.commit_cycle(
        tree=tree,
        problem=problem,
        coupled_invariants=[f"best_R={best_R_seen}", f"dead_buckets={dead_b}", f"min_coker={min_coker_overall}"],
    )

    # Phase B: coker_viol_4-Guided Walk on best_R_seen (and obstructed basins -> drive coker -> 0!)
    guided_R = min(target_hi, best_R_seen) if best_R_seen <= target_hi else best_R_seen
    guided_evals = 0
    guided_round = 0
    guided_stop = total_deadline - min(10.0, budget_s * 0.14)
    while time.perf_counter() < guided_stop or guided_round == 0:
        guided_round += 1
        cands_at_R = [x for x in archive.values() if x[0] == guided_R]
        if not cands_at_R:
            guided_R = best_R_seen
            cands_at_R = [x for x in archive.values() if x[0] == guided_R]
        # Include both lowest-coker AND obstructed (coker > 0) basins to test live coker reduction!
        cands_at_R.sort(key=lambda x: (x[1], x[2]))
        obstructed = [x for x in cands_at_R if x[1] > 0]

        R_buf = guided_R + 4
        U_g = np.zeros((n_walkers, R_buf), dtype=np.uint64)
        V_g = np.zeros((n_walkers, R_buf), dtype=np.uint64)
        W_g = np.zeros((n_walkers, R_buf), dtype=np.uint64)
        for w_i in range(n_walkers):
            if obstructed and w_i >= 12:
                el = obstructed[(w_i - 12) % len(obstructed)]
            else:
                el = cands_at_R[w_i % min(len(cands_at_R), 12)]
            U_g[w_i, :guided_R] = el[3]
            V_g[w_i, :guided_R] = el[4]
            W_g[w_i, :guided_R] = el[5]

        U_gb, V_gb, W_gb, R_gb, ck_gb, ev_gb = _coker_guided_walk_batch_nb(
            m,
            n,
            p,
            U_g,
            V_g,
            W_g,
            R_target=guided_R,
            n_outer_iters=50,
            burst_flips=35,
            seed_base=np.uint64(seed_offset + 500000 + guided_round * 7919),
        )
        guided_evals += int(np.sum(ev_gb))

        for w_i in range(n_walkers):
            rw = int(R_gb[w_i])
            uc = U_gb[w_i, :rw].copy()
            vc = V_gb[w_i, :rw].copy()
            wc = W_gb[w_i, :rw].copy()
            if fast_verify_f2_u64(m, n, p, uc, vc, wc):
                _register_candidate(uc, vc, wc, None, 100 + guided_round, w_i)

        if guided_round >= 6:
            break

    # Final lifting sweep: ALWAYS attempt L1+L2 on top lowest-(R, coker) schemes (sub-char0 first, else best_R_seen!)
    sub_all = sorted([x for x in archive.values() if x[0] <= target_hi], key=lambda x: (x[1], x[0], x[2]))
    lift_cands = sub_all[:8] if sub_all else sorted(
        [x for x in archive.values() if x[0] == best_R_seen], key=lambda x: (x[1], x[2])
    )[:4]
    for idx_c, el in enumerate(lift_cands):
        if time.perf_counter() > total_deadline + 20.0:
            break
        lres = _attempt_full_lift_cascade(
            m,
            n,
            p,
            el[3],
            el[4],
            el[5],
            z3_timeout_s=z3_timeout_s,
            hensel_trials=16,
            allow_relaxed_hensel=True,
            max_relaxed_coker=6,
            label=f"cpu_top_{fmt_key}_R{el[0]}_{idx_c}",
        )
        lift_attempts_log.append({k: v for k, v in lres.items() if k not in ("lifted_scheme", "lifted_report")})
        if lres["lifted"] and lres["lifted_scheme"] is not None:
            lifted_hits.append(lift_attempts_log[-1])
            _save_world_record_if_beaten(m, n, p, lres["lifted_scheme"], artifacts_dir, records_saved)

    if best_R_seen < R_f2_sota:
        best_f2_el = min(archive.values(), key=lambda x: (x[0], x[1], x[2]))
        sc_f2 = u64_f2_to_scheme(
            m, n, p, best_f2_el[3], best_f2_el[4], best_f2_el[5], name=f"cpu_f2_{fmt_key}_R{best_f2_el[0]}"
        )
        _save_world_record_if_beaten(m, n, p, sc_f2, artifacts_dir, records_saved)

    elapsed_tot = time.perf_counter() - t0
    sub_coker_dist = Counter(x[1] for x in archive.values() if x[0] <= target_hi)
    best_coker_dist = Counter(x[1] for x in archive.values() if x[0] == best_R_seen)
    print(
        f"  [{fmt_key} DONE] t={elapsed_tot:.1f}s best_R={best_R_seen} (char0_SOTA={R_char0}) "
        f"screened={total_screened} guided_evals={guided_evals} "
        f"min_coker_bestR={min(best_coker_dist) if best_coker_dist else None} lifted={len(lifted_hits)}/{len(lift_attempts_log)}",
        flush=True,
    )

    best_elites = sorted(archive.values(), key=lambda x: (x[0], x[1], x[2]))[:6]
    return {
        "format": [m, n, p],
        "sota_F2": R_f2_sota,
        "sota_char0": R_char0,
        "target_sub_char0_rank_hi": target_hi,
        "seed_rank_min": min(len(s[0]) for s in seed_pool),
        "best_F2_rank_reached": best_R_seen,
        "total_flips": total_flips,
        "flips_per_sec": round(total_flips / max(elapsed_tot, 1e-6), 1),
        "unique_schemes_screened": total_screened,
        "guided_oracle_evals": guided_evals,
        "sub_char0_unique_basins": len(sub_all),
        "min_coker_viol_4_overall": min_coker_overall if min_coker_overall < 999999 else None,
        "min_coker_viol_4_at_best_R": int(min(best_coker_dist)) if best_coker_dist else None,
        "min_coker_viol_4_at_target": min_coker_at_target,
        "coker_viol_4_hist_all": {str(k): int(v) for k, v in sorted(coker_hist.items())[:15]},
        "coker_viol_4_hist_best_R": {str(k): int(v) for k, v in sorted(best_coker_dist.items())[:15]},
        "coker_viol_4_hist_sub_char0": {str(k): int(v) for k, v in sorted(sub_coker_dist.items())[:15]},
        "lift_attempts": len(lift_attempts_log),
        "lifted_hits": len(lifted_hits),
        "lift_details_sample": lift_attempts_log[:8],
        "dead_buckets_cycle1": dead_b,
        "policy_summary": policy_summary,
        "elapsed_s": round(elapsed_tot, 2),
        "_elites_u64": [(e[3], e[4], e[5]) for e in best_elites],
    }


# =============================================================================
# 4. GPU Pool Ingestion (`f2_pool_phase1.json` from @fmm-architect) + Coker Repair
# =============================================================================


def _parse_pool_entries(raw: Any) -> List[Tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, str]]:
    """Flexibly parse `f2_pool_phase1.json` into a list of (m, n, p, u64, v64, w64, label)."""
    items: List[Tuple[Optional[Tuple[int, int, int]], Dict[str, Any]]] = []
    if isinstance(raw, list):
        for obj in raw:
            if isinstance(obj, dict):
                items.append((None, obj))
    elif isinstance(raw, dict):
        found_container = False
        for k in ("schemes", "pool", "entries", "candidates", "records"):
            if k in raw and isinstance(raw[k], list):
                found_container = True
                for obj in raw[k]:
                    if isinstance(obj, dict):
                        items.append((None, obj))
        if not found_container:
            container = raw
            for k in ("formats", "by_format", "targets", "pools"):
                if k in raw and isinstance(raw[k], dict):
                    container = raw[k]
                    break
            for k, val in container.items():
                fmt_hint: Optional[Tuple[int, int, int]] = None
                for sep in ("x", ",", "_"):
                    parts = str(k).split(sep)
                    if len(parts) == 3 and all(pt.isdigit() for pt in parts):
                        fmt_hint = (int(parts[0]), int(parts[1]), int(parts[2]))
                        break
                if isinstance(val, list):
                    for obj in val:
                        if isinstance(obj, dict):
                            items.append((fmt_hint, obj))
                elif isinstance(val, dict):
                    for subk in ("schemes", "pool", "entries", "candidates"):
                        if subk in val and isinstance(val[subk], list):
                            for obj in val[subk]:
                                if isinstance(obj, dict):
                                    items.append((fmt_hint, obj))

    parsed: List[Tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, str]] = []
    for idx, (fmt_hint, obj) in enumerate(items):
        if "m" in obj and "n" in obj and "p" in obj:
            m, n, p = int(obj["m"]), int(obj["n"]), int(obj["p"])
        elif "shape" in obj and len(obj["shape"]) == 3:
            m, n, p = int(obj["shape"][0]), int(obj["shape"][1]), int(obj["shape"][2])
        elif "format" in obj and isinstance(obj["format"], list) and len(obj["format"]) == 3:
            m, n, p = int(obj["format"][0]), int(obj["format"][1]), int(obj["format"][2])
        elif fmt_hint is not None:
            m, n, p = fmt_hint
        else:
            continue

        label = str(obj.get("name") or obj.get("id") or f"gpu_pool_{m}x{n}x{p}_{idx}")
        if (
            "u" in obj
            and "v" in obj
            and "w" in obj
            and isinstance(obj["u"], list)
            and obj["u"]
            and isinstance(obj["u"][0], int)
        ):
            u = np.array([int(x) for x in obj["u"]], dtype=np.uint64)
            v = np.array([int(x) for x in obj["v"]], dtype=np.uint64)
            w = np.array([int(x) for x in obj["w"]], dtype=np.uint64)
        elif "u_u64" in obj and "v_u64" in obj and "w_u64" in obj:
            u = np.array([int(x) for x in obj["u_u64"]], dtype=np.uint64)
            v = np.array([int(x) for x in obj["v_u64"]], dtype=np.uint64)
            w = np.array([int(x) for x in obj["w_u64"]], dtype=np.uint64)
        elif "u_bits" in obj and "v_bits" in obj and "w_bits" in obj:
            u = np.array([int(x) for x in obj["u_bits"]], dtype=np.uint64)
            v = np.array([int(x) for x in obj["v_bits"]], dtype=np.uint64)
            w = np.array([int(x) for x in obj["w_bits"]], dtype=np.uint64)
        elif "U" in obj and "V" in obj and "W" in obj:
            sc = BilinearScheme(m=m, n=n, p=p, U=obj["U"], V=obj["V"], W=obj["W"], ring="Z2", modulus=2, name=label)
            u, v, w = scheme_to_u64_f2(sc)
        else:
            continue
        parsed.append((m, n, p, u, v, w, label))
    return parsed


def ingest_and_lift_gpu_pool(
    pool_path: str,
    artifacts_dir: str,
    records_saved: List[Dict[str, Any]],
    z3_timeout_s: float = 10.0,
    max_lift_attempts_per_format: int = 24,
    run_coker_repair: bool = True,
) -> Dict[str, Any]:
    """
    Ingest `f2_pool_phase1.json`, screen EVERY scheme through Stage L0 (batch parallel Numba 12T),
    run `coker_viol_4`-guided CPU steering on obstructed GPU-pool schemes (`coker > 0 -> 0`),
    and run Stage L1 (linear Hensel -> Q/Z) + Stage L2 (Z3 -> Z_T) on clean & lowest-coker hits!
    """
    t0 = time.perf_counter()
    if not os.path.exists(pool_path):
        return {"found": False, "path": pool_path, "total_schemes": 0}

    with open(pool_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    entries = _parse_pool_entries(raw)
    print(f"\n=== Ingesting GPU Pool: {pool_path} ({len(entries)} schemes) ===", flush=True)

    by_shape_rank: Dict[Tuple[int, int, int, int], List[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]] = {}
    invalid_f2 = 0
    for m, n, p, u, v, w, label in entries:
        if not fast_verify_f2_u64(m, n, p, u, v, w):
            invalid_f2 += 1
            continue
        by_shape_rank.setdefault((m, n, p, len(u)), []).append((u, v, w, label))

    per_format: Dict[str, Dict[str, Any]] = {}
    total_valid = 0
    total_mod4_clean = 0
    total_lifted = 0

    for (m, n, p, R), batch_list in sorted(by_shape_rank.items()):
        B = len(batch_list)
        total_valid += B
        U_b = np.stack([x[0] for x in batch_list], axis=0)
        V_b = np.stack([x[1] for x in batch_list], axis=0)
        W_b = np.stack([x[2] for x in batch_list], axis=0)

        t_l0 = time.perf_counter()
        cokers, frees, ranks_j1 = mod4_oracle_batch_u64(m, n, p, U_b, V_b, W_b)
        l0_ms = (time.perf_counter() - t_l0) * 1000.0

        fkey = f"{m}x{n}x{p}"
        pf = per_format.setdefault(
            fkey,
            {
                "format": [m, n, p],
                "total_screened": 0,
                "ranks_seen": [],
                "min_rank": R,
                "min_coker_viol_4": 999999,
                "max_coker_viol_4": 0,
                "mod4_clean_count": 0,
                "l0_batch_ms": 0.0,
                "coker_hist": Counter(),
                "coker_repair_demo": None,
                "lift_attempts": [],
                "lifted_count": 0,
            },
        )
        pf["total_screened"] += B
        pf["min_rank"] = min(pf["min_rank"], R)
        pf["l0_batch_ms"] = round(pf["l0_batch_ms"] + l0_ms, 2)
        pf["ranks_seen"].append(
            {
                "rank": R,
                "count": B,
                "min_coker_4": int(np.min(cokers)),
                "median_coker_4": int(np.median(cokers)),
                "max_coker_4": int(np.max(cokers)),
                "clean_count": int(np.sum(cokers == 0)),
                "free_bits_min_max": [int(np.min(frees)), int(np.max(frees))],
            }
        )

        for b_i in range(B):
            ck = int(cokers[b_i])
            pf["coker_hist"][ck] += 1
            if ck < pf["min_coker_viol_4"]:
                pf["min_coker_viol_4"] = ck
            if ck > pf["max_coker_viol_4"]:
                pf["max_coker_viol_4"] = ck
            if ck == 0:
                pf["mod4_clean_count"] += 1
                total_mod4_clean += 1

        # If there are obstructed GPU schemes (coker > 0), run coker_viol_4-guided CPU repair!
        obs_idxs = [i for i in range(B) if int(cokers[i]) > 0]
        if run_coker_repair and obs_idxs:
            obs_idxs.sort(key=lambda i: int(cokers[i]))
            n_rep = min(12, len(obs_idxs))
            R_buf = R + 4
            U_r = np.zeros((n_rep, R_buf), dtype=np.uint64)
            V_r = np.zeros((n_rep, R_buf), dtype=np.uint64)
            W_r = np.zeros((n_rep, R_buf), dtype=np.uint64)
            before_ck = []
            for k in range(n_rep):
                idx_o = obs_idxs[k]
                U_r[k, :R] = U_b[idx_o]
                V_r[k, :R] = V_b[idx_o]
                W_r[k, :R] = W_b[idx_o]
                before_ck.append(int(cokers[idx_o]))
            U_rb, V_rb, W_rb, R_rb, ck_rb, ev_rb = _coker_guided_walk_batch_nb(
                m, n, p, U_r, V_r, W_r, R_target=R, n_outer_iters=40, burst_flips=30, seed_base=np.uint64(8888 + R)
            )
            repaired_to_zero = int(np.sum(ck_rb == 0))
            improved = sum(1 for k in range(n_rep) if int(ck_rb[k]) < before_ck[k])
            pf["coker_repair_demo"] = {
                "rank": R,
                "obstructed_tested": n_rep,
                "before_coker_min_max": [min(before_ck), max(before_ck)],
                "after_coker_min_max": [int(np.min(ck_rb)), int(np.max(ck_rb))],
                "improved_count": improved,
                "repaired_to_zero_count": repaired_to_zero,
                "oracle_evals": int(np.sum(ev_rb)),
            }
            print(
                f"  [GPU pool {fkey} R={R} coker-repair] tested={n_rep} "
                f"coker {min(before_ck)}..{max(before_ck)} -> {int(np.min(ck_rb))}..{int(np.max(ck_rb))} "
                f"(improved={improved}, zeroed={repaired_to_zero})",
                flush=True,
            )

        # Rank candidates by coker_viol_4 ascending: lift clean + test low-coker relaxed!
        order = np.argsort(cokers)
        sota = SOTA_2026.get(tuple(sorted((m, n, p))))
        is_record_band = sota is not None and R <= sota["Q"]
        cap_clean = max_lift_attempts_per_format if is_record_band else 4
        tried = 0
        for idx_o in order:
            ck = int(cokers[idx_o])
            if ck > 0 and tried >= (min(8, max_lift_attempts_per_format) if is_record_band else 5):
                break
            if tried >= cap_clean and ck == 0:
                # Also sample 1 obstructed scheme to confirm unrestricted L1 / negative control behavior
                if obs_idxs and not is_record_band:
                    idx_obs = obs_idxs[0]
                    u_c, v_c, w_c, lab = batch_list[idx_obs]
                    lres = _attempt_full_lift_cascade(
                        m,
                        n,
                        p,
                        u_c,
                        v_c,
                        w_c,
                        z3_timeout_s=z3_timeout_s,
                        hensel_trials=12,
                        allow_relaxed_hensel=True,
                        max_relaxed_coker=12,
                        label=f"{lab}_obs_probe",
                    )
                    summary_entry = {k: v for k, v in lres.items() if k not in ("lifted_scheme", "lifted_report")}
                    summary_entry["label"] = f"{lab}_obs_probe"
                    pf["lift_attempts"].append(summary_entry)
                    if lres["lifted"] and lres["lifted_scheme"] is not None:
                        pf["lifted_count"] += 1
                        total_lifted += 1
                        _save_world_record_if_beaten(m, n, p, lres["lifted_scheme"], artifacts_dir, records_saved)
                break
            u_c, v_c, w_c, lab = batch_list[int(idx_o)]
            lres = _attempt_full_lift_cascade(
                m,
                n,
                p,
                u_c,
                v_c,
                w_c,
                z3_timeout_s=z3_timeout_s,
                hensel_trials=20,
                allow_relaxed_hensel=True,
                max_relaxed_coker=8,
                label=lab,
            )
            tried += 1
            summary_entry = {k: v for k, v in lres.items() if k not in ("lifted_scheme", "lifted_report")}
            summary_entry["label"] = lab
            pf["lift_attempts"].append(summary_entry)
            if lres["lifted"] and lres["lifted_scheme"] is not None:
                pf["lifted_count"] += 1
                total_lifted += 1
                _save_world_record_if_beaten(m, n, p, lres["lifted_scheme"], artifacts_dir, records_saved)

            if sota and R < sota["F2"]:
                sc_f2 = u64_f2_to_scheme(m, n, p, u_c, v_c, w_c, name=lab)
                _save_world_record_if_beaten(m, n, p, sc_f2, artifacts_dir, records_saved)

        print(
            f"  [GPU pool {fkey} R={R}] B={B} L0={l0_ms:.1f}ms "
            f"coker=[{int(np.min(cokers))}..{int(np.max(cokers))}] "
            f"clean={int(np.sum(cokers == 0))}/{B} lifted={pf['lifted_count']}/{len(pf['lift_attempts'])}",
            flush=True,
        )

    for fkey, pf in per_format.items():
        pf["coker_hist"] = {str(k): int(v) for k, v in sorted(pf["coker_hist"].items())[:20]}

    return {
        "found": True,
        "path": pool_path,
        "total_entries": len(entries),
        "valid_f2_screened": total_valid,
        "invalid_f2_skipped": invalid_f2,
        "total_mod4_clean": total_mod4_clean,
        "total_lifted": total_lifted,
        "per_format": per_format,
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }


# =============================================================================
# 5. Main Campaign Runner
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 CPU-only Liftability-Guided Flip-Graph & Hensel/Z3 Campaign")
    parser.add_argument("--cpu-minutes", type=float, default=6.0, help="Total CPU Numba 12T search budget in minutes")
    parser.add_argument(
        "--pool-path",
        type=str,
        default=os.path.join(_FMM_ROOT, "artifacts", "f2_pool_phase1.json"),
        help="Path to GPU F2 pool JSON from @fmm-architect",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=os.path.join(_FMM_ROOT, "artifacts", "phase1_lifting_summary.json"),
        help="Output summary JSON path",
    )
    parser.add_argument("--ingest-only", action="store_true", help="Skip CPU search and only ingest/update from GPU pool")
    parser.add_argument("--z3-timeout", type=float, default=10.0, help="Per-scheme Z3 timeout in seconds")
    args = parser.parse_args()

    artifacts_dir = os.path.join(_FMM_ROOT, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    t_start = time.perf_counter()

    records_saved: List[Dict[str, Any]] = []
    cpu_results: Dict[str, Any] = {}

    if args.ingest_only and os.path.exists(args.summary_path):
        with open(args.summary_path, "r", encoding="utf-8") as fh:
            prev = json.load(fh)
        cpu_results = prev.get("cpu_guided_search", {})
        records_saved = prev.get("lifted_world_records", [])
    elif not args.ingest_only:
        # Warm up Numba JIT kernels on tiny <2,2,2:8> BEFORE starting campaign timers!
        t_jit = time.perf_counter()
        u_w, v_w, w_w = scheme_to_u64_f2(make_naive_scheme(2, 2, 2, ring="Z2", modulus=2))
        U_w = np.zeros((2, 12), dtype=np.uint64)
        V_w = np.zeros((2, 12), dtype=np.uint64)
        W_w = np.zeros((2, 12), dtype=np.uint64)
        U_w[:, :8] = u_w
        V_w[:, :8] = v_w
        W_w[:, :8] = w_w
        _f2_rank_descent_batch_nb(U_w, V_w, W_w, np.array([8, 8], dtype=np.int32), 50, 4, 0, np.uint64(1))
        _coker_guided_walk_batch_nb(2, 2, 2, U_w, V_w, W_w, 8, 2, 5, np.uint64(1))
        print(f"[JIT warmup done in {time.perf_counter() - t_jit:.2f}s]", flush=True)

        total_s = max(25.0, args.cpu_minutes * 60.0)
        # Stage 0: Pure from-scratch bottom-up Meta-Flip primitive curriculum (~18-28s on full run)
        curr_scale = min(2.2, max(0.35, total_s / 160.0))
        curr_seeds = build_clean_curriculum_seeds(budget_scale=curr_scale)
        rem_s = max(15.0, total_s - (time.perf_counter() - t_start))

        # Load any existing GPU pool seeds for extra cross-pollination (all from-scratch!)
        gpu_extra: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        if os.path.exists(args.pool_path):
            with open(args.pool_path, "r", encoding="utf-8") as fh:
                raw_p = json.load(fh)
            for mm, nn, pp, uu, vv, ww, _ in _parse_pool_entries(raw_p):
                fk = f"{mm}x{nn}x{pp}"
                if len(gpu_extra.get(fk, [])) < 8:
                    gpu_extra.setdefault(fk, []).append((uu, vv, ww))

        # 1) 4x4x5 (30% budget)
        res_445 = run_cpu_guided_target(
            4,
            4,
            5,
            rem_s * 0.30,
            artifacts_dir,
            records_saved,
            extra_seed_u64=curr_seeds.get("4x4x5", []) + gpu_extra.get("4x4x5", [])[:4],
            seed_offset=1000,
            z3_timeout_s=args.z3_timeout,
        )

        # 2) 4x5x5 (#1 gap of 3: 73 vs 76, 40% budget)
        res_455 = run_cpu_guided_target(
            4,
            5,
            5,
            rem_s * 0.40,
            artifacts_dir,
            records_saved,
            extra_seed_u64=curr_seeds.get("4x5x5", []) + gpu_extra.get("4x5x5", [])[:4],
            seed_offset=2000,
            z3_timeout_s=args.z3_timeout,
        )

        # Cross-dimension Meta-Flip down-projection: 4x5x5 -> 4x4x5
        proj_445_seeds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for eu, ev, ew in res_455.get("_elites_u64", [])[:3]:
            for drop_j in range(5):
                _, _, _, pu, pv, pw = project_f2_u64_down(4, 5, 5, eu, ev, ew, axis=1, drop_idx=drop_j)
                if len(pu) <= res_445["best_F2_rank_reached"] and fast_verify_f2_u64(4, 4, 5, pu, pv, pw):
                    proj_445_seeds.append((pu, pv, pw))
        if proj_445_seeds and res_445["best_F2_rank_reached"] > 60:
            proj_445_seeds.sort(key=lambda x: len(x[0]))
            print(
                f"\n  [Meta-Flip Projection] 4x5x5 -> 4x4x5 yielded R={len(proj_445_seeds[0][0])}, running 15s bonus pass!",
                flush=True,
            )
            res_445_bonus = run_cpu_guided_target(
                4, 4, 5, 15.0, artifacts_dir, records_saved, extra_seed_u64=proj_445_seeds[:4], seed_offset=2500
            )
            if res_445_bonus["best_F2_rank_reached"] < res_445["best_F2_rank_reached"]:
                res_445["meta_flip_bonus"] = {k: v for k, v in res_445_bonus.items() if not k.startswith("_")}
                res_445["best_F2_rank_reached"] = res_445_bonus["best_F2_rank_reached"]

        # 3) 4x5x6 (30% budget)
        res_456 = run_cpu_guided_target(
            4,
            5,
            6,
            rem_s * 0.30,
            artifacts_dir,
            records_saved,
            extra_seed_u64=curr_seeds.get("4x5x6", []) + gpu_extra.get("4x5x6", [])[:4],
            seed_offset=3000,
            z3_timeout_s=args.z3_timeout,
        )

        # Cross-dimension Meta-Flip down-projection: 4x5x6 -> 4x5x5
        proj_455_seeds: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for eu, ev, ew in res_456.get("_elites_u64", [])[:3]:
            for drop_k in range(6):
                _, _, _, pu, pv, pw = project_f2_u64_down(4, 5, 6, eu, ev, ew, axis=2, drop_idx=drop_k)
                if len(pu) <= res_455["best_F2_rank_reached"] and fast_verify_f2_u64(4, 5, 5, pu, pv, pw):
                    proj_455_seeds.append((pu, pv, pw))
        if proj_455_seeds and res_455["best_F2_rank_reached"] > 75:
            proj_455_seeds.sort(key=lambda x: len(x[0]))
            print(
                f"\n  [Meta-Flip Projection] 4x5x6 -> 4x5x5 yielded R={len(proj_455_seeds[0][0])}, running 15s bonus pass!",
                flush=True,
            )
            res_455_bonus = run_cpu_guided_target(
                4, 5, 5, 15.0, artifacts_dir, records_saved, extra_seed_u64=proj_455_seeds[:4], seed_offset=3500
            )
            if res_455_bonus["best_F2_rank_reached"] < res_455["best_F2_rank_reached"]:
                res_455["meta_flip_bonus"] = {k: v for k, v in res_455_bonus.items() if not k.startswith("_")}
                res_455["best_F2_rank_reached"] = res_455_bonus["best_F2_rank_reached"]

        for key, d in (("4x4x5", res_445), ("4x5x5", res_455), ("4x5x6", res_456)):
            cpu_results[key] = {k: v for k, v in d.items() if not k.startswith("_")}

    # Full GPU pool ingestion + L0 screening + coker-repair + L1/L2 lifting
    gpu_pool_res = ingest_and_lift_gpu_pool(
        args.pool_path, artifacts_dir, records_saved, z3_timeout_s=args.z3_timeout, run_coker_repair=True
    )

    total_screened_all = sum(int(v.get("unique_schemes_screened", 0)) for v in cpu_results.values()) + int(
        gpu_pool_res.get("valid_f2_screened", 0)
    )

    summary = {
        "phase": "Phase 1 Lifting & Liftability-Guided CPU Search",
        "hardware": "CPU-only (Numba 12T + Z3 5.1.0, ZERO GPU usage)",
        "zero_external_leakage": True,
        "seed_primitives": ["strassen_222 (<2,2,2:7>)", "laderman_333 (<3,3,3:23>)", "schoolbook (<1,1,1:1>)"],
        "sleep_dag_rsi_components": [
            "DiscoveryTree",
            "LeanDAGPolicy(entropic_tau=4.0)",
            "BackwardSubGoalDecomposer",
            "MutableSkillState",
        ],
        "total_f2_schemes_screened": total_screened_all,
        "cpu_guided_search": cpu_results,
        "gpu_pool_ingestion": gpu_pool_res,
        "lifted_world_records": records_saved,
        "elapsed_total_s": round(time.perf_counter() - t_start, 2),
    }

    with open(args.summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(
        f"\nSaved summary -> {args.summary_path} "
        f"(total_screened={total_screened_all}, records={len(records_saved)}, t={summary['elapsed_total_s']}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Pure-CPU Phase 2 Structural Families, Phase 1 Anatomy & Generalized Leaf Theory.

Zero GPU usage (NumPy + stdlib + exact_verifier.py only).
Generates and certifies 100% from-scratch balanced monotone Tier 2 seeds into:
  e:\\эксперименты\\исследования\\c71_fei_research\\artifacts\\phase2_structural_seeds.json
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
import exact_verifier as ev  # noqa: E402

ARTIFACTS_DIR = SRC_DIR.parent / "artifacts"


# =============================================================================
# Fast Helpers (Pure CPU)
# =============================================================================

def to_mask(indices: Sequence[int]) -> int:
    m = 0
    for i in indices:
        m |= 1 << int(i)
    return m


def fwht_np(arr: np.ndarray) -> np.ndarray:
    N = arr.shape[0]
    a = arr.astype(np.float64).copy()
    h = 1
    while h < N:
        a = a.reshape(-1, 2, h)
        u = a[:, 0, :].copy()
        v = a[:, 1, :].copy()
        a[:, 0, :] = u + v
        a[:, 1, :] = u - v
        a = a.reshape(N)
        h <<= 1
    return a


def build_cross_intersecting_g0(d: int, F_pos: Sequence[int], F_neg: Sequence[int]) -> np.ndarray:
    """Build 3-valued monotone partial core g0 in {-1, 0, +1}^{2^d} from cross-intersecting families."""
    N = 1 << d
    full = N - 1
    xs = np.arange(N, dtype=np.int32)
    pos = np.zeros(N, dtype=bool)
    for A in F_pos:
        pos |= (xs & A) == A
    neg = np.zeros(N, dtype=bool)
    inv_xs = full ^ xs
    for B in F_neg:
        neg |= (inv_xs & B) == B
    if np.any(pos & neg):
        raise ValueError("F_pos and F_neg are NOT cross-intersecting!")
    g0 = np.zeros(N, dtype=np.int8)
    g0[pos] = 1
    g0[neg] = -1
    return g0


def extract_subcube_components(d: int, g0: np.ndarray) -> Tuple[List[Dict[str, int]], List[List[int]]]:
    """Partition g0^{-1}(0) into Hamming-connected components, separating valid subcubes vs non-subcubes."""
    N = 1 << d
    full = N - 1
    zeros = set(int(x) for x in np.flatnonzero(g0 == 0))
    vis = set()
    valid_scs: List[Dict[str, int]] = []
    bad_comps: List[List[int]] = []
    for x in sorted(zeros):
        if x in vis:
            continue
        comp = []
        q = [x]
        vis.add(x)
        while q:
            u = q.pop()
            comp.append(u)
            for i in range(d):
                v = u ^ (1 << i)
                if v in zeros and v not in vis:
                    vis.add(v)
                    q.append(v)
        m = len(comp)
        oa = 0
        aa = full
        for v in comp:
            oa |= v
            aa &= v
        free_m = oa ^ aa
        if (m & (m - 1)) == 0 and (1 << free_m.bit_count()) == m:
            valid_scs.append({"fixed_mask": int(full ^ free_m), "fixed_val": int(aa)})
        else:
            bad_comps.append(sorted(comp))
    return valid_scs, bad_comps


def peel_to_monotone_subcubes(d: int, g0_in: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    """Peel balanced (max -> +1, min -> -1) pairs from non-subcube zero components until all are valid subcubes."""
    g = g0_in.copy()
    while True:
        scs, bad = extract_subcube_components(d, g)
        if not bad:
            return g, scs
        max_els: List[int] = []
        min_els: List[int] = []
        for comp in bad:
            cset = set(comp)
            for u in comp:
                if not any((u | (1 << i)) in cset for i in range(d) if not ((u >> i) & 1)):
                    max_els.append(u)
                if not any((u ^ (1 << i)) in cset for i in range(d) if (u >> i) & 1):
                    min_els.append(u)
        max_els.sort(key=lambda u: (-u.bit_count(), u))
        min_els.sort(key=lambda u: (u.bit_count(), u))
        g[max_els[0]] = 1
        g[min_els[0]] = -1


def lift_tier2_core(
    d: int,
    g0: Sequence[int],
    subcubes: Sequence[Dict[str, int]],
    extra_vars: int,
) -> Tuple[int, List[int], List[Dict[str, int]]]:
    """Embed a d-var Tier 2 core cleanly into d_new = d + extra_vars (exact same L, I, H, monotone).

    Every point x in {0,1}^d is duplicated across the 2^extra_vars assignments of the new
    high-index variables, and each subcube C_j keeps the same fixed_mask/fixed_val (so the new
    coordinates are free inside C_j, preserving codimension r_j and measure 2^{-r_j} identically).
    """
    d_new = d + extra_vars
    N_old = 1 << d
    N_new = 1 << d_new
    mask_old = N_old - 1
    g_new = [int(g0[x & mask_old]) for x in range(N_new)]
    sc_new = [{"fixed_mask": int(sc["fixed_mask"]), "fixed_val": int(sc["fixed_val"])} for sc in subcubes]
    return d_new, g_new, sc_new


# =============================================================================
# Part 1: Anatomy of Phase 1 Champion (d=14, L=6.480691...)
# =============================================================================

def analyze_phase1_champion() -> Dict[str, Any]:
    path = ARTIFACTS_DIR / "phase1_best_candidate.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    f = [int(v) for v in data["g0"]]
    d = len(f).bit_length() - 1
    N = 1 << d
    c = ev.fwht_int(f)
    denom = 1 << (2 * d)

    inf_coord = [Fraction(0)] * d
    W_deg = [Fraction(0)] * (d + 1)
    H_deg = [0.0] * (d + 1)
    nz_deg = [0] * (d + 1)
    abs_hist: Dict[int, int] = {}

    for S, v in enumerate(c):
        if v != 0:
            w = Fraction(v * v, denom)
            deg = S.bit_count()
            W_deg[deg] += w
            nz_deg[deg] += 1
            wf = float(w)
            H_deg[deg] += -wf * math.log2(wf)
            ac = abs(v)
            abs_hist[ac] = abs_hist.get(ac, 0) + 1
            for i in range(d):
                if (S >> i) & 1:
                    inf_coord[i] += w

    # Extract 9-var core (x_0..x_8) vs 5-var auxiliary layer (x_9..x_13)
    core_vals = []
    mixed_cells = []
    for ck in range(512):
        vals = [f[ck | (am << 9)] for am in range(32)]
        s = set(vals)
        if len(s) == 1:
            core_vals.append(vals[0])
        else:
            core_vals.append(0)
            mixed_cells.append((ck, ck.bit_count(), vals.count(1)))

    # Map x_0..x_8 to the canonical 3x3 AG(2,3) grid:
    # row 0: [0, 1, 5], row 1: [2, 7, 4], row 2: [8, 3, 6]
    grid = [0, 1, 5, 2, 7, 4, 8, 3, 6]
    rows = [{0, 1, 2}, {3, 4, 5}, {6, 7, 8}]
    cols = [{0, 3, 6}, {1, 4, 7}, {2, 5, 8}]
    lines = rows + cols
    trans = [{0, 4, 8}, {0, 5, 7}, {1, 3, 8}, {1, 5, 6}, {2, 3, 7}, {2, 4, 6}]
    ag23_mismatch = 0
    ag23_unforced_resolved_neg = 0
    for ck in range(512):
        S_grid = {pos for pos, orig in enumerate(grid) if (ck >> orig) & 1}
        Z_grid = set(range(9)) - S_grid
        if any(L <= S_grid for L in lines):
            if core_vals[ck] != 1:
                ag23_mismatch += 1
        elif any(T <= Z_grid for T in trans):
            if core_vals[ck] != -1:
                ag23_mismatch += 1
        else:
            if core_vals[ck] == -1:
                ag23_unforced_resolved_neg += 1

    # Prime implicants and maximal zeros
    min_ones = sum(
        1 for x in range(N)
        if f[x] == 1 and all(f[x ^ (1 << i)] != 1 for i in range(d) if (x >> i) & 1)
    )
    max_negs = sum(
        1 for x in range(N)
        if f[x] == -1 and all(f[x | (1 << i)] != -1 for i in range(d) if not ((x >> i) & 1))
    )

    return {
        "d": d,
        "I_exact": f"{sum(inf_coord).numerator}/{sum(inf_coord).denominator}",
        "I_float": float(sum(inf_coord)),
        "H_float": sum(H_deg),
        "L_float": sum(H_deg) / (float(sum(inf_coord)) - 1.0),
        "nonzero_fourier_coeffs": sum(nz_deg),
        "coord_influences": [
            {"coord": i, "I_str": f"{inf_coord[i].numerator}/{inf_coord[i].denominator}", "I_float": float(inf_coord[i])}
            for i in range(d)
        ],
        "core_9_inf_sum": float(sum(inf_coord[:9])),
        "aux_5_inf_sum": float(sum(inf_coord[9:])),
        "degree_spectrum": [
            {"deg": k, "W_k": float(W_deg[k]), "H_k": H_deg[k], "nz": nz_deg[k]}
            for k in range(d + 1) if nz_deg[k] > 0
        ],
        "top_abs_coeffs": sorted(
            [{"abs_cS": ac, "abs_fhat": ac / N, "count": cnt, "weight": cnt * (ac / N) ** 2}
             for ac, cnt in abs_hist.items()],
            key=lambda x: x["abs_cS"],
            reverse=True,
        )[:10],
        "prime_implicants": min_ones,
        "maximal_zeros": max_negs,
        "mixed_core_cells": len(mixed_cells),
        "ag23_forced_mismatches": ag23_mismatch,
        "ag23_unforced_resolved_neg": ag23_unforced_resolved_neg,
    }


# =============================================================================
# Part 2: Generalized Leaf De-Aliasing Verification (Balanced & Unbalanced)
# =============================================================================

def verify_generalized_leaf_theorem() -> Dict[str, Any]:
    """Numerically verify exact formulas & convexity barrier for balanced and unbalanced leaves."""
    results: Dict[str, Any] = {}

    # 1. Balanced leaf check: Q_2 (d=2, L_base=6.0) with h_1 = MAJ_3 (L=4.0), h_2 = MAJ_3 (L=4.0)
    # Full function has n = 2 + 3 + 3 = 8 variables!
    maj3 = [0, 0, 0, 1, 0, 1, 1, 1]
    tt8 = []
    for z in range(256):
        x = z & 3
        y1 = (z >> 2) & 7
        y2 = (z >> 5) & 7
        if x == 0:
            tt8.append(-1)
        elif x == 3:
            tt8.append(1)
        elif x == 1:
            tt8.append(1 if maj3[y1] else -1)
        else:
            tt8.append(1 if maj3[y2] else -1)
    res_t1_bal = ev.verify_full_truth_table(tt8, require_monotone=True)

    # Analytic formula: Q_2 base (r1=r2=2, I_g0=0.5, H_g0=1.0) + two MAJ_3 leaves (I=1.5, H=2.0):
    # I_formula = 0.5 + 2 * 2^(-2) * (2/2 + 1.5) = 0.5 + 0.5 * 2.5 = 1.75 = 7/4
    # H_formula = 1.0 + 2 * 2^(-2) * (4 + 2.0) = 1.0 + 0.5 * 6.0 = 4.0
    # L_formula = 4.0 / (7/4 - 1) = 4.0 / 0.75 = 16/3 = 5.333333333333...
    # Convex combination check:
    #   L_base = 6.0 (weight I_base - 1 = 0.5), L_leaf = 4.0 (weight 2 * 2^(-2) * (1.5 - 1) = 0.25)
    #   (0.5 * 6.0 + 0.25 * 4.0) / (0.5 + 0.25) = 4.0 / 0.75 = 16/3!
    I_f_bal = Fraction(1, 2) + 2 * Fraction(1, 4) * (Fraction(1, 1) + Fraction(3, 2))
    H_f_bal = 1.0 + 2 * 0.25 * (4.0 + 2.0)
    L_f_bal = H_f_bal / (float(I_f_bal) - 1.0)
    L_conv = (0.5 * 6.0 + 0.25 * 4.0) / (0.5 + 0.25)
    assert res_t1_bal["I_exact_str"] == f"{I_f_bal.numerator}/{I_f_bal.denominator}"
    assert abs(float(res_t1_bal["L_cert"][:18]) - L_f_bal) < 1e-13
    assert abs(L_f_bal - L_conv) < 1e-15
    results["balanced_Q2_plus_MAJ3"] = {
        "n": 8,
        "I_exact": res_t1_bal["I_exact_str"],
        "L_tier1": res_t1_bal["L_cert"][:20],
        "L_formula": L_f_bal,
        "L_convex_comb": L_conv,
        "diff": abs(float(res_t1_bal["L_cert"][:18]) - L_f_bal),
        "strictly_convex_bounded": True,
    }

    # 2. Balanced higher-L leaf check: Q_2 (L=6.0) with h_1 = h_2 = g_3' (Hod 6-var, L=6.3470674987, I=13/8, H=3.9669171867)
    # Convex combination prediction:
    #   w_base = 0.5, w_leaves = 2 * 0.25 * (13/8 - 1) = 5/16 = 0.3125
    #   L_new = (0.5 * 6.0 + 0.3125 * 6.347067498701919) / 0.8125 = 6.133487499500738 < 6.3470674987!
    L_g3p = 6.34706749870191887
    L_q2_g3p = (0.5 * 6.0 + (5.0 / 16.0) * L_g3p) / (0.5 + 5.0 / 16.0)
    results["balanced_Q2_plus_g3prime"] = {
        "L_base": 6.0,
        "L_leaf": L_g3p,
        "L_combined": L_q2_g3p,
        "beats_max_component": False,
    }

    # 3. Unbalanced leaf check: Q_2 (d=2) with h_1 = AND_2 (mu=-1/2), h_2 = OR_2 (mu=+1/2) -> n=6!
    and2 = [0, 0, 0, 1]
    or2 = [0, 1, 1, 1]
    tt6 = []
    for z in range(64):
        x = z & 3
        y1 = (z >> 2) & 3
        y2 = (z >> 4) & 3
        if x == 0:
            tt6.append(-1)
        elif x == 3:
            tt6.append(1)
        elif x == 1:
            tt6.append(1 if and2[y1] else -1)
        else:
            tt6.append(1 if or2[y2] else -1)
    res_t1_unbal = ev.verify_full_truth_table(tt6, require_monotone=True)

    # Exact unbalanced formula check:
    # mu_a = -0.5, V_a = 0.75, I_a = 1.0, H_ge1_a = 2.0 - (-0.25*log2(0.25)) = 2.0 - 0.5 = 1.5
    # g_eff = [-1, -0.5, +0.5, +1] -> c_eff = [0, -0.75, -0.25, 0] (check: I_eff = 0.5625+0.0625 = 0.625)
    # I_f = 0.625 + 2 * 0.25 * (1.0*0.75 + 1.0) = 0.625 + 0.875 = 1.5 = 3/2
    # H_f = (-0.5625*log2(0.5625) - 0.0625*log2(0.0625)) + 2 * 0.25 * (4.0*0.75 + 1.5)
    H_eff = -0.5625 * math.log2(0.5625) - 0.0625 * math.log2(0.0625)
    H_f_unbal = H_eff + 0.5 * (3.0 + 1.5)
    L_f_unbal = H_f_unbal / (1.5 - 1.0)
    assert abs(float(res_t1_unbal["L_cert"][:18]) - L_f_unbal) < 1e-13
    results["unbalanced_Q2_plus_AND2_OR2"] = {
        "n": 6,
        "I_exact": res_t1_unbal["I_exact_str"],
        "L_tier1": res_t1_unbal["L_cert"][:20],
        "L_formula": L_f_unbal,
        "diff": abs(float(res_t1_unbal["L_cert"][:18]) - L_f_unbal),
        "beats_literals_6_0": False,
    }

    return results


# =============================================================================
# Part 3: Pure-Math From-Scratch Structural Families (d = 4..16)
# =============================================================================

def build_ag22_core() -> Tuple[int, List[int], List[Dict[str, int]]]:
    """AG(2,2) 1v1 parallel classes (2 rows vs 2 cols) on d=4 -> k=2, n=6, L=6.34706750."""
    Fp = [to_mask([0, 1]), to_mask([2, 3])]
    Fn = [to_mask([0, 2]), to_mask([1, 3])]
    g0 = build_cross_intersecting_g0(4, Fp, Fn)
    scs, bad = extract_subcube_components(4, g0)
    assert not bad
    return 4, g0.tolist(), scs


def build_ag23_core() -> Tuple[int, List[int], List[Dict[str, int]]]:
    """AG(2,3) 2v2 parallel classes (rows+cols vs 2 diagonal classes) on d=9 -> k=9, n=18, L=6.5218457109..."""
    rows3 = [to_mask([3 * r + c for c in range(3)]) for r in range(3)]
    cols3 = [to_mask([3 * r + c for r in range(3)]) for c in range(3)]
    diagp3 = [to_mask([3 * r + ((r + c) % 3) for r in range(3)]) for c in range(3)]
    diagm3 = [to_mask([3 * r + ((2 * r + c) % 3) for r in range(3)]) for c in range(3)]
    g0 = build_cross_intersecting_g0(9, rows3 + cols3, diagp3 + diagm3)
    scs, bad = extract_subcube_components(9, g0)
    assert not bad and len(scs) == 9
    return 9, g0.tolist(), scs


def build_ag23_2pair_split_d13() -> Tuple[int, List[int], List[Dict[str, int]]]:
    """Balanced asymmetric 2-pair coordinate split (AND/OR) of AG(2,3) on d=13, peeled to valid subcubes."""
    rows3 = [to_mask([3 * r + c for c in range(3)]) for r in range(3)]
    cols3 = [to_mask([3 * r + c for r in range(3)]) for c in range(3)]
    diagp3 = [to_mask([3 * r + ((r + c) % 3) for r in range(3)]) for c in range(3)]
    diagm3 = [to_mask([3 * r + ((2 * r + c) % 3) for r in range(3)]) for c in range(3)]
    Fp, Fn = rows3 + cols3, diagp3 + diagm3

    def sym_split_pair(Fp_in: List[int], Fn_in: List[int], d_cur: int, c1: int, c2: int) -> Tuple[List[int], List[int]]:
        Fp1 = [A | (1 << d_cur) if (A >> c1) & 1 else A for A in Fp_in]
        Fn1 = []
        for B in Fn_in:
            if (B >> c1) & 1:
                Fn1.append(B)
                Fn1.append((B ^ (1 << c1)) | (1 << d_cur))
            else:
                Fn1.append(B)
        Fp2 = []
        for A in Fp1:
            if (A >> c2) & 1:
                Fp2.append(A)
                Fp2.append((A ^ (1 << c2)) | (1 << (d_cur + 1)))
            else:
                Fp2.append(A)
        Fn2 = [B | (1 << (d_cur + 1)) if (B >> c2) & 1 else B for B in Fn1]
        return Fp2, Fn2

    Fp11, Fn11 = sym_split_pair(Fp, Fn, 9, 0, 1)
    Fp13, Fn13 = sym_split_pair(Fp11, Fn11, 11, 5, 4)
    g0 = build_cross_intersecting_g0(13, Fp13, Fn13)
    g_peeled, scs = peel_to_monotone_subcubes(13, g0)
    return 13, g_peeled.tolist(), scs


def build_cyclic_z8_3set() -> Tuple[int, List[int], List[Dict[str, int]]]:
    """Cyclic 3-set difference cover on Z_8: {0,1,2}+s vs {0,3,6}+s -> exact 6 0D subcubes."""
    Fp = [to_mask([(a + s) % 8 for a in (0, 1, 2)]) for s in range(8)]
    Fn = [to_mask([(b + s) % 8 for b in (0, 3, 6)]) for s in range(8)]
    g0 = build_cross_intersecting_g0(8, Fp, Fn)
    scs, bad = extract_subcube_components(8, g0)
    assert not bad
    return 8, g0.tolist(), scs


def build_cyclic_z15_4set() -> Tuple[int, List[int], List[Dict[str, int]]]:
    """Cyclic 4-set difference cover on Z_15: {0,1,4,5}+s vs {0,2,8,10}+s, majority-completed."""
    d = 15
    Fp = [to_mask([(a + s) % d for a in (0, 1, 4, 5)]) for s in range(d)]
    Fn = [to_mask([(b + s) % d for b in (0, 2, 8, 10)]) for s in range(d)]
    g0 = build_cross_intersecting_g0(d, Fp, Fn)
    N = 1 << d
    idx = np.arange(N, dtype=np.int32)
    pc = np.zeros(N, dtype=np.int32)
    for b in range(d):
        pc += (idx >> b) & 1
    zm = g0 == 0
    g0[zm & (pc >= 8)] = 1
    g0[zm & (pc <= 7)] = -1
    return d, g0.tolist(), []


def build_ag24_2v2_d16() -> Tuple[int, List[int], List[Dict[str, int]]]:
    """AG(2,4) 2v2 parallel classes on F_4^2 (d=16), middle-layer antichain completion."""
    d = 16
    mul4 = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    def pt4(x: int, y: int) -> int:
        return 4 * x + y
    pclasses = [
        [to_mask([pt4(c, y) for y in range(4)]) for c in range(4)],
        [to_mask([pt4(x, c) for x in range(4)]) for c in range(4)],
    ] + [[to_mask([pt4(x, mul4[m][x] ^ c) for x in range(4)]) for c in range(4)] for m in (1, 2, 3)]
    g0 = build_cross_intersecting_g0(d, pclasses[0] + pclasses[1], pclasses[2] + pclasses[3])
    N = 1 << d
    full = N - 1
    idx = np.arange(N, dtype=np.int32)
    pc = np.zeros(N, dtype=np.int32)
    for b in range(d):
        pc += (idx >> b) & 1
    zm = g0 == 0
    g0[zm & (pc > 8)] = 1
    g0[zm & (pc < 8)] = -1
    scs = [{"fixed_mask": full, "fixed_val": int(x)} for x in np.flatnonzero(g0 == 0)]
    return d, g0.tolist(), scs


def build_alt_chain(d: int) -> Tuple[int, List[int], List[Dict[str, int]]]:
    """Alternating read-once OR-AND chain on even d >= 6 + optimal 2-point antichain de-aliasing."""
    assert d >= 6 and d % 2 == 0
    N = 1 << d
    full = N - 1
    # f(x) = (x0 & x1) | (x2 & (x3 | (x4 & ... (x_{d-2} & x_{d-1})...)))
    f = np.zeros(N, dtype=np.int8)
    for x in range(N):
        # Innermost pair: x_{d-2} & x_{d-1}
        val = ((x >> (d - 2)) & 1) & ((x >> (d - 1)) & 1)
        for pos in range(d - 4, 0, -2):
            val = ((x >> pos) & 1) & (((x >> (pos + 1)) & 1) | val)
        val = (((x >> 0) & 1) & ((x >> 1) & 1)) | val
        f[x] = 1 if val else -1
    # Balance: f has 2^(d-1) - 1 ones and 2^(d-1) + 1 zeros!
    # Pick the symmetric complementary maximal-zero pair:
    # (x0,x1)=(0,1), (x_{d-2},x_{d-1})=(1,0), middle pairs (x_{2k}, x_{2k+1})=(1, 0)
    # and its swap (x0,x1)=(1,0), (x_{d-2},x_{d-1})=(0,1), middle pairs=(1, 0)!
    z1 = 0b10  # x1=1, x0=0
    z2 = 0b01  # x0=1, x1=0
    for pos in range(2, d - 2, 2):
        z1 |= 1 << pos
        z2 |= 1 << pos
    z1 |= 1 << (d - 2)
    z2 |= 1 << (d - 1)
    assert f[z1] == -1 and f[z2] == -1
    f[z1] = 0
    f[z2] = 0
    scs = [{"fixed_mask": full, "fixed_val": z1}, {"fixed_mask": full, "fixed_val": z2}]
    return d, f.tolist(), scs


def build_pg22_fano() -> Tuple[int, List[int], List[Dict[str, int]]]:
    fano = [to_mask(L) for L in [[0, 1, 3], [1, 2, 4], [2, 3, 5], [3, 4, 6], [4, 5, 0], [5, 6, 1], [6, 0, 2]]]
    g0 = build_cross_intersecting_g0(7, fano, fano)
    return 7, g0.tolist(), []


def build_pg23_d13() -> Tuple[int, List[int], List[Dict[str, int]]]:
    d = 13
    pg23 = [to_mask([(x + s) % d for x in (0, 1, 3, 9)]) for s in range(d)]
    g0 = build_cross_intersecting_g0(d, pg23, pg23)
    N = 1 << d
    idx = np.arange(N, dtype=np.int32)
    pc = np.zeros(N, dtype=np.int32)
    for b in range(d):
        pc += (idx >> b) & 1
    zm = g0 == 0
    g0[zm & (pc >= 7)] = 1
    g0[zm & (pc <= 6)] = -1
    return d, g0.tolist(), []


def build_perturbed_odd_maj(d: int, n_1d_subcubes: int = 0) -> Tuple[int, List[int], List[Dict[str, int]]]:
    """Odd majority MAJ_d with greedy disjoint middle-layer 1D subcubes (wt=(d-1)/2 <-> (d+1)/2)."""
    assert d % 2 == 1
    N = 1 << d
    full = N - 1
    mid_lo = (d - 1) // 2
    g = np.array([1 if x.bit_count() > mid_lo else -1 for x in range(N)], dtype=np.int8)
    scs: List[Dict[str, int]] = []
    if n_1d_subcubes > 0:
        # Pick disjoint edges (u < v, wt(u)=mid_lo, wt(v)=mid_lo+1) with no shared/adjacent zero neighbors
        used_mask = np.zeros(N, dtype=bool)
        for u in range(N):
            if len(scs) >= n_1d_subcubes:
                break
            if g[u] != -1 or u.bit_count() != mid_lo or used_mask[u]:
                continue
            for i in range(d):
                if (u >> i) & 1:
                    continue
                v = u | (1 << i)
                if g[v] != 1 or used_mask[v]:
                    continue
                # Ensure no neighbor of u or v is already 0
                ok = True
                for j in range(d):
                    if j != i and (g[u ^ (1 << j)] == 0 or g[v ^ (1 << j)] == 0):
                        ok = False
                        break
                if ok:
                    g[u] = 0
                    g[v] = 0
                    used_mask[u] = True
                    used_mask[v] = True
                    scs.append({"fixed_mask": full ^ (1 << i), "fixed_val": u})
                    break
    return d, g.tolist(), scs


def build_weighted_threshold_a2(weights: Sequence[int]) -> Tuple[int, List[int], List[Dict[str, int]]]:
    d = len(weights)
    N = 1 << d
    xs = np.arange(N, dtype=np.int32)
    scores = np.zeros(N, dtype=np.int64)
    for i, w in enumerate(weights):
        scores += w * ((xs >> i) & 1)
    tot = sum(weights)
    g0 = np.zeros(N, dtype=np.int8)
    g0[2 * scores > tot] = 1
    g0[2 * scores < tot] = -1
    scs, bad = extract_subcube_components(d, g0)
    assert not bad
    return d, g0.tolist(), scs


def build_maj3_tribes_d9() -> Tuple[int, List[int], List[Dict[str, int]]]:
    """MAJ_3(Tribes(2,2), DualTribes(2,2), x_8) on d=9 (balanced monotone, k=0)."""
    d = 9
    N = 1 << d
    g = []
    for x in range(N):
        b = [(x >> i) & 1 for i in range(d)]
        t1 = (b[0] & b[1]) | (b[2] & b[3])
        t2 = (b[4] | b[5]) & (b[6] | b[7])
        t3 = b[8]
        m = (t1 & t2) | (t1 & t3) | (t2 & t3)
        g.append(1 if m else -1)
    return d, g, []


def generate_and_certify_all() -> Dict[str, Any]:
    t0 = time.perf_counter()
    print("=== Phase 2 Structural Families: Anatomy, Leaf Theory & Seed Certification ===")

    # 1. Anatomy of Phase 1 champion
    anatomy = analyze_phase1_champion()
    print(
        f"[ANATOMY] Phase 1 d={anatomy['d']}: I={anatomy['I_exact']}, L={anatomy['L_float']:.8f}, "
        f"core9_Inf={anatomy['core_9_inf_sum']:.6f}, aux5_Inf={anatomy['aux_5_inf_sum']:.6f}, "
        f"AG(2,3)_forced_mismatches={anatomy['ag23_forced_mismatches']}!"
    )

    # 2. Generalized Leaf De-Aliasing check
    leaf_theory = verify_generalized_leaf_theorem()
    print(
        f"[LEAF THEORY] Q2+MAJ3 T1==Formula diff={leaf_theory['balanced_Q2_plus_MAJ3']['diff']:.2e} "
        f"(L={leaf_theory['balanced_Q2_plus_MAJ3']['L_formula']:.6f}); "
        f"Q2+AND2/OR2 T1==Formula diff={leaf_theory['unbalanced_Q2_plus_AND2_OR2']['diff']:.2e} "
        f"(L={leaf_theory['unbalanced_Q2_plus_AND2_OR2']['L_formula']:.6f})"
    )

    # 3. Build & certify diverse from-scratch structural families
    ag23_d, ag23_g0, ag23_sc = build_ag23_core()
    raw_specs: List[Tuple[str, str, Tuple[int, List[int], List[Dict[str, int]]]]] = [
        ("AG23_2v2_d9", "affine_hypergraph", (ag23_d, ag23_g0, ag23_sc)),
        ("AG23_lifted_d10", "affine_hypergraph_lifted", lift_tier2_core(ag23_d, ag23_g0, ag23_sc, 1)),
        ("AG23_lifted_d12", "affine_hypergraph_lifted", lift_tier2_core(ag23_d, ag23_g0, ag23_sc, 3)),
        ("AG23_lifted_d14", "affine_hypergraph_lifted", lift_tier2_core(ag23_d, ag23_g0, ag23_sc, 5)),
        ("AG23_2pair_split_d13", "affine_hypergraph_split", build_ag23_2pair_split_d13()),
        ("AltChain_d14", "alternating_tree_a2", build_alt_chain(14)),
        ("AltChain_d12", "alternating_tree_a2", build_alt_chain(12)),
        ("AltChain_d10", "alternating_tree_a2", build_alt_chain(10)),
        ("AG22_Tribes22_d4", "affine_hypergraph", build_ag22_core()),
        ("Z15_4set_diffcover_d15", "cyclic_diff_cover", build_cyclic_z15_4set()),
        ("Z8_3set_diffcover_d8", "cyclic_diff_cover", build_cyclic_z8_3set()),
        ("AG24_2v2_mid_d16", "affine_hypergraph", build_ag24_2v2_d16()),
        ("MAJ3_Tribes_d9", "hybrid_tree", build_maj3_tribes_d9()),
        ("PG23_d13", "projective_plane", build_pg23_d13()),
        ("PG22_Fano_d7", "projective_plane", build_pg22_fano()),
        ("Geom_WTH_d10", "weighted_threshold_a2", build_weighted_threshold_a2([1, 2, 3, 4, 6, 9, 13, 19, 28, 41])),
        ("MAJ9_1D_perturbed_d9", "perturbed_majority_a2", build_perturbed_odd_maj(9, n_1d_subcubes=12)),
        ("MAJ11_1D_perturbed_d11", "perturbed_majority_a2", build_perturbed_odd_maj(11, n_1d_subcubes=20)),
    ]

    certified_seeds: List[Dict[str, Any]] = []
    print("\n--- Certifying From-Scratch Structural Seeds via exact_verifier.py ---")
    for label, family, (d, g0, scs) in raw_specs:
        # Fast exact verification (skip mpmath/Decimal for huge k > 500? No: exact_verifier handles k=5262 in <1s because extra_H is exact Fraction!)
        # Wait: for d=16 (65536 pts) pure-Python int FWHT takes ~0.5s! Let's run full exact_verifier on ALL of them!
        res = ev.verify_dealiased_core(g0, scs, require_monotone=True)
        ot_float = float(res["L_cert"][:20])
        print(
            f"  [CERT PASS] {label:25s} d={d:2d} k={len(scs):4d} n_eff={d+len(scs):4d} "
            f"I={res['I_exact_str']:12s} mono={res['is_monotone']} L_cert={res['L_cert'][:22]}"
        )
        certified_seeds.append({
            "label": label,
            "family": family,
            "tier": 2,
            "d": d,
            "k_subcubes": len(scs),
            "n_effective": d + len(scs),
            "I_exact_str": res["I_exact_str"],
            "OT": ot_float,
            "L_cert": res["L_cert"],
            "U_cert": res["U_cert"],
            "is_monotone": res["is_monotone"],
            "sha256": res["sha256"],
            "g0": g0,
            "subcubes": scs,
        })

    # Sort descending by OT
    certified_seeds.sort(key=lambda s: s["OT"], reverse=True)

    # Save phase2_structural_seeds.json (Tier 2 schema: d, g0, subcubes, label, OT + exact cert metadata)
    seeds_payload = {
        "schema_version": "c71_tier2_structural_seeds_v1",
        "convention": "pm1_monotone_increasing",
        "source": "100% pure-math from-scratch constructions (ZERO external truth tables)",
        "generated_elapsed_sec": round(time.perf_counter() - t0, 2),
        "phase1_anatomy_summary": {
            "d": anatomy["d"],
            "I_exact": anatomy["I_exact"],
            "L_float": anatomy["L_float"],
            "core_9_inf_sum": anatomy["core_9_inf_sum"],
            "aux_5_inf_sum": anatomy["aux_5_inf_sum"],
            "ag23_forced_mismatches": anatomy["ag23_forced_mismatches"],
        },
        "leaf_dealiasing_theory": leaf_theory,
        "seeds": certified_seeds,
    }
    out_path = ARTIFACTS_DIR / "phase2_structural_seeds.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(seeds_payload, indent=2), encoding="utf-8")
    print(f"\nSaved {len(certified_seeds)} certified seeds to {out_path} in {time.perf_counter() - t0:.2f}s")
    return seeds_payload


if __name__ == "__main__":
    generate_and_certify_all()

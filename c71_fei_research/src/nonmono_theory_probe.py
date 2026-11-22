"""Phase 4 Pure-CPU Non-Monotone (Strictly Non-Unate) & A2-NonMono Theory Probe.

Zero GPU usage (NumPy + stdlib + exact_verifier.py only).

Tasks implemented:
1. Exact A2-NonMono verification on strictly non-unate / comparable-subcube
   functions confirming Tier 1 == Tier 2 > 55 decimal digits in exact_verifier.py.
2. Exact is_unate_tier1 & is_unate_tier2 checkers (cross-verified against each
   other on all small benchmarks), distinguishing genuinely non-unate functions
   from sign-flipped copies of monotone functions.
3. Systematic CPU probe of 4 non-unate families (d = 5..12):
   - Family 1: MUX/ITE & non-monotone decision cascades (incl. MUX on AG(2,3),
     Tribes(2,2), AltTree6).
   - Family 2: Localized subcube sign-flips, localized XOR masks, and comparable
     zero-subcube splits on AG(2,3) (d=9) and Z10 (d=10).
   - Family 3: Small-d (d = 5, 6, 7, 8) strictly non-unate vs monotone/unate SA
     + steepest 2-swap + A2-NonMono benchmark mapping the exact non-unate gap.
   - Family 4: Non-monotone Bent / Reed-Muller / quadratic-on-subcube hybrids
     with low influence + A2-NonMono (d = 6, 8, 10).
4. Saves certified strictly non-unate Tier 2 seeds + benchmark summary to:
   e:\\эксперименты\\исследования\\c71_fei_research\\artifacts\\phase4_nonmono_theory.json
"""

from __future__ import annotations

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
import structural_families as sf  # noqa: E402
import hypergraph_catalog as hc  # noqa: E402

ARTIFACTS_DIR = SRC_DIR.parent / "artifacts"


# =============================================================================
# Core Exact Unateness Checkers & Fast A2-NonMono Evaluators
# =============================================================================

def is_unate_tier1(values_pm1: Sequence[int]) -> Tuple[bool, int]:
    """Exact unateness check for full Boolean f: {0,1}^n -> {-1, +1}.

    Returns (is_unate, non_unate_coord_count).
    """
    f = np.asarray(values_pm1, dtype=np.int8)
    N = f.shape[0]
    n = N.bit_length() - 1
    bad_coords = 0
    for i in range(n):
        v = f.reshape(-1, 2, 1 << i)
        diff = v[:, 1, :] - v[:, 0, :]
        if np.any(diff > 0) and np.any(diff < 0):
            bad_coords += 1
    return (bad_coords == 0), bad_coords


def is_unate_tier2(
    d: int,
    g0: Sequence[int],
    subcubes: Sequence[Dict[str, int]],
) -> Tuple[bool, int]:
    """Exact unateness check for Tier 2 F(x, y) = g0(x) + sum_j 1_{C_j}(x) * y_j.

    A core coordinate i is non-unate iff EITHER:
      - some edge along i has g0(u) < g0(v) AND another edge has g0(u') > g0(v'), OR
      - some edge along i connects two DISTINCT zero-subcubes C_a != C_b (where
        varying y_a, y_b produces both +2 and -2 edge transitions).
    Auxiliary coordinates y_j are always positive-unate.
    Returns (is_unate, non_unate_coord_count).
    """
    g = np.asarray(g0, dtype=np.int8)
    N = 1 << d
    cid = np.full(N, -1, dtype=np.int32)
    for j, sc in enumerate(subcubes):
        fm, fv = int(sc["fixed_mask"]), int(sc["fixed_val"])
        for x in ev._iter_subcube_points(fm, fv, d):
            cid[x] = j

    bad_coords = 0
    for i in range(d):
        step = 1 << i
        gv = g.reshape(-1, 2, step)
        cv = cid.reshape(-1, 2, step)
        g_lo, g_hi = gv[:, 0, :], gv[:, 1, :]
        c_lo, c_hi = cv[:, 0, :], cv[:, 1, :]
        # Distinct zero-subcube adjacency along coordinate i => both + and - occur!
        if np.any((g_lo == 0) & (g_hi == 0) & (c_lo != c_hi)):
            bad_coords += 1
            continue
        has_inc = bool(np.any(g_lo < g_hi))
        has_dec = bool(np.any(g_lo > g_hi))
        if has_inc and has_dec:
            bad_coords += 1
    return (bad_coords == 0), bad_coords


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


_PC_CACHE: Dict[int, np.ndarray] = {}


def get_pc(d: int) -> np.ndarray:
    if d not in _PC_CACHE:
        idx = np.arange(1 << d, dtype=np.uint32)
        pc = np.zeros(1 << d, dtype=np.float64)
        for b in range(d):
            pc += (idx >> b) & 1
        _PC_CACHE[d] = pc
    return _PC_CACHE[d]


def auto_maximal_subcubes_nonmono(d: int, g0: np.ndarray) -> List[Dict[str, int]]:
    """Partition g0^{-1}(0) into maximal disjoint subcubes for arbitrary (non-monotone) g0.

    Connected components that are valid subcubes stay whole (avoiding the deltaH/deltaI=4
    penalty of splitting adjacent cells); non-subcube components are greedily tiled by
    largest disjoint subcubes (2D -> 1D -> 0D).
    """
    N = 1 << d
    full = N - 1
    zeros = set(int(x) for x in np.flatnonzero(g0 == 0))
    vis = set()
    scs: List[Dict[str, int]] = []
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
        fm = oa ^ aa
        if (m & (m - 1)) == 0 and (1 << fm.bit_count()) == m:
            scs.append({"fixed_mask": int(full ^ fm), "fixed_val": int(aa)})
        else:
            rem = set(comp)
            # First greedily pull any 2D subcubes (size 4), then 1D (size 2), then 0D (size 1)
            for u in sorted(list(rem)):
                if u not in rem:
                    continue
                found2 = False
                for i in range(d):
                    u_i = u ^ (1 << i)
                    if u_i not in rem:
                        continue
                    for j in range(i + 1, d):
                        u_j = u ^ (1 << j)
                        u_ij = u_i ^ (1 << j)
                        if u_j in rem and u_ij in rem:
                            fm2 = full ^ (1 << i) ^ (1 << j)
                            fv2 = u & fm2
                            scs.append({"fixed_mask": int(fm2), "fixed_val": int(fv2)})
                            rem.difference_update({u, u_i, u_j, u_ij})
                            found2 = True
                            break
                    if found2:
                        break
            while rem:
                u = min(rem)
                paired = False
                for i in range(d):
                    v = u ^ (1 << i)
                    if v in rem:
                        scs.append({"fixed_mask": int(full ^ (1 << i)), "fixed_val": int(u & v)})
                        rem.remove(u)
                        rem.remove(v)
                        paired = True
                        break
                if not paired:
                    scs.append({"fixed_mask": full, "fixed_val": u})
                    rem.remove(u)
    return scs


def eval_tier2_fast(d: int, g0: np.ndarray, subcubes: Sequence[Dict[str, int]]) -> Tuple[float, float, float]:
    N = 1 << d
    a = fwht_np(g0)
    w = (a / N) ** 2
    pc = get_pc(d)
    I_t = float(np.dot(pc, w)) + sum(
        (2 + int(s["fixed_mask"]).bit_count()) / (1 << (int(s["fixed_mask"]).bit_count() + 1))
        for s in subcubes
    )
    nz = w > 1e-300
    H_t = float(-np.sum(w[nz] * np.log2(w[nz]))) + sum(
        (2 * int(s["fixed_mask"]).bit_count()) / (1 << int(s["fixed_mask"]).bit_count())
        for s in subcubes
    )
    if I_t <= 1.0 + 1e-12:
        return 0.0, I_t, H_t
    return H_t / (I_t - 1.0), I_t, H_t


# =============================================================================
# Task 1: Verify A2-NonMono Exactness (Tier 1 == Tier 2 > 55 dps, Non-Unate)
# =============================================================================

def run_task1_a2_nonmono_exactness() -> List[Dict[str, Any]]:
    print("=== Task 1: A2-NonMono Exactness (Comparable / Strictly Non-Unate Subcubes) ===")
    cases: List[Tuple[str, int, List[int], List[Dict[str, int]]]] = [
        # Case 1: d=3, k=2 (n=5), comparable 0D points {001, 011} (adjacent edge between C_0 != C_1)
        (
            "A2NonMono_Q3_comparable_0D_n5",
            3,
            [-1, 0, -1, 0, -1, 1, 1, 1],
            [{"fixed_mask": 7, "fixed_val": 1}, {"fixed_mask": 7, "fixed_val": 3}],
        ),
        # Case 2: d=4, k=3 (n=7), strictly non-unate g0 with 1D subcube {0000,0001} + 2 comparable 0D points {0010, 0110}
        (
            "A2NonMono_Q4_mixed1D_comparable_n7",
            4,
            # Zeros at 0,1 (1D subcube), 2 (0D), 6 (0D) => 4 zeros, need 6 (+1) and 6 (-1)!
            # Make g0 itself non-unate by putting +1 at 3,4,5,8,9,10 and -1 at 7,11,12,13,14,15
            [0, 0, 0, 1, 1, 1, 0, -1, 1, 1, 1, -1, -1, -1, -1, -1],
            [
                {"fixed_mask": 14, "fixed_val": 0},
                {"fixed_mask": 15, "fixed_val": 2},
                {"fixed_mask": 15, "fixed_val": 6},
            ],
        ),
        # Case 3: d=4, k=2 (n=6), Tribes(2,2) with 1 non-monotone swap + comparable 0D pair {0011, 0111}
        (
            "A2NonMono_Q4_Tribes_nonunate_n6",
            4,
            # Start from Tribes22_dealiased, flip (0000: -1 -> +1) and (1111: +1 -> -1), zeros at 3 and 7 (comparable!)
            [1, -1, -1, 0, -1, -1, 1, 0, -1, 1, -1, 1, 1, 1, 1, -1],
            [{"fixed_mask": 15, "fixed_val": 3}, {"fixed_mask": 15, "fixed_val": 7}],
        ),
    ]

    verified = []
    for name, d, g0, scs in cases:
        res_t2 = ev.verify_dealiased_core(g0, scs, require_monotone=False)
        full_tt = ev.expand_dealiased_to_full(g0, scs)
        res_t1 = ev.verify_full_truth_table(full_tt, require_monotone=False)
        u2, bad2 = is_unate_tier2(d, g0, scs)
        u1, bad1 = is_unate_tier1(full_tt)
        assert u2 == u1 and bad2 == bad1, f"{name}: is_unate mismatch T2=({u2},{bad2}) vs T1=({u1},{bad1})"
        assert u2 is False, f"{name} expected strictly non-unate!"
        assert res_t2["is_monotone"] is False and res_t1["is_monotone"] is False
        assert res_t2["I_exact_str"] == res_t1["I_exact_str"]
        assert res_t2["L_cert"][:55] == res_t1["L_cert"][:55]
        assert res_t2["U_cert"][:55] == res_t1["U_cert"][:55]
        print(
            f"  [PASS] {name:34s} d={d} k={len(scs)} n={res_t1['n']} "
            f"is_unate={u2} (bad_coords={bad2}) I={res_t2['I_exact_str']:6s} "
            f"L_cert={res_t2['L_cert'][:18]} (T1==T2 >55 dps!)"
        )
        verified.append({
            "name": name,
            "d": d,
            "k": len(scs),
            "n_effective": res_t1["n"],
            "is_unate": u2,
            "non_unate_coords": bad2,
            "I_exact_str": res_t2["I_exact_str"],
            "L_cert": res_t2["L_cert"],
            "tier1_equals_tier2_55dps": True,
        })
    return verified


# =============================================================================
# Task 3: Four Non-Unate Families (d = 5..12)
# =============================================================================

def permute_tt(arr: np.ndarray, d: int, perm: Sequence[int], sign_flip_mask: int = 0, negate_out: bool = False) -> np.ndarray:
    N = 1 << d
    out = np.empty(N, dtype=np.int8)
    for x in range(N):
        y = 0
        xf = x ^ sign_flip_mask
        for i in range(d):
            if (xf >> i) & 1:
                y |= 1 << perm[i]
        out[x] = -arr[y] if negate_out else arr[y]
    return out


def probe_family1_mux(ag9: np.ndarray) -> List[Dict[str, Any]]:
    print("\n=== Family 1: MUX/ITE & Non-Monotone Decision Cascades (d = 5..12) ===")
    out = []

    def add_mux(label: str, d_base: int, f1: np.ndarray, f2: np.ndarray) -> None:
        d = d_base + 1
        g = np.concatenate([f1, f2])
        assert int(np.sum(g)) == 0
        scs = auto_maximal_subcubes_nonmono(d, g)
        L, I_t, H_t = eval_tier2_fast(d, g, scs)
        unate, bad = is_unate_tier2(d, g.tolist(), scs)
        print(f"  {label:36s} d={d:2d} k={len(scs):2d} unate={str(unate):5s}({bad}) I={I_t:.5f} H={H_t:.5f} L={L:.8f}")
        out.append({
            "label": label, "family": "family1_mux", "d": d, "k_subcubes": len(scs),
            "is_unate": unate, "non_unate_coords": bad, "I": I_t, "H": H_t, "OT": L,
            "g0": g.tolist(), "subcubes": scs,
        })

    # MUX on AG(2,3) (d=9 -> d=10)
    add_mux("MUX_d10_AG23_neg", 9, ag9, -ag9)
    p_swap = [1, 0, 2, 3, 4, 5, 6, 7, 8]
    add_mux("MUX_d10_AG23_swap01", 9, ag9, permute_tt(ag9, 9, p_swap))
    p_3cyc = [1, 2, 0, 3, 4, 5, 6, 7, 8]
    add_mux("MUX_d10_AG23_3cyc_row0", 9, ag9, permute_tt(ag9, 9, p_3cyc))
    add_mux("MUX_d10_AG23_flip_x0", 9, ag9, permute_tt(ag9, 9, list(range(9)), sign_flip_mask=1))
    p_dual = [3 * ((r + c) % 3) + ((r - c) % 3) for r in range(3) for c in range(3)]
    add_mux("MUX_d10_AG23_dual", 9, ag9, permute_tt(ag9, 9, p_dual))

    # MUX on Tribes(2,2) de-aliased (d=4 -> d=5 -> d=6)
    _, t22_list, _ = sf.build_ag22_core()
    t22 = np.array(t22_list, dtype=np.int8)
    add_mux("MUX_d5_Tribes22_swap02", 4, t22, permute_tt(t22, 4, [2, 1, 0, 3]))
    add_mux("MUX_d5_Tribes22_flip_x0", 4, t22, permute_tt(t22, 4, [0, 1, 2, 3], sign_flip_mask=1))
    # 2-level MUX on Tribes(2,2) -> d=6
    m5_a = np.concatenate([t22, permute_tt(t22, 4, [2, 1, 0, 3])])
    m5_b = np.concatenate([permute_tt(t22, 4, [0, 2, 1, 3]), permute_tt(t22, 4, [3, 1, 2, 0])])
    add_mux("MUX2_d6_Tribes22_4perms", 5, m5_a, m5_b)

    # MUX on AltChain d=6 -> d=7
    _, alt6_list, _ = sf.build_alt_chain(6)
    alt6 = np.array(alt6_list, dtype=np.int8)
    add_mux("MUX_d7_AltTree6_swap02", 6, alt6, permute_tt(alt6, 6, [2, 1, 0, 3, 4, 5]))
    add_mux("MUX_d7_AltTree6_flip_x0", 6, alt6, permute_tt(alt6, 6, list(range(6)), sign_flip_mask=1))

    # MUX on Z10_step2_2orb_34 (d=10 -> d=11)
    _, z10_fp, z10_fn = hc.build_cyclic_suborbit_pair(10, 2, [(0, 3, 7), (0, 2, 4, 7)], 3, 0)
    z10 = hc.build_g0_from_hypergraph(10, z10_fp, z10_fn)
    p10_swap = [1, 0] + list(range(2, 10))
    add_mux("MUX_d11_Z10_swap01", 10, z10, permute_tt(z10, 10, p10_swap))

    return out


def probe_family2_subcube_flips_and_splits(ag9: np.ndarray, ag9_sc: List[Dict[str, int]]) -> List[Dict[str, Any]]:
    print("\n=== Family 2: Subcube Sign-Flips, Localized XOR Masks & Comparable A2-NonMono Splits ===")
    out = []

    # 2a. Comparable 0D split of 1 of AG(2,3)'s 9 1D subcubes -> strictly non-unate!
    # Keep 8 1D subcubes, split the first 1D subcube {a, b} into two adjacent 0D subcubes {a}, {b}!
    fm0, fv0 = ag9_sc[0]["fixed_mask"], ag9_sc[0]["fixed_val"]
    pts0 = list(ev._iter_subcube_points(fm0, fv0, 9))
    assert len(pts0) == 2
    sc_split1 = [{"fixed_mask": 511, "fixed_val": pts0[0]}, {"fixed_mask": 511, "fixed_val": pts0[1]}] + ag9_sc[1:]
    L, I_t, H_t = eval_tier2_fast(9, ag9, sc_split1)
    unate, bad = is_unate_tier2(9, ag9.tolist(), sc_split1)
    print(f"  AG23_d9_1comparable_0D_split         d= 9 k={len(sc_split1):2d} unate={str(unate):5s}({bad}) I={I_t:.6f} H={H_t:.6f} L={L:.10f}")
    out.append({
        "label": "AG23_d9_1comparable_0D_split", "family": "family2_comparable_split",
        "d": 9, "k_subcubes": len(sc_split1), "is_unate": unate, "non_unate_coords": bad,
        "I": I_t, "H": H_t, "OT": L, "g0": ag9.tolist(), "subcubes": sc_split1,
    })

    # 2b. Best 1-pair non-monotone swap on AG(2,3) (d=9): pick 1 (+1) and 1 (-1) creating non-unate violation
    # Sample 400 diverse non-monotone 1-pair swaps near the wt-4/wt-5 boundary!
    pos_idx = [int(x) for x in np.flatnonzero(ag9 == 1)]
    neg_idx = [int(x) for x in np.flatnonzero(ag9 == -1)]
    rng = random.Random(2026)
    best_swap = None
    for _ in range(500):
        u = rng.choice(pos_idx)
        v = rng.choice(neg_idx)
        g_try = ag9.copy()
        g_try[u] = -1
        g_try[v] = 1
        unate_t, bad_t = is_unate_tier2(9, g_try.tolist(), ag9_sc)
        if unate_t:
            continue
        L_t, I_t, H_t = eval_tier2_fast(9, g_try, ag9_sc)
        if best_swap is None or L_t > best_swap[0]:
            best_swap = (L_t, I_t, H_t, bad_t, g_try.tolist())
    assert best_swap is not None
    print(f"  AG23_d9_best_nonmono_2swap           d= 9 k={len(ag9_sc):2d} unate=False({best_swap[3]}) I={best_swap[1]:.6f} H={best_swap[2]:.6f} L={best_swap[0]:.10f}")
    out.append({
        "label": "AG23_d9_best_nonmono_2swap", "family": "family2_nonmono_swap",
        "d": 9, "k_subcubes": len(ag9_sc), "is_unate": False, "non_unate_coords": best_swap[3],
        "I": best_swap[1], "H": best_swap[2], "OT": best_swap[0], "g0": best_swap[4], "subcubes": ag9_sc,
    })

    # 2c. Best balanced codim-r subcube sign-flip g0[C] *= -1 on AG(2,3) (d=9)
    best_flip = None
    for r_sc in [2, 3, 4]:
        checked = 0
        for coords in itertools.combinations(range(9), r_sc):
            fm = sum(1 << c for c in coords)
            for vb in range(1 << r_sc):
                fv = 0
                for bi, c in enumerate(coords):
                    if (vb >> bi) & 1:
                        fv |= 1 << c
                pts = list(ev._iter_subcube_points(fm, fv, 9))
                if int(np.sum(ag9[pts])) != 0:
                    continue
                g_try = ag9.copy()
                g_try[pts] = -g_try[pts]
                scs = auto_maximal_subcubes_nonmono(9, g_try)
                u_t, b_t = is_unate_tier2(9, g_try.tolist(), scs)
                checked += 1
                if not u_t:
                    L_t, I_t, H_t = eval_tier2_fast(9, g_try, scs)
                    if best_flip is None or L_t > best_flip[0]:
                        best_flip = (L_t, I_t, H_t, b_t, g_try.tolist(), scs, r_sc)
                if checked >= 150:
                    break
            if checked >= 150:
                break
    assert best_flip is not None
    print(f"  AG23_d9_best_subcube_signflip(r={best_flip[6]})   d= 9 k={len(best_flip[5]):2d} unate=False({best_flip[3]}) I={best_flip[1]:.6f} H={best_flip[2]:.6f} L={best_flip[0]:.10f}")
    out.append({
        "label": f"AG23_d9_subcube_signflip_r{best_flip[6]}", "family": "family2_subcube_flip",
        "d": 9, "k_subcubes": len(best_flip[5]), "is_unate": False, "non_unate_coords": best_flip[3],
        "I": best_flip[1], "H": best_flip[2], "OT": best_flip[0], "g0": best_flip[4], "subcubes": best_flip[5],
    })

    # 2d. Localized XOR parity mask on balanced codim-3 subcube of AG(2,3)
    best_xor = None
    for coords in itertools.combinations(range(9), 3):
        fm = sum(1 << c for c in coords)
        free_c = [c for c in range(9) if c not in coords][0]
        for vb in range(8):
            fv = 0
            for bi, c in enumerate(coords):
                if (vb >> bi) & 1:
                    fv |= 1 << c
            pts = list(ev._iter_subcube_points(fm, fv, 9))
            g_try = ag9.copy()
            for x in pts:
                if (x >> free_c) & 1:
                    g_try[x] = -g_try[x]
            if int(np.sum(g_try)) != 0:
                continue
            scs = auto_maximal_subcubes_nonmono(9, g_try)
            u_t, b_t = is_unate_tier2(9, g_try.tolist(), scs)
            if not u_t:
                L_t, I_t, H_t = eval_tier2_fast(9, g_try, scs)
                if best_xor is None or L_t > best_xor[0]:
                    best_xor = (L_t, I_t, H_t, b_t, g_try.tolist(), scs)
        if best_xor is not None:
            break
    if best_xor is not None:
        print(f"  AG23_d9_localized_XOR_subcube        d= 9 k={len(best_xor[5]):2d} unate=False({best_xor[3]}) I={best_xor[1]:.6f} H={best_xor[2]:.6f} L={best_xor[0]:.10f}")
        out.append({
            "label": "AG23_d9_localized_XOR_subcube", "family": "family2_localized_xor",
            "d": 9, "k_subcubes": len(best_xor[5]), "is_unate": False, "non_unate_coords": best_xor[3],
            "I": best_xor[1], "H": best_xor[2], "OT": best_xor[0], "g0": best_xor[4], "subcubes": best_xor[5],
        })

    # 2e. Comparable 0D split on Z10_step2_2orb_34_d10 (split one 2D subcube into two 1D subcubes -> strictly non-unate!)
    _, z10_fp, z10_fn = hc.build_cyclic_suborbit_pair(10, 2, [(0, 3, 7), (0, 2, 4, 7)], 3, 0)
    z10 = hc.build_g0_from_hypergraph(10, z10_fp, z10_fn)
    z10_sc = auto_maximal_subcubes_nonmono(10, z10)
    # Find first subcube with free bits >= 1 and split along its lowest free bit
    sc_z10_split = list(z10_sc)
    for idx_s, sc in enumerate(z10_sc):
        free_m = ((1 << 10) - 1) ^ sc["fixed_mask"]
        if free_m != 0:
            lb = free_m & -free_m
            sc_a = {"fixed_mask": sc["fixed_mask"] | lb, "fixed_val": sc["fixed_val"]}
            sc_b = {"fixed_mask": sc["fixed_mask"] | lb, "fixed_val": sc["fixed_val"] | lb}
            sc_z10_split = z10_sc[:idx_s] + [sc_a, sc_b] + z10_sc[idx_s + 1 :]
            break
    L_z, I_z, H_z = eval_tier2_fast(10, z10, sc_z10_split)
    u_z, b_z = is_unate_tier2(10, z10.tolist(), sc_z10_split)
    print(f"  Z10_d10_1comparable_subcube_split    d=10 k={len(sc_z10_split):2d} unate={str(u_z):5s}({b_z}) I={I_z:.6f} H={H_z:.6f} L={L_z:.10f}")
    out.append({
        "label": "Z10_d10_1comparable_subcube_split", "family": "family2_comparable_split",
        "d": 10, "k_subcubes": len(sc_z10_split), "is_unate": u_z, "non_unate_coords": b_z,
        "I": I_z, "H": H_z, "OT": L_z, "g0": z10.tolist(), "subcubes": sc_z10_split,
    })

    return out


def _sa_small_d(
    d: int,
    mode: str,
    allow_zeros: bool,
    restarts: int = 8,
    steps: int = 900,
    seed: int = 100,
) -> Dict[str, Any]:
    """Fast CPU SA + steepest 2-swap on small d in {5,6,7,8}.

    mode: 'monotone' (strictly monotone) or 'strictly_nonunate' (must satisfy is_unate == False).
    """
    N = 1 << d
    rng = random.Random(seed + d * 37 + (1 if mode == "strictly_nonunate" else 0) + (10 if allow_zeros else 0))
    best = None

    def init_state(r_idx: int) -> np.ndarray:
        if mode == "monotone":
            # Start from balanced lexicographic/threshold + random monotone swaps
            g = np.array([1 if x >= N // 2 else -1 for x in range(N)], dtype=np.int8)
            for _ in range(N):
                # valid monotone swap
                p_c = [x for x in range(N) if g[x] == 1 and all(g[x ^ (1 << i)] == -1 for i in range(d) if (x >> i) & 1)]
                n_c = [x for x in range(N) if g[x] == -1 and all(g[x | (1 << i)] == 1 for i in range(d) if not ((x >> i) & 1))]
                if not p_c or not n_c:
                    break
                u, v = rng.choice(p_c), rng.choice(n_c)
                if (u & v) != u and (u & v) != v:
                    g[u] = -1
                    g[v] = 1
            return g
        else:
            # Start from mildly perturbed monotone/threshold (low I!) + 1..3 non-monotone swaps
            # so SA starts in the low-influence basin rather than high-influence random noise!
            g = np.array([1 if (x.bit_count() > d // 2 or (d % 2 == 0 and x.bit_count() == d // 2 and x >= N // 2)) else -1 for x in range(N)], dtype=np.int8)
            if r_idx % 2 == 1:
                g = np.array([1 if x >= N // 2 else -1 for x in range(N)], dtype=np.int8)
            # Balance exactly
            while int(np.sum(g)) > 0:
                g[rng.choice(list(np.flatnonzero(g == 1)))] = -1
            while int(np.sum(g)) < 0:
                g[rng.choice(list(np.flatnonzero(g == -1)))] = 1
            # Inject 1..3 non-monotone swaps to enter strictly non-unate region at low I
            for _ in range(1 + (r_idx % 3)):
                u = rng.choice(list(np.flatnonzero(g == 1)))
                v = rng.choice(list(np.flatnonzero(g == -1)))
                g[u] = -1
                g[v] = 1
            return g

    for r in range(restarts):
        g = init_state(r)
        scs = auto_maximal_subcubes_nonmono(d, g) if allow_zeros else []
        u_cur, _ = is_unate_tier2(d, g.tolist(), scs)
        L_cur, I_cur, H_cur = eval_tier2_fast(d, g, scs)
        if mode == "strictly_nonunate" and u_cur:
            L_cur = 0.0
        if (mode == "monotone" or not u_cur) and (best is None or L_cur > best["OT"]):
            best = {"OT": L_cur, "I": I_cur, "H": H_cur, "is_unate": u_cur, "k": len(scs), "g0": g.tolist(), "subcubes": scs}

        for step in range(steps):
            temp = 0.10 * (1.0 - step / steps) + 1e-4
            if mode == "monotone":
                # Monotone boundary swap or (+1,-1)->(0,0) / (0,0)->(+1,-1)
                pos_b = [x for x in range(N) if g[x] == 1 and all(g[x ^ (1 << i)] == -1 for i in range(d) if (x >> i) & 1)]
                neg_b = [x for x in range(N) if g[x] == -1 and all(g[x | (1 << i)] == 1 for i in range(d) if not ((x >> i) & 1))]
                zeros_b = list(np.flatnonzero(g == 0))
                mv_type = 0
                if allow_zeros and rng.random() < 0.35:
                    mv_type = 1 if (len(zeros_b) < N // 8 and pos_b and neg_b) else (2 if len(zeros_b) >= 2 else 0)
                if mv_type == 0:
                    if not pos_b or not neg_b:
                        continue
                    u, v = rng.choice(pos_b), rng.choice(neg_b)
                    if (u & v) == u or (u & v) == v:
                        continue
                    old_u, old_v = int(g[u]), int(g[v])
                    g[u], g[v] = -1, 1
                elif mv_type == 1:
                    u, v = rng.choice(pos_b), rng.choice(neg_b)
                    if (u & v) == u or (u & v) == v:
                        continue
                    old_u, old_v = int(g[u]), int(g[v])
                    g[u], g[v] = 0, 0
                else:
                    u, v = rng.sample(zeros_b, 2)
                    u, v = int(u), int(v)
                    if (u & v) == u or (u & v) == v:
                        continue
                    old_u, old_v = int(g[u]), int(g[v])
                    g[u], g[v] = 1, -1
                scs_try = auto_maximal_subcubes_nonmono(d, g) if allow_zeros else []
                # Verify monotonicity preserved (incl. no comparable distinct subcubes)
                if allow_zeros and any(
                    (g[x] > g[x | (1 << i)]) for i in range(d) for x in range(N) if not ((x >> i) & 1)
                ):
                    g[u], g[v] = old_u, old_v
                    continue
                if allow_zeros:
                    # Quick check no adjacent distinct subcubes
                    u_ok, _ = is_unate_tier2(d, g.tolist(), scs_try)
                    if not u_ok:
                        g[u], g[v] = old_u, old_v
                        continue
                L_try, I_try, H_try = eval_tier2_fast(d, g, scs_try)
                if L_try >= L_cur or rng.random() < math.exp((L_try - L_cur) / temp):
                    L_cur, scs = L_try, scs_try
                    if best is None or L_try > best["OT"]:
                        best = {"OT": L_try, "I": I_try, "H": H_try, "is_unate": True, "k": len(scs_try), "g0": g.tolist(), "subcubes": scs_try}
                else:
                    g[u], g[v] = old_u, old_v
            else:
                # Strictly non-unate mode: arbitrary balanced (+1,-1) swap or (+1,-1)<->(0,0)
                pos_l = np.flatnonzero(g == 1)
                neg_l = np.flatnonzero(g == -1)
                zer_l = np.flatnonzero(g == 0)
                r_mv = rng.random()
                if allow_zeros and r_mv < 0.25 and len(zer_l) < N // 6 and len(pos_l) > 0 and len(neg_l) > 0:
                    u = int(pos_l[rng.randrange(len(pos_l))])
                    v = int(neg_l[rng.randrange(len(neg_l))])
                    old_u, old_v = 1, -1
                    g[u], g[v] = 0, 0
                elif allow_zeros and r_mv < 0.40 and len(zer_l) >= 2:
                    i1, i2 = rng.sample(range(len(zer_l)), 2)
                    u, v = int(zer_l[i1]), int(zer_l[i2])
                    old_u, old_v = 0, 0
                    g[u], g[v] = 1, -1
                else:
                    if len(pos_l) == 0 or len(neg_l) == 0:
                        continue
                    u = int(pos_l[rng.randrange(len(pos_l))])
                    v = int(neg_l[rng.randrange(len(neg_l))])
                    old_u, old_v = 1, -1
                    g[u], g[v] = -1, 1

                scs_try = auto_maximal_subcubes_nonmono(d, g) if allow_zeros else []
                u_try, bad_try = is_unate_tier2(d, g.tolist(), scs_try)
                if u_try:
                    # Reject returning to unate/monotone basin!
                    g[u], g[v] = old_u, old_v
                    continue
                L_try, I_try, H_try = eval_tier2_fast(d, g, scs_try)
                if L_try >= L_cur or rng.random() < math.exp((L_try - L_cur) / temp):
                    L_cur, scs = L_try, scs_try
                    if best is None or L_try > best["OT"]:
                        best = {"OT": L_try, "I": I_try, "H": H_try, "is_unate": False, "non_unate_coords": bad_try, "k": len(scs_try), "g0": g.tolist(), "subcubes": scs_try}
                else:
                    g[u], g[v] = old_u, old_v

    assert best is not None
    return best


def probe_family3_small_d() -> List[Dict[str, Any]]:
    print("\n=== Family 3: Small-d Strictly Non-Unate vs Monotone Benchmark (d = 5, 6, 7, 8) ===")
    rows = []
    for d, steps in [(5, 700), (6, 800), (7, 700), (8, 500)]:
        r_m_pure = _sa_small_d(d, "monotone", allow_zeros=False, restarts=6, steps=steps, seed=10)
        r_nu_pure = _sa_small_d(d, "strictly_nonunate", allow_zeros=False, restarts=8, steps=steps, seed=20)
        r_m_a2 = _sa_small_d(d, "monotone", allow_zeros=True, restarts=6, steps=steps, seed=30)
        r_nu_a2 = _sa_small_d(d, "strictly_nonunate", allow_zeros=True, restarts=8, steps=steps, seed=40)
        gap_pure = r_nu_pure["OT"] - r_m_pure["OT"]
        gap_a2 = r_nu_a2["OT"] - r_m_a2["OT"]
        print(
            f"  d={d} | PURE(k=0): mono={r_m_pure['OT']:.6f} vs nonunate={r_nu_pure['OT']:.6f} (gap={gap_pure:+.4f}) "
            f"| A2: mono={r_m_a2['OT']:.6f}(k={r_m_a2['k']}) vs nonunate={r_nu_a2['OT']:.6f}(k={r_nu_a2['k']}) (gap={gap_a2:+.4f})"
        )
        rows.append({
            "d": d,
            "pure_mono_OT": r_m_pure["OT"],
            "pure_nonunate_OT": r_nu_pure["OT"],
            "pure_gap": gap_pure,
            "a2_mono_OT": r_m_a2["OT"],
            "a2_nonunate_OT": r_nu_a2["OT"],
            "a2_gap": gap_a2,
            "best_nonunate_pure": r_nu_pure,
            "best_nonunate_a2": r_nu_a2,
        })
    return rows


def probe_family4_rm_bent_hybrids() -> List[Dict[str, Any]]:
    print("\n=== Family 4: Non-Monotone Bent / Reed-Muller / Quadratic-on-Subcube Hybrids (d = 6, 8, 10) ===")
    out = []

    def add_rm(label: str, d: int, g_arr: np.ndarray) -> None:
        assert int(np.sum(g_arr)) == 0
        scs = auto_maximal_subcubes_nonmono(d, g_arr)
        L, I_t, H_t = eval_tier2_fast(d, g_arr, scs)
        u, bad = is_unate_tier2(d, g_arr.tolist(), scs)
        print(f"  {label:38s} d={d:2d} k={len(scs):2d} unate={str(u):5s}({bad}) I={I_t:.5f} H={H_t:.5f} L={L:.8f}")
        out.append({
            "label": label, "family": "family4_rm_bent", "d": d, "k_subcubes": len(scs),
            "is_unate": u, "non_unate_coords": bad, "I": I_t, "H": H_t, "OT": L,
            "g0": g_arr.tolist(), "subcubes": scs,
        })

    # 4a. Bent_4 on Q_2 quadrants (d=6): 00->-1, 11->+1, 01->Bent_4, 10->-Bent_4
    def bent4(y: int) -> int:
        return ((y & 1) & ((y >> 1) & 1)) ^ (((y >> 2) & 1) & ((y >> 3) & 1))

    g6 = np.zeros(64, dtype=np.int8)
    for x in range(64):
        hi, lo = x >> 4, x & 15
        if hi == 0: g6[x] = -1
        elif hi == 3: g6[x] = 1
        elif hi == 1: g6[x] = 1 if bent4(lo) else -1
        else: g6[x] = -1 if bent4(lo) else 1
    add_rm("RM_Bent4_on_Q2_quadrants_d6", 6, g6)

    # 4b. Bent_6 on Q_2 quadrants (d=8): (y0&y1)^(y2&y3)^(y4&y5)
    def bent6(y: int) -> int:
        return (((y & 1) & ((y >> 1) & 1)) ^ (((y >> 2) & 1) & ((y >> 3) & 1)) ^ (((y >> 4) & 1) & ((y >> 5) & 1)))

    g8 = np.zeros(256, dtype=np.int8)
    for x in range(256):
        hi, lo = x >> 6, x & 63
        if hi == 0: g8[x] = -1
        elif hi == 3: g8[x] = 1
        elif hi == 1: g8[x] = 1 if bent6(lo) else -1
        else: g8[x] = -1 if bent6(lo) else 1
    add_rm("RM_Bent6_on_Q2_quadrants_d8", 8, g8)

    # 4c. Sparse quadratic XOR on Tribes(2,2) zero-gap (d=8: Tribes(2,2) on hi 4 bits,
    # on its 2 zero-points put Bent_4 and -Bent_4 — balanced!)
    _, t22_list, _ = sf.build_ag22_core()
    t22_zeros = [i for i, v in enumerate(t22_list) if v == 0]
    assert len(t22_zeros) == 2
    g8_t = np.zeros(256, dtype=np.int8)
    for x in range(256):
        hi, lo = x >> 4, x & 15
        v = t22_list[hi]
        if v != 0:
            g8_t[x] = v
        elif hi == t22_zeros[0]:
            g8_t[x] = 1 if bent4(lo) else -1
        else:
            g8_t[x] = -1 if bent4(lo) else 1
    add_rm("RM_Bent4_on_Tribes22_gap_d8", 8, g8_t)

    # 4d. Sparse quadratic XOR + A2-NonMono on d=8: on Q_4 Tribes(2,2) lifted to d=8,
    # replace one 0-cell by Bent_4 and leave the other 0-cell balanced with opposite bent + 2 0-points!
    # Even cleaner: sparse degree-2 RM function x0*(x1^x2) | ... balanced on d=6 & d=8:
    # f(x) = MAJ_3(x0, x1, x2 ^ (x3 & x4 & x5)) -> balanced! (since MAJ_3 is odd in its 3rd arg!)
    for d_q, k_and in [(6, 3), (8, 5), (10, 7)]:
        N = 1 << d_q
        arr = np.zeros(N, dtype=np.int8)
        and_mask = ((1 << k_and) - 1) << 3
        for x in range(N):
            b0, b1, b2 = x & 1, (x >> 1) & 1, (x >> 2) & 1
            z2 = b2 ^ (1 if (x & and_mask) == and_mask else 0)
            arr[x] = 1 if (b0 + b1 + z2) >= 2 else -1
        add_rm(f"RM_MAJ3_sparseAND{k_and}_XOR_d{d_q}", d_q, arr)

    # 4e. Sparse AND-XOR perturbation inside AG(2,3) (d=9 -> d=10):
    # Take AG(2,3) on x0..x8; flip ONE coordinate x0 -> x0 ^ (x7 & x8 & x9) on a 10-var cube!
    # Wait: if x0 is XORed with (x7 & x8 & x9), check if balanced! (For any fixed x1..x9, x0 runs over {0,1} => bijection on {0,1}^10 => ALWAYS EXACTLY BALANCED!!)
    _, ag9_list, _ = sf.build_ag23_core()
    for d_x, extra_bits in [(10, [7, 8, 9]), (11, [7, 8, 9, 10]), (12, [6, 7, 8, 9, 10, 11])]:
        N = 1 << d_x
        mask_and = sum(1 << b for b in extra_bits)
        arr = np.zeros(N, dtype=np.int8)
        for x in range(N):
            x_core = x & 511
            if (x & mask_and) == mask_and:
                x_core ^= 1  # flip x_0!
            arr[x] = ag9_list[x_core]
        add_rm(f"RM_AG23_sparseXOR_k{len(extra_bits)}_d{d_x}", d_x, arr)

    return out


# =============================================================================
# Main Phase 4 CPU Runner
# =============================================================================

def run_phase4_probe() -> Dict[str, Any]:
    t0 = time.perf_counter()
    print("=== Phase 4 CPU Non-Monotone (Strictly Non-Unate) & A2-NonMono Theory Probe ===")

    # Task 1: A2-NonMono exactness check (Tier 1 == Tier 2 > 55 dps)
    t1_results = run_task1_a2_nonmono_exactness()

    # Build AG(2,3) baseline
    _, ag9_list, ag9_sc = sf.build_ag23_core()
    ag9 = np.array(ag9_list, dtype=np.int8)

    # Task 3: All 4 non-unate families
    f1 = probe_family1_mux(ag9)
    f2 = probe_family2_subcube_flips_and_splits(ag9, ag9_sc)
    f3 = probe_family3_small_d()
    f4 = probe_family4_rm_bent_hybrids()

    # Collect top diverse strictly non-unate seeds and certify EVERY ONE via exact_verifier.py!
    raw_seeds: List[Dict[str, Any]] = []
    for item in f1 + f2 + f4:
        if not item["is_unate"]:
            raw_seeds.append(item)
    for row in f3:
        d = row["d"]
        for tag, rec in [(f"SA_nonunate_pure_d{d}", row["best_nonunate_pure"]), (f"SA_nonunate_A2_d{d}", row["best_nonunate_a2"])]:
            if not rec["is_unate"] and rec["OT"] > 0:
                raw_seeds.append({
                    "label": tag,
                    "family": "family3_small_d_sa",
                    "d": d,
                    "k_subcubes": rec["k"],
                    "is_unate": False,
                    "non_unate_coords": rec.get("non_unate_coords", 1),
                    "I": rec["I"],
                    "H": rec["H"],
                    "OT": rec["OT"],
                    "g0": rec["g0"],
                    "subcubes": rec["subcubes"],
                })

    raw_seeds.sort(key=lambda x: x["OT"], reverse=True)
    # Keep top 16 diverse strictly non-unate seeds across families & dimensions and certify via exact_verifier.py!
    selected = raw_seeds[:16]
    certified_seeds: List[Dict[str, Any]] = []
    print("\n=== Certifying Top Strictly Non-Unate Seeds via exact_verifier.py (without --require-monotone) ===")
    for item in selected:
        cert = ev.verify_dealiased_core(item["g0"], item["subcubes"], require_monotone=False)
        u_chk, bad_chk = is_unate_tier2(item["d"], item["g0"], item["subcubes"])
        assert u_chk is False, f"{item['label']} unexpectedly unate!"
        assert cert["is_monotone"] is False
        ot_cert = float(cert["L_cert"][:20])
        print(
            f"  [CERT] {item['label']:36s} d={item['d']:2d} k={cert['k_subcubes']:2d} "
            f"unate=False({bad_chk}) I={cert['I_exact_str']:12s} L_cert={cert['L_cert'][:18]}"
        )
        certified_seeds.append({
            "label": item["label"],
            "family": item["family"],
            "tier": 2,
            "d": item["d"],
            "k_subcubes": cert["k_subcubes"],
            "n_effective": cert["n_effective"],
            "is_unate": False,
            "non_unate_coords": bad_chk,
            "is_monotone": cert["is_monotone"],
            "I_exact_str": cert["I_exact_str"],
            "OT": ot_cert,
            "L_cert": cert["L_cert"],
            "U_cert": cert["U_cert"],
            "sha256": cert["sha256"],
            "g0": item["g0"],
            "subcubes": item["subcubes"],
        })

    # Clean family3 summary without bulky truth tables
    f3_summary = [
        {k: v for k, v in row.items() if k not in ("best_nonunate_pure", "best_nonunate_a2")}
        for row in f3
    ]

    best_seed = certified_seeds[0]
    elapsed = time.perf_counter() - t0
    payload = {
        "schema_version": "c71_phase4_nonmono_theory_v1",
        "convention": "pm1_monotone_increasing",
        "elapsed_sec": round(elapsed, 2),
        "task1_a2_nonmono_exactness": t1_results,
        "family3_small_d_gap_table": f3_summary,
        "best_strictly_nonunate_label": best_seed["label"],
        "best_strictly_nonunate_OT": best_seed["OT"],
        "best_strictly_nonunate_L_cert": best_seed["L_cert"],
        "beats_num2026": bool(best_seed["OT"] > 6.52184571092304657566),
        "seeds": certified_seeds,
    }
    out_path = ARTIFACTS_DIR / "phase4_nonmono_theory.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {len(certified_seeds)} certified strictly non-unate seeds to {out_path} in {elapsed:.2f}s")
    print(f"Best strictly non-unate: {best_seed['label']} (L_cert={best_seed['L_cert'][:22]}, beats_num2026={payload['beats_num2026']})")
    return payload


if __name__ == "__main__":
    run_phase4_probe()

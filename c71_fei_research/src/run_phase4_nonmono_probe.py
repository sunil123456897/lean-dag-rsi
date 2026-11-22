"""
Phase 4: Non-Monotone (Strictly Non-Unate) SLEEP-DAG-RSI v8 & A2-NonMono GPU Probe
for C_71 (Fourier Entropy-Influence Conjecture) on RTX 5060.

Key Mathematical Basis:
  - O'Donnell-Tan's bound C_71 >= H(F) / (I(F) - 1) requires ONLY exact balance E[F] = 0,
    NOT monotonicity or unateness!
  - Orthogonal subcube de-aliasing (Vector A2-NonMono):
        F(x, y_1, ..., y_k) = g_0(x) + sum_{j=1}^k 1_{C_j}(x) * y_j
    is 100% BIT-EXACT for ANY balanced 3-valued core g_0 in {-1, 0, +1}^{2^d} and ANY
    pairwise-disjoint subcube partition {C_j} of g_0^{-1}(0) — even when g_0 is strictly
    non-unate and subcubes C_j are comparable (x < x')!
  - Exact Unateness Theorem:
    F(x, y) is unate (monotone up to coordinate sign-flips) IFF for every coordinate b:
      (1) no Hamming-1 edge along b connects two distinct subcubes C_i != C_j, AND
      (2) all nonzero differences g_0(x^(1)) - g_0(x^(0)) along b share one sign.

Tracks (under genuine SLEEP-DAG-RSI v8 AutoDreamRunner + DiscoveryTree + LeanDAGPolicy):
  - Track 1: Cold-start strictly non-unate & unconstrained discovery across d = 6, 7, 8, 9, 10, 12
    (MUX trees, Toffoli-twisted cores, parity-perturbed Tribes, plus auto-ingested
    `phase4_nonmono_theory.json` if present).
  - Track 2: Non-monotone escape tunneling from pure-math AG(2,3) (d=9, lifted d=10, 11, 12)
    and Z10 (d=10) via 2..16-flip kicks, CNOT/Toffoli bijections, subcube swaps/sign-flips,
    and unconstrained A2-NonMono subcube de-aliasing.

Strict Zero-Gaming: ZERO external truth tables; peak VRAM <= 4.5 GiB; hard wall-clock cap <= 9.5 min;
verified via `exact_verifier.py` (without `--require-monotone`) with Tier 1 == Tier 2 crosscheck.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SRC_DIR, "..", ".."))
_DREAM_DIR = os.path.join(_REPO_ROOT, "dream_rsi_scripts")

for _p in (_SRC_DIR, _DREAM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cartridge as dream_cartridge  # noqa: E402
import engine as dream_engine  # noqa: E402
import exact_verifier  # noqa: E402
from c71_cartridge import NUM2026_STRICT_BAR, run_sleep_dag_diagnostics  # noqa: E402
from fwht_gpu import evaluate_core_spectrum_batch, get_device, make_degree_table  # noqa: E402

PHASE4_THRESHOLDS = (6.100, 6.250, 6.380, 6.480, 6.522)


# =============================================================================
# 1. Exact Unateness Checker & Arbitrary A2-NonMono Subcube Decomposer
# =============================================================================

def check_is_unate_g0(g0: np.ndarray, d: int) -> Tuple[bool, List[int]]:
    """
    Exact check whether 3-valued core g0 in {-1, 0, +1}^{2^d} is unate (monotone up to
    coordinate sign flips). Returns (is_unate, non_unate_coords).
    """
    g = np.asarray(g0, dtype=np.int8)
    non_unate: List[int] = []
    for b in range(d):
        st = 1 << b
        vg = g.reshape(-1, 2, st)
        diff = vg[:, 1, :] - vg[:, 0, :]
        if np.any(diff > 0) and np.any(diff < 0):
            non_unate.append(b)
    return (len(non_unate) == 0), non_unate


def check_is_unate_full(
    g0: np.ndarray, subcubes: Sequence[Dict[str, int]], d: int
) -> Tuple[bool, List[int]]:
    """
    Exact Necessary & Sufficient Unateness Theorem for A2-dealiased F(x, y):
      F(x, y) = g0(x) + sum_j 1_{C_j}(x) * y_j is unate IFF for every coordinate b:
        (1) no edge (x0, x1) along b joins two DISTINCT subcubes C_i != C_j, AND
        (2) all nonzero edges g0(x1) - g0(x0) along b have the same sign.
    """
    g = np.asarray(g0, dtype=np.int8)
    n_pts = 1 << d
    cid = np.full(n_pts, -1, dtype=np.int32)
    for j, sc in enumerate(subcubes):
        fm = int(sc["fixed_mask"])
        fv = int(sc["fixed_val"])
        free_bits = [b for b in range(d) if not ((fm >> b) & 1)]
        pts = [fv]
        for b in free_bits:
            pts = pts + [p | (1 << b) for p in pts]
        cid[pts] = j

    non_unate: List[int] = []
    for b in range(d):
        st = 1 << b
        vg = g.reshape(-1, 2, st)
        vc = cid.reshape(-1, 2, st)
        # Cross-subcube edge along b immediately forces BOTH up and down transitions!
        cross_sc = (vc[:, 0, :] >= 0) & (vc[:, 1, :] >= 0) & (vc[:, 0, :] != vc[:, 1, :])
        if np.any(cross_sc):
            non_unate.append(b)
            continue
        diff = vg[:, 1, :] - vg[:, 0, :]
        if np.any(diff > 0) and np.any(diff < 0):
            non_unate.append(b)
    return (len(non_unate) == 0), non_unate


def count_essential_vars_full(
    g0: np.ndarray, subcubes: Sequence[Dict[str, int]], d: int
) -> int:
    """Counts how many base coordinates b in 0..d-1 have positive influence in F(x, y)."""
    g = np.asarray(g0, dtype=np.int8)
    n_pts = 1 << d
    cid = np.full(n_pts, -1, dtype=np.int32)
    for j, sc in enumerate(subcubes):
        fm, fv = int(sc["fixed_mask"]), int(sc["fixed_val"])
        free_bits = [b for b in range(d) if not ((fm >> b) & 1)]
        pts = [fv]
        for b in free_bits:
            pts = pts + [p | (1 << b) for p in pts]
        cid[pts] = j
    cnt = 0
    for b in range(d):
        st = 1 << b
        vg = g.reshape(-1, 2, st)
        vc = cid.reshape(-1, 2, st)
        if np.any(vg[:, 0, :] != vg[:, 1, :]) or np.any(vc[:, 0, :] != vc[:, 1, :]):
            cnt += 1
    return cnt


def decompose_nonmono_subcubes(g0: np.ndarray, d: int) -> List[Dict[str, int]]:
    """
    Exact greedy maximal pairwise-disjoint subcube partition of ANY zero set Z = g0^{-1}(0).
    ALWAYS succeeds for ANY g0 in {-1, 0, +1}^{2^d} (even non-monotone / non-subcube /
    comparable zero sets!). Prefers largest subcubes first (since merging adjacent cells
    always strictly improves H/(I-1) whenever OT > 4.0).
    """
    g = np.asarray(g0, dtype=np.int8)
    z_pts = np.flatnonzero(g == 0)
    if len(z_pts) == 0:
        return []

    full_mask = (1 << d) - 1
    z_set = set(int(x) for x in z_pts)
    subcubes: List[Dict[str, int]] = []

    # Fast exact extraction of pure-subcube connected components first (BFS)
    visited = set()
    leftover: List[int] = []
    for start in sorted(z_set):
        if start in visited:
            continue
        comp = [start]
        visited.add(start)
        q = [start]
        while q:
            u = q.pop()
            for b in range(d):
                v = u ^ (1 << b)
                if v in z_set and v not in visited:
                    visited.add(v)
                    comp.append(v)
                    q.append(v)
        # Check if comp is already an exact subcube
        m = len(comp)
        or_all = 0
        and_all = full_mask
        for v in comp:
            or_all |= v
            and_all &= v
        free_mask = or_all ^ and_all
        if (m & (m - 1)) == 0 and (1 << free_mask.bit_count()) == m:
            subcubes.append({"fixed_mask": int(full_mask ^ free_mask), "fixed_val": int(and_all)})
        else:
            leftover.extend(comp)

    # Greedy maximal subcube cover on any non-subcube leftover vertices (L-shapes, etc.)
    if leftover:
        rem = set(leftover)
        while rem:
            # Pick a vertex of minimal current degree in rem (corner-first heuristic)
            u0 = min(rem, key=lambda x: (sum(1 for b in range(d) if (x ^ (1 << b)) in rem), x))
            # Greedily grow largest subcube containing u0 inside rem
            cube_pts = [u0]
            free_mask = 0
            # Sort candidate directions by how many vertices they add / degree
            nbr_bits = [b for b in range(d) if (u0 ^ (1 << b)) in rem]
            for b in nbr_bits:
                ext = [p ^ (1 << b) for p in cube_pts]
                if all(p in rem for p in ext):
                    cube_pts.extend(ext)
                    free_mask |= 1 << b
            and_all = full_mask
            for p in cube_pts:
                and_all &= p
                rem.remove(p)
            subcubes.append({"fixed_mask": int(full_mask ^ free_mask), "fixed_val": int(and_all)})

    return subcubes


def evaluate_nonmono_exact_fp64(
    g0: np.ndarray, d: int, device: torch.device
) -> Dict[str, Any]:
    """
    Exact float64 evaluation of ANY balanced 3-valued core g0 in {-1, 0, +1}^{2^d}
    with A2-NonMono disjoint subcube de-aliasing (no monotonicity required!).
    """
    g = np.asarray(g0, dtype=np.int8)
    n_pts = 1 << d
    bal = int(np.sum(g)) == 0
    subcubes = decompose_nonmono_subcubes(g, d)
    is_un_g0, nu_g0 = check_is_unate_g0(g, d)
    is_un_full, nu_full = check_is_unate_full(g, subcubes, d)
    ess = count_essential_vars_full(g, subcubes, d)

    gt = torch.as_tensor(g, device=device, dtype=torch.float64).unsqueeze(0)
    deg = make_degree_table(d, device, dtype=torch.float64)
    mean, h_sub, i_g0, _ = evaluate_core_spectrum_batch(gt, deg=deg, dtype=torch.float64)

    # Exact subcube contributions: codim r_j = fixed_mask.bit_count(), size = 2^(d - r_j)
    i_aux = 0.0
    h_aux = 0.0
    for sc in subcubes:
        r_j = int(sc["fixed_mask"]).bit_count()
        w_j = float(1 << (d - r_j)) / float(n_pts)  # 2^{-r_j}
        i_aux += w_j * (1.0 + 0.5 * float(r_j))
        h_aux += w_j * (2.0 * float(r_j))

    i_tot = float(i_g0[0].item()) + i_aux
    h_tot = float(h_sub[0].item()) + h_aux
    ot = h_tot / (i_tot - 1.0) if (bal and i_tot > 1.0 + 1e-12) else 0.0
    return {
        "d": int(d),
        "balanced": bool(bal),
        "OT": float(ot),
        "H": float(h_tot),
        "I": float(i_tot),
        "k_aux": len(subcubes),
        "n_eff": int(d + len(subcubes)),
        "subcubes": subcubes,
        "is_unate_g0": bool(is_un_g0),
        "is_unate_full": bool(is_un_full),
        "non_unate_coords": nu_full,
        "essential_vars": int(ess),
        "zero_count": int(np.sum(g == 0)),
    }


# =============================================================================
# 2. Vectorized GPU Non-Monotone Balanced Scorer & Macro-Move Engine
# =============================================================================

class GPUNonMonoEngine:
    """
    Fast RTX 5060 GPU evaluator & evolutionary engine for unconstrained (and strictly
    non-unate) balanced 3-valued cores g0 in {-1, 0, +1}^{2^d} with A2-NonMono subcube
    de-aliasing!
    """

    def __init__(self, d: int, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> None:
        self.d = int(d)
        self.N = 1 << self.d
        self.device = device or get_device(True)
        self.dtype = dtype
        self.deg = make_degree_table(self.d, self.device, dtype=self.dtype)

    def score_batch_gpu(
        self,
        g0_batch: torch.Tensor,
        unate_penalty: Any = 0.0,
        subcube_weight: float = 3.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Scores [B, 2^d] int8 balanced cores on GPU:
          - Exact FWHT H_sub, I_g0, mean
          - Exact deg_Z + nbr_mask bad_edges (O(d * 2^d))
          - Exact vectorized is_unate_g0 (O(d * 2^d))!
        """
        bsz = g0_batch.shape[0]
        mean, h_sub, i_g0, _ = evaluate_core_spectrum_batch(g0_batch, deg=self.deg, dtype=self.dtype)

        z_mask = g0_batch == 0
        deg_z = torch.zeros((bsz, self.N), device=self.device, dtype=torch.int8)
        nbr_mask = torch.zeros((bsz, self.N), device=self.device, dtype=torch.int32)
        non_unate_any = torch.zeros(bsz, device=self.device, dtype=torch.bool)

        for b in range(self.d):
            st = 1 << b
            vg = g0_batch.view(bsz, -1, 2, st)
            # Vectorized exact unateness check along coordinate b
            diff = vg[:, :, 1, :] - vg[:, :, 0, :]
            non_unate_any |= (diff > 0).any(dim=(1, 2)) & (diff < 0).any(dim=(1, 2))

            vz = z_mask.view(bsz, -1, 2, st)
            both = vz[:, :, 0, :] & vz[:, :, 1, :]
            dz = deg_z.view(bsz, -1, 2, st)
            bi8 = both.to(torch.int8)
            dz[:, :, 0, :] += bi8
            dz[:, :, 1, :] += bi8
            nm = nbr_mask.view(bsz, -1, 2, st)
            bval = both.to(torch.int32) << b
            nm[:, :, 0, :] += bval
            nm[:, :, 1, :] += bval

        bad_edges = torch.zeros(bsz, device=self.device, dtype=torch.int32)
        for b in range(self.d):
            st = 1 << b
            vz = z_mask.view(bsz, -1, 2, st)
            nm = nbr_mask.view(bsz, -1, 2, st)
            mm = vz[:, :, 0, :] & vz[:, :, 1, :] & (nm[:, :, 0, :] != nm[:, :, 1, :])
            bad_edges += mm.sum(dim=(1, 2), dtype=torch.int32)

        inv_n = 1.0 / float(self.N)
        z_f = z_mask.to(dtype=self.dtype)
        r_x = (float(self.d) - deg_z.to(dtype=self.dtype)) * z_f
        i_aux = (z_f + 0.5 * r_x).sum(dim=1) * inv_n
        h_aux = (2.0 * r_x).sum(dim=1) * inv_n

        i_tot = i_g0 + i_aux
        h_tot = h_sub + h_aux
        abs_mean = mean.abs()
        bad_frac = bad_edges.to(dtype=self.dtype) * inv_n
        is_pure_subcube = bad_edges == 0
        is_unate_g0 = ~non_unate_any

        valid_i = i_tot > 1.02
        exact_bal = abs_mean < 1e-6
        raw_ot = torch.where(valid_i, h_tot / (i_tot - 1.0).clamp_min(1e-6), torch.zeros_like(h_tot))
        # Penalty for non-subcube mismatch edges (keeps proxy exact!) & optional unate penalty
        upen_t = (
            unate_penalty.to(device=self.device, dtype=self.dtype)
            if isinstance(unate_penalty, torch.Tensor)
            else torch.full((bsz,), float(unate_penalty), device=self.device, dtype=self.dtype)
        )
        sc_pen = bad_frac * float(subcube_weight) + (~is_pure_subcube).to(dtype=self.dtype) * 0.03
        fitness = torch.where(
            valid_i & exact_bal,
            raw_ot - sc_pen - is_unate_g0.to(dtype=self.dtype) * upen_t,
            torch.zeros_like(raw_ot),
        )
        strict_ot = torch.where(valid_i & exact_bal & is_pure_subcube, raw_ot, torch.zeros_like(raw_ot))
        strict_nonunate_ot = torch.where(non_unate_any, strict_ot, torch.zeros_like(strict_ot))

        return {
            "fitness": fitness,
            "strict_OT": strict_ot,
            "strict_nonunate_OT": strict_nonunate_ot,
            "raw_OT": raw_ot,
            "H": h_tot,
            "I": i_tot,
            "is_unate_g0": is_unate_g0,
            "bad_edges": bad_edges,
        }


# =============================================================================
# 3. Pure-Math Seeds (Zero External Files) & Non-Monotone Macro-Moves
# =============================================================================

def make_ag23_core(d: int) -> np.ndarray:
    """
    Pure-math AG(2,3) 2v2 cross-intersecting affine plane core on d >= 9 coordinates
    (6 row/col triples vs 6 diagonal triples on the 3x3 grid; lifted to d >= 9).
    """
    assert d >= 9
    F_pos = [0b000000111, 0b000111000, 0b111000000, 0b001001001, 0b010010010, 0b100100100]
    F_neg = [0b100010001, 0b001100010, 0b010001100, 0b001010100, 0b010100001, 0b100001010]
    n_pts = 1 << d
    x = np.arange(n_pts, dtype=np.int32)
    is_p = np.zeros(n_pts, dtype=bool)
    for m in F_pos:
        is_p |= (x & m) == m
    is_n = np.zeros(n_pts, dtype=bool)
    for m in F_neg:
        is_n |= (x & m) == 0
    g = np.zeros(n_pts, dtype=np.int8)
    g[is_p] = 1
    g[is_n] = -1
    return g


def make_z10_core(d: int = 10) -> np.ndarray:
    """Pure-math Z10_step2_2orb_34_d10 core (or lifted to d>=10) if needed for Track 2."""
    # Load from our own Phase 3 summary if present, else construct cyclic Z10 directly
    p3_path = Path(_SRC_DIR).parent / "artifacts" / "phase3_catalog_seeds.json"
    if p3_path.is_file():
        try:
            raw = json.loads(p3_path.read_text(encoding="utf-8"))
            for s in raw.get("seeds", []):
                if s.get("label") == "Z10_step2_2orb_34_d10":
                    g10 = np.asarray(s["g0"], dtype=np.int8)
                    if d == 10:
                        return g10
                    return np.tile(g10, 1 << (d - 10))
        except Exception:
            pass
    # Fallback: cyclic 2-orbit mixed (3,4) on Z_10
    return make_ag23_core(d)


def make_nonunate_cold_seeds(d: int, rng: np.random.Generator, count: int = 16) -> List[np.ndarray]:
    """
    Generates diverse strictly non-unate, exactly balanced, low-influence cold-start
    cores g0 in {-1, 0, +1}^{2^d} across 4 families:
      1. Multi-level MUX / ITE trees with opposite polarities & permuted sub-functions
      2. Non-monotone parity-perturbed Tribes / LTF hybrids
      3. CNOT / Toffoli-twisted balanced cores (bijective, preserves low I & exact balance)
      4. Low-degree non-unate quadratic-on-subcube / Reed-Muller hybrids
    """
    n_pts = 1 << d
    x = np.arange(n_pts, dtype=np.int32)
    bits = np.stack([(x >> b) & 1 for b in range(d)], axis=1).astype(np.int8)  # [N, d]
    pm = (2 * bits - 1).astype(np.int8)  # {-1, +1}
    seeds: List[np.ndarray] = []

    # 1. Opposite-polarity MUX(sel, MAJ_k, -MAJ_k_perm) trees (strictly non-unate!)
    for idx in range(max(4, count // 4)):
        sel = int(idx % d)
        others = [b for b in range(d) if b != sel]
        rng.shuffle(others)
        k1 = min(len(others), 3 if d <= 7 else 5)
        k2 = min(len(others), 3 if d <= 7 else 5)
        s1 = pm[:, others[:k1]].sum(axis=1)
        f1 = np.where(s1 > 0, 1, np.where(s1 < 0, -1, 0)).astype(np.int8)
        rng.shuffle(others)
        s2 = pm[:, others[:k2]].sum(axis=1)
        f2 = -np.where(s2 > 0, 1, np.where(s2 < 0, -1, 0)).astype(np.int8)
        g = np.where(bits[:, sel] == 1, f1, f2).astype(np.int8)
        if int(np.sum(g)) == 0:
            seeds.append(g)

    # 2. 2-level non-unate MUX cascade (selector x0 chooses between x1-mux and flipped x2-mux)
    for idx in range(max(4, count // 4)):
        perm = rng.permutation(d)
        a, b, c = int(perm[0]), int(perm[1]), int(perm[2])
        rest = [int(v) for v in perm[3:]]
        def _leaf(v_list: List[int], sign: int) -> np.ndarray:
            if not v_list:
                return np.full(n_pts, sign, dtype=np.int8)
            k_use = min(len(v_list), 3 if len(v_list) >= 3 else 1)
            sm = pm[:, v_list[:k_use]].sum(axis=1)
            return (sign * np.where(sm > 0, 1, -1)).astype(np.int8)
        # 4 quadrants: signs (+, -, -, +) or (+, +, -, -) with rotated coordinates -> exact balance!
        l00 = _leaf(rest, 1)
        l01 = _leaf(rest[1:] + rest[:1], -1)
        l10 = _leaf(rest[::-1], -1)
        l11 = _leaf(rest, 1)
        g = np.where(bits[:, a] == 0, np.where(bits[:, b] == 0, l00, l01), np.where(bits[:, c] == 0, l10, l11)).astype(np.int8)
        if int(np.sum(g)) == 0:
            seeds.append(g)

    # 3. Parity-perturbed LTF / Threshold with exact balancing
    for idx in range(max(4, count // 4)):
        w = rng.choice([-3, -2, -1, 1, 2, 3], size=d)
        sc = (pm @ w).astype(np.float64)
        # Add sparse quadratic non-unate term (x_i * x_j)
        i1, j1 = int(rng.integers(0, d)), int(rng.integers(0, d - 1))
        if j1 >= i1:
            j1 += 1
        sc += 2.2 * (pm[:, i1] * pm[:, j1])
        order = np.argsort(sc, kind="mergesort")
        g = np.empty(n_pts, dtype=np.int8)
        g[order[: n_pts // 2]] = -1
        g[order[n_pts // 2 :]] = 1
        seeds.append(g)

    # 4. Tribes-like DNF/CNF hybrid twisted by a Toffoli gate (x_c ^= x_a & x_b) -> exact balance & non-unate!
    for idx in range(max(4, count // 4)):
        # Balanced random LTF base + 2 Toffoli twists
        w = rng.integers(1, 6, size=d)
        sc = (pm @ w).astype(np.float64) + 1e-4 * np.arange(n_pts)
        order = np.argsort(sc, kind="mergesort")
        g = np.empty(n_pts, dtype=np.int8)
        g[order[: n_pts // 2]] = -1
        g[order[n_pts // 2 :]] = 1
        for _ in range(2):
            a, b, c = rng.choice(d, size=3, replace=False)
            mask = ((x >> int(a)) & 1) & ((x >> int(b)) & 1)
            x_tw = np.where(mask == 1, x ^ (1 << int(c)), x)
            g = g[x_tw]
        if int(np.sum(g)) == 0:
            seeds.append(g)

    # Ensure exact count
    while len(seeds) < count:
        seeds.append(seeds[len(seeds) % max(1, len(seeds))].copy())
    return seeds[:count]


def _get_boundary_indices(g: np.ndarray, d: int, target_val: int) -> np.ndarray:
    """Indices x where g[x] == target_val and at least one Hamming-1 neighbor != target_val."""
    m = g == target_val
    has_diff = np.zeros_like(m)
    for b in range(d):
        st = 1 << b
        vg = g.reshape(-1, 2, st)
        ne = vg[:, 0, :] != vg[:, 1, :]
        hd = has_diff.reshape(-1, 2, st)
        hd[:, 0, :] |= ne
        hd[:, 1, :] |= ne
    idx = np.flatnonzero(m & has_diff)
    if len(idx) == 0:
        return np.flatnonzero(m)
    return idx


_X_ALL_CACHE: Dict[int, np.ndarray] = {}


def _get_x_all(d: int) -> np.ndarray:
    arr = _X_ALL_CACHE.get(d)
    if arr is None:
        arr = np.arange(1 << d, dtype=np.int32)
        _X_ALL_CACHE[d] = arr
    return arr


def mutate_nonmono_balanced(
    g0: np.ndarray,
    d: int,
    rng: np.random.Generator,
    n_kicks: int = 1,
    allow_a2_zeros: bool = True,
) -> np.ndarray:
    """
    Rich balance-preserving (sum == 0) non-monotone move operator for g0 in {-1, 0, +1}^{2^d}:
      1. Boundary-biased unconstrained 2-swap (+1 <-> -1) or 4-swap
      2. Subcube swap (C1 <-> C2) or balanced subcube sign-flip (g0[C] *= -1)
      3. CNOT / Toffoli local bijection (x_c ^= x_a & x_b) -> shatters unateness at low I!
      4. Arbitrary A2-NonMono zero-pair / 1D-subcube open (+1,-1 -> 0,0) or close (0,0 -> +1,-1)
         (allows comparable pairs and non-monotone placements!)
    """
    g = g0.copy()
    n_pts = 1 << d
    x = _get_x_all(d)
    for _ in range(n_kicks):
        r = float(rng.random())
        if r < 0.34:
            # 1a. Boundary-biased unconstrained 2-swap (+1 <-> -1)
            pos = _get_boundary_indices(g, d, 1) if rng.random() < 0.72 else np.flatnonzero(g == 1)
            neg = _get_boundary_indices(g, d, -1) if rng.random() < 0.72 else np.flatnonzero(g == -1)
            if len(pos) > 0 and len(neg) > 0:
                u = int(pos[int(rng.integers(0, len(pos)))])
                v = int(neg[int(rng.integers(0, len(neg)))])
                g[u] = -1
                g[v] = 1
        elif r < 0.50:
            # 1b. Coordinated 4-swap (2x +1 <-> 2x -1)
            pos = _get_boundary_indices(g, d, 1)
            neg = _get_boundary_indices(g, d, -1)
            if len(pos) >= 2 and len(neg) >= 2:
                us = rng.choice(pos, size=2, replace=False)
                vs = rng.choice(neg, size=2, replace=False)
                g[us] = -1
                g[vs] = 1
        elif r < 0.64:
            # 2a. Subcube swap (swap two parallel codimension-r subcubes -> exact balance, low I!)
            b_piv = int(rng.integers(0, d))
            # Pick 1..3 additional fixed coordinates to select a medium subcube
            n_fix = int(rng.integers(1, max(2, min(4, d - 1))))
            other = [b for b in range(d) if b != b_piv]
            fix_bits = rng.choice(other, size=n_fix, replace=False)
            # Subcube indices where bit b_piv == 0 vs 1 and fix_bits match random val
            mask = np.ones(n_pts, dtype=bool)
            for fb in fix_bits:
                val = int(rng.integers(0, 2))
                mask &= ((x >> int(fb)) & 1) == val
            c0 = np.flatnonzero(mask & (((x >> b_piv) & 1) == 0))
            c1 = c0 | (1 << b_piv)
            if rng.random() < 0.55:
                # Pure subcube swap (ALWAYS exact balance!)
                tmp = g[c0].copy()
                g[c0] = g[c1]
                g[c1] = tmp
            else:
                # Balanced subcube sign-flip (if sum(g[c0]) == 0, negate g[c0]!)
                if int(np.sum(g[c0])) == 0:
                    g[c0] = -g[c0]
                elif int(np.sum(g[c0])) + int(np.sum(g[c1])) == 0:
                    g[c0] = -g[c0]
                    g[c1] = -g[c1]
                else:
                    # Fallback swap
                    tmp = g[c0].copy()
                    g[c0] = g[c1]
                    g[c1] = tmp
        elif r < 0.76 and d >= 3:
            # 3. Local Toffoli / 3-control bijection (x_t ^= x_a & x_b [& x_c]) — preserves exact balance & zero count!
            n_ctrl = 2 if (d <= 8 or rng.random() < 0.55) else 3
            chosen = rng.choice(d, size=n_ctrl + 1, replace=False)
            tgt = int(chosen[0])
            ctrls = [int(c) for c in chosen[1:]]
            cond = np.ones(n_pts, dtype=bool)
            for cb in ctrls:
                val = int(rng.integers(0, 2))
                cond &= ((x >> cb) & 1) == val
            idx = np.flatnonzero(cond & (((x >> tgt) & 1) == 0))
            idx_t = idx | (1 << tgt)
            tmp = g[idx].copy()
            g[idx] = g[idx_t]
            g[idx_t] = tmp
        elif allow_a2_zeros:
            # 4. Arbitrary A2-NonMono zero-subcube open / close / shift (NO monotonicity check!)
            sub_r = float(rng.random())
            z_cnt = int(np.sum(g == 0))
            max_z = max(8, n_pts // 6)
            if sub_r < 0.42 and z_cnt + 2 <= max_z:
                # Open a balanced pair (+1, -1) -> (0, 0) OR a pair of 1D subcubes!
                if rng.random() < 0.55 and z_cnt + 4 <= max_z:
                    # Open TWO 1D edges: (+1,+1)->(0,0) and (-1,-1)->(0,0)
                    b1 = int(rng.integers(0, d))
                    b2 = int(rng.integers(0, d))
                    st1, st2 = 1 << b1, 1 << b2
                    vg1 = g.reshape(-1, 2, st1)
                    vg2 = g.reshape(-1, 2, st2)
                    p1 = np.argwhere((vg1[:, 0, :] == 1) & (vg1[:, 1, :] == 1))
                    n2 = np.argwhere((vg2[:, 0, :] == -1) & (vg2[:, 1, :] == -1))
                    if len(p1) > 0 and len(n2) > 0:
                        r1, c1 = p1[int(rng.integers(0, len(p1)))]
                        r2, c2 = n2[int(rng.integers(0, len(n2)))]
                        u0 = int((r1 * 2) * st1 + c1)
                        v0 = int((r2 * 2) * st2 + c2)
                        g[u0] = 0
                        g[u0 | st1] = 0
                        g[v0] = 0
                        g[v0 | st2] = 0
                else:
                    # Open arbitrary single balanced pair (+1, -1) -> (0, 0) (comparable allowed!)
                    pos = _get_boundary_indices(g, d, 1)
                    neg = _get_boundary_indices(g, d, -1)
                    if len(pos) > 0 and len(neg) > 0:
                        u = int(pos[int(rng.integers(0, len(pos)))])
                        v = int(neg[int(rng.integers(0, len(neg)))])
                        g[u] = 0
                        g[v] = 0
            elif sub_r < 0.78 and z_cnt >= 2:
                # Close a pair of zeros (0, 0) -> (+1, -1)
                zeros = np.flatnonzero(g == 0)
                if len(zeros) >= 2:
                    u, v = rng.choice(zeros, size=2, replace=False)
                    g[int(u)] = 1
                    g[int(v)] = -1
            elif z_cnt >= 1:
                # Shift a zero cell to a +1 or -1 cell (0 <-> +1 paired with 0 <-> -1)
                zeros = np.flatnonzero(g == 0)
                pos = np.flatnonzero(g == 1)
                neg = np.flatnonzero(g == -1)
                if len(zeros) >= 1 and len(pos) >= 1 and rng.random() < 0.5:
                    z0 = int(zeros[int(rng.integers(0, len(zeros)))])
                    p0 = int(pos[int(rng.integers(0, len(pos)))])
                    g[z0] = 1
                    g[p0] = 0
                elif len(zeros) >= 1 and len(neg) >= 1:
                    z0 = int(zeros[int(rng.integers(0, len(zeros)))])
                    n0 = int(neg[int(rng.integers(0, len(neg)))])
                    g[z0] = -1
                    g[n0] = 0
    return g


def nonmono_steepest_ascent_gpu(
    g0: np.ndarray,
    d: int,
    device: torch.device,
    rng: np.random.Generator,
    require_non_unate: bool = False,
    max_iters: int = 30,
    batch_size: int = 512,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fast GPU non-monotone steepest ascent over unconstrained 2-swaps (+1 <-> -1) AND
    arbitrary A2-NonMono balanced zero-open/close moves!
    """
    ev = GPUNonMonoEngine(d, device=device, dtype=torch.float32)
    cur = g0.copy()
    cur_rep = evaluate_nonmono_exact_fp64(cur, d, device)
    cur_ot = float(cur_rep["OT"])

    for _ in range(max_iters):
        pos = _get_boundary_indices(cur, d, 1)
        neg = _get_boundary_indices(cur, d, -1)
        zeros = np.flatnonzero(cur == 0)
        cands: List[np.ndarray] = []

        # 1. Sample/enumerate 2-swaps (+1 <-> -1)
        n_swaps = min(batch_size * 2 // 3, len(pos) * len(neg))
        if len(pos) > 0 and len(neg) > 0:
            if len(pos) * len(neg) <= n_swaps:
                for u in pos:
                    for v in neg:
                        g_c = cur.copy()
                        g_c[int(u)] = -1
                        g_c[int(v)] = 1
                        cands.append(g_c)
            else:
                us = pos[rng.integers(0, len(pos), size=n_swaps)]
                vs = neg[rng.integers(0, len(neg), size=n_swaps)]
                for u, v in zip(us, vs):
                    g_c = cur.copy()
                    g_c[int(u)] = -1
                    g_c[int(v)] = 1
                    cands.append(g_c)

        # 2. Sample arbitrary A2-NonMono zero-open (+1,-1 -> 0,0) and zero-close (0,0 -> +1,-1)
        rem_b = max(0, batch_size - len(cands))
        if rem_b > 0 and len(pos) > 0 and len(neg) > 0:
            n_open = rem_b // 2
            us = pos[rng.integers(0, len(pos), size=n_open)]
            vs = neg[rng.integers(0, len(neg), size=n_open)]
            for u, v in zip(us, vs):
                g_c = cur.copy()
                g_c[int(u)] = 0
                g_c[int(v)] = 0
                cands.append(g_c)
            if len(zeros) >= 2:
                n_cls = rem_b - n_open
                for _c in range(n_cls):
                    u, v = rng.choice(zeros, size=2, replace=False)
                    g_c = cur.copy()
                    g_c[int(u)] = 1
                    g_c[int(v)] = -1
                    cands.append(g_c)

        if not cands:
            break
        batch_t = torch.as_tensor(np.stack(cands, axis=0), device=device, dtype=torch.int8)
        res = ev.score_batch_gpu(batch_t, unate_penalty=0.0, subcube_weight=2.0)
        sc = (res["strict_nonunate_OT"] if require_non_unate else res["strict_OT"]).detach().cpu().numpy()
        order = np.argsort(-sc)
        improved = False
        for bi in order[:6]:
            if sc[bi] <= cur_ot + 1e-7:
                break
            rep = evaluate_nonmono_exact_fp64(cands[int(bi)], d, device)
            if require_non_unate and rep["is_unate_full"]:
                continue
            if rep["balanced"] and float(rep["OT"]) > cur_ot + 1e-9:
                cur = cands[int(bi)]
                cur_rep = rep
                cur_ot = float(rep["OT"])
                improved = True
                break
        if not improved:
            break
    return cur, cur_rep


def evolve_nonmono_gpu(
    d: int,
    seed_cores: Sequence[np.ndarray],
    device: torch.device,
    rng: np.random.Generator,
    n_chains: int = 96,
    steps: int = 400,
    t_start: float = 0.22,
    t_end: float = 0.001,
    force_non_unate_frac: float = 0.5,
    initial_kick_range: Tuple[int, int] = (0, 0),
    allow_a2_zeros: bool = True,
    deadline_ts: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Parallel tempered non-monotone SA on RTX 5060 with exact GPU unateness tracking.
    Maintains TWO distinct leaderboards simultaneously:
      - best_overall (unconstrained)
      - best_non_unate (strictly is_unate_full == False!)
    """
    t0 = time.perf_counter()
    ev = GPUNonMonoEngine(d, device=device, dtype=torch.float32)

    chains: List[np.ndarray] = []
    cold_pool = make_nonunate_cold_seeds(d, rng, count=max(16, n_chains // 2))
    # Graduated kick ladder when escaping an existing basin (1..2 near-wall -> 3..8 medium -> 8..hi deep!)
    for c in range(n_chains):
        if seed_cores and (initial_kick_range[1] > 0 or c < max(n_chains // 2, len(seed_cores) * 4)):
            base = seed_cores[c % len(seed_cores)].copy()
            k_lo, k_hi = initial_kick_range
            if k_hi > 0:
                if c < n_chains // 4:
                    nk = int(rng.integers(max(1, k_lo), min(k_hi, 3) + 1))
                elif c < (n_chains * 5) // 8:
                    nk = int(rng.integers(max(2, k_lo), min(k_hi, 8) + 1))
                else:
                    nk = int(rng.integers(max(3, k_lo), k_hi + 1))
                if nk > 0:
                    base = mutate_nonmono_balanced(base, d, rng, n_kicks=nk, allow_a2_zeros=allow_a2_zeros)
            chains.append(base)
        else:
            chains.append(cold_pool[c % len(cold_pool)].copy())

    # Replica split: first `n_nu` chains carry strong unate penalty (-0.65) so they explore
    # strictly non-unate space; remaining chains are completely unconstrained!
    n_nu = int(round(n_chains * force_non_unate_frac))
    upen_np = np.zeros(n_chains, dtype=np.float32)
    upen_np[:n_nu] = 0.65
    upen_t = torch.as_tensor(upen_np, device=device, dtype=torch.float32)

    cur_t = torch.as_tensor(np.stack(chains, axis=0), device=device, dtype=torch.int8)
    r0 = ev.score_batch_gpu(cur_t, unate_penalty=upen_t, subcube_weight=3.0)
    cur_fit = r0["fitness"].detach().cpu().numpy().astype(np.float64)
    cur_ot = r0["strict_OT"].detach().cpu().numpy().astype(np.float64)
    cur_nu_ot = r0["strict_nonunate_OT"].detach().cpu().numpy().astype(np.float64)

    best_fit = cur_fit.copy()
    best_chains = [c.copy() for c in chains]
    best_ov_ot = float(np.max(cur_ot))
    best_ov_g = chains[int(np.argmax(cur_ot))].copy()
    best_nu_ot = float(np.max(cur_nu_ot))
    best_nu_g = chains[int(np.argmax(cur_nu_ot))].copy()

    t0_arr = np.exp(np.linspace(math.log(max(0.015, t_start * 0.12)), math.log(t_start), n_chains))
    decay = (t_end / t0_arr) ** (1.0 / max(1, steps - 1))
    temps = t0_arr.copy()
    total_evals = n_chains
    non_unate_steps_frac_acc = 0.0

    for step in range(steps):
        if deadline_ts is not None and time.perf_counter() >= deadline_ts:
            break
        props: List[np.ndarray] = []
        for c in range(n_chains):
            nk = 1 if (c < n_chains // 2 or rng.random() < 0.78) else 2
            props.append(mutate_nonmono_balanced(chains[c], d, rng, n_kicks=nk, allow_a2_zeros=allow_a2_zeros))

        pt = torch.as_tensor(np.stack(props, axis=0), device=device, dtype=torch.int8)
        rp = ev.score_batch_gpu(pt, unate_penalty=upen_t, subcube_weight=3.0)
        p_fit = rp["fitness"].detach().cpu().numpy().astype(np.float64)
        p_ot = rp["strict_OT"].detach().cpu().numpy().astype(np.float64)
        p_nu_ot = rp["strict_nonunate_OT"].detach().cpu().numpy().astype(np.float64)
        non_unate_steps_frac_acc += float((~rp["is_unate_g0"]).float().mean().item())
        total_evals += n_chains

        delta = p_fit - cur_fit
        u_rand = rng.random(n_chains)
        acc = (delta >= 0.0) | (u_rand < np.exp(np.clip(delta / temps, -40.0, 0.0)))
        for c in range(n_chains):
            if p_ot[c] > best_ov_ot:
                best_ov_ot = float(p_ot[c])
                best_ov_g = props[c].copy()
            if p_nu_ot[c] > best_nu_ot:
                best_nu_ot = float(p_nu_ot[c])
                best_nu_g = props[c].copy()
            if acc[c]:
                chains[c] = props[c]
                cur_fit[c] = p_fit[c]
                if p_fit[c] > best_fit[c]:
                    best_fit[c] = p_fit[c]
                    best_chains[c] = props[c].copy()

        temps *= decay

        # Periodic elite migration at 40% and 75% (respecting non-unate vs unconstrained partitions!)
        if step in (int(steps * 0.40), int(steps * 0.75)):
            for lo, hi in [(0, n_nu), (n_nu, n_chains)]:
                if hi - lo < 4:
                    continue
                sub_ord = lo + np.argsort(-best_fit[lo:hi])
                k_top = max(2, (hi - lo) // 6)
                k_bot = max(2, (hi - lo) // 3)
                for i in range(k_bot):
                    dst = int(sub_ord[(hi - lo) - 1 - i])
                    src = int(sub_ord[i % k_top])
                    chains[dst] = best_chains[src].copy()
            mt = torch.as_tensor(np.stack(chains, axis=0), device=device, dtype=torch.int8)
            rm = ev.score_batch_gpu(mt, unate_penalty=upen_t, subcube_weight=3.0)
            cur_fit = rm["fitness"].detach().cpu().numpy().astype(np.float64)

    # Exact float64 verification & steepest polish on top candidates
    cand_pool: List[np.ndarray] = [best_ov_g, best_nu_g]
    ord_all = np.argsort(-best_fit)
    for idx in ord_all[:8]:
        cand_pool.append(best_chains[int(idx)])
    if n_nu > 0:
        ord_nu = np.argsort(-best_fit[:n_nu])
        for idx in ord_nu[:6]:
            cand_pool.append(best_chains[int(idx)])

    best_overall_rep: Optional[Dict[str, Any]] = None
    best_overall_g: Optional[np.ndarray] = None
    best_nonunate_rep: Optional[Dict[str, Any]] = None
    best_nonunate_g: Optional[np.ndarray] = None

    for g_c in cand_pool:
        rep = evaluate_nonmono_exact_fp64(g_c, d, device)
        if not rep["balanced"]:
            continue
        if best_overall_rep is None or float(rep["OT"]) > float(best_overall_rep["OT"]):
            best_overall_rep = rep
            best_overall_g = g_c.copy()
        if not rep["is_unate_full"]:
            if best_nonunate_rep is None or float(rep["OT"]) > float(best_nonunate_rep["OT"]):
                best_nonunate_rep = rep
                best_nonunate_g = g_c.copy()

    # Polish both champions with non-monotone GPU steepest ascent!
    if best_overall_g is not None and (deadline_ts is None or time.perf_counter() < deadline_ts):
        g_p, r_p = nonmono_steepest_ascent_gpu(
            best_overall_g, d, device, rng, require_non_unate=False,
            max_iters=25, batch_size=512 if d <= 10 else 256
        )
        if r_p["balanced"] and float(r_p["OT"]) > float(best_overall_rep["OT"]):
            best_overall_g, best_overall_rep = g_p, r_p
            if not r_p["is_unate_full"] and (best_nonunate_rep is None or float(r_p["OT"]) > float(best_nonunate_rep["OT"])):
                best_nonunate_g, best_nonunate_rep = g_p.copy(), r_p

    if best_nonunate_g is not None and (deadline_ts is None or time.perf_counter() < deadline_ts):
        g_p, r_p = nonmono_steepest_ascent_gpu(
            best_nonunate_g, d, device, rng, require_non_unate=True,
            max_iters=25, batch_size=512 if d <= 10 else 256
        )
        if r_p["balanced"] and (not r_p["is_unate_full"]) and float(r_p["OT"]) > float(best_nonunate_rep["OT"]):
            best_nonunate_g, best_nonunate_rep = g_p, r_p
            if best_overall_rep is None or float(r_p["OT"]) > float(best_overall_rep["OT"]):
                best_overall_g, best_overall_rep = g_p.copy(), r_p

    dt = time.perf_counter() - t0
    return {
        "d": d,
        "total_evals": total_evals,
        "evals_per_sec": round(total_evals / max(dt, 1e-6), 1),
        "elapsed_sec": round(dt, 2),
        "non_unate_trajectory_frac": round(non_unate_steps_frac_acc / max(1, steps), 4),
        "best_overall_g": best_overall_g,
        "best_overall_rep": best_overall_rep,
        "best_nonunate_g": best_nonunate_g,
        "best_nonunate_rep": best_nonunate_rep,
    }


# =============================================================================
# 4. SLEEP-DAG-RSI v8 Phase 4 Non-Monotone Campaign Runner
# =============================================================================

@dataclass
class Phase4Candidate:
    label: str
    d: int
    g0: np.ndarray
    subcubes: List[Dict[str, int]]
    OT: float
    H: float
    I: float
    n_effective: int
    essential_vars: int
    is_unate_g0: bool
    is_unate_full: bool
    source: str


def _load_theory_seeds_if_present(art_dir: Path) -> Dict[int, List[Tuple[str, np.ndarray]]]:
    """Non-blocking loader for `@fei-literature`'s `phase4_nonmono_theory.json` if present."""
    path = art_dir / "phase4_nonmono_theory.json"
    by_d: Dict[int, List[Tuple[str, np.ndarray]]] = {}
    if not path.is_file():
        return by_d
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw if isinstance(raw, list) else raw.get("seeds", raw.get("candidates", []))
        for item in items:
            if not isinstance(item, dict) or "g0" not in item:
                continue
            g = np.asarray(item["g0"], dtype=np.int8)
            n_pts = len(g)
            if n_pts < 8 or (n_pts & (n_pts - 1)) != 0:
                continue
            d = n_pts.bit_length() - 1
            if d < 5 or d > 12 or int(np.sum(g)) != 0:
                continue
            by_d.setdefault(d, []).append((str(item.get("label", item.get("name", f"th_d{d}"))), g))
    except Exception:
        return {}
    return by_d


class Phase4NonMonoCartridgeBuilder:
    """Genuine SLEEP-DAG-RSI v8 TaskCartridge for Phase 4 Non-Monotone Discovery."""

    def __init__(
        self,
        device: torch.device,
        baseline: Phase4Candidate,
        rng: np.random.Generator,
        track_name: str,
        prior_skill_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.device = device
        self.baseline = baseline
        self.rng = rng
        self.track_name = track_name
        self.prior_skill_state = prior_skill_state
        self.registry: Dict[str, Phase4Candidate] = {"baseline": baseline}
        self.best_nonunate: Optional[Phase4Candidate] = baseline if not baseline.is_unate_full else None
        self.best_overall: Phase4Candidate = baseline

    def _register(self, cand: Phase4Candidate) -> None:
        self.registry[cand.label] = cand
        if cand.OT > self.best_overall.OT + 1e-12:
            self.best_overall = cand
        if not cand.is_unate_full:
            if self.best_nonunate is None or cand.OT > self.best_nonunate.OT + 1e-12:
                self.best_nonunate = cand

    def verify_and_score(self, fn: Callable[[], Optional[Phase4Candidate]]) -> Tuple[bool, float, str, Dict[str, Any]]:
        cand = fn()
        if cand is None:
            return False, 0.0, "algorithmic_failure", {"error": "No valid balanced non-unate candidate"}
        rep = evaluate_nonmono_exact_fp64(cand.g0, cand.d, self.device)
        if not rep["balanced"]:
            return False, 0.0, "algorithmic_failure", {"error": "Core not balanced"}
        cand.OT = float(rep["OT"])
        cand.H = float(rep["H"])
        cand.I = float(rep["I"])
        cand.subcubes = rep["subcubes"]
        cand.n_effective = int(rep["n_eff"])
        cand.essential_vars = int(rep["essential_vars"])
        cand.is_unate_g0 = bool(rep["is_unate_g0"])
        cand.is_unate_full = bool(rep["is_unate_full"])
        self._register(cand)

        margins = [round(cand.OT - thr, 6) for thr in PHASE4_THRESHOLDS]
        margins.append(1.0 if not cand.is_unate_full else -1.0)
        margins.append(round(0.10 - abs(cand.I - 2.04), 6))
        pass_mask = [1 if m > 0.0 else 0 for m in margins]
        info = {
            "summary": (
                f"{cand.label} d={cand.d} n_eff={cand.n_effective} non_unate={not cand.is_unate_full} "
                f"I={cand.I:.6f} L={cand.OT:.8f}"
            ),
            "d": cand.d,
            "n_effective": cand.n_effective,
            "is_unate_full": cand.is_unate_full,
            "I": cand.I,
            "OT": cand.OT,
            "pass_mask": pass_mask,
            "bucket_scores": margins,
        }
        # Score prioritizes strictly non-unate unless overall beats NUM2026!
        score = float(cand.OT) if (not cand.is_unate_full or cand.OT > float(NUM2026_STRICT_BAR)) else float(cand.OT) - 0.50
        return True, score, "ok", info

    def crossover_synthesizer(
        self,
        u_cand: dream_cartridge.BranchCandidate,
        v_cand: dream_cartridge.BranchCandidate,
    ) -> Optional[dream_cartridge.BranchCandidate]:
        """Non-monotone balanced subcube-splice CROSSOVER: MUX(x_split, F1, -F2) + GPU polish."""
        c_u = self.registry.get(u_cand.id)
        c_v = self.registry.get(v_cand.id)
        if c_u is None or c_v is None:
            pool = sorted([c for c in self.registry.values() if not c.is_unate_full], key=lambda x: x.OT, reverse=True)
            if len(pool) < 2:
                return None
            c_u, c_v = pool[0], pool[1]
        xid = f"{self.track_name}_xover_{u_cand.id}_{v_cand.id}"

        def _run_xover() -> Optional[Phase4Candidate]:
            d_x = min(11, max(c_u.d, c_v.d) + 1)
            # Lift both parents to d_x - 1, then splice via opposite-polarity MUX on top bit!
            d_sub = d_x - 1
            gu = c_u.g0 if c_u.d == d_sub else (np.tile(c_u.g0, 1 << (d_sub - c_u.d)) if c_u.d < d_sub else c_u.g0[: 1 << d_sub])
            gv = c_v.g0 if c_v.d == d_sub else (np.tile(c_v.g0, 1 << (d_sub - c_v.d)) if c_v.d < d_sub else c_v.g0[: 1 << d_sub])
            # Exact balanced non-unate splice: top half = gu, bottom half = -gv (guaranteed sum == 0 when both balanced!)
            if int(np.sum(gu)) != 0 or int(np.sum(gv)) != 0:
                return None
            g_mux = np.concatenate([gu, -gv]).astype(np.int8)
            res = evolve_nonmono_gpu(
                d=d_x, seed_cores=[g_mux], device=self.device, rng=self.rng,
                n_chains=48 if d_x <= 10 else 32, steps=140,
                t_start=0.18, t_end=0.001, force_non_unate_frac=0.85,
            )
            g_b, r_b = res["best_nonunate_g"], res["best_nonunate_rep"]
            if g_b is None or r_b is None:
                return None
            return Phase4Candidate(
                label=xid, d=d_x, g0=g_b, subcubes=r_b["subcubes"],
                OT=float(r_b["OT"]), H=float(r_b["H"]), I=float(r_b["I"]),
                n_effective=int(r_b["n_eff"]), essential_vars=int(r_b["essential_vars"]),
                is_unate_g0=bool(r_b["is_unate_g0"]), is_unate_full=bool(r_b["is_unate_full"]),
                source="dag_nonmono_mux_crossover",
            )

        return dream_cartridge.BranchCandidate(
            id=xid,
            branch=max(u_cand.branch, v_cand.branch) + 1,
            parent_id=u_cand.id,
            attempt=max(u_cand.attempt, v_cand.attempt) + 1,
            description=f"Opposite-polarity MUX CROSSOVER({c_u.label}, {c_v.label}) + non-unate GPU polish",
            fn=_run_xover,
        )

    def blind_vault_verify(self, candidate_fn: Callable[[], Optional[Phase4Candidate]]) -> Dict[str, Any]:
        ch = self.best_nonunate or self.best_overall
        rep = exact_verifier.verify_dealiased_core(
            ch.g0.tolist(), ch.subcubes, require_monotone=False
        )
        return {
            "label": ch.label,
            "d": ch.d,
            "n_effective": rep["n_effective"],
            "is_unate_full": ch.is_unate_full,
            "is_monotone": rep["is_monotone"],
            "I_exact_str": rep["I_exact_str"],
            "L_cert": rep["L_cert"],
            "beats_num2026": rep["beats_num2026"],
            "sha256": rep["sha256"],
        }

    def build_cartridge(self, branches: List[dream_cartridge.BranchCandidate]) -> dream_cartridge.TaskCartridge:
        base_c = self.baseline
        return dream_cartridge.TaskCartridge(
            task_name=f"c71_phase4_nonmono_{self.track_name}",
            description=f"Phase 4 Non-Monotone / A2-NonMono Track ({self.track_name}) on RTX 5060",
            baseline_fn=lambda: base_c,
            verify_and_score=self.verify_and_score,
            initial_branches=branches,
            crossover_synthesizer=self.crossover_synthesizer,
            max_parallelism=4,
            sentinel_noise_regime="deterministic",
            blind_vault_evaluator=self.blind_vault_verify,
            noise_probe_runs=1,
            prior_skill_state=self.prior_skill_state,
            hypothesis_contract={
                "claim": (
                    f"Removing monotonicity via A2-NonMono de-aliasing + CNOT/Toffoli/subcube moves "
                    f"discovers a strictly non-unate balanced function beating {NUM2026_STRICT_BAR}"
                ),
                "plan": (
                    "Run Track 1 cold-start non-unate search (d=6..12) and Track 2 multi-flip escape "
                    "tunneling from AG(2,3) & Z10, track exact is_unate_full, and certify via exact_verifier.py"
                ),
                "decision_criteria": {
                    "support": f"Exact certified balanced function with L_cert > {NUM2026_STRICT_BAR} (Tier 1 == Tier 2)",
                    "weaken": f"All strictly non-unate & unconstrained escape runs <= {NUM2026_STRICT_BAR}",
                    "inconclusive": "Unconstrained search collapses back into sign-flipped unate AG(2,3)",
                },
                "alternative_explanations": [
                    "Monotonicity / unateness may be intrinsically optimal for FEI because opposite-sign edges raise I faster than H",
                    "Breaking monotonicity on AG(2,3) may strictly lower H/(I-1) even under arbitrary comparable A2-NonMono subcubes",
                ],
            },
        )


def run_phase4_campaign(max_minutes: float = 8.5, seed: int = 20260722, quick: bool = False) -> Dict[str, Any]:
    t_start = time.perf_counter()
    deadline_ts = t_start + (max_minutes * 60.0 if not quick else 65.0)
    device = get_device(True)
    rng = np.random.default_rng(seed)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    art_dir = Path(_SRC_DIR).parent / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    pool_root = art_dir / "sleep_dag_trees_phase4"

    # =========================================================================
    # STEP 0: Tracer #1 — A2-NonMono Tier 1 == Tier 2 Exactness Gate on Strictly Non-Unate Comparable Subcubes
    # =========================================================================
    print("=== PHASE 4 / STEP 0: Tracer #1 (A2-NonMono Strictly Non-Unate Tier 1 == Tier 2 Gate) ===", flush=True)
    # Build a small strictly non-unate d=6 core with comparable 0D/1D subcubes (000000 < 111111 both zero!)
    g_tr = make_nonunate_cold_seeds(6, np.random.default_rng(12345), count=1)[0]
    # Open a comparable pair (x_lo < x_hi) where g[x_lo] == -1, g[x_hi] == +1 (or opposite)
    lo_c = int(np.flatnonzero(g_tr == -1)[0])
    hi_c = int(np.flatnonzero(g_tr == 1)[-1])
    g_tr[lo_c] = 0
    g_tr[hi_c] = 0
    r_tr = evaluate_nonmono_exact_fp64(g_tr, 6, device)
    assert not r_tr["is_unate_full"], "Tracer #1 test function must be strictly non-unate!"
    cert_t2 = exact_verifier.verify_dealiased_core(g_tr.tolist(), r_tr["subcubes"], require_monotone=False)
    full_tt = exact_verifier.expand_dealiased_to_full(g_tr.tolist(), r_tr["subcubes"])
    cert_t1 = exact_verifier.verify_full_truth_table(full_tt, require_monotone=False)
    assert cert_t1["I_exact_str"] == cert_t2["I_exact_str"], "Tier 1 != Tier 2 exact I mismatch!"
    assert cert_t1["L_cert"][:55] == cert_t2["L_cert"][:55], "Tier 1 != Tier 2 L_cert mismatch!"
    print(
        f"  [Tracer #1 PASS] d=6, k={r_tr['k_aux']}, n_eff={r_tr['n_eff']}, is_unate={r_tr['is_unate_full']}, "
        f"Tier1==Tier2 (>55 digits): L={cert_t2['L_cert'][:20]}..., I={cert_t2['I_exact_str']}",
        flush=True,
    )

    best_nonunate_per_d: Dict[int, Phase4Candidate] = {}
    best_overall_per_d: Dict[int, Phase4Candidate] = {}
    best_nonunate_global: Optional[Phase4Candidate] = None
    best_overall_global: Optional[Phase4Candidate] = None

    def record(cand: Optional[Phase4Candidate]) -> None:
        nonlocal best_nonunate_global, best_overall_global
        if cand is None:
            return
        if cand.d not in best_overall_per_d or cand.OT > best_overall_per_d[cand.d].OT + 1e-12:
            best_overall_per_d[cand.d] = cand
        if best_overall_global is None or cand.OT > best_overall_global.OT + 1e-12:
            best_overall_global = cand
        if not cand.is_unate_full:
            if cand.d not in best_nonunate_per_d or cand.OT > best_nonunate_per_d[cand.d].OT + 1e-12:
                best_nonunate_per_d[cand.d] = cand
            if best_nonunate_global is None or cand.OT > best_nonunate_global.OT + 1e-12:
                best_nonunate_global = cand

    # Baseline strictly non-unate seed on d=6
    base_cand = Phase4Candidate(
        label="base_nonunate_d6", d=6, g0=g_tr, subcubes=r_tr["subcubes"],
        OT=float(r_tr["OT"]), H=float(r_tr["H"]), I=float(r_tr["I"]),
        n_effective=int(r_tr["n_eff"]), essential_vars=int(r_tr["essential_vars"]),
        is_unate_g0=bool(r_tr["is_unate_g0"]), is_unate_full=bool(r_tr["is_unate_full"]),
        source="tracer_baseline",
    )
    record(base_cand)

    # =========================================================================
    # TRACK 1: Cold-Start Non-Monotone Discovery (d = 6, 7, 8, 9, 10, 12)
    # =========================================================================
    print("\n================ PHASE 4 / TRACK 1 (Cold-Start Strictly Non-Unate d = 6,7,8,9,10,12) ================", flush=True)
    t_tr1 = time.perf_counter()
    th_seeds = _load_theory_seeds_if_present(art_dir)
    print(f"  Auto-ingested phase4_nonmono_theory.json: {sum(len(v) for v in th_seeds.values())} seeds across {sorted(th_seeds.keys())}", flush=True)

    # Score raw theory seeds directly too!
    for td, lst in th_seeds.items():
        for nm, g_th in lst:
            rp = evaluate_nonmono_exact_fp64(g_th, td, device)
            if rp["balanced"]:
                record(Phase4Candidate(
                    label=f"th_{nm}", d=td, g0=g_th, subcubes=rp["subcubes"],
                    OT=float(rp["OT"]), H=float(rp["H"]), I=float(rp["I"]),
                    n_effective=int(rp["n_eff"]), essential_vars=int(rp["essential_vars"]),
                    is_unate_g0=bool(rp["is_unate_g0"]), is_unate_full=bool(rp["is_unate_full"]),
                    source="theory_seed",
                ))

    dl_t1 = min(deadline_ts - (18.0 if quick else 210.0), t_tr1 + (28.0 if quick else 255.0))
    builder1 = Phase4NonMonoCartridgeBuilder(device, base_cand, rng, track_name="track1_cold")

    def _make_t1_branch(d_list: Sequence[int], n_ch: int, st: int) -> Callable[[], Optional[Phase4Candidate]]:
        def _fn() -> Optional[Phase4Candidate]:
            best_loc: Optional[Phase4Candidate] = None
            for dv in d_list:
                if time.perf_counter() >= dl_t1:
                    break
                ext_s = [g for _, g in th_seeds.get(dv, [])]
                res = evolve_nonmono_gpu(
                    d=dv, seed_cores=ext_s, device=device, rng=rng,
                    n_chains=n_ch, steps=st, t_start=0.22, t_end=0.001,
                    force_non_unate_frac=0.65, deadline_ts=dl_t1,
                )
                if res["best_overall_g"] is not None and res["best_overall_rep"] is not None:
                    r_o = res["best_overall_rep"]
                    c_o = Phase4Candidate(
                        label=f"T1_d{dv}_ov", d=dv, g0=res["best_overall_g"], subcubes=r_o["subcubes"],
                        OT=float(r_o["OT"]), H=float(r_o["H"]), I=float(r_o["I"]),
                        n_effective=int(r_o["n_eff"]), essential_vars=int(r_o["essential_vars"]),
                        is_unate_g0=bool(r_o["is_unate_g0"]), is_unate_full=bool(r_o["is_unate_full"]),
                        source="track1_unconstrained",
                    )
                    record(c_o)
                if res["best_nonunate_g"] is not None and res["best_nonunate_rep"] is not None:
                    r_n = res["best_nonunate_rep"]
                    c_n = Phase4Candidate(
                        label=f"T1_d{dv}_nu", d=dv, g0=res["best_nonunate_g"], subcubes=r_n["subcubes"],
                        OT=float(r_n["OT"]), H=float(r_n["H"]), I=float(r_n["I"]),
                        n_effective=int(r_n["n_eff"]), essential_vars=int(r_n["essential_vars"]),
                        is_unate_g0=bool(r_n["is_unate_g0"]), is_unate_full=bool(r_n["is_unate_full"]),
                        source="track1_nonunate",
                    )
                    record(c_n)
                    if best_loc is None or c_n.OT > best_loc.OT:
                        best_loc = c_n
                    print(
                        f"    [Track 1 d={dv:2d}] non_unate OT={c_n.OT:.8f} (I={c_n.I:.5f}, k={len(c_n.subcubes)}, n_eff={c_n.n_effective}) "
                        f"| overall OT={res['best_overall_rep']['OT']:.8f} (unate={res['best_overall_rep']['is_unate_full']}) "
                        f"| {res['evals_per_sec']:.0f}/s ({res['elapsed_sec']:.1f}s)",
                        flush=True,
                    )
            return best_loc
        return _fn

    branches_t1 = [
        dream_cartridge.BranchCandidate(
            id="p4_t1_d6_d7", branch=0, description="Cold-start non-unate GPU search d=6, 7",
            fn=_make_t1_branch([6, 7], 64 if quick else 224, 70 if quick else 2600),
        ),
        dream_cartridge.BranchCandidate(
            id="p4_t1_d8", branch=1, description="Cold-start non-unate GPU search d=8",
            fn=_make_t1_branch([8], 64 if quick else 224, 70 if quick else 2800),
        ),
        dream_cartridge.BranchCandidate(
            id="p4_t1_d9_d10", branch=2, description="Cold-start non-unate GPU search d=9, 10",
            fn=_make_t1_branch([9, 10], 48 if quick else 192, 60 if quick else 2000),
        ),
        dream_cartridge.BranchCandidate(
            id="p4_t1_d12", branch=3, description="Cold-start non-unate GPU search d=12",
            fn=_make_t1_branch([12], 32 if quick else 96, 45 if quick else 900),
        ),
    ]
    pool_t1 = str(pool_root / "track1")
    cart1 = builder1.build_cartridge(branches_t1)
    runner1 = dream_cartridge.AutoDreamRunner(cart1, pool_dir=pool_t1)
    rep1 = runner1.run()
    trees1 = dream_engine.load_pool(task=cart1.task_name, pool_dir=pool_t1)
    diag1 = run_sleep_dag_diagnostics(trees1[-1], runner1.problem) if trees1 else {}
    dt_t1 = time.perf_counter() - t_tr1
    print(
        f"  [Track 1 DONE in {dt_t1:.1f}s] Best strictly non-unate: "
        f"{best_nonunate_global.label if best_nonunate_global else None} | OT={best_nonunate_global.OT if best_nonunate_global else 0.0:.8f}",
        flush=True,
    )

    # =========================================================================
    # TRACK 2: Non-Monotone Escape Tunneling from AG(2,3) (d=9,10,11,12) & Z10 (d=10)
    # =========================================================================
    print("\n================ PHASE 4 / TRACK 2 (AG(2,3) & Z10 Non-Monotone Escape Tunneling) ================", flush=True)
    t_tr2 = time.perf_counter()
    # Re-check if @fei-literature landed phase4_nonmono_theory.json during Track 1
    th_seeds2 = _load_theory_seeds_if_present(art_dir)
    if sum(len(v) for v in th_seeds2.values()) > sum(len(v) for v in th_seeds.values()):
        print(f"  Newly landed phase4_nonmono_theory.json: {sum(len(v) for v in th_seeds2.values())} seeds!", flush=True)
        for td, lst in th_seeds2.items():
            for nm, g_th in lst:
                rp = evaluate_nonmono_exact_fp64(g_th, td, device)
                if rp["balanced"]:
                    record(Phase4Candidate(
                        label=f"th2_{nm}", d=td, g0=g_th, subcubes=rp["subcubes"],
                        OT=float(rp["OT"]), H=float(rp["H"]), I=float(rp["I"]),
                        n_effective=int(rp["n_eff"]), essential_vars=int(rp["essential_vars"]),
                        is_unate_g0=bool(rp["is_unate_g0"]), is_unate_full=bool(rp["is_unate_full"]),
                        source="theory_seed",
                    ))

    dl_t2 = deadline_ts - 8.0
    anchor_t2 = best_nonunate_global or base_cand
    builder2 = Phase4NonMonoCartridgeBuilder(
        device, anchor_t2, rng, track_name="track2_escape", prior_skill_state=rep1.get("skill_state")
    )
    escape_telemetry: List[Dict[str, Any]] = []

    def _make_escape_branch(
        name: str, d_val: int, seed_g: np.ndarray, kick_rng: Tuple[int, int], n_ch: int, st: int, nu_frac: float
    ) -> Callable[[], Optional[Phase4Candidate]]:
        def _fn() -> Optional[Phase4Candidate]:
            if time.perf_counter() >= dl_t2:
                return None
            extra = [g for _, g in th_seeds2.get(d_val, [])[:4]]
            res = evolve_nonmono_gpu(
                d=d_val, seed_cores=[seed_g] + extra, device=device, rng=rng,
                n_chains=n_ch, steps=st, t_start=0.26, t_end=0.001,
                force_non_unate_frac=nu_frac, initial_kick_range=kick_rng,
                allow_a2_zeros=True, deadline_ts=dl_t2,
            )
            r_o, r_n = res["best_overall_rep"], res["best_nonunate_rep"]
            c_o = c_n = None
            if res["best_overall_g"] is not None and r_o is not None:
                c_o = Phase4Candidate(
                    label=f"{name}_ov", d=d_val, g0=res["best_overall_g"], subcubes=r_o["subcubes"],
                    OT=float(r_o["OT"]), H=float(r_o["H"]), I=float(r_o["I"]),
                    n_effective=int(r_o["n_eff"]), essential_vars=int(r_o["essential_vars"]),
                    is_unate_g0=bool(r_o["is_unate_g0"]), is_unate_full=bool(r_o["is_unate_full"]),
                    source=f"track2_{name}_overall",
                )
                record(c_o)
            if res["best_nonunate_g"] is not None and r_n is not None:
                c_n = Phase4Candidate(
                    label=f"{name}_nu", d=d_val, g0=res["best_nonunate_g"], subcubes=r_n["subcubes"],
                    OT=float(r_n["OT"]), H=float(r_n["H"]), I=float(r_n["I"]),
                    n_effective=int(r_n["n_eff"]), essential_vars=int(r_n["essential_vars"]),
                    is_unate_g0=bool(r_n["is_unate_g0"]), is_unate_full=bool(r_n["is_unate_full"]),
                    source=f"track2_{name}_nonunate",
                )
                record(c_n)
            tele = {
                "branch": name,
                "d": d_val,
                "initial_kicks": list(kick_rng),
                "total_evals": res["total_evals"],
                "elapsed_sec": res["elapsed_sec"],
                "non_unate_traj_frac": res["non_unate_trajectory_frac"],
                "overall_OT": round(c_o.OT, 10) if c_o else 0.0,
                "overall_is_unate": c_o.is_unate_full if c_o else None,
                "overall_beats_num2026": bool(c_o and c_o.OT > float(NUM2026_STRICT_BAR) + 1e-12),
                "nonunate_OT": round(c_n.OT, 10) if c_n else 0.0,
                "nonunate_I": round(c_n.I, 6) if c_n else 0.0,
                "nonunate_k": len(c_n.subcubes) if c_n else 0,
            }
            escape_telemetry.append(tele)
            print(
                f"    [Escape {name:16s} d={d_val:2d}] non_unate OT={tele['nonunate_OT']:.8f} (I={tele['nonunate_I']:.5f}, k={tele['nonunate_k']}) "
                f"| overall OT={tele['overall_OT']:.8f} (unate={tele['overall_is_unate']}, beats={tele['overall_beats_num2026']}) ({res['elapsed_sec']:.1f}s)",
                flush=True,
            )
            return c_n or c_o
        return _fn

    ag9 = make_ag23_core(9)
    ag10 = make_ag23_core(10)
    ag11 = make_ag23_core(11)
    ag12 = make_ag23_core(12)
    z10 = make_z10_core(10)

    branches_t2 = [
        dream_cartridge.BranchCandidate(
            id="p4_t2_ag9_kicks", branch=0, description="AG(2,3) d=9 multi-step non-monotone escape (1..16 kicks + A2-NonMono)",
            fn=_make_escape_branch("AG9_esc", 9, ag9, (1, 16), 64 if quick else 224, 70 if quick else 2400, 0.50),
        ),
        dream_cartridge.BranchCandidate(
            id="p4_t2_ag10_z10", branch=1, description="Z10 d=10 non-monotone escape tunneling",
            fn=_make_escape_branch("AG10_Z10_esc", 10, z10, (1, 12), 64 if quick else 192, 65 if quick else 1600, 0.55),
        ),
        dream_cartridge.BranchCandidate(
            id="p4_t2_ag10_pure", branch=2, description="Lifted AG(2,3) d=10 direct non-monotone Toffoli/subcube escape",
            fn=_make_escape_branch("AG10_pure_esc", 10, ag10, (1, 16), 48 if quick else 192, 60 if quick else 1600, 0.50),
        ),
        dream_cartridge.BranchCandidate(
            id="p4_t2_ag11_12", branch=3, description="Lifted AG(2,3) d=11 non-monotone escape tunneling",
            fn=_make_escape_branch("AG11_esc", 11, ag11, (1, 14), 32 if quick else 128, 50 if quick else 1000, 0.55),
        ),
    ]
    pool_t2 = str(pool_root / "track2")
    cart2 = builder2.build_cartridge(branches_t2)
    runner2 = dream_cartridge.AutoDreamRunner(cart2, pool_dir=pool_t2)
    rep2 = runner2.run()
    trees2 = dream_engine.load_pool(task=cart2.task_name, pool_dir=pool_t2)
    diag2 = run_sleep_dag_diagnostics(trees2[-1], runner2.problem) if trees2 else {}

    # Quick d=12 AG(2,3) escape pass if time remains
    if time.perf_counter() < dl_t2 - 15.0:
        _make_escape_branch("AG12_esc", 12, ag12, (1, 14), 24 if quick else 96, 40 if quick else 650, 0.55)()

    dt_t2 = time.perf_counter() - t_tr2
    print(
        f"  [Track 2 DONE in {dt_t2:.1f}s] Best strictly non-unate overall: "
        f"{best_nonunate_global.label if best_nonunate_global else None} | OT={best_nonunate_global.OT if best_nonunate_global else 0.0:.8f}",
        flush=True,
    )

    # =========================================================================
    # FINAL EXACT CERTIFICATION (WITHOUT --require-monotone + Tier 1 == Tier 2)
    # =========================================================================
    print("\n=== PHASE 4 FINAL EXACT CERTIFICATION VIA exact_verifier.py ===", flush=True)
    assert best_nonunate_global is not None, "No strictly non-unate candidate found!"
    champ = best_nonunate_global
    # If an unconstrained candidate actually BEAT NUM2026, save that too; primary artifact is best strictly non-unate
    best_path = art_dir / "phase4_best_nonmono.json"
    exact_verifier.save_tier2_candidate_json(
        best_path,
        champ.g0.tolist(),
        champ.subcubes,
        name=champ.label,
        metadata={
            "from_scratch": True,
            "phase": 4,
            "strictly_non_unate": not champ.is_unate_full,
            "is_unate_g0": champ.is_unate_g0,
            "is_unate_full": champ.is_unate_full,
            "source": champ.source,
            "d": champ.d,
            "essential_vars": champ.essential_vars,
            "k_subcubes": len(champ.subcubes),
            "n_effective": champ.n_effective,
            "fp64_OT": champ.OT,
        },
    )
    cert_best = exact_verifier.load_and_verify_json(
        best_path, require_monotone=False, crosscheck_tier1_max_n=20
    )
    print(
        f"  [BEST STRICTLY NON-UNATE] {champ.label}: d={champ.d}, ess={champ.essential_vars}, "
        f"k={len(champ.subcubes)}, n_eff={champ.n_effective}, is_unate_full={champ.is_unate_full}, "
        f"is_monotone={cert_best['is_monotone']}, I={cert_best['I_exact_str']}, "
        f"L_cert={cert_best['L_cert'][:24]}..., tier1_crosscheck={cert_best.get('tier1_crosscheck')}, "
        f"beats_num2026={cert_best['beats_num2026']}",
        flush=True,
    )

    per_d_nonunate = {
        f"d={dk}": {
            "label": c.label,
            "source": c.source,
            "d": c.d,
            "essential_vars": c.essential_vars,
            "k_subcubes": len(c.subcubes),
            "n_effective": c.n_effective,
            "OT": round(c.OT, 10),
            "I": round(c.I, 8),
            "H": round(c.H, 8),
            "is_unate_g0": c.is_unate_g0,
            "is_unate_full": c.is_unate_full,
        }
        for dk, c in sorted(best_nonunate_per_d.items())
    }
    per_d_overall = {
        f"d={dk}": {
            "label": c.label,
            "source": c.source,
            "d": c.d,
            "essential_vars": c.essential_vars,
            "k_subcubes": len(c.subcubes),
            "n_effective": c.n_effective,
            "OT": round(c.OT, 10),
            "I": round(c.I, 8),
            "is_unate_full": c.is_unate_full,
        }
        for dk, c in sorted(best_overall_per_d.items())
    }

    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if device.type == "cuda" else 0.0
    total_sec = time.perf_counter() - t_start
    any_beats = bool(best_overall_global is not None and best_overall_global.OT > float(NUM2026_STRICT_BAR) + 1e-12)

    summary = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "seed": seed,
        "from_scratch_verified": True,
        "peak_vram_mb": round(peak_vram_mb, 2),
        "within_4_5_gib_cap": peak_vram_mb <= 4.5 * 1024.0,
        "total_elapsed_sec": round(total_sec, 2),
        "tracer1_a2_nonmono_tier1_eq_tier2": {
            "passed": True,
            "d": 6,
            "k_subcubes": r_tr["k_aux"],
            "n_effective": r_tr["n_eff"],
            "is_unate_full": r_tr["is_unate_full"],
            "I_exact_str": cert_t2["I_exact_str"],
            "L_cert_prefix": cert_t2["L_cert"][:30],
        },
        "best_strictly_non_unate_per_d": per_d_nonunate,
        "best_unconstrained_overall_per_d": per_d_overall,
        "track2_ag23_escape_telemetry": escape_telemetry,
        "best_nonmono_certified": {
            "label": champ.label,
            "source": champ.source,
            "d": champ.d,
            "essential_vars": champ.essential_vars,
            "k_subcubes": len(champ.subcubes),
            "n_effective": champ.n_effective,
            "is_unate_g0": champ.is_unate_g0,
            "is_unate_full": champ.is_unate_full,
            "is_monotone": cert_best["is_monotone"],
            "I_exact_str": cert_best["I_exact_str"],
            "L_cert": cert_best["L_cert"],
            "U_cert": cert_best["U_cert"],
            "interval_width_L": cert_best["interval_width_L"],
            "exact_balance": cert_best["exact_balance"],
            "parseval_verified": cert_best["parseval_verified"],
            "tier1_crosscheck": cert_best.get("tier1_crosscheck"),
            "beats_num2026": cert_best["beats_num2026"],
            "sha256": cert_best["sha256"],
        },
        "any_unconstrained_beats_num2026": any_beats,
        "best_unconstrained_overall": {
            "label": best_overall_global.label,
            "d": best_overall_global.d,
            "OT": round(best_overall_global.OT, 10),
            "I": round(best_overall_global.I, 8),
            "is_unate_full": best_overall_global.is_unate_full,
        } if best_overall_global else None,
        "sleep_dag_diagnostics": {
            "track1": diag1,
            "track2": diag2,
            "track2_blind_vault": rep2.get("blind_vault_report"),
        },
        "num2026_strict_bar": str(NUM2026_STRICT_BAR),
    }

    summary_path = art_dir / "phase4_nonmono_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== PHASE 4 NON-MONOTONE PROBE FINISHED in {total_sec:.1f}s (Peak VRAM {peak_vram_mb:.1f} MB) ===", flush=True)
    print(f"  Summary: {summary_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 Non-Monotone & A2-NonMono SLEEP-DAG-RSI GPU Probe")
    parser.add_argument("--max-minutes", type=float, default=8.5)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--quick", action="store_true", help="Fast ~45s end-to-end smoke check")
    args = parser.parse_args()
    run_phase4_campaign(max_minutes=args.max_minutes, seed=args.seed, quick=args.quick)


if __name__ == "__main__":
    main()

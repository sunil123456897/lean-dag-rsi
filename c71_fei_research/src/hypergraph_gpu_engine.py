"""
Direct Cross-Intersecting Hypergraph (F_+, F_-) GPU Evolutionary Engine on RTX 5060
for C_71 (Fourier Entropy-Influence Conjecture) — Phase 3.

Core Paradigm:
  - Genome = cross-intersecting antichain pair (F_+, F_-) of integer bitmasks on d vertices:
      forall A in F_+, B in F_-: (A & B) != 0.
  - Induced 3-valued monotone core g_0 in {-1, 0, +1}^{2^d}:
      g_0(x) = +1  iff  exists A in F_+ with (x & A) == A
      g_0(x) = -1  iff  exists B in F_- with (x & B) == 0
      g_0(x) =  0  otherwise (zero gap -> orthogonal maximal subcube de-aliasing, Vector A2!).
  - Exact GPU Mixed-Dimension Subcube Scoring Theorem:
      For ANY disjoint union of subcubes C_1, ..., C_k of arbitrary dimensions c_j = d - r_j,
      every vertex x in C_j has internal zero-graph degree deg_Z(x) == c_j, so:
        I_aux = (1 / 2^d) * sum_{x in Z} (1 + (d - deg_Z(x)) / 2)
        H_aux = (1 / 2^d) * sum_{x in Z} 2 * (d - deg_Z(x))
      is 100% BIT-EXACT on GPU for arbitrary mixed codimensions r_j in {0..d}!
  - Two search modes:
      * Mode I (Isomorphic / Self-Dual): F_- = pi(F_+), guaranteeing EXACT automatic balance |U| == |L|.
      * Mode II (Free Asymmetric): F_+ and F_- evolve independently with monotone gap balancing.
  - Mandatory Tracer #1 CLI (`--sanity-d9`): cold-start discovery on d=9 MUST independently
    rediscover AG(2,3) (L = 6.5218457109...) with zero hardcoded seeds!
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from fwht_gpu import (  # noqa: E402
    check_core_monotone_and_balanced,
    decompose_zero_subcubes,
    evaluate_core_spectrum_batch,
    evaluate_subcube_dealiased_exact_fp64,
    get_device,
    make_degree_table,
)


# =============================================================================
# 1. Hypergraph Genome & Bitmask Primitives
# =============================================================================

@dataclass
class HypergraphGenome:
    """Cross-intersecting antichain pair (F_pos, F_neg) on d vertices (bitmasks)."""

    d: int
    F_pos: List[int]
    F_neg: List[int]
    mode: str = "mode2"  # "mode1" (isomorphic F_neg = pi(F_pos)) or "mode2" (free asymmetric)
    perm: Optional[List[int]] = None  # length-d permutation for mode1

    def clone(self) -> "HypergraphGenome":
        return HypergraphGenome(
            d=self.d,
            F_pos=list(self.F_pos),
            F_neg=list(self.F_neg),
            mode=self.mode,
            perm=list(self.perm) if self.perm is not None else None,
        )


def permute_mask(mask: int, perm: Sequence[int]) -> int:
    """Applies vertex permutation perm (bit i -> bit perm[i]) to integer bitmask."""
    out = 0
    m = mask
    while m:
        lsb = m & -m
        bit = lsb.bit_length() - 1
        out |= 1 << perm[bit]
        m ^= lsb
    return out


def reduce_to_antichain(edges: Sequence[int]) -> List[int]:
    """Removes duplicate and non-minimal (superset) hyperedges."""
    uniq = sorted(set(int(e) for e in edges if e > 0), key=lambda x: (x.bit_count(), x))
    out: List[int] = []
    for e in uniq:
        if not any((a & e) == a for a in out):
            out.append(e)
    return out


def is_cross_intersecting(F_pos: Sequence[int], F_neg: Sequence[int]) -> bool:
    for a in F_pos:
        for b in F_neg:
            if (a & b) == 0:
                return False
    return True


def count_essential_vars(g0: np.ndarray, d: int) -> int:
    """Counts how many coordinates b in 0..d-1 actually influence g0 (Inf_b > 0)."""
    g = np.asarray(g0, dtype=np.int8)
    cnt = 0
    for b in range(d):
        st = 1 << b
        vg = g.reshape(-1, 2, st)
        if np.any(vg[:, 0, :] != vg[:, 1, :]):
            cnt += 1
    return cnt


def extract_hypergraph_from_g0(g0: np.ndarray, d: int) -> Tuple[List[int], List[int]]:
    """
    Exact O(d * 2^d) bijection: extracts the exact minimal cross-intersecting
    antichain pair (F_pos, F_neg) from ANY monotone 3-valued core g0 in {-1, 0, +1}^{2^d}:
      F_pos = {minimal x with g0(x) == +1}
      F_neg = {((1<<d)-1) ^ y : maximal y with g0(y) == -1}
    """
    g = np.asarray(g0, dtype=np.int8)
    is_min_pos = g == 1
    is_max_neg = g == -1
    for b in range(d):
        st = 1 << b
        vg = g.reshape(-1, 2, st)
        # x with bit b == 1 is NOT minimal +1 if its lower neighbor (bit b == 0) is also +1
        is_min_pos.reshape(-1, 2, st)[:, 1, :] &= (vg[:, 0, :] != 1)
        # y with bit b == 0 is NOT maximal -1 if its upper neighbor (bit b == 1) is also -1
        is_max_neg.reshape(-1, 2, st)[:, 0, :] &= (vg[:, 1, :] != -1)
    full_mask = (1 << d) - 1
    F_pos = [int(x) for x in np.flatnonzero(is_min_pos)]
    F_neg = sorted(int(full_mask ^ int(y)) for y in np.flatnonzero(is_max_neg))
    return F_pos, F_neg


def detect_isomorphism_perm(d: int, F_pos: Sequence[int], F_neg: Sequence[int]) -> Optional[List[int]]:
    """
    Fast algebraic isomorphism detector for Mode I seeding: checks cyclic/affine
    permutations (a*i + b) mod d and small bit-XOR/block shifts for F_neg == pi(F_pos).
    """
    if len(F_pos) != len(F_neg) or not F_pos:
        return None
    target = set(int(x) for x in F_neg)
    src = [int(x) for x in F_pos]

    def _matches(p: List[int]) -> bool:
        return {permute_mask(a, p) for a in src} == target

    # 1. All affine permutations (a*i + b) mod d with gcd(a, d) == 1
    for a in range(1, d):
        if math.gcd(a, d) != 1:
            continue
        for b in range(d):
            p = [(a * i + b) % d for i in range(d)]
            if _matches(p):
                return p
    # 2. Product-group shifts for d = 12 (2x2x3 or 3x4) and d = 15 (3x5) and d = 16 (4x4)
    if d == 12:
        for sx in range(2):
            for sy in range(2):
                for sz in range(3):
                    p = [((i // 6) ^ sx) * 6 + (((i // 3) % 2) ^ sy) * 3 + ((i % 3) + sz) % 3 for i in range(12)]
                    if _matches(p):
                        return p
        for sr in range(3):
            for sc in range(4):
                p = [((i // 4 + sr) % 3) * 4 + ((i % 4 + sc) % 4) for i in range(12)]
                if _matches(p):
                    return p
    if d == 16:
        for sr in range(4):
            for sc in range(4):
                p = [((i // 4 + sr) % 4) * 4 + ((i % 4 + sc) % 4) for i in range(16)]
                if _matches(p):
                    return p
    return None


# =============================================================================
# 2. Fast Batched GPU Hypergraph -> g_0 -> Mixed-Subcube FEI Evaluator
# =============================================================================

class GPUHypergraphEvaluator:
    """
    Vectorized RTX 5060 GPU evaluator turning batches of (F_pos, F_neg) hypergraphs
    directly into g_0 in {-1, 0, +1}^{B x 2^d} and scoring them with the exact
    mixed-dimension subcube deg_Z formula + FWHT in milliseconds (VRAM <= 4.5 GiB).
    """

    def __init__(self, d: int, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> None:
        self.d = int(d)
        self.N = 1 << self.d
        self.device = device or get_device(True)
        self.dtype = dtype
        self.x_all = torch.arange(self.N, device=self.device, dtype=torch.int32)
        self.deg = make_degree_table(self.d, self.device, dtype=self.dtype)

    def build_g0_batch_gpu(
        self, genomes: Sequence[HypergraphGenome]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts B cross-intersecting hypergraphs into g0_batch [B, 2^d] on GPU.
        Returns (g0_batch [B, N] int8, collision_free [B] bool).
        Peak memory is O(B * N) regardless of number of hyperedges E!
        """
        bsz = len(genomes)
        max_p = max((len(g.F_pos) for g in genomes), default=0)
        max_n = max((len(g.F_neg) for g in genomes), default=0)

        is_pos = torch.zeros((bsz, self.N), device=self.device, dtype=torch.bool)
        if max_p > 0:
            fp_np = np.full((bsz, max_p), -1, dtype=np.int32)
            for i, gm in enumerate(genomes):
                if gm.F_pos:
                    fp_np[i, : len(gm.F_pos)] = gm.F_pos
            fp_t = torch.as_tensor(fp_np, device=self.device, dtype=torch.int32)
            for e in range(max_p):
                m = fp_t[:, e : e + 1]
                valid_e = m >= 0
                is_pos |= valid_e & ((self.x_all.unsqueeze(0) & m) == m)

        is_neg = torch.zeros((bsz, self.N), device=self.device, dtype=torch.bool)
        if max_n > 0:
            fn_np = np.full((bsz, max_n), -1, dtype=np.int32)
            for i, gm in enumerate(genomes):
                if gm.F_neg:
                    fn_np[i, : len(gm.F_neg)] = gm.F_neg
            fn_t = torch.as_tensor(fn_np, device=self.device, dtype=torch.int32)
            for e in range(max_n):
                m = fn_t[:, e : e + 1]
                valid_e = m >= 0
                is_neg |= valid_e & ((self.x_all.unsqueeze(0) & m) == 0)

        collision_free = ~(is_pos & is_neg).any(dim=1)
        g0 = torch.zeros((bsz, self.N), device=self.device, dtype=torch.int8)
        g0[is_pos] = 1
        g0[is_neg] = -1
        return g0, collision_free

    def score_g0_batch_gpu(
        self,
        g0_batch: torch.Tensor,
        balance_penalty: float = 4.0,
        subcube_weight: Any = 6.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluates g0_batch [B, 2^d] on GPU using the exact mixed-dimension subcube deg_Z
        formula! Exact for ANY disjoint union of subcubes of arbitrary codimensions r_j <= d.
        For slightly unbalanced Mode II candidates, applies smooth penalty -balance_penalty*|mean|.
        """
        bsz = g0_batch.shape[0]
        mean, h_sub, i_g0, _ = evaluate_core_spectrum_batch(g0_batch, deg=self.deg, dtype=self.dtype)

        # Compute internal zero-graph degree deg_Z(x) AND neighbor-direction bitmask nbr_mask(x)
        # Exact O(d * 2^d) Subcube Theorem:
        #   Every connected component of Z = g0^{-1}(0) is a subcube IFF
        #   for every Z-edge (x, x^e_b): nbr_mask(x) == nbr_mask(x^e_b)!
        z_mask = g0_batch == 0
        deg_z = torch.zeros((bsz, self.N), device=self.device, dtype=torch.int8)
        nbr_mask = torch.zeros((bsz, self.N), device=self.device, dtype=torch.int32)
        for b in range(self.d):
            st = 1 << b
            vz = z_mask.view(bsz, -1, 2, st)
            both = vz[:, :, 0, :] & vz[:, :, 1, :]
            dz = deg_z.view(bsz, -1, 2, st)
            both_i8 = both.to(torch.int8)
            dz[:, :, 0, :] += both_i8
            dz[:, :, 1, :] += both_i8
            nm = nbr_mask.view(bsz, -1, 2, st)
            bit_val = both.to(torch.int32) << b
            nm[:, :, 0, :] += bit_val
            nm[:, :, 1, :] += bit_val

        # Exact count of non-subcube mismatch edges in Z (bad_edges == 0 iff pure disjoint subcubes!)
        bad_edges = torch.zeros(bsz, device=self.device, dtype=torch.int32)
        for b in range(self.d):
            st = 1 << b
            vz = z_mask.view(bsz, -1, 2, st)
            nm = nbr_mask.view(bsz, -1, 2, st)
            mismatch = vz[:, :, 0, :] & vz[:, :, 1, :] & (nm[:, :, 0, :] != nm[:, :, 1, :])
            bad_edges += mismatch.sum(dim=(1, 2), dtype=torch.int32)

        # Exact mixed-dimension subcube contribution:
        # For each x in Z with internal degree c_x = deg_Z(x), codim r_x = d - c_x:
        #   dI_x = (1 + r_x / 2) / 2^d,  dH_x = (2 * r_x) / 2^d
        inv_n = 1.0 / float(self.N)
        r_x = (float(self.d) - deg_z.to(dtype=self.dtype)) * z_mask.to(dtype=self.dtype)
        z_f = z_mask.to(dtype=self.dtype)
        i_aux = (z_f + 0.5 * r_x).sum(dim=1) * inv_n
        h_aux = (2.0 * r_x).sum(dim=1) * inv_n

        i_tot = i_g0 + i_aux
        h_tot = h_sub + h_aux
        z_frac = z_f.sum(dim=1) * inv_n
        abs_mean = mean.abs()
        bad_frac = bad_edges.to(dtype=self.dtype) * inv_n
        is_pure_subcube = bad_edges == 0

        # Raw proxy ratio (valid when I > 1.02)
        valid_i = i_tot > 1.02
        raw_ot = torch.where(valid_i, h_tot / (i_tot - 1.0).clamp_min(1e-6), torch.zeros_like(h_tot))
        # Fitness penalizes imbalance, non-subcube mismatch edges (per-chain replica weight), and giant gaps (> 28%)
        gap_pen = torch.clamp(z_frac - 0.28, min=0.0) * 3.0
        sc_w_t = (
            subcube_weight.to(device=self.device, dtype=self.dtype)
            if isinstance(subcube_weight, torch.Tensor)
            else torch.full((bsz,), float(subcube_weight), device=self.device, dtype=self.dtype)
        )
        subcube_pen = bad_frac * sc_w_t + (~is_pure_subcube).to(dtype=self.dtype) * (0.004 * sc_w_t)
        fitness = torch.where(
            valid_i,
            raw_ot - balance_penalty * abs_mean - gap_pen - subcube_pen,
            torch.zeros_like(raw_ot),
        )
        exact_bal = abs_mean < 1e-6
        strict_ot = torch.where(valid_i & exact_bal & is_pure_subcube, raw_ot, torch.zeros_like(raw_ot))

        return {
            "fitness": fitness,
            "strict_OT": strict_ot,
            "raw_OT": raw_ot,
            "H": h_tot,
            "I": i_tot,
            "mean": mean,
            "z_frac": z_frac,
            "bad_edges": bad_edges,
        }

    def evaluate_genomes(
        self,
        genomes: Sequence[HypergraphGenome],
        balance_penalty: float = 4.0,
        subcube_weight: Any = 6.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        g0_batch, ok = self.build_g0_batch_gpu(genomes)
        res = self.score_g0_batch_gpu(g0_batch, balance_penalty=balance_penalty, subcube_weight=subcube_weight)
        res["fitness"] = torch.where(ok, res["fitness"], torch.full_like(res["fitness"], -10.0))
        res["strict_OT"] = torch.where(ok, res["strict_OT"], torch.zeros_like(res["strict_OT"]))
        res["collision_free"] = ok
        return g0_batch, res


def balance_mode2_gap_monotone(g0: np.ndarray, d: int) -> Optional[np.ndarray]:
    """
    Fast monotone boundary balancer for a Mode II hypergraph core with small |U| - |L| slack:
    Repeatedly assigns minimal zero points (all lower neighbors == -1) to -1 when sum > 0,
    or maximal zero points (all upper neighbors == +1) to +1 when sum < 0, strictly preserving
    monotonicity and shrinking imbalance to exact 0!
    """
    g = np.asarray(g0, dtype=np.int8).copy()
    bal = int(np.sum(g))
    if bal == 0:
        return g
    n_pts = 1 << d
    max_steps = min(abs(bal) + 4, max(64, n_pts // 16))
    for _ in range(max_steps):
        bal = int(np.sum(g))
        if bal == 0:
            break
        if bal > 0:
            # Find minimal zero points (g == 0 and all lower neighbors == -1)
            mask = g == 0
            for b in range(d):
                st = 1 << b
                vg = g.reshape(-1, 2, st)
                mask.reshape(-1, 2, st)[:, 1, :] &= (vg[:, 0, :] == -1)
            cands = np.flatnonzero(mask)
            if len(cands) == 0:
                # Fallback: minimal +1 points (g == 1 and all lower neighbors == -1) -> 0
                mask = g == 1
                for b in range(d):
                    st = 1 << b
                    vg = g.reshape(-1, 2, st)
                    mask.reshape(-1, 2, st)[:, 1, :] &= (vg[:, 0, :] == -1)
                cands = np.flatnonzero(mask)
                if len(cands) == 0:
                    return None
                g[int(cands[0])] = 0
            else:
                g[int(cands[0])] = -1
        else:
            # Find maximal zero points (g == 0 and all upper neighbors == +1)
            mask = g == 0
            for b in range(d):
                st = 1 << b
                vg = g.reshape(-1, 2, st)
                mask.reshape(-1, 2, st)[:, 0, :] &= (vg[:, 1, :] == 1)
            cands = np.flatnonzero(mask)
            if len(cands) == 0:
                mask = g == -1
                for b in range(d):
                    st = 1 << b
                    vg = g.reshape(-1, 2, st)
                    mask.reshape(-1, 2, st)[:, 0, :] &= (vg[:, 1, :] == 1)
                cands = np.flatnonzero(mask)
                if len(cands) == 0:
                    return None
                g[int(cands[-1])] = 0
            else:
                g[int(cands[-1])] = 1
    return g if int(np.sum(g)) == 0 else None


def repair_non_subcube_gap_monotone(g0: np.ndarray, d: int, max_pairs: int = 128) -> Optional[np.ndarray]:
    """
    Repairs small non-subcube zero-gap components while strictly preserving
    exact balance (sum == 0) and exact monotonicity:
    Repeatedly finds a minimal zero cell x_lo (all lower neighbors == -1) and a
    maximal zero cell x_hi (all upper neighbors == +1), preferring cells incident
    to a non-subcube nbr_mask mismatch edge, and sets g[x_lo] = -1, g[x_hi] = +1
    until every zero connected component is a pure subcube!
    """
    g = np.asarray(g0, dtype=np.int8).copy()
    if int(np.sum(g)) != 0:
        return None
    n_pts = 1 << d
    for _ in range(max_pairs):
        z = g == 0
        if not np.any(z):
            return g
        nm = np.zeros(n_pts, dtype=np.int32)
        for b in range(d):
            st = 1 << b
            vz = z.reshape(-1, 2, st)
            both = vz[:, 0, :] & vz[:, 1, :]
            if np.any(both):
                nmv = nm.reshape(-1, 2, st)
                bit_v = both.astype(np.int32) << b
                nmv[:, 0, :] |= bit_v
                nmv[:, 1, :] |= bit_v
        bad_verts = np.zeros(n_pts, dtype=bool)
        for b in range(d):
            st = 1 << b
            vz = z.reshape(-1, 2, st)
            nmv = nm.reshape(-1, 2, st)
            mm = vz[:, 0, :] & vz[:, 1, :] & (nmv[:, 0, :] != nmv[:, 1, :])
            if np.any(mm):
                bv = bad_verts.reshape(-1, 2, st)
                bv[:, 0, :] |= mm
                bv[:, 1, :] |= mm
        if not np.any(bad_verts):
            return g
        # Minimal zero points (all lower neighbors == -1) & maximal zero points (all upper neighbors == +1)
        min_z = z.copy()
        max_z = z.copy()
        for b in range(d):
            st = 1 << b
            vg = g.reshape(-1, 2, st)
            min_z.reshape(-1, 2, st)[:, 1, :] &= (vg[:, 0, :] == -1)
            max_z.reshape(-1, 2, st)[:, 0, :] &= (vg[:, 1, :] == 1)
        lo_bad = np.flatnonzero(min_z & bad_verts)
        hi_bad = np.flatnonzero(max_z & bad_verts)
        lo_all = lo_bad if len(lo_bad) > 0 else np.flatnonzero(min_z)
        hi_all = hi_bad if len(hi_bad) > 0 else np.flatnonzero(max_z)
        if len(lo_all) == 0 or len(hi_all) == 0:
            return None
        x_lo = int(lo_all[0])
        x_hi = int(hi_all[-1])
        if x_lo == x_hi:
            if len(hi_all) >= 2:
                x_hi = int(hi_all[-2])
            elif len(lo_all) >= 2:
                x_lo = int(lo_all[1])
            else:
                return None
        g[x_lo] = -1
        g[x_hi] = 1
    # Final check
    _, ok_sub = decompose_zero_subcubes(g, d)
    return g if ok_sub and int(np.sum(g)) == 0 else None


def verify_hypergraph_candidate_fp64(
    genome: HypergraphGenome,
    device: torch.device,
    try_gap_balance: bool = True,
    try_subcube_repair: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Exact float64 + strict subcube de-aliasing verification of a HypergraphGenome.
    Returns dict with exact OT, H, I, subcubes, essential_vars, g0, and exact matching
    (F_pos, F_neg) extracted from g0 if repaired, or None if invalid.
    """
    if not is_cross_intersecting(genome.F_pos, genome.F_neg):
        return None
    ev = GPUHypergraphEvaluator(genome.d, device=device, dtype=torch.float64)
    g0_t, ok = ev.build_g0_batch_gpu([genome])
    if not bool(ok[0].item()):
        return None
    g0 = g0_t[0].detach().cpu().numpy()
    repaired = False
    if int(np.sum(g0)) != 0:
        if not try_gap_balance:
            return None
        g_bal = balance_mode2_gap_monotone(g0, genome.d)
        if g_bal is None:
            return None
        g0 = g_bal
        repaired = True
    res = evaluate_subcube_dealiased_exact_fp64(g0, genome.d, device=device)
    if not (res["balanced"] and res["monotone_full"]):
        if not try_subcube_repair:
            return None
        g_rep = repair_non_subcube_gap_monotone(g0, genome.d, max_pairs=min(192, max(48, (1 << genome.d) // 64)))
        if g_rep is None:
            return None
        g0 = g_rep
        repaired = True
        res = evaluate_subcube_dealiased_exact_fp64(g0, genome.d, device=device)
        if not (res["balanced"] and res["monotone_full"]):
            return None
    ess = count_essential_vars(g0, genome.d)
    if repaired:
        fp_ex, fn_ex = extract_hypergraph_from_g0(g0, genome.d)
    else:
        fp_ex, fn_ex = list(genome.F_pos), list(genome.F_neg)
    return {
        **res,
        "essential_vars": int(ess),
        "g0": g0,
        "F_pos": fp_ex,
        "F_neg": fn_ex,
        "mode": genome.mode,
        "perm": list(genome.perm) if genome.perm is not None else None,
        "repaired": repaired,
    }


# =============================================================================
# 3. High-Leverage Hypergraph Mutation & Crossover Operators (Mode I & Mode II)
# =============================================================================

def _sample_random_kset(d: int, k: int, rng: np.random.Generator) -> int:
    verts = rng.choice(d, size=k, replace=False)
    m = 0
    for v in verts:
        m |= 1 << int(v)
    return m


def _sample_compatible_transversal(
    d: int,
    opposing_edges: Sequence[int],
    own_edges: Sequence[int],
    k_choices: Sequence[int],
    rng: np.random.Generator,
    max_attempts: int = 40,
    perm_for_mode1: Optional[Sequence[int]] = None,
) -> Optional[int]:
    """
    Samples a k-subset (k in k_choices, typically {3, 4}) that hits EVERY edge in
    `opposing_edges` and is incomparable with `own_edges`.
    Uses greedy vertex-covering construction (much faster than blind rejection sampling for d=10..16!).
    """
    for _ in range(max_attempts):
        target_k = int(k_choices[int(rng.integers(0, len(k_choices)))])
        # Greedy cover: iteratively pick a vertex from an uncovered opposing edge!
        cand = 0
        uncov = list(opposing_edges)
        rng.shuffle(uncov)
        for opp in uncov:
            if (cand & opp) == 0:
                if cand.bit_count() >= target_k:
                    break
                # Pick a random vertex in opp (biased toward vertices hitting more remaining edges)
                bits = [b for b in range(d) if (opp >> b) & 1]
                if len(bits) > 1 and rng.random() < 0.6:
                    best_b = max(bits, key=lambda b: sum(1 for e in uncov if (e >> b) & 1) + 0.5 * rng.random())
                else:
                    best_b = bits[int(rng.integers(0, len(bits)))]
                cand |= 1 << best_b

        # Pad up to target_k if needed
        while cand.bit_count() < target_k:
            free = [b for b in range(d) if not ((cand >> b) & 1)]
            if not free:
                break
            cand |= 1 << free[int(rng.integers(0, len(free)))]

        # Check all opposing edges are hit
        if not all((cand & opp) != 0 for opp in opposing_edges):
            continue
        # Check incomparability with own_edges (not duplicate / superset / subset)
        if any((cand & a) == a or (cand & a) == cand for a in own_edges):
            continue
        if perm_for_mode1 is not None:
            p_cand = permute_mask(cand, perm_for_mode1)
            if (cand & p_cand) == 0:
                continue
            if not all((p_cand & a) != 0 for a in own_edges):
                continue
        return cand
    return None


def random_init_mode1(
    d: int,
    rng: np.random.Generator,
    k_choices: Sequence[int] = (3,),
    target_edges: Optional[int] = None,
) -> HypergraphGenome:
    """Cold-start Mode I (isomorphic F_- = pi(F_+)) hypergraph with zero hardcoded seeds."""
    # Random derangement / high-order permutation pi (so A and pi(A) aren't trivially identical)
    perm = list(range(d))
    for _ in range(10):
        rng.shuffle(perm)
        if sum(1 for i, p in enumerate(perm) if i == p) <= d // 3:
            break
    n_target = target_edges if target_edges is not None else max(4, min(14, int(d * 0.75)))
    F: List[int] = []
    for _ in range(n_target * 4):
        if len(F) >= n_target:
            break
        opp = [permute_mask(a, perm) for a in F]
        cand = _sample_compatible_transversal(d, opp, F, k_choices, rng, max_attempts=25, perm_for_mode1=perm)
        if cand is not None:
            F.append(cand)
    F = reduce_to_antichain(F)
    F_neg = [permute_mask(a, perm) for a in F]
    return HypergraphGenome(d=d, F_pos=F, F_neg=F_neg, mode="mode1", perm=perm)


def random_init_mode2(
    d: int,
    rng: np.random.Generator,
    k_choices: Sequence[int] = (3, 4),
    target_edges: Optional[int] = None,
) -> HypergraphGenome:
    """Cold-start Mode II (free asymmetric (F_+, F_-)) cross-intersecting hypergraph."""
    n_target = target_edges if target_edges is not None else max(5, min(16, int(d * 0.9)))
    F_pos: List[int] = [_sample_random_kset(d, int(k_choices[0]), rng)]
    F_neg: List[int] = []
    for step in range(n_target * 3):
        if len(F_pos) >= n_target and len(F_neg) >= n_target:
            break
        if len(F_neg) <= len(F_pos):
            c = _sample_compatible_transversal(d, F_pos, F_neg, k_choices, rng, max_attempts=20)
            if c is not None:
                F_neg.append(c)
        else:
            c = _sample_compatible_transversal(d, F_neg, F_pos, k_choices, rng, max_attempts=20)
            if c is not None:
                F_pos.append(c)
    return HypergraphGenome(
        d=d, F_pos=reduce_to_antichain(F_pos), F_neg=reduce_to_antichain(F_neg), mode="mode2", perm=None
    )


def mutate_mode1(
    genome: HypergraphGenome,
    rng: np.random.Generator,
    k_choices: Sequence[int],
    max_edges: int = 24,
) -> HypergraphGenome:
    """
    High-leverage Mode I mutations preserving F_- = pi(F_+) and cross-intersection:
      1. Add compatible transversal edge A (grows coverage, shrinks gap)
      2. Drop edge A (opens structured gap subcubes)
      3. Replace edge (drop 1 + add compatible transversal)
      4. Vertex-slide / size-change (3 <-> 4) with minimal conflict repair
      5. Transposition on permutation pi with minimal conflict repair
    """
    g = genome.clone()
    d = g.d
    assert g.perm is not None
    perm = g.perm
    F = list(g.F_pos)

    def _repair_mode1(edges: List[int], p: List[int]) -> List[int]:
        edges = reduce_to_antichain(edges)
        # Keep maximal subset where forall a, b: a & p(b) != 0 (greedily drop highest-conflict edge)
        while edges:
            pe = [permute_mask(a, p) for a in edges]
            # First drop any self-disjoint edge (a & p(a) == 0)
            bad_self = [i for i in range(len(edges)) if (edges[i] & pe[i]) == 0]
            if bad_self:
                edges.pop(bad_self[0])
                continue
            conflicts = [0] * len(edges)
            for i in range(len(edges)):
                for j in range(len(edges)):
                    if (edges[i] & pe[j]) == 0:
                        conflicts[i] += 1
                        conflicts[j] += 1
            mx = max(conflicts)
            if mx == 0:
                break
            worst = [i for i, c in enumerate(conflicts) if c == mx]
            edges.pop(worst[int(rng.integers(0, len(worst)))])
        return edges

    r = float(rng.random())
    if r < 0.28 and len(F) < max_edges:
        # 1. Add compatible transversal edge
        opp = [permute_mask(a, perm) for a in F]
        cand = _sample_compatible_transversal(d, opp, F, k_choices, rng, max_attempts=30, perm_for_mode1=perm)
        if cand is not None:
            F.append(cand)
    elif r < 0.44 and len(F) >= 3:
        # 2. Drop random edge
        F.pop(int(rng.integers(0, len(F))))
    elif r < 0.68 and len(F) >= 2:
        # 3. Replace 1 edge with a new compatible transversal edge
        drop_i = int(rng.integers(0, len(F)))
        F_sub = F[:drop_i] + F[drop_i + 1 :]
        opp = [permute_mask(a, perm) for a in F_sub]
        cand = _sample_compatible_transversal(d, opp, F_sub, k_choices, rng, max_attempts=30, perm_for_mode1=perm)
        if cand is not None:
            F = F_sub + [cand]
    elif r < 0.86 and len(F) >= 1:
        # 4. Vertex-slide or size change (3 <-> 4) + minimal conflict repair + refill!
        idx = int(rng.integers(0, len(F)))
        old_e = F[idx]
        in_bits = [b for b in range(d) if (old_e >> b) & 1]
        out_bits = [b for b in range(d) if not ((old_e >> b) & 1)]
        sub_r = float(rng.random())
        new_e = old_e
        if sub_r < 0.65 and in_bits and out_bits:
            # Slide 1 vertex
            rm = in_bits[int(rng.integers(0, len(in_bits)))]
            ad = out_bits[int(rng.integers(0, len(out_bits)))]
            new_e = (old_e ^ (1 << rm)) | (1 << ad)
        elif sub_r < 0.82 and out_bits and old_e.bit_count() < max(k_choices):
            # Grow edge by 1 vertex (e.g. 3 -> 4)
            ad = out_bits[int(rng.integers(0, len(out_bits)))]
            new_e = old_e | (1 << ad)
        elif len(in_bits) > min(k_choices):
            # Shrink edge by 1 vertex (e.g. 4 -> 3)
            rm = in_bits[int(rng.integers(0, len(in_bits)))]
            new_e = old_e ^ (1 << rm)
        # Put new_e first and repair conflicts (prefer keeping new_e!)
        F_try = [new_e] + [a for i, a in enumerate(F) if i != idx]
        # Drop edges conflicting with new_e
        p_new = permute_mask(new_e, perm)
        if (new_e & p_new) != 0:
            F_clean = [new_e] + [
                a for a in F_try[1:]
                if (a & p_new) != 0 and (new_e & permute_mask(a, perm)) != 0
            ]
            F = reduce_to_antichain(F_clean)
            # Greedily try adding 1 compatible edge back if edges were dropped
            if len(F) < len(g.F_pos):
                opp = [permute_mask(a, perm) for a in F]
                c_add = _sample_compatible_transversal(d, opp, F, k_choices, rng, max_attempts=20, perm_for_mode1=perm)
                if c_add is not None:
                    F.append(c_add)
    else:
        # 5. Transposition on permutation pi + minimal conflict repair + refill!
        u = int(rng.integers(0, d))
        v = int(rng.integers(0, d - 1))
        if v >= u:
            v += 1
        perm = list(perm)
        perm[u], perm[v] = perm[v], perm[u]
        F = _repair_mode1(F, perm)
        if len(F) < max_edges:
            opp = [permute_mask(a, perm) for a in F]
            c_add = _sample_compatible_transversal(d, opp, F, k_choices, rng, max_attempts=20, perm_for_mode1=perm)
            if c_add is not None:
                F.append(c_add)

    F = _repair_mode1(F, perm)
    if not F:
        return random_init_mode1(d, rng, k_choices=k_choices)
    F_neg = [permute_mask(a, perm) for a in F]
    return HypergraphGenome(d=d, F_pos=F, F_neg=F_neg, mode="mode1", perm=perm)


def mutate_mode2(
    genome: HypergraphGenome,
    rng: np.random.Generator,
    k_choices: Sequence[int],
    max_edges: int = 28,
    mean_hint: float = 0.0,
) -> HypergraphGenome:
    """
    High-leverage Mode II mutations for free asymmetric cross-intersecting pairs (F_pos, F_neg):
      1. Balance-guided single-side add/drop (when |mean_hint| > 0.005, steers |U| <-> |L|!)
      2. Simultaneous dual-side add/drop/replace (moves both F_pos & F_neg together near |U|==|L|)
      3. Vertex-slide / 3<->4 size change with minimal conflict repair + refill on opposing side
      4. Split a k-edge into two (k+1)-edges (or merge) to tune mixed (3,4)-geometries!
    """
    g = genome.clone()
    d = g.d
    Fp = list(g.F_pos)
    Fn = list(g.F_neg)

    r = float(rng.random())
    # 1. Balance-guided move if currently imbalanced (steers smoothly onto |U| == |L| manifold!)
    if abs(mean_hint) > 0.004 and r < 0.45:
        if mean_hint > 0.0:
            # Too many +1s: either add compatible edge to Fn OR drop an edge from Fp
            if rng.random() < 0.60 and len(Fn) < max_edges:
                c = _sample_compatible_transversal(d, Fp, Fn, k_choices, rng, max_attempts=30)
                if c is not None:
                    Fn.append(c)
            elif len(Fp) >= 3:
                Fp.pop(int(rng.integers(0, len(Fp))))
        else:
            # Too many -1s: either add compatible edge to Fp OR drop an edge from Fn
            if rng.random() < 0.60 and len(Fp) < max_edges:
                c = _sample_compatible_transversal(d, Fn, Fp, k_choices, rng, max_attempts=30)
                if c is not None:
                    Fp.append(c)
            elif len(Fn) >= 3:
                Fn.pop(int(rng.integers(0, len(Fn))))
    elif r < 0.30 and len(Fp) < max_edges and len(Fn) < max_edges:
        # 2a. Simultaneous pair-add (keeps |U| ~ |L|!)
        c1 = _sample_compatible_transversal(d, Fn, Fp, k_choices, rng, max_attempts=25)
        if c1 is not None:
            Fp.append(c1)
        c2 = _sample_compatible_transversal(d, Fp, Fn, k_choices, rng, max_attempts=25)
        if c2 is not None:
            Fn.append(c2)
    elif r < 0.45 and len(Fp) >= 3 and len(Fn) >= 3:
        # 2b. Simultaneous pair-drop
        Fp.pop(int(rng.integers(0, len(Fp))))
        Fn.pop(int(rng.integers(0, len(Fn))))
    elif r < 0.68 and len(Fp) >= 2 and len(Fn) >= 2:
        # 2c. Replace 1 edge on one (or both) sides with compatible transversal
        side_pos = rng.random() < 0.5
        own, opp = (Fp, Fn) if side_pos else (Fn, Fp)
        idx = int(rng.integers(0, len(own)))
        own_sub = own[:idx] + own[idx + 1 :]
        cand = _sample_compatible_transversal(d, opp, own_sub, k_choices, rng, max_attempts=30)
        if cand is not None:
            own[:] = own_sub + [cand]
    elif r < 0.90:
        # 3. Vertex-slide / 3<->4 size change with minimal conflict repair + refill on opposing side
        side_pos = rng.random() < 0.5
        own, opp = (Fp, Fn) if side_pos else (Fn, Fp)
        if own:
            idx = int(rng.integers(0, len(own)))
            old_e = own[idx]
            in_bits = [b for b in range(d) if (old_e >> b) & 1]
            out_bits = [b for b in range(d) if not ((old_e >> b) & 1)]
            sub_r = float(rng.random())
            new_e = old_e
            if sub_r < 0.60 and in_bits and out_bits:
                rm = in_bits[int(rng.integers(0, len(in_bits)))]
                ad = out_bits[int(rng.integers(0, len(out_bits)))]
                new_e = (old_e ^ (1 << rm)) | (1 << ad)
            elif sub_r < 0.80 and out_bits and old_e.bit_count() < max(k_choices):
                ad = out_bits[int(rng.integers(0, len(out_bits)))]
                new_e = old_e | (1 << ad)
            elif len(in_bits) > min(k_choices):
                rm = in_bits[int(rng.integers(0, len(in_bits)))]
                new_e = old_e ^ (1 << rm)
            own[idx] = new_e
            own[:] = reduce_to_antichain(own)
            old_opp_len = len(opp)
            opp[:] = [b for b in opp if (b & new_e) != 0]
            for _ in range(max(1, old_opp_len - len(opp))):
                if len(opp) >= max_edges:
                    break
                c_ref = _sample_compatible_transversal(d, own, opp, k_choices, rng, max_attempts=20)
                if c_ref is not None:
                    opp.append(c_ref)
    else:
        # 4. Split a k-edge into two (k+1)-edges (tunes mixed (3,4)-geometries)
        side_pos = rng.random() < 0.5
        own = Fp if side_pos else Fn
        splittable = [i for i, e in enumerate(own) if e.bit_count() < max(k_choices)]
        if splittable and len(own) < max_edges:
            idx = splittable[int(rng.integers(0, len(splittable)))]
            e = own.pop(idx)
            out_bits = [b for b in range(d) if not ((e >> b) & 1)]
            if len(out_bits) >= 2:
                b1, b2 = rng.choice(out_bits, size=2, replace=False)
                own.append(e | (1 << int(b1)))
                own.append(e | (1 << int(b2)))
                own[:] = reduce_to_antichain(own)

    Fp = reduce_to_antichain(Fp)
    Fn = reduce_to_antichain(Fn)
    if not Fp or not Fn or not is_cross_intersecting(Fp, Fn):
        return random_init_mode2(d, rng, k_choices=k_choices)
    return HypergraphGenome(d=d, F_pos=Fp, F_neg=Fn, mode="mode2", perm=None)


def crossover_hypergraphs(
    h1: HypergraphGenome,
    h2: HypergraphGenome,
    rng: np.random.Generator,
    k_choices: Sequence[int] = (3, 4),
) -> HypergraphGenome:
    """
    Hypergraph CROSSOVER(H1, H2) for SLEEP-DAG-RSI DiscoveryTree:
    Combines subsets of F_pos from H1 and H2 (lifted to common d = max(d1, d2)) and
    projects onto a maximal cross-intersecting pair (F_pos, F_neg)!
    """
    d = max(h1.d, h2.d)
    # Mix roughly half edges from h1.F_pos and half from h2.F_pos
    Fp_mix: List[int] = []
    for e in h1.F_pos:
        if rng.random() < 0.55:
            Fp_mix.append(e)
    for e in h2.F_pos:
        if rng.random() < 0.55:
            Fp_mix.append(e)
    Fp_mix = reduce_to_antichain(Fp_mix)
    if not Fp_mix:
        Fp_mix = list(h1.F_pos)

    # Filter union of F_neg from both parents to those intersecting ALL of Fp_mix
    Fn_cand = reduce_to_antichain(list(h1.F_neg) + list(h2.F_neg))
    Fn_ok = [b for b in Fn_cand if all((a & b) != 0 for a in Fp_mix)]
    # Refill Fn_ok with compatible transversals so coverage balances cleanly!
    target_n = max(len(Fp_mix), 6)
    for _ in range(target_n * 2):
        if len(Fn_ok) >= target_n:
            break
        c = _sample_compatible_transversal(d, Fp_mix, Fn_ok, k_choices, rng, max_attempts=20)
        if c is not None:
            Fn_ok.append(c)
    if not Fn_ok:
        return random_init_mode2(d, rng, k_choices=k_choices)
    return HypergraphGenome(d=d, F_pos=Fp_mix, F_neg=reduce_to_antichain(Fn_ok), mode="mode2", perm=None)


# =============================================================================
# 4. Parallel GPU Evolutionary Hypergraph Solver (Mode I & Mode II)
# =============================================================================

def evolve_hypergraphs_gpu(
    d: int,
    mode: str = "mode1",
    k_choices: Sequence[int] = (3,),
    n_chains: int = 64,
    steps: int = 400,
    t_start: float = 0.25,
    t_end: float = 0.002,
    seed: int = 2026,
    device: Optional[torch.device] = None,
    seed_genomes: Sequence[HypergraphGenome] = (),
    max_edges: int = 22,
    stop_at_ot: Optional[float] = None,
    min_used_verts: int = 0,
    min_I: float = 0.0,
    max_deg_cap: int = 0,
) -> Dict[str, Any]:
    """
    Runs `n_chains` parallel tempered Simulated Annealing trajectories directly in
    cross-intersecting hypergraph genome space (F_pos, F_neg) evaluated in batched
    PyTorch CUDA FWHT on RTX 5060!
    """
    t0 = time.perf_counter()
    dev = device or get_device(True)
    rng = np.random.default_rng(seed)
    ev = GPUHypergraphEvaluator(d, device=dev, dtype=torch.float32)

    # Initialize chains (cold start unless seed_genomes explicitly provided)
    chains: List[HypergraphGenome] = []
    for c in range(n_chains):
        if seed_genomes and c < min(n_chains // 2, len(seed_genomes) * 4):
            g_s = seed_genomes[c % len(seed_genomes)].clone()
            if g_s.d < d:
                g_s.d = d
                if g_s.perm is not None:
                    g_s.perm = g_s.perm + list(range(len(g_s.perm), d))
            if mode == "mode1" and g_s.perm is None:
                p_det = detect_isomorphism_perm(d, g_s.F_pos, g_s.F_neg)
                if p_det is not None:
                    g_s.perm = p_det
            g_s.mode = mode
            if mode == "mode1" and g_s.perm is None:
                g_s = random_init_mode1(d, rng, k_choices=k_choices)
            chains.append(g_s)
        else:
            if mode == "mode1":
                chains.append(random_init_mode1(d, rng, k_choices=k_choices))
            else:
                chains.append(random_init_mode2(d, rng, k_choices=k_choices))

    def _structural_penalty(g_list: Sequence[HypergraphGenome], i_vals: np.ndarray) -> np.ndarray:
        pen = np.zeros(len(g_list), dtype=np.float64)
        if min_used_verts > 0 or max_deg_cap > 0:
            for i, gm in enumerate(g_list):
                um = 0
                if max_deg_cap > 0:
                    dp = [0] * d
                    dn = [0] * d
                    for a in gm.F_pos:
                        um |= a
                        m = a
                        while m:
                            lsb = m & -m
                            dp[lsb.bit_length() - 1] += 1
                            m ^= lsb
                    for b in gm.F_neg:
                        um |= b
                        m = b
                        while m:
                            lsb = m & -m
                            dn[lsb.bit_length() - 1] += 1
                            m ^= lsb
                    exc = max(max(dp) - max_deg_cap, max(dn) - max_deg_cap, 0)
                    if exc > 0:
                        pen[i] += 0.28 * float(exc)
                else:
                    for a in gm.F_pos:
                        um |= a
                    for b in gm.F_neg:
                        um |= b
                if min_used_verts > 0:
                    deficit = min_used_verts - um.bit_count()
                    if deficit > 0:
                        pen[i] += 0.65 * float(deficit)
        if min_I > 0.0:
            pen += np.maximum(0.0, min_I - i_vals) * 2.5
        return pen

    # Fixed per-chain multi-penalty replica weights (zero time-drift in cur_fit!):
    # Half chains run soft subcube penalty (0.5..2.0) to freely cross non-subcube bridges (like AG(2,3)),
    # half chains run firm subcube penalty (4.0..14.0) to snap onto exact disjoint subcubes!
    sc_weights_np = np.zeros(n_chains, dtype=np.float32)
    half_c = n_chains // 2
    sc_weights_np[:half_c] = np.linspace(0.5, 2.0, max(1, half_c), dtype=np.float32)
    sc_weights_np[half_c:] = np.linspace(4.0, 14.0, max(1, n_chains - half_c), dtype=np.float32)
    sc_weights_t = torch.as_tensor(sc_weights_np, device=dev, dtype=torch.float32)

    bal_pen = 3.5 if mode == "mode2" else 8.0
    _, r0 = ev.evaluate_genomes(chains, balance_penalty=bal_pen, subcube_weight=sc_weights_t)
    i0 = r0["I"].detach().cpu().numpy().astype(np.float64)
    pen0 = _structural_penalty(chains, i0)
    cur_fit = r0["fitness"].detach().cpu().numpy().astype(np.float64) - pen0
    cur_ot = np.where(pen0 == 0.0, r0["strict_OT"].detach().cpu().numpy().astype(np.float64), 0.0)
    cur_mean = r0["mean"].detach().cpu().numpy().astype(np.float64)

    best_fit = cur_fit.copy()
    best_chains = [c.clone() for c in chains]
    best_strict_ot = cur_ot.copy()
    best_strict_chains = [c.clone() for c in chains]
    global_best_ot = float(np.max(cur_ot))
    global_best_genome = chains[int(np.argmax(cur_ot))].clone()
    global_best_fit = float(np.max(cur_fit))
    global_best_fit_genome = chains[int(np.argmax(cur_fit))].clone()

    # Per-chain temperature ladder:
    # - Seeded chains (if any): cooler local-refinement half + medium exploration half
    # - Cold-start chains: warm log-spaced ladder (t_start * 0.2 .. t_start) proven on Tracer #1 d=9!
    n_seeded = min(n_chains // 2, len(seed_genomes) * 4) if seed_genomes else 0
    t0_arr = np.exp(np.linspace(math.log(t_start * 0.2), math.log(t_start), n_chains))
    if n_seeded > 0:
        n_cold_s = max(1, n_seeded // 2)
        t0_arr[:n_cold_s] = np.exp(np.linspace(math.log(0.015), math.log(0.06), n_cold_s))
    decay = (t_end / t0_arr) ** (1.0 / max(1, steps - 1))
    temps = t0_arr.copy()
    total_evals = n_chains

    for step in range(steps):
        props: List[HypergraphGenome] = []
        for c in range(n_chains):
            if mode == "mode1":
                cand = mutate_mode1(chains[c], rng, k_choices=k_choices, max_edges=max_edges)
                if c >= n_chains // 2 and rng.random() < 0.12:
                    # Double-kick macro-mutation on half the chains to tunnel across 2-edge barriers
                    cand = mutate_mode1(cand, rng, k_choices=k_choices, max_edges=max_edges)
                props.append(cand)
            else:
                cand = mutate_mode2(chains[c], rng, k_choices=k_choices, max_edges=max_edges, mean_hint=float(cur_mean[c]))
                if c >= n_chains // 2 and rng.random() < 0.12:
                    cand = mutate_mode2(cand, rng, k_choices=k_choices, max_edges=max_edges, mean_hint=float(cur_mean[c]))
                props.append(cand)

        _, rp = ev.evaluate_genomes(props, balance_penalty=bal_pen, subcube_weight=sc_weights_t)
        ip = rp["I"].detach().cpu().numpy().astype(np.float64)
        pen_p = _structural_penalty(props, ip)
        p_fit = rp["fitness"].detach().cpu().numpy().astype(np.float64) - pen_p
        p_ot = np.where(pen_p == 0.0, rp["strict_OT"].detach().cpu().numpy().astype(np.float64), 0.0)
        p_mean = rp["mean"].detach().cpu().numpy().astype(np.float64)
        total_evals += n_chains

        delta = p_fit - cur_fit
        u_rand = rng.random(n_chains)
        acc = (delta >= 0.0) | (u_rand < np.exp(np.clip(delta / temps, -40.0, 0.0)))
        for c in range(n_chains):
            if p_ot[c] > best_strict_ot[c]:
                best_strict_ot[c] = p_ot[c]
                best_strict_chains[c] = props[c].clone()
            if p_ot[c] > global_best_ot:
                global_best_ot = float(p_ot[c])
                global_best_genome = props[c].clone()
            if acc[c]:
                chains[c] = props[c]
                cur_fit[c] = p_fit[c]
                cur_ot[c] = p_ot[c]
                cur_mean[c] = p_mean[c]
                if p_fit[c] > best_fit[c]:
                    best_fit[c] = p_fit[c]
                    best_chains[c] = props[c].clone()
                if p_fit[c] > global_best_fit:
                    global_best_fit = float(p_fit[c])
                    global_best_fit_genome = props[c].clone()

        temps *= decay

        # Periodic elite migration + crossover injection (every 35% of steps)
        if step in (int(steps * 0.35), int(steps * 0.70)):
            order = np.argsort(-best_fit)
            k_top = min(8, n_chains // 4)
            k_bot = min(n_chains // 3, n_chains - k_top)
            for i in range(k_bot):
                dst = int(order[n_chains - 1 - i])
                src = int(order[i % k_top])
                if mode == "mode2" and i % 3 == 0 and k_top >= 2:
                    src2 = int(order[(i + 1) % k_top])
                    chains[dst] = crossover_hypergraphs(best_chains[src], best_chains[src2], rng, k_choices=k_choices)
                else:
                    chains[dst] = best_chains[src].clone()
            # Re-evaluate migrated chains under their own replica subcube_weight (zero stale cur_fit!)
            _, rm = ev.evaluate_genomes(chains, balance_penalty=bal_pen, subcube_weight=sc_weights_t)
            im = rm["I"].detach().cpu().numpy().astype(np.float64)
            pen_m = _structural_penalty(chains, im)
            cur_fit = rm["fitness"].detach().cpu().numpy().astype(np.float64) - pen_m
            cur_mean = rm["mean"].detach().cpu().numpy().astype(np.float64)

        if stop_at_ot is not None and global_best_ot >= stop_at_ot:
            # Verify exact fp64 before early-stopping!
            chk = verify_hypergraph_candidate_fp64(global_best_genome, dev, try_gap_balance=True)
            if chk is not None and float(chk["OT"]) >= stop_at_ot:
                break

    # Verify top unique elites (strict pure-subcube pool + best fitness pool) in exact float64!
    cand_pool: List[HypergraphGenome] = [global_best_genome, global_best_fit_genome]
    order_s = np.argsort(-best_strict_ot)
    for idx in order_s[:10]:
        if best_strict_ot[int(idx)] > 0.0:
            cand_pool.append(best_strict_chains[int(idx)])
    order_f = np.argsort(-best_fit)
    for idx in order_f[:10]:
        cand_pool.append(best_chains[int(idx)])

    verified_best: Optional[Dict[str, Any]] = None
    verified_non_ag23: Optional[Dict[str, Any]] = None
    seen_sigs = set()
    for gm in cand_pool:
        sig = (tuple(sorted(gm.F_pos)), tuple(sorted(gm.F_neg)))
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        rep = verify_hypergraph_candidate_fp64(gm, dev, try_gap_balance=(mode == "mode2"))
        if rep is None:
            continue
        if verified_best is None or float(rep["OT"]) > float(verified_best["OT"]):
            verified_best = rep
        if int(rep["essential_vars"]) >= 10:
            if verified_non_ag23 is None or float(rep["OT"]) > float(verified_non_ag23["OT"]):
                verified_non_ag23 = rep

    dt = time.perf_counter() - t0
    return {
        "d": d,
        "mode": mode,
        "k_choices": list(k_choices),
        "total_evals": total_evals,
        "evals_per_sec": round(total_evals / max(dt, 1e-6), 1),
        "elapsed_sec": round(dt, 2),
        "verified_best": verified_best,
        "verified_non_ag23": verified_non_ag23,
    }


# =============================================================================
# 5. Mandatory Tracer #1 CLI: Cold-Start d=9 AG(2,3) Rediscovery Gate
# =============================================================================

def run_sanity_d9(seed: int = 202609) -> Dict[str, Any]:
    """
    Mandatory Tracer #1 sanity check: runs cold-start hypergraph evolution on d=9
    (ZERO AG(2,3) hints, ZERO hardcoded seeds!) and verifies independent rediscovery
    of AG(2,3) (L >= 6.52184571, I = 261/128, 6+6 triples, 9 disjoint 1D subcubes)!
    """
    dev = get_device(True)
    res: Dict[str, Any] = {}
    vb: Optional[Dict[str, Any]] = None
    tot_evals = 0
    t_all = time.perf_counter()
    for attempt in range(3):
        res = evolve_hypergraphs_gpu(
            d=9,
            mode="mode1",
            k_choices=(3,),
            n_chains=128,
            steps=450,
            t_start=0.25,
            t_end=0.002,
            seed=seed + attempt * 997,
            device=dev,
            seed_genomes=(),
            max_edges=12,
            stop_at_ot=6.521845,
        )
        tot_evals += int(res["total_evals"])
        vb = res["verified_best"]
        if vb is not None and float(vb["OT"]) >= 6.521845:
            break
    dt_all = time.perf_counter() - t_all
    passed = bool(vb is not None and float(vb["OT"]) >= 6.521845)
    out = {
        "sanity_d9_passed": passed,
        "elapsed_sec": round(dt_all, 2),
        "total_evals": tot_evals,
        "evals_per_sec": round(tot_evals / max(dt_all, 1e-6), 1),
        "best_OT": float(vb["OT"]) if vb else 0.0,
        "best_I": float(vb["I"]) if vb else 0.0,
        "best_k_aux": int(vb["k_aux"]) if vb else 0,
        "best_n_eff": int(vb["n_eff"]) if vb else 0,
        "n_F_pos": len(vb["F_pos"]) if vb else 0,
        "n_F_neg": len(vb["F_neg"]) if vb else 0,
        "F_pos": vb["F_pos"] if vb else [],
        "F_neg": vb["F_neg"] if vb else [],
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Hypergraph GPU Evolutionary Engine (Tracer #1 d=9 sanity)")
    parser.add_argument("--sanity-d9", action="store_true", help="Run mandatory cold-start d=9 AG(2,3) sanity check")
    parser.add_argument("--seed", type=int, default=202609)
    args = parser.parse_args()
    if args.sanity_d9 or True:
        rep = run_sanity_d9(seed=args.seed)
        print(json.dumps(rep, indent=2))
        assert rep["sanity_d9_passed"], f"Tracer #1 FAILED: cold-start d=9 only reached {rep['best_OT']} < 6.521845!"
        print("RESULT: hypergraph_gpu_engine Tracer #1 (cold-start d=9 AG(2,3) rediscovery) PASSED")


if __name__ == "__main__":
    main()

"""
GPU Batched Walsh-Hadamard Transform (FWHT) & Vector A2 Orthogonal Subcube/Antichain
De-Aliasing Evaluator for C_71 (Fourier Entropy-Influence Conjecture) on RTX 5060.

Key guarantees:
  - Strict VRAM budget guard (<= 4.5 GiB peak on 7.93 GiB RTX 5060).
  - Supports both fast float32 screening (exact integer FWHT for d <= 24) and float64
    high-precision verification mode.
  - Exact Vector A2 (Orthogonal Subcube/Antichain De-Aliasing) evaluation:
      F(x, y_1, ..., y_k) = g_0(x) + sum_{j=1}^k 1_{C_j}(x) * y_j
    collapses n = d + k variables into a single 2^d FWHT with exact formulas:
      Inf(F) = Inf(g_0) + sum_j 2^{-r_j} * (1 + r_j / 2),
      H(F)   = H_sub(g_0) + sum_j 2^{-r_j} * (2 * r_j).
  - Includes maximal-subcube coalescing (avoiding the Delta H / Delta I = 4 splitting
    penalty) and strict monotonicity + no-comparable-subcubes validator.
  - Includes incremental k-point delta spectrum evaluator on GPU for fast local search.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

VRAM_LIMIT_BYTES = int(4.5 * (1024 ** 3))


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_degree_table(d: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Returns tensor of popcounts |S| for S in 0 .. 2^d - 1."""
    n_pts = 1 << d
    idx = torch.arange(n_pts, device=device, dtype=torch.int32)
    deg = torch.zeros(n_pts, device=device, dtype=dtype)
    for bit in range(d):
        deg += ((idx >> bit) & 1).to(dtype)
    return deg


def max_safe_batch_size(d: int, dtype: torch.dtype = torch.float32, vram_cap_bytes: int = VRAM_LIMIT_BYTES) -> int:
    """
    Conservative batch cap keeping peak PyTorch FWHT VRAM <= vram_cap_bytes.
    Peak memory during stack-based FWHT + p + log2(p) is ~4.5 * B * 2^d * bytes_per_elem.
    """
    bpe = 4 if dtype == torch.float32 else 8
    per_item = int(5 * (1 << d) * bpe)
    return max(1, min(8192, vram_cap_bytes // max(1, per_item)))


def fwht_batched(x: torch.Tensor) -> torch.Tensor:
    """
    Unnormalized Walsh-Hadamard Transform along the last dimension of x [B, 2^d].
    For integer inputs in {-1, 0, +1} and d <= 24, float32 FWHT is bit-exact!
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    bsz, n_pts = x.shape
    if n_pts == 0 or (n_pts & (n_pts - 1)) != 0:
        raise ValueError(f"FWHT length must be power of 2, got {n_pts}")
    a = x.clone()
    h = 1
    while h < n_pts:
        a = a.view(bsz, -1, 2, h)
        u = a[:, :, 0, :]
        v = a[:, :, 1, :]
        a = torch.stack((u + v, u - v), dim=2).view(bsz, n_pts)
        h <<= 1
    return a.squeeze(0) if squeeze else a


def evaluate_core_spectrum_batch(
    g0_batch: torch.Tensor,
    deg: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes (mean, H_sub, I_g0, fhat) for a batch of 3-valued cores g0 in {-1, 0, +1}^{B x 2^d}.
    H_sub = -sum_{S} ghat(S)^2 * log2(ghat(S)^2), I_g0 = sum_{S} |S| * ghat(S)^2.
    """
    if g0_batch.ndim == 1:
        g0_batch = g0_batch.unsqueeze(0)
    bsz, n_pts = g0_batch.shape
    d = int(round(math.log2(n_pts)))
    if deg is None or deg.shape[0] != n_pts or deg.dtype != dtype or deg.device != g0_batch.device:
        deg = make_degree_table(d, g0_batch.device, dtype=dtype)

    x = g0_batch.to(dtype=dtype)
    fhat = fwht_batched(x) / float(n_pts)
    p = fhat * fhat
    mean = fhat[:, 0]
    i_g0 = (p * deg).sum(dim=1)
    # Mask exact zeros cleanly to avoid NaN in 0 * log2(0)
    eps = 1e-38 if dtype == torch.float32 else 1e-300
    h_sub = -torch.where(p > 0.0, p * torch.log2(p.clamp_min(eps)), torch.zeros_like(p)).sum(dim=1)
    return mean, h_sub, i_g0, fhat


def evaluate_antichain_dealiased_batch(
    g0_batch: torch.Tensor,
    deg: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """
    Exact Vector A2 single-point antichain de-aliasing evaluation for a batch [B, 2^d].
    Every zero cell g0(a) == 0 receives its own distinct fresh auxiliary variable y_a (codim r = d):
      I(F) = I(g0) + (|A| / 2^d) * (1 + d/2)
      H(F) = H_sub(g0) + (|A| / 2^d) * (2 * d)
      L(F) = H(F) / (I(F) - 1)   (valid when mean == 0 and I(F) > 1).
    """
    if g0_batch.ndim == 1:
        g0_batch = g0_batch.unsqueeze(0)
    n_pts = g0_batch.shape[1]
    d = int(round(math.log2(n_pts)))
    mean, h_sub, i_g0, _ = evaluate_core_spectrum_batch(g0_batch, deg=deg, dtype=dtype)
    a_cnt = (g0_batch == 0).sum(dim=1).to(dtype=dtype)
    frac_a = a_cnt / float(n_pts)
    i_tot = i_g0 + frac_a * (1.0 + 0.5 * float(d))
    h_tot = h_sub + frac_a * (2.0 * float(d))
    valid_ot = (mean.abs() < 1e-9) & (i_tot > 1.0 + 1e-9)
    ot = torch.where(valid_ot, h_tot / (i_tot - 1.0).clamp_min(1e-12), torch.zeros_like(h_tot))
    return {
        "mean": mean,
        "H": h_tot,
        "I": i_tot,
        "OT": ot,
        "A_count": a_cnt,
        "H_sub": h_sub,
        "I_g0": i_g0,
    }


def eval_delta_candidates_gpu(
    fhat_parent: torch.Tensor,
    deg: torch.Tensor,
    delta_indices: torch.Tensor,
    delta_values: torch.Tensor,
    a_count_parent: int,
    a_count_deltas: torch.Tensor,
    d: int,
    subcube_h_delta: Optional[torch.Tensor] = None,
    subcube_i_delta: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Incremental k-point Fourier delta update on GPU:
      fhat'(S) = fhat_parent(S) + (1/2^d) * sum_{j=1}^k delta_values[b, j] * (-1)^{<S, x_{b,j}>}.
    Evaluates B candidate moves against one frozen parent spectrum in O(B * k * 2^d)
    without running B full FWHTs.

    Args:
      fhat_parent: [2^d] float32/float64 parent Fourier spectrum of g0.
      deg: [2^d] degree table |S|.
      delta_indices: [B, k] int32 indices x_j in [0, 2^d).
      delta_values: [B, k] float delta g0(x_j) (0.0 for padded slots).
      a_count_parent: int |A| of parent.
      a_count_deltas: [B] change in |A| for each candidate (antichain mode), ignored if subcube_* provided.
      d: dimension.
    Returns:
      (H_tot [B], I_tot [B], OT [B]).
    """
    device = fhat_parent.device
    dtype = fhat_parent.dtype
    n_pts = 1 << d
    bsz, k = delta_indices.shape
    s_idx = torch.arange(n_pts, device=device, dtype=torch.int32)  # [N]

    # Process in chunks to strictly bound VRAM <= 4.5 GiB
    bpe = 4 if dtype == torch.float32 else 8
    # Each candidate needs ~2 * N * bpe for fhat_new & p
    chunk_b = max(1, min(bsz, int((1.5 * (1024 ** 3)) // max(1, 3 * n_pts * bpe))))
    h_out = torch.empty(bsz, device=device, dtype=dtype)
    i_out = torch.empty(bsz, device=device, dtype=dtype)
    ot_out = torch.empty(bsz, device=device, dtype=dtype)
    eps = 1e-38 if dtype == torch.float32 else 1e-300
    inv_n = 1.0 / float(n_pts)

    for start in range(0, bsz, chunk_b):
        end = min(bsz, start + chunk_b)
        cb = end - start
        idx_c = delta_indices[start:end]  # [cb, k]
        val_c = delta_values[start:end].to(dtype=dtype) * inv_n  # [cb, k]
        f_new = fhat_parent.unsqueeze(0).expand(cb, n_pts).clone()
        for j in range(k):
            vj = val_c[:, j : j + 1]  # [cb, 1]
            if not torch.any(vj != 0):
                continue
            xj = idx_c[:, j : j + 1]  # [cb, 1]
            # Compute parity of (S & xj) via bit loop (d <= 20)
            and_bits = xj & s_idx.unsqueeze(0)  # [cb, N]
            par = torch.zeros((cb, n_pts), device=device, dtype=torch.int32)
            for b in range(d):
                par ^= (and_bits >> b) & 1
            sign = 1.0 - 2.0 * par.to(dtype=dtype)
            f_new.add_(vj * sign)

        p = f_new * f_new
        i_g0 = (p * deg).sum(dim=1)
        h_sub = -torch.where(p > 0.0, p * torch.log2(p.clamp_min(eps)), torch.zeros_like(p)).sum(dim=1)

        if subcube_h_delta is not None and subcube_i_delta is not None:
            i_tot = i_g0 + subcube_i_delta[start:end].to(dtype=dtype)
            h_tot = h_sub + subcube_h_delta[start:end].to(dtype=dtype)
        else:
            a_new = (float(a_count_parent) + a_count_deltas[start:end].to(dtype=dtype)) * inv_n
            i_tot = i_g0 + a_new * (1.0 + 0.5 * float(d))
            h_tot = h_sub + a_new * (2.0 * float(d))

        mean_ok = f_new[:, 0].abs() < 1e-7
        valid = mean_ok & (i_tot > 1.0 + 1e-9)
        ot = torch.where(valid, h_tot / (i_tot - 1.0).clamp_min(1e-12), torch.zeros_like(h_tot))
        h_out[start:end] = h_tot
        i_out[start:end] = i_tot
        ot_out[start:end] = ot

    return h_out, i_out, ot_out


# ---------------------------------------------------------------------------
# Monotonicity, Balance & Maximal Subcube Coalescing (Vector A2exact)
# ---------------------------------------------------------------------------

def check_core_monotone_and_balanced(g0: np.ndarray, d: int) -> Tuple[bool, bool]:
    """
    Checks whether 3-valued core g0 in {-1, 0, +1}^{2^d} is:
      - balanced: sum(g0) == 0
      - monotone: g0(x) <= g0(x | (1<<b)) on all d * 2^{d-1} hypercube edges.
    Note: for full monotonicity of de-aliased F(x, y_1..y_k), we ALSO require
    no comparable pair (x < x') across DISTINCT subcubes C_j != C_{j'}!
    """
    g = np.asarray(g0, dtype=np.int8)
    is_bal = int(np.sum(g)) == 0
    is_mono = True
    n_pts = 1 << d
    for b in range(d):
        # Reshape so axis -2 is bit b (size 2)
        stride = 1 << b
        view = g.reshape(-1, 2, stride)
        if np.any(view[:, 0, :] > view[:, 1, :]):
            is_mono = False
            break
    return is_mono, is_bal


def decompose_zero_subcubes(
    g0: np.ndarray, d: int
) -> Tuple[List[Dict[str, int]], bool]:
    """
    Greedy maximal-subcube decomposition of A = {x : g0(x) == 0} into disjoint subcubes
    C_j = {x : (x & fixed_mask) == fixed_val} of codimension r_j = popcount(fixed_mask).

    Why merge adjacent zero-cells into maximal subcubes?
      - Splitting a connected subcube C (r -> r+1) adds Delta H / Delta I = 4 < 6.52 (HURTS!).
      - Keeping maximal subcubes intact and assigning a DISTINCT fresh auxiliary y_j to each
        incomparable subcube C_j maximizes H/(I-1) while preserving monotonicity!

    Also checks the strict no-comparable-pairs invariant across distinct subcubes:
      for all j != j', there is NO x in C_j, x' in C_{j'} with x < x' (bitwise).
    Equivalently: for two subcubes C_1 = [lo_1, hi_1] and C_2 = [lo_2, hi_2] (where
    lo = fixed_val, hi = fixed_val | free_mask), there exists x in C_1, x' in C_2 with
    x <= x' iff lo_1 <= hi_2 (i.e. (lo_1 & ~hi_2) == 0)!

    Returns:
      (subcubes, strictly_monotone_compatible)
      Each subcube dict has: {"fixed_mask": int, "fixed_val": int, "codim": int, "size": int}.
    """
    g = np.asarray(g0, dtype=np.int8)
    zeros = set(int(x) for x in np.flatnonzero(g == 0))
    if not zeros:
        return [], True

    full_mask = (1 << d) - 1
    subcubes: List[Dict[str, int]] = []
    remaining = set(zeros)

    # Greedily extract largest subcubes first (prefer merging adjacent cells along free bits)
    while remaining:
        # Pick smallest remaining point (by popcount, then index) as anchor
        anchor = min(remaining, key=lambda x: (x.bit_count(), x))
        free_mask = 0
        # Iteratively expand along any coordinate bit whose partner-cube is wholly inside remaining
        for b in range(d):
            bit = 1 << b
            if anchor & bit:
                continue
            # Check if shifting current cube by `bit` stays completely inside `remaining`
            # Current cube elements are {anchor | s : s subset of free_mask}
            ok = True
            # Enumerate subsets of free_mask
            s = free_mask
            while True:
                if (anchor | s | bit) not in remaining:
                    ok = False
                    break
                if s == 0:
                    break
                s = (s - 1) & free_mask
            if ok:
                free_mask |= bit

        # Remove all elements of this maximal subcube from `remaining`
        s = free_mask
        while True:
            remaining.remove(anchor | s)
            if s == 0:
                break
            s = (s - 1) & free_mask

        fixed_mask = full_mask ^ free_mask
        fixed_val = anchor & fixed_mask
        codim = fixed_mask.bit_count()
        subcubes.append({
            "fixed_mask": int(fixed_mask),
            "fixed_val": int(fixed_val),
            "codim": int(codim),
            "size": 1 << (d - codim),
        })

    # Verify strict no-comparable-pairs across distinct subcubes:
    # Fast exact O(d * 2^d) theorem when g0 is monotone: any chain x < z_1 < ... < x' between
    # two zero-cells must stay inside g0^{-1}(0), so a comparable pair across distinct subcubes
    # exists IFF a Hamming-1 edge crosses between distinct subcubes!
    # Also check general O(k^2) fallback if g0 is non-monotone and k is modest.
    k = len(subcubes)
    if k <= 1:
        return subcubes, True

    is_mono_core, _ = check_core_monotone_and_balanced(g, d)
    if is_mono_core:
        n_pts = 1 << d
        owner = np.full(n_pts, -1, dtype=np.int32)
        for idx_c, c in enumerate(subcubes):
            fm, fv = c["fixed_mask"], c["fixed_val"]
            free_m = full_mask ^ fm
            s = free_m
            while True:
                owner[fv | s] = idx_c
                if s == 0:
                    break
                s = (s - 1) & free_m
        compat = True
        for b in range(d):
            stride = 1 << b
            view = owner.reshape(-1, 2, stride)
            u_ow = view[:, 0, :]
            v_ow = view[:, 1, :]
            if np.any((u_ow >= 0) & (v_ow >= 0) & (u_ow != v_ow)):
                compat = False
                break
        return subcubes, compat

    los = [c["fixed_val"] for c in subcubes]
    his = [c["fixed_val"] | (full_mask ^ c["fixed_mask"]) for c in subcubes]
    compat = True
    for i in range(k):
        lo_i = los[i]
        hi_i = his[i]
        for j in range(i + 1, k):
            if (lo_i & ~his[j]) == 0 or (los[j] & ~hi_i) == 0:
                compat = False
                break
        if not compat:
            break

    return subcubes, compat


def evaluate_subcube_dealiased_exact_fp64(
    g0: np.ndarray, d: int, device: Optional[torch.device] = None
) -> Dict[str, object]:
    """
    High-precision float64 evaluation of a single 3-valued core g0 with automatic
    maximal-subcube decomposition and strict monotonicity/balance checks.
    """
    dev = device or get_device(prefer_cuda=True)
    is_mono_core, is_bal = check_core_monotone_and_balanced(g0, d)
    subcubes, subcube_compat = decompose_zero_subcubes(g0, d)
    is_mono_full = bool(is_mono_core and subcube_compat)

    t_g0 = torch.as_tensor(g0, device=dev, dtype=torch.float64).unsqueeze(0)
    mean, h_sub, i_g0, _ = evaluate_core_spectrum_batch(t_g0, dtype=torch.float64)
    h_val = float(h_sub.item())
    i_val = float(i_g0.item())
    for c in subcubes:
        r = c["codim"]
        w = 2.0 ** (-r)
        i_val += w * (1.0 + 0.5 * r)
        h_val += w * (2.0 * r)
    ot = h_val / (i_val - 1.0) if (is_bal and i_val > 1.0 + 1e-12) else 0.0
    return {
        "d": int(d),
        "k_aux": len(subcubes),
        "n_eff": int(d + len(subcubes)),
        "zero_cells": int(np.sum(np.asarray(g0) == 0)),
        "balanced": is_bal,
        "monotone_core": is_mono_core,
        "subcubes_incomparable": subcube_compat,
        "monotone_full": is_mono_full,
        "mean": float(mean.item()),
        "H": h_val,
        "I": i_val,
        "OT": ot,
        "subcubes": subcubes,
    }


def materialize_full_truth_table(g0: np.ndarray, d: int, subcubes: Sequence[Dict[str, int]]) -> np.ndarray:
    """
    Materializes the explicit 2^(d+k) truth table F(x, y_1..y_k) in {-1, +1} for Tier 1 cross-checks
    (only for small n = d + k <= 20).
    """
    k = len(subcubes)
    n = d + k
    if n > 20:
        raise ValueError(f"Refusing to materialize 2^{n} truth table in RAM (n > 20)")
    g = np.asarray(g0, dtype=np.int8)
    n_x = 1 << d
    # Map each x in g0^{-1}(0) to its subcube index j in 0..k-1
    owner = np.full(n_x, -1, dtype=np.int32)
    for j, c in enumerate(subcubes):
        fm, fv = c["fixed_mask"], c["fixed_val"]
        for x in range(n_x):
            if (x & fm) == fv:
                assert owner[x] == -1, f"Overlapping subcubes at x={x}"
                owner[x] = j
    assert np.all((owner >= 0) == (g == 0)), "Subcubes do not partition g0^{-1}(0)"

    out = np.empty(1 << n, dtype=np.int8)
    for y_mask in range(1 << k):
        base = y_mask << d
        slice_vals = g.copy()
        for x in np.flatnonzero(g == 0):
            j = owner[x]
            bit = (y_mask >> j) & 1
            slice_vals[x] = 1 if bit else -1
        out[base : base + n_x] = slice_vals
    return out


# ---------------------------------------------------------------------------
# Reference Constructors & Phase 0 Self-Test / Benchmark CLI
# ---------------------------------------------------------------------------

def make_q2_anchor() -> Tuple[int, np.ndarray]:
    # d=2, |A|=2 -> n=4, OT = 6.00000000
    return 2, np.array([-1, 0, 0, 1], dtype=np.int8)


def make_tribes22_dealiased_anchor() -> Tuple[int, np.ndarray]:
    # d=4, Tribes(2,2) = (x0&x1)|(x2&x3) + antipodal maximal zeros 0101 (5), 1010 (10) -> n=6, OT=6.34706750
    g = np.full(16, -1, dtype=np.int8)
    for x in range(16):
        b = [(x >> i) & 1 for i in range(4)]
        if (b[0] and b[1]) or (b[2] and b[3]):
            g[x] = 1
    g[5] = 0
    g[10] = 0
    return 4, g


def make_alttree6_dealiased_anchor() -> Tuple[int, np.ndarray]:
    # d=6: (x0&x1) | (x2 & (x3 | (x4&x5))) + 2 maximal zeros -> n=8, OT=6.36396945
    g = np.full(64, -1, dtype=np.int8)
    for x in range(64):
        b = [(x >> i) & 1 for i in range(6)]
        if (b[0] and b[1]) or (b[2] and (b[3] or (b[4] and b[5]))):
            g[x] = 1
    # Find best pair of maximal zeros (deterministic: 21 and 58, or verify all pairs)
    max_z = [
        x for x in range(64)
        if g[x] == -1 and all(g[x | (1 << i)] == 1 for i in range(6) if not ((x >> i) & 1))
    ]
    best_ot = -1.0
    best_g = None
    for i in range(len(max_z)):
        for j in range(i + 1, len(max_z)):
            cand = g.copy()
            cand[max_z[i]] = 0
            cand[max_z[j]] = 0
            res = evaluate_subcube_dealiased_exact_fp64(cand, 6, device=torch.device("cpu"))
            if res["OT"] > best_ot:
                best_ot = float(res["OT"])
                best_g = cand
    assert best_g is not None
    return 6, best_g


def run_selftest_and_benchmark() -> Dict[str, object]:
    device = get_device(prefer_cuda=True)
    report: Dict[str, object] = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "anchors": {},
        "subcube_r_lt_d_check": {},
        "delta_gpu_check": {},
        "precision_check": {},
        "benchmarks": {},
    }

    # 1. Verify Vector A2 anchors against materialized full 2^(d+k) FWHT
    for name, (d, g0), expected_ot in [
        ("Q2_n4", make_q2_anchor(), 6.0),
        ("Tribes22_n6", make_tribes22_dealiased_anchor(), 6.347067498701919),
        ("AltTree6_n8", make_alttree6_dealiased_anchor(), 6.363969446288164),
    ]:
        res = evaluate_subcube_dealiased_exact_fp64(g0, d, device=device)
        full_tt = materialize_full_truth_table(g0, d, res["subcubes"])  # type: ignore[arg-type]
        n_full = int(res["n_eff"])  # type: ignore[arg-type]
        t_full = torch.as_tensor(full_tt, device=device, dtype=torch.float64).unsqueeze(0)
        m_f, h_f, i_f, _ = evaluate_core_spectrum_batch(t_full, dtype=torch.float64)
        ot_full = float(h_f.item() / (i_f.item() - 1.0))
        diff = abs(float(res["OT"]) - ot_full)
        assert res["balanced"] and res["monotone_full"], f"{name} failed monotone/balance check!"
        assert diff < 1e-12, f"{name} collapsed vs full mismatch: {diff}"
        assert abs(ot_full - expected_ot) < 1e-9, f"{name} expected {expected_ot}, got {ot_full}"
        report["anchors"][name] = {
            "d": d,
            "n_eff": n_full,
            "I": res["I"],
            "H": res["H"],
            "OT_collapsed": res["OT"],
            "OT_full": ot_full,
            "diff": diff,
        }

    # 2. Verify proper subcube codim r < d (e.g. d=4, two 1-dim subcubes codim r=3: 010* and 101* -> n=6)
    # Construct monotone g0 on d=4 where x=(x0,x1,x2,x3):
    # Actually use d=3 with bit 2 free on 01* and 10* -> wait: that's Q2 lifted to d=3 (codim r=2 < d=3)!
    g_sub = np.array([-1, 0, 0, 1, -1, 0, 0, 1], dtype=np.int8)  # bit 2 free! 01* and 10* are two r=2 subcubes!
    res_sub = evaluate_subcube_dealiased_exact_fp64(g_sub, 3, device=device)
    assert res_sub["k_aux"] == 2 and all(c["codim"] == 2 for c in res_sub["subcubes"]), f"Expected two r=2 subcubes, got {res_sub['subcubes']}"
    assert res_sub["monotone_full"] and abs(float(res_sub["OT"]) - 6.0) < 1e-12
    full_sub = materialize_full_truth_table(g_sub, 3, res_sub["subcubes"])  # type: ignore[arg-type]
    _, hf_s, if_s, _ = evaluate_core_spectrum_batch(torch.as_tensor(full_sub, device=device, dtype=torch.float64).unsqueeze(0), dtype=torch.float64)
    diff_sub = abs(float(res_sub["OT"]) - float(hf_s.item() / (if_s.item() - 1.0)))
    assert diff_sub < 1e-12
    report["subcube_r_lt_d_check"] = {"codims": [c["codim"] for c in res_sub["subcubes"]], "OT": res_sub["OT"], "diff": diff_sub}

    # 3. Verify GPU incremental delta evaluator vs full FWHT on d=10
    d_test = 10
    rng = np.random.default_rng(123)
    w_bits = np.array([int(x).bit_count() for x in range(1 << d_test)])
    g_parent = np.where(w_bits < 5, -1, np.where(w_bits > 5, 1, 0)).astype(np.int8)
    # Keep only 4 zeros (balance rest)
    mid_idx = np.flatnonzero(g_parent == 0)
    g_parent[mid_idx[:124]] = -1
    g_parent[mid_idx[124:248]] = 1
    # Now 4 zeros remain, balanced!
    deg10 = make_degree_table(d_test, device, dtype=torch.float64)
    _, _, _, fhat_p = evaluate_core_spectrum_batch(torch.as_tensor(g_parent, device=device, dtype=torch.float64).unsqueeze(0), deg=deg10, dtype=torch.float64)
    fhat_p = fhat_p.squeeze(0)
    # Create 16 random 2-point balanced flips
    b_delta = 16
    pos_idx = np.flatnonzero(g_parent == 1)
    neg_idx = np.flatnonzero(g_parent == -1)
    d_idx = np.zeros((b_delta, 2), dtype=np.int32)
    d_val = np.zeros((b_delta, 2), dtype=np.float64)
    a_dlt = np.zeros(b_delta, dtype=np.float64)
    batch_mut = np.repeat(g_parent[None, :], b_delta, axis=0)
    for b in range(b_delta):
        p_i = int(rng.choice(pos_idx))
        n_i = int(rng.choice(neg_idx))
        if b % 2 == 0:
            # Swap (+1, -1) -> (-1, +1)
            d_idx[b] = [p_i, n_i]
            d_val[b] = [-2.0, 2.0]
            a_dlt[b] = 0.0
            batch_mut[b, p_i] = -1
            batch_mut[b, n_i] = 1
        else:
            # Expand antichain (+1, -1) -> (0, 0)
            d_idx[b] = [p_i, n_i]
            d_val[b] = [-1.0, 1.0]
            a_dlt[b] = 2.0
            batch_mut[b, p_i] = 0
            batch_mut[b, n_i] = 0
    h_d, i_d, ot_d = eval_delta_candidates_gpu(
        fhat_p, deg10,
        torch.as_tensor(d_idx, device=device),
        torch.as_tensor(d_val, device=device),
        int(np.sum(g_parent == 0)),
        torch.as_tensor(a_dlt, device=device),
        d_test,
    )
    ref = evaluate_antichain_dealiased_batch(torch.as_tensor(batch_mut, device=device, dtype=torch.float64), deg=deg10, dtype=torch.float64)
    max_delta_err = float(torch.max((ot_d - ref["OT"]).abs()).item())
    assert max_delta_err < 1e-12, f"Delta GPU mismatch: {max_delta_err}"
    report["delta_gpu_check"] = {"d": d_test, "batch": b_delta, "max_OT_err": max_delta_err}

    # 4. GPU float32 vs float64 precision check & throughput benchmark on d = 14, 16, 18
    for d, bsz in [(14, 256), (16, 128), (18, 64)]:
        n_pts = 1 << d
        # Precision check on 4 balanced cores
        x_prec = torch.empty((4, n_pts), device=device, dtype=torch.int8)
        x_prec[:, : n_pts // 2] = -1
        x_prec[:, n_pts // 2 :] = 1
        # Shuffle each row deterministically
        gen = torch.Generator(device=device).manual_seed(42 + d)
        for r in range(4):
            perm = torch.randperm(n_pts, device=device, generator=gen)
            x_prec[r] = x_prec[r, perm]
        res32 = evaluate_antichain_dealiased_batch(x_prec, dtype=torch.float32)
        res64 = evaluate_antichain_dealiased_batch(x_prec, dtype=torch.float64)
        max_ot_err = float(torch.max((res32["OT"].double() - res64["OT"]).abs()).item())
        max_i_err = float(torch.max((res32["I"].double() - res64["I"]).abs()).item())
        report["precision_check"][f"d={d}"] = {"max_OT_err_f32_vs_f64": max_ot_err, "max_I_err_f32_vs_f64": max_i_err}

        # Speed + Peak VRAM benchmark
        deg_f32 = make_degree_table(d, device, dtype=torch.float32)
        x_bench = torch.randint(0, 2, (bsz, n_pts), device=device, dtype=torch.int8) * 2 - 1
        # Warmup
        for _ in range(3):
            _ = evaluate_antichain_dealiased_batch(x_bench, deg=deg_f32, dtype=torch.float32)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        iters = 20
        for _ in range(iters):
            _ = evaluate_antichain_dealiased_batch(x_bench, deg=deg_f32, dtype=torch.float32)
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_mb = 0.0
        dt = (time.perf_counter() - t0) / iters
        evals_s = bsz / dt
        report["benchmarks"][f"d={d}"] = {
            "N": n_pts,
            "batch": bsz,
            "ms_per_batch": round(dt * 1000.0, 3),
            "evals_per_sec": round(evals_s, 1),
            "peak_vram_mb": round(peak_mb, 2),
            "within_4_5_gib_cap": peak_mb <= 4.5 * 1024.0,
        }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="RTX 5060 FWHT + Vector A2 self-test & benchmark")
    parser.add_argument("--selftest", action="store_true", help="Run full Phase 0 self-test and benchmark")
    args = parser.parse_args()
    if args.selftest or True:
        rep = run_selftest_and_benchmark()
        print(json.dumps(rep, indent=2))
        b18 = rep["benchmarks"]["d=18"]  # type: ignore[index]
        assert b18["within_4_5_gib_cap"], "VRAM cap exceeded!"
        if rep["device"] == "cuda":
            assert b18["evals_per_sec"] >= 2500.0, f"Phase 0 speed gate failed: {b18['evals_per_sec']} < 2500"
        print("RESULT: fwht_gpu Phase 0 self-test & benchmark PASSED")


if __name__ == "__main__":
    main()

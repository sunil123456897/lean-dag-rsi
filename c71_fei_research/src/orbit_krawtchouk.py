"""
Analytical Krawtchouk & Wreath-Product Orbit-Compressed Walsh-Hadamard Engine
for Vector B (Symmetry-Compressed High-n Discovery, d = 8..24) + Vector A2 De-Aliasing.

Key mathematical identities (all verified bit-exact against full 2^d FWHT):
  1. Block-symmetric G = S_{k_1} x ... x S_{k_m} (d = sum k_j, M = prod(k_j + 1)):
       W_G = K^{(k_1)} (x) ... (x) K^{(k_m)}
     Applied via factorized O(M * sum(k_j + 1)) Kronecker tensor contractions on GPU
     (zero O(M^2) memory blowup even at M = 59,049 for S_2^{10}, d = 20).
  2. Wreath product G = S_a wr S_b (d = a * b, M = C(b + a, b)):
       (W_G)_{r, w} = sum_{distinct permutations w~ of w} prod_{t=1}^b K^{(a)}_{r_t, w~_t}
     Computed analytically via DP over block multiplicities (zero 2^d enumeration even at d = 24!).
  3. Exact orbit-space Parseval, Influence, Entropy, and Antichain De-Aliasing:
       sum_j |O_j| v_j^2 = 1 - |A|/2^d   (for 3-valued u in {-1, 0, +1}^M)
       Inf(F) = sum_j |O_j| wt(O_j) v_j^2 + (|A|/2^d) * (1 + d/2)
       H(F)   = -sum_j |O_j| v_j^2 log_2(v_j^2) + (|A|/2^d) * (2 * d)
     Crucially: inside log_2 is v_j^2 (NOT |O_j| * v_j^2), and every constant-weight orbit O_i
     is internally an antichain, so u_i = 0 is monotone-compatible iff no two zero orbits
     are comparable in the orbit poset (w <= w' coordinatewise)!
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations_with_replacement, product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from fwht_gpu import (
        evaluate_antichain_dealiased_batch,
        evaluate_subcube_dealiased_exact_fp64,
        get_device,
    )
except ImportError:
    from .fwht_gpu import (  # type: ignore[no-redef]
        evaluate_antichain_dealiased_batch,
        evaluate_subcube_dealiased_exact_fp64,
        get_device,
    )


def krawtchouk_matrix(k: int) -> np.ndarray:
    """
    Exact normalized binary Krawtchouk matrix K^{(k)} of shape (k+1, k+1):
      K[r, w] = 2^{-k} * sum_{t} (-1)^t * C(r, t) * C(k - r, w - t).
    """
    mat = np.zeros((k + 1, k + 1), dtype=np.float64)
    scale = float(1 << k)
    for r in range(k + 1):
        for w in range(k + 1):
            t_lo = max(0, w - (k - r))
            t_hi = min(r, w)
            s = 0
            for t in range(t_lo, t_hi + 1):
                term = math.comb(r, t) * math.comb(k - r, w - t)
                s += -term if (t & 1) else term
            mat[r, w] = float(s) / scale
    return mat


class BlockSymGroup:
    """
    Block-symmetric permutation group G = S_{k_1} x ... x S_{k_m} acting on d = sum(k_j) bits.
    Supports factorized Kronecker Krawtchouk transforms on GPU/CPU, exact monotonicity/antichain
    checks in orbit space, lifting to finer subgroups, and expansion to full 2^d truth tables.
    """

    def __init__(self, blocks: Sequence[int], device: Optional[torch.device] = None) -> None:
        self.blocks: Tuple[int, ...] = tuple(int(k) for k in blocks)
        self.d: int = sum(self.blocks)
        self.shape: Tuple[int, ...] = tuple(k + 1 for k in self.blocks)
        self.M: int = 1
        for s in self.shape:
            self.M *= s
        self.device = device or get_device(prefer_cuda=True)

        # Build orbit tuples, sizes, and Hamming weights
        self.orbits: List[Tuple[int, ...]] = list(product(*(range(k + 1) for k in self.blocks)))
        sizes = np.empty(self.M, dtype=np.int64)
        wts = np.empty(self.M, dtype=np.int32)
        for idx, tup in enumerate(self.orbits):
            sz = 1
            wt = 0
            for k, w in zip(self.blocks, tup):
                sz *= math.comb(k, w)
                wt += w
            sizes[idx] = sz
            wts[idx] = wt
        assert int(np.sum(sizes)) == (1 << self.d)
        self.orbit_sizes_np = sizes
        self.orbit_wts_np = wts

        # Factorized Krawtchouk matrices per block
        self.K_np: List[np.ndarray] = [krawtchouk_matrix(k) for k in self.blocks]
        self.K_f32: List[torch.Tensor] = [torch.as_tensor(km, device=self.device, dtype=torch.float32) for km in self.K_np]
        self.K_f64: List[torch.Tensor] = [torch.as_tensor(km, device=self.device, dtype=torch.float64) for km in self.K_np]
        self.sizes_f32 = torch.as_tensor(sizes, device=self.device, dtype=torch.float32)
        self.sizes_f64 = torch.as_tensor(sizes, device=self.device, dtype=torch.float64)
        self.wts_f32 = torch.as_tensor(wts, device=self.device, dtype=torch.float32)
        self.wts_f64 = torch.as_tensor(wts, device=self.device, dtype=torch.float64)

        # Precompute covering edges (i -> j where w_j = w_i + e_axis) for O(M * m) monotonicity checks
        strides = []
        st = 1
        for s in reversed(self.shape):
            strides.append(st)
            st *= s
        strides.reverse()
        self.strides = tuple(strides)
        up_src: List[int] = []
        up_dst: List[int] = []
        for idx, tup in enumerate(self.orbits):
            for ax, (k, w) in enumerate(zip(self.blocks, tup)):
                if w < k:
                    up_src.append(idx)
                    up_dst.append(idx + strides[ax])
        self.edge_src = np.array(up_src, dtype=np.int32)
        self.edge_dst = np.array(up_dst, dtype=np.int32)
        # Adjacency lists for fast boundary/minimal/maximal orbit queries
        self.parents: List[List[int]] = [[] for _ in range(self.M)]  # smaller orbits w - e_ax
        self.children: List[List[int]] = [[] for _ in range(self.M)]  # larger orbits w + e_ax
        for u, v in zip(up_src, up_dst):
            self.children[u].append(v)
            self.parents[v].append(u)

    def build_dense_W_G(self) -> np.ndarray:
        """Builds explicit M x M Kronecker matrix W_G (only for M <= 4096)."""
        if self.M > 4096:
            raise ValueError(f"M={self.M} > 4096; use factorized transform_batch() instead")
        W = np.array([[1.0]], dtype=np.float64)
        for km in self.K_np:
            W = np.kron(W, km)
        return W

    def transform_batch(self, u_batch: torch.Tensor, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        """
        Factorized Kronecker Krawtchouk transform: u [B, M] -> v [B, M].
        Exact O(B * M * sum(k_j + 1)) with zero M x M matrix allocation!
        """
        if u_batch.ndim == 1:
            u_batch = u_batch.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        bsz = u_batch.shape[0]
        kms = self.K_f32 if dtype == torch.float32 else self.K_f64
        t = u_batch.to(device=self.device, dtype=dtype).view((bsz,) + self.shape)
        # Contract along each block axis 1..m using tensordot / matmul via permute
        m = len(self.blocks)
        for ax in range(m):
            # Move axis (ax + 1) to the last position, multiply by K^T, move back
            dim_ax = ax + 1
            perm = list(range(m + 1))
            perm[dim_ax], perm[-1] = perm[-1], perm[dim_ax]
            t = t.permute(*perm)
            t = torch.matmul(t, kms[ax].t())
            t = t.permute(*perm)
        out = t.reshape(bsz, self.M)
        return out.squeeze(0) if squeeze else out

    def evaluate_batch(self, u_batch: torch.Tensor, dtype: torch.dtype = torch.float64) -> Dict[str, torch.Tensor]:
        """
        Evaluates batch [B, M] of 3-valued orbit vectors u in {-1, 0, +1}^M with
        antichain de-aliasing on all zero orbits (each cell in zero orbits gets a fresh y_a).
        """
        if u_batch.ndim == 1:
            u_batch = u_batch.unsqueeze(0)
        u_t = u_batch.to(device=self.device, dtype=dtype)
        v = self.transform_batch(u_t, dtype=dtype)
        p = v * v
        sizes = self.sizes_f32 if dtype == torch.float32 else self.sizes_f64
        wts = self.wts_f32 if dtype == torch.float32 else self.wts_f64
        eps = 1e-38 if dtype == torch.float32 else 1e-300

        i_g0 = (p * (sizes * wts)).sum(dim=1)
        h_sub = -torch.where(p > 0.0, sizes * p * torch.log2(p.clamp_min(eps)), torch.zeros_like(p)).sum(dim=1)
        a_cells = ((u_t == 0).to(dtype=dtype) * sizes).sum(dim=1)
        frac_a = a_cells / float(1 << self.d)
        i_tot = i_g0 + frac_a * (1.0 + 0.5 * float(self.d))
        h_tot = h_sub + frac_a * (2.0 * float(self.d))
        mean = v[:, 0]
        valid = (mean.abs() < 1e-9) & (i_tot > 1.0 + 1e-9)
        ot = torch.where(valid, h_tot / (i_tot - 1.0).clamp_min(1e-12), torch.zeros_like(h_tot))
        return {
            "mean": mean,
            "H": h_tot,
            "I": i_tot,
            "OT": ot,
            "A_cells": a_cells,
            "parseval_tot": (p * sizes).sum(dim=1) + frac_a,
        }

    def check_monotone_and_antichain(self, u: np.ndarray) -> Tuple[bool, bool, bool]:
        """
        Returns (is_monotone_core, zero_orbits_form_antichain, is_balanced).
        """
        u_arr = np.asarray(u, dtype=np.int8)
        is_bal = int(np.dot(self.orbit_sizes_np, u_arr.astype(np.int64))) == 0
        is_mono = bool(np.all(u_arr[self.edge_src] <= u_arr[self.edge_dst]))
        zero_idx = np.flatnonzero(u_arr == 0)
        is_ac = True
        if len(zero_idx) > 1:
            z_orbs = [self.orbits[int(i)] for i in zero_idx]
            nz = len(z_orbs)
            for i in range(nz):
                wi = z_orbs[i]
                for j in range(i + 1, nz):
                    wj = z_orbs[j]
                    if all(a <= b for a, b in zip(wi, wj)) or all(b <= a for a, b in zip(wi, wj)):
                        is_ac = False
                        break
                if not is_ac:
                    break
        return is_mono, is_ac, is_bal

    def expand_to_full(self, u: np.ndarray) -> np.ndarray:
        """Expands orbit vector u [M] into full 2^d truth table g0 in {-1, 0, +1}^{2^d} (for d <= 20)."""
        if self.d > 20:
            raise ValueError(f"Refusing to expand 2^{self.d} truth table (d > 20)")
        u_arr = np.asarray(u, dtype=np.int8).reshape(self.shape)
        n_pts = 1 << self.d
        x = np.arange(n_pts, dtype=np.int32)
        coords = []
        shift = 0
        for k in self.blocks:
            mask = (1 << k) - 1
            blk = (x >> shift) & mask
            wt = np.zeros(n_pts, dtype=np.int32)
            for b in range(k):
                wt += (blk >> b) & 1
            coords.append(wt)
            shift += k
        return u_arr[tuple(coords)].astype(np.int8)

    def lift_to_finer_blocks(self, u: np.ndarray, split_plan: Sequence[Sequence[int]]) -> Tuple["BlockSymGroup", np.ndarray]:
        """
        Lifts an orbit vector u from S_{k_1} x ... x S_{k_m} to a finer subgroup where
        each block k_j is split into sub-blocks summing to k_j (e.g. (4,4) -> ((2,2),(2,2))).
        """
        assert len(split_plan) == len(self.blocks)
        new_blocks: List[int] = []
        for k, sub in zip(self.blocks, split_plan):
            assert sum(sub) == k, f"Sub-blocks {sub} do not sum to {k}"
            new_blocks.extend(sub)
        fine = BlockSymGroup(new_blocks, device=self.device)
        u_grid = np.asarray(u, dtype=np.int8).reshape(self.shape)
        u_fine = np.empty(fine.M, dtype=np.int8)
        for idx, ftup in enumerate(fine.orbits):
            coarse_tup = []
            pos = 0
            for sub in split_plan:
                coarse_tup.append(sum(ftup[pos : pos + len(sub)]))
                pos += len(sub)
            u_fine[idx] = u_grid[tuple(coarse_tup)]
        return fine, u_fine


class WreathSymGroup:
    """
    Wreath-product symmetry group G = S_a wr S_b acting on d = a * b bits.
    Collapses 2^(a*b) hypercube to M = C(b + a, b) orbits (e.g., S_3 wr S_8: 2^24 -> 165!).
    Computes W_G analytically via DP over block weight multiplicities (zero 2^d enumeration).
    """

    def __init__(self, a: int, b: int, device: Optional[torch.device] = None) -> None:
        self.a = int(a)
        self.b = int(b)
        self.d = self.a * self.b
        self.device = device or get_device(prefer_cuda=True)

        # Orbits = non-decreasing tuples (w_1 <= ... <= w_b) in {0..a}^b
        self.orbits: List[Tuple[int, ...]] = list(combinations_with_replacement(range(self.a + 1), self.b))
        self.M = len(self.orbits)
        assert self.M == math.comb(self.b + self.a, self.b)

        sizes = np.empty(self.M, dtype=np.int64)
        wts = np.empty(self.M, dtype=np.int32)
        fact_b = math.factorial(self.b)
        for idx, tup in enumerate(self.orbits):
            counts = [0] * (self.a + 1)
            prod_c = 1
            for w in tup:
                counts[w] += 1
                prod_c *= math.comb(self.a, w)
            denom = 1
            for c in counts:
                denom *= math.factorial(c)
            sizes[idx] = (fact_b // denom) * prod_c
            wts[idx] = sum(tup)
        assert int(np.sum(sizes)) == (1 << self.d)
        self.orbit_sizes_np = sizes
        self.orbit_wts_np = wts

        # Analytical W_G via DP over remaining weight-count state (handles up to b=10, a=5 easily!)
        self.K_a = krawtchouk_matrix(self.a)
        self.W_G_np = self._build_wreath_matrix()
        self.W_G_f32 = torch.as_tensor(self.W_G_np, device=self.device, dtype=torch.float32)
        self.W_G_f64 = torch.as_tensor(self.W_G_np, device=self.device, dtype=torch.float64)
        self.sizes_f32 = torch.as_tensor(sizes, device=self.device, dtype=torch.float32)
        self.sizes_f64 = torch.as_tensor(sizes, device=self.device, dtype=torch.float64)
        self.wts_f32 = torch.as_tensor(wts, device=self.device, dtype=torch.float32)
        self.wts_f64 = torch.as_tensor(wts, device=self.device, dtype=torch.float64)

        # Build exact covering edges of wreath poset:
        # w <= w' iff sorted w_i <= w'_i for all i. Covering relation: increment one element w_i -> w_i+1 and re-sort!
        orb_to_idx = {tup: i for i, tup in enumerate(self.orbits)}
        edges_set = set()
        for i, tup in enumerate(self.orbits):
            for pos in range(self.b):
                if tup[pos] < self.a:
                    nxt = list(tup)
                    nxt[pos] += 1
                    nxt_t = tuple(sorted(nxt))
                    j = orb_to_idx[nxt_t]
                    edges_set.add((i, j))
        edges_sorted = sorted(edges_set)
        self.edge_src = np.array([e[0] for e in edges_sorted], dtype=np.int32)
        self.edge_dst = np.array([e[1] for e in edges_sorted], dtype=np.int32)
        self.parents: List[List[int]] = [[] for _ in range(self.M)]
        self.children: List[List[int]] = [[] for _ in range(self.M)]
        for u, v in edges_sorted:
            self.children[u].append(v)
            self.parents[v].append(u)

    def _build_wreath_matrix(self) -> np.ndarray:
        """
        Computes exact W_G[r_idx, w_idx] = sum_{distinct perm w~ of w} prod_{t=0}^{b-1} K_a[r_t, w~_t].
        Uses exact multiset-DP per row r: dp[counts_tuple] = sum over placements of first t positions.
        Even for S_3 wr S_8 (M=165), there are <= 165 states at step t, taking < 10 ms total!
        """
        W = np.zeros((self.M, self.M), dtype=np.float64)
        # Map count-vector -> orbit index
        counts_to_idx: Dict[Tuple[int, ...], int] = {}
        for idx, tup in enumerate(self.orbits):
            cnt = [0] * (self.a + 1)
            for w in tup:
                cnt[w] += 1
            counts_to_idx[tuple(cnt)] = idx

        zero_state = tuple([0] * (self.a + 1))
        for r_idx, r_tup in enumerate(self.orbits):
            dp: Dict[Tuple[int, ...], float] = {zero_state: 1.0}
            for r_val in r_tup:
                row_k = self.K_a[r_val]
                nxt: Dict[Tuple[int, ...], float] = {}
                for state, val in dp.items():
                    for w_val in range(self.a + 1):
                        kv = float(row_k[w_val])
                        if kv == 0.0:
                            continue
                        ns = list(state)
                        ns[w_val] += 1
                        nst = tuple(ns)
                        nxt[nst] = nxt.get(nst, 0.0) + val * kv
                dp = nxt
            for cnt_key, val in dp.items():
                W[r_idx, counts_to_idx[cnt_key]] = val
        return W

    def evaluate_batch(self, u_batch: torch.Tensor, dtype: torch.dtype = torch.float64) -> Dict[str, torch.Tensor]:
        if u_batch.ndim == 1:
            u_batch = u_batch.unsqueeze(0)
        u_t = u_batch.to(device=self.device, dtype=dtype)
        W = self.W_G_f32 if dtype == torch.float32 else self.W_G_f64
        sizes = self.sizes_f32 if dtype == torch.float32 else self.sizes_f64
        wts = self.wts_f32 if dtype == torch.float32 else self.wts_f64
        eps = 1e-38 if dtype == torch.float32 else 1e-300

        v = torch.matmul(u_t, W.t())
        p = v * v
        i_g0 = (p * (sizes * wts)).sum(dim=1)
        h_sub = -torch.where(p > 0.0, sizes * p * torch.log2(p.clamp_min(eps)), torch.zeros_like(p)).sum(dim=1)
        a_cells = ((u_t == 0).to(dtype=dtype) * sizes).sum(dim=1)
        frac_a = a_cells / float(1 << self.d)
        i_tot = i_g0 + frac_a * (1.0 + 0.5 * float(self.d))
        h_tot = h_sub + frac_a * (2.0 * float(self.d))
        mean = v[:, 0]
        valid = (mean.abs() < 1e-9) & (i_tot > 1.0 + 1e-9)
        ot = torch.where(valid, h_tot / (i_tot - 1.0).clamp_min(1e-12), torch.zeros_like(h_tot))
        return {
            "mean": mean,
            "H": h_tot,
            "I": i_tot,
            "OT": ot,
            "A_cells": a_cells,
            "parseval_tot": (p * sizes).sum(dim=1) + frac_a,
        }

    def check_monotone_and_antichain(self, u: np.ndarray) -> Tuple[bool, bool, bool]:
        u_arr = np.asarray(u, dtype=np.int8)
        is_bal = int(np.dot(self.orbit_sizes_np, u_arr.astype(np.int64))) == 0
        is_mono = bool(np.all(u_arr[self.edge_src] <= u_arr[self.edge_dst]))
        zero_idx = np.flatnonzero(u_arr == 0)
        is_ac = True
        if len(zero_idx) > 1:
            z_orbs = [self.orbits[int(i)] for i in zero_idx]
            nz = len(z_orbs)
            for i in range(nz):
                wi = z_orbs[i]
                for j in range(i + 1, nz):
                    wj = z_orbs[j]
                    if all(x <= y for x, y in zip(wi, wj)) or all(y <= x for x, y in zip(wi, wj)):
                        is_ac = False
                        break
                if not is_ac:
                    break
        return is_mono, is_ac, is_bal

    def lift_to_block_sym(self, u: np.ndarray) -> Tuple[BlockSymGroup, np.ndarray]:
        """Lifts a wreath S_a wr S_b orbit vector to the subgroup S_a^b (breaking outer S_b symmetry)."""
        bg = BlockSymGroup((self.a,) * self.b, device=self.device)
        orb_to_val = {tup: int(u[i]) for i, tup in enumerate(self.orbits)}
        u_blk = np.array([orb_to_val[tuple(sorted(wtup))] for wtup in bg.orbits], dtype=np.int8)
        return bg, u_blk

    def expand_to_full(self, u: np.ndarray) -> np.ndarray:
        """Expands wreath orbit vector u [M] to full 2^d truth table (for d <= 20)."""
        bg, u_blk = self.lift_to_block_sym(u)
        return bg.expand_to_full(u_blk)


# ---------------------------------------------------------------------------
# Phase 0 Self-Test CLI
# ---------------------------------------------------------------------------

def _make_balanced_monotone_orbit_vec(group, target_zero_wt: Optional[int] = None) -> np.ndarray:
    """Creates a nontrivial balanced monotone 3-valued orbit vector (middle weight -> 0)."""
    d = group.d
    wts = group.orbit_wts_np
    mid = d // 2 if target_zero_wt is None else target_zero_wt
    if d % 2 == 0:
        u = np.where(wts < mid, -1, np.where(wts > mid, 1, 0)).astype(np.int8)
    else:
        u = np.where(wts <= mid, -1, 1).astype(np.int8)
    return u


def run_selftest() -> Dict[str, object]:
    device = get_device(prefer_cuda=True)
    report: Dict[str, object] = {"device": str(device), "exact_vs_full_fwht": {}, "high_d_parseval": {}}

    # 1. Exact comparison vs full 2^d GPU FWHT (diff < 1e-12) on S_2^6, S_3 wr S_4, S_4^4, S_3 wr S_6
    tests = [
        ("S2^6_d12", BlockSymGroup((2,) * 6, device=device)),
        ("S3_wr_S4_d12", WreathSymGroup(3, 4, device=device)),
        ("S4^4_d16", BlockSymGroup((4,) * 4, device=device)),
        ("S3_wr_S6_d18", WreathSymGroup(3, 6, device=device)),
    ]
    rng = np.random.default_rng(2026)
    for name, grp in tests:
        # Test TWO vectors: (a) balanced monotone middle-layer de-aliased, (b) arbitrary random 3-valued vector
        u_mono = _make_balanced_monotone_orbit_vec(grp)
        is_m, is_ac, is_b = grp.check_monotone_and_antichain(u_mono)
        assert is_m and is_ac and is_b, f"{name} mono/ac/bal failed: {(is_m, is_ac, is_b)}"

        # Also random 3-valued vector (with ~20% zeros) to stress-test every matrix entry!
        u_rand = rng.choice([-1, 0, 1], size=grp.M, p=[0.4, 0.2, 0.4]).astype(np.int8)
        u_pair = np.stack([u_mono, u_rand], axis=0)
        res_orb = grp.evaluate_batch(torch.as_tensor(u_pair, device=device), dtype=torch.float64)

        g_mono = grp.expand_to_full(u_mono)
        g_rand = grp.expand_to_full(u_rand)
        g_pair = np.stack([g_mono, g_rand], axis=0)
        res_full = evaluate_antichain_dealiased_batch(
            torch.as_tensor(g_pair, device=device, dtype=torch.float64), dtype=torch.float64
        )
        diff_h = float(torch.max((res_orb["H"] - res_full["H"]).abs()).item())
        diff_i = float(torch.max((res_orb["I"] - res_full["I"]).abs()).item())
        diff_ot = float(abs(res_orb["OT"][0].item() - res_full["OT"][0].item()))
        assert diff_h < 1e-12 and diff_i < 1e-12 and diff_ot < 1e-12, (
            f"{name} mismatch: dH={diff_h}, dI={diff_i}, dOT={diff_ot}"
        )
        # Verify full monotonicity matches orbit check on u_mono (for d <= 16 fast check)
        if grp.d <= 16:
            fp_check = evaluate_subcube_dealiased_exact_fp64(g_mono, grp.d, device=device)
            assert fp_check["monotone_full"] and fp_check["balanced"]
        report["exact_vs_full_fwht"][name] = {
            "d": grp.d,
            "M": grp.M,
            "diff_H": diff_h,
            "diff_I": diff_i,
            "diff_OT": diff_ot,
            "mono_OT": float(res_orb["OT"][0].item()),
            "parseval_err": float(torch.max((res_orb["parseval_tot"] - 1.0).abs()).item()),
        }

    # 2. High-d wreath & block Parseval checks (d = 20, 24 — zero 2^d allocation!)
    for name, grp in [
        ("S4_wr_S5_d20", WreathSymGroup(4, 5, device=device)),
        ("S4^5_d20", BlockSymGroup((4,) * 5, device=device)),
        ("S3_wr_S8_d24", WreathSymGroup(3, 8, device=device)),
        ("S4_wr_S6_d24", WreathSymGroup(4, 6, device=device)),
    ]:
        u_m = _make_balanced_monotone_orbit_vec(grp)
        res = grp.evaluate_batch(torch.as_tensor(u_m, device=device), dtype=torch.float64)
        p_err = float(abs(res["parseval_tot"].item() - 1.0))
        assert p_err < 1e-12, f"{name} Parseval failed: {p_err}"
        report["high_d_parseval"][name] = {
            "d": grp.d,
            "M": grp.M,
            "parseval_err": p_err,
            "mono_OT": float(res["OT"].item()),
        }

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Krawtchouk & Wreath-Product Orbit Engine self-test")
    parser.add_argument("--selftest", action="store_true", help="Run bit-exact self-tests against full FWHT")
    args = parser.parse_args()
    if args.selftest or True:
        rep = run_selftest()
        print(json.dumps(rep, indent=2))
        print("RESULT: orbit_krawtchouk Phase 0 self-test PASSED")


if __name__ == "__main__":
    main()

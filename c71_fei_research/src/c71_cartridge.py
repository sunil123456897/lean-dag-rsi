"""
C_71 Fourier Entropy-Influence Discovery Cartridge for SLEEP-DAG-RSI v8 on RTX 5060.

Provides:
  1. High-leverage RTX 5060 GPU operators (VRAM strictly <= 4.5 GiB):
       - Exact batched analytic O(2^d) GPU delta-swap steepest ascent + 2-step monotone tunneling
       - 100% GPU-resident parallel tempered SA (pure balanced monotone & 3-valued antichain)
       - Zero-cost disconnected critical-set variable splitting (Delta I == 0, Delta H > 0)
       - Curriculum dimension lifting (duplication, optimal component-split lift d -> d+1,
         and exact monotone balanced ite(x_new, max(u, v), min(u, v)) crossover)
       - Core+Subcube extraction & optimal orthogonal Vector A2 de-aliasing
       - From-scratch perturbed MAJ_9 / MAJ_11 + structural seed loader
  2. Genuine SLEEP-DAG-RSI v8 integration wiring directly into dream_rsi_scripts:
       TaskCartridge, BranchCandidate, AutoDreamRunner, BlindVaultGuard, DiscoveryTree,
       Node, Problem, LeanDAGWorld, LeanDAGPolicy(entropic_tau=4.0),
       compute_max_tilted_objective(tau=4.0), BackwardSubGoalDecomposer,
       MutableSkillState, WaldMicroSentinel.

Strict Zero-Gaming: ZERO external truth tables or published [MI2026]/[Num2026] data loaded.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
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
from fwht_gpu import (  # noqa: E402
    check_core_monotone_and_balanced,
    decompose_zero_subcubes,
    eval_delta_candidates_gpu,
    evaluate_antichain_dealiased_batch,
    evaluate_core_spectrum_batch,
    evaluate_subcube_dealiased_exact_fp64,
    fwht_batched,
    get_device,
    make_degree_table,
)
import exact_verifier  # noqa: E402

NUM2026_STRICT_BAR = 6.52184571092304657566
STEPPING_THRESHOLDS = (6.480, 6.495, 6.510, 6.522, 6.535)


# =============================================================================
# 1. Candidate Container & Fourier Degree-Shell Profiling
# =============================================================================

@dataclass
class FEICandidate:
    """Self-contained balanced monotone Boolean candidate (Tier 1 or Tier 2)."""

    label: str
    d: int
    g0: np.ndarray  # int8 in {-1, 0, +1}^{2^d}
    subcubes: List[Dict[str, int]] = field(default_factory=list)
    OT: float = 0.0
    H: float = 0.0
    I: float = 0.0
    n_effective: int = 0
    degree_shells: List[float] = field(default_factory=list)
    source: str = "from_scratch"

    def is_pure(self) -> bool:
        return len(self.subcubes) == 0 and not np.any(self.g0 == 0)


def evaluate_fei_candidate_fp64(
    g0: np.ndarray,
    d: int,
    label: str = "cand",
    source: str = "from_scratch",
    device: Optional[torch.device] = None,
) -> Optional[FEICandidate]:
    """
    Evaluates g0 in float64 with automatic maximal-subcube decomposition, strict
    balance/monotonicity checks, and Fourier degree-shell profile [W1, W2, W3, W4, W5+].
    """
    dev = device or get_device(True)
    res = evaluate_subcube_dealiased_exact_fp64(g0, d, device=dev)
    if not (res["balanced"] and res["monotone_full"]):
        return None
    # Compute Fourier degree-shell weights for bucket complementarity
    t_g = torch.as_tensor(g0, device=dev, dtype=torch.float64).unsqueeze(0)
    deg = make_degree_table(d, dev, dtype=torch.float64)
    _, _, _, fhat = evaluate_core_spectrum_batch(t_g, deg=deg, dtype=torch.float64)
    p = (fhat.squeeze(0) ** 2)
    shells = []
    for k_deg in (1, 2, 3, 4):
        shells.append(float(p[deg == float(k_deg)].sum().item()))
    shells.append(float(p[deg >= 5.0].sum().item()))
    return FEICandidate(
        label=label,
        d=int(d),
        g0=np.asarray(g0, dtype=np.int8).copy(),
        subcubes=list(res["subcubes"]),  # type: ignore[arg-type]
        OT=float(res["OT"]),
        H=float(res["H"]),
        I=float(res["I"]),
        n_effective=int(res["n_eff"]),
        degree_shells=shells,
        source=source,
    )


def compute_bucket_scores(cand: FEICandidate) -> Tuple[List[float], List[int]]:
    """
    Constructs Node.bucket_scores (continuous margins) and Node.pass_mask (0/1):
      - 5 stepping-stone threshold margins: [L - 6.480, L - 6.495, L - 6.510, L - 6.522, L - 6.535]
      - 5 Fourier degree-shell / influence-ridge complementarity buckets so CROSSOVER(u, v)
        identifies genuinely complementary asymmetric cores!
    """
    margins: List[float] = [round(cand.OT - thr, 6) for thr in STEPPING_THRESHOLDS]
    # Complementary structural buckets:
    # B5: tight influence ridge (I in [2.005, 2.055]) -> positive when inside ridge
    margins.append(round(0.025 - abs(cand.I - 2.030), 6))
    # B6..B9: spectral spread across degree shells W1..W4 (positive when shell has strong spread)
    sh = cand.degree_shells if len(cand.degree_shells) == 5 else [0.0] * 5
    # Target shell profile for high-FEI asymmetric cores: strong W1, W2, W3, W4 balance
    targets = (0.35, 0.22, 0.14, 0.08)
    for val, tgt in zip(sh[:4], targets):
        margins.append(round(val - tgt, 6))
    pass_mask = [1 if m > 0.0 else 0 for m in margins]
    return margins, pass_mask


# =============================================================================
# 2. High-Leverage GPU Operators (RTX 5060, VRAM <= 4.5 GiB)
# =============================================================================

def _boundary_masks_gpu(g_1d: torch.Tensor, d: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Exact monotone boundary indices for a single 3-valued core g_1d [2^d] on GPU:
      - strict_min_pos: g == +1 and all lower neighbors == -1
      - strict_max_neg: g == -1 and all upper neighbors == +1
      - zeros: g == 0
    """
    mp = g_1d == 1
    mn = g_1d == -1
    for b in range(d):
        st = 1 << b
        vg = g_1d.view(-1, 2, st)
        mp.view(-1, 2, st)[:, 1, :] &= (vg[:, 0, :] == -1)
        mn.view(-1, 2, st)[:, 0, :] &= (vg[:, 1, :] == 1)
    return torch.nonzero(mp, as_tuple=False).squeeze(1), torch.nonzero(mn, as_tuple=False).squeeze(1), torch.nonzero(g_1d == 0, as_tuple=False).squeeze(1)


def gpu_delta_steepest_ascent(
    g_init: np.ndarray,
    d: int,
    device: torch.device,
    rng: np.random.Generator,
    max_iters: int = 80,
    batch_candidates: int = 256,
    allow_antichain: bool = True,
    allow_tunneling: bool = True,
    max_tunnels: int = 6,
) -> Tuple[np.ndarray, float, int]:
    """
    Exact batched analytic O(2^d) GPU delta-swap steepest ascent + 2-step monotone tunneling!
    Uses eval_delta_candidates_gpu() to evaluate up to `batch_candidates` balanced monotone
    boundary swaps/antichain moves per iteration in parallel against ONE frozen parent
    spectrum fhat (zero full FWHT per swap!).
    """
    n_pts = 1 << d
    # Use float32 for d >= 15 speed (within 3e-7) and float64 for d <= 14 exactness
    work_dtype = torch.float64 if d <= 14 else torch.float32
    deg = make_degree_table(d, device, dtype=work_dtype)
    g_cur = torch.as_tensor(g_init, device=device, dtype=torch.int8).clone()

    def _compute_parent_spec(g_t: torch.Tensor) -> Tuple[torch.Tensor, float]:
        fhat = fwht_batched(g_t.to(dtype=work_dtype)) / float(n_pts)
        p = fhat * fhat
        a_cnt = int((g_t == 0).sum().item())
        inv_n = 1.0 / float(n_pts)
        eps = 1e-300 if work_dtype == torch.float64 else 1e-38
        i_tot = float((p * deg).sum().item()) + a_cnt * inv_n * (1.0 + 0.5 * d)
        h_tot = float(-torch.where(p > 0.0, p * torch.log2(p.clamp_min(eps)), torch.zeros_like(p)).sum().item()) + a_cnt * inv_n * (2.0 * d)
        ot = h_tot / (i_tot - 1.0) if (abs(float(fhat[0].item())) < 1e-7 and i_tot > 1.0 + 1e-9) else 0.0
        return fhat, ot

    fhat_cur, cur_ot = _compute_parent_spec(g_cur)
    best_g = g_cur.clone()
    best_ot = cur_ot
    total_evals = 1
    tunnels_used = 0
    # Bound candidate batch size on high d to stay well under 4.5 GiB VRAM
    eff_batch = min(batch_candidates, 512 if d <= 12 else (256 if d <= 14 else (128 if d <= 16 else 64)))

    for _ in range(max_iters):
        mp, mn, zs = _boundary_masks_gpu(g_cur, d)
        mp_np = mp.detach().cpu().numpy()
        mn_np = mn.detach().cpu().numpy()
        zs_np = zs.detach().cpu().numpy()
        n_mp, n_mn, n_z = len(mp_np), len(mn_np), len(zs_np)
        if n_mp == 0 and n_mn == 0 and n_z < 2:
            break

        # Assemble candidate delta batch [B, 2]
        d_idx_list: List[Tuple[int, int]] = []
        d_val_list: List[Tuple[float, float]] = []
        a_dlt_list: List[float] = []
        move_ops: List[Tuple[int, int, int, int]] = []  # (x1, new_v1, x2, new_v2)

        # 1. Pure swaps (+1, -1) -> (-1, +1): exhaustive if small, else stratified sample!
        if n_mp > 0 and n_mn > 0:
            max_swaps = eff_batch if not allow_antichain else int(eff_batch * 0.55)
            if n_mp * n_mn <= max_swaps:
                for a in mp_np:
                    ai = int(a)
                    for b in mn_np:
                        bi = int(b)
                        if (bi & ai) != bi:
                            d_idx_list.append((ai, bi))
                            d_val_list.append((-2.0, 2.0))
                            a_dlt_list.append(0.0)
                            move_ops.append((ai, -1, bi, 1))
            else:
                for _ in range(max_swaps * 2):
                    if len(d_idx_list) >= max_swaps:
                        break
                    ai = int(mp_np[int(rng.integers(0, n_mp))])
                    bi = int(mn_np[int(rng.integers(0, n_mn))])
                    if (bi & ai) != bi:
                        d_idx_list.append((ai, bi))
                        d_val_list.append((-2.0, 2.0))
                        a_dlt_list.append(0.0)
                        move_ops.append((ai, -1, bi, 1))

        # 2. Antichain expand (+1,-1)->(0,0), contract (0,0)->(+1,-1), slide zero
        if allow_antichain:
            rem = eff_batch - len(d_idx_list)
            if rem > 0 and n_mp > 0 and n_mn > 0:
                n_exp = max(1, int(rem * 0.45))
                added = 0
                for _ in range(n_exp * 2):
                    if added >= n_exp or len(d_idx_list) >= eff_batch:
                        break
                    ai = int(mp_np[int(rng.integers(0, n_mp))])
                    bi = int(mn_np[int(rng.integers(0, n_mn))])
                    if (bi & ai) != bi:
                        d_idx_list.append((ai, bi))
                        d_val_list.append((-1.0, 1.0))
                        a_dlt_list.append(2.0)
                        move_ops.append((ai, 0, bi, 0))
                        added += 1
            if len(d_idx_list) < eff_batch and n_z >= 2:
                n_con = min(eff_batch - len(d_idx_list), max(1, eff_batch // 6))
                for _ in range(n_con):
                    i1 = int(rng.integers(0, n_z))
                    i2 = int(rng.integers(0, n_z - 1))
                    if i2 >= i1:
                        i2 += 1
                    z1, z2 = int(zs_np[i1]), int(zs_np[i2])
                    d_idx_list.append((z1, z2))
                    d_val_list.append((1.0, -1.0))
                    a_dlt_list.append(-2.0)
                    move_ops.append((z1, 1, z2, -1))
            if len(d_idx_list) < eff_batch and n_z >= 1 and (n_mp > 0 or n_mn > 0):
                while len(d_idx_list) < eff_batch:
                    zi = int(zs_np[int(rng.integers(0, n_z))])
                    if n_mp > 0 and (n_mn == 0 or rng.random() < 0.5):
                        ai = int(mp_np[int(rng.integers(0, n_mp))])
                        d_idx_list.append((zi, ai))
                        d_val_list.append((1.0, -1.0))
                        a_dlt_list.append(0.0)
                        move_ops.append((zi, 1, ai, 0))
                    elif n_mn > 0:
                        bi = int(mn_np[int(rng.integers(0, n_mn))])
                        d_idx_list.append((zi, bi))
                        d_val_list.append((-1.0, 1.0))
                        a_dlt_list.append(0.0)
                        move_ops.append((zi, -1, bi, 0))
                    else:
                        break

        if not d_idx_list:
            break

        t_idx = torch.as_tensor(d_idx_list, device=device, dtype=torch.int32)
        t_val = torch.as_tensor(d_val_list, device=device, dtype=work_dtype)
        t_adlt = torch.as_tensor(a_dlt_list, device=device, dtype=work_dtype)
        _, _, ot_cands = eval_delta_candidates_gpu(
            fhat_cur, deg, t_idx, t_val, n_z, t_adlt, d
        )
        total_evals += len(d_idx_list)

        best_k = int(ot_cands.argmax().item())
        cand_ot = float(ot_cands[best_k].item())

        if cand_ot > cur_ot + 1e-8:
            x1, v1, x2, v2 = move_ops[best_k]
            g_cur[x1] = v1
            g_cur[x2] = v2
            fhat_cur, cur_ot = _compute_parent_spec(g_cur)
            if cur_ot > best_ot + 1e-8:
                best_ot = cur_ot
                best_g = g_cur.clone()
        elif allow_tunneling and tunnels_used < max_tunnels:
            # 2-step monotone tunneling: accept a near-neutral move (within -0.004) to cross saddle!
            tunnels_used += 1
            if cand_ot >= cur_ot - 0.004:
                x1, v1, x2, v2 = move_ops[best_k]
                g_cur[x1] = v1
                g_cur[x2] = v2
                fhat_cur, cur_ot = _compute_parent_spec(g_cur)
            else:
                break
        else:
            break

    g_out = best_g.detach().cpu().numpy()
    r64 = evaluate_subcube_dealiased_exact_fp64(g_out, d, device=device)
    if r64["balanced"] and r64["monotone_full"]:
        return g_out, float(r64["OT"]), total_evals
    return g_init.astype(np.int8).copy(), float(evaluate_subcube_dealiased_exact_fp64(g_init, d, device=device)["OT"]), total_evals


def optimize_mixed_12_hosts_on_core(
    g_core: np.ndarray,
    d: int,
    device: torch.device,
    rng: np.random.Generator,
    max_rounds: int = 60,
    batch_limit: int = 384,
    allow_sa_kicks: bool = True,
) -> Tuple[np.ndarray, Optional[FEICandidate]]:
    """
    GPU-batched Iterated Local Search (steepest ascent + SA tunneling) for placing,
    relocating, and co-optimizing 2-cell subcube hosts (codim d-1), 1-cell antichain hosts
    (codim d), AND core boundary swaps directly on top of a monotone core g_core (d = 9..13)!
    Generalizes the [MI2026b]/[Num2026] 'perturbed core + 2-cell hosts' architecture to ANY d
    and ANY number of orthogonal hosts k_aux at O(2^d) GPU cost!
    """
    n_pts = 1 << d
    deg_f64 = make_degree_table(d, device, dtype=torch.float64)
    g_cur = np.asarray(g_core, dtype=np.int8).copy()
    best_cand = evaluate_fei_candidate_fp64(g_cur, d, label=f"host12_d{d}", source="mixed_12_hosts", device=device)
    if best_cand is None:
        return g_cur, None
    best_ot = best_cand.OT
    cur_ot = best_ot
    g_best = g_cur.copy()
    stagnation = 0
    kicks_left = 10 if allow_sa_kicks else 0

    for _ in range(max_rounds):
        # Compute neighbor violation counts on g_cur
        bad_lo = np.zeros(n_pts, dtype=np.int16)      # lower neighbors != -1
        bad_lo_pos = np.zeros(n_pts, dtype=np.int16)  # lower neighbors == +1
        bad_hi = np.zeros(n_pts, dtype=np.int16)      # upper neighbors != +1
        bad_hi_neg = np.zeros(n_pts, dtype=np.int16)  # upper neighbors == -1
        deg_z_arr = np.zeros(n_pts, dtype=np.int16)
        z_partner = np.full(n_pts, -1, dtype=np.int32)
        for b in range(d):
            st = 1 << b
            vg = g_cur.reshape(-1, 2, st)
            bad_lo.reshape(-1, 2, st)[:, 1, :] += (vg[:, 0, :] != -1)
            bad_lo_pos.reshape(-1, 2, st)[:, 1, :] += (vg[:, 0, :] == 1)
            bad_hi.reshape(-1, 2, st)[:, 0, :] += (vg[:, 1, :] != 1)
            bad_hi_neg.reshape(-1, 2, st)[:, 0, :] += (vg[:, 1, :] == -1)
            both_z = (vg[:, 0, :] == 0) & (vg[:, 1, :] == 0)
            if np.any(both_z):
                dz = deg_z_arr.reshape(-1, 2, st)
                dz[:, 0, :] += both_z
                dz[:, 1, :] += both_z
                hi_i, lo_i = np.nonzero(both_z)
                for h_i, l_i in zip(hi_i, lo_i):
                    x0 = int((h_i << (b + 1)) | l_i)
                    x1 = x0 | st
                    z_partner[x0] = x1
                    z_partner[x1] = x0

        # Strict 1-cell boundary sets (incomparable with existing zeros)
        mp1 = np.flatnonzero((g_cur == 1) & (bad_lo == 0))
        mn1 = np.flatnonzero((g_cur == -1) & (bad_hi == 0))
        # Relaxed 1-cell sets (allowed to touch ONE existing isolated zero -> merges into a 2-cell subcube!)
        mp1_merge = np.flatnonzero((g_cur == 1) & (bad_lo_pos == 0) & (bad_lo == 1))
        mn1_merge = np.flatnonzero((g_cur == -1) & (bad_hi_neg == 0) & (bad_hi == 1))
        # Existing 1-cell zeros & 2-cell zero edges (for contraction / relocation!)
        z1_pts = np.flatnonzero((g_cur == 0) & (deg_z_arr == 0))
        z2_edges: List[Tuple[int, int]] = []
        for x in np.flatnonzero((g_cur == 0) & (deg_z_arr == 1)):
            xi = int(x)
            pi = int(z_partner[xi])
            if xi < pi:
                z2_edges.append((xi, pi))

        # Strict 2-cell boundary edges (codim d-1 subcubes, disjoint from existing zeros)
        p_edges: List[Tuple[int, int]] = []
        n_edges: List[Tuple[int, int]] = []
        for b in range(d):
            st = 1 << b
            vg = g_cur.reshape(-1, 2, st)
            bl = bad_lo.reshape(-1, 2, st)
            bh = bad_hi.reshape(-1, 2, st)
            ok_p = (vg[:, 0, :] == 1) & (vg[:, 1, :] == 1) & (bl[:, 0, :] == 0) & (bl[:, 1, :] == 1)
            if np.any(ok_p):
                hi_i, lo_i = np.nonzero(ok_p)
                for h_i, l_i in zip(hi_i, lo_i):
                    x0 = int((h_i << (b + 1)) | l_i)
                    p_edges.append((x0, x0 | st))
            ok_n = (vg[:, 0, :] == -1) & (vg[:, 1, :] == -1) & (bh[:, 1, :] == 0) & (bh[:, 0, :] == 1)
            if np.any(ok_n):
                hi_i, lo_i = np.nonzero(ok_n)
                for h_i, l_i in zip(hi_i, lo_i):
                    x0 = int((h_i << (b + 1)) | l_i)
                    n_edges.append((x0, x0 | st))

        cands_list: List[np.ndarray] = []

        # Move Type 1 (35%): Add 2-cell subcube host pair (+1,+1)->0 & (-1,-1)->0
        if p_edges and n_edges:
            quota = int(batch_limit * 0.35)
            for _ in range(quota * 2):
                if len(cands_list) >= quota:
                    break
                p0, p1 = p_edges[int(rng.integers(0, len(p_edges)))]
                n0, n1 = n_edges[int(rng.integers(0, len(n_edges)))]
                if (n0 & p1) == n0:
                    continue
                arr = g_cur.copy()
                arr[p0] = 0; arr[p1] = 0; arr[n0] = 0; arr[n1] = 0
                cands_list.append(arr)

        # Move Type 2 (20%): Expand 1-cell zero into 2-cell subcube (using mp1_merge / mn1_merge!)
        q2_target = len(cands_list) + int(batch_limit * 0.20)
        if (len(mp1_merge) > 0 and (len(mn1) > 0 or len(mn1_merge) > 0)) or (len(mn1_merge) > 0 and len(mp1) > 0):
            all_p = np.concatenate([mp1_merge, mp1]) if len(mp1_merge) > 0 else mp1
            all_n = np.concatenate([mn1_merge, mn1]) if len(mn1_merge) > 0 else mn1
            for _ in range(int(batch_limit * 0.40)):
                if len(cands_list) >= q2_target:
                    break
                if len(mp1_merge) > 0 and (len(mn1_merge) == 0 or rng.random() < 0.5):
                    a = int(mp1_merge[int(rng.integers(0, len(mp1_merge)))])
                    b = int(all_n[int(rng.integers(0, len(all_n)))])
                else:
                    a = int(all_p[int(rng.integers(0, len(all_p)))])
                    b = int(mn1_merge[int(rng.integers(0, len(mn1_merge)))])
                if (b & a) == b:
                    continue
                arr = g_cur.copy()
                arr[a] = 0; arr[b] = 0
                cands_list.append(arr)

        # Move Type 3 (15%): Relocate / contract 2-cell hosts (z2_edge -> +1/-1 paired with opposite edge or z2_edge)
        q3_target = len(cands_list) + int(batch_limit * 0.15)
        if len(z2_edges) >= 1:
            for _ in range(int(batch_limit * 0.30)):
                if len(cands_list) >= q3_target:
                    break
                r_t = float(rng.random())
                if r_t < 0.40 and len(z2_edges) >= 2:
                    i1 = int(rng.integers(0, len(z2_edges)))
                    i2 = int(rng.integers(0, len(z2_edges) - 1))
                    if i2 >= i1:
                        i2 += 1
                    e1, e2 = z2_edges[i1], z2_edges[i2]
                    arr = g_cur.copy()
                    arr[e1[0]] = 1; arr[e1[1]] = 1; arr[e2[0]] = -1; arr[e2[1]] = -1
                    cands_list.append(arr)
                elif r_t < 0.70 and p_edges:
                    e1 = z2_edges[int(rng.integers(0, len(z2_edges)))]
                    ep = p_edges[int(rng.integers(0, len(p_edges)))]
                    arr = g_cur.copy()
                    arr[e1[0]] = 1; arr[e1[1]] = 1; arr[ep[0]] = 0; arr[ep[1]] = 0
                    cands_list.append(arr)
                elif n_edges:
                    e1 = z2_edges[int(rng.integers(0, len(z2_edges)))]
                    en = n_edges[int(rng.integers(0, len(n_edges)))]
                    arr = g_cur.copy()
                    arr[e1[0]] = -1; arr[e1[1]] = -1; arr[en[0]] = 0; arr[en[1]] = 0
                    cands_list.append(arr)

        # Move Type 4 (30%): 1-cell host add/contract/slide + pure core boundary 2-swaps!
        if len(mp1) > 0 or len(mn1) > 0 or len(z1_pts) >= 2:
            for _ in range(batch_limit):
                if len(cands_list) >= batch_limit:
                    break
                rm = float(rng.random())
                if rm < 0.45 and len(mp1) > 0 and len(mn1) > 0:
                    a = int(mp1[int(rng.integers(0, len(mp1)))])
                    b = int(mn1[int(rng.integers(0, len(mn1)))])
                    if (b & a) == b:
                        continue
                    arr = g_cur.copy()
                    arr[a] = -1; arr[b] = 1
                    cands_list.append(arr)
                elif rm < 0.75 and len(mp1) > 0 and len(mn1) > 0:
                    a = int(mp1[int(rng.integers(0, len(mp1)))])
                    b = int(mn1[int(rng.integers(0, len(mn1)))])
                    if (b & a) == b:
                        continue
                    arr = g_cur.copy()
                    arr[a] = 0; arr[b] = 0
                    cands_list.append(arr)
                elif rm < 0.88 and len(z1_pts) >= 2:
                    i1 = int(rng.integers(0, len(z1_pts)))
                    i2 = int(rng.integers(0, len(z1_pts) - 1))
                    if i2 >= i1:
                        i2 += 1
                    arr = g_cur.copy()
                    arr[int(z1_pts[i1])] = 1; arr[int(z1_pts[i2])] = -1
                    cands_list.append(arr)
                elif len(z1_pts) >= 1 and (len(mp1) > 0 or len(mn1) > 0):
                    zi = int(z1_pts[int(rng.integers(0, len(z1_pts)))])
                    arr = g_cur.copy()
                    if len(mp1) > 0 and (len(mn1) == 0 or rng.random() < 0.5):
                        arr[zi] = 1; arr[int(mp1[int(rng.integers(0, len(mp1)))])] = 0
                    else:
                        arr[zi] = -1; arr[int(mn1[int(rng.integers(0, len(mn1)))])] = 0
                    cands_list.append(arr)

        if not cands_list:
            break

        batch_t = torch.as_tensor(np.stack(cands_list, axis=0), device=device, dtype=torch.int8)
        c_bsz = batch_t.shape[0]
        mean, h_sub, i_g0, _ = evaluate_core_spectrum_batch(batch_t, deg=deg_f64, dtype=torch.float64)

        # Strictly verify BOTH core monotonicity AND deg_z <= 1 on GPU!
        z_mask = batch_t == 0
        deg_z = torch.zeros((c_bsz, n_pts), device=device, dtype=torch.int8)
        edges_zz = torch.zeros(c_bsz, device=device, dtype=torch.float64)
        mono_ok = torch.ones(c_bsz, device=device, dtype=torch.bool)
        for b in range(d):
            st = 1 << b
            vb = batch_t.view(c_bsz, -1, 2, st)
            mono_ok &= (vb[:, :, 0, :] <= vb[:, :, 1, :]).all(dim=(1, 2))
            vz = z_mask.view(c_bsz, -1, 2, st)
            both = vz[:, :, 0, :] & vz[:, :, 1, :]
            edges_zz += both.sum(dim=(1, 2)).double()
            dz = deg_z.view(c_bsz, -1, 2, st)
            dz[:, :, 0, :] += both.to(torch.int8)
            dz[:, :, 1, :] += both.to(torch.int8)

        valid_sc = mono_ok & (deg_z <= 1).all(dim=1) & (mean.abs() < 1e-9)
        z_cnt = z_mask.sum(dim=1).double()
        inv_n = 1.0 / float(n_pts)
        i_tot = i_g0 + (z_cnt * (1.0 + 0.5 * d) - edges_zz) * inv_n
        h_tot = h_sub + (z_cnt * (2.0 * d) - 4.0 * edges_zz) * inv_n
        ot_vals = torch.where(valid_sc & (i_tot > 1.0 + 1e-9), h_tot / (i_tot - 1.0).clamp_min(1e-12), torch.zeros_like(h_tot))

        best_k = int(ot_vals.argmax().item())
        cand_ot = float(ot_vals[best_k].item())
        if cand_ot > cur_ot + 1e-9:
            g_cur = cands_list[best_k]
            cur_ot = cand_ot
            stagnation = 0
            if cur_ot > best_ot + 1e-9:
                cand_obj = evaluate_fei_candidate_fp64(
                    g_cur, d, label=f"host12_d{d}", source="mixed_12_hosts", device=device
                )
                if cand_obj is not None and cand_obj.OT > best_ot + 1e-9:
                    g_best = g_cur.copy()
                    best_cand = cand_obj
                    best_ot = cand_obj.OT
        elif kicks_left > 0 and cand_ot >= cur_ot - 0.005:
            # Tunnel across saddle!
            kicks_left -= 1
            stagnation += 1
            g_cur = cands_list[best_k]
            cur_ot = cand_ot
            if stagnation >= 4:
                g_cur = g_best.copy()
                cur_ot = best_ot
                stagnation = 0
        else:
            break

    return g_best, best_cand


def find_disconnected_critical_components(g: np.ndarray, d: int, bit_m: int) -> List[List[int]]:
    """
    Finds all connected components of the critical set
      R_m = {x_{setminus m} in Q_{d-1} : g(x_{setminus m}, 0) != g(x_{setminus m}, 1)}
    in Q_{d-1} for a pure {-1, +1} monotone function g.
    Theorem: For any monotone g, disconnected components of R_m are automatically mutually
    incomparable, so splitting x_m across fresh variables has Delta I == 0, preserves
    monotonicity and balance, and strictly increases H whenever Fourier blocks interfere!
    """
    st = 1 << bit_m
    v = g.reshape(-1, 2, st)
    crit = (v[:, 0, :] != v[:, 1, :]).reshape(-1)
    pts = set(int(i) for i in np.flatnonzero(crit))
    comps: List[List[int]] = []
    d_sub = d - 1
    while pts:
        start = next(iter(pts))
        pts.remove(start)
        q = deque([start])
        comp = [start]
        while q:
            u = q.popleft()
            for b in range(d_sub):
                nb = u ^ (1 << b)
                if nb in pts:
                    pts.remove(nb)
                    q.append(nb)
                    comp.append(nb)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def split_one_var_lift_d_plus_1(
    g: np.ndarray,
    d: int,
    device: torch.device,
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Exact zero-influence-cost curriculum dimension lift (d -> d+1) generalizing [Num2026]'s
    4-cell -> 2+2 auxiliary split to ANY balanced monotone pure function g on Q_d!
    Scans all coordinates m=0..d-1 with >= 2 disconnected critical components R_m = R_1 U ... U R_c,
    partitions the components into two halves (kept on bit m vs moved to new bit d), and selects
    the split maximizing Delta H (with exact Delta I == 0, 100% monotone, 100% balanced!).
    """
    assert not np.any(g == 0), "split_one_var_lift expects pure {-1, +1} function"
    r0 = evaluate_subcube_dealiased_exact_fp64(g, d, device=device)
    base_ot = float(r0["OT"])
    best_g = np.concatenate([g, g]).astype(np.int8)  # fallback: plain duplication
    best_ot = base_ot
    best_info: Dict[str, Any] = {"split_bit": None, "gain_OT": 0.0, "n_comps": 1}

    n_pts = 1 << d
    for m in range(d):
        comps = find_disconnected_critical_components(g, d, m)
        if len(comps) < 2:
            continue
        # Partition components into two sets (keep largest in A, greedily balance sizes into A vs B)
        set_a: List[int] = []
        set_b: List[int] = []
        sa, sb = 0, 0
        for comp in comps:
            if sa <= sb:
                set_a.extend(comp)
                sa += len(comp)
            else:
                set_b.extend(comp)
                sb += len(comp)
        if not set_a or not set_b:
            continue

        # Construct lifted truth table on d+1 variables:
        # On R_m (set_b), replace dependence on x_m with dependence on new bit x_d!
        # Outside set_b, output is independent of x_d (equal to original g).
        # Specifically: in Q_{d-1} index idx_sub, map back to Q_d indices (x_lo with bit m=0, x_hi with bit m=1):
        low_mask = (1 << m) - 1
        g_new0 = g.copy()  # slice x_d = 0
        g_new1 = g.copy()  # slice x_d = 1
        for idx_sub in set_b:
            lo_part = idx_sub & low_mask
            hi_part = idx_sub >> m
            x_m0 = (hi_part << (m + 1)) | lo_part
            x_m1 = x_m0 | (1 << m)
            # On set_b, original had g(x_m0)=-1, g(x_m1)=+1; now both depend ONLY on x_d!
            g_new0[x_m0] = -1
            g_new0[x_m1] = -1
            g_new1[x_m0] = 1
            g_new1[x_m1] = 1
        cand = np.concatenate([g_new0, g_new1]).astype(np.int8)
        rc = evaluate_subcube_dealiased_exact_fp64(cand, d + 1, device=device)
        if rc["balanced"] and rc["monotone_full"] and float(rc["OT"]) > best_ot + 1e-12:
            best_ot = float(rc["OT"])
            best_g = cand
            best_info = {
                "split_bit": m,
                "gain_OT": round(best_ot - base_ot, 8),
                "n_comps": len(comps),
                "sizes_A_B": [sa, sb],
            }

    return best_g, best_ot, best_info


def greedy_multi_split_up_to_n(
    g: np.ndarray,
    d: int,
    max_n: int,
    device: torch.device,
) -> Tuple[np.ndarray, int, float, List[Dict[str, Any]]]:
    """
    Repeatedly applies exact zero-influence-cost variable splitting (d -> d+1 -> ... <= max_n)
    as long as disconnected critical components yield strict Delta H > 0!
    """
    cur_g = g.astype(np.int8).copy()
    cur_d = d
    cur_ot = float(evaluate_subcube_dealiased_exact_fp64(cur_g, cur_d, device=device)["OT"])
    history: List[Dict[str, Any]] = []
    while cur_d < max_n:
        nxt_g, nxt_ot, info = split_one_var_lift_d_plus_1(cur_g, cur_d, device=device)
        if nxt_ot > cur_ot + 1e-9 and info["split_bit"] is not None:
            cur_g = nxt_g
            cur_d += 1
            cur_ot = nxt_ot
            history.append({"new_d": cur_d, "OT": round(cur_ot, 8), **info})
        else:
            break
    return cur_g, cur_d, cur_ot, history


def extract_core_subcube_dealiasing(
    g: np.ndarray,
    d: int,
    device: torch.device,
    inf_threshold: float = 0.05,
) -> Optional[FEICandidate]:
    """
    Extracts the high-influence core variables C_core (Inf_m > inf_threshold) from a pure
    d-variable balanced monotone function g, projects out the low-influence auxiliary tail
    variables by replacing balanced tail slices with 0, and applies FULL orthogonal maximal-subcube
    de-aliasing (Vector A2) on Q_{|C_core|}!
    Note: only succeeds cleanly if every core cell's tail slice is constant (-1/+1) or exact mean-0.
    """
    # Compute per-coordinate influences
    infs = []
    for m in range(d):
        st = 1 << m
        v = g.reshape(-1, 2, st)
        infs.append(float(np.mean(v[:, 0, :] != v[:, 1, :])))
    core_bits = [m for m, val in enumerate(infs) if val > inf_threshold]
    aux_bits = [m for m, val in enumerate(infs) if val <= inf_threshold]
    if not aux_bits or len(core_bits) < 4 or len(core_bits) > 16:
        return None

    # Permute axes of g so core_bits come first (LSB) and aux_bits last
    # Reshape to (2,)*d with bit (d-1) outermost, bit 0 innermost -> reverse order
    arr = g.reshape((2,) * d)
    # Bit m corresponds to axis (d - 1 - m)
    perm_axes = [d - 1 - m for m in reversed(aux_bits)] + [d - 1 - m for m in reversed(core_bits)]
    arr_p = np.transpose(arr, perm_axes).reshape(1 << len(aux_bits), 1 << len(core_bits))
    col_means = arr_p.mean(axis=0)

    # Check if every core cell is constant (+1/-1) or exact mean 0 (balanced auxiliary host!)
    g_core = np.zeros(1 << len(core_bits), dtype=np.int8)
    for i, cm in enumerate(col_means):
        if abs(cm - 1.0) < 1e-9:
            g_core[i] = 1
        elif abs(cm + 1.0) < 1e-9:
            g_core[i] = -1
        elif abs(cm) < 1e-9:
            g_core[i] = 0
        else:
            # Non-balanced partial auxiliary (e.g. AND/OR of 2 aux vars): snap sign or abort
            return None

    return evaluate_fei_candidate_fp64(
        g_core, len(core_bits),
        label=f"core_d{len(core_bits)}_extracted_from_d{d}",
        source="core_subcube_extract",
        device=device,
    )


def align_coordinates_by_influence(g: np.ndarray, d: int) -> np.ndarray:
    """Permutes coordinates of g in descending order of coordinate influence Inf_m(g)."""
    infs = []
    for m in range(d):
        st = 1 << m
        v = g.reshape(-1, 2, st)
        infs.append(float(np.mean(v[:, 0, :] != v[:, 1, :])))
    # Order bits so bit 0 has highest influence, ..., bit d-1 lowest
    order = sorted(range(d), key=lambda m: -infs[m])
    if order == list(range(d)):
        return g.copy()
    arr = g.reshape((2,) * d)
    # New bit k (axis d-1-k) comes from old bit order[k] (axis d-1-order[k])
    perm_axes = [d - 1 - order[k] for k in reversed(range(d))]
    return np.transpose(arr, perm_axes).reshape(-1).astype(np.int8)


def monotone_crossover_ite(
    g_u: np.ndarray,
    d_u: int,
    g_v: np.ndarray,
    d_v: int,
    device: torch.device,
) -> Optional[Tuple[np.ndarray, int]]:
    """
    Exact balanced monotone crossover operator for DiscoveryTree CROSSOVER(u, v):
      1. Lifts both parents to common dimension d = max(d_u, d_v) (resolving zeros if any).
      2. Aligns coordinate influence ordering of both parents so their cores overlap!
      3. Constructs F(x, x_new) = ite(x_new, max(f_u(x), f_v(x)), min(f_u(x), f_v(x))) on d+1 bits.
    Theorem: Automatically 100% logic-monotone and 100% strictly balanced for ANY two balanced
    monotone functions f_u, f_v! If f_u == f_v after alignment, tries cyclic shift of f_v.
    """
    d = max(d_u, d_v)
    if d >= 18:
        return None

    def _to_pure(g_raw: np.ndarray, ds: int, dt: int) -> np.ndarray:
        arr = g_raw.astype(np.int8)
        cur = ds
        if np.any(arr == 0) and cur < dt:
            arr = np.concatenate([np.where(arr == 0, -1, arr), np.where(arr == 0, 1, arr)]).astype(np.int8)
            cur += 1
        elif np.any(arr == 0):
            # Resolve zeros symmetrically by index
            zs = np.flatnonzero(arr == 0)
            arr = arr.copy()
            arr[zs[: len(zs) // 2]] = -1
            arr[zs[len(zs) // 2 :]] = 1
        while cur < dt:
            arr = np.concatenate([arr, arr]).astype(np.int8)
            cur += 1
        return arr

    fu = align_coordinates_by_influence(_to_pure(g_u, d_u, d), d)
    fv = align_coordinates_by_influence(_to_pure(g_v, d_v, d), d)
    if np.array_equal(fu, fv):
        # Swap top two bits of fv to break exact identity
        arr_v = fv.reshape((2,) * d)
        axes = list(range(d))
        axes[-1], axes[-2] = axes[-2], axes[-1]
        fv = np.transpose(arr_v, axes).reshape(-1).astype(np.int8)

    f_lo = np.minimum(fu, fv).astype(np.int8)
    f_hi = np.maximum(fu, fv).astype(np.int8)
    f_xover = np.concatenate([f_lo, f_hi]).astype(np.int8)
    is_m, is_b = check_core_monotone_and_balanced(f_xover, d + 1)
    if is_m and is_b:
        return f_xover, d + 1
    return None


def gpu_parallel_pure_sa_fast(
    d: int,
    init_chains_np: np.ndarray,
    device: torch.device,
    steps: int = 8000,
    t_high: float = 0.18,
    t_low: float = 1e-4,
    seed: int = 2026,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    100% GPU-resident Parallel Tempered SA over pure balanced monotone functions on Q_d.
    """
    torch.manual_seed(seed)
    n_chains, n_pts = init_chains_np.shape
    deg = make_degree_table(d, device, dtype=torch.float32)

    chains = torch.as_tensor(init_chains_np, device=device, dtype=torch.int8)
    _, h0, i0, _ = evaluate_core_spectrum_batch(chains, deg=deg, dtype=torch.float32)
    cur_ot = torch.where(i0 > 1.0 + 1e-6, h0 / (i0 - 1.0).clamp_min(1e-6), torch.zeros_like(h0))
    best_ot = cur_ot.clone()
    best_chains = chains.clone()

    # Log-spaced temperature ladder: bottom 25% cool (local polish), top 75% warm->cold (exploration)
    chain_t0 = torch.linspace(math.log(max(t_low * 10.0, t_high * 0.05)), math.log(t_high), n_chains, device=device).exp()
    decay = (torch.full_like(chain_t0, t_low) / chain_t0).pow(1.0 / max(1, steps - 1))
    temps = chain_t0.clone()
    row_idx = torch.arange(n_chains, device=device)

    for step in range(steps):
        mp = chains == 1
        mn = chains == -1
        for b in range(d):
            st = 1 << b
            vc = chains.view(n_chains, -1, 2, st)
            mp.view(n_chains, -1, 2, st)[:, :, 1, :] &= (vc[:, :, 0, :] == -1)
            mn.view(n_chains, -1, 2, st)[:, :, 0, :] &= (vc[:, :, 1, :] == 1)

        a_idx = (torch.rand((n_chains, n_pts), device=device) * mp.float()).argmax(dim=1)
        b_idx = (torch.rand((n_chains, n_pts), device=device) * mn.float()).argmax(dim=1)
        valid = mp.any(dim=1) & mn.any(dim=1) & ((b_idx & a_idx) != b_idx)

        prop = chains.clone()
        prop[row_idx[valid], a_idx[valid]] = -1
        prop[row_idx[valid], b_idx[valid]] = 1

        _, hp, ip, _ = evaluate_core_spectrum_batch(prop, deg=deg, dtype=torch.float32)
        prop_ot = torch.where(ip > 1.0 + 1e-6, hp / (ip - 1.0).clamp_min(1e-6), torch.zeros_like(hp))

        delta = prop_ot - cur_ot
        acc = valid & ((delta >= 0.0) | (torch.rand(n_chains, device=device) < torch.exp((delta / temps).clamp(-40.0, 0.0))))
        chains = torch.where(acc.unsqueeze(1), prop, chains)
        cur_ot = torch.where(acc, prop_ot, cur_ot)

        imp = cur_ot > best_ot
        if imp.any():
            best_ot = torch.where(imp, cur_ot, best_ot)
            best_chains = torch.where(imp.unsqueeze(1), chains, best_chains)

        temps.mul_(decay)

        # Elite migration at 40% and 75% of steps (re-seeds bottom third from top-6 elites)
        if step in (int(steps * 0.40), int(steps * 0.75)):
            k_top = min(6, n_chains)
            k_bot = min(n_chains // 3, n_chains - k_top)
            if k_bot > 0:
                top_idx = torch.topk(best_ot, k=k_top).indices
                bot_idx = torch.topk(best_ot, k=k_bot, largest=False).indices
                src = top_idx[torch.arange(k_bot, device=device) % k_top]
                chains[bot_idx] = best_chains[src]
                cur_ot[bot_idx] = best_ot[src]

    return (
        best_chains.detach().cpu().numpy(),
        best_ot.detach().cpu().numpy().astype(np.float64),
        int(n_chains * (steps + 1)),
    )


def make_perturbed_odd_maj_seed(d: int, rng: np.random.Generator, asym_strength: float = 0.22) -> np.ndarray:
    """From-scratch perturbed asymmetric MAJ_9 / MAJ_11 core on Q_d."""
    n_pts = 1 << d
    x = np.arange(n_pts, dtype=np.int32)
    w = 1.0 + rng.uniform(-asym_strength, asym_strength, size=d)
    w = np.sort(w)[::-1]
    sc = np.zeros(n_pts, dtype=np.float64)
    for b in range(d):
        sc += w[b] * ((x >> b) & 1)
    sc += 1e-9 * x
    arr = np.full(n_pts, -1, dtype=np.int8)
    arr[np.argsort(sc)[n_pts // 2 :]] = 1
    return arr


def load_structural_seeds_if_present(art_dir: Path) -> List[Tuple[str, int, np.ndarray]]:
    """
    Auto-ingests `phase2_structural_seeds.json` if @fei-literature has already written it
    (non-blocking!). Validates exact balance & monotonicity before accepting any seed.
    """
    path = art_dir / "phase2_structural_seeds.json"
    if not path.is_file():
        return []
    out: List[Tuple[str, int, np.ndarray]] = []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw if isinstance(raw, list) else raw.get("seeds", raw.get("candidates", []))
        for item in items:
            if not isinstance(item, dict) or "g0" not in item:
                continue
            g0 = np.asarray(item["g0"], dtype=np.int8)
            d = int(item.get("d", int(round(math.log2(len(g0))))))
            if len(g0) != (1 << d) or d > 18:
                continue
            is_m, is_b = check_core_monotone_and_balanced(g0, d)
            if is_m and is_b:
                _, compat = decompose_zero_subcubes(g0, d)
                if compat:
                    out.append((str(item.get("label", f"struct_d{d}")), d, g0))
    except Exception:
        return []
    return out


# =============================================================================
# 3. Genuine SLEEP-DAG-RSI v8 Cartridge Factory (`TaskCartridge` + `AutoDreamRunner`)
# =============================================================================

class C71SleepDAGCartridgeBuilder:
    """
    Wires C_71 GPU search branches into dream_rsi_scripts's genuine `TaskCartridge`,
    `BranchCandidate`, `AutoDreamRunner`, `BlindVaultGuard`, `DiscoveryTree`,
    `LeanDAGPolicy(entropic_tau=4.0)`, `compute_max_tilted_objective(tau=4.0)`,
    `BackwardSubGoalDecomposer`, `MutableSkillState`, and `WaldMicroSentinel`.
    """

    def __init__(
        self,
        device: torch.device,
        baseline_cand: FEICandidate,
        rng: np.random.Generator,
        cycle_idx: int = 1,
        prior_skill_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.device = device
        self.baseline_cand = baseline_cand
        self.rng = rng
        self.cycle_idx = cycle_idx
        self.prior_skill_state = prior_skill_state
        self.evaluated_registry: Dict[str, FEICandidate] = {
            "baseline": baseline_cand,
        }
        self.best_overall: FEICandidate = baseline_cand
        self.best_tier1: Optional[FEICandidate] = baseline_cand if baseline_cand.n_effective <= 24 else None

    def _register(self, cand: FEICandidate) -> None:
        self.evaluated_registry[cand.label] = cand
        if cand.OT > self.best_overall.OT + 1e-12:
            self.best_overall = cand
        if cand.n_effective <= 24:
            if self.best_tier1 is None or cand.OT > self.best_tier1.OT + 1e-12:
                self.best_tier1 = cand

    def verify_and_score(self, fn: Callable[[], Optional[FEICandidate]]) -> Tuple[bool, float, str, Dict[str, Any]]:
        cand = fn()
        if cand is None:
            return False, 0.0, "algorithmic_failure", {"error": "Candidate failed monotonicity or balance check"}
        # Re-verify strictly in float64
        checked = evaluate_fei_candidate_fp64(cand.g0, cand.d, label=cand.label, source=cand.source, device=self.device)
        if checked is None:
            return False, 0.0, "algorithmic_failure", {"error": "Float64 re-verification failed"}
        self._register(checked)
        margins, pass_mask = compute_bucket_scores(checked)
        info = {
            "summary": f"{checked.label} d={checked.d} n_eff={checked.n_effective} I={checked.I:.6f} L={checked.OT:.8f}",
            "d": checked.d,
            "k_subcubes": len(checked.subcubes),
            "n_effective": checked.n_effective,
            "I": checked.I,
            "H": checked.H,
            "OT": checked.OT,
            "pass_mask": pass_mask,
            "bucket_scores": margins,
            "candidate_label": checked.label,
        }
        return True, float(checked.OT), "ok", info

    def crossover_synthesizer(
        self,
        u_cand: dream_cartridge.BranchCandidate,
        v_cand: dream_cartridge.BranchCandidate,
    ) -> Optional[dream_cartridge.BranchCandidate]:
        """
        Synthesizes a genuine monotone balanced hybrid via ite(x_new, max(u, v), min(u, v))
        followed by exact GPU delta-swap steepest ascent + zero-cost component splitting!
        """
        c_u = self.evaluated_registry.get(u_cand.id) or self.evaluated_registry.get(u_cand.description)
        c_v = self.evaluated_registry.get(v_cand.id) or self.evaluated_registry.get(v_cand.description)
        if c_u is None or c_v is None:
            # Pick top-2 distinct evaluated candidates
            all_c = sorted(self.evaluated_registry.values(), key=lambda c: c.OT, reverse=True)
            if len(all_c) < 2:
                return None
            c_u, c_v = all_c[0], all_c[1]

        xover_id = f"c{self.cycle_idx}_xover_{u_cand.id}_{v_cand.id}"

        def _run_xover() -> Optional[FEICandidate]:
            res = monotone_crossover_ite(c_u.g0, c_u.d, c_v.g0, c_v.d, device=self.device)
            if res is None:
                return None
            g_x, d_x = res
            # Quick GPU SA + exact delta steepest ascent polish on the hybrid!
            if d_x <= 16:
                g_pol, _, _ = gpu_delta_steepest_ascent(
                    g_x, d_x, self.device, self.rng,
                    max_iters=45, batch_candidates=192, allow_antichain=False, allow_tunneling=True,
                )
            else:
                g_pol = g_x
            # Try zero-cost component split if d_x < 18
            cand_main = evaluate_fei_candidate_fp64(g_pol, d_x, label=xover_id, source="dag_crossover_ite", device=self.device)
            if cand_main is not None and cand_main.is_pure() and d_x <= 17:
                g_sp, d_sp, ot_sp, _ = greedy_multi_split_up_to_n(g_pol, d_x, max_n=min(18, d_x + 2), device=self.device)
                if ot_sp > cand_main.OT + 1e-9:
                    c_sp = evaluate_fei_candidate_fp64(g_sp, d_sp, label=xover_id, source="dag_crossover_ite_split", device=self.device)
                    if c_sp is not None:
                        cand_main = c_sp
            return cand_main

        return dream_cartridge.BranchCandidate(
            id=xover_id,
            branch=max(u_cand.branch, v_cand.branch) + 1,
            parent_id=u_cand.id,
            attempt=max(u_cand.attempt, v_cand.attempt) + 1,
            description=f"Monotone ite(x_new, max, min) crossover({c_u.label}, {c_v.label}) + GPU delta polish",
            fn=_run_xover,
        )

    def blind_vault_exact_verify(self, candidate_fn: Callable[[], Optional[FEICandidate]]) -> Dict[str, Any]:
        """
        Single-shot BlindVaultGuard hook invoking exact_verifier.py (pure Python int FWHT
        + dual 100-dps outward-rounded intervals) on the final selected champion!
        """
        cand = self.best_overall
        if cand.is_pure() and cand.d <= 20:
            rep = exact_verifier.verify_full_truth_table(cand.g0.tolist(), require_monotone=True)
        else:
            rep = exact_verifier.verify_dealiased_core(
                cand.g0.tolist(),
                [{"fixed_mask": int(c["fixed_mask"]), "fixed_val": int(c["fixed_val"])} for c in cand.subcubes],
                require_monotone=True,
            )
        return {
            "exact_L_cert": rep["L_cert"],
            "exact_U_cert": rep["U_cert"],
            "I_exact_str": rep["I_exact_str"],
            "is_monotone": rep["is_monotone"],
            "exact_balance": rep["exact_balance"],
            "parseval_verified": rep["parseval_verified"],
            "beats_num2026": rep["beats_num2026"],
            "sha256": rep["sha256"],
        }

    def build_cartridge(
        self,
        initial_branches: List[dream_cartridge.BranchCandidate],
    ) -> dream_cartridge.TaskCartridge:
        base_c = self.baseline_cand

        def _base_fn() -> FEICandidate:
            return base_c

        return dream_cartridge.TaskCartridge(
            task_name=f"c71_fei_phase2_cycle{self.cycle_idx}",
            description=(
                f"C_71 FEI Phase 2 Cycle {self.cycle_idx}: Asymmetric Vector A2 + "
                f"GPU Delta Steepest Ascent + Zero-Cost Critical-Set Splitting on RTX 5060"
            ),
            baseline_fn=_base_fn,
            verify_and_score=self.verify_and_score,
            initial_branches=initial_branches,
            crossover_synthesizer=self.crossover_synthesizer,
            max_parallelism=4,
            sentinel_noise_regime="deterministic",
            blind_vault_evaluator=self.blind_vault_exact_verify,
            noise_probe_runs=1,
            prior_skill_state=self.prior_skill_state,
            hypothesis_contract={
                "claim": (
                    f"From-scratch asymmetric GPU SA + analytic delta steepest ascent + "
                    f"zero-cost critical-set splitting exceeds baseline {base_c.OT:.6f} toward > {NUM2026_STRICT_BAR}"
                ),
                "plan": (
                    "Explore parallel branches (d=12..18 lifting, perturbed MAJ_9/MAJ_11, "
                    "zero-cost component splits), combine complementary degree shells via "
                    "monotone ite(x_new, max, min) CROSSOVER, and certify via exact_verifier.py"
                ),
                "decision_criteria": {
                    "support": f"Exact dual-contour certified L_cert > {base_c.OT:.6f} with 0 monotonicity violations",
                    "weaken": f"All branches <= {base_c.OT:.6f} or balance/monotonicity violation",
                    "inconclusive": "Gain <= 1e-6 without structural novelty",
                },
                "alternative_explanations": [
                    "float32 accumulation drift vs pure-Python exact integer FWHT",
                    "Local basin saturation on 9-core auxiliary ridge requiring wider core d >= 10",
                ],
            },
        )


def run_sleep_dag_diagnostics(
    tree: dream_engine.DiscoveryTree,
    problem: dream_engine.Problem,
) -> Dict[str, Any]:
    """
    Runs genuine LeanDAGPolicy(entropic_tau=4.0), compute_max_tilted_objective(tau=4.0),
    and BackwardSubGoalDecomposer on the completed DiscoveryTree!
    """
    scores = [
        problem.score(n)
        for nid, n in tree.nodes.items()
        if nid != tree.root_id and problem.is_success(n)
    ]
    max_tilt = dream_engine.compute_max_tilted_objective(scores, tau=4.0) if scores else {"J_tau": 0.0}

    policy = dream_engine.LeanDAGPolicy(
        beta=0.6,
        use_counterfactual=True,
        use_crossover=True,
        entropic_tau=4.0,
    )
    replay_rep = dream_engine.replay(tree, policy, problem)

    decomposer = dream_engine.BackwardSubGoalDecomposer()
    bucket_specs = [
        {"name": f"thr_{thr:.3f}", "scale_N": 16, "tol_eps": 1e-4}
        for thr in STEPPING_THRESHOLDS
    ] + [
        {"name": "influence_ridge", "scale_N": 16, "tol_eps": 1e-3},
        {"name": "shell_W1", "scale_N": 16, "tol_eps": 1e-2},
        {"name": "shell_W2", "scale_N": 16, "tol_eps": 1e-2},
        {"name": "shell_W3", "scale_N": 16, "tol_eps": 1e-2},
        {"name": "shell_W4", "scale_N": 16, "tol_eps": 1e-2},
    ]
    subgoals = decomposer.decompose([tree], problem=problem, bucket_specs=bucket_specs)

    return {
        "max_tilted_objective_tau4": {
            "J_tau": float(max_tilt.get("j_tau", 0.0)),
            "n_scores": len(scores),
            "max_score": max(scores) if scores else 0.0,
        },
        "lean_dag_policy_replay": {
            "entropic_tau": policy.entropic_tau,
            "quality": float(replay_rep.get("quality", 0.0)),
            "v_lean": float(replay_rep.get("v", 0.0)),
            "probes": int(replay_rep.get("probes", 0)),
        },
        "backward_subgoal_decomposer": {
            "solved_buckets": subgoals.get("solved_buckets", []),
            "dead_buckets": subgoals.get("dead_buckets", []),
            "next_stepping_stone": (
                subgoals["stepping_stone_subgoals"][0]["assert_expr"]
                if subgoals.get("stepping_stone_subgoals")
                else "ALL_SOLVED"
            ),
        },
    }

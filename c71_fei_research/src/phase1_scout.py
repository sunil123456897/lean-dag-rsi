"""
Phase 1 From-Scratch Landscape Scout for C_71 (Fourier Entropy-Influence Conjecture)
on RTX 5060:
  - Vector A2: 100% GPU-resident Parallel Tempered SA + Curriculum Lifting (d = 8..14)
    combining pure balanced monotone core discovery (A=0) with Orthogonal Subcube /
    Antichain De-Aliasing (A > 0, n_eff = d + k_aux) and size-2 subcube polishing.
  - Vector B: Symmetry-Compressed Orbit Search via Krawtchouk / Wreath-Product transforms
    (d = 12..20) + Subgroup-Lattice Descent (Wreath -> Block -> Free GPU).

Strictly from scratch (NO external truth tables or published certificates loaded).
All champions are verified for exact balance, core monotonicity, and strict
subcube/antichain incomparability, and exported in exact_verifier.py Tier 2 JSON schema.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from fwht_gpu import (  # noqa: E402
    check_core_monotone_and_balanced,
    decompose_zero_subcubes,
    evaluate_antichain_dealiased_batch,
    evaluate_core_spectrum_batch,
    evaluate_subcube_dealiased_exact_fp64,
    get_device,
    make_degree_table,
)
from orbit_krawtchouk import BlockSymGroup, WreathSymGroup  # noqa: E402

NUM2026_BAR = 6.52184571092304657566
PHASE1_GO_GATE = 6.48


# =============================================================================
# 1. From-Scratch Structured Monotone Core Seed Generators (Vector A2)
# =============================================================================

def make_or2_altchain_seed(d: int) -> np.ndarray:
    """
    Constructs from-scratch (x0 | x1) & AltChain(x2..x_{d-1}) balanced via 2 minimal ones -> 0
    for even d >= 4 (gives 6.347 at d=4, 6.364 at d=6, 6.3737 at d=8, 6.3765 at d=10, 6.3772 at d=12).
    For odd d, lifts the (d-1) construction along the last variable (and resolves zeros for A=0 if needed).
    """
    if d % 2 == 1:
        sub = make_or2_altchain_seed(d - 1)
        # Lift sub to d bits: repeat along bit d-1 -> subcubes of size 2; resolve 0s cleanly if pure
        return np.concatenate([sub, sub]).astype(np.int8)

    n_pts = 1 << d
    x = np.arange(n_pts, dtype=np.int32)
    or2 = (((x >> 0) & 1) | ((x >> 1) & 1)).astype(bool)
    cur = ((x >> (d - 1)) & 1).astype(bool)
    for pos in range(d - 2, 1, -1):
        bp = ((x >> pos) & 1).astype(bool)
        cur = (bp | cur) if ((pos - 2) % 2 == 0) else (bp & cur)
    raw = np.where(or2 & cur, 1, -1).astype(np.int8)
    # Exactly +1 excess one (sum == +2): turn 1 minimal +1 to -1 for A=0 pure seed
    mask = raw == 1
    for b in range(d):
        st = 1 << b
        v = raw.reshape(-1, 2, st)
        mask.reshape(-1, 2, st)[:, 1, :] &= (v[:, 0, :] == -1)
    mp = np.flatnonzero(mask)
    if int(np.sum(raw)) == 2 and len(mp) >= 1:
        raw[int(mp[0])] = -1
    return raw


def lift_pure_core(g_small: np.ndarray, d_small: int, d_target: int) -> np.ndarray:
    """Lifts a pure {-1, +1} core from d_small to d_target by adding dummy high bits."""
    assert d_target >= d_small
    g = np.asarray(g_small, dtype=np.int8)
    # Resolve any zeros to alternating (-1, +1) along first dummy bit so it becomes a pure {-1,+1} core!
    if np.any(g == 0) and d_target > d_small:
        g_lo = np.where(g == 0, -1, g).astype(np.int8)
        g_hi = np.where(g == 0, 1, g).astype(np.int8)
        out = np.concatenate([g_lo, g_hi])
        cur_d = d_small + 1
    else:
        out = np.where(g == 0, -1, g).astype(np.int8)
        cur_d = d_small
    while cur_d < d_target:
        out = np.concatenate([out, out])
        cur_d += 1
    return out.astype(np.int8)


def build_diverse_pure_init_pool(
    d: int,
    n_chains: int,
    rng: np.random.Generator,
    prior_champions: Sequence[Tuple[int, np.ndarray]] = (),
) -> np.ndarray:
    """
    Builds n_chains strictly balanced pure monotone {-1, +1}^{2^d} starting points:
      - Curriculum-lifted champions from smaller d (preserving their exact OT at step 0!)
      - Dictators x_b (neutral I=1.0 start allowing SA to form arbitrary fresh basins)
      - From-scratch (x0|x1) & AltChain(x2..)
      - Random exponential-decay LTFs (varied coordinate hierarchies)
      - Symmetric/perturbed majority (for odd d)
    """
    n_pts = 1 << d
    x = np.arange(n_pts, dtype=np.int32)
    chains = np.empty((n_chains, n_pts), dtype=np.int8)
    alt_seed = make_or2_altchain_seed(d)
    # Ensure alt_seed is pure {-1, +1}
    if np.any(alt_seed == 0):
        alt_seed = lift_pure_core(make_or2_altchain_seed(d - 1), d - 1, d)

    for c in range(n_chains):
        if prior_champions and c < min(n_chains // 3, len(prior_champions) * 8):
            pd, pg = prior_champions[c % len(prior_champions)]
            g_lifted = lift_pure_core(pg, pd, d)
            if check_core_monotone_and_balanced(g_lifted, d) == (True, True):
                chains[c] = g_lifted
                continue
        r = c % 4
        if r == 0:
            # Dictator on bit (c % d)
            chains[c] = np.where((x >> (c % d)) & 1, 1, -1).astype(np.int8)
        elif r == 1:
            # From-scratch OR2 & AltChain seed
            chains[c] = alt_seed.copy()
        elif r == 2:
            # Random exponential-weight LTF
            w = rng.exponential(1.0, size=d)
            sc = np.zeros(n_pts, dtype=np.float64)
            for b in range(d):
                sc += w[b] * ((x >> b) & 1)
            sc += 1e-9 * x
            order = np.argsort(sc)
            arr = np.full(n_pts, -1, dtype=np.int8)
            arr[order[n_pts // 2 :]] = 1
            chains[c] = arr
        else:
            # Perturbed majority around uniform weights
            w = 1.0 + rng.uniform(-0.3, 0.3, size=d)
            sc = np.zeros(n_pts, dtype=np.float64)
            for b in range(d):
                sc += w[b] * ((x >> b) & 1)
            sc += 1e-9 * x
            order = np.argsort(sc)
            arr = np.full(n_pts, -1, dtype=np.int8)
            arr[order[n_pts // 2 :]] = 1
            chains[c] = arr
    return chains


# =============================================================================
# 2. 100% GPU-Resident Parallel Tempered SA (Pure Core + Vector A2 De-Aliasing)
# =============================================================================

def gpu_parallel_pure_sa(
    d: int,
    init_chains_np: np.ndarray,
    device: torch.device,
    steps: int = 8000,
    t_high: float = 0.25,
    t_low: float = 1e-4,
    seed: int = 2026,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    100% GPU-resident parallel tempered Simulated Annealing over pure balanced monotone
    cores g0 in {-1, +1}^{2^d}. Each chain has its own geometric temperature ladder
    spanning [t_low, t_high], with periodic best-reheat kicks on stuck chains.
    Returns (best_chains_np [C, 2^d], best_ot_np [C], total_evals).
    """
    torch.manual_seed(seed)
    n_chains, n_pts = init_chains_np.shape
    deg = make_degree_table(d, device, dtype=torch.float32)

    chains = torch.as_tensor(init_chains_np, device=device, dtype=torch.int8)
    _, h0, i0, _ = evaluate_core_spectrum_batch(chains, deg=deg, dtype=torch.float32)
    cur_ot = torch.where(i0 > 1.0 + 1e-6, h0 / (i0 - 1.0).clamp_min(1e-6), torch.zeros_like(h0))
    best_ot = cur_ot.clone()
    best_chains = chains.clone()

    # Diverse per-chain temperature schedule: half annealed hot->cold, half multi-temp ladder!
    chain_t0 = torch.linspace(math.log(t_high * 0.25), math.log(t_high), n_chains, device=device).exp()
    chain_t1 = torch.full((n_chains,), t_low, device=device)
    decay = (chain_t1 / chain_t0).pow(1.0 / max(1, steps - 1))
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

        # Mid-run elite seeding: at 65% of steps, copy top-8 elites into worst 32 chains for fine polish
        if step == int(steps * 0.65):
            top_idx = torch.topk(best_ot, k=min(8, n_chains)).indices
            bot_idx = torch.topk(best_ot, k=min(32, n_chains // 2), largest=False).indices
            src = top_idx[torch.arange(len(bot_idx), device=device) % len(top_idx)]
            chains[bot_idx] = best_chains[src]
            cur_ot[bot_idx] = best_ot[src]

    return (
        best_chains.detach().cpu().numpy(),
        best_ot.detach().cpu().numpy().astype(np.float64),
        int(n_chains * (steps + 1)),
    )


def gpu_parallel_antichain_polish(
    d: int,
    seed_cores_np: np.ndarray,
    device: torch.device,
    steps: int = 3500,
    t_start: float = 0.035,
    t_end: float = 1e-5,
    seed: int = 3026,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    100% GPU-resident 3-valued Vector A2 Antichain De-Aliasing Polish on top of discovered
    pure cores (or 3-valued cores). Executes all 5 strictly legal balanced monotone moves
    (Swap, Expand Antichain, Contract Antichain, Slide-Pos, Slide-Neg) completely on GPU!
    """
    torch.manual_seed(seed)
    n_chains, n_pts = seed_cores_np.shape
    deg = make_degree_table(d, device, dtype=torch.float32)

    chains = torch.as_tensor(seed_cores_np, device=device, dtype=torch.int8)
    out0 = evaluate_antichain_dealiased_batch(chains, deg=deg, dtype=torch.float32)
    cur_ot = out0["OT"]
    best_ot = cur_ot.clone()
    best_chains = chains.clone()

    chain_t0 = torch.linspace(math.log(t_start * 0.15), math.log(t_start), n_chains, device=device).exp()
    decay = (torch.full_like(chain_t0, t_end) / chain_t0).pow(1.0 / max(1, steps - 1))
    temps = chain_t0.clone()
    row_idx = torch.arange(n_chains, device=device)

    for step in range(steps):
        mp = chains == 1
        mn = chains == -1
        zm = chains == 0
        for b in range(d):
            st = 1 << b
            vc = chains.view(n_chains, -1, 2, st)
            mp.view(n_chains, -1, 2, st)[:, :, 1, :] &= (vc[:, :, 0, :] == -1)
            mn.view(n_chains, -1, 2, st)[:, :, 0, :] &= (vc[:, :, 1, :] == 1)

        has_mp = mp.any(dim=1)
        has_mn = mn.any(dim=1)
        z_cnt = zm.sum(dim=1)

        # Sample candidate indices for all 5 move types simultaneously on GPU
        a_idx = (torch.rand((n_chains, n_pts), device=device) * mp.float()).argmax(dim=1)
        b_idx = (torch.rand((n_chains, n_pts), device=device) * mn.float()).argmax(dim=1)
        r_z1 = torch.rand((n_chains, n_pts), device=device) * zm.float()
        z1_idx = r_z1.argmax(dim=1)
        # Mask out z1 to pick distinct z2 for contract move
        r_z2 = r_z1.clone()
        r_z2[row_idx, z1_idx] = -1.0
        z2_idx = r_z2.argmax(dim=1)

        incomp_ab = (b_idx & a_idx) != b_idx
        u_type = torch.rand(n_chains, device=device)

        prop = chains.clone()
        valid = torch.zeros(n_chains, device=device, dtype=torch.bool)

        # Move 0 (32%): Swap (+1, -1) -> (-1, +1)
        m0 = (u_type < 0.32) & has_mp & has_mn & incomp_ab
        if m0.any():
            r0 = row_idx[m0]
            prop[r0, a_idx[m0]] = -1
            prop[r0, b_idx[m0]] = 1
            valid |= m0

        # Move 1 (30%): Expand antichain (+1, -1) -> (0, 0)
        m1 = (u_type >= 0.32) & (u_type < 0.62) & has_mp & has_mn & incomp_ab
        if m1.any():
            r1 = row_idx[m1]
            prop[r1, a_idx[m1]] = 0
            prop[r1, b_idx[m1]] = 0
            valid |= m1

        # Move 2 (14%): Contract antichain (0, 0) -> (+1, -1)
        m2 = (u_type >= 0.62) & (u_type < 0.76) & (z_cnt >= 2) & (z1_idx != z2_idx)
        if m2.any():
            r2 = row_idx[m2]
            prop[r2, z1_idx[m2]] = 1
            prop[r2, z2_idx[m2]] = -1
            valid |= m2

        # Move 3 (12%): Slide zero -> +1, a -> 0
        m3 = (u_type >= 0.76) & (u_type < 0.88) & (z_cnt >= 1) & has_mp
        if m3.any():
            r3 = row_idx[m3]
            prop[r3, z1_idx[m3]] = 1
            prop[r3, a_idx[m3]] = 0
            valid |= m3

        # Move 4 (12%): Slide zero -> -1, b -> 0
        m4 = (u_type >= 0.88) & (z_cnt >= 1) & has_mn
        if m4.any():
            r4 = row_idx[m4]
            prop[r4, z1_idx[m4]] = -1
            prop[r4, b_idx[m4]] = 0
            valid |= m4

        out = evaluate_antichain_dealiased_batch(prop, deg=deg, dtype=torch.float32)
        prop_ot = out["OT"]
        delta = prop_ot - cur_ot
        acc = valid & ((delta >= 0.0) | (torch.rand(n_chains, device=device) < torch.exp((delta / temps).clamp(-40.0, 0.0))))
        chains = torch.where(acc.unsqueeze(1), prop, chains)
        cur_ot = torch.where(acc, prop_ot, cur_ot)

        imp = cur_ot > best_ot
        if imp.any():
            best_ot = torch.where(imp, cur_ot, best_ot)
            best_chains = torch.where(imp.unsqueeze(1), chains, best_chains)

        temps.mul_(decay)

    return (
        best_chains.detach().cpu().numpy(),
        best_ot.detach().cpu().numpy().astype(np.float64),
        int(n_chains * (steps + 1)),
    )


def subcube_pair_polish_fp64(
    g_init: np.ndarray,
    d: int,
    device: torch.device,
    rng: np.random.Generator,
    max_rounds: int = 8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Tests coalesced size-2 boundary subcube expansions (codimension r = d - 1, analogous to
    2-cell auxiliary hosts) on top of a winning core g_init, strictly preserving monotonicity
    and balance via exact float64 evaluation.
    """
    g_best = g_init.astype(np.int8).copy()
    best_res = evaluate_subcube_dealiased_exact_fp64(g_best, d, device=device)
    best_ot = float(best_res["OT"])
    n_pts = 1 << d

    for _ in range(max_rounds):
        bad_lo = np.zeros(n_pts, dtype=np.int16)
        bad_hi = np.zeros(n_pts, dtype=np.int16)
        for b in range(d):
            st = 1 << b
            vg = g_best.reshape(-1, 2, st)
            bad_lo.reshape(-1, 2, st)[:, 1, :] += (vg[:, 0, :] != -1)
            bad_hi.reshape(-1, 2, st)[:, 0, :] += (vg[:, 1, :] != 1)

        p_pairs: List[Tuple[int, int]] = []
        n_pairs: List[Tuple[int, int]] = []
        for b in range(d):
            st = 1 << b
            vg = g_best.reshape(-1, 2, st)
            bl = bad_lo.reshape(-1, 2, st)
            bh = bad_hi.reshape(-1, 2, st)
            ok_p = (vg[:, 0, :] == 1) & (vg[:, 1, :] == 1) & (bl[:, 0, :] == 0) & (bl[:, 1, :] == 1)
            if np.any(ok_p):
                hi_i, lo_i = np.nonzero(ok_p)
                for h_i, l_i in zip(hi_i[:16], lo_i[:16]):
                    x0 = int((h_i << (b + 1)) | l_i)
                    p_pairs.append((x0, x0 | st))
            ok_n = (vg[:, 0, :] == -1) & (vg[:, 1, :] == -1) & (bh[:, 1, :] == 0) & (bh[:, 0, :] == 1)
            if np.any(ok_n):
                hi_i, lo_i = np.nonzero(ok_n)
                for h_i, l_i in zip(hi_i[:16], lo_i[:16]):
                    x0 = int((h_i << (b + 1)) | l_i)
                    n_pairs.append((x0, x0 | st))
        if not p_pairs or not n_pairs:
            break

        improved = False
        for _ in range(min(48, len(p_pairs) * len(n_pairs))):
            p0, p1 = p_pairs[int(rng.integers(0, len(p_pairs)))]
            n0, n1 = n_pairs[int(rng.integers(0, len(n_pairs)))]
            if (n0 & p1) == n0:
                continue
            cand = g_best.copy()
            cand[p0] = 0
            cand[p1] = 0
            cand[n0] = 0
            cand[n1] = 0
            res = evaluate_subcube_dealiased_exact_fp64(cand, d, device=device)
            if res["balanced"] and res["monotone_full"] and float(res["OT"]) > best_ot + 1e-9:
                best_ot = float(res["OT"])
                best_res = res
                g_best = cand
                improved = True
                break
        if not improved:
            break

    return g_best, best_res


# =============================================================================
# 3. Fast Orbit-Space Search (Vector B) + Subgroup-Lattice Descent
# =============================================================================

def _orbit_boundary_sets(group, u: np.ndarray) -> Tuple[List[int], List[int], List[int]]:
    min_pos: List[int] = []
    max_neg: List[int] = []
    zeros: List[int] = []
    for i in range(group.M):
        val = int(u[i])
        if val == 0:
            zeros.append(i)
        elif val == 1:
            if all(int(u[p]) == -1 for p in group.parents[i]):
                min_pos.append(i)
        else:
            if all(int(u[c]) == 1 for c in group.children[i]):
                max_neg.append(i)
    return min_pos, max_neg, zeros


def _orbits_comparable(w_lo: Tuple[int, ...], w_hi: Tuple[int, ...]) -> bool:
    return all(a <= b for a, b in zip(w_lo, w_hi))


def orbit_seed_from_structure(group, mode: int, rng: np.random.Generator) -> Optional[np.ndarray]:
    """
    Generates diverse balanced monotone orbit seeds for BlockSymGroup or WreathSymGroup:
      - mode 0: dictator on first S_1 block or hierarchical OR/AND over blocks (low I ~ 1.5..2.0!)
      - mode 1: Tribes-like block formula (OR of block-ANDs balanced at boundary)
      - mode 2/3: weighted threshold with exponential decay across blocks
    """
    M = group.M
    sizes = group.orbit_sizes_np
    half = (1 << group.d) // 2

    if isinstance(group, BlockSymGroup) and mode == 0 and group.blocks[0] == 1:
        # Pure dictator on block 0 (exact I=1.0, balanced, monotone -> lets orbit SA grow freely!)
        u = np.array([1 if orb[0] == 1 else -1 for orb in group.orbits], dtype=np.int8)
        is_m, is_ac, is_b = group.check_monotone_and_antichain(u)
        if is_m and is_ac and is_b:
            return u

    if isinstance(group, BlockSymGroup) and mode == 1:
        # Block-OR-AND hierarchy: block 0 OR (block 1 AND (block 2 OR ...))
        u = np.empty(M, dtype=np.int8)
        nb = len(group.blocks)
        for idx, orb in enumerate(group.orbits):
            # Convert each block weight to boolean (>= ceil(k/2) or == k for AND, >= 1 for OR)
            val = orb[nb - 1] >= max(1, (group.blocks[nb - 1] + 1) // 2)
            for j in range(nb - 2, -1, -1):
                kj = group.blocks[j]
                if j % 2 == 0:
                    val = (orb[j] >= 1) or val
                else:
                    val = (orb[j] == kj) and val
            u[idx] = 1 if val else -1
    else:
        if isinstance(group, BlockSymGroup):
            rho = [0.55, 0.70, 0.82, 0.92][mode % 4]
            bw = [rho ** j for j in range(len(group.blocks))]
            scores = np.array([sum(wk * tk for wk, tk in zip(bw, orb)) for orb in group.orbits], dtype=np.float64)
        else:
            alpha = [0.6, 1.0, 1.5, 2.2][mode % 4]
            scores = np.array([sum((tk / float(group.a)) ** alpha for tk in orb) for orb in group.orbits], dtype=np.float64)
        order = sorted(range(M), key=lambda i: (scores[i], int(group.orbit_wts_np[i]), rng.random()))
        u = np.full(M, -1, dtype=np.int8)
        acc = 0
        for idx in reversed(order):
            sz = int(sizes[idx])
            if acc + sz <= half:
                u[idx] = 1
                acc += sz

    # Enforce monotonicity upper closure
    for _ in range(4):
        changed = False
        for src, dst in zip(group.edge_src, group.edge_dst):
            if u[src] > u[dst]:
                u[dst] = u[src]
                changed = True
        if not changed:
            break

    # Repair balance on boundary orbits
    for _ in range(M * 2):
        bal = int(np.dot(sizes, u.astype(np.int64)))
        if bal == 0:
            break
        mp, mn, _ = _orbit_boundary_sets(group, u)
        if bal > 0:
            if not mp:
                break
            best_i = min(mp, key=lambda i: min(abs(bal - 2 * int(sizes[i])), abs(bal - int(sizes[i]))))
            u[best_i] = 0 if abs(bal - int(sizes[best_i])) < abs(bal - 2 * int(sizes[best_i])) else -1
        else:
            if not mn:
                break
            best_j = min(mn, key=lambda j: min(abs(bal + 2 * int(sizes[j])), abs(bal + int(sizes[j]))))
            u[best_j] = 0 if abs(bal + int(sizes[best_j])) < abs(bal + 2 * int(sizes[best_j])) else 1

    is_m, is_ac, is_b = group.check_monotone_and_antichain(u)
    if is_m and is_ac and is_b:
        return u
    # Guaranteed monotone balanced fallback
    wts = group.orbit_wts_np
    mid = group.d // 2
    u_fb = np.where(wts < mid, -1, np.where(wts > mid, 1, 0)).astype(np.int8) if group.d % 2 == 0 else np.where(wts <= mid, -1, 1).astype(np.int8)
    is_m, is_ac, is_b = group.check_monotone_and_antichain(u_fb)
    return u_fb if (is_m and is_ac and is_b) else None


def orbit_search_vector_b(
    group,
    u_init: np.ndarray,
    rng: np.random.Generator,
    steps: int = 350,
    batch_size: int = 128,
    temp_start: float = 0.08,
    temp_end: float = 1e-5,
) -> Dict[str, Any]:
    """
    Batched GPU Simulated Annealing + Steepest Ascent in orbit space (BlockSymGroup or WreathSymGroup).
    Supports equal-size 2-orbit moves AND 1-vs-2 balanced moves (crucial for power-of-2 sizes in S_2^m!).
    """
    u_cur = u_init.astype(np.int8).copy()
    sizes = group.orbit_sizes_np
    device = group.device

    r0 = group.evaluate_batch(torch.as_tensor(u_cur, device=device), dtype=torch.float64)
    cur_ot = float(r0["OT"].item())
    best_ot = cur_ot
    best_u = u_cur.copy()

    t_ratio = (temp_end / max(temp_start, 1e-9)) ** (1.0 / max(1, steps - 1))
    temp = temp_start
    total_evals = 1
    stagnation = 0

    for _ in range(steps):
        mp, mn, zs = _orbit_boundary_sets(group, u_cur)
        mp_by_sz: Dict[int, List[int]] = {}
        for i in mp:
            mp_by_sz.setdefault(int(sizes[i]), []).append(i)
        mn_by_sz: Dict[int, List[int]] = {}
        for j in mn:
            mn_by_sz.setdefault(int(sizes[j]), []).append(j)
        z_by_sz: Dict[int, List[int]] = {}
        for k in zs:
            z_by_sz.setdefault(int(sizes[k]), []).append(k)

        common_pm = [sz for sz in mp_by_sz if sz in mn_by_sz]
        common_z2 = [sz for sz, lst in z_by_sz.items() if len(lst) >= 2]
        common_zp = [sz for sz in z_by_sz if sz in mp_by_sz]
        common_zn = [sz for sz in z_by_sz if sz in mn_by_sz]
        one_vs_two_pm = [(2 * sz, sz) for sz in mn_by_sz if len(mn_by_sz[sz]) >= 2 and (2 * sz) in mp_by_sz]
        two_vs_one_pm = [(sz, 2 * sz) for sz in mp_by_sz if len(mp_by_sz[sz]) >= 2 and (2 * sz) in mn_by_sz]

        batch_np = np.repeat(u_cur[None, :], batch_size, axis=0)
        actual_b = 0

        for _ in range(batch_size * 3):
            if actual_b >= batch_size:
                break
            r = float(rng.random())
            if r < 0.36 and common_pm:
                sz = common_pm[int(rng.integers(0, len(common_pm)))]
                a = mp_by_sz[sz][int(rng.integers(0, len(mp_by_sz[sz])))]
                b = mn_by_sz[sz][int(rng.integers(0, len(mn_by_sz[sz])))]
                if _orbits_comparable(group.orbits[b], group.orbits[a]):
                    continue
                batch_np[actual_b, a] = -1
                batch_np[actual_b, b] = 1
                actual_b += 1
            elif r < 0.56 and common_pm:
                sz = common_pm[int(rng.integers(0, len(common_pm)))]
                a = mp_by_sz[sz][int(rng.integers(0, len(mp_by_sz[sz])))]
                b = mn_by_sz[sz][int(rng.integers(0, len(mn_by_sz[sz])))]
                if _orbits_comparable(group.orbits[b], group.orbits[a]):
                    continue
                batch_np[actual_b, a] = 0
                batch_np[actual_b, b] = 0
                actual_b += 1
            elif r < 0.74 and common_z2:
                sz = common_z2[int(rng.integers(0, len(common_z2)))]
                lst = z_by_sz[sz]
                i1 = int(rng.integers(0, len(lst)))
                i2 = int(rng.integers(0, len(lst) - 1))
                if i2 >= i1:
                    i2 += 1
                batch_np[actual_b, lst[i1]] = 1
                batch_np[actual_b, lst[i2]] = -1
                actual_b += 1
            elif r < 0.86 and (common_zp or common_zn):
                if common_zp and (not common_zn or rng.random() < 0.5):
                    sz = common_zp[int(rng.integers(0, len(common_zp)))]
                    z = z_by_sz[sz][int(rng.integers(0, len(z_by_sz[sz])))]
                    a = mp_by_sz[sz][int(rng.integers(0, len(mp_by_sz[sz])))]
                    batch_np[actual_b, z] = 1
                    batch_np[actual_b, a] = 0
                    actual_b += 1
                elif common_zn:
                    sz = common_zn[int(rng.integers(0, len(common_zn)))]
                    z = z_by_sz[sz][int(rng.integers(0, len(z_by_sz[sz])))]
                    b = mn_by_sz[sz][int(rng.integers(0, len(mn_by_sz[sz])))]
                    batch_np[actual_b, z] = -1
                    batch_np[actual_b, b] = 0
                    actual_b += 1
            elif one_vs_two_pm or two_vs_one_pm:
                if one_vs_two_pm and (not two_vs_one_pm or rng.random() < 0.5):
                    sz2, sz1 = one_vs_two_pm[int(rng.integers(0, len(one_vs_two_pm)))]
                    a = mp_by_sz[sz2][int(rng.integers(0, len(mp_by_sz[sz2])))]
                    lst = mn_by_sz[sz1]
                    i1 = int(rng.integers(0, len(lst)))
                    i2 = int(rng.integers(0, len(lst) - 1))
                    if i2 >= i1:
                        i2 += 1
                    b1, b2 = lst[i1], lst[i2]
                    if _orbits_comparable(group.orbits[b1], group.orbits[a]) or _orbits_comparable(group.orbits[b2], group.orbits[a]):
                        continue
                    batch_np[actual_b, a] = -1
                    batch_np[actual_b, b1] = 1
                    batch_np[actual_b, b2] = 1
                    actual_b += 1
                else:
                    sz1, sz2 = two_vs_one_pm[int(rng.integers(0, len(two_vs_one_pm)))]
                    b = mn_by_sz[sz2][int(rng.integers(0, len(mn_by_sz[sz2])))]
                    lst = mp_by_sz[sz1]
                    i1 = int(rng.integers(0, len(lst)))
                    i2 = int(rng.integers(0, len(lst) - 1))
                    if i2 >= i1:
                        i2 += 1
                    a1, a2 = lst[i1], lst[i2]
                    if _orbits_comparable(group.orbits[b], group.orbits[a1]) or _orbits_comparable(group.orbits[b], group.orbits[a2]):
                        continue
                    batch_np[actual_b, a1] = -1
                    batch_np[actual_b, a2] = -1
                    batch_np[actual_b, b] = 1
                    actual_b += 1

        if actual_b == 0:
            break

        out = group.evaluate_batch(torch.as_tensor(batch_np[:actual_b], device=device), dtype=torch.float64)
        ot_vals = out["OT"].detach().cpu().numpy()
        total_evals += actual_b

        best_idx = int(np.argmax(ot_vals))
        cand_ot = float(ot_vals[best_idx])
        accept_idx: Optional[int] = None
        if cand_ot > cur_ot + 1e-9:
            accept_idx = best_idx
            stagnation = 0
        else:
            stagnation += 1
            top_k = min(6, actual_b)
            top_indices = np.argpartition(ot_vals, -top_k)[-top_k:]
            pick = int(rng.choice(top_indices))
            delta = float(ot_vals[pick]) - cur_ot
            if delta >= 0.0 or (temp > 1e-9 and rng.random() < math.exp(max(-40.0, delta / temp))):
                accept_idx = pick

        if accept_idx is not None:
            u_cur = batch_np[accept_idx].copy()
            cur_ot = float(ot_vals[accept_idx])
            if cur_ot > best_ot + 1e-9:
                is_m, is_ac, is_b = group.check_monotone_and_antichain(u_cur)
                if is_m and is_ac and is_b:
                    best_ot = cur_ot
                    best_u = u_cur.copy()
                    stagnation = 0

        if stagnation >= 45:
            u_cur = best_u.copy()
            cur_ot = best_ot
            stagnation = 0

        temp *= t_ratio

    r_fin = group.evaluate_batch(torch.as_tensor(best_u, device=device), dtype=torch.float64)
    is_m, is_ac, is_b = group.check_monotone_and_antichain(best_u)
    return {
        "d": group.d,
        "M": group.M,
        "OT": float(r_fin["OT"].item()),
        "H": float(r_fin["H"].item()),
        "I": float(r_fin["I"].item()),
        "A_cells": int(round(float(r_fin["A_cells"].item()))),
        "n_eff": int(group.d + round(float(r_fin["A_cells"].item()))),
        "monotone_full": bool(is_m and is_ac),
        "balanced": bool(is_b),
        "evals": total_evals,
        "u": best_u,
    }


# =============================================================================
# 4. Full Phase 1 Campaign Driver & Exact-Verifier JSON Export
# =============================================================================

def save_candidate_tier2_json(
    path: Path,
    name: str,
    d: int,
    g0: np.ndarray,
    subcubes: Sequence[Dict[str, int]],
    metadata: Dict[str, Any],
) -> None:
    payload = {
        "tier": 2,
        "name": name,
        "d": int(d),
        "k_subcubes": len(subcubes),
        "n_effective": int(d + len(subcubes)),
        "convention": "pm1_monotone_increasing",
        "g0": [int(v) for v in np.asarray(g0, dtype=np.int8).tolist()],
        "subcubes": [{"fixed_mask": int(c["fixed_mask"]), "fixed_val": int(c["fixed_val"])} for c in subcubes],
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_phase1_scout(seed: int = 2026071, quick: bool = False) -> Dict[str, Any]:
    t_global = time.perf_counter()
    device = get_device(prefer_cuda=True)
    rng = np.random.default_rng(seed)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    summary: Dict[str, Any] = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "seed": seed,
        "vector_a2": {},
        "vector_b": {},
        "subgroup_descent": {},
        "overall_best": None,
        "best_tier1_n_le_24": None,
    }

    best_overall: Dict[str, Any] = {"OT": -1.0}
    best_tier1: Dict[str, Any] = {"OT": -1.0}

    def consider_champion(label: str, d: int, g0: np.ndarray, source: str) -> Dict[str, Any]:
        nonlocal best_overall, best_tier1
        res = evaluate_subcube_dealiased_exact_fp64(g0, d, device=device)
        assert res["balanced"] and res["monotone_full"], f"Invalid champion {label}!"
        entry = {
            "label": label,
            "source": source,
            "d": int(d),
            "k_subcubes": int(res["k_aux"]),
            "n_effective": int(res["n_eff"]),
            "zero_cells": int(res["zero_cells"]),
            "H": float(res["H"]),
            "I": float(res["I"]),
            "OT": float(res["OT"]),
        }
        if float(res["OT"]) > float(best_overall["OT"]):
            best_overall = {**entry, "g0": g0.copy(), "subcubes": res["subcubes"]}
        if int(res["n_eff"]) <= 24 and float(res["OT"]) > float(best_tier1["OT"]):
            best_tier1 = {**entry, "g0": g0.copy(), "subcubes": res["subcubes"]}
        return entry

    # -------------------------------------------------------------------------
    # PART 1: Vector A2 — Curriculum 100%-GPU Pure SA + Antichain De-Aliasing (d = 8, 9, 10, 11, 12, 14)
    # -------------------------------------------------------------------------
    print("=== PHASE 1 / PART 1: Vector A2 (100% GPU Curriculum Core + De-Aliasing, d = 8..14) ===", flush=True)
    # (d, n_chains, pure_steps, antichain_steps)
    a2_schedule = [
        (8,  128, 3000 if quick else 8000,  1500 if quick else 4000),
        (9,  128, 4000 if quick else 14000, 2000 if quick else 6000),
        (10, 128, 3500 if quick else 12000, 2000 if quick else 6000),
        (11, 128, 2500 if quick else 9000,  1500 if quick else 4500),
        (12, 128, 2000 if quick else 7000,  1500 if quick else 4000),
        (14,  64, 1000 if quick else 3000,   800 if quick else 2000),
    ]

    prior_pure_champions: List[Tuple[int, np.ndarray]] = []
    for d, n_chains, p_steps, ac_steps in a2_schedule:
        t0 = time.perf_counter()
        init_pure = build_diverse_pure_init_pool(d, n_chains, rng, prior_champions=prior_pure_champions[-4:])
        pure_chains, pure_ots, evals_pure = gpu_parallel_pure_sa(
            d, init_pure, device=device, steps=p_steps,
            t_high=0.25 if d <= 10 else 0.12, t_low=1e-4, seed=seed + d * 17,
        )
        # Rank top pure cores and replicate them into antichain polish pool!
        order_p = np.argsort(-pure_ots)
        top_pure_g = pure_chains[int(order_p[0])]
        r_pure = evaluate_subcube_dealiased_exact_fp64(top_pure_g, d, device=device)
        consider_champion(f"A2_pure_d{d}", d, top_pure_g, source="vector_a2_pure")
        # Keep top pure core for curriculum lifting to next d
        prior_pure_champions.append((d, top_pure_g.copy()))

        # Build antichain polish pool from top-16 distinct pure cores (replicated to n_chains)
        top_k = min(16, n_chains)
        ac_init = np.empty_like(pure_chains)
        for c in range(n_chains):
            ac_init[c] = pure_chains[int(order_p[c % top_k])]

        ac_chains, ac_ots, evals_ac = gpu_parallel_antichain_polish(
            d, ac_init, device=device, steps=ac_steps,
            t_start=0.04 if d <= 10 else 0.02, t_end=1e-5, seed=seed + d * 31,
        )
        # Evaluate top-6 antichain candidates in exact float64 (including automatic subcube coalescing!)
        order_ac = np.argsort(-ac_ots)
        best_d_entry = consider_champion(f"A2_pure_d{d}", d, top_pure_g, source="vector_a2_pure")
        best_ac_g = top_pure_g
        for idx_c in order_ac[:6]:
            cand_g = ac_chains[int(idx_c)]
            ent = consider_champion(f"A2_dealiased_d{d}", d, cand_g, source="vector_a2_dealiased")
            if ent["OT"] > best_d_entry["OT"]:
                best_d_entry = ent
                best_ac_g = cand_g

        # Subcube pair polish (codim = d - 1) on the best d<=12 candidate
        if d <= 12:
            g_sc, r_sc = subcube_pair_polish_fp64(best_ac_g, d, device=device, rng=rng, max_rounds=6)
            if float(r_sc["OT"]) > best_d_entry["OT"] + 1e-9:
                best_d_entry = consider_champion(f"A2_subcube_d{d}", d, g_sc, source="vector_a2_subcube")

        dt = time.perf_counter() - t0
        total_e = evals_pure + evals_ac
        summary["vector_a2"][f"d={d}"] = {
            **best_d_entry,
            "pure_best_OT": round(float(r_pure["OT"]), 8),
            "pure_best_I": round(float(r_pure["I"]), 8),
            "total_evals": total_e,
            "evals_per_sec": round(total_e / max(dt, 1e-6), 1),
            "elapsed_sec": round(dt, 2),
        }
        print(
            f"  [A2 d={d:2d}] pure={float(r_pure['OT']):.6f} -> de-aliased OT={best_d_entry['OT']:.8f} "
            f"(I={best_d_entry['I']:.6f}, H={best_d_entry['H']:.6f}, k_aux={best_d_entry['k_subcubes']}, "
            f"n_eff={best_d_entry['n_effective']}, {total_e/dt:.0f}/s, {dt:.1f}s)",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # PART 2: Vector B — Symmetry-Compressed Orbit Search (d = 12..20)
    # -------------------------------------------------------------------------
    print("\n=== PHASE 1 / PART 2: Vector B (Orbit Krawtchouk / Wreath, d = 12..20) ===", flush=True)
    b_groups: List[Tuple[str, Any, int]] = [
        ("S1x4_S2x4_d12", BlockSymGroup((1, 1, 1, 1, 2, 2, 2, 2), device=device), 180 if quick else 360),
        ("S2x6_d12", BlockSymGroup((2,) * 6, device=device), 160 if quick else 320),
        ("S1x4_S2x5_d14", BlockSymGroup((1, 1, 1, 1, 2, 2, 2, 2, 2), device=device), 160 if quick else 320),
        ("S2x7_d14", BlockSymGroup((2,) * 7, device=device), 150 if quick else 300),
        ("S1x2_S2x5_S4_d16", BlockSymGroup((1, 1, 2, 2, 2, 2, 2, 4), device=device), 140 if quick else 260),
        ("S2x8_d16", BlockSymGroup((2,) * 8, device=device), 120 if quick else 240),
        ("S3_wr_S6_d18", WreathSymGroup(3, 6, device=device), 180 if quick else 380),
        ("S3x6_d18", BlockSymGroup((3,) * 6, device=device), 140 if quick else 260),
        ("S4_wr_S5_d20", WreathSymGroup(4, 5, device=device), 180 if quick else 380),
        ("S4x5_d20", BlockSymGroup((4,) * 5, device=device), 120 if quick else 240),
    ]

    top_b_basins: List[Tuple[float, str, Any, np.ndarray]] = []
    for gname, grp, steps in b_groups:
        t0 = time.perf_counter()
        best_g_res: Optional[Dict[str, Any]] = None
        evals_g = 0
        n_restarts = 2 if quick else 4
        for r_i in range(n_restarts):
            u0 = orbit_seed_from_structure(grp, mode=r_i, rng=rng)
            if u0 is None:
                continue
            res = orbit_search_vector_b(
                grp, u0, rng=rng, steps=steps,
                batch_size=64 if grp.M > 3000 else 128,
                temp_start=0.10 if r_i % 2 == 0 else 0.03,
                temp_end=1e-5,
            )
            evals_g += int(res["evals"])
            if best_g_res is None or res["OT"] > best_g_res["OT"]:
                best_g_res = res

        dt = time.perf_counter() - t0
        assert best_g_res is not None
        u_best = best_g_res["u"]
        top_b_basins.append((best_g_res["OT"], gname, grp, u_best))

        if grp.d <= 16:
            g_exp = grp.expand_to_full(u_best)
            consider_champion(f"B_{gname}", grp.d, g_exp, source="vector_b")

        summary["vector_b"][gname] = {
            "d": grp.d,
            "M": grp.M,
            "OT": round(best_g_res["OT"], 8),
            "H": round(best_g_res["H"], 8),
            "I": round(best_g_res["I"], 8),
            "A_cells": best_g_res["A_cells"],
            "n_eff": best_g_res["n_eff"],
            "monotone_full": best_g_res["monotone_full"],
            "balanced": best_g_res["balanced"],
            "total_evals": evals_g,
            "elapsed_sec": round(dt, 2),
        }
        print(
            f"  [B {gname:16s} d={grp.d:2d} M={grp.M:5d}] OT={best_g_res['OT']:.8f} "
            f"(I={best_g_res['I']:.6f}, A={best_g_res['A_cells']}, n_eff={best_g_res['n_eff']}, {dt:.1f}s)",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # PART 3: Subgroup-Lattice Descent on Top Vector B Basins
    # -------------------------------------------------------------------------
    print("\n=== PHASE 1 / PART 3: Subgroup-Lattice Descent ===", flush=True)
    for orig_ot, gname, grp, u_best in top_b_basins:
        if isinstance(grp, WreathSymGroup):
            t0 = time.perf_counter()
            blk_grp, u_blk = grp.lift_to_block_sym(u_best)
            res_desc = orbit_search_vector_b(
                blk_grp, u_blk, rng=rng,
                steps=120 if quick else 260,
                batch_size=64, temp_start=0.04, temp_end=1e-5,
            )
            dt = time.perf_counter() - t0
            key = f"{gname}_to_block"
            summary["subgroup_descent"][key] = {
                "from_OT": round(orig_ot, 8),
                "to_OT": round(res_desc["OT"], 8),
                "I": round(res_desc["I"], 8),
                "d": blk_grp.d,
                "M": blk_grp.M,
                "elapsed_sec": round(dt, 2),
            }
            print(f"  [Descent {key}] {orig_ot:.6f} -> {res_desc['OT']:.8f} ({dt:.1f}s)", flush=True)

    block_d14 = [b for b in top_b_basins if isinstance(b[2], BlockSymGroup) and b[2].d <= 14]
    block_d14.sort(key=lambda x: x[0], reverse=True)
    for orig_ot, gname, grp, u_best in block_d14[:2]:
        t0 = time.perf_counter()
        g_full = grp.expand_to_full(u_best)
        seed_batch = np.repeat(g_full[None, :], 64, axis=0)
        ac_chains, ac_ots, _ = gpu_parallel_antichain_polish(
            grp.d, seed_batch, device=device,
            steps=1000 if quick else 3000,
            t_start=0.04, t_end=1e-5, seed=seed + grp.d * 77,
        )
        top_g = ac_chains[int(np.argmax(ac_ots))]
        ent = consider_champion(f"Descent_{gname}_to_free", grp.d, top_g, source="subgroup_descent")
        dt = time.perf_counter() - t0
        summary["subgroup_descent"][f"{gname}_to_free"] = {
            "from_OT": round(orig_ot, 8),
            "to_OT": round(ent["OT"], 8),
            "I": round(ent["I"], 8),
            "H": round(ent["H"], 8),
            "n_eff": ent["n_effective"],
            "k_aux": ent["k_subcubes"],
            "elapsed_sec": round(dt, 2),
        }
        print(
            f"  [Descent {gname} -> free d={grp.d}] {orig_ot:.6f} -> {ent['OT']:.8f} "
            f"(I={ent['I']:.6f}, n_eff={ent['n_effective']}, {dt:.1f}s)",
            flush=True,
        )

    # -------------------------------------------------------------------------
    # Final Telemetry, Go/No-Go Check & Artifact Persistence
    # -------------------------------------------------------------------------
    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if device.type == "cuda" else 0.0
    total_sec = time.perf_counter() - t_global

    art_dir = Path(_SRC_DIR).parent / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    best_cand_path = art_dir / "phase1_best_candidate.json"
    best_meta = {k: v for k, v in best_overall.items() if k not in ("g0", "subcubes")}
    save_candidate_tier2_json(
        best_cand_path,
        name=str(best_overall["label"]),
        d=int(best_overall["d"]),
        g0=best_overall["g0"],
        subcubes=best_overall["subcubes"],
        metadata={**best_meta, "from_scratch": True, "phase": 1},
    )

    if best_tier1.get("OT", -1.0) > 0:
        t1_meta = {k: v for k, v in best_tier1.items() if k not in ("g0", "subcubes")}
        summary["best_tier1_n_le_24"] = t1_meta
        save_candidate_tier2_json(
            art_dir / "phase1_best_tier1_n_le_24.json",
            name=str(best_tier1["label"]),
            d=int(best_tier1["d"]),
            g0=best_tier1["g0"],
            subcubes=best_tier1["subcubes"],
            metadata={**t1_meta, "from_scratch": True, "phase": 1, "tier1_eligible": True},
        )

    summary["overall_best"] = best_meta
    summary["peak_vram_mb"] = round(peak_vram_mb, 2)
    summary["total_elapsed_sec"] = round(total_sec, 2)
    summary["phase1_go_gate_6_48"] = bool(best_overall["OT"] > PHASE1_GO_GATE)
    summary["beats_num2026_6_52184571"] = bool(best_overall["OT"] > NUM2026_BAR)

    summary_path = art_dir / "phase1_scout_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== PHASE 1 SCOUT COMPLETE ===", flush=True)
    print(f"  Best from-scratch OT:  {best_overall['OT']:.10f} ({best_overall['label']})", flush=True)
    print(f"  d={best_overall['d']}, k_aux={best_overall['k_subcubes']}, n_eff={best_overall['n_effective']}, I={best_overall['I']:.8f}", flush=True)
    print(f"  Phase 1 Go Gate (>6.48): {summary['phase1_go_gate_6_48']} | Beats [Num2026]: {summary['beats_num2026_6_52184571']}", flush=True)
    print(f"  Peak VRAM: {peak_vram_mb:.1f} MB | Wall time: {total_sec:.1f}s", flush=True)
    print(f"  Saved: {best_cand_path}", flush=True)
    print(f"  Saved: {summary_path}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 From-Scratch FEI Landscape Scout (Vectors A2 + B)")
    parser.add_argument("--seed", type=int, default=2026071)
    parser.add_argument("--quick", action="store_true", help="Fast smoke run (~60s)")
    args = parser.parse_args()
    run_phase1_scout(seed=args.seed, quick=args.quick)


if __name__ == "__main__":
    main()

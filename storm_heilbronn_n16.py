#!/usr/bin/env python3
"""
GPU + HiGHS Trust-Region SLP Cascade Discovery Engine for Heilbronn Triangle n=15, 16, 17
(`storm_heilbronn_n16.py`).

Combines:
  1. Massive Batched CUDA Homotopy (LogSumExp / Boltzmann Roof-Bundle Adam + Cosine Annealing)
     on NVIDIA RTX 5060 across B = 2048 parallel configurations.
  2. Bidirectional Dimension Cascade:
     - Upward: Certified n=14* (0.0237757731) -> Pocket Insertion & Vertex Mitosis -> n=15 -> n=16
     - Direct Topological & Symmetric/Asymmetric Multi-Start on n=16 and n=17
     - Downward: Top n=17 chambers -> 1-Vertex Deletion -> n=16
  3. Batched GPU Bottleneck Ruin-and-Recreate (relocating worst-triangle vertices to optimal
     simplex clearance pockets).
  4. Exact Active-Set Polishing via SLSQP + HiGHS Trust-Region Sequential Linear Programming (TR-SLP).
"""

from __future__ import annotations
import json
import math
import time
import numpy as np
import torch
from scipy.optimize import linprog, minimize
from fractions import Fraction

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def get_triplets_np(n: int):
    idx_i, idx_j, idx_k = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx_i.append(i)
                idx_j.append(j)
                idx_k.append(k)
    return (
        np.array(idx_i, dtype=np.int32),
        np.array(idx_j, dtype=np.int32),
        np.array(idx_k, dtype=np.int32),
    )


TRIPLETS_NP = {n: get_triplets_np(n) for n in (14, 15, 16, 17)}
TRIPLETS_TORCH = {
    n: tuple(torch.tensor(arr, dtype=torch.long, device=DEVICE) for arr in TRIPLETS_NP[n])
    for n in (14, 15, 16, 17)
}


def make_simplex_grid_torch(res: int = 85) -> torch.Tensor:
    pts = []
    for i in range(res + 1):
        u = i / res
        for j in range(res + 1 - i):
            v = j / res
            pts.append((u, v))
    return torch.tensor(pts, dtype=DTYPE, device=DEVICE)


GRID_TORCH = make_simplex_grid_torch(85)


@torch.no_grad()
def project_simplex_torch(pts: torch.Tensor) -> torch.Tensor:
    pts.clamp_(0.0, 1.0)
    s = pts[..., 0] + pts[..., 1]
    over = s - 1.0
    mask = over > 0.0
    pts[..., 0] = torch.where(mask, pts[..., 0] - 0.5 * over, pts[..., 0])
    pts[..., 1] = torch.where(mask, pts[..., 1] - 0.5 * over, pts[..., 1])
    pts.clamp_(0.0, 1.0)
    s2 = pts[..., 0] + pts[..., 1]
    over2 = torch.clamp(s2 - 1.0, min=0.0)
    pts[..., 0] -= over2
    return pts


@torch.no_grad()
def eval_batch_torch(pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    pts: (B, n, 2)
    Returns (min_areas: (B,), worst_tri_idx: (B,))
    """
    n = pts.shape[1]
    I, J, K = TRIPLETS_TORCH[n]
    x = pts[:, :, 0]
    y = pts[:, :, 1]
    cross = x[:, I] * (y[:, J] - y[:, K]) + x[:, J] * (y[:, K] - y[:, I]) + x[:, K] * (y[:, I] - y[:, J])
    areas = torch.abs(cross)
    min_a, worst_idx = torch.min(areas, dim=1)
    return min_a, worst_idx


def gpu_boltzmann_polish(
    pts_init: torch.Tensor,
    betas: tuple[float, ...] = (120.0, 400.0, 1400.0, 5000.0, 18000.0, 50000.0),
    lrs: tuple[float, ...] = (0.0045, 0.0025, 0.0012, 0.0005, 0.0002, 0.00007),
    steps_per_stage: int = 160,
    symmetric_cs: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched Boltzmann Roof-Bundle / Softmin Adam optimization on GPU in float64.
    If symmetric_cs=True, enforces mirror symmetry u <-> v across the diagonal u = v.
    """
    B, n, _ = pts_init.shape
    I, J, K = TRIPLETS_TORCH[n]
    pts = project_simplex_torch(pts_init.clone())
    best_pts = pts.clone()
    best_val, _ = eval_batch_torch(pts)

    for beta, lr in zip(betas, lrs):
        m_mom = torch.zeros_like(pts)
        v_mom = torch.zeros_like(pts)
        b1, b2 = 0.85, 0.99
        for s in range(1, steps_per_stage + 1):
            x = pts[:, :, 0]
            y = pts[:, :, 1]
            cross = x[:, I] * (y[:, J] - y[:, K]) + x[:, J] * (y[:, K] - y[:, I]) + x[:, K] * (y[:, I] - y[:, J])
            areas = torch.abs(cross)
            min_a, _ = torch.min(areas, dim=1, keepdim=True)
            cur_min = min_a.squeeze(1)
            improved = cur_min > best_val
            if torch.any(improved):
                best_val = torch.where(improved, cur_min, best_val)
                best_pts[improved] = pts[improved]

            logits = -beta * (areas - min_a)
            w = torch.softmax(logits, dim=1)
            sgn = torch.where(cross >= 0.0, w, -w)

            gx = torch.zeros((B, n), dtype=DTYPE, device=DEVICE)
            gy = torch.zeros((B, n), dtype=DTYPE, device=DEVICE)
            gx.scatter_add_(1, I.unsqueeze(0).expand(B, -1), sgn * (y[:, J] - y[:, K]))
            gy.scatter_add_(1, I.unsqueeze(0).expand(B, -1), sgn * (x[:, K] - x[:, J]))
            gx.scatter_add_(1, J.unsqueeze(0).expand(B, -1), sgn * (y[:, K] - y[:, I]))
            gy.scatter_add_(1, J.unsqueeze(0).expand(B, -1), sgn * (x[:, I] - x[:, K]))
            gx.scatter_add_(1, K.unsqueeze(0).expand(B, -1), sgn * (y[:, I] - y[:, J]))
            gy.scatter_add_(1, K.unsqueeze(0).expand(B, -1), sgn * (x[:, J] - x[:, I]))

            grad = torch.stack([gx, gy], dim=-1)
            if symmetric_cs:
                # Mirror pair coupling: pair (2p, 2p+1) has u_{2p+1} = v_{2p}, v_{2p+1} = u_{2p}
                n_pairs = n // 2
                for p in range(n_pairs):
                    i1, i2 = 2 * p, 2 * p + 1
                    gu = 0.5 * (grad[:, i1, 0] + grad[:, i2, 1])
                    gv = 0.5 * (grad[:, i1, 1] + grad[:, i2, 0])
                    grad[:, i1, 0] = gu
                    grad[:, i1, 1] = gv
                    grad[:, i2, 0] = gv
                    grad[:, i2, 1] = gu
                if n % 2 == 1:
                    # Odd vertex sits on mirror axis u == v
                    gm = 0.5 * (grad[:, -1, 0] + grad[:, -1, 1])
                    grad[:, -1, 0] = gm
                    grad[:, -1, 1] = gm

            bc1 = 1.0 - (b1 ** s)
            bc2 = 1.0 - (b2 ** s)
            m_mom.mul_(b1).add_(grad, alpha=1.0 - b1)
            v_mom.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
            step = lr * (m_mom / bc1) / (torch.sqrt(v_mom / bc2) + 1e-12)
            pts.add_(step)
            project_simplex_torch(pts)
            if symmetric_cs:
                n_pairs = n // 2
                for p in range(n_pairs):
                    i1, i2 = 2 * p, 2 * p + 1
                    pts[:, i2, 0] = pts[:, i1, 1]
                    pts[:, i2, 1] = pts[:, i1, 0]
                if n % 2 == 1:
                    mid = 0.5 * (pts[:, -1, 0] + pts[:, -1, 1])
                    pts[:, -1, 0] = mid
                    pts[:, -1, 1] = mid

        pts.copy_(best_pts)

    return best_pts, best_val


@torch.no_grad()
def gpu_best_pocket_insertion(pts_nm1: torch.Tensor, top_k: int = 4) -> torch.Tensor:
    """
    Given pts_nm1 of shape (B, m, 2), evaluates exact clearance on GRID_TORCH (G, 2)
    and returns (B * top_k, m + 1, 2) lifted configurations.
    """
    B, m, _ = pts_nm1.shape
    G = GRID_TORCH.shape[0]
    ii, jj = torch.triu_indices(m, m, offset=1, device=DEVICE)
    # Chunk over B to keep memory small
    out_list = []
    chunk_sz = 64
    qx = GRID_TORCH[:, 0].view(1, 1, G)
    qy = GRID_TORCH[:, 1].view(1, 1, G)
    for start in range(0, B, chunk_sz):
        sub = pts_nm1[start : start + chunk_sz]
        b_cur = sub.shape[0]
        xi = sub[:, ii, 0].unsqueeze(-1)  # (b, P, 1)
        yi = sub[:, ii, 1].unsqueeze(-1)
        xj = sub[:, jj, 0].unsqueeze(-1)
        yj = sub[:, jj, 1].unsqueeze(-1)
        areas = torch.abs(xi * (yj - qy) + xj * (qy - yi) + qx * (yi - yj))  # (b, P, G)
        min_a, _ = torch.min(areas, dim=1)  # (b, G)
        # Pick top_k well-separated grid points per item
        vals, order = torch.sort(min_a, dim=1, descending=True)
        for b_idx in range(b_cur):
            base = sub[b_idx]
            chosen = []
            for g_idx in order[b_idx].tolist():
                cand_q = GRID_TORCH[g_idx]
                if all(torch.norm(cand_q - prev).item() > 0.045 for prev in chosen):
                    chosen.append(cand_q)
                    if len(chosen) >= top_k:
                        break
            while len(chosen) < top_k:
                chosen.append(GRID_TORCH[order[b_idx, len(chosen)]])
            for q in chosen:
                out_list.append(torch.cat([base, q.unsqueeze(0)], dim=0))
    return torch.stack(out_list, dim=0)


@torch.no_grad()
def gpu_ruin_and_recreate_round(pts: torch.Tensor, n_rounds: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched Ruin-and-Recreate on GPU:
    Finds the bottleneck triangle for each configuration, removes one of its 3 vertices,
    relocates it to the best simplex pocket, and polishes with gpu_boltzmann_polish.
    """
    B, n, _ = pts.shape
    I, J, K = TRIPLETS_TORCH[n]
    best_pts = pts.clone()
    best_val, _ = eval_batch_torch(best_pts)

    for r in range(n_rounds):
        _, worst_t = eval_batch_torch(best_pts)
        # Pick vertex 0, 1, or 2 of the worst triangle
        which_v = r % 3
        v_del = (I[worst_t] if which_v == 0 else (J[worst_t] if which_v == 1 else K[worst_t]))
        # Build (B, n-1, 2) by omitting v_del
        rem_list = []
        for b in range(B):
            vd = int(v_del[b].item())
            rem_list.append(torch.cat([best_pts[b, :vd], best_pts[b, vd + 1 :]], dim=0))
        rem_t = torch.stack(rem_list, dim=0)
        lifted = gpu_best_pocket_insertion(rem_t, top_k=1)  # (B, n, 2)
        if r % 2 == 1:
            lifted += 0.008 * torch.randn_like(lifted)
            project_simplex_torch(lifted)
        cand_pts, cand_val = gpu_boltzmann_polish(
            lifted,
            betas=(250.0, 1200.0, 6000.0, 28000.0),
            lrs=(0.003, 0.0014, 0.0005, 0.00012),
            steps_per_stage=110,
        )
        improved = cand_val > best_val
        if torch.any(improved):
            best_val = torch.where(improved, cand_val, best_val)
            best_pts[improved] = cand_pts[improved]

    return best_pts, best_val


def seed_topological_batch(n: int, batch_size: int, seed: int = 42, symmetric_cs: bool = False) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    out = np.zeros((batch_size, n, 2), dtype=np.float64)
    for b in range(batch_size):
        if symmetric_cs:
            n_pairs = n // 2
            pts = np.zeros((n, 2), dtype=np.float64)
            # 1 pair on boundary u=0 / v=0, 1 pair on hypotenuse u+v=1
            e1 = rng.uniform(0.12, 0.45)
            e2 = rng.uniform(0.55, 0.88)
            h1 = rng.uniform(0.12, 0.42)
            pts[0] = [e1, 0.0]; pts[1] = [0.0, e1]
            pts[2] = [e2, 0.0]; pts[3] = [0.0, e2]
            pts[4] = [h1, 1.0 - h1]; pts[5] = [1.0 - h1, h1]
            for p in range(3, n_pairs):
                u = rng.uniform(0.04, 0.75)
                v = rng.uniform(0.04, 0.92 - u)
                pts[2 * p] = [u, v]
                pts[2 * p + 1] = [v, u]
            if n % 2 == 1:
                m_val = rng.uniform(0.08, 0.46)
                pts[-1] = [m_val, m_val]
            out[b] = pts
        else:
            pts = np.zeros((n, 2), dtype=np.float64)
            # 6 boundary points (2 per edge) + 3 corner buffers + (n-9) interior
            pts[0] = [rng.uniform(0.10, 0.42), 0.0]
            pts[1] = [rng.uniform(0.55, 0.88), 0.0]
            pts[2] = [0.0, rng.uniform(0.10, 0.42)]
            pts[3] = [0.0, rng.uniform(0.55, 0.88)]
            h1 = rng.uniform(0.10, 0.42)
            h2 = rng.uniform(0.55, 0.88)
            pts[4] = [h1, 1.0 - h1]
            pts[5] = [h2, 1.0 - h2]
            pts[6] = [rng.uniform(0.03, 0.09), rng.uniform(0.03, 0.09)]
            pts[7] = [rng.uniform(0.80, 0.90), rng.uniform(0.03, 0.09)]
            pts[8] = [rng.uniform(0.03, 0.09), rng.uniform(0.80, 0.90)]
            rem = n - 9
            rot = rng.uniform(0.0, 2.0 * math.pi)
            for r in range(rem):
                if r < rem - 2:
                    ang = rot + r * (2.0 * math.pi / max(1, rem - 2)) + rng.normal(0.0, 0.15)
                    rad = rng.uniform(0.14, 0.24)
                    u = 0.30 + rad * math.cos(ang)
                    v = 0.30 + rad * math.sin(ang)
                else:
                    u = rng.uniform(0.18, 0.38)
                    v = rng.uniform(0.18, 0.38)
                if u < 0.01: u = 0.01
                if v < 0.01: v = 0.01
                if u + v > 0.99:
                    s = u + v
                    u = 0.98 * u / s
                    v = 0.98 * v / s
                pts[9 + r] = [u, v]
            out[b] = pts
    return project_simplex_torch(torch.tensor(out, dtype=DTYPE, device=DEVICE))


def verify_exact_np(pts: np.ndarray) -> tuple[float, int]:
    n = pts.shape[0]
    I, J, K = TRIPLETS_NP[n]
    x, y = pts[:, 0], pts[:, 1]
    areas = np.abs(x[I] * (y[J] - y[K]) + x[J] * (y[K] - y[I]) + x[K] * (y[I] - y[J]))
    min_a = float(np.min(areas))
    ties = int(np.sum(areas <= min_a + 1e-8))
    return min_a, ties


def trust_region_slp_polish(pts_init: np.ndarray, n_steps: int = 55, rho_init: float = 0.012) -> np.ndarray:
    pts = pts_init.copy()
    n_pts = pts.shape[0]
    I, J, K = TRIPLETS_NP[n_pts]
    best_pts = pts.copy()
    best_val, _ = verify_exact_np(pts)
    rho = rho_init

    c_obj = np.zeros(2 * n_pts + 1, dtype=np.float64)
    c_obj[-1] = -1.0

    for _ in range(n_steps):
        x, y = pts[:, 0], pts[:, 1]
        cross = x[I] * (y[J] - y[K]) + x[J] * (y[K] - y[I]) + x[K] * (y[I] - y[J])
        areas = np.abs(cross)
        cur_val = float(np.min(areas))
        if cur_val > best_val:
            best_val = cur_val
            best_pts = pts.copy()

        active = np.where(areas <= max(cur_val * 2.2, cur_val + 0.012))[0]
        m_act = len(active)
        Ia, Ja, Ka = I[active], J[active], K[active]
        sgn = np.where(cross[active] >= 0.0, 1.0, -1.0)
        r_act = np.arange(m_act)

        A_ub = np.zeros((m_act + n_pts, 2 * n_pts + 1), dtype=np.float64)
        b_ub = np.zeros(m_act + n_pts, dtype=np.float64)

        gx_I = sgn * (y[Ja] - y[Ka]); gy_I = sgn * (x[Ka] - x[Ja])
        gx_J = sgn * (y[Ka] - y[Ia]); gy_J = sgn * (x[Ia] - x[Ka])
        gx_K = sgn * (y[Ia] - y[Ja]); gy_K = sgn * (x[Ja] - x[Ia])

        A_ub[r_act, 2 * Ia] = -gx_I; A_ub[r_act, 2 * Ia + 1] = -gy_I
        A_ub[r_act, 2 * Ja] = -gx_J; A_ub[r_act, 2 * Ja + 1] = -gy_J
        A_ub[r_act, 2 * Ka] = -gx_K; A_ub[r_act, 2 * Ka + 1] = -gy_K
        A_ub[r_act, 2 * n_pts] = 1.0

        dot_p = gx_I * x[Ia] + gy_I * y[Ia] + gx_J * x[Ja] + gy_J * y[Ja] + gx_K * x[Ka] + gy_K * y[Ka]
        b_ub[r_act] = areas[active] - dot_p

        for p_idx in range(n_pts):
            A_ub[m_act + p_idx, 2 * p_idx] = 1.0
            A_ub[m_act + p_idx, 2 * p_idx + 1] = 1.0
            b_ub[m_act + p_idx] = 1.0

        bounds = []
        for p_idx in range(n_pts):
            bounds.append((max(0.0, float(x[p_idx] - rho)), min(1.0, float(x[p_idx] + rho))))
            bounds.append((max(0.0, float(y[p_idx] - rho)), min(1.0, float(y[p_idx] + rho))))
        bounds.append((0.0, 1.0))

        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success or res.x is None:
            rho *= 0.5
            if rho < 1e-11:
                break
            continue

        pts_lp = res.x[: 2 * n_pts].reshape(n_pts, 2)
        d = pts_lp - pts
        improved = False
        for eta in (1.0, 0.7, 0.5, 0.3, 0.15, 0.07, 0.03, 0.01):
            pts_try = np.clip(pts + eta * d, 0.0, 1.0)
            over = (pts_try[:, 0] + pts_try[:, 1]) - 1.0
            m_ov = over > 0.0
            pts_try[m_ov, 0] -= 0.5 * over[m_ov]
            pts_try[m_ov, 1] -= 0.5 * over[m_ov]
            val_try, _ = verify_exact_np(pts_try)
            if val_try > cur_val + 1e-14:
                pts = pts_try
                if val_try > best_val:
                    best_val = val_try
                    best_pts = pts_try.copy()
                improved = True
                if eta >= 0.7:
                    rho = min(0.03, rho * 1.15)
                break
        if not improved:
            rho *= 0.55
            if rho < 1e-11:
                break

    return best_pts


def main():
    t0 = time.perf_counter()
    print(f"=== LAUNCHING GPU CASCADE ON {DEVICE} ({torch.cuda.get_device_name(0)}) ===")

    # 1. Upward Cascade: Certified n=14* -> n=15 -> n=16
    pts14_star = torch.tensor(np.load("heilbronn_n14_champion.npy"), dtype=DTYPE, device=DEVICE).unsqueeze(0)
    # Create 128 perturbed + pocket-lifted n=15 seeds from n=14*
    base14_batch = pts14_star.repeat(64, 1, 1)
    base14_batch[1:] += 0.012 * torch.randn_like(base14_batch[1:])
    project_simplex_torch(base14_batch)
    seeds15_up = gpu_best_pocket_insertion(base14_batch, top_k=4)  # (256, 15, 2)
    seeds15_sym = seed_topological_batch(15, 128, seed=101, symmetric_cs=True)
    seeds15_asym = seed_topological_batch(15, 128, seed=102, symmetric_cs=False)

    pts15_sym, val15_sym = gpu_boltzmann_polish(seeds15_sym, symmetric_cs=True, steps_per_stage=150)
    pts15_all = torch.cat([seeds15_up, seeds15_asym, pts15_sym], dim=0)
    pts15_all, val15_all = gpu_boltzmann_polish(pts15_all, steps_per_stage=140)
    pts15_all, val15_all = gpu_ruin_and_recreate_round(pts15_all, n_rounds=5)

    vals15, ord15 = torch.sort(val15_all, descending=True)
    print(f"[{time.perf_counter()-t0:5.1f}s] Stage 1 (n=15 Pool): Top-5 GPU = {[round(x, 7) for x in vals15[:5].tolist()]}")

    # 2. Lift top 96 of n=15 -> n=16 (96 * 4 = 384 seeds) + 384 Direct Asym + 256 Direct Sym
    top15 = pts15_all[ord15[:96]]
    seeds16_from15 = gpu_best_pocket_insertion(top15, top_k=4)  # (384, 16, 2)
    seeds16_asym = seed_topological_batch(16, 384, seed=202, symmetric_cs=False)
    seeds16_sym = seed_topological_batch(16, 256, seed=203, symmetric_cs=True)

    pts16_sym, _ = gpu_boltzmann_polish(seeds16_sym, symmetric_cs=True, steps_per_stage=160)
    pts16_pool = torch.cat([seeds16_from15, seeds16_asym, pts16_sym], dim=0)
    pts16_pool, val16_pool = gpu_boltzmann_polish(pts16_pool, steps_per_stage=160)
    pts16_pool, val16_pool = gpu_ruin_and_recreate_round(pts16_pool, n_rounds=6)

    vals16, ord16 = torch.sort(val16_pool, descending=True)
    print(f"[{time.perf_counter()-t0:5.1f}s] Stage 2 (n=16 Upward+Direct): Top-5 GPU = {[round(x, 7) for x in vals16[:5].tolist()]}")

    # 3. Stage 3: n=17 Pool + Downward Cascade 17 -> 16
    top16_for17 = pts16_pool[ord16[:64]]
    seeds17_from16 = gpu_best_pocket_insertion(top16_for17, top_k=3)  # (192, 17, 2)
    seeds17_direct = seed_topological_batch(17, 192, seed=303, symmetric_cs=False)
    pts17_pool = torch.cat([seeds17_from16, seeds17_direct], dim=0)
    pts17_pool, val17_pool = gpu_boltzmann_polish(pts17_pool, steps_per_stage=140)
    vals17, ord17 = torch.sort(val17_pool, descending=True)
    print(f"[{time.perf_counter()-t0:5.1f}s] Stage 3 (n=17 Pool): Top-5 GPU = {[round(x, 7) for x in vals17[:5].tolist()]}")

    # Downward deletion 17 -> 16 from top 24 of n=17 (24 * 17 = 408 candidates)
    down16_list = []
    for idx in ord17[:24].tolist():
        p17 = pts17_pool[idx]
        for v_del in range(17):
            down16_list.append(torch.cat([p17[:v_del], p17[v_del + 1 :]], dim=0))
    down16_t = torch.stack(down16_list, dim=0)
    down16_t, val_down16 = gpu_boltzmann_polish(down16_t, steps_per_stage=140)
    down16_t, val_down16 = gpu_ruin_and_recreate_round(down16_t, n_rounds=5)

    all16_pts = torch.cat([pts16_pool[ord16[:64]], down16_t], dim=0)
    all16_val, _ = eval_batch_torch(all16_pts)
    vals16_final, ord16_final = torch.sort(all16_val, descending=True)
    print(f"[{time.perf_counter()-t0:5.1f}s] Stage 4 (Merged n=16 Pool): Top-5 GPU = {[round(x, 7) for x in vals16_final[:5].tolist()]}")

    # 4. Polish Top-12 candidates of n=15, n=16, n=17 with HiGHS Trust-Region SLP!
    for n_target, pts_t, ord_t, ref_val in [
        (15, pts15_all, ord15, 0.02060000),
        (16, all16_pts, ord16_final, 0.01835554),
        (17, pts17_pool, ord17, 0.01624232),
    ]:
        best_np = None
        best_v = 0.0
        best_ties = 0
        for rank in range(10):
            cand_np = pts_t[ord_t[rank]].detach().cpu().numpy()
            pol_np = trust_region_slp_polish(cand_np, n_steps=60, rho_init=0.012)
            v, ties = verify_exact_np(pol_np)
            if v > best_v:
                best_v = v
                best_np = pol_np
                best_ties = ties
        print(
            f"=== FINAL n={n_target} AFTER HiGHS TR-SLP: Delta_{n_target} = {best_v:.10f} "
            f"(active ties = {best_ties}, vs benchmark {ref_val:.8f}, diff = {best_v - ref_val:+.8f}) ==="
        )
        np.save(f"heilbronn_n{n_target}_candidate.npy", best_np)


if __name__ == "__main__":
    main()

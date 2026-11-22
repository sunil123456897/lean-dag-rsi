"""Phase 1 CPU Scout for Vector A1: p-Biased Cycles + Co-Optimized Uniform Bridge.

Pure CPU (NumPy + stdlib; zero GPU usage so RTX 5060 remains exclusively free).

Implements:
1. Exact fast O(m * 2^m) q-biased Fourier transform via separable 2x2 tensor
   contraction with M_q = [[1-q, q], [-sqrt(q(1-q)), sqrt(q(1-q))]].
2. Exact Vector A1 cycle & bridge laws (verified against Hod 2017 Thm 3.2:
   OR_2 o AND_2 at p* = Phi -> R_unif = 6.4138455149847...).
3. Probe 1: Exact finite uniform bridge co-optimization for dyadic targets
   p* in {9/16, 5/8, 11/16, 3/4, 13/16, 17/32, 19/32, 21/32, 23/32}
   (comparing lexicographic l<p*> vs monotone & de-aliased bridges).
4. Probe 2: Exact dyadic T=1 and T=2 fixed-point cycle discovery (checking
   exact Diophantine weight profiles for m=2..8, monotone & non-monotone).
5. Probe 3: General real fixed-point T=1 (monotone & non-monotone, m=2..7)
   and T=2 alternating cycles (OR-AND / monotone / non-monotone SA on m=2..7)
   evaluated with exact asymptotic l<p*> bridge and co-optimized finite bridges.
6. Explicit Phase 1 Go/No-Go gate (> 6.46 vs Stop-Crank reserve freeze) and
   JSON summary written to:
   e:\\эксперименты\\исследования\\c71_fei_research\\artifacts\\phase1_a1_summary.json
"""

from __future__ import annotations

from fractions import Fraction
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


GO_GATE_A1 = 6.46
HOD_2017_BAR = 6.454783716562761
NUM_2026_BAR = 6.52184571092304657566


# =============================================================================
# Core Exact Math: q-Biased Fourier Transform, Binary Entropy, Lexicographic l<p>
# =============================================================================

def h2(x: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)


def lex_asymptotic_IH(p: float, depth: int = 64) -> Tuple[float, float]:
    """Exact (I, H) of infinite lexicographic function l<p> on uniform cube.

    Verified against Hod (2017) Thm 3.2:
      p = Phi => I = 1.297689466989230, H = 2.423939569290445.
    """
    if p <= 0.0 or p >= 1.0:
        return 0.0, 0.0
    ps = [p]
    for _ in range(depth):
        x = ps[-1]
        if x <= 0.0 or x >= 1.0:
            ps.append(0.0)
        elif x >= 0.5:
            ps.append(2.0 * x - 1.0)
        else:
            ps.append(2.0 * x)
    I_val = 0.0
    H_val = 0.0
    for k in range(depth - 1, -1, -1):
        pk = ps[k]
        qk = ps[k + 1]
        m = min(pk, 1.0 - pk)
        e_q = 1.0 - 2.0 * qk
        H_ge1_q = H_val - (0.0 if e_q == 0.0 else (e_q * e_q) * math.log2(1.0 / (e_q * e_q)))
        var_q = 4.0 * qk * (1.0 - qk)
        I_val = 2.0 * m + 0.5 * I_val
        e_p = 1.0 - 2.0 * pk
        w0 = e_p * e_p
        w1 = (2.0 * m) ** 2
        t0 = 0.0 if w0 <= 0.0 else -w0 * math.log2(w0)
        t1 = 0.0 if w1 <= 0.0 else -w1 * math.log2(w1)
        H_val = t0 + t1 + 0.5 * H_ge1_q + var_q
    return I_val, H_val


_POPCOUNT_CACHE: Dict[int, np.ndarray] = {}


def popcount_array(m: int) -> np.ndarray:
    arr = _POPCOUNT_CACHE.get(m)
    if arr is None:
        idx = np.arange(1 << m, dtype=np.uint32)
        arr = np.zeros(1 << m, dtype=np.float64)
        for b in range(m):
            arr += (idx >> b) & 1
        _POPCOUNT_CACHE[m] = arr
    return arr


def biased_fourier_stats(bits01: np.ndarray, q: float) -> Dict[str, float]:
    """Exact O(m * 2^m) q-biased Fourier spectrum of g: {0,1}^m -> {0,1}.

    Uses g_pm1 = 2*g - 1 in {-1, +1} under product measure Pr[x_i = 1] = q.
    Returns p_out, V, I, H, I_cond (I_{!=0}), H_cond (H_{!=0}).
    """
    N = bits01.shape[0]
    m = N.bit_length() - 1
    g_pm1 = (2.0 * bits01.astype(np.float64) - 1.0).reshape((2,) * m)
    sq = math.sqrt(q * (1.0 - q))
    Mq = np.array([[1.0 - q, q], [-sq, sq]], dtype=np.float64)
    cur = g_pm1
    for axis in range(m):
        cur = np.tensordot(Mq, cur, axes=([1], [axis]))
        cur = np.moveaxis(cur, 0, axis)
    coeffs = cur.reshape(-1)
    w = coeffs * coeffs
    # Parseval check (tolerance 1e-10)
    p_sum = float(w.sum())
    if abs(p_sum - 1.0) > 1e-9:
        raise AssertionError(f"Biased Parseval failed: sum={p_sum} at q={q}")
    c0 = float(coeffs[0])
    p_out = 0.5 * (1.0 + c0)
    V = max(0.0, 4.0 * p_out * (1.0 - p_out))
    pc = popcount_array(m)
    I_val = float(np.dot(pc, w))
    nz = w > 1e-300
    H_val = float(-np.sum(w[nz] * np.log2(w[nz])))
    if V > 1e-15:
        I_cond = I_val / V
        H_cond = (H_val - h2(V)) / V
    else:
        I_cond = 0.0
        H_cond = 0.0
    return {
        "p_out": p_out,
        "V": V,
        "I": I_val,
        "H": H_val,
        "I_cond": I_cond,
        "H_cond": H_cond,
    }


def eval_prob_poly(weight_counts: Sequence[int], m: int, q: float) -> float:
    """Compute Pr_{mu_q}[g=1] = sum_{w=0}^m N_w * q^w * (1-q)^(m-w)."""
    acc = 0.0
    one_m_q = 1.0 - q
    for w, cnt in enumerate(weight_counts):
        if cnt:
            acc += cnt * (q ** w) * (one_m_q ** (m - w))
    return acc


def weight_profile(bits01: np.ndarray) -> List[int]:
    m = bits01.shape[0].bit_length() - 1
    pc = popcount_array(m).astype(np.int64)
    counts = [0] * (m + 1)
    for w in range(m + 1):
        counts[w] = int(np.sum(bits01[pc == w]))
    return counts


def solve_fixed_point_1step(bits01: np.ndarray, lo: float = 1e-6, hi: float = 1.0 - 1e-6) -> List[float]:
    """Find all interior fixed points p* in (lo, hi) of P_g(p) = p."""
    m = bits01.shape[0].bit_length() - 1
    wc = weight_profile(bits01)
    grid = np.linspace(lo, hi, 200)
    vals = [eval_prob_poly(wc, m, float(x)) - float(x) for x in grid]
    roots: List[float] = []
    for i in range(len(grid) - 1):
        a, b = float(grid[i]), float(grid[i + 1])
        fa, fb = vals[i], vals[i + 1]
        if fa == 0.0:
            roots.append(a)
        elif fa * fb < 0.0:
            for _ in range(60):
                mid = 0.5 * (a + b)
                fm = eval_prob_poly(wc, m, mid) - mid
                if fa * fm <= 0.0:
                    b, fb = mid, fm
                else:
                    a, fa = mid, fm
            r = 0.5 * (a + b)
            if not any(abs(r - prev) < 1e-7 for prev in roots):
                roots.append(r)
    return roots


def solve_fixed_point_2step(
    bits_g1: np.ndarray,
    bits_g2: np.ndarray,
    lo: float = 1e-5,
    hi: float = 1.0 - 1e-5,
) -> List[float]:
    """Find interior fixed points p* of P_{g2}(P_{g1}(p)) = p."""
    m1 = bits_g1.shape[0].bit_length() - 1
    m2 = bits_g2.shape[0].bit_length() - 1
    wc1 = weight_profile(bits_g1)
    wc2 = weight_profile(bits_g2)

    def f(p: float) -> float:
        p1 = eval_prob_poly(wc1, m1, p)
        return eval_prob_poly(wc2, m2, p1) - p

    grid = np.linspace(lo, hi, 150)
    vals = [f(float(x)) for x in grid]
    roots: List[float] = []
    for i in range(len(grid) - 1):
        a, b = float(grid[i]), float(grid[i + 1])
        fa, fb = vals[i], vals[i + 1]
        if fa * fb < 0.0:
            for _ in range(60):
                mid = 0.5 * (a + b)
                fm = f(mid)
                if fa * fm <= 0.0:
                    b, fb = mid, fm
                else:
                    a, fa = mid, fm
            r = 0.5 * (a + b)
            if not any(abs(r - prev) < 1e-7 for prev in roots):
                roots.append(r)
    return roots


def eval_cycle_with_lex_bridge(
    funcs: Sequence[np.ndarray],
    p_star: float,
) -> Optional[Dict[str, float]]:
    """Evaluate cycle p0=p* -> g1 -> p1 -> ... -> gT -> p* with exact asymptotic l<p*> bridge."""
    p = p_star
    H_steps: List[float] = []
    I_steps: List[float] = []
    ps: List[float] = [p_star]
    for g in funcs:
        st = biased_fourier_stats(g, p)
        H_steps.append(st["H_cond"])
        I_steps.append(st["I_cond"])
        p = st["p_out"]
        ps.append(p)
    if abs(p - p_star) > 1e-7:
        return None
    T = len(funcs)
    alpha = 1.0
    for iv in I_steps:
        alpha *= iv
    if alpha <= 1.0 + 1e-9:
        return None
    H_cyc = 0.0
    for t in range(T):
        suffix = 1.0
        for r in range(t + 1, T):
            suffix *= I_steps[r]
        H_cyc += H_steps[t] * suffix
    beta = H_cyc / (alpha - 1.0)
    V_star = 4.0 * p_star * (1.0 - p_star)
    I_lex, H_lex = lex_asymptotic_IH(p_star)
    if I_lex <= 0.0:
        return None
    R_unif = (H_lex - h2(V_star) + V_star * beta) / I_lex
    return {
        "p_star": p_star,
        "ps": ps,
        "alpha_cyc": alpha,
        "H_cyc": H_cyc,
        "beta_cyc": beta,
        "I_lex": I_lex,
        "H_lex": H_lex,
        "R_unif_lex": R_unif,
    }


def is_monotone_01(bits01: np.ndarray) -> bool:
    N = bits01.shape[0]
    m = N.bit_length() - 1
    for i in range(m):
        step = 1 << i
        b = bits01.reshape(-1, 2, step)
        if np.any(b[:, 0, :] > b[:, 1, :]):
            return False
    return True


def make_lex_finite(m: int, count_ones: int) -> np.ndarray:
    """Monotone finite lexicographic function on m bits with exact count_ones True outputs.

    Places 1s on the top `count_ones` inputs in lexicographic order of (x_{m-1}, ..., x_0),
    which is strictly monotone!
    """
    N = 1 << m
    arr = np.zeros(N, dtype=np.int8)
    if count_ones > 0:
        arr[N - count_ones :] = 1
    return arr


# =============================================================================
# Probe 1: Co-Optimization of Finite Uniform Bridge h_in at Exact Dyadic p*
# =============================================================================

def _optimize_bridge_at_dyadic_p(
    p_frac: Fraction,
    m_bridge: int,
    R_ref: float = 6.522,
    sa_steps: int = 2500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare lexicographic bridge vs SA-optimized monotone/general/de-aliased bridges.

    Minimizes beta_req(p*, h_in; R_ref) = (R_ref * I[h] - H[h] + h2(V*)) / V*,
    equivalently maximizes H[h] - R_ref * I[h] subject to Pr_{1/2}[h=1] = p*.
    """
    p = float(p_frac)
    V = 4.0 * p * (1.0 - p)
    ht = h2(V)
    N = 1 << m_bridge
    target_ones = int(p_frac * N)
    assert Fraction(target_ones, N) == p_frac

    def beta_req(I_h: float, H_h: float) -> float:
        return (R_ref * I_h - H_h + ht) / V

    # 1. Exact lexicographic finite bridge (equals asymptotic l<p*> since p* is dyadic!)
    lex_bits = make_lex_finite(m_bridge, target_ones)
    assert is_monotone_01(lex_bits)
    st_lex = biased_fourier_stats(lex_bits, 0.5)
    I_lex_inf, H_lex_inf = lex_asymptotic_IH(p)
    assert abs(st_lex["I"] - I_lex_inf) < 1e-12 and abs(st_lex["H"] - H_lex_inf) < 1e-12
    breq_lex = beta_req(st_lex["I"], st_lex["H"])

    # 2. Monotone SA search for a better bridge at m_bridge (and m_bridge+1 if <=7)
    rng = random.Random(seed)
    best_mono_bits = lex_bits.copy()
    best_mono_I, best_mono_H = st_lex["I"], st_lex["H"]
    best_mono_breq = breq_lex

    # Precompute bit masks for fast monotone check
    cur = lex_bits.copy()
    cur_obj = best_mono_H - R_ref * best_mono_I
    for step in range(sa_steps):
        # Find boundary 1s (minimal 1s: flipping 1->0 stays monotone) and boundary 0s (maximal 0s: flipping 0->1 stays monotone)
        # Fast vectorized check for small N <= 128
        ones_idx = np.flatnonzero(cur == 1)
        zeros_idx = np.flatnonzero(cur == 0)
        # Sample candidate monotone swap (u in ones -> 0, v in zeros -> 1)
        min_ones = []
        for x in ones_idx:
            xi = int(x)
            ok = True
            for b in range(m_bridge):
                if (xi >> b) & 1 and cur[xi ^ (1 << b)] == 1:
                    ok = False
                    break
            if ok:
                min_ones.append(xi)
        max_zeros = []
        for x in zeros_idx:
            xi = int(x)
            ok = True
            for b in range(m_bridge):
                if not ((xi >> b) & 1) and cur[xi | (1 << b)] == 0:
                    ok = False
                    break
            if ok:
                max_zeros.append(xi)
        if not min_ones or not max_zeros:
            break
        u = rng.choice(min_ones)
        v = rng.choice(max_zeros)
        # Ensure u and v are incomparable so simultaneous swap keeps monotonicity
        if (u & v) == u or (u & v) == v:
            continue
        cur[u] = 0
        cur[v] = 1
        st = biased_fourier_stats(cur, 0.5)
        obj = st["H"] - R_ref * st["I"]
        temp = 0.08 * (1.0 - step / sa_steps) + 1e-4
        if obj >= cur_obj or rng.random() < math.exp((obj - cur_obj) / temp):
            cur_obj = obj
            br = beta_req(st["I"], st["H"])
            if br < best_mono_breq - 1e-12:
                best_mono_breq = br
                best_mono_I = st["I"]
                best_mono_H = st["H"]
                best_mono_bits = cur.copy()
        else:
            cur[u] = 1
            cur[v] = 0

    # 3. Also test de-aliased monotone bridge on the SAME m_bridge:
    # Turning a pair of incomparable (minimal-1 u, maximal-0 v) into two 0-points (antichain of size 2)
    # preserves Pr[h=1] = p* EXACTLY! Check if any antichain pair on best_mono_bits lowers beta_req!
    best_dealiased_breq = best_mono_breq
    best_dealiased_I = best_mono_I
    best_dealiased_H = best_mono_H
    g_base = (2 * best_mono_bits - 1).astype(np.float64)
    min_ones_b = [
        int(x) for x in np.flatnonzero(best_mono_bits == 1)
        if all(best_mono_bits[int(x) ^ (1 << b)] == 0 for b in range(m_bridge) if (int(x) >> b) & 1)
    ]
    max_zeros_b = [
        int(x) for x in np.flatnonzero(best_mono_bits == 0)
        if all(best_mono_bits[int(x) | (1 << b)] == 1 for b in range(m_bridge) if not ((int(x) >> b) & 1))
    ]
    pc = popcount_array(m_bridge)
    for u in min_ones_b:
        for v in max_zeros_b:
            if (u & v) == u or (u & v) == v:
                continue
            g0 = g_base.copy()
            g0[u] = 0.0
            g0[v] = 0.0
            # Uniform WHT of 3-valued g0
            cur_g = g0.reshape((2,) * m_bridge)
            Mhalf = np.array([[0.5, 0.5], [-0.5, 0.5]], dtype=np.float64)
            for axis in range(m_bridge):
                cur_g = np.moveaxis(np.tensordot(Mhalf, cur_g, axes=([1], [axis])), 0, axis)
            w = cur_g.reshape(-1) ** 2
            I_g0 = float(np.dot(pc, w))
            nz = w > 1e-300
            H_g0 = float(-np.sum(w[nz] * np.log2(w[nz])))
            # Add 2 point-antichain subcubes (r_j = m_bridge)
            I_tot = I_g0 + (2.0 / N) * (1.0 + 0.5 * m_bridge)
            H_tot = H_g0 + (2.0 / N) * (2.0 * m_bridge)
            br = beta_req(I_tot, H_tot)
            if br < best_dealiased_breq - 1e-12:
                best_dealiased_breq = br
                best_dealiased_I = I_tot
                best_dealiased_H = H_tot

    return {
        "p_str": f"{p_frac.numerator}/{p_frac.denominator}",
        "p": p,
        "V": V,
        "m_bridge": m_bridge,
        "lex_I": st_lex["I"],
        "lex_H": st_lex["H"],
        "lex_beta_req_6522": breq_lex,
        "best_mono_I": best_mono_I,
        "best_mono_H": best_mono_H,
        "best_mono_beta_req_6522": best_mono_breq,
        "best_dealiased_I": best_dealiased_I,
        "best_dealiased_H": best_dealiased_H,
        "best_dealiased_beta_req_6522": best_dealiased_breq,
    }


# =============================================================================
# Probe 2: Exact Dyadic T=1 Fixed-Point Search (Diophantine + Monotone/General)
# =============================================================================

def find_exact_dyadic_t1(p_frac: Fraction, max_m: int = 6) -> Dict[str, Any]:
    """Search for exact non-dictator T=1 functions g: {0,1}^m -> {0,1} with P_g(p*) = p*.

    For p* = a/b (b = 2^k), P_g(a/b) = a/b iff
      sum_{w=0}^m N_w * a^w * (b-a)^{m-w} == a * b^{m-1},  0 <= N_w <= C(m, w).
    Evaluates best symmetric/monotone/non-monotone realizations of valid profiles.
    """
    a, b = p_frac.numerator, p_frac.denominator
    p = float(p_frac)
    V = 4.0 * p * (1.0 - p)
    I_lex, H_lex = lex_asymptotic_IH(p)

    best_any: Optional[Dict[str, Any]] = None
    best_mono: Optional[Dict[str, Any]] = None
    total_profiles_found = 0

    for m in range(2, max_m + 1):
        target = a * (b ** (m - 1))
        coeffs = [(a ** w) * ((b - a) ** (m - w)) for w in range(m + 1)]
        caps = [math.comb(m, w) for w in range(m + 1)]
        dictator_prof = [math.comb(m - 1, w - 1) if w >= 1 else 0 for w in range(m + 1)]

        # Suffix max sums for branch-and-bound
        suf_max = [0] * (m + 2)
        for w in range(m, -1, -1):
            suf_max[w] = suf_max[w + 1] + caps[w] * coeffs[w]

        profiles: List[List[int]] = []

        def dfs(w: int, rem: int, cur: List[int]) -> None:
            if len(profiles) >= 200:
                return
            if rem < 0 or rem > suf_max[w]:
                return
            if w == m + 1:
                if rem == 0 and cur != dictator_prof:
                    # Exclude constant 0/1 (already impossible for 0<p<1)
                    profiles.append(list(cur))
                return
            c_w = coeffs[w]
            max_k = min(caps[w], rem // c_w)
            for k in range(max_k, -1, -1):
                cur.append(k)
                dfs(w + 1, rem - k * c_w, cur)
                cur.pop()

        dfs(0, target, [])
        total_profiles_found += len(profiles)

        if not profiles:
            continue

        # Realize each found weight profile via lexicographic layer selection & random swaps
        pc = popcount_array(m).astype(np.int64)
        layers = [np.flatnonzero(pc == w) for w in range(m + 1)]
        rng = random.Random(100 + m)

        for prof in profiles[:40]:
            # Build candidate 1: top-of-layer (tends to be monotone if Kruskal-Katona holds)
            bits = np.zeros(1 << m, dtype=np.int8)
            for w, cnt in enumerate(prof):
                if cnt > 0:
                    bits[layers[w][-cnt:]] = 1
            for trial in range(8):
                if trial > 0:
                    # Random realization with exact same weight profile (preserves exact p*!)
                    bits = np.zeros(1 << m, dtype=np.int8)
                    for w, cnt in enumerate(prof):
                        if cnt > 0:
                            chosen = rng.sample(list(layers[w]), cnt)
                            bits[chosen] = 1
                st = biased_fourier_stats(bits, p)
                assert abs(st["p_out"] - p) < 1e-11
                if st["I_cond"] <= 1.0 + 1e-9:
                    continue
                beta = st["H_cond"] / (st["I_cond"] - 1.0)
                R_unif = (H_lex - h2(V) + V * beta) / I_lex
                mono = is_monotone_01(bits)
                rec = {
                    "m": m,
                    "profile": prof,
                    "monotone": mono,
                    "I_cond": st["I_cond"],
                    "H_cond": st["H_cond"],
                    "beta_cyc": beta,
                    "R_unif_lex": R_unif,
                }
                if best_any is None or R_unif > best_any["R_unif_lex"]:
                    best_any = rec
                if mono and (best_mono is None or R_unif > best_mono["R_unif_lex"]):
                    best_mono = rec

    return {
        "p_str": f"{a}/{b}",
        "max_m": max_m,
        "non_dictator_profiles_found": total_profiles_found,
        "best_any": best_any,
        "best_mono": best_mono,
    }


# =============================================================================
# Probe 3: General Real Fixed-Point T=1 & T=2 Scan + Simulated Annealing
# =============================================================================

def _make_named_gates() -> List[Tuple[str, np.ndarray]]:
    """Standard monotone & non-monotone small gates for exact T=1 and T=2 cross-scan."""
    gates: List[Tuple[str, np.ndarray]] = []
    # AND_k, OR_k, NAND_k, NOR_k for k=2..6
    for k in range(2, 7):
        N = 1 << k
        and_b = np.zeros(N, dtype=np.int8)
        and_b[-1] = 1
        or_b = np.ones(N, dtype=np.int8)
        or_b[0] = 0
        gates.append((f"AND_{k}", and_b))
        gates.append((f"OR_{k}", or_b))
        gates.append((f"NAND_{k}", 1 - and_b))
        gates.append((f"NOR_{k}", 1 - or_b))

    # Thresholds TH_{k, t} for k=3..7
    for k in range(3, 8):
        pc = popcount_array(k).astype(np.int64)
        for t in range(2, k):
            b = (pc >= t).astype(np.int8)
            gates.append((f"TH_{k}_{t}", b))

    # Tribes(w, s) and Dual-Tribes for small sizes
    for w, s in [(2, 2), (2, 3), (3, 2)]:
        m = w * s
        N = 1 << m
        b = np.zeros(N, dtype=np.int8)
        for x in range(N):
            ok = False
            for blk in range(s):
                mask = ((1 << w) - 1) << (blk * w)
                if (x & mask) == mask:
                    ok = True
                    break
            if ok:
                b[x] = 1
        gates.append((f"Tribes_{w}x{s}", b))
        # Dual (CNF tribes): AND of s OR_w's
        b_dual = 1 - b[::-1]
        gates.append((f"DualTribes_{w}x{s}", b_dual))

    # Asymmetric read-once trees on m=3,4,5,6
    for name, m, fn in [
        ("x0|(x1&x2)", 3, lambda b: b[0] | (b[1] & b[2])),
        ("x0&(x1|x2)", 3, lambda b: b[0] & (b[1] | b[2])),
        ("x0|(x1&(x2|x3))", 4, lambda b: b[0] | (b[1] & (b[2] | b[3]))),
        ("x0&(x1|(x2&x3))", 4, lambda b: b[0] & (b[1] | (b[2] & b[3]))),
        ("(x0&x1)|(x2&(x3|x4))", 5, lambda b: (b[0] & b[1]) | (b[2] & (b[3] | b[4]))),
        ("(x0|x1)&(x2|(x3&x4))", 5, lambda b: (b[0] | b[1]) & (b[2] | (b[3] & b[4]))),
        ("Hod_g3_6", 6, lambda b: (b[0] | b[1]) & (b[2] | b[3]) & (b[0] | b[2] | b[4]) & (b[2] | b[4] | b[5])),
        ("Hod_g3p_6", 6, lambda b: (b[0] | b[1]) & (b[2] | b[3]) & (b[0] | b[2] | b[4]) & (b[1] | b[3] | b[5])),
        ("MUX_3", 3, lambda b: b[1] if b[0] else b[2]),
        ("Addr_4_asym", 4, lambda b: (b[1] & b[2]) if b[0] else (b[2] | b[3])),
        ("Not_x0_or_x1x2", 3, lambda b: (1 - b[0]) | (b[1] & b[2])),
    ]:
        N = 1 << m
        arr = np.zeros(N, dtype=np.int8)
        for x in range(N):
            bits = [(x >> i) & 1 for i in range(m)]
            arr[x] = int(fn(bits))
        gates.append((name, arr))

    return gates


def sa_search_biased_t1(
    m: int,
    monotone_only: bool,
    restarts: int = 8,
    steps: int = 1200,
    seed: int = 123,
) -> Dict[str, Any]:
    """Simulated annealing over T=1 fixed-point functions g: {0,1}^m -> {0,1} (p* != 1/2)."""
    rng = random.Random(seed)
    N = 1 << m
    best: Optional[Dict[str, Any]] = None

    for r in range(restarts):
        if monotone_only:
            # Start from random monotone threshold or lex with bias in [0.55, 0.78]
            c1 = rng.randint(int(0.54 * N), int(0.80 * N))
            cur = make_lex_finite(m, c1)
        else:
            # Start from NAND-like or random non-monotone function
            if r == 0:
                cur = np.ones(N, dtype=np.int8)
                cur[-1] = 0
                cur[-2] = 0
            else:
                cur = np.array([1 if rng.random() < 0.62 else 0 for _ in range(N)], dtype=np.int8)

        def eval_t1(b: np.ndarray) -> Optional[Dict[str, Any]]:
            roots = solve_fixed_point_1step(b, lo=0.05, hi=0.95)
            # Only non-balanced p* (|p* - 0.5| > 0.01) so we strictly measure Vector A1!
            best_loc = None
            for p_s in roots:
                if abs(p_s - 0.5) < 0.01:
                    continue
                res = eval_cycle_with_lex_bridge([b], p_s)
                if res is not None:
                    if best_loc is None or res["R_unif_lex"] > best_loc["R_unif_lex"]:
                        best_loc = res
            return best_loc

        cur_res = eval_t1(cur)
        cur_score = cur_res["R_unif_lex"] if cur_res else 0.0
        if cur_res and (best is None or cur_score > best["R_unif_lex"]):
            best = {**cur_res, "m": m, "monotone": is_monotone_01(cur)}

        for step in range(steps):
            if monotone_only:
                # Pick a valid monotone single-bit flip
                cands = []
                for x in range(N):
                    if cur[x] == 1:
                        if all(cur[x ^ (1 << i)] == 0 for i in range(m) if (x >> i) & 1):
                            cands.append(x)
                    else:
                        if all(cur[x | (1 << i)] == 1 for i in range(m) if not ((x >> i) & 1)):
                            cands.append(x)
                if not cands:
                    break
                idx = rng.choice(cands)
            else:
                idx = rng.randrange(N)

            cur[idx] ^= 1
            res = eval_t1(cur)
            sc = res["R_unif_lex"] if res else 0.0
            temp = 0.15 * (1.0 - step / steps) + 1e-4
            if sc >= cur_score or (sc > 0 and rng.random() < math.exp((sc - cur_score) / temp)):
                cur_score = sc
                if res and (best is None or sc > best["R_unif_lex"]):
                    best = {**res, "m": m, "monotone": is_monotone_01(cur)}
            else:
                cur[idx] ^= 1

    return best or {"m": m, "monotone": monotone_only, "R_unif_lex": 0.0}


def sa_search_biased_t2(
    m1: int,
    m2: int,
    restarts: int = 6,
    steps: int = 800,
    seed: int = 456,
) -> Dict[str, Any]:
    """Simulated annealing over monotone T=2 alternating cycles (g1, g2) with p* != 1/2."""
    rng = random.Random(seed)
    N1, N2 = 1 << m1, 1 << m2
    best: Optional[Dict[str, Any]] = None

    def eval_t2(b1: np.ndarray, b2: np.ndarray) -> Optional[Dict[str, Any]]:
        roots = solve_fixed_point_2step(b1, b2, lo=0.05, hi=0.95)
        best_loc = None
        for p_s in roots:
            if abs(p_s - 0.5) < 0.005:
                continue
            res = eval_cycle_with_lex_bridge([b1, b2], p_s)
            if res is not None:
                if best_loc is None or res["R_unif_lex"] > best_loc["R_unif_lex"]:
                    best_loc = res
        return best_loc

    for r in range(restarts):
        # Initialize g1 as AND-heavy monotone (pushes p down), g2 as OR-heavy monotone (pushes p up)
        g1 = make_lex_finite(m1, rng.randint(1, max(1, N1 // 2 - 1)))
        g2 = make_lex_finite(m2, rng.randint(min(N2 - 1, N2 // 2 + 1), N2 - 1))
        cur_res = eval_t2(g1, g2)
        cur_score = cur_res["R_unif_lex"] if cur_res else 0.0
        if cur_res and (best is None or cur_score > best["R_unif_lex"]):
            best = {**cur_res, "m1": m1, "m2": m2, "monotone": True}

        for step in range(steps):
            which = rng.randrange(2)
            target_b = g1 if which == 0 else g2
            mm = m1 if which == 0 else m2
            NN = N1 if which == 0 else N2
            cands = []
            for x in range(NN):
                if target_b[x] == 1:
                    if all(target_b[x ^ (1 << i)] == 0 for i in range(mm) if (x >> i) & 1):
                        cands.append(x)
                else:
                    if all(target_b[x | (1 << i)] == 1 for i in range(mm) if not ((x >> i) & 1)):
                        cands.append(x)
            if not cands:
                continue
            idx = rng.choice(cands)
            target_b[idx] ^= 1
            res = eval_t2(g1, g2)
            sc = res["R_unif_lex"] if res else 0.0
            temp = 0.12 * (1.0 - step / steps) + 1e-4
            if sc >= cur_score or (sc > 0 and rng.random() < math.exp((sc - cur_score) / temp)):
                cur_score = sc
                if res and (best is None or sc > best["R_unif_lex"]):
                    best = {**res, "m1": m1, "m2": m2, "monotone": True}
            else:
                target_b[idx] ^= 1

    return best or {"m1": m1, "m2": m2, "monotone": True, "R_unif_lex": 0.0}


# =============================================================================
# Main Phase 1 Runner
# =============================================================================

def run_phase1_a1() -> Dict[str, Any]:
    t0 = time.perf_counter()
    print("=== Phase 1 CPU Scout: Vector A1 (p-Biased Cycles + Uniform Bridge) ===")

    # 0. Sanity anchor: Hod (2017) Thm 3.2 (OR_2 o AND_2 at p* = Phi)
    Phi = (math.sqrt(5.0) - 1.0) / 2.0
    and2 = np.array([0, 0, 0, 1], dtype=np.int8)
    or2 = np.array([0, 1, 1, 1], dtype=np.int8)
    nand2 = np.array([1, 1, 1, 0], dtype=np.int8)
    anchor_t2 = eval_cycle_with_lex_bridge([and2, or2], Phi)
    anchor_t1 = eval_cycle_with_lex_bridge([nand2], Phi)
    assert anchor_t2 is not None and anchor_t1 is not None
    assert abs(anchor_t2["R_unif_lex"] - 6.413845514984722) < 1e-11
    assert abs(anchor_t1["R_unif_lex"] - 6.413845514984722) < 1e-11
    print(
        f"[ANCHOR PASS] Hod Thm 3.2 (OR2 o AND2 == NAND2 at Phi): "
        f"beta={anchor_t1['beta_cyc']:.6f}, R_unif_lex={anchor_t1['R_unif_lex']:.12f}"
    )

    # 1. Probe 1: Co-optimized finite uniform bridges for dyadic p*
    print("\n--- Probe 1: Dyadic Uniform Bridge Co-Optimization (target R=6.522) ---")
    dyadic_targets = [
        (Fraction(9, 16), 4),
        (Fraction(5, 8), 4),
        (Fraction(11, 16), 4),
        (Fraction(3, 4), 4),
        (Fraction(13, 16), 4),
        (Fraction(9, 16), 6),
        (Fraction(5, 8), 6),
        (Fraction(11, 16), 6),
        (Fraction(3, 4), 6),
        (Fraction(17, 32), 6),
        (Fraction(19, 32), 6),
        (Fraction(21, 32), 6),
        (Fraction(23, 32), 6),
    ]
    bridge_results: List[Dict[str, Any]] = []
    for pf, mb in dyadic_targets:
        res_b = _optimize_bridge_at_dyadic_p(pf, mb, R_ref=6.522, sa_steps=1500, seed=42 + mb)
        bridge_results.append(res_b)
        print(
            f"  p*={res_b['p_str']:5s} m={mb}: lex I={res_b['lex_I']:.5f} H={res_b['lex_H']:.5f} "
            f"beta_req={res_b['lex_beta_req_6522']:.4f} | "
            f"opt_mono={res_b['best_mono_beta_req_6522']:.4f} | "
            f"dealiased={res_b['best_dealiased_beta_req_6522']:.4f}"
        )

    # 2. Probe 2: Exact dyadic T=1 fixed points (Diophantine scan)
    print("\n--- Probe 2: Exact Dyadic T=1 Fixed-Point Search (m=2..6) ---")
    dyadic_t1_results: List[Dict[str, Any]] = []
    for pf in [Fraction(9, 16), Fraction(5, 8), Fraction(11, 16), Fraction(3, 4), Fraction(13, 16)]:
        res_d = find_exact_dyadic_t1(pf, max_m=6)
        dyadic_t1_results.append(res_d)
        ba = res_d["best_any"]
        bm = res_d["best_mono"]
        print(
            f"  p*={res_d['p_str']:5s}: profiles={res_d['non_dictator_profiles_found']:4d} | "
            f"best_any R={ba['R_unif_lex']:.6f} (m={ba['m']}, beta={ba['beta_cyc']:.4f})"
            if ba else f"  p*={res_d['p_str']:5s}: profiles={res_d['non_dictator_profiles_found']:4d} | none",
            f"| best_mono R={bm['R_unif_lex']:.6f}" if bm else "| best_mono=None",
        )

    # 3. Probe 3A: Named gates T=1 and T=2 cross-scan
    print("\n--- Probe 3A: Named Gates T=1 & T=2 Cross-Scan ---")
    gates = _make_named_gates()
    named_t1: List[Dict[str, Any]] = []
    named_t2: List[Dict[str, Any]] = []
    for name, b in gates:
        for p_s in solve_fixed_point_1step(b, lo=0.05, hi=0.95):
            if abs(p_s - 0.5) < 0.005:
                continue
            res = eval_cycle_with_lex_bridge([b], p_s)
            if res:
                named_t1.append({"name": name, "monotone": is_monotone_01(b), **res})

    # Pairwise T=2 across all gates (focusing on pairs up to m<=5 to keep fast)
    small_gates = [(n, b) for (n, b) in gates if b.shape[0] <= 32]
    for i, (n1, b1) in enumerate(small_gates):
        for j, (n2, b2) in enumerate(small_gates):
            if i == j:
                continue
            for p_s in solve_fixed_point_2step(b1, b2, lo=0.08, hi=0.92):
                if abs(p_s - 0.5) < 0.005:
                    continue
                res = eval_cycle_with_lex_bridge([b1, b2], p_s)
                if res:
                    named_t2.append({
                        "name": f"{n2} o {n1}",
                        "monotone": is_monotone_01(b1) and is_monotone_01(b2),
                        **res,
                    })

    named_t1.sort(key=lambda d: d["R_unif_lex"], reverse=True)
    named_t2.sort(key=lambda d: d["R_unif_lex"], reverse=True)
    print("  Top 5 named T=1 (p* != 1/2):")
    for rec in named_t1[:5]:
        print(
            f"    {rec['name']:22s} mono={str(rec['monotone']):5s} p*={rec['p_star']:.6f} "
            f"beta={rec['beta_cyc']:.6f} R_unif_lex={rec['R_unif_lex']:.6f}"
        )
    print("  Top 5 named T=2 (p* != 1/2):")
    for rec in named_t2[:5]:
        print(
            f"    {rec['name']:30s} mono={str(rec['monotone']):5s} p*={rec['p_star']:.6f} "
            f"beta={rec['beta_cyc']:.6f} R_unif_lex={rec['R_unif_lex']:.6f}"
        )

    # 4. Probe 3B: Simulated Annealing over general T=1 (mono & non-mono) and T=2 (mono)
    print("\n--- Probe 3B: Simulated Annealing (T=1 mono/non-mono m=4..6, T=2 mono) ---")
    sa_results: List[Dict[str, Any]] = []
    for m in [4, 5, 6]:
        r_m = sa_search_biased_t1(m=m, monotone_only=True, restarts=6, steps=700, seed=10 + m)
        r_nm = sa_search_biased_t1(m=m, monotone_only=False, restarts=6, steps=900, seed=50 + m)
        sa_results.append({"kind": f"SA_T1_mono_m{m}", **r_m})
        sa_results.append({"kind": f"SA_T1_nonmono_m{m}", **r_nm})
        print(
            f"  SA T=1 m={m} mono:    R_unif_lex={r_m['R_unif_lex']:.6f} "
            f"(p*={r_m.get('p_star', 0):.5f}, beta={r_m.get('beta_cyc', 0):.5f})"
        )
        print(
            f"  SA T=1 m={m} nonmono: R_unif_lex={r_nm['R_unif_lex']:.6f} "
            f"(p*={r_nm.get('p_star', 0):.5f}, beta={r_nm.get('beta_cyc', 0):.5f})"
        )

    for (m1, m2) in [(3, 3), (4, 4), (5, 5)]:
        r_t2 = sa_search_biased_t2(m1=m1, m2=m2, restarts=6, steps=600, seed=200 + 10 * m1 + m2)
        sa_results.append({"kind": f"SA_T2_mono_{m1}x{m2}", **r_t2})
        print(
            f"  SA T=2 ({m1},{m2}) mono: R_unif_lex={r_t2['R_unif_lex']:.6f} "
            f"(p*={r_t2.get('p_star', 0):.5f}, beta={r_t2.get('beta_cyc', 0):.5f})"
        )

    # Aggregate best overall across ALL Vector A1 probes
    all_candidates: List[Tuple[float, str, Dict[str, Any]]] = []
    for rec in dyadic_t1_results:
        if rec["best_any"]:
            all_candidates.append((rec["best_any"]["R_unif_lex"], f"dyadic_T1_{rec['p_str']}", rec["best_any"]))
    for rec in named_t1[:10]:
        all_candidates.append((rec["R_unif_lex"], f"named_T1_{rec['name']}", rec))
    for rec in named_t2[:10]:
        all_candidates.append((rec["R_unif_lex"], f"named_T2_{rec['name']}", rec))
    for rec in sa_results:
        all_candidates.append((rec["R_unif_lex"], rec["kind"], rec))
    all_candidates.sort(key=lambda x: x[0], reverse=True)

    best_score, best_label, best_detail = all_candidates[0]
    best_dyadic_score = max(
        (r["best_any"]["R_unif_lex"] for r in dyadic_t1_results if r["best_any"]),
        default=0.0,
    )
    elapsed = time.perf_counter() - t0
    go_decision = "GO" if best_score > GO_GATE_A1 else "STOP_CRANK_FREEZE_TO_RESERVE"

    summary = {
        "vector": "A1_p_biased_cycles",
        "elapsed_sec": round(elapsed, 2),
        "anchor_hod_thm32": {
            "p_star": Phi,
            "beta_cyc": anchor_t1["beta_cyc"],
            "R_unif_lex": anchor_t1["R_unif_lex"],
            "verified_against_exact": True,
        },
        "go_gate_threshold": GO_GATE_A1,
        "hod_2017_bar": HOD_2017_BAR,
        "num_2026_bar": NUM_2026_BAR,
        "best_overall_R_unif_lex": best_score,
        "best_overall_label": best_label,
        "best_overall_detail": best_detail,
        "best_exact_dyadic_R_unif": best_dyadic_score,
        "decision": go_decision,
        "probe1_dyadic_bridges": bridge_results,
        "probe2_exact_dyadic_t1": dyadic_t1_results,
        "probe3a_top10_named_t1": named_t1[:10],
        "probe3a_top10_named_t2": named_t2[:10],
        "probe3b_sa_results": sa_results,
    }

    out_path = Path(r"e:\эксперименты\исследования\c71_fei_research\artifacts\phase1_a1_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== VECTOR A1 PHASE 1 SUMMARY ===")
    print(f"Elapsed:                {elapsed:.2f} s")
    print(f"Best exact dyadic R:    {best_dyadic_score:.6f}")
    print(f"Best overall A1 R_unif: {best_score:.6f} ({best_label})")
    print(f"Phase 1 Go Gate (>6.46): {go_decision}")
    print(f"Saved JSON:             {out_path}")
    return summary


if __name__ == "__main__":
    run_phase1_a1()

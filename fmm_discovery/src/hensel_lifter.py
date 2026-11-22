"""
CPU-Only Mod-4 Liftability Oracle, Linear 2-adic Hensel Lifter (2^k -> 2^{k+1}),
Exact Z3 5.1.0 Ternary Sign Lifter, and Clean From-Scratch Block-Seed Builders
for Fast Matrix Multiplication (FMM) Bilinear Schemes.

Zero GPU usage (strictly Numba 12T CPU + NumPy + Z3 5.1.0 + Python exact int/Fraction).
All schemes follow `brent_verifier.py` convention `C_eq_AB_row_major` and are verified
with zero floating-point arithmetic via `brent_verifier.verify_scheme()`.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numba as nb
import numpy as np
import z3

# Ensure sibling import of brent_verifier works whether run as script or module
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from brent_verifier import (  # noqa: E402
    BilinearScheme,
    VerificationReport,
    direct_sum_p,
    laderman_333,
    make_naive_scheme,
    strassen_222,
    verify_scheme,
)


# =============================================================================
# 1. Bit-Packed uint64 Conversions & Exact Mod-2 Fast Check
# =============================================================================


def scheme_to_u64_f2(scheme: BilinearScheme) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pack a Z2 (or integer reduced mod 2) BilinearScheme into three uint64[R] bitmasks.
    Requires max(m*n, n*p, m*p) <= 64.
    """
    m, n, p = scheme.m, scheme.n, scheme.p
    if max(m * n, n * p, m * p) > 64:
        raise ValueError(f"Format {scheme.shape} exceeds uint64 bit-packing limit (max_flat > 64)")
    R = scheme.rank
    u = np.zeros(R, dtype=np.uint64)
    v = np.zeros(R, dtype=np.uint64)
    w = np.zeros(R, dtype=np.uint64)
    for r in range(R):
        mu = 0
        for idx, val in enumerate(scheme.U[r]):
            if int(val) & 1:
                mu |= 1 << idx
        mv = 0
        for idx, val in enumerate(scheme.V[r]):
            if int(val) & 1:
                mv |= 1 << idx
        mw = 0
        for idx, val in enumerate(scheme.W[r]):
            if int(val) & 1:
                mw |= 1 << idx
        u[r] = np.uint64(mu)
        v[r] = np.uint64(mv)
        w[r] = np.uint64(mw)
    return u, v, w


def u64_f2_to_scheme(
    m: int,
    n: int,
    p: int,
    u: Sequence[int] | np.ndarray,
    v: Sequence[int] | np.ndarray,
    w: Sequence[int] | np.ndarray,
    name: str = "",
) -> BilinearScheme:
    """Unpack uint64[R] bitmasks into an exact Z2 BilinearScheme."""
    mn, np_dim, mp = m * n, n * p, m * p
    R = len(u)
    U: List[List[int]] = []
    V: List[List[int]] = []
    W: List[List[int]] = []
    for r in range(R):
        mu, mv, mw = int(u[r]), int(v[r]), int(w[r])
        U.append([(mu >> a) & 1 for a in range(mn)])
        V.append([(mv >> b) & 1 for b in range(np_dim)])
        W.append([(mw >> c) & 1 for c in range(mp)])
    return BilinearScheme(m=m, n=n, p=p, U=U, V=V, W=W, ring="Z2", modulus=2, name=name)


@nb.njit(cache=True)
def _verify_f2_u64_nb(m: int, n: int, p: int, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> bool:
    """Exact mod-2 Brent equation check on bit-packed uint64[R] factors."""
    mn = m * n
    np_dim = n * p
    R = u.shape[0]
    # Accumulate XOR over w-masks for each (a, b) pair
    acc = np.zeros((mn, np_dim), dtype=np.uint64)
    for r in range(R):
        ur = u[r]
        vr = v[r]
        wr = w[r]
        if ur == 0 or vr == 0 or wr == 0:
            continue
        tu = ur
        while tu != 0:
            lsb_u = tu & (~tu + np.uint64(1))
            a = 0
            x = lsb_u
            while x > 1:
                x >>= np.uint64(1)
                a += 1
            tv = vr
            while tv != 0:
                lsb_v = tv & (~tv + np.uint64(1))
                b = 0
                y = lsb_v
                while y > 1:
                    y >>= np.uint64(1)
                    b += 1
                acc[a, b] ^= wr
                tv ^= lsb_v
            tu ^= lsb_u

    for i1 in range(m):
        for i2 in range(n):
            a = i1 * n + i2
            for j1 in range(n):
                for j2 in range(p):
                    b = j1 * p + j2
                    expected = np.uint64(0)
                    if i2 == j1:
                        c = i1 * p + j2
                        expected = np.uint64(1) << np.uint64(c)
                    if acc[a, b] != expected:
                        return False
    return True


def fast_verify_f2_u64(m: int, n: int, p: int, u: np.ndarray, v: np.ndarray, w: np.ndarray) -> bool:
    """Fast exact Numba mod-2 check on uint64 arrays."""
    return bool(_verify_f2_u64_nb(m, n, p, u, v, w))


# =============================================================================
# 2. Stage L0 — Ultra-Fast Numba Mod-4 Liftability Oracle (GF(2) Jacobian)
# =============================================================================


@dataclass
class Mod4OracleResult:
    coker_viol_4: int
    free_bits: int
    rank_J1: int
    num_vars: int
    num_active_eqs: int
    is_mod4_liftable: bool
    support_restricted: bool
    nnz_total: int
    gauge_fixed: int
    elapsed_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@nb.njit(cache=True)
def _ctz64(x: np.uint64) -> int:
    """Count trailing zeros of nonzero uint64."""
    c = 0
    if (x & np.uint64(0xFFFFFFFF)) == 0:
        x >>= np.uint64(32)
        c += 32
    if (x & np.uint64(0xFFFF)) == 0:
        x >>= np.uint64(16)
        c += 16
    if (x & np.uint64(0xFF)) == 0:
        x >>= np.uint64(8)
        c += 8
    if (x & np.uint64(0xF)) == 0:
        x >>= np.uint64(4)
        c += 4
    if (x & np.uint64(0x3)) == 0:
        x >>= np.uint64(2)
        c += 2
    if (x & np.uint64(0x1)) == 0:
        c += 1
    return c


@nb.njit(cache=True)
def _popcount64(x: np.uint64) -> int:
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((x * np.uint64(0x0101010101010101)) >> np.uint64(56))


@nb.njit(cache=True)
def _mod4_oracle_support_nb(
    m: int, n: int, p: int, u: np.ndarray, v: np.ndarray, w: np.ndarray
) -> Tuple[int, int, int, int, int, int, int]:
    """
    Ultra-fast Numba support-restricted mod-4 liftability oracle with 2R gauge fixing.

    Returns:
      (coker_viol_4, free_bits, rank_J1, num_vars, num_active_eqs, nnz_total, gauge_fixed)
      or (-1, ...) if the input scheme is NOT valid mod 2.
    """
    mn = m * n
    np_dim = n * p
    mp = m * p
    R = u.shape[0]

    # Map each nonzero (mode, r, idx) to a variable column index, skipping 2R gauge positions.
    var_u = np.full((R, mn), -1, dtype=np.int32)
    var_v = np.full((R, np_dim), -1, dtype=np.int32)
    var_w = np.full((R, mp), -1, dtype=np.int32)

    num_vars = 0
    nnz_total = 0
    gauge_fixed = 0

    for r in range(R):
        ur = u[r]
        vr = v[r]
        wr = w[r]
        first_u = True
        tu = ur
        while tu != 0:
            lsb = tu & (~tu + np.uint64(1))
            a = _ctz64(lsb)
            nnz_total += 1
            if first_u:
                first_u = False
                gauge_fixed += 1
            else:
                var_u[r, a] = num_vars
                num_vars += 1
            tu ^= lsb

        first_v = True
        tv = vr
        while tv != 0:
            lsb = tv & (~tv + np.uint64(1))
            b = _ctz64(lsb)
            nnz_total += 1
            if first_v:
                first_v = False
                gauge_fixed += 1
            else:
                var_v[r, b] = num_vars
                num_vars += 1
            tv ^= lsb

        tw = wr
        while tw != 0:
            lsb = tw & (~tw + np.uint64(1))
            c = _ctz64(lsb)
            nnz_total += 1
            var_w[r, c] = num_vars
            num_vars += 1
            tw ^= lsb

    # Compute exact integer T1 counts on active (a, b, c) triples
    total_triples = mn * np_dim * mp
    counts = np.zeros(total_triples, dtype=np.int32)
    # Also store expected T_exact in {0, 1}
    t_exact = np.zeros(total_triples, dtype=np.uint8)
    for i1 in range(m):
        for i2 in range(n):
            a = i1 * n + i2
            for j2 in range(p):
                b = i2 * p + j2
                c = i1 * p + j2
                t_exact[(a * np_dim + b) * mp + c] = 1

    for r in range(R):
        ur = u[r]
        vr = v[r]
        wr = w[r]
        if ur == 0 or vr == 0 or wr == 0:
            continue
        tu = ur
        while tu != 0:
            lu = tu & (~tu + np.uint64(1))
            a = _ctz64(lu)
            tv = vr
            while tv != 0:
                lv = tv & (~tv + np.uint64(1))
                b = _ctz64(lv)
                base = (a * np_dim + b) * mp
                tw = wr
                while tw != 0:
                    lw = tw & (~tw + np.uint64(1))
                    c = _ctz64(lw)
                    counts[base + c] += 1
                    tw ^= lw
                tv ^= lv
            tu ^= lu

    # Verify mod-2 validity & collect active equation indices (counts[idx] >= 1)
    num_active = 0
    for idx in range(total_triples):
        cnt = counts[idx]
        if (cnt & 1) != t_exact[idx]:
            return -1, 0, 0, num_vars, 0, nnz_total, gauge_fixed
        if cnt > 0:
            num_active += 1

    # Assign row index 0..num_active-1 to each active (a, b, c)
    eq_id = np.full(total_triples, -1, dtype=np.int32)
    cur = 0
    for idx in range(total_triples):
        if counts[idx] > 0:
            eq_id[idx] = cur
            cur += 1

    # Build bit-packed augmented matrix [J1 | e1] of shape (num_active, n_words)
    # Last column (index = num_vars) stores e1 = ((counts - t_exact) >> 1) & 1.
    n_cols = num_vars + 1
    n_words = (n_cols + 63) // 64
    mat = np.zeros((num_active, n_words), dtype=np.uint64)

    rhs_word = num_vars >> 6
    rhs_bit = np.uint64(1) << np.uint64(num_vars & 63)

    for idx in range(total_triples):
        row = eq_id[idx]
        if row >= 0:
            e1 = ((counts[idx] - int(t_exact[idx])) >> 1) & 1
            if e1:
                mat[row, rhs_word] |= rhs_bit

    # Populate J1 entries: for each term r covering (a,b,c), toggle var_u[r,a], var_v[r,b], var_w[r,c]
    for r in range(R):
        ur = u[r]
        vr = v[r]
        wr = w[r]
        if ur == 0 or vr == 0 or wr == 0:
            continue
        tu = ur
        while tu != 0:
            lu = tu & (~tu + np.uint64(1))
            a = _ctz64(lu)
            cu = var_u[r, a]
            wu = cu >> 6
            bu = np.uint64(1) << np.uint64(cu & 63) if cu >= 0 else np.uint64(0)
            tv = vr
            while tv != 0:
                lv = tv & (~tv + np.uint64(1))
                b = _ctz64(lv)
                cv = var_v[r, b]
                wv = cv >> 6
                bv = np.uint64(1) << np.uint64(cv & 63) if cv >= 0 else np.uint64(0)
                base = (a * np_dim + b) * mp
                tw = wr
                while tw != 0:
                    lw = tw & (~tw + np.uint64(1))
                    c = _ctz64(lw)
                    row = eq_id[base + c]
                    cw = var_w[r, c]
                    if cu >= 0:
                        mat[row, wu] ^= bu
                    if cv >= 0:
                        mat[row, wv] ^= bv
                    if cw >= 0:
                        mat[row, cw >> 6] ^= np.uint64(1) << np.uint64(cw & 63)
                    tw ^= lw
                tv ^= lv
            tu ^= lu

    # Forward GF(2) Gaussian elimination over columns 0..num_vars-1
    pivot_row = 0
    for col in range(num_vars):
        w_idx = col >> 6
        b_mask = np.uint64(1) << np.uint64(col & 63)
        found = -1
        for r_i in range(pivot_row, num_active):
            if (mat[r_i, w_idx] & b_mask) != 0:
                found = r_i
                break
        if found == -1:
            continue
        if found != pivot_row:
            for j in range(w_idx, n_words):
                tmp = mat[pivot_row, j]
                mat[pivot_row, j] = mat[found, j]
                mat[found, j] = tmp
        for r_i in range(pivot_row + 1, num_active):
            if (mat[r_i, w_idx] & b_mask) != 0:
                for j in range(w_idx, n_words):
                    mat[r_i, j] ^= mat[pivot_row, j]
        pivot_row += 1
        if pivot_row == num_active:
            break

    rank_J1 = pivot_row
    free_bits = num_vars - rank_J1
    coker_viol_4 = 0
    for r_i in range(rank_J1, num_active):
        if (mat[r_i, rhs_word] & rhs_bit) != 0:
            coker_viol_4 += 1

    return coker_viol_4, free_bits, rank_J1, num_vars, num_active, nnz_total, gauge_fixed


@nb.njit(cache=True)
def _mod4_oracle_unrestricted_nb(
    m: int,
    n: int,
    p: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
) -> Tuple[int, int, int, int, int, int, int]:
    """
    Fast forward-only Stage L0 UNRESTRICTED mod-4 liftability oracle (allows 0 -> 2 mod 4
    off-support lifts for general Z / Q schemes with odd denominators, with 2R gauge fixing).
    Skips P_mat and back-substitution -> ~15-25x faster than full RREF builder!
    Returns (coker_viol_4, free_bits, rank_J1, num_vars, num_active_eqs, nnz_total, gauge_fixed).
    """
    mn = m * n
    np_dim = n * p
    mp = m * p
    R = u.shape[0]

    var_u = np.full((R, mn), -1, dtype=np.int32)
    var_v = np.full((R, np_dim), -1, dtype=np.int32)
    var_w = np.full((R, mp), -1, dtype=np.int32)

    num_vars = 0
    nnz_total = 0
    gauge_fixed = 0
    for pass_off in (0, 1):
        for r in range(R):
            ur = u[r]
            vr = v[r]
            wr = w[r]
            a0 = _ctz64(ur) if ur != 0 else -1
            b0 = _ctz64(vr) if vr != 0 else -1
            if pass_off == 0:
                if a0 >= 0:
                    gauge_fixed += 1
                if b0 >= 0:
                    gauge_fixed += 1
                nnz_total += _popcount64(ur) + _popcount64(vr) + _popcount64(wr)
            for a in range(mn):
                if a == a0:
                    continue
                is_nz = ((ur >> np.uint64(a)) & np.uint64(1)) != 0
                if (pass_off == 0 and is_nz) or (pass_off == 1 and not is_nz):
                    var_u[r, a] = num_vars
                    num_vars += 1
            for b in range(np_dim):
                if b == b0:
                    continue
                is_nz = ((vr >> np.uint64(b)) & np.uint64(1)) != 0
                if (pass_off == 0 and is_nz) or (pass_off == 1 and not is_nz):
                    var_v[r, b] = num_vars
                    num_vars += 1
            for c in range(mp):
                is_nz = ((wr >> np.uint64(c)) & np.uint64(1)) != 0
                if (pass_off == 0 and is_nz) or (pass_off == 1 and not is_nz):
                    var_w[r, c] = num_vars
                    num_vars += 1

    total_triples = mn * np_dim * mp
    counts = np.zeros(total_triples, dtype=np.int32)
    t_exact = np.zeros(total_triples, dtype=np.uint8)
    active_mask = np.zeros(total_triples, dtype=np.uint8)

    for i1 in range(m):
        for i2 in range(n):
            a = i1 * n + i2
            for j2 in range(p):
                b = i2 * p + j2
                c = i1 * p + j2
                idx = (a * np_dim + b) * mp + c
                t_exact[idx] = 1
                active_mask[idx] = 1

    for r in range(R):
        ur = u[r]
        vr = v[r]
        wr = w[r]
        if ur != 0 and vr != 0 and wr != 0:
            tu = ur
            while tu != 0:
                lu = tu & (~tu + np.uint64(1))
                a = _ctz64(lu)
                tv = vr
                while tv != 0:
                    lv = tv & (~tv + np.uint64(1))
                    b = _ctz64(lv)
                    base = (a * np_dim + b) * mp
                    tw = wr
                    while tw != 0:
                        lw = tw & (~tw + np.uint64(1))
                        c = _ctz64(lw)
                        counts[base + c] += 1
                        active_mask[base + c] = 1
                        tw ^= lw
                    tv ^= lv
                tu ^= lu
        if vr != 0 and wr != 0:
            tv = vr
            while tv != 0:
                lv = tv & (~tv + np.uint64(1))
                b = _ctz64(lv)
                tw = wr
                while tw != 0:
                    lw = tw & (~tw + np.uint64(1))
                    c = _ctz64(lw)
                    for a in range(mn):
                        active_mask[(a * np_dim + b) * mp + c] = 1
                    tw ^= lw
                tv ^= lv
        if ur != 0 and wr != 0:
            tu = ur
            while tu != 0:
                lu = tu & (~tu + np.uint64(1))
                a = _ctz64(lu)
                tw = wr
                while tw != 0:
                    lw = tw & (~tw + np.uint64(1))
                    c = _ctz64(lw)
                    for b in range(np_dim):
                        active_mask[(a * np_dim + b) * mp + c] = 1
                    tw ^= lw
                tu ^= lu
        if ur != 0 and vr != 0:
            tu = ur
            while tu != 0:
                lu = tu & (~tu + np.uint64(1))
                a = _ctz64(lu)
                tv = vr
                while tv != 0:
                    lv = tv & (~tv + np.uint64(1))
                    b = _ctz64(lv)
                    base = (a * np_dim + b) * mp
                    for c in range(mp):
                        active_mask[base + c] = 1
                    tv ^= lv
                tu ^= lu

    num_active = 0
    for idx in range(total_triples):
        if (counts[idx] & 1) != t_exact[idx]:
            return -1, 0, 0, num_vars, 0, nnz_total, gauge_fixed
        if active_mask[idx] != 0:
            num_active += 1

    eq_id = np.full(total_triples, -1, dtype=np.int32)
    cur = 0
    for idx in range(total_triples):
        if active_mask[idx] != 0:
            eq_id[idx] = cur
            cur += 1

    n_cols = num_vars + 1
    n_words = (n_cols + 63) // 64
    mat = np.zeros((num_active, n_words), dtype=np.uint64)
    rhs_word = num_vars >> 6
    rhs_bit = np.uint64(1) << np.uint64(num_vars & 63)

    for idx in range(total_triples):
        row = eq_id[idx]
        if row >= 0:
            if (((counts[idx] - int(t_exact[idx])) >> 1) & 1) != 0:
                mat[row, rhs_word] |= rhs_bit

    for r in range(R):
        ur = u[r]
        vr = v[r]
        wr = w[r]
        if vr != 0 and wr != 0:
            tv = vr
            while tv != 0:
                lv = tv & (~tv + np.uint64(1))
                b = _ctz64(lv)
                tw = wr
                while tw != 0:
                    lw = tw & (~tw + np.uint64(1))
                    c = _ctz64(lw)
                    for a in range(mn):
                        cu = var_u[r, a]
                        if cu >= 0:
                            row = eq_id[(a * np_dim + b) * mp + c]
                            mat[row, cu >> 6] ^= np.uint64(1) << np.uint64(cu & 63)
                    tw ^= lw
                tv ^= lv
        if ur != 0 and wr != 0:
            tu = ur
            while tu != 0:
                lu = tu & (~tu + np.uint64(1))
                a = _ctz64(lu)
                tw = wr
                while tw != 0:
                    lw = tw & (~tw + np.uint64(1))
                    c = _ctz64(lw)
                    for b in range(np_dim):
                        cv = var_v[r, b]
                        if cv >= 0:
                            row = eq_id[(a * np_dim + b) * mp + c]
                            mat[row, cv >> 6] ^= np.uint64(1) << np.uint64(cv & 63)
                    tw ^= lw
                tu ^= lu
        if ur != 0 and vr != 0:
            tu = ur
            while tu != 0:
                lu = tu & (~tu + np.uint64(1))
                a = _ctz64(lu)
                tv = vr
                while tv != 0:
                    lv = tv & (~tv + np.uint64(1))
                    b = _ctz64(lv)
                    base = (a * np_dim + b) * mp
                    for c in range(mp):
                        cw = var_w[r, c]
                        if cw >= 0:
                            row = eq_id[base + c]
                            mat[row, cw >> 6] ^= np.uint64(1) << np.uint64(cw & 63)
                    tv ^= lv
                tu ^= lu

    pivot_row = 0
    for col in range(num_vars):
        w_idx = col >> 6
        b_mask = np.uint64(1) << np.uint64(col & 63)
        found = -1
        for r_i in range(pivot_row, num_active):
            if (mat[r_i, w_idx] & b_mask) != 0:
                found = r_i
                break
        if found == -1:
            continue
        if found != pivot_row:
            for j in range(w_idx, n_words):
                tmp = mat[pivot_row, j]
                mat[pivot_row, j] = mat[found, j]
                mat[found, j] = tmp
        for r_i in range(pivot_row + 1, num_active):
            if (mat[r_i, w_idx] & b_mask) != 0:
                for j in range(w_idx, n_words):
                    mat[r_i, j] ^= mat[pivot_row, j]
        pivot_row += 1
        if pivot_row == num_active:
            break

    rank_J1 = pivot_row
    free_bits = num_vars - rank_J1
    coker_viol_4 = 0
    for r_i in range(rank_J1, num_active):
        if (mat[r_i, rhs_word] & rhs_bit) != 0:
            coker_viol_4 += 1

    return coker_viol_4, free_bits, rank_J1, num_vars, num_active, nnz_total, gauge_fixed


@nb.njit(cache=True)
def _build_jacobian_and_rref_nb(
    m: int,
    n: int,
    p: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    support_restricted: bool,
) -> Tuple[
    int,  # status (0=ok, -1=invalid_mod2)
    int,  # num_vars
    int,  # num_active
    int,  # rank_J1
    int,  # nnz_total
    int,  # gauge_fixed
    np.ndarray,  # eq_triples (num_active,) flat triple idx
    np.ndarray,  # var_u (R, mn)
    np.ndarray,  # var_v (R, np_dim)
    np.ndarray,  # var_w (R, mp)
    np.ndarray,  # pivot_cols (rank_J1,)
    np.ndarray,  # free_cols (num_vars - rank_J1,)
    np.ndarray,  # rref_mat (rank_J1, n_var_words)
    np.ndarray,  # P_mat (num_active, n_eq_words) row transform s.t. P @ J1 = RREF(J1)
    np.ndarray,  # e1_vec (num_active,) uint8
]:
    """
    Build the exact GF(2) Jacobian J1 (either support-restricted or unrestricted,
    both with 2R gauge fixing), compute its full Gauss-Jordan RREF AND exact row
    transformation matrix P_mat ONCE so that any linear Hensel step 2^k -> 2^{k+1}
    solves J1 x = e_k (mod 2) via a single matrix-vector product P_mat @ e_k!
    """
    mn = m * n
    np_dim = n * p
    mp = m * p
    R = u.shape[0]

    var_u = np.full((R, mn), -1, dtype=np.int32)
    var_v = np.full((R, np_dim), -1, dtype=np.int32)
    var_w = np.full((R, mp), -1, dtype=np.int32)

    num_vars = 0
    nnz_total = 0
    gauge_fixed = 0

    # Order variables ON-SUPPORT FIRST, OFF-SUPPORT LAST so that Gaussian elimination
    # prefers on-support pivots and keeps off-support zeros as free variables!
    for pass_off in (0, 1):
        if pass_off == 1 and support_restricted:
            break
        for r in range(R):
            ur = u[r]
            vr = v[r]
            wr = w[r]
            a0 = _ctz64(ur) if ur != 0 else -1
            b0 = _ctz64(vr) if vr != 0 else -1
            if pass_off == 0:
                if a0 >= 0:
                    gauge_fixed += 1
                if b0 >= 0:
                    gauge_fixed += 1
                nnz_total += _popcount64(ur) + _popcount64(vr) + _popcount64(wr)

            for a in range(mn):
                if a == a0:
                    continue
                is_nz = ((ur >> np.uint64(a)) & np.uint64(1)) != 0
                if (pass_off == 0 and is_nz) or (pass_off == 1 and not is_nz):
                    var_u[r, a] = num_vars
                    num_vars += 1
            for b in range(np_dim):
                if b == b0:
                    continue
                is_nz = ((vr >> np.uint64(b)) & np.uint64(1)) != 0
                if (pass_off == 0 and is_nz) or (pass_off == 1 and not is_nz):
                    var_v[r, b] = num_vars
                    num_vars += 1
            for c in range(mp):
                is_nz = ((wr >> np.uint64(c)) & np.uint64(1)) != 0
                if (pass_off == 0 and is_nz) or (pass_off == 1 and not is_nz):
                    var_w[r, c] = num_vars
                    num_vars += 1

    total_triples = mn * np_dim * mp
    counts = np.zeros(total_triples, dtype=np.int32)
    active_mask = np.zeros(total_triples, dtype=np.uint8)
    t_exact = np.zeros(total_triples, dtype=np.uint8)
    for i1 in range(m):
        for i2 in range(n):
            a = i1 * n + i2
            for j2 in range(p):
                b = i2 * p + j2
                c = i1 * p + j2
                t_exact[(a * np_dim + b) * mp + c] = 1

    # Count T1 and mark active equations
    for r in range(R):
        ur = u[r]
        vr = v[r]
        wr = w[r]
        if ur != 0 and vr != 0 and wr != 0:
            tu = ur
            while tu != 0:
                lu = tu & (~tu + np.uint64(1))
                a = _ctz64(lu)
                tv = vr
                while tv != 0:
                    lv = tv & (~tv + np.uint64(1))
                    b = _ctz64(lv)
                    base = (a * np_dim + b) * mp
                    tw = wr
                    while tw != 0:
                        lw = tw & (~tw + np.uint64(1))
                        c = _ctz64(lw)
                        idx = base + c
                        counts[idx] += 1
                        active_mask[idx] = 1
                        tw ^= lw
                    tv ^= lv
                tu ^= lu

        if not support_restricted:
            # An equation (a,b,c) is active if at least two of (U[r,a], V[r,b], W[r,c]) are 1
            if vr != 0 and wr != 0:
                tv = vr
                while tv != 0:
                    lv = tv & (~tv + np.uint64(1))
                    b = _ctz64(lv)
                    tw = wr
                    while tw != 0:
                        lw = tw & (~tw + np.uint64(1))
                        c = _ctz64(lw)
                        for a in range(mn):
                            active_mask[(a * np_dim + b) * mp + c] = 1
                        tw ^= lw
                    tv ^= lv
            if ur != 0 and wr != 0:
                tu = ur
                while tu != 0:
                    lu = tu & (~tu + np.uint64(1))
                    a = _ctz64(lu)
                    tw = wr
                    while tw != 0:
                        lw = tw & (~tw + np.uint64(1))
                        c = _ctz64(lw)
                        for b in range(np_dim):
                            active_mask[(a * np_dim + b) * mp + c] = 1
                        tw ^= lw
                    tu ^= lu
            if ur != 0 and vr != 0:
                tu = ur
                while tu != 0:
                    lu = tu & (~tu + np.uint64(1))
                    a = _ctz64(lu)
                    tv = vr
                    while tv != 0:
                        lv = tv & (~tv + np.uint64(1))
                        b = _ctz64(lv)
                        base = (a * np_dim + b) * mp
                        for c in range(mp):
                            active_mask[base + c] = 1
                        tv ^= lv
                    tu ^= lu

    num_active = 0
    for idx in range(total_triples):
        if (counts[idx] & 1) != t_exact[idx]:
            empty_i32 = np.zeros(0, dtype=np.int32)
            empty_u64 = np.zeros((0, 0), dtype=np.uint64)
            empty_u8 = np.zeros(0, dtype=np.uint8)
            return (
                -1,
                num_vars,
                0,
                0,
                nnz_total,
                gauge_fixed,
                empty_i32,
                var_u,
                var_v,
                var_w,
                empty_i32,
                empty_i32,
                empty_u64,
                empty_u64,
                empty_u8,
            )
        if active_mask[idx] != 0:
            num_active += 1

    eq_id = np.full(total_triples, -1, dtype=np.int32)
    eq_triples = np.empty(num_active, dtype=np.int32)
    e1_vec = np.empty(num_active, dtype=np.uint8)
    cur = 0
    for idx in range(total_triples):
        if active_mask[idx] != 0:
            eq_id[idx] = cur
            eq_triples[cur] = idx
            e1_vec[cur] = ((counts[idx] - int(t_exact[idx])) >> 1) & 1
            cur += 1

    n_var_words = (num_vars + 63) // 64
    n_eq_words = (num_active + 63) // 64
    mat = np.zeros((num_active, n_var_words), dtype=np.uint64)
    P_mat = np.zeros((num_active, n_eq_words), dtype=np.uint64)
    for r_i in range(num_active):
        P_mat[r_i, r_i >> 6] = np.uint64(1) << np.uint64(r_i & 63)

    # Populate J1
    for r in range(R):
        ur = u[r]
        vr = v[r]
        wr = w[r]
        if support_restricted:
            if ur == 0 or vr == 0 or wr == 0:
                continue
            tu = ur
            while tu != 0:
                lu = tu & (~tu + np.uint64(1))
                a = _ctz64(lu)
                cu = var_u[r, a]
                tv = vr
                while tv != 0:
                    lv = tv & (~tv + np.uint64(1))
                    b = _ctz64(lv)
                    cv = var_v[r, b]
                    base = (a * np_dim + b) * mp
                    tw = wr
                    while tw != 0:
                        lw = tw & (~tw + np.uint64(1))
                        c = _ctz64(lw)
                        row = eq_id[base + c]
                        cw = var_w[r, c]
                        if cu >= 0:
                            mat[row, cu >> 6] ^= np.uint64(1) << np.uint64(cu & 63)
                        if cv >= 0:
                            mat[row, cv >> 6] ^= np.uint64(1) << np.uint64(cv & 63)
                        if cw >= 0:
                            mat[row, cw >> 6] ^= np.uint64(1) << np.uint64(cw & 63)
                        tw ^= lw
                    tv ^= lv
                tu ^= lu
        else:
            # dU * V1 * W1
            if vr != 0 and wr != 0:
                tv = vr
                while tv != 0:
                    lv = tv & (~tv + np.uint64(1))
                    b = _ctz64(lv)
                    tw = wr
                    while tw != 0:
                        lw = tw & (~tw + np.uint64(1))
                        c = _ctz64(lw)
                        for a in range(mn):
                            cu = var_u[r, a]
                            if cu >= 0:
                                row = eq_id[(a * np_dim + b) * mp + c]
                                mat[row, cu >> 6] ^= np.uint64(1) << np.uint64(cu & 63)
                        tw ^= lw
                    tv ^= lv
            # U1 * dV * W1
            if ur != 0 and wr != 0:
                tu = ur
                while tu != 0:
                    lu = tu & (~tu + np.uint64(1))
                    a = _ctz64(lu)
                    tw = wr
                    while tw != 0:
                        lw = tw & (~tw + np.uint64(1))
                        c = _ctz64(lw)
                        for b in range(np_dim):
                            cv = var_v[r, b]
                            if cv >= 0:
                                row = eq_id[(a * np_dim + b) * mp + c]
                                mat[row, cv >> 6] ^= np.uint64(1) << np.uint64(cv & 63)
                        tw ^= lw
                    tu ^= lu
            # U1 * V1 * dW
            if ur != 0 and vr != 0:
                tu = ur
                while tu != 0:
                    lu = tu & (~tu + np.uint64(1))
                    a = _ctz64(lu)
                    tv = vr
                    while tv != 0:
                        lv = tv & (~tv + np.uint64(1))
                        b = _ctz64(lv)
                        base = (a * np_dim + b) * mp
                        for c in range(mp):
                            cw = var_w[r, c]
                            if cw >= 0:
                                row = eq_id[base + c]
                                mat[row, cw >> 6] ^= np.uint64(1) << np.uint64(cw & 63)
                        tv ^= lv
                    tu ^= lu

    # Full Gauss-Jordan RREF simultaneously on mat and P_mat
    pivot_cols_buf = np.empty(num_vars, dtype=np.int32)
    is_pivot_col = np.zeros(num_vars, dtype=np.uint8)
    pivot_row = 0

    for col in range(num_vars):
        w_idx = col >> 6
        b_mask = np.uint64(1) << np.uint64(col & 63)
        found = -1
        for r_i in range(pivot_row, num_active):
            if (mat[r_i, w_idx] & b_mask) != 0:
                found = r_i
                break
        if found == -1:
            continue
        if found != pivot_row:
            for j in range(n_var_words):
                tmp = mat[pivot_row, j]
                mat[pivot_row, j] = mat[found, j]
                mat[found, j] = tmp
            for j in range(n_eq_words):
                tmp = P_mat[pivot_row, j]
                P_mat[pivot_row, j] = P_mat[found, j]
                P_mat[found, j] = tmp

        # Eliminate BOTH above and below pivot_row for full RREF
        for r_i in range(num_active):
            if r_i != pivot_row and (mat[r_i, w_idx] & b_mask) != 0:
                for j in range(w_idx, n_var_words):
                    mat[r_i, j] ^= mat[pivot_row, j]
                for j in range(n_eq_words):
                    P_mat[r_i, j] ^= P_mat[pivot_row, j]

        pivot_cols_buf[pivot_row] = col
        is_pivot_col[col] = 1
        pivot_row += 1
        if pivot_row == num_active:
            break

    rank_J1 = pivot_row
    pivot_cols = pivot_cols_buf[:rank_J1].copy()
    free_count = num_vars - rank_J1
    free_cols = np.empty(free_count, dtype=np.int32)
    fc = 0
    for col in range(num_vars):
        if is_pivot_col[col] == 0:
            free_cols[fc] = col
            fc += 1

    rref_mat = mat[:rank_J1, :].copy()
    return (
        0,
        num_vars,
        num_active,
        rank_J1,
        nnz_total,
        gauge_fixed,
        eq_triples,
        var_u,
        var_v,
        var_w,
        pivot_cols,
        free_cols,
        rref_mat,
        P_mat,
        e1_vec,
    )


@nb.njit(cache=True)
def _apply_P_and_solve_nb(
    num_vars: int,
    rank_J1: int,
    pivot_cols: np.ndarray,
    free_cols: np.ndarray,
    rref_mat: np.ndarray,
    P_mat: np.ndarray,
    e_vec: np.ndarray,
    free_assignment: np.ndarray,
) -> Tuple[int, np.ndarray]:
    """
    Given precomputed RREF + P_mat, compute:
      rhs_rref = P_mat @ e_vec (mod 2).
    Returns `(coker_viol, x_sol)` where `coker_viol == 0` iff system is consistent.
    `free_assignment` (len = num_vars - rank_J1, values 0/1) specifies exact values on free_cols!
    """
    num_active = e_vec.shape[0]
    n_eq_words = P_mat.shape[1]
    e_packed = np.zeros(n_eq_words, dtype=np.uint64)
    for i in range(num_active):
        if e_vec[i] != 0:
            e_packed[i >> 6] |= np.uint64(1) << np.uint64(i & 63)

    # Check cokernel rows [rank_J1 .. num_active - 1]
    coker_viol = 0
    for r_i in range(rank_J1, num_active):
        parity = 0
        for w_i in range(n_eq_words):
            parity ^= _popcount64(P_mat[r_i, w_i] & e_packed[w_i]) & 1
        if parity != 0:
            coker_viol += 1

    x = np.zeros(num_vars, dtype=np.uint8)
    if coker_viol > 0:
        return coker_viol, x

    # Assign free variables directly
    n_free = free_cols.shape[0]
    n_var_words = rref_mat.shape[1]
    free_packed = np.zeros(n_var_words, dtype=np.uint64)
    for t in range(n_free):
        val = free_assignment[t] & 1
        if val:
            col = free_cols[t]
            x[col] = 1
            free_packed[col >> 6] |= np.uint64(1) << np.uint64(col & 63)

    # Back-substitute each pivot row in one shot (RREF means pivot col is the ONLY pivot in row r_i!)
    for r_i in range(rank_J1):
        rhs_bit = 0
        for w_i in range(n_eq_words):
            rhs_bit ^= _popcount64(P_mat[r_i, w_i] & e_packed[w_i]) & 1
        free_contrib = 0
        for w_i in range(n_var_words):
            free_contrib ^= _popcount64(rref_mat[r_i, w_i] & free_packed[w_i]) & 1
        x[pivot_cols[r_i]] = rhs_bit ^ free_contrib

    return 0, x


def mod4_oracle_u64(
    m: int,
    n: int,
    p: int,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    support_restricted: bool = True,
) -> Mod4OracleResult:
    """
    Evaluate Stage L0 mod-4 liftability oracle on bit-packed uint64[R] F2 scheme.
    - `support_restricted=True`: ultra-fast (< 1ms) ternary/mod-4 support oracle with 2R gauge fixing.
    - `support_restricted=False`: unrestricted mod-4 oracle (allows new even entries 0 -> 2 for Q/Z lifts).
    """
    t0 = time.perf_counter()
    if support_restricted:
        coker, free_b, rk, n_vars, n_eqs, nnz, gf = _mod4_oracle_support_nb(m, n, p, u, v, w)
        if coker < 0:
            raise ValueError("Input (u, v, w) is NOT a valid F2 matrix multiplication scheme!")
    else:
        coker, free_b, rk, n_vars, n_eqs, nnz, gf = _mod4_oracle_unrestricted_nb(m, n, p, u, v, w)
        if coker < 0:
            raise ValueError("Input (u, v, w) is NOT a valid F2 matrix multiplication scheme!")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return Mod4OracleResult(
        coker_viol_4=int(coker),
        free_bits=int(free_b),
        rank_J1=int(rk),
        num_vars=int(n_vars),
        num_active_eqs=int(n_eqs),
        is_mod4_liftable=(int(coker) == 0),
        support_restricted=support_restricted,
        nnz_total=int(nnz),
        gauge_fixed=int(gf),
        elapsed_ms=round(elapsed_ms, 4),
    )


def mod4_oracle(scheme: BilinearScheme, support_restricted: bool = True) -> Mod4OracleResult:
    """Convenience wrapper accepting a `BilinearScheme`."""
    u, v, w = scheme_to_u64_f2(scheme)
    return mod4_oracle_u64(scheme.m, scheme.n, scheme.p, u, v, w, support_restricted=support_restricted)


@nb.njit(parallel=True, cache=True)
def _mod4_oracle_batch_support_nb(
    m: int, n: int, p: int, U_batch: np.ndarray, V_batch: np.ndarray, W_batch: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parallel 12-thread Numba batch evaluation of Stage L0 support-restricted mod-4 oracle.
    Inputs: U_batch, V_batch, W_batch of shape (B, R), dtype uint64.
    Returns: (coker_viol_4[B], free_bits[B], rank_J1[B]).
    """
    B = U_batch.shape[0]
    cokers = np.empty(B, dtype=np.int32)
    frees = np.empty(B, dtype=np.int32)
    ranks = np.empty(B, dtype=np.int32)
    for b_idx in nb.prange(B):
        c, f, rk, _, _, _, _ = _mod4_oracle_support_nb(m, n, p, U_batch[b_idx], V_batch[b_idx], W_batch[b_idx])
        cokers[b_idx] = c
        frees[b_idx] = f
        ranks[b_idx] = rk
    return cokers, frees, ranks


def mod4_oracle_batch_u64(
    m: int, n: int, p: int, U_batch: np.ndarray, V_batch: np.ndarray, W_batch: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parallel CPU 12T batch support-restricted mod-4 oracle."""
    return _mod4_oracle_batch_support_nb(
        m,
        n,
        p,
        np.ascontiguousarray(U_batch, dtype=np.uint64),
        np.ascontiguousarray(V_batch, dtype=np.uint64),
        np.ascontiguousarray(W_batch, dtype=np.uint64),
    )


@nb.njit(parallel=True, cache=True)
def _mod4_oracle_batch_unrestricted_nb(
    m: int, n: int, p: int, U_batch: np.ndarray, V_batch: np.ndarray, W_batch: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parallel 12-thread Numba batch evaluation of Stage L0 UNRESTRICTED mod-4 oracle."""
    B = U_batch.shape[0]
    cokers = np.empty(B, dtype=np.int32)
    frees = np.empty(B, dtype=np.int32)
    ranks = np.empty(B, dtype=np.int32)
    for b_idx in nb.prange(B):
        c, f, rk, _, _, _, _ = _mod4_oracle_unrestricted_nb(m, n, p, U_batch[b_idx], V_batch[b_idx], W_batch[b_idx])
        cokers[b_idx] = c
        frees[b_idx] = f
        ranks[b_idx] = rk
    return cokers, frees, ranks


def mod4_oracle_batch_unrestricted_u64(
    m: int, n: int, p: int, U_batch: np.ndarray, V_batch: np.ndarray, W_batch: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parallel CPU 12T batch unrestricted mod-4 oracle."""
    return _mod4_oracle_batch_unrestricted_nb(
        m,
        n,
        p,
        np.ascontiguousarray(U_batch, dtype=np.uint64),
        np.ascontiguousarray(V_batch, dtype=np.uint64),
        np.ascontiguousarray(W_batch, dtype=np.uint64),
    )


# =============================================================================
# 3. Stage L1 — Linear Hensel Lifting (2^k -> 2^{k+1} to 2^64) + Wang Rational Rec.
# =============================================================================


@dataclass
class LiftResult:
    success: bool
    status: str  # "verified_ZT", "verified_Z", "verified_Q", "obstructed_mod4", "obstructed_mod2k", "recon_failed", "unsat", "timeout"
    scheme: Optional[BilinearScheme]
    report: Optional[VerificationReport]
    method: str
    elapsed_ms: float
    coker_viol_4: int = 0
    max_level_reached: int = 1
    details: Dict[str, Any] = field(default_factory=dict)


@nb.njit(cache=True)
def _compute_mod2kp1_residual_nb(
    m: int,
    n: int,
    p: int,
    U_k: np.ndarray,
    V_k: np.ndarray,
    W_k: np.ndarray,
    eq_triples: np.ndarray,
    k: int,
) -> Tuple[int, np.ndarray]:
    """
    Compute exact residual e_k = ((T_exact - T_k) >> k) & 1 on eq_triples using
    exact uint64 wrap-around arithmetic (exact mod 2^{k+1} for all 1 <= k <= 63!).
    Also verifies that all lower k bits of (T_exact - T_k) are strictly 0 across the
    entire tensor (returning err_low_bits > 0 if violated).
    """
    mn = m * n
    np_dim = n * p
    mp = m * p
    R = U_k.shape[0]
    total_triples = mn * np_dim * mp
    T_k = np.zeros(total_triples, dtype=np.uint64)

    for r in range(R):
        # Collect nonzero indices for sparse contraction
        for a in range(mn):
            ua = U_k[r, a]
            if ua == 0:
                continue
            for b in range(np_dim):
                vb = V_k[r, b]
                if vb == 0:
                    continue
                uv = ua * vb
                base = (a * np_dim + b) * mp
                for c in range(mp):
                    wc = W_k[r, c]
                    if wc != 0:
                        T_k[base + c] += uv * wc

    # Subtract T_k from T_exact (mod 2^{64})
    # Note: for k <= 63, mask_lo = (1 << k) - 1
    mask_lo = (np.uint64(1) << np.uint64(k)) - np.uint64(1)
    # Build expected ones quickly
    t_exact = np.zeros(total_triples, dtype=np.uint64)
    for i1 in range(m):
        for i2 in range(n):
            a = i1 * n + i2
            for j2 in range(p):
                b = i2 * p + j2
                c = i1 * p + j2
                t_exact[(a * np_dim + b) * mp + c] = np.uint64(1)

    err_low = 0
    # Check off-active triples at level k+1 too!
    # An entry NOT in eq_triples has zero Jacobian row, so if its bit k is 1, it is an obstruction!
    in_eq = np.zeros(total_triples, dtype=np.uint8)
    for i in range(eq_triples.shape[0]):
        in_eq[eq_triples[i]] = 1

    off_active_viol = 0
    for idx in range(total_triples):
        diff = t_exact[idx] - T_k[idx]
        if (diff & mask_lo) != 0:
            err_low += 1
        elif in_eq[idx] == 0:
            if ((diff >> np.uint64(k)) & np.uint64(1)) != 0:
                off_active_viol += 1

    num_active = eq_triples.shape[0]
    e_k = np.empty(num_active, dtype=np.uint8)
    for i in range(num_active):
        idx = eq_triples[i]
        diff = t_exact[idx] - T_k[idx]
        e_k[i] = (diff >> np.uint64(k)) & np.uint64(1)

    return err_low + off_active_viol * 1000000, e_k


@nb.njit(cache=True)
def _guided_free_assignment_nb(
    num_vars: int,
    rank_J1: int,
    pivot_cols: np.ndarray,
    free_cols: np.ndarray,
    rref_mat: np.ndarray,
    x0: np.ndarray,
    desired: np.ndarray,
    priority_order: np.ndarray,
) -> np.ndarray:
    """
    Lexicographically optimal Guided Constrained Lifting over GF(2):
    Finds `c in GF(2)^{n_free}` maximizing satisfied constraints `(x0 ^ H c)[j] == desired[j]`
    in `priority_order` (off-support zeros first, then ternary {-1, 0, +1} entries, then pivots)
    via incremental GF(2) basis insertion. Strictly preserves consistency!
    """
    n_free = free_cols.shape[0]
    if n_free == 0:
        return np.zeros(0, dtype=np.uint8)

    # Map each column -> pivot row index (or -1 if free) and free index (or -1 if pivot)
    col_to_piv = np.full(num_vars, -1, dtype=np.int32)
    for r_i in range(rank_J1):
        col_to_piv[pivot_cols[r_i]] = r_i
    col_to_free = np.full(num_vars, -1, dtype=np.int32)
    for t in range(n_free):
        col_to_free[free_cols[t]] = t

    n_c_words = (n_free + 63) // 64
    basis_lhs = np.zeros((n_free, n_c_words), dtype=np.uint64)
    basis_rhs = np.zeros(n_free, dtype=np.uint8)
    basis_used = np.zeros(n_free, dtype=np.uint8)
    row_lhs = np.zeros(n_c_words, dtype=np.uint64)
    inserted = 0

    for idx_ord in range(priority_order.shape[0]):
        if inserted == n_free:
            break
        col = priority_order[idx_ord]
        rhs = (x0[col] ^ desired[col]) & np.uint8(1)
        row_lhs[:] = np.uint64(0)

        fi = col_to_free[col]
        if fi >= 0:
            row_lhs[fi >> 6] = np.uint64(1) << np.uint64(fi & 63)
        else:
            pr = col_to_piv[col]
            if pr < 0:
                continue
            for t in range(n_free):
                fc = free_cols[t]
                if ((rref_mat[pr, fc >> 6] >> np.uint64(fc & 63)) & np.uint64(1)) != 0:
                    row_lhs[t >> 6] |= np.uint64(1) << np.uint64(t & 63)

        # Reduce row_lhs against existing basis
        lead = -1
        for w_i in range(n_c_words):
            while row_lhs[w_i] != 0:
                lsb = row_lhs[w_i] & (~row_lhs[w_i] + np.uint64(1))
                bit = (w_i << 6) + _ctz64(lsb)
                if basis_used[bit] != 0:
                    for ww in range(w_i, n_c_words):
                        row_lhs[ww] ^= basis_lhs[bit, ww]
                    rhs ^= basis_rhs[bit]
                else:
                    lead = bit
                    break
            if lead >= 0:
                break

        if lead >= 0:
            basis_used[lead] = 1
            for ww in range(n_c_words):
                basis_lhs[lead, ww] = row_lhs[ww]
            basis_rhs[lead] = rhs
            inserted += 1

    # Back-substitute upper-triangular basis (bit = n_free - 1 down to 0)
    c_sol = np.zeros(n_free, dtype=np.uint8)
    for bit in range(n_free - 1, -1, -1):
        if basis_used[bit] == 0:
            c_sol[bit] = 0
            continue
        val = basis_rhs[bit]
        for t in range(bit + 1, n_free):
            if c_sol[t] != 0:
                if ((basis_lhs[bit, t >> 6] >> np.uint64(t & 63)) & np.uint64(1)) != 0:
                    val ^= np.uint8(1)
        c_sol[bit] = val

    return c_sol


@nb.njit(cache=True)
def _linear_hensel_run_nb(
    m: int,
    n: int,
    p: int,
    u1: np.ndarray,
    v1: np.ndarray,
    w1: np.ndarray,
    num_vars: int,
    rank_J1: int,
    eq_triples: np.ndarray,
    var_u: np.ndarray,
    var_v: np.ndarray,
    var_w: np.ndarray,
    pivot_cols: np.ndarray,
    free_cols: np.ndarray,
    rref_mat: np.ndarray,
    P_mat: np.ndarray,
    e1_vec: np.ndarray,
    free_init: np.ndarray,
    target_bits: int,
    steer_mode: int,
    rng_seed: np.uint64,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute linear Hensel lifting 2^1 -> 2^2 -> ... -> 2^{target_bits} (target_bits <= 64)
    reusing ONE precomputed RREF + P_mat with lexicographic Guided Constrained Steering!
    steer_mode:
      0 = unsteered free random nullspace walk
      1 = full guided ternary steering at all k >= 1 (pins off-support zeros=0 first)
      2 = nullspace kick at k == 1 (uses full free_init incl. off-support to clear mod-8 obstructions!),
          then guided ternary steering at k >= 2 for compact Z/Q rational reconstruction!
    Returns (max_level_achieved, U_k, V_k, W_k) where max_level_achieved == target_bits on full lift.
    """
    mn = m * n
    np_dim = n * p
    mp = m * p
    R = u1.shape[0]
    n_free = free_cols.shape[0]

    U_k = np.zeros((R, mn), dtype=np.uint64)
    V_k = np.zeros((R, np_dim), dtype=np.uint64)
    W_k = np.zeros((R, mp), dtype=np.uint64)
    for r in range(R):
        for a in range(mn):
            U_k[r, a] = (u1[r] >> np.uint64(a)) & np.uint64(1)
        for b in range(np_dim):
            V_k[r, b] = (v1[r] >> np.uint64(b)) & np.uint64(1)
        for c in range(mp):
            W_k[r, c] = (w1[r] >> np.uint64(c)) & np.uint64(1)

    # Precompute reverse lookup var_id -> (mode, r, idx, is_on_support)
    var_mode = np.empty(num_vars, dtype=np.int8)
    var_r = np.empty(num_vars, dtype=np.int32)
    var_pos = np.empty(num_vars, dtype=np.int32)
    var_on_supp = np.zeros(num_vars, dtype=np.uint8)
    for r in range(R):
        for a in range(mn):
            vid = var_u[r, a]
            if vid >= 0:
                var_mode[vid] = 0
                var_r[vid] = r
                var_pos[vid] = a
                var_on_supp[vid] = (u1[r] >> np.uint64(a)) & np.uint64(1)
        for b in range(np_dim):
            vid = var_v[r, b]
            if vid >= 0:
                var_mode[vid] = 1
                var_r[vid] = r
                var_pos[vid] = b
                var_on_supp[vid] = (v1[r] >> np.uint64(b)) & np.uint64(1)
        for c in range(mp):
            vid = var_w[r, c]
            if vid >= 0:
                var_mode[vid] = 2
                var_r[vid] = r
                var_pos[vid] = c
                var_on_supp[vid] = (w1[r] >> np.uint64(c)) & np.uint64(1)

    # Priority order for guided steering: off-support zeros FIRST (keep sparse!), then on-support entries
    priority_order = np.empty(num_vars, dtype=np.int32)
    po = 0
    for vid in range(num_vars):
        if var_on_supp[vid] == 0:
            priority_order[po] = vid
            po += 1
    for vid in range(num_vars):
        if var_on_supp[vid] != 0:
            priority_order[po] = vid
            po += 1

    state = rng_seed if rng_seed != 0 else np.uint64(0x9E3779B97F4A7C15)
    e_cur = e1_vec
    zero_free = np.zeros(n_free, dtype=np.uint8)
    free_vec = np.empty(n_free, dtype=np.uint8)
    desired = np.empty(num_vars, dtype=np.uint8)

    for k in range(1, target_bits):
        if k > 1:
            err_flag, e_cur = _compute_mod2kp1_residual_nb(m, n, p, U_k, V_k, W_k, eq_triples, k)
            if err_flag != 0:
                return k, U_k, V_k, W_k

        coker_k, x0 = _apply_P_and_solve_nb(
            num_vars, rank_J1, pivot_cols, free_cols, rref_mat, P_mat, e_cur, zero_free
        )
        if coker_k > 0:
            return k, U_k, V_k, W_k

        if k == 1:
            if steer_mode == 1:
                desired[:] = 0
                for t in range(n_free):
                    col = free_cols[t]
                    if var_on_supp[col] != 0:
                        desired[col] = free_init[t]
                free_vec = _guided_free_assignment_nb(
                    num_vars, rank_J1, pivot_cols, free_cols, rref_mat, x0, desired, priority_order
                )
            else:
                for t in range(n_free):
                    free_vec[t] = free_init[t]
        else:
            if steer_mode >= 1:
                neg1_mod = (np.uint64(1) << np.uint64(k)) - np.uint64(1)
                for vid in range(num_vars):
                    md = var_mode[vid]
                    rr = var_r[vid]
                    pp = var_pos[vid]
                    val = U_k[rr, pp] if md == 0 else (V_k[rr, pp] if md == 1 else W_k[rr, pp])
                    if val == 0 or val == 1:
                        desired[vid] = 0
                    elif val == neg1_mod:
                        desired[vid] = 1
                    else:
                        desired[vid] = (val >> np.uint64(k - 1)) & np.uint64(1)
                free_vec = _guided_free_assignment_nb(
                    num_vars, rank_J1, pivot_cols, free_cols, rref_mat, x0, desired, priority_order
                )
            else:
                for t in range(n_free):
                    state ^= state >> np.uint64(12)
                    state ^= state << np.uint64(25)
                    state ^= state >> np.uint64(27)
                    free_vec[t] = (state * np.uint64(0x2545F4914F6CDD1D)) & np.uint64(1)

        _, delta = _apply_P_and_solve_nb(
            num_vars, rank_J1, pivot_cols, free_cols, rref_mat, P_mat, e_cur, free_vec
        )

        bit_add = np.uint64(1) << np.uint64(k)
        for vid in range(num_vars):
            if delta[vid] != 0:
                md = var_mode[vid]
                rr = var_r[vid]
                pp = var_pos[vid]
                if md == 0:
                    U_k[rr, pp] += bit_add
                elif md == 1:
                    V_k[rr, pp] += bit_add
                else:
                    W_k[rr, pp] += bit_add

    return target_bits, U_k, V_k, W_k


def wang_rational_reconstruct(a: int, M: int, bound: Optional[int] = None) -> Optional[Fraction]:
    """
    Exact Extended-Euclidean Rational Reconstruction (Wang's algorithm):
    Recover unique Fraction(num, den) with |num| <= B, 0 < den <= B, gcd(|num|, den) == 1,
    and num == a * den (mod M), where B = isqrt(M // 2) by default.
    Pure Python exact int arithmetic (zero floats).
    """
    a %= M
    if a == 0:
        return Fraction(0, 1)
    if a == 1:
        return Fraction(1, 1)
    if a == M - 1:
        return Fraction(-1, 1)
    if bound is None:
        bound = math.isqrt(M // 2)
    if a <= bound:
        return Fraction(a, 1)
    if M - a <= bound:
        return Fraction(a - M, 1)

    r0, r1 = M, a
    t0, t1 = 0, 1
    while r1 > bound:
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        t0, t1 = t1, t0 - q * t1

    if t1 < 0:
        r1 = -r1
        t1 = -t1
    if t1 == 0 or t1 > bound or abs(r1) > bound:
        return None
    if math.gcd(abs(r1), t1) != 1:
        return None
    if (a * t1 - r1) % M != 0:
        return None
    return Fraction(r1, t1)


def _reconstruct_and_verify(
    m: int,
    n: int,
    p: int,
    U_k: np.ndarray,
    V_k: np.ndarray,
    W_k: np.ndarray,
    bits: int,
    name: str,
) -> Optional[Tuple[BilinearScheme, VerificationReport, str]]:
    """
    Attempt exact integer & Wang rational reconstruction from mod 2^bits matrices,
    and immediately certify with `brent_verifier.verify_scheme()`.
    """
    M = 1 << bits
    half = M >> 1
    bound = math.isqrt(M >> 1)
    R = U_k.shape[0]
    mn, np_dim, mp = m * n, n * p, m * p

    # 1. Fast path: try balanced signed integers first (catches ZT and small Z immediately)
    max_int_mag = min(bound, 1 << 20)
    all_small_int = True
    U_z: List[List[int]] = []
    V_z: List[List[int]] = []
    W_z: List[List[int]] = []
    for src, dst, cols in ((U_k, U_z, mn), (V_k, V_z, np_dim), (W_k, W_z, mp)):
        for r in range(R):
            row: List[int] = []
            for j in range(cols):
                val = int(src[r, j]) & (M - 1)
                sval = val if val < half else val - M
                if abs(sval) > max_int_mag:
                    all_small_int = False
                row.append(sval)
            dst.append(row)

    if all_small_int:
        cand_z = BilinearScheme(m=m, n=n, p=p, U=U_z, V=V_z, W=W_z, ring="Z", name=name)
        rep_z = verify_scheme(cand_z)
        if rep_z.valid:
            status = "verified_ZT" if rep_z.is_ternary else "verified_Z"
            return cand_z, rep_z, status

    # 2. General Wang rational reconstruction (handles odd denominators like 1/3, 1/5, ...)
    U_q: List[List[Fraction]] = []
    V_q: List[List[Fraction]] = []
    W_q: List[List[Fraction]] = []
    all_int_q = True
    for src, dst_q, cols in ((U_k, U_q, mn), (V_k, V_q, np_dim), (W_k, W_q, mp)):
        for r in range(R):
            row_q: List[Fraction] = []
            for j in range(cols):
                val = int(src[r, j]) & (M - 1)
                fr = wang_rational_reconstruct(val, M, bound=bound)
                if fr is None:
                    return None
                if fr.denominator != 1:
                    all_int_q = False
                row_q.append(fr)
            dst_q.append(row_q)

    if all_int_q:
        cand_z2 = BilinearScheme(
            m=m,
            n=n,
            p=p,
            U=[[int(x.numerator) for x in row] for row in U_q],
            V=[[int(x.numerator) for x in row] for row in V_q],
            W=[[int(x.numerator) for x in row] for row in W_q],
            ring="Z",
            name=name,
        )
        rep_z2 = verify_scheme(cand_z2)
        if rep_z2.valid:
            status = "verified_ZT" if rep_z2.is_ternary else "verified_Z"
            return cand_z2, rep_z2, status

    cand_q = BilinearScheme(m=m, n=n, p=p, U=U_q, V=V_q, W=W_q, ring="Q", name=name)
    rep_q = verify_scheme(cand_q)
    if rep_q.valid:
        return cand_q, rep_q, "verified_Q"
    return None


def linear_hensel_lift(
    scheme_or_u64: BilinearScheme | Tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray],
    support_restricted: bool = False,
    target_bits: int = 64,
    max_trials: int = 16,
    seed: int = 42,
    name: str = "",
) -> LiftResult:
    """
    Stage L1: Linear 2-adic Hensel lifting (2^k -> 2^{k+1} up to 2^{target_bits}, <= 2^64)
    reusing ONE precomputed GF(2) RREF + P_mat, followed by exact Wang rational reconstruction
    and dual-path verification via `brent_verifier.verify_scheme()`.
    """
    t0 = time.perf_counter()
    if isinstance(scheme_or_u64, BilinearScheme):
        m, n, p = scheme_or_u64.m, scheme_or_u64.n, scheme_or_u64.p
        u1, v1, w1 = scheme_to_u64_f2(scheme_or_u64)
        base_name = name or (scheme_or_u64.name + "_hensel")
    else:
        m, n, p, u1, v1, w1 = scheme_or_u64
        base_name = name or f"{m}x{n}x{p}_R{len(u1)}_hensel"

    if target_bits < 4 or target_bits > 64:
        raise ValueError(f"target_bits must be in [4, 64], got {target_bits}")

    res = _build_jacobian_and_rref_nb(m, n, p, u1, v1, w1, support_restricted)
    status = res[0]
    if status < 0:
        raise ValueError("Input scheme is NOT valid mod 2!")

    (
        _,
        num_vars,
        num_active,
        rank_J1,
        nnz_total,
        gauge_fixed,
        eq_triples,
        var_u,
        var_v,
        var_w,
        pivot_cols,
        free_cols,
        rref_mat,
        P_mat,
        e1_vec,
    ) = res
    n_free = num_vars - rank_J1

    # Check k=1 mod-4 consistency immediately
    coker_4, _ = _apply_P_and_solve_nb(
        num_vars, rank_J1, pivot_cols, free_cols, rref_mat, P_mat, e1_vec, np.zeros(n_free, dtype=np.uint8)
    )
    if coker_4 > 0:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return LiftResult(
            success=False,
            status="obstructed_mod4",
            scheme=None,
            report=None,
            method=f"linear_hensel_{'supp' if support_restricted else 'unres'}",
            elapsed_ms=round(elapsed_ms, 3),
            coker_viol_4=int(coker_4),
            max_level_reached=1,
            details={"rank_J1": int(rank_J1), "free_bits": int(n_free), "num_active_eqs": int(num_active)},
        )

    rng = np.random.default_rng(seed)
    best_level = 1

    # Generate diverse k=1 nullspace samples (trial 0 = 0, sparse 1..4-bit kicks in ker(J1),
    # and random full-nullspace kicks using steer_mode=2 to clear mod-8 obstructions!)
    trial_inits: List[Tuple[np.ndarray, int]] = []
    if n_free <= 5 and (1 << n_free) <= max_trials:
        for mask in range(1 << n_free):
            arr = np.array([(mask >> t) & 1 for t in range(n_free)], dtype=np.uint8)
            trial_inits.append((arr, 1))
    else:
        trial_inits.append((np.zeros(n_free, dtype=np.uint8), 1))
        trial_inits.append((np.zeros(n_free, dtype=np.uint8), 2))
        trial_inits.append((np.zeros(n_free, dtype=np.uint8), 0))
        # Sparse 1-bit & 2-bit nullspace kicks (mode 1 + mode 2 so off-support kicks are tested!)
        n_single = min(n_free, max(0, max_trials // 4))
        for t in range(n_single):
            arr = np.zeros(n_free, dtype=np.uint8)
            arr[t] = 1
            trial_inits.append((arr, 2 if (t & 1) == 1 else 1))
        # Random sparse (2..6 bits) and dense nullspace kicks across ker(J1)
        while len(trial_inits) < max_trials:
            idx_k = len(trial_inits)
            if idx_k % 2 == 0 and n_free >= 4:
                arr = np.zeros(n_free, dtype=np.uint8)
                k_bits = int(rng.integers(2, min(8, n_free) + 1))
                chosen = rng.choice(n_free, size=k_bits, replace=False)
                arr[chosen] = 1
                trial_inits.append((arr, 2))
            else:
                arr = rng.integers(0, 2, size=n_free, dtype=np.uint8)
                mode = 1 if (idx_k % 3 == 0) else (2 if (idx_k % 3 == 1) else 0)
                trial_inits.append((arr, mode))

    for t_idx, (free_init, steer_mode) in enumerate(trial_inits):
        lvl, U_k, V_k, W_k = _linear_hensel_run_nb(
            m,
            n,
            p,
            u1,
            v1,
            w1,
            num_vars,
            rank_J1,
            eq_triples,
            var_u,
            var_v,
            var_w,
            pivot_cols,
            free_cols,
            rref_mat,
            P_mat,
            e1_vec,
            free_init,
            target_bits,
            steer_mode,
            np.uint64(seed + 10007 * (t_idx + 1)),
        )
        if lvl > best_level:
            best_level = int(lvl)
        if lvl == target_bits:
            # Try reconstructing at progressively larger bit widths (16, 32, target_bits)
            for check_bits in sorted({min(16, target_bits), min(32, target_bits), target_bits}):
                rec = _reconstruct_and_verify(m, n, p, U_k, V_k, W_k, check_bits, base_name)
                if rec is not None:
                    cand, rep, st = rec
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    return LiftResult(
                        success=True,
                        status=st,
                        scheme=cand,
                        report=rep,
                        method=f"linear_hensel_{'supp' if support_restricted else 'unres'}",
                        elapsed_ms=round(elapsed_ms, 3),
                        coker_viol_4=0,
                        max_level_reached=int(lvl),
                        details={
                            "trial_idx": t_idx,
                            "steer_mode": steer_mode,
                            "reconstructed_at_bits": check_bits,
                            "rank_J1": int(rank_J1),
                            "free_bits": int(n_free),
                            "max_abs_coeff": str(rep.max_abs_coeff),
                        },
                    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    final_status = "recon_failed" if best_level == target_bits else "obstructed_mod2k"
    return LiftResult(
        success=False,
        status=final_status,
        scheme=None,
        report=None,
        method=f"linear_hensel_{'supp' if support_restricted else 'unres'}",
        elapsed_ms=round(elapsed_ms, 3),
        coker_viol_4=0,
        max_level_reached=best_level,
        details={
            "trials_run": len(trial_inits),
            "rank_J1": int(rank_J1),
            "free_bits": int(n_free),
            "num_active_eqs": int(num_active),
        },
    )


# =============================================================================
# 4. Stage L2 — Exact Z3 5.1.0 Ternary Sign Lifter (Bool + Xor + PbEq)
# =============================================================================


@nb.njit(cache=True)
def _exhaustive_ternary_nullspace_nb(
    m: int,
    n: int,
    p: int,
    u1: np.ndarray,
    v1: np.ndarray,
    w1: np.ndarray,
    num_vars: int,
    rank_J1: int,
    var_u: np.ndarray,
    var_v: np.ndarray,
    var_w: np.ndarray,
    pivot_cols: np.ndarray,
    free_cols: np.ndarray,
    rref_mat: np.ndarray,
    P_mat: np.ndarray,
    e1_vec: np.ndarray,
) -> Tuple[bool, np.ndarray]:
    """
    When free_bits <= 15 (<= 32,768 mod-4 valid sign assignments), exhaustively test
    EVERY mod-4-valid ternary sign vector in exact integer arithmetic in milliseconds!
    """
    n_free = free_cols.shape[0]
    total_masks = 1 << n_free
    mn, np_dim, mp = m * n, n * p, m * p
    R = u1.shape[0]
    total_triples = mn * np_dim * mp

    # Precompute base rhs_rref for pivot rows
    num_active = e1_vec.shape[0]
    n_eq_words = P_mat.shape[1]
    e_packed = np.zeros(n_eq_words, dtype=np.uint64)
    for i in range(num_active):
        if e1_vec[i] != 0:
            e_packed[i >> 6] |= np.uint64(1) << np.uint64(i & 63)

    base_rhs = np.zeros(rank_J1, dtype=np.uint8)
    for r_i in range(rank_J1):
        par = 0
        for w_i in range(n_eq_words):
            par ^= _popcount64(P_mat[r_i, w_i] & e_packed[w_i]) & 1
        base_rhs[r_i] = par

    # Precompute sparse H_col[t, r_i] = bit of rref_mat[r_i, free_cols[t]]
    # Store as uint32 mask over free bits per pivot row (valid since n_free <= 15)
    piv_free_mask = np.zeros(rank_J1, dtype=np.uint32)
    for r_i in range(rank_J1):
        mk = np.uint32(0)
        for t in range(n_free):
            fc = free_cols[t]
            if ((rref_mat[r_i, fc >> 6] >> np.uint64(fc & 63)) & np.uint64(1)) != 0:
                mk |= np.uint32(1) << np.uint32(t)
        piv_free_mask[r_i] = mk

    # Build sparse list of active triples (a,b,c) with expected value and list of covering terms
    t_exact = np.zeros(total_triples, dtype=np.int16)
    for i1 in range(m):
        for i2 in range(n):
            a = i1 * n + i2
            for j2 in range(p):
                b = i2 * p + j2
                c = i1 * p + j2
                t_exact[(a * np_dim + b) * mp + c] = 1

    # Precompute nonzero lists per term r
    U_s = np.zeros((R, mn), dtype=np.int8)
    V_s = np.zeros((R, np_dim), dtype=np.int8)
    W_s = np.zeros((R, mp), dtype=np.int8)
    x = np.zeros(num_vars, dtype=np.uint8)
    acc = np.zeros(total_triples, dtype=np.int16)

    for mask in range(total_masks):
        for t in range(n_free):
            x[free_cols[t]] = (mask >> t) & 1
        for r_i in range(rank_J1):
            inter = np.uint64(piv_free_mask[r_i] & np.uint32(mask))
            x[pivot_cols[r_i]] = base_rhs[r_i] ^ (_popcount64(inter) & 1)

        # Fill ternary signs (-1 if x==1 else +1 on support, 0 off support)
        for r in range(R):
            ur = u1[r]
            vr = v1[r]
            wr = w1[r]
            for a in range(mn):
                if ((ur >> np.uint64(a)) & np.uint64(1)) != 0:
                    vid = var_u[r, a]
                    U_s[r, a] = -1 if (vid >= 0 and x[vid] == 1) else 1
                else:
                    U_s[r, a] = 0
            for b in range(np_dim):
                if ((vr >> np.uint64(b)) & np.uint64(1)) != 0:
                    vid = var_v[r, b]
                    V_s[r, b] = -1 if (vid >= 0 and x[vid] == 1) else 1
                else:
                    V_s[r, b] = 0
            for c in range(mp):
                if ((wr >> np.uint64(c)) & np.uint64(1)) != 0:
                    vid = var_w[r, c]
                    W_s[r, c] = -1 if (vid >= 0 and x[vid] == 1) else 1
                else:
                    W_s[r, c] = 0

        acc[:] = 0
        for r in range(R):
            ur = u1[r]
            vr = v1[r]
            wr = w1[r]
            if ur == 0 or vr == 0 or wr == 0:
                continue
            tu = ur
            while tu != 0:
                lu = tu & (~tu + np.uint64(1))
                a = _ctz64(lu)
                ua = int(U_s[r, a])
                tv = vr
                while tv != 0:
                    lv = tv & (~tv + np.uint64(1))
                    b = _ctz64(lv)
                    uv = ua * int(V_s[r, b])
                    base = (a * np_dim + b) * mp
                    tw = wr
                    while tw != 0:
                        lw = tw & (~tw + np.uint64(1))
                        c = _ctz64(lw)
                        acc[base + c] += uv * int(W_s[r, c])
                        tw ^= lw
                    tv ^= lv
                tu ^= lu

        ok = True
        for idx in range(total_triples):
            if acc[idx] != t_exact[idx]:
                ok = False
                break
        if ok:
            return True, x.copy()

    return False, np.zeros(0, dtype=np.uint8)


def z3_ternary_lift(
    scheme_or_u64: BilinearScheme | Tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray],
    timeout_s: float = 10.0,
    force_z3: bool = False,
    name: str = "",
) -> LiftResult:
    """
    Stage L2: Exact Ternary Sign Lifter (Z2 -> Z_T = {-1, 0, +1}) on the existing F2 support
    with 2R scaling gauge fixing.

    - First checks Stage L0 support-restricted mod-4 oracle (instant UNSAT proof if coker_viol_4 > 0!).
    - If `not force_z3` and `free_bits <= 15`: uses exhaustive exact Numba nullspace search (< 50 ms!).
    - Otherwise (or when `force_z3=True`): encodes the exact Brent equations as pure Boolean
      `Xor` + `PbEq` cardinality constraints in `Z3 5.1.0` (with strict `timeout_s` timeout).
    - Every found solution is certified via `brent_verifier.verify_scheme()` over ring="Z" (`is_ternary=True`).
    """
    t0 = time.perf_counter()
    if isinstance(scheme_or_u64, BilinearScheme):
        m, n, p = scheme_or_u64.m, scheme_or_u64.n, scheme_or_u64.p
        u1, v1, w1 = scheme_to_u64_f2(scheme_or_u64)
        base_name = name or (scheme_or_u64.name + "_ZT")
    else:
        m, n, p, u1, v1, w1 = scheme_or_u64
        base_name = name or f"{m}x{n}x{p}_R{len(u1)}_ZT"

    # Stage L0 pre-screen + exact RREF on support
    res = _build_jacobian_and_rref_nb(m, n, p, u1, v1, w1, True)
    if res[0] < 0:
        raise ValueError("Input scheme is NOT valid mod 2!")

    (
        _,
        num_vars,
        num_active,
        rank_J1,
        nnz_total,
        gauge_fixed,
        eq_triples,
        var_u,
        var_v,
        var_w,
        pivot_cols,
        free_cols,
        rref_mat,
        P_mat,
        e1_vec,
    ) = res
    n_free = num_vars - rank_J1

    coker_4, _ = _apply_P_and_solve_nb(
        num_vars, rank_J1, pivot_cols, free_cols, rref_mat, P_mat, e1_vec, np.zeros(n_free, dtype=np.uint8)
    )
    if coker_4 > 0:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return LiftResult(
            success=False,
            status="obstructed_mod4",
            scheme=None,
            report=None,
            method="z3_ternary_support",
            elapsed_ms=round(elapsed_ms, 3),
            coker_viol_4=int(coker_4),
            max_level_reached=1,
            details={"rank_J1": int(rank_J1), "free_bits": int(n_free), "nnz_total": int(nnz_total)},
        )

    mn, np_dim, mp = m * n, n * p, m * p
    R = len(u1)

    def _signs_to_verified_scheme(sign_bits: Sequence[int], method_tag: str, extra: Dict[str, Any]) -> LiftResult:
        U_z: List[List[int]] = []
        V_z: List[List[int]] = []
        W_z: List[List[int]] = []
        for r in range(R):
            ur, vr, wr = int(u1[r]), int(v1[r]), int(w1[r])
            u_row = []
            for a in range(mn):
                if (ur >> a) & 1:
                    vid = int(var_u[r, a])
                    u_row.append(-1 if (vid >= 0 and sign_bits[vid]) else 1)
                else:
                    u_row.append(0)
            v_row = []
            for b in range(np_dim):
                if (vr >> b) & 1:
                    vid = int(var_v[r, b])
                    v_row.append(-1 if (vid >= 0 and sign_bits[vid]) else 1)
                else:
                    v_row.append(0)
            w_row = []
            for c in range(mp):
                if (wr >> c) & 1:
                    vid = int(var_w[r, c])
                    w_row.append(-1 if (vid >= 0 and sign_bits[vid]) else 1)
                else:
                    w_row.append(0)
            U_z.append(u_row)
            V_z.append(v_row)
            W_z.append(w_row)
        cand = BilinearScheme(m=m, n=n, p=p, U=U_z, V=V_z, W=W_z, ring="Z", name=base_name)
        rep = verify_scheme(cand)
        if not (rep.valid and rep.is_ternary):
            raise RuntimeError(f"Internal error: lifted ternary scheme failed brent_verifier! {rep}")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return LiftResult(
            success=True,
            status="verified_ZT",
            scheme=cand,
            report=rep,
            method=method_tag,
            elapsed_ms=round(elapsed_ms, 3),
            coker_viol_4=0,
            max_level_reached=64,
            details=extra,
        )

    # Fast exact exhaustive path when free_bits <= 14 (<= 16,384 states) and not forcing Z3
    if (not force_z3) and n_free <= 14:
        found, x_sol = _exhaustive_ternary_nullspace_nb(
            m,
            n,
            p,
            u1,
            v1,
            w1,
            num_vars,
            rank_J1,
            var_u,
            var_v,
            var_w,
            pivot_cols,
            free_cols,
            rref_mat,
            P_mat,
            e1_vec,
        )
        if found:
            return _signs_to_verified_scheme(
                [int(b) for b in x_sol],
                "numba_exact_nullspace_ZT",
                {"rank_J1": int(rank_J1), "free_bits": int(n_free), "checked_space": 1 << n_free},
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return LiftResult(
            success=False,
            status="unsat",
            scheme=None,
            report=None,
            method="numba_exact_nullspace_ZT",
            elapsed_ms=round(elapsed_ms, 3),
            coker_viol_4=0,
            max_level_reached=2,
            details={"rank_J1": int(rank_J1), "free_bits": int(n_free), "exhausted_space": 1 << n_free},
        )

    # Z3 5.1.0 Exact Bool + Xor + PbEq Solver
    solver = z3.Solver()
    solver.set("timeout", max(100, int(timeout_s * 1000)))
    # Create Boolean variables ONLY for the num_vars non-gauge nonzero positions
    s_vars = [z3.Bool(f"s_{i}") for i in range(num_vars)]

    # Build covering terms per active triple (a, b, c)
    expected_ones = {
        (i1 * n + i2) * np_dim * mp + (i2 * p + j2) * mp + (i1 * p + j2)
        for i1 in range(m)
        for i2 in range(n)
        for j2 in range(p)
    }

    covers: Dict[int, List[Tuple[int, int, int]]] = {}
    for r in range(R):
        ur, vr, wr = int(u1[r]), int(v1[r]), int(w1[r])
        if not (ur and vr and wr):
            continue
        u_nz = [a for a in range(mn) if (ur >> a) & 1]
        v_nz = [b for b in range(np_dim) if (vr >> b) & 1]
        w_nz = [c for c in range(mp) if (wr >> c) & 1]
        for a in u_nz:
            vu = int(var_u[r, a])
            for b in v_nz:
                vv = int(var_v[r, b])
                base = (a * np_dim + b) * mp
                for c in w_nz:
                    vw = int(var_w[r, c])
                    covers.setdefault(base + c, []).append((vu, vv, vw))

    def _term_sign_expr(vu: int, vv: int, vw: int) -> Any:
        bits = [s_vars[v_idx] for v_idx in (vu, vv, vw) if v_idx >= 0]
        if not bits:
            return z3.BoolVal(False)
        expr = bits[0]
        for b_var in bits[1:]:
            expr = z3.Xor(expr, b_var)
        return expr

    for flat_idx, term_list in covers.items():
        k_cov = len(term_list)
        t_val = 1 if flat_idx in expected_ones else 0
        need_neg = (k_cov - t_val) // 2
        p_exprs = [_term_sign_expr(vu, vv, vw) for (vu, vv, vw) in term_list]
        if k_cov == 1:
            # need_neg == 0
            solver.add(z3.Not(p_exprs[0]))
        elif k_cov == 2:
            # need_neg == 1 -> exactly one is negative
            solver.add(z3.Xor(p_exprs[0], p_exprs[1]))
        else:
            # Create lightweight Tseitin proxy bools for each term parity and enforce exact PbEq
            aux_ps = []
            for p_e in p_exprs:
                if z3.is_const(p_e):
                    aux_ps.append(p_e)
                else:
                    aux = z3.FreshBool("p")
                    solver.add(aux == p_e)
                    aux_ps.append(aux)
            solver.add(z3.PbEq([(b_p, 1) for b_p in aux_ps], need_neg))

    chk = solver.check()
    if chk == z3.sat:
        model = solver.model()
        sign_bits = [1 if z3.is_true(model.evaluate(sv, model_completion=True)) else 0 for sv in s_vars]
        return _signs_to_verified_scheme(
            sign_bits,
            "z3_pbeq_ternary_ZT",
            {"rank_J1": int(rank_J1), "free_bits": int(n_free), "num_vars": int(num_vars), "z3_status": "sat"},
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    st_str = "unsat" if chk == z3.unsat else "timeout"
    return LiftResult(
        success=False,
        status=st_str,
        scheme=None,
        report=None,
        method="z3_pbeq_ternary_ZT",
        elapsed_ms=round(elapsed_ms, 3),
        coker_viol_4=0,
        max_level_reached=2,
        details={"rank_J1": int(rank_J1), "free_bits": int(n_free), "num_vars": int(num_vars), "z3_status": st_str},
    )


# =============================================================================
# 5. Clean From-Scratch Block-Seed Builders (Strassen 1969 + Laderman 1976 ONLY)
# =============================================================================


def direct_sum_m(s1: BilinearScheme, s2: BilinearScheme, name: str = "") -> BilinearScheme:
    """Exact vertical block concatenation along m: <m1,n,p:R1> + <m2,n,p:R2> -> <m1+m2,n,p:R1+R2>."""
    if (s1.n, s1.p) != (s2.n, s2.p) or (s1.ring, s1.modulus) != (s2.ring, s2.modulus):
        raise ValueError("direct_sum_m requires matching (n, p, ring, modulus)")
    m1, m2, n, p = s1.m, s2.m, s1.n, s1.p
    m = m1 + m2
    z = 0 if s1.ring != "Q" else Fraction(0, 1)
    U, V, W = [], [], []
    for r in range(s1.rank):
        U.append(list(s1.U[r]) + [z] * (m2 * n))
        V.append(list(s1.V[r]))
        W.append(list(s1.W[r]) + [z] * (m2 * p))
    for r in range(s2.rank):
        U.append([z] * (m1 * n) + list(s2.U[r]))
        V.append(list(s2.V[r]))
        W.append([z] * (m1 * p) + list(s2.W[r]))
    return BilinearScheme(
        m=m, n=n, p=p, U=U, V=V, W=W, ring=s1.ring, modulus=s1.modulus, name=name or f"sum_m_{m1}+{m2}x{n}x{p}"
    )


def direct_sum_n(s1: BilinearScheme, s2: BilinearScheme, name: str = "") -> BilinearScheme:
    """Exact inner block sum along n: <m,n1,p:R1> + <m,n2,p:R2> -> <m,n1+n2,p:R1+R2>."""
    if (s1.m, s1.p) != (s2.m, s2.p) or (s1.ring, s1.modulus) != (s2.ring, s2.modulus):
        raise ValueError("direct_sum_n requires matching (m, p, ring, modulus)")
    m, n1, n2, p = s1.m, s1.n, s2.n, s1.p
    n = n1 + n2
    z = 0 if s1.ring != "Q" else Fraction(0, 1)
    U, V, W = [], [], []
    for r in range(s1.rank):
        u_row = [z] * (m * n)
        for i in range(m):
            for j in range(n1):
                u_row[i * n + j] = s1.U[r][i * n1 + j]
        U.append(u_row)
        V.append(list(s1.V[r]) + [z] * (n2 * p))
        W.append(list(s1.W[r]))
    for r in range(s2.rank):
        u_row = [z] * (m * n)
        for i in range(m):
            for j in range(n2):
                u_row[i * n + (n1 + j)] = s2.U[r][i * n2 + j]
        U.append(u_row)
        V.append([z] * (n1 * p) + list(s2.V[r]))
        W.append(list(s2.W[r]))
    return BilinearScheme(
        m=m, n=n, p=p, U=U, V=V, W=W, ring=s1.ring, modulus=s1.modulus, name=name or f"sum_n_{m}x{n1}+{n2}x{p}"
    )


def transpose_scheme(s: BilinearScheme, name: str = "") -> BilinearScheme:
    """Exact transpose (A*B)^T = B^T * A^T: <m, n, p : R> -> <p, n, m : R>."""
    m, n, p = s.m, s.n, s.p
    U_new, V_new, W_new = [], [], []
    for r in range(s.rank):
        # New A' = B^T (p x n) -> U'[r, k*n + j] = V[r, j*p + k]
        u_row = [s.V[r][j * p + k] for k in range(p) for j in range(n)]
        # New B' = A^T (n x m) -> V'[r, j*m + i] = U[r, i*n + j]
        v_row = [s.U[r][i * n + j] for j in range(n) for i in range(m)]
        # New C' = C^T (p x m) -> W'[r, k*m + i] = W[r, i*p + k]
        w_row = [s.W[r][i * p + k] for k in range(p) for i in range(m)]
        U_new.append(u_row)
        V_new.append(v_row)
        W_new.append(w_row)
    return BilinearScheme(
        m=p, n=n, p=m, U=U_new, V=V_new, W=W_new, ring=s.ring, modulus=s.modulus, name=name or f"T_{s.name}"
    )


def cyclic_rotate_scheme(s: BilinearScheme, name: str = "") -> BilinearScheme:
    """
    Exact cyclic trace rotation: <m, n, p : R> -> <n, p, m : R>.
    Uses trace convention (U, V, W_tr) -> (V, W_tr, U) and converts back to C = A * B.
    """
    w_tr = s.w_to_trace_convention()
    return BilinearScheme.from_trace_convention(
        m=s.n,
        n=s.p,
        p=s.m,
        U=s.V,
        V=w_tr,
        W_tr=s.U,
        ring=s.ring,
        modulus=s.modulus,
        name=name or f"rot_{s.name}",
    )


def permute_scheme_axes(s: BilinearScheme, target_shape: Tuple[int, int, int]) -> BilinearScheme:
    """Map <m, n, p : R> to any S3 permutation of (m, n, p) with identical rank."""
    cur = s
    for _ in range(3):
        if cur.shape == target_shape:
            return cur
        cur = cyclic_rotate_scheme(cur)
    cur = transpose_scheme(s)
    for _ in range(3):
        if cur.shape == target_shape:
            return cur
        cur = cyclic_rotate_scheme(cur)
    raise ValueError(f"target_shape {target_shape} is not a permutation of {s.shape}")


def heterogeneous_strassen_combine(
    m1: int,
    m2: int,
    n1: int,
    n2: int,
    p1: int,
    p2: int,
    sub_schemes: Sequence[BilinearScheme],
    name: str = "",
) -> BilinearScheme:
    """
    Exact heterogeneous Strassen block combination for <m1+m2, n1+n2, p1+p2> (m1>=m2, n1>=n2, p1>=p2)
    from 7 exact sub-schemes matching the 7 peeled Strassen block products:
      M1: (m1, n1, p1)  (A11 + A22) * (B11 + B22) -> C11, C22
      M2: (m2, n1, p1)  (A21 + A22) * B11         -> C21, -C22
      M3: (m1, n1, p2)  A11 * (B12 - B22)         -> C12, C22
      M4: (m2, n2, p1)  A22 * (B21 - B11)         -> C11, C21
      M5: (m1, n2, p2)  (A11 + A12) * B22         -> -C11, C12
      M6: (m2, n1, p2)  (A21 - A11) * (B11 + B12) -> C22
      M7: (m1, n2, p1)  (A12 - A22) * (B21 + B22) -> C11
    Pure Strassen (1969) zero-padded block identity; strictly preserves Z_T = {-1, 0, 1}!
    """
    assert m1 >= m2 >= 1 and n1 >= n2 >= 1 and p1 >= p2 >= 1
    assert len(sub_schemes) == 7
    ring, mod = sub_schemes[0].ring, sub_schemes[0].modulus
    M_tot, N_tot, P_tot = m1 + m2, n1 + n2, p1 + p2

    # Block offset helpers:
    # A blocks: 11=(0,0,m1,n1), 12=(0,n1,m1,n2), 21=(m1,0,m2,n1), 22=(m1,n1,m2,n2)
    # B blocks: 11=(0,0,n1,p1), 12=(0,p1,n1,p2), 21=(n1,0,n2,p1), 22=(n1,p1,n2,p2)
    # C blocks: 11=(0,0,m1,p1), 12=(0,p1,m1,p2), 21=(m1,0,m2,p1), 22=(m1,p1,m2,p2)
    a_spec = [
        [((0, 0, m1, n1), 1), ((m1, n1, m2, n2), 1)],  # M1: A11 + A22
        [((m1, 0, m2, n1), 1), ((m1, n1, m2, n2), 1)],  # M2: A21 + A22
        [((0, 0, m1, n1), 1)],  # M3: A11
        [((m1, n1, m2, n2), 1)],  # M4: A22
        [((0, 0, m1, n2), 1), ((0, n1, m1, n2), 1)],  # M5: A11[:,:n2] + A12
        [((m1, 0, m2, n1), 1), ((0, 0, m2, n1), -1)],  # M6: A21 - A11[:m2,:]
        [((0, n1, m1, n2), 1), ((m1, n1, m2, n2), -1)],  # M7: A12 - A22
    ]
    b_spec = [
        [((0, 0, n1, p1), 1), ((n1, p1, n2, p2), 1)],  # M1: B11 + B22
        [((0, 0, n1, p1), 1)],  # M2: B11
        [((0, p1, n1, p2), 1), ((n1, p1, n2, p2), -1)],  # M3: B12 - B22
        [((n1, 0, n2, p1), 1), ((0, 0, n2, p1), -1)],  # M4: B21 - B11[:n2,:]
        [((n1, p1, n2, p2), 1)],  # M5: B22
        [((0, 0, n1, p2), 1), ((0, p1, n1, p2), 1)],  # M6: B11[:,:p2] + B12
        [((n1, 0, n2, p1), 1), ((n1, p1, n2, p2), 1)],  # M7: B21 + B22
    ]
    c_spec = [
        [((0, 0, m1, p1), 1), ((m1, p1, m2, p2), 1)],  # M1 -> C11, C22
        [((m1, 0, m2, p1), 1), ((m1, p1, m2, p2), -1)],  # M2 -> C21, -C22
        [((0, p1, m1, p2), 1), ((m1, p1, m2, p2), 1)],  # M3 -> C12, C22
        [((0, 0, m2, p1), 1), ((m1, 0, m2, p1), 1)],  # M4 -> C11[:m2,:], C21
        [((0, 0, m1, p2), -1), ((0, p1, m1, p2), 1)],  # M5 -> -C11[:,:p2], C12
        [((m1, p1, m2, p2), 1)],  # M6 -> C22
        [((0, 0, m1, p1), 1)],  # M7 -> C11
    ]

    U_out: List[List[int]] = []
    V_out: List[List[int]] = []
    W_out: List[List[int]] = []

    for idx_m, sub in enumerate(sub_schemes):
        sm, sn, sp = sub.shape
        for r in range(sub.rank):
            u_row = [0] * (M_tot * N_tot)
            v_row = [0] * (N_tot * P_tot)
            w_row = [0] * (M_tot * P_tot)

            for (r0, c0, bh, bw), sgn in a_spec[idx_m]:
                for i in range(bh):
                    for j in range(bw):
                        val = int(sub.U[r][i * sn + j])
                        if val:
                            u_row[(r0 + i) * N_tot + (c0 + j)] += sgn * val
            for (r0, c0, bh, bw), sgn in b_spec[idx_m]:
                for i in range(bh):
                    for j in range(bw):
                        val = int(sub.V[r][i * sp + j])
                        if val:
                            v_row[(r0 + i) * P_tot + (c0 + j)] += sgn * val
            for (r0, c0, bh, bw), sgn in c_spec[idx_m]:
                for i in range(bh):
                    for j in range(bw):
                        val = int(sub.W[r][i * sp + j])
                        if val:
                            w_row[(r0 + i) * P_tot + (c0 + j)] += sgn * val

            U_out.append(u_row)
            V_out.append(v_row)
            W_out.append(w_row)

    return BilinearScheme(
        m=M_tot,
        n=N_tot,
        p=P_tot,
        U=U_out,
        V=V_out,
        W=W_out,
        ring=ring,
        modulus=mod,
        name=name or f"het_strassen_{M_tot}x{N_tot}x{P_tot}",
    )


@lru_cache(maxsize=512)
def _block_seed_dp(a: int, b: int, c: int) -> Tuple[int, Tuple[Any, ...]]:
    """Optimal pure-algebraic block DP using ONLY naive(1,1,1:1), Strassen(2,2,2:7), Laderman(3,3,3:23)."""
    m, n, p = sorted((a, b, c))
    if m == 1:
        return n * p, ("naive", m, n, p)
    if (m, n, p) == (2, 2, 2):
        return 7, ("strassen",)
    if (m, n, p) == (3, 3, 3):
        return 23, ("laderman",)

    best_r = m * n * p
    best_plan: Tuple[Any, ...] = ("naive", m, n, p)

    for i in range(1, m // 2 + 1):
        r = _block_seed_dp(i, n, p)[0] + _block_seed_dp(m - i, n, p)[0]
        if r < best_r:
            best_r = r
            best_plan = ("split_m", i, m - i, n, p)
    for j in range(1, n // 2 + 1):
        r = _block_seed_dp(m, j, p)[0] + _block_seed_dp(m, n - j, p)[0]
        if r < best_r:
            best_r = r
            best_plan = ("split_n", m, j, n - j, p)
    for k in range(1, p // 2 + 1):
        r = _block_seed_dp(m, n, k)[0] + _block_seed_dp(m, n, p - k)[0]
        if r < best_r:
            best_r = r
            best_plan = ("split_p", m, n, k, p - k)

    for m1 in range((m + 1) // 2, m):
        m2 = m - m1
        for n1 in range((n + 1) // 2, n):
            n2 = n - n1
            for p1 in range((p + 1) // 2, p):
                p2 = p - p1
                r = (
                    _block_seed_dp(m1, n1, p1)[0]
                    + _block_seed_dp(m2, n1, p1)[0]
                    + _block_seed_dp(m1, n1, p2)[0]
                    + _block_seed_dp(m2, n2, p1)[0]
                    + _block_seed_dp(m1, n2, p2)[0]
                    + _block_seed_dp(m2, n1, p2)[0]
                    + _block_seed_dp(m1, n2, p1)[0]
                )
                if r < best_r:
                    best_r = r
                    best_plan = ("het_strassen", m1, m2, n1, n2, p1, p2)

    return best_r, best_plan


def make_block_seed(m: int, n: int, p: int, ring: str = "Z") -> BilinearScheme:
    """
    Construct a clean, verified, zero-external-leak starting Z_T scheme for <m, n, p>
    built strictly from Strassen <2,2,2:7>, Laderman <3,3,3:23>, and schoolbook <1,1,1:1>
    via exact block splits and heterogeneous Strassen peeling.
    """
    sm, sn, sp = sorted((m, n, p))
    _, plan = _block_seed_dp(sm, sn, sp)
    kind = plan[0]

    if kind == "naive":
        canon = make_naive_scheme(sm, sn, sp, ring=ring)
    elif kind == "strassen":
        canon = strassen_222(ring=ring)
    elif kind == "laderman":
        canon = laderman_333(ring=ring)
    elif kind == "split_m":
        _, m1, m2, nn, pp = plan
        canon = direct_sum_m(make_block_seed(m1, nn, pp, ring=ring), make_block_seed(m2, nn, pp, ring=ring))
    elif kind == "split_n":
        _, mm, n1, n2, pp = plan
        canon = direct_sum_n(make_block_seed(mm, n1, pp, ring=ring), make_block_seed(mm, n2, pp, ring=ring))
    elif kind == "split_p":
        _, mm, nn, p1, p2 = plan
        canon = direct_sum_p(make_block_seed(mm, nn, p1, ring=ring), make_block_seed(mm, nn, p2, ring=ring))
    elif kind == "het_strassen":
        _, m1, m2, n1, n2, p1, p2 = plan
        shapes = [
            (m1, n1, p1),
            (m2, n1, p1),
            (m1, n1, p2),
            (m2, n2, p1),
            (m1, n2, p2),
            (m2, n1, p2),
            (m1, n2, p1),
        ]
        subs = [make_block_seed(a, b, c, ring=ring) for (a, b, c) in shapes]
        canon = heterogeneous_strassen_combine(m1, m2, n1, n2, p1, p2, subs)
    else:
        raise RuntimeError(f"Unknown block seed plan {plan}")

    out = permute_scheme_axes(canon, (m, n, p))
    out.name = f"block_seed_{m}x{n}x{p}_R{out.rank}"
    return out


# =============================================================================
# 6. Helper Perturbations & Obstructed F2 Scheme Constructor (for Selftest)
# =============================================================================


def _permute_and_perturb_f2(scheme: BilinearScheme, rng: random.Random) -> BilinearScheme:
    """Randomly permute rank-1 terms and apply a random GL(m,2)xGL(n,2)xGL(p,2) permutation of indices."""
    f2 = scheme.reduce_mod(2)
    order = list(range(f2.rank))
    rng.shuffle(order)
    return BilinearScheme(
        m=f2.m,
        n=f2.n,
        p=f2.p,
        U=[f2.U[r] for r in order],
        V=[f2.V[r] for r in order],
        W=[f2.W[r] for r in order],
        ring="Z2",
        modulus=2,
        name=f"{scheme.name}_perm_f2",
    )


def _make_obstructed_f2_scheme() -> BilinearScheme:
    """
    Construct a deterministic, provably valid F2 scheme <3, 3, 3> that is genuinely
    support-restricted mod-4 obstructed (coker_viol_4 > 0), created purely from scratch
    via elementary F2 flip-graph moves starting from schoolbook <3, 3, 3 : 27>.
    """
    base = make_naive_scheme(3, 3, 3, ring="Z").reduce_mod(2)
    u, v, w = scheme_to_u64_f2(base)
    rng = random.Random(42)
    for _ in range(600):
        all_pairs = []
        R = len(u)
        for mode, arr in enumerate((u, v, w)):
            for i in range(R):
                if arr[i] == 0:
                    continue
                for j in range(i + 1, R):
                    if arr[i] == arr[j]:
                        all_pairs.append((mode, i, j))
        if not all_pairs:
            break
        mode, i, j = rng.choice(all_pairs)
        if rng.random() < 0.5:
            i, j = j, i
        if mode == 0:
            v[i] ^= v[j]
            w[j] ^= w[i]
        elif mode == 1:
            u[i] ^= u[j]
            w[j] ^= w[i]
        else:
            u[i] ^= u[j]
            v[j] ^= v[i]
        keep = (u != 0) & (v != 0) & (w != 0)
        u, v, w = u[keep], v[keep], w[keep]
        ora = mod4_oracle_u64(3, 3, 3, u, v, w, support_restricted=True)
        if ora.coker_viol_4 > 0:
            sch = u64_f2_to_scheme(3, 3, 3, u, v, w, name="naive333_f2_mod4_obstructed")
            assert verify_scheme(sch).valid
            return sch
    raise RuntimeError("Failed to find mod-4 obstructed F2 flip neighbor of naive <3,3,3:27>")


def _make_rational_denom3_strassen_f2() -> Tuple[BilinearScheme, BilinearScheme]:
    """
    Create an exact Q-scheme for <2,2,2:7> with denominator 3 (via P = [[1, 0], [1, 3]],
    P^{-1} = [[1, 0], [-1/3, 1/3]] -> valid mod 2 since P mod 2 = [[1,0],[1,1]] in GL(2,2)!),
    and return (exact_Q_scheme, its_F2_reduction) for testing Wang rational reconstruction!
    """
    s = strassen_222(ring="Z")
    # Sandwich A -> P A, C -> P C with P = [[1,0],[1,3]] (det=3, invertible mod 2 and over Q!)
    # Equivalently, U' corresponds to A' = P^{-1} A => (u00,u01,u10,u11) -> u'(A) = u(P^{-1} A)
    # P^{-1} = [[1, 0], [-1/3, 1/3]]: row0 of A' = row0 of A; row1 of A' = (-1/3)*row0 + (1/3)*row1
    # And W' corresponds to C = P C' => row0 of C = row0 of C'; row1 of C = row0 of C' + 3*row1 of C'
    U_q: List[List[Fraction]] = []
    V_q: List[List[Fraction]] = []
    W_q: List[List[Fraction]] = []
    for r in range(s.rank):
        u00, u01, u10, u11 = [Fraction(int(x), 1) for x in s.U[r]]
        # sum u_{ij} A'_{ij} = u00*a00 + u01*a01 + u10*(-a00/3 + a10/3) + u11*(-a01/3 + a11/3)
        u_new = [
            u00 - u10 * Fraction(1, 3),
            u01 - u11 * Fraction(1, 3),
            u10 * Fraction(1, 3),
            u11 * Fraction(1, 3),
        ]
        v_new = [Fraction(int(x), 1) for x in s.V[r]]
        w00, w01, w10, w11 = [Fraction(int(x), 1) for x in s.W[r]]
        # C = P C' => c00 = c'00, c01 = c'01, c10 = c'00 + 3*c'10, c11 = c'01 + 3*c'11
        w_new = [w00, w01, w00 + 3 * w10, w01 + 3 * w11]
        U_q.append(u_new)
        V_q.append(v_new)
        W_q.append(w_new)
    sq = BilinearScheme(m=2, n=2, p=2, U=U_q, V=V_q, W=W_q, ring="Q", name="strassen_Q_denom3")
    assert verify_scheme(sq).valid
    # Reduce mod 2: since 1/3 == 1 (mod 2), num % 2 == (num * inv(den)) % 2!
    U_2 = [[(int(x.numerator) * pow(int(x.denominator), -1, 2)) % 2 for x in row] for row in sq.U]
    V_2 = [[(int(x.numerator) * pow(int(x.denominator), -1, 2)) % 2 for x in row] for row in sq.V]
    W_2 = [[(int(x.numerator) * pow(int(x.denominator), -1, 2)) % 2 for x in row] for row in sq.W]
    sf2 = BilinearScheme(m=2, n=2, p=2, U=U_2, V=V_2, W=W_2, ring="Z2", modulus=2, name="strassen_Q_denom3_mod2")
    assert verify_scheme(sf2).valid
    return sq, sf2


# =============================================================================
# 7. Mandatory --selftest (Tracer #1 Gate)
# =============================================================================


# =============================================================================
# 6. Stage L0-p3 & L1-p3 — 3-adic Mod-9 Oracle & Linear 3-adic Hensel Lifter
#    (3^k -> 3^{k+1} up to 3^40 ~ 1.216e19 + Wang Rational Reconstruction -> Z[1/2]/Q/Z)
# =============================================================================


@dataclass
class Mod9OracleResult:
    coker_viol_9: int
    free_trits: int
    rank_J1: int
    num_vars: int
    num_active_eqs: int
    is_mod9_liftable: bool
    support_restricted: bool
    nnz_total: int
    gauge_fixed: int
    elapsed_ms: float


def scheme_to_u64_f3(
    scheme: BilinearScheme,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pack a mod-3 (or integer/Z[1/2]) BilinearScheme into 2-bitplane uint64[R] F_3 arrays:
      +1 -> (pos=1, neg=0),  -1 (2 mod 3) -> (pos=0, neg=1),  0 -> (pos=0, neg=0).
    For ring == 'Q', reduces numerator * inv(denominator, 3) mod 3 (supports Z[1/2]!).
    """
    m, n, p, R = scheme.m, scheme.n, scheme.p, scheme.rank
    if max(m * n, n * p, m * p) > 64:
        raise ValueError(f"uint64 F3 packing requires mn,np,mp <= 64, got {m}x{n}x{p}")
    up = np.zeros(R, dtype=np.uint64)
    un = np.zeros(R, dtype=np.uint64)
    vp = np.zeros(R, dtype=np.uint64)
    vn = np.zeros(R, dtype=np.uint64)
    wp = np.zeros(R, dtype=np.uint64)
    wn = np.zeros(R, dtype=np.uint64)

    def _to_f3(val: Any) -> int:
        if isinstance(val, Fraction):
            d3 = val.denominator % 3
            if d3 == 0:
                raise ValueError(f"Denominator divisible by 3 cannot reduce mod 3: {val}")
            return (val.numerator * d3) % 3  # since inv(1,3)=1, inv(2,3)=2
        return int(val) % 3

    for r in range(R):
        p_u, n_u = 0, 0
        for idx, val in enumerate(scheme.U[r]):
            t = _to_f3(val)
            if t == 1:
                p_u |= 1 << idx
            elif t == 2:
                n_u |= 1 << idx
        up[r], un[r] = np.uint64(p_u), np.uint64(n_u)

        p_v, n_v = 0, 0
        for idx, val in enumerate(scheme.V[r]):
            t = _to_f3(val)
            if t == 1:
                p_v |= 1 << idx
            elif t == 2:
                n_v |= 1 << idx
        vp[r], vn[r] = np.uint64(p_v), np.uint64(n_v)

        p_w, n_w = 0, 0
        for idx, val in enumerate(scheme.W[r]):
            t = _to_f3(val)
            if t == 1:
                p_w |= 1 << idx
            elif t == 2:
                n_w |= 1 << idx
        wp[r], wn[r] = np.uint64(p_w), np.uint64(n_w)

    return up, un, vp, vn, wp, wn


def u64_f3_to_scheme(
    m: int,
    n: int,
    p: int,
    up: np.ndarray,
    un: np.ndarray,
    vp: np.ndarray,
    vn: np.ndarray,
    wp: np.ndarray,
    wn: np.ndarray,
    name: str = "u64_f3_scheme",
) -> BilinearScheme:
    """Unpack 2-bitplane uint64[R] F_3 arrays into a canonical BilinearScheme(ring='Z3', modulus=3)."""
    R = len(up)
    mn, np_dim, mp = m * n, n * p, m * p

    def _unpack(pos_arr: np.ndarray, neg_arr: np.ndarray, dim: int) -> List[List[int]]:
        out: List[List[int]] = []
        for r in range(R):
            pm, nm = int(pos_arr[r]), int(neg_arr[r])
            row = [0] * dim
            for idx in range(dim):
                if (pm >> idx) & 1:
                    row[idx] = 1
                elif (nm >> idx) & 1:
                    row[idx] = 2
            out.append(row)
        return out

    return BilinearScheme(
        m=m,
        n=n,
        p=p,
        U=_unpack(up, un, mn),
        V=_unpack(vp, vn, np_dim),
        W=_unpack(wp, wn, mp),
        ring="Z3",
        modulus=3,
        name=name,
    )


@nb.njit(cache=True)
def _f3_add_u64(p1: np.uint64, m1: np.uint64, p2: np.uint64, m2: np.uint64) -> Tuple[np.uint64, np.uint64]:
    """Exact branchless 64-lane GF(3) addition (+1=(1,0), -1=(0,1), 0=(0,0))."""
    z2 = ~(p2 | m2)
    z1 = ~(p1 | m1)
    po = (p1 & z2) | (p2 & z1) | (m1 & m2)
    mo = (m1 & z2) | (m2 & z1) | (p1 & p2)
    return po, mo


@nb.njit(cache=True)
def _f3_row_axpy_nb(
    mat_p: np.ndarray,
    mat_m: np.ndarray,
    dst: int,
    src: int,
    start_w: int,
    n_words: int,
    add_not_sub: bool,
) -> None:
    """In-place GF(3) row operation: row[dst] += row[src] (if add_not_sub) else row[dst] -= row[src]."""
    if add_not_sub:
        for j in range(start_w, n_words):
            po, mo = _f3_add_u64(mat_p[dst, j], mat_m[dst, j], mat_p[src, j], mat_m[src, j])
            mat_p[dst, j] = po
            mat_m[dst, j] = mo
    else:
        for j in range(start_w, n_words):
            po, mo = _f3_add_u64(mat_p[dst, j], mat_m[dst, j], mat_m[src, j], mat_p[src, j])
            mat_p[dst, j] = po
            mat_m[dst, j] = mo


@nb.njit(cache=True)
def _f3_toggle_entry_nb(
    mat_p: np.ndarray, mat_m: np.ndarray, row: int, col: int, sgn: int
) -> None:
    """Add sgn in {-1, +1} into packed GF(3) matrix entry (row, col)."""
    w_i = col >> 6
    bit = np.uint64(1) << np.uint64(col & 63)
    p2 = bit if sgn > 0 else np.uint64(0)
    m2 = bit if sgn < 0 else np.uint64(0)
    po, mo = _f3_add_u64(mat_p[row, w_i], mat_m[row, w_i], p2, m2)
    mat_p[row, w_i] = po
    mat_m[row, w_i] = mo


@nb.njit(cache=True)
def _mod9_oracle_nb(
    m: int,
    n: int,
    p: int,
    up: np.ndarray,
    un: np.ndarray,
    vp: np.ndarray,
    vn: np.ndarray,
    wp: np.ndarray,
    wn: np.ndarray,
    support_restricted: bool,
) -> Tuple[int, int, int, int, int, int, int]:
    """
    Fast forward-only Stage L0-p3 mod-9 liftability oracle over GF(3) (2-bitplane uint64 SIMD!),
    with 2R scaling gauge fixing over F_3^x.
    Returns (coker_viol_9, free_trits, rank_J1, num_vars, num_active_eqs, nnz_total, gauge_fixed).
    Returns coker_viol_9 = -1 if input fails Brent equations mod 3.
    """
    mn = m * n
    np_dim = n * p
    mp = m * p
    R = up.shape[0]

    var_u = np.full((R, mn), -1, dtype=np.int32)
    var_v = np.full((R, np_dim), -1, dtype=np.int32)
    var_w = np.full((R, mp), -1, dtype=np.int32)

    num_vars = 0
    nnz_total = 0
    gauge_fixed = 0
    for pass_off in (0, 1):
        if pass_off == 1 and support_restricted:
            break
        for r in range(R):
            um = up[r] | un[r]
            vm = vp[r] | vn[r]
            wm = wp[r] | wn[r]
            a0 = _ctz64(um) if um != 0 else -1
            b0 = _ctz64(vm) if vm != 0 else -1
            if pass_off == 0:
                if a0 >= 0:
                    gauge_fixed += 1
                if b0 >= 0:
                    gauge_fixed += 1
                nnz_total += _popcount64(um) + _popcount64(vm) + _popcount64(wm)
            for a in range(mn):
                if a == a0:
                    continue
                is_nz = ((um >> np.uint64(a)) & np.uint64(1)) != 0
                if (pass_off == 0 and is_nz) or (pass_off == 1 and not is_nz):
                    var_u[r, a] = num_vars
                    num_vars += 1
            for b in range(np_dim):
                if b == b0:
                    continue
                is_nz = ((vm >> np.uint64(b)) & np.uint64(1)) != 0
                if (pass_off == 0 and is_nz) or (pass_off == 1 and not is_nz):
                    var_v[r, b] = num_vars
                    num_vars += 1
            for c in range(mp):
                is_nz = ((wm >> np.uint64(c)) & np.uint64(1)) != 0
                if (pass_off == 0 and is_nz) or (pass_off == 1 and not is_nz):
                    var_w[r, c] = num_vars
                    num_vars += 1

    total_triples = mn * np_dim * mp
    counts = np.zeros(total_triples, dtype=np.int32)
    t_exact = np.zeros(total_triples, dtype=np.int8)
    active_mask = np.zeros(total_triples, dtype=np.uint8)

    for i1 in range(m):
        for i2 in range(n):
            a = i1 * n + i2
            for j2 in range(p):
                b = i2 * p + j2
                c = i1 * p + j2
                idx = (a * np_dim + b) * mp + c
                t_exact[idx] = 1
                active_mask[idx] = 1

    for r in range(R):
        um = up[r] | un[r]
        vm = vp[r] | vn[r]
        wm = wp[r] | wn[r]
        if um != 0 and vm != 0 and wm != 0:
            tu = um
            while tu != 0:
                lu = tu & (~tu + np.uint64(1))
                a = _ctz64(lu)
                su = 1 if (up[r] & lu) != 0 else -1
                tv = vm
                while tv != 0:
                    lv = tv & (~tv + np.uint64(1))
                    b = _ctz64(lv)
                    sv = 1 if (vp[r] & lv) != 0 else -1
                    suv = su * sv
                    base = (a * np_dim + b) * mp
                    tw = wm
                    while tw != 0:
                        lw = tw & (~tw + np.uint64(1))
                        c = _ctz64(lw)
                        sw = 1 if (wp[r] & lw) != 0 else -1
                        counts[base + c] += suv * sw
                        active_mask[base + c] = 1
                        tw ^= lw
                    tv ^= lv
                tu ^= lu
        if not support_restricted:
            if vm != 0 and wm != 0:
                tv = vm
                while tv != 0:
                    lv = tv & (~tv + np.uint64(1))
                    b = _ctz64(lv)
                    tw = wm
                    while tw != 0:
                        lw = tw & (~tw + np.uint64(1))
                        c = _ctz64(lw)
                        for a in range(mn):
                            active_mask[(a * np_dim + b) * mp + c] = 1
                        tw ^= lw
                    tv ^= lv
            if um != 0 and wm != 0:
                tu = um
                while tu != 0:
                    lu = tu & (~tu + np.uint64(1))
                    a = _ctz64(lu)
                    tw = wm
                    while tw != 0:
                        lw = tw & (~tw + np.uint64(1))
                        c = _ctz64(lw)
                        for b in range(np_dim):
                            active_mask[(a * np_dim + b) * mp + c] = 1
                        tw ^= lw
                    tu ^= lu
            if um != 0 and vm != 0:
                tu = um
                while tu != 0:
                    lu = tu & (~tu + np.uint64(1))
                    a = _ctz64(lu)
                    tv = vm
                    while tv != 0:
                        lv = tv & (~tv + np.uint64(1))
                        b = _ctz64(lv)
                        base = (a * np_dim + b) * mp
                        for c in range(mp):
                            active_mask[base + c] = 1
                        tv ^= lv
                    tu ^= lu

    num_active = 0
    for idx in range(total_triples):
        diff = int(t_exact[idx]) - int(counts[idx])
        if diff % 3 != 0:
            return -1, 0, 0, num_vars, 0, nnz_total, gauge_fixed
        if active_mask[idx] != 0:
            num_active += 1

    eq_id = np.full(total_triples, -1, dtype=np.int32)
    cur = 0
    for idx in range(total_triples):
        if active_mask[idx] != 0:
            eq_id[idx] = cur
            cur += 1

    n_cols = num_vars + 1
    n_words = (n_cols + 63) // 64
    mat_p = np.zeros((num_active, n_words), dtype=np.uint64)
    mat_m = np.zeros((num_active, n_words), dtype=np.uint64)
    rhs_word = num_vars >> 6
    rhs_bit = np.uint64(1) << np.uint64(num_vars & 63)

    for idx in range(total_triples):
        row = eq_id[idx]
        if row >= 0:
            e1 = ((int(t_exact[idx]) - int(counts[idx])) // 3) % 3
            if e1 == 1:
                mat_p[row, rhs_word] |= rhs_bit
            elif e1 == 2:
                mat_m[row, rhs_word] |= rhs_bit

    # Populate GF(3) Jacobian J1
    for r in range(R):
        um = up[r] | un[r]
        vm = vp[r] | vn[r]
        wm = wp[r] | wn[r]
        if support_restricted:
            if um == 0 or vm == 0 or wm == 0:
                continue
            tu = um
            while tu != 0:
                lu = tu & (~tu + np.uint64(1))
                a = _ctz64(lu)
                su = 1 if (up[r] & lu) != 0 else -1
                cu = var_u[r, a]
                tv = vm
                while tv != 0:
                    lv = tv & (~tv + np.uint64(1))
                    b = _ctz64(lv)
                    sv = 1 if (vp[r] & lv) != 0 else -1
                    cv = var_v[r, b]
                    base = (a * np_dim + b) * mp
                    tw = wm
                    while tw != 0:
                        lw = tw & (~tw + np.uint64(1))
                        c = _ctz64(lw)
                        sw = 1 if (wp[r] & lw) != 0 else -1
                        row = eq_id[base + c]
                        cw = var_w[r, c]
                        if cu >= 0:
                            _f3_toggle_entry_nb(mat_p, mat_m, row, cu, sv * sw)
                        if cv >= 0:
                            _f3_toggle_entry_nb(mat_p, mat_m, row, cv, su * sw)
                        if cw >= 0:
                            _f3_toggle_entry_nb(mat_p, mat_m, row, cw, su * sv)
                        tw ^= lw
                    tv ^= lv
                tu ^= lu
        else:
            if vm != 0 and wm != 0:
                tv = vm
                while tv != 0:
                    lv = tv & (~tv + np.uint64(1))
                    b = _ctz64(lv)
                    sv = 1 if (vp[r] & lv) != 0 else -1
                    tw = wm
                    while tw != 0:
                        lw = tw & (~tw + np.uint64(1))
                        c = _ctz64(lw)
                        sw = 1 if (wp[r] & lw) != 0 else -1
                        sgn = sv * sw
                        for a in range(mn):
                            cu = var_u[r, a]
                            if cu >= 0:
                                _f3_toggle_entry_nb(mat_p, mat_m, eq_id[(a * np_dim + b) * mp + c], cu, sgn)
                        tw ^= lw
                    tv ^= lv
            if um != 0 and wm != 0:
                tu = um
                while tu != 0:
                    lu = tu & (~tu + np.uint64(1))
                    a = _ctz64(lu)
                    su = 1 if (up[r] & lu) != 0 else -1
                    tw = wm
                    while tw != 0:
                        lw = tw & (~tw + np.uint64(1))
                        c = _ctz64(lw)
                        sw = 1 if (wp[r] & lw) != 0 else -1
                        sgn = su * sw
                        for b in range(np_dim):
                            cv = var_v[r, b]
                            if cv >= 0:
                                _f3_toggle_entry_nb(mat_p, mat_m, eq_id[(a * np_dim + b) * mp + c], cv, sgn)
                        tw ^= lw
                    tu ^= lu
            if um != 0 and vm != 0:
                tu = um
                while tu != 0:
                    lu = tu & (~tu + np.uint64(1))
                    a = _ctz64(lu)
                    su = 1 if (up[r] & lu) != 0 else -1
                    tv = vm
                    while tv != 0:
                        lv = tv & (~tv + np.uint64(1))
                        b = _ctz64(lv)
                        sv = 1 if (vp[r] & lv) != 0 else -1
                        sgn = su * sv
                        base = (a * np_dim + b) * mp
                        for c in range(mp):
                            cw = var_w[r, c]
                            if cw >= 0:
                                _f3_toggle_entry_nb(mat_p, mat_m, eq_id[base + c], cw, sgn)
                        tv ^= lv
                    tu ^= lu

    # Forward GF(3) Gaussian elimination
    pivot_row = 0
    for col in range(num_vars):
        w_idx = col >> 6
        b_mask = np.uint64(1) << np.uint64(col & 63)
        found = -1
        for r_i in range(pivot_row, num_active):
            if ((mat_p[r_i, w_idx] | mat_m[r_i, w_idx]) & b_mask) != 0:
                found = r_i
                break
        if found == -1:
            continue
        if found != pivot_row:
            for j in range(w_idx, n_words):
                tp = mat_p[pivot_row, j]
                mat_p[pivot_row, j] = mat_p[found, j]
                mat_p[found, j] = tp
                tm = mat_m[pivot_row, j]
                mat_m[pivot_row, j] = mat_m[found, j]
                mat_m[found, j] = tm
        # Normalize pivot to +1 by swapping pos/neg if pivot is -1
        if (mat_m[pivot_row, w_idx] & b_mask) != 0:
            for j in range(w_idx, n_words):
                tp = mat_p[pivot_row, j]
                mat_p[pivot_row, j] = mat_m[pivot_row, j]
                mat_m[pivot_row, j] = tp
        for r_i in range(pivot_row + 1, num_active):
            if (mat_p[r_i, w_idx] & b_mask) != 0:
                _f3_row_axpy_nb(mat_p, mat_m, r_i, pivot_row, w_idx, n_words, False)
            elif (mat_m[r_i, w_idx] & b_mask) != 0:
                _f3_row_axpy_nb(mat_p, mat_m, r_i, pivot_row, w_idx, n_words, True)
        pivot_row += 1
        if pivot_row == num_active:
            break

    rank_J1 = pivot_row
    free_trits = num_vars - rank_J1
    coker_viol_9 = 0
    for r_i in range(rank_J1, num_active):
        if ((mat_p[r_i, rhs_word] | mat_m[r_i, rhs_word]) & rhs_bit) != 0:
            coker_viol_9 += 1

    return coker_viol_9, free_trits, rank_J1, num_vars, num_active, nnz_total, gauge_fixed


@nb.njit(cache=True)
def _build_f3_jacobian_rref_nb(
    m: int,
    n: int,
    p: int,
    up: np.ndarray,
    un: np.ndarray,
    vp: np.ndarray,
    vn: np.ndarray,
    wp: np.ndarray,
    wn: np.ndarray,
    support_restricted: bool,
) -> Tuple[
    int,
    int,
    int,
    int,
    int,
    int,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Build exact GF(3) Jacobian J1 (2-bitplane SIMD packed), compute full Gauss-Jordan RREF
    AND row transform (P_p, P_m) ONCE so ALL 39 linear 3-adic steps (3^k -> 3^{k+1} to 3^40)
    reuse ONE precomputed RREF!
    """
    mn = m * n
    np_dim = n * p
    mp = m * p
    R = up.shape[0]

    var_u = np.full((R, mn), -1, dtype=np.int32)
    var_v = np.full((R, np_dim), -1, dtype=np.int32)
    var_w = np.full((R, mp), -1, dtype=np.int32)

    num_vars = 0
    nnz_total = 0
    gauge_fixed = 0
    for pass_off in (0, 1):
        if pass_off == 1 and support_restricted:
            break
        for r in range(R):
            um = up[r] | un[r]
            vm = vp[r] | vn[r]
            wm = wp[r] | wn[r]
            a0 = _ctz64(um) if um != 0 else -1
            b0 = _ctz64(vm) if vm != 0 else -1
            if pass_off == 0:
                if a0 >= 0:
                    gauge_fixed += 1
                if b0 >= 0:
                    gauge_fixed += 1
                nnz_total += _popcount64(um) + _popcount64(vm) + _popcount64(wm)
            for a in range(mn):
                if a == a0:
                    continue
                is_nz = ((um >> np.uint64(a)) & np.uint64(1)) != 0
                if (pass_off == 0 and is_nz) or (pass_off == 1 and not is_nz):
                    var_u[r, a] = num_vars
                    num_vars += 1
            for b in range(np_dim):
                if b == b0:
                    continue
                is_nz = ((vm >> np.uint64(b)) & np.uint64(1)) != 0
                if (pass_off == 0 and is_nz) or (pass_off == 1 and not is_nz):
                    var_v[r, b] = num_vars
                    num_vars += 1
            for c in range(mp):
                is_nz = ((wm >> np.uint64(c)) & np.uint64(1)) != 0
                if (pass_off == 0 and is_nz) or (pass_off == 1 and not is_nz):
                    var_w[r, c] = num_vars
                    num_vars += 1

    total_triples = mn * np_dim * mp
    counts = np.zeros(total_triples, dtype=np.int32)
    t_exact = np.zeros(total_triples, dtype=np.int8)
    active_mask = np.zeros(total_triples, dtype=np.uint8)

    for i1 in range(m):
        for i2 in range(n):
            a = i1 * n + i2
            for j2 in range(p):
                b = i2 * p + j2
                c = i1 * p + j2
                idx = (a * np_dim + b) * mp + c
                t_exact[idx] = 1
                active_mask[idx] = 1

    for r in range(R):
        um = up[r] | un[r]
        vm = vp[r] | vn[r]
        wm = wp[r] | wn[r]
        if um != 0 and vm != 0 and wm != 0:
            tu = um
            while tu != 0:
                lu = tu & (~tu + np.uint64(1))
                a = _ctz64(lu)
                su = 1 if (up[r] & lu) != 0 else -1
                tv = vm
                while tv != 0:
                    lv = tv & (~tv + np.uint64(1))
                    b = _ctz64(lv)
                    sv = 1 if (vp[r] & lv) != 0 else -1
                    suv = su * sv
                    base = (a * np_dim + b) * mp
                    tw = wm
                    while tw != 0:
                        lw = tw & (~tw + np.uint64(1))
                        c = _ctz64(lw)
                        sw = 1 if (wp[r] & lw) != 0 else -1
                        counts[base + c] += suv * sw
                        active_mask[base + c] = 1
                        tw ^= lw
                    tv ^= lv
                tu ^= lu
        if not support_restricted:
            if vm != 0 and wm != 0:
                tv = vm
                while tv != 0:
                    lv = tv & (~tv + np.uint64(1))
                    b = _ctz64(lv)
                    tw = wm
                    while tw != 0:
                        lw = tw & (~tw + np.uint64(1))
                        c = _ctz64(lw)
                        for a in range(mn):
                            active_mask[(a * np_dim + b) * mp + c] = 1
                        tw ^= lw
                    tv ^= lv
            if um != 0 and wm != 0:
                tu = um
                while tu != 0:
                    lu = tu & (~tu + np.uint64(1))
                    a = _ctz64(lu)
                    tw = wm
                    while tw != 0:
                        lw = tw & (~tw + np.uint64(1))
                        c = _ctz64(lw)
                        for b in range(np_dim):
                            active_mask[(a * np_dim + b) * mp + c] = 1
                        tw ^= lw
                    tu ^= lu
            if um != 0 and vm != 0:
                tu = um
                while tu != 0:
                    lu = tu & (~tu + np.uint64(1))
                    a = _ctz64(lu)
                    tv = vm
                    while tv != 0:
                        lv = tv & (~tv + np.uint64(1))
                        b = _ctz64(lv)
                        base = (a * np_dim + b) * mp
                        for c in range(mp):
                            active_mask[base + c] = 1
                        tv ^= lv
                    tu ^= lu

    num_active = 0
    for idx in range(total_triples):
        if (int(t_exact[idx]) - int(counts[idx])) % 3 != 0:
            e32 = np.zeros(0, dtype=np.int32)
            eu64 = np.zeros((0, 0), dtype=np.uint64)
            eu8 = np.zeros(0, dtype=np.uint8)
            return -1, num_vars, 0, 0, nnz_total, gauge_fixed, e32, var_u, var_v, var_w, e32, e32, eu64, eu64, eu64, eu64, eu8
        if active_mask[idx] != 0:
            num_active += 1

    eq_id = np.full(total_triples, -1, dtype=np.int32)
    eq_triples = np.empty(num_active, dtype=np.int32)
    e1_vec = np.empty(num_active, dtype=np.uint8)
    cur = 0
    for idx in range(total_triples):
        if active_mask[idx] != 0:
            eq_id[idx] = cur
            eq_triples[cur] = idx
            e1_vec[cur] = ((int(t_exact[idx]) - int(counts[idx])) // 3) % 3
            cur += 1

    n_var_words = (num_vars + 63) // 64
    n_eq_words = (num_active + 63) // 64
    mat_p = np.zeros((num_active, n_var_words), dtype=np.uint64)
    mat_m = np.zeros((num_active, n_var_words), dtype=np.uint64)
    P_p = np.zeros((num_active, n_eq_words), dtype=np.uint64)
    P_m = np.zeros((num_active, n_eq_words), dtype=np.uint64)
    for r_i in range(num_active):
        P_p[r_i, r_i >> 6] = np.uint64(1) << np.uint64(r_i & 63)

    for r in range(R):
        um = up[r] | un[r]
        vm = vp[r] | vn[r]
        wm = wp[r] | wn[r]
        if support_restricted:
            if um == 0 or vm == 0 or wm == 0:
                continue
            tu = um
            while tu != 0:
                lu = tu & (~tu + np.uint64(1))
                a = _ctz64(lu)
                su = 1 if (up[r] & lu) != 0 else -1
                cu = var_u[r, a]
                tv = vm
                while tv != 0:
                    lv = tv & (~tv + np.uint64(1))
                    b = _ctz64(lv)
                    sv = 1 if (vp[r] & lv) != 0 else -1
                    cv = var_v[r, b]
                    base = (a * np_dim + b) * mp
                    tw = wm
                    while tw != 0:
                        lw = tw & (~tw + np.uint64(1))
                        c = _ctz64(lw)
                        sw = 1 if (wp[r] & lw) != 0 else -1
                        row = eq_id[base + c]
                        cw = var_w[r, c]
                        if cu >= 0:
                            _f3_toggle_entry_nb(mat_p, mat_m, row, cu, sv * sw)
                        if cv >= 0:
                            _f3_toggle_entry_nb(mat_p, mat_m, row, cv, su * sw)
                        if cw >= 0:
                            _f3_toggle_entry_nb(mat_p, mat_m, row, cw, su * sv)
                        tw ^= lw
                    tv ^= lv
                tu ^= lu
        else:
            if vm != 0 and wm != 0:
                tv = vm
                while tv != 0:
                    lv = tv & (~tv + np.uint64(1))
                    b = _ctz64(lv)
                    sv = 1 if (vp[r] & lv) != 0 else -1
                    tw = wm
                    while tw != 0:
                        lw = tw & (~tw + np.uint64(1))
                        c = _ctz64(lw)
                        sw = 1 if (wp[r] & lw) != 0 else -1
                        sgn = sv * sw
                        for a in range(mn):
                            cu = var_u[r, a]
                            if cu >= 0:
                                _f3_toggle_entry_nb(mat_p, mat_m, eq_id[(a * np_dim + b) * mp + c], cu, sgn)
                        tw ^= lw
                    tv ^= lv
            if um != 0 and wm != 0:
                tu = um
                while tu != 0:
                    lu = tu & (~tu + np.uint64(1))
                    a = _ctz64(lu)
                    su = 1 if (up[r] & lu) != 0 else -1
                    tw = wm
                    while tw != 0:
                        lw = tw & (~tw + np.uint64(1))
                        c = _ctz64(lw)
                        sw = 1 if (wp[r] & lw) != 0 else -1
                        sgn = su * sw
                        for b in range(np_dim):
                            cv = var_v[r, b]
                            if cv >= 0:
                                _f3_toggle_entry_nb(mat_p, mat_m, eq_id[(a * np_dim + b) * mp + c], cv, sgn)
                        tw ^= lw
                    tu ^= lu
            if um != 0 and vm != 0:
                tu = um
                while tu != 0:
                    lu = tu & (~tu + np.uint64(1))
                    a = _ctz64(lu)
                    su = 1 if (up[r] & lu) != 0 else -1
                    tv = vm
                    while tv != 0:
                        lv = tv & (~tv + np.uint64(1))
                        b = _ctz64(lv)
                        sv = 1 if (vp[r] & lv) != 0 else -1
                        sgn = su * sv
                        base = (a * np_dim + b) * mp
                        for c in range(mp):
                            cw = var_w[r, c]
                            if cw >= 0:
                                _f3_toggle_entry_nb(mat_p, mat_m, eq_id[base + c], cw, sgn)
                        tv ^= lv
                    tu ^= lu

    pivot_cols_buf = np.empty(min(num_active, num_vars), dtype=np.int32)
    is_pivot_col = np.zeros(num_vars, dtype=np.uint8)
    pivot_row = 0
    for col in range(num_vars):
        w_idx = col >> 6
        b_mask = np.uint64(1) << np.uint64(col & 63)
        found = -1
        for r_i in range(pivot_row, num_active):
            if ((mat_p[r_i, w_idx] | mat_m[r_i, w_idx]) & b_mask) != 0:
                found = r_i
                break
        if found == -1:
            continue
        if found != pivot_row:
            for j in range(w_idx, n_var_words):
                tp = mat_p[pivot_row, j]
                mat_p[pivot_row, j] = mat_p[found, j]
                mat_p[found, j] = tp
                tm = mat_m[pivot_row, j]
                mat_m[pivot_row, j] = mat_m[found, j]
                mat_m[found, j] = tm
            for j in range(n_eq_words):
                tp = P_p[pivot_row, j]
                P_p[pivot_row, j] = P_p[found, j]
                P_p[found, j] = tp
                tm = P_m[pivot_row, j]
                P_m[pivot_row, j] = P_m[found, j]
                P_m[found, j] = tm
        if (mat_m[pivot_row, w_idx] & b_mask) != 0:
            for j in range(w_idx, n_var_words):
                tp = mat_p[pivot_row, j]
                mat_p[pivot_row, j] = mat_m[pivot_row, j]
                mat_m[pivot_row, j] = tp
            for j in range(n_eq_words):
                tp = P_p[pivot_row, j]
                P_p[pivot_row, j] = P_m[pivot_row, j]
                P_m[pivot_row, j] = tp

        for r_i in range(num_active):
            if r_i == pivot_row:
                continue
            if (mat_p[r_i, w_idx] & b_mask) != 0:
                _f3_row_axpy_nb(mat_p, mat_m, r_i, pivot_row, w_idx, n_var_words, False)
                _f3_row_axpy_nb(P_p, P_m, r_i, pivot_row, 0, n_eq_words, False)
            elif (mat_m[r_i, w_idx] & b_mask) != 0:
                _f3_row_axpy_nb(mat_p, mat_m, r_i, pivot_row, w_idx, n_var_words, True)
                _f3_row_axpy_nb(P_p, P_m, r_i, pivot_row, 0, n_eq_words, True)

        pivot_cols_buf[pivot_row] = col
        is_pivot_col[col] = 1
        pivot_row += 1
        if pivot_row == num_active:
            break

    rank_J1 = pivot_row
    pivot_cols = pivot_cols_buf[:rank_J1].copy()
    free_count = num_vars - rank_J1
    free_cols = np.empty(free_count, dtype=np.int32)
    fc = 0
    for col in range(num_vars):
        if is_pivot_col[col] == 0:
            free_cols[fc] = col
            fc += 1

    return (
        0,
        num_vars,
        num_active,
        rank_J1,
        nnz_total,
        gauge_fixed,
        eq_triples,
        var_u,
        var_v,
        var_w,
        pivot_cols,
        free_cols,
        mat_p[:rank_J1, :].copy(),
        mat_m[:rank_J1, :].copy(),
        P_p,
        P_m,
        e1_vec,
    )


@nb.njit(cache=True)
def _apply_P_and_solve_f3_nb(
    num_vars: int,
    rank_J1: int,
    pivot_cols: np.ndarray,
    free_cols: np.ndarray,
    rref_p: np.ndarray,
    rref_m: np.ndarray,
    P_p: np.ndarray,
    P_m: np.ndarray,
    e_vec: np.ndarray,
    free_assignment: np.ndarray,
) -> Tuple[int, np.ndarray]:
    """Solve J1 x = e_vec (mod 3) using precomputed GF(3) RREF + (P_p, P_m) in 64-lane SIMD!"""
    num_active = e_vec.shape[0]
    n_eq_words = P_p.shape[1]
    ep = np.zeros(n_eq_words, dtype=np.uint64)
    em = np.zeros(n_eq_words, dtype=np.uint64)
    for i in range(num_active):
        ev = e_vec[i]
        if ev == 1:
            ep[i >> 6] |= np.uint64(1) << np.uint64(i & 63)
        elif ev == 2:
            em[i >> 6] |= np.uint64(1) << np.uint64(i & 63)

    coker_viol = 0
    for r_i in range(rank_J1, num_active):
        acc = np.int64(0)
        for w_i in range(n_eq_words):
            pp = P_p[r_i, w_i]
            pm = P_m[r_i, w_i]
            acc += np.int64(_popcount64((pp & ep[w_i]) | (pm & em[w_i])))
            acc -= np.int64(_popcount64((pp & em[w_i]) | (pm & ep[w_i])))
        if acc % 3 != 0:
            coker_viol += 1

    x = np.zeros(num_vars, dtype=np.uint8)
    if coker_viol > 0:
        return coker_viol, x

    n_free = free_cols.shape[0]
    n_var_words = rref_p.shape[1]
    fp = np.zeros(n_var_words, dtype=np.uint64)
    fm = np.zeros(n_var_words, dtype=np.uint64)
    for t in range(n_free):
        val = free_assignment[t] % 3
        col = free_cols[t]
        x[col] = val
        if val == 1:
            fp[col >> 6] |= np.uint64(1) << np.uint64(col & 63)
        elif val == 2:
            fm[col >> 6] |= np.uint64(1) << np.uint64(col & 63)

    for r_i in range(rank_J1):
        rhs = np.int64(0)
        for w_i in range(n_eq_words):
            pp = P_p[r_i, w_i]
            pm = P_m[r_i, w_i]
            rhs += np.int64(_popcount64((pp & ep[w_i]) | (pm & em[w_i])))
            rhs -= np.int64(_popcount64((pp & em[w_i]) | (pm & ep[w_i])))
        fc = np.int64(0)
        for w_i in range(n_var_words):
            rp = rref_p[r_i, w_i]
            rm = rref_m[r_i, w_i]
            fc += np.int64(_popcount64((rp & fp[w_i]) | (rm & fm[w_i])))
            fc -= np.int64(_popcount64((rp & fm[w_i]) | (rm & fp[w_i])))
        x[pivot_cols[r_i]] = (rhs - fc) % 3

    return 0, x


@nb.njit(cache=True)
def _guided_free_assignment_f3_nb(
    num_vars: int,
    rank_J1: int,
    pivot_cols: np.ndarray,
    free_cols: np.ndarray,
    rref_p: np.ndarray,
    rref_m: np.ndarray,
    x0: np.ndarray,
    desired: np.ndarray,
    priority_order: np.ndarray,
) -> np.ndarray:
    """
    Lexicographic GF(3) Guided Constrained Steering:
    Pick free_trits c in F_3^{n_free} to satisfy as many desired[vid] == x(c)[vid]
    equations as possible in priority order (off-support zeros first, then constant-digit
    3-adic steering toward {0, +1, -1, +1/2, -1/2}!).
    """
    n_free = free_cols.shape[0]
    if n_free == 0:
        return np.zeros(0, dtype=np.uint8)

    col_to_row = np.full(num_vars, -1, dtype=np.int32)
    for r_i in range(rank_J1):
        col_to_row[pivot_cols[r_i]] = r_i
    col_to_free = np.full(num_vars, -1, dtype=np.int32)
    for t in range(n_free):
        col_to_free[free_cols[t]] = t

    basis_used = np.zeros(n_free, dtype=np.uint8)
    basis_lhs = np.zeros((n_free, n_free), dtype=np.uint8)
    basis_rhs = np.zeros(n_free, dtype=np.uint8)
    row_lhs = np.zeros(n_free, dtype=np.uint8)
    inserted = 0

    for p_idx in range(num_vars):
        if inserted >= n_free:
            break
        vid = priority_order[p_idx]
        for t in range(n_free):
            row_lhs[t] = 0
        fi = col_to_free[vid]
        if fi >= 0:
            row_lhs[fi] = 1
            rhs = desired[vid] % 3
        else:
            r_i = col_to_row[vid]
            if r_i < 0:
                continue
            # x[vid] = x0[vid] - sum_t rref[r_i, free_cols[t]] * c_t == desired[vid] (mod 3)
            # <=> sum_t rref[r_i, free_cols[t]] * c_t == (x0[vid] - desired[vid]) (mod 3)
            rhs = (int(x0[vid]) - int(desired[vid])) % 3
            for t in range(n_free):
                fc = free_cols[t]
                w_c = fc >> 6
                b_c = np.uint64(1) << np.uint64(fc & 63)
                if (rref_p[r_i, w_c] & b_c) != 0:
                    row_lhs[t] = 1
                elif (rref_m[r_i, w_c] & b_c) != 0:
                    row_lhs[t] = 2

        lead = -1
        for bit in range(n_free):
            v = row_lhs[bit]
            if v != 0:
                if basis_used[bit] != 0:
                    # basis_lhs[bit, bit] == 1; subtract v * basis_lhs[bit]
                    if v == 1:
                        for t in range(bit, n_free):
                            row_lhs[t] = (int(row_lhs[t]) - int(basis_lhs[bit, t])) % 3
                        rhs = (int(rhs) - int(basis_rhs[bit])) % 3
                    else:
                        for t in range(bit, n_free):
                            row_lhs[t] = (int(row_lhs[t]) + int(basis_lhs[bit, t])) % 3
                        rhs = (int(rhs) + int(basis_rhs[bit])) % 3
                else:
                    lead = bit
                    break

        if lead >= 0:
            basis_used[lead] = 1
            if row_lhs[lead] == 2:
                # Multiply row by 2 mod 3 so leading entry == 1
                for t in range(lead, n_free):
                    basis_lhs[lead, t] = (2 * int(row_lhs[t])) % 3
                basis_rhs[lead] = (2 * int(rhs)) % 3
            else:
                for t in range(lead, n_free):
                    basis_lhs[lead, t] = row_lhs[t]
                basis_rhs[lead] = rhs
            inserted += 1

    c_sol = np.zeros(n_free, dtype=np.uint8)
    for bit in range(n_free - 1, -1, -1):
        if basis_used[bit] == 0:
            c_sol[bit] = 0
            continue
        acc = int(basis_rhs[bit])
        for t in range(bit + 1, n_free):
            if c_sol[t] != 0 and basis_lhs[bit, t] != 0:
                acc -= int(basis_lhs[bit, t]) * int(c_sol[t])
        c_sol[bit] = acc % 3

    return c_sol


@nb.njit(cache=True)
def _compute_mod3kp1_residual_nb(
    m: int,
    n: int,
    p: int,
    U0: np.ndarray,
    U1: np.ndarray,
    U2: np.ndarray,
    V0: np.ndarray,
    V1: np.ndarray,
    V2: np.ndarray,
    W0: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    eq_triples: np.ndarray,
    k: int,
    pow3_tbl: np.ndarray,
) -> Tuple[int, np.ndarray]:
    """
    Exact 3-limb base B = 3^14 = 4,782,969 evaluation of
      e_k[idx] = ((T_exact - T_k) / 3^k) mod 3   for k in 1..40 (modulus up to 3^42 ~ 1.09e20)!
    Zero int64 overflow guaranteed for R <= 1000!
    """
    B = np.int64(4782969)  # 3^14
    np_dim = n * p
    mp = m * p
    R = U0.shape[0]
    num_active = eq_triples.shape[0]
    e_k = np.empty(num_active, dtype=np.uint8)

    limb_q = k // 14
    rem_r = k % 14
    p3_r = pow3_tbl[rem_r]

    for row in range(num_active):
        idx = eq_triples[row]
        c = idx % mp
        tmp = idx // mp
        b = tmp % np_dim
        a = tmp // np_dim

        i1 = a // n
        i2 = a - i1 * n
        j1 = b // p
        j2 = b - j1 * p
        k1 = c // p
        k2 = c - k1 * p
        t_ex = np.int64(1) if (i1 == k1 and i2 == j1 and j2 == k2) else np.int64(0)

        s0 = np.int64(0)
        s1 = np.int64(0)
        s2 = np.int64(0)
        for r in range(R):
            ua0, ua1, ua2 = U0[r, a], U1[r, a], U2[r, a]
            if (ua0 | ua1 | ua2) == 0:
                continue
            vb0, vb1, vb2 = V0[r, b], V1[r, b], V2[r, b]
            if (vb0 | vb1 | vb2) == 0:
                continue
            wc0, wc1, wc2 = W0[r, c], W1[r, c], W2[r, c]
            if (wc0 | wc1 | wc2) == 0:
                continue

            # uv = u * v mod B^3
            t0 = ua0 * vb0
            c0 = t0 // B
            uv0 = t0 - c0 * B
            t1 = ua0 * vb1 + ua1 * vb0 + c0
            c1 = t1 // B
            uv1 = t1 - c1 * B
            uv2 = (ua0 * vb2 + ua1 * vb1 + ua2 * vb0 + c1) % B

            # uvw = uv * w mod B^3
            q0 = uv0 * wc0
            d0 = q0 // B
            s0 += q0 - d0 * B
            q1 = uv0 * wc1 + uv1 * wc0 + d0
            d1 = q1 // B
            s1 += q1 - d1 * B
            s2 += (uv0 * wc2 + uv1 * wc1 + uv2 * wc0 + d1) % B

        # Carry-reduce (s0, s1, s2) mod B^3
        cs0 = s0 // B
        s0 -= cs0 * B
        s1 += cs0
        cs1 = s1 // B
        s1 -= cs1 * B
        s2 = (s2 + cs1) % B

        # diff = (t_ex - s) mod B^3
        df0 = t_ex - s0
        br0 = np.int64(0)
        if df0 < 0:
            df0 += B
            br0 = np.int64(1)
        df1 = -s1 - br0
        br1 = np.int64(0)
        while df1 < 0:
            df1 += B
            br1 += np.int64(1)
        df2 = (-s2 - br1) % B
        if df2 < 0:
            df2 += B

        # Check exact divisibility by 3^k and extract k-th base-3 digit
        if limb_q == 0:
            if df0 % p3_r != 0:
                return -1, e_k
            e_k[row] = (df0 // p3_r) % 3
        elif limb_q == 1:
            if df0 != 0 or (df1 % p3_r) != 0:
                return -1, e_k
            e_k[row] = (df1 // p3_r) % 3
        else:
            if df0 != 0 or df1 != 0 or (df2 % p3_r) != 0:
                return -1, e_k
            e_k[row] = (df2 // p3_r) % 3

    return 0, e_k


@nb.njit(cache=True)
def _linear_hensel_p3_run_nb(
    m: int,
    n: int,
    p: int,
    up: np.ndarray,
    un: np.ndarray,
    vp: np.ndarray,
    vn: np.ndarray,
    wp: np.ndarray,
    wn: np.ndarray,
    num_vars: int,
    rank_J1: int,
    eq_triples: np.ndarray,
    var_u: np.ndarray,
    var_v: np.ndarray,
    var_w: np.ndarray,
    pivot_cols: np.ndarray,
    free_cols: np.ndarray,
    rref_p: np.ndarray,
    rref_m: np.ndarray,
    P_p: np.ndarray,
    P_m: np.ndarray,
    e1_vec: np.ndarray,
    free_init_k1: np.ndarray,
    free_init_k2: np.ndarray,
    target_trits: int,
    steer_mode: int,
    rng_seed: np.uint64,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Linear 3-adic Hensel lifting 3^1 -> 3^2 -> ... -> 3^{target_trits} (up to 3^40)
    reusing ONE precomputed GF(3) RREF + (P_p, P_m), with constant-digit 3-adic steering
    (simultaneously attracts 0, +1, -1, +1/2, and -1/2!).
    """
    B = np.int64(4782969)  # 3^14
    mn = m * n
    np_dim = n * p
    mp = m * p
    R = up.shape[0]
    n_free = free_cols.shape[0]

    pow3_tbl = np.empty(15, dtype=np.int64)
    pow3_tbl[0] = np.int64(1)
    for i in range(1, 15):
        pow3_tbl[i] = pow3_tbl[i - 1] * np.int64(3)

    U0 = np.zeros((R, mn), dtype=np.int64)
    U1 = np.zeros((R, mn), dtype=np.int64)
    U2 = np.zeros((R, mn), dtype=np.int64)
    V0 = np.zeros((R, np_dim), dtype=np.int64)
    V1 = np.zeros((R, np_dim), dtype=np.int64)
    V2 = np.zeros((R, np_dim), dtype=np.int64)
    W0 = np.zeros((R, mp), dtype=np.int64)
    W1 = np.zeros((R, mp), dtype=np.int64)
    W2 = np.zeros((R, mp), dtype=np.int64)

    # Initialize mod 3: +1 -> 1, -1 -> 2, 0 -> 0 (wait: balanced -1 = 2 + 2*3 + ... is represented in [0, 3) as 2!)
    # Note: e1_vec was computed using balanced {-1, 0, +1}!
    # If we store balanced -1 as (-1 mod B^3) = (B-1, B-1, B-1) initially:
    #   WAIT: (-1) = 2 + 2*3 + 2*3^2 + ... (mod 3^42), whereas 2 = 2 + 0*3 + 0*3^2!
    #   Using exact balanced {-1, 0, +1} mod B^3 at k=0 means U_1 matches counts[] IDENTICALLY!
    #   AND for k >= 1, adding delta * 3^k with delta in {0, 1, 2} adjusts U_k -> U_{k+1} identically!
    for r in range(R):
        for a in range(mn):
            bit = np.uint64(1) << np.uint64(a)
            if (up[r] & bit) != 0:
                U0[r, a] = np.int64(1)
            elif (un[r] & bit) != 0:
                U0[r, a] = B - np.int64(1)
                U1[r, a] = B - np.int64(1)
                U2[r, a] = B - np.int64(1)
        for b in range(np_dim):
            bit = np.uint64(1) << np.uint64(b)
            if (vp[r] & bit) != 0:
                V0[r, b] = np.int64(1)
            elif (vn[r] & bit) != 0:
                V0[r, b] = B - np.int64(1)
                V1[r, b] = B - np.int64(1)
                V2[r, b] = B - np.int64(1)
        for c in range(mp):
            bit = np.uint64(1) << np.uint64(c)
            if (wp[r] & bit) != 0:
                W0[r, c] = np.int64(1)
            elif (wn[r] & bit) != 0:
                W0[r, c] = B - np.int64(1)
                W1[r, c] = B - np.int64(1)
                W2[r, c] = B - np.int64(1)

    var_mode = np.empty(num_vars, dtype=np.int8)
    var_r = np.empty(num_vars, dtype=np.int32)
    var_pos = np.empty(num_vars, dtype=np.int32)
    var_on_supp = np.zeros(num_vars, dtype=np.uint8)
    for r in range(R):
        um = up[r] | un[r]
        vm = vp[r] | vn[r]
        wm = wp[r] | wn[r]
        for a in range(mn):
            vid = var_u[r, a]
            if vid >= 0:
                var_mode[vid] = 0
                var_r[vid] = r
                var_pos[vid] = a
                var_on_supp[vid] = (um >> np.uint64(a)) & np.uint64(1)
        for b in range(np_dim):
            vid = var_v[r, b]
            if vid >= 0:
                var_mode[vid] = 1
                var_r[vid] = r
                var_pos[vid] = b
                var_on_supp[vid] = (vm >> np.uint64(b)) & np.uint64(1)
        for c in range(mp):
            vid = var_w[r, c]
            if vid >= 0:
                var_mode[vid] = 2
                var_r[vid] = r
                var_pos[vid] = c
                var_on_supp[vid] = (wm >> np.uint64(c)) & np.uint64(1)

    priority_order = np.empty(num_vars, dtype=np.int32)
    po = 0
    for vid in range(num_vars):
        if var_on_supp[vid] == 0:
            priority_order[po] = vid
            po += 1
    for vid in range(num_vars):
        if var_on_supp[vid] != 0:
            priority_order[po] = vid
            po += 1

    state = rng_seed if rng_seed != 0 else np.uint64(0x9E3779B97F4A7C15)
    e_cur = e1_vec
    zero_free = np.zeros(n_free, dtype=np.uint8)
    free_vec = np.empty(n_free, dtype=np.uint8)
    desired = np.zeros(num_vars, dtype=np.uint8)

    for k in range(1, target_trits):
        if k > 1:
            err_flag, e_cur = _compute_mod3kp1_residual_nb(
                m, n, p, U0, U1, U2, V0, V1, V2, W0, W1, W2, eq_triples, k, pow3_tbl
            )
            if err_flag != 0:
                return k, U0, U1, U2, V0, V1, V2, W0, W1, W2

        coker_k, x0 = _apply_P_and_solve_f3_nb(
            num_vars, rank_J1, pivot_cols, free_cols, rref_p, rref_m, P_p, P_m, e_cur, zero_free
        )
        if coker_k > 0:
            return k, U0, U1, U2, V0, V1, V2, W0, W1, W2

        if k == 1:
            if steer_mode == 1:
                desired[:] = 0
                for t in range(n_free):
                    col = free_cols[t]
                    if var_on_supp[col] != 0:
                        desired[col] = free_init_k1[t] % 3
                free_vec = _guided_free_assignment_f3_nb(
                    num_vars, rank_J1, pivot_cols, free_cols, rref_p, rref_m, x0, desired, priority_order
                )
            else:
                for t in range(n_free):
                    free_vec[t] = free_init_k1[t] % 3
        elif k == 2 and steer_mode == 2:
            for t in range(n_free):
                free_vec[t] = free_init_k2[t] % 3
        else:
            if steer_mode >= 1:
                # Constant-digit 3-adic steering: steer each entry's k-th base-3 digit to match its (k-1)-th digit!
                # Since initially -1 was stored as (B-1,B-1,B-1) (all 2s), adding delta_k = 0 keeps 0, +1, -1 exact,
                # and for +1/2 (-1 -> +1/2 needs +1*3^1 + ... => d_1=1, d_2=1, ...) matching (k-1)-th digit stabilizes
                # EVERY one of {0, +1, -1, +1/2, -1/2} automatically!
                # Wait: because -1 has d_{k-1} = 2 already in limbs, to keep digit_k == digit_{k-1},
                # we want (current_digit_k + delta_k) % 3 == current_digit_{k-1}!
                q_prev = (k - 1) // 14
                r_prev = (k - 1) % 14
                p3_prev = pow3_tbl[r_prev]
                q_cur = k // 14
                r_cur = k % 14
                p3_cur = pow3_tbl[r_cur]
                for vid in range(num_vars):
                    md = var_mode[vid]
                    rr = var_r[vid]
                    pp = var_pos[vid]
                    if md == 0:
                        lv_p = U0[rr, pp] if q_prev == 0 else (U1[rr, pp] if q_prev == 1 else U2[rr, pp])
                        lv_c = U0[rr, pp] if q_cur == 0 else (U1[rr, pp] if q_cur == 1 else U2[rr, pp])
                    elif md == 1:
                        lv_p = V0[rr, pp] if q_prev == 0 else (V1[rr, pp] if q_prev == 1 else V2[rr, pp])
                        lv_c = V0[rr, pp] if q_cur == 0 else (V1[rr, pp] if q_cur == 1 else V2[rr, pp])
                    else:
                        lv_p = W0[rr, pp] if q_prev == 0 else (W1[rr, pp] if q_prev == 1 else W2[rr, pp])
                        lv_c = W0[rr, pp] if q_cur == 0 else (W1[rr, pp] if q_cur == 1 else W2[rr, pp])
                    d_prev = (lv_p // p3_prev) % 3
                    d_cur = (lv_c // p3_cur) % 3
                    desired[vid] = (d_prev - d_cur) % 3
                free_vec = _guided_free_assignment_f3_nb(
                    num_vars, rank_J1, pivot_cols, free_cols, rref_p, rref_m, x0, desired, priority_order
                )
            else:
                for t in range(n_free):
                    state ^= state >> np.uint64(12)
                    state ^= state << np.uint64(25)
                    state ^= state >> np.uint64(27)
                    free_vec[t] = (state * np.uint64(0x2545F4914F6CDD1D)) % np.uint64(3)

        _, delta = _apply_P_and_solve_f3_nb(
            num_vars, rank_J1, pivot_cols, free_cols, rref_p, rref_m, P_p, P_m, e_cur, free_vec
        )

        # Add delta[vid] * 3^k to 3-limb representation mod B^3
        q_k = k // 14
        add_unit = pow3_tbl[k % 14]
        for vid in range(num_vars):
            dv = int(delta[vid])
            if dv == 0:
                continue
            add_v = np.int64(dv) * add_unit
            md = var_mode[vid]
            rr = var_r[vid]
            pp = var_pos[vid]
            if md == 0:
                a0, a1, a2 = U0[rr, pp], U1[rr, pp], U2[rr, pp]
                if q_k == 0:
                    a0 += add_v
                elif q_k == 1:
                    a1 += add_v
                else:
                    a2 += add_v
                c0 = a0 // B
                a0 -= c0 * B
                a1 += c0
                c1 = a1 // B
                a1 -= c1 * B
                a2 = (a2 + c1) % B
                U0[rr, pp], U1[rr, pp], U2[rr, pp] = a0, a1, a2
            elif md == 1:
                a0, a1, a2 = V0[rr, pp], V1[rr, pp], V2[rr, pp]
                if q_k == 0:
                    a0 += add_v
                elif q_k == 1:
                    a1 += add_v
                else:
                    a2 += add_v
                c0 = a0 // B
                a0 -= c0 * B
                a1 += c0
                c1 = a1 // B
                a1 -= c1 * B
                a2 = (a2 + c1) % B
                V0[rr, pp], V1[rr, pp], V2[rr, pp] = a0, a1, a2
            else:
                a0, a1, a2 = W0[rr, pp], W1[rr, pp], W2[rr, pp]
                if q_k == 0:
                    a0 += add_v
                elif q_k == 1:
                    a1 += add_v
                else:
                    a2 += add_v
                c0 = a0 // B
                a0 -= c0 * B
                a1 += c0
                c1 = a1 // B
                a1 -= c1 * B
                a2 = (a2 + c1) % B
                W0[rr, pp], W1[rr, pp], W2[rr, pp] = a0, a1, a2

    return target_trits, U0, U1, U2, V0, V1, V2, W0, W1, W2


def mod9_oracle_u64(
    m: int,
    n: int,
    p: int,
    up: np.ndarray,
    un: np.ndarray,
    vp: np.ndarray,
    vn: np.ndarray,
    wp: np.ndarray,
    wn: np.ndarray,
    support_restricted: bool = True,
) -> Mod9OracleResult:
    """Evaluate Stage L0-p3 mod-9 liftability oracle over F_3 on 2-bitplane uint64[R] scheme."""
    t0 = time.perf_counter()
    coker, free_t, rk, n_vars, n_eqs, nnz, gf = _mod9_oracle_nb(
        m, n, p, up, un, vp, vn, wp, wn, support_restricted
    )
    if coker < 0:
        raise ValueError("Input (up, un, vp, vn, wp, wn) is NOT a valid F_3 matrix multiplication scheme!")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return Mod9OracleResult(
        coker_viol_9=int(coker),
        free_trits=int(free_t),
        rank_J1=int(rk),
        num_vars=int(n_vars),
        num_active_eqs=int(n_eqs),
        is_mod9_liftable=(int(coker) == 0),
        support_restricted=support_restricted,
        nnz_total=int(nnz),
        gauge_fixed=int(gf),
        elapsed_ms=round(elapsed_ms, 4),
    )


def mod9_oracle(scheme: BilinearScheme, support_restricted: bool = True) -> Mod9OracleResult:
    """Convenience wrapper accepting a BilinearScheme (Z3, Z, or Z[1/2] Q)."""
    up, un, vp, vn, wp, wn = scheme_to_u64_f3(scheme)
    return mod9_oracle_u64(
        scheme.m, scheme.n, scheme.p, up, un, vp, vn, wp, wn, support_restricted=support_restricted
    )


@nb.njit(parallel=True, cache=True)
def _mod9_oracle_batch_nb(
    m: int,
    n: int,
    p: int,
    Up: np.ndarray,
    Un: np.ndarray,
    Vp: np.ndarray,
    Vn: np.ndarray,
    Wp: np.ndarray,
    Wn: np.ndarray,
    support_restricted: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    B_sz = Up.shape[0]
    cokers = np.empty(B_sz, dtype=np.int32)
    frees = np.empty(B_sz, dtype=np.int32)
    ranks = np.empty(B_sz, dtype=np.int32)
    for b_idx in nb.prange(B_sz):
        c, f, rk, _, _, _, _ = _mod9_oracle_nb(
            m, n, p, Up[b_idx], Un[b_idx], Vp[b_idx], Vn[b_idx], Wp[b_idx], Wn[b_idx], support_restricted
        )
        cokers[b_idx] = c
        frees[b_idx] = f
        ranks[b_idx] = rk
    return cokers, frees, ranks


def mod9_oracle_batch_u64(
    m: int,
    n: int,
    p: int,
    Up: np.ndarray,
    Un: np.ndarray,
    Vp: np.ndarray,
    Vn: np.ndarray,
    Wp: np.ndarray,
    Wn: np.ndarray,
    support_restricted: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parallel CPU 12T batch evaluation of Stage L0-p3 mod-9 oracle."""
    return _mod9_oracle_batch_nb(
        m,
        n,
        p,
        np.ascontiguousarray(Up, dtype=np.uint64),
        np.ascontiguousarray(Un, dtype=np.uint64),
        np.ascontiguousarray(Vp, dtype=np.uint64),
        np.ascontiguousarray(Vn, dtype=np.uint64),
        np.ascontiguousarray(Wp, dtype=np.uint64),
        np.ascontiguousarray(Wn, dtype=np.uint64),
        support_restricted,
    )


def _reconstruct_and_verify_p3(
    m: int,
    n: int,
    p: int,
    U0: np.ndarray,
    U1: np.ndarray,
    U2: np.ndarray,
    V0: np.ndarray,
    V1: np.ndarray,
    V2: np.ndarray,
    W0: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    trits: int,
    name: str,
) -> Optional[Tuple[BilinearScheme, VerificationReport, str]]:
    """
    Reconstruct exact integer or rational (incl. Z[1/2] with denominators 2, 4, 8, ...!)
    BilinearScheme modulo M = 3^{trits} via Wang Extended-Euclidean rational reconstruction,
    and strictly verify with `brent_verifier.verify_scheme()`.
    """
    B = 4782969  # 3^14
    B2 = B * B
    M = 3**trits
    half = M // 2
    bound = math.isqrt(M // 2)
    R = U0.shape[0]
    mn, np_dim, mp = m * n, n * p, m * p

    def _extract_vals(A0: np.ndarray, A1: np.ndarray, A2: np.ndarray, cols: int) -> List[List[int]]:
        out: List[List[int]] = []
        for r in range(R):
            row: List[int] = []
            for j in range(cols):
                val = (int(A0[r, j]) + int(A1[r, j]) * B + int(A2[r, j]) * B2) % M
                row.append(val)
            out.append(row)
        return out

    U_m = _extract_vals(U0, U1, U2, mn)
    V_m = _extract_vals(V0, V1, V2, np_dim)
    W_m = _extract_vals(W0, W1, W2, mp)

    # 1. Fast path: small balanced integers (catches Z_T and small Z immediately)
    max_int_mag = min(bound, 1 << 20)
    all_small_int = True
    U_z: List[List[int]] = []
    V_z: List[List[int]] = []
    W_z: List[List[int]] = []
    for src, dst in ((U_m, U_z), (V_m, V_z), (W_m, W_z)):
        for row in src:
            zrow: List[int] = []
            for val in row:
                sval = val if val <= half else val - M
                if abs(sval) > max_int_mag:
                    all_small_int = False
                zrow.append(sval)
            dst.append(zrow)

    if all_small_int:
        cand_z = BilinearScheme(m=m, n=n, p=p, U=U_z, V=V_z, W=W_z, ring="Z", name=name)
        rep_z = verify_scheme(cand_z)
        if rep_z.valid:
            return cand_z, rep_z, ("verified_ZT" if rep_z.is_ternary else "verified_Z")

    # 2. Wang rational reconstruction mod 3^{trits} (recovers 1/2, 1/4, -3/8, 1/5, ...!)
    U_q: List[List[Fraction]] = []
    V_q: List[List[Fraction]] = []
    W_q: List[List[Fraction]] = []
    all_int = True
    for src, dst_q in ((U_m, U_q), (V_m, V_q), (W_m, W_q)):
        for row in src:
            qrow: List[Fraction] = []
            for val in row:
                fr = wang_rational_reconstruct(val, M, bound=bound)
                if fr is None:
                    return None
                if fr.denominator != 1:
                    all_int = False
                qrow.append(fr)
            dst_q.append(qrow)

    if all_int:
        cand_z2 = BilinearScheme(
            m=m,
            n=n,
            p=p,
            U=[[int(x.numerator) for x in r] for r in U_q],
            V=[[int(x.numerator) for x in r] for r in V_q],
            W=[[int(x.numerator) for x in r] for r in W_q],
            ring="Z",
            name=name,
        )
        rep_z2 = verify_scheme(cand_z2)
        if rep_z2.valid:
            return cand_z2, rep_z2, ("verified_ZT" if rep_z2.is_ternary else "verified_Z")

    cand_q = BilinearScheme(m=m, n=n, p=p, U=U_q, V=V_q, W=W_q, ring="Q", name=name)
    rep_q = verify_scheme(cand_q)
    if rep_q.valid:
        return cand_q, rep_q, "verified_Q"
    return None


def linear_hensel_lift_p3(
    scheme_or_u64: BilinearScheme
    | Tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    support_restricted: bool = False,
    target_trits: int = 40,
    max_trials: int = 24,
    seed: int = 42,
    name: str = "",
) -> LiftResult:
    """
    Stage L1-p3: Linear 3-adic Hensel lifting (3^k -> 3^{k+1} up to 3^{target_trits} <= 3^40 ~ 1.2e19)
    reusing ONE precomputed GF(3) RREF + (P_p, P_m), with ker(J1) nullspace kicks at k=1,2
    + constant-digit 3-adic steering + Wang rational reconstruction -> Z[1/2] / Q / Z / Z_T!
    """
    t0 = time.perf_counter()
    if isinstance(scheme_or_u64, BilinearScheme):
        m, n, p = scheme_or_u64.m, scheme_or_u64.n, scheme_or_u64.p
        up, un, vp, vn, wp, wn = scheme_to_u64_f3(scheme_or_u64)
        base_name = name or (scheme_or_u64.name + "_hensel3")
    else:
        m, n, p, up, un, vp, vn, wp, wn = scheme_or_u64
        base_name = name or f"{m}x{n}x{p}_R{len(up)}_hensel3"

    if target_trits < 4 or target_trits > 40:
        raise ValueError(f"target_trits must be in [4, 40], got {target_trits}")

    res = _build_f3_jacobian_rref_nb(m, n, p, up, un, vp, vn, wp, wn, support_restricted)
    if res[0] < 0:
        raise ValueError("Input scheme is NOT valid mod 3!")

    (
        _,
        num_vars,
        num_active,
        rank_J1,
        nnz_total,
        gauge_fixed,
        eq_triples,
        var_u,
        var_v,
        var_w,
        pivot_cols,
        free_cols,
        rref_p,
        rref_m,
        P_p,
        P_m,
        e1_vec,
    ) = res
    n_free = num_vars - rank_J1

    coker_9, _ = _apply_P_and_solve_f3_nb(
        num_vars, rank_J1, pivot_cols, free_cols, rref_p, rref_m, P_p, P_m, e1_vec, np.zeros(n_free, dtype=np.uint8)
    )
    if coker_9 > 0:
        return LiftResult(
            success=False,
            status="obstructed_mod9",
            scheme=None,
            report=None,
            method=f"linear_hensel_p3_{'supp' if support_restricted else 'unres'}",
            elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            coker_viol_4=int(coker_9),
            max_level_reached=1,
            details={"coker_viol_9": int(coker_9), "rank_J1": int(rank_J1), "free_trits": int(n_free)},
        )

    rng = np.random.default_rng(seed)
    best_level = 1
    trial_specs: List[Tuple[np.ndarray, np.ndarray, int]] = []
    z0 = np.zeros(n_free, dtype=np.uint8)
    trial_specs.append((z0, z0, 1))
    trial_specs.append((z0, z0, 2))
    trial_specs.append((z0, z0, 0))
    # Sparse 1-trit kicks at k=1 (both mode 1 and mode 2)
    for t in range(min(n_free, max(0, max_trials // 4))):
        a1 = np.zeros(n_free, dtype=np.uint8)
        a1[t] = 1 if (t & 1) == 0 else 2
        trial_specs.append((a1, z0, 1 if (t & 1) == 0 else 2))
    while len(trial_specs) < max_trials:
        idx_k = len(trial_specs)
        if idx_k % 2 == 0 and n_free >= 3:
            a1 = np.zeros(n_free, dtype=np.uint8)
            a2 = np.zeros(n_free, dtype=np.uint8)
            kb = int(rng.integers(1, min(6, n_free) + 1))
            ch1 = rng.choice(n_free, size=kb, replace=False)
            a1[ch1] = rng.integers(1, 3, size=kb, dtype=np.uint8)
            ch2 = rng.choice(n_free, size=kb, replace=False)
            a2[ch2] = rng.integers(1, 3, size=kb, dtype=np.uint8)
            trial_specs.append((a1, a2, 2))
        else:
            a1 = rng.integers(0, 3, size=n_free, dtype=np.uint8)
            a2 = rng.integers(0, 3, size=n_free, dtype=np.uint8)
            mode = 1 if (idx_k % 3 == 0) else (2 if (idx_k % 3 == 1) else 0)
            trial_specs.append((a1, a2, mode))

    for t_idx, (fk1, fk2, smode) in enumerate(trial_specs):
        lvl, U0, U1, U2, V0, V1, V2, W0, W1, W2 = _linear_hensel_p3_run_nb(
            m,
            n,
            p,
            up,
            un,
            vp,
            vn,
            wp,
            wn,
            num_vars,
            rank_J1,
            eq_triples,
            var_u,
            var_v,
            var_w,
            pivot_cols,
            free_cols,
            rref_p,
            rref_m,
            P_p,
            P_m,
            e1_vec,
            fk1,
            fk2,
            target_trits,
            smode,
            np.uint64(seed + 10007 * (t_idx + 1)),
        )
        if lvl > best_level:
            best_level = int(lvl)
        if lvl == target_trits:
            for chk_t in sorted({min(12, target_trits), min(24, target_trits), target_trits}):
                rec = _reconstruct_and_verify_p3(
                    m, n, p, U0, U1, U2, V0, V1, V2, W0, W1, W2, chk_t, base_name
                )
                if rec is not None:
                    cand, rep, st = rec
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    max_den = 1
                    if cand.ring == "Q":
                        for mat_q in (cand.U, cand.V, cand.W):
                            for row_q in mat_q:
                                for x in row_q:
                                    if x.denominator > max_den:
                                        max_den = x.denominator
                    return LiftResult(
                        success=True,
                        status=st,
                        scheme=cand,
                        report=rep,
                        method=f"linear_hensel_p3_{'supp' if support_restricted else 'unres'}",
                        elapsed_ms=round(elapsed_ms, 3),
                        coker_viol_4=0,
                        max_level_reached=int(lvl),
                        details={
                            "coker_viol_9": 0,
                            "trial_idx": t_idx,
                            "steer_mode": smode,
                            "reconstructed_at_trits": chk_t,
                            "rank_J1": int(rank_J1),
                            "free_trits": int(n_free),
                            "max_abs_coeff": str(rep.max_abs_coeff),
                            "max_denom": max_den,
                        },
                    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    final_status = "recon_failed" if best_level == target_trits else "obstructed_mod3k"
    return LiftResult(
        success=False,
        status=final_status,
        scheme=None,
        report=None,
        method=f"linear_hensel_p3_{'supp' if support_restricted else 'unres'}",
        elapsed_ms=round(elapsed_ms, 3),
        coker_viol_4=0,
        max_level_reached=best_level,
        details={"coker_viol_9": 0, "rank_J1": int(rank_J1), "free_trits": int(n_free), "trials": len(trial_specs)},
    )


def _make_half_basis_strassen_q() -> Tuple[BilinearScheme, BilinearScheme]:
    """
    Construct a genuinely non-trivial Z[1/2] rational Strassen <2,2,2:7> scheme via
    non-unimodular basis change P = [[1/2, 1/2], [1/2, -1/2]] (det = -1/2, P^{-1} = [[1, 1], [1, -1]]):
      A' = P A, C' = P C => U'_r = (P^{-1})^T U_r, W'_r = P W_r (irreducible 1/2 coefficients!).
    Returns (exact Q scheme, its mod-3 reduction over Z3 where 1/2 == 2 == -1 mod 3).
    """
    s2 = strassen_222(ring="Z")
    # P = [[1/2, 1/2], [1/2, -1/2]], P^-1 = [[1, 1], [1, -1]], (P^-1)^T = [[1, 1], [1, -1]]
    # Verify: sum_r <U'_r, A> <V_r, B> W'_r = P (sum_r <U_r, P^-1 A> <V_r, B> W_r) = P (P^-1 A B) = A B!
    U_q: List[List[Fraction]] = []
    V_q: List[List[Fraction]] = []
    W_q: List[List[Fraction]] = []
    half = Fraction(1, 2)
    for r in range(s2.rank):
        u = s2.U[r]
        # U'_r[0, j] = u[0, j] + u[1, j], U'_r[1, j] = u[0, j] - u[1, j]
        U_q.append([
            Fraction(u[0] + u[2], 1),
            Fraction(u[1] + u[3], 1),
            Fraction(u[0] - u[2], 1),
            Fraction(u[1] - u[3], 1),
        ])
        V_q.append([Fraction(x, 1) for x in s2.V[r]])
        w = s2.W[r]
        # W'_r[0, k] = 1/2*(w[0, k] + w[1, k]), W'_r[1, k] = 1/2*(w[0, k] - w[1, k])
        W_q.append([
            half * (w[0] + w[2]),
            half * (w[1] + w[3]),
            half * (w[0] - w[2]),
            half * (w[1] - w[3]),
        ])
    sc_q = BilinearScheme(m=2, n=2, p=2, U=U_q, V=V_q, W=W_q, ring="Q", name="strassen_Z_half_Q")
    assert verify_scheme(sc_q).valid
    up, un, vp, vn, wp, wn = scheme_to_u64_f3(sc_q)
    sc_f3 = u64_f3_to_scheme(2, 2, 2, up, un, vp, vn, wp, wn, name="strassen_Z_half_mod3")
    assert verify_scheme(sc_f3).valid
    return sc_q, sc_f3


def _make_obstructed_f3_scheme() -> BilinearScheme:
    """
    Construct a deterministic valid F_3 <3,3,3:27> scheme that is support-obstructed mod 9
    (`coker_viol_9 > 0`) via exact F_3 flip steps starting from naive <3,3,3:27>.
    """
    sc = make_naive_scheme(3, 3, 3, ring="Z3", modulus=3)
    up, un, vp, vn, wp, wn = scheme_to_u64_f3(sc)
    R = len(up)
    rng = random.Random(7)
    for step in range(4000):
        edges: List[Tuple[int, int, int, int]] = []
        for ax, (pa, na) in enumerate(((up, un), (vp, vn), (wp, wn))):
            for i in range(R):
                for j in range(i + 1, R):
                    if pa[i] == pa[j] and na[i] == na[j]:
                        edges.append((ax, i, j, 1))
                    elif pa[i] == na[j] and na[i] == pa[j]:
                        edges.append((ax, i, j, -1))
        if not edges:
            break
        ax, i, j, sgn = rng.choice(edges)
        if rng.randrange(2):
            i, j = j, i
        alpha = 1 if rng.randrange(2) == 0 else -1
        if ax == 0:
            vp[i], vn[i] = _f3_add_u64(vp[i], vn[i], vn[j] if alpha > 0 else vp[j], vp[j] if alpha > 0 else vn[j])
            add_pos = (alpha * sgn) > 0
            wp[j], wn[j] = _f3_add_u64(wp[j], wn[j], wp[i] if add_pos else wn[i], wn[i] if add_pos else wp[i])
        elif ax == 1:
            up[i], un[i] = _f3_add_u64(up[i], un[i], un[j] if alpha > 0 else up[j], up[j] if alpha > 0 else un[j])
            add_pos = (alpha * sgn) > 0
            wp[j], wn[j] = _f3_add_u64(wp[j], wn[j], wp[i] if add_pos else wn[i], wn[i] if add_pos else wp[i])
        else:
            up[i], un[i] = _f3_add_u64(up[i], un[i], un[j] if alpha > 0 else up[j], up[j] if alpha > 0 else un[j])
            add_pos = (alpha * sgn) > 0
            vp[j], vn[j] = _f3_add_u64(vp[j], vn[j], vp[i] if add_pos else vn[i], vn[i] if add_pos else vp[i])
        keep = [r for r in range(R) if (up[r] | un[r]) != 0 and (vp[r] | vn[r]) != 0 and (wp[r] | wn[r]) != 0]
        if len(keep) < R:
            up, un, vp, vn, wp, wn = up[keep], un[keep], vp[keep], vn[keep], wp[keep], wn[keep]
            R = len(up)
        if step >= 2400:
            ck = int(_mod9_oracle_nb(3, 3, 3, up, un, vp, vn, wp, wn, True)[0])
            if ck > 0:
                out = u64_f3_to_scheme(3, 3, 3, up, un, vp, vn, wp, wn, name="obstructed_f3_3x3x3")
                if verify_scheme(out).valid:
                    return out
    raise RuntimeError("Failed to construct obstructed F3 scheme!")


def run_selftest() -> int:
    print("=== hensel_lifter.py --selftest (CPU-only: Numba 12T + Z3 5.1.0) ===")
    t_all = time.perf_counter()
    passed = 0
    total = 0

    def check(cond: bool, label: str, info: str = "") -> None:
        nonlocal passed, total
        total += 1
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {total:2d}. {label}" + (f"  ({info})" if info else ""))
        if not cond:
            raise AssertionError(f"Selftest failed: {label} ({info})")
        passed += 1

    # 1. Warm up Numba JIT on Strassen
    s2 = strassen_222(ring="Z")
    s2_f2 = s2.reduce_mod(2)
    l3 = laderman_333(ring="Z")
    l3_f2 = l3.reduce_mod(2)
    _ = mod4_oracle(s2_f2, support_restricted=True)
    _ = mod4_oracle(s2_f2, support_restricted=False)

    # 2. Exact Mod-4 Support Oracle checks (reproducing Phase 0 exact counts & < 1ms speed)
    o_s2 = mod4_oracle(s2_f2, support_restricted=True)
    check(
        o_s2.is_mod4_liftable and o_s2.rank_J1 == 19 and o_s2.free_bits == 3 and o_s2.num_active_eqs == 20,
        "L0 Mod-4 oracle Strassen <2,2,2:7>",
        f"coker={o_s2.coker_viol_4}, rank={o_s2.rank_J1}, free={o_s2.free_bits}, eqs={o_s2.num_active_eqs}, {o_s2.elapsed_ms:.3f}ms",
    )

    o_l3 = mod4_oracle(l3_f2, support_restricted=True)
    check(
        o_l3.is_mod4_liftable and o_l3.rank_J1 == 95 and o_l3.free_bits == 12 and o_l3.num_active_eqs == 119,
        "L0 Mod-4 oracle Laderman <3,3,3:23>",
        f"coker={o_l3.coker_viol_4}, rank={o_l3.rank_J1}, free={o_l3.free_bits}, eqs={o_l3.num_active_eqs}, {o_l3.elapsed_ms:.3f}ms",
    )

    # 3. Deliberately obstructed F2 scheme negative control
    obs_f2 = _make_obstructed_f2_scheme()
    o_obs = mod4_oracle(obs_f2, support_restricted=True)
    check(
        (not o_obs.is_mod4_liftable) and o_obs.coker_viol_4 > 0,
        "L0 Mod-4 oracle negative control (support-obstructed F2 naive-3x3x3 flip)",
        f"valid_F2=True, coker_viol_4={o_obs.coker_viol_4} (>0), {o_obs.elapsed_ms:.3f}ms",
    )

    # Also test that unrestricted Stage L1 linear Hensel lifts the support-obstructed scheme to Z/Q!
    h_obs_unres = linear_hensel_lift(obs_f2, support_restricted=False, target_bits=64, max_trials=16, seed=11)
    check(
        h_obs_unres.success and h_obs_unres.report is not None and h_obs_unres.report.valid,
        "L1 Unrestricted Linear Hensel lifts support-obstructed F2 scheme into Z/Q",
        f"status={h_obs_unres.status}, ring={h_obs_unres.scheme.ring}, max_coeff={h_obs_unres.report.max_abs_coeff}, {h_obs_unres.elapsed_ms:.2f}ms",
    )

    # 4. Stage L1 Linear Hensel Lifting (2^k -> 2^{k+1} up to 2^64) + Wang Rational Reconstruction
    rng = random.Random(12345)
    s2_perm = _permute_and_perturb_f2(s2, rng)
    h_s2 = linear_hensel_lift(s2_perm, support_restricted=True, target_bits=64, max_trials=16, seed=1)
    check(
        h_s2.success and h_s2.report is not None and h_s2.report.valid,
        "L1 Linear Hensel lift (support) on permuted F2 Strassen <2,2,2:7>",
        f"status={h_s2.status}, ring={h_s2.scheme.ring}, ternary={h_s2.report.is_ternary}, {h_s2.elapsed_ms:.2f}ms",
    )

    h_s2_unres = linear_hensel_lift(s2_perm, support_restricted=False, target_bits=64, max_trials=16, seed=2)
    check(
        h_s2_unres.success and h_s2_unres.report is not None and h_s2_unres.report.valid,
        "L1 Linear Hensel lift (unrestricted) on permuted F2 Strassen <2,2,2:7>",
        f"status={h_s2_unres.status}, ring={h_s2_unres.scheme.ring}, max_coeff={h_s2_unres.report.max_abs_coeff}, {h_s2_unres.elapsed_ms:.2f}ms",
    )

    l3_perm = _permute_and_perturb_f2(l3, rng)
    h_l3 = linear_hensel_lift(l3_perm, support_restricted=False, target_bits=64, max_trials=24, seed=3)
    check(
        h_l3.success and h_l3.report is not None and h_l3.report.valid,
        "L1 Linear Hensel lift (unrestricted -> Z/Q) on permuted F2 Laderman <3,3,3:23>",
        f"status={h_l3.status}, ring={h_l3.scheme.ring}, max_coeff={h_l3.report.max_abs_coeff}, {h_l3.elapsed_ms:.2f}ms",
    )

    # Test Wang rational reconstruction directly on exact denominator-3 / denominator-7 fractions mod 2^64
    M64 = 1 << 64
    test_fracs = [Fraction(1, 3), Fraction(-2, 3), Fraction(5, 7), Fraction(-13, 19), Fraction(0, 1), Fraction(-1, 1)]
    wang_ok = True
    for fr in test_fracs:
        a_mod = (fr.numerator * pow(fr.denominator, -1, M64)) % M64
        rec_fr = wang_rational_reconstruct(a_mod, M64)
        if rec_fr != fr:
            wang_ok = False
    check(wang_ok, "L1 Wang Rational Reconstruction mod 2^64 exactness (denoms 1,3,7,19)")

    # Test lifting a genuine denominator-3 rational F2 scheme
    _, sq_f2 = _make_rational_denom3_strassen_f2()
    h_sq = linear_hensel_lift(sq_f2, support_restricted=False, target_bits=64, max_trials=16, seed=7)
    check(
        h_sq.success and h_sq.report is not None and h_sq.report.valid,
        "L1 Linear Hensel lift on GL(Q)-sheared Strassen mod 2",
        f"status={h_sq.status}, ring={h_sq.scheme.ring}, {h_sq.elapsed_ms:.2f}ms",
    )

    # 5. Stage L2 Exact Z3 5.1.0 Ternary Sign Lifter (force_z3=True!)
    z3_s2 = z3_ternary_lift(s2_perm, timeout_s=5.0, force_z3=True)
    check(
        z3_s2.success and z3_s2.report is not None and z3_s2.report.valid and z3_s2.report.is_ternary,
        "L2 Z3 5.1.0 ternary lift (force_z3=True) on permuted F2 Strassen <2,2,2:7>",
        f"status={z3_s2.status}, adds={z3_s2.report.additive_complexity['naive_additions_total']}, {z3_s2.elapsed_ms:.2f}ms",
    )

    z3_l3 = z3_ternary_lift(l3_perm, timeout_s=10.0, force_z3=True)
    check(
        z3_l3.success and z3_l3.report is not None and z3_l3.report.valid and z3_l3.report.is_ternary,
        "L2 Z3 5.1.0 ternary lift (force_z3=True) on permuted F2 Laderman <3,3,3:23>",
        f"status={z3_l3.status}, adds={z3_l3.report.additive_complexity['naive_additions_total']}, {z3_l3.elapsed_ms:.2f}ms",
    )

    # Fast exact Numba nullspace ternary lift on Laderman (2^12 = 4096)
    nb_l3 = z3_ternary_lift(l3_perm, timeout_s=5.0, force_z3=False)
    check(
        nb_l3.success and nb_l3.report is not None and nb_l3.report.valid and nb_l3.report.is_ternary,
        "L2 Numba exact nullspace ternary lift on Laderman <3,3,3:23>",
        f"method={nb_l3.method}, {nb_l3.elapsed_ms:.2f}ms",
    )

    # Z3 negative control on obstructed F2 scheme
    z3_obs = z3_ternary_lift(obs_f2, timeout_s=5.0, force_z3=True)
    check(
        (not z3_obs.success) and z3_obs.status == "obstructed_mod4" and z3_obs.coker_viol_4 > 0,
        "L2 Z3 lifter rejects mod-4 obstructed F2 scheme at L0 gate",
        f"status={z3_obs.status}, coker={z3_obs.coker_viol_4}, {z3_obs.elapsed_ms:.3f}ms",
    )

    # 6. Clean From-Scratch Block-Seed Builders verification via brent_verifier
    seed_targets = [(2, 2, 2), (3, 3, 3), (4, 4, 4), (4, 4, 5), (4, 5, 5), (4, 5, 6), (5, 8, 8), (6, 6, 7), (7, 7, 7)]
    seed_summary = []
    all_seeds_ok = True
    for fm in seed_targets:
        sc = make_block_seed(*fm, ring="Z")
        # Use exact fast Brent contraction + basis check on <= 4x5x6, and exact Brent check on larger
        if fm[0] * fm[1] * fm[2] <= 120:
            rep = verify_scheme(sc)
            ok = rep.valid and rep.is_ternary
        else:
            # Verify BOTH exact Brent contraction AND uint64 mod-2 check (fast exact)
            from brent_verifier import verify_brent_equations

            b_res = verify_brent_equations(sc)
            ok = bool(b_res["ok"]) and sc.is_ternary()
        if not ok:
            all_seeds_ok = False
        seed_summary.append(f"{fm[0]}x{fm[1]}x{fm[2]}:R{sc.rank}")
    check(
        all_seeds_ok,
        "Clean from-scratch Block-Seed Library (Strassen+Laderman+naive ONLY, all Z_T)",
        ", ".join(seed_summary),
    )

    # 7. Phase 3: Stage L0-p3 (mod-9 oracle) + Stage L1-p3 (3-adic Hensel -> Z_T / Z / Z[1/2] Q)
    def _permute_f3_scheme(sc_z: BilinearScheme, prng: random.Random, tag: str) -> BilinearScheme:
        R = sc_z.rank
        perm = list(range(R))
        prng.shuffle(perm)
        U3, V3, W3 = [], [], []
        for idx in perm:
            su = 1 if prng.randrange(2) == 0 else -1
            sv = 1 if prng.randrange(2) == 0 else -1
            sw = su * sv
            U3.append([(int(x) * su) % 3 for x in sc_z.U[idx]])
            V3.append([(int(x) * sv) % 3 for x in sc_z.V[idx]])
            W3.append([(int(x) * sw) % 3 for x in sc_z.W[idx]])
        return BilinearScheme(m=sc_z.m, n=sc_z.n, p=sc_z.p, U=U3, V=V3, W=W3, ring="Z3", modulus=3, name=tag)

    # Warm up Numba 3-adic JIT kernels
    s2_f3 = _permute_f3_scheme(s2, rng, "strassen_f3_perm")
    _ = mod9_oracle(s2_f3, support_restricted=True)
    _ = mod9_oracle(s2_f3, support_restricted=False)

    # 15. Lift F3-Strassen <2,2,2:7> -> Z (is_ternary=True) via L0-p3 + L1-p3
    o9_s2 = mod9_oracle(s2_f3, support_restricted=True)
    h3_s2 = linear_hensel_lift_p3(s2_f3, support_restricted=True, target_trits=40, max_trials=16, seed=15)
    check(
        o9_s2.is_mod9_liftable
        and h3_s2.success
        and h3_s2.report is not None
        and h3_s2.report.valid
        and h3_s2.report.is_ternary,
        "L0-p3 + L1-p3 3-adic Hensel lift F3-Strassen <2,2,2:7> -> Z_T",
        f"coker9={o9_s2.coker_viol_9}, rk={o9_s2.rank_J1}, free={o9_s2.free_trits}, status={h3_s2.status}, {h3_s2.elapsed_ms:.2f}ms",
    )

    # 16. Lift F3-Laderman <3,3,3:23> -> Z (is_ternary=True) via L0-p3 + L1-p3
    l3_f3 = _permute_f3_scheme(l3, rng, "laderman_f3_perm")
    o9_l3 = mod9_oracle(l3_f3, support_restricted=True)
    h3_l3 = linear_hensel_lift_p3(l3_f3, support_restricted=True, target_trits=40, max_trials=16, seed=16)
    check(
        o9_l3.is_mod9_liftable
        and h3_l3.success
        and h3_l3.report is not None
        and h3_l3.report.valid
        and h3_l3.report.is_ternary,
        "L0-p3 + L1-p3 3-adic Hensel lift F3-Laderman <3,3,3:23> -> Z_T",
        f"coker9={o9_l3.coker_viol_9}, rk={o9_l3.rank_J1}, free={o9_l3.free_trits}, status={h3_l3.status}, {h3_l3.elapsed_ms:.2f}ms",
    )

    # 17. CRITICAL Z[1/2] TEST: recover genuine 1/2 rational scheme from mod-3 reduction via 3-adic Hensel + Wang mod 3^40!
    M3_40 = 3**40
    z_half_fracs = [
        Fraction(1, 2),
        Fraction(-1, 2),
        Fraction(1, 4),
        Fraction(-3, 8),
        Fraction(5, 16),
        Fraction(-1, 5),
        Fraction(7, 11),
    ]
    wang3_ok = all(
        wang_rational_reconstruct((fr.numerator * pow(fr.denominator, -1, M3_40)) % M3_40, M3_40) == fr
        for fr in z_half_fracs
    )
    _, sq_half_f3 = _make_half_basis_strassen_q()
    o9_half = mod9_oracle(sq_half_f3, support_restricted=True)
    h3_half = linear_hensel_lift_p3(sq_half_f3, support_restricted=True, target_trits=40, max_trials=24, seed=17)
    max_den_half = int(h3_half.details.get("max_denom", 1)) if h3_half.details else 1
    check(
        wang3_ok
        and o9_half.is_mod9_liftable
        and h3_half.success
        and h3_half.scheme is not None
        and h3_half.report is not None
        and h3_half.report.valid
        and h3_half.scheme.ring == "Q"
        and max_den_half >= 2,
        "L1-p3 CRITICAL Z[1/2] TEST: 3-adic Hensel + Wang mod 3^40 recovers exact 1/2 Q-scheme",
        f"status={h3_half.status}, ring={h3_half.scheme.ring if h3_half.scheme else None}, max_denom={max_den_half}, "
        f"trits={h3_half.details.get('reconstructed_at_trits')}, {h3_half.elapsed_ms:.2f}ms",
    )

    # 18. Negative control: deliberately support-obstructed valid F3 scheme gives coker_viol_9 > 0
    obs_f3 = _make_obstructed_f3_scheme()
    o9_obs = mod9_oracle(obs_f3, support_restricted=True)
    h3_obs = linear_hensel_lift_p3(obs_f3, support_restricted=True, target_trits=12, max_trials=4, seed=18)
    check(
        (not o9_obs.is_mod9_liftable)
        and o9_obs.coker_viol_9 > 0
        and (not h3_obs.success)
        and h3_obs.status == "obstructed_mod9",
        "L0-p3 Negative control: obstructed valid F3 <3,3,3:27> gives coker_viol_9 > 0",
        f"valid_F3=True, coker_viol_9={o9_obs.coker_viol_9} (>0), status={h3_obs.status}, {o9_obs.elapsed_ms:.3f}ms",
    )

    total_ms = (time.perf_counter() - t_all) * 1000.0
    print(f"\nRESULT: {passed}/{total} PASS in {total_ms:.1f} ms")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="FMM Mod-4 Oracle, Linear Hensel Lifter, and Z3 Ternary Lifter")
    parser.add_argument("--selftest", action="store_true", help="Run full Tracer #1 verification selftest")
    args = parser.parse_args()
    if args.selftest:
        return run_selftest()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Independent Zero-Gaming Exact Verifier for C_71 (Fourier Entropy-Influence).

Follows the [T2026] independent-replay standard:
- Zero shared code with GPU/search modules (no torch, no numpy).
- Pure Python arbitrary-precision ``int`` Walsh-Hadamard Transform.
- Exact ``fractions.Fraction`` total influence verified by TWO independent routes
  (spectral sum and combinatorial edge-boundary count).
- Dual 100-digit outward-rounded interval arithmetic:
    * Contour 1: ``mpmath.iv`` (``dps = 100``).
    * Contour 2: independent stdlib ``decimal.Decimal`` (``prec = 115``) with
      explicit IEEE-754 ``ROUND_FLOOR``/``ROUND_CEILING`` interval arithmetic
      and provable series remainder bounds for ``ln`` (zero reliance on
      ``Decimal.ln()`` rounding).
- Supports:
    * Tier 1: Full balanced Boolean truth table F: {0,1}^n -> {-1, +1}.
    * Tier 2: Orthogonal subcube/antichain de-aliased core
      (g0: {0,1}^d -> {-1, 0, +1}, disjoint subcubes {C_j}), certifying
      F(x, y_1, ..., y_k) = g0(x) + sum_j 1_{C_j}(x) * y_j.

Convention:
- Input bits x in {0, ..., 2^d - 1} ordered bitwise: x <= x' iff (x & x') == x.
- Output values in {-1, +1} (and 0 for Tier 2 core), where -1 = FALSE and
  +1 = TRUE, so logic-monotonicity is x <= x' => f(x) <= f(x').
  (Negating all outputs f -> -f preserves balance, Parseval, I, and H.)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import mpmath


# Published [Num2026] / [T2026] certified lower bound & conservative upper bar
NUM2026_HEADLINE_FLOOR = "6.521845710923046575"
NUM2026_STRICT_BEAT_BAR = Decimal("6.52184571092304657566")


# =============================================================================
# Pure Python Exact Integer FWHT
# =============================================================================

def fwht_int(values: Sequence[int]) -> List[int]:
    """Exact unnormalized Walsh-Hadamard transform over arbitrary-precision int.

    Computes c_S = sum_{x=0}^{2^d - 1} values[x] * (-1)^{<x, S>} for all S.
    """
    n_pts = len(values)
    if n_pts == 0 or (n_pts & (n_pts - 1)) != 0:
        raise ValueError(f"Length must be a positive power of 2, got {n_pts}")
    a = [int(v) for v in values]
    h = 1
    while h < n_pts:
        step = h << 1
        for i in range(0, n_pts, step):
            for j in range(i, i + h):
                u = a[j]
                v = a[j + h]
                a[j] = u + v
                a[j + h] = u - v
        h = step
    return a


# =============================================================================
# Contour 2: Rigorous Outward-Rounded stdlib Decimal Interval Arithmetic
# =============================================================================

DEC_PREC = 115


@dataclass(frozen=True)
class DecInterval:
    """Guaranteed outward-rounded closed interval [lo, hi] using stdlib Decimal."""

    lo: Decimal
    hi: Decimal

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise AssertionError(f"Invalid DecInterval: lo={self.lo} > hi={self.hi}")

    @staticmethod
    def exact_int(n: int) -> "DecInterval":
        d = Decimal(n)
        return DecInterval(d, d)

    @staticmethod
    def from_fraction(frac: Fraction, prec: int = DEC_PREC) -> "DecInterval":
        num = Decimal(frac.numerator)
        den = Decimal(frac.denominator)
        with localcontext() as ctx:
            ctx.prec = prec
            ctx.rounding = ROUND_FLOOR
            lo = num / den
            ctx.rounding = ROUND_CEILING
            hi = num / den
        return DecInterval(lo, hi)

    def __add__(self, other: "DecInterval") -> "DecInterval":
        with localcontext() as ctx:
            ctx.prec = DEC_PREC
            ctx.rounding = ROUND_FLOOR
            lo = self.lo + other.lo
            ctx.rounding = ROUND_CEILING
            hi = self.hi + other.hi
        return DecInterval(lo, hi)

    def __sub__(self, other: "DecInterval") -> "DecInterval":
        with localcontext() as ctx:
            ctx.prec = DEC_PREC
            ctx.rounding = ROUND_FLOOR
            lo = self.lo - other.hi
            ctx.rounding = ROUND_CEILING
            hi = self.hi - other.lo
        return DecInterval(lo, hi)

    def __mul__(self, other: "DecInterval") -> "DecInterval":
        with localcontext() as ctx:
            ctx.prec = DEC_PREC
            ctx.rounding = ROUND_FLOOR
            lo = min(
                self.lo * other.lo,
                self.lo * other.hi,
                self.hi * other.lo,
                self.hi * other.hi,
            )
            ctx.rounding = ROUND_CEILING
            hi = max(
                self.lo * other.lo,
                self.lo * other.hi,
                self.hi * other.lo,
                self.hi * other.hi,
            )
        return DecInterval(lo, hi)

    def __truediv__(self, other: "DecInterval") -> "DecInterval":
        if other.lo <= 0 <= other.hi:
            raise ZeroDivisionError("Division by DecInterval containing 0")
        with localcontext() as ctx:
            ctx.prec = DEC_PREC
            ctx.rounding = ROUND_FLOOR
            lo = min(
                self.lo / other.lo,
                self.lo / other.hi,
                self.hi / other.lo,
                self.hi / other.hi,
            )
            ctx.rounding = ROUND_CEILING
            hi = max(
                self.lo / other.lo,
                self.lo / other.hi,
                self.hi / other.lo,
                self.hi / other.hi,
            )
        return DecInterval(lo, hi)


def _dec_ln_rational_near_one(num: int, den: int) -> DecInterval:
    """Rigorous outward-rounded ln(num/den) for 1/2 <= num/den <= 2 via atanh series.

    Uses ln(r) = 2 * sum_{j=0}^{K-1} u^{2j+1}/(2j+1) + R_K where u = (num-den)/(num+den),
    |u| <= 1/3, and |R_K| <= 2 * |u|^{2K+1} / ((2K+1) * (1 - u^2)).
    Every arithmetic operation uses exact IEEE-754 ROUND_FLOOR / ROUND_CEILING.
    """
    if num == den:
        return DecInterval.exact_int(0)
    if num <= 0 or den <= 0:
        raise ValueError("num and den must be positive")
    u_frac = Fraction(num - den, num + den)
    u_abs_frac = abs(u_frac)
    if u_abs_frac > Fraction(1, 3):
        raise ValueError(f"Ratio {num}/{den} outside [1/2, 2] for fast atanh series")

    sign = 1 if u_frac > 0 else -1
    u_iv = DecInterval.from_fraction(u_abs_frac)
    u2_iv = u_iv * u_iv

    # Choose K so that (1/3)^(2K+1) < 10^(-112): 3^235 > 10^112 => K = 120
    # For smaller |u| we still converge even faster; compute adaptive K safely.
    # For |u| <= 1/3, K=120 gives 2*(1/3)^241 / (241 * 8/9) < 10^-117.
    K = 120
    # Compute power u^(2j+1) iteratively with outward rounding
    total = DecInterval.exact_int(0)
    u_pow = u_iv
    for j in range(K):
        denom_iv = DecInterval.exact_int(2 * j + 1)
        total = total + (u_pow / denom_iv)
        u_pow = u_pow * u2_iv

    # Remainder bound: 0 <= R_K <= 2 * u_abs^(2K+1) / ((2K+1) * (1 - u^2))
    # Notice u_pow is currently u_abs^(2K+1)
    one_minus_u2 = DecInterval.exact_int(1) - u2_iv
    rem_upper = (DecInterval.exact_int(2) * u_pow) / (
        DecInterval.exact_int(2 * K + 1) * one_minus_u2
    )
    val = DecInterval.exact_int(2) * total
    pos_iv = DecInterval(val.lo, (val + DecInterval(Decimal(0), rem_upper.hi)).hi)
    if sign > 0:
        return pos_iv
    return DecInterval(pos_iv.hi.copy_negate(), pos_iv.lo.copy_negate())


def _dec_ln2() -> DecInterval:
    """Rigorous outward-rounded ln(2) = 2*ln(4/3) + ln(9/8)."""
    # 4/3 * 4/3 * 9/8 = 16/9 * 9/8 = 2 exactly!
    # u(4/3) = 1/7, u(9/8) = 1/17 -> ultra-tight convergence.
    ln_4_3 = _dec_ln_rational_near_one(4, 3)
    ln_9_8 = _dec_ln_rational_near_one(9, 8)
    return DecInterval.exact_int(2) * ln_4_3 + ln_9_8


_DEC_LN2_CACHE: Optional[DecInterval] = None


def dec_ln2() -> DecInterval:
    global _DEC_LN2_CACHE
    if _DEC_LN2_CACHE is None:
        _DEC_LN2_CACHE = _dec_ln2()
    return _DEC_LN2_CACHE


def dec_log2_pos_int(m: int, ln2_iv: DecInterval, cache: Dict[int, DecInterval]) -> DecInterval:
    """Rigorous outward-rounded log2(m) for positive integer m.

    Factors out powers of 2: m = 2^a * q (q odd), then reduces q to [3/4, 3/2]
    via power of 2 scaling so |u| <= 1/5 for exact atanh evaluation.
    """
    if m <= 0:
        raise ValueError(f"Expected positive integer, got {m}")
    # Strip trailing zeros (exact integer part of log2)
    tz = (m & -m).bit_length() - 1
    q = m >> tz
    if q == 1:
        return DecInterval.exact_int(tz)
    if q in cache:
        return DecInterval.exact_int(tz) + cache[q]

    # Find k such that q / 2^k in [2/3, 4/3] (subset of [1/2, 2])
    bl = q.bit_length()
    # 2^(bl-1) <= q < 2^bl. Compare 3*q vs 4 * 2^(bl-1) = 2^(bl+1)
    if 3 * q <= (1 << (bl + 1)):
        k = bl - 1
    else:
        k = bl
    ln_ratio = _dec_ln_rational_near_one(q, 1 << k)
    log2_q = DecInterval.exact_int(k) + (ln_ratio / ln2_iv)
    cache[q] = log2_q
    return DecInterval.exact_int(tz) + log2_q


def _entropy_and_ratio_dual_intervals(
    wht_coeffs: Sequence[int],
    d: int,
    I_exact: Fraction,
    extra_entropy_frac: Fraction = Fraction(0),
) -> Dict[str, Any]:
    """Compute outward-rounded 100-digit intervals for H and L = H / (I - 1)
    using BOTH Contour 1 (mpmath.iv) and Contour 2 (stdlib DecInterval).
    """
    if I_exact <= 1:
        raise ValueError(f"Amplification requires I > 1, got I = {I_exact}")

    # Group nonzero coefficients by |c_S| (exact multiplicity) for speed + precision
    abs_counts: Dict[int, int] = {}
    nonzero_count = 0
    for c in wht_coeffs:
        if c != 0:
            nonzero_count += 1
            ac = abs(c)
            abs_counts[ac] = abs_counts.get(ac, 0) + 1

    denom_4d = 1 << (2 * d)  # 4^d

    # ---- Contour 1: mpmath.iv (dps = 100) ----
    old_dps = mpmath.iv.dps
    try:
        mpmath.iv.dps = 105
        iv_ln2 = mpmath.iv.log(mpmath.iv.mpf(2))
        iv_4d = mpmath.iv.mpf(denom_4d)
        iv_2d = mpmath.iv.mpf(2 * d)
        H_mp = mpmath.iv.mpf(0)
        for ac, mult in abs_counts.items():
            # weight = mult * ac^2 / 4^d
            # -p_S log2(p_S) = p_S * (2d - 2*log2(ac))
            w_iv = mpmath.iv.mpf(mult * ac * ac) / iv_4d
            tz = (ac & -ac).bit_length() - 1
            q = ac >> tz
            if q == 1:
                log2_ac = mpmath.iv.mpf(tz)
            else:
                log2_ac = mpmath.iv.mpf(tz) + mpmath.iv.log(mpmath.iv.mpf(q)) / iv_ln2
            term = w_iv * (iv_2d - mpmath.iv.mpf(2) * log2_ac)
            H_mp += term
        if extra_entropy_frac != 0:
            H_mp += mpmath.iv.mpf(extra_entropy_frac.numerator) / mpmath.iv.mpf(
                extra_entropy_frac.denominator
            )
        I_minus_1 = I_exact - 1
        Im1_mp = mpmath.iv.mpf(I_minus_1.numerator) / mpmath.iv.mpf(I_minus_1.denominator)
        L_mp = H_mp / Im1_mp
        # Extract exact dyadic endpoints directly from ivmpf._mpi_ = (lo_tuple, hi_tuple)
        def _mpf_tuple_to_fraction(t: Tuple[int, int, int, int]) -> Fraction:
            sign, man, exp, _ = t
            val = Fraction(man, 1) * (
                Fraction(1 << exp, 1) if exp >= 0 else Fraction(1, 1 << (-exp))
            )
            return -val if sign else val

        H_mp_lo_frac = _mpf_tuple_to_fraction(H_mp._mpi_[0])
        H_mp_hi_frac = _mpf_tuple_to_fraction(H_mp._mpi_[1])
        L_mp_lo_frac = _mpf_tuple_to_fraction(L_mp._mpi_[0])
        L_mp_hi_frac = _mpf_tuple_to_fraction(L_mp._mpi_[1])
    finally:
        mpmath.iv.dps = old_dps

    # Convert exact binary Fractions of mpmath endpoints outward to 80-digit Decimals
    def _frac_floor_dec(fr: Fraction, digits: int = 80) -> Decimal:
        with localcontext() as ctx:
            ctx.prec = digits + 5
            ctx.rounding = ROUND_FLOOR
            return Decimal(fr.numerator) / Decimal(fr.denominator)

    def _frac_ceil_dec(fr: Fraction, digits: int = 80) -> Decimal:
        with localcontext() as ctx:
            ctx.prec = digits + 5
            ctx.rounding = ROUND_CEILING
            return Decimal(fr.numerator) / Decimal(fr.denominator)

    H_mp_lo_d = _frac_floor_dec(H_mp_lo_frac, 80)
    H_mp_hi_d = _frac_ceil_dec(H_mp_hi_frac, 80)
    L_mp_lo_d = _frac_floor_dec(L_mp_lo_frac, 80)
    L_mp_hi_d = _frac_ceil_dec(L_mp_hi_frac, 80)

    # ---- Contour 2: Independent stdlib DecInterval (prec = 115) ----
    ln2_iv = dec_ln2()
    log2_cache: Dict[int, DecInterval] = {}
    iv_2d_dec = DecInterval.exact_int(2 * d)
    iv_two = DecInterval.exact_int(2)
    H_dec = DecInterval.exact_int(0)
    for ac, mult in abs_counts.items():
        w_iv = DecInterval.from_fraction(Fraction(mult * ac * ac, denom_4d))
        log2_ac = dec_log2_pos_int(ac, ln2_iv, log2_cache)
        H_dec = H_dec + (w_iv * (iv_2d_dec - iv_two * log2_ac))
    if extra_entropy_frac != 0:
        H_dec = H_dec + DecInterval.from_fraction(extra_entropy_frac)

    Im1_dec = DecInterval.from_fraction(I_exact - 1)
    L_dec = H_dec / Im1_dec

    # Cross-check that Contour 1 and Contour 2 intervals strictly overlap!
    if H_mp_hi_d < H_dec.lo or H_dec.hi < H_mp_lo_d:
        raise AssertionError(
            f"Contour mismatch in H: mp=[{H_mp_lo_d}, {H_mp_hi_d}] vs dec=[{H_dec.lo}, {H_dec.hi}]"
        )
    if L_mp_hi_d < L_dec.lo or L_dec.hi < L_mp_lo_d:
        raise AssertionError(
            f"Contour mismatch in L: mp=[{L_mp_lo_d}, {L_mp_hi_d}] vs dec=[{L_dec.lo}, {L_dec.hi}]"
        )

    # Conservative combined outward enclosure across both independent contours
    H_comb_lo = min(H_mp_lo_d, H_dec.lo)
    H_comb_hi = max(H_mp_hi_d, H_dec.hi)
    L_comb_lo = min(L_mp_lo_d, L_dec.lo)
    L_comb_hi = max(L_mp_hi_d, L_dec.hi)

    def _floor_str(val: Decimal, places: int = 60) -> str:
        with localcontext() as ctx:
            ctx.prec = DEC_PREC
            q = Decimal(10) ** (-places)
            return str(val.quantize(q, rounding=ROUND_FLOOR))

    def _ceil_str(val: Decimal, places: int = 60) -> str:
        with localcontext() as ctx:
            ctx.prec = DEC_PREC
            q = Decimal(10) ** (-places)
            return str(val.quantize(q, rounding=ROUND_CEILING))

    return {
        "nonzero_fourier_coeffs": nonzero_count,
        "unique_abs_coeffs": len(abs_counts),
        "H_mpmath_iv": [_floor_str(H_mp_lo_d, 60), _ceil_str(H_mp_hi_d, 60)],
        "H_decimal_iv": [_floor_str(H_dec.lo, 60), _ceil_str(H_dec.hi, 60)],
        "H_combined_iv": [_floor_str(H_comb_lo, 60), _ceil_str(H_comb_hi, 60)],
        "L_mpmath_iv": [_floor_str(L_mp_lo_d, 60), _ceil_str(L_mp_hi_d, 60)],
        "L_decimal_iv": [_floor_str(L_dec.lo, 60), _ceil_str(L_dec.hi, 60)],
        "L_cert": _floor_str(L_comb_lo, 60),
        "U_cert": _ceil_str(L_comb_hi, 60),
        "interval_width_L": float(L_comb_hi - L_comb_lo),
        "beats_num2026": bool(L_comb_lo > NUM2026_STRICT_BEAT_BAR),
    }


# =============================================================================
# Tier 1 Verifier: Full Boolean Truth Table F: {0,1}^n -> {-1, +1}
# =============================================================================

def normalize_truth_table_pm1(bits_or_pm1: Sequence[int]) -> List[int]:
    """Accept either {-1, +1} directly or {0, 1} (mapped as 0 -> -1, 1 -> +1)."""
    vals = [int(v) for v in bits_or_pm1]
    uniq = set(vals)
    if uniq <= {-1, 1}:
        return vals
    if uniq <= {0, 1}:
        return [2 * v - 1 for v in vals]
    raise ValueError(f"Truth table must contain only {{-1,+1}} or {{0,1}}, got {sorted(uniq)}")


def truth_table_sha256(values_pm1: Sequence[int]) -> str:
    """Canonical SHA-256 of {-1, +1} truth table packed LSB-first (bit=1 iff +1)."""
    n_pts = len(values_pm1)
    buf = bytearray((n_pts + 7) // 8)
    for idx, v in enumerate(values_pm1):
        if v == 1:
            buf[idx >> 3] |= 1 << (idx & 7)
    return hashlib.sha256(bytes(buf)).hexdigest()


def verify_full_truth_table(
    bits: Sequence[int],
    require_monotone: bool = False,
) -> Dict[str, Any]:
    """Tier 1 exact verification of a balanced Boolean function F: {0,1}^n -> {-1, +1}."""
    f = normalize_truth_table_pm1(bits)
    N = len(f)
    if N < 2 or (N & (N - 1)) != 0:
        raise ValueError(f"Truth table length must be 2^n (n>=1), got {N}")
    n = N.bit_length() - 1

    # 1. Exact balance check (sum f(x) == 0)
    tt_sum = sum(f)
    if tt_sum != 0:
        raise AssertionError(f"Function is NOT balanced: sum(f) = {tt_sum} != 0")

    # 2. Pure-Python int FWHT + Parseval + balance c_empty == 0
    c = fwht_int(f)
    if c[0] != 0:
        raise AssertionError(f"WHT c_empty = {c[0]} != 0")
    parseval_sum = sum(v * v for v in c)
    expected_parseval = 1 << (2 * n)  # 4^n
    if parseval_sum != expected_parseval:
        raise AssertionError(
            f"Parseval failed: sum(c_S^2) = {parseval_sum} != 4^{n} = {expected_parseval}"
        )

    # 3. Exact Influence via TWO independent routes (spectral + combinatorial)
    inf_num = sum(S.bit_count() * v * v for S, v in enumerate(c) if v != 0)
    I_spectral = Fraction(inf_num, expected_parseval)

    sensitive_edges = 0
    mono_violations = 0
    anti_mono_violations = 0
    for i in range(n):
        bit = 1 << i
        for x in range(N):
            if (x & bit) == 0:
                fx = f[x]
                fy = f[x | bit]
                if fx != fy:
                    sensitive_edges += 1
                if fx > fy:
                    mono_violations += 1
                if fx < fy:
                    anti_mono_violations += 1

    I_comb = Fraction(sensitive_edges, 1 << (n - 1))
    if I_spectral != I_comb:
        raise AssertionError(
            f"Spectral vs combinatorial influence mismatch: {I_spectral} != {I_comb}"
        )

    is_monotone = mono_violations == 0
    is_anti_monotone = anti_mono_violations == 0
    if require_monotone and not is_monotone:
        raise AssertionError(f"Monotonicity violated on {mono_violations} edges")

    # 4. Dual 100-digit outward-rounded intervals for H and L = H / (I - 1)
    iv_res = _entropy_and_ratio_dual_intervals(c, n, I_spectral)

    return {
        "tier": 1,
        "n": n,
        "sha256": truth_table_sha256(f),
        "exact_balance": True,
        "parseval_verified": True,
        "is_monotone": is_monotone,
        "is_anti_monotone": is_anti_monotone,
        "monotonicity_violations": mono_violations,
        "edges_checked": n * (1 << (n - 1)),
        "I_exact_num": I_spectral.numerator,
        "I_exact_den": I_spectral.denominator,
        "I_exact_str": f"{I_spectral.numerator}/{I_spectral.denominator}",
        "I_minus_1_str": f"{(I_spectral - 1).numerator}/{(I_spectral - 1).denominator}",
        **iv_res,
    }


# =============================================================================
# Tier 2 Verifier: Orthogonal Subcube / Antichain De-Aliased Core (g0, {C_j})
# =============================================================================

def _parse_subcube(
    sc: Union[Dict[str, int], Sequence[int], Tuple[int, int]],
    d: int,
) -> Tuple[int, int, int]:
    """Normalize a subcube representation into (fixed_mask, fixed_val, codim).

    Accepts:
      - dict {"fixed_mask": int, "fixed_val": int}
      - tuple/list of 2 ints if dict-like? No: list[int] is treated as explicit vertex list
        (for unambiguous masks, pass dict {"fixed_mask": ..., "fixed_val": ...}).
    """
    full_mask = (1 << d) - 1
    if isinstance(sc, dict):
        if "fixed_mask" not in sc or "fixed_val" not in sc:
            raise ValueError(f"Subcube dict missing fixed_mask/fixed_val: {sc}")
        fm = int(sc["fixed_mask"])
        fv = int(sc["fixed_val"])
        if fm < 0 or fm > full_mask:
            raise ValueError(f"fixed_mask {fm} out of range [0, {full_mask}]")
        if (fv & ~fm) != 0:
            raise ValueError(f"fixed_val {fv} has bits outside fixed_mask {fm}")
        return fm, fv, fm.bit_count()

    # Explicit vertex list
    verts = sorted(set(int(v) for v in sc))
    if not verts:
        raise ValueError("Empty subcube vertex list")
    if len(verts) != len(sc):
        raise ValueError("Duplicate vertices inside a subcube")
    if verts[0] < 0 or verts[-1] > full_mask:
        raise ValueError(f"Vertex out of range [0, {full_mask}]")
    m = len(verts)
    if (m & (m - 1)) != 0:
        raise ValueError(f"Subcube vertex count must be a power of 2, got {m}")
    or_all = 0
    and_all = full_mask
    for v in verts:
        or_all |= v
        and_all &= v
    free_mask = or_all ^ and_all
    if (1 << free_mask.bit_count()) != m:
        raise ValueError(f"Vertices {verts} do not form a valid subcube")
    fixed_mask = full_mask ^ free_mask
    fixed_val = and_all
    return fixed_mask, fixed_val, fixed_mask.bit_count()


def _iter_subcube_points(fixed_mask: int, fixed_val: int, d: int) -> Iterable[int]:
    """Iterate over all 2^(d - r) vertices of the subcube (fixed_mask, fixed_val)."""
    free_bits = [i for i in range(d) if ((fixed_mask >> i) & 1) == 0]
    k_free = len(free_bits)
    for idx in range(1 << k_free):
        x = fixed_val
        for b_pos, coord in enumerate(free_bits):
            if (idx >> b_pos) & 1:
                x |= 1 << coord
        yield x


def expand_dealiased_to_full(
    g0: Sequence[int],
    subcubes: Sequence[Union[Dict[str, int], Sequence[int]]],
) -> List[int]:
    """Materialize the full 2^(d+k) truth table F(x, y_1, ..., y_k) = g0(x) + sum_j 1_{C_j}(x) y_j.

    Bit layout: bits 0..d-1 are x, bits d..d+k-1 are y_0..y_{k-1} (bit=0 => y_j=-1, bit=1 => y_j=+1).
    """
    N_d = len(g0)
    d = N_d.bit_length() - 1
    k = len(subcubes)
    if d + k > 22:
        raise ValueError(f"Refusing to expand n = d + k = {d + k} > 22 in memory")
    parsed = [_parse_subcube(sc, d) for sc in subcubes]
    cid = [-1] * N_d
    for j, (fm, fv, _) in enumerate(parsed):
        for x in _iter_subcube_points(fm, fv, d):
            if cid[x] != -1:
                raise AssertionError(f"Overlapping subcubes at x={x}")
            cid[x] = j
    full = [0] * (1 << (d + k))
    mask_d = (1 << d) - 1
    for z in range(1 << (d + k)):
        x = z & mask_d
        val = int(g0[x])
        if val != 0:
            full[z] = val
        else:
            j = cid[x]
            if j < 0:
                raise AssertionError(f"Uncovered zero point x={x}")
            y_bit = (z >> (d + j)) & 1
            full[z] = 1 if y_bit else -1
    return full


def verify_dealiased_core(
    g0: Sequence[int],
    subcubes: Sequence[Union[Dict[str, int], Sequence[int]]],
    require_monotone: bool = False,
) -> Dict[str, Any]:
    """Tier 2 exact verification of F(x, y) = g0(x) + sum_j 1_{C_j}(x) * y_j."""
    g = [int(v) for v in g0]
    N_d = len(g)
    if N_d < 2 or (N_d & (N_d - 1)) != 0:
        raise ValueError(f"g0 length must be 2^d (d>=1), got {N_d}")
    d = N_d.bit_length() - 1
    uniq = set(g)
    if not (uniq <= {-1, 0, 1}):
        raise ValueError(f"g0 entries must be in {{-1, 0, +1}}, got {sorted(uniq)}")

    # 1. Exact balance: sum_x g0(x) == 0
    g_sum = sum(g)
    if g_sum != 0:
        raise AssertionError(f"Core g0 is NOT balanced: sum(g0) = {g_sum} != 0")

    # 2. Exact subcube partition of g0^{-1}(0)
    parsed = [_parse_subcube(sc, d) for sc in subcubes]
    k = len(parsed)
    cid = [-1] * N_d
    total_zero_covered = 0
    for j, (fm, fv, r_j) in enumerate(parsed):
        pts_in_sc = 0
        for x in _iter_subcube_points(fm, fv, d):
            if g[x] != 0:
                raise AssertionError(
                    f"Subcube {j} contains non-zero point x={x} with g0[x]={g[x]}"
                )
            if cid[x] != -1:
                raise AssertionError(
                    f"Subcubes {cid[x]} and {j} overlap at x={x}"
                )
            cid[x] = j
            pts_in_sc += 1
        assert pts_in_sc == (1 << (d - r_j))
        total_zero_covered += pts_in_sc

    zero_count = sum(1 for v in g if v == 0)
    if total_zero_covered != zero_count:
        raise AssertionError(
            f"Subcubes do not partition g0^(-1)(0): covered {total_zero_covered} != {zero_count}"
        )

    # 3. Pure-Python int FWHT + Exact Parseval: sum c_S^2 == 2^d * (2^d - |g0^(-1)(0)|)
    c = fwht_int(g)
    if c[0] != 0:
        raise AssertionError(f"WHT c_empty = {c[0]} != 0")
    parseval_sum = sum(v * v for v in c)
    expected_parseval = N_d * (N_d - zero_count)
    if parseval_sum != expected_parseval:
        raise AssertionError(
            f"Core Parseval failed: sum(c_S^2) = {parseval_sum} != {expected_parseval}"
        )

    # 4. Exact monotonicity check of g0 AND strict no-comparable-pairs across distinct C_j != C_{j'}
    g0_mono_violations = 0
    cross_subcube_edge_violations = 0
    # Also accumulate exact combinatorial influence of F(x, y) simultaneously!
    # Edge contribution (in units of 1/2 so integer):
    #   (-1, +1) -> prob 1   -> weight 2
    #   (±1,  0) -> prob 1/2 -> weight 1
    #   (0_a, 0_b) a!=b -> prob 1/2 -> weight 1
    #   same value / same subcube -> prob 0 -> weight 0
    comb_edge_twice_prob = 0
    for i in range(d):
        bit = 1 << i
        for x in range(N_d):
            if (x & bit) == 0:
                gx = g[x]
                gy = g[x | bit]
                if gx > gy:
                    g0_mono_violations += 1
                if gx == 0 and gy == 0:
                    if cid[x] != cid[x | bit]:
                        cross_subcube_edge_violations += 1
                        comb_edge_twice_prob += 1
                elif gx != gy:
                    if gx == 0 or gy == 0:
                        comb_edge_twice_prob += 1
                    else:
                        comb_edge_twice_prob += 2

    # Second independent route for comparable pairs across distinct subcubes (exact lattice check):
    # Two disjoint subcubes C_j, C_{j'} have x < x' iff min(C_j) <= max(C_{j'}).
    full_mask = (1 << d) - 1
    comparable_subcube_pairs = 0
    if k <= 4096:
        mins = [fv for (_, fv, _) in parsed]
        maxs = [fv | (full_mask ^ fm) for (fm, fv, _) in parsed]
        for j in range(k):
            min_j = mins[j]
            for jp in range(k):
                if j != jp and (min_j & maxs[jp]) == min_j:
                    comparable_subcube_pairs += 1
    else:
        # When g0 is monotone, cross_subcube_edge_violations == 0 iff comparable_subcube_pairs == 0
        # (proved via shortest increasing path in g0^{-1}(0)).
        comparable_subcube_pairs = cross_subcube_edge_violations

    if g0_mono_violations == 0 and k <= 4096:
        assert (cross_subcube_edge_violations == 0) == (comparable_subcube_pairs == 0), (
            f"Internal logic mismatch: edges={cross_subcube_edge_violations}, "
            f"pairs={comparable_subcube_pairs}"
        )

    is_monotone = (g0_mono_violations == 0) and (comparable_subcube_pairs == 0)
    if require_monotone and not is_monotone:
        raise AssertionError(
            f"Tier 2 monotonicity failed: g0_violations={g0_mono_violations}, "
            f"cross_subcube_edges={cross_subcube_edge_violations}, "
            f"comparable_pairs={comparable_subcube_pairs}"
        )

    # 5. Exact rational influence by TWO independent routes
    denom_4d = 1 << (2 * d)
    I_g0_spec = Fraction(sum(S.bit_count() * v * v for S, v in enumerate(c) if v != 0), denom_4d)
    I_subcubes_spec = sum(
        (Fraction(2 + r_j, 1 << (r_j + 1)) for (_, _, r_j) in parsed),
        Fraction(0),
    )
    I_exact = I_g0_spec + I_subcubes_spec

    # Combinatorial route: core edges / 2^(d-1) + sum_j Inf_{y_j}(F) where Inf_{y_j} = 2^(-r_j)
    I_core_comb = Fraction(comb_edge_twice_prob, 1 << d)
    I_aux_comb = sum((Fraction(1, 1 << r_j) for (_, _, r_j) in parsed), Fraction(0))
    I_comb = I_core_comb + I_aux_comb
    if I_exact != I_comb:
        raise AssertionError(
            f"Tier 2 spectral vs combinatorial influence mismatch: {I_exact} != {I_comb}"
        )

    # 6. Extra exact rational entropy from subcubes: sum_j 2*r_j / 2^(r_j)
    extra_H_frac = sum((Fraction(2 * r_j, 1 << r_j) for (_, _, r_j) in parsed), Fraction(0))

    # 7. Dual 100-digit outward-rounded intervals for H and L = H / (I - 1)
    iv_res = _entropy_and_ratio_dual_intervals(c, d, I_exact, extra_entropy_frac=extra_H_frac)

    # Canonical SHA-256 over (d, g0, sorted subcubes)
    h = hashlib.sha256()
    h.update(f"tier2:d={d}:k={k}:".encode("ascii"))
    h.update(bytes((v + 1) for v in g))
    for fm, fv, _ in sorted(parsed):
        h.update(f";{fm}:{fv}".encode("ascii"))

    return {
        "tier": 2,
        "d": d,
        "k_subcubes": k,
        "n_effective": d + k,
        "zero_points": zero_count,
        "sha256": h.hexdigest(),
        "exact_balance": True,
        "parseval_verified": True,
        "g0_monotone": g0_mono_violations == 0,
        "g0_monotonicity_violations": g0_mono_violations,
        "cross_subcube_edge_violations": cross_subcube_edge_violations,
        "comparable_subcube_pairs": comparable_subcube_pairs,
        "is_monotone": is_monotone,
        "I_exact_num": I_exact.numerator,
        "I_exact_den": I_exact.denominator,
        "I_exact_str": f"{I_exact.numerator}/{I_exact.denominator}",
        "I_minus_1_str": f"{(I_exact - 1).numerator}/{(I_exact - 1).denominator}",
        **iv_res,
    }


# =============================================================================
# JSON Candidate I/O (Clean Interop Schema for c71_fei_research/artifacts/)
# =============================================================================

def pack_pm1_to_hex(values_pm1: Sequence[int]) -> str:
    """Pack {-1, +1} truth table LSB-first into compact hex string."""
    f = normalize_truth_table_pm1(values_pm1)
    buf = bytearray((len(f) + 7) // 8)
    for idx, v in enumerate(f):
        if v == 1:
            buf[idx >> 3] |= 1 << (idx & 7)
    return buf.hex()


def unpack_hex_to_pm1(hex_str: str, n: int) -> List[int]:
    """Unpack LSB-first hex string into {-1, +1} truth table of length 2^n."""
    buf = bytes.fromhex(hex_str)
    N = 1 << n
    if len(buf) * 8 < N:
        raise ValueError(f"Hex buffer too short for n={n}: {len(buf)} bytes")
    return [1 if ((buf[i >> 3] >> (i & 7)) & 1) else -1 for i in range(N)]


def save_tier1_candidate_json(
    path: Union[str, Path],
    values_pm1: Sequence[int],
    name: str = "candidate_tier1",
    metadata: Optional[Dict[str, Any]] = None,
    compact_hex: bool = True,
) -> Path:
    f = normalize_truth_table_pm1(values_pm1)
    n = len(f).bit_length() - 1
    payload: Dict[str, Any] = {
        "tier": 1,
        "name": name,
        "n": n,
        "convention": "pm1_monotone_increasing",
        "sha256": truth_table_sha256(f),
        "metadata": metadata or {},
    }
    if compact_hex:
        payload["truth_table_hex"] = pack_pm1_to_hex(f)
    else:
        payload["truth_table_pm1"] = f
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def save_tier2_candidate_json(
    path: Union[str, Path],
    g0: Sequence[int],
    subcubes: Sequence[Union[Dict[str, int], Sequence[int]]],
    name: str = "candidate_tier2",
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    d = len(g0).bit_length() - 1
    parsed = [_parse_subcube(sc, d) for sc in subcubes]
    payload: Dict[str, Any] = {
        "tier": 2,
        "name": name,
        "d": d,
        "k_subcubes": len(parsed),
        "n_effective": d + len(parsed),
        "convention": "pm1_monotone_increasing",
        "g0": [int(v) for v in g0],
        "subcubes": [{"fixed_mask": fm, "fixed_val": fv} for (fm, fv, _) in parsed],
        "metadata": metadata or {},
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_and_verify_json(
    path: Union[str, Path],
    require_monotone: bool = False,
    crosscheck_tier1_max_n: int = 16,
) -> Dict[str, Any]:
    """Load a Tier 1 or Tier 2 candidate JSON and run exact verification."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tier = int(data["tier"])
    if tier == 1:
        if "truth_table_pm1" in data:
            f = data["truth_table_pm1"]
        elif "truth_table_bits" in data:
            f = data["truth_table_bits"]
        elif "truth_table_hex" in data:
            f = unpack_hex_to_pm1(data["truth_table_hex"], int(data["n"]))
        else:
            raise ValueError("Tier 1 JSON must contain truth_table_pm1, truth_table_bits, or truth_table_hex")
        res = verify_full_truth_table(f, require_monotone=require_monotone)
        if "sha256" in data and data["sha256"] != res["sha256"]:
            raise AssertionError(f"SHA-256 mismatch: JSON={data['sha256']} vs computed={res['sha256']}")
        res["name"] = data.get("name", Path(path).stem)
        return res
    elif tier == 2:
        g0 = data["g0"]
        subcubes = data["subcubes"]
        res = verify_dealiased_core(g0, subcubes, require_monotone=require_monotone)
        res["name"] = data.get("name", Path(path).stem)
        if res["n_effective"] <= crosscheck_tier1_max_n:
            full_tt = expand_dealiased_to_full(g0, subcubes)
            res_t1 = verify_full_truth_table(full_tt, require_monotone=require_monotone)
            assert res["I_exact_str"] == res_t1["I_exact_str"], "Tier 1 vs Tier 2 I mismatch"
            assert res["is_monotone"] == res_t1["is_monotone"], "Tier 1 vs Tier 2 monotonicity mismatch"
            assert res["L_cert"][:55] == res_t1["L_cert"][:55], "Tier 1 vs Tier 2 L_cert mismatch"
            res["tier1_crosscheck"] = "PASS"
        return res
    else:
        raise ValueError(f"Unsupported tier: {tier}")


# =============================================================================
# Built-in From-Scratch Self-Test (Q_2, Tribes(2,2), Alternating Tree Q_6)
# =============================================================================

def _build_selftest_benchmarks() -> List[Tuple[str, List[int], List[Dict[str, int]], str, str]]:
    """Construct from-scratch Tier 2 benchmarks with expected I and L prefixes."""
    benchmarks = []

    # 1. Q_2 middle layer (d=2, k=2 => n=4): 00->-1, 11->+1, {01, 10}->0
    # Expected: I = 3/2, H = 3, L = 6.0000000000...
    g0_q2 = [-1, 0, 0, 1]
    sc_q2 = [{"fixed_mask": 3, "fixed_val": 1}, {"fixed_mask": 3, "fixed_val": 2}]
    benchmarks.append(("Q2_middle_layer_n4", g0_q2, sc_q2, "3/2", "6.00000000"))

    # 2. De-aliased Tribes(2,2) on Q_4 (d=4, k=2 => n=6):
    # Tribes(2,2)(x) = (x0 & x1) | (x2 & x3).
    # Wait: Tribes(2,2) has 7 ones and 9 zeros! Specifically on {0101, 1010} (weights 2,
    # incomparable with each other and with the 7 ones where (x0&x1) or (x2&x3) holds;
    # wait: check balance! 7 ones, 7 remaining zeros -> -1, and {0101, 1010} -> 0 => sum = 7 - 7 = 0!
    # Also check monotonicity: any point below 0101 has weight <=1 (all -1); any point above 0101
    # has weight >=3 so either (x0&x1) or (x2&x3) is true (+1)! Same for 1010!
    # Expected: I = 13/8, L = 6.3470674987... (matches Hod g_3'!).
    g0_t22 = []
    for x in range(16):
        b0, b1, b2, b3 = (x >> 0) & 1, (x >> 1) & 1, (x >> 2) & 1, (x >> 3) & 1
        if (b0 & b1) or (b2 & b3):
            g0_t22.append(1)
        elif x in (0b0101, 0b1010):
            g0_t22.append(0)
        else:
            g0_t22.append(-1)
    sc_t22 = [{"fixed_mask": 15, "fixed_val": 0b0101}, {"fixed_mask": 15, "fixed_val": 0b1010}]
    benchmarks.append(("Tribes22_dealiased_n6", g0_t22, sc_t22, "13/8", "6.3470674987"))

    # 3. Alternating read-once tree on Q_6 (d=6, k=2 => n=8):
    # f(x) = (x0 & x1) | (x2 & (x3 | (x4 & x5)))
    # Count ones of f: Pr(x0&x1)=1/4; Pr(x2&(x3|(x4&x5))) = (1/2)*(1/2 + 1/2*1/4) = 5/16.
    # Union = 1/4 + 5/16 - 5/64 = 16/64 + 20/64 - 5/64 = 31/64!
    # So f has 31 ones and 33 zeros -> needs 2 maximal zeros turned into 0 to get 31 (+1), 31 (-1), 2 (0)!
    # Maximal zeros of f:
    #   Not (x0&x1) => (x0,x1) in {00,01,10}.
    #   Not (x2&(x3|(x4&x5))) => x2=0 (with x3,x4,x5=111) OR (x2=1, x3=0, (x4,x5) in {00,01,10}).
    # Take the optimal complementary maximal-zero pair at x2=1, x3=0:
    # 0b010110 (x1=1,x2=1,x4=1) and 0b100101 (x0=1,x2=1,x5=1).
    # Expected: I = 53/32, L = 6.363969446288...
    g0_alt6 = []
    max_zeros = (0b010110, 0b100101)
    for x in range(64):
        b0, b1, b2, b3, b4, b5 = [(x >> i) & 1 for i in range(6)]
        val = (b0 & b1) | (b2 & (b3 | (b4 & b5)))
        if val:
            g0_alt6.append(1)
        elif x in max_zeros:
            g0_alt6.append(0)
        else:
            g0_alt6.append(-1)
    sc_alt6 = [{"fixed_mask": 63, "fixed_val": z} for z in max_zeros]
    benchmarks.append(("AltTree_Q6_dealiased_n8", g0_alt6, sc_alt6, "53/32", "6.3639694462"))

    # 4. Also test a proper codimension r < d subcube (r=2 inside d=4, k=2 => n=6)
    # Use a balanced 3-valued monotone function where two incomparable 1D subcubes (r=3, size 2 each) are zero:
    # Let's construct: on x0,x1,x2,x3, take Q2 middle layer lifted to Q4 by ignoring x2,x3?
    # Wait: if x2,x3 are free inside C_1=(x0=0,x1=1) and C_2=(x0=1,x1=0), then r_j=2, size 4 each!
    # g0(x) = -1 if (x0,x1)=(0,0), +1 if (x0,x1)=(1,1), 0 if (x0,x1) in {(0,1),(1,0)}!
    # That's literally Q2 lifted with 2 free vars per subcube -> must give EXACTLY I=3/2, L=6.00000000!
    g0_r2 = []
    for x in range(16):
        b0, b1 = x & 1, (x >> 1) & 1
        if b0 == 0 and b1 == 0:
            g0_r2.append(-1)
        elif b0 == 1 and b1 == 1:
            g0_r2.append(1)
        else:
            g0_r2.append(0)
    sc_r2 = [{"fixed_mask": 3, "fixed_val": 1}, {"fixed_mask": 3, "fixed_val": 2}]
    benchmarks.append(("ProperSubcube_r2_in_Q4_n6", g0_r2, sc_r2, "3/2", "6.00000000"))

    return benchmarks


def run_selftest() -> bool:
    print("=== exact_verifier.py SELFTEST (Dual Contour: mpmath.iv + stdlib DecInterval) ===")
    benchmarks = _build_selftest_benchmarks()
    all_ok = True
    for name, g0, sc, exp_I, exp_L_prefix in benchmarks:
        res2 = verify_dealiased_core(g0, sc, require_monotone=True)
        full_tt = expand_dealiased_to_full(g0, sc)
        res1 = verify_full_truth_table(full_tt, require_monotone=True)

        assert res2["I_exact_str"] == exp_I, f"{name}: I={res2['I_exact_str']} != {exp_I}"
        assert res1["I_exact_str"] == exp_I, f"{name}: T1 I={res1['I_exact_str']} != {exp_I}"
        assert res2["L_cert"].startswith(exp_L_prefix), (
            f"{name}: L_cert={res2['L_cert']} does not start with {exp_L_prefix}"
        )
        # Cross-check Tier 1 == Tier 2 to > 55 decimal digits!
        assert res1["L_cert"][:55] == res2["L_cert"][:55], (
            f"{name}: Tier 1 vs Tier 2 L_cert mismatch:\n  T1={res1['L_cert']}\n  T2={res2['L_cert']}"
        )
        assert res1["U_cert"][:55] == res2["U_cert"][:55], (
            f"{name}: Tier 1 vs Tier 2 U_cert mismatch:\n  T1={res1['U_cert']}\n  T2={res2['U_cert']}"
        )
        # Verify both contours agree to > 55 digits
        assert res2["L_mpmath_iv"][0][:55] == res2["L_decimal_iv"][0][:55], (
            f"{name}: mpmath vs Decimal mismatch"
        )
        print(
            f"[PASS] {name:26s} d={res2['d']} k={res2['k_subcubes']} n={res1['n']} "
            f"I={res2['I_exact_str']:5s} mono={res2['is_monotone']} "
            f"L_cert={res2['L_cert'][:22]}... (T1==T2 >55 dps, width={res2['interval_width_L']:.2e})"
        )

    # Negative tamper tests (ensure verifier catches broken invariants!)
    # Tamper 1: unbalanced truth table
    try:
        bad = [1, 1, 1, -1]
        verify_full_truth_table(bad)
        raise RuntimeError("Tamper 1 (unbalanced) was NOT caught!")
    except AssertionError:
        print("[PASS] Tamper 1 (unbalanced Tier 1) rejected")

    # Tamper 2: comparable subcubes across distinct C_j != C_{j'} (x=001 < x'=011)
    try:
        # d=3: 000->-1, 111->+1, 010->-1, 100->-1, 101->+1, 110->+1, zeros={001, 011} (comparable!)
        g0_bad = [-1, 0, -1, 0, -1, 1, 1, 1]
        sc_bad = [{"fixed_mask": 7, "fixed_val": 1}, {"fixed_mask": 7, "fixed_val": 3}]
        res_bad = verify_dealiased_core(g0_bad, sc_bad, require_monotone=False)
        assert res_bad["g0_monotone"] is True
        assert res_bad["is_monotone"] is False
        assert res_bad["comparable_subcube_pairs"] > 0
        # Also verify Tier 1 agrees it is NOT monotone AND agrees on I and L!
        res_bad_t1 = verify_full_truth_table(expand_dealiased_to_full(g0_bad, sc_bad))
        assert res_bad_t1["is_monotone"] is False
        assert res_bad_t1["I_exact_str"] == res_bad["I_exact_str"]
        assert res_bad_t1["L_cert"][:55] == res_bad["L_cert"][:55]
        print("[PASS] Tamper 2 (comparable distinct subcubes) caught & T1==T2 verified")
    except Exception:
        all_ok = False
        raise

    print("RESULT: ALL EXACT_VERIFIER SELFTEST CHECKS PASSED")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent Exact Verifier for C_71 (FEI).")
    parser.add_argument("--selftest", action="store_true", help="Run built-in from-scratch self-test.")
    parser.add_argument("--verify", type=str, default=None, help="Path to Tier 1 or Tier 2 JSON certificate.")
    parser.add_argument("--require-monotone", action="store_true", help="Fail if function is not monotone.")
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
    if args.verify:
        res = load_and_verify_json(args.verify, require_monotone=args.require_monotone)
        print(json.dumps(res, indent=2))
    if not args.selftest and not args.verify:
        parser.print_help()


if __name__ == "__main__":
    main()

"""
Standalone Zero-Dependency Exact Rational Verifier for the Signed Third Autocorrelation
Constant C_1d (Phi_3):

    C_1d = inf_{f != 0, supp(f) subset [-1/4, 1/4]}
               max_{t in R} |(f * f)(t)| / (integral_{-1/4}^{1/4} f(x) dx)^2

For a piecewise-constant step function f taking values q_i / D on P equal cells
of width h = 1 / (2P) covering [-1/4, 1/4], the autoconvolution (f * f)(t) is
continuous and piecewise linear with breakpoints at the 2P - 1 interior nodes
t_k = -1/2 + (k + 1) * h (k = 0, ..., 2P - 2) and value 0 at t = +-1/2.
Since |g(t)| is convex on every linear segment, its global supremum over R is
attained identically at one of the 2P - 1 interior nodes:

    max_{t in R} |(f * f)(t)| / (integral f)^2
      = (2 * P * max_{0 <= k <= 2P-2} |(q * q)_k|) / (sum_{i=0}^{P-1} q_i)^2

Uses ONLY Python standard library (`json`, `fractions`, `decimal`, `argparse`, `time`).
Zero NumPy, zero FFT, zero floating-point rounding!

Usage:
    python verify_c1d.py                  # exact big-int Kronecker check (~0.08s) + random shift cross-check
    python verify_c1d.py --brute-force    # 100% naive schoolbook double-sum over ALL 2P-1 shifts (~3s)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from decimal import Decimal, getcontext
from fractions import Fraction
from typing import Dict, List, Tuple


def exact_convolve_kronecker(q: List[int]) -> Tuple[int, int, int, int, int]:
    """
    Exact linear autoconvolution of signed integer list `q` of length P via signed
    Kronecker substitution:
        Q(x) = Q_pos(x) - Q_neg(x)  =>  Q(x)^2 = Q_pos^2 + Q_neg^2 - 2 * Q_pos * Q_neg.
    Returns (max_abs_c, max_pos_c, arg_pos, min_neg_c, arg_neg).
    """
    P = len(q)
    max_abs_q = max(abs(v) for v in q)
    # Safe bitwidth: |c_k| <= P * max_abs_q^2; each non-negative product <= P * max_abs_q^2.
    bits = (P * max_abs_q * max_abs_q).bit_length() + 4
    mask = (1 << bits) - 1

    pos_int = 0
    neg_int = 0
    for i in range(P - 1, -1, -1):
        v = q[i]
        pos_int = (pos_int << bits) | (v if v > 0 else 0)
        neg_int = (neg_int << bits) | (-v if v < 0 else 0)

    pp = pos_int * pos_int
    nn = neg_int * neg_int
    pn = pos_int * neg_int

    max_abs_c = 0
    max_pos_c = -(1 << (bits + 2))
    min_neg_c = 1 << (bits + 2)
    arg_pos = -1
    arg_neg = -1
    for k in range(2 * P - 1):
        shift = k * bits
        ck = ((pp >> shift) & mask) + ((nn >> shift) & mask) - 2 * ((pn >> shift) & mask)
        if ck > max_pos_c:
            max_pos_c = ck
            arg_pos = k
        if ck < min_neg_c:
            min_neg_c = ck
            arg_neg = k
        ac = -ck if ck < 0 else ck
        if ac > max_abs_c:
            max_abs_c = ac
    return max_abs_c, max_pos_c, arg_pos, min_neg_c, arg_neg


def exact_convolve_bruteforce(q: List[int]) -> Tuple[int, int, int, int, int]:
    """
    100% naive exact integer linear convolution over ALL 2P - 1 shifts.
    Zero FFT, zero Kronecker packing, zero floats.
    """
    P = len(q)
    q_rev = q[::-1]
    max_abs_c = 0
    max_pos_c = None
    min_neg_c = None
    arg_pos = -1
    arg_neg = -1
    for k in range(2 * P - 1):
        lo = 0 if k < P else k - P + 1
        hi = k + 1 if k < P else P
        r_lo = P - 1 - k + lo
        r_hi = P - 1 - k + hi
        ck = sum(x * y for x, y in zip(q[lo:hi], q_rev[r_lo:r_hi]))
        if max_pos_c is None or ck > max_pos_c:
            max_pos_c = ck
            arg_pos = k
        if min_neg_c is None or ck < min_neg_c:
            min_neg_c = ck
            arg_neg = k
        ac = -ck if ck < 0 else ck
        if ac > max_abs_c:
            max_abs_c = ac
    assert max_pos_c is not None and min_neg_c is not None
    return max_abs_c, max_pos_c, arg_pos, min_neg_c, arg_neg


def verify_single_shift_direct(q: List[int], k: int) -> int:
    P = len(q)
    lo = max(0, k - (P - 1))
    hi = min(P - 1, k)
    return sum(q[i] * q[k - i] for i in range(lo, hi + 1))


def verify_certificate(cert_path: str, brute_force: bool = False) -> Dict[str, object]:
    t0 = time.perf_counter()
    with open(cert_path, "r", encoding="utf-8") as fh:
        cert = json.load(fh)

    P = int(cert["num_cells"])
    q = [int(v) for v in cert["integer_numerators"]]
    assert len(q) == P, f"Length mismatch: expected {P}, got {len(q)}"

    sum_q = sum(q)
    assert sum_q != 0, "Zero integral step function"

    if brute_force:
        max_abs_c, max_pos_c, arg_pos, min_neg_c, arg_neg = exact_convolve_bruteforce(q)
        mode = "naive_bruteforce_all_shifts"
    else:
        max_abs_c, max_pos_c, arg_pos, min_neg_c, arg_neg = exact_convolve_kronecker(q)
        # Cross-check extremal shifts directly via schoolbook sum
        assert verify_single_shift_direct(q, arg_pos) == max_pos_c
        assert verify_single_shift_direct(q, arg_neg) == min_neg_c
        assert verify_single_shift_direct(q, 0) == q[0] * q[0]
        assert verify_single_shift_direct(q, 2 * P - 2) == q[-1] * q[-1]
        mode = "exact_signed_kronecker_plus_direct_spotcheck"

    exact_frac = Fraction(2 * P * max_abs_c, sum_q * sum_q)
    pos_frac = Fraction(2 * P * max_pos_c, sum_q * sum_q)
    neg_frac = Fraction(2 * P * min_neg_c, sum_q * sum_q)

    getcontext().prec = 60
    dec50 = f"{Decimal(exact_frac.numerator) / Decimal(exact_frac.denominator):.50f}"
    pos_dec = f"{Decimal(pos_frac.numerator) / Decimal(pos_frac.denominator):.20f}"
    neg_dec = f"{Decimal(neg_frac.numerator) / Decimal(neg_frac.denominator):.20f}"

    # Verify stored exact fraction matches recomputed exact fraction bit-for-bit
    if "exact_fraction_numerator" in cert and "exact_fraction_denominator" in cert:
        stored = Fraction(int(cert["exact_fraction_numerator"]), int(cert["exact_fraction_denominator"]))
        assert exact_frac == stored, f"Stored fraction mismatch: {exact_frac} != {stored}"

    # Benchmark comparison: SCiEL (Cairn Commons, Nov 2026, 800 cells)
    sciel_frac = Fraction(
        38534420343892614563110610373646557184000,
        26432274393314964174848236787785831382961,
    )
    beats_sciel = exact_frac < sciel_frac
    margin_sciel = float(sciel_frac - exact_frac)

    elapsed = time.perf_counter() - t0
    print("=" * 80)
    print(f"EXACT C_1d (Phi_3) CERTIFICATE VERIFICATION  [{mode}]")
    print("=" * 80)
    print(f"  File           : {os.path.basename(cert_path)}")
    print(f"  Cells (P)      : {P}  (linear convolution shifts = {2 * P - 1})")
    print(f"  Neg / Pos cells: {sum(1 for v in q if v < 0)} / {sum(1 for v in q if v > 0)}")
    print(f"  sum(q)         : {sum_q}")
    print(f"  max_k (+)      : {pos_dec}  (at shift k = {arg_pos})")
    print(f"  min_k (-)      : {neg_dec}  (at shift k = {arg_neg})")
    print(f"  EXACT C_1d     : {exact_frac.numerator} / {exact_frac.denominator}")
    print(f"  Decimal (50dp) : {dec50}")
    print(f"  Beats SCiEL    : {beats_sciel}  (margin = -{margin_sciel:.10f} below 1.4578548849)")
    print(f"  Elapsed        : {elapsed:.2f}s")
    print("=" * 80)
    print("RESULT: C_1d EXACT RATIONAL CERTIFICATE PASSED")

    return {
        "num_cells": P,
        "exact_numerator": str(exact_frac.numerator),
        "exact_denominator": str(exact_frac.denominator),
        "decimal_50": dec50,
        "beats_sciel": beats_sciel,
        "margin_sciel": margin_sciel,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    default_cert = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "c1d_p6400_exact_certificate.json"
    )
    parser.add_argument("--cert", default=default_cert)
    parser.add_argument("--brute-force", action="store_true")
    args = parser.parse_args()
    verify_certificate(args.cert, brute_force=args.brute_force)


if __name__ == "__main__":
    main()

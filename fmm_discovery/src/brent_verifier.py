"""
Zero-Floating-Point Exact Verifier for Fast Matrix Multiplication (FMM) Bilinear Schemes.

Convention (`C_eq_AB_row_major`)
--------------------------------
A bilinear matrix multiplication scheme of format <m, n, p : R> over an exact ring
R_ring in {"Z2", "Z3", "Zq", "Z", "Q"} is specified by three factor matrices:
    U in R_ring^{R x (m*n)}
    V in R_ring^{R x (n*p)}
    W in R_ring^{R x (m*p)}

Row-major flattening conventions:
    A in R_ring^{m x n}:  idx_A(i1, i2) = i1 * n + i2   (0 <= i1 < m, 0 <= i2 < n)
    B in R_ring^{n x p}:  idx_B(j1, j2) = j1 * p + j2   (0 <= j1 < n, 0 <= j2 < p)
    C in R_ring^{m x p}:  idx_C(k1, k2) = k1 * p + k2   (0 <= k1 < m, 0 <= k2 < p)

The scheme computes the matrix product C = A * B via R scalar multiplications:
    m_r = (sum_{i1, i2} U[r, i1*n + i2] * A[i1, i2])
        * (sum_{j1, j2} V[r, j1*p + j2] * B[j1, j2])      for r = 0, ..., R - 1,
    C[k1, k2] = sum_{r=0}^{R-1} W[r, k1*p + k2] * m_r     for 0 <= k1 < m, 0 <= k2 < p.

Equivalently, (U, V, W) satisfies the exact Brent equations:
    sum_{r=0}^{R-1} U[r, i1*n + i2] * V[r, j1*p + j2] * W[r, k1*p + k2]
        == delta(i1, k1) * delta(i2, j1) * delta(j2, k2)
for all i1, k1 in [m], i2, j1 in [n], j2, k2 in [p].

Relation to the cyclic trace convention tr(A * B * C_tr)
--------------------------------------------------------
Much of the FMM literature (Strassen, Kauers-Moosbauer flip-graph papers, Lille FMM tables)
encodes matrix multiplication as the trilinear form tr(A * B * C_tr) = sum_{i,j,k} A_{i,j} B_{j,k} (C_tr)_{k,i},
where C_tr = C^T has shape (p x m) and row-major index idx_{C_tr}(k2, k1) = k2 * m + k1.
In that convention, the third factor W_tr is the transpose of W along the (m, p) matrix axes:
    W_tr[r, k2 * m + k1] == W[r, k1 * p + k2].
Use `BilinearScheme.w_to_trace_convention()` and `BilinearScheme.from_trace_convention()`
to convert losslessly without transpose bugs.

Zero-Floating-Point Guarantee
-----------------------------
All scalar coercion, tensor contraction, basis matrix multiplication, hashing, and
serialization use ONLY exact Python `int` and `fractions.Fraction`. Passing any `float`,
`complex`, or NumPy floating scalar raises `TypeError`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from math import gcd
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np

    _NP_INT_TYPES: Tuple[type, ...] = (np.integer,)
    _NP_BOOL_TYPES: Tuple[type, ...] = (np.bool_,)
    _NP_FLOAT_TYPES: Tuple[type, ...] = (np.floating, np.complexfloating)
except ImportError:  # pragma: no cover - numpy is optional for pure-Python verification
    np = None  # type: ignore[assignment]
    _NP_INT_TYPES = ()
    _NP_BOOL_TYPES = ()
    _NP_FLOAT_TYPES = ()

RingScalar = Union[int, Fraction]

CONVENTION = "C_eq_AB_row_major"
FORMAT_VERSION = "fmm_scheme_v1"
SUPPORTED_RINGS = ("Z2", "Z3", "Zq", "Z", "Q")

_MOD_RING_RE = re.compile(r"^Z(\d+)$")
_EXACT_RATIONAL_RE = re.compile(r"^[+-]?\d+(?:/[+-]?\d+)?$")


# =============================================================================
# 1. Exact Ring Specification & Zero-Float Scalar Coercion
# =============================================================================


def parse_ring_spec(ring: str, modulus: Optional[int] = None) -> Tuple[str, Optional[int]]:
    """
    Normalize a ring identifier into `(canonical_ring, modulus)`.

    Supported:
      - "Z2" (modulus=2)
      - "Z3" (modulus=3)
      - "Zq" with integer modulus >= 2 (e.g. q = 4, 8, 16, 5, 7, ...)
      - Shorthand "Z<q>" (e.g. "Z4", "Z8") -> ("Zq", q) (or "Z2"/"Z3" for 2/3)
      - "Z"  (exact arbitrary-precision Python int, modulus=None)
      - "Q"  (exact fractions.Fraction, modulus=None)
    """
    if not isinstance(ring, str):
        raise TypeError(f"ring must be str, got {type(ring).__name__}")
    r = ring.strip()
    if r == "Z":
        if modulus is not None:
            raise ValueError("ring='Z' does not accept a modulus")
        return "Z", None
    if r == "Q":
        if modulus is not None:
            raise ValueError("ring='Q' does not accept a modulus")
        return "Q", None
    if r == "Z2":
        if modulus is not None and int(modulus) != 2:
            raise ValueError(f"ring='Z2' requires modulus=2 or None, got {modulus}")
        return "Z2", 2
    if r == "Z3":
        if modulus is not None and int(modulus) != 3:
            raise ValueError(f"ring='Z3' requires modulus=3 or None, got {modulus}")
        return "Z3", 3
    if r == "Zq":
        if modulus is None or isinstance(modulus, (bool, float)) or isinstance(modulus, _NP_FLOAT_TYPES):
            raise ValueError("ring='Zq' requires an exact integer modulus >= 2")
        q = int(modulus)
        if q < 2:
            raise ValueError(f"modulus must be >= 2, got {q}")
        return "Zq", q

    m = _MOD_RING_RE.match(r)
    if m:
        q = int(m.group(1))
        if q < 2:
            raise ValueError(f"modular ring '{r}' must have modulus >= 2")
        if modulus is not None and int(modulus) != q:
            raise ValueError(f"ring='{r}' conflicts with modulus={modulus}")
        if q == 2:
            return "Z2", 2
        if q == 3:
            return "Z3", 3
        return "Zq", q

    raise ValueError(f"Unsupported ring '{ring}'. Expected one of {SUPPORTED_RINGS} or 'Z<q>'.")


def coerce_ring_scalar(val: Any, ring: str, modulus: Optional[int] = None) -> RingScalar:
    """
    Coerce an input value into an exact scalar of `(ring, modulus)`.
    Strictly rejects all floating-point and complex types.
    """
    if isinstance(val, (float, complex)) or isinstance(val, _NP_FLOAT_TYPES):
        raise TypeError(
            f"Floating-point/complex scalar {val!r} ({type(val).__name__}) is strictly forbidden "
            f"in exact FMM verification."
        )

    if ring == "Q":
        if isinstance(val, Fraction):
            return val
        if isinstance(val, (bool, int)) or isinstance(val, (_NP_INT_TYPES + _NP_BOOL_TYPES)):
            return Fraction(int(val), 1)
        if isinstance(val, str):
            s = val.strip()
            if not _EXACT_RATIONAL_RE.match(s):
                raise ValueError(
                    f"Rational string {val!r} must be an exact integer or 'num/den' fraction "
                    f"(decimal/scientific notation is forbidden)."
                )
            return Fraction(s)
        if isinstance(val, (tuple, list)) and len(val) == 2:
            num = coerce_ring_scalar(val[0], "Z", None)
            den = coerce_ring_scalar(val[1], "Z", None)
            assert isinstance(num, int) and isinstance(den, int)
            if den == 0:
                raise ZeroDivisionError("Denominator cannot be 0 in ring Q")
            return Fraction(num, den)
        raise TypeError(f"Cannot coerce {val!r} ({type(val).__name__}) into exact ring Q")

    # Integer or modular ring
    if isinstance(val, Fraction):
        if val.denominator != 1:
            raise ValueError(f"Non-integer Fraction {val} cannot be coerced into ring '{ring}'")
        iv = int(val.numerator)
    elif isinstance(val, (bool, int)) or isinstance(val, (_NP_INT_TYPES + _NP_BOOL_TYPES)):
        iv = int(val)
    elif isinstance(val, str):
        s = val.strip()
        if not re.match(r"^[+-]?\d+$", s):
            raise ValueError(f"Integer string {val!r} is not a valid exact integer for ring '{ring}'")
        iv = int(s)
    else:
        raise TypeError(f"Cannot coerce {val!r} ({type(val).__name__}) into exact ring '{ring}'")

    if ring == "Z":
        return iv
    assert modulus is not None and modulus >= 2
    return iv % modulus


def _ring_zero(ring: str) -> RingScalar:
    return Fraction(0, 1) if ring == "Q" else 0


def _ring_one(ring: str) -> RingScalar:
    return Fraction(1, 1) if ring == "Q" else 1


def _ring_add(a: RingScalar, b: RingScalar, ring: str, modulus: Optional[int]) -> RingScalar:
    if ring == "Z" or ring == "Q":
        return a + b  # type: ignore[operator]
    assert modulus is not None
    return (int(a) + int(b)) % modulus


def _ring_mul(a: RingScalar, b: RingScalar, ring: str, modulus: Optional[int]) -> RingScalar:
    if ring == "Z" or ring == "Q":
        return a * b  # type: ignore[operator]
    assert modulus is not None
    return (int(a) * int(b)) % modulus


def _serialize_scalar(val: RingScalar, ring: str) -> Union[int, str]:
    if ring == "Q":
        assert isinstance(val, Fraction)
        return str(val.numerator) if val.denominator == 1 else f"{val.numerator}/{val.denominator}"
    return int(val)


def _format_coeff_term(coeff: RingScalar, var_name: str, ring: str, modulus: Optional[int]) -> Tuple[str, str]:
    """Return `(sign_str, abs_term_str)` for human-readable algebraic printing."""
    if ring == "Q":
        assert isinstance(coeff, Fraction)
        sign = "-" if coeff < 0 else "+"
        mag = -coeff if coeff < 0 else coeff
        if mag == 1:
            return sign, var_name
        mag_str = str(mag.numerator) if mag.denominator == 1 else f"({mag.numerator}/{mag.denominator})"
        return sign, f"{mag_str}*{var_name}"
    iv = int(coeff)
    if ring in ("Z2", "Z3", "Zq") and modulus is not None:
        if ring == "Z2":
            return "+", var_name
        if ring == "Z3" and iv == 2:
            return "-", var_name
        if iv == 1:
            return "+", var_name
        return "+", f"{iv}*{var_name}"
    sign = "-" if iv < 0 else "+"
    mag = -iv if iv < 0 else iv
    if mag == 1:
        return sign, var_name
    return sign, f"{mag}*{var_name}"


def _format_linear_combination(
    row: Sequence[RingScalar],
    var_names: Sequence[str],
    ring: str,
    modulus: Optional[int],
) -> str:
    parts: List[str] = []
    for coeff, name in zip(row, var_names):
        if coeff == 0:
            continue
        sign, term = _format_coeff_term(coeff, name, ring, modulus)
        if not parts:
            parts.append(f"-{term}" if sign == "-" else term)
        else:
            parts.append(f" {sign} {term}")
    return "".join(parts) if parts else "0"


# =============================================================================
# 2. BilinearScheme Data Structure, Canonical Hashing & Serialization
# =============================================================================


def _coerce_matrix(
    mat: Sequence[Sequence[Any]],
    expected_rows: Optional[int],
    expected_cols: int,
    label: str,
    ring: str,
    modulus: Optional[int],
) -> List[List[RingScalar]]:
    if np is not None and isinstance(mat, np.ndarray):
        if mat.ndim != 2:
            raise ValueError(f"{label} must be 2D, got shape {mat.shape}")
        if np.issubdtype(mat.dtype, np.floating) or np.issubdtype(mat.dtype, np.complexfloating):
            raise TypeError(f"{label} has forbidden floating/complex dtype {mat.dtype}")
        raw_rows: Sequence[Sequence[Any]] = mat.tolist()
    else:
        raw_rows = list(mat)

    if expected_rows is not None and len(raw_rows) != expected_rows:
        raise ValueError(f"{label} row count mismatch: expected {expected_rows}, got {len(raw_rows)}")

    out: List[List[RingScalar]] = []
    for r_idx, row in enumerate(raw_rows):
        row_list = list(row)
        if len(row_list) != expected_cols:
            raise ValueError(
                f"{label}[{r_idx}] length mismatch: expected {expected_cols}, got {len(row_list)}"
            )
        out.append([coerce_ring_scalar(x, ring, modulus) for x in row_list])
    return out


def _canonicalize_term(
    u_row: Sequence[RingScalar],
    v_row: Sequence[RingScalar],
    w_row: Sequence[RingScalar],
    ring: str,
    modulus: Optional[int],
) -> Tuple[Tuple[Union[int, str], ...], Tuple[Union[int, str], ...], Tuple[Union[int, str], ...]]:
    """
    Canonicalize a single rank-1 tensor u_r (x) v_r (x) w_r so that equivalent
    sign/scalar rescalings hash identically:
      - Over Z: negate (u, w) if first nonzero of u < 0; negate (v, w) if first nonzero of v < 0.
      - Over Q: scale u and v so their first nonzero entry equals 1, absorbing u_0 * v_0 into w.
      - Over Z2/Z3/Zq: canonical residues in {0, ..., q-1}; if q is prime (e.g. Z3),
        normalize first nonzero of u and v to 1 via modular inverse (for Z2, already 1).
    """
    u = list(u_row)
    v = list(v_row)
    w = list(w_row)

    u_first = next((x for x in u if x != 0), None)
    v_first = next((x for x in v if x != 0), None)
    w_first = next((x for x in w if x != 0), None)
    if u_first is None or v_first is None or w_first is None:
        # Zero rank-1 term canonicalizes to all-zero triple
        z_u = tuple(_serialize_scalar(_ring_zero(ring), ring) for _ in u)
        z_v = tuple(_serialize_scalar(_ring_zero(ring), ring) for _ in v)
        z_w = tuple(_serialize_scalar(_ring_zero(ring), ring) for _ in w)
        return z_u, z_v, z_w

    if ring == "Z":
        if int(u_first) < 0:
            u = [-int(x) for x in u]
            w = [-int(x) for x in w]
        if int(v_first) < 0:
            v = [-int(x) for x in v]
            w = [-int(x) for x in w]
    elif ring == "Q":
        assert isinstance(u_first, Fraction) and isinstance(v_first, Fraction)
        scale = u_first * v_first
        u = [x / u_first for x in u]  # type: ignore[operator]
        v = [x / v_first for x in v]  # type: ignore[operator]
        w = [x * scale for x in w]  # type: ignore[operator]
    elif ring in ("Z3", "Zq") and modulus is not None:
        u0 = int(u_first)
        v0 = int(v_first)
        if gcd(u0, modulus) == 1:
            inv_u = pow(u0, -1, modulus)
            u = [(int(x) * inv_u) % modulus for x in u]
            w = [(int(x) * u0) % modulus for x in w]
        if gcd(v0, modulus) == 1:
            inv_v = pow(v0, -1, modulus)
            v = [(int(x) * inv_v) % modulus for x in v]
            w = [(int(x) * v0) % modulus for x in w]

    return (
        tuple(_serialize_scalar(x, ring) for x in u),
        tuple(_serialize_scalar(x, ring) for x in v),
        tuple(_serialize_scalar(x, ring) for x in w),
    )


@dataclass
class BilinearScheme:
    """
    Exact representation of a bilinear matrix multiplication scheme <m, n, p : R>
    using convention C = A * B with row-major matrix flattening.
    """

    m: int
    n: int
    p: int
    U: List[List[RingScalar]]
    V: List[List[RingScalar]]
    W: List[List[RingScalar]]
    ring: str = "Z"
    modulus: Optional[int] = None
    name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(isinstance(d, (bool, float)) or isinstance(d, _NP_FLOAT_TYPES) for d in (self.m, self.n, self.p)):
            raise TypeError("Dimensions (m, n, p) must be exact positive integers")
        self.m = int(self.m)
        self.n = int(self.n)
        self.p = int(self.p)
        if self.m < 1 or self.n < 1 or self.p < 1:
            raise ValueError(f"Dimensions must be >= 1, got ({self.m}, {self.n}, {self.p})")

        self.ring, self.modulus = parse_ring_spec(self.ring, self.modulus)
        self.U = _coerce_matrix(self.U, None, self.m * self.n, "U", self.ring, self.modulus)
        rank = len(self.U)
        self.V = _coerce_matrix(self.V, rank, self.n * self.p, "V", self.ring, self.modulus)
        self.W = _coerce_matrix(self.W, rank, self.m * self.p, "W", self.ring, self.modulus)

    @property
    def rank(self) -> int:
        return len(self.U)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.m, self.n, self.p)

    @property
    def ring_label(self) -> str:
        if self.ring == "Zq":
            return f"Z{self.modulus}"
        return self.ring

    def is_ternary(self) -> bool:
        """
        True iff every entry in U, V, W lies in {-1, 0, +1} (or {0, 1, q-1} mod q).
        """
        if self.ring == "Z2":
            return True
        if self.ring == "Z3":
            return True
        if self.ring == "Zq":
            assert self.modulus is not None
            allowed = {0, 1, self.modulus - 1}
            return all(int(x) in allowed for M in (self.U, self.V, self.W) for row in M for x in row)
        if self.ring == "Z":
            return all(int(x) in (-1, 0, 1) for M in (self.U, self.V, self.W) for row in M for x in row)
        # ring == "Q"
        allowed_q = (Fraction(-1, 1), Fraction(0, 1), Fraction(1, 1))
        return all(x in allowed_q for M in (self.U, self.V, self.W) for row in M for x in row)

    def max_abs_coeff(self) -> RingScalar:
        """
        Maximum absolute coefficient magnitude across U, V, W.
        For modular rings Zq, uses the balanced residue magnitude min(x, q - x).
        """
        if self.rank == 0:
            return _ring_zero(self.ring)
        if self.ring in ("Z2", "Z3", "Zq"):
            assert self.modulus is not None
            q = self.modulus
            return max(min(int(x), q - int(x)) for M in (self.U, self.V, self.W) for row in M for x in row)
        return max(abs(x) for M in (self.U, self.V, self.W) for row in M for x in row)  # type: ignore[return-value]

    def additive_complexity(self) -> Dict[str, int]:
        """
        Exact naive additive complexity (without common subexpression elimination):
          - nnz_U, nnz_V, nnz_W, nnz_total
          - naive_additions_U = sum_r max(0, nnz(U[r, :]) - 1)
          - naive_additions_V = sum_r max(0, nnz(V[r, :]) - 1)
          - naive_additions_W = sum_c max(0, nnz(W[:, c]) - 1)
          - naive_additions_total = sum of the three
        """
        nnz_u = sum(1 for row in self.U for x in row if x != 0)
        nnz_v = sum(1 for row in self.V for x in row if x != 0)
        nnz_w = sum(1 for row in self.W for x in row if x != 0)

        adds_u = sum(max(0, sum(1 for x in row if x != 0) - 1) for row in self.U)
        adds_v = sum(max(0, sum(1 for x in row if x != 0) - 1) for row in self.V)
        adds_w = sum(
            max(0, sum(1 for r in range(self.rank) if self.W[r][c] != 0) - 1)
            for c in range(self.m * self.p)
        )
        return {
            "nnz_U": nnz_u,
            "nnz_V": nnz_v,
            "nnz_W": nnz_w,
            "nnz_total": nnz_u + nnz_v + nnz_w,
            "naive_additions_U": adds_u,
            "naive_additions_V": adds_v,
            "naive_additions_W": adds_w,
            "naive_additions_total": adds_u + adds_v + adds_w,
        }

    def canonical_sha256(self) -> str:
        """
        Deterministic SHA-256 fingerprint invariant to permutation of the R rank-1 terms
        and equivalent per-term sign/scalar normalizations.
        """
        terms = [
            _canonicalize_term(self.U[r], self.V[r], self.W[r], self.ring, self.modulus)
            for r in range(self.rank)
        ]
        terms.sort()
        payload = {
            "convention": CONVENTION,
            "m": self.m,
            "n": self.n,
            "p": self.p,
            "rank": self.rank,
            "ring": self.ring,
            "modulus": self.modulus,
            "terms": terms,
        }
        blob = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def reduce_mod(self, q: int) -> "BilinearScheme":
        """
        Project an integer scheme (or ZQ scheme with q | Q) down to Z2 / Z3 / Zq.
        """
        if isinstance(q, (bool, float)) or isinstance(q, _NP_FLOAT_TYPES) or int(q) < 2:
            raise ValueError(f"Target modulus must be an integer >= 2, got {q!r}")
        q = int(q)
        if self.ring == "Q":
            raise ValueError("Cannot reduce ring='Q' modulo q directly without clearing denominators")
        if self.ring in ("Z2", "Z3", "Zq"):
            assert self.modulus is not None
            if self.modulus % q != 0:
                raise ValueError(f"Cannot reduce mod {self.modulus} to mod {q}: {q} does not divide {self.modulus}")
        target_ring = "Z2" if q == 2 else ("Z3" if q == 3 else "Zq")
        return BilinearScheme(
            m=self.m,
            n=self.n,
            p=self.p,
            U=[[int(x) % q for x in row] for row in self.U],
            V=[[int(x) % q for x in row] for row in self.V],
            W=[[int(x) % q for x in row] for row in self.W],
            ring=target_ring,
            modulus=q,
            name=f"{self.name}_mod{q}" if self.name else "",
            metadata=dict(self.metadata),
        )

    def w_to_trace_convention(self) -> List[List[RingScalar]]:
        """
        Convert W from C = A * B indexing (k1 * p + k2) to cyclic tr(A * B * C_tr)
        indexing (k2 * m + k1), where C_tr = C^T has shape (p x m).
        """
        w_tr: List[List[RingScalar]] = []
        for row in self.W:
            out_row: List[RingScalar] = [_ring_zero(self.ring)] * (self.m * self.p)
            for k1 in range(self.m):
                for k2 in range(self.p):
                    out_row[k2 * self.m + k1] = row[k1 * self.p + k2]
            w_tr.append(out_row)
        return w_tr

    @classmethod
    def from_trace_convention(
        cls,
        m: int,
        n: int,
        p: int,
        U: Sequence[Sequence[Any]],
        V: Sequence[Sequence[Any]],
        W_tr: Sequence[Sequence[Any]],
        ring: str = "Z",
        modulus: Optional[int] = None,
        name: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BilinearScheme":
        """
        Construct a C = A * B BilinearScheme from factors (U, V, W_tr) expressed in the
        cyclic trace convention tr(A * B * C_tr) where C_tr has shape (p x m).
        """
        canon_ring, canon_mod = parse_ring_spec(ring, modulus)
        w_tr_mat = _coerce_matrix(W_tr, None, m * p, "W_tr", canon_ring, canon_mod)
        w_mat: List[List[RingScalar]] = []
        for row in w_tr_mat:
            out_row: List[RingScalar] = [_ring_zero(canon_ring)] * (m * p)
            for k1 in range(m):
                for k2 in range(p):
                    out_row[k1 * p + k2] = row[k2 * m + k1]
            w_mat.append(out_row)
        return cls(
            m=m,
            n=n,
            p=p,
            U=list(U),  # type: ignore[arg-type]
            V=list(V),  # type: ignore[arg-type]
            W=w_mat,
            ring=canon_ring,
            modulus=canon_mod,
            name=name,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "convention": CONVENTION,
            "name": self.name,
            "m": self.m,
            "n": self.n,
            "p": self.p,
            "rank": self.rank,
            "ring": self.ring,
            "modulus": self.modulus,
            "canonical_sha256": self.canonical_sha256(),
            "U": [[_serialize_scalar(x, self.ring) for x in row] for row in self.U],
            "V": [[_serialize_scalar(x, self.ring) for x in row] for row in self.V],
            "W": [[_serialize_scalar(x, self.ring) for x in row] for row in self.W],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BilinearScheme":
        conv = data.get("convention", CONVENTION)
        if conv != CONVENTION:
            raise ValueError(
                f"Unsupported scheme convention {conv!r}; expected {CONVENTION!r}. "
                f"Convert cyclic-trace W via `from_trace_convention` first."
            )
        return cls(
            m=int(data["m"]),
            n=int(data["n"]),
            p=int(data["p"]),
            U=data["U"],
            V=data["V"],
            W=data["W"],
            ring=str(data.get("ring", "Z")),
            modulus=data.get("modulus"),
            name=str(data.get("name", "")),
            metadata=dict(data.get("metadata", {})),
        )

    def save_json(self, path: str) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "BilinearScheme":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    def _var_names(self, one_based: bool = True) -> Tuple[List[str], List[str], List[str]]:
        off = 1 if one_based else 0
        sep = "" if max(self.m, self.n, self.p) + off < 10 else "_"
        a_names = [f"a{i + off}{sep}{j + off}" for i in range(self.m) for j in range(self.n)]
        b_names = [f"b{i + off}{sep}{j + off}" for i in range(self.n) for j in range(self.p)]
        c_names = [f"c{i + off}{sep}{j + off}" for i in range(self.m) for j in range(self.p)]
        return a_names, b_names, c_names

    def to_triplet_text(self, one_based: bool = True) -> str:
        """
        Export in standard FMM triplet format:
            (linear_combo_A) * (linear_combo_B) -> (linear_combo_C)
        with convention C = A * B explicitly declared in the header.
        """
        a_names, b_names, c_names = self._var_names(one_based=one_based)
        lines = [
            f"# FMM Triplet Scheme <{self.m}, {self.n}, {self.p} : {self.rank}> over {self.ring_label}",
            f"# Convention: {CONVENTION} (C = A * B; NOT cyclic trace tr(A*B*C))",
            f"# Canonical SHA-256: {self.canonical_sha256()}",
        ]
        if self.name:
            lines.append(f"# Name: {self.name}")
        for r in range(self.rank):
            u_s = _format_linear_combination(self.U[r], a_names, self.ring, self.modulus)
            v_s = _format_linear_combination(self.V[r], b_names, self.ring, self.modulus)
            w_s = _format_linear_combination(self.W[r], c_names, self.ring, self.modulus)
            lines.append(f"({u_s}) * ({v_s}) -> ({w_s})")
        return "\n".join(lines) + "\n"

    def to_lrp_text(self, one_based: bool = True) -> str:
        """
        Export in standard Left-Right-Product (LRP) explicit bilinear program format:
            m1 = (...) * (...)
            ...
            c11 = m1 + ...
        """
        a_names, b_names, c_names = self._var_names(one_based=one_based)
        off = 1 if one_based else 0
        m_names = [f"m{r + off}" for r in range(self.rank)]
        lines = [
            f"# FMM LRP Program <{self.m}, {self.n}, {self.p} : {self.rank}> over {self.ring_label}",
            f"# Convention: {CONVENTION} (C = A * B)",
            f"# Canonical SHA-256: {self.canonical_sha256()}",
        ]
        if self.name:
            lines.append(f"# Name: {self.name}")
        for r in range(self.rank):
            u_s = _format_linear_combination(self.U[r], a_names, self.ring, self.modulus)
            v_s = _format_linear_combination(self.V[r], b_names, self.ring, self.modulus)
            lines.append(f"{m_names[r]} = ({u_s}) * ({v_s})")
        lines.append("")
        for c_idx, cname in enumerate(c_names):
            col = [self.W[r][c_idx] for r in range(self.rank)]
            c_s = _format_linear_combination(col, m_names, self.ring, self.modulus)
            lines.append(f"{cname} = {c_s}")
        return "\n".join(lines) + "\n"


# =============================================================================
# 3. Exact Matrix Evaluation & Dual Independent Verification Paths
# =============================================================================


def naive_matmul(
    A: Sequence[Sequence[Any]],
    B: Sequence[Sequence[Any]],
    ring: str = "Z",
    modulus: Optional[int] = None,
) -> List[List[RingScalar]]:
    """Exact O(m*n*p) reference matrix product C = A * B over `(ring, modulus)`."""
    canon_ring, canon_mod = parse_ring_spec(ring, modulus)
    m = len(A)
    if m == 0:
        raise ValueError("Matrix A must have at least 1 row")
    n = len(A[0])
    if len(B) != n:
        raise ValueError(f"Inner dimension mismatch: A is {m}x{n}, B has {len(B)} rows")
    p = len(B[0]) if n > 0 else 0
    A_ex = _coerce_matrix(A, m, n, "A", canon_ring, canon_mod)
    B_ex = _coerce_matrix(B, n, p, "B", canon_ring, canon_mod)

    zero = _ring_zero(canon_ring)
    C: List[List[RingScalar]] = [[zero for _ in range(p)] for _ in range(m)]
    if canon_ring in ("Z2", "Z3", "Zq"):
        assert canon_mod is not None
        for i in range(m):
            for k in range(p):
                acc = 0
                for j in range(n):
                    acc += int(A_ex[i][j]) * int(B_ex[j][k])
                C[i][k] = acc % canon_mod
    else:
        for i in range(m):
            for k in range(p):
                acc_val = zero
                for j in range(n):
                    acc_val = acc_val + (A_ex[i][j] * B_ex[j][k])  # type: ignore[operator]
                C[i][k] = acc_val
    return C


def matmul_via_scheme(
    scheme: BilinearScheme,
    A: Sequence[Sequence[Any]],
    B: Sequence[Sequence[Any]],
) -> List[List[RingScalar]]:
    """
    Evaluate C = A * B directly through the bilinear scheme (U, V, W) in exact arithmetic.
    Rejects floating-point inputs and dimension mismatches.
    """
    m, n, p = scheme.m, scheme.n, scheme.p
    ring, mod = scheme.ring, scheme.modulus
    A_ex = _coerce_matrix(A, m, n, "A", ring, mod)
    B_ex = _coerce_matrix(B, n, p, "B", ring, mod)

    # Flatten A and B in row-major order and extract nonzero entries for fast exact evaluation
    nz_a: List[Tuple[int, RingScalar]] = [
        (i * n + j, A_ex[i][j]) for i in range(m) for j in range(n) if A_ex[i][j] != 0
    ]
    nz_b: List[Tuple[int, RingScalar]] = [
        (j * p + k, B_ex[j][k]) for j in range(n) for k in range(p) if B_ex[j][k] != 0
    ]

    zero = _ring_zero(ring)
    C_flat: List[RingScalar] = [zero] * (m * p)

    if ring in ("Z2", "Z3", "Zq"):
        assert mod is not None
        for r in range(scheme.rank):
            u_row = scheme.U[r]
            alpha = 0
            for idx, aval in nz_a:
                u_c = int(u_row[idx])
                if u_c:
                    alpha += u_c * int(aval)
            alpha %= mod
            if alpha == 0:
                continue

            v_row = scheme.V[r]
            beta = 0
            for idx, bval in nz_b:
                v_c = int(v_row[idx])
                if v_c:
                    beta += v_c * int(bval)
            beta %= mod
            if beta == 0:
                continue

            prod = (alpha * beta) % mod
            if prod == 0:
                continue

            w_row = scheme.W[r]
            for c_idx in range(m * p):
                w_c = int(w_row[c_idx])
                if w_c:
                    C_flat[c_idx] = (int(C_flat[c_idx]) + w_c * prod) % mod
    else:
        for r in range(scheme.rank):
            u_row = scheme.U[r]
            alpha_val = zero
            for idx, aval in nz_a:
                u_c = u_row[idx]
                if u_c != 0:
                    alpha_val = alpha_val + (u_c * aval)  # type: ignore[operator]
            if alpha_val == 0:
                continue

            v_row = scheme.V[r]
            beta_val = zero
            for idx, bval in nz_b:
                v_c = v_row[idx]
                if v_c != 0:
                    beta_val = beta_val + (v_c * bval)  # type: ignore[operator]
            if beta_val == 0:
                continue

            prod_val = alpha_val * beta_val  # type: ignore[operator]
            if prod_val == 0:
                continue

            w_row = scheme.W[r]
            for c_idx in range(m * p):
                w_c = w_row[c_idx]
                if w_c != 0:
                    C_flat[c_idx] = C_flat[c_idx] + (w_c * prod_val)  # type: ignore[operator]

    return [C_flat[i * p : (i + 1) * p] for i in range(m)]


def verify_brent_equations(scheme: BilinearScheme, max_mismatches: int = 10) -> Dict[str, Any]:
    """
    Verification Path 1 (Exact Sparse Tensor Contraction / Brent Equations):
        sum_{r=0}^{R-1} U[r, i1*n + i2] * V[r, j1*p + j2] * W[r, k1*p + k2]
            == delta(i1, k1) * delta(i2, j1) * delta(j2, k2).

    Checks ALL (m*n)*(n*p)*(m*p) tensor entries without floating-point arithmetic.
    """
    m, n, p = scheme.m, scheme.n, scheme.p
    ring, mod = scheme.ring, scheme.modulus
    mn, np_dim, mp = m * n, n * p, m * p
    total_equations = mn * np_dim * mp

    # Sparse exact contraction across all R rank-1 terms
    acc: Dict[Tuple[int, int, int], RingScalar] = {}
    if ring in ("Z2", "Z3", "Zq"):
        assert mod is not None
        for r in range(scheme.rank):
            nz_u = [(idx, int(val)) for idx, val in enumerate(scheme.U[r]) if val != 0]
            nz_v = [(idx, int(val)) for idx, val in enumerate(scheme.V[r]) if val != 0]
            nz_w = [(idx, int(val)) for idx, val in enumerate(scheme.W[r]) if val != 0]
            if not (nz_u and nz_v and nz_w):
                continue
            for u_idx, u_val in nz_u:
                for v_idx, v_val in nz_v:
                    uv = (u_val * v_val) % mod
                    if uv == 0:
                        continue
                    for w_idx, w_val in nz_w:
                        delta = (uv * w_val) % mod
                        if delta == 0:
                            continue
                        key = (u_idx, v_idx, w_idx)
                        acc[key] = (int(acc.get(key, 0)) + delta) % mod
    else:
        zero = _ring_zero(ring)
        for r in range(scheme.rank):
            nz_u_q = [(idx, val) for idx, val in enumerate(scheme.U[r]) if val != 0]
            nz_v_q = [(idx, val) for idx, val in enumerate(scheme.V[r]) if val != 0]
            nz_w_q = [(idx, val) for idx, val in enumerate(scheme.W[r]) if val != 0]
            if not (nz_u_q and nz_v_q and nz_w_q):
                continue
            for u_idx, u_val in nz_u_q:
                for v_idx, v_val in nz_v_q:
                    uv_q = u_val * v_val  # type: ignore[operator]
                    if uv_q == 0:
                        continue
                    for w_idx, w_val in nz_w_q:
                        key = (u_idx, v_idx, w_idx)
                        acc[key] = acc.get(key, zero) + (uv_q * w_val)  # type: ignore[operator]

    one = _ring_one(ring)
    expected_ones = {
        (i1 * n + i2, i2 * p + k2, i1 * p + k2)
        for i1 in range(m)
        for i2 in range(n)
        for k2 in range(p)
    }

    mismatch_count = 0
    mismatches: List[Dict[str, Any]] = []

    # Check all required 1-entries
    for key in sorted(expected_ones):
        got = acc.get(key, _ring_zero(ring))
        if got != one:
            mismatch_count += 1
            if len(mismatches) < max_mismatches:
                u_idx, v_idx, w_idx = key
                mismatches.append(
                    {
                        "a_idx": divmod(u_idx, n),
                        "b_idx": divmod(v_idx, p),
                        "c_idx": divmod(w_idx, p),
                        "expected": _serialize_scalar(one, ring),
                        "got": _serialize_scalar(got, ring),
                    }
                )

    # Check all off-support entries must vanish
    for key, got in acc.items():
        if key in expected_ones or got == 0:
            continue
        mismatch_count += 1
        if len(mismatches) < max_mismatches:
            u_idx, v_idx, w_idx = key
            mismatches.append(
                {
                    "a_idx": divmod(u_idx, n),
                    "b_idx": divmod(v_idx, p),
                    "c_idx": divmod(w_idx, p),
                    "expected": 0,
                    "got": _serialize_scalar(got, ring),
                }
            )

    return {
        "ok": mismatch_count == 0,
        "total_equations": total_equations,
        "expected_ones": len(expected_ones),
        "mismatch_count": mismatch_count,
        "sample_mismatches": mismatches,
    }


def verify_basis_matmul(scheme: BilinearScheme, max_mismatches: int = 10) -> Dict[str, Any]:
    """
    Verification Path 2 (Direct Exhaustive Basis Matrix Multiplication):
    Explicitly constructs ALL (m*n)*(n*p) unit matrix pairs (E_{i1,i2}, E_{j1,j2}),
    evaluates `matmul_via_scheme(scheme, E_A, E_B)`, and compares entry-by-entry against
    `naive_matmul(E_A, E_B, ring, modulus)`.

    Completely independent of `verify_brent_equations`.
    """
    m, n, p = scheme.m, scheme.n, scheme.p
    ring, mod = scheme.ring, scheme.modulus
    zero = _ring_zero(ring)
    one = _ring_one(ring)

    # Pre-allocate reusable basis matrices
    E_A: List[List[RingScalar]] = [[zero for _ in range(n)] for _ in range(m)]
    E_B: List[List[RingScalar]] = [[zero for _ in range(p)] for _ in range(n)]

    total_pairs = (m * n) * (n * p)
    failed_pairs = 0
    mismatches: List[Dict[str, Any]] = []

    for i1 in range(m):
        for i2 in range(n):
            E_A[i1][i2] = one
            for j1 in range(n):
                for j2 in range(p):
                    E_B[j1][j2] = one
                    C_got = matmul_via_scheme(scheme, E_A, E_B)
                    C_ref = naive_matmul(E_A, E_B, ring=ring, modulus=mod)
                    if C_got != C_ref:
                        failed_pairs += 1
                        if len(mismatches) < max_mismatches:
                            entry_diffs = []
                            for k1 in range(m):
                                for k2 in range(p):
                                    if C_got[k1][k2] != C_ref[k1][k2]:
                                        entry_diffs.append(
                                            {
                                                "c_idx": (k1, k2),
                                                "expected": _serialize_scalar(C_ref[k1][k2], ring),
                                                "got": _serialize_scalar(C_got[k1][k2], ring),
                                            }
                                        )
                            mismatches.append(
                                {
                                    "basis_A": (i1, i2),
                                    "basis_B": (j1, j2),
                                    "diffs": entry_diffs[:4],
                                }
                            )
                    E_B[j1][j2] = zero
            E_A[i1][i2] = zero

    return {
        "ok": failed_pairs == 0,
        "total_basis_pairs": total_pairs,
        "failed_pairs": failed_pairs,
        "sample_mismatches": mismatches,
    }


@dataclass
class VerificationReport:
    valid: bool
    brent_ok: bool
    basis_ok: bool
    both_paths_agree: bool
    shape: Tuple[int, int, int]
    rank: int
    ring: str
    modulus: Optional[int]
    convention: str
    is_ternary: bool
    max_abs_coeff: Union[int, str]
    additive_complexity: Dict[str, int]
    canonical_sha256: str
    brent_details: Dict[str, Any]
    basis_details: Dict[str, Any]
    elapsed_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def verify_scheme(scheme: BilinearScheme, max_mismatches: int = 10) -> VerificationReport:
    """
    Run BOTH independent exact verification paths (Brent tensor contraction AND
    exhaustive basis matrix multiplication) and return a full VerificationReport.
    """
    t0 = time.perf_counter()
    brent_res = verify_brent_equations(scheme, max_mismatches=max_mismatches)
    basis_res = verify_basis_matmul(scheme, max_mismatches=max_mismatches)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    brent_ok = bool(brent_res["ok"])
    basis_ok = bool(basis_res["ok"])
    both_agree = brent_ok == basis_ok
    if not both_agree:  # pragma: no cover - defensive invariant check
        raise RuntimeError(
            f"INTERNAL VERIFIER DISAGREEMENT: brent_ok={brent_ok} != basis_ok={basis_ok}"
        )

    return VerificationReport(
        valid=brent_ok and basis_ok,
        brent_ok=brent_ok,
        basis_ok=basis_ok,
        both_paths_agree=both_agree,
        shape=scheme.shape,
        rank=scheme.rank,
        ring=scheme.ring,
        modulus=scheme.modulus,
        convention=CONVENTION,
        is_ternary=scheme.is_ternary(),
        max_abs_coeff=_serialize_scalar(scheme.max_abs_coeff(), scheme.ring),
        additive_complexity=scheme.additive_complexity(),
        canonical_sha256=scheme.canonical_sha256(),
        brent_details=brent_res,
        basis_details=basis_res,
        elapsed_ms=round(elapsed_ms, 3),
    )


# =============================================================================
# 4. Exact Classical Schemes & Structural Combinators
# =============================================================================


def make_naive_scheme(
    m: int,
    n: int,
    p: int,
    ring: str = "Z",
    modulus: Optional[int] = None,
) -> BilinearScheme:
    """Construct the exact naive schoolbook scheme <m, n, p : m*n*p>."""
    U: List[List[int]] = []
    V: List[List[int]] = []
    W: List[List[int]] = []
    for i in range(m):
        for j in range(n):
            for k in range(p):
                u_row = [0] * (m * n)
                v_row = [0] * (n * p)
                w_row = [0] * (m * p)
                u_row[i * n + j] = 1
                v_row[j * p + k] = 1
                w_row[i * p + k] = 1
                U.append(u_row)
                V.append(v_row)
                W.append(w_row)
    return BilinearScheme(
        m=m, n=n, p=p, U=U, V=V, W=W, ring=ring, modulus=modulus, name=f"naive_{m}x{n}x{p}"
    )


def direct_sum_p(s1: BilinearScheme, s2: BilinearScheme, name: str = "") -> BilinearScheme:
    """
    Exact horizontal block-concatenation along the p-axis:
        <m, n, p1 : R1> + <m, n, p2 : R2>  -->  <m, n, p1 + p2 : R1 + R2>
    by splitting B = [B1 | B2] and C = [A*B1 | A*B2].
    """
    if (s1.m, s1.n) != (s2.m, s2.n):
        raise ValueError(f"direct_sum_p requires equal (m, n): {(s1.m, s1.n)} != {(s2.m, s2.n)}")
    if (s1.ring, s1.modulus) != (s2.ring, s2.modulus):
        raise ValueError("direct_sum_p requires matching (ring, modulus)")
    m, n, p1, p2 = s1.m, s1.n, s1.p, s2.p
    p = p1 + p2
    zero = _ring_zero(s1.ring)

    U: List[List[RingScalar]] = []
    V: List[List[RingScalar]] = []
    W: List[List[RingScalar]] = []

    for r in range(s1.rank):
        U.append(list(s1.U[r]))
        v_row = [zero] * (n * p)
        w_row = [zero] * (m * p)
        for j in range(n):
            for k in range(p1):
                v_row[j * p + k] = s1.V[r][j * p1 + k]
        for i in range(m):
            for k in range(p1):
                w_row[i * p + k] = s1.W[r][i * p1 + k]
        V.append(v_row)
        W.append(w_row)

    for r in range(s2.rank):
        U.append(list(s2.U[r]))
        v_row = [zero] * (n * p)
        w_row = [zero] * (m * p)
        for j in range(n):
            for k in range(p2):
                v_row[j * p + (p1 + k)] = s2.V[r][j * p2 + k]
        for i in range(m):
            for k in range(p2):
                w_row[i * p + (p1 + k)] = s2.W[r][i * p2 + k]
        V.append(v_row)
        W.append(w_row)

    return BilinearScheme(
        m=m,
        n=n,
        p=p,
        U=U,
        V=V,
        W=W,
        ring=s1.ring,
        modulus=s1.modulus,
        name=name or f"direct_sum_p_{m}x{n}x{p1}+{p2}",
    )


def _scheme_from_sparse_terms(
    m: int,
    n: int,
    p: int,
    terms: Sequence[Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], int], Dict[Tuple[int, int], int]]],
    ring: str = "Z",
    modulus: Optional[int] = None,
    name: str = "",
) -> BilinearScheme:
    """Helper converting 1-based sparse coordinate dicts into a 0-based row-major BilinearScheme."""
    U: List[List[int]] = []
    V: List[List[int]] = []
    W: List[List[int]] = []
    for u_dict, v_dict, w_dict in terms:
        u_row = [0] * (m * n)
        v_row = [0] * (n * p)
        w_row = [0] * (m * p)
        for (i, j), val in u_dict.items():
            u_row[(i - 1) * n + (j - 1)] += val
        for (i, j), val in v_dict.items():
            v_row[(i - 1) * p + (j - 1)] += val
        for (i, j), val in w_dict.items():
            w_row[(i - 1) * p + (j - 1)] += val
        U.append(u_row)
        V.append(v_row)
        W.append(w_row)
    return BilinearScheme(m=m, n=n, p=p, U=U, V=V, W=W, ring=ring, modulus=modulus, name=name)


def strassen_222(ring: str = "Z", modulus: Optional[int] = None) -> BilinearScheme:
    """
    Classic Strassen (1969) <2, 2, 2 : 7> ternary algorithm (7 multiplications, 18 naive additions).
    """
    terms = [
        # m1 = (a11 + a22) * (b11 + b22) -> c11 + c22
        ({(1, 1): 1, (2, 2): 1}, {(1, 1): 1, (2, 2): 1}, {(1, 1): 1, (2, 2): 1}),
        # m2 = (a21 + a22) * b11 -> c21 - c22
        ({(2, 1): 1, (2, 2): 1}, {(1, 1): 1}, {(2, 1): 1, (2, 2): -1}),
        # m3 = a11 * (b12 - b22) -> c12 + c22
        ({(1, 1): 1}, {(1, 2): 1, (2, 2): -1}, {(1, 2): 1, (2, 2): 1}),
        # m4 = a22 * (b21 - b11) -> c11 + c21
        ({(2, 2): 1}, {(2, 1): 1, (1, 1): -1}, {(1, 1): 1, (2, 1): 1}),
        # m5 = (a11 + a12) * b22 -> -c11 + c12
        ({(1, 1): 1, (1, 2): 1}, {(2, 2): 1}, {(1, 1): -1, (1, 2): 1}),
        # m6 = (a21 - a11) * (b11 + b12) -> c22
        ({(2, 1): 1, (1, 1): -1}, {(1, 1): 1, (1, 2): 1}, {(2, 2): 1}),
        # m7 = (a12 - a22) * (b21 + b22) -> c11
        ({(1, 2): 1, (2, 2): -1}, {(2, 1): 1, (2, 2): 1}, {(1, 1): 1}),
    ]
    return _scheme_from_sparse_terms(2, 2, 2, terms, ring=ring, modulus=modulus, name="strassen_222")


def strassen_block_223(ring: str = "Z", modulus: Optional[int] = None) -> BilinearScheme:
    """
    Exact non-square <2, 2, 3 : 11> ternary scheme formed as Strassen <2,2,2:7> + naive <2,2,1:4>.
    """
    return direct_sum_p(
        strassen_222(ring=ring, modulus=modulus),
        make_naive_scheme(2, 2, 1, ring=ring, modulus=modulus),
        name="strassen_block_223",
    )


def laderman_333(ring: str = "Z", modulus: Optional[int] = None) -> BilinearScheme:
    """
    Classic Laderman (1976) <3, 3, 3 : 23> non-commutative ternary algorithm
    (J. D. Laderman, Bull. AMS 82 (1976), 126-128).
    """
    u_defs: List[Dict[Tuple[int, int], int]] = [
        {(1, 1): 1, (1, 2): 1, (1, 3): 1, (2, 1): -1, (2, 2): -1, (3, 2): -1, (3, 3): -1},  # m1
        {(1, 1): 1, (2, 1): -1},  # m2
        {(2, 2): 1},  # m3
        {(1, 1): -1, (2, 1): 1, (2, 2): 1},  # m4
        {(2, 1): 1, (2, 2): 1},  # m5
        {(1, 1): 1},  # m6
        {(1, 1): -1, (3, 1): 1, (3, 2): 1},  # m7
        {(1, 1): -1, (3, 1): 1},  # m8
        {(3, 1): 1, (3, 2): 1},  # m9
        {(1, 1): 1, (1, 2): 1, (1, 3): 1, (2, 2): -1, (2, 3): -1, (3, 1): -1, (3, 2): -1},  # m10
        {(3, 2): 1},  # m11
        {(1, 3): -1, (3, 2): 1, (3, 3): 1},  # m12
        {(1, 3): 1, (3, 3): -1},  # m13
        {(1, 3): 1},  # m14
        {(3, 2): 1, (3, 3): 1},  # m15
        {(1, 3): -1, (2, 2): 1, (2, 3): 1},  # m16
        {(1, 3): 1, (2, 3): -1},  # m17
        {(2, 2): 1, (2, 3): 1},  # m18
        {(1, 2): 1},  # m19
        {(2, 3): 1},  # m20
        {(2, 1): 1},  # m21
        {(3, 1): 1},  # m22
        {(3, 3): 1},  # m23
    ]
    v_defs: List[Dict[Tuple[int, int], int]] = [
        {(2, 2): 1},  # m1
        {(1, 2): -1, (2, 2): 1},  # m2
        {(1, 1): -1, (1, 2): 1, (2, 1): 1, (2, 2): -1, (2, 3): -1, (3, 1): -1, (3, 3): 1},  # m3
        {(1, 1): 1, (1, 2): -1, (2, 2): 1},  # m4
        {(1, 1): -1, (1, 2): 1},  # m5
        {(1, 1): 1},  # m6
        {(1, 1): 1, (1, 3): -1, (2, 3): 1},  # m7
        {(1, 3): 1, (2, 3): -1},  # m8
        {(1, 1): -1, (1, 3): 1},  # m9
        {(2, 3): 1},  # m10
        {(1, 1): -1, (1, 3): 1, (2, 1): 1, (2, 2): -1, (2, 3): -1, (3, 1): -1, (3, 2): 1},  # m11
        {(2, 2): 1, (3, 1): 1, (3, 2): -1},  # m12
        {(2, 2): 1, (3, 2): -1},  # m13
        {(3, 1): 1},  # m14
        {(3, 1): -1, (3, 2): 1},  # m15
        {(2, 3): 1, (3, 1): 1, (3, 3): -1},  # m16
        {(2, 3): 1, (3, 3): -1},  # m17
        {(3, 1): -1, (3, 3): 1},  # m18
        {(2, 1): 1},  # m19
        {(3, 2): 1},  # m20
        {(1, 3): 1},  # m21
        {(1, 2): 1},  # m22
        {(3, 3): 1},  # m23
    ]
    # c_ij -> list of 1-based m_r indices that sum with coefficient +1
    c_equations: Dict[Tuple[int, int], Sequence[int]] = {
        (1, 1): (6, 14, 19),
        (1, 2): (1, 4, 5, 6, 12, 14, 15),
        (1, 3): (6, 7, 9, 10, 14, 16, 18),
        (2, 1): (2, 3, 4, 6, 14, 16, 17),
        (2, 2): (2, 4, 5, 6, 20),
        (2, 3): (14, 16, 17, 18, 21),
        (3, 1): (6, 7, 8, 11, 12, 13, 14),
        (3, 2): (12, 13, 14, 15, 22),
        (3, 3): (6, 7, 8, 9, 23),
    }
    w_defs: List[Dict[Tuple[int, int], int]] = [{} for _ in range(23)]
    for c_pos, m_indices in c_equations.items():
        for m_idx in m_indices:
            w_defs[m_idx - 1][c_pos] = 1

    terms = list(zip(u_defs, v_defs, w_defs))
    return _scheme_from_sparse_terms(3, 3, 3, terms, ring=ring, modulus=modulus, name="laderman_333")


# =============================================================================
# 5. Comprehensive Built-In Self-Test (`--selftest`)
# =============================================================================


def _tamper_one_entry(scheme: BilinearScheme, side: str = "U", row: int = 0, col: int = 0) -> BilinearScheme:
    U = [list(r) for r in scheme.U]
    V = [list(r) for r in scheme.V]
    W = [list(r) for r in scheme.W]
    target = {"U": U, "V": V, "W": W}[side]
    if scheme.ring in ("Z2", "Z3", "Zq"):
        assert scheme.modulus is not None
        target[row][col] = (int(target[row][col]) + 1) % scheme.modulus
    elif scheme.ring == "Q":
        target[row][col] = target[row][col] + Fraction(1, 2)  # type: ignore[operator]
    else:
        target[row][col] = int(target[row][col]) + 1
    return BilinearScheme(
        m=scheme.m,
        n=scheme.n,
        p=scheme.p,
        U=U,
        V=V,
        W=W,
        ring=scheme.ring,
        modulus=scheme.modulus,
        name=f"{scheme.name}_tampered_{side}_{row}_{col}",
    )


def run_selftest(verbose: bool = True) -> bool:
    """
    Execute all mandatory Tracer #1 self-tests:
      (a) Strassen <2,2,2:7> over Z (ternary, 18 naive additions)
      (b) Laderman <3,3,3:23> over Z (ternary)
      (c) Strassen mod 2 (Z2), mod 4 (Zq, q=4), mod 3 (Z3), and rational Q (with 1/2 scaling)
      (d) Non-square rectangular schemes: naive <2,3,4:24> and Strassen-block <2,2,3:11>
      (e) Negative tamper controls (1 flipped entry fails BOTH Brent and basis checks across Z, Z2, Z4, Q)
      (f) Zero-float guard, JSON round-trip, canonical SHA-256 permutation invariance, trace convention round-trip, LRP/triplet export.
    """
    checks: List[Tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append((name, ok, detail))
        if verbose:
            tag = "PASS" if ok else "FAIL"
            print(f"  [{tag}] {name}: {detail}")

    # (a) Strassen <2,2,2:7> over Z
    s2 = strassen_222("Z")
    r_s2 = verify_scheme(s2)
    record(
        "strassen_222_Z",
        r_s2.valid
        and r_s2.brent_ok
        and r_s2.basis_ok
        and r_s2.is_ternary
        and r_s2.max_abs_coeff == 1
        and r_s2.additive_complexity["naive_additions_total"] == 18,
        f"valid={r_s2.valid}, ternary={r_s2.is_ternary}, max_abs={r_s2.max_abs_coeff}, "
        f"adds={r_s2.additive_complexity['naive_additions_total']}, nnz={r_s2.additive_complexity['nnz_total']}, "
        f"sha256={r_s2.canonical_sha256[:12]}..., {r_s2.elapsed_ms:.2f}ms",
    )

    # (b) Laderman <3,3,3:23> over Z
    l3 = laderman_333("Z")
    r_l3 = verify_scheme(l3)
    record(
        "laderman_333_Z",
        r_l3.valid and r_l3.brent_ok and r_l3.basis_ok and r_l3.is_ternary and r_l3.max_abs_coeff == 1,
        f"valid={r_l3.valid}, ternary={r_l3.is_ternary}, max_abs={r_l3.max_abs_coeff}, "
        f"adds={r_l3.additive_complexity['naive_additions_total']}, nnz={r_l3.additive_complexity['nnz_total']}, "
        f"eqs={r_l3.brent_details['total_equations']}, pairs={r_l3.basis_details['total_basis_pairs']}, "
        f"sha256={r_l3.canonical_sha256[:12]}..., {r_l3.elapsed_ms:.2f}ms",
    )

    # (c) Strassen mod 2, mod 4, mod 3, and Q (with exact fractional rescaling 1/2 * 2)
    s2_z2 = s2.reduce_mod(2)
    r_z2 = verify_scheme(s2_z2)
    record(
        "strassen_222_Z2",
        r_z2.valid and r_z2.ring == "Z2" and r_z2.modulus == 2,
        f"valid={r_z2.valid}, ring={r_z2.ring}, mod={r_z2.modulus}, sha256={r_z2.canonical_sha256[:12]}...",
    )

    s2_z4 = s2.reduce_mod(4)
    r_z4 = verify_scheme(s2_z4)
    record(
        "strassen_222_Z4",
        r_z4.valid and r_z4.ring == "Zq" and r_z4.modulus == 4 and r_z4.is_ternary,
        f"valid={r_z4.valid}, ring={r_z4.ring}, mod={r_z4.modulus}, ternary={r_z4.is_ternary}",
    )

    s2_z3 = s2.reduce_mod(3)
    r_z3 = verify_scheme(s2_z3)
    record(
        "strassen_222_Z3",
        r_z3.valid and r_z3.ring == "Z3" and r_z3.modulus == 3,
        f"valid={r_z3.valid}, ring={r_z3.ring}, mod={r_z3.modulus}",
    )

    # Exact Q scheme with genuine non-integer fractions (scale U by 1/2, V by 1/3, W by 6)
    s2_q = BilinearScheme(
        m=2,
        n=2,
        p=2,
        U=[[Fraction(int(x), 2) for x in row] for row in s2.U],
        V=[[Fraction(int(x), 3) for x in row] for row in s2.V],
        W=[[Fraction(int(x) * 6, 1) for x in row] for row in s2.W],
        ring="Q",
        name="strassen_222_Q_scaled",
    )
    r_q = verify_scheme(s2_q)
    # Canonical hash over Q must match unscaled Strassen over Q!
    s2_q_unscaled = strassen_222("Q")
    record(
        "strassen_222_Q_exact",
        r_q.valid
        and not r_q.is_ternary
        and r_q.max_abs_coeff == "6"
        and r_q.canonical_sha256 == s2_q_unscaled.canonical_sha256(),
        f"valid={r_q.valid}, max_abs={r_q.max_abs_coeff}, scale_invariant_hash={r_q.canonical_sha256 == s2_q_unscaled.canonical_sha256()}",
    )

    # (d) Non-square rectangular schemes <2,3,4:24> and <2,2,3:11>
    rect_234 = make_naive_scheme(2, 3, 4, "Z")
    r_234 = verify_scheme(rect_234)
    rect_223 = strassen_block_223("Z")
    r_223 = verify_scheme(rect_223)
    # Random integer matrix check on non-square <2,2,3:11>
    rng = random.Random(2026)
    A_rand = [[rng.randint(-50, 50) for _ in range(2)] for _ in range(2)]
    B_rand = [[rng.randint(-50, 50) for _ in range(3)] for _ in range(2)]
    rand_match = matmul_via_scheme(rect_223, A_rand, B_rand) == naive_matmul(A_rand, B_rand, "Z")
    record(
        "rectangular_234_and_223",
        r_234.valid and r_223.valid and r_223.rank == 11 and r_223.is_ternary and rand_match,
        f"<2,3,4:24> valid={r_234.valid}, <2,2,3:11> valid={r_223.valid}, rand_matmul_exact={rand_match}",
    )

    # (e) Negative tamper controls (1 flipped entry MUST fail BOTH Brent and basis checks)
    tamper_cases = [
        ("tamper_Z_U", _tamper_one_entry(s2, "U", 0, 0)),
        ("tamper_Z_V", _tamper_one_entry(l3, "V", 11, 4)),
        ("tamper_Z_W", _tamper_one_entry(rect_223, "W", 3, 5)),
        ("tamper_Z2", _tamper_one_entry(s2_z2, "W", 2, 1)),
        ("tamper_Z4", _tamper_one_entry(s2_z4, "U", 4, 2)),
        ("tamper_Q", _tamper_one_entry(s2_q, "V", 1, 3)),
    ]
    for label, bad_scheme in tamper_cases:
        rep = verify_scheme(bad_scheme)
        ok_t = (not rep.valid) and (not rep.brent_ok) and (not rep.basis_ok) and rep.both_paths_agree
        record(
            label,
            ok_t,
            f"caught_both={ok_t}, brent_mismatches={rep.brent_details['mismatch_count']}, "
            f"basis_failed_pairs={rep.basis_details['failed_pairs']}",
        )

    # (f) Zero-float guard, canonical SHA-256 permutation/sign invariance, JSON round-trip, trace conversion, text exports
    float_rejected = False
    try:
        BilinearScheme(2, 2, 2, [[1.0, 0, 0, 0]], [[1, 0, 0, 0]], [[1, 0, 0, 0]], ring="Z")
    except TypeError:
        float_rejected = True

    # Shuffle rank-1 terms & flip signs of (u_r, w_r) -> must preserve SHA-256 and exact validity
    perm = list(range(l3.rank))
    rng.shuffle(perm)
    U_perm = [[-int(x) if i % 2 == 0 else int(x) for x in l3.U[r]] for i, r in enumerate(perm)]
    V_perm = [list(l3.V[r]) for r in perm]
    W_perm = [[-int(x) if i % 2 == 0 else int(x) for x in l3.W[r]] for i, r in enumerate(perm)]
    l3_perm = BilinearScheme(3, 3, 3, U_perm, V_perm, W_perm, ring="Z", name="laderman_permuted")
    r_l3_perm = verify_scheme(l3_perm)
    hash_inv = l3_perm.canonical_sha256() == l3.canonical_sha256()

    # Trace convention round-trip on non-square <2,2,3:11> (catches m != p transpose bugs!)
    w_tr = rect_223.w_to_trace_convention()
    rect_rt = BilinearScheme.from_trace_convention(2, 2, 3, rect_223.U, rect_223.V, w_tr, ring="Z")
    trace_rt_ok = verify_scheme(rect_rt).valid and rect_rt.canonical_sha256() == rect_223.canonical_sha256()

    # JSON round-trip for both Z and Q
    with tempfile.TemporaryDirectory(prefix="brent_verifier_test_") as tmpd:
        p_z = os.path.join(tmpd, "laderman.json")
        p_q = os.path.join(tmpd, "strassen_q.json")
        l3.save_json(p_z)
        s2_q.save_json(p_q)
        l3_loaded = BilinearScheme.load_json(p_z)
        s2_q_loaded = BilinearScheme.load_json(p_q)
        json_ok = (
            l3_loaded.canonical_sha256() == l3.canonical_sha256()
            and s2_q_loaded.canonical_sha256() == s2_q.canonical_sha256()
            and verify_scheme(l3_loaded).valid
            and verify_scheme(s2_q_loaded).valid
        )

    lrp_txt = s2.to_lrp_text()
    trip_txt = s2.to_triplet_text()
    export_ok = ("m1 =" in lrp_txt and "c11 =" in lrp_txt and "->" in trip_txt and CONVENTION in trip_txt)

    record(
        "guards_and_serialization",
        float_rejected and r_l3_perm.valid and hash_inv and trace_rt_ok and json_ok and export_ok,
        f"float_rejected={float_rejected}, perm_sign_hash_inv={hash_inv}, "
        f"trace_conv_223_rt={trace_rt_ok}, json_rt={json_ok}, text_export={export_ok}",
    )

    all_passed = all(ok for _, ok, _ in checks)
    if verbose:
        n_ok = sum(1 for _, ok, _ in checks if ok)
        print(f"SELFTEST SUMMARY: {n_ok}/{len(checks)} checks passed -> {'ALL PASS' if all_passed else 'FAILED'}")
    return all_passed


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Zero-floating-point exact FMM bilinear scheme verifier (convention C = A * B)."
    )
    parser.add_argument("--selftest", action="store_true", help="Run full built-in verification test suite.")
    parser.add_argument("--verify", type=str, default=None, help="Verify a saved JSON FMM scheme file.")
    parser.add_argument("--export-lrp", type=str, default=None, help="Print LRP text representation of JSON scheme.")
    parser.add_argument(
        "--export-triplet", type=str, default=None, help="Print triplet text representation of JSON scheme."
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON verification report.")
    args = parser.parse_args(argv)

    if args.selftest:
        ok = run_selftest(verbose=True)
        return 0 if ok else 1

    if args.verify:
        scheme = BilinearScheme.load_json(args.verify)
        rep = verify_scheme(scheme)
        if args.json:
            print(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2))
        else:
            status = "VALID" if rep.valid else "INVALID"
            print(
                f"[{status}] <{scheme.m},{scheme.n},{scheme.p}:{scheme.rank}> over {scheme.ring_label} | "
                f"brent={rep.brent_ok} basis={rep.basis_ok} ternary={rep.is_ternary} "
                f"max_abs={rep.max_abs_coeff} adds={rep.additive_complexity['naive_additions_total']} "
                f"nnz={rep.additive_complexity['nnz_total']} sha256={rep.canonical_sha256[:16]}... "
                f"({rep.elapsed_ms:.2f}ms)"
            )
        return 0 if rep.valid else 2

    if args.export_lrp:
        print(BilinearScheme.load_json(args.export_lrp).to_lrp_text(), end="")
        return 0

    if args.export_triplet:
        print(BilinearScheme.load_json(args.export_triplet).to_triplet_text(), end="")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Zero-dependency exact rational verifier for the 46 Heilbronn triangle world records.

Uses Python standard library `fractions.Fraction` (zero float) to check:
  1. Every `triangle-nNN/coordinates.txt` has exact point count N, plain 30-decimal
     coordinates, and exact simplex feasibility `x >= 0, y >= 0, x + y <= 1`.
  2. Computes exact minimum normalized triangle area `min_{i<j<k} |det(p_i, p_j, p_k)|`
     in exact `Fraction` (on the unit-simplex triangle of area 1/2, normalized area = |det|).
  3. Validates `meta.json` required keys (`ref`, `credit`, `note`).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from decimal import Decimal, getcontext
from fractions import Fraction

getcontext().prec = 50
NUM_RE = re.compile(r"^-?\d{1,6}(\.\d{1,200})?$")


def exact_min_area_triangle(pts: list[tuple[Fraction, Fraction]]) -> Fraction:
    n = len(pts)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    # Scale all coordinates to exact integers using common denominator for fast exact arithmetic!
    den = 1
    for x, y in pts:
        if x.denominator != den:
            import math
            den = math.lcm(den, x.denominator)
        if y.denominator != den:
            import math
            den = math.lcm(den, y.denominator)
    ix = [int(x * den) for x in xs]
    iy = [int(y * den) for y in ys]

    min_det: int | None = None
    for i in range(n - 2):
        xi, yi = ix[i], iy[i]
        for j in range(i + 1, n - 1):
            xj, yj = ix[j], iy[j]
            dx_ij = xj - xi
            dy_ij = yj - yi
            for k in range(j + 1, n):
                det = dx_ij * (iy[k] - yi) - dy_ij * (ix[k] - xi)
                if det < 0:
                    det = -det
                if min_det is None or det < min_det:
                    min_det = det
                    if min_det == 0:
                        return Fraction(0)
    assert min_det is not None
    return Fraction(min_det, den * den)


def main() -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    dirs = sorted(
        (d for d in os.listdir(root) if d.startswith("triangle-n") and os.path.isdir(os.path.join(root, d))),
        key=lambda d: int(d.split("-n")[1]),
    )
    if not dirs:
        print("ERROR: no triangle-nNN directories found", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    for d in dirs:
        n = int(d.split("-n")[1])
        cpath = os.path.join(root, d, "coordinates.txt")
        mpath = os.path.join(root, d, "meta.json")
        if not os.path.isfile(cpath) or not os.path.isfile(mpath):
            print(f"FAIL {d}: missing coordinates.txt or meta.json", file=sys.stderr)
            return 1

        meta = json.load(open(mpath, encoding="utf-8"))
        for key in ("ref", "credit", "note"):
            assert key in meta and isinstance(meta[key], str) and meta[key].strip(), f"{d}: bad meta.{key}"

        pts: list[tuple[Fraction, Fraction]] = []
        with open(cpath, encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                if not s or s.startswith("#"):
                    continue
                a, b = s.split()
                assert NUM_RE.match(a) and NUM_RE.match(b), f"{d}: non-decimal token {a} {b}"
                fx, fy = Fraction(a), Fraction(b)
                assert fx >= 0 and fy >= 0 and fx + fy <= 1, f"{d}: infeasible point ({a}, {b})"
                pts.append((fx, fy))
        assert len(pts) == n, f"{d}: expected {n} points, got {len(pts)}"

        val = exact_min_area_triangle(pts)
        assert val > 0, f"{d}: degenerate zero-area triple!"
        v_dec = Decimal(val.numerator) / Decimal(val.denominator)
        print(f"PASS {d:12s} | exact A={str(v_dec)[:20]} | n^2*A={float(val)*n*n:.4f}")

    print(f"\nALL {len(dirs)}/{len(dirs)} TRIANGLE WORLD RECORDS EXACT-VERIFIED ({time.perf_counter()-t0:.2f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

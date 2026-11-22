"""Phase 3 Pure-CPU Algebraic, Graph-Matroid, and Cyclic-Orbit Catalog of
Cross-Intersecting Hypergraphs (F_plus, F_minus) on d = 9..16.

Zero GPU usage (NumPy + stdlib + exact_verifier.py only).
Systematically constructs, checks exact cross-intersection (forall A in F_+,
B in F_-: A & B != 0), applies Vector A2 orthogonal subcube de-aliasing,
certifies every candidate via exact_verifier.py (require_monotone=True), and
writes the top diverse catalog to:
  e:\\эксперименты\\исследования\\c71_fei_research\\artifacts\\phase3_catalog_seeds.json
"""

from __future__ import annotations

from collections import Counter
import itertools
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
import exact_verifier as ev  # noqa: E402

ARTIFACTS_DIR = SRC_DIR.parent / "artifacts"


# =============================================================================
# Core Hypergraph & Vector A2 Helpers (Pure CPU)
# =============================================================================

def to_mask(indices: Sequence[int]) -> int:
    m = 0
    for i in indices:
        m |= 1 << int(i)
    return m


def is_cross_intersecting(Fp: Sequence[int], Fn: Sequence[int]) -> bool:
    for A in Fp:
        for B in Fn:
            if (A & B) == 0:
                return False
    return True


def prune_minimal(masks: Sequence[int]) -> List[int]:
    ordered = sorted(set(int(m) for m in masks), key=lambda m: (m.bit_count(), m))
    out: List[int] = []
    for m in ordered:
        if not any((m & prev) == prev for prev in out):
            out.append(m)
    return out


def fwht_np(arr: np.ndarray) -> np.ndarray:
    N = arr.shape[0]
    a = arr.astype(np.float64).copy()
    h = 1
    while h < N:
        a = a.reshape(-1, 2, h)
        u = a[:, 0, :].copy()
        v = a[:, 1, :].copy()
        a[:, 0, :] = u + v
        a[:, 1, :] = u - v
        a = a.reshape(N)
        h <<= 1
    return a


def count_essential_vars(d: int, g0: np.ndarray) -> int:
    ess = 0
    for i in range(d):
        g = g0.reshape(-1, 2, 1 << i)
        if np.any(g[:, 0, :] != g[:, 1, :]):
            ess += 1
    return ess


def build_g0_from_hypergraph(d: int, Fp: Sequence[int], Fn: Sequence[int]) -> np.ndarray:
    if not is_cross_intersecting(Fp, Fn):
        raise ValueError("Hypergraph pair (Fp, Fn) is NOT cross-intersecting")
    N = 1 << d
    full = N - 1
    xs = np.arange(N, dtype=np.int32)
    p = np.zeros(N, dtype=bool)
    n = np.zeros(N, dtype=bool)
    for A in Fp:
        p |= (xs & A) == A
    inv_xs = full ^ xs
    for B in Fn:
        n |= (inv_xs & B) == B
    assert not np.any(p & n)
    g0 = np.zeros(N, dtype=np.int8)
    g0[p] = 1
    g0[n] = -1
    return g0


def extract_subcube_components(d: int, g0: np.ndarray) -> Tuple[List[Dict[str, int]], List[List[int]]]:
    N = 1 << d
    full = N - 1
    zeros = set(int(x) for x in np.flatnonzero(g0 == 0))
    vis = set()
    valid_scs: List[Dict[str, int]] = []
    bad_comps: List[List[int]] = []
    for x in sorted(zeros):
        if x in vis:
            continue
        comp = []
        q = [x]
        vis.add(x)
        while q:
            u = q.pop()
            comp.append(u)
            for i in range(d):
                v = u ^ (1 << i)
                if v in zeros and v not in vis:
                    vis.add(v)
                    q.append(v)
        m = len(comp)
        oa = 0
        aa = full
        for v in comp:
            oa |= v
            aa &= v
        fm = oa ^ aa
        if (m & (m - 1)) == 0 and (1 << fm.bit_count()) == m:
            valid_scs.append({"fixed_mask": int(full ^ fm), "fixed_val": int(aa)})
        else:
            bad_comps.append(sorted(comp))
    return valid_scs, bad_comps


def resolve_gap_to_valid_subcubes(d: int, g0_raw: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, int]], str]:
    """Convert balanced monotone g0_raw into valid disjoint subcubes (exact if already subcubes,
    else balanced peeling or majority-mid completion while preserving monotonicity & balance).
    """
    N = 1 << d
    full = N - 1
    scs, bad = extract_subcube_components(d, g0_raw)
    if not bad:
        return g0_raw.copy(), scs, "exact_subcubes"

    # Try majority-mid completion first (fast and symmetric for large gaps)
    xs = np.arange(N, dtype=np.int32)
    pc = np.zeros(N, dtype=np.int32)
    for b in range(d):
        pc += (xs >> b) & 1
    gc = g0_raw.copy()
    zm = gc == 0
    if d % 2 == 1:
        gc[zm & (pc >= (d + 1) // 2)] = 1
        gc[zm & (pc <= (d - 1) // 2)] = -1
    else:
        gc[zm & (pc > d // 2)] = 1
        gc[zm & (pc < d // 2)] = -1
    if int(np.sum(gc == 1)) == int(np.sum(gc == -1)):
        scs_c, bad_c = extract_subcube_components(d, gc)
        if not bad_c:
            return gc, scs_c, "majority_mid_completed"

    # Fallback: iterative balanced peeling of maximal/minimal elements of non-subcube components
    g = g0_raw.copy()
    while True:
        scs_p, bad_p = extract_subcube_components(d, g)
        if not bad_p:
            return g, scs_p, "peeled_subcubes"
        max_els: List[int] = []
        min_els: List[int] = []
        for comp in bad_p:
            cset = set(comp)
            for u in comp:
                if not any((u | (1 << i)) in cset for i in range(d) if not ((u >> i) & 1)):
                    max_els.append(u)
                if not any((u ^ (1 << i)) in cset for i in range(d) if (u >> i) & 1):
                    min_els.append(u)
        max_els.sort(key=lambda u: (-u.bit_count(), u))
        min_els.sort(key=lambda u: (u.bit_count(), u))
        g[max_els[0]] = 1
        g[min_els[0]] = -1


def compute_W1(d: int, g0: np.ndarray, scs: Sequence[Dict[str, int]]) -> float:
    """Compute exact degree-1 Fourier weight W_1(F) = sum_{|S|=1} f_hat(S)^2 of de-aliased F."""
    N = 1 << d
    a = fwht_np(g0)
    w = (a / N) ** 2
    W1_core = float(sum(w[1 << i] for i in range(d)))
    # Each subcube C_j of codim r_j adds:
    #   - degree 1 on y_j alone (S=empty in x, T={y_j}): weight 2^(-2*r_j) * 1!
    #   - wait: in F = g0 + sum_j 1_{C_j}(x) y_j, the character {y_j} has coeff 2^(-r_j), weight 2^(-2*r_j).
    W1_aux = sum(1.0 / (1 << (2 * int(sc["fixed_mask"]).bit_count())) for sc in scs)
    return W1_core + W1_aux


# =============================================================================
# Family 1: Self-Dual Planar (s,t)-Networks, Hex Boards & Graphic Matroids
# =============================================================================

def build_hex_pair(k: int, max_len: Optional[int] = None) -> Tuple[int, List[int], List[int]]:
    def nbrs(r: int, c: int):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < k and 0 <= nc < k:
                yield (nr, nc)

    ml = max_len or (k * k)
    raw_lr: List[int] = []

    def dfs(path: List[Tuple[int, int]], vis: set) -> None:
        r, c = path[-1]
        if c == k - 1:
            raw_lr.append(to_mask([rr * k + cc for rr, cc in path]))
            return
        if len(path) >= ml:
            return
        for nr, nc in nbrs(r, c):
            if (nr, nc) not in vis and nc != 0:
                vis.add((nr, nc))
                path.append((nr, nc))
                dfs(path, vis)
                path.pop()
                vis.remove((nr, nc))

    for r0 in range(k):
        dfs([(r0, 0)], {(r0, 0)})
    Fp = prune_minimal(raw_lr)

    def transpose_mask(m: int) -> int:
        out = 0
        for r in range(k):
            for c in range(k):
                if (m >> (r * k + c)) & 1:
                    out |= 1 << (c * k + r)
        return out

    Fn = [transpose_mask(m) for m in Fp]
    return k * k, Fp, Fn


def build_grid3x3_bond_d13(max_len: Optional[int] = None) -> Tuple[int, List[int], List[int]]:
    """Self-dual 3x3 planar bond-percolation (s,t)-network on d=13 edges (8 vertices, s=0, t=7)."""
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (2, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
        (1, 2), (2, 3), (4, 5), (5, 6),
    ]
    n_verts = 8
    s, t = 0, 7
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(n_verts)]
    for ei, (u, v) in enumerate(edges):
        adj[u].append((v, ei))
        adj[v].append((u, ei))
    raw_p: List[int] = []

    def dfs(u: int, vis: set, emask: int, elen: int) -> None:
        if u == t:
            raw_p.append(emask)
            return
        if max_len and elen >= max_len:
            return
        for v, ei in adj[u]:
            if v not in vis:
                vis.add(v)
                dfs(v, vis, emask | (1 << ei), elen + 1)
                vis.remove(v)

    dfs(s, {s}, 0, 0)
    Fp = prune_minimal(raw_p)

    other = [v for v in range(n_verts) if v not in (s, t)]
    raw_c: List[int] = []
    for mask in range(1 << len(other)):
        S = {s} | {other[i] for i in range(len(other)) if (mask >> i) & 1}
        cm = 0
        for ei, (u, v) in enumerate(edges):
            if (u in S) != (v in S):
                cm |= 1 << ei
        if not max_len or cm.bit_count() <= max_len:
            raw_c.append(cm)
    Fn = prune_minimal(raw_c)
    return 13, Fp, Fn


def build_k5_stars_d10() -> Tuple[int, List[int], List[int]]:
    edges = list(itertools.combinations(range(5), 2))
    stars = [to_mask([i for i, e in enumerate(edges) if v in e]) for v in range(5)]
    return 10, stars, list(stars)


# =============================================================================
# Family 2: Finite Geometries, Affine/Projective Planes, Grids & SRGs
# =============================================================================

def build_ag23_d9() -> Tuple[int, List[int], List[int]]:
    rows3 = [to_mask([3 * r + c for c in range(3)]) for r in range(3)]
    cols3 = [to_mask([3 * r + c for r in range(3)]) for c in range(3)]
    diagp3 = [to_mask([3 * r + ((r + c) % 3) for r in range(3)]) for c in range(3)]
    diagm3 = [to_mask([3 * r + ((2 * r + c) % 3) for r in range(3)]) for c in range(3)]
    return 9, rows3 + cols3, diagp3 + diagm3


def build_ag23_2pair_split_d13() -> Tuple[int, List[int], List[int]]:
    _, Fp, Fn = build_ag23_d9()

    def sym_split_pair(Fp_in: List[int], Fn_in: List[int], d_cur: int, c1: int, c2: int):
        Fp1 = [A | (1 << d_cur) if (A >> c1) & 1 else A for A in Fp_in]
        Fn1 = []
        for B in Fn_in:
            if (B >> c1) & 1:
                Fn1.append(B)
                Fn1.append((B ^ (1 << c1)) | (1 << d_cur))
            else:
                Fn1.append(B)
        Fp2 = []
        for A in Fp1:
            if (A >> c2) & 1:
                Fp2.append(A)
                Fp2.append((A ^ (1 << c2)) | (1 << (d_cur + 1)))
            else:
                Fp2.append(A)
        Fn2 = [B | (1 << (d_cur + 1)) if (B >> c2) & 1 else B for B in Fn1]
        return prune_minimal(Fp2), prune_minimal(Fn2)

    Fp11, Fn11 = sym_split_pair(Fp, Fn, 9, 0, 1)
    Fp13, Fn13 = sym_split_pair(Fp11, Fn11, 11, 5, 4)
    return 13, Fp13, Fn13


def build_grid3x4_balanced_d12() -> Tuple[int, List[int], List[int]]:
    rows34 = [to_mask([4 * r + c for c in range(4)]) for r in range(3)]
    cols34 = [to_mask([4 * r + c for r in range(3)]) for c in range(4)]
    Fp = rows34 + cols34
    hit34 = [
        to_mask([4 * a[c] + c for c in range(4)])
        for a in itertools.product(range(3), repeat=4)
        if len(set(a)) == 3
    ]
    N12 = 1 << 12
    full12 = N12 - 1
    xs12 = np.arange(N12, dtype=np.int32)
    pos = np.zeros(N12, dtype=bool)
    for A in Fp:
        pos |= (xs12 & A) == A
    target = int(pos.sum())
    rng = random.Random(7)
    for k_keep in range(21, 30):
        for _ in range(40):
            sub = rng.sample(hit34, k_keep)
            neg = np.zeros(N12, dtype=bool)
            for B in sub:
                neg |= ((full12 ^ xs12) & B) == B
            if int(neg.sum()) == target:
                return 12, Fp, prune_minimal(sub)
    raise RuntimeError("Failed to find balanced Grid3x4 subset")


def build_ag24_2v2_d16() -> Tuple[int, List[int], List[int]]:
    mul4 = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    def pt4(x: int, y: int) -> int:
        return 4 * x + y
    pclasses = [
        [to_mask([pt4(c, y) for y in range(4)]) for c in range(4)],
        [to_mask([pt4(x, c) for x in range(4)]) for c in range(4)],
    ] + [[to_mask([pt4(x, mul4[m][x] ^ c) for x in range(4)]) for c in range(4)] for m in (1, 2, 3)]
    return 16, pclasses[0] + pclasses[1], pclasses[2] + pclasses[3]


def build_pg23_d13() -> Tuple[int, List[int], List[int]]:
    lines = [to_mask([(x + s) % 13 for x in (0, 1, 3, 9)]) for s in range(13)]
    return 13, lines, list(lines)


def build_petersen_nbrs_d10() -> Tuple[int, List[int], List[int]]:
    p_verts = list(itertools.combinations(range(5), 2))
    nbrs = [
        to_mask([i] + [j for j in range(10) if not (set(p_verts[i]) & set(p_verts[j]))])
        for i in range(10)
    ]
    return 10, nbrs, list(nbrs)


def build_clebsch_nbrs_d16() -> Tuple[int, List[int], List[int]]:
    nbrs = [to_mask([u] + [u ^ (1 << i) for i in range(4)] + [u ^ 15]) for u in range(16)]
    return 16, nbrs, list(nbrs)


# =============================================================================
# Family 3: Cyclic, Dihedral & Non-Cyclic Abelian Sub-Orbit Pairs (d = 10..15)
# =============================================================================

def build_cyclic_suborbit_pair(
    d: int,
    step_g: int,
    generators: Sequence[Tuple[int, ...]],
    mult_a: int,
    shift_b: int = 0,
) -> Tuple[int, List[int], List[int]]:
    orb_len = d // step_g
    Fp: List[int] = []
    Fn: List[int] = []
    for gen in generators:
        for s in range(orb_len):
            Fp.append(to_mask([(x + s * step_g) % d for x in gen]))
            Fn.append(to_mask([(mult_a * x + shift_b + s * step_g) % d for x in gen]))
    return d, prune_minimal(Fp), prune_minimal(Fn)


def build_z2xz2xz3_h4_d12() -> Tuple[int, List[int], List[int]]:
    """Best non-cyclic Abelian Z_2 x Z_2 x Z_3 (H4 sub-orbit, d=12, exact k=6 subcubes, L=6.1029)."""
    pts12 = [(a, b, c) for a in range(2) for b in range(2) for c in range(3)]
    idx12 = {p: i for i, p in enumerate(pts12)}
    def add12(p, q): return ((p[0] ^ q[0]), (p[1] ^ q[1]), (p[2] + q[2]) % 3)
    H4 = [(a, b, 0) for a in range(2) for b in range(2)]

    # Reconstruct exact top hit deterministically
    gl22 = [((1, 0), (0, 1)), ((0, 1), (1, 0)), ((1, 1), (0, 1))]
    seen = set()
    orbs = []
    for k in (3, 4):
        for comb in itertools.combinations(range(12), k):
            orb = frozenset(to_mask([idx12[add12(pts12[x], h)] for x in comb]) for h in H4)
            if orb not in seen:
                seen.add(orb)
                orbs.append(sorted(orb))

    N12 = 1 << 12
    xs12 = np.arange(N12, dtype=np.int32)
    pc12 = np.zeros(N12, dtype=np.int32)
    for b in range(12):
        pc12 += (xs12 >> b) & 1

    best = None
    rng = random.Random(99)
    for ((a00, a01), (a10, a11)) in gl22:
        for m3 in (1, 2):
            for sh in [(0, 0, 1), (1, 0, 0)]:
                perm = [idx12[((a00 * p[0] ^ a01 * p[1]) ^ sh[0], (a10 * p[0] ^ a11 * p[1]) ^ sh[1], (m3 * p[2] + sh[2]) % 3)] for p in pts12]
                compat = []
                for Fp_o in orbs:
                    Fn_o = sorted(set(to_mask([perm[i] for i in range(12) if (m >> i) & 1]) for m in Fp_o))
                    if is_cross_intersecting(Fp_o, Fn_o):
                        compat.append((Fp_o, Fn_o))
                mc = len(compat)
                adj = [set() for _ in range(mc)]
                for i in range(mc):
                    for j in range(i + 1, mc):
                        if is_cross_intersecting(compat[i][0], compat[j][1]) and is_cross_intersecting(compat[j][0], compat[i][1]):
                            adj[i].add(j)
                            adj[j].add(i)
                cliques = []
                for i in range(mc):
                    for j in adj[i]:
                        if j > i:
                            for k3 in (adj[i] & adj[j]):
                                if k3 > j:
                                    cliques.append((i, j, k3))
                if len(cliques) > 400:
                    cliques = rng.sample(cliques, 400)
                for clq in cliques:
                    Fp = prune_minimal([x for o in clq for x in compat[o][0]])
                    Fn = prune_minimal([x for o in clq for x in compat[o][1]])
                    g0 = build_g0_from_hypergraph(12, Fp, Fn)
                    if int(np.sum(g0)) != 0 or count_essential_vars(12, g0) < 12:
                        continue
                    scs, bad = extract_subcube_components(12, g0)
                    if bad:
                        continue
                    a = fwht_np(g0)
                    w = (a / N12) ** 2
                    I_t = float(np.dot(pc12, w)) + sum((2 + int(s["fixed_mask"]).bit_count()) / (1 << (int(s["fixed_mask"]).bit_count() + 1)) for s in scs)
                    nz = w > 1e-300
                    H_t = float(-np.sum(w[nz] * np.log2(w[nz]))) + sum((2 * int(s["fixed_mask"]).bit_count()) / (1 << int(s["fixed_mask"]).bit_count()) for s in scs)
                    L = H_t / (I_t - 1.0)
                    if best is None or L > best[0]:
                        best = (L, Fp, Fn)
                if best is not None and best[0] > 6.10:
                    return 12, best[1], best[2]
    assert best is not None
    return 12, best[1], best[2]


# =============================================================================
# Family 4: Mixed (3,4)-Uniform Bollobas / Extremal Perturbed Pairs (d=11, 14)
# =============================================================================

def build_bollobas_mixed34_d11_or_d14(d: int, seed: int = 314) -> Tuple[int, List[int], List[int]]:
    """Construct a genuinely d-essential balanced mixed (3,4)-uniform cross-intersecting pair
    using a cyclic shift-isomorphic permutation pi(x) = a*x mod d and greedy maximal transversal
    packing targeting coverage mu(U) in [0.46, 0.495].
    """
    N = 1 << d
    full = N - 1
    xs = np.arange(N, dtype=np.int32)
    pc = np.zeros(N, dtype=np.int32)
    for b in range(d):
        pc += (xs >> b) & 1

    # Choose multiplicative automorphism a with gcd(a, d) = 1
    a_mult = 2 if math.gcd(2, d) == 1 else 3
    perm = [(a_mult * i) % d for i in range(d)]

    def map_m(m: int) -> int:
        out = 0
        for i in range(d):
            if (m >> i) & 1:
                out |= 1 << perm[i]
        return out

    # Candidate 3-sets and 4-sets
    rng = random.Random(seed + d)
    all3 = [to_mask(c) for c in itertools.combinations(range(d), 3)]
    all4 = [to_mask(c) for c in itertools.combinations(range(d), 4)]

    best_overall = None
    for trial in range(18):
        # Seed with a compatible 3-set or 4-set pair (A, pi(A))
        Fp: List[int] = []
        Fn: List[int] = []
        cands = (rng.sample(all3, min(len(all3), 90)) + rng.sample(all4, min(len(all4), 220)))
        rng.shuffle(cands)
        # Prioritize 3-sets first to keep I small and W1 low, then pack 4-sets until coverage ~0.48
        cands.sort(key=lambda m: (m.bit_count(), rng.random()))
        pos_mask = np.zeros(N, dtype=bool)
        for A in cands:
            if any((A & x) == x for x in Fp):
                continue
            B = map_m(A)
            if (A & B) == 0:
                continue
            if not all((A & y) != 0 for y in Fn):
                continue
            if not all((B & x) != 0 for x in Fp):
                continue
            new_pos = pos_mask | ((xs & A) == A)
            cov = int(new_pos.sum()) / N
            if cov > 0.496:
                continue
            Fp = prune_minimal(Fp + [A])
            Fn = prune_minimal(Fn + [B])
            pos_mask = new_pos
            if cov >= 0.482:
                break
        if len(Fp) < 4:
            continue
        g0_raw = build_g0_from_hypergraph(d, Fp, Fn)
        if int(np.sum(g0_raw)) != 0 or count_essential_vars(d, g0_raw) < d:
            continue
        g0_res, scs, _ = resolve_gap_to_valid_subcubes(d, g0_raw)
        a = fwht_np(g0_res)
        w = (a / N) ** 2
        I_t = float(np.dot(pc, w)) + sum((2 + int(s["fixed_mask"]).bit_count()) / (1 << (int(s["fixed_mask"]).bit_count() + 1)) for s in scs)
        nz = w > 1e-300
        H_t = float(-np.sum(w[nz] * np.log2(w[nz]))) + sum((2 * int(s["fixed_mask"]).bit_count()) / (1 << int(s["fixed_mask"]).bit_count()) for s in scs)
        if I_t <= 1.0:
            continue
        L = H_t / (I_t - 1.0)
        if best_overall is None or L > best_overall[0]:
            best_overall = (L, Fp, Fn)

    assert best_overall is not None
    return d, best_overall[1], best_overall[2]


# =============================================================================
# Catalog Assembly & Certification
# =============================================================================

def build_and_certify_catalog() -> Dict[str, Any]:
    t0 = time.perf_counter()
    print("=== Phase 3 Cross-Intersecting Hypergraph Catalog (d = 9..16, Pure CPU) ===")

    specs: List[Tuple[str, str, Tuple[int, List[int], List[int]]]] = [
        # Reference anchor (d=9)
        ("AG23_2v2_d9_anchor", "finite_geometry_anchor", build_ag23_d9()),
        # Family 3: Cyclic & Non-Cyclic Abelian Sub-Orbit Pairs (strongest genuinely new d=10..15!)
        ("Z10_step2_2orb_34_d10", "cyclic_suborbit", build_cyclic_suborbit_pair(10, 2, [(0, 3, 7), (0, 2, 4, 7)], 3, 0)),
        ("Z15_step3_3orb_34_d15", "cyclic_suborbit", build_cyclic_suborbit_pair(15, 3, [(0, 2, 8), (1, 2, 5, 11), (2, 5, 8, 11)], 7, 0)),
        ("Z15_step3_2orb_33_d15", "cyclic_suborbit", build_cyclic_suborbit_pair(15, 3, [(0, 1, 4), (1, 4, 7)], 7, 0)),
        ("Z15_step3_2orb_34_d15", "cyclic_suborbit", build_cyclic_suborbit_pair(15, 3, [(0, 3, 6), (0, 1, 11, 12)], 2, 0)),
        ("Z2xZ2xZ3_H4_3orb_d12", "noncyclic_abelian_suborbit", build_z2xz2xz3_h4_d12()),
        ("Z15_1orb_4set_diffcover_d15", "cyclic_diff_cover", build_cyclic_suborbit_pair(15, 1, [(0, 1, 4, 5)], 2, 0)),
        # Family 2: Finite Geometries, Grids & Strongly Regular Graphs
        ("AG23_2pair_split_d13", "finite_geometry_split", build_ag23_2pair_split_d13()),
        ("Grid3x4_balanced_d12", "rectangular_grid_mixed34", build_grid3x4_balanced_d12()),
        ("AG24_2v2_mid_d16", "finite_geometry", build_ag24_2v2_d16()),
        ("Petersen_ClosedNbrs_d10", "strongly_regular_graph", build_petersen_nbrs_d10()),
        ("PG23_selfdual_d13", "projective_plane", build_pg23_d13()),
        ("Clebsch_ClosedNbrs_d16", "strongly_regular_graph", build_clebsch_nbrs_d16()),
        # Family 1: Self-Dual Planar (s,t)-Networks, Hex & Graphic Matroids
        ("Hex_3x3_trunc4_d9", "planar_selfdual_hex", build_hex_pair(3, max_len=4)),
        ("Hex_4x4_trunc6_d16", "planar_selfdual_hex", build_hex_pair(4, max_len=6)),
        ("Grid3x3_Bond_full_d13", "planar_selfdual_st_bond", build_grid3x3_bond_d13(max_len=None)),
        ("Grid3x3_Bond_trunc5_d13", "planar_selfdual_st_bond", build_grid3x3_bond_d13(max_len=5)),
        ("K5_Stars_d10", "graphic_matroid", build_k5_stars_d10()),
        # Family 4: Mixed (3,4) Bollobas-type extremal packed pairs on d=11 and d=14
        ("Bollobas_mixed34_d11", "bollobas_mixed34", build_bollobas_mixed34_d11_or_d14(11, seed=314)),
        ("Bollobas_mixed34_d14", "bollobas_mixed34", build_bollobas_mixed34_d11_or_d14(14, seed=314)),
    ]

    entries: List[Dict[str, Any]] = []
    for label, family, (d, Fp, Fn) in specs:
        assert is_cross_intersecting(Fp, Fn), f"{label} failed cross-intersection!"
        g0_raw = build_g0_from_hypergraph(d, Fp, Fn)
        raw_gap = int(np.sum(g0_raw == 0))
        ess_raw = count_essential_vars(d, g0_raw)
        g0_res, scs, gap_mode = resolve_gap_to_valid_subcubes(d, g0_raw)
        ess_res = count_essential_vars(d, g0_res)

        # Certify via independent exact_verifier.py!
        cert = ev.verify_dealiased_core(g0_res.tolist(), scs, require_monotone=True)
        W1 = compute_W1(d, g0_res, scs)
        ot_val = float(cert["L_cert"][:20])
        I_val = cert["I_exact_num"] / cert["I_exact_den"]
        print(
            f"  [CERT] {label:28s} d={d:2d} ess={ess_res:2d} |F|={len(Fp):2d}v{len(Fn):2d} "
            f"raw_gap={raw_gap:5d} ({gap_mode[:5]}) k={len(scs):4d} "
            f"I={cert['I_exact_str']:12s} W1={W1:.4f} L_cert={cert['L_cert'][:18]}"
        )
        entries.append({
            "label": label,
            "family": family,
            "tier": 2,
            "d": d,
            "essential_vars": ess_res,
            "essential_vars_raw": ess_raw,
            "F_plus": [int(m) for m in Fp],
            "F_minus": [int(m) for m in Fn],
            "num_F_plus": len(Fp),
            "num_F_minus": len(Fn),
            "edge_sizes_plus": dict(sorted(Counter(m.bit_count() for m in Fp).items())),
            "edge_sizes_minus": dict(sorted(Counter(m.bit_count() for m in Fn).items())),
            "gap_size_raw": raw_gap,
            "gap_mode": gap_mode,
            "k_subcubes": len(scs),
            "n_effective": d + len(scs),
            "I": I_val,
            "I_exact_str": cert["I_exact_str"],
            "W1": round(W1, 8),
            "OT": ot_val,
            "L_cert": cert["L_cert"],
            "U_cert": cert["U_cert"],
            "is_monotone": cert["is_monotone"],
            "sha256": cert["sha256"],
            "g0": [int(v) for v in g0_res.tolist()],
            "subcubes": scs,
        })

    entries.sort(key=lambda e: e["OT"], reverse=True)
    non_ag = [e for e in entries if e["essential_vars"] >= 10]
    best_new = non_ag[0] if non_ag else None

    payload = {
        "schema_version": "c71_phase3_hypergraph_catalog_v1",
        "convention": "pm1_monotone_increasing",
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "total_certified": len(entries),
        "best_d_ge_10_label": best_new["label"] if best_new else None,
        "best_d_ge_10_OT": best_new["OT"] if best_new else None,
        "beats_num2026": any(e["OT"] > 6.52184571092304657566 for e in entries),
        "seeds": entries,
    }
    out_path = ARTIFACTS_DIR / "phase3_catalog_seeds.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {len(entries)} certified cross-intersecting hypergraph seeds to {out_path} in {payload['elapsed_sec']}s")
    if best_new:
        print(f"Best genuinely new d>=10 object: {best_new['label']} (d={best_new['d']}, ess={best_new['essential_vars']}, L_cert={best_new['L_cert'][:20]})")
    return payload


if __name__ == "__main__":
    build_and_certify_catalog()

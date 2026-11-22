"""
Phase 2 (Option B - Track 1): 100% Cold-Start From-Scratch Z_T World-Record Assault
on NVIDIA GeForce RTX 5060 (Blackwell SM 12.0) using Flip-Graph v2 + SLEEP-DAG-RSI v8.

Zero External Schemes Policy
----------------------------
EVERY seed is built strictly from scratch starting from schoolbook naive schemes plus
classic Strassen (1969) <2,2,2:7> and Laderman (1976) <3,3,3:23> (built into
`brent_verifier.py`), evolving our own verified Z_T primitive library on GPU:
  2x2x2:7, 2x2x3:11, 2x3x3:15, 2x3x4:20, 3x3x3:23, 2x4x4:26, 3x3x4:29, 3x4x4:38, 4x4x4:49.

Exact Verified Starting Ranks (asserted via `brent_verifier.py` BEFORE GPU launch):
  - 5x8x8: R_start == 230 (EXACT 2026 SOTA; +7.0 convexity ridge -> target <= 229!)
  - 7x7x7: R_start == 250 (EXACT 2026 Z_T SOTA; Q=249 -> target <= 249 ZT / <= 248 Q+ZT!)
  - 5x7x8: R_start <= 206 (+ fused 5x8x8 meta-flip projections -> target <= 203!)
  - 6x6x7: R_start <= 185 (+ fused 6x6x8:203 & 7x7x7 meta-flip projections -> target <= 182!)
  - Focused shot: 4x5x5 (Z_T target <= 75, F2 <= 72) & 4x5x6 (Z_T target <= 89, F2 <= 88)

Every world-record candidate is immediately dual-path certified by `brent_verifier.py`
and saved to `fmm_discovery/artifacts/record_zt_<m>x<n>x<p>_R<R>.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_FMM_DIR = os.path.abspath(os.path.join(_SRC_DIR, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_FMM_DIR, ".."))
_DREAM_DIR = os.path.join(_REPO_ROOT, "dream_rsi_scripts")
_ARTIFACTS_DIR = os.path.join(_FMM_DIR, "artifacts")

for _p in (_SRC_DIR, _DREAM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import brent_verifier as bv  # noqa: E402
import cartridge as dream_cartridge  # noqa: E402
import engine as dream_engine  # noqa: E402
import flip_graph_gpu as fg  # noqa: E402
import sentinel as dream_sentinel  # noqa: E402
import skill_state as dream_skill  # noqa: E402
import tree as dream_tree  # noqa: E402
import z3  # noqa: E402

SOTA_2026: Dict[str, Dict[str, int]] = {
    "5x8x8": {"m": 5, "n": 8, "p": 8, "z2": 230, "zt": 230, "q": 230},
    "7x7x7": {"m": 7, "n": 7, "p": 7, "z2": 245, "zt": 250, "q": 249},
    "5x7x8": {"m": 5, "n": 7, "p": 8, "z2": 204, "zt": 204, "q": 204},
    "6x6x7": {"m": 6, "n": 6, "p": 7, "z2": 183, "zt": 183, "q": 183},
    "4x5x5": {"m": 4, "n": 5, "p": 5, "z2": 73, "zt": 76, "q": 76},
    "4x5x6": {"m": 4, "n": 5, "p": 6, "z2": 89, "zt": 90, "q": 90},
}


def _fmt_key(m: int, n: int, p: int) -> str:
    return f"{m}x{n}x{p}"


def _verify_and_save_if_record(
    scheme: bv.BilinearScheme,
    records_saved: List[Dict[str, Any]],
    tag: str = "",
) -> bv.VerificationReport:
    """Strictly verify via dual-path brent_verifier; auto-save exact certificate if beating 2026 SOTA."""
    rep = bv.verify_scheme(scheme)
    if not rep.valid:
        raise RuntimeError(f"VERIFICATION FAILURE on {scheme.name}: {rep.brent_details}")
    if scheme.ring == "Z" and not rep.is_ternary:
        raise RuntimeError(f"Non-ternary Z scheme encountered: {scheme.name}")

    fk = _fmt_key(*scheme.shape)
    ref = SOTA_2026.get(fk)
    if ref is not None:
        is_rec = False
        rec_kind = ""
        if scheme.ring == "Z" and scheme.rank < ref["zt"]:
            is_rec = True
            rec_kind = f"ZT_and_Q_beats_{ref['q']}" if scheme.rank < ref["q"] else f"ZT_beats_{ref['zt']}"
        elif scheme.ring == "Z2" and scheme.rank < ref["z2"]:
            is_rec = True
            rec_kind = f"Z2_beats_{ref['z2']}"

        if is_rec:
            os.makedirs(_ARTIFACTS_DIR, exist_ok=True)
            if scheme.ring == "Z":
                fname = f"record_zt_{fk}_R{scheme.rank}.json"
                fpath = os.path.join(_ARTIFACTS_DIR, fname)
                should_write = True
                if os.path.exists(fpath):
                    try:
                        prev = bv.BilinearScheme.load_json(fpath)
                        if prev.additive_complexity()["nnz_total"] <= rep.additive_complexity["nnz_total"]:
                            should_write = False
                    except Exception:
                        pass
                if should_write:
                    scheme.metadata = {
                        **dict(scheme.metadata),
                        "record_kind": rec_kind,
                        "sota_before": ref,
                        "discovery_tag": tag,
                        "provenance": "100%_from_scratch_zero_external_seeds",
                        "verified_brent": True,
                        "verified_basis": True,
                        "is_ternary": rep.is_ternary,
                        "additive_complexity": rep.additive_complexity,
                        "discovered_at": datetime.now(timezone.utc).isoformat(),
                    }
                    scheme.save_json(fpath)
            else:
                fname = "(logged_in_summary_only)"
                fpath = ""
            info = {
                "format": fk,
                "ring": "Z_T" if scheme.ring == "Z" else scheme.ring,
                "rank": scheme.rank,
                "record_kind": rec_kind,
                "nnz": rep.additive_complexity["nnz_total"],
                "naive_additions": rep.additive_complexity["naive_additions_total"],
                "canonical_sha256": rep.canonical_sha256,
                "path": fpath,
            }
            records_saved.append(info)
            print(
                f"\n  *** [NEW 2026 WORLD RECORD CERTIFIED!] {fk} {info['ring']} R={scheme.rank} "
                f"({rec_kind}) nnz={info['nnz']} adds={info['naive_additions']} -> {fname} ***\n",
                flush=True,
            )
    return rep


def try_sign_lift_zt_bool(sc: bv.BilinearScheme, timeout_ms: int = 800) -> Optional[bv.BilinearScheme]:
    """
    Exact support-preserving F2 -> Z_T = {-1, 0, +1} sign-lifting via Z3 Bool-PB (ZERO external data).
    Each active entry in (U, V, W) is one Bool (False -> +1, True -> -1), triple product is XOR,
    and each exact integer Brent equation reduces to a single Pseudo-Boolean equality PbEq.
    """
    m, n, p, R = sc.m, sc.n, sc.p, sc.rank
    s = z3.Solver()
    s.set("timeout", int(timeout_ms))
    u_v: Dict[Tuple[int, int], Any] = {}
    v_v: Dict[Tuple[int, int], Any] = {}
    w_v: Dict[Tuple[int, int], Any] = {}
    by_a: List[List[int]] = [[] for _ in range(m * n)]
    by_b: List[List[int]] = [[] for _ in range(n * p)]
    by_c: List[List[int]] = [[] for _ in range(m * p)]
    for r in range(R):
        for a, val in enumerate(sc.U[r]):
            if int(val) & 1:
                u_v[(r, a)] = z3.Bool(f"u_{r}_{a}")
                by_a[a].append(r)
        for b, val in enumerate(sc.V[r]):
            if int(val) & 1:
                v_v[(r, b)] = z3.Bool(f"v_{r}_{b}")
                by_b[b].append(r)
        for c, val in enumerate(sc.W[r]):
            if int(val) & 1:
                w_v[(r, c)] = z3.Bool(f"w_{r}_{c}")
                by_c[c].append(r)
    # Fix 4^R sign-gauge symmetry: first nonzero entry of U[r] and V[r] = +1 (False)
    for r in range(R):
        for a in range(m * n):
            if (r, a) in u_v:
                s.add(z3.Not(u_v[(r, a)]))
                break
        for b in range(n * p):
            if (r, b) in v_v:
                s.add(z3.Not(v_v[(r, b)]))
                break
    for i1 in range(m):
        for i2 in range(n):
            a = i1 * n + i2
            sa = set(by_a[a])
            if not sa:
                continue
            for j1 in range(n):
                for j2 in range(p):
                    b = j1 * p + j2
                    sab = sa.intersection(by_b[b])
                    if not sab:
                        continue
                    for k1 in range(m):
                        for k2 in range(p):
                            c = k1 * p + k2
                            sabc = sorted(sab.intersection(by_c[c]))
                            rhs = 1 if (i1 == k1 and i2 == j1 and j2 == k2) else 0
                            k = len(sabc)
                            if k == 0:
                                if rhs != 0:
                                    return None
                                continue
                            if (k - rhs) & 1:
                                return None
                            need_neg = (k - rhs) // 2
                            if k == 1:
                                r0 = sabc[0]
                                neg = z3.Xor(z3.Xor(u_v[(r0, a)], v_v[(r0, b)]), w_v[(r0, c)])
                                s.add(neg if need_neg == 1 else z3.Not(neg))
                            else:
                                negs = [z3.Xor(z3.Xor(u_v[(r, a)], v_v[(r, b)]), w_v[(r, c)]) for r in sabc]
                                s.add(z3.PbEq([(x, 1) for x in negs], need_neg))
    if s.check() != z3.sat:
        return None
    mod = s.model()

    def _val(d: Dict[Tuple[int, int], Any], key: Tuple[int, int]) -> int:
        if key not in d:
            return 0
        return -1 if z3.is_true(mod.eval(d[key], model_completion=True)) else 1

    U = [[_val(u_v, (r, a)) for a in range(m * n)] for r in range(R)]
    V = [[_val(v_v, (r, b)) for b in range(n * p)] for r in range(R)]
    W = [[_val(w_v, (r, c)) for c in range(m * p)] for r in range(R)]
    cand = bv.BilinearScheme(m=m, n=n, p=p, U=U, V=V, W=W, ring="Z", name=f"zt_lifted_{m}x{n}x{p}_R{R}")
    rep = bv.verify_scheme(cand)
    return cand if (rep.valid and rep.is_ternary) else None


def _evolve_f2_and_lift_zt(
    engine: fg.FlipGraphGPUEngine,
    shape: Tuple[int, int, int],
    target_rank: int,
    want_count: int = 8,
    seed: int = 2027,
) -> List[bv.BilinearScheme]:
    """100% from-scratch GPU F2 cold-start -> Bool-PB exact Z_T sign-lifting + Z_T sparsification."""
    m, n, p = shape
    naive_f2 = bv.make_naive_scheme(m, n, p, "Z2")
    lifted: List[bv.BilinearScheme] = []
    seen: set[str] = set()
    for att in range(5):
        r_f2 = engine.evolve_f2(
            [naive_f2], target_rank=target_rank, n_rounds=8, n_walkers=4096,
            n_steps=35000, plus_period=1200, top_k=32, seed=seed + att * 1009,
        )
        if r_f2.best_rank > target_rank:
            continue
        f2_hits = [s for s in r_f2.top_schemes if s.rank == target_rank]
        r_div = engine.run_f2_batch(
            f2_hits, n_walkers=4096, n_steps=15000, plus_period=1200,
            target_rank=target_rank, diverse_mode=True, top_k=48, seed=seed + 500 + att * 97,
        )
        for sc_f2 in r_div.top_schemes:
            if sc_f2.rank != target_rank:
                continue
            zt = try_sign_lift_zt_bool(sc_f2, timeout_ms=700)
            if zt is not None:
                h = zt.canonical_sha256()
                if h not in seen:
                    seen.add(h)
                    lifted.append(zt)
                    if len(lifted) >= want_count:
                        break
        if len(lifted) >= want_count:
            break
    if lifted:
        # One fast 2s Z_T sparsification & diversification wave to minimize nnz before composition!
        r_sp = engine.run_zt_batch(
            lifted, n_walkers=2048, n_steps=20000, plus_period=1500, plus_headroom=1,
            top_k=16, seed=seed + 7777,
        )
        lifted = _keep_strict_min_rank(lifted + r_sp.top_schemes, max_keep=12)
    return lifted


# =============================================================================
# 1. Stage 0: Strict Min-Rank From-Scratch Z_T Primitive Block Library
# =============================================================================


def _keep_strict_min_rank(schemes: Sequence[bv.BilinearScheme], max_keep: int = 12) -> List[bv.BilinearScheme]:
    """Filter strictly to verified ternary schemes achieving the EXACT minimum rank."""
    valid_list = []
    for sc in schemes:
        rep = bv.verify_scheme(sc)
        if rep.valid and rep.is_ternary:
            valid_list.append(sc)
    if not valid_list:
        raise RuntimeError("No valid ternary schemes in pool")
    min_r = min(sc.rank for sc in valid_list)
    filtered = [sc for sc in valid_list if sc.rank == min_r]
    return filtered[:max_keep]


def build_strict_min_rank_library(
    engine: fg.FlipGraphGPUEngine, seed: int = 2027
) -> Dict[str, List[bv.BilinearScheme]]:
    """
    Evolve and strictly filter all primitive Z_T blocks to their EXACT 2026 optimal ranks
    purely from schoolbook naive + Strassen <2,2,2:7> + Laderman <3,3,3:23>.
    Asserts exact ranks: 2x3x4=20, 2x4x4=26, 3x3x4=29, 3x4x4=38, 4x4x4=49!
    """
    t0 = time.perf_counter()
    print("[Stage 0] Building strict min-rank from-scratch Z_T primitive library on RTX 5060 v2...", flush=True)
    lib: Dict[str, List[bv.BilinearScheme]] = {}

    s222 = bv.strassen_222("Z")
    s223 = bv.strassen_block_223("Z")
    s333 = bv.laderman_333("Z")
    s444 = fg.kronecker_product(s222, s222, name="strassen_444")
    assert bv.verify_scheme(s444).valid

    lib["2x2x2"] = [s222]
    lib["2x2x3"] = [s223]
    lib["4x4x4"] = [s444]

    # Diverse <3,3,3:23>
    r_333 = engine.evolve_zt(
        [bv.make_naive_scheme(3, 3, 3, "Z")], target_rank=23, n_rounds=5, n_walkers=4096,
        n_steps=20000, plus_period=600, top_k=24, seed=seed + 11,
    )
    lib["3x3x3"] = [s333] + [sc for sc in _keep_strict_min_rank(r_333.top_schemes, 10) if sc.rank == 23]

    # <2,3,3:15> from naive 18
    r_233 = engine.evolve_zt(
        [bv.make_naive_scheme(2, 3, 3, "Z")], target_rank=15, n_rounds=4, n_walkers=2048,
        n_steps=15000, plus_period=500, top_k=16, seed=seed + 22,
    )
    lib["2x3x3"] = _keep_strict_min_rank(r_233.top_schemes, 10)
    assert lib["2x3x3"][0].rank == 15, f"Expected 2x3x3=15, got {lib['2x3x3'][0].rank}"

    # <2,4,4:26> from naive 32 (GUARANTEE exact 26!)
    cur = [bv.make_naive_scheme(2, 4, 4, "Z")]
    best_244: List[bv.BilinearScheme] = []
    for attempt in range(5):
        r_244 = engine.evolve_zt(
            cur if attempt < 2 else [bv.make_naive_scheme(2, 4, 4, "Z")] + cur[:8],
            target_rank=26, n_rounds=6, n_walkers=4096,
            n_steps=35000, plus_period=1800, top_k=24, seed=seed + 33 + attempt * 1009,
        )
        cur = r_244.top_schemes
        best_244.extend(cur[:8])
        if r_244.best_rank <= 26:
            break
    lib["2x4x4"] = _keep_strict_min_rank(best_244, 12)
    assert lib["2x4x4"][0].rank <= 26, f"Expected 2x4x4<=26, got {lib['2x4x4'][0].rank}"

    # <2,3,4:20> from naive 24 + meta-flip projection of <2,4,4:26> (Hopcroft-Kerr exact rank 20!)
    s234_seeds = [bv.make_naive_scheme(2, 3, 4, "Z")] + [
        fg.best_project_slice(sc, "n") for sc in lib["2x4x4"][:6]
    ]
    cur = s234_seeds
    best_234: List[bv.BilinearScheme] = list(s234_seeds)
    for attempt in range(4):
        r_234 = engine.evolve_zt(
            cur, target_rank=20, n_rounds=5, n_walkers=4096,
            n_steps=30000, plus_period=1200, top_k=24, seed=seed + 27 + attempt * 1009,
        )
        cur = r_234.top_schemes
        best_234.extend(cur[:8])
        if r_234.best_rank <= 20:
            break
    lib["2x3x4"] = _keep_strict_min_rank(best_234, 10)
    assert lib["2x3x4"][0].rank <= 20, f"Expected 2x3x4<=20, got {lib['2x3x4'][0].rank}"

    # <3,3,4:29> from naive 36 (direct Z_T + from-scratch F2->Bool-PB sign-lift fallback)
    cur = [bv.make_naive_scheme(3, 3, 4, "Z")]
    best_334: List[bv.BilinearScheme] = []
    for attempt in range(3):
        r_334 = engine.evolve_zt(
            cur if attempt < 2 else [bv.make_naive_scheme(3, 3, 4, "Z")] + cur[:8],
            target_rank=29, n_rounds=6, n_walkers=4096,
            n_steps=40000, plus_period=2000, top_k=24, seed=seed + 44 + attempt * 1009,
        )
        cur = r_334.top_schemes
        best_334.extend(cur[:8])
        if r_334.best_rank <= 29:
            break
    if not best_334 or min(s.rank for s in best_334) > 29:
        best_334.extend(_evolve_f2_and_lift_zt(engine, (3, 3, 4), target_rank=29, want_count=8, seed=seed + 444))
    lib["3x3x4"] = _keep_strict_min_rank(best_334, 12)
    assert lib["3x3x4"][0].rank <= 29, f"Expected 3x3x4<=29, got {lib['3x3x4'][0].rank}"

    # <3,4,4:38> 100% from-scratch GPU F2 cold-start (190 M/s) -> Bool-PB exact Z_T sign-lifting (~58ms/scheme!)
    # + 2s Z_T sparsification wave for compact, diverse exact <3,4,4:38> primitives!
    lifted_344 = _evolve_f2_and_lift_zt(engine, (3, 4, 4), target_rank=38, want_count=10, seed=42)
    lib["3x4x4"] = _keep_strict_min_rank(lifted_344, 12)
    assert lib["3x4x4"][0].rank <= 38, f"Expected 3x4x4<=38, got {lib['3x4x4'][0].rank}"

    # Bonus fluid <4,5,2> & <4,5,3> for 4x5x5 / 4x5x6 direct-sum seeds (direct Z_T + from-scratch F2->Z_T boost)
    r_452 = engine.evolve_zt(
        [bv.make_naive_scheme(4, 5, 2, "Z")], target_rank=33, n_rounds=5, n_walkers=4096,
        n_steps=35000, plus_period=1800, top_k=16, seed=seed + 66,
    )
    pool_452 = list(r_452.top_schemes)
    if r_452.best_rank > 33:
        pool_452.extend(_evolve_f2_and_lift_zt(engine, (4, 5, 2), target_rank=33, want_count=4, seed=seed + 666))
    lib["4x5x2"] = _keep_strict_min_rank(pool_452, 8)

    r_453 = engine.evolve_zt(
        [bv.make_naive_scheme(4, 5, 3, "Z")], target_rank=46, n_rounds=5, n_walkers=4096,
        n_steps=35000, plus_period=1800, top_k=16, seed=seed + 77,
    )
    lib["4x5x3"] = _keep_strict_min_rank(r_453.top_schemes, 8)

    dt = time.perf_counter() - t0
    summary = {k: (v[0].rank, len(v)) for k, v in lib.items()}
    print(f"[Stage 0] Done in {dt:.1f}s | STRICT min-rank library (rank, count): {summary}", flush=True)
    return lib


def _pick_min_sub(
    lib: Dict[str, List[bv.BilinearScheme]],
    shape: Tuple[int, int, int],
    rng: random.Random,
    randomize: bool = True,
) -> bv.BilinearScheme:
    """Strictly pick ONLY from exact min-rank verified Z_T sub-schemes (fixes Phase 1 mixing bug!)."""
    canon = tuple(sorted(shape))
    key = _fmt_key(*canon)
    if key in lib and lib[key]:
        min_r = min(s.rank for s in lib[key])
        best_pool = [s for s in lib[key] if s.rank == min_r]
        chosen = rng.choice(best_pool) if randomize else best_pool[0]
        sc = fg.permute_axes(chosen, shape)
    else:
        sc = bv.make_naive_scheme(*shape, "Z")
    return fg.random_monomial_symmetry(sc, rng) if randomize else sc


def build_exact_sota_seeds(
    fmt_key: str,
    lib: Dict[str, List[bv.BilinearScheme]],
    rng: random.Random,
    extra_meta_seeds: Optional[Sequence[Tuple[str, bv.BilinearScheme]]] = None,
) -> List[Tuple[str, bv.BilinearScheme]]:
    """
    Construct verified exact-SOTA / near-SOTA from-scratch Z_T seed portfolios across
    orthogonal S_3 orientations + cross-dimension meta-flip projections!
    """
    seeds: List[Tuple[str, bv.BilinearScheme]] = []

    if fmt_key == "5x8x8":
        # Orientation A: ALIGNED canonical (5, 8, 8) split (2+3, 4+4, 4+4) -> 4*38 + 3*26 = 230 EXACT (max triple-collisions!)
        subs_a = [_pick_min_sub(lib, sh, rng, randomize=False) for sh in [(3,4,4),(3,4,4),(2,4,4),(3,4,4),(2,4,4),(3,4,4),(2,4,4)]]
        seeds.append(("ps_588_230_aligned", fg.padded_strassen_compose(2, 3, 4, 4, 4, 4, subs_a, name="5x8x8_230_aligned")))
        # Orientation B: (8, 5, 8) split (4+4, 3+2, 4+4) -> 4*38 + 3*26 = 230 EXACT!
        subs_b = [_pick_min_sub(lib, sh, rng) for sh in [(4,3,4),(4,3,4),(4,3,4),(4,2,4),(4,2,4),(4,3,4),(4,2,4)]]
        s_858 = fg.padded_strassen_compose(4, 4, 3, 2, 4, 4, subs_b, name="8x5x8_230")
        seeds.append(("ps_858_s3_230_B", fg.permute_axes(s_858, (5, 8, 8), name="5x8x8_230_B")))
        # Orientation C: (8, 8, 5) split (4+4, 4+4, 2+3) -> 4*38 + 3*26 = 230 EXACT!
        subs_c = [_pick_min_sub(lib, sh, rng) for sh in [(4,4,3),(4,4,2),(4,4,3),(4,4,2),(4,4,3),(4,4,3),(4,4,2)]]
        s_885 = fg.padded_strassen_compose(4, 4, 4, 4, 2, 3, subs_c, name="8x8x5_230")
        seeds.append(("ps_885_s3_230_C", fg.permute_axes(s_885, (5, 8, 8), name="5x8x8_230_C")))
        # Orientation D: independent symmetry-randomized (5, 8, 8) = 230 EXACT!
        subs_d = [_pick_min_sub(lib, sh, rng) for sh in [(3,4,4),(3,4,4),(2,4,4),(3,4,4),(2,4,4),(3,4,4),(2,4,4)]]
        seeds.append(("ps_588_230_D", fg.padded_strassen_compose(2, 3, 4, 4, 4, 4, subs_d, name="5x8x8_230_D")))

    elif fmt_key == "7x7x7":
        # v0 = aligned canonical (max cross-block triple-collisions!), v1..v3 = symmetry-rotated/transposed 250 EXACT!
        for k_i in range(4):
            subs = [_pick_min_sub(lib, sh, rng, randomize=(k_i > 0)) for sh in [(4,4,4),(3,4,4),(4,4,3),(3,3,4),(4,3,3),(3,4,3),(4,3,4)]]
            sc = fg.padded_strassen_compose(4, 3, 4, 3, 4, 3, subs, name=f"7x7x7_250_v{k_i}")
            if k_i == 1:
                sc = fg.cyclic_rotate_scheme(sc, name="7x7x7_250_c3")
            elif k_i == 2:
                sc = fg.transpose_scheme(sc, name="7x7x7_250_tr")
            seeds.append((f"ps_777_250_v{k_i}", sc))

    elif fmt_key == "5x7x8":
        # Optimal padded Strassen (7, 8, 5) split (3+4, 4+4, 2+3) -> 2*38 + 2*29 + 2*26 + 20 = 206!
        for k_i in range(2):
            subs = [_pick_min_sub(lib, sh, rng) for sh in [(4,4,3),(4,4,2),(3,4,3),(4,4,2),(3,4,3),(4,4,3),(3,4,2)]]
            s_785 = fg.padded_strassen_compose(3, 4, 4, 4, 2, 3, subs, name=f"7x8x5_206_v{k_i}")
            seeds.append((f"ps_785_s3_206_v{k_i}", fg.permute_axes(s_785, (5, 7, 8), name=f"5x7x8_206_v{k_i}")))
        # Second optimal permutation (5, 7, 8) split (3+2, 4+3, 4+4) -> 2*38 + 2*29 + 2*26 + 20 = 206!
        subs_b = [_pick_min_sub(lib, sh, rng) for sh in [(3,4,4),(2,4,4),(3,4,4),(2,3,4),(3,3,4),(2,4,4),(3,3,4)]]
        seeds.append(("ps_578_206_B", fg.padded_strassen_compose(3, 2, 4, 3, 4, 4, subs_b, name="5x7x8_206_B")))

    elif fmt_key == "6x6x7":
        # Optimal padded Strassen (6, 6, 7) split (3+3, 3+3, 3+4) -> 4*29 + 3*23 = 185!
        for k_i in range(2):
            subs = [_pick_min_sub(lib, sh, rng) for sh in [(3,3,4),(3,3,3),(3,3,4),(3,3,3),(3,3,4),(3,3,4),(3,3,3)]]
            seeds.append((f"ps_667_185_v{k_i}", fg.padded_strassen_compose(3, 3, 3, 3, 3, 4, subs, name=f"6x6x7_185_v{k_i}")))
        # S_3 permuted (7, 6, 6) split (4+3, 3+3, 3+3) -> 185 (different seam!)
        subs_m = [_pick_min_sub(lib, sh, rng) for sh in [(4,3,3),(3,3,3),(4,3,3),(3,3,3),(4,3,3),(3,3,3),(4,3,3)]]
        s_766 = fg.padded_strassen_compose(4, 3, 3, 3, 3, 3, subs_m, name="7x6x6_185")
        seeds.append(("ps_766_s3_185", fg.permute_axes(s_766, (6, 6, 7), name="6x6x7_185_s3")))

    elif fmt_key == "4x5x5":
        subs_a = [_pick_min_sub(lib, sh, rng) for sh in [(2,3,3),(2,3,3),(2,3,2),(2,2,3),(2,2,2),(2,3,2),(2,2,3)]]
        seeds.append(("ps_455_81_A", fg.padded_strassen_compose(2, 2, 3, 2, 3, 2, subs_a, name="4x5x5_81_A")))
        subs_b = [_pick_min_sub(lib, sh, rng) for sh in [(3,2,3),(3,2,2),(2,2,3),(3,2,2),(2,2,3),(3,2,3),(2,2,2)]]
        s_545 = fg.padded_strassen_compose(2, 3, 2, 2, 2, 3, subs_b, name="5x4x5_81")
        seeds.append(("ps_545_s3_81_B", fg.permute_axes(s_545, (4, 5, 5), name="4x5x5_81_B")))
        seeds.append(("ds_452_453", bv.direct_sum_p(lib["4x5x2"][0], lib["4x5x3"][0], name="4x5x5_ds")))
        seeds.append(("naive_455_100", bv.make_naive_scheme(4, 5, 5, "Z")))

    elif fmt_key == "4x5x6":
        subs_a = [_pick_min_sub(lib, sh, rng) for sh in [(2,3,3),(2,3,3),(2,3,3),(2,2,3),(2,2,3),(2,3,3),(2,2,3)]]
        seeds.append(("ps_456_93_A", fg.padded_strassen_compose(2, 2, 3, 2, 3, 3, subs_a, name="4x5x6_93_A")))
        subs_b = [_pick_min_sub(lib, sh, rng) for sh in [(2,3,3),(2,3,2),(2,3,3),(2,3,2),(2,3,3),(2,3,3),(2,3,2)]]
        s_465 = fg.padded_strassen_compose(2, 2, 3, 3, 2, 3, subs_b, name="4x6x5_93")
        seeds.append(("ps_465_s3_93_B", fg.permute_axes(s_465, (4, 5, 6), name="4x5x6_93_B")))
        seeds.append(("ds_453_453", bv.direct_sum_p(lib["4x5x3"][0], fg.random_monomial_symmetry(lib["4x5x3"][-1], rng), name="4x5x6_ds")))

    else:
        raise ValueError(f"Unsupported format {fmt_key}")

    if extra_meta_seeds:
        for lbl, sc in extra_meta_seeds:
            seeds.append((lbl, sc))

    # Strictly verify every seed before GPU launch
    for lbl, sc in seeds:
        rep = bv.verify_scheme(sc)
        if not (rep.valid and rep.is_ternary):
            raise RuntimeError(f"Seed {fmt_key}/{lbl} failed exact Z_T brent_verifier!")
    return seeds


def generate_meta_flip_seeds(
    engine: fg.FlipGraphGPUEngine,
    parent_elites: Sequence[bv.BilinearScheme],
    target_shape: Tuple[int, int, int],
    axis: str,
    label_prefix: str,
    lib: Optional[Dict[str, List[bv.BilinearScheme]]] = None,
    seed: int = 9001,
) -> List[Tuple[str, bv.BilinearScheme]]:
    """
    Cross-dimension meta-flip projection: take fused high-rank elites (or a freshly mixed
    6x6x8:203 parent) and drop 1 coordinate along `axis` via `best_project_slice` to land in
    brand-new non-Strassen basins!
    """
    out: List[Tuple[str, bv.BilinearScheme]] = []
    candidates = list(parent_elites)

    # Special booster for 6x6x7: build exact 6x6x8:203 (7 * 3x3x4:29 = 203!) -> mix on GPU -> project to 6x6x7!
    if target_shape == (6, 6, 7) and lib is not None:
        rng = random.Random(seed)
        subs_668 = [_pick_min_sub(lib, (3, 3, 4), rng) for _ in range(7)]
        s668 = fg.padded_strassen_compose(3, 3, 3, 3, 4, 4, subs_668, name="meta_6x6x8_203")
        assert s668.rank == 203 and bv.verify_scheme(s668).valid
        # Quick 3s GPU mixing wave with aggressive triple-collision splits so blocks fuse before slicing!
        res_mix = engine.run_zt_batch(
            [s668], n_walkers=1024, n_steps=12000, plus_period=250, plus_headroom=3, top_k=12, seed=seed
        )
        for sc_m in res_mix.top_schemes[:8]:
            if bv.verify_scheme(sc_m).valid:
                candidates.append(sc_m)

    best_proj: List[Tuple[int, int, bv.BilinearScheme]] = []
    seen = set()
    for p_sc in candidates:
        proj = fg.best_project_slice(p_sc, axis)
        if proj.shape != target_shape:
            proj = fg.permute_axes(proj, target_shape)
        rep = bv.verify_scheme(proj)
        if rep.valid and rep.is_ternary and rep.canonical_sha256 not in seen:
            seen.add(rep.canonical_sha256)
            best_proj.append((proj.rank, rep.additive_complexity["nnz_total"], proj))

    best_proj.sort(key=lambda x: (x[0], x[1]))
    for idx, (rk, nnz, sc) in enumerate(best_proj[:2]):
        out.append((f"{label_prefix}_R{rk}_{idx}", sc))
        print(f"  [Meta-Flip Projection] -> {target_shape} seed `{label_prefix}` R={rk} nnz={nnz}", flush=True)
    return out


# =============================================================================
# 2. Genuine SLEEP-DAG-RSI v8 High-R Z_T Assault Loop
# =============================================================================


def _score_zt(rank: int, nnz: int, sota_zt: int, ref_start_r: int) -> Tuple[float, List[float], List[int]]:
    score = 100.0 * float(ref_start_r - rank) - 0.01 * float(nnz) / max(rank, 1)
    buckets = [
        float(ref_start_r - 0.5 - rank),
        float(sota_zt + 2.5 - rank),
        float(sota_zt + 0.5 - rank),
        float(sota_zt - 0.5 - rank),
        float(sota_zt - 1.5 - rank),
    ]
    return score, buckets, [1 if v > 0.0 else 0 for v in buckets]


def run_sleep_dag_zt_assault_target(
    engine: fg.FlipGraphGPUEngine,
    fmt_key: str,
    lib: Dict[str, List[bv.BilinearScheme]],
    time_budget_s: float,
    skill_state: dream_skill.MutableSkillState,
    records_saved: List[Dict[str, Any]],
    extra_meta_seeds: Optional[Sequence[Tuple[str, bv.BilinearScheme]]] = None,
    assert_max_start_r: Optional[int] = None,
    seed: int = 2027,
) -> Tuple[Dict[str, Any], List[bv.BilinearScheme]]:
    """
    Run genuine SLEEP-DAG-RSI v8 Z_T assault on `fmt_key`, returning `(report_dict, top_elites)`.
    """
    t0 = time.perf_counter()
    rng = random.Random(seed)
    ref = SOTA_2026[fmt_key]
    m, n, p = ref["m"], ref["n"], ref["p"]
    sota_zt = ref["zt"]

    named_seeds = build_exact_sota_seeds(fmt_key, lib, rng, extra_meta_seeds=extra_meta_seeds)
    ref_start_r = min(sc.rank for _, sc in named_seeds)
    if assert_max_start_r is not None and ref_start_r > assert_max_start_r:
        raise AssertionError(
            f"Starting rank assertion failed for {fmt_key}: got {ref_start_r} > required {assert_max_start_r}"
        )

    # Check if any initial meta-flip projection ALREADY beat SOTA!
    for lbl, sc in named_seeds:
        _verify_and_save_if_record(sc, records_saved, tag=f"init_{fmt_key}_{lbl}")

    problem = dream_tree.Problem()
    problem.name = f"fmm_p2_zt_{fmt_key}"
    init_rep0 = bv.verify_scheme(named_seeds[0][1])
    base_score, base_buckets, base_mask = _score_zt(
        ref_start_r, init_rep0.additive_complexity["nnz_total"], sota_zt, ref_start_r
    )
    tree = dream_tree.DiscoveryTree(task=problem.name, baseline_score=base_score)
    tree.add(
        dream_tree.Node(
            id=tree.root_id, parent_id=None, branch=-1, attempt=0,
            score=base_score, fail_class="ok", valid=True,
            pass_mask=base_mask, bucket_scores=base_buckets,
            meta={"stage": "root", "rank": ref_start_r},
        )
    )

    policy = dream_engine.LeanDAGPolicy(
        beta=0.6, slope_weight=0.55, max_recovery=1,
        use_counterfactual=True, use_crossover=True, entropic_tau=4.0,
    )
    decomposer = dream_skill.BackwardSubGoalDecomposer()
    sentinel = dream_sentinel.WaldMicroSentinel(noise_regime="deterministic")

    branch_pools: Dict[int, List[bv.BilinearScheme]] = {}
    branch_best_node: Dict[int, str] = {}
    branch_meta: Dict[int, Dict[str, Any]] = {}
    all_verified_elites: Dict[str, Tuple[int, int, bv.BilinearScheme]] = {}

    for b_idx, (lbl, sc) in enumerate(named_seeds):
        rep = bv.verify_scheme(sc)
        nnz = rep.additive_complexity["nnz_total"]
        sc_val, b_sc, p_mk = _score_zt(sc.rank, nnz, sota_zt, ref_start_r)
        nid = f"b{b_idx}_init"
        tree.add(
            dream_tree.Node(
                id=nid, parent_id=tree.root_id, branch=b_idx, attempt=0,
                score=sc_val, fail_class="ok", valid=True,
                delta_vs_parent=sc_val - base_score, delta_vs_baseline=sc_val - base_score,
                pass_mask=p_mk, bucket_scores=b_sc,
                meta={"label": lbl, "rank": sc.rank, "nnz": nnz, "stage": "init"},
            )
        )
        branch_pools[b_idx] = [sc]
        branch_best_node[b_idx] = nid
        branch_meta[b_idx] = {"label": lbl, "best_r": sc.rank, "best_nnz": nnz, "stalls": 0}
        all_verified_elites[rep.canonical_sha256] = (sc.rank, nnz, sc)

    # High-R v2 throughput lets us run 40k-60k steps per wave cleanly in 2-4s!
    if ref_start_r <= 105:
        total_walkers_wave = 4096
        steps_wave = 60000
    elif ref_start_r <= 195:
        total_walkers_wave = 3072
        steps_wave = 40000
    else:
        total_walkers_wave = 2560
        steps_wave = 35000

    total_flips = 0
    total_plus = 0
    total_reduces = 0
    peak_vram = 0.0
    wave = 0
    best_r, best_nnz, best_scheme = min(all_verified_elites.values(), key=lambda x: (x[0], x[1]))
    decomposer_triggers = 0

    print(
        f"\n[Phase 2 Z_T | {fmt_key}] start_R={ref_start_r} (SOTA ZT={sota_zt}, Q={ref['q']}) | "
        f"{len(named_seeds)} branches {[lbl + ':R' + str(sc.rank) for lbl, sc in named_seeds]} | budget={time_budget_s:.0f}s",
        flush=True,
    )

    while (time.perf_counter() - t0) < time_budget_s:
        wave += 1
        b_ids = sorted(branch_pools.keys())
        b_scores = [tree.nodes[branch_best_node[b]].score for b in b_ids]
        tilted = dream_engine.compute_max_tilted_objective(
            b_scores, tau=policy.entropic_tau, baseline=base_score, normalize_by_scale=True
        )
        weights = tilted["tilted_weights_normalized"]
        alloc_walkers = [
            max(256, int(round(w * total_walkers_wave / 64.0)) * 64) for w in weights
        ]

        wave_improved = False
        for b_idx, nw in zip(b_ids, alloc_walkers):
            if (time.perf_counter() - t0) >= time_budget_s:
                break
            stalls = branch_meta[b_idx]["stalls"]
            # Aggressive triple-collision Split-and-Flip cadence to shatter Strassen shells!
            plus_headroom = 2 if stalls < 2 else (3 if stalls < 4 else 4)
            base_pp = 350 if ref_start_r >= 180 else 700
            plus_period = max(160, base_pp // (1 + min(stalls, 2)))

            res = engine.run_zt_batch(
                branch_pools[b_idx],
                n_walkers=nw,
                n_steps=steps_wave,
                plus_period=plus_period,
                plus_headroom=plus_headroom,
                top_k=16,
                seed=seed + wave * 4099 + b_idx * 131,
            )
            total_flips += res.executed_flips
            total_plus += res.executed_plus
            total_reduces += res.executed_reduces
            peak_vram = max(peak_vram, res.peak_vram_mib)

            cand = res.top_schemes[0]
            rep = _verify_and_save_if_record(cand, records_saved, tag=f"p2_zt_{fmt_key}_w{wave}_b{b_idx}")
            nnz = rep.additive_complexity["nnz_total"]
            sc_val, b_sc, p_mk = _score_zt(cand.rank, nnz, sota_zt, ref_start_r)

            prev_nid = branch_best_node[b_idx]
            prev_score = tree.nodes[prev_nid].score
            nid = f"b{b_idx}_w{wave}"
            tree.add(
                dream_tree.Node(
                    id=nid, parent_id=prev_nid, branch=b_idx, attempt=wave,
                    score=sc_val, fail_class="ok", valid=True,
                    delta_vs_parent=sc_val - prev_score, delta_vs_baseline=sc_val - base_score,
                    pass_mask=p_mk, bucket_scores=b_sc,
                    meta={
                        "rank": cand.rank, "nnz": nnz, "plus_headroom": plus_headroom,
                        "plus_period": plus_period, "n_walkers": nw, "stage": "gpu_zt_v2_wave",
                    },
                )
            )

            verified_top = []
            for sc_i in res.top_schemes[:12]:
                r_i = bv.verify_scheme(sc_i)
                if r_i.valid and r_i.is_ternary:
                    verified_top.append(sc_i)
                    h_i = r_i.canonical_sha256
                    if h_i not in all_verified_elites:
                        all_verified_elites[h_i] = (sc_i.rank, r_i.additive_complexity["nnz_total"], sc_i)

            if (cand.rank, nnz) < (branch_meta[b_idx]["best_r"], branch_meta[b_idx]["best_nnz"]):
                branch_meta[b_idx]["best_r"] = cand.rank
                branch_meta[b_idx]["best_nnz"] = nnz
                branch_meta[b_idx]["stalls"] = 0
                branch_best_node[b_idx] = nid
            else:
                branch_meta[b_idx]["stalls"] += 1

            if verified_top:
                prev_anchor = branch_pools[b_idx][0]
                branch_pools[b_idx] = verified_top[:10] + ([prev_anchor] if prev_anchor.rank <= verified_top[0].rank else [])

            if (cand.rank, nnz) < (best_r, best_nnz):
                if cand.rank < best_r:
                    wave_improved = True
                    print(
                        f"  [Z_T {fmt_key} | wave {wave} | b{b_idx}] RANK DROP: {best_r} -> {cand.rank} "
                        f"(nnz={nnz}, adds={rep.additive_complexity['naive_additions_total']}, "
                        f"SOTA={sota_zt}, t={time.perf_counter()-t0:.1f}s)",
                        flush=True,
                    )
                best_r, best_nnz, best_scheme = cand.rank, nnz, cand

        # BackwardSubGoalDecomposer every 4 waves if beat_sota bucket is still unsolved
        if wave % 4 == 0 and best_r >= sota_zt and (time.perf_counter() - t0) < (time_budget_s - 6.0):
            decomp = decomposer.decompose([tree], problem)
            if decomp.get("dead_buckets"):
                decomposer_triggers += 1
                worst_b = max(b_ids, key=lambda b: (branch_meta[b]["best_r"], branch_meta[b]["stalls"]))
                fresh_seeds = build_exact_sota_seeds(fmt_key, lib, rng)
                _, fresh_sc = rng.choice(fresh_seeds)
                branch_pools[worst_b] = [best_scheme, fresh_sc]
                branch_meta[worst_b]["stalls"] = 0

        if wave == 1 or wave_improved or wave % 4 == 0:
            dt_now = time.perf_counter() - t0
            print(
                f"  [Z_T {fmt_key} | wave {wave}] best_R={best_r} (nnz={best_nnz}, SOTA={sota_zt}) | "
                f"elites={len(all_verified_elites)} | {total_flips/max(dt_now,1e-9)/1e6:.1f}M flips/s | "
                f"plus={total_plus:,} red={total_reduces:,} | VRAM={peak_vram:.1f}MiB | t={dt_now:.1f}/{time_budget_s:.0f}s",
                flush=True,
            )

    # Final AutoDreamRunner + BlindVaultGuard(max_calls=1) sealed certification cycle
    top_elites_tuples = sorted(all_verified_elites.values(), key=lambda x: (x[0], x[1]))[:4]
    vault_scheme_ref: List[Optional[bv.BilinearScheme]] = [None]

    def _make_fn(sc_obj: bv.BilinearScheme):
        return lambda: sc_obj

    def _verify_and_score_c(fn):
        sc_obj = fn()
        rep_c = bv.verify_scheme(sc_obj)
        if not (rep_c.valid and rep_c.is_ternary):
            return False, -1e6, "algorithmic_failure", {"valid": False}
        nnz_c = rep_c.additive_complexity["nnz_total"]
        s_c, b_c, pm_c = _score_zt(sc_obj.rank, nnz_c, sota_zt, ref_start_r)
        return True, s_c, "ok", {
            "rank": sc_obj.rank,
            "nnz": nnz_c,
            "naive_additions": rep_c.additive_complexity["naive_additions_total"],
            "bucket_scores": b_c,
            "pass_mask": pm_c,
            "canonical_sha256": rep_c.canonical_sha256,
        }

    def _xover_synth(u_c: dream_cartridge.BranchCandidate, v_c: dream_cartridge.BranchCandidate):
        res_x = engine.run_zt_batch(
            [u_c.fn(), v_c.fn()], n_walkers=1024, n_steps=10000, plus_period=300, plus_headroom=3,
            top_k=4, seed=seed + 99991,
        )
        cand_x = res_x.top_schemes[0]
        _verify_and_save_if_record(cand_x, records_saved, tag=f"p2_autodream_xover_{fmt_key}")
        return dream_cartridge.BranchCandidate(
            id=f"xover_{u_c.id}_{v_c.id}", branch=99,
            description=f"GPU v2 crossover of {u_c.id} and {v_c.id}",
            fn=_make_fn(cand_x), parent_id=u_c.id,
        )

    def _vault_cert(fn):
        sc_obj = fn()
        vault_scheme_ref[0] = sc_obj
        rep_v = bv.verify_scheme(sc_obj)
        return {
            "valid": rep_v.valid,
            "brent_ok": rep_v.brent_ok,
            "basis_ok": rep_v.basis_ok,
            "is_ternary": rep_v.is_ternary,
            "rank": rep_v.rank,
            "nnz": rep_v.additive_complexity["nnz_total"],
            "naive_additions": rep_v.additive_complexity["naive_additions_total"],
            "canonical_sha256": rep_v.canonical_sha256,
        }

    init_branches = [
        dream_cartridge.BranchCandidate(
            id=f"elite_{idx}_R{rk}", branch=idx,
            description=f"GPU v2 Z_T elite {fmt_key} R={rk} nnz={nz}",
            fn=_make_fn(sc_e),
        )
        for idx, (rk, nz, sc_e) in enumerate(top_elites_tuples)
    ]
    cartridge = dream_cartridge.TaskCartridge(
        task_name=f"fmm_p2_zt_{fmt_key}",
        description=f"Phase 2 From-Scratch Z_T Assault <{m},{n},{p}> (SOTA={sota_zt})",
        baseline_fn=_make_fn(named_seeds[0][1]),
        verify_and_score=_verify_and_score_c,
        initial_branches=init_branches,
        crossover_synthesizer=_xover_synth,
        sentinel_noise_regime="deterministic",
        blind_vault_evaluator=_vault_cert,
    )
    ad_report = dream_cartridge.AutoDreamRunner(cartridge).run()
    vault = ad_report.get("blind_vault_report") or {}
    if vault.get("valid") and vault_scheme_ref[0] is not None:
        v_r, v_nnz = int(vault.get("rank", 9999)), int(vault.get("nnz", 10**9))
        if (v_r, v_nnz) < (best_r, best_nnz):
            best_r, best_nnz, best_scheme = v_r, v_nnz, vault_scheme_ref[0]

    skill_state.commit_cycle(tree, problem)
    champ_b = min(b_ids, key=lambda b: (branch_meta[b]["best_r"], branch_meta[b]["best_nnz"]))
    champ_node = tree.nodes[branch_best_node[champ_b]]
    sentinel_ok = sentinel.evaluate_gate(
        tree.nodes[tree.root_id], champ_node, base_buckets,
        lambda idxs: [champ_node.bucket_scores[i] for i in idxs],
    )

    dt_total = time.perf_counter() - t0
    final_rep = bv.verify_scheme(best_scheme)
    final_rank = final_rep.rank
    final_nnz = final_rep.additive_complexity["nnz_total"]

    report = {
        "format": fmt_key,
        "shape": [m, n, p],
        "sota_zt": sota_zt,
        "sota_q": ref["q"],
        "start_rank": ref_start_r,
        "best_rank": final_rank,
        "best_nnz": final_nnz,
        "best_naive_additions": final_rep.additive_complexity["naive_additions_total"],
        "best_sha256": final_rep.canonical_sha256,
        "beat_sota_zt": final_rank < sota_zt,
        "beat_sota_q": final_rank < ref["q"],
        "waves": wave,
        "distinct_elites": len(all_verified_elites),
        "decomposer_triggers": decomposer_triggers,
        "elapsed_s": round(dt_total, 2),
        "executed_flips": total_flips,
        "executed_plus": total_plus,
        "executed_reduces": total_reduces,
        "flips_per_s": round(total_flips / max(dt_total, 1e-9), 1),
        "peak_vram_mib": round(peak_vram, 2),
        "autodream_champion_score": round(float(ad_report["champion_score"]), 4),
        "blind_vault_verified": bool(vault.get("valid", False)),
        "blind_vault_rank": vault.get("rank"),
        "sentinel_gate_passed": bool(sentinel_ok.get("passed", True)),
        "tree_node_count": len(tree.nodes),
    }
    elites_out = [sc for _, _, sc in sorted(all_verified_elites.values(), key=lambda x: (x[0], x[1]))[:12]]
    return report, elites_out


def run_f2_km_sniper_shot(
    engine: fg.FlipGraphGPUEngine,
    fmt_key: str,
    lib: Dict[str, List[bv.BilinearScheme]],
    time_budget_s: float,
    records_saved: List[Dict[str, Any]],
    seed: int = 2027,
) -> Dict[str, Any]:
    """
    Focused cold-start F2 KM-sniper shot on 4x5x5 / 4x5x6 starting from naive + fluid direct sums,
    coupled with instant Bool-PB exact Z_T sign-lifting whenever any F2 scheme beats the Z_T SOTA!
    """
    t0 = time.perf_counter()
    rng = random.Random(seed)
    ref = SOTA_2026[fmt_key]
    m, n, p = ref["m"], ref["n"], ref["p"]

    zt_seeds = build_exact_sota_seeds(fmt_key, lib, rng)
    streams = [
        ("naive", [bv.make_naive_scheme(m, n, p, "Z2")], time_budget_s * 0.55),
        ("structured", [sc.reduce_mod(2) for lbl, sc in zt_seeds if "naive" not in lbl], time_budget_s * 0.45),
    ]
    best_r = m * n * p
    best_sc = bv.make_naive_scheme(m, n, p, "Z2")
    total_flips = 0
    peak_vram = 0.0
    waves = 0
    lift_attempts = 0
    lift_successes = 0
    lifted_hashes: set[str] = set()
    for st_name, init_s, st_budget in streams:
        ts = time.perf_counter()
        cur = list(init_s)
        while (time.perf_counter() - ts) < st_budget:
            waves += 1
            res = engine.run_f2_batch(
                cur, n_walkers=4096, n_steps=60000, plus_period=1200, plus_headroom=3,
                top_k=28, seed=seed + waves * 1777,
            )
            total_flips += res.executed_flips
            peak_vram = max(peak_vram, res.peak_vram_mib)
            cand = res.top_schemes[0]
            _verify_and_save_if_record(cand, records_saved, tag=f"p2_f2_{st_name}_{fmt_key}_w{waves}")
            if cand.rank < best_r:
                best_r, best_sc = cand.rank, cand
            # Instant Bool-PB Z_T sign-lifting attempt whenever ANY F2 scheme beats Z_T SOTA!
            for sc_f2 in res.top_schemes[:8]:
                if sc_f2.rank < ref["zt"]:
                    h_f2 = sc_f2.canonical_sha256()
                    if h_f2 not in lifted_hashes:
                        lifted_hashes.add(h_f2)
                        lift_attempts += 1
                        zt_cand = try_sign_lift_zt_bool(sc_f2, timeout_ms=500)
                        if zt_cand is not None:
                            lift_successes += 1
                            _verify_and_save_if_record(zt_cand, records_saved, tag=f"p2_f2_lifted_zt_{fmt_key}_w{waves}")
            cur = res.top_schemes
    rep = bv.verify_scheme(best_sc)
    dt = time.perf_counter() - t0
    return {
        "format": fmt_key,
        "ring": "Z2",
        "sota_z2": ref["z2"],
        "best_rank": rep.rank,
        "best_nnz": rep.additive_complexity["nnz_total"],
        "beat_sota_z2": rep.rank < ref["z2"],
        "zt_lift_attempts_below_zt_sota": lift_attempts,
        "zt_lift_successes": lift_successes,
        "waves": waves,
        "elapsed_s": round(dt, 2),
        "executed_flips": total_flips,
        "flips_per_s": round(total_flips / max(dt, 1e-9), 1),
        "peak_vram_mib": round(peak_vram, 2),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 RTX 5060 From-Scratch Z_T World-Record Assault.")
    parser.add_argument("--budget", type=float, default=850.0, help="Assault wall-clock budget (s) after Stage 0.")
    parser.add_argument("--seed", type=int, default=2027, help="Master PRNG seed.")
    args = parser.parse_args(argv)

    t_start = time.perf_counter()
    engine = fg.FlipGraphGPUEngine()
    os.makedirs(_ARTIFACTS_DIR, exist_ok=True)
    records_saved: List[Dict[str, Any]] = []

    # 1. Stage 0: Strict min-rank primitive library (ZERO external schemes!)
    lib = build_strict_min_rank_library(engine, seed=args.seed)

    # 2. Mandatory High-R Throughput Benchmark at EXACT R=230 (5x8x8) and R=250 (7x7x7)
    rng_b = random.Random(args.seed + 1)
    s588_230 = build_exact_sota_seeds("5x8x8", lib, rng_b)[0][1]
    s777_250 = build_exact_sota_seeds("7x7x7", lib, rng_b)[0][1]
    assert s588_230.rank == 230 and bv.verify_scheme(s588_230).valid
    assert s777_250.rank == 250 and bv.verify_scheme(s777_250).valid

    b_230 = engine.run_zt_batch([s588_230], n_walkers=2048, n_steps=20000, plus_period=400, plus_headroom=2, top_k=4, seed=args.seed + 2)
    b_250 = engine.run_zt_batch([s777_250], n_walkers=2048, n_steps=20000, plus_period=400, plus_headroom=2, top_k=4, seed=args.seed + 3)
    assert bv.verify_scheme(b_230.top_schemes[0]).valid and bv.verify_scheme(b_250.top_schemes[0]).valid
    _verify_and_save_if_record(b_230.top_schemes[0], records_saved, tag="bench_5x8x8")
    _verify_and_save_if_record(b_250.top_schemes[0], records_saved, tag="bench_7x7x7")

    high_r_bench = {
        "5x8x8_R230": {
            "start_rank": 230, "best_rank": b_230.best_rank,
            "flips_per_s": round(b_230.executed_flips_per_s, 1),
            "flips_per_s_M": round(b_230.executed_flips_per_s / 1e6, 2),
            "phase1_baseline_M": 1.28,
            "speedup_vs_phase1": round((b_230.executed_flips_per_s / 1e6) / 1.28, 2),
            "peak_vram_mib": round(b_230.peak_vram_mib, 2),
        },
        "7x7x7_R250": {
            "start_rank": 250, "best_rank": b_250.best_rank,
            "flips_per_s": round(b_250.executed_flips_per_s, 1),
            "flips_per_s_M": round(b_250.executed_flips_per_s / 1e6, 2),
            "phase1_baseline_M": 1.04,
            "speedup_vs_phase1": round((b_250.executed_flips_per_s / 1e6) / 1.04, 2),
            "peak_vram_mib": round(b_250.peak_vram_mib, 2),
        },
    }
    print(
        f"[High-R Benchmark] 5x8x8 R=230: {high_r_bench['5x8x8_R230']['flips_per_s_M']} M flips/s "
        f"({high_r_bench['5x8x8_R230']['speedup_vs_phase1']}x vs Phase 1!) | "
        f"7x7x7 R=250: {high_r_bench['7x7x7_R250']['flips_per_s_M']} M flips/s "
        f"({high_r_bench['7x7x7_R250']['speedup_vs_phase1']}x vs Phase 1!) | VRAM={b_250.peak_vram_mib:.1f} MiB",
        flush=True,
    )

    # 3. Main ~14-15 min SLEEP-DAG-RSI v8 Z_T Combat Campaign
    skill_state = dream_skill.MutableSkillState(task="fmm_phase2_zt_assault")
    zt_reports: Dict[str, Any] = {}
    f2_reports: Dict[str, Any] = {}

    # (a) 5x8x8 (starts AT EXACT 230 SOTA, +7.0 ridge!)
    rep_588, elites_588 = run_sleep_dag_zt_assault_target(
        engine, "5x8x8", lib, time_budget_s=args.budget * 0.22,
        skill_state=skill_state, records_saved=records_saved,
        assert_max_start_r=230, seed=args.seed + 100,
    )
    zt_reports["5x8x8"] = rep_588

    # (b) 7x7x7 (starts AT EXACT 250 Z_T SOTA!)
    rep_777, elites_777 = run_sleep_dag_zt_assault_target(
        engine, "7x7x7", lib, time_budget_s=args.budget * 0.22,
        skill_state=skill_state, records_saved=records_saved,
        assert_max_start_r=250, seed=args.seed + 200,
    )
    zt_reports["7x7x7"] = rep_777

    # (c) 5x7x8 (starts <= 206 + meta-flip projections from fused 5x8x8 elites!)
    meta_578 = generate_meta_flip_seeds(
        engine, elites_588, (5, 7, 8), axis="n", label_prefix="meta_from_5x8x8", seed=args.seed + 301
    )
    rep_578, _ = run_sleep_dag_zt_assault_target(
        engine, "5x7x8", lib, time_budget_s=args.budget * 0.19,
        skill_state=skill_state, records_saved=records_saved,
        extra_meta_seeds=meta_578, assert_max_start_r=206, seed=args.seed + 300,
    )
    zt_reports["5x7x8"] = rep_578

    # (d) 6x6x7 (starts <= 185 + meta-flip projections from fused 6x6x8:203 & 7x7x7!)
    meta_667 = generate_meta_flip_seeds(
        engine, [], (6, 6, 7), axis="p", label_prefix="meta_from_6x6x8", lib=lib, seed=args.seed + 401
    )
    # Also project fused 7x7x7 -> 6x7x7 -> 6x6x7
    if elites_777:
        p677 = fg.best_project_slice(elites_777[0], "m")
        p667 = fg.best_project_slice(p677, "n")
        if bv.verify_scheme(p667).valid:
            meta_667.append((f"meta_from_7x7x7_R{p667.rank}", p667))
            print(f"  [Meta-Flip Projection] 7x7x7 -> 6x6x7 seed R={p667.rank}", flush=True)
    rep_667, _ = run_sleep_dag_zt_assault_target(
        engine, "6x6x7", lib, time_budget_s=args.budget * 0.19,
        skill_state=skill_state, records_saved=records_saved,
        extra_meta_seeds=meta_667, assert_max_start_r=185, seed=args.seed + 400,
    )
    zt_reports["6x6x7"] = rep_667

    # (e) Focused ~2.5 min cold-start Z_T + F2->Bool-PB sign-lifting shot on 4x5x5 and 4x5x6
    rep_455, _ = run_sleep_dag_zt_assault_target(
        engine, "4x5x5", lib, time_budget_s=args.budget * 0.05,
        skill_state=skill_state, records_saved=records_saved, seed=args.seed + 500,
    )
    zt_reports["4x5x5"] = rep_455

    rep_456, _ = run_sleep_dag_zt_assault_target(
        engine, "4x5x6", lib, time_budget_s=args.budget * 0.05,
        skill_state=skill_state, records_saved=records_saved, seed=args.seed + 600,
    )
    zt_reports["4x5x6"] = rep_456

    f2_reports["4x5x5"] = run_f2_km_sniper_shot(
        engine, "4x5x5", lib, time_budget_s=args.budget * 0.04,
        records_saved=records_saved, seed=args.seed + 700,
    )
    f2_reports["4x5x6"] = run_f2_km_sniper_shot(
        engine, "4x5x6", lib, time_budget_s=args.budget * 0.04,
        records_saved=records_saved, seed=args.seed + 800,
    )

    total_wall = round(time.perf_counter() - t_start, 2)
    total_zt_flips = sum(r["executed_flips"] for r in zt_reports.values())
    total_f2_flips = sum(r["executed_flips"] for r in f2_reports.values())
    peak_vram_all = max(
        [r["peak_vram_mib"] for r in zt_reports.values()]
        + [r["peak_vram_mib"] for r in f2_reports.values()]
        + [high_r_bench["5x8x8_R230"]["peak_vram_mib"], high_r_bench["7x7x7_R250"]["peak_vram_mib"]]
    )

    summary = {
        "phase": "phase2_optionB_track1_cold_start_zt",
        "device": engine.backend.device_name,
        "arch": engine.backend.arch,
        "provenance": "100%_from_scratch_zero_external_seeds",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_wall_s": total_wall,
        "peak_vram_mib": round(peak_vram_all, 2),
        "total_zt_flips": total_zt_flips,
        "total_f2_flips": total_f2_flips,
        "block_library_ranks": {k: v[0].rank for k, v in lib.items()},
        "high_r_benchmark": high_r_bench,
        "zt_targets": zt_reports,
        "f2_sniper_shots": f2_reports,
        "skill_state_summary": {
            "round_index": skill_state.round_index,
            "invariants_count": len(skill_state.best_invariants),
            "unsolved_buckets_count": len(skill_state.unsolved_buckets),
            "serialized_bytes": skill_state.serialized_byte_size(),
        },
        "records_saved": records_saved,
    }
    out_path = os.path.join(_ARTIFACTS_DIR, "phase2_zt_summary.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"\n[Phase 2 Complete in {total_wall:.1f}s] Summary saved -> {out_path}", flush=True)
    print(f"  Total Z_T flips: {total_zt_flips:,} | Total F2 flips: {total_f2_flips:,} | Peak VRAM: {peak_vram_all:.1f} MiB", flush=True)
    for fk, r in zt_reports.items():
        print(
            f"  Z_T {fk}: start={r['start_rank']} -> best={r['best_rank']} (SOTA ZT={r['sota_zt']}, Q={r['sota_q']}, "
            f"beat_ZT={r['beat_sota_zt']}) | nnz={r['best_nnz']}, adds={r['best_naive_additions']}, "
            f"{r['flips_per_s']/1e6:.1f}M/s, vault={r['blind_vault_verified']}",
            flush=True,
        )
    for fk, r in f2_reports.items():
        print(
            f"  F2  {fk}: best={r['best_rank']} (SOTA Z2={r['sota_z2']}, beat_Z2={r['beat_sota_z2']}) | "
            f"nnz={r['best_nnz']}, {r['flips_per_s']/1e6:.1f}M/s",
            flush=True,
        )
    print(f"  NEW 2026 WORLD RECORDS SAVED: {len(records_saved)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

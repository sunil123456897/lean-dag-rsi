"""
Phase 1 RTX 5060 SM 12.0 SLEEP-DAG-RSI v8 FMM Discovery Campaign.

Runs strictly from scratch (naive schoolbook + Strassen 1969 + Laderman 1976 + GPU-evolved
sub-blocks; ZERO external scheme imports!) in two coordinated stages:

  Stage 1 (`--phase f2-pool`, ~2.5-3.5 min):
    Track A-Bridge: Generate hundreds of structurally diverse, brent_verifier-certified
    F2 schemes across 4x4x5 (R<=60), 4x5x5 (R<=75), 4x5x6 (R<=89), 4x4x8 (R<=95),
    and flush `fmm_discovery/artifacts/f2_pool_phase1.json` immediately so `@fmm-sota`
    can launch parallel CPU Mod-4/Hensel/Z3 lifting.

  Stage 2 (`--phase zt-assault`, ~7-8 min):
    Track B/A-Direct: Direct 2-bitplane Z_T ternary GPU assault orchestrated by genuine
    `SLEEP-DAG-RSI v8` (`DiscoveryTree`, `LeanDAGPolicy(entropic_tau=4.0)`,
    `compute_max_tilted_objective`, `BackwardSubGoalDecomposer`, `MutableSkillState`,
    `WaldMicroSentinel`, `TaskCartridge`, `AutoDreamRunner`, `BlindVaultGuard`)
    targeting 2026 open char-0 records:
      - 4x5x5 (SOTA ZT=76 -> target <= 75)
      - 4x5x6 (SOTA ZT=90 -> target <= 89)
      - 5x8x8 (SOTA ZT=230 -> target <= 229, +7.0 ridge!)
      - 6x6x7 (SOTA ZT=183 -> target <= 182, +5.0 ridge!)
      - 7x7x7 (SOTA ZT=250, Q=249 -> target <= 248; <= 249 beats ZT!)

Every reported/saved scheme is certified by `brent_verifier.verify_scheme()` (zero floats).
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

# Verified 2026 SOTA reference bars (from Phase 0 #60/#61)
SOTA_REF: Dict[str, Dict[str, int]] = {
    "4x4x5": {"m": 4, "n": 4, "p": 5, "z2": 60, "zt": 61, "q": 61},
    "4x5x5": {"m": 4, "n": 5, "p": 5, "z2": 73, "zt": 76, "q": 76},
    "4x5x6": {"m": 4, "n": 5, "p": 6, "z2": 89, "zt": 90, "q": 90},
    "4x4x8": {"m": 4, "n": 4, "p": 8, "z2": 94, "zt": 96, "q": 95},
    "5x8x8": {"m": 5, "n": 8, "p": 8, "z2": 230, "zt": 230, "q": 230},
    "6x6x7": {"m": 6, "n": 6, "p": 7, "z2": 183, "zt": 183, "q": 183},
    "7x7x7": {"m": 7, "n": 7, "p": 7, "z2": 245, "zt": 250, "q": 249},
}


def _fmt_key(m: int, n: int, p: int) -> str:
    return f"{m}x{n}x{p}"


def _verify_and_maybe_save_record(
    scheme: bv.BilinearScheme,
    records_saved: List[Dict[str, Any]],
    tag: str = "",
) -> bv.VerificationReport:
    """Strictly verify scheme via dual-path brent_verifier and auto-save if it beats 2026 SOTA."""
    rep = bv.verify_scheme(scheme)
    if not rep.valid:
        raise RuntimeError(f"VERIFICATION FAILURE on {scheme.name}: {rep.brent_details}")
    if scheme.ring == "Z" and not rep.is_ternary:
        raise RuntimeError(f"Non-ternary Z scheme returned from Z_T kernel: {scheme.name}")

    fk = _fmt_key(*scheme.shape)
    ref = SOTA_REF.get(fk)
    if ref is not None:
        is_record = False
        record_kind = ""
        if scheme.ring == "Z2" and scheme.rank < ref["z2"]:
            is_record = True
            record_kind = f"Z2_beats_{ref['z2']}"
        elif scheme.ring == "Z" and scheme.rank < ref["zt"]:
            is_record = True
            record_kind = (
                f"ZT_and_Q_beats_{ref['q']}" if scheme.rank < ref["q"] else f"ZT_beats_{ref['zt']}"
            )
        if is_record:
            os.makedirs(_ARTIFACTS_DIR, exist_ok=True)
            ring_tag = "ZT" if scheme.ring == "Z" else scheme.ring
            fname = f"record_{fk}_{ring_tag}_R{scheme.rank}_{rep.canonical_sha256[:8]}.json"
            fpath = os.path.join(_ARTIFACTS_DIR, fname)
            scheme.metadata = {
                **dict(scheme.metadata),
                "record_kind": record_kind,
                "sota_before": ref,
                "discovery_tag": tag,
                "verified_brent": True,
                "verified_basis": True,
                "additive_complexity": rep.additive_complexity,
                "discovered_at": datetime.now(timezone.utc).isoformat(),
            }
            scheme.save_json(fpath)
            rec_info = {
                "format": fk,
                "ring": ring_tag,
                "rank": scheme.rank,
                "record_kind": record_kind,
                "nnz": rep.additive_complexity["nnz_total"],
                "naive_additions": rep.additive_complexity["naive_additions_total"],
                "canonical_sha256": rep.canonical_sha256,
                "path": fpath,
            }
            records_saved.append(rec_info)
            print(
                f"  *** [NEW 2026 WORLD RECORD CERTIFIED!] {fk} {ring_tag} R={scheme.rank} "
                f"({record_kind}) nnz={rec_info['nnz']} adds={rec_info['naive_additions']} -> {fname} ***",
                flush=True,
            )
    return rep


# =============================================================================
# 1. Stage 0: Zero-External-Leak Primitive Block Library (Built From Scratch)
# =============================================================================


def build_from_scratch_block_library(
    engine: fg.FlipGraphGPUEngine, seed: int = 2026
) -> Dict[str, List[bv.BilinearScheme]]:
    """
    Build clean, diverse ternary Z_T primitive sub-schemes purely from schoolbook naive +
    Strassen <2,2,2:7> + Laderman <3,3,3:23> via short RTX 5060 GPU evolution.
    Returns dict mapping format key -> list of verified diverse Z_T schemes.
    """
    t0 = time.perf_counter()
    print("[Stage 0] Building from-scratch Z_T primitive block library on RTX 5060...", flush=True)
    lib: Dict[str, List[bv.BilinearScheme]] = {}

    s222 = bv.strassen_222("Z")
    s333 = bv.laderman_333("Z")
    s223 = bv.strassen_block_223("Z")
    s444 = fg.kronecker_product(s222, s222, name="strassen_444")
    assert bv.verify_scheme(s444).valid

    lib["2x2x2"] = [s222]
    lib["2x2x3"] = [s223]
    lib["3x3x3"] = [s333]
    lib["4x4x4"] = [s444]

    # Evolve diverse <3,3,3:23> variants from naive (adds diversity for 6x6x7!)
    r_333 = engine.evolve_zt(
        [bv.make_naive_scheme(3, 3, 3, "Z")], target_rank=23, n_rounds=4, n_walkers=4096,
        n_steps=20000, plus_period=600, top_k=16, seed=seed + 11,
    )
    for sc in r_333.top_schemes[:6]:
        if sc.rank == 23 and bv.verify_scheme(sc).valid:
            lib["3x3x3"].append(sc)

    # Evolve <2,3,3:15> from naive 18
    r_233 = engine.evolve_zt(
        [bv.make_naive_scheme(2, 3, 3, "Z")], target_rank=15, n_rounds=3, n_walkers=2048,
        n_steps=12000, plus_period=400, top_k=12, seed=seed + 22,
    )
    lib["2x3x3"] = [sc for sc in r_233.top_schemes[:8] if bv.verify_scheme(sc).valid]

    # Evolve <2,4,4:26> from pure naive 32 (high initial flip connectivity, avoids Strassen isolation!)
    r_244 = engine.evolve_zt(
        [bv.make_naive_scheme(2, 4, 4, "Z")], target_rank=26, n_rounds=6, n_walkers=4096,
        n_steps=35000, plus_period=2000, top_k=24, seed=seed + 33,
    )
    lib["2x4x4"] = [sc for sc in r_244.top_schemes[:12] if bv.verify_scheme(sc).valid]

    # Evolve <3,3,4:29> from pure naive 36
    r_334 = engine.evolve_zt(
        [bv.make_naive_scheme(3, 3, 4, "Z")], target_rank=29, n_rounds=6, n_walkers=4096,
        n_steps=40000, plus_period=2200, top_k=24, seed=seed + 44,
    )
    lib["3x3x4"] = [sc for sc in r_334.top_schemes[:12] if bv.verify_scheme(sc).valid]

    # Evolve <3,4,4:38> from BOTH pure naive 48 and fluid <3,3,4> + <3,1,4>
    base_334 = lib["3x3x4"][0]
    s344_init = fg.direct_sum_n(base_334, bv.make_naive_scheme(3, 1, 4, "Z"), name="3x4x4_init")
    r_344 = engine.evolve_zt(
        [bv.make_naive_scheme(3, 4, 4, "Z"), s344_init], target_rank=38, n_rounds=6, n_walkers=4096,
        n_steps=35000, plus_period=2000, top_k=24, seed=seed + 55,
    )
    lib["3x4x4"] = [sc for sc in r_344.top_schemes[:12] if bv.verify_scheme(sc).valid]

    # Evolve <4,5,2:33> from pure naive 40 and <4,5,3:46> from pure naive 60
    r_452 = engine.evolve_zt(
        [bv.make_naive_scheme(4, 5, 2, "Z")], target_rank=33, n_rounds=5, n_walkers=4096,
        n_steps=35000, plus_period=2000, top_k=16, seed=seed + 66,
    )
    lib["4x5x2"] = [sc for sc in r_452.top_schemes[:10] if bv.verify_scheme(sc).valid]

    r_453 = engine.evolve_zt(
        [bv.make_naive_scheme(4, 5, 3, "Z")], target_rank=46, n_rounds=6, n_walkers=4096,
        n_steps=35000, plus_period=2000, top_k=16, seed=seed + 77,
    )
    lib["4x5x3"] = [sc for sc in r_453.top_schemes[:10] if bv.verify_scheme(sc).valid]

    dt = time.perf_counter() - t0
    summary = {k: (min(s.rank for s in v), len(v)) for k, v in lib.items()}
    print(f"[Stage 0] Done in {dt:.1f}s | block library (best_R, count): {summary}", flush=True)
    return lib


def _pick_sub(
    lib: Dict[str, List[bv.BilinearScheme]],
    shape: Tuple[int, int, int],
    rng: random.Random,
    randomize: bool = True,
) -> bv.BilinearScheme:
    """Retrieve a verified Z_T sub-scheme of exact shape via S_3 permutation from `lib`."""
    m, n, p = shape
    canon = tuple(sorted(shape))
    key = _fmt_key(*canon)
    if key in lib and lib[key]:
        base = rng.choice(lib[key])
        sc = fg.permute_axes(base, shape)
    else:
        sc = bv.make_naive_scheme(m, n, p, "Z")
    return fg.random_monomial_symmetry(sc, rng) if randomize else sc


def build_target_zt_seeds(
    fmt_key: str, lib: Dict[str, List[bv.BilinearScheme]], rng: random.Random
) -> List[Tuple[str, bv.BilinearScheme]]:
    """
    Construct multiple structurally orthogonal from-scratch Z_T starting seeds for each target:
      - Padded-Strassen compositions (exact 2026 composite record for 5x8x8=230, 7x7x7=250!)
      - Direct-sum sub-tensor gluings along different axes
      - S_3 symmetry-permuted padded-Strassen (shifts which axis carries the odd dimension!)
    """
    seeds: List[Tuple[str, bv.BilinearScheme]] = []

    if fmt_key == "4x4x5":
        # (a) <4,4,4:49> + <4,4,1:16> = 65
        s_a = bv.direct_sum_p(lib["4x4x4"][0], bv.make_naive_scheme(4, 4, 1, "Z"), name="4x4x5_strassen49_plus_16")
        seeds.append(("strassen49_plus_naive16", s_a))
        # (b) <4,4,3> + <4,4,2> via S_3 from <3,4,4:38> and <2,4,4:26> => 38 + 26 = 64!
        s_b = bv.direct_sum_p(
            _pick_sub(lib, (4, 4, 3), rng, False),
            _pick_sub(lib, (4, 4, 2), rng, False),
            name="4x4x5_38plus26",
        )
        seeds.append(("direct_sum_38_26", s_b))
        # (c) Padded Strassen (2+2, 2+2, 3+2): 4*<2,2,3:11> + 3*<2,2,2:7> = 65
        subs = [_pick_sub(lib, sh, rng) for sh in [(2,2,3),(2,2,3),(2,2,2),(2,2,3),(2,2,2),(2,2,2),(2,2,3)]]
        seeds.append(("padded_strassen_22_32", fg.padded_strassen_compose(2, 2, 2, 2, 3, 2, subs, name="4x4x5_ps")))

    elif fmt_key == "4x5x5":
        # (a) Direct sum <4,5,2> + <4,5,3> (~33 + 46 = 79!)
        s_a = bv.direct_sum_p(lib["4x5x2"][0], lib["4x5x3"][0], name="4x5x5_ds_p_2_3")
        seeds.append(("direct_sum_452_453", s_a))
        # (b) Padded Strassen (2+2, 3+2, 3+2): M1..M7 = 15+15+11+11+7+11+11 = 81
        subs_b = [_pick_sub(lib, sh, rng) for sh in [(2,3,3),(2,3,3),(2,3,2),(2,2,3),(2,2,2),(2,3,2),(2,2,3)]]
        seeds.append(("padded_strassen_22_32_32", fg.padded_strassen_compose(2, 2, 3, 2, 3, 2, subs_b, name="4x5x5_ps1")))
        # (c) Randomized second padded Strassen with different sub-block symmetries
        subs_c = [_pick_sub(lib, sh, rng) for sh in [(2,3,3),(2,3,3),(2,3,2),(2,2,3),(2,2,2),(2,3,2),(2,2,3)]]
        seeds.append(("padded_strassen_sym2", fg.padded_strassen_compose(2, 2, 3, 2, 3, 2, subs_c, name="4x5x5_ps2")))
        # (d) Direct sum along m: <2,5,5> + <2,5,5> built from <2,3,3> & <2,2,3>
        s252 = fg.direct_sum_n(_pick_sub(lib, (2, 3, 2), rng), _pick_sub(lib, (2, 2, 2), rng))
        s253 = fg.direct_sum_n(_pick_sub(lib, (2, 3, 3), rng), _pick_sub(lib, (2, 2, 3), rng))
        s255 = bv.direct_sum_p(s252, s253)
        seeds.append(("direct_sum_m_255_255", fg.direct_sum_m(s255, fg.random_monomial_symmetry(s255, rng), name="4x5x5_dsm")))

    elif fmt_key == "4x5x6":
        # (a) Direct sum <4,5,3> + <4,5,3> (~46 + 46 = 92!)
        s_a = bv.direct_sum_p(lib["4x5x3"][0], fg.random_monomial_symmetry(lib["4x5x3"][-1], rng), name="4x5x6_ds_453x2")
        seeds.append(("direct_sum_453_453", s_a))
        # (b) Padded Strassen (2+2, 3+2, 3+3): 4*<2,3,3:15> + 3*<2,2,3:11> = 93
        subs_b = [_pick_sub(lib, sh, rng) for sh in [(2,3,3),(2,3,3),(2,3,3),(2,2,3),(2,2,3),(2,3,3),(2,2,3)]]
        seeds.append(("padded_strassen_22_32_33", fg.padded_strassen_compose(2, 2, 3, 2, 3, 3, subs_b, name="4x5x6_ps1")))
        # (c) Padded Strassen along S_3 permuted (4, 6, 5) -> permute back to (4, 5, 6)!
        subs_c = [_pick_sub(lib, sh, rng) for sh in [(2,3,3),(2,3,3),(2,3,2),(2,3,3),(2,3,2),(2,3,2),(2,3,3)]]
        ps_465 = fg.padded_strassen_compose(2, 2, 3, 3, 3, 2, subs_c, name="4x6x5_ps")
        seeds.append(("padded_strassen_s3_465", fg.permute_axes(ps_465, (4, 5, 6), name="4x5x6_ps_s3")))

    elif fmt_key == "4x4x8":
        # (a) Direct sum <4,4,4:49> + <4,4,4:49> = 98
        s_a = bv.direct_sum_p(lib["4x4x4"][0], fg.random_monomial_symmetry(lib["4x4x4"][0], rng), name="4x4x8_49x2")
        seeds.append(("direct_sum_49_49", s_a))
        # (b) Padded Strassen (2+2, 2+2, 4+4) = 7 * <2,2,4:14> = 98
        s224 = bv.direct_sum_p(lib["2x2x2"][0], lib["2x2x2"][0])
        subs_b = [fg.random_monomial_symmetry(s224, rng) for _ in range(7)]
        seeds.append(("strassen_7x_224", fg.padded_strassen_compose(2, 2, 2, 2, 4, 4, subs_b, name="4x4x8_ps")))

    elif fmt_key == "5x8x8":
        # (a) Exact 2026 composite record: padded Strassen (2+3, 4+4, 4+4) => 3*26 + 4*38 = 230!
        for k_v in range(3):
            subs = [_pick_sub(lib, sh, rng) for sh in [(3,4,4),(3,4,4),(2,4,4),(3,4,4),(2,4,4),(3,4,4),(2,4,4)]]
            seeds.append((f"padded_strassen_230_v{k_v}", fg.padded_strassen_compose(2, 3, 4, 4, 4, 4, subs, name=f"5x8x8_ps_v{k_v}")))
        # (b) S_3 permuted padded Strassen (8, 8, 5) with (4+4, 4+4, 2+3) -> permute back to (5, 8, 8) (also R=230, distinct boundary!)
        subs_p = [_pick_sub(lib, sh, rng) for sh in [(4,4,3),(4,4,2),(4,4,3),(4,4,2),(4,4,3),(4,4,3),(4,4,2)]]
        ps_885 = fg.padded_strassen_compose(4, 4, 4, 4, 2, 3, subs_p, name="8x8x5_ps")
        seeds.append(("padded_strassen_s3_885", fg.permute_axes(ps_885, (5, 8, 8), name="5x8x8_ps_s3")))

    elif fmt_key == "6x6x7":
        # (a) Padded Strassen (3+3, 3+3, 4+3) => 4*<3,3,4:29> + 3*<3,3,3:23> = 185!
        for k_v in range(3):
            subs = [_pick_sub(lib, sh, rng) for sh in [(3,3,4),(3,3,4),(3,3,3),(3,3,4),(3,3,3),(3,3,3),(3,3,4)]]
            seeds.append((f"padded_strassen_185_v{k_v}", fg.padded_strassen_compose(3, 3, 3, 3, 4, 3, subs, name=f"6x6x7_ps_v{k_v}")))
        # (b) S_3 permuted (7, 6, 6) with (4+3, 3+3, 3+3) -> distinct cross-block boundary!
        subs_m = [_pick_sub(lib, sh, rng) for sh in [(4,3,3),(3,3,3),(4,3,3),(3,3,3),(4,3,3),(3,3,3),(4,3,3)]]
        ps_766 = fg.padded_strassen_compose(4, 3, 3, 3, 3, 3, subs_m, name="7x6x6_ps")
        seeds.append(("padded_strassen_s3_766", fg.permute_axes(ps_766, (6, 6, 7), name="6x6x7_ps_s3")))

    elif fmt_key == "7x7x7":
        # Exact 2026 Z_T composite record: padded Strassen (4+3, 4+3, 4+3) => 49 + 3*38 + 3*29 = 250!
        for k_v in range(4):
            subs = [_pick_sub(lib, sh, rng) for sh in [(4,4,4),(3,4,4),(4,4,3),(3,3,4),(4,3,3),(3,4,3),(4,3,4)]]
            seeds.append((f"padded_strassen_250_v{k_v}", fg.padded_strassen_compose(4, 3, 4, 3, 4, 3, subs, name=f"7x7x7_ps_v{k_v}")))

    else:
        raise ValueError(f"Unknown target format {fmt_key}")

    # Strictly verify every constructed seed before GPU launch
    for lbl, sc in seeds:
        rep = bv.verify_scheme(sc)
        if not (rep.valid and rep.is_ternary):
            raise RuntimeError(f"Seed {fmt_key}/{lbl} failed brent_verifier!")
    return seeds


# =============================================================================
# 2. Stage 1: Track A-Bridge (Massive Diverse F2 Pool Generator)
# =============================================================================


def run_f2_pool_stage(
    engine: fg.FlipGraphGPUEngine,
    lib: Dict[str, List[bv.BilinearScheme]],
    time_budget_s: float = 170.0,
    seed: int = 2026,
) -> Dict[str, Any]:
    """
    Generate hundreds of structurally diverse, verified F2 schemes for:
      - 4x4x5 (target R <= 60)
      - 4x5x5 (target R in {73, 74, 75})
      - 4x5x6 (target R <= 89)
      - 4x4x8 (target R in {94, 95})
    Save immediately to `fmm_discovery/artifacts/f2_pool_phase1.json`.
    """
    t_stage = time.perf_counter()
    rng = random.Random(seed + 1000)
    records_saved: List[Dict[str, Any]] = []

    # Per-format config: (key, collect_max_rank, stretch_rank, budget_frac, max_pool_size)
    targets = [
        ("4x4x5", 60, 60, 0.22, 160),
        ("4x5x5", 75, 73, 0.30, 180),
        ("4x5x6", 89, 89, 0.26, 160),
        ("4x4x8", 95, 94, 0.22, 160),
    ]

    pool_payload: Dict[str, Any] = {
        "format_version": "f2_pool_v1",
        "convention": bv.CONVENTION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "packing_spec": (
            "Each u[r], v[r], w[r] is an exact integer bitmask (uint64) in C_eq_AB_row_major order: "
            "bit (i*cols + j) is entry (i,j) mod 2. Unpack via flip_graph_gpu.f2_packed_to_scheme(m, n, p, u, v, w)."
        ),
        "formats": {},
    }
    stage_stats: Dict[str, Any] = {}

    print(f"\n[Stage 1 | Track A-Bridge] Generating diverse F2 pool (budget ~{time_budget_s:.0f}s)...", flush=True)
    # First evolve a fluid non-Strassen F2 <4,4,4:<=49> from pure naive 64 (~10s) for 4x4x5 & 4x4x8!
    fluid_444_f2 = engine.evolve_f2(
        [bv.make_naive_scheme(4, 4, 4, "Z2")],
        target_rank=48,
        n_rounds=6,
        n_walkers=4096,
        n_steps=60000,
        plus_period=3500,
        plus_headroom=2,
        top_k=24,
        seed=seed + 7777,
    )
    fluid_444_list = [sc for sc in fluid_444_f2.top_schemes[:8] if bv.verify_scheme(sc).valid]
    print(
        f"  [F2 Fluid Bridge] naive 4x4x4:64 -> best_R={fluid_444_f2.best_rank} "
        f"(nnz={fluid_444_f2.best_nnz}, {len(fluid_444_list)} verified non-Strassen seeds)",
        flush=True,
    )

    for fk, collect_max_r, stretch_r, frac, max_pool in targets:
        t_f0 = time.perf_counter()
        f_budget = time_budget_s * frac
        ref = SOTA_REF[fk]
        m, n, p = ref["m"], ref["n"], ref["p"]

        # Run TWO uncontaminated streams so lower-rank frozen direct-sums never kill fluid descents:
        #   Stream A (50% budget): pure naive m*n*p (or fluid non-Strassen 4x4x4 bridge for 4x4x5 & 4x4x8!)
        #   Stream B (50% budget): fluid Stage-0 direct-sum compositions reduced mod 2
        zt_seeds = build_target_zt_seeds(fk, lib, rng)
        ds_seeds = [sc.reduce_mod(2) for lbl, sc in zt_seeds if "direct_sum" in lbl or "padded" in lbl]
        if fk == "4x4x5" and fluid_444_list:
            for sc444 in fluid_444_list[:4]:
                ds_seeds.insert(0, bv.direct_sum_p(sc444, bv.make_naive_scheme(4, 4, 1, "Z2")))
        elif fk == "4x4x8" and fluid_444_list:
            for sc444 in fluid_444_list[:4]:
                ds_seeds.insert(0, bv.direct_sum_p(sc444, fg.random_monomial_symmetry(sc444, rng)))
        elif fk == "4x5x5" and fluid_444_list:
            # Fluid 4x4x4(<=49) + 4x4x1(16) + 4x1x5(20) = <=85 fluid non-Strassen seed!
            sc444 = fluid_444_list[0]
            s445_fl = bv.direct_sum_p(sc444, bv.make_naive_scheme(4, 4, 1, "Z2"))
            ds_seeds.insert(0, fg.direct_sum_n(s445_fl, bv.make_naive_scheme(4, 1, 5, "Z2")))
        if not ds_seeds:
            ds_seeds = [zt_seeds[0][1].reduce_mod(2)]
        streams: List[Tuple[str, List[bv.BilinearScheme], float]] = [
            ("fluid_bridge", ds_seeds[:6], f_budget * 0.55),
            ("naive", [bv.make_naive_scheme(m, n, p, "Z2")], f_budget * 0.45),
        ]

        pool_by_hash: Dict[str, Tuple[int, int, bv.BilinearScheme]] = {}
        best_r = min(m * n * p, min(s.rank for s in ds_seeds))
        total_flips = 0
        total_plus = 0
        total_reduces = 0
        peak_vram = 0.0
        wave = 0
        best_overall_seen: List[bv.BilinearScheme] = []

        for st_name, st_init, st_budget in streams:
            t_st0 = time.perf_counter()
            cur_seeds = list(st_init)
            st_best_r = min(s.rank for s in cur_seeds)
            while (time.perf_counter() - t_st0) < st_budget:
                wave += 1
                diverse_now = st_best_r <= collect_max_r
                n_walkers = 4096 if st_best_r <= 115 else 2048
                n_steps = 65000 if st_best_r <= 85 else 50000
                plus_p = 1800 if diverse_now else 4500
                res = engine.run_f2_batch(
                    cur_seeds,
                    n_walkers=n_walkers,
                    n_steps=n_steps,
                    plus_period=plus_p,
                    plus_headroom=2 if not diverse_now else 3,
                    target_rank=collect_max_r,
                    diverse_mode=diverse_now,
                    top_k=48 if diverse_now else 32,
                    seed=seed + wave * 1337 + m * 100 + p,
                )
                total_flips += res.executed_flips
                total_plus += res.executed_plus
                total_reduces += res.executed_reduces
                peak_vram = max(peak_vram, res.peak_vram_mib)

                if res.best_rank < st_best_r:
                    st_best_r = res.best_rank
                if res.best_rank < best_r:
                    best_r = res.best_rank

                best_overall_seen.extend(res.top_schemes[:8])
                for sc in res.top_schemes:
                    if sc.rank <= collect_max_r + 4:
                        h = sc.canonical_sha256()
                        if h not in pool_by_hash:
                            rep = _verify_and_maybe_save_record(sc, records_saved, tag=f"f2_{st_name}_{fk}_w{wave}")
                            pool_by_hash[h] = (sc.rank, rep.additive_complexity["nnz_total"], sc)

                # Anti-monoculture re-seeding within this stream
                next_seeds: List[bv.BilinearScheme] = list(res.top_schemes[:28])
                if pool_by_hash:
                    near_keys = [k for k, v in pool_by_hash.items() if v[0] <= st_best_r + 1]
                    if near_keys:
                        for sk in rng.sample(near_keys, min(8, len(near_keys))):
                            next_seeds.append(pool_by_hash[sk][2])
                cur_seeds = next_seeds

                n_at_target = sum(1 for v in pool_by_hash.values() if v[0] <= collect_max_r)
                if n_at_target >= max_pool and best_r <= stretch_r:
                    break

        # If pool fell slightly short of collect_max_r + 4, still save best verified schemes found
        if not pool_by_hash:
            best_overall_seen.sort(key=lambda s: s.rank)
            for sc in best_overall_seen[:32]:
                h = sc.canonical_sha256()
                if h not in pool_by_hash:
                    rep = _verify_and_maybe_save_record(sc, records_saved, tag=f"f2_best_{fk}")
                    pool_by_hash[h] = (sc.rank, rep.additive_complexity["nnz_total"], sc)

        # Sort pool by (rank, nnz) and serialize compactly as uint64 lists
        sorted_items = sorted(pool_by_hash.values(), key=lambda x: (x[0], x[1]))[:max_pool]
        rank_hist: Dict[str, int] = {}
        serialized: List[Dict[str, Any]] = []
        for idx, (rk, nnz, sc) in enumerate(sorted_items):
            rank_hist[str(rk)] = rank_hist.get(str(rk), 0) + 1
            u_arr, v_arr, w_arr = fg.scheme_to_f2_packed(sc)
            serialized.append(
                {
                    "id": f"{fk}_R{rk}_{idx:03d}",
                    "m": m,
                    "n": n,
                    "p": p,
                    "rank": rk,
                    "nnz": nnz,
                    "canonical_sha256": sc.canonical_sha256(),
                    "u": [int(x) for x in u_arr],
                    "v": [int(x) for x in v_arr],
                    "w": [int(x) for x in w_arr],
                }
            )

        dt_f = time.perf_counter() - t_f0
        pool_payload["formats"][fk] = {
            "shape": [m, n, p],
            "sota_z2": ref["z2"],
            "sota_char0": ref["zt"],
            "collect_max_rank": collect_max_r,
            "best_rank_found": best_r,
            "count": len(serialized),
            "rank_histogram": rank_hist,
            "schemes": serialized,
        }
        stage_stats[fk] = {
            "best_rank": best_r,
            "pool_count": len(serialized),
            "rank_histogram": rank_hist,
            "waves": wave,
            "elapsed_s": round(dt_f, 2),
            "executed_flips": total_flips,
            "flips_per_s": round(total_flips / max(dt_f, 1e-9), 1),
            "peak_vram_mib": round(peak_vram, 2),
        }
        print(
            f"  [F2 Pool | {fk}] best_R={best_r} (SOTA Z2={ref['z2']}) | pool={len(serialized)} "
            f"hist={rank_hist} | {wave} waves, {dt_f:.1f}s, {total_flips/max(dt_f,1e-9)/1e6:.1f}M flips/s, "
            f"VRAM={peak_vram:.1f}MiB",
            flush=True,
        )

    os.makedirs(_ARTIFACTS_DIR, exist_ok=True)
    pool_path = os.path.join(_ARTIFACTS_DIR, "f2_pool_phase1.json")
    with open(pool_path, "w", encoding="utf-8") as fh:
        json.dump(pool_payload, fh, ensure_ascii=False)
    total_schemes = sum(v["count"] for v in pool_payload["formats"].values())
    print(
        f"[Stage 1] Saved {total_schemes} verified diverse F2 schemes to {pool_path} "
        f"in {time.perf_counter() - t_stage:.1f}s!",
        flush=True,
    )
    return {
        "pool_path": pool_path,
        "total_schemes": total_schemes,
        "elapsed_s": round(time.perf_counter() - t_stage, 2),
        "formats": stage_stats,
        "records_saved": records_saved,
    }


# =============================================================================
# 3. Stage 2: Track B/A-Direct Z_T Assault under Genuine SLEEP-DAG-RSI v8
# =============================================================================


def _score_zt_scheme(rank: int, nnz: int, sota_zt: int, ref_start_r: int) -> Tuple[float, List[float], List[int]]:
    """
    Monotone scalar score + multi-bucket margin vector for `DiscoveryTree` / `LeanDAGPolicy`
    and `BackwardSubGoalDecomposer`.
    """
    score = 100.0 * float(ref_start_r - rank) - 0.01 * float(nnz) / max(rank, 1)
    # Buckets: [beat_start, reach_sota+2, reach_sota, beat_sota_ZT, beat_sota_stretch]
    buckets = [
        float(ref_start_r - 0.5 - rank),
        float(sota_zt + 2.5 - rank),
        float(sota_zt + 0.5 - rank),
        float(sota_zt - 0.5 - rank),
        float(sota_zt - 1.5 - rank),
    ]
    pass_mask = [1 if m_val > 0.0 else 0 for m_val in buckets]
    return score, buckets, pass_mask


def run_sleep_dag_zt_target(
    engine: fg.FlipGraphGPUEngine,
    fmt_key: str,
    lib: Dict[str, List[bv.BilinearScheme]],
    time_budget_s: float,
    skill_state: dream_skill.MutableSkillState,
    records_saved: List[Dict[str, Any]],
    seed: int = 2026,
) -> Dict[str, Any]:
    """
    Execute a genuine SLEEP-DAG-RSI v8 Z_T discovery campaign for one target format:
      - Initializes `DiscoveryTree`, `Problem`, `LeanDAGPolicy(entropic_tau=4.0)`,
        `BackwardSubGoalDecomposer`, `WaldMicroSentinel(noise_regime="deterministic")`.
      - Runs multi-branch RTX 5060 Z_T waves with entropic max-tilted walker allocation
        (`compute_max_tilted_objective(tau=4.0)`).
      - Uses `BackwardSubGoalDecomposer` on stalled dead buckets (`beat_sota`) to trigger
        fresh symmetry-remixed sub-tensor `CROSSOVER` & deeper `plus`-tunneling.
      - Concludes with a formal `TaskCartridge` + `AutoDreamRunner` + `BlindVaultGuard(max_calls=1)`
        sealed verification cycle over the top discovered branches!
    """
    t0 = time.perf_counter()
    rng = random.Random(seed)
    ref = SOTA_REF[fmt_key]
    m, n, p = ref["m"], ref["n"], ref["p"]
    sota_zt = ref["zt"]

    named_seeds = build_target_zt_seeds(fmt_key, lib, rng)
    ref_start_r = min(sc.rank for _, sc in named_seeds)

    # 1. Initialize SLEEP-DAG-RSI v8 structures
    problem = dream_tree.Problem()
    problem.name = f"fmm_zt_{fmt_key}"
    base_score, base_buckets, base_mask = _score_zt_scheme(
        ref_start_r, named_seeds[0][1].additive_complexity()["nnz_total"], sota_zt, ref_start_r
    )
    tree = dream_tree.DiscoveryTree(task=problem.name, baseline_score=base_score)
    tree.add(
        dream_tree.Node(
            id=tree.root_id,
            parent_id=None,
            branch=-1,
            attempt=0,
            score=base_score,
            fail_class="ok",
            valid=True,
            pass_mask=base_mask,
            bucket_scores=base_buckets,
            meta={"stage": "root", "rank": ref_start_r},
        )
    )

    policy = dream_engine.LeanDAGPolicy(
        beta=0.6, slope_weight=0.55, max_recovery=1,
        use_counterfactual=True, use_crossover=True, entropic_tau=4.0,
    )
    decomposer = dream_skill.BackwardSubGoalDecomposer()
    sentinel = dream_sentinel.WaldMicroSentinel(noise_regime="deterministic")

    # Track per-branch active elite pools
    branch_pools: Dict[int, List[bv.BilinearScheme]] = {}
    branch_best_node: Dict[int, str] = {}
    branch_meta: Dict[int, Dict[str, Any]] = {}
    all_verified_elites: Dict[str, Tuple[int, int, bv.BilinearScheme]] = {}

    for b_idx, (lbl, sc) in enumerate(named_seeds):
        rep = bv.verify_scheme(sc)
        assert rep.valid and rep.is_ternary
        nnz = rep.additive_complexity["nnz_total"]
        sc_val, b_sc, p_mk = _score_zt_scheme(sc.rank, nnz, sota_zt, ref_start_r)
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

    # Adaptive GPU workload sizing: deep continuous waves prevent premature stalls
    if ref_start_r <= 105:
        total_walkers_wave = 4096
        steps_wave = 45000
    elif ref_start_r <= 195:
        total_walkers_wave = 2048
        steps_wave = 20000
    else:
        total_walkers_wave = 1536
        steps_wave = 16000

    total_flips = 0
    total_plus = 0
    total_reduces = 0
    peak_vram = 0.0
    wave = 0
    best_r, best_nnz, best_scheme = min(all_verified_elites.values(), key=lambda x: (x[0], x[1]))
    decomposer_triggers = 0

    print(
        f"\n[Stage 2 | Z_T SLEEP-DAG-RSI | {fmt_key}] start_R={ref_start_r} (SOTA ZT={sota_zt}, Q={ref['q']}) | "
        f"{len(named_seeds)} branches, budget={time_budget_s:.0f}s",
        flush=True,
    )

    while (time.perf_counter() - t0) < time_budget_s:
        wave += 1

        # Compute TTT-Discover Max-Tilted entropic weights (tau = 4.0) across branches!
        b_ids = sorted(branch_pools.keys())
        b_scores = [tree.nodes[branch_best_node[b]].score for b in b_ids]
        tilted = dream_engine.compute_max_tilted_objective(
            b_scores, tau=policy.entropic_tau, baseline=base_score, normalize_by_scale=True
        )
        weights = tilted["tilted_weights_normalized"]
        # Allocate GPU walkers proportionally to entropic weights (floor 256 per branch for exploration)
        alloc_walkers = [
            max(256, int(round(w * total_walkers_wave / 64.0)) * 64) for w in weights
        ]

        wave_improved = False
        for b_idx, nw in zip(b_ids, alloc_walkers):
            if (time.perf_counter() - t0) >= time_budget_s:
                break
            stalls = branch_meta[b_idx]["stalls"]
            # Escalate plus-tunneling headroom when stalled
            plus_headroom = 2 if stalls < 2 else (3 if stalls < 4 else 4)
            plus_period = max(250, (1200 if ref_start_r <= 100 else 700) // (1 + min(stalls, 3)))

            res = engine.run_zt_batch(
                branch_pools[b_idx],
                n_walkers=nw,
                n_steps=steps_wave,
                plus_period=plus_period,
                plus_headroom=plus_headroom,
                top_k=16,
                seed=seed + wave * 4099 + b_idx * 97,
            )
            total_flips += res.executed_flips
            total_plus += res.executed_plus
            total_reduces += res.executed_reduces
            peak_vram = max(peak_vram, res.peak_vram_mib)

            cand = res.top_schemes[0]
            rep = _verify_and_maybe_save_record(cand, records_saved, tag=f"zt_{fmt_key}_w{wave}_b{b_idx}")
            nnz = rep.additive_complexity["nnz_total"]
            sc_val, b_sc, p_mk = _score_zt_scheme(cand.rank, nnz, sota_zt, ref_start_r)

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
                        "plus_period": plus_period, "n_walkers": nw, "stage": "gpu_zt_wave",
                    },
                )
            )

            # Update branch pool with verified diverse elites
            verified_top = []
            for sc_i in res.top_schemes[:12]:
                r_i = bv.verify_scheme(sc_i)
                if r_i.valid and r_i.is_ternary:
                    verified_top.append(sc_i)
                    h_i = r_i.canonical_sha256
                    if h_i not in all_verified_elites:
                        all_verified_elites[h_i] = (sc_i.rank, r_i.additive_complexity["nnz_total"], sc_i)

            if cand.rank < branch_meta[b_idx]["best_r"] or (
                cand.rank == branch_meta[b_idx]["best_r"] and nnz < branch_meta[b_idx]["best_nnz"]
            ):
                branch_meta[b_idx]["best_r"] = cand.rank
                branch_meta[b_idx]["best_nnz"] = nnz
                branch_meta[b_idx]["stalls"] = 0
                branch_best_node[b_idx] = nid
            else:
                branch_meta[b_idx]["stalls"] += 1

            if verified_top:
                # Keep top best-rank schemes + previous anchor so we never lose ground
                prev_best = branch_pools[b_idx][0]
                branch_pools[b_idx] = verified_top[:10] + ([prev_best] if prev_best.rank <= verified_top[0].rank else [])

            if cand.rank < best_r or (cand.rank == best_r and nnz < best_nnz):
                if cand.rank < best_r:
                    wave_improved = True
                    print(
                        f"  [Z_T {fmt_key} | wave {wave} | b{b_idx}] RANK DROP: {best_r} -> {cand.rank} "
                        f"(nnz={nnz}, adds={rep.additive_complexity['naive_additions_total']}, "
                        f"SOTA={sota_zt}, t={time.perf_counter()-t0:.1f}s)",
                        flush=True,
                    )
                best_r = cand.rank
                best_nnz = nnz
                best_scheme = cand

        # BackwardSubGoalDecomposer check every 3 waves when beat_sota bucket is still unsolved
        if wave % 3 == 0 and best_r >= sota_zt and (time.perf_counter() - t0) < (time_budget_s - 4.0):
            decomp = decomposer.decompose([tree], problem)
            if decomp.get("dead_buckets"):
                decomposer_triggers += 1
                # Inject a freshly symmetry-remixed sub-tensor seed into the worst stalled branch!
                worst_b = max(b_ids, key=lambda b: (branch_meta[b]["best_r"], branch_meta[b]["stalls"]))
                fresh_seeds = build_target_zt_seeds(fmt_key, lib, rng)
                _, fresh_sc = rng.choice(fresh_seeds)
                # Blend fresh composite seed with current global champion for structural diversity
                branch_pools[worst_b] = [best_scheme, fresh_sc]
                branch_meta[worst_b]["stalls"] = 0

        if wave == 1 or wave_improved or wave % 4 == 0:
            dt_now = time.perf_counter() - t0
            print(
                f"  [Z_T {fmt_key} | wave {wave}] best_R={best_r} (nnz={best_nnz}, SOTA={sota_zt}) | "
                f"elites={len(all_verified_elites)} | {total_flips/max(dt_now,1e-9)/1e6:.1f}M flips/s | "
                f"VRAM={peak_vram:.1f}MiB | t={dt_now:.1f}/{time_budget_s:.0f}s",
                flush=True,
            )

    # 2. Final sealed AutoDreamRunner + BlindVaultGuard(max_calls=1) certification cycle!
    top_elites = sorted(all_verified_elites.values(), key=lambda x: (x[0], x[1]))[:4]

    def _make_scheme_fn(sc_obj: bv.BilinearScheme):
        return lambda: sc_obj

    def _verify_and_score_cartridge(fn):
        sc_obj = fn()
        rep_c = bv.verify_scheme(sc_obj)
        if not (rep_c.valid and rep_c.is_ternary):
            return False, -1e6, "algorithmic_failure", {"valid": False}
        nnz_c = rep_c.additive_complexity["nnz_total"]
        s_c, b_c, pm_c = _score_zt_scheme(sc_obj.rank, nnz_c, sota_zt, ref_start_r)
        return True, s_c, "ok", {
            "rank": sc_obj.rank,
            "nnz": nnz_c,
            "naive_additions": rep_c.additive_complexity["naive_additions_total"],
            "bucket_scores": b_c,
            "pass_mask": pm_c,
            "canonical_sha256": rep_c.canonical_sha256,
        }

    def _crossover_synth(u_c: dream_cartridge.BranchCandidate, v_c: dream_cartridge.BranchCandidate):
        # Structural symmetry-crossover + quick 1-wave GPU polish between two elite basins
        sc_u = u_c.fn()
        sc_v = v_c.fn()
        res_x = engine.run_zt_batch(
            [sc_u, sc_v], n_walkers=1024, n_steps=4000, plus_period=400, plus_headroom=2,
            top_k=4, seed=seed + 99991,
        )
        cand_x = res_x.top_schemes[0]
        _verify_and_maybe_save_record(cand_x, records_saved, tag=f"autodream_xover_{fmt_key}")
        return dream_cartridge.BranchCandidate(
            id=f"xover_{u_c.id}_{v_c.id}",
            branch=99,
            description=f"GPU basin crossover of {u_c.id} and {v_c.id}",
            fn=_make_scheme_fn(cand_x),
            parent_id=u_c.id,
        )

    vault_scheme_ref: List[Optional[bv.BilinearScheme]] = [None]

    def _blind_vault_cert(fn):
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
            id=f"elite_{idx}_R{rk}",
            branch=idx,
            description=f"GPU Z_T elite {fmt_key} R={rk} nnz={nz}",
            fn=_make_scheme_fn(sc_e),
        )
        for idx, (rk, nz, sc_e) in enumerate(top_elites)
    ]
    cartridge = dream_cartridge.TaskCartridge(
        task_name=f"fmm_zt_{fmt_key}",
        description=f"Ternary Z_T FMM discovery for <{m},{n},{p}> (SOTA={sota_zt})",
        baseline_fn=_make_scheme_fn(named_seeds[0][1]),
        verify_and_score=_verify_and_score_cartridge,
        initial_branches=init_branches,
        crossover_synthesizer=_crossover_synth,
        sentinel_noise_regime="deterministic",
        blind_vault_evaluator=_blind_vault_cert,
    )
    runner = dream_cartridge.AutoDreamRunner(cartridge)
    ad_report = runner.run()

    # Sync any crossover breakthrough back to champion (including best_scheme itself!)
    vault = ad_report.get("blind_vault_report") or {}
    if vault.get("valid") and vault_scheme_ref[0] is not None:
        v_r = int(vault.get("rank", 9999))
        v_nnz = int(vault.get("nnz", 10**9))
        if (v_r, v_nnz) < (best_r, best_nnz):
            best_r = v_r
            best_nnz = v_nnz
            best_scheme = vault_scheme_ref[0]

    # Commit discovery tree into bounded MutableSkillState (SKILL.state)
    skill_state.commit_cycle(tree, problem)
    # Audit champion vs root via WaldMicroSentinel
    champ_branch = min(b_ids, key=lambda b: (branch_meta[b]["best_r"], branch_meta[b]["best_nnz"]))
    champ_node = tree.nodes[branch_best_node[champ_branch]]
    root_node = tree.nodes[tree.root_id]
    sentinel_ok = sentinel.evaluate_gate(
        root_node,
        champ_node,
        base_buckets,
        lambda idxs: [champ_node.bucket_scores[i] for i in idxs],
    )

    dt_total = time.perf_counter() - t0
    final_rep = bv.verify_scheme(best_scheme)
    final_rank = final_rep.rank
    final_nnz = final_rep.additive_complexity["nnz_total"]
    return {
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 1 RTX 5060 SLEEP-DAG-RSI FMM Scout.")
    parser.add_argument(
        "--phase",
        choices=("f2-pool", "zt-assault", "all"),
        default="all",
        help="Run Stage 1 (f2-pool), Stage 2 (zt-assault), or both sequentially.",
    )
    parser.add_argument("--f2-budget", type=float, default=160.0, help="Wall-clock budget (s) for F2 pool stage.")
    parser.add_argument("--zt-budget", type=float, default=440.0, help="Wall-clock budget (s) for Z_T assault stage.")
    parser.add_argument("--seed", type=int, default=2026, help="Master PRNG seed.")
    args = parser.parse_args(argv)

    t_all = time.perf_counter()
    engine = fg.FlipGraphGPUEngine()
    os.makedirs(_ARTIFACTS_DIR, exist_ok=True)
    summary_path = os.path.join(_ARTIFACTS_DIR, "phase1_gpu_summary.json")

    # Load existing summary if running split stages
    summary: Dict[str, Any] = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as fh:
                summary = json.load(fh)
        except Exception:
            summary = {}

    summary.setdefault("device", engine.backend.device_name)
    summary.setdefault("arch", engine.backend.arch)
    summary.setdefault("records_saved", [])

    lib = build_from_scratch_block_library(engine, seed=args.seed)
    summary["block_library_ranks"] = {k: min(s.rank for s in v) for k, v in lib.items()}

    if args.phase in ("f2-pool", "all"):
        f2_res = run_f2_pool_stage(engine, lib, time_budget_s=args.f2_budget, seed=args.seed)
        summary["f2_pool_stage"] = f2_res
        summary["records_saved"].extend(f2_res["records_saved"])
        summary["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)

    if args.phase in ("zt-assault", "all"):
        t_zt0 = time.perf_counter()
        skill_state = dream_skill.MutableSkillState(task="fmm_phase1_zt_assault")
        records_zt: List[Dict[str, Any]] = []

        # Budget allocation across the 5 high-value Z_T targets (emphasizing +7.0/+5.0 ridges & 4x5x5/4x5x6!)
        zt_targets = [
            ("5x8x8", 0.24),  # starts RIGHT AT 230 SOTA (+7.0 ridge!) -> 1 drop = WORLD RECORD!
            ("7x7x7", 0.22),  # starts RIGHT AT 250 ZT SOTA -> 1 drop beats ZT, 2 drops beats Q!
            ("6x6x7", 0.20),  # starts at 185 (+2 from 183 SOTA, +5.0 ridge!)
            ("4x5x5", 0.18),  # SOTA ZT=76 -> target <= 75
            ("4x5x6", 0.16),  # SOTA ZT=90 -> target <= 89
        ]
        zt_results: Dict[str, Any] = {}
        for idx, (fk, frac) in enumerate(zt_targets):
            zt_results[fk] = run_sleep_dag_zt_target(
                engine=engine,
                fmt_key=fk,
                lib=lib,
                time_budget_s=args.zt_budget * frac,
                skill_state=skill_state,
                records_saved=records_zt,
                seed=args.seed + 5000 + idx * 777,
            )

        summary["zt_assault_stage"] = {
            "elapsed_s": round(time.perf_counter() - t_zt0, 2),
            "targets": zt_results,
            "skill_state_summary": {
                "round_index": skill_state.round_index,
                "invariants_count": len(skill_state.best_invariants),
                "unsolved_buckets_count": len(skill_state.unsolved_buckets),
                "serialized_bytes": skill_state.serialized_byte_size(),
            },
        }
        summary["records_saved"].extend(records_zt)
        summary["total_wall_s"] = round(time.perf_counter() - t_all, 2)
        summary["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)

        print(f"\n[Stage 2] Complete! Summary saved to {summary_path}", flush=True)
        for fk, res in zt_results.items():
            print(
                f"  SUMMARY {fk}: start={res['start_rank']} -> best_ZT={res['best_rank']} "
                f"(SOTA ZT={res['sota_zt']}, Q={res['sota_q']}, beat_ZT={res['beat_sota_zt']}) | "
                f"nnz={res['best_nnz']}, adds={res['best_naive_additions']}, "
                f"vault_ok={res['blind_vault_verified']}, VRAM={res['peak_vram_mib']:.1f}MiB",
                flush=True,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())

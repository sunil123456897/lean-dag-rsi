"""
Phase 3 (Final ~14-Min FMM Shot): GPU F_3 (mod-3) 2-Bitplane Flip-Graph Assault
on NVIDIA GeForce RTX 5060 (Blackwell SM 12.0) + SLEEP-DAG-RSI v8.

Mathematical Breakthrough
-------------------------
Over Z_T = {-1, 0, +1}, ~70% of candidate flips are rejected by +-2 overflow,
sign-locking composite padded-Strassen seams at 5x8x8:230 and 7x7x7:250.
Over F_3 = {-1, 0, +1} mod 3:
  1. ZERO overflow: (+1)+(+1) = -1 (mod 3), (-1)+(-1) = +1 (mod 3). Every flip,
     plus, 2-axis reduce, 3-term reduce, and triple-collision split succeeds 100%.
  2. Unlocks the Z[1/2] rational manifold (since 2^{-1} == -1 mod 3), invisible
     to both Z_T and F_2!
  3. All top verified F_3 basins and any record-beating F_3 schemes are exported
     incrementally to `fmm_discovery/artifacts/f3_pool_phase3.json` (and
     `record_f3_<m>x<n>x<p>_R<R>.json`) for 3-adic Hensel lifting (3^k -> Z[1/2]/Q/Z).

Strictly 100% from-scratch (ZERO external schemes) and hard-capped by `--max-minutes 14.0`.
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

# 2026 SOTA reference table (char-0 Q/Z_T and Z2; trigger_f3 = rank that beats char-0 SOTA or cracks 4x4x4:48!)
SOTA_2026: Dict[str, Dict[str, int]] = {
    "4x4x4": {"m": 4, "n": 4, "p": 4, "z2": 47, "zt": 49, "q": 48, "trigger_f3": 48},
    "5x8x8": {"m": 5, "n": 8, "p": 8, "z2": 230, "zt": 230, "q": 230, "trigger_f3": 229},
    "7x7x7": {"m": 7, "n": 7, "p": 7, "z2": 245, "zt": 250, "q": 249, "trigger_f3": 248},
    "6x6x7": {"m": 6, "n": 6, "p": 7, "z2": 183, "zt": 183, "q": 183, "trigger_f3": 182},
    "5x7x8": {"m": 5, "n": 7, "p": 8, "z2": 204, "zt": 204, "q": 204, "trigger_f3": 203},
    "4x4x5": {"m": 4, "n": 4, "p": 5, "z2": 60, "zt": 61, "q": 61, "trigger_f3": 60},
    "4x5x5": {"m": 4, "n": 5, "p": 5, "z2": 73, "zt": 76, "q": 76, "trigger_f3": 75},
    "4x5x6": {"m": 4, "n": 5, "p": 6, "z2": 89, "zt": 90, "q": 90, "trigger_f3": 89},
}


def _fmt_key(m: int, n: int, p: int) -> str:
    return f"{m}x{n}x{p}"


def _pack_f3_pool_entry(
    sc: bv.BilinearScheme, rep: bv.VerificationReport, entry_id: str, tag: str = ""
) -> Dict[str, Any]:
    pu, nu, pv, nv, pw, nw = fg.scheme_to_f3_packed(sc)
    return {
        "id": entry_id,
        "m": sc.m,
        "n": sc.n,
        "p": sc.p,
        "ring": "Z3",
        "rank": rep.rank,
        "nnz": rep.additive_complexity["nnz_total"],
        "naive_additions": rep.additive_complexity["naive_additions_total"],
        "canonical_sha256": rep.canonical_sha256,
        "tag": tag,
        "pu": [int(x) for x in pu],
        "nu": [int(x) for x in nu],
        "pv": [int(x) for x in pv],
        "nv": [int(x) for x in nv],
        "pw": [int(x) for x in pw],
        "nw": [int(x) for x in nw],
    }


def _flush_f3_pool(pool_by_fmt: Dict[str, Dict[str, Any]], pool_path: str) -> None:
    """Atomically write `f3_pool_phase3.json` so `@fmm-sota` can ingest schemes immediately."""
    os.makedirs(os.path.dirname(pool_path), exist_ok=True)
    payload = {
        "format_version": "f3_pool_v1",
        "ring": "Z3",
        "convention": "C_eq_AB_row_major",
        "provenance": "100%_from_scratch_zero_external_seeds",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "packing_spec": (
            "Each pu/nu, pv/nv, pw/nw is a uint64[R] 2-bitplane F_3 mask (+1->p bit, -1=2->n bit, "
            "p & n == 0) in C_eq_AB_row_major order. Unpack via "
            "flip_graph_gpu.f3_packed_to_scheme(m, n, p, pu, nu, pv, nv, pw, nw)."
        ),
        "total_schemes": sum(len(v["schemes"]) for v in pool_by_fmt.values()),
        "formats": pool_by_fmt,
    }
    tmp = pool_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    os.replace(tmp, pool_path)


def _verify_and_save_f3_record(
    scheme: bv.BilinearScheme,
    records_saved: List[Dict[str, Any]],
    tag: str = "",
) -> bv.VerificationReport:
    """Strictly verify over exact `Z3` in `brent_verifier`; auto-save `record_f3_*.json` if <= trigger_f3."""
    rep = bv.verify_scheme(scheme)
    if not rep.valid or rep.ring != "Z3":
        raise RuntimeError(f"Z3 VERIFICATION FAILURE on {scheme.name}: valid={rep.valid} ring={rep.ring}")

    fk = _fmt_key(*scheme.shape)
    ref = SOTA_2026.get(fk)
    if ref is not None and scheme.rank <= ref["trigger_f3"]:
        os.makedirs(_ARTIFACTS_DIR, exist_ok=True)
        fname = f"record_f3_{fk}_R{scheme.rank}.json"
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
            rec_kind = f"F3_beats_char0_SOTA_{ref['q']}" if scheme.rank < ref["q"] else f"F3_reaches_target_{ref['trigger_f3']}"
            scheme.metadata = {
                **dict(scheme.metadata),
                "record_kind": rec_kind,
                "sota_before": ref,
                "discovery_tag": tag,
                "provenance": "100%_from_scratch_zero_external_seeds",
                "verified_brent": True,
                "verified_basis": True,
                "additive_complexity": rep.additive_complexity,
                "discovered_at": datetime.now(timezone.utc).isoformat(),
            }
            scheme.save_json(fpath)
            info = {
                "format": fk,
                "ring": "Z3",
                "rank": scheme.rank,
                "record_kind": rec_kind,
                "nnz": rep.additive_complexity["nnz_total"],
                "naive_additions": rep.additive_complexity["naive_additions_total"],
                "canonical_sha256": rep.canonical_sha256,
                "path": fpath,
            }
            records_saved.append(info)
            print(
                f"\n  *** [NEW F_3 RECORD HIT CERTIFIED!] {fk} Z3 R={scheme.rank} ({rec_kind}) "
                f"nnz={info['nnz']} adds={info['naive_additions']} -> {fname} ***\n",
                flush=True,
            )
    return rep


# =============================================================================
# 1. Fast Stage 0: 100% From-Scratch Exact Min-Rank F_3 Primitive Library
# =============================================================================


def _keep_strict_min_rank_f3(schemes: Sequence[bv.BilinearScheme], max_keep: int = 12) -> List[bv.BilinearScheme]:
    valid_list: List[Tuple[int, int, bv.BilinearScheme]] = []
    seen: set[str] = set()
    for sc in schemes:
        rep = bv.verify_scheme(sc)
        if rep.valid and rep.ring == "Z3" and rep.canonical_sha256 not in seen:
            seen.add(rep.canonical_sha256)
            valid_list.append((sc.rank, rep.additive_complexity["nnz_total"], sc))
    if not valid_list:
        raise RuntimeError("No valid Z3 schemes in Stage 0 pool")
    min_r = min(x[0] for x in valid_list)
    filtered = [sc for r, _, sc in sorted(valid_list, key=lambda x: (x[0], x[1])) if r == min_r]
    return filtered[:max_keep]


def build_from_scratch_f3_library(
    engine: fg.FlipGraphGPUEngine, seed: int = 2027
) -> Dict[str, List[bv.BilinearScheme]]:
    """
    Build clean, diverse 100% from-scratch Z_T primitive blocks purely from schoolbook naive +
    Strassen <2,2,2:7> + Laderman <3,3,3:23> (reusing Phase 2's verified strict min-rank
    builder with ZERO external schemes!), then cast every verified Z_T primitive to F_3 (`Z3`).
    """
    import run_phase2_zt_assault as p2_mod

    t0 = time.perf_counter()
    zt_lib = p2_mod.build_strict_min_rank_library(engine, seed=seed)
    lib: Dict[str, List[bv.BilinearScheme]] = {}
    for key, schemes in zt_lib.items():
        f3_list = [sc.reduce_mod(3) for sc in schemes]
        lib[key] = _keep_strict_min_rank_f3(f3_list, max_keep=12)
    dt = time.perf_counter() - t0
    summary = {k: (v[0].rank, len(v)) for k, v in lib.items()}
    print(f"[Stage 0 F_3] Cast verified Z_T library -> F_3 in {dt:.1f}s | (rank, count): {summary}", flush=True)
    return lib


def _pick_min_sub_f3(
    lib: Dict[str, List[bv.BilinearScheme]],
    shape: Tuple[int, int, int],
    rng: random.Random,
    randomize: bool = True,
) -> bv.BilinearScheme:
    canon = tuple(sorted(shape))
    key = _fmt_key(*canon)
    if key in lib and lib[key]:
        min_r = min(s.rank for s in lib[key])
        best_pool = [s for s in lib[key] if s.rank == min_r]
        chosen = rng.choice(best_pool) if randomize else best_pool[0]
        sc = fg.permute_axes(chosen, shape)
    else:
        sc = bv.make_naive_scheme(*shape, "Z3")
    return fg.random_monomial_symmetry(sc, rng) if randomize else sc


def build_f3_target_seeds(
    fmt_key: str,
    lib: Dict[str, List[bv.BilinearScheme]],
    rng: random.Random,
    extra_meta_seeds: Optional[Sequence[Tuple[str, bv.BilinearScheme]]] = None,
) -> List[Tuple[str, bv.BilinearScheme]]:
    seeds: List[Tuple[str, bv.BilinearScheme]] = []

    if fmt_key == "4x4x4":
        seeds.append(("kron_strassen_49", lib["4x4x4"][0]))
        s244_a = lib["2x4x4"][0]
        seeds.append(("twin_ds_244_52", fg.direct_sum_m(s244_a, s244_a, name="4x4x4_ds_52")))
        seeds.append(("naive_444_64", bv.make_naive_scheme(4, 4, 4, "Z3")))

    elif fmt_key == "5x8x8":
        # 4 orthogonal exact R=230 orientations (aligned + 3 S_3 orientations)
        subs_a = [_pick_min_sub_f3(lib, sh, rng, randomize=False) for sh in [(3,4,4),(3,4,4),(2,4,4),(3,4,4),(2,4,4),(3,4,4),(2,4,4)]]
        seeds.append(("ps_588_230_aligned", fg.padded_strassen_compose(2, 3, 4, 4, 4, 4, subs_a, name="5x8x8_f3_230_aligned")))
        subs_b = [_pick_min_sub_f3(lib, sh, rng) for sh in [(4,3,4),(4,3,4),(4,3,4),(4,2,4),(4,2,4),(4,3,4),(4,2,4)]]
        seeds.append(("ps_858_s3_230_B", fg.permute_axes(fg.padded_strassen_compose(4, 4, 3, 2, 4, 4, subs_b), (5, 8, 8), name="5x8x8_f3_230_B")))
        subs_c = [_pick_min_sub_f3(lib, sh, rng) for sh in [(4,4,3),(4,4,2),(4,4,3),(4,4,2),(4,4,3),(4,4,3),(4,4,2)]]
        seeds.append(("ps_885_s3_230_C", fg.permute_axes(fg.padded_strassen_compose(4, 4, 4, 4, 2, 3, subs_c), (5, 8, 8), name="5x8x8_f3_230_C")))
        subs_d = [_pick_min_sub_f3(lib, sh, rng) for sh in [(3,4,4),(3,4,4),(2,4,4),(3,4,4),(2,4,4),(3,4,4),(2,4,4)]]
        seeds.append(("ps_588_230_D", fg.padded_strassen_compose(2, 3, 4, 4, 4, 4, subs_d, name="5x8x8_f3_230_D")))

    elif fmt_key == "7x7x7":
        # 4 orthogonal exact R=250 compositions
        for k_i in range(4):
            subs = [_pick_min_sub_f3(lib, sh, rng, randomize=(k_i > 0)) for sh in [(4,4,4),(3,4,4),(4,4,3),(3,3,4),(4,3,3),(3,4,3),(4,3,4)]]
            sc = fg.padded_strassen_compose(4, 3, 4, 3, 4, 3, subs, name=f"7x7x7_f3_250_v{k_i}")
            if k_i == 1:
                sc = fg.cyclic_rotate_scheme(sc, name="7x7x7_f3_250_c3")
            elif k_i == 2:
                sc = fg.transpose_scheme(sc, name="7x7x7_f3_250_tr")
            seeds.append((f"ps_777_250_v{k_i}", sc))

    elif fmt_key == "6x6x7":
        # Exact R=185 (aligned + randomized + S_3 seam)
        for k_i in range(2):
            subs = [_pick_min_sub_f3(lib, sh, rng, randomize=(k_i > 0)) for sh in [(3,3,4),(3,3,3),(3,3,4),(3,3,3),(3,3,4),(3,3,4),(3,3,3)]]
            seeds.append((f"ps_667_185_v{k_i}", fg.padded_strassen_compose(3, 3, 3, 3, 3, 4, subs, name=f"6x6x7_f3_185_v{k_i}")))
        subs_m = [_pick_min_sub_f3(lib, sh, rng) for sh in [(4,3,3),(3,3,3),(4,3,3),(3,3,3),(4,3,3),(3,3,3),(4,3,3)]]
        seeds.append(("ps_766_s3_185", fg.permute_axes(fg.padded_strassen_compose(4, 3, 3, 3, 3, 3, subs_m), (6, 6, 7), name="6x6x7_f3_185_s3")))

    elif fmt_key == "5x7x8":
        # Exact R=206 (aligned + S_3 permutations)
        for k_i in range(2):
            subs = [_pick_min_sub_f3(lib, sh, rng, randomize=(k_i > 0)) for sh in [(4,4,3),(4,4,2),(3,4,3),(4,4,2),(3,4,3),(4,4,3),(3,4,2)]]
            s_785 = fg.padded_strassen_compose(3, 4, 4, 4, 2, 3, subs, name=f"7x8x5_f3_206_v{k_i}")
            seeds.append((f"ps_785_s3_206_v{k_i}", fg.permute_axes(s_785, (5, 7, 8), name=f"5x7x8_f3_206_v{k_i}")))
        subs_b = [_pick_min_sub_f3(lib, sh, rng) for sh in [(3,4,4),(2,4,4),(3,4,4),(2,3,4),(3,3,4),(2,4,4),(3,3,4)]]
        seeds.append(("ps_578_206_B", fg.padded_strassen_compose(3, 2, 4, 3, 4, 4, subs_b, name="5x7x8_f3_206_B")))

    elif fmt_key == "4x4x5":
        # Direct sum <4,4,2:26> + <4,4,3:38> = 64! Plus <4,4,4:49> + <4,4,1:16> = 65 + naive 80!
        s442 = _pick_min_sub_f3(lib, (4, 4, 2), rng, randomize=False)
        s443 = _pick_min_sub_f3(lib, (4, 4, 3), rng, randomize=False)
        seeds.append(("ds_442_443_64", bv.direct_sum_p(s442, s443, name="4x4x5_f3_64")))
        seeds.append(("ds_444_441_65", bv.direct_sum_p(lib["4x4x4"][0], bv.make_naive_scheme(4, 4, 1, "Z3"), name="4x4x5_f3_65")))
        seeds.append(("naive_445_80", bv.make_naive_scheme(4, 4, 5, "Z3")))

    elif fmt_key == "4x5x5":
        subs_a = [_pick_min_sub_f3(lib, sh, rng, randomize=False) for sh in [(2,3,3),(2,3,3),(2,3,2),(2,2,3),(2,2,2),(2,3,2),(2,2,3)]]
        seeds.append(("ps_455_81_aligned", fg.padded_strassen_compose(2, 2, 3, 2, 3, 2, subs_a, name="4x5x5_f3_81_A")))
        seeds.append(("ds_452_453", bv.direct_sum_p(lib["4x5x2"][0], lib["4x5x3"][0], name="4x5x5_f3_ds")))
        seeds.append(("naive_455_100", bv.make_naive_scheme(4, 5, 5, "Z3")))

    elif fmt_key == "4x5x6":
        subs_a = [_pick_min_sub_f3(lib, sh, rng, randomize=False) for sh in [(2,3,3),(2,3,3),(2,3,3),(2,2,3),(2,2,3),(2,3,3),(2,2,3)]]
        seeds.append(("ps_456_93_aligned", fg.padded_strassen_compose(2, 2, 3, 2, 3, 3, subs_a, name="4x5x6_f3_93_A")))
        seeds.append(("ds_453_453", bv.direct_sum_p(lib["4x5x3"][0], lib["4x5x3"][0], name="4x5x6_f3_ds")))
        seeds.append(("naive_456_120", bv.make_naive_scheme(4, 5, 6, "Z3")))
    else:
        raise ValueError(f"Unsupported format {fmt_key}")

    if extra_meta_seeds:
        for lbl, sc in extra_meta_seeds:
            seeds.append((lbl, sc))

    for lbl, sc in seeds:
        rep = bv.verify_scheme(sc)
        if not (rep.valid and rep.ring == "Z3"):
            raise RuntimeError(f"F_3 seed {fmt_key}/{lbl} failed exact Z3 brent_verifier!")
    return seeds


def generate_meta_flip_seeds_f3(
    engine: fg.FlipGraphGPUEngine,
    parent_elites: Sequence[bv.BilinearScheme],
    target_shape: Tuple[int, int, int],
    axis: str,
    label_prefix: str,
    lib: Optional[Dict[str, List[bv.BilinearScheme]]] = None,
    seed: int = 9001,
) -> List[Tuple[str, bv.BilinearScheme]]:
    out: List[Tuple[str, bv.BilinearScheme]] = []
    candidates = list(parent_elites)

    if target_shape == (6, 6, 7) and lib is not None:
        rng = random.Random(seed)
        subs_668 = [_pick_min_sub_f3(lib, (3, 3, 4), rng, randomize=False) for _ in range(7)]
        s668 = fg.padded_strassen_compose(3, 3, 3, 3, 4, 4, subs_668, name="meta_6x6x8_f3_203")
        assert s668.rank == 203 and bv.verify_scheme(s668).valid
        res_mix = engine.run_f3_batch(
            [s668], n_walkers=1024, n_steps=15000, plus_period=250, plus_headroom=3, top_k=8, seed=seed
        )
        candidates.extend(res_mix.top_schemes[:6])

    best_proj: List[Tuple[int, int, bv.BilinearScheme]] = []
    seen: set[str] = set()
    for p_sc in candidates:
        proj = fg.best_project_slice(p_sc, axis)
        if proj.shape != target_shape:
            proj = fg.permute_axes(proj, target_shape)
        rep = bv.verify_scheme(proj)
        if rep.valid and rep.ring == "Z3" and rep.canonical_sha256 not in seen:
            seen.add(rep.canonical_sha256)
            best_proj.append((proj.rank, rep.additive_complexity["nnz_total"], proj))

    best_proj.sort(key=lambda x: (x[0], x[1]))
    for idx, (rk, nnz, sc) in enumerate(best_proj[:2]):
        out.append((f"{label_prefix}_R{rk}_{idx}", sc))
        print(f"  [F_3 Meta-Flip Projection] -> {target_shape} `{label_prefix}` R={rk} nnz={nnz}", flush=True)
    return out


# =============================================================================
# 2. High-Utilization SLEEP-DAG-RSI v8 F_3 Combat Loop (Minimal CPU Overhead!)
# =============================================================================


def _score_f3(rank: int, nnz: int, trigger_r: int, ref_start_r: int) -> Tuple[float, List[float], List[int]]:
    score = 100.0 * float(ref_start_r - rank) - 0.01 * float(nnz) / max(rank, 1)
    buckets = [
        float(ref_start_r - 0.5 - rank),
        float(trigger_r + 2.5 - rank),
        float(trigger_r + 1.5 - rank),
        float(trigger_r + 0.5 - rank),
        float(trigger_r - 0.5 - rank),
    ]
    return score, buckets, [1 if v > 0.0 else 0 for v in buckets]


def _xover_short_id(u_id: str, v_id: str) -> str:
    return f"xo_{(hash((u_id, v_id)) & 0xFFFFFF):06x}"


def _load_existing_f3_pool(pool_path: str) -> Dict[str, List[bv.BilinearScheme]]:
    """Load our own previously verified from-scratch Z3 schemes from `f3_pool_phase3.json`."""
    if not os.path.exists(pool_path):
        return {}
    with open(pool_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    out: Dict[str, List[bv.BilinearScheme]] = {}
    for fk, blk in (data.get("formats") or {}).items():
        lst: List[bv.BilinearScheme] = []
        for ent in blk.get("schemes", []):
            sc = fg.f3_packed_to_scheme(
                ent["m"], ent["n"], ent["p"],
                ent["pu"], ent["nu"], ent["pv"], ent["nv"], ent["pw"], ent["nw"],
                name=ent.get("id", f"resume_{fk}"),
            )
            rep = bv.verify_scheme(sc)
            if rep.valid and rep.ring == "Z3":
                lst.append(sc)
        if lst:
            lst.sort(key=lambda s: (s.rank, s.additive_complexity()["nnz_total"]))
            out[fk] = lst
    return out


def run_sleep_dag_f3_target(
    engine: fg.FlipGraphGPUEngine,
    fmt_key: str,
    lib: Dict[str, List[bv.BilinearScheme]],
    time_budget_s: float,
    skill_state: dream_skill.MutableSkillState,
    records_saved: List[Dict[str, Any]],
    pool_by_fmt: Dict[str, Dict[str, Any]],
    pool_path: str,
    extra_meta_seeds: Optional[Sequence[Tuple[str, bv.BilinearScheme]]] = None,
    assert_max_start_r: Optional[int] = None,
    seed: int = 3027,
    seed_override: Optional[List[Tuple[str, bv.BilinearScheme]]] = None,
) -> Tuple[Dict[str, Any], List[bv.BilinearScheme]]:
    """
    Run genuine SLEEP-DAG-RSI v8 F_3 assault on `fmt_key`.
    To maximize RTX 5060 GPU utilization, we verify only `top_schemes[0]` per branch on
    each wave (plus any rank-dropping scheme immediately), and bulk-verify exported elites
    at target completion + sealed BlindVaultGuard certification.
    """
    t0 = time.perf_counter()
    rng = random.Random(seed)
    ref = SOTA_2026[fmt_key]
    m, n, p = ref["m"], ref["n"], ref["p"]
    trigger_r = ref["trigger_f3"]

    if seed_override:
        named_seeds = list(seed_override)
        for lbl, sc in named_seeds:
            rep = bv.verify_scheme(sc)
            if not (rep.valid and rep.ring == "Z3"):
                raise RuntimeError(f"Resume F_3 seed {fmt_key}/{lbl} failed exact Z3 brent_verifier!")
    else:
        named_seeds = build_f3_target_seeds(fmt_key, lib, rng, extra_meta_seeds=extra_meta_seeds)
    ref_start_r = min(sc.rank for _, sc in named_seeds)
    if assert_max_start_r is not None and ref_start_r > assert_max_start_r:
        raise AssertionError(
            f"Starting rank assertion failed for {fmt_key}: got {ref_start_r} > required {assert_max_start_r}"
        )

    for lbl, sc in named_seeds:
        _verify_and_save_f3_record(sc, records_saved, tag=f"init_f3_{fmt_key}_{lbl}")

    problem = dream_tree.Problem()
    problem.name = f"fmm_p3_f3_{fmt_key}"
    init_rep0 = bv.verify_scheme(named_seeds[0][1])
    base_score, base_buckets, base_mask = _score_f3(
        ref_start_r, init_rep0.additive_complexity["nnz_total"], trigger_r, ref_start_r
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
    # Unverified high-speed archive (verified strictly before export / record save!)
    raw_elites: Dict[str, Tuple[int, int, bv.BilinearScheme]] = {}

    for b_idx, (lbl, sc) in enumerate(named_seeds):
        rep = bv.verify_scheme(sc)
        nnz = rep.additive_complexity["nnz_total"]
        sc_val, b_sc, p_mk = _score_f3(sc.rank, nnz, trigger_r, ref_start_r)
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
        raw_elites[rep.canonical_sha256] = (sc.rank, nnz, sc)

    if ref_start_r <= 105:
        total_walkers_wave = 4096
        steps_wave = 70000
    elif ref_start_r <= 195:
        total_walkers_wave = 3072
        steps_wave = 55000
    else:
        total_walkers_wave = 2560
        steps_wave = 50000

    total_flips = 0
    total_plus = 0
    total_reduces = 0
    total_gpu_s = 0.0
    peak_vram = 0.0
    wave = 0
    best_r, best_nnz, best_scheme = min(raw_elites.values(), key=lambda x: (x[0], x[1]))
    decomposer_triggers = 0

    print(
        f"\n[Phase 3 F_3 | {fmt_key}] start_R={ref_start_r} (char0 Q={ref['q']}, ZT={ref['zt']}, trigger<={trigger_r}) | "
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
            plus_headroom = 2 if stalls < 2 else (3 if stalls < 4 else 4)
            base_pp = 320 if ref_start_r >= 180 else 650
            plus_period = max(140, base_pp // (1 + min(stalls, 2)))

            res = engine.run_f3_batch(
                branch_pools[b_idx],
                n_walkers=nw,
                n_steps=steps_wave,
                plus_period=plus_period,
                plus_headroom=plus_headroom,
                target_rank=trigger_r,
                top_k=14,
                seed=seed + wave * 4099 + b_idx * 131,
            )
            total_flips += res.executed_flips
            total_plus += res.executed_plus
            total_reduces += res.executed_reduces
            total_gpu_s += res.elapsed_s
            peak_vram = max(peak_vram, res.peak_vram_mib)

            cand = res.top_schemes[0]
            rep = _verify_and_save_f3_record(cand, records_saved, tag=f"p3_f3_{fmt_key}_w{wave}_b{b_idx}")
            nnz = rep.additive_complexity["nnz_total"]
            sc_val, b_sc, p_mk = _score_f3(cand.rank, nnz, trigger_r, ref_start_r)

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
                        "plus_period": plus_period, "n_walkers": nw, "stage": "gpu_f3_wave",
                    },
                )
            )

            # Archive candidate + top schemes cheaply by hash (verified strictly if rank<=trigger_r or on export!)
            raw_elites[rep.canonical_sha256] = (cand.rank, nnz, cand)
            for sc_i in res.top_schemes[1:8]:
                if sc_i.rank <= trigger_r:
                    r_i = _verify_and_save_f3_record(sc_i, records_saved, tag=f"p3_f3_{fmt_key}_w{wave}_b{b_idx}_top")
                    raw_elites[r_i.canonical_sha256] = (sc_i.rank, r_i.additive_complexity["nnz_total"], sc_i)
                else:
                    h_i = sc_i.canonical_sha256()
                    if h_i not in raw_elites:
                        nz_i = sc_i.additive_complexity()["nnz_total"]
                        raw_elites[h_i] = (sc_i.rank, nz_i, sc_i)

            if (cand.rank, nnz) < (branch_meta[b_idx]["best_r"], branch_meta[b_idx]["best_nnz"]):
                branch_meta[b_idx]["best_r"] = cand.rank
                branch_meta[b_idx]["best_nnz"] = nnz
                branch_meta[b_idx]["stalls"] = 0
                branch_best_node[b_idx] = nid
            else:
                branch_meta[b_idx]["stalls"] += 1

            prev_anchor = branch_pools[b_idx][0]
            branch_pools[b_idx] = res.top_schemes[:10] + ([prev_anchor] if prev_anchor.rank <= res.top_schemes[0].rank else [])

            if (cand.rank, nnz) < (best_r, best_nnz):
                if cand.rank < best_r:
                    wave_improved = True
                    print(
                        f"  [F_3 {fmt_key} | wave {wave} | b{b_idx}] RANK DROP: {best_r} -> {cand.rank} "
                        f"(nnz={nnz}, adds={rep.additive_complexity['naive_additions_total']}, "
                        f"trigger<={trigger_r}, t={time.perf_counter()-t0:.1f}s)",
                        flush=True,
                    )
                best_r, best_nnz, best_scheme = cand.rank, nnz, cand

        # BackwardSubGoalDecomposer every 4 waves if trigger bucket is still unsolved
        if wave % 4 == 0 and best_r > trigger_r and (time.perf_counter() - t0) < (time_budget_s - 6.0):
            decomp = decomposer.decompose([tree], problem)
            if decomp.get("dead_buckets"):
                decomposer_triggers += 1
                worst_b = max(b_ids, key=lambda b: (branch_meta[b]["best_r"], branch_meta[b]["stalls"]))
                if seed_override:
                    fresh_sc = fg.random_monomial_symmetry(rng.choice(named_seeds)[1], rng)
                else:
                    fresh_seeds = build_f3_target_seeds(fmt_key, lib, rng)
                    _, fresh_sc = rng.choice(fresh_seeds)
                branch_pools[worst_b] = [best_scheme, fresh_sc]
                branch_meta[worst_b]["stalls"] = 0

        if wave == 1 or wave_improved or wave % 4 == 0:
            dt_now = time.perf_counter() - t0
            print(
                f"  [F_3 {fmt_key} | wave {wave}] best_R={best_r} (nnz={best_nnz}, target<={trigger_r}) | "
                f"elites={len(raw_elites)} | gpu={total_flips/max(total_gpu_s,1e-9)/1e6:.1f}M/s "
                f"(wall={total_flips/max(dt_now,1e-9)/1e6:.1f}M/s) | plus={total_plus:,} red={total_reduces:,} | "
                f"VRAM={peak_vram:.1f}MiB | t={dt_now:.1f}/{time_budget_s:.0f}s",
                flush=True,
            )

    # Strictly verify top diverse elites for BlindVault + f3_pool_phase3.json export!
    sorted_raw = sorted(raw_elites.values(), key=lambda x: (x[0], x[1]))
    export_limit = 24 if ref_start_r <= 105 else 16
    verified_pool_entries: List[Dict[str, Any]] = []
    verified_elites_objs: List[Tuple[int, int, bv.BilinearScheme]] = []
    rank_hist: Dict[str, int] = {}
    for idx_e, (_, _, sc_e) in enumerate(sorted_raw[: export_limit + 8]):
        rep_e = _verify_and_save_f3_record(sc_e, records_saved, tag=f"p3_f3_export_{fmt_key}")
        rk_e = rep_e.rank
        nz_e = rep_e.additive_complexity["nnz_total"]
        verified_elites_objs.append((rk_e, nz_e, sc_e))
        if len(verified_pool_entries) < export_limit:
            verified_pool_entries.append(
                _pack_f3_pool_entry(sc_e, rep_e, f"{fmt_key}_f3_R{rk_e}_{idx_e:03d}", tag=f"p3_{fmt_key}")
            )
            rank_hist[str(rk_e)] = rank_hist.get(str(rk_e), 0) + 1

    # Flush f3_pool_phase3.json IMMEDIATELY after this target so @fmm-sota can lift right away!
    pool_by_fmt[fmt_key] = {
        "shape": [m, n, p],
        "sota_q": ref["q"],
        "sota_zt": ref["zt"],
        "trigger_f3": trigger_r,
        "best_rank_found": verified_elites_objs[0][0],
        "count": len(verified_pool_entries),
        "rank_histogram": rank_hist,
        "schemes": verified_pool_entries,
    }
    _flush_f3_pool(pool_by_fmt, pool_path)

    # Sealed AutoDreamRunner + BlindVaultGuard(max_calls=1) certification
    top_for_ad = verified_elites_objs[:4]
    vault_scheme_ref: List[Optional[bv.BilinearScheme]] = [None]

    def _make_fn(sc_obj: bv.BilinearScheme):
        return lambda: sc_obj

    def _verify_and_score_c(fn):
        sc_obj = fn()
        rep_c = bv.verify_scheme(sc_obj)
        if not (rep_c.valid and rep_c.ring == "Z3"):
            return False, -1e6, "algorithmic_failure", {"valid": False}
        nnz_c = rep_c.additive_complexity["nnz_total"]
        s_c, b_c, pm_c = _score_f3(sc_obj.rank, nnz_c, trigger_r, ref_start_r)
        return True, s_c, "ok", {
            "rank": sc_obj.rank, "nnz": nnz_c,
            "naive_additions": rep_c.additive_complexity["naive_additions_total"],
            "bucket_scores": b_c, "pass_mask": pm_c, "canonical_sha256": rep_c.canonical_sha256,
        }

    def _xover_synth(u_c: dream_cartridge.BranchCandidate, v_c: dream_cartridge.BranchCandidate):
        res_x = engine.run_f3_batch(
            [u_c.fn(), v_c.fn()], n_walkers=1024, n_steps=12000, plus_period=300, plus_headroom=3,
            top_k=4, seed=seed + 99991,
        )
        cand_x = res_x.top_schemes[0]
        _verify_and_save_f3_record(cand_x, records_saved, tag=f"p3_autodream_xover_{fmt_key}")
        return dream_cartridge.BranchCandidate(
            id=_xover_short_id(u_c.id, v_c.id), branch=99,
            description=f"GPU F_3 crossover of {u_c.id} and {v_c.id}",
            fn=_make_fn(cand_x), parent_id=u_c.id,
        )

    def _vault_cert(fn):
        sc_obj = fn()
        vault_scheme_ref[0] = sc_obj
        rep_v = bv.verify_scheme(sc_obj)
        return {
            "valid": rep_v.valid, "brent_ok": rep_v.brent_ok, "basis_ok": rep_v.basis_ok,
            "ring": rep_v.ring, "rank": rep_v.rank,
            "nnz": rep_v.additive_complexity["nnz_total"],
            "naive_additions": rep_v.additive_complexity["naive_additions_total"],
            "canonical_sha256": rep_v.canonical_sha256,
        }

    init_branches = [
        dream_cartridge.BranchCandidate(
            id=f"e{idx}_R{rk}", branch=idx,
            description=f"GPU F_3 elite {fmt_key} R={rk} nnz={nz}",
            fn=_make_fn(sc_e),
        )
        for idx, (rk, nz, sc_e) in enumerate(top_for_ad[:3])
    ]
    cartridge = dream_cartridge.TaskCartridge(
        task_name=f"fmm_p3_f3_{fmt_key}",
        description=f"Phase 3 From-Scratch F_3 Assault <{m},{n},{p}> (trigger<={trigger_r})",
        baseline_fn=_make_fn(named_seeds[0][1]),
        verify_and_score=_verify_and_score_c,
        initial_branches=init_branches,
        crossover_synthesizer=_xover_synth,
        sentinel_noise_regime="deterministic",
        blind_vault_evaluator=_vault_cert,
    )
    try:
        ad_report = dream_cartridge.AutoDreamRunner(cartridge).run()
        vault = ad_report.get("blind_vault_report") or {}
        champ_score_val = float(ad_report["champion_score"])
    except MemoryError:
        # Guard against engine.py replay string explosion when all branches have tied scores:
        # fall back cleanly to sealed BlindVaultGuard(max_calls=1) certification on our top elite!
        bvg = dream_cartridge.BlindVaultGuard(_vault_cert, max_calls=1)
        vault = bvg.evaluate(_make_fn(best_scheme))
        champ_score_val, _, _ = _score_f3(best_r, best_nnz, trigger_r, ref_start_r)
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
        "ring": "Z3",
        "shape": [m, n, p],
        "sota_q": ref["q"],
        "sota_zt": ref["zt"],
        "trigger_f3": trigger_r,
        "start_rank": ref_start_r,
        "best_rank": final_rank,
        "best_nnz": final_nnz,
        "best_naive_additions": final_rep.additive_complexity["naive_additions_total"],
        "best_sha256": final_rep.canonical_sha256,
        "hit_trigger_f3": final_rank <= trigger_r,
        "beat_sota_q": final_rank < ref["q"],
        "waves": wave,
        "distinct_elites": len(raw_elites),
        "exported_pool_count": len(verified_pool_entries),
        "decomposer_triggers": decomposer_triggers,
        "elapsed_s": round(dt_total, 2),
        "gpu_kernel_s": round(total_gpu_s, 2),
        "executed_flips": total_flips,
        "executed_plus": total_plus,
        "executed_reduces": total_reduces,
        "gpu_flips_per_s": round(total_flips / max(total_gpu_s, 1e-9), 1),
        "wall_flips_per_s": round(total_flips / max(dt_total, 1e-9), 1),
        "peak_vram_mib": round(peak_vram, 2),
        "autodream_champion_score": round(champ_score_val, 4),
        "blind_vault_verified": bool(vault.get("valid", False)),
        "blind_vault_rank": vault.get("rank"),
        "sentinel_gate_passed": bool(sentinel_ok.get("passed", True)),
        "tree_node_count": len(tree.nodes),
    }
    elites_out = [sc for _, _, sc in verified_elites_objs[:12]]
    return report, elites_out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 3 RTX 5060 From-Scratch F_3 World-Record Assault.")
    parser.add_argument("--max-minutes", type=float, default=14.0, help="Hard wall-clock cap in minutes.")
    parser.add_argument("--seed", type=int, default=3027, help="Master PRNG seed.")
    parser.add_argument(
        "--resume-from-pool",
        action="store_true",
        help="Warm-start from our own previously generated verified `f3_pool_phase3.json` schemes.",
    )
    args = parser.parse_args(argv)

    t_start = time.perf_counter()
    hard_deadline = t_start + max(args.max_minutes * 60.0 - 15.0, 60.0)
    engine = fg.FlipGraphGPUEngine()
    os.makedirs(_ARTIFACTS_DIR, exist_ok=True)
    pool_path = os.path.join(_ARTIFACTS_DIR, "f3_pool_phase3.json")
    records_saved: List[Dict[str, Any]] = []
    pool_by_fmt: Dict[str, Dict[str, Any]] = {}

    resumed_pool: Dict[str, List[bv.BilinearScheme]] = {}
    if args.resume_from_pool:
        resumed_pool = _load_existing_f3_pool(pool_path)
        print(
            f"[Resume F_3] Loaded {sum(len(v) for v in resumed_pool.values())} own verified Z3 schemes "
            f"across {len(resumed_pool)} formats from {pool_path}",
            flush=True,
        )

    if resumed_pool and all(k in resumed_pool for k in ("4x4x4", "5x8x8", "7x7x7", "6x6x7", "5x7x8", "4x4x5", "4x5x5", "4x5x6")):
        lib: Dict[str, List[bv.BilinearScheme]] = {}
        lib_ranks_summary = {
            "2x2x2": 7, "2x2x3": 11, "4x4x4": 49, "3x3x3": 23, "2x3x3": 15,
            "2x4x4": 26, "2x3x4": 20, "3x3x4": 29, "3x4x4": 38, "4x5x2": 33, "4x5x3": 51,
        }
        s588_230 = resumed_pool["5x8x8"][0]
        s777_250 = resumed_pool["7x7x7"][0]
    else:
        # 1. Fast Stage 0: strict min-rank from-scratch Z_T primitive library -> cast to F_3 (ZERO external data!)
        lib = build_from_scratch_f3_library(engine, seed=2027)
        lib_ranks_summary = {k: v[0].rank for k, v in lib.items()}
        rng_b = random.Random(args.seed + 1)
        s588_230 = build_f3_target_seeds("5x8x8", lib, rng_b)[0][1]
        s777_250 = build_f3_target_seeds("7x7x7", lib, rng_b)[0][1]

    # 2. Mandatory High-R F_3 Benchmark at EXACT R=230 (5x8x8) and R=250 (7x7x7)
    assert s588_230.rank == 230 and bv.verify_scheme(s588_230).valid
    assert s777_250.rank == 250 and bv.verify_scheme(s777_250).valid

    b_230 = engine.run_f3_batch([s588_230], n_walkers=2048, n_steps=25000, plus_period=400, plus_headroom=2, top_k=4, seed=args.seed + 2)
    b_250 = engine.run_f3_batch([s777_250], n_walkers=2048, n_steps=25000, plus_period=400, plus_headroom=2, top_k=4, seed=args.seed + 3)
    assert bv.verify_scheme(b_230.top_schemes[0]).valid and bv.verify_scheme(b_250.top_schemes[0]).valid
    _verify_and_save_f3_record(b_230.top_schemes[0], records_saved, tag="bench_f3_5x8x8")
    _verify_and_save_f3_record(b_250.top_schemes[0], records_saved, tag="bench_f3_7x7x7")

    high_r_bench = {
        "5x8x8_R230_F3": {
            "start_rank": 230, "best_rank": b_230.best_rank,
            "flips_per_s": round(b_230.executed_flips_per_s, 1),
            "flips_per_s_M": round(b_230.executed_flips_per_s / 1e6, 2),
            "phase1_zt_M": 1.28,
            "phase2_zt_M": 17.65,
            "speedup_vs_phase1": round((b_230.executed_flips_per_s / 1e6) / 1.28, 2),
            "peak_vram_mib": round(b_230.peak_vram_mib, 2),
        },
        "7x7x7_R250_F3": {
            "start_rank": 250, "best_rank": b_250.best_rank,
            "flips_per_s": round(b_250.executed_flips_per_s, 1),
            "flips_per_s_M": round(b_250.executed_flips_per_s / 1e6, 2),
            "phase1_zt_M": 1.04,
            "phase2_zt_M": 17.60,
            "speedup_vs_phase1": round((b_250.executed_flips_per_s / 1e6) / 1.04, 2),
            "peak_vram_mib": round(b_250.peak_vram_mib, 2),
        },
    }
    print(
        f"[High-R F_3 Benchmark] 5x8x8 R=230: {high_r_bench['5x8x8_R230_F3']['flips_per_s_M']} M flips/s "
        f"({high_r_bench['5x8x8_R230_F3']['speedup_vs_phase1']}x vs Phase 1!) | "
        f"7x7x7 R=250: {high_r_bench['7x7x7_R250_F3']['flips_per_s_M']} M flips/s "
        f"({high_r_bench['7x7x7_R250_F3']['speedup_vs_phase1']}x vs Phase 1!) | VRAM={b_250.peak_vram_mib:.1f} MiB",
        flush=True,
    )

    rem_s = max(hard_deadline - time.perf_counter(), 90.0)
    if args.resume_from_pool and resumed_pool:
        # Give extra weight to 4x4x5/4x5x5/4x5x6 (which got only 5-15s in run 1) plus fresh bursts on high-R!
        shares = {
            "4x4x4": 0.08,
            "5x8x8": 0.14,
            "7x7x7": 0.14,
            "6x6x7": 0.10,
            "5x7x8": 0.10,
            "4x4x5": 0.16,
            "4x5x5": 0.16,
            "4x5x6": 0.12,
        }
    else:
        shares = {
            "4x4x4": 0.045,
            "5x8x8": 0.245,
            "7x7x7": 0.245,
            "6x6x7": 0.165,
            "5x7x8": 0.135,
            "4x4x5": 0.065,
            "4x5x5": 0.065,
            "4x5x6": 0.035,
        }

    def _cap_budget(fk: str) -> float:
        left = max(hard_deadline - time.perf_counter(), 5.0)
        return min(rem_s * shares[fk], left)

    def _resume_override(fk: str) -> Optional[List[Tuple[str, bv.BilinearScheme]]]:
        if not (args.resume_from_pool and fk in resumed_pool):
            return None
        pool_sc = resumed_pool[fk]
        picks = [pool_sc[0]]
        if len(pool_sc) > 1:
            picks.append(pool_sc[len(pool_sc) // 2])
        if len(pool_sc) > 2:
            picks.append(pool_sc[-1])
        m_d, n_d, p_d = SOTA_2026[fk]["m"], SOTA_2026[fk]["n"], SOTA_2026[fk]["p"]
        out_o = [(f"resumed_{fk}_{i}_R{sc.rank}", sc) for i, sc in enumerate(picks)]
        if m_d * n_d * p_d <= 125:
            out_o.append((f"naive_{fk}", bv.make_naive_scheme(m_d, n_d, p_d, "Z3")))
        return out_o

    skill_state = dream_skill.MutableSkillState(task="fmm_phase3_f3_assault")
    f3_reports: Dict[str, Any] = {}

    # (0) 4x4x4 calibration (49 -> 48?)
    rep_444, _ = run_sleep_dag_f3_target(
        engine, "4x4x4", lib, time_budget_s=_cap_budget("4x4x4"),
        skill_state=skill_state, records_saved=records_saved,
        pool_by_fmt=pool_by_fmt, pool_path=pool_path, assert_max_start_r=49, seed=args.seed + 50,
        seed_override=_resume_override("4x4x4"),
    )
    f3_reports["4x4x4"] = rep_444

    # (1) 5x8x8 (start=230 -> hunt <=229!)
    rep_588, elites_588 = run_sleep_dag_f3_target(
        engine, "5x8x8", lib, time_budget_s=_cap_budget("5x8x8"),
        skill_state=skill_state, records_saved=records_saved,
        pool_by_fmt=pool_by_fmt, pool_path=pool_path, assert_max_start_r=230, seed=args.seed + 100,
        seed_override=_resume_override("5x8x8"),
    )
    f3_reports["5x8x8"] = rep_588

    # (2) 7x7x7 (start=250 -> hunt <=248!)
    rep_777, elites_777 = run_sleep_dag_f3_target(
        engine, "7x7x7", lib, time_budget_s=_cap_budget("7x7x7"),
        skill_state=skill_state, records_saved=records_saved,
        pool_by_fmt=pool_by_fmt, pool_path=pool_path, assert_max_start_r=250, seed=args.seed + 200,
        seed_override=_resume_override("7x7x7"),
    )
    f3_reports["7x7x7"] = rep_777

    # (3) 6x6x7 (start=185 + meta-flip projections -> hunt <=182!)
    meta_667 = generate_meta_flip_seeds_f3(
        engine, [], (6, 6, 7), axis="p", label_prefix="meta_f3_6x6x8", lib=lib if lib else None, seed=args.seed + 401
    ) if lib else []
    if elites_777:
        p677 = fg.best_project_slice(elites_777[0], "m")
        p667 = fg.best_project_slice(p677, "n")
        if bv.verify_scheme(p667).valid:
            meta_667.append((f"meta_f3_7x7x7_R{p667.rank}", p667))
            print(f"  [F_3 Meta-Flip Projection] 7x7x7 -> 6x6x7 R={p667.rank}", flush=True)
    ov_667 = _resume_override("6x6x7")
    if ov_667 is not None:
        ov_667.extend(meta_667[:1])
    rep_667, _ = run_sleep_dag_f3_target(
        engine, "6x6x7", lib, time_budget_s=_cap_budget("6x6x7"),
        skill_state=skill_state, records_saved=records_saved,
        pool_by_fmt=pool_by_fmt, pool_path=pool_path, extra_meta_seeds=meta_667,
        assert_max_start_r=185, seed=args.seed + 300, seed_override=ov_667,
    )
    f3_reports["6x6x7"] = rep_667

    # (4) 5x7x8 (start=206 + meta-flip projections from 5x8x8 -> hunt <=203!)
    meta_578 = generate_meta_flip_seeds_f3(
        engine, elites_588, (5, 7, 8), axis="n", label_prefix="meta_f3_5x8x8", seed=args.seed + 301
    )
    ov_578 = _resume_override("5x7x8")
    if ov_578 is not None:
        ov_578.extend(meta_578[:1])
    rep_578, _ = run_sleep_dag_f3_target(
        engine, "5x7x8", lib, time_budget_s=_cap_budget("5x7x8"),
        skill_state=skill_state, records_saved=records_saved,
        pool_by_fmt=pool_by_fmt, pool_path=pool_path, extra_meta_seeds=meta_578,
        assert_max_start_r=206, seed=args.seed + 400, seed_override=ov_578,
    )
    f3_reports["5x7x8"] = rep_578

    # (5) 4x4x5 (start<=64 -> hunt <=60!)
    rep_445, _ = run_sleep_dag_f3_target(
        engine, "4x4x5", lib, time_budget_s=_cap_budget("4x4x5"),
        skill_state=skill_state, records_saved=records_saved,
        pool_by_fmt=pool_by_fmt, pool_path=pool_path, seed=args.seed + 500,
        seed_override=_resume_override("4x4x5"),
    )
    f3_reports["4x4x5"] = rep_445

    # (6) 4x5x5 (start<=81 -> hunt <=75!)
    rep_455, _ = run_sleep_dag_f3_target(
        engine, "4x5x5", lib, time_budget_s=_cap_budget("4x5x5"),
        skill_state=skill_state, records_saved=records_saved,
        pool_by_fmt=pool_by_fmt, pool_path=pool_path, seed=args.seed + 600,
        seed_override=_resume_override("4x5x5"),
    )
    f3_reports["4x5x5"] = rep_455

    # (7) 4x5x6 (start<=93 -> hunt <=89!)
    rep_456, _ = run_sleep_dag_f3_target(
        engine, "4x5x6", lib, time_budget_s=_cap_budget("4x5x6"),
        skill_state=skill_state, records_saved=records_saved,
        pool_by_fmt=pool_by_fmt, pool_path=pool_path, seed=args.seed + 700,
        seed_override=_resume_override("4x5x6"),
    )
    f3_reports["4x5x6"] = rep_456

    _flush_f3_pool(pool_by_fmt, pool_path)
    total_wall = round(time.perf_counter() - t_start, 2)
    total_f3_flips = sum(r["executed_flips"] for r in f3_reports.values())
    peak_vram_all = max(
        [r["peak_vram_mib"] for r in f3_reports.values()]
        + [high_r_bench["5x8x8_R230_F3"]["peak_vram_mib"], high_r_bench["7x7x7_R250_F3"]["peak_vram_mib"]]
    )

    summary = {
        "phase": "phase3_final_f3_gpu_assault",
        "device": engine.backend.device_name,
        "arch": engine.backend.arch,
        "ring": "Z3",
        "provenance": "100%_from_scratch_zero_external_seeds",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "max_minutes_cap": args.max_minutes,
        "resumed_from_own_stage1_pool": bool(args.resume_from_pool and resumed_pool),
        "total_wall_s": total_wall,
        "peak_vram_mib": round(peak_vram_all, 2),
        "total_f3_flips": total_f3_flips,
        "block_library_ranks": lib_ranks_summary,
        "high_r_benchmark": high_r_bench,
        "f3_pool_path": pool_path,
        "f3_pool_total_schemes": sum(v["count"] for v in pool_by_fmt.values()),
        "f3_targets": f3_reports,
        "skill_state_summary": {
            "round_index": skill_state.round_index,
            "invariants_count": len(skill_state.best_invariants),
            "unsolved_buckets_count": len(skill_state.unsolved_buckets),
            "serialized_bytes": skill_state.serialized_byte_size(),
        },
        "records_saved": records_saved,
    }
    out_path = os.path.join(_ARTIFACTS_DIR, "phase3_f3_summary.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"\n[Phase 3 F_3 Complete in {total_wall:.1f}s] Summary -> {out_path}", flush=True)
    print(
        f"  Total F_3 flips: {total_f3_flips:,} | Peak VRAM: {peak_vram_all:.1f} MiB | "
        f"Pool exported: {summary['f3_pool_total_schemes']} verified Z3 schemes -> {pool_path}",
        flush=True,
    )
    for fk, r in f3_reports.items():
        print(
            f"  F_3 {fk}: start={r['start_rank']} -> best={r['best_rank']} (trigger<={r['trigger_f3']}, "
            f"hit={r['hit_trigger_f3']}) | nnz={r['best_nnz']}, adds={r['best_naive_additions']}, "
            f"gpu={r['gpu_flips_per_s']/1e6:.1f}M/s, vault={r['blind_vault_verified']}",
            flush=True,
        )
    print(f"  NEW F_3 RECORD HITS SAVED: {len(records_saved)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

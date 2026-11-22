"""
Phase 3 Combat Campaign: Direct Cross-Intersecting Hypergraph (F_+, F_-) Evolution
via SLEEP-DAG-RSI v8 on RTX 5060 across d = 10..16 to discover a genuinely new
combinatorial object beating AG(2,3) (C_71 > 6.52184571092304657566).

Pipeline:
  - Step 0 (Mandatory Tracer #1): Cold-start d=9 AG(2,3) rediscovery gate (>= 6.521845).
  - Cycle 1 (SLEEP-DAG-RSI AutoDreamRunner): Cold-start Mode I + Mode II hunt on d=10,11,12,13
    with min_used_verts = d (preventing collapse to lifted AG(2,3)), post-discovery fine polish,
    and Round 3 hypergraph CROSSOVER(H1, H2).
  - Cycle 2 (SLEEP-DAG-RSI AutoDreamRunner): High-d hunt on d=14,15,16 + non-blocking
    auto-ingest of `phase3_catalog_seeds.json` + carried MutableSkillState.
  - Cycle 3 (SLEEP-DAG-RSI AutoDreamRunner): Deep focused assault on the top-3 dimensions
    d* in [10..16] + deep fine polish + single-shot BlindVaultGuard exact certification!

Strict Zero-Gaming: ZERO external truth tables; distinguishes genuinely new non-AG(2,3)
objects (essential_vars >= 10) from lifted AG(2,3) clones; validates every champion via
exact_verifier.py before persisting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SRC_DIR, "..", ".."))
_DREAM_DIR = os.path.join(_REPO_ROOT, "dream_rsi_scripts")

for _p in (_SRC_DIR, _DREAM_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cartridge as dream_cartridge  # noqa: E402
import engine as dream_engine  # noqa: E402
import exact_verifier  # noqa: E402
from c71_cartridge import (  # noqa: E402
    NUM2026_STRICT_BAR,
    evaluate_fei_candidate_fp64,
    gpu_delta_steepest_ascent,
    optimize_mixed_12_hosts_on_core,
    run_sleep_dag_diagnostics,
)
from fwht_gpu import evaluate_subcube_dealiased_exact_fp64, get_device  # noqa: E402
from hypergraph_gpu_engine import (  # noqa: E402
    HypergraphGenome,
    count_essential_vars,
    crossover_hypergraphs,
    detect_isomorphism_perm,
    evolve_hypergraphs_gpu,
    extract_hypergraph_from_g0,
    is_cross_intersecting,
    reduce_to_antichain,
    run_sanity_d9,
    verify_hypergraph_candidate_fp64,
)

PHASE3_THRESHOLDS = (6.100, 6.250, 6.400, 6.500, 6.522)


@dataclass
class Phase3Champion:
    label: str
    d: int
    essential_vars: int
    g0: np.ndarray
    subcubes: List[Dict[str, int]]
    F_pos: List[int]
    F_neg: List[int]
    mode: str
    perm: Optional[List[int]]
    OT: float
    H: float
    I: float
    n_effective: int
    source: str
    polished: bool = False

    def to_genome(self) -> HypergraphGenome:
        return HypergraphGenome(
            d=self.d,
            F_pos=list(self.F_pos),
            F_neg=list(self.F_neg),
            mode=self.mode,
            perm=list(self.perm) if self.perm is not None else None,
        )


def _load_catalog_seeds_if_present(art_dir: Path) -> Dict[int, List[Tuple[str, HypergraphGenome]]]:
    """
    Non-blocking loader for `phase3_catalog_seeds.json` produced by @fei-literature.
    Groups verified cross-intersecting HypergraphGenomes by dimension d in [10..16].
    """
    path = art_dir / "phase3_catalog_seeds.json"
    by_d: Dict[int, List[Tuple[str, HypergraphGenome]]] = {}
    if not path.is_file():
        return by_d
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw if isinstance(raw, list) else raw.get("seeds", raw.get("candidates", []))
        for item in items:
            if not isinstance(item, dict):
                continue
            d = int(item.get("d", 0))
            fp = item.get("F_plus", item.get("F_pos"))
            fn = item.get("F_minus", item.get("F_neg"))
            if d < 10 or d > 16 or not fp or not fn:
                continue
            fp_c = reduce_to_antichain([int(x) for x in fp])
            fn_c = reduce_to_antichain([int(x) for x in fn])
            if not fp_c or not fn_c or not is_cross_intersecting(fp_c, fn_c):
                continue
            perm = detect_isomorphism_perm(d, fp_c, fn_c)
            gm = HypergraphGenome(
                d=d, F_pos=fp_c, F_neg=fn_c, mode="mode1" if perm is not None else "mode2", perm=perm
            )
            by_d.setdefault(d, []).append((str(item.get("label", f"cat_d{d}")), gm))
    except Exception:
        return {}
    return by_d


def _rep_to_champion(
    rep: Dict[str, Any], label: str, source: str, polished: bool = False
) -> Phase3Champion:
    return Phase3Champion(
        label=label,
        d=int(rep["d"]),
        essential_vars=int(rep["essential_vars"]),
        g0=np.asarray(rep["g0"], dtype=np.int8).copy(),
        subcubes=[{"fixed_mask": int(c["fixed_mask"]), "fixed_val": int(c["fixed_val"])} for c in rep["subcubes"]],
        F_pos=[int(x) for x in rep["F_pos"]],
        F_neg=[int(x) for x in rep["F_neg"]],
        mode=str(rep.get("mode", "mode2")),
        perm=list(rep["perm"]) if rep.get("perm") is not None else None,
        OT=float(rep["OT"]),
        H=float(rep["H"]),
        I=float(rep["I"]),
        n_effective=int(rep["n_eff"]),
        source=source,
        polished=polished,
    )


def _fine_polish_champion(
    champ: Phase3Champion,
    device: torch.device,
    rng: np.random.Generator,
    quick: bool = False,
) -> Phase3Champion:
    """
    Post-discovery fine polish on a newly discovered hypergraph core g0:
      - For d <= 13: runs exact 1+2-cell subcube host ILS (`optimize_mixed_12_hosts_on_core`)
      - For d <= 16: runs exact GPU analytic delta steepest ascent (`gpu_delta_steepest_ascent`)
    Strictly preserves balance, full monotonicity, and essential_vars >= 10!
    """
    best = champ
    if champ.d <= 13:
        g_p, c_p = optimize_mixed_12_hosts_on_core(
            champ.g0, champ.d, device, rng,
            max_rounds=15 if quick else 45,
            batch_limit=192 if quick else 384,
            allow_sa_kicks=True,
        )
        if c_p is not None and c_p.OT > best.OT + 1e-9:
            ess = count_essential_vars(c_p.g0, champ.d)
            if ess >= min(10, champ.d):
                fp_new, fn_new = extract_hypergraph_from_g0(c_p.g0, champ.d)
                perm_new = detect_isomorphism_perm(champ.d, fp_new, fn_new)
                best = Phase3Champion(
                    label=f"{champ.label}_pol",
                    d=champ.d,
                    essential_vars=ess,
                    g0=c_p.g0.copy(),
                    subcubes=[{"fixed_mask": int(c["fixed_mask"]), "fixed_val": int(c["fixed_val"])} for c in c_p.subcubes],
                    F_pos=fp_new,
                    F_neg=fn_new,
                    mode="mode1" if perm_new is not None else "mode2",
                    perm=perm_new,
                    OT=float(c_p.OT),
                    H=float(c_p.H),
                    I=float(c_p.I),
                    n_effective=int(c_p.n_effective),
                    source=f"{champ.source}_host12_pol",
                    polished=True,
                )
    elif champ.d <= 16:
        g_p, ot_p, _ = gpu_delta_steepest_ascent(
            best.g0, champ.d, device, rng,
            max_iters=15 if quick else 40,
            batch_candidates=96 if champ.d >= 15 else 160,
            allow_antichain=False, allow_tunneling=True, max_tunnels=3,
        )
        if ot_p > best.OT + 1e-9:
            r64 = evaluate_subcube_dealiased_exact_fp64(g_p, champ.d, device=device)
            ess = count_essential_vars(g_p, champ.d)
            if r64["balanced"] and r64["monotone_full"] and ess >= min(10, champ.d):
                fp_new, fn_new = extract_hypergraph_from_g0(g_p, champ.d)
                perm_new = detect_isomorphism_perm(champ.d, fp_new, fn_new)
                best = Phase3Champion(
                    label=f"{champ.label}_pol",
                    d=champ.d,
                    essential_vars=ess,
                    g0=g_p.copy(),
                    subcubes=[{"fixed_mask": int(c["fixed_mask"]), "fixed_val": int(c["fixed_val"])} for c in r64["subcubes"]],
                    F_pos=fp_new,
                    F_neg=fn_new,
                    mode="mode1" if perm_new is not None else "mode2",
                    perm=perm_new,
                    OT=float(r64["OT"]),
                    H=float(r64["H"]),
                    I=float(r64["I"]),
                    n_effective=int(r64["n_eff"]),
                    source=f"{champ.source}_delta_pol",
                    polished=True,
                )
    return best


def _hunt_dimension(
    d: int,
    device: torch.device,
    rng: np.random.Generator,
    seed_base: int,
    n_chains: int,
    steps_m1: int,
    steps_m2: int,
    k_profiles: Sequence[Tuple[int, ...]],
    catalog_seeds: Sequence[HypergraphGenome],
    prior_elites: Sequence[HypergraphGenome],
    quick: bool,
    deadline_ts: float,
) -> Optional[Phase3Champion]:
    """
    Hunts genuinely new (essential_vars >= 10) cross-intersecting hypergraphs at dimension d
    across Mode I (isomorphic, exact balance) and Mode II (free asymmetric, seeded from best
    Mode I hits + catalog seeds + cold starts), followed by fine monotone polish!
    """
    best_champ: Optional[Phase3Champion] = None
    mode1_elites: List[HypergraphGenome] = []
    max_e = min(44, max(16, int(d * 2.2)))

    # 0. Evaluate raw catalog seeds at dimension d directly (in case an algebraic seed is already strong!)
    for idx_s, sg in enumerate(catalog_seeds):
        rep_s = verify_hypergraph_candidate_fp64(sg, device, try_gap_balance=True, try_subcube_repair=True)
        if rep_s is not None and int(rep_s["essential_vars"]) >= 10:
            ch_s = _rep_to_champion(rep_s, label=f"Cat_d{d}_{idx_s}", source="catalog_seed")
            if ch_s.mode == "mode1" and ch_s.perm is not None:
                mode1_elites.append(ch_s.to_genome())
            if best_champ is None or ch_s.OT > best_champ.OT:
                best_champ = ch_s

    # 1. Mode I hunt across k_profiles (mix of cold starts + isomorphic catalog/prior seeds)
    m1_seeds = [g for g in prior_elites if g.perm is not None] + [g for g in catalog_seeds if g.perm is not None]
    for p_i, kp in enumerate(k_profiles):
        if time.perf_counter() >= deadline_ts:
            break
        res1 = evolve_hypergraphs_gpu(
            d=d, mode="mode1", k_choices=kp,
            n_chains=n_chains, steps=steps_m1,
            t_start=0.25, t_end=0.002,
            seed=seed_base + d * 101 + p_i * 17,
            device=device,
            seed_genomes=m1_seeds[:6],
            max_edges=max_e,
            min_used_verts=d,
        )
        v_non = res1["verified_non_ag23"]
        if v_non is not None:
            ch = _rep_to_champion(v_non, label=f"H_d{d}_m1_k{''.join(map(str, kp))}", source=f"mode1_k{kp}")
            mode1_elites.append(ch.to_genome())
            if best_champ is None or ch.OT > best_champ.OT:
                best_champ = ch

    # 2. Mode II hunt (seeded from top Mode I elites + catalog seeds + prior elites + cold starts!)
    m2_seeds = list(mode1_elites[:4]) + list(catalog_seeds[:6]) + list(prior_elites[:4])
    for p_i, kp in enumerate(k_profiles):
        if time.perf_counter() >= deadline_ts:
            break
        res2 = evolve_hypergraphs_gpu(
            d=d, mode="mode2", k_choices=kp,
            n_chains=n_chains, steps=steps_m2,
            t_start=0.25, t_end=0.002,
            seed=seed_base + d * 211 + p_i * 29,
            device=device,
            seed_genomes=m2_seeds,
            max_edges=max_e,
            min_used_verts=d,
        )
        v_non = res2["verified_non_ag23"]
        if v_non is not None:
            ch = _rep_to_champion(v_non, label=f"H_d{d}_m2_k{''.join(map(str, kp))}", source=f"mode2_k{kp}")
            if best_champ is None or ch.OT > best_champ.OT:
                best_champ = ch

    if best_champ is not None and time.perf_counter() < deadline_ts:
        best_champ = _fine_polish_champion(best_champ, device, rng, quick=quick)

    return best_champ


class Phase3HypergraphCartridgeBuilder:
    """Genuine SLEEP-DAG-RSI v8 TaskCartridge wrapper for Phase 3 Hypergraph Evolution."""

    def __init__(
        self,
        device: torch.device,
        baseline_champ: Phase3Champion,
        rng: np.random.Generator,
        cycle_idx: int,
        prior_skill_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.device = device
        self.baseline = baseline_champ
        self.rng = rng
        self.cycle_idx = cycle_idx
        self.prior_skill_state = prior_skill_state
        self.registry: Dict[str, Phase3Champion] = {"baseline": baseline_champ}
        self.best_non_ag23: Phase3Champion = baseline_champ

    def _register(self, ch: Phase3Champion) -> None:
        self.registry[ch.label] = ch
        if ch.essential_vars >= 10 and ch.OT > self.best_non_ag23.OT + 1e-12:
            self.best_non_ag23 = ch

    def verify_and_score(self, fn: Callable[[], Optional[Phase3Champion]]) -> Tuple[bool, float, str, Dict[str, Any]]:
        ch = fn()
        if ch is None:
            return False, 0.0, "algorithmic_failure", {"error": "No valid balanced monotone hypergraph found"}
        # Strict float64 + essential_vars re-verification
        r64 = evaluate_subcube_dealiased_exact_fp64(ch.g0, ch.d, device=self.device)
        if not (r64["balanced"] and r64["monotone_full"]):
            return False, 0.0, "algorithmic_failure", {"error": "Failed strict float64 monotonicity/balance"}
        ch.OT = float(r64["OT"])
        ch.H = float(r64["H"])
        ch.I = float(r64["I"])
        ch.n_effective = int(r64["n_eff"])
        ch.essential_vars = count_essential_vars(ch.g0, ch.d)
        self._register(ch)

        # Bucket scores across stepping stones + influence ridge + sparsity/gap structure
        margins = [round(ch.OT - thr, 6) for thr in PHASE3_THRESHOLDS]
        margins.append(round(0.08 - abs(ch.I - 2.04), 6))
        margins.append(round(float(ch.essential_vars - 10) / 6.0, 6))
        margins.append(round(1.0 - float(len(ch.subcubes)) / 64.0, 6))
        pass_mask = [1 if m > 0.0 else 0 for m in margins]
        info = {
            "summary": f"{ch.label} d={ch.d} ess={ch.essential_vars} n_eff={ch.n_effective} I={ch.I:.6f} L={ch.OT:.8f}",
            "d": ch.d,
            "essential_vars": ch.essential_vars,
            "k_subcubes": len(ch.subcubes),
            "n_effective": ch.n_effective,
            "I": ch.I,
            "H": ch.H,
            "OT": ch.OT,
            "pass_mask": pass_mask,
            "bucket_scores": margins,
        }
        return True, float(ch.OT), "ok", info

    def crossover_synthesizer(
        self,
        u_cand: dream_cartridge.BranchCandidate,
        v_cand: dream_cartridge.BranchCandidate,
    ) -> Optional[dream_cartridge.BranchCandidate]:
        """Hypergraph CROSSOVER(H1, H2) + Mode II GPU evolution polish."""
        c_u = self.registry.get(u_cand.id)
        c_v = self.registry.get(v_cand.id)
        if c_u is None or c_v is None:
            pool = sorted([c for c in self.registry.values() if c.essential_vars >= 10], key=lambda x: x.OT, reverse=True)
            if len(pool) < 2:
                return None
            c_u, c_v = pool[0], pool[1]
        xid = f"c{self.cycle_idx}_hxover_{u_cand.id}_{v_cand.id}"

        def _run_hxover() -> Optional[Phase3Champion]:
            d_x = max(c_u.d, c_v.d)
            if d_x > 16:
                return None
            seeds_x = [
                crossover_hypergraphs(c_u.to_genome(), c_v.to_genome(), self.rng, k_choices=(3, 4))
                for _ in range(8)
            ]
            res = evolve_hypergraphs_gpu(
                d=d_x, mode="mode2", k_choices=(3, 4),
                n_chains=48 if d_x <= 13 else 32,
                steps=160 if d_x <= 13 else 100,
                t_start=0.20, t_end=0.002,
                seed=int(self.rng.integers(1, 1_000_000)),
                device=self.device,
                seed_genomes=seeds_x,
                max_edges=min(28, int(d_x * 1.6)),
                min_used_verts=d_x,
            )
            v_non = res["verified_non_ag23"]
            if v_non is None:
                return None
            return _rep_to_champion(v_non, label=xid, source="dag_hypergraph_crossover")

        return dream_cartridge.BranchCandidate(
            id=xid,
            branch=max(u_cand.branch, v_cand.branch) + 1,
            parent_id=u_cand.id,
            attempt=max(u_cand.attempt, v_cand.attempt) + 1,
            description=f"Hypergraph CROSSOVER({c_u.label}, {c_v.label}) + GPU Mode II evolution",
            fn=_run_hxover,
        )

    def blind_vault_verify(self, candidate_fn: Callable[[], Optional[Phase3Champion]]) -> Dict[str, Any]:
        ch = self.best_non_ag23
        rep = exact_verifier.verify_dealiased_core(
            ch.g0.tolist(),
            [{"fixed_mask": int(c["fixed_mask"]), "fixed_val": int(c["fixed_val"])} for c in ch.subcubes],
            require_monotone=True,
        )
        return {
            "label": ch.label,
            "d": ch.d,
            "essential_vars": ch.essential_vars,
            "n_effective": rep["n_effective"],
            "I_exact_str": rep["I_exact_str"],
            "L_cert": rep["L_cert"],
            "U_cert": rep["U_cert"],
            "is_monotone": rep["is_monotone"],
            "exact_balance": rep["exact_balance"],
            "parseval_verified": rep["parseval_verified"],
            "beats_num2026": rep["beats_num2026"],
            "sha256": rep["sha256"],
        }

    def build_cartridge(self, branches: List[dream_cartridge.BranchCandidate]) -> dream_cartridge.TaskCartridge:
        base_c = self.baseline
        return dream_cartridge.TaskCartridge(
            task_name=f"c71_phase3_hypergraph_cycle{self.cycle_idx}",
            description=f"Phase 3 Cycle {self.cycle_idx}: Cross-Intersecting Hypergraph Evolution on RTX 5060",
            baseline_fn=lambda: base_c,
            verify_and_score=self.verify_and_score,
            initial_branches=branches,
            crossover_synthesizer=self.crossover_synthesizer,
            max_parallelism=4,
            sentinel_noise_regime="deterministic",
            blind_vault_evaluator=self.blind_vault_verify,
            noise_probe_runs=1,
            prior_skill_state=self.prior_skill_state,
            hypothesis_contract={
                "claim": (
                    f"Direct cross-intersecting hypergraph (F_+, F_-) GPU evolution on d=10..16 "
                    f"discovers a genuinely new non-AG(2,3) geometry exceeding {base_c.OT:.6f} toward > {NUM2026_STRICT_BAR}"
                ),
                "plan": (
                    "Evolve Mode I (F_-=pi(F_+)) and Mode II (asymmetric) hypergraphs with exact "
                    "mixed-subcube GPU scoring, splice complementary geometries via hypergraph CROSSOVER, "
                    "fine-polish best cores, and certify via exact_verifier.py"
                ),
                "decision_criteria": {
                    "support": f"Exact certified non-AG(2,3) (essential_vars >= 10) L_cert > {base_c.OT:.6f} with 0 monotonicity violations",
                    "weaken": f"All d=10..16 branches <= {base_c.OT:.6f} or fail cross-intersection/monotonicity",
                    "inconclusive": "Gain <= 1e-6 or collapse to lifted d=9 AG(2,3) core",
                },
                "alternative_explanations": [
                    "Mixed (3,4)-set saturation gaps on d=10..16 may have non-subcube components requiring connected-component hosts",
                    "AG(2,3) may remain the global extremal cross-intersecting hypergraph for small d <= 16",
                ],
            },
        )


def _save_and_verify_phase3_json(path: Path, champ: Phase3Champion, extra_meta: Dict[str, Any]) -> Dict[str, Any]:
    meta = {
        "from_scratch": True,
        "phase": 3,
        "source": champ.source,
        "d": champ.d,
        "essential_vars": champ.essential_vars,
        "k_subcubes": len(champ.subcubes),
        "n_effective": champ.n_effective,
        "mode": champ.mode,
        "F_pos": champ.F_pos,
        "F_neg": champ.F_neg,
        "n_F_pos": len(champ.F_pos),
        "n_F_neg": len(champ.F_neg),
        "polished": champ.polished,
        "fp64_OT": champ.OT,
        **extra_meta,
    }
    exact_verifier.save_tier2_candidate_json(
        path,
        champ.g0.tolist(),
        [{"fixed_mask": int(c["fixed_mask"]), "fixed_val": int(c["fixed_val"])} for c in champ.subcubes],
        name=champ.label,
        metadata=meta,
    )
    return exact_verifier.load_and_verify_json(path, require_monotone=True, crosscheck_tier1_max_n=16)


def run_phase3_campaign(max_minutes: float = 14.0, seed: int = 20260713, quick: bool = False) -> Dict[str, Any]:
    t_start = time.perf_counter()
    deadline_ts = t_start + (max_minutes * 60.0 if not quick else 90.0)
    device = get_device(True)
    rng = np.random.default_rng(seed)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    art_dir = Path(_SRC_DIR).parent / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    pool_root = art_dir / "sleep_dag_trees_phase3"

    # =========================================================================
    # STEP 0: Mandatory Tracer #1 — Cold-Start d=9 AG(2,3) Rediscovery Gate
    # =========================================================================
    print("=== PHASE 3 / STEP 0: Mandatory Cold-Start d=9 AG(2,3) Sanity Gate ===", flush=True)
    sanity_d9 = run_sanity_d9(seed=seed + 9)
    print(
        f"  [Tracer #1 d=9] passed={sanity_d9['sanity_d9_passed']} | OT={sanity_d9['best_OT']:.10f} "
        f"| I={sanity_d9['best_I']:.6f} | k_aux={sanity_d9['best_k_aux']} | {sanity_d9['evals_per_sec']:.0f} hg/s "
        f"({sanity_d9['elapsed_sec']:.2f}s)",
        flush=True,
    )
    assert sanity_d9["sanity_d9_passed"], "Mandatory Tracer #1 d=9 AG(2,3) rediscovery gate FAILED!"

    # Track best genuinely new (essential_vars >= 10) champion per dimension d in 10..16
    best_per_d: Dict[int, Phase3Champion] = {}
    elites_per_d: Dict[int, List[HypergraphGenome]] = {d: [] for d in range(10, 17)}
    best_non_ag23_overall: Optional[Phase3Champion] = None
    best_non_ag23_tier1: Optional[Phase3Champion] = None

    def record_champion(ch: Optional[Phase3Champion]) -> None:
        nonlocal best_non_ag23_overall, best_non_ag23_tier1
        if ch is None or ch.essential_vars < 10:
            return
        if ch.d not in best_per_d or ch.OT > best_per_d[ch.d].OT + 1e-12:
            best_per_d[ch.d] = ch
        elites_per_d[ch.d].append(ch.to_genome())
        elites_per_d[ch.d] = elites_per_d[ch.d][-6:]
        if best_non_ag23_overall is None or ch.OT > best_non_ag23_overall.OT + 1e-12:
            best_non_ag23_overall = ch
        if ch.n_effective <= 24:
            if best_non_ag23_tier1 is None or ch.OT > best_non_ag23_tier1.OT + 1e-12:
                best_non_ag23_tier1 = ch

    # Quick baseline on d=10 (cold start) so Cycle 1 TaskCartridge has a valid non-AG(2,3) root
    base_res = evolve_hypergraphs_gpu(
        d=10, mode="mode1", k_choices=(3, 4), n_chains=48, steps=60, seed=seed + 1000, device=device, min_used_verts=10
    )
    if base_res["verified_non_ag23"] is not None:
        base_champ = _rep_to_champion(base_res["verified_non_ag23"], label="base_cold_d10", source="cold_baseline")
    else:
        # Fallback: guaranteed balanced 10-var monotone core (d=10 AltChain or random)
        from phase1_scout import make_or2_altchain_seed
        g10 = make_or2_altchain_seed(10)
        r10 = evaluate_subcube_dealiased_exact_fp64(g10, 10, device=device)
        base_champ = Phase3Champion(
            label="base_altchain_d10", d=10, essential_vars=10, g0=g10,
            subcubes=[{"fixed_mask": int(c["fixed_mask"]), "fixed_val": int(c["fixed_val"])} for c in r10["subcubes"]],
            F_pos=[], F_neg=[], mode="mode2", perm=None,
            OT=float(r10["OT"]), H=float(r10["H"]), I=float(r10["I"]), n_effective=int(r10["n_eff"]),
            source="fallback_baseline",
        )
    record_champion(base_champ)
    print(f"  [Baseline non-AG(2,3)] {base_champ.label}: d={base_champ.d}, ess={base_champ.essential_vars}, OT={base_champ.OT:.8f}", flush=True)

    prior_skill_state: Optional[Dict[str, Any]] = None
    cycle_reports: List[Dict[str, Any]] = []

    # =========================================================================
    # CYCLE 1: Cold-Start Hypergraph Evolution across d = 10, 11, 12, 13
    # =========================================================================
    print("\n================ PHASE 3 / CYCLE 1 (d = 10, 11, 12, 13) ================", flush=True)
    t_c1 = time.perf_counter()
    cat_c1 = _load_catalog_seeds_if_present(art_dir)
    builder1 = Phase3HypergraphCartridgeBuilder(device, base_champ, rng, cycle_idx=1, prior_skill_state=None)

    def _make_d_branch_fn(d_val: int, k_profs: Sequence[Tuple[int, ...]], n_ch: int, s1: int, s2: int, seed_off: int, cat_map: Dict[int, List[Tuple[str, HypergraphGenome]]], dl: float):
        def _fn() -> Optional[Phase3Champion]:
            t_b = time.perf_counter()
            c_seeds = [g for _, g in cat_map.get(d_val, [])]
            ch = _hunt_dimension(
                d=d_val, device=device, rng=rng, seed_base=seed + seed_off,
                n_chains=n_ch, steps_m1=s1, steps_m2=s2, k_profiles=k_profs,
                catalog_seeds=c_seeds, prior_elites=elites_per_d[d_val],
                quick=quick, deadline_ts=dl,
            )
            record_champion(ch)
            if ch is not None:
                print(
                    f"    [Hunt d={d_val:2d}] best={ch.label:22s} | OT={ch.OT:.8f} | I={ch.I:.6f} "
                    f"| ess={ch.essential_vars} | k_aux={len(ch.subcubes):3d} | n_eff={ch.n_effective:3d} "
                    f"| |F+|={len(ch.F_pos)}, |F-|={len(ch.F_neg)} ({time.perf_counter()-t_b:.1f}s)",
                    flush=True,
                )
            return ch
        return _fn

    dl_c1 = min(deadline_ts - 60.0, t_c1 + (30.0 if quick else 320.0))
    branches_c1 = [
        dream_cartridge.BranchCandidate(
            id="p3_c1_d10", branch=0, description="Hypergraph evolution d=10 (mixed 3,4-sets)",
            fn=_make_d_branch_fn(10, [(3, 4), (3,)], 48 if quick else 96, 80 if quick else 550, 80 if quick else 500, 100, cat_c1, dl_c1),
        ),
        dream_cartridge.BranchCandidate(
            id="p3_c1_d11", branch=1, description="Hypergraph evolution d=11 (mixed 3,4-sets)",
            fn=_make_d_branch_fn(11, [(3, 4), (4,)], 48 if quick else 96, 80 if quick else 550, 80 if quick else 500, 200, cat_c1, dl_c1),
        ),
        dream_cartridge.BranchCandidate(
            id="p3_c1_d12", branch=2, description="Hypergraph evolution d=12 (mixed 3,4 & 4-uniform)",
            fn=_make_d_branch_fn(12, [(3, 4), (4,)], 48 if quick else 96, 70 if quick else 500, 70 if quick else 450, 300, cat_c1, dl_c1),
        ),
        dream_cartridge.BranchCandidate(
            id="p3_c1_d13", branch=3, description="Hypergraph evolution d=13 (mixed 3,4 & 4-uniform)",
            fn=_make_d_branch_fn(13, [(3, 4), (4,)], 40 if quick else 80, 60 if quick else 450, 60 if quick else 400, 400, cat_c1, dl_c1),
        ),
    ]
    pool_c1 = str(pool_root / "cycle1")
    cart1 = builder1.build_cartridge(branches_c1)
    runner1 = dream_cartridge.AutoDreamRunner(cart1, pool_dir=pool_c1)
    rep1 = runner1.run()
    record_champion(builder1.best_non_ag23)
    trees1 = dream_engine.load_pool(task=cart1.task_name, pool_dir=pool_c1)
    diag1 = run_sleep_dag_diagnostics(trees1[-1], runner1.problem) if trees1 else {}
    prior_skill_state = rep1.get("skill_state")
    dt_c1 = time.perf_counter() - t_c1
    cycle_reports.append({
        "cycle": 1,
        "elapsed_sec": round(dt_c1, 2),
        "best_non_ag23_OT": round(best_non_ag23_overall.OT, 8) if best_non_ag23_overall else 0.0,
        "best_non_ag23_label": best_non_ag23_overall.label if best_non_ag23_overall else None,
        "sleep_dag_diagnostics": diag1,
        "journal_tail": rep1["journal"][-6:],
    })
    print(
        f"  [Cycle 1 DONE in {dt_c1:.1f}s] Best non-AG(2,3): {best_non_ag23_overall.label if best_non_ag23_overall else None} "
        f"| OT={best_non_ag23_overall.OT if best_non_ag23_overall else 0.0:.8f}",
        flush=True,
    )

    # =========================================================================
    # CYCLE 2: High-d Hunt (d = 14, 15, 16) + Auto-Ingest Catalog Seeds
    # =========================================================================
    if time.perf_counter() < deadline_ts - 25.0:
        print("\n================ PHASE 3 / CYCLE 2 (d = 14, 15, 16 + Catalog Seeds) ================", flush=True)
        t_c2 = time.perf_counter()
        cat_c2 = _load_catalog_seeds_if_present(art_dir)
        n_cat_tot = sum(len(v) for v in cat_c2.values())
        print(f"  Auto-ingested phase3_catalog_seeds.json: {n_cat_tot} verified seeds across {sorted(cat_c2.keys())}", flush=True)

        anchor_c2 = best_non_ag23_overall or base_champ
        builder2 = Phase3HypergraphCartridgeBuilder(device, anchor_c2, rng, cycle_idx=2, prior_skill_state=prior_skill_state)
        for ch_el in best_per_d.values():
            builder2._register(ch_el)

        dl_c2 = min(deadline_ts - 40.0, t_c2 + (28.0 if quick else 320.0))

        # Branch 3: Mutate/evolve top catalog seeds across all dimensions (if present!) or extra d=12 hunt
        def _branch_catalog_evolve() -> Optional[Phase3Champion]:
            if not cat_c2:
                return _make_d_branch_fn(12, [(3, 4)], 48 if quick else 80, 60 if quick else 450, 60 if quick else 400, 550, {}, dl_c2)()
            # Score ALL catalog seeds first and pick top-3 dimensions to evolve!
            scored_cat: List[Tuple[float, int, str, HypergraphGenome]] = []
            for cd, lst in cat_c2.items():
                for nm, gm in lst:
                    rp = verify_hypergraph_candidate_fp64(gm, device, try_gap_balance=True)
                    if rp is not None and int(rp["essential_vars"]) >= 10:
                        ch_raw = _rep_to_champion(rp, label=f"cat_{nm}", source="catalog_raw")
                        record_champion(ch_raw)
                        scored_cat.append((ch_raw.OT, cd, nm, gm))
            scored_cat.sort(key=lambda x: x[0], reverse=True)
            print(f"    [Catalog Direct Top-3]: {[(round(x[0], 6), x[1], x[2]) for x in scored_cat[:3]]}", flush=True)
            best_loc: Optional[Phase3Champion] = None
            for idx_top, (orig_ot, cd, nm, gm) in enumerate(scored_cat[: (1 if quick else 3)]):
                if time.perf_counter() >= dl_c2:
                    break
                ch_ev = _hunt_dimension(
                    d=cd, device=device, rng=rng, seed_base=seed + 700 + idx_top * 31,
                    n_chains=40 if quick else 64,
                    steps_m1=40 if quick else 220, steps_m2=60 if quick else 350,
                    k_profiles=[(3, 4)] if cd <= 14 else [(4,), (3, 4)],
                    catalog_seeds=[gm], prior_elites=elites_per_d[cd],
                    quick=quick, deadline_ts=dl_c2,
                )
                record_champion(ch_ev)
                if ch_ev is not None and (best_loc is None or ch_ev.OT > best_loc.OT):
                    best_loc = ch_ev
            return best_loc or best_non_ag23_overall

        branches_c2 = [
            dream_cartridge.BranchCandidate(
                id="p3_c2_d14", branch=0, description="Hypergraph evolution d=14 (mixed 3,4 & 4-uniform)",
                fn=_make_d_branch_fn(14, [(3, 4), (4,)], 32 if quick else 64, 50 if quick else 360, 50 if quick else 340, 500, cat_c2, dl_c2),
            ),
            dream_cartridge.BranchCandidate(
                id="p3_c2_d15", branch=1, description="Hypergraph evolution d=15 (4-uniform & mixed 3,4)",
                fn=_make_d_branch_fn(15, [(4,), (3, 4)], 24 if quick else 48, 40 if quick else 260, 40 if quick else 240, 600, cat_c2, dl_c2),
            ),
            dream_cartridge.BranchCandidate(
                id="p3_c2_d16", branch=2, description="Hypergraph evolution d=16 (4-uniform)",
                fn=_make_d_branch_fn(16, [(4,), (3, 4)], 16 if quick else 32, 30 if quick else 180, 30 if quick else 160, 700, cat_c2, dl_c2),
            ),
            dream_cartridge.BranchCandidate(
                id="p3_c2_catalog", branch=3, description="Catalog seed scoring & GPU hypergraph mutation",
                fn=_branch_catalog_evolve,
            ),
        ]
        pool_c2 = str(pool_root / "cycle2")
        cart2 = builder2.build_cartridge(branches_c2)
        runner2 = dream_cartridge.AutoDreamRunner(cart2, pool_dir=pool_c2)
        rep2 = runner2.run()
        record_champion(builder2.best_non_ag23)
        trees2 = dream_engine.load_pool(task=cart2.task_name, pool_dir=pool_c2)
        diag2 = run_sleep_dag_diagnostics(trees2[-1], runner2.problem) if trees2 else {}
        prior_skill_state = rep2.get("skill_state")
        dt_c2 = time.perf_counter() - t_c2
        cycle_reports.append({
            "cycle": 2,
            "elapsed_sec": round(dt_c2, 2),
            "catalog_seeds_ingested": n_cat_tot,
            "best_non_ag23_OT": round(best_non_ag23_overall.OT, 8) if best_non_ag23_overall else 0.0,
            "best_non_ag23_label": best_non_ag23_overall.label if best_non_ag23_overall else None,
            "sleep_dag_diagnostics": diag2,
            "journal_tail": rep2["journal"][-6:],
        })
        print(
            f"  [Cycle 2 DONE in {dt_c2:.1f}s] Best non-AG(2,3): {best_non_ag23_overall.label if best_non_ag23_overall else None} "
            f"| OT={best_non_ag23_overall.OT if best_non_ag23_overall else 0.0:.8f}",
            flush=True,
        )

    # =========================================================================
    # CYCLE 3: Deep Focused Assault on Top-3 Dimensions + Final Polish + BlindVault
    # =========================================================================
    if time.perf_counter() < deadline_ts - 15.0:
        print("\n================ PHASE 3 / CYCLE 3 (Deep Assault on Top-3 Dimensions) ================", flush=True)
        t_c3 = time.perf_counter()
        cat_c3 = _load_catalog_seeds_if_present(art_dir)
        ranked_ds = sorted(best_per_d.keys(), key=lambda d_k: best_per_d[d_k].OT, reverse=True)
        top_ds = ranked_ds[:3] if ranked_ds else [10, 11, 12]
        print(f"  Focusing deep Cycle 3 assault on top dimensions: {top_ds}", flush=True)

        anchor_c3 = best_non_ag23_overall or base_champ
        builder3 = Phase3HypergraphCartridgeBuilder(device, anchor_c3, rng, cycle_idx=3, prior_skill_state=prior_skill_state)
        for ch_el in best_per_d.values():
            builder3._register(ch_el)

        dl_c3 = deadline_ts - 10.0
        branches_c3: List[dream_cartridge.BranchCandidate] = []
        for b_i, d_star in enumerate(top_ds):
            k_p = [(3, 4), (3,)] if d_star <= 11 else ([(3, 4), (4,)] if d_star <= 14 else [(4,), (3, 4)])
            n_ch = (32 if quick else 96) if d_star <= 12 else ((24 if quick else 64) if d_star <= 14 else (16 if quick else 40))
            st1 = (60 if quick else 700) if d_star <= 12 else ((45 if quick else 450) if d_star <= 14 else (30 if quick else 240))
            st2 = (60 if quick else 700) if d_star <= 12 else ((45 if quick else 450) if d_star <= 14 else (30 if quick else 240))
            branches_c3.append(
                dream_cartridge.BranchCandidate(
                    id=f"p3_c3_deep_d{d_star}",
                    branch=b_i,
                    description=f"Deep Cycle 3 hypergraph assault + polish on d={d_star}",
                    fn=_make_d_branch_fn(d_star, k_p, n_ch, st1, st2, 900 + b_i * 73, cat_c3, dl_c3),
                )
            )

        pool_c3 = str(pool_root / "cycle3")
        cart3 = builder3.build_cartridge(branches_c3)
        runner3 = dream_cartridge.AutoDreamRunner(cart3, pool_dir=pool_c3)
        rep3 = runner3.run()
        record_champion(builder3.best_non_ag23)
        trees3 = dream_engine.load_pool(task=cart3.task_name, pool_dir=pool_c3)
        diag3 = run_sleep_dag_diagnostics(trees3[-1], runner3.problem) if trees3 else {}
        prior_skill_state = rep3.get("skill_state")
        dt_c3 = time.perf_counter() - t_c3
        cycle_reports.append({
            "cycle": 3,
            "elapsed_sec": round(dt_c3, 2),
            "focused_dims": top_ds,
            "best_non_ag23_OT": round(best_non_ag23_overall.OT, 8) if best_non_ag23_overall else 0.0,
            "best_non_ag23_label": best_non_ag23_overall.label if best_non_ag23_overall else None,
            "blind_vault_report": rep3.get("blind_vault_report"),
            "sleep_dag_diagnostics": diag3,
            "journal_tail": rep3["journal"][-6:],
        })
        print(
            f"  [Cycle 3 DONE in {dt_c3:.1f}s] Best non-AG(2,3): {best_non_ag23_overall.label if best_non_ag23_overall else None} "
            f"| OT={best_non_ag23_overall.OT if best_non_ag23_overall else 0.0:.8f}",
            flush=True,
        )

    # =========================================================================
    # FINAL EXACT CERTIFICATION & SUMMARY PERSISTENCE
    # =========================================================================
    print("\n=== PHASE 3 FINAL CERTIFICATION VIA exact_verifier.py ===", flush=True)
    assert best_non_ag23_overall is not None, "No valid non-AG(2,3) candidate found!"

    best_path = art_dir / "phase3_best_candidate.json"
    cert_best = _save_and_verify_phase3_json(best_path, best_non_ag23_overall, {"campaign": "phase3_hypergraph"})
    print(
        f"  [BEST GENUINELY NEW (d>=10)] {best_non_ag23_overall.label}: d={best_non_ag23_overall.d}, "
        f"ess={best_non_ag23_overall.essential_vars}, k_aux={len(best_non_ag23_overall.subcubes)}, "
        f"n_eff={best_non_ag23_overall.n_effective}, I={cert_best['I_exact_str']}, "
        f"L_cert={cert_best['L_cert'][:24]}..., mono={cert_best['is_monotone']}, beats_num2026={cert_best['beats_num2026']}",
        flush=True,
    )

    cert_t1: Optional[Dict[str, Any]] = None
    t1_champ = best_non_ag23_tier1
    if t1_champ is not None:
        t1_path = art_dir / "phase3_best_tier1.json"
        cert_t1 = _save_and_verify_phase3_json(t1_path, t1_champ, {"campaign": "phase3_hypergraph", "tier1_eligible": True})
        print(
            f"  [BEST TIER 1 (n_eff<=24)] {t1_champ.label}: d={t1_champ.d}, ess={t1_champ.essential_vars}, "
            f"n_eff={t1_champ.n_effective}, I={cert_t1['I_exact_str']}, L_cert={cert_t1['L_cert'][:24]}...",
            flush=True,
        )

    per_d_summary: Dict[str, Any] = {}
    for d_k in range(10, 17):
        if d_k in best_per_d:
            c = best_per_d[d_k]
            per_d_summary[f"d={d_k}"] = {
                "label": c.label,
                "source": c.source,
                "d": c.d,
                "essential_vars": c.essential_vars,
                "k_subcubes": len(c.subcubes),
                "n_effective": c.n_effective,
                "OT": round(c.OT, 10),
                "I": round(c.I, 8),
                "H": round(c.H, 8),
                "mode": c.mode,
                "n_F_pos": len(c.F_pos),
                "n_F_neg": len(c.F_neg),
                "F_pos": c.F_pos,
                "F_neg": c.F_neg,
                "polished": c.polished,
            }

    peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if device.type == "cuda" else 0.0
    total_sec = time.perf_counter() - t_start

    summary = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "seed": seed,
        "from_scratch_verified": True,
        "peak_vram_mb": round(peak_vram_mb, 2),
        "within_4_5_gib_cap": peak_vram_mb <= 4.5 * 1024.0,
        "total_elapsed_sec": round(total_sec, 2),
        "tracer1_sanity_d9": sanity_d9,
        "best_per_d_non_ag23": per_d_summary,
        "cycles": cycle_reports,
        "best_non_ag23_certified": {
            "label": best_non_ag23_overall.label,
            "source": best_non_ag23_overall.source,
            "d": best_non_ag23_overall.d,
            "essential_vars": best_non_ag23_overall.essential_vars,
            "k_subcubes": len(best_non_ag23_overall.subcubes),
            "n_effective": best_non_ag23_overall.n_effective,
            "F_pos": best_non_ag23_overall.F_pos,
            "F_neg": best_non_ag23_overall.F_neg,
            "I_exact_str": cert_best["I_exact_str"],
            "L_cert": cert_best["L_cert"],
            "U_cert": cert_best["U_cert"],
            "interval_width_L": cert_best["interval_width_L"],
            "is_monotone": cert_best["is_monotone"],
            "exact_balance": cert_best["exact_balance"],
            "parseval_verified": cert_best["parseval_verified"],
            "beats_num2026": cert_best["beats_num2026"],
            "sha256": cert_best["sha256"],
        },
        "best_tier1_non_ag23_certified": (
            {
                "label": t1_champ.label,
                "d": t1_champ.d,
                "essential_vars": t1_champ.essential_vars,
                "k_subcubes": len(t1_champ.subcubes),
                "n_effective": t1_champ.n_effective,
                "F_pos": t1_champ.F_pos,
                "F_neg": t1_champ.F_neg,
                "I_exact_str": cert_t1["I_exact_str"],
                "L_cert": cert_t1["L_cert"],
                "is_monotone": cert_t1["is_monotone"],
                "beats_num2026": cert_t1["beats_num2026"],
                "sha256": cert_t1["sha256"],
            }
            if (t1_champ is not None and cert_t1 is not None)
            else None
        ),
        "num2026_strict_bar": str(NUM2026_STRICT_BAR),
        "final_skill_state": prior_skill_state,
    }

    summary_path = art_dir / "phase3_campaign_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== PHASE 3 CAMPAIGN FINISHED in {total_sec:.1f}s (Peak VRAM {peak_vram_mb:.1f} MB) ===", flush=True)
    print(f"  Summary: {summary_path}", flush=True)
    return summary


def run_antihub_ridge_cycle4(max_minutes: float = 4.5, seed: int = 20260799) -> Dict[str, Any]:
    """
    Cycle 4 (Anti-Hub High-Influence Ridge Assault):
    Cycles 1-3 revealed that unconstrained hypergraph SA at d=11..16 falls into a
    4-hub sunflower basin (I ~ 1.63, OT ~ 6.30-6.34, 4 vertices with deg 10..15).
    Cycle 4 bans that hub trap via (min_I >= 1.88, max_deg_cap in {4..6}) across
    d = 10..15, forcing all RTX 5060 chains onto the AG(2,3) / Z10 regular-hypergraph
    ridge (I ~ 1.95..2.30), then merges any improvements into phase3_campaign_summary.json
    and re-certifies phase3_best_candidate.json + phase3_best_tier1.json!
    """
    t0 = time.perf_counter()
    deadline_ts = t0 + max_minutes * 60.0
    device = get_device(True)
    rng = np.random.default_rng(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    art_dir = Path(_SRC_DIR).parent / "artifacts"
    summary_path = art_dir / "phase3_campaign_summary.json"
    assert summary_path.is_file(), "Run Cycles 1-3 first before --antihub-cycle4!"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cat_map = _load_catalog_seeds_if_present(art_dir)

    print("\n================ PHASE 3 / CYCLE 4 (Anti-Hub High-Influence Ridge: I >= 1.88) ================", flush=True)
    configs = [
        # (d, max_deg_cap, max_edges, k_profiles, n_chains, steps_m1, steps_m2)
        (10, 4, 16, [(3, 4), (3,)], 128, 750, 750),
        (11, 4, 18, [(3, 4), (4,)], 128, 700, 700),
        (12, 5, 20, [(3, 4), (4,)], 112, 600, 600),
        (13, 5, 22, [(3, 4), (4,)], 96, 500, 500),
        (14, 6, 26, [(4,), (3, 4)], 64, 350, 350),
        (15, 6, 30, [(4,), (3, 4)], 48, 220, 220),
    ]

    ridge_results: Dict[str, Any] = {}
    best_per_d = summary.get("best_per_d_non_ag23", {})
    cur_best_ot = float(summary["best_non_ag23_certified"]["L_cert"])
    cur_t1_ot = (
        float(summary["best_tier1_non_ag23_certified"]["L_cert"])
        if summary.get("best_tier1_non_ag23_certified")
        else 0.0
    )
    new_overall_champ: Optional[Phase3Champion] = None
    new_tier1_champ: Optional[Phase3Champion] = None

    for d, deg_cap, max_e, k_profs, n_ch, s1, s2 in configs:
        if time.perf_counter() >= deadline_ts:
            break
        t_d = time.perf_counter()
        c_seeds = [g for _, g in cat_map.get(d, [])]
        # Also include prior best at d if it had I >= 1.85
        prev_d = best_per_d.get(f"d={d}")
        if prev_d and float(prev_d.get("I", 0.0)) >= 1.85 and prev_d.get("F_pos") and prev_d.get("F_neg"):
            fp0, fn0 = [int(x) for x in prev_d["F_pos"]], [int(x) for x in prev_d["F_neg"]]
            p0 = detect_isomorphism_perm(d, fp0, fn0)
            c_seeds.insert(0, HypergraphGenome(d=d, F_pos=fp0, F_neg=fn0, mode="mode1" if p0 else "mode2", perm=p0))

        best_d_ch: Optional[Phase3Champion] = None
        m1_elites: List[HypergraphGenome] = [g for g in c_seeds if g.perm is not None]

        for p_i, kp in enumerate(k_profs):
            if time.perf_counter() >= deadline_ts:
                break
            r1 = evolve_hypergraphs_gpu(
                d=d, mode="mode1", k_choices=kp,
                n_chains=n_ch, steps=s1,
                t_start=0.25, t_end=0.002,
                seed=seed + d * 131 + p_i * 19,
                device=device,
                seed_genomes=m1_elites[:6],
                max_edges=max_e,
                min_used_verts=d,
                min_I=1.88,
                max_deg_cap=deg_cap,
            )
            vn = r1["verified_non_ag23"]
            if vn is not None and float(vn["I"]) >= 1.85:
                ch = _rep_to_champion(vn, label=f"Ridge_d{d}_m1_k{''.join(map(str, kp))}", source=f"c4_ridge_m1_k{kp}")
                m1_elites.append(ch.to_genome())
                if best_d_ch is None or ch.OT > best_d_ch.OT:
                    best_d_ch = ch

        m2_seeds = m1_elites[:4] + c_seeds[:6]
        for p_i, kp in enumerate(k_profs):
            if time.perf_counter() >= deadline_ts:
                break
            r2 = evolve_hypergraphs_gpu(
                d=d, mode="mode2", k_choices=kp,
                n_chains=n_ch, steps=s2,
                t_start=0.25, t_end=0.002,
                seed=seed + d * 251 + p_i * 31,
                device=device,
                seed_genomes=m2_seeds,
                max_edges=max_e,
                min_used_verts=d,
                min_I=1.88,
                max_deg_cap=deg_cap,
            )
            vn = r2["verified_non_ag23"]
            if vn is not None and float(vn["I"]) >= 1.85:
                ch = _rep_to_champion(vn, label=f"Ridge_d{d}_m2_k{''.join(map(str, kp))}", source=f"c4_ridge_m2_k{kp}")
                if best_d_ch is None or ch.OT > best_d_ch.OT:
                    best_d_ch = ch

        if best_d_ch is not None and time.perf_counter() < deadline_ts:
            best_d_ch = _fine_polish_champion(best_d_ch, device, rng, quick=False)

        dt_d = time.perf_counter() - t_d
        if best_d_ch is not None:
            deg_p = [sum(1 for e in best_d_ch.F_pos if (e >> b) & 1) for b in range(d)]
            print(
                f"    [Cycle4 Ridge d={d:2d}] {best_d_ch.label:20s} | OT={best_d_ch.OT:.8f} | I={best_d_ch.I:.6f} "
                f"| ess={best_d_ch.essential_vars} | k={len(best_d_ch.subcubes):2d} | n_eff={best_d_ch.n_effective:2d} "
                f"| |F+|={len(best_d_ch.F_pos)} max_deg={max(deg_p) if deg_p else 0} ({dt_d:.1f}s)",
                flush=True,
            )
            entry = {
                "label": best_d_ch.label,
                "source": best_d_ch.source,
                "d": best_d_ch.d,
                "essential_vars": best_d_ch.essential_vars,
                "k_subcubes": len(best_d_ch.subcubes),
                "n_effective": best_d_ch.n_effective,
                "OT": round(best_d_ch.OT, 10),
                "I": round(best_d_ch.I, 8),
                "H": round(best_d_ch.H, 8),
                "mode": best_d_ch.mode,
                "n_F_pos": len(best_d_ch.F_pos),
                "n_F_neg": len(best_d_ch.F_neg),
                "F_pos": best_d_ch.F_pos,
                "F_neg": best_d_ch.F_neg,
                "max_deg_pos": max(deg_p) if deg_p else 0,
                "polished": best_d_ch.polished,
            }
            ridge_results[f"d={d}"] = entry
            if f"d={d}" not in best_per_d or best_d_ch.OT > float(best_per_d[f"d={d}"]["OT"]) + 1e-12:
                best_per_d[f"d={d}"] = entry
            if best_d_ch.OT > cur_best_ot + 1e-12:
                cur_best_ot = best_d_ch.OT
                new_overall_champ = best_d_ch
            if best_d_ch.n_effective <= 24 and best_d_ch.OT > cur_t1_ot + 1e-12:
                cur_t1_ot = best_d_ch.OT
                new_tier1_champ = best_d_ch
        else:
            print(f"    [Cycle4 Ridge d={d:2d}] no valid I>=1.85 pure-subcube hit ({dt_d:.1f}s)", flush=True)

    # Re-certify if Cycle 4 beat the overall or Tier 1 non-AG(2,3) champion!
    if new_overall_champ is not None:
        best_path = art_dir / "phase3_best_candidate.json"
        cert_b = _save_and_verify_phase3_json(best_path, new_overall_champ, {"campaign": "phase3_cycle4_antihub"})
        summary["best_non_ag23_certified"] = {
            "label": new_overall_champ.label,
            "source": new_overall_champ.source,
            "d": new_overall_champ.d,
            "essential_vars": new_overall_champ.essential_vars,
            "k_subcubes": len(new_overall_champ.subcubes),
            "n_effective": new_overall_champ.n_effective,
            "F_pos": new_overall_champ.F_pos,
            "F_neg": new_overall_champ.F_neg,
            "I_exact_str": cert_b["I_exact_str"],
            "L_cert": cert_b["L_cert"],
            "U_cert": cert_b["U_cert"],
            "interval_width_L": cert_b["interval_width_L"],
            "is_monotone": cert_b["is_monotone"],
            "exact_balance": cert_b["exact_balance"],
            "parseval_verified": cert_b["parseval_verified"],
            "beats_num2026": cert_b["beats_num2026"],
            "sha256": cert_b["sha256"],
        }
        print(
            f"  [CYCLE 4 NEW OVERALL CHAMPION!] {new_overall_champ.label}: d={new_overall_champ.d}, "
            f"I={cert_b['I_exact_str']}, L_cert={cert_b['L_cert'][:24]}..., beats_num2026={cert_b['beats_num2026']}",
            flush=True,
        )

    if new_tier1_champ is not None:
        t1_path = art_dir / "phase3_best_tier1.json"
        cert_t1 = _save_and_verify_phase3_json(t1_path, new_tier1_champ, {"campaign": "phase3_cycle4_antihub", "tier1_eligible": True})
        summary["best_tier1_non_ag23_certified"] = {
            "label": new_tier1_champ.label,
            "d": new_tier1_champ.d,
            "essential_vars": new_tier1_champ.essential_vars,
            "k_subcubes": len(new_tier1_champ.subcubes),
            "n_effective": new_tier1_champ.n_effective,
            "F_pos": new_tier1_champ.F_pos,
            "F_neg": new_tier1_champ.F_neg,
            "I_exact_str": cert_t1["I_exact_str"],
            "L_cert": cert_t1["L_cert"],
            "is_monotone": cert_t1["is_monotone"],
            "beats_num2026": cert_t1["beats_num2026"],
            "sha256": cert_t1["sha256"],
        }

    dt_c4 = time.perf_counter() - t0
    peak_c4 = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if device.type == "cuda" else 0.0
    summary["best_per_d_non_ag23"] = best_per_d
    summary["cycle4_antihub_ridge"] = {
        "elapsed_sec": round(dt_c4, 2),
        "peak_vram_mb": round(peak_c4, 2),
        "per_d": ridge_results,
    }
    summary["peak_vram_mb"] = round(max(float(summary.get("peak_vram_mb", 0.0)), peak_c4), 2)
    summary["total_elapsed_sec"] = round(float(summary.get("total_elapsed_sec", 0.0)) + dt_c4, 2)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"=== CYCLE 4 ANTI-HUB RIDGE DONE in {dt_c4:.1f}s (Peak VRAM {peak_c4:.1f} MB) ===", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Cross-Intersecting Hypergraph SLEEP-DAG-RSI Campaign")
    parser.add_argument("--max-minutes", type=float, default=14.0)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--quick", action="store_true", help="Fast ~60s end-to-end smoke check")
    parser.add_argument("--antihub-cycle4", action="store_true", help="Run Cycle 4 Anti-Hub High-Influence Ridge addon")
    args = parser.parse_args()
    if args.antihub_cycle4:
        run_antihub_ridge_cycle4(max_minutes=args.max_minutes, seed=args.seed)
    else:
        run_phase3_campaign(max_minutes=args.max_minutes, seed=args.seed, quick=args.quick)


if __name__ == "__main__":
    main()

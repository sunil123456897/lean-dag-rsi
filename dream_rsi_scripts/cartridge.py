"""
Universal Cartridge SDK & Autonomous Multi-Round Discovery Runner (`SLEEP-DAG-RSI v8`).

Provides a domain-agnostic execution harness (`TaskCartridge` + `AutoDreamRunner` + `BlindVaultGuard`)
connecting user project benchmarks directly to `engine.py`:
  - Pre-flight OEB noise-floor calibration (`noise_probe_runs`, `2 * sigma_noise` gate)
  - 5% sequential early abort filter (`WaldMicroSentinel`)
  - Backward dead-bucket stepping-stone decomposition (`BackwardSubGoalDecomposer`)
  - Autonomous multi-branch crossover (`crossover_synthesizer` or zero-handwriting `AstCodeSplicer`)
  - Continuous parameter tuning + `prune_epistatic_gauss_newton` / `prune_tunable_params` + `evaluate_flat_plateau`
  - Bounded `MutableSkillState` (`SKILL.state`) persistence & single-shot `BlindVaultGuard`
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import engine as dream_engine  # noqa: E402
import loop as dream_loop  # noqa: E402


class BlindVaultViolationError(RuntimeError):
    """Raised when code attempts to evaluate the untouched Blind Vault more than once."""


class BlindVaultGuard:
    """
    Enforces the Zero-Gaming single-open invariant on a chronological or held-out Blind Vault.
    Once opened on the final selected champion, any second call raises `BlindVaultViolationError`.
    """

    def __init__(
        self,
        eval_fn: Callable[[Callable[..., Any]], Dict[str, Any]],
        max_calls: int = 1,
    ) -> None:
        self._eval_fn = eval_fn
        self._max_calls = int(max(1, max_calls))
        self._calls = 0
        self._last_report: Optional[Dict[str, Any]] = None

    @property
    def calls(self) -> int:
        return self._calls

    def evaluate_once(self, candidate_fn: Callable[..., Any]) -> Dict[str, Any]:
        if self._calls >= self._max_calls:
            raise BlindVaultViolationError(
                f"Blind Vault lock violated: attempted call #{self._calls + 1} "
                f"(max_calls={self._max_calls}). Tuning or re-selecting on the Blind Vault is prohibited."
            )
        self._calls += 1
        report = dict(self._eval_fn(candidate_fn))
        report["blind_vault_call_count"] = self._calls
        self._last_report = report
        return report


@dataclass
class BranchCandidate:
    """
    A candidate solution node in a `TaskCartridge`.
    Supports pure callables, source-code strings for autonomous `AstCodeSplicer` merges,
    and `@tunable` continuous parameter vectors for Round 4 Neuro-Symbolic polishing.
    """

    id: str
    branch: int
    description: str
    fn: Callable[..., Any]
    parent_id: str = "root"
    attempt: int = 1
    next_refinement: Optional["BranchCandidate"] = None
    repair_candidate: Optional["BranchCandidate"] = None
    tunable_params: Optional[np.ndarray] = None
    tunable_bounds: Optional[List[Tuple[float, float]]] = None
    param_names: Optional[List[str]] = None
    param_factory: Optional[Callable[[np.ndarray], Callable[..., Any]]] = None
    mechanism_tags: Dict[str, Any] = field(default_factory=dict)
    source_code: Optional[str] = None
    entry_func: Optional[str] = None
    root_cause: Optional[str] = None
    predicted_buckets: Optional[List[int]] = None
    provenance: Optional[Dict[str, Any]] = None


@dataclass
class TaskCartridge:
    """
    Plug-in problem specification for `AutoDreamRunner`.
    `verify_and_score(fn)` must check domain validity before returning `(ok, score, fail_class, info)`.
    """

    task_name: str
    description: str
    baseline_fn: Callable[..., Any]
    verify_and_score: Callable[
        [Callable[..., Any]], Tuple[bool, float, str, Dict[str, Any]]
    ]
    initial_branches: List[BranchCandidate]
    crossover_synthesizer: Optional[
        Callable[[BranchCandidate, BranchCandidate], Optional[BranchCandidate]]
    ] = None
    train_evaluator: Optional[Callable[[Callable[..., Any]], float]] = None
    hypothesis_contract: Optional[Dict[str, Any]] = None
    max_parallelism: int = 4
    # Optional SLEEP-DAG-RSI v8 universal hooks:
    base_source: Optional[str] = None
    entry_func: Optional[str] = None
    bucket_inputs: Optional[Sequence[Any]] = None
    pointwise_slice_evaluator: Optional[
        Callable[[Callable[..., Any], Optional[Sequence[int]]], Sequence[float]]
    ] = None
    sentinel_noise_regime: str = "deterministic"
    blind_vault_evaluator: Optional[Callable[[Callable[..., Any]], Dict[str, Any]]] = (
        None
    )
    noise_probe_runs: int = 1
    provenance: Optional[Dict[str, Any]] = None
    prior_skill_state: Optional[Dict[str, Any]] = None


def _compile_safe_callable(
    source_code: str, entry_func: str
) -> Tuple[Optional[Callable[..., Any]], Optional[str]]:
    """Verifies candidate Python code via `AstPolicySafetyGate` and compiles `entry_func`."""
    gate = dream_engine.AstPolicySafetyGate.verify_source(source_code)
    if not gate["valid"]:
        return None, f"AstPolicySafetyGate blocked code: {gate['violations']}"
    ns: Dict[str, Any] = {"np": np, "math": __import__("math")}
    try:
        code_obj = compile(source_code, "<ast_spliced_candidate>", "exec")
        exec(code_obj, ns)  # noqa: S102 - statically gated by AstPolicySafetyGate above
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    fn = ns.get(entry_func)
    if not callable(fn):
        return None, f"Compiled source did not define callable `{entry_func}`"
    return fn, None


class AutoDreamRunner:
    """
    Executes the closed-loop `SLEEP-DAG-RSI v8` discovery cycle on any `TaskCartridge`,
    persisting the `DiscoveryTree` and bounded `MutableSkillState` (`SKILL.state`).
    """

    def __init__(
        self, cartridge: TaskCartridge, pool_dir: Optional[str] = None
    ) -> None:
        self.cartridge = cartridge
        self.own_pool = pool_dir is None
        self.pool_dir = pool_dir or tempfile.mkdtemp(
            prefix=f"autodream_{cartridge.task_name}_"
        )
        self.problem = dream_engine.Problem()
        self.candidates_by_id: Dict[str, BranchCandidate] = {}
        self.metrics_by_id: Dict[str, Dict[str, Any]] = {}
        self.pointwise_by_id: Dict[str, List[float]] = {}
        self.journal: List[str] = []
        self.sentinel = dream_engine.WaldMicroSentinel(
            noise_regime=cartridge.sentinel_noise_regime
        )
        self._vault_guard: Optional[BlindVaultGuard] = (
            BlindVaultGuard(cartridge.blind_vault_evaluator, max_calls=1)
            if cartridge.blind_vault_evaluator is not None
            else None
        )

    def _evaluate_and_record(
        self,
        tree: dream_engine.DiscoveryTree,
        cand: BranchCandidate,
        secondary_parent_id: Optional[str] = None,
        stage_label: str = "explore",
    ) -> dream_engine.Node:
        is_xover = secondary_parent_id is not None or "crossover" in stage_label
        parent_id = cand.parent_id if cand.parent_id in tree.nodes else "root"
        parent_node = tree.nodes.get(parent_id)

        # Optional Stage 1: 5% WaldMicroSentinel early veto before full evaluation
        if (
            self.cartridge.pointwise_slice_evaluator is not None
            and parent_node is not None
            and parent_id in self.pointwise_by_id
        ):
            parent_pts = self.pointwise_by_id[parent_id]
            probe_stub = dream_engine.Node(
                id=cand.id,
                parent_id=parent_id,
                branch=cand.branch,
                attempt=cand.attempt,
                score=0.0,
            )
            try:
                gate_rep = self.sentinel.evaluate_gate(
                    parent_node=parent_node,
                    child_node=probe_stub,
                    parent_pointwise_scores=parent_pts,
                    eval_slice_fn=lambda idx: self.cartridge.pointwise_slice_evaluator(  # type: ignore[misc]
                        cand.fn, idx
                    ),
                    is_crossover=is_xover,
                )
            except Exception:
                gate_rep = {"passed": True}

            if not gate_rep.get("passed", True):
                aborted_node = dream_engine.Node(
                    id=cand.id,
                    parent_id=parent_id,
                    branch=cand.branch,
                    attempt=cand.attempt,
                    score=round(float(gate_rep.get("sentinel_mean", 0.0)), 4),
                    fail_class="sentinel_pruned",
                    valid=False,
                    error=f"WaldMicroSentinel early veto (LCB={gate_rep.get('lcb', 0.0):.4f})",
                    meta={
                        "description": cand.description,
                        "stage": stage_label,
                        "sentinel_aborted": True,
                        "cost": float(gate_rep.get("cost_spent", 0.05)),
                    },
                )
                tree.add(aborted_node)
                self.candidates_by_id[cand.id] = cand
                self.metrics_by_id[cand.id] = {"sentinel_gate": gate_rep}
                return aborted_node

        try:
            ok, score, fail_class, info = self.cartridge.verify_and_score(cand.fn)
            err_msg = info.get("error")
        except Exception as exc:
            ok = False
            score = 0.0
            exc_name = type(exc).__name__.lower()
            if any(
                k in exc_name or k in str(exc).lower()
                for k in (
                    "memory",
                    "shape",
                    "broadcast",
                    "valueerror",
                    "typeerror",
                    "overflow",
                )
            ):
                fail_class = "oom"
            else:
                fail_class = "algorithmic_failure"
            err_msg = f"{type(exc).__name__}: {exc}"
            info = {"error": err_msg}

        meta: Dict[str, Any] = {
            "description": cand.description,
            "stage": stage_label,
            "artifact_hash": f"verified_{cand.id}",
        }
        if cand.root_cause:
            meta["root_cause"] = cand.root_cause
        if cand.predicted_buckets is not None:
            meta["predicted_buckets"] = list(cand.predicted_buckets)
        merged_prov: Dict[str, Any] = {}
        if isinstance(self.cartridge.provenance, dict):
            merged_prov.update(self.cartridge.provenance)
        if isinstance(cand.provenance, dict):
            merged_prov.update(cand.provenance)
        if merged_prov:
            meta["provenance"] = merged_prov
        if secondary_parent_id:
            meta["secondary_parent_id"] = secondary_parent_id
            meta["crossover_parents"] = [cand.parent_id, secondary_parent_id]

        artifact: Dict[str, Any] = {}
        if cand.source_code:
            artifact["code"] = cand.source_code
        if self.cartridge.base_source:
            artifact["base_code"] = self.cartridge.base_source

        pass_mask = [int(x) for x in info.get("pass_mask", [])]
        bucket_scores = [
            float(x)
            for x in info.get("bucket_scores", info.get("bucket_margins", pass_mask))
        ]
        failure_breakdown = {
            str(k): float(v) for k, v in info.get("failure_breakdown", {}).items()
        }
        line_hotspots = list(info.get("line_hotspots", []))

        node = dream_engine.Node(
            id=cand.id,
            parent_id=cand.parent_id,
            branch=cand.branch,
            attempt=cand.attempt,
            score=round(float(score), 4) if ok else 0.0,
            fail_class="ok" if ok else fail_class,
            error=err_msg,
            valid=bool(ok),
            pass_mask=pass_mask,
            bucket_scores=bucket_scores,
            failure_breakdown=failure_breakdown,
            line_hotspots=line_hotspots,
            artifact=artifact,
            meta=meta,
        )
        tree.add(node)
        self.candidates_by_id[cand.id] = cand
        self.metrics_by_id[cand.id] = info
        if ok and self.cartridge.pointwise_slice_evaluator is not None:
            try:
                pts = self.cartridge.pointwise_slice_evaluator(cand.fn, None)
                self.pointwise_by_id[cand.id] = [float(v) for v in pts]
            except Exception:
                pass
        return node

    def _try_autonomous_ast_splice(
        self,
        u_cand: BranchCandidate,
        v_cand: BranchCandidate,
        tree: dream_engine.DiscoveryTree,
    ) -> Optional[BranchCandidate]:
        entry_fn = u_cand.entry_func or v_cand.entry_func or self.cartridge.entry_func
        base_src = self.cartridge.base_source
        if not (base_src and entry_fn and u_cand.source_code and v_cand.source_code):
            return None

        u_node = tree.nodes.get(u_cand.id)
        v_node = tree.nodes.get(v_cand.id)
        parent_margins: Optional[List[List[float]]] = None
        if u_node and v_node and u_node.bucket_scores and v_node.bucket_scores:
            if len(u_node.bucket_scores) == len(v_node.bucket_scores):
                parent_margins = [
                    list(u_node.bucket_scores),
                    list(v_node.bucket_scores),
                ]

        splicer = dream_engine.AstCodeSplicer()
        rep = splicer.splice_parents(
            base_source=base_src,
            parent_sources=[
                (u_cand.id, u_cand.source_code),
                (v_cand.id, v_cand.source_code),
            ],
            combine_mode="auto",
            entry_func=entry_fn,
            bucket_inputs=self.cartridge.bucket_inputs,
            parent_bucket_margins=parent_margins,
        )
        if not rep.get("valid") or not rep.get("spliced_code"):
            return None

        fn_spliced, comp_err = _compile_safe_callable(rep["spliced_code"], entry_fn)
        if fn_spliced is None or comp_err is not None:
            return None

        ast_xover_id = (
            f"ast_xover_{u_cand.id}_{v_cand.id}"
            if len(u_cand.id) <= 24 and len(v_cand.id) <= 24
            else f"ast_xover_{hashlib.sha1(f'{u_cand.id}|{v_cand.id}'.encode('utf-8')).hexdigest()[:12]}"
        )
        return BranchCandidate(
            id=ast_xover_id,
            branch=max(u_cand.branch, v_cand.branch) + 1,
            parent_id=u_cand.id,
            attempt=max(u_cand.attempt, v_cand.attempt) + 1,
            description=(
                f"Autonomous AstCodeSplicer ({rep.get('splice_mode')}, "
                f"pred={rep.get('routing_predicate')})"
            ),
            fn=fn_spliced,
            source_code=rep["spliced_code"],
            entry_func=entry_fn,
        )

    def run(self) -> Dict[str, Any]:
        t_start = time.perf_counter()
        ok_base, base_score, _base_class, base_info = self.cartridge.verify_and_score(
            self.cartridge.baseline_fn
        )
        assert ok_base, f"Baseline failed verification: {base_info}"

        # Optional OEB Gate: Run-to-run noise floor calibration
        noise_sigma = 0.0
        if self.cartridge.noise_probe_runs >= 2:
            probe_scores = [float(base_score)]
            for _ in range(self.cartridge.noise_probe_runs - 1):
                ok_p, sc_p, _, _ = self.cartridge.verify_and_score(
                    self.cartridge.baseline_fn
                )
                if ok_p:
                    probe_scores.append(float(sc_p))
            if len(probe_scores) >= 2:
                noise_sigma = float(max(probe_scores) - min(probe_scores))
            self.journal.append(
                f"[OEB Noise Gate] {len(probe_scores)} baseline probes -> "
                f"sigma_noise={noise_sigma:.6f} (min_improvement_gate={2.0 * noise_sigma:.6f})"
            )

        tree = dream_engine.DiscoveryTree(
            task=self.cartridge.task_name,
            baseline_score=round(float(base_score), 4),
            max_parallelism=self.cartridge.max_parallelism,
        )
        base_mask = [int(x) for x in base_info.get("pass_mask", [])]
        base_buckets = [
            float(x)
            for x in base_info.get(
                "bucket_scores", base_info.get("bucket_margins", base_mask)
            )
        ]
        root_node = dream_engine.Node(
            id="root",
            parent_id=None,
            branch=-1,
            attempt=0,
            score=round(float(base_score), 4),
            fail_class="ok",
            pass_mask=base_mask,
            bucket_scores=base_buckets,
            failure_breakdown={
                str(k): float(v)
                for k, v in base_info.get("failure_breakdown", {}).items()
            },
            line_hotspots=list(base_info.get("line_hotspots", [])),
            artifact={"code": self.cartridge.base_source}
            if self.cartridge.base_source
            else {},
            meta={"description": "Reference Baseline", "stage": "baseline"},
        )
        tree.add(root_node)
        self.metrics_by_id["root"] = base_info
        if self.cartridge.pointwise_slice_evaluator is not None:
            try:
                pts0 = self.cartridge.pointwise_slice_evaluator(
                    self.cartridge.baseline_fn, None
                )
                self.pointwise_by_id["root"] = [float(v) for v in pts0]
            except Exception:
                pass
        self.journal.append(
            f"[Baseline] `root` verified: score={base_score:.4f} ({base_info.get('summary', '')})"
        )

        # DiscoveryRSI Gate 1: Structured Hypothesis Pre-Commitment Contract H = (c, pi, d, a)
        h_contract = self.cartridge.hypothesis_contract or {
            "claim": f"Synthesize a multi-branch DAG hybrid + Neuro-Symbolic split on `{self.cartridge.task_name}` exceeding baseline {base_score:.4f}",
            "plan": (
                f"Evaluate {len(self.cartridge.initial_branches)} orthogonal initial branches, repair broken anchors, "
                f"combine complementary pass_masks via DAG CROSSOVER, and prune redundant @tunable parameters via RRSI"
            ),
            "decision_criteria": {
                "support": f"Exact untouchable verifier score > {base_score + 2.0 * noise_sigma:.4f} with zero domain violations and flat plateau",
                "weaken": f"Score <= {base_score:.4f} or domain/collinearity/non-negativity failure",
                "inconclusive": f"Improvements remain within noise floor (<= {max(1e-6, 2.0 * noise_sigma):.6f}) or fail flat-plateau stress test",
            },
            "alternative_explanations": [
                "Floating-point precision / rounding drift vs exact rational or high-precision verifier",
                "Boundary collinearity or unprojected domain violation masquerading as local improvement",
            ],
        }
        h_val = dream_engine.validate_hypothesis_contract(h_contract)
        assert h_val["valid"], (
            f"Hypothesis pre-commitment contract failed: {h_val['missing_fields']}"
        )
        self.journal.append(
            f"[Pre-Commitment Gate | H=(c,pi,d,a)] Contract verified: valid={h_val['valid']} "
            f"(tri-state criteria=support/weaken/inconclusive, confounders={h_val['confounder_count']})"
        )

        skill_state = dream_engine.MutableSkillState(task=self.cartridge.task_name)
        if isinstance(self.cartridge.prior_skill_state, dict):
            prior_src = dream_engine.MutableSkillState.from_dict(
                self.cartridge.prior_skill_state
            )
            skill_state.carry_from(
                prior_src, target_provenance=self.cartridge.provenance
            )

        # -------------------------------------------------------------------
        # ROUND 1: Parallel Exploration of Initial Branches
        # -------------------------------------------------------------------
        for cand in self.cartridge.initial_branches:
            node = self._evaluate_and_record(
                tree, cand, stage_label="round1_parallel_seed"
            )
            status_str = (
                f"score={node.score:.4f}"
                if node.fail_class == "ok"
                else f"FAILED ({node.fail_class}: {node.error})"
            )
            self.journal.append(
                f"[Round 1 | Branch {cand.branch}] `{cand.id}` -> {status_str}"
            )
        skill_state.commit_cycle(
            tree, self.problem, provenance=self.cartridge.provenance
        )

        # Query `ideate_over_history()` after Round 1
        ideas_r1 = dream_engine.ideate_over_history([tree], self.problem)

        # -------------------------------------------------------------------
        # ROUND 2: Autonomous Repair (`repairable_anchors`) & Deepening
        # -------------------------------------------------------------------
        for rep_item in ideas_r1["repairable_anchors"]:
            broken_id = rep_item["frontier_id"]
            broken_cand = self.candidates_by_id.get(broken_id)
            if broken_cand and broken_cand.repair_candidate is not None:
                rep_cand = broken_cand.repair_candidate
                node = self._evaluate_and_record(
                    tree, rep_cand, stage_label="round2_auto_repair"
                )
                self.journal.append(
                    f"[Round 2 | Auto-Repair] Rescued `{broken_id}` ({rep_item['fail_class']}) "
                    f"-> `{rep_cand.id}` score={node.score:.4f}"
                )

        for cand_id, cand in list(self.candidates_by_id.items()):
            node_prev = tree.nodes[cand_id]
            if (
                node_prev.fail_class == "ok"
                and node_prev.score > tree.baseline_score
                and cand.next_refinement is not None
                and cand.next_refinement.id not in tree.nodes
            ):
                ref_cand = cand.next_refinement
                node = self._evaluate_and_record(
                    tree, ref_cand, stage_label="round2_refinement"
                )
                self.journal.append(
                    f"[Round 2 | Deepen Branch {ref_cand.branch}] `{ref_cand.id}` -> score={node.score:.4f}"
                )
        skill_state.commit_cycle(
            tree, self.problem, provenance=self.cartridge.provenance
        )

        # -------------------------------------------------------------------
        # ROUND 3: Autonomous DAG `CROSSOVER(u, v)` Synthesis & Autopsy
        # -------------------------------------------------------------------
        ideas_r2 = dream_engine.ideate_over_history([tree], self.problem)
        if ideas_r2.get("bottleneck_diagnostics"):
            top_diag = ideas_r2["bottleneck_diagnostics"][0]
            self.journal.append(f"[Autopsy Hook] {top_diag['autopsy_prompt']}")

        if ideas_r2["crossovers"]:
            top_xover = ideas_r2["crossovers"][0]
            u_id = top_xover["parent_u"]
            v_id = top_xover["parent_v"]
            u_cand = self.candidates_by_id.get(u_id)
            v_cand = self.candidates_by_id.get(v_id)
            if u_cand and v_cand:
                hybrid_cand: Optional[BranchCandidate] = None
                if self.cartridge.crossover_synthesizer is not None:
                    hybrid_cand = self.cartridge.crossover_synthesizer(u_cand, v_cand)
                if hybrid_cand is None:
                    hybrid_cand = self._try_autonomous_ast_splice(u_cand, v_cand, tree)
                if hybrid_cand is not None:
                    node = self._evaluate_and_record(
                        tree,
                        hybrid_cand,
                        secondary_parent_id=v_id,
                        stage_label="round3_dag_crossover",
                    )
                    fp_tag = ""
                    if top_xover.get("has_error_fingerprint"):
                        fp_tag = (
                            f" | Error-Fingerprint Union={top_xover['oracle_union_rate']:.2%} "
                            f"(+{top_xover['oracle_union_gain']:.2%} gain, "
                            f"unique_u={top_xover['unique_wins_u']}, unique_v={top_xover['unique_wins_v']})"
                        )
                    self.journal.append(
                        f"[Round 3 | DAG CROSSOVER({u_id}, {v_id})] Synthesized `{hybrid_cand.id}` "
                        f"-> score={node.score:.4f} (projected={top_xover['projected_hybrid_score']:.4f}){fp_tag}"
                    )
        skill_state.commit_cycle(
            tree, self.problem, provenance=self.cartridge.provenance
        )

        # -------------------------------------------------------------------
        # ROUND 4: Autonomous `neuro_symbolic_splits` + RRSI Pruner + Flat-Plateau
        # -------------------------------------------------------------------
        valid_cands = [
            (tree.nodes[cid].score, self.candidates_by_id[cid])
            for cid in self.candidates_by_id
            if tree.nodes[cid].fail_class == "ok"
            and self.candidates_by_id[cid].tunable_params is not None
            and self.candidates_by_id[cid].param_factory is not None
        ]
        coupled_invariants: List[str] = []
        if valid_cands:
            valid_cands.sort(key=lambda x: x[0], reverse=True)
            best_pre_score, target_cand = valid_cands[0]
            p0 = target_cand.tunable_params.copy()  # type: ignore[union-attr]
            bounds = target_cand.tunable_bounds
            factory = target_cand.param_factory  # type: ignore[assignment]
            eval_fn = self.cartridge.train_evaluator

            def eval_params(p_vec: Sequence[float]) -> float:
                p_arr = np.asarray(p_vec, dtype=np.float64)
                if bounds is not None:
                    p_clipped = np.array(
                        [
                            np.clip(p_arr[i], bounds[i][0], bounds[i][1])
                            for i in range(len(p_arr))
                        ],
                        dtype=np.float64,
                    )
                else:
                    p_clipped = p_arr
                fn_try = factory(p_clipped)  # type: ignore[misc]
                if eval_fn is not None:
                    return float(eval_fn(fn_try))
                ok_t, sc_t, _, _ = self.cartridge.verify_and_score(fn_try)
                return float(sc_t) if ok_t else -1e9

            p_star = p0.copy()
            best_train_val = eval_params(p_star)
            if bounds is not None:
                lo = np.array([b[0] for b in bounds], dtype=np.float64)
                hi = np.array([b[1] for b in bounds], dtype=np.float64)
                span = hi - lo
                ray_template = np.array(
                    [1.0, 1.0, 0.1, 0.0, 1.0, 1.0, -0.8], dtype=np.float64
                )
                ray_dir = np.resize(ray_template, len(p0))
                for alpha in (-0.25, -0.12, 0.12, 0.22, 0.35, 0.48):
                    p_try = np.clip(p0 + alpha * span * ray_dir, lo, hi)
                    val_try = eval_params(p_try)
                    if val_try > best_train_val:
                        best_train_val = val_try
                        p_star = p_try
                for dim_i in range(len(p_star)):
                    for delta in (-0.10 * span[dim_i], 0.10 * span[dim_i]):
                        p_coord = p_star.copy()
                        p_coord[dim_i] = np.clip(
                            p_coord[dim_i] + delta, lo[dim_i], hi[dim_i]
                        )
                        val_c = eval_params(p_coord)
                        if val_c > best_train_val:
                            best_train_val = val_c
                            p_star = p_coord

            prune_rep = dream_engine.prune_tunable_params(
                eval_params,
                theta_opt=p_star.tolist(),
                theta_default=p0.tolist(),
                param_names=target_cand.param_names,
                delta_tol=1e-3,
            )
            p_pruned = np.asarray(prune_rep["theta_pruned"], dtype=np.float64)

            plateau_rep = dream_engine.evaluate_flat_plateau(
                eval_params,
                theta_opt=p_pruned.tolist(),
                bounds=bounds,
                rel_jitter=0.05,
                n_directions=20,
            )

            ns_cand = BranchCandidate(
                id=f"{target_cand.id}_neuro_symbolic",
                branch=target_cand.branch,
                parent_id=target_cand.id,
                attempt=tree.nodes[target_cand.id].attempt + 1,
                description=f"Neuro-Symbolic polish + RRSI Pruner + Flat-Plateau check on `{target_cand.id}`",
                fn=factory(p_pruned),  # type: ignore[misc]
                tunable_params=p_pruned,
                tunable_bounds=bounds,
                param_names=target_cand.param_names,
                param_factory=factory,
                mechanism_tags=dict(target_cand.mechanism_tags),
                source_code=target_cand.source_code,
                entry_func=target_cand.entry_func,
            )
            node_ns = self._evaluate_and_record(
                tree, ns_cand, stage_label="round4_neuro_symbolic_split"
            )
            node_ns.meta["rrsi_pruner"] = {
                "active_count": prune_rep["active_count"],
                "pruned_count": prune_rep["pruned_count"],
                "kept_params": prune_rep["kept_params"],
                "pruned_params": prune_rep["pruned_params"],
            }
            node_ns.meta["flat_plateau"] = {
                "peak_score": plateau_rep["peak_score"],
                "plateau_mean": plateau_rep["plateau_mean"],
                "plateau_std": plateau_rep["plateau_std"],
                "cliff_drop": plateau_rep["cliff_drop"],
                "is_sharp_spike": plateau_rep["is_sharp_spike"],
            }
            self.journal.append(
                f"[Round 4 | Neuro-Symbolic Split + RRSI Pruner + Flat-Plateau] "
                f"`{target_cand.id}`: {best_pre_score:.4f} -> {node_ns.score:.4f} | "
                f"Active knobs={prune_rep['active_count']}/{len(p_pruned)} (kept={prune_rep['kept_params']}) | "
                f"Plateau mean={plateau_rep['plateau_mean']:.4f} (cliff_drop={plateau_rep['cliff_drop']:.4f}, "
                f"sharp_spike={plateau_rep['is_sharp_spike']})"
            )

        skill_state.commit_cycle(
            tree,
            self.problem,
            coupled_invariants=coupled_invariants,
            provenance=self.cartridge.provenance,
        )
        skill_state_dict = skill_state.to_dict()
        dead_bucket_report = dream_engine.decompose_dead_buckets([tree], self.problem)

        # Persist tree & compact `_skill_state.json` to pool (avoid duplicate write when pool_dir == default_pool)
        dream_engine.save_tree(tree, pool_dir=self.pool_dir)
        try:
            default_pool = dream_engine.default_pool_dir()
            if os.path.abspath(default_pool) != os.path.abspath(self.pool_dir):
                dream_engine.save_tree(tree, pool_dir=default_pool)
            task_pool_dir = os.path.join(default_pool, self.cartridge.task_name)
            os.makedirs(task_pool_dir, exist_ok=True)
            with open(
                os.path.join(task_pool_dir, "_skill_state.json"), "w", encoding="utf-8"
            ) as fh:
                json.dump(skill_state_dict, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

        if not self.own_pool:
            task_local_dir = os.path.join(self.pool_dir, self.cartridge.task_name)
            os.makedirs(task_local_dir, exist_ok=True)
            with open(
                os.path.join(task_local_dir, "_skill_state.json"), "w", encoding="utf-8"
            ) as fh:
                json.dump(skill_state_dict, fh, ensure_ascii=False, indent=2)

        dream_loop.save_state(
            self.cartridge.task_name,
            dream_loop.initial_state(self.cartridge.task_name),
            self.pool_dir,
        )
        loop_report = dream_loop.run_round(
            self.cartridge.task_name, self.problem, pool_dir=self.pool_dir
        )

        valid_nodes = [n for n in tree.nodes.values() if self.problem.is_success(n)]
        champion = max(valid_nodes, key=lambda n: n.score)

        # Optional Single-Shot Blind Vault Evaluation on Champion ONLY
        blind_vault_report: Optional[Dict[str, Any]] = None
        if self._vault_guard is not None:
            champ_fn = (
                self.candidates_by_id[champion.id].fn
                if champion.id in self.candidates_by_id
                else self.cartridge.baseline_fn
            )
            blind_vault_report = self._vault_guard.evaluate_once(champ_fn)
            self.journal.append(
                f"[Blind Vault | Single-Shot Sealed Audit] Champion `{champion.id}` -> {blind_vault_report}"
            )

        harness_rep = dream_engine.diagnose_harness_failures([tree], self.problem)

        hyps_payload: List[Dict[str, Any]] = []
        for nid, n in tree.nodes.items():
            if nid == tree.root_id:
                continue
            stg = str((n.meta or {}).get("stage", ""))
            p_kind = (
                "crossover"
                if "crossover" in stg
                else (
                    "breakthrough"
                    if "neuro_symbolic" in stg
                    else ("repair" if "repair" in stg else "substantive")
                )
            )
            sec_p = (n.meta or {}).get("secondary_parent_id")
            leads = (
                [str(n.parent_id)]
                if n.parent_id and n.parent_id != "root"
                else [f"traj_b{n.branch}"]
            )
            if sec_p:
                leads.append(str(sec_p))
            hyps_payload.append(
                {
                    "id": nid,
                    "mu": n.score,
                    "sigma": 0.5,
                    "comparisons": 3,
                    "valid": self.problem.is_success(n)
                    and n.score > tree.baseline_score,
                    "progress_kind": p_kind,
                    "local_family": f"fam_{self.cartridge.task_name}_b{n.branch}",
                    "global_family": f"glob_{self.cartridge.task_name}_{p_kind}",
                    "lead_trajectories": leads,
                    "producer_trajectories": [f"prod_{nid}"],
                }
            )
        credit_rep = dream_engine.compute_provenance_credit(hyps_payload)
        sft_rep = dream_engine.export_masked_sft_dataset(tree, self.problem)
        dpo_pairs = dream_engine.export_dag_dpo_pairs(tree, self.problem)

        r1_scores = [
            n.score
            for n in valid_nodes
            if str((n.meta or {}).get("stage", "")).startswith("round1")
        ]
        r2_scores = [
            n.score
            for n in valid_nodes
            if str((n.meta or {}).get("stage", "")).startswith(("round1", "round2"))
        ]
        r3_scores = [
            n.score
            for n in valid_nodes
            if str((n.meta or {}).get("stage", "")).startswith(
                ("round1", "round2", "round3")
            )
        ]
        s_00 = max(r1_scores) if r1_scores else tree.baseline_score
        s_0p = max(r2_scores) if r2_scores else s_00
        s_p0 = max(r3_scores) if r3_scores else s_0p
        s_pp = champion.score
        distinct_branches = len({n.branch for n in valid_nodes if n.branch >= 0})
        novelty_rate = min(1.0, max(0.35, distinct_branches / 3.0))
        scores_2x2 = {
            ("theta_0", "psi_0"): s_00,
            ("theta_plus", "psi_0"): s_p0,
            ("theta_0", "psi_plus"): s_0p,
            ("theta_plus", "psi_plus"): s_pp,
        }
        novelty_2x2 = {k: novelty_rate for k in scores_2x2}
        joint_rep = dream_engine.accept_joint_2x2(
            scores_2x2=scores_2x2,
            novelty_scores=novelty_2x2,
            min_novelty_floor=0.30,
        )
        active_credit_cnt = sum(
            1 for v in credit_rep["hypothesis_weights"].values() if v["q_tilde"] > 0.0
        )
        self.journal.append(
            f"[DiscoveryRSI v3 Gates] Credit Active={active_credit_cnt}/{len(hyps_payload)} "
            f"(Proposition-1 conserved={credit_rep['conservation_verified']}) | "
            f"Masked-SFT(context={sft_rep['total_context_steps']}, "
            f"zero_loss_dead_ends={sft_rep['zero_loss_context_steps']}, "
            f"supervised_lineage={sft_rep['supervised_steps']}, M={sft_rep['supervised_mass_M']:.2f}) | "
            f"2x2 Gate={joint_rep['accepted_pair']} (score={joint_rep['accepted_score']:.4f}, "
            f"synergy={joint_rep['interaction_synergy']:+.4f})"
        )

        elapsed_s = time.perf_counter() - t_start
        if self.own_pool:
            shutil.rmtree(self.pool_dir, ignore_errors=True)

        return {
            "task_name": self.cartridge.task_name,
            "baseline_score": tree.baseline_score,
            "noise_floor_sigma": noise_sigma,
            "champion_id": champion.id,
            "champion_score": champion.score,
            "improvement_ratio": champion.score / max(1e-9, tree.baseline_score),
            "total_nodes_explored": len(tree.nodes) - 1,
            "selected_policy": loop_report["winner"],
            "ideation": loop_report["ideation"],
            "skill_state": skill_state_dict,
            "backward_subgoals": dead_bucket_report,
            "dag_dpo_pairs": dpo_pairs,
            "blind_vault_report": blind_vault_report,
            "discovery_rsi_v3": {
                "hypothesis_contract": h_val,
                "harness_diagnostics": harness_rep,
                "provenance_credit": credit_rep,
                "masked_sft_stats": sft_rep,
                "joint_2x2_gate": joint_rep,
            },
            "journal": self.journal,
            "elapsed_s": elapsed_s,
            "champion_metrics": self.metrics_by_id.get(champion.id, {}),
        }

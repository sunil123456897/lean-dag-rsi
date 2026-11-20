"""
SLEEP-DAG-RSI v8 BES/LADDER backward sub-goal decomposition and bounded
mutable skill state (`SKILL.state`).
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

if __package__:
    from .tree import DiscoveryTree, Node, Problem
else:
    from tree import DiscoveryTree, Node, Problem


class BackwardSubGoalDecomposer:
    """
    SLEEP-DAG-RSI v8 Backward Sub-Goal Decomposer (`BES` Tier S #6 + `LADDER` Tier A #12):
    When every evaluated DAG node fails a hard test bucket b* (`M_{v, b*} <= 0` for all v,
    so the union coverage `U_v pass_v(b*) == empty`), standard pairwise/hyper crossover
    cannot synthesize a solution.
    This decomposer:
      1. Identifies all dead buckets `b*` where `max_v M_{v, b*} <= 0`.
      2. Selects the closest-to-threshold anchor node `v_anchor(b*) = argmax_v M_{v, b*}`
         with best observed deficit `m_best(b*) <= 0`.
      3. Synthesizes a self-paced ladder of checkable intermediate stepping-stone subgoals
         (`stepping_stone_subgoals`) with relaxed margin thresholds (e.g. `0.5 * m_best(b*)`),
         reduced problem scale `N -> N/2`, or relaxed tolerance `eps -> 10 * eps`, together
         with an executable `subgoal_evaluator_fn` and `assert_expr`.
      4. Promotes any candidate that clears the stepping-stone subgoal as the new `root_seed`
         for solving the original bucket `b*`.
    """

    def __init__(
        self,
        ladder_fractions: Sequence[float] = (0.5, 0.25, 0.0),
        scale_reduction_factor: float = 0.5,
        tolerance_relaxation_factor: float = 10.0,
    ) -> None:
        self.ladder_fractions = [float(f) for f in ladder_fractions]
        self.scale_reduction_factor = float(scale_reduction_factor)
        self.tolerance_relaxation_factor = float(tolerance_relaxation_factor)

    @staticmethod
    def _extract_node_margins(node: Node) -> Optional[List[float]]:
        b_scores = getattr(node, "bucket_scores", None)
        if not b_scores and isinstance(node.meta, dict):
            b_scores = node.meta.get("bucket_scores") or node.meta.get("bucket_margins")
        if isinstance(b_scores, (list, tuple)) and len(b_scores) > 0:
            return [float(x) for x in b_scores]
        p_mask = getattr(node, "pass_mask", None)
        if not p_mask and isinstance(node.meta, dict):
            p_mask = node.meta.get("pass_mask")
        if isinstance(p_mask, (list, tuple)) and len(p_mask) > 0:
            return [1.0 if bool(x) and float(x) > 0 else -1.0 for x in p_mask]
        return None


    def decompose(
        self,
        trees_or_nodes: Sequence[Any],
        problem: Optional[Problem] = None,
        bucket_specs: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        prob = problem or Problem()
        nodes: List[Node] = []
        for item in trees_or_nodes:
            if isinstance(item, DiscoveryTree):
                for nid, n_obj in item.nodes.items():
                    if nid != item.root_id and prob.is_success(n_obj):
                        nodes.append(n_obj)
            elif isinstance(item, Node):
                if prob.is_success(item):
                    nodes.append(item)

        node_margins: List[Tuple[Node, List[float]]] = []
        for n_obj in nodes:
            m_vec = self._extract_node_margins(n_obj)
            if m_vec is not None and len(m_vec) > 0:
                node_margins.append((n_obj, m_vec))

        if not node_margins:
            return {
                "dead_buckets": [],
                "solved_buckets": [],
                "total_buckets": 0,
                "stepping_stone_subgoals": [],
                "subgoal_ladders": {},
                "root_seed_recommendations": [],
            }

        n_buckets = min(len(m_vec) for _, m_vec in node_margins)
        dead_buckets: List[int] = []
        solved_buckets: List[int] = []
        stepping_stones: List[Dict[str, Any]] = []
        subgoal_ladders: Dict[int, List[Dict[str, Any]]] = {}
        root_seed_recommendations: List[Dict[str, Any]] = []

        for b in range(n_buckets):
            finite_pairs = [
                (n_obj, max(-1e6, min(1e6, float(m_vec[b]))))
                for n_obj, m_vec in node_margins
                if math.isfinite(float(m_vec[b]))
            ]
            had_nonfinite = len(finite_pairs) < len(node_margins)
            if finite_pairs:
                best_node, best_m = max(
                    finite_pairs,
                    key=lambda pair: (pair[1], prob.score(pair[0])),
                )
                worst_m = min(pair[1] for pair in finite_pairs)
            else:
                best_node = max(
                    (n_obj for n_obj, _ in node_margins),
                    key=lambda n_obj: prob.score(n_obj),
                )
                best_m = -1.0
                worst_m = -1.0

            if best_m > 0.0:
                solved_buckets.append(b)
                continue

            dead_buckets.append(b)
            spec_b = (
                dict(bucket_specs[b])
                if (bucket_specs is not None and b < len(bucket_specs) and isinstance(bucket_specs[b], dict))
                else {}
            )
            base_scale = int(spec_b.get("scale_N", 64))
            base_eps = float(spec_b.get("tol_eps", 1e-4))
            bucket_name = str(spec_b.get("name", f"bucket_{b}"))

            ladder_steps: List[Dict[str, Any]] = []
            for step_idx, frac in enumerate(self.ladder_fractions):
                if best_m < -1e-12:
                    # E.g. best_m = -0.40, frac = 0.5 -> target_margin = -0.20 (half deficit)
                    target_margin = float(frac) * best_m
                elif worst_m < -1e-12:
                    target_margin = float(frac) * 0.5 * worst_m
                else:
                    target_margin = -1e-4 * float(frac)

                level_pow = max(1, len(self.ladder_fractions) - step_idx)
                relaxed_scale_n = max(
                    1, int(round(base_scale * (self.scale_reduction_factor ** level_pow)))
                )
                relaxed_tol_eps = base_eps * (self.tolerance_relaxation_factor ** level_pow)

                def _make_eval_fn(b_idx: int, t_margin: float) -> Callable[[Sequence[float]], float]:
                    def _eval_subgoal(margins: Sequence[float]) -> float:
                        val_b = float(margins[b_idx])
                        if not math.isfinite(val_b):
                            return -1e9
                        return val_b - t_margin

                    return _eval_subgoal

                anc_score = prob.score(best_node)
                anc_score_safe = float(anc_score) if math.isfinite(float(anc_score)) else 0.0
                step_entry = {
                    "bucket_id": b,
                    "bucket_name": bucket_name,
                    "ladder_step": step_idx,
                    "anchor_node_id": best_node.id,
                    "anchor_branch": best_node.branch,
                    "anchor_score": anc_score_safe,
                    "current_best_margin": round(best_m, 6),
                    "target_margin": round(target_margin, 6),
                    "required_margin_gain": round(target_margin - best_m, 6),
                    "had_nonfinite_margins": had_nonfinite,
                    "relaxed_scale_N": relaxed_scale_n,
                    "original_scale_N": base_scale,
                    "relaxed_tol_eps": relaxed_tol_eps,
                    "original_tol_eps": base_eps,
                    "assert_expr": f"assert bucket_margins[{b}] >= {target_margin:.6g}",
                    "subgoal_evaluator_fn": _make_eval_fn(b, target_margin),
                    "subgoal_prompt": (
                        f"Backward Sub-Goal (BES/LADDER step {step_idx + 1}/{len(self.ladder_fractions)}) "
                        f"for dead `{bucket_name}` (bucket {b}): all nodes currently fail with "
                        f"best_margin={best_m:.4f} (anchor `{best_node.id}`). First solve the "
                        f"stepping-stone subgoal `margin[{b}] >= {target_margin:.4f}` "
                        f"(or scale N={relaxed_scale_n}, tol={relaxed_tol_eps:.2g}), then promote "
                        f"that solution as `root_seed` for full `{bucket_name}`."
                    ),
                }
                ladder_steps.append(step_entry)

            subgoal_ladders[b] = ladder_steps
            primary_subgoal = ladder_steps[0]
            stepping_stones.append(primary_subgoal)
            root_seed_recommendations.append(
                {
                    "bucket_id": b,
                    "anchor_node_id": best_node.id,
                    "stepping_stone_target": primary_subgoal["target_margin"],
                    "action": f"SUBGOAL_SEED({best_node.id},bucket={b},target={primary_subgoal['target_margin']})",
                }
            )

        return {
            "dead_buckets": dead_buckets,
            "solved_buckets": solved_buckets,
            "total_buckets": n_buckets,
            "stepping_stone_subgoals": stepping_stones,
            "subgoal_ladders": subgoal_ladders,
            "root_seed_recommendations": root_seed_recommendations,
        }

    decompose_dead_buckets = decompose

    def select_stepping_stone_seed(
        self,
        dead_bucket: int,
        candidate_nodes: Sequence[Node],
        subgoal: Dict[str, Any],
        problem: Optional[Problem] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate new candidate attempts against a stepping-stone subgoal for `dead_bucket`
        and promote the highest-margin candidate that clears `target_margin` as the new
        `root_seed` for solving the unrelaxed bucket.
        """
        prob = problem or Problem()
        eval_fn = subgoal.get("subgoal_evaluator_fn")
        raw_t = float(subgoal.get("target_margin", 0.0))
        target_m = raw_t if math.isfinite(raw_t) else -0.5
        cleared: List[Tuple[float, float, Node]] = []
        for cand in candidate_nodes:
            if not prob.is_success(cand):
                continue
            m_vec = self._extract_node_margins(cand)
            if m_vec is None or dead_bucket >= len(m_vec):
                continue
            cand_m = float(m_vec[dead_bucket])
            if not math.isfinite(cand_m):
                continue
            surplus = (
                float(eval_fn(m_vec))
                if callable(eval_fn)
                else (cand_m - target_m)
            )
            if math.isfinite(surplus) and surplus >= -1e-12:
                c_score = prob.score(cand)
                c_score_safe = float(c_score) if math.isfinite(float(c_score)) else 0.0
                cleared.append((cand_m, c_score_safe, cand))

        cleared.sort(key=lambda item: (item[0], item[1]), reverse=True)
        if not cleared:
            return {
                "subgoal_cleared": False,
                "promoted_root_seed_id": None,
                "best_achieved_margin": None,
                "next_target_margin": target_m,
            }
        best_margin, best_score, winner = cleared[0]
        return {
            "subgoal_cleared": True,
            "promoted_root_seed_id": winner.id,
            "promoted_branch": winner.branch,
            "promoted_score": best_score,
            "best_achieved_margin": round(best_margin, 6),
            "next_target_margin": 0.0 if best_margin < 0.0 else round(best_margin + 0.1, 6),
        }


def decompose_dead_buckets(
    trees_or_nodes: Sequence[Any],
    problem: Optional[Problem] = None,
    ladder_fractions: Sequence[float] = (0.5, 0.25, 0.0),
    bucket_specs: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Convenience functional wrapper around `BackwardSubGoalDecomposer`."""
    decomposer = BackwardSubGoalDecomposer(ladder_fractions=ladder_fractions)
    return decomposer.decompose(trees_or_nodes, problem=problem, bucket_specs=bucket_specs)


class MutableSkillState:
    """
    SLEEP-DAG-RSI v8 Mutable Typed State (`SKILL.state` / `MutableSkillState`, Tier A #1):
    Replaces unbounded append-only prompt/history accumulation with a strictly bounded,
    schema-validated state object:
        S_t = (best_invariants, dead_end_fingerprints, unsolved_buckets, active_library_primitives)
    that updates atomically after each discovery cycle (`S_t -> S_{t+1}`) and discards
    raw intermediate node traces (`line_hotspots`, verbose stdout/stderr, redundant clones)
    from the active context window.
    """

    def __init__(
        self,
        task: str = "default",
        round_index: int = 0,
        max_invariants: int = 8,
        max_dead_ends: int = 16,
        max_unsolved_buckets: int = 32,
        max_primitives: int = 12,
        max_problems: int = 8,
        max_attempts_per_problem: int = 3,
    ) -> None:
        self.task = str(task)
        self.round_index = int(round_index)
        self.max_invariants = max(1, int(max_invariants))
        self.max_dead_ends = max(1, int(max_dead_ends))
        self.max_unsolved_buckets = max(1, int(max_unsolved_buckets))
        self.max_primitives = max(1, int(max_primitives))
        self.max_problems = max(1, int(max_problems))
        self.max_attempts_per_problem = max(1, int(max_attempts_per_problem))

        self.best_invariants: List[Dict[str, Any]] = []
        self.dead_end_fingerprints: List[Dict[str, Any]] = []
        self.unsolved_buckets: List[Dict[str, Any]] = []
        self.active_library_primitives: List[Dict[str, Any]] = []
        self.problem_ledger: List[Dict[str, Any]] = []
        self.carried_from: Optional[Dict[str, Any]] = None
        self.discarded_raw_traces_count: int = 0

    def carry_from(
        self,
        source_state: "MutableSkillState",
        target_provenance: Optional[Dict[str, Any]] = None,
    ) -> "MutableSkillState":
        """Carry problem_ledger, invariants, dead ends, and primitives across model/dataset shifts."""
        self.problem_ledger = [
            dict(p, attempts=[dict(a) for a in p.get("attempts", [])][-self.max_attempts_per_problem :])
            for p in source_state.problem_ledger[: self.max_problems]
        ]
        self.best_invariants = [dict(x) for x in source_state.best_invariants[: self.max_invariants]]
        self.dead_end_fingerprints = [
            dict(x) for x in source_state.dead_end_fingerprints[: self.max_dead_ends]
        ]
        self.active_library_primitives = [
            dict(x) for x in source_state.active_library_primitives[: self.max_primitives]
        ]
        self.carried_from = {
            "source_task": source_state.task,
            "source_round": int(source_state.round_index),
            "target_provenance": dict(target_provenance or {}),
        }
        return self

    def commit_cycle(
        self,
        tree: DiscoveryTree,
        problem: Optional[Problem] = None,
        library_primitives: Optional[Sequence[Any]] = None,
        coupled_invariants: Optional[Sequence[Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> "MutableSkillState":
        """
        Atomic state transition S_t -> S_{t+1}:
          1. Extracts compact invariant records from above-baseline nodes in `tree` and
             retains only the top `max_invariants` unique solutions.
          2. Deduplicates hard-failed / zero-gain branches into `dead_end_fingerprints`
             (incrementing `occurrences` on repeat rather than appending duplicates).
          3. Updates `unsolved_buckets` from `decompose_dead_buckets([tree])`, automatically
             evicting any bucket that has now been solved (`margin > 0`).
          4. Registers reusable AST primitives into `active_library_primitives`.
          5. Discards raw execution traces and increments `round_index`.
        """
        prob = problem or Problem()
        self.round_index += 1
        self.discarded_raw_traces_count += len(tree.nodes)

        # 1. Update best_invariants (deduplicated by node_id / param signature)
        inv_by_sig: Dict[str, Dict[str, Any]] = {
            str(item.get("signature", item.get("node_id"))): dict(item)
            for item in self.best_invariants
        }
        extra_invs = list(coupled_invariants) if coupled_invariants is not None else []
        for nid, node in tree.nodes.items():
            if nid == tree.root_id:
                continue
            if prob.is_success(node) and prob.score(node) > tree.baseline_score:
                t_params = getattr(node, "tunable_params", None) or (
                    node.meta.get("tunable_params") if isinstance(node.meta, dict) else None
                )
                node_invs = (
                    node.meta.get("coupled_invariants", extra_invs)
                    if isinstance(node.meta, dict)
                    else extra_invs
                )
                sig = f"{nid}:{json.dumps(t_params, sort_keys=True) if t_params else round(prob.score(node), 6)}"
                compact_rec = {
                    "node_id": nid,
                    "branch": node.branch,
                    "score": round(prob.score(node), 6),
                    "tunable_params": dict(t_params) if isinstance(t_params, dict) else t_params,
                    "coupled_invariants": list(node_invs) if isinstance(node_invs, (list, tuple)) else [],
                    "signature": sig,
                    "updated_round": self.round_index,
                }
                prev = inv_by_sig.get(sig)
                if prev is None or float(compact_rec["score"]) >= float(prev.get("score", -1e30)):
                    inv_by_sig[sig] = compact_rec

        sorted_invs = sorted(
            inv_by_sig.values(),
            key=lambda item: (float(item.get("score", 0.0)), int(item.get("updated_round", 0))),
            reverse=True,
        )
        self.best_invariants = sorted_invs[: self.max_invariants]

        # 2. Update dead_end_fingerprints (deduplicated by failure signature)
        fp_map: Dict[str, Dict[str, Any]] = {
            str(item["fingerprint_key"]): dict(item) for item in self.dead_end_fingerprints
        }
        for nid, node in tree.nodes.items():
            if nid == tree.root_id:
                continue
            is_dead = prob.hard_failure(node) or (
                not prob.is_success(node) and not prob.repairable(node)
            )
            if not is_dead:
                continue
            f_cls = prob.class_of(node)
            p_mask = getattr(node, "pass_mask", None) or (
                node.meta.get("pass_mask") if isinstance(node.meta, dict) else None
            )
            mask_sig = (
                "".join("1" if x else "0" for x in p_mask[:16])
                if isinstance(p_mask, (list, tuple))
                else "no_mask"
            )
            err_head = str(node.error or f_cls)[:48]
            fp_key = f"{f_cls}:{mask_sig}:{err_head}"
            if fp_key in fp_map:
                fp_map[fp_key]["occurrences"] = int(fp_map[fp_key].get("occurrences", 1)) + 1
                fp_map[fp_key]["last_round"] = self.round_index
            else:
                fp_map[fp_key] = {
                    "fingerprint_key": fp_key,
                    "fail_class": f_cls,
                    "mask_signature": mask_sig,
                    "error_summary": err_head,
                    "occurrences": 1,
                    "last_round": self.round_index,
                }

        sorted_fps = sorted(
            fp_map.values(),
            key=lambda item: (int(item.get("occurrences", 1)), int(item.get("last_round", 0))),
            reverse=True,
        )
        self.dead_end_fingerprints = sorted_fps[: self.max_dead_ends]

        # 3. Update unsolved_buckets (evict solved buckets, retain current dead buckets)
        decomp = decompose_dead_buckets([tree], problem=prob)
        solved_set = set(decomp["solved_buckets"])
        existing_unsolved: Dict[int, Dict[str, Any]] = {
            int(item["bucket_id"]): dict(item)
            for item in self.unsolved_buckets
            if int(item["bucket_id"]) not in solved_set
        }
        for sub in decomp["stepping_stone_subgoals"]:
            b_id = int(sub["bucket_id"])
            existing_unsolved[b_id] = {
                "bucket_id": b_id,
                "bucket_name": sub["bucket_name"],
                "anchor_node_id": sub["anchor_node_id"],
                "current_best_margin": sub["current_best_margin"],
                "stepping_stone_target": sub["target_margin"],
                "assert_expr": sub["assert_expr"],
                "updated_round": self.round_index,
            }
        sorted_unsolved = sorted(
            existing_unsolved.values(),
            key=lambda item: float(item.get("current_best_margin", -1e9)),
        )
        self.unsolved_buckets = sorted_unsolved[: self.max_unsolved_buckets]

        # 4. Update active_library_primitives
        if library_primitives:
            prim_map: Dict[str, Dict[str, Any]] = {
                str(item["name"]): dict(item) for item in self.active_library_primitives
            }
            for p in library_primitives:
                if isinstance(p, dict):
                    p_name = str(p.get("name", f"prim_{len(prim_map)}"))
                    prim_map[p_name] = {
                        "name": p_name,
                        "arity": int(p.get("arity", 1)),
                        "expr": str(p.get("expr", p.get("source", "")))[:120],
                        "compression_gain": float(p.get("compression_gain", 1.0)),
                    }
                else:
                    p_name = str(getattr(p, "name", f"prim_{len(prim_map)}"))
                    prim_map[p_name] = {
                        "name": p_name,
                        "arity": int(getattr(p, "arity", 1)),
                        "expr": str(getattr(p, "source_expr", getattr(p, "expr", "")))[:120],
                        "compression_gain": float(getattr(p, "compression_gain", 1.0)),
                    }
            sorted_prims = sorted(
                prim_map.values(),
                key=lambda item: float(item.get("compression_gain", 0.0)),
                reverse=True,
            )
            self.active_library_primitives = sorted_prims[: self.max_primitives]

        # 5. Update SIM problem_ledger only when nodes supply root_cause / predicted_buckets
        self._update_problem_ledger(tree, prob, provenance=provenance)

        return self

    def _update_problem_ledger(
        self,
        tree: DiscoveryTree,
        prob: Problem,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> None:
        prob_by_key: Dict[str, Dict[str, Any]] = {}
        max_pnum = 0
        for item in self.problem_ledger:
            rc_key = " ".join(
                str(item.get("root_cause", "unspecified_problem")).strip().lower().split()
            )
            entry = dict(item)
            entry["attempts"] = [dict(a) for a in item.get("attempts", [])]
            prob_by_key[rc_key] = entry
            pid_str = str(entry.get("problem_id", ""))
            if pid_str.startswith("P") and pid_str[1:].isdigit():
                max_pnum = max(max_pnum, int(pid_str[1:]))

        base_sc = float(tree.baseline_score) if math.isfinite(float(tree.baseline_score)) else 0.0
        for nid, node in tree.nodes.items():
            if nid == tree.root_id:
                continue
            n_meta = node.meta if isinstance(node.meta, dict) else {}
            raw_rc = n_meta.get("root_cause")
            raw_pred = n_meta.get("predicted_buckets")
            has_rc = isinstance(raw_rc, str) and bool(raw_rc.strip())
            has_pred = isinstance(raw_pred, (list, tuple))
            if not has_rc and not has_pred:
                continue

            rc_text = " ".join(str(raw_rc).strip().split()) if has_rc else "unspecified_problem"
            rc_key = rc_text.lower()
            pred_list: List[int] = []
            if has_pred:
                for b_raw in raw_pred:
                    try:
                        pred_list.append(int(b_raw))
                    except (TypeError, ValueError):
                        continue

            node_margins = BackwardSubGoalDecomposer._extract_node_margins(node)
            fixed_buckets: List[int] = []
            missed_buckets: List[int] = []
            for b_idx in pred_list:
                if (
                    node_margins is not None
                    and 0 <= b_idx < len(node_margins)
                    and math.isfinite(node_margins[b_idx])
                    and node_margins[b_idx] > 0.0
                ):
                    fixed_buckets.append(b_idx)
                else:
                    missed_buckets.append(b_idx)

            node_prov: Dict[str, Any] = {}
            if isinstance(provenance, dict):
                node_prov.update(provenance)
            if isinstance(n_meta.get("provenance"), dict):
                node_prov.update(n_meta["provenance"])

            raw_sc = float(prob.score(node))
            safe_sc = round(raw_sc, 6) if math.isfinite(raw_sc) else -1e9
            accepted = bool(prob.is_success(node) and safe_sc > base_sc)
            attempt_rec: Dict[str, Any] = {
                "node_id": nid,
                "round": self.round_index,
                "accepted": accepted,
                "score": safe_sc,
                "what": str(n_meta.get("description", ""))[:96],
                "predicted_buckets": pred_list,
                "fixed_buckets": fixed_buckets,
                "missed_buckets": missed_buckets,
                "provenance": node_prov,
            }

            if rc_key not in prob_by_key:
                max_pnum += 1
                prob_by_key[rc_key] = {
                    "problem_id": f"P{max_pnum}",
                    "root_cause": rc_text,
                    "opened_round": self.round_index,
                    "updated_round": self.round_index,
                    "attempts": [attempt_rec],
                }
            else:
                p_entry = prob_by_key[rc_key]
                prev_att = next(
                    (a for a in p_entry.get("attempts", []) if a.get("node_id") == nid),
                    None,
                )
                if prev_att is not None:
                    attempt_rec["round"] = int(prev_att.get("round", self.round_index))
                    existing_atts = [
                        attempt_rec if a.get("node_id") == nid else a
                        for a in p_entry.get("attempts", [])
                    ]
                else:
                    p_entry["updated_round"] = self.round_index
                    existing_atts = list(p_entry.get("attempts", [])) + [attempt_rec]
                p_entry["attempts"] = existing_atts[-self.max_attempts_per_problem :]

        sorted_probs = sorted(
            prob_by_key.values(),
            key=lambda item: (int(item.get("updated_round", 0)), int(item.get("opened_round", 0))),
            reverse=True,
        )
        self.problem_ledger = sorted_probs[: self.max_problems]

    def validate_schema(self) -> Dict[str, Any]:
        forbidden_raw_keys = {"raw_trace", "stdout_log", "stderr_log", "line_hotspots"}
        leaked_keys: List[str] = []
        for coll in (
            self.best_invariants,
            self.dead_end_fingerprints,
            self.unsolved_buckets,
            self.active_library_primitives,
            self.problem_ledger,
        ):
            for entry in coll:
                if isinstance(entry, dict):
                    overlap = forbidden_raw_keys.intersection(entry.keys())
                    if overlap:
                        leaked_keys.extend(sorted(overlap))
                    for sub in entry.get("attempts", []):
                        if isinstance(sub, dict):
                            sub_overlap = forbidden_raw_keys.intersection(sub.keys())
                            if sub_overlap:
                                leaked_keys.extend(sorted(sub_overlap))

        within_caps = (
            len(self.best_invariants) <= self.max_invariants
            and len(self.dead_end_fingerprints) <= self.max_dead_ends
            and len(self.unsolved_buckets) <= self.max_unsolved_buckets
            and len(self.active_library_primitives) <= self.max_primitives
            and len(self.problem_ledger) <= self.max_problems
            and all(
                len(p.get("attempts", [])) <= self.max_attempts_per_problem
                for p in self.problem_ledger
                if isinstance(p, dict)
            )
        )
        valid = within_caps and len(leaked_keys) == 0
        return {
            "valid": valid,
            "within_caps": within_caps,
            "leaked_raw_keys": leaked_keys,
            "counts": {
                "best_invariants": len(self.best_invariants),
                "dead_end_fingerprints": len(self.dead_end_fingerprints),
                "unsolved_buckets": len(self.unsolved_buckets),
                "active_library_primitives": len(self.active_library_primitives),
                "problem_ledger": len(self.problem_ledger),
            },
            "discarded_raw_traces_count": self.discarded_raw_traces_count,
            "serialized_byte_size": self.serialized_byte_size(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "round_index": self.round_index,
            "best_invariants": list(self.best_invariants),
            "dead_end_fingerprints": list(self.dead_end_fingerprints),
            "unsolved_buckets": list(self.unsolved_buckets),
            "active_library_primitives": list(self.active_library_primitives),
            "problem_ledger": [
                dict(p, attempts=[dict(a) for a in p.get("attempts", [])])
                for p in self.problem_ledger
            ],
            "carried_from": dict(self.carried_from) if self.carried_from is not None else None,
            "discarded_raw_traces_count": self.discarded_raw_traces_count,
            "caps": {
                "max_invariants": self.max_invariants,
                "max_dead_ends": self.max_dead_ends,
                "max_unsolved_buckets": self.max_unsolved_buckets,
                "max_primitives": self.max_primitives,
                "max_problems": self.max_problems,
                "max_attempts_per_problem": self.max_attempts_per_problem,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MutableSkillState":
        caps = data.get("caps", {}) if isinstance(data.get("caps"), dict) else {}
        obj = cls(
            task=str(data.get("task", "default")),
            round_index=int(data.get("round_index", 0)),
            max_invariants=int(caps.get("max_invariants", 8)),
            max_dead_ends=int(caps.get("max_dead_ends", 16)),
            max_unsolved_buckets=int(caps.get("max_unsolved_buckets", 32)),
            max_primitives=int(caps.get("max_primitives", 12)),
            max_problems=int(caps.get("max_problems", 8)),
            max_attempts_per_problem=int(caps.get("max_attempts_per_problem", 3)),
        )
        obj.best_invariants = list(data.get("best_invariants", []))[: obj.max_invariants]
        obj.dead_end_fingerprints = list(data.get("dead_end_fingerprints", []))[: obj.max_dead_ends]
        obj.unsolved_buckets = list(data.get("unsolved_buckets", []))[: obj.max_unsolved_buckets]
        obj.active_library_primitives = list(data.get("active_library_primitives", []))[
            : obj.max_primitives
        ]
        raw_probs = data.get("problem_ledger", [])
        if isinstance(raw_probs, list):
            obj.problem_ledger = [
                dict(
                    p,
                    attempts=[dict(a) for a in p.get("attempts", [])][-obj.max_attempts_per_problem :],
                )
                for p in raw_probs[: obj.max_problems]
                if isinstance(p, dict)
            ]
        cf = data.get("carried_from")
        obj.carried_from = dict(cf) if isinstance(cf, dict) else None
        obj.discarded_raw_traces_count = int(data.get("discarded_raw_traces_count", 0))
        return obj

    def serialized_byte_size(self) -> int:
        return len(json.dumps(self.to_dict(), sort_keys=True).encode("utf-8"))

    def compact_prompt_context(self) -> str:
        inner: Dict[str, Any] = {
            "task": self.task,
            "round": self.round_index,
            "best_invariants": self.best_invariants,
            "dead_end_fingerprints": self.dead_end_fingerprints,
            "unsolved_buckets": self.unsolved_buckets,
            "active_library_primitives": self.active_library_primitives,
        }
        if self.problem_ledger:
            inner["problem_ledger"] = self.problem_ledger
        if self.carried_from is not None:
            inner["carried_from"] = self.carried_from
        return json.dumps({"SKILL.state": inner}, indent=2, sort_keys=True)

__all__ = [
    "BackwardSubGoalDecomposer",
    "decompose_dead_buckets",
    "MutableSkillState",
]

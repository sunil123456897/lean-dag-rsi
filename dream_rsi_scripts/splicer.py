"""
Lean-DAG-RSI v4/v7 AST call-graph metrics, hyper-crossover selection,
bucket JEPA predictor, and autonomous AST code splicer (`AstCodeSplicer`).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

if __package__:
    from .tree import Node
else:
    from tree import Node


def _ast_iou(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return float(inter) / float(max(1, union))


def _extract_node_margins(node: Node) -> List[float]:
    mv = node.meta.get("margin_vector") if isinstance(node.meta, dict) else None
    if isinstance(mv, (list, tuple)) and len(mv) > 0:
        return [float(x) for x in mv]
    pm = getattr(node, "pass_mask", None) or (node.meta.get("pass_mask") if isinstance(node.meta, dict) else None)
    if isinstance(pm, (list, tuple)) and len(pm) > 0:
        return [1.0 if x else 0.0 for x in pm]
    return []


def _extract_node_ast_symbols(node: Node) -> Set[str]:
    syms: Set[str] = set()
    if isinstance(node.meta, dict):
        raw = node.meta.get("ast_touches") or node.meta.get("modified_symbols")
        if isinstance(raw, (list, tuple, set)):
            syms.update(str(s) for s in raw)
    hotspots = getattr(node, "line_hotspots", None)
    if isinstance(hotspots, list):
        for h in hotspots:
            if isinstance(h, dict) and "code" in h:
                syms.add(str(h["code"]).strip())
    return syms


def compute_hyper_crossover(
    candidates: Sequence[Node],
    baseline: float = 0.0,
    lambda_collision: float = 0.35,
    max_parents: int = 4,
    min_marginal_gain: float = 1e-4,
) -> Dict[str, Any]:
    """
    Lean-DAG-RSI v4 Multi-Parent Submodular Hyper-Crossover with Excess Margin Gain:
      Delta_Hyper(S) = (1/M) sum_{j=1}^M max_{u in S} phi_j(u)
                       - max_{u in S} ((1/M) sum_{j=1}^M phi_j(u))
                       - lambda_collision * sum_{{u,v} subseteq S} IoU_AST(u, v)

    Crucial mathematical properties:
      1. Subtracting the best single parent max_{u in S} mean_j(phi_u) guarantees
         Delta_Hyper({u*}) == 0 and prevents pairing a dominant parent u* with a
         useless disjoint node v_junk (Junk-Parent Rejection).
      2. Enforcing strict positive marginal gain Delta_Hyper(S U {w}) - Delta_Hyper(S) > tol
         at every greedy step filters out clones (0 marginal gain) and AST-colliding
         branches (negative marginal gain when IoU_AST penalty exceeds bucket gain).
    """
    valid_nodes: List[Tuple[Node, List[float], Set[str], float]] = []
    m_len: Optional[int] = None
    for n in candidates:
        margins = _extract_node_margins(n)
        if not margins:
            continue
        if m_len is None:
            m_len = len(margins)
        if len(margins) != m_len:
            continue
        mean_m = sum(margins) / float(m_len)
        valid_nodes.append((n, margins, _extract_node_ast_symbols(n), mean_m))

    if len(valid_nodes) < 2 or m_len is None or m_len == 0:
        return {
            "selected_ids": [],
            "action": "",
            "excess_hyper_gain": 0.0,
            "raw_union_mean": 0.0,
            "best_single_mean": 0.0,
            "total_ast_collision": 0.0,
            "rejected_candidates": {},
        }

    # Sort by mean margin descending, pick anchor u* = argmax_u mean_j(phi_j(u))
    valid_nodes.sort(key=lambda item: (item[3], item[0].score or 0.0), reverse=True)
    anchor_id = valid_nodes[0][0].id
    anchor_single_mean = valid_nodes[0][3]

    def _eval_subset(subset_indices: List[int]) -> Tuple[float, float, float, float]:
        # For any subset S anchored at u* (index 0), max_{u in S} mean_j(phi_j(u)) == mean_j(phi_j(u*))
        # is constant across S, so the anchored coverage gain is strictly monotone submodular.
        best_single = (
            anchor_single_mean
            if (0 in subset_indices)
            else max(valid_nodes[idx][3] for idx in subset_indices)
        )
        union_mean = (
            sum(
                max(valid_nodes[idx][1][j] for idx in subset_indices)
                for j in range(m_len)
            )
            / float(m_len)
        )
        collision_sum = 0.0
        n_sub = len(subset_indices)
        n_pairs = max(1, (n_sub * (n_sub - 1)) // 2)
        for a_pos in range(n_sub):
            for b_pos in range(a_pos + 1, n_sub):
                u_idx = subset_indices[a_pos]
                v_idx = subset_indices[b_pos]
                # Same-branch nodes also incur full collision
                if (
                    valid_nodes[u_idx][0].branch is not None
                    and valid_nodes[u_idx][0].branch == valid_nodes[v_idx][0].branch
                ):
                    collision_sum += 1.0
                else:
                    collision_sum += _ast_iou(
                        valid_nodes[u_idx][2], valid_nodes[v_idx][2]
                    )
        mean_pair_collision = collision_sum / float(n_pairs)
        excess = (union_mean - best_single) - float(lambda_collision) * mean_pair_collision
        return excess, union_mean, best_single, mean_pair_collision

    selected_idx: List[int] = [0]
    current_excess, current_union, current_best_single, current_coll = _eval_subset(
        selected_idx
    )
    rejected_reasons: Dict[str, str] = {}

    while len(selected_idx) < max_parents:
        best_w: Optional[int] = None
        best_marginal = 0.0
        best_metrics = (current_excess, current_union, current_best_single, current_coll)

        for w_idx in range(1, len(valid_nodes)):
            if w_idx in selected_idx:
                continue
            trial_idx = selected_idx + [w_idx]
            exc, uni, b_sing, coll = _eval_subset(trial_idx)
            marginal = exc - current_excess
            w_id = valid_nodes[w_idx][0].id
            if marginal > max(min_marginal_gain, best_marginal):
                best_marginal = marginal
                best_w = w_idx
                best_metrics = (exc, uni, b_sing, coll)
            elif w_id not in rejected_reasons:
                raw_gain = uni - current_union
                if raw_gain <= 1e-9:
                    rejected_reasons[w_id] = "zero_bucket_gain_or_clone"
                else:
                    rejected_reasons[w_id] = "ast_collision_exceeds_gain"

        if best_w is not None:
            w_id = valid_nodes[best_w][0].id
            rejected_reasons.pop(w_id, None)
            selected_idx.append(best_w)
            current_excess, current_union, current_best_single, current_coll = (
                best_metrics
            )
        else:
            break

    selected_ids = [valid_nodes[idx][0].id for idx in selected_idx]
    action_str = (
        f"HYPER_CROSSOVER({','.join(selected_ids)})"
        if len(selected_ids) >= 2
        else ""
    )
    spliced_artifact: Optional[Dict[str, Any]] = None
    if len(selected_ids) >= 2:
        parent_code_pairs: List[Tuple[str, str]] = []
        base_src: Optional[str] = None
        for idx in selected_idx:
            n_obj = valid_nodes[idx][0]
            art = n_obj.artifact if isinstance(n_obj.artifact, dict) else {}
            meta_d = n_obj.meta if isinstance(n_obj.meta, dict) else {}
            code_str = art.get("code") or meta_d.get("code")
            if isinstance(code_str, str) and code_str.strip():
                parent_code_pairs.append((n_obj.id, code_str))
            if base_src is None:
                b_cand = art.get("base_code") or meta_d.get("base_code")
                if isinstance(b_cand, str) and b_cand.strip():
                    base_src = b_cand
        if len(parent_code_pairs) >= 2:
            splicer = AstCodeSplicer()
            spliced_artifact = splicer.splice_parents(
                base_source=base_src or parent_code_pairs[0][1],
                parent_sources=parent_code_pairs,
            )

    return {
        "selected_ids": selected_ids if len(selected_ids) >= 2 else [],
        "anchor_id": anchor_id,
        "is_anchored_submodular": True,
        "action": action_str,
        "parent_count": len(selected_ids) if len(selected_ids) >= 2 else 0,
        "excess_hyper_gain": round(current_excess, 6),
        "raw_union_mean": round(current_union, 6),
        "best_single_mean": round(current_best_single, 6),
        "total_ast_collision": round(current_coll, 6),
        "rejected_candidates": rejected_reasons,
        "spliced_artifact": spliced_artifact,
    }


def predict_bucket_jepa(
    observed_nodes: Sequence[Node],
    candidate_parent: Node,
    candidate_ast_touches: Set[str],
    candidate_step_delta: float = 0.0,
) -> Dict[str, Any]:
    """
    Lean-DAG-RSI v4 Zero-Cold-Start Bucket-Factored World Model W_phi(z_u, delta_AST, h_bucket(j)):
    Predicts per-bucket pass probabilities hat_m(v) in [0, 1]^M by conditioning on each
    bucket's empirical difficulty profile h_bucket(j) = (mean_pass_j, var_pass_j, parent_pass_j)
    and AST novelty relative to prior nodes. Works for any M >= 1 from Round 1 onward.
    """
    parent_margins = _extract_node_margins(candidate_parent)
    if not parent_margins:
        return {"predicted_margins": [], "expected_union_gain": 0.0}
    m_len = len(parent_margins)

    hist_Node_margins: List[List[float]] = []
    hist_ast_sets: List[Set[str]] = []
    for n in observed_nodes:
        pm = _extract_node_margins(n)
        if len(pm) == m_len:
            hist_Node_margins.append(pm)
            hist_ast_sets.append(_extract_node_ast_symbols(n))

    if not hist_Node_margins:
        hist_Node_margins = [parent_margins]

    # Compute AST novelty (1 - max IoU with existing nodes)
    max_iou = max((_ast_iou(candidate_ast_touches, s) for s in hist_ast_sets), default=0.0)
    ast_novelty = 1.0 - max_iou

    predicted_margins: List[float] = []
    for j in range(m_len):
        col = [row[j] for row in hist_Node_margins]
        mean_j = sum(col) / float(len(col))
        parent_j = parent_margins[j]
        # Latent transition update: buckets failing in parent (parent_j < 0.5) with non-zero
        # historical solvability (mean_j > 0) gain probability proportional to AST novelty
        lift = 0.35 * ast_novelty * max(0.0, mean_j * 0.5 + 0.25) + 0.10 * math.tanh(candidate_step_delta)
        pred_j = min(1.0, max(0.0, 0.75 * parent_j + 0.25 * mean_j + (lift if parent_j < 0.8 else 0.02 * lift)))
        predicted_margins.append(round(pred_j, 6))

    expected_union = sum(max(parent_margins[j], predicted_margins[j]) for j in range(m_len)) / float(m_len)
    parent_mean = sum(parent_margins) / float(m_len)
    return {
        "predicted_margins": predicted_margins,
        "predicted_mean": round(sum(predicted_margins) / float(m_len), 6),
        "expected_union_gain": round(max(0.0, expected_union - parent_mean), 6),
        "ast_novelty": round(ast_novelty, 4),
    }


class AstCodeSplicer:
    """
    SLEEP-DAG-RSI v7 Autonomous Multi-Parent AST Structural Splicer:
    Turns `compute_hyper_crossover` recommendations into verified, executable Python
    source code without requiring an external LLM or manual merge step:
      1. Parses the lowest common ancestor (`base_source`) and 2..K parent source strings
         into abstract syntax trees (`ast.parse`).
      2. Extracts structural AST diffs per parent:
         - Top-level helper functions (`ast.FunctionDef`), imports, and module constants.
         - Intra-function statement mutations indexed by assignment target variable
           (`ast.Assign` / `ast.AugAssign`) or hook block.
      3. When parent mutations touch disjoint AST spans (`ast_collision_iou < 0.30`),
         splices all non-colliding top-level helpers and intra-function variable updates
         into a single unified AST (`splice_mode = "structural_ast_merge"`).
      4. When parents modify the same return expression on complementary buckets,
         automatically synthesizes renamed parent sub-functions `_parent_<id>_<fn>` and
         a verified combiner/dispatcher (`splice_mode = "ensemble_dispatch_merge"`).
      5. Verifies the spliced code through `compile(..., "<ast_splice>", "exec")` and
         static AST safety checks (blocking `eval`, `exec`, `globals`, etc.).
    """

    FORBIDDEN_CALLS = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "system",
        "popen",
        "rmtree",
        "globals",
        "locals",
        "setattr",
        "delattr",
    }

    @staticmethod
    def _stmt_key(stmt: Any, idx: int) -> str:
        import ast

        if isinstance(stmt, ast.Assign) and stmt.targets:
            t0 = stmt.targets[0]
            if isinstance(t0, ast.Name):
                return f"assign:{t0.id}"
            return f"assign:{ast.unparse(t0)}"
        if isinstance(stmt, ast.AugAssign):
            if isinstance(stmt.target, ast.Name):
                return f"augassign:{stmt.target.id}"
            return f"augassign:{ast.unparse(stmt.target)}"
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            return f"assign:{stmt.target.id}"
        if isinstance(stmt, ast.Return):
            return "return"
        return f"stmt_{idx}:{type(stmt).__name__}"

    def extract_structural_touches(
        self, source_code: str, base_source: Optional[str] = None
    ) -> Set[str]:
        import ast

        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return set()

        base_funcs: Dict[str, Any] = {}
        base_assigns: Dict[str, str] = {}
        if base_source:
            try:
                b_tree = ast.parse(base_source)
                for node in b_tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        base_funcs[node.name] = node
                    elif isinstance(node, ast.Assign) and node.targets:
                        k = self._stmt_key(node, 0)
                        base_assigns[k] = ast.unparse(node)
            except SyntaxError:
                pass

        touches: Set[str] = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name not in base_funcs:
                    touches.add(f"func_new:{node.name}")
                else:
                    b_fn = base_funcs[node.name]
                    b_map = {
                        self._stmt_key(s, idx): ast.unparse(s)
                        for idx, s in enumerate(b_fn.body)
                    }
                    for idx, s in enumerate(node.body):
                        k = self._stmt_key(s, idx)
                        s_str = ast.unparse(s)
                        if b_map.get(k) != s_str:
                            touches.add(f"fn:{node.name}:{k}")
            elif isinstance(node, ast.Assign) and node.targets:
                k = self._stmt_key(node, 0)
                if base_assigns.get(k) != ast.unparse(node):
                    touches.add(f"mod_{k}")
        return touches

    def _synthesize_input_routing_predicate(
        self,
        arg_name: str,
        bucket_inputs: Sequence[Any],
        parent_bucket_margins: Sequence[Sequence[float]],
        custom_predicates: Optional[Sequence[Tuple[str, Callable[[Any], float]]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Synthesizes an Out-of-Sample input-space routing predicate (`if <pred>: return p_win(x) else: return p_alt(x)`)
        by screening a 1-sparse dictionary of input-space features:
          - Tier 0 (1D coordinate cuts): `x[c] >= tau`
          - Tier 1 (1D magnitude cuts): `abs(x[c]) >= tau`
          - Tier 2 (Custom / SymbolicLibrary primitives): `lib_k(x) >= tau`
          - Tier 3 (Pairwise bilinear quadrant / XOR splits): `(x[i] * x[j]) >= tau`
          - Tier 4 (Pairwise radial / circular norm splits): `(x[i] * x[i] + x[j] * x[j]) >= tau`
        Uses an MDL complexity tie-break (`mdl_eps = 1e-5 * tier`) so simple 1D coordinate cuts
        win on ties, while nonlinear quadrant/radial/library splits automatically take over
        whenever axis-aligned hyperplanes cannot separate the parent specialties.
        """
        if len(parent_bucket_margins) < 2 or len(bucket_inputs) == 0:
            return None
        m0 = [float(v) for v in parent_bucket_margins[0]]
        m1 = [float(v) for v in parent_bucket_margins[1]]
        n_b = min(len(bucket_inputs), len(m0), len(m1))
        if n_b < 2:
            return None

        is_scalar = isinstance(bucket_inputs[0], (int, float))
        rows: List[List[float]] = []
        for idx in range(n_b):
            inp = bucket_inputs[idx]
            if isinstance(inp, (int, float)):
                rows.append([float(inp)])
            else:
                rows.append([float(v) for v in inp])
        d_in = len(rows[0]) if rows else 0
        if d_in == 0:
            return None

        # Build candidate 1D feature projections: (lhs_expr, values_per_bucket, complexity_tier)
        candidates: List[Tuple[str, List[float], int]] = []
        for c in range(d_in):
            lhs = arg_name if is_scalar else f"{arg_name}[{c}]"
            col_vals = [rows[b][c] for b in range(n_b)]
            candidates.append((lhs, col_vals, 0))
            candidates.append((f"abs({lhs})", [abs(v) for v in col_vals], 1))

        if custom_predicates:
            for expr_tmpl, fn_eval in custom_predicates:
                try:
                    c_vals = [float(fn_eval(bucket_inputs[b])) for b in range(n_b)]
                    lhs_custom = expr_tmpl.replace("(x)", f"({arg_name})") if "(x)" in expr_tmpl else expr_tmpl
                    candidates.append((lhs_custom, c_vals, 2))
                except Exception:
                    continue

        if not is_scalar and d_in >= 2:
            max_pair_d = min(d_in, 8)
            for i in range(max_pair_d):
                for j in range(i + 1, max_pair_d):
                    prod_vals = [rows[b][i] * rows[b][j] for b in range(n_b)]
                    candidates.append((f"({arg_name}[{i}] * {arg_name}[{j}])", prod_vals, 3))
                    rad_vals = [rows[b][i] * rows[b][i] + rows[b][j] * rows[b][j] for b in range(n_b)]
                    candidates.append(
                        (f"({arg_name}[{i}] * {arg_name}[{i}] + {arg_name}[{j}] * {arg_name}[{j}])", rad_vals, 4)
                    )

        best_mdl_obj = -float("inf")
        best_gain = -float("inf")
        best_loo_gain = -float("inf")
        best_expr: Optional[str] = None
        best_p0_when_true = True
        best_acc = 0.0
        best_tier = 0
        max_single_parent = max(sum(m0[:n_b]) / float(n_b), sum(m1[:n_b]) / float(n_b))

        for lhs_expr, feat_vals, tier in candidates:
            vals = sorted(set(feat_vals))
            if len(vals) < 2:
                continue
            thresholds: List[float] = []
            for k in range(len(vals) - 1):
                if vals[k] < 0.0 < vals[k + 1]:
                    thresholds.append(0.0)
                thresholds.append(0.5 * (vals[k] + vals[k + 1]))
            for tau in thresholds:
                for p0_on_ge in (True, False):
                    routed_sum = 0.0
                    correct = 0
                    informative = 0
                    for b in range(n_b):
                        cond = feat_vals[b] >= tau
                        use_p0 = cond if p0_on_ge else (not cond)
                        routed_sum += m0[b] if use_p0 else m1[b]
                        if abs(m0[b] - m1[b]) > 1e-9:
                            informative += 1
                            if (use_p0 and m0[b] > m1[b]) or ((not use_p0) and m1[b] > m0[b]):
                                correct += 1
                    routed_mean = routed_sum / float(n_b)

                    # Out-of-fold / leave-one-bucket-out (LOO) holdout check for small M (n_b <= 8)
                    # on high-degree Tier 3/4 predicates to prevent random quadratic shattering.
                    loo_mean = routed_mean
                    if n_b <= 8 and tier >= 3 and n_b >= 3:
                        loo_sum = 0.0
                        for hold_b in range(n_b):
                            p0_f = [
                                feat_vals[i]
                                for i in range(n_b)
                                if i != hold_b and (m0[i] > m1[i] if p0_on_ge else m1[i] > m0[i])
                            ]
                            p1_f = [
                                feat_vals[i]
                                for i in range(n_b)
                                if i != hold_b and (m1[i] > m0[i] if p0_on_ge else m0[i] > m1[i])
                            ]
                            if p0_f and p1_f and min(p0_f) > max(p1_f):
                                tau_loo = 0.0 if (max(p1_f) < 0.0 < min(p0_f)) else 0.5 * (min(p0_f) + max(p1_f))
                                cond_h = feat_vals[hold_b] >= tau_loo
                                use_p0_h = cond_h if p0_on_ge else (not cond_h)
                                loo_sum += m0[hold_b] if use_p0_h else m1[hold_b]
                            else:
                                loo_sum += max_single_parent
                        loo_mean = loo_sum / float(n_b)
                        if loo_mean <= max_single_parent + 1e-6:
                            continue

                    mdl_obj = routed_mean - 1e-5 * float(tier)
                    acc = correct / float(max(1, informative))
                    if mdl_obj > best_mdl_obj + 1e-9:
                        best_mdl_obj = mdl_obj
                        best_gain = routed_mean
                        best_loo_gain = loo_mean
                        best_p0_when_true = p0_on_ge
                        best_acc = acc
                        best_tier = tier
                        best_expr = f"{lhs_expr} >= {round(tau, 6)}"

        if best_expr is None or best_gain <= max_single_parent + 1e-6:
            return None
        return {
            "predicate_expr": best_expr,
            "p0_when_true": best_p0_when_true,
            "routed_mean_score": round(best_gain, 6),
            "loo_holdout_score": round(best_loo_gain, 6),
            "routing_accuracy": round(best_acc, 6),
            "complexity_tier": best_tier,
        }

    def splice_parents(
        self,
        base_source: str,
        parent_sources: Sequence[Tuple[str, str]],
        combine_mode: str = "auto",
        entry_func: Optional[str] = None,
        bucket_inputs: Optional[Sequence[Any]] = None,
        parent_bucket_margins: Optional[Sequence[Sequence[float]]] = None,
        routing_predicate_expr: Optional[str] = None,
        custom_predicates: Optional[Sequence[Tuple[str, Callable[[Any], float]]]] = None,
    ) -> Dict[str, Any]:
        import ast
        import copy
        import re

        if len(parent_sources) < 2:
            return {
                "valid": False,
                "spliced_code": base_source,
                "splice_mode": "insufficient_parents",
                "merged_symbols": [],
                "ast_collision_iou": 0.0,
                "violations": ["need_at_least_2_parents"],
            }

        try:
            base_tree = ast.parse(base_source)
        except SyntaxError as exc:
            return {
                "valid": False,
                "spliced_code": "",
                "splice_mode": "syntax_error_base",
                "merged_symbols": [],
                "ast_collision_iou": 0.0,
                "violations": [f"base_syntax_error:{exc}"],
            }

        parsed_parents: List[Tuple[str, Any, Set[str]]] = []
        for pid, p_src in parent_sources:
            try:
                p_tree = ast.parse(p_src)
            except SyntaxError as exc:
                return {
                    "valid": False,
                    "spliced_code": "",
                    "splice_mode": "syntax_error_parent",
                    "merged_symbols": [],
                    "ast_collision_iou": 0.0,
                    "violations": [f"parent_syntax_error:{pid}:{exc}"],
                }
            touches = self.extract_structural_touches(p_src, base_source=base_source)
            parsed_parents.append((pid, p_tree, touches))

        # Compute pairwise AST touch IoU (ignoring shared `return` if combine_mode handles it)
        non_return_sets = [
            {t for t in touches if not t.endswith(":return")}
            for _, _, touches in parsed_parents
        ]
        max_iou = 0.0
        for i in range(len(non_return_sets)):
            for j in range(i + 1, len(non_return_sets)):
                max_iou = max(max_iou, _ast_iou(non_return_sets[i], non_return_sets[j]))

        merged_tree = copy.deepcopy(base_tree)
        base_top_funcs: Dict[str, Any] = {}
        base_top_assigns: Dict[str, Any] = {}
        for node in base_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                base_top_funcs[node.name] = node
            elif isinstance(node, ast.Assign):
                base_top_assigns[self._stmt_key(node, 0)] = node

        merged_symbols: List[str] = []
        new_top_nodes: List[Any] = []
        seen_new_names: Set[str] = set()

        # 1. Collect orthogonal new top-level imports, constants, and helper functions from parents
        for pid, p_tree, _ in parsed_parents:
            for node in p_tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    u_str = ast.unparse(node)
                    if not any(
                        isinstance(b, (ast.Import, ast.ImportFrom)) and ast.unparse(b) == u_str
                        for b in merged_tree.body + new_top_nodes
                    ):
                        new_top_nodes.append(copy.deepcopy(node))
                elif isinstance(node, ast.Assign):
                    k = self._stmt_key(node, 0)
                    u_str = ast.unparse(node)
                    if k not in base_top_assigns and k not in seen_new_names:
                        seen_new_names.add(k)
                        new_top_nodes.append(copy.deepcopy(node))
                        merged_symbols.append(f"{pid}:{k}")
                    elif k in base_top_assigns and u_str != ast.unparse(base_top_assigns[k]):
                        for idx_b, b_node in enumerate(merged_tree.body):
                            if isinstance(b_node, ast.Assign) and self._stmt_key(b_node, 0) == k:
                                merged_tree.body[idx_b] = copy.deepcopy(node)
                                merged_symbols.append(f"{pid}:{k}")
                                break
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name not in base_top_funcs and node.name not in seen_new_names:
                        seen_new_names.add(node.name)
                        new_top_nodes.append(copy.deepcopy(node))
                        merged_symbols.append(f"{pid}:func:{node.name}")

        # 2. Merge intra-function statement modifications across shared functions
        return_collision_detected = False
        target_fn_name = entry_func
        for idx_b, b_node in enumerate(merged_tree.body):
            if not isinstance(b_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            fn_name = b_node.name
            if target_fn_name is None:
                target_fn_name = fn_name
            orig_fn = base_top_funcs.get(fn_name)
            if orig_fn is None:
                continue

            orig_stmt_map: Dict[str, str] = {}
            orig_order: List[str] = []
            stmt_slots: Dict[str, Any] = {}
            for s_idx, stmt in enumerate(orig_fn.body):
                k = self._stmt_key(stmt, s_idx)
                orig_stmt_map[k] = ast.unparse(stmt)
                orig_order.append(k)
                stmt_slots[k] = copy.deepcopy(stmt)

            extra_inserted_stmts: List[Any] = []
            extra_seen_keys: Set[str] = set()
            modified_returns: List[Tuple[str, Any]] = []

            for pid, p_tree, _ in parsed_parents:
                p_fn = next(
                    (
                        n
                        for n in p_tree.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and n.name == fn_name
                    ),
                    None,
                )
                if p_fn is None:
                    continue
                for s_idx, p_stmt in enumerate(p_fn.body):
                    k = self._stmt_key(p_stmt, s_idx)
                    p_str = ast.unparse(p_stmt)
                    if k == "return":
                        if orig_stmt_map.get("return") != p_str:
                            modified_returns.append((pid, copy.deepcopy(p_stmt)))
                    elif k in orig_stmt_map:
                        if orig_stmt_map[k] != p_str:
                            stmt_slots[k] = copy.deepcopy(p_stmt)
                            merged_symbols.append(f"{pid}:{fn_name}.{k}")
                    elif k not in extra_seen_keys:
                        extra_seen_keys.add(k)
                        extra_inserted_stmts.append(copy.deepcopy(p_stmt))
                        merged_symbols.append(f"{pid}:{fn_name}.{k}")

            if len(modified_returns) == 1:
                pid_r, ret_stmt = modified_returns[0]
                stmt_slots["return"] = ret_stmt
                merged_symbols.append(f"{pid_r}:{fn_name}.return")
            elif len(modified_returns) >= 2:
                # Check if one parent's return is just additive or if both modified return expressions
                ret_exprs = [ast.unparse(r_stmt.value) for _, r_stmt in modified_returns if r_stmt.value is not None]
                orig_ret_Node = orig_fn.body[-1] if orig_fn.body and isinstance(orig_fn.body[-1], ast.Return) else None
                orig_ret_str = ast.unparse(orig_ret_Node.value) if (orig_ret_Node and orig_ret_Node.value) else ""
                # Try additive delta composition if return is a sum or expression over base return
                additive_deltas: List[str] = []
                for _, r_stmt in modified_returns:
                    if r_stmt.value is not None:
                        r_s = ast.unparse(r_stmt.value)
                        if orig_ret_str and r_s.startswith(orig_ret_str + " + "):
                            additive_deltas.append(r_s[len(orig_ret_str) + 3 :])
                if len(additive_deltas) == len(modified_returns) and orig_ret_str:
                    combined_ret_src = orig_ret_str + "".join(f" + ({d_s})" for d_s in additive_deltas)
                    stmt_slots["return"] = ast.parse(f"return {combined_ret_src}").body[0]
                    for pid_r, _ in modified_returns:
                        merged_symbols.append(f"{pid_r}:{fn_name}.return_delta")
                elif (
                    combine_mode == "ensemble"
                    or max_iou > 0.30
                    or routing_predicate_expr is not None
                    or (bucket_inputs is not None and parent_bucket_margins is not None)
                ):
                    return_collision_detected = True
                else:
                    # Combine complementary return expressions according to combine_mode
                    if combine_mode == "max":
                        comb_expr = f"max({', '.join(f'({e})' for e in ret_exprs)})"
                    elif combine_mode == "min":
                        comb_expr = f"min({', '.join(f'({e})' for e in ret_exprs)})"
                    else:
                        w = round(1.0 / float(len(ret_exprs)), 6)
                        comb_expr = " + ".join(f"{w} * ({e})" for e in ret_exprs)
                    stmt_slots["return"] = ast.parse(f"return {comb_expr}").body[0]
                    for pid_r, _ in modified_returns:
                        merged_symbols.append(f"{pid_r}:{fn_name}.return_combined")

            # Rebuild function body with extra non-return statements inserted right before `return`
            new_body: List[Any] = []
            for k in orig_order:
                if k == "return":
                    new_body.extend(extra_inserted_stmts)
                new_body.append(stmt_slots[k])
            if "return" not in orig_order and extra_inserted_stmts:
                new_body.extend(extra_inserted_stmts)
            b_node.body = new_body

        splice_mode = "structural_ast_merge"
        learned_routing: Optional[Dict[str, Any]] = None
        if return_collision_detected and target_fn_name is not None:
            # Mode C: Full sub-function renaming & ensemble/input-predicate dispatch
            splice_mode = "ensemble_dispatch_merge"
            helper_fn_names: List[str] = []
            ensemble_body_nodes: List[Any] = []
            for pid, p_tree, _ in parsed_parents:
                safe_pid = re.sub(r"[^a-zA-Z0-9_]", "_", str(pid))
                for node in p_tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target_fn_name:
                        sub_fn = copy.deepcopy(node)
                        sub_fn.name = f"_parent_{safe_pid}_{target_fn_name}"
                        helper_fn_names.append(sub_fn.name)
                        ensemble_body_nodes.append(sub_fn)
                        merged_symbols.append(f"{pid}:dispatch:{sub_fn.name}")
            if helper_fn_names:
                orig_fn = base_top_funcs[target_fn_name]
                arg_names = [a.arg for a in orig_fn.args.args]
                call_args = ", ".join(arg_names)
                first_arg = arg_names[0] if arg_names else "x"
                if (
                    routing_predicate_expr is None
                    and bucket_inputs is not None
                    and parent_bucket_margins is not None
                ):
                    learned_routing = self._synthesize_input_routing_predicate(
                        first_arg,
                        bucket_inputs,
                        parent_bucket_margins,
                        custom_predicates=custom_predicates,
                    )
                    if learned_routing is not None:
                        routing_predicate_expr = str(learned_routing["predicate_expr"])

                if routing_predicate_expr is not None and len(helper_fn_names) >= 2:
                    p0_true = True if learned_routing is None else bool(learned_routing.get("p0_when_true", True))
                    fn_true = helper_fn_names[0] if p0_true else helper_fn_names[1]
                    fn_false = helper_fn_names[1] if p0_true else helper_fn_names[0]
                    wrapper_src = (
                        f"def {target_fn_name}({call_args}):\n"
                        f"    if {routing_predicate_expr}:\n"
                        f"        return {fn_true}({call_args})\n"
                        f"    else:\n"
                        f"        return {fn_false}({call_args})\n"
                    )
                    merged_symbols.append(f"oos_predicate:{routing_predicate_expr}")
                else:
                    w = round(1.0 / float(len(helper_fn_names)), 6)
                    calls_expr = " + ".join(f"{w} * {h_name}({call_args})" for h_name in helper_fn_names)
                    wrapper_src = f"def {target_fn_name}({call_args}):\n    return {calls_expr}\n"
                wrapper_fn = ast.parse(wrapper_src).body[0]
                merged_tree.body = new_top_nodes + ensemble_body_nodes + [wrapper_fn]
        else:
            # Insert new helper functions/imports above existing functions
            imports = [n for n in new_top_nodes if isinstance(n, (ast.Import, ast.ImportFrom))]
            helpers = [n for n in new_top_nodes if not isinstance(n, (ast.Import, ast.ImportFrom))]
            merged_tree.body = imports + helpers + merged_tree.body

        ast.fix_missing_locations(merged_tree)
        spliced_code = ast.unparse(merged_tree)

        violations: List[str] = []
        try:
            compile(spliced_code, "<ast_splice>", "exec")
        except SyntaxError as exc:
            violations.append(f"compile_error:{exc}")

        for node in ast.walk(merged_tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in self.FORBIDDEN_CALLS:
                    violations.append(f"forbidden_call:{node.func.id}")
                elif isinstance(node.func, ast.Attribute) and node.func.attr in self.FORBIDDEN_CALLS:
                    violations.append(f"forbidden_method:{node.func.attr}")

        return {
            "valid": len(violations) == 0,
            "spliced_code": spliced_code,
            "splice_mode": splice_mode,
            "merged_symbols": merged_symbols,
            "ast_collision_iou": round(max_iou, 6),
            "routing_predicate": routing_predicate_expr,
            "learned_routing": learned_routing,
            "violations": violations,
        }

    def verify_spliced_epistasis(
        self,
        spliced_code: str,
        entry_func: str,
        eval_score_fn: Callable[[Callable[..., Any]], float],
        parent_scores: Sequence[float],
        tol: float = 1e-6,
        extra_globals: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Post-Splice Epistasis Execution Gate:
        Even when syntactic `ast_collision_iou == 0`, two merged statements can interact
        nonlinearly via shared data-flow variables (semantic epistasis). This gate
        compiles `spliced_code`, runs `eval_score_fn(fn)` on the sentinel/validation harness,
        and verifies `spliced_score >= max(parent_scores) - tol`.
        """
        ns: Dict[str, Any] = {"math": math}
        if extra_globals:
            ns.update(extra_globals)
        try:
            exec(compile(spliced_code, "<ast_splice>", "exec"), ns)  # noqa: S102
            fn = ns.get(entry_func)
            if not callable(fn):
                return {"passed": False, "reason": "missing_entry_func", "spliced_score": 0.0}
            spliced_score = float(eval_score_fn(fn))
            best_parent = max((float(s) for s in parent_scores), default=0.0)
            passed = bool(spliced_score >= best_parent - tol)
            return {
                "passed": passed,
                "spliced_score": round(spliced_score, 6),
                "best_parent_score": round(best_parent, 6),
                "synergy_gain": round(spliced_score - best_parent, 6),
            }
        except Exception as exc:
            return {"passed": False, "reason": f"runtime_error:{exc}", "spliced_score": 0.0}

__all__ = [
    "compute_hyper_crossover",
    "predict_bucket_jepa",
    "AstCodeSplicer",
]

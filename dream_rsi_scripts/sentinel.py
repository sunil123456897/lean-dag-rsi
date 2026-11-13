"""
SLEEP-DAG-RSI v6/v7 Wald multi-fidelity micro-sentinel for early mutation pruning.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

if __package__:
    from .tree import Node
else:
    from tree import Node


@dataclass
class SentinelSlice:
    """
    SLEEP-DAG-RSI v6/v7 Multi-Fidelity Sentinel Slice (`S_sentinel = S_adv U S_strat`).
    Holds the 2.5% parent adversarial failure indices and 2.5% uniform stratified
    anchor indices, with automatic exemption of `S_adv` for macro-innovations
    (`crossover`, `frame_transform`, or high-temperature `beta >= 0.8`), plus
    optional Winsorized top-quantile outlier trimming (`outlier_trim_quantile > 0`)
    for stochastic/noisy domains (e.g. financial time series).
    """

    domain_size: int
    adversarial_indices: List[int]
    stratified_indices: List[int]
    combined_indices: List[int]
    exempt_adversarial: bool
    cost_fraction: float
    trimmed_outlier_indices: List[int] = field(default_factory=list)


class WaldMicroSentinel:
    """
    Stage 1 of SLEEP-DAG-RSI v6/v7: Sequential Lower-Confidence-Bound (LCB) Early Veto.
    Evaluates a candidate mutation on a ~5% sentinel slice first:
      - Standard mutations evaluate on `S_adv U S_strat` (catching regressions on
        the parent's fragile manifold plus global anchor points).
      - In deterministic math (`noise_regime="deterministic"`, `outlier_trim_quantile=0.0`),
        `S_adv` targets the worst 2.5% bottlenecks (`97.5%–100%` error quantile).
      - In stochastic/noisy domains (`noise_regime="stochastic"` or `outlier_trim_quantile=0.05`),
        `S_adv` trims the top 5% irreducible outlier spikes (e.g. flash crashes) and selects
        from the informative high-error quantile band (`[1 - trim_q - adv_frac, 1 - trim_q]`),
        preventing outlier overfitting.
      - Macro-innovations (`is_crossover`, `is_frame_transform`, `beta >= 0.8`)
        evaluate ONLY on `S_strat` so global reorganizations are not killed for
        trading off the parent's local error patch.
      - Acts strictly as a ONE-WAY VETO: survivors must still pass 100% full-domain
        evaluation for final ranking and DAG acceptance (preventing slice Goodharting).
      - Aborted nodes receive `fail_class = "sentinel_pruned"` and `meta["sentinel_aborted"] = True`,
        shielding `V_Lean` replay while feeding `export_dag_dpo_pairs()` negative targets.
    """

    def __init__(
        self,
        adv_frac: float = 0.025,
        strat_frac: float = 0.025,
        gamma_lcb: float = 1.96,
        tol_margin: float = 0.01,
        noise_regime: str = "deterministic",
        outlier_trim_quantile: Optional[float] = None,
        hard_divergence_ceiling: float = 1e6,
    ) -> None:
        self.adv_frac = adv_frac
        self.strat_frac = strat_frac
        self.gamma_lcb = gamma_lcb
        self.tol_margin = tol_margin
        self.noise_regime = str(noise_regime).lower()
        self.hard_divergence_ceiling = float(hard_divergence_ceiling)
        if outlier_trim_quantile is not None:
            self.outlier_trim_quantile = max(0.0, min(0.40, float(outlier_trim_quantile)))
        else:
            self.outlier_trim_quantile = 0.05 if self.noise_regime == "stochastic" else 0.0

    def build_slice(
        self,
        parent_scores_or_errors: Sequence[float],
        higher_is_better: bool = True,
        is_crossover: bool = False,
        is_frame_transform: bool = False,
        beta: float = 0.5,
    ) -> SentinelSlice:
        n_total = len(parent_scores_or_errors)
        if n_total == 0:
            return SentinelSlice(0, [], [], [], True, 0.0, [])

        k_adv = max(1, int(round(self.adv_frac * n_total)))
        k_strat = max(1, int(round(self.strat_frac * n_total)))

        # Adversarial core: worst-performing indices of the parent (with optional outlier trimming)
        if higher_is_better:
            ranked_worst = sorted(range(n_total), key=lambda idx: float(parent_scores_or_errors[idx]))
        else:
            ranked_worst = sorted(
                range(n_total), key=lambda idx: abs(float(parent_scores_or_errors[idx])), reverse=True
            )

        trim_offset = 0
        if self.outlier_trim_quantile > 0.0 and n_total > k_adv:
            trim_offset = min(
                n_total - k_adv,
                max(0, int(round(self.outlier_trim_quantile * n_total))),
            )
        trimmed_outliers = sorted(ranked_worst[:trim_offset]) if trim_offset > 0 else []
        adv_idx = sorted(ranked_worst[trim_offset : trim_offset + k_adv])

        # Deterministic uniform stratified anchor sampled without replacement from the complement of S_adv U S_trim
        stride = max(1, n_total // k_strat)
        raw_strat = sorted({min(n_total - 1, i * stride) for i in range(k_strat)})
        excluded = set(adv_idx).union(trimmed_outliers)
        raw_disjoint = [s_i for s_i in raw_strat if s_i not in excluded]
        if not trimmed_outliers and len(raw_disjoint) >= min(2, k_strat):
            strat_idx = raw_disjoint
        else:
            strat_set = set()
            for s_i in raw_strat:
                if s_i not in excluded and s_i not in strat_set:
                    strat_set.add(s_i)
                else:
                    for offset in range(1, n_total):
                        cand_hi = s_i + offset
                        if cand_hi < n_total and cand_hi not in excluded and cand_hi not in strat_set:
                            strat_set.add(cand_hi)
                            break
                        cand_lo = s_i - offset
                        if cand_lo >= 0 and cand_lo not in excluded and cand_lo not in strat_set:
                            strat_set.add(cand_lo)
                            break
            strat_idx = sorted(strat_set)

        exempt = bool(is_crossover or is_frame_transform or beta >= 0.8)
        if exempt:
            # Double stratified density when adversarial slice is exempted so sample size stays ~5%
            k_strat_ex = max(2, int(round((self.adv_frac + self.strat_frac) * n_total)))
            stride_ex = max(1, n_total // k_strat_ex)
            combined = sorted({min(n_total - 1, i * stride_ex) for i in range(k_strat_ex)})
        else:
            combined = sorted(set(adv_idx).union(strat_idx))

        cost_frac = round(len(combined) / float(n_total), 6)
        return SentinelSlice(
            domain_size=n_total,
            adversarial_indices=adv_idx,
            stratified_indices=strat_idx,
            combined_indices=combined,
            exempt_adversarial=exempt,
            cost_fraction=cost_frac,
            trimmed_outlier_indices=trimmed_outliers,
        )

    def evaluate_gate(
        self,
        parent_node: Node,
        child_node: Node,
        parent_pointwise_scores: Sequence[float],
        eval_slice_fn: Callable[[Sequence[int]], Sequence[float]],
        eval_full_fn: Optional[Callable[[], Sequence[float]]] = None,
        is_crossover: bool = False,
        is_frame_transform: bool = False,
        beta: float = 0.5,
        warmup_grace_remaining: int = 0,
        aggregation: str = "mean",
    ) -> Dict[str, Any]:
        """
        Evaluates candidate `child_node` on the 5% sentinel slice:
          - `aggregation="mean"`: Wald Sequential Upper-Confidence-Bound (UCB95) early-veto test
            on mean gain `UCB95(Delta) = delta_mu + gamma * SE_strat >= -tol_margin`
            (optimism under uncertainty: vetoes a candidate only when even the optimistic
            95% upper confidence bound of its gain falls below `-tol_margin`).
          - `aggregation="min"`: Exact extremal upper-bound filter for max-min problems (e.g. Heilbronn):
            since `min_{all}(s_child) <= min_{sentinel}(s_child)`, any candidate with
            `min_{sentinel}(s_child) < min_{all}(s_parent) - tol_margin` is provably suboptimal (0 false negatives).
          - `aggregation="max"`: Exact extremal lower-bound filter for min-max problems (e.g. Autocorrelation C_2).
        """
        higher_is_better = aggregation != "max"
        sentinel_slice = self.build_slice(
            parent_pointwise_scores,
            higher_is_better=higher_is_better,
            is_crossover=is_crossover,
            is_frame_transform=is_frame_transform,
            beta=beta,
        )
        idxs = list(sentinel_slice.combined_indices)
        if not idxs:
            return {"passed": True, "aborted": False, "cost_spent": 1.0, "node": child_node}

        # In stochastic mode, carve at most n_canary = max(1, floor(0.005 * N)) extreme parent-trim
        # indices from within the 5% sentinel budget (2.25% S_adv + 2.25% S_strat + 0.5% canary = 5.0%)
        # so Stage 1 executes a single eval_slice_fn call of total size <= 5% of N while preserving
        # balanced strata (|S_adv|, |S_strat| >= 2 for N >= N_min = 80).
        trimmed_idxs = list(sentinel_slice.trimmed_outlier_indices or [])
        canary_idxs: List[int] = []
        active_idxs = list(idxs)
        adv_active = list(sentinel_slice.adversarial_indices)
        strat_active = list(sentinel_slice.stratified_indices)
        if trimmed_idxs:
            if higher_is_better:
                ranked_trim = sorted(trimmed_idxs, key=lambda idx: float(parent_pointwise_scores[idx]))
            else:
                ranked_trim = sorted(
                    trimmed_idxs, key=lambda idx: abs(float(parent_pointwise_scores[idx])), reverse=True
                )
            n_canary = min(len(ranked_trim), max(1, int(0.005 * sentinel_slice.domain_size)))
            canary_idxs = ranked_trim[:n_canary]
            if not sentinel_slice.exempt_adversarial and (len(adv_active) + len(strat_active) > n_canary + 2):
                min_keep = 2 if sentinel_slice.domain_size >= 80 else 1
                max_drop_adv = max(0, len(adv_active) - min_keep)
                max_drop_strat = max(0, len(strat_active) - min_keep)
                drop_adv = min(max_drop_adv, (n_canary + 1) // 2)
                drop_strat = min(max_drop_strat, n_canary - drop_adv)
                rem_drop = n_canary - (drop_adv + drop_strat)
                if rem_drop > 0 and drop_adv < max_drop_adv:
                    extra = min(rem_drop, max_drop_adv - drop_adv)
                    drop_adv += extra
                    rem_drop -= extra
                if rem_drop > 0 and drop_strat < max_drop_strat:
                    extra = min(rem_drop, max_drop_strat - drop_strat)
                    drop_strat += extra
                if drop_adv > 0:
                    # Drop the least-extreme adversarial indices at the end of the ranked band
                    if higher_is_better:
                        adv_ranked = sorted(adv_active, key=lambda idx: float(parent_pointwise_scores[idx]))
                    else:
                        adv_ranked = sorted(
                            adv_active, key=lambda idx: abs(float(parent_pointwise_scores[idx])), reverse=True
                        )
                    adv_active = sorted(adv_ranked[: len(adv_ranked) - drop_adv])
                if drop_strat > 0:
                    strat_active = strat_active[: len(strat_active) - drop_strat]
                active_idxs = sorted(set(adv_active).union(strat_active))
            elif len(active_idxs) > n_canary + 2:
                active_idxs = active_idxs[: len(active_idxs) - n_canary]

        eval_idxs = active_idxs + [c_i for c_i in canary_idxs if c_i not in set(active_idxs)]
        raw_eval_scores = [float(v) for v in eval_slice_fn(eval_idxs)]
        eval_map = {eval_idxs[j]: raw_eval_scores[j] for j in range(len(eval_idxs))}

        child_slice_scores = [eval_map[i] for i in active_idxs]
        parent_slice_scores = [float(parent_pointwise_scores[i]) for i in active_idxs]
        k = len(active_idxs)
        actual_stage1_cost = round(len(eval_idxs) / float(max(1, sentinel_slice.domain_size)), 6)

        catastrophic_blowup = any(
            (not math.isfinite(raw_eval_scores[j]))
            or (abs(raw_eval_scores[j] - float(parent_pointwise_scores[eval_idxs[j]])) > self.hard_divergence_ceiling)
            for j in range(len(eval_idxs))
        )
        diffs = [
            (child_slice_scores[j] - parent_slice_scores[j])
            if math.isfinite(child_slice_scores[j])
            else -self.hard_divergence_ceiling
            for j in range(k)
        ]
        pooled_mu = sum(diffs) / float(k)
        if k > 1:
            var_d = sum((d - pooled_mu) ** 2 for d in diffs) / float(k - 1)
            pooled_se = math.sqrt(max(0.0, var_d) / float(k))
        else:
            pooled_se = 0.0

        # Unbiased stratified variance estimator across disjoint S_adv and S_strat
        idx_to_diff = {active_idxs[j]: diffs[j] for j in range(k)}
        n_df = k
        if not sentinel_slice.exempt_adversarial:
            diffs_adv = [idx_to_diff[i] for i in adv_active if i in idx_to_diff]
            diffs_strat = [idx_to_diff[i] for i in strat_active if i in idx_to_diff]
            if len(diffs_adv) >= 2 and len(diffs_strat) >= 2:
                mu_adv = sum(diffs_adv) / float(len(diffs_adv))
                var_adv = sum((d - mu_adv) ** 2 for d in diffs_adv) / float(len(diffs_adv) - 1)
                se_adv_sq = max(0.0, var_adv) / float(len(diffs_adv))
                mu_strat = sum(diffs_strat) / float(len(diffs_strat))
                var_strat = sum((d - mu_strat) ** 2 for d in diffs_strat) / float(len(diffs_strat) - 1)
                se_strat_sq = max(0.0, var_strat) / float(len(diffs_strat))
                w_adv = 0.5
                w_strat = 0.5
                delta_mu = w_adv * mu_adv + w_strat * mu_strat
                se_mu = math.sqrt((w_adv ** 2) * se_adv_sq + (w_strat ** 2) * se_strat_sq)
                n_df = min(len(diffs_adv), len(diffs_strat))
            else:
                delta_mu = pooled_mu
                se_mu = pooled_se
        else:
            delta_mu = pooled_mu
            se_mu = pooled_se

        student_t_inflation = math.sqrt(float(k) / float(k - 1.5)) if 2 <= k < 30 else 1.0
        gamma_eff = self.gamma_lcb * student_t_inflation

        if aggregation == "min":
            parent_ext = min(float(v) for v in parent_pointwise_scores)
            child_slice_ext = min(child_slice_scores)
            ucb_gain = child_slice_ext - parent_ext
            passed = bool(
                (not catastrophic_blowup)
                and (warmup_grace_remaining > 0 or ucb_gain >= -self.tol_margin)
            )
        elif aggregation == "max":
            parent_ext = max(float(v) for v in parent_pointwise_scores)
            child_slice_ext = max(child_slice_scores)
            ucb_gain = parent_ext - child_slice_ext
            passed = bool(
                (not catastrophic_blowup)
                and (warmup_grace_remaining > 0 or ucb_gain >= -self.tol_margin)
            )
        else:
            ucb_gain = delta_mu + gamma_eff * se_mu
            passed = bool(
                (not catastrophic_blowup)
                and (warmup_grace_remaining > 0 or ucb_gain >= -self.tol_margin)
            )

        if not passed:
            child_node.fail_class = "sentinel_pruned"
            child_node.valid = False
            child_node.score = round(float(parent_node.score) + delta_mu, 6)
            child_node.delta_vs_parent = round(delta_mu, 6)
            child_node.meta = dict(child_node.meta or {})
            child_node.meta["sentinel_aborted"] = True
            child_node.meta["sentinel_cost"] = actual_stage1_cost
            child_node.meta["sentinel_ucb"] = round(ucb_gain, 6)
            child_node.meta["sentinel_lcb"] = round(ucb_gain, 6)
            child_node.meta["exempt_adversarial"] = sentinel_slice.exempt_adversarial
            return {
                "passed": False,
                "aborted": True,
                "cost_spent": actual_stage1_cost,
                "compute_saved": round(1.0 - actual_stage1_cost, 6),
                "delta_mean": round(delta_mu, 6),
                "se": round(se_mu, 6),
                "se_stratified": round(se_mu, 6),
                "se_pooled": round(pooled_se, 6),
                "student_t_inflation": round(student_t_inflation, 4),
                "ucb95_gain": round(ucb_gain, 6),
                "ucb": round(ucb_gain, 6),
                "lcb95_gain": round(ucb_gain, 6),
                "lcb": round(ucb_gain, 6),
                "parent_trim_locked": True,
                "canary_trim_probed": len(canary_idxs),
                "catastrophic_blowup": catastrophic_blowup,
                "exempt_adversarial": sentinel_slice.exempt_adversarial,
                "slice_size": len(eval_idxs),
                "node": child_node,
            }

        # Survivor: evaluate on 100% full benchmark so ranking never Goodharts the sentinel slice
        full_cost = 1.0
        if eval_full_fn is not None:
            full_scores = [float(v) for v in eval_full_fn()]
            if any(
                (not math.isfinite(full_scores[i]))
                or (
                    i < len(parent_pointwise_scores)
                    and abs(full_scores[i] - float(parent_pointwise_scores[i])) > self.hard_divergence_ceiling
                )
                for i in range(len(full_scores))
            ):
                catastrophic_blowup = True
                child_node.fail_class = "sentinel_pruned"
                child_node.valid = False
                child_node.score = round(float(parent_node.score) - self.hard_divergence_ceiling, 6)
                child_node.delta_vs_parent = round(-self.hard_divergence_ceiling, 6)
                child_node.meta = dict(child_node.meta or {})
                child_node.meta["sentinel_aborted"] = True
                child_node.meta["sentinel_cost"] = full_cost
                child_node.meta["sentinel_ucb"] = round(ucb_gain, 6)
                child_node.meta["sentinel_lcb"] = round(ucb_gain, 6)
                child_node.meta["exempt_adversarial"] = sentinel_slice.exempt_adversarial
                return {
                    "passed": False,
                    "aborted": True,
                    "cost_spent": full_cost,
                    "compute_saved": 0.0,
                    "delta_mean": round(delta_mu, 6),
                    "se": round(se_mu, 6),
                    "se_stratified": round(se_mu, 6),
                    "se_pooled": round(pooled_se, 6),
                    "ucb95_gain": round(ucb_gain, 6),
                    "ucb": round(ucb_gain, 6),
                    "lcb95_gain": round(ucb_gain, 6),
                    "lcb": round(ucb_gain, 6),
                    "parent_trim_locked": True,
                    "catastrophic_blowup": True,
                    "exempt_adversarial": sentinel_slice.exempt_adversarial,
                    "slice_size": k,
                    "node": child_node,
                }
            full_mean = sum(full_scores) / float(max(1, len(full_scores)))
            child_node.score = round(full_mean, 6)
            child_node.delta_vs_parent = round(full_mean - float(parent_node.score), 6)
        child_node.fail_class = "ok"
        child_node.valid = True
        child_node.meta = dict(child_node.meta or {})
        child_node.meta["sentinel_aborted"] = False
        child_node.meta["sentinel_cost"] = 1.0
        child_node.meta["sentinel_ucb"] = round(ucb_gain, 6)
        child_node.meta["sentinel_lcb"] = round(ucb_gain, 6)
        child_node.meta["exempt_adversarial"] = sentinel_slice.exempt_adversarial
        return {
            "passed": True,
            "aborted": False,
            "cost_spent": full_cost,
            "compute_saved": 0.0,
            "delta_mean": round(delta_mu, 6),
            "se": round(se_mu, 6),
            "se_stratified": round(se_mu, 6),
            "se_pooled": round(pooled_se, 6),
            "ucb95_gain": round(ucb_gain, 6),
            "ucb": round(ucb_gain, 6),
            "lcb95_gain": round(ucb_gain, 6),
            "lcb": round(ucb_gain, 6),
            "parent_trim_locked": True,
            "catastrophic_blowup": catastrophic_blowup,
            "exempt_adversarial": sentinel_slice.exempt_adversarial,
            "slice_size": k,
            "node": child_node,
        }

__all__ = [
    "SentinelSlice",
    "WaldMicroSentinel",
]

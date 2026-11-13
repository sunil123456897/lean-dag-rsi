"""
SLEEP-DAG-RSI v7/v8 Active boundary counterexample & VeriHarness
dual-head disagreement/consensus synthesizer.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class ActiveBoundarySynthesizer:
    """
    SLEEP-DAG-RSI v7 Active Disagreement & Boundary Counterexample Synthesizer:
    Breaks the passive historical replay ceiling by actively synthesizing inputs
    `x* in B(X_anchor, r)` that maximize committee disagreement or straddle the
    zero/decision boundary between competing policies, models, or experts:
      1. Bichromatic Boundary Bisection: locates anchor pairs `(x_i, x_j)` where
         committee models reverse rank or flip sign, and bisects along the connecting
         chord to land directly on the decision boundary.
      2. Projected Gradient Ascent on Disagreement `D(x)`: takes central finite-difference
         steps along `grad_x Var_k(f_k(x))` clamped within the local trust-region hyperbox
         `[x_anchor - r * sigma, x_anchor + r * sigma]` and global bounds.
      3. Feeds high-disagreement boundary states into `WaldMicroSentinel` (active veto)
         and `NeuroSymbolicDistiller` (active tail distillation).
    """

    def __init__(
        self,
        trust_radius_sigma: float = 1.5,
        ascent_steps: int = 8,
        step_lr: float = 0.20,
        fd_eps: float = 1e-3,
        probe_zero_boundary: bool = True,
        max_active_coords: int = 16,
    ) -> None:
        self.trust_radius_sigma = float(trust_radius_sigma)
        self.ascent_steps = max(1, int(ascent_steps))
        self.step_lr = float(step_lr)
        self.fd_eps = float(fd_eps)
        self.probe_zero_boundary = bool(probe_zero_boundary)
        self.max_active_coords = max(1, int(max_active_coords))

    def _screen_active_coordinates(
        self,
        base_x: Sequence[float],
        col_stds: Sequence[float],
        committee_fns: Sequence[Callable[[Sequence[float]], Any]],
    ) -> List[int]:
        d = len(base_x)
        if d <= self.max_active_coords:
            return list(range(d))
        fn0 = committee_fns[0]
        base_v = self._to_vec(fn0(base_x))
        sens: List[Tuple[float, int]] = []
        for c in range(d):
            h = max(1e-5, self.fd_eps * col_stds[c])
            x_p = list(base_x)
            x_p[c] += h
            vp = self._to_vec(fn0(x_p))
            diff = sum(abs(vp[k] - base_v[k]) for k in range(min(len(vp), len(base_v))))
            sens.append((diff * col_stds[c], c))
        sens.sort(reverse=True)
        return sorted(idx for _, idx in sens[: self.max_active_coords])

    @staticmethod
    def _to_vec(val: Any) -> List[float]:
        if isinstance(val, (int, float)):
            return [float(val)]
        if hasattr(val, "tolist"):
            flat = val.tolist()
            if isinstance(flat, (int, float)):
                return [float(flat)]
            if isinstance(flat, list) and flat and isinstance(flat[0], list):
                return [float(x) for x in flat[0]]
            return [float(x) for x in flat]
        if isinstance(val, (list, tuple)):
            return [float(x) for x in val]
        return [float(val)]

    def evaluate_disagreement(
        self,
        x: Sequence[float],
        committee_fns: Sequence[Callable[[Sequence[float]], Any]],
    ) -> Dict[str, Any]:
        preds = [self._to_vec(fn(x)) for fn in committee_fns]
        k_models = len(preds)
        if k_models < 2 or not preds[0]:
            return {"objective": 0.0, "variance": 0.0, "sign_conflict": False, "preds": preds}
        d_out = len(preds[0])
        mean_vec = [
            sum(preds[m][c] for m in range(k_models)) / float(k_models)
            for c in range(d_out)
        ]
        variance = (
            sum(
                sum((preds[m][c] - mean_vec[c]) ** 2 for c in range(d_out))
                for m in range(k_models)
            )
            / float(k_models)
        )
        # Check if committee members disagree on the sign of channel 0 or have large gap
        signs = {1 if preds[m][0] >= 0.0 else -1 for m in range(k_models)}
        sign_conflict = len(signs) > 1
        boundary_bonus = 0.0
        if self.probe_zero_boundary:
            mean_mag = math.sqrt(sum(v * v for v in mean_vec) / float(d_out))
            boundary_bonus = 0.15 * variance * math.exp(-mean_mag)
        return {
            "objective": variance + boundary_bonus,
            "variance": variance,
            "sign_conflict": sign_conflict,
            "preds": preds,
        }

    def synthesize_boundary_points(
        self,
        anchor_points: Sequence[Sequence[float]],
        committee_fns: Sequence[Callable[[Sequence[float]], Any]],
        num_synthetic: int = 8,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        n_anc = len(anchor_points)
        if n_anc == 0 or len(committee_fns) < 2:
            return {
                "synthetic_points": [],
                "anchor_mean_disagreement": 0.0,
                "synthetic_mean_disagreement": 0.0,
                "disagreement_amplification": 1.0,
                "boundary_crossings_found": 0,
            }

        d = len(anchor_points[0])
        anchors = [[float(v) for v in row] for row in anchor_points]

        # Compute per-coordinate std and bounds across anchors
        col_means = [sum(anchors[r][c] for r in range(n_anc)) / float(n_anc) for c in range(d)]
        col_stds = [
            max(
                1e-4,
                math.sqrt(
                    sum((anchors[r][c] - col_means[c]) ** 2 for r in range(n_anc))
                    / float(max(1, n_anc - 1))
                ),
            )
            for c in range(d)
        ]
        if bounds is not None and len(bounds) == d:
            global_lows = [float(b[0]) for b in bounds]
            global_highs = [float(b[1]) for b in bounds]
        else:
            global_lows = [
                min(anchors[r][c] for r in range(n_anc)) - self.trust_radius_sigma * col_stds[c]
                for c in range(d)
            ]
            global_highs = [
                max(anchors[r][c] for r in range(n_anc)) + self.trust_radius_sigma * col_stds[c]
                for c in range(d)
            ]

        anchor_evals = [self.evaluate_disagreement(x, committee_fns) for x in anchors]
        anchor_mean_dis = (
            sum(ev["variance"] for ev in anchor_evals) / float(n_anc)
        )

        # 1. Build initial seed pool: top high-disagreement anchors + bichromatic chord midpoints
        seeds: List[Tuple[List[float], List[float]]] = []  # (seed_x, trust_center_x)
        ranked_anc = sorted(
            range(n_anc), key=lambda i: anchor_evals[i]["objective"], reverse=True
        )
        for i in ranked_anc[: max(2, num_synthetic // 2)]:
            seeds.append((list(anchors[i]), list(anchors[i])))

        # Bichromatic chord search between anchors where difference (f_0 - f_1) flips sign
        diff_vals = [
            anchor_evals[i]["preds"][0][0] - anchor_evals[i]["preds"][1][0]
            for i in range(n_anc)
        ]
        for i in range(min(n_anc, 24)):
            for j in range(i + 1, min(n_anc, 24)):
                if diff_vals[i] * diff_vals[j] < 0.0:
                    denom = abs(diff_vals[i]) + abs(diff_vals[j]) + 1e-12
                    alpha = abs(diff_vals[j]) / denom
                    mid = [
                        alpha * anchors[i][c] + (1.0 - alpha) * anchors[j][c]
                        for c in range(d)
                    ]
                    seeds.append((mid, mid))
                    if len(seeds) >= num_synthetic * 2:
                        break
            if len(seeds) >= num_synthetic * 2:
                break

        # Pad seeds if needed
        while len(seeds) < num_synthetic:
            base_idx = len(seeds) % n_anc
            pert = [
                min(
                    global_highs[c],
                    max(
                        global_lows[c],
                        anchors[base_idx][c]
                        + 0.25 * col_stds[c] * (1.0 if (len(seeds) + c) % 2 == 0 else -1.0),
                    ),
                )
                for c in range(d)
            ]
            seeds.append((pert, list(anchors[base_idx])))

        active_dims = self._screen_active_coordinates(anchors[0], col_stds, committee_fns)

        # 2. Projected Gradient Ascent on committee disagreement D(x)
        optimized_candidates: List[Tuple[float, bool, List[float]]] = []
        for seed_x, center_x in seeds[: max(num_synthetic, len(seeds))]:
            curr_x = list(seed_x)
            lows_loc = [
                max(global_lows[c], center_x[c] - self.trust_radius_sigma * col_stds[c])
                for c in range(d)
            ]
            highs_loc = [
                min(global_highs[c], center_x[c] + self.trust_radius_sigma * col_stds[c])
                for c in range(d)
            ]
            for _ in range(self.ascent_steps):
                grad = [0.0] * d
                for c in active_dims:
                    h = max(1e-5, self.fd_eps * col_stds[c])
                    x_p = list(curr_x)
                    x_m = list(curr_x)
                    x_p[c] = min(highs_loc[c], curr_x[c] + h)
                    x_m[c] = max(lows_loc[c], curr_x[c] - h)
                    denom = max(1e-12, x_p[c] - x_m[c])
                    d_p = self.evaluate_disagreement(x_p, committee_fns)["objective"]
                    d_m = self.evaluate_disagreement(x_m, committee_fns)["objective"]
                    grad[c] = (d_p - d_m) / denom
                g_norm = math.sqrt(sum(g * g for g in grad))
                if g_norm < 1e-12:
                    break
                for c in active_dims:
                    step = self.step_lr * col_stds[c] * (grad[c] / g_norm)
                    curr_x[c] = min(highs_loc[c], max(lows_loc[c], curr_x[c] + step))

            ev_final = self.evaluate_disagreement(curr_x, committee_fns)
            optimized_candidates.append(
                (ev_final["variance"], ev_final["sign_conflict"], curr_x)
            )

        optimized_candidates.sort(key=lambda item: item[0], reverse=True)
        chosen = optimized_candidates[:num_synthetic]
        synth_points = [item[2] for item in chosen]
        synth_mean_dis = (
            sum(item[0] for item in chosen) / float(max(1, len(chosen)))
        )
        crossings = sum(1 for item in chosen if item[1])
        amp = synth_mean_dis / max(anchor_mean_dis, 1e-9)

        return {
            "synthetic_points": synth_points,
            "anchor_mean_disagreement": round(anchor_mean_dis, 6),
            "synthetic_mean_disagreement": round(synth_mean_dis, 6),
            "disagreement_amplification": round(amp, 4),
            "boundary_crossings_found": crossings,
        }

    def evaluate_consensus_fragility(
        self,
        x: Sequence[float],
        committee_fns: Sequence[Callable[[Sequence[float]], Any]],
        consensus_var_tol: float = 0.05,
    ) -> Dict[str, Any]:
        """
        SLEEP-DAG-RSI v8 Consensus Challenger Evaluation (`VeriHarness`, Tier A #8):
        Inspects points where all committee models agree (`variance <= consensus_var_tol`
        and unanimous sign), yet the shared stability margin `M_cons(x) = min_k |f_k(x)_0|`
        is small. High fragility score indicates a collective blind spot where unanimous
        agreement masks proximity to a shared failure cliff.
        """
        preds = [self._to_vec(fn(x)) for fn in committee_fns]
        k_models = len(preds)
        if k_models < 1 or not preds[0]:
            return {
                "fragility": 0.0,
                "consensus_margin": 0.0,
                "mean_pred": 0.0,
                "variance": 0.0,
                "is_unanimous_sign": False,
                "is_consensus_state": False,
                "preds": preds,
            }
        d_out = len(preds[0])
        mean_vec = [
            sum(preds[m][c] for m in range(k_models)) / float(k_models)
            for c in range(d_out)
        ]
        variance = (
            sum(
                sum((preds[m][c] - mean_vec[c]) ** 2 for c in range(d_out))
                for m in range(k_models)
            )
            / float(k_models)
        )
        signs = {1 if preds[m][0] >= 0.0 else -1 for m in range(k_models)}
        is_unanimous = len(signs) == 1
        consensus_margin = min(abs(preds[m][0]) for m in range(k_models))
        is_consensus = bool(is_unanimous and variance <= consensus_var_tol)
        # Fragility rises sharply as consensus_margin -> 0 while variance remains near 0
        fragility = math.exp(-10.0 * variance) / (consensus_margin + 1e-3)
        return {
            "fragility": fragility,
            "consensus_margin": consensus_margin,
            "mean_pred": mean_vec[0],
            "variance": variance,
            "is_unanimous_sign": is_unanimous,
            "is_consensus_state": is_consensus,
            "preds": preds,
        }

    def challenge_consensus_points(
        self,
        anchor_points: Sequence[Sequence[float]],
        committee_fns: Sequence[Callable[[Sequence[float]], Any]],
        num_challengers: int = 4,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
        consensus_var_tol: float = 0.05,
        lambda_var: float = 2.0,
        ground_truth_oracle_fn: Optional[Callable[[Sequence[float]], float]] = None,
    ) -> Dict[str, Any]:
        """
        VeriHarness Second Head — `Consensus Challenger` (Tier A #8, Eq. 16):
        Targets regions where all top candidates agree (`variance ~ 0`), so standard
        disagreement maximization (`grad_x Var_k`) has zero gradient and is blind.
        Minimizes the smooth signed consensus margin anchored at `s_anc = sign(f_bar(x_anc))`:
            M_cons(x) = s_anc * f_bar(x) + lambda_var * sqrt(Var_k(f_k(x)) + eps)
        so the gradient does not flip when crossing `f_bar(x) = 0`, pushing the unanimous
        committee across the decision boundary together while keeping inter-model variance low.
        When `ground_truth_oracle_fn` is provided, verifies whether the synthesized point
        is a genuine collective blind spot where unanimous committee predictions disagree
        with the exact oracle.
        """
        n_anc = len(anchor_points)
        if n_anc == 0 or len(committee_fns) < 1:
            return {
                "challenger_points": [],
                "anchor_mean_consensus_margin": 0.0,
                "challenged_mean_consensus_margin": 0.0,
                "margin_collapse_ratio": 1.0,
                "collective_blind_spots_found": 0,
                "unanimous_flips_found": 0,
                "oracle_verified_blind_spots": 0,
            }

        d = len(anchor_points[0])
        anchors = [[float(v) for v in row] for row in anchor_points]
        col_means = [sum(anchors[r][c] for r in range(n_anc)) / float(n_anc) for c in range(d)]
        col_stds = [
            max(
                1e-4,
                math.sqrt(
                    sum((anchors[r][c] - col_means[c]) ** 2 for r in range(n_anc))
                    / float(max(1, n_anc - 1))
                ),
            )
            for c in range(d)
        ]
        if bounds is not None and len(bounds) == d:
            global_lows = [float(b[0]) for b in bounds]
            global_highs = [float(b[1]) for b in bounds]
        else:
            global_lows = [
                min(anchors[r][c] for r in range(n_anc)) - self.trust_radius_sigma * col_stds[c]
                for c in range(d)
            ]
            global_highs = [
                max(anchors[r][c] for r in range(n_anc)) + self.trust_radius_sigma * col_stds[c]
                for c in range(d)
            ]

        anc_evals = [
            self.evaluate_consensus_fragility(x, committee_fns, consensus_var_tol=consensus_var_tol)
            for x in anchors
        ]
        consensus_indices = [
            i for i in range(n_anc) if anc_evals[i]["is_consensus_state"]
        ]
        if not consensus_indices:
            # Fall back to lowest-variance anchors
            consensus_indices = sorted(range(n_anc), key=lambda i: anc_evals[i]["variance"])[
                : max(1, num_challengers)
            ]

        # Rank consensus anchors by highest fragility (lowest safety margin at low variance)
        consensus_indices.sort(key=lambda i: anc_evals[i]["fragility"], reverse=True)
        anchor_mean_margin = sum(
            anc_evals[i]["consensus_margin"] for i in consensus_indices
        ) / float(max(1, len(consensus_indices)))

        active_dims = self._screen_active_coordinates(anchors[consensus_indices[0]], col_stds, committee_fns)

        challenged_candidates: List[Tuple[float, float, bool, bool, bool, List[float]]] = []
        for idx_pos in range(min(num_challengers * 2, max(num_challengers, len(consensus_indices)))):
            anc_i = consensus_indices[idx_pos % len(consensus_indices)]
            center_x = list(anchors[anc_i])
            curr_x = list(center_x)
            orig_sign = 1.0 if anc_evals[anc_i]["mean_pred"] >= 0.0 else -1.0
            orig_margin = max(1e-9, anc_evals[anc_i]["consensus_margin"])

            lows_loc = [
                max(global_lows[c], center_x[c] - self.trust_radius_sigma * col_stds[c])
                for c in range(d)
            ]
            highs_loc = [
                min(global_highs[c], center_x[c] + self.trust_radius_sigma * col_stds[c])
                for c in range(d)
            ]

            for _ in range(self.ascent_steps):
                # Minimize smooth signed consensus margin (Eq. 16):
                #   M_cons(x) = s_anc * mean_pred(x) + lambda_var * sqrt(variance(x) + eps)
                grad = [0.0] * d
                for c in active_dims:
                    h = max(1e-5, self.fd_eps * col_stds[c])
                    x_p = list(curr_x)
                    x_m = list(curr_x)
                    x_p[c] = min(highs_loc[c], curr_x[c] + h)
                    x_m[c] = max(lows_loc[c], curr_x[c] - h)
                    denom = max(1e-12, x_p[c] - x_m[c])
                    ev_p = self.evaluate_consensus_fragility(
                        x_p, committee_fns, consensus_var_tol=consensus_var_tol
                    )
                    ev_m = self.evaluate_consensus_fragility(
                        x_m, committee_fns, consensus_var_tol=consensus_var_tol
                    )
                    loss_p = orig_sign * ev_p["mean_pred"] + float(lambda_var) * math.sqrt(
                        ev_p["variance"] + 1e-12
                    )
                    loss_m = orig_sign * ev_m["mean_pred"] + float(lambda_var) * math.sqrt(
                        ev_m["variance"] + 1e-12
                    )
                    grad[c] = (loss_p - loss_m) / denom

                g_norm = math.sqrt(sum(g * g for g in grad))
                if g_norm < 1e-12:
                    break
                for c in active_dims:
                    step = -self.step_lr * col_stds[c] * (grad[c] / g_norm)
                    curr_x[c] = min(highs_loc[c], max(lows_loc[c], curr_x[c] + step))

            ev_fin = self.evaluate_consensus_fragility(
                curr_x, committee_fns, consensus_var_tol=consensus_var_tol
            )
            new_sign = 1.0 if ev_fin["mean_pred"] >= 0.0 else -1.0
            unanimous_flip = bool(ev_fin["is_unanimous_sign"] and (new_sign != orig_sign))
            margin_collapsed = bool(
                unanimous_flip or (ev_fin["consensus_margin"] < 0.5 * orig_margin)
            )
            oracle_blind_spot = False
            if ground_truth_oracle_fn is not None:
                oracle_val = float(ground_truth_oracle_fn(curr_x))
                oracle_sign = 1.0 if oracle_val >= 0.0 else -1.0
                oracle_blind_spot = bool(
                    ev_fin["is_unanimous_sign"] and (new_sign != oracle_sign)
                )
            score_key = (
                (1000.0 + abs(ev_fin["mean_pred"]))
                if unanimous_flip
                else ev_fin["fragility"]
            )
            challenged_candidates.append(
                (
                    score_key,
                    ev_fin["consensus_margin"],
                    margin_collapsed,
                    unanimous_flip,
                    oracle_blind_spot,
                    curr_x,
                )
            )

        challenged_candidates.sort(key=lambda item: item[0], reverse=True)
        chosen = challenged_candidates[:num_challengers]
        chal_points = [item[5] for item in chosen]
        chal_mean_margin = sum(item[1] for item in chosen) / float(max(1, len(chosen)))
        blind_spots = sum(1 for item in chosen if item[2])
        unanimous_flips = sum(1 for item in chosen if item[3])
        oracle_blind_spots = sum(1 for item in chosen if item[4])
        collapse_ratio = chal_mean_margin / max(anchor_mean_margin, 1e-9)

        return {
            "challenger_points": chal_points,
            "anchor_mean_consensus_margin": round(anchor_mean_margin, 6),
            "challenged_mean_consensus_margin": round(chal_mean_margin, 6),
            "margin_collapse_ratio": round(collapse_ratio, 6),
            "collective_blind_spots_found": blind_spots,
            "unanimous_flips_found": unanimous_flips,
            "oracle_verified_blind_spots": oracle_blind_spots,
        }

    def synthesize_dual_head_verification(
        self,
        anchor_points: Sequence[Sequence[float]],
        committee_fns: Sequence[Callable[[Sequence[float]], Any]],
        num_disagreement: int = 4,
        num_consensus_challengers: int = 4,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
        consensus_var_tol: float = 0.05,
        num_challenger: Optional[int] = None,
        ground_truth_oracle_fn: Optional[Callable[[Sequence[float]], float]] = None,
    ) -> Dict[str, Any]:
        """
        VeriHarness Dual-Head Verification (`Disagreement Resolver + Consensus Challenger`, Tier A #8):
        Combines:
          - Head 1 (`Disagreement Resolver`): finds boundary inputs maximizing committee
            disagreement `Var_k(f_k(x))` to separate competing candidates.
          - Head 2 (`Consensus Challenger`): finds perturbations at unanimous low-margin
            states (`Var_k ~ 0`) that collapse or flip collective predictions.
        """
        n_chal = int(num_challenger) if num_challenger is not None else int(num_consensus_challengers)
        head_disagreement = self.synthesize_boundary_points(
            anchor_points=anchor_points,
            committee_fns=committee_fns,
            num_synthetic=num_disagreement,
            bounds=bounds,
        )
        head_consensus = self.challenge_consensus_points(
            anchor_points=anchor_points,
            committee_fns=committee_fns,
            num_challengers=n_chal,
            bounds=bounds,
            consensus_var_tol=consensus_var_tol,
            ground_truth_oracle_fn=ground_truth_oracle_fn,
        )
        combined = list(head_disagreement["synthetic_points"]) + list(
            head_consensus["challenger_points"]
        )
        return {
            "disagreement_head": head_disagreement,
            "disagreement_resolver_head": head_disagreement,
            "consensus_challenger_head": head_consensus,
            "combined_verification_points": combined,
            "total_verification_points": len(combined),
            "total_boundary_crossings": int(head_disagreement["boundary_crossings_found"]),
            "collective_blind_spots_found": int(
                head_consensus["collective_blind_spots_found"]
            ),
            "unanimous_flips_found": int(head_consensus["unanimous_flips_found"]),
            "oracle_verified_blind_spots": int(
                head_consensus["oracle_verified_blind_spots"]
            ),
        }

__all__ = [
    "ActiveBoundarySynthesizer",
]

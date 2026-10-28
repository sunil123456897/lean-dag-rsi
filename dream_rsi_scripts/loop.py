"""
Dream-RSI closed loop.

Ties the four stages together so recursive improvement actually runs:

    explore (elsewhere, recorded as trees)
        -> replay simulator          (engine.replay)
        -> propose candidate policies (this module: parametric or LLM-authored)
        -> select by mean replay score over ALL recorded trees   (engine.select_policy)
        -> persist the winner, redeploy, repeat

Policy state is stored as JSON, never as code, so a pool entry is data and can be
diffed, reviewed, and rolled back.

Two ways to propose candidates:

  A. parametric -- small variations around the incumbent. Needs no model at all
     and is the cheapest thing that still counts as dreaming.
  B. authored -- render a prompt (prompts/policy_improvement.md), hand it to a
     coding agent, take back a policy spec or a policy file. This is the half
     that makes it recursive self-improvement rather than policy search.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from engine import (
    DiscoveryTree,
    DynamicPortfolio,
    Policy,
    Problem,
    build_policy,
    default_pool_dir,
    ideate_over_history,
    load_pool,
    policy_spec,
    replay_over_history,
    select_policy,
    sweep_beta,
)

STATE_NAME = "loop_state.json"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def state_path(task: str, pool_dir: Optional[str] = None) -> str:
    root = pool_dir or default_pool_dir()
    return os.path.join(root, _safe(task), STATE_NAME)


def state_dir(task: str, pool_dir: Optional[str] = None) -> str:
    root = pool_dir or default_pool_dir()
    return os.path.join(root, _safe(task))


def load_state(task: str, pool_dir: Optional[str] = None) -> Dict[str, Any]:
    path = state_path(task, pool_dir)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_state(task: str, state: Dict[str, Any], pool_dir: Optional[str] = None) -> str:
    path = state_path(task, pool_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, ensure_ascii=False)
    return path


def initial_state(
    task: str,
    current: Optional[Policy] = None,
    beta1: float = 0.01,
    beta2: float = 0.05,
) -> Dict[str, Any]:
    pol = current or DynamicPortfolio(beta=0.6)
    return {
        "task": task,
        "round": 0,
        "current": policy_spec(pol, label="incumbent"),
        "beta1": beta1,
        "beta2": beta2,
        "history": [],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------


def parametric_candidates(
    current: Policy, include_incumbent: bool = True
) -> Dict[str, Policy]:
    """
    Cheap dreaming: vary the incumbent's own knobs.

    Always includes the incumbent, which is what makes selection monotone.
    """
    base = dict(getattr(current, "params", lambda: {"beta": current.beta})())
    out: Dict[str, Policy] = {}
    if include_incumbent:
        out["incumbent"] = build_policy(
            {"kind": getattr(current, "kind", "dynamic_portfolio"), **base}
        )

    beta = float(base.get("beta", 0.6))
    for delta in (-0.3, -0.15, 0.15, 0.3):
        v = min(1.0, max(0.0, round(beta + delta, 3)))
        spec = dict(base)
        spec["beta"] = v
        try:
            out[f"beta={v}"] = build_policy(
                {"kind": getattr(current, "kind", "dynamic_portfolio"), **spec}
            )
        except (TypeError, ValueError):
            continue

    for th in (0.2, 0.5, 0.7):
        spec = dict(base)
        spec["explore_threshold"] = th
        try:
            out[f"explore_threshold={th}"] = build_policy(
                {"kind": getattr(current, "kind", "dynamic_portfolio"), **spec}
            )
        except (TypeError, ValueError):
            continue

    for mr in (0, 2):
        spec = dict(base)
        spec["max_recovery"] = mr
        try:
            out[f"max_recovery={mr}"] = build_policy(
                {"kind": getattr(current, "kind", "dynamic_portfolio"), **spec}
            )
        except (TypeError, ValueError):
            continue

    for kind in ("optimal_policy", "lean_dag", "deepest_first", "widest_first"):
        try:
            out[kind] = build_policy({"kind": kind, "beta": beta})
        except ValueError:
            continue
    return out


def parse_proposal(payload: Any) -> Dict[str, Dict[str, Any]]:
    """
    Accept what a coding agent is most likely to return.

    - {"kind": ..., "beta": ...}                  -> one candidate
    - [ {...}, {...} ]                            -> several
    - {"candidates": [...], "incumbent": {...}}   -> the documented shape
    """
    if payload is None:
        return {}
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict) and "candidates" in payload:
        items = list(payload.get("candidates") or [])
        if payload.get("incumbent"):
            items = [payload["incumbent"]] + items
    elif isinstance(payload, dict):
        items = [payload]
    else:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for i, item in enumerate(items):
        if not isinstance(item, dict) or "kind" not in item:
            continue
        label = str(item.get("label") or f"proposal_{i}")
        spec = {k: v for k, v in item.items() if k != "label"}
        out[label] = spec
    return out


def candidates_from_specs(specs: Dict[str, Dict[str, Any]]) -> Dict[str, Policy]:
    out: Dict[str, Policy] = {}
    for label, spec in specs.items():
        try:
            out[label] = build_policy(spec)
        except (ValueError, TypeError):
            continue
    return out


# ---------------------------------------------------------------------------
# One round
# ---------------------------------------------------------------------------


def run_round(
    task: str,
    problem: Optional[Problem] = None,
    pool_dir: Optional[str] = None,
    provided_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    include_parametric: bool = True,
    persist: bool = True,
) -> Dict[str, Any]:
    """
    One full loop iteration: propose -> replay -> select -> persist.

    Returns a report with the ranking, the winner, and whether the incumbent was
    improved upon.
    """
    state = load_state(task, pool_dir)
    if not state:
        state = initial_state(task)

    trees: List[DiscoveryTree] = load_pool(task=task, pool_dir=pool_dir)
    if not trees:
        return {
            "task": task,
            "error": "no recorded trees for this task; record an exploration round first",
            "trees": 0,
        }

    problem = problem or Problem()
    beta1 = float(state.get("beta1", 0.01))
    beta2 = float(state.get("beta2", 0.05))

    incumbent: Policy = build_policy(state["current"])
    incumbent_mean = replay_over_history(trees, incumbent, problem, beta1, beta2)[
        "mean_v"
    ]

    candidates: Dict[str, Policy] = {}
    if include_parametric:
        candidates.update(parametric_candidates(incumbent, include_incumbent=True))
    if provided_specs:
        candidates.update(candidates_from_specs(provided_specs))
    if "incumbent" not in candidates:
        candidates["incumbent"] = incumbent

    selection = select_policy(candidates, trees, problem, beta1, beta2)
    winner_name = selection["best"]
    winner_spec = policy_spec(candidates[winner_name], label=winner_name)

    improved = selection["best_mean_v"] > incumbent_mean + 1e-12
    state["round"] = int(state.get("round", 0)) + 1
    state["current"] = winner_spec
    state["history"].append(
        {
            "round": state["round"],
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "incumbent_mean_v": round(incumbent_mean, 6),
            "winner": winner_name,
            "winner_mean_v": round(selection["best_mean_v"], 6),
            "improved": improved,
            "ranking": selection["ranking"],
            "trees": len(trees),
        }
    )
    if persist:
        save_state(task, state, pool_dir)

    return {
        "task": task,
        "round": state["round"],
        "trees": len(trees),
        "candidates": len(candidates),
        "incumbent_mean_v": round(incumbent_mean, 6),
        "winner": winner_name,
        "winner_spec": winner_spec,
        "winner_mean_v": round(selection["best_mean_v"], 6),
        "improved": improved,
        "ranking": selection["ranking"],
        "beta_sweep": sweep_beta(trees, problem, values=[i / 10 for i in range(11)])[
            :4
        ],
        "ideation": ideate_over_history(trees, problem),
    }


def run_loop(
    task: str,
    rounds: int = 3,
    problem: Optional[Problem] = None,
    pool_dir: Optional[str] = None,
    specs_per_round: Optional[List[Dict[str, Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Several rounds back to back. Monotone, so later rounds cannot regress."""
    reports: List[Dict[str, Any]] = []
    for i in range(rounds):
        specs = None
        if specs_per_round and i < len(specs_per_round):
            specs = specs_per_round[i]
        reports.append(run_round(task, problem, pool_dir, provided_specs=specs))
    return reports


# ---------------------------------------------------------------------------
# Prompt rendering (the authored-proposal path)
# ---------------------------------------------------------------------------

DEFAULT_PROMPT = "prompts/policy_improvement.md"


def _skill_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def render_prompt(
    task: str,
    pool_dir: Optional[str] = None,
    problem: Optional[Problem] = None,
    template: Optional[str] = None,
    top: int = 8,
) -> str:
    """
    Produce the prompt that asks a coding agent for the next policy version.

    Fails loudly with an explicit reason when there is nothing to learn from,
    rather than handing the agent a prompt it cannot act on.
    """
    trees = load_pool(task=task, pool_dir=pool_dir)
    problem = problem or Problem()

    path = template or os.path.join(_skill_root(), DEFAULT_PROMPT)
    if not os.path.exists(path):
        return f"ERROR: prompt template not found at {path}"

    with open(path, "r", encoding="utf-8") as fh:
        body = fh.read()

    state = load_state(task, pool_dir) or initial_state(task)
    current = state["current"]

    # Facts about the recorded history, derived rather than guessed.
    total_nodes = sum(len(t.nodes) for t in trees)
    branches = sum(
        len({n.branch for n in t.nodes.values() if n.branch >= 0}) for t in trees
    )
    classes: Dict[str, int] = {}
    for t in trees:
        for n in t.nodes.values():
            cls = problem.class_of(n)
            classes[cls] = classes.get(cls, 0) + 1
    failures = sorted(
        ((c, k) for c, k in classes.items() if c != "ok"),
        key=lambda kv: kv[1],
        reverse=True,
    )[:top]

    incumbent_mean = (
        replay_over_history(
            trees,
            build_policy(current),
            problem,
            float(state.get("beta1", 0.01)),
            float(state.get("beta2", 0.05)),
        )["mean_v"]
        if trees
        else 0.0
    )
    sweep = (
        sweep_beta(trees, problem, values=[i / 10 for i in range(11)])[:top]
        if trees
        else []
    )

    repl = {
        "task": task,
        "trees": str(len(trees)),
        "nodes": str(total_nodes),
        "branches": str(branches),
        "round": str(state.get("round", 0)),
        "current_policy_json": json.dumps(current, ensure_ascii=False, indent=2),
        "failure_classes": ", ".join(f"{c}={k}" for c, k in failures)
        or "none recorded",
        "incumbent_mean_v": f"{incumbent_mean:.6f}",
        "beta_sweep": ", ".join(f"{b}:{v}" for b, v in sweep) or "not available",
        "max_parallelism": str(max((t.max_parallelism for t in trees), default=4)),
        "kinds": ", ".join(
            sorted(
                {getattr(build_policy(current), "kind", "custom")}
                | {
                    "deepest_first",
                    "widest_first",
                    "dynamic_portfolio",
                    "optimal_policy",
                    "lean_dag",
                }
            )
        ),
        "pool_dir": pool_dir or default_pool_dir(),
    }
    for key, value in repl.items():
        body = body.replace("{{" + key + "}}", value)
    return body


def _safe(text: str) -> str:
    keep = "-_. "
    return (
        "".join(c if (c.isalnum() or c in keep) else "_" for c in text).strip()
        or "task"
    )

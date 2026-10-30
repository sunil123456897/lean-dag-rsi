"""
Dream-RSI universal engine.

The replay simulator is domain-agnostic: it needs a tree of scored attempts and
nothing else. All task-specific knowledge lives behind the Problem interface, so
the same engine drives algorithm tuning, prompt iteration, config search,
schedule optimisation, hypothesis elimination, or bug diagnosis.

Guarantees
----------
- Deterministic: replay never calls a model and never executes anything.
- Prefix-only: a policy sees only nodes it has already revealed.
- Append-only history: trees are never rewritten, so past experience stays valid.

Design constraints inherited from the paper (Zheng et al., 2026, §3 & Appendix B.2)
-----------------------------------------------------------------------------------
- Only the exploration-policy code is meant to change; the evaluator, the task,
  and the interfaces stay fixed. This module implements the replay half of that
  loop.
- beta is a scalar fixed for the duration of one replay episode and swept by the
  caller, never inferred from observations mid-episode.
- Every behavioral threshold in `OptimalPolicy` routes through `_schedule(beta)`
  so that low beta prunes/stops early (fewer probes N) and high beta explores
  wider/deeper, making the offline beta sweep non-degenerate.
- Opened branches are ranked via full ordered prefix trajectories
  (`successful_anchor`, parent->child gain, stagnation steps, failure/repair
  history), never solely by the last leaf.
- Every policy implements `plan_grid(context) -> GridPlan(branch_count, refine_count, reason)`
  for next-cycle width/depth allocation before a new live grid is created.
"""

from __future__ import annotations

import json
import linecache
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Task adapter
# ---------------------------------------------------------------------------


class Problem:
    """
    Everything the engine needs to know about a domain.

    Subclass and override what you need. The defaults suit any task whose
    attempts produce a single number where larger is better.
    """

    #: Optional human label, used in reports.
    name: str = "generic"

    def score(self, node: "Node") -> float:
        """Quality of one attempt. Larger must mean better."""
        return float(node.score)

    def class_of(self, node: "Node") -> str:
        """Failure class of an attempt. Defaults to what was recorded."""
        return node.fail_class or "ok"

    def is_success(self, node: "Node") -> bool:
        """
        Whether an attempt counts as a valid result.

        The paper's convention (Appendix B.2): a recorded observation with
        `error is None` and `fail_class == "ok"` is a success even when `valid`
        is False. Override for domains where validity gates success.
        """
        return node.error is None and self.class_of(node) == "ok"

    def repairable(self, node: "Node") -> bool:
        """
        Whether a failed attempt is worth repairing rather than pruning.

        Widen this for domains with cheap retries (flaky networks, transient
        resource limits, prompts that merely parsed badly).
        """
        return self.class_of(node) in (
            "syntax_error",
            "shape_mismatch",
            "oom",
            "timeout",
            "transient",
            "parse_error",
        )

    def hard_failure(self, node: "Node") -> bool:
        """A clean run that is still wrong or aborted by the sentinel: an algorithmic dead end."""
        return self.class_of(node) in ("algorithmic_failure", "sentinel_pruned")


# ---------------------------------------------------------------------------
# Records & Grid Planning (Appendix B.2)
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """One attempt. Kept deliberately generic so any domain can be recorded."""

    id: str
    parent_id: Optional[str]
    branch: int
    attempt: int
    score: float
    fail_class: str = "ok"
    error: Optional[str] = None
    evaluated: bool = True
    valid: bool = True
    delta_vs_parent: float = 0.0
    delta_vs_baseline: float = 0.0
    #: Per-example binary outcome vector (0/1) for Error Fingerprint crossover.
    pass_mask: List[int] = field(default_factory=list)
    #: Grouped failure counts or error rates by example tag (e.g. {"opcode=add_alpha": 8.0}).
    failure_breakdown: Dict[str, float] = field(default_factory=dict)
    #: Top CPU hotspot source lines ([{"line_no": int, "code": str, "pct_time": float, "time_ms": float}]).
    line_hotspots: List[Dict[str, Any]] = field(default_factory=list)
    #: Domain-specific extras (diff, prompt text, params, logs, ...).
    artifact: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GridPlan:
    """
    Appendix B.2 (`GridPlan(branch_count=W, refine_count=R, reason=...)`):
    Specifies how many branches (0..W-1) and how many refinements after each
    root (0..R) to allocate for the next live discovery cycle.
    """

    branch_count: int
    refine_count: int
    reason: str = ""


@dataclass
class GridPlanningContext:
    """
    Prefix-safe historical context passed to `Policy.plan_grid(context)` before
    a new live grid is created (Appendix B.2 lines 1192-1248).
    """

    history: List[Dict[str, Any]] = field(default_factory=list)
    hard_max_branch_count: int = 8
    hard_max_refine_count: int = 8
    default_branch_count: int = 3
    default_refine_count: int = 3
    worker_cap: int = 3
    trace_branch_count: Optional[int] = None
    trace_refine_count: Optional[int] = None


@dataclass
class BranchTrajectory:
    """
    Ordered prefix trajectory reconstructed for one opened branch
    (Appendix B.2 lines 1051-1067).
    """

    branch_id: int
    nodes: List[Node]
    frontier_id: str
    depth: int
    successful_anchor: float
    latest_score: float
    last_gain: float
    mean_gain: float
    stagnation_steps: int
    hard_failed: bool
    repairable_failed: bool
    consecutive_failures: int


@dataclass
class DiscoveryTree:
    task: str = "task"
    root_id: str = "root"
    baseline_score: float = 0.0
    max_parallelism: int = 4
    nodes: Dict[str, Node] = field(default_factory=dict)
    children_map: Dict[str, List[str]] = field(default_factory=dict)
    created: str = ""

    def __post_init__(self) -> None:
        if not self.created:
            self.created = time.strftime("%Y-%m-%dT%H:%M:%S")

    def add(self, node: Node) -> Node:
        self.nodes[node.id] = node
        if node.parent_id:
            self.children_map.setdefault(node.parent_id, []).append(node.id)
        return node

    # -- io -----------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "root_id": self.root_id,
            "baseline_score": self.baseline_score,
            "max_parallelism": self.max_parallelism,
            "created": self.created,
            "nodes": {nid: asdict(n) for nid, n in self.nodes.items()},
            "children_map": self.children_map,
        }

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: str) -> "DiscoveryTree":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        tree = cls(
            task=data.get("task", "task"),
            root_id=data.get("root_id", "root"),
            baseline_score=float(data.get("baseline_score", 0.0)),
            max_parallelism=int(data.get("max_parallelism", 4)),
            created=data.get("created", ""),
        )
        for nid, nd in data.get("nodes", {}).items():
            known = {k: v for k, v in nd.items() if k in Node.__dataclass_fields__}
            tree.nodes[nid] = Node(**known)
        tree.children_map = {
            k: list(v) for k, v in data.get("children_map", {}).items()
        }
        return tree


# ---------------------------------------------------------------------------
# History pool: append-only, multi-domain
# ---------------------------------------------------------------------------


def default_pool_dir() -> str:
    """Single pool root. Override per call or with DREAM_RSI_POOL."""
    env = os.environ.get("DREAM_RSI_POOL")
    if env:
        return env
    home = os.path.expanduser("~")
    return os.path.join(home, ".gemini", "antigravity", "dream_rsi_pool")


def save_tree(tree: DiscoveryTree, pool_dir: Optional[str] = None) -> str:
    """Append a finished tree to the pool. Never overwrites an existing one."""
    root = pool_dir or default_pool_dir()
    task_dir = os.path.join(root, _safe(tree.task))
    os.makedirs(task_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(task_dir, f"{stamp}_{_safe(tree.root_id)}.json")
    n = 1
    while os.path.exists(path):
        n += 1
        path = os.path.join(task_dir, f"{stamp}_{_safe(tree.root_id)}_{n}.json")
    return tree.save(path)


def load_pool(
    task: Optional[str] = None, pool_dir: Optional[str] = None
) -> List[DiscoveryTree]:
    """
    Load recorded trees. task=None loads every domain in the pool, which is how
    cross-task pre-calibration works.
    """
    root = pool_dir or default_pool_dir()
    if not os.path.isdir(root):
        return []
    if task:
        dirs = [os.path.join(root, _safe(task))]
    else:
        dirs = [
            os.path.join(root, d)
            for d in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, d))
        ]
    trees: List[DiscoveryTree] = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".json"):
                continue
            # Skip the loop's own bookkeeping files; only recorded trees belong here.
            if name in ("loop_state.json",) or name.startswith("_"):
                continue
            try:
                trees.append(DiscoveryTree.load(os.path.join(d, name)))
            except (OSError, ValueError, TypeError, KeyError):
                continue
    return trees


def _safe(text: str) -> str:
    keep = "-_. "
    return (
        "".join(c if (c.isalnum() or c in keep) else "_" for c in text).strip()
        or "task"
    )


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def _percentile_linear(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(n - 1, lo + 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def compute_tukey_cap(
    tree: DiscoveryTree,
    problem: Optional[Problem] = None,
    tau_cap: float = 1.5,
) -> float:
    """
    Non-parametric Tukey upper fence on positive parent->child gains:
        U(T) = med(Delta s^+) + tau_cap * IQR(Delta s^+)
    Used to bound counterfactual continuation gains on right-censored leaves.
    """
    prob = problem or Problem()
    pos_gains: List[float] = []
    for node in tree.nodes.values():
        if node.parent_id is None or node.parent_id not in tree.nodes:
            continue
        if not prob.is_success(node):
            continue
        parent = tree.nodes[node.parent_id]
        if parent.id != tree.root_id and not prob.is_success(parent):
            continue
        gain = prob.score(node) - prob.score(parent)
        if gain > 0:
            pos_gains.append(gain)
    if not pos_gains:
        return 0.0
    pos_gains.sort()
    med = _percentile_linear(pos_gains, 0.50)
    q25 = _percentile_linear(pos_gains, 0.25)
    q75 = _percentile_linear(pos_gains, 0.75)
    return med + tau_cap * (q75 - q25)


def _get_pass_mask(node: Node) -> List[int]:
    """Extract normalized 0/1 per-example outcome vector from `node.pass_mask` or `node.meta`."""
    raw = node.pass_mask if node.pass_mask else node.meta.get("pass_mask", [])
    if not raw or not isinstance(raw, (list, tuple)):
        return []
    return [1 if int(x) > 0 else 0 for x in raw]


def compute_error_fingerprint(
    node_u: Node,
    node_v: Node,
    baseline_score: float,
    score_u: float,
    score_v: float,
    tukey_cap: float = 0.0,
    crossover_synergy: float = 0.35,
    gamma: float = 0.65,
) -> Dict[str, Any]:
    """
    Error-Fingerprint Complementarity Analyzer for `CROSSOVER(u, v)`:
    If both `node_u` and `node_v` carry aligned binary `pass_mask` vectors of length M:
      - Computes exact union coverage `len(pass_u | pass_v) / M`, unique wins `pass_u & ~pass_v`
        and `pass_v & ~pass_u`, and `oracle_union_gain = union_rate - max(rate_u, rate_v)`.
      - When two branches succeed/fail on the exact same examples (`oracle_union_gain == 0`),
        their synergy is strictly `0.0` (preventing blind score-based crossover of clones).
      - When `oracle_union_gain > 0`, projects hybrid gain proportional to the non-overlapping
        wins of the complementary branch.
    Falls back to scalar `crossover_synergy * min(gain_u, gain_v)` only when `pass_mask` is absent.
    """
    mask_u = _get_pass_mask(node_u)
    mask_v = _get_pass_mask(node_v)
    if mask_u and mask_v and len(mask_u) == len(mask_v):
        m = len(mask_u)
        wins_u = sum(mask_u)
        wins_v = sum(mask_v)
        union_mask = [1 if (a or b) else 0 for a, b in zip(mask_u, mask_v)]
        union_wins = sum(union_mask)
        unique_u = sum(1 for a, b in zip(mask_u, mask_v) if (a and not b))
        unique_v = sum(1 for a, b in zip(mask_u, mask_v) if (b and not a))
        rate_u = wins_u / m
        rate_v = wins_v / m
        union_rate = union_wins / m
        oracle_union_gain = max(0.0, union_rate - max(rate_u, rate_v))
        max_rate = max(rate_u, rate_v, 1e-6)
        max_score = max(score_u, score_v)
        score_per_unit = max(0.0, max_score - baseline_score) / max_rate
        comp_gain = oracle_union_gain * score_per_unit
        projected_score = round(max_score + gamma * comp_gain, 4)
        return {
            "has_fingerprint": True,
            "mask_len": m,
            "rate_u": round(rate_u, 4),
            "rate_v": round(rate_v, 4),
            "union_rate": round(union_rate, 4),
            "oracle_union_gain": round(oracle_union_gain, 4),
            "unique_u": unique_u,
            "unique_v": unique_v,
            "comp_gain": round(comp_gain, 4),
            "projected_score": projected_score,
            "union_mask": union_mask,
        }

    # Clone guard when pass_mask is absent: if nodes belong to the same branch,
    # share identical non-empty params, or share identical code/artifact hashes,
    # synergy is strictly 0.0 (preserving the Zero-Clone-Synergy invariant).
    same_branch = (node_u.branch == node_v.branch) and (node_u.id != "root")
    pu = getattr(node_u, "tunable_params", None) or node_u.meta.get("params")
    pv = getattr(node_v, "tunable_params", None) or node_v.meta.get("params")
    same_params = bool(pu and pv and list(pu) == list(pv))
    hash_u = node_u.meta.get("artifact_hash") or node_u.meta.get("code_hash")
    hash_v = node_v.meta.get("artifact_hash") or node_v.meta.get("code_hash")
    same_hash = bool(hash_u and hash_v and hash_u == hash_v)

    gain_u = max(0.0, score_u - baseline_score)
    gain_v = max(0.0, score_v - baseline_score)
    if same_branch or same_params or same_hash:
        comp_gain = 0.0
    else:
        raw_synergy = crossover_synergy * min(gain_u, gain_v)
        comp_gain = min(tukey_cap, raw_synergy) if tukey_cap > 0.0 else raw_synergy
    projected_score = round(max(score_u, score_v) + gamma * comp_gain, 4)
    return {
        "has_fingerprint": False,
        "mask_len": 0,
        "rate_u": 0.0,
        "rate_v": 0.0,
        "union_rate": 0.0,
        "oracle_union_gain": 0.0,
        "unique_u": 0,
        "unique_v": 0,
        "comp_gain": round(comp_gain, 4),
        "projected_score": projected_score,
        "union_mask": [],
    }


class World:
    """
    Frozen replay world over one recorded tree, enforcing prefix-only view.

    Supports Lean-DAG-RSI extensions when `allow_counterfactual=True` or
    `allow_crossover=True` (enabled by default in `LeanDAGPolicy` and `LeanDAGWorld`):
    - Right-censored leaf continuation (`s_cf(v)`) bounded by the Tukey fence `U(T)`
      with geometric decay `gamma^h`.
    - Directed Acyclic Graph hybrid synthesis `CROSSOVER(u,v)` between distinct
      above-baseline branches, ranked by Error Fingerprint complementarity (`pass_u | pass_v`).
    """

    def __init__(
        self,
        tree: DiscoveryTree,
        problem: Optional[Problem] = None,
        allow_counterfactual: bool = False,
        allow_crossover: bool = False,
        gamma: float = 0.65,
        tau_cap: float = 1.5,
        crossover_synergy: float = 0.35,
    ):
        self.tree = tree
        self.problem = problem or Problem()
        self.allow_counterfactual = allow_counterfactual
        self.allow_crossover = allow_crossover
        self.gamma = gamma
        self.tau_cap = tau_cap
        self.crossover_synergy = crossover_synergy
        self.tukey_cap = compute_tukey_cap(tree, self.problem, tau_cap)
        self.max_depth = max((n.attempt for n in tree.nodes.values()), default=1)
        self.reset()

    def reset(self) -> None:
        self.revealed: Dict[str, Node] = {
            self.tree.root_id: self.tree.nodes[self.tree.root_id]
        }
        self.rounds = 0
        self.probes = 0
        self._cf_counter = 0
        self._used_crossovers: Set[tuple] = set()

    # -- observable state ---------------------------------------------------

    def observed(self) -> Dict[str, Node]:
        return dict(self.revealed)

    def best_so_far(self) -> float:
        valid_scores = [self.tree.baseline_score] + [
            self.problem.score(n)
            for nid, n in self.revealed.items()
            if nid != self.tree.root_id and self.problem.is_success(n)
        ]
        return max(valid_scores)

    def opened_branches(self) -> Set[int]:
        return {n.branch for n in self.revealed.values() if n.branch >= 0}

    def legal_roots(self) -> List[str]:
        """Unopened root continuations remaining in the recorded tree (Appendix B.2)."""
        root_kids = [
            c
            for c in self.tree.children_map.get(self.tree.root_id, [])
            if c not in self.revealed
        ]
        return [self.tree.root_id] if root_kids else []

    def unopened_root_count(self) -> int:
        """Number of unopened branches hanging off the root."""
        return len(
            [
                c
                for c in self.tree.children_map.get(self.tree.root_id, [])
                if c not in self.revealed
            ]
        )

    def _all_parents_revealed(self, cid: str) -> bool:
        """
        Enforce strict DAG causality for recorded merge nodes: a recorded child `cid`
        with `secondary_parent_id` or `parent_ids` in `meta` can only be revealed
        once ALL of its recorded parents are in `self.revealed`.
        """
        node = self.tree.nodes.get(cid)
        if node is None:
            return False
        if node.parent_id is not None and node.parent_id not in self.revealed:
            return False
        sec_pid = node.meta.get("secondary_parent_id")
        if isinstance(sec_pid, str) and sec_pid and sec_pid not in self.revealed:
            return False
        extra_pids = node.meta.get("parent_ids")
        if isinstance(extra_pids, (list, tuple)):
            for pid in extra_pids:
                if isinstance(pid, str) and pid and pid not in self.revealed:
                    return False
        return True

    def _last_positive_gain(self, nid: str) -> float:
        node = self.revealed.get(nid)
        if node is None or not self.problem.is_success(node) or node.parent_id is None:
            return 0.0
        parent = self.revealed.get(node.parent_id) or self.tree.nodes.get(
            node.parent_id
        )
        if parent is None:
            return 0.0
        return max(0.0, self.problem.score(node) - self.problem.score(parent))

    def legal_actions(
        self,
        allow_counterfactual: Optional[bool] = None,
        allow_crossover: Optional[bool] = None,
    ) -> List[str]:
        """
        Admissible continuation points:
        - `root_id` when an unopened recorded branch exists,
        - any revealed leaf that still has an unrevealed recorded child,
        - (when `allow_counterfactual=True`) right-censored leaves at `attempt < max_depth`
          with positive last gain,
        - (when `allow_crossover=True`) `CROSSOVER(u,v)` for the most complementary pair
          of distinct above-baseline frontier leaves (using `pass_mask` Error Fingerprints
          when present).
        """
        use_cf = (
            self.allow_counterfactual
            if allow_counterfactual is None
            else allow_counterfactual
        )
        use_xo = self.allow_crossover if allow_crossover is None else allow_crossover

        actions: List[str] = []
        root_kids = [
            c
            for c in self.tree.children_map.get(self.tree.root_id, [])
            if c not in self.revealed
        ]
        if root_kids:
            actions.append(self.tree.root_id)

        branch_frontiers: Dict[int, str] = {}
        for nid, node in self.revealed.items():
            if nid == self.tree.root_id:
                continue
            kids = self.tree.children_map.get(nid, [])
            has_revealed_child = any(c in self.revealed for c in kids) or any(
                other.parent_id == nid for other in self.revealed.values()
            )
            if has_revealed_child:
                continue
            branch_frontiers[node.branch] = nid
            if any(
                c not in self.revealed and self._all_parents_revealed(c) for c in kids
            ):
                actions.append(nid)
            elif use_cf:
                if (
                    self.problem.is_success(node)
                    and node.attempt < self.max_depth
                    and self._last_positive_gain(nid) > 0.0
                ):
                    actions.append(nid)

        if use_xo and len(branch_frontiers) >= 2:
            base = self.tree.baseline_score
            promising = [
                nid
                for nid in branch_frontiers.values()
                if self.problem.is_success(self.revealed[nid])
                and self.problem.score(self.revealed[nid]) > base
            ]
            promising.sort(
                key=lambda x: self.problem.score(self.revealed[x]), reverse=True
            )
            if len(promising) >= 2:
                candidate_pairs: List[Tuple[Tuple[float, float, float], str, str]] = []
                for i in range(len(promising)):
                    for j in range(i + 1, len(promising)):
                        u_id, v_id = promising[i], promising[j]
                        pair = (min(u_id, v_id), max(u_id, v_id))
                        if pair in self._used_crossovers:
                            continue
                        nu, nv = self.revealed[u_id], self.revealed[v_id]
                        su, sv = self.problem.score(nu), self.problem.score(nv)
                        fp = compute_error_fingerprint(
                            nu,
                            nv,
                            base,
                            su,
                            sv,
                            tukey_cap=self.tukey_cap,
                            crossover_synergy=self.crossover_synergy,
                            gamma=self.gamma,
                        )
                        if fp["has_fingerprint"] and fp["oracle_union_gain"] <= 0.0:
                            prio = (-1.0, fp["projected_score"], su + sv)
                        else:
                            prio = (
                                fp["oracle_union_gain"],
                                fp["projected_score"],
                                su + sv,
                            )
                        candidate_pairs.append((prio, u_id, v_id))
                if candidate_pairs:
                    candidate_pairs.sort(key=lambda item: item[0], reverse=True)
                    _, best_u, best_v = candidate_pairs[0]
                    actions.append(f"CROSSOVER({best_u},{best_v})")

        return actions

    # -- transitions --------------------------------------------------------

    def probe(self, parent_ids: Sequence[str]) -> List[Node]:
        """
        CONTINUE(v) or CROSSOVER(u,v) for each selected action.
        Deterministic: reveals the next recorded child (or counterfactual/crossover
        node when enabled). No model, no execution.
        """
        if len(parent_ids) > self.tree.max_parallelism:
            raise ValueError(
                f"batch of {len(parent_ids)} exceeds max_parallelism {self.tree.max_parallelism}"
            )
        out: List[Node] = []
        for pid in parent_ids:
            if pid.startswith("CROSSOVER(") and pid.endswith(")"):
                inner = pid[len("CROSSOVER(") : -1]
                u, v = [x.strip() for x in inner.split(",", 1)]
                if u not in self.revealed or v not in self.revealed:
                    raise ValueError(
                        f"crossover parents {pid!r} outside revealed prefix"
                    )
                node_u, node_v = self.revealed[u], self.revealed[v]
                pair = (min(u, v), max(u, v))
                self._used_crossovers.add(pair)
                self._cf_counter += 1
                cid = f"xover_{self._cf_counter}_{u}_{v}"
                su = self.problem.score(node_u)
                sv = self.problem.score(node_v)
                fp = compute_error_fingerprint(
                    node_u,
                    node_v,
                    self.tree.baseline_score,
                    su,
                    sv,
                    tukey_cap=self.tukey_cap,
                    crossover_synergy=self.crossover_synergy,
                    gamma=self.gamma,
                )
                new_score = fp["projected_score"]
                child = Node(
                    id=cid,
                    parent_id=u,
                    branch=node_u.branch,
                    attempt=max(node_u.attempt, node_v.attempt) + 1,
                    score=new_score,
                    fail_class="ok",
                    pass_mask=fp["union_mask"],
                    meta={
                        "secondary_parent_id": v,
                        "counterfactual": "crossover",
                        "oracle_union_gain": fp["oracle_union_gain"],
                        "union_rate": fp["union_rate"],
                    },
                )
                self.revealed[cid] = child
                self.probes += 1
                out.append(child)
                continue

            if pid not in self.revealed:
                raise ValueError(f"action {pid!r} is outside the revealed prefix")
            pending = [
                c
                for c in self.tree.children_map.get(pid, [])
                if c not in self.revealed and self._all_parents_revealed(c)
            ]
            if pending:
                node = self.tree.nodes[pending[0]]
                self.revealed[node.id] = node
                self.probes += 1
                out.append(node)
            elif self.allow_counterfactual:
                parent_node = self.revealed[pid]
                ds = self._last_positive_gain(pid)
                if ds > 0.0 and parent_node.attempt < self.max_depth:
                    capped_step = min(ds, self.tukey_cap)
                    cf_step_num = int(parent_node.meta.get("cf_step", 0)) + 1
                    gain = (self.gamma**cf_step_num) * capped_step
                    self._cf_counter += 1
                    cid = f"cf_{self._cf_counter}_{pid}"
                    child = Node(
                        id=cid,
                        parent_id=pid,
                        branch=parent_node.branch,
                        attempt=parent_node.attempt + 1,
                        score=round(self.problem.score(parent_node) + gain, 4),
                        fail_class="ok",
                        meta={"cf_step": cf_step_num, "counterfactual": "extrapolated"},
                    )
                    self.revealed[cid] = child
                    self.probes += 1
                    out.append(child)
        if out:
            self.rounds += 1
        return out

    # -- scoring ------------------------------------------------------------

    def reward(self, beta1: float = 0.01, beta2: float = 0.05) -> Dict[str, float]:
        """
        Monotone Work- and Latency-Penalized Replay Objective (Lean-DAG-RSI Fix):
            quality = max({baseline_score} U {score(n) : n in revealed, is_success(n)})
            V = quality - beta1 * N_probes - beta2 * (rounds / max(1, N_probes))
        Wait: to guarantee dV/dN <= -beta1 < 0 strictly for ALL k >= 1 (so adding
        wasted probes in the same round NEVER increases V), we penalize both total
        probes `N_probes` and sequential decision rounds `rounds`:
            V = quality - beta1 * N_probes - beta2 * rounds
        Properties:
        - At fixed rounds `k`, adding a wasted probe changes V by `-beta1 < 0`.
        - At fixed total probes `N`, batching them into fewer rounds `k` increases V
          by `+beta2` per saved round (rewarding true parallelism without reward hacking).
        - Failed attempts (`is_success(n) == False`) never corrupt `quality`, and
          `quality` never drops below `baseline_score`.
        """
        attempts = [n for nid, n in self.revealed.items() if nid != self.tree.root_id]
        valid_scores = [self.tree.baseline_score] + [
            self.problem.score(n) for n in attempts if self.problem.is_success(n)
        ]
        quality = max(valid_scores)
        n_probes = len(attempts)
        k = max(1, self.rounds)
        efficiency = n_probes / k
        return {
            "v": quality - beta1 * n_probes - beta2 * self.rounds,
            "quality": quality,
            "probes": float(n_probes),
            "rounds": float(k),
            "parallelism_efficiency": efficiency,
        }


class LeanDAGWorld(World):
    """Replay world with counterfactual right-censored leaf estimation and DAG crossover enabled."""

    def __init__(
        self,
        tree: DiscoveryTree,
        problem: Optional[Problem] = None,
        gamma: float = 0.65,
        tau_cap: float = 1.5,
        crossover_synergy: float = 0.35,
    ):
        super().__init__(
            tree,
            problem,
            allow_counterfactual=True,
            allow_crossover=True,
            gamma=gamma,
            tau_cap=tau_cap,
            crossover_synergy=crossover_synergy,
        )


def reconstruct_trajectories(world: World) -> Dict[int, BranchTrajectory]:
    """
    Appendix B.2 (lines 1051-1067):
    Reconstruct each opened branch's ordered prefix trajectory from `world.observed()`:
    - `successful_anchor`: best score among successful (`error is None` and `class == 'ok'`)
      evaluations in the branch. A later repairable failure NEVER erases this anchor!
    - `last_gain` and `mean_gain` across valid refinements.
    - `stagnation_steps`: consecutive valid steps where marginal gain was negligible.
    - `hard_failed` and `repairable_failed` status of the current branch frontier.
    """
    problem = world.problem
    observed = world.observed()
    by_branch: Dict[int, List[Node]] = {}
    for nid, node in observed.items():
        if nid == world.tree.root_id or node.branch < 0:
            continue
        by_branch.setdefault(node.branch, []).append(node)

    # Estimate scale from revealed prefix + baseline (strictly prefix-safe)
    revealed_scores = [
        problem.score(n) for n in observed.values() if problem.is_success(n)
    ]
    max_obs = max(revealed_scores, default=world.tree.baseline_score)
    span = max(
        abs(max_obs - world.tree.baseline_score),
        abs(world.tree.baseline_score) * 0.1,
        1e-6,
    )
    small_gain_thresh = 0.03 * span

    trajectories: Dict[int, BranchTrajectory] = {}
    for bid, b_nodes in by_branch.items():
        b_nodes.sort(key=lambda n: n.attempt)
        frontier = b_nodes[-1]
        anchor = world.tree.baseline_score
        prev_valid_score = world.tree.baseline_score
        gains: List[float] = []
        stagnation = 0
        consec_fail = 0

        for n in b_nodes:
            if problem.is_success(n):
                s = problem.score(n)
                gain = s - prev_valid_score
                gains.append(gain)
                if s > anchor:
                    anchor = s
                if gain <= small_gain_thresh:
                    stagnation += 1
                else:
                    stagnation = 0
                prev_valid_score = s
                consec_fail = 0
            else:
                consec_fail += 1

        last_gain = gains[-1] if gains else 0.0
        mean_gain = sum(gains) / len(gains) if gains else 0.0
        hard_failed = problem.hard_failure(frontier) or (
            consec_fail >= 2 and not problem.repairable(frontier)
        )
        repairable_failed = (
            not problem.is_success(frontier)
            and problem.repairable(frontier)
            and not hard_failed
        )

        trajectories[bid] = BranchTrajectory(
            branch_id=bid,
            nodes=b_nodes,
            frontier_id=frontier.id,
            depth=len(b_nodes),
            successful_anchor=anchor,
            latest_score=problem.score(frontier),
            last_gain=last_gain,
            mean_gain=mean_gain,
            stagnation_steps=stagnation,
            hard_failed=hard_failed,
            repairable_failed=repairable_failed,
            consecutive_failures=consec_fail,
        )
    return trajectories


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class Policy:
    """Base exploration policy. Subclass and implement batch() and optionally plan_grid()."""

    #: Fixed for one episode; swept by the caller, never inferred mid-episode.
    beta: float = 0.6

    def __init__(self, beta: float = 0.6):
        self.beta = beta

    def batch(self, world: World) -> List[str]:
        raise NotImplementedError

    def plan_grid(self, context: GridPlanningContext) -> GridPlan:
        """
        Appendix B.2 (lines 1192-1248):
        Deterministic next-cycle grid planner. Runs BEFORE a new live grid is created
        and uses only prefix-safe completed cycle history.
        """
        if not context.history:
            w = max(1, min(context.default_branch_count, context.hard_max_branch_count))
            r = max(0, min(context.default_refine_count, context.hard_max_refine_count))
            return GridPlan(
                branch_count=w,
                refine_count=r,
                reason="Conservative bootstrap grid: insufficient prior live cycle history.",
            )

        last = context.history[-1]
        w = int(last.get("branch_count", context.default_branch_count))
        r = int(last.get("refine_count", context.default_refine_count))
        late_depth_gains = bool(last.get("late_depth_gains", False))
        early_roots_improved = bool(last.get("early_roots_improved", False))
        hard_failure_rate = float(last.get("hard_failure_rate", 0.0))

        if hard_failure_rate > 0.4:
            w = max(1, w - 1)
            r = max(1, r - 1)
            reason = (
                "Reduced width and depth conservatively due to high hard-failure rate."
            )
        elif late_depth_gains and not early_roots_improved:
            r = min(context.hard_max_refine_count, r + 1)
            reason = "High gains arrived late on deep branches; holding width and increasing refine depth."
        elif early_roots_improved and not late_depth_gains:
            w = min(context.hard_max_branch_count, w + 1)
            r = max(1, r - 1)
            reason = "Many distinct roots improved early while deep refinements stalled; increasing width."
        else:
            reason = "Maintaining balanced width/depth grid from prior live cycle."

        return GridPlan(
            branch_count=max(1, min(w, context.hard_max_branch_count)),
            refine_count=max(0, min(r, context.hard_max_refine_count)),
            reason=reason,
        )


class DynamicPortfolio(Policy):
    """
    Exploitation + exploration + at most one recovery slot.

    Parameterised rather than hardcoded, so a candidate policy is just a name
    plus a small dict of numbers. That keeps policy proposals serialisable (no
    arbitrary code persisted in the pool) while still allowing real variation.
    """

    kind = "dynamic_portfolio"

    def __init__(
        self,
        beta: float = 0.6,
        explore_threshold: float = 0.4,
        recovery_threshold: float = 0.3,
        max_recovery: int = 1,
    ):
        super().__init__(beta)
        self.explore_threshold = explore_threshold
        self.recovery_threshold = recovery_threshold
        self.max_recovery = max_recovery

    def params(self) -> Dict[str, Any]:
        return {
            "beta": self.beta,
            "explore_threshold": self.explore_threshold,
            "recovery_threshold": self.recovery_threshold,
            "max_recovery": self.max_recovery,
        }

    def batch(self, world: World) -> List[str]:
        legal = world.legal_actions()
        if not legal:
            return []

        slots = world.tree.max_parallelism
        problem = world.problem
        observed = world.observed()
        trajectories = reconstruct_trajectories(world)
        b = max(0.0, min(1.0, float(self.beta)))
        max_depth_for_beta = max(1, int(1 + round(6.0 * b)))

        root_actions = [a for a in legal if a == world.tree.root_id]
        leaf_actions = [a for a in legal if a != world.tree.root_id]

        exploit: List[tuple] = []
        recover: List[str] = []
        explore: List[str] = list(root_actions)

        for leaf in leaf_actions:
            if leaf.startswith("CROSSOVER("):
                if b >= self.explore_threshold:
                    explore.append(leaf)
                continue
            node = observed[leaf]
            traj = trajectories.get(node.branch)
            if problem.repairable(node) and not problem.is_success(node):
                recover.append(leaf)
            elif (
                problem.is_success(node)
                and problem.score(node) >= world.tree.baseline_score
            ):
                # Gate deep diminishing refinements by beta so beta_sweep is non-degenerate
                if traj is not None and traj.depth > max_depth_for_beta:
                    continue
                exploit.append((problem.score(node), leaf))
            elif self.beta > self.explore_threshold and not problem.hard_failure(node):
                explore.append(leaf)

        exploit.sort(reverse=True)
        chosen: List[str] = []
        reserve = (
            1
            if (
                recover
                and self.beta >= self.recovery_threshold
                and self.max_recovery > 0
            )
            else 0
        )
        for _, leaf in exploit:
            if len(chosen) < slots - reserve:
                chosen.append(leaf)
        for cand in recover[: self.max_recovery]:
            if len(chosen) < slots:
                chosen.append(cand)
        for cand in explore:
            if len(chosen) >= slots:
                break
            if cand not in chosen:
                chosen.append(cand)

        # Parallel root opening on Round 1 when slots and unopened roots remain
        if (
            world.tree.root_id in root_actions
            and not trajectories
            and len(chosen) < slots
        ):
            target_roots = min(
                slots,
                world.unopened_root_count(),
                max(1, int(1 + round((slots - 1) * max(0.3, b)))),
            )
            while len(chosen) < target_roots:
                chosen.append(world.tree.root_id)

        return chosen[:slots]


class OptimalPolicy(Policy):
    """
    Paper-faithful `OptimalPolicy` from Appendix B.2 of Dream-RSI (Zheng et al., 2026):
    1. Reconstructs full per-branch ordered prefix trajectories (`reconstruct_trajectories`)
       so a repairable failure never erases a branch's `successful_anchor`.
    2. Routes all behavioral thresholds through `_schedule(beta) -> dict`:
       - Low beta (0.0..0.3): fewer open branches, stricter marginal-gain & stagnation
         pruning, early portfolio stop (`batch = []`) -> fewer total probes N.
       - High beta (0.7..1.0): wider root opening, deeper patience through plateaus,
         active recovery slot -> higher peak attainment at higher probe cost N.
    3. Ensures the offline `beta_sweep` is non-degenerate whenever branches differ in
       slope or plateau depth.
    """

    kind = "optimal_policy"

    def __init__(
        self,
        beta: float = 0.6,
        slope_weight: float = 0.45,
        max_recovery: int = 1,
    ):
        super().__init__(float(beta))
        self.slope_weight = float(slope_weight)
        self.max_recovery = int(max_recovery)

    def params(self) -> Dict[str, Any]:
        return {
            "beta": self.beta,
            "slope_weight": self.slope_weight,
            "max_recovery": self.max_recovery,
        }

    @staticmethod
    def _schedule(beta: float, span: float = 1.0) -> Dict[str, float]:
        b = max(0.0, min(1.0, float(beta)))
        return {
            "max_open_branches": int(2 + round(4.0 * b)),
            "min_continue_gain": 0.08 * span * (1.0 - 0.85 * b),
            "max_stagnation": 0 if b < 0.25 else (1 if b < 0.65 else 3),
            "max_anchor_gap": (0.06 + 0.35 * b) * span,
            "allow_recovery": 1.0 if b >= 0.20 else 0.0,
            "reserve_root_slot": 1.0 if b >= 0.35 else 0.0,
        }

    def batch(self, world: World) -> List[str]:
        legal = world.legal_actions()
        if not legal:
            return []

        trajectories = reconstruct_trajectories(world)
        best_anchor = max(
            (t.successful_anchor for t in trajectories.values()),
            default=world.tree.baseline_score,
        )
        span = max(
            abs(best_anchor - world.tree.baseline_score),
            abs(world.tree.baseline_score) * 0.1,
            1.0,
        )
        sched = self._schedule(self.beta, span=span)
        slots = world.tree.max_parallelism
        legal_set = set(legal)

        can_open_root = (
            world.tree.root_id in legal_set
            and len(trajectories) < sched["max_open_branches"]
        )

        exploit_queue: List[tuple] = []
        recovery_queue: List[tuple] = []
        underexplored_queue: List[tuple] = []

        for act in legal:
            if act.startswith("CROSSOVER("):
                exploit_queue.append((best_anchor + 0.25 * span, act))

        for _, traj in trajectories.items():
            if traj.frontier_id not in legal_set or traj.hard_failed:
                continue
            if traj.repairable_failed:
                if sched["allow_recovery"] > 0.5 and self.max_recovery > 0:
                    rec_prio = traj.successful_anchor + (0.05 * span) / max(
                        1, traj.depth
                    )
                    recovery_queue.append((rec_prio, traj.frontier_id))
                continue

            anchor_gap = best_anchor - traj.successful_anchor
            if traj.depth >= 2 and traj.stagnation_steps > sched["max_stagnation"]:
                if traj.last_gain < sched["min_continue_gain"]:
                    continue
            if traj.depth >= 2 and anchor_gap > sched["max_anchor_gap"]:
                continue

            prio = (
                traj.successful_anchor
                + self.slope_weight * max(0.0, traj.last_gain)
                - 0.5 * anchor_gap
            )
            if traj.depth <= 1:
                underexplored_queue.append((prio, traj.frontier_id))
            else:
                exploit_queue.append((prio, traj.frontier_id))

        exploit_queue.sort(reverse=True)
        recovery_queue.sort(reverse=True)
        underexplored_queue.sort(reverse=True)

        chosen: List[str] = []
        reserve_rec = 1 if (recovery_queue and slots >= 2) else 0
        reserve_exp = (
            1
            if (
                (can_open_root or underexplored_queue)
                and sched["reserve_root_slot"] > 0.5
                and slots >= 2
            )
            else 0
        )
        max_exploit = max(0, slots - reserve_rec - reserve_exp)

        for _, fid in exploit_queue:
            if len(chosen) < max_exploit and fid not in chosen:
                chosen.append(fid)
        for _, fid in recovery_queue[: self.max_recovery]:
            if len(chosen) < slots - reserve_exp and fid not in chosen:
                chosen.append(fid)
        if can_open_root and len(chosen) < slots:
            if not trajectories:
                # Parallel root opening on Round 1 up to max_open_branches and unopened roots
                n_init = min(
                    slots - len(chosen),
                    world.unopened_root_count(),
                    int(sched["max_open_branches"]),
                )
                chosen.extend([world.tree.root_id] * max(1, n_init))
            else:
                chosen.append(world.tree.root_id)
        for _, fid in underexplored_queue:
            if len(chosen) < slots and fid not in chosen:
                chosen.append(fid)
        for _, fid in exploit_queue + recovery_queue[: self.max_recovery]:
            if len(chosen) >= slots:
                break
            if fid not in chosen:
                chosen.append(fid)

        return chosen[:slots]


class LeanDAGPolicy(OptimalPolicy):
    """
    Upgraded Lean-DAG-RSI Exploration Policy:
    - Enables Tukey-capped counterfactual leaf continuation (`s_cf(v)`) on right-censored
      high-velocity branches so premature cutoffs in prior trees do not blind selection.
    - Enables Directed Acyclic Graph hybrid synthesis `CROSSOVER(u,v)` between the top
      independent above-baseline branches.
    """

    kind = "lean_dag"
    use_counterfactual: bool = True
    use_crossover: bool = True

    def __init__(
        self,
        beta: float = 0.6,
        slope_weight: float = 0.55,
        max_recovery: int = 1,
        use_counterfactual: bool = True,
        use_crossover: bool = True,
    ):
        super().__init__(
            beta=beta, slope_weight=slope_weight, max_recovery=max_recovery
        )
        self.use_counterfactual = bool(use_counterfactual)
        self.use_crossover = bool(use_crossover)

    def params(self) -> Dict[str, Any]:
        return {
            "beta": self.beta,
            "slope_weight": self.slope_weight,
            "max_recovery": self.max_recovery,
            "use_counterfactual": self.use_counterfactual,
            "use_crossover": self.use_crossover,
        }


class DeepestFirst(Policy):
    """Always push the deepest reachable frontier (up to a beta-controlled horizon)."""

    kind = "deepest_first"

    def params(self) -> Dict[str, Any]:
        return {"beta": self.beta}

    def batch(self, world: World) -> List[str]:
        legal = world.legal_actions()
        if not legal:
            return []
        b = max(0.0, min(1.0, float(self.beta)))
        max_depth = max(2, int(2 + round(6.0 * b)))
        depth = _depths(world.tree, world.revealed)
        filtered = [
            n
            for n in legal
            if n == world.tree.root_id
            or n.startswith("CROSSOVER(")
            or (
                not world.problem.hard_failure(world.revealed[n])
                and depth.get(n, 0) <= max_depth
            )
        ]
        if not filtered:
            return []
        ranked = sorted(
            filtered,
            key=lambda n: (
                depth.get(n, 0),
                world.problem.score(world.revealed[n]) if n in world.revealed else 0.0,
            ),
            reverse=True,
        )
        if ranked == [world.tree.root_id] and not world.opened_branches():
            n_init = min(world.tree.max_parallelism, world.unopened_root_count())
            return [world.tree.root_id] * max(1, n_init)
        return ranked[: world.tree.max_parallelism]


class WidestFirst(Policy):
    """Open every root branch in parallel before deepening any of them (up to beta horizon)."""

    kind = "widest_first"

    def params(self) -> Dict[str, Any]:
        return {"beta": self.beta}

    def batch(self, world: World) -> List[str]:
        legal = world.legal_actions()
        if not legal:
            return []
        b = max(0.0, min(1.0, float(self.beta)))
        max_depth = max(1, int(1 + round(6.0 * b)))
        depth = _depths(world.tree, world.revealed)
        slots = world.tree.max_parallelism
        root = world.tree.root_id
        batch: List[str] = []
        if root in legal:
            n_roots = min(slots, max(1, world.unopened_root_count()))
            batch.extend([root] * n_roots)
        for nid in legal:
            if len(batch) >= slots:
                break
            if nid != root and depth.get(nid, 0) <= max_depth:
                if nid in world.revealed and world.problem.hard_failure(
                    world.revealed[nid]
                ):
                    continue
                batch.append(nid)
        return batch


def _depths(tree: DiscoveryTree, revealed: Dict[str, Node]) -> Dict[str, int]:
    depth: Dict[str, int] = {tree.root_id: 0}
    for nid in revealed:
        chain = 0
        cur = nid
        seen = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            node = tree.nodes.get(cur) or revealed.get(cur)
            if node is None or node.parent_id is None:
                break
            chain += 1
            cur = node.parent_id
        depth[nid] = chain
    return depth


#: Registry used to rebuild a policy from persisted parameters.
POLICY_KINDS = {
    DynamicPortfolio.kind: DynamicPortfolio,
    OptimalPolicy.kind: OptimalPolicy,
    LeanDAGPolicy.kind: LeanDAGPolicy,
    DeepestFirst.kind: DeepestFirst,
    WidestFirst.kind: WidestFirst,
}


def build_policy(spec: Dict[str, Any]) -> Policy:
    """Rebuild a policy from a JSON-serialisable spec produced by Policy.spec()."""
    kind = spec.get("kind", DynamicPortfolio.kind)
    cls = POLICY_KINDS.get(kind)
    if cls is None:
        raise ValueError(f"unknown policy kind: {kind!r}")
    # `label` and any other metadata are not constructor arguments.
    params = {k: v for k, v in spec.items() if k not in ("kind", "label", "rationale")}
    return cls(**params)


def policy_spec(policy: Policy, label: Optional[str] = None) -> Dict[str, Any]:
    """A policy as plain JSON. This is what the loop persists, not code."""
    spec: Dict[str, Any] = {"kind": getattr(policy, "kind", "custom")}
    if hasattr(policy, "params"):
        spec.update(policy.params())
    else:
        spec["beta"] = policy.beta
    if label:
        spec["label"] = label
    return spec


def replay(
    tree: DiscoveryTree,
    policy: Policy,
    problem: Optional[Problem] = None,
    beta1: float = 0.01,
    beta2: float = 0.05,
    max_rounds: int = 40,
) -> Dict[str, float]:
    """Replay one policy against one frozen tree. Pure computation."""
    use_cf = bool(getattr(policy, "use_counterfactual", False))
    use_xo = bool(getattr(policy, "use_crossover", False))
    world = World(
        tree,
        problem,
        allow_counterfactual=use_cf,
        allow_crossover=use_xo,
    )
    for _ in range(max_rounds):
        batch = policy.batch(world)
        if not batch:
            break
        world.probe(batch)
    return world.reward(beta1, beta2)


def replay_over_history(
    trees: Sequence[DiscoveryTree],
    policy: Policy,
    problem: Optional[Problem] = None,
    beta1: float = 0.01,
    beta2: float = 0.05,
    max_rounds: int = 40,
) -> Dict[str, Any]:
    """
    Mean reward over every recorded tree, which is what makes policy selection
    monotone: because the incumbent is always in the candidate set, the chosen
    policy can never score worse on the frozen history than the current one.
    """
    if not trees:
        return {"mean_v": 0.0, "per_tree": [], "trees": 0}
    per = [replay(t, policy, problem, beta1, beta2, max_rounds) for t in trees]
    return {
        "mean_v": sum(p["v"] for p in per) / len(per),
        "mean_quality": sum(p["quality"] for p in per) / len(per),
        "mean_probes": sum(p["probes"] for p in per) / len(per),
        "per_tree": per,
        "trees": len(per),
    }


def select_policy(
    candidates: Dict[str, Policy],
    trees: Sequence[DiscoveryTree],
    problem: Optional[Problem] = None,
    beta1: float = 0.01,
    beta2: float = 0.05,
    max_rounds: int = 40,
) -> Dict[str, Any]:
    """
    argmax over candidate policies by mean replay score.

    Include the incumbent under some key to obtain the paper's guarantee: the
    returned policy is never worse than the incumbent on the frozen history.
    """
    scored = {
        name: replay_over_history(trees, pol, problem, beta1, beta2, max_rounds)
        for name, pol in candidates.items()
    }
    best = max(scored, key=lambda k: scored[k]["mean_v"]) if scored else None
    ranked = sorted(scored.items(), key=lambda kv: kv[1]["mean_v"], reverse=True)
    return {
        "best": best,
        "best_mean_v": scored[best]["mean_v"] if best else 0.0,
        "ranking": [(n, round(s["mean_v"], 6)) for n, s in ranked],
        "detail": scored,
    }


def sweep_beta(
    trees: Sequence[DiscoveryTree],
    problem: Optional[Problem] = None,
    policy_cls=DynamicPortfolio,
    values: Optional[Sequence[float]] = None,
    beta1: float = 0.01,
    beta2: float = 0.05,
) -> List[tuple]:
    """Offline beta sweep. This is the 'dreaming' step in its simplest form."""
    vals = list(values) if values is not None else [i / 10 for i in range(11)]
    out = []
    for v in vals:
        res = replay_over_history(trees, policy_cls(beta=v), problem, beta1, beta2)
        out.append((round(v, 3), round(res["mean_v"], 6)))
    return sorted(out, key=lambda kv: kv[1], reverse=True)


def pareto_sweep(
    trees: Sequence[DiscoveryTree],
    problem: Optional[Problem] = None,
    policy_cls=OptimalPolicy,
    values: Optional[Sequence[float]] = None,
    lam: float = 0.5,
    beta1: float = 0.01,
    beta2: float = 0.05,
) -> Dict[str, Any]:
    """
    Appendix B.2 Pareto sweep evaluation:
        pareto.reward = pareto.auc - lambda * parallel_penalty
    where:
    - `pareto.auc` measures mean quality-minus-work across the beta grid, and
    - `parallel_penalty` is the mean of `rounds / max(1, probes)` (1.0 for serial
      probes, approaching `1 / W` for full parallel batches).
    """
    vals = list(values) if values is not None else [i / 10 for i in range(11)]
    prob = problem or Problem()
    curve: List[Dict[str, float]] = []

    for b in vals:
        res = replay_over_history(trees, policy_cls(beta=b), prob, beta1, beta2)
        par_pens = (
            [pt["rounds"] / max(1.0, pt["probes"]) for pt in res["per_tree"]]
            if res["per_tree"]
            else [1.0]
        )
        mean_par_pen = sum(par_pens) / len(par_pens)
        curve.append(
            {
                "beta": round(b, 3),
                "mean_v": round(res["mean_v"], 6),
                "mean_quality": round(res["mean_quality"], 6),
                "mean_probes": round(res["mean_probes"], 4),
                "parallel_penalty": round(mean_par_pen, 6),
            }
        )

    auc = sum(p["mean_quality"] - beta1 * p["mean_probes"] for p in curve) / max(
        1, len(curve)
    )
    mean_par_penalty = sum(p["parallel_penalty"] for p in curve) / max(1, len(curve))
    reward = auc - lam * mean_par_penalty
    return {
        "pareto_reward": round(reward, 6),
        "pareto_auc": round(auc, 6),
        "parallel_penalty": round(mean_par_penalty, 6),
        "curve": curve,
    }


def evaluate_flat_plateau(
    eval_fn: Callable[[Sequence[float]], float],
    theta_opt: Sequence[float],
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    rel_jitter: float = 0.05,
    n_directions: int = 20,
    max_drop_tol: Optional[float] = None,
    max_drop_ratio: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Neuro-Symbolic Anti-Spike Flatness / Jitter Test:
    Perturbs all parameters `theta_opt` by +/- `rel_jitter` (default +/- 5%)
    across `n_directions` (default 20) deterministic seeded directions in R^d.
    Detects sharp needle overfitting ('игольное ушко') where a tiny parameter shift
    causes a cliff drop in performance.
    """
    base_vec = [float(x) for x in theta_opt]
    d = len(base_vec)
    peak_score = float(eval_fn(base_vec))
    if d == 0 or n_directions <= 0:
        return {
            "peak_score": round(peak_score, 6),
            "plateau_mean": round(peak_score, 6),
            "plateau_std": 0.0,
            "plateau_min": round(peak_score, 6),
            "cliff_drop": 0.0,
            "worst_drop": 0.0,
            "is_sharp_spike": False,
            "n_directions": 0,
            "rel_jitter": rel_jitter,
            "perturbed_scores": [],
        }

    radii: List[float] = []
    for j in range(d):
        if abs(base_vec[j]) > 1e-9:
            r_j = rel_jitter * abs(base_vec[j])
        elif bounds is not None and j < len(bounds):
            r_j = rel_jitter * abs(float(bounds[j][1]) - float(bounds[j][0]))
        else:
            r_j = rel_jitter
        radii.append(r_j)

    rng = random.Random(seed)
    perturbed_scores: List[float] = []
    for k in range(n_directions):
        if k < min(2 * d, n_directions // 2):
            # Deterministic axis-coupled +/- 100% corner/diagonal shifts
            axis = k // 2
            sign = 1.0 if (k % 2 == 0) else -1.0
            raw_dir = [
                sign
                if j == axis
                else (0.5 * sign if ((j + k) % 2 == 0) else -0.5 * sign)
                for j in range(d)
            ]
        else:
            raw_dir = [rng.uniform(-1.0, 1.0) for _ in range(d)]
            max_abs = max((abs(u) for u in raw_dir), default=1.0)
            if max_abs > 1e-9:
                raw_dir = [u / max_abs for u in raw_dir]

        trial_vec: List[float] = []
        for j in range(d):
            val = base_vec[j] + radii[j] * raw_dir[j]
            if bounds is not None and j < len(bounds):
                lo_j, hi_j = float(bounds[j][0]), float(bounds[j][1])
                val = max(lo_j, min(hi_j, val))
            trial_vec.append(val)
        perturbed_scores.append(float(eval_fn(trial_vec)))

    plateau_mean = sum(perturbed_scores) / len(perturbed_scores)
    plateau_min = min(perturbed_scores)
    variance = sum((s - plateau_mean) ** 2 for s in perturbed_scores) / len(
        perturbed_scores
    )
    plateau_std = math.sqrt(variance)
    cliff_drop = peak_score - plateau_mean
    worst_drop = peak_score - plateau_min
    tol = (
        float(max_drop_tol)
        if max_drop_tol is not None
        else max_drop_ratio * max(abs(peak_score), 1.0)
    )
    is_sharp_spike = bool(cliff_drop > tol)
    return {
        "peak_score": round(peak_score, 6),
        "plateau_mean": round(plateau_mean, 6),
        "plateau_std": round(plateau_std, 6),
        "plateau_min": round(plateau_min, 6),
        "cliff_drop": round(cliff_drop, 6),
        "worst_drop": round(worst_drop, 6),
        "is_sharp_spike": is_sharp_spike,
        "n_directions": n_directions,
        "rel_jitter": rel_jitter,
        "perturbed_scores": [round(s, 6) for s in perturbed_scores],
    }


def prune_tunable_params(
    eval_fn: Callable[[Sequence[float]], float],
    theta_opt: Sequence[float],
    theta_default: Sequence[float],
    param_names: Optional[Sequence[str]] = None,
    delta_tol: float = 1e-3,
) -> Dict[str, Any]:
    """
    RRSI Complexity Pruner (Секатор параметров из статьи RRSI):
    Freezes each tuned parameter in `theta_opt` back to `theta_default[i]` one by one.
    Greedily eliminates parameters whose removal reduces the objective by at most
    `delta_tol` (or improves it), locking them permanently to their default values.
    """
    current = [float(x) for x in theta_opt]
    defaults = [float(x) for x in theta_default]
    d = len(current)
    if len(defaults) != d:
        raise ValueError(
            f"theta_opt length ({d}) != theta_default length ({len(defaults)})"
        )
    names = (
        list(param_names)
        if param_names is not None and len(param_names) == d
        else [f"param_{i}" for i in range(d)]
    )
    full_score = float(eval_fn(current))

    ablation_table: List[Dict[str, Any]] = []
    active_indices: List[int] = []
    for i in range(d):
        if abs(current[i] - defaults[i]) <= 1e-12:
            ablation_table.append(
                {
                    "index": i,
                    "name": names[i],
                    "opt_val": current[i],
                    "default_val": defaults[i],
                    "score_when_frozen": round(full_score, 6),
                    "drop": 0.0,
                    "status": "pruned_already_default",
                }
            )
        else:
            active_indices.append(i)
            trial = list(current)
            trial[i] = defaults[i]
            s_frozen = float(eval_fn(trial))
            drop = full_score - s_frozen
            ablation_table.append(
                {
                    "index": i,
                    "name": names[i],
                    "opt_val": current[i],
                    "default_val": defaults[i],
                    "score_when_frozen": round(s_frozen, 6),
                    "drop": round(drop, 6),
                    "status": "candidate",
                }
            )

    # Greedy backward elimination of low-impact parameters
    pruned_indices: Set[int] = {i for i in range(d) if i not in active_indices}
    while active_indices:
        best_idx = None
        best_trial_score = -float("inf")
        best_drop = float("inf")
        for idx in active_indices:
            trial = list(current)
            trial[idx] = defaults[idx]
            s_try = float(eval_fn(trial))
            drop_from_full = full_score - s_try
            if drop_from_full < best_drop:
                best_drop = drop_from_full
                best_trial_score = s_try
                best_idx = idx
        if best_idx is not None and best_drop <= delta_tol:
            current[best_idx] = defaults[best_idx]
            active_indices.remove(best_idx)
            pruned_indices.add(best_idx)
            if best_trial_score > full_score:
                full_score = best_trial_score
        else:
            break

    for row in ablation_table:
        idx = int(row["index"])
        if idx in pruned_indices and row["status"] == "candidate":
            row["status"] = "pruned_redundant"
        elif idx not in pruned_indices:
            row["status"] = "kept_essential"

    pruned_score = float(eval_fn(current))
    kept_names = [names[i] for i in range(d) if i not in pruned_indices]
    pruned_names = [names[i] for i in range(d) if i in pruned_indices]
    return {
        "theta_pruned": current,
        "full_score": round(full_score, 6),
        "pruned_score": round(pruned_score, 6),
        "active_count": len(kept_names),
        "pruned_count": len(pruned_names),
        "kept_params": kept_names,
        "pruned_params": pruned_names,
        "ablation_table": ablation_table,
    }


def profile_callable_lines(
    fn: Callable[..., Any],
    *args: Any,
    top_k: int = 3,
    max_line_events: int = 50000,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Automated Bottleneck Autopsy Line Profiler:
    Executes `fn(*args, **kwargs)` under a lightweight `sys.settrace` line timer
    (`time.perf_counter_ns`) scoped to the candidate's source file, returning the
    function result along with the `top_k` source lines consuming the most CPU time.
    Automatically detaches `sys.settrace` after `max_line_events` (default 50,000)
    events so pure-Python inner loops never suffer unbounded tracing overhead.
    """
    code_obj = getattr(fn, "__code__", None)
    target_file = getattr(code_obj, "co_filename", None)
    frame_last: Dict[int, Tuple[str, int, int]] = {}
    line_stats: Dict[Tuple[str, int], List[int]] = {}
    event_counter = [0]
    capped = [False]

    def _trace(frame: Any, event: str, arg: Any) -> Any:
        now = time.perf_counter_ns()
        fid = id(frame)
        if fid in frame_last:
            prev_file, prev_line, prev_t = frame_last[fid]
            dt = max(0, now - prev_t)
            key = (prev_file, prev_line)
            if key not in line_stats:
                line_stats[key] = [dt, 1]
            else:
                line_stats[key][0] += dt
                line_stats[key][1] += 1

        event_counter[0] += 1
        if max_line_events > 0 and event_counter[0] >= max_line_events:
            capped[0] = True
            frame_last.clear()
            sys.settrace(old_trace)
            return None

        f_code = frame.f_code
        f_file = f_code.co_filename
        if event == "call":
            if target_file is not None and f_file != target_file:
                return None
            if f_code.co_name == "profile_callable_lines":
                return None
            frame_last[fid] = (f_file, frame.f_lineno, time.perf_counter_ns())
            return _trace
        elif event == "line":
            frame_last[fid] = (f_file, frame.f_lineno, time.perf_counter_ns())
            return _trace
        elif event in ("return", "exception"):
            frame_last.pop(fid, None)
            return _trace
        return _trace

    old_trace = sys.gettrace()
    t0 = time.perf_counter_ns()
    try:
        sys.settrace(_trace)
        result = fn(*args, **kwargs)
    finally:
        sys.settrace(old_trace)
    wall_ns = max(1, time.perf_counter_ns() - t0)

    total_line_ns = max(1, sum(v[0] for v in line_stats.values()))
    ranked = sorted(line_stats.items(), key=lambda kv: kv[1][0], reverse=True)[
        : max(1, top_k)
    ]
    hotspots: List[Dict[str, Any]] = []
    for (fname, lineno), (ns, hits) in ranked:
        src_line = linecache.getline(fname, lineno).strip()
        hotspots.append(
            {
                "line_no": int(lineno),
                "code": src_line,
                "pct_time": round(100.0 * ns / total_line_ns, 2),
                "time_ms": round(ns / 1e6, 4),
                "hits": int(hits),
            }
        )
    return {
        "result": result,
        "total_time_ms": round(wall_ns / 1e6, 4),
        "line_events_sampled": int(event_counter[0]),
        "trace_capped": bool(capped[0]),
        "line_hotspots": hotspots,
    }


def ideate_over_history(
    trees: Sequence[DiscoveryTree],
    problem: Optional[Problem] = None,
) -> Dict[str, Any]:
    """
    Lean-DAG-RSI Ideation Synthesizer:
    Analyzes all recorded discovery trees and produces concrete, actionable
    mechanisms for generating NEW ideas in the next online round:
      1. `crossovers`: pairs of distinct above-baseline branches (`CROSSOVER(u,v)`)
         ranked by Error-Fingerprint union coverage (`len(pass_u | pass_v)`) when
         `pass_mask` is recorded, avoiding blind crossover of identical-outcome clones.
      2. `censored_breakthroughs`: fast-rising branches (`last_gain > 0`) that were
         cut off early (`depth < max_depth`) and should be deepened first (`s_cf(v)`).
      3. `repairable_anchors`: branches with a strong `successful_anchor` whose
         latest attempt failed with a repairable error (`syntax_error`, `oom`, etc.).
      4. `neuro_symbolic_splits`: branches experiencing diminishing tweaks (`stagnation >= 1`)
         where continuous parameters/thresholds should be handed off to a numerical
         optimizer with Flat-Plateau Jitter (`evaluate_flat_plateau`) and RRSI Parameter
         Pruning (`prune_tunable_params`).
      5. `bottleneck_diagnostics`: automated forensic pinpointing from `failure_breakdown`
         (dominant error tag) and `line_hotspots` (top CPU hotspot source line).
      6. `dead_ends`: confirmed `algorithmic_failure` branches to never repeat.
    """
    prob = problem or Problem()
    crossovers: List[Dict[str, Any]] = []
    censored_breakthroughs: List[Dict[str, Any]] = []
    repairable_anchors: List[Dict[str, Any]] = []
    neuro_symbolic_splits: List[Dict[str, Any]] = []
    bottleneck_diagnostics: List[Dict[str, Any]] = []
    dead_ends: List[Dict[str, Any]] = []

    for idx, tree in enumerate(trees):
        tukey_cap = compute_tukey_cap(tree, prob)
        w_full = World(tree, prob, allow_counterfactual=True, allow_crossover=True)
        # Reveal all recorded nodes to inspect the full realized tree structure
        for nid, node in tree.nodes.items():
            w_full.revealed[nid] = node
        trajs = reconstruct_trajectories(w_full)
        max_d = max((t.depth for t in trajs.values()), default=1)

        promising_branches: List[BranchTrajectory] = []
        for bid, tr in sorted(trajs.items()):
            if tr.hard_failed:
                dead_ends.append(
                    {
                        "tree_index": idx,
                        "branch": bid,
                        "frontier_id": tr.frontier_id,
                        "fail_class": prob.class_of(tree.nodes[tr.frontier_id]),
                        "score": tr.latest_score,
                    }
                )
                continue
            if tr.repairable_failed and tr.successful_anchor >= tree.baseline_score:
                frontier_node = tree.nodes[tr.frontier_id]
                repairable_anchors.append(
                    {
                        "tree_index": idx,
                        "branch": bid,
                        "frontier_id": tr.frontier_id,
                        "successful_anchor": tr.successful_anchor,
                        "fail_class": prob.class_of(frontier_node),
                        "error": frontier_node.error,
                    }
                )
            if tr.successful_anchor > tree.baseline_score:
                promising_branches.append(tr)
                if tr.depth < max_d and tr.last_gain > 0:
                    est_cf = round(
                        tr.successful_anchor + 0.65 * min(tr.last_gain, tukey_cap), 4
                    )
                    censored_breakthroughs.append(
                        {
                            "tree_index": idx,
                            "branch": bid,
                            "frontier_id": tr.frontier_id,
                            "depth": tr.depth,
                            "max_tree_depth": max_d,
                            "current_score": tr.successful_anchor,
                            "last_gain": round(tr.last_gain, 4),
                            "projected_s_cf": est_cf,
                        }
                    )
                if tr.stagnation_steps >= 1 and tr.depth >= 2:
                    neuro_symbolic_splits.append(
                        {
                            "tree_index": idx,
                            "branch": bid,
                            "frontier_id": tr.frontier_id,
                            "anchor_score": tr.successful_anchor,
                            "last_gain": round(tr.last_gain, 4),
                            "recommendation": (
                                f"Branch {bid} ({tr.frontier_id}) plateaued after {tr.depth} steps "
                                f"(last_gain={tr.last_gain:.3f}). Freeze discrete structure, optimize "
                                "continuous parameters via L-BFGS-B / coordinate search, and validate "
                                "with evaluate_flat_plateau() + prune_tunable_params()."
                            ),
                        }
                    )

        # Automated Bottleneck Autopsy over recorded nodes (`failure_breakdown` & `line_hotspots`)
        valid_tree_nodes = [
            n
            for nid, n in tree.nodes.items()
            if nid != tree.root_id and prob.is_success(n)
        ]
        valid_tree_nodes.sort(key=lambda n: prob.score(n), reverse=True)
        for n in valid_tree_nodes:
            fb = (
                n.failure_breakdown
                if n.failure_breakdown
                else n.meta.get("failure_breakdown", {})
            )
            lh = n.line_hotspots if n.line_hotspots else n.meta.get("line_hotspots", [])
            if not fb and not lh:
                continue
            top_tag = None
            top_tag_val = 0.0
            top_tag_share_pct = 0.0
            if isinstance(fb, dict) and fb:
                total_fb = sum(max(0.0, float(v)) for v in fb.values())
                sorted_fb = sorted(
                    fb.items(), key=lambda kv: float(kv[1]), reverse=True
                )
                top_tag = str(sorted_fb[0][0])
                top_tag_val = float(sorted_fb[0][1])
                top_tag_share_pct = (
                    round(100.0 * top_tag_val / total_fb, 2) if total_fb > 0 else 0.0
                )

            top_hotspot = None
            if isinstance(lh, list) and lh:
                sorted_lh = sorted(
                    lh, key=lambda item: float(item.get("pct_time", 0.0)), reverse=True
                )
                top_hotspot = sorted_lh[0]

            parts: List[str] = []
            if top_hotspot is not None:
                parts.append(
                    f"{float(top_hotspot.get('pct_time', 0.0)):.1f}% of CPU time is in line "
                    f"{top_hotspot.get('line_no')} (`{top_hotspot.get('code', '')}`)"
                )
            if top_tag is not None:
                parts.append(
                    f"{top_tag_share_pct:.1f}% of failures occur in tag `{top_tag}` ({top_tag_val:g})"
                )
            bottleneck_diagnostics.append(
                {
                    "tree_index": idx,
                    "node_id": n.id,
                    "branch": n.branch,
                    "score": prob.score(n),
                    "top_failure_tag": top_tag,
                    "top_failure_value": top_tag_val,
                    "top_failure_share_pct": top_tag_share_pct,
                    "top_hotspot": top_hotspot,
                    "failure_breakdown": dict(fb) if isinstance(fb, dict) else {},
                    "line_hotspots": list(lh) if isinstance(lh, list) else [],
                    "autopsy_prompt": f"Node {n.id} (score={prob.score(n):.4f}): "
                    + " and ".join(parts)
                    + ".",
                }
            )

        promising_branches.sort(key=lambda t: t.successful_anchor, reverse=True)
        if len(promising_branches) >= 2:
            # Evaluate all pairs (u_tr, v_tr) using Error Fingerprint union coverage
            pair_candidates: List[
                Tuple[Tuple[float, float, float], Dict[str, Any]]
            ] = []
            for i in range(len(promising_branches)):
                for j in range(i + 1, len(promising_branches)):
                    u_tr, v_tr = promising_branches[i], promising_branches[j]
                    u_node = max(
                        (n for n in u_tr.nodes if prob.is_success(n)),
                        key=lambda n: prob.score(n),
                        default=tree.nodes[u_tr.frontier_id],
                    )
                    v_node = max(
                        (n for n in v_tr.nodes if prob.is_success(n)),
                        key=lambda n: prob.score(n),
                        default=tree.nodes[v_tr.frontier_id],
                    )
                    fp = compute_error_fingerprint(
                        u_node,
                        v_node,
                        tree.baseline_score,
                        u_tr.successful_anchor,
                        v_tr.successful_anchor,
                        tukey_cap=tukey_cap,
                    )
                    if fp["has_fingerprint"] and fp["oracle_union_gain"] <= 0.0:
                        prio = (
                            -1.0,
                            fp["projected_score"],
                            u_tr.successful_anchor + v_tr.successful_anchor,
                        )
                    else:
                        prio = (
                            fp["oracle_union_gain"],
                            fp["projected_score"],
                            u_tr.successful_anchor + v_tr.successful_anchor,
                        )
                    if fp["has_fingerprint"]:
                        synth_prompt = (
                            f"Error-Fingerprint Crossover: combine {u_tr.frontier_id} "
                            f"(branch {u_tr.branch_id}, score={u_tr.successful_anchor}, "
                            f"pass_rate={fp['rate_u']:.2%}, unique_wins={fp['unique_u']}) with "
                            f"{v_tr.frontier_id} (branch {v_tr.branch_id}, score={v_tr.successful_anchor}, "
                            f"pass_rate={fp['rate_v']:.2%}, unique_wins={fp['unique_v']}) -> "
                            f"oracle union={fp['union_rate']:.2%} (+{fp['oracle_union_gain']:.2%} gain)."
                        )
                    else:
                        synth_prompt = (
                            f"Combine the mechanism of {u_tr.frontier_id} (branch {u_tr.branch_id}, "
                            f"score={u_tr.successful_anchor}) with the complementary mechanism of "
                            f"{v_tr.frontier_id} (branch {v_tr.branch_id}, score={v_tr.successful_anchor})."
                        )
                    entry = {
                        "tree_index": idx,
                        "action": f"CROSSOVER({u_tr.frontier_id},{v_tr.frontier_id})",
                        "parent_u": u_tr.frontier_id,
                        "parent_v": v_tr.frontier_id,
                        "score_u": u_tr.successful_anchor,
                        "score_v": v_tr.successful_anchor,
                        "has_error_fingerprint": fp["has_fingerprint"],
                        "oracle_union_rate": fp["union_rate"],
                        "oracle_union_gain": fp["oracle_union_gain"],
                        "unique_wins_u": fp["unique_u"],
                        "unique_wins_v": fp["unique_v"],
                        "projected_hybrid_score": fp["projected_score"],
                        "synthesis_prompt": synth_prompt,
                    }
                    pair_candidates.append((prio, entry))

            pair_candidates.sort(key=lambda item: item[0], reverse=True)
            if pair_candidates:
                crossovers.append(pair_candidates[0][1])

    harness_diagnostics = diagnose_harness_failures(trees, prob)
    return {
        "crossovers": crossovers,
        "censored_breakthroughs": censored_breakthroughs,
        "repairable_anchors": repairable_anchors,
        "neuro_symbolic_splits": neuro_symbolic_splits,
        "bottleneck_diagnostics": bottleneck_diagnostics,
        "harness_diagnostics": harness_diagnostics,
        "dead_ends": dead_ends,
    }


# ---------------------------------------------------------------------------
# DiscoveryRSI (arXiv:2610.23839v1) Upgrades:
#   1. Pre-Commitment Hypothesis Contract H = (c, pi, d, a) (Def. 1, Eq. 1)
#   2. Harness Replay Failure Diagnostics (Sec. 3.4, App. A.4)
#   3. Provenance Credit + Anti-Clone Family Caps + Masked SFT (App. B & C)
#   4. 2x2 Joint Model x Harness Acceptance with Novelty Guard (Eq. 6, Alg. 1)
# ---------------------------------------------------------------------------


def validate_hypothesis_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    """
    Operational Hypothesis Pre-Commitment Gate H = (c, pi, d, a)
    (DiscoveryRSI Definition 1, Eq. 1, p. 5 & Figure 3, p. 7):
    Verifies that a candidate hypothesis declares all four required components
    BEFORE compiling or executing code on data:
      - `claim` (c): evidence-grounded claim specifying population/system and metric.
      - `plan` (pi): investigation plan specifying data splits, baseline comparisons, and controls.
      - `decision_criteria` (d): pre-committed rules for `support`, `weaken`, and `inconclusive` (`unclear`).
      - `alternative_explanations` (a): list/dict of confounders, biases, or artifacts + control checks.
    """
    if not isinstance(contract, dict):
        return {
            "valid": False,
            "missing_fields": [
                "claim",
                "plan",
                "decision_criteria",
                "alternative_explanations",
            ],
            "confounder_count": 0,
            "has_tri_state_criteria": False,
            "summary": "Invalid contract type (expected dict).",
        }

    missing: List[str] = []
    claim = str(contract.get("claim", "")).strip()
    if len(claim) < 5:
        missing.append("claim")

    plan = contract.get("plan")
    if isinstance(plan, dict):
        if not plan:
            missing.append("plan")
    elif isinstance(plan, str):
        if len(plan.strip()) < 5:
            missing.append("plan")
    else:
        missing.append("plan")

    criteria = contract.get("decision_criteria")
    has_tri_state = False
    if isinstance(criteria, dict):
        keys_lower = {str(k).lower() for k in criteria.keys()}
        has_support = "support" in keys_lower
        has_weaken = "weaken" in keys_lower
        has_unclear = ("inconclusive" in keys_lower) or ("unclear" in keys_lower)
        has_tri_state = bool(
            has_support
            and has_weaken
            and has_unclear
            and all(str(v).strip() for v in criteria.values())
        )
        if not has_tri_state:
            missing.append("decision_criteria(support/weaken/inconclusive)")
    else:
        missing.append("decision_criteria")

    alt = contract.get("alternative_explanations")
    confounder_count = 0
    if isinstance(alt, (list, tuple)):
        valid_alts = [x for x in alt if str(x).strip()]
        confounder_count = len(valid_alts)
    elif isinstance(alt, dict):
        valid_alts = [k for k, v in alt.items() if str(k).strip() and str(v).strip()]
        confounder_count = len(valid_alts)
    if confounder_count == 0:
        missing.append("alternative_explanations")

    is_valid = len(missing) == 0
    return {
        "valid": is_valid,
        "missing_fields": missing,
        "confounder_count": confounder_count,
        "has_tri_state_criteria": has_tri_state,
        "summary": (
            f"PASS: H=(c,pi,d,a) pre-committed with {confounder_count} confounder control(s)."
            if is_valid
            else f"REJECTED: missing or incomplete {missing}."
        ),
    }


def diagnose_harness_failures(
    trees: Sequence[DiscoveryTree],
    problem: Optional[Problem] = None,
) -> Dict[str, Any]:
    """
    Harness Replay Diagnostic Engine (DiscoveryRSI Sec. 3.4 & App. A.4, pp. 9, 25):
    Joins decision, contribution, and execution records across `trees` to locate
    four recurring workflow/harness breakdowns:
      1. `repeated_exploration`: distinct nodes/branches with 100% overlapping
         `pass_mask` (`oracle_union_gain == 0`) or identical score + tunable parameters.
      2. `failed_tool_use`: repairable execution failures (`timeout`, `syntax_error`, `oom`)
         that were never followed up by a repair child node.
      3. `lost_artifacts`: above-baseline successful nodes (`score > baseline_score`)
         whose `artifact` and `tunable_params` are both empty (unreproducible state).
      4. `premature_closure`: branches abandoned at depth 1 despite either positive gain
         (`last_gain > 0`) or a single repairable failure while the tree grew deeper.
    """
    prob = problem or Problem()
    repeated_exploration: List[Dict[str, Any]] = []
    failed_tool_use: List[Dict[str, Any]] = []
    lost_artifacts: List[Dict[str, Any]] = []
    premature_closure: List[Dict[str, Any]] = []

    for idx, tree in enumerate(trees):
        children_map = tree.children_map
        w_full = World(tree, prob, allow_counterfactual=True, allow_crossover=True)
        for nid, node in tree.nodes.items():
            w_full.revealed[nid] = node
        trajs = reconstruct_trajectories(w_full)
        max_d = max((t.depth for t in trajs.values()), default=1)

        # 1. Check nodes for failed_tool_use and lost_artifacts
        success_nodes: List[Node] = []
        for nid, node in tree.nodes.items():
            if nid == tree.root_id:
                continue
            ch_ids = children_map.get(nid, [])
            if not prob.is_success(node) and prob.repairable(node) and len(ch_ids) == 0:
                failed_tool_use.append(
                    {
                        "tree_index": idx,
                        "node_id": nid,
                        "branch": node.branch,
                        "fail_class": prob.class_of(node),
                        "error": node.error,
                        "remedy": f"Queue a Repair slot on `{nid}` ({prob.class_of(node)}).",
                    }
                )
            if prob.is_success(node) and prob.score(node) > tree.baseline_score:
                success_nodes.append(node)
                t_params = getattr(node, "tunable_params", None) or node.meta.get(
                    "tunable_params"
                )
                has_art = (
                    bool(node.artifact)
                    or bool(t_params)
                    or bool(
                        node.meta.get("artifact_hash") or node.meta.get("checkpoint")
                    )
                )
                if not has_art:
                    lost_artifacts.append(
                        {
                            "tree_index": idx,
                            "node_id": nid,
                            "branch": node.branch,
                            "score": prob.score(node),
                            "remedy": (
                                f"Node `{nid}` beat baseline ({prob.score(node)} > {tree.baseline_score}) "
                                "without persisting `artifact` or `tunable_params`."
                            ),
                        }
                    )

        # 2. Check pairwise repeated_exploration among successful nodes
        for i in range(len(success_nodes)):
            for j in range(i + 1, len(success_nodes)):
                u, v = success_nodes[i], success_nodes[j]
                if u.branch == v.branch:
                    continue
                fp = compute_error_fingerprint(
                    u,
                    v,
                    tree.baseline_score,
                    prob.score(u),
                    prob.score(v),
                )
                is_clone_fp = bool(
                    fp["has_fingerprint"] and fp["oracle_union_gain"] <= 1e-9
                )
                u_tp = getattr(u, "tunable_params", None) or u.meta.get(
                    "tunable_params"
                )
                v_tp = getattr(v, "tunable_params", None) or v.meta.get(
                    "tunable_params"
                )
                is_clone_params = bool(
                    not fp["has_fingerprint"]
                    and abs(prob.score(u) - prob.score(v)) <= 1e-9
                    and u_tp
                    and u_tp == v_tp
                )
                if is_clone_fp or is_clone_params:
                    repeated_exploration.append(
                        {
                            "tree_index": idx,
                            "node_u": u.id,
                            "node_v": v.id,
                            "branch_u": u.branch,
                            "branch_v": v.branch,
                            "reason": (
                                "zero_oracle_union_gain"
                                if is_clone_fp
                                else "identical_score_and_params"
                            ),
                            "remedy": (
                                f"Branches {u.branch} (`{u.id}`) and {v.branch} (`{v.id}`) "
                                "duplicate the same outcome slice; suppress clone credit via family cap."
                            ),
                        }
                    )

        # 3. Check trajectories for premature_closure
        for bid, tr in sorted(trajs.items()):
            if tr.depth == 1 and max_d >= 2 and not tr.hard_failed:
                if tr.last_gain > 0 or tr.repairable_failed:
                    premature_closure.append(
                        {
                            "tree_index": idx,
                            "branch": bid,
                            "frontier_id": tr.frontier_id,
                            "depth": tr.depth,
                            "max_tree_depth": max_d,
                            "last_gain": round(tr.last_gain, 4),
                            "repairable_failed": tr.repairable_failed,
                            "remedy": (
                                f"Branch {bid} (`{tr.frontier_id}`) closed at depth 1 "
                                f"(max_depth={max_d}); resume continuation or recovery."
                            ),
                        }
                    )

    total_issues = (
        len(repeated_exploration)
        + len(failed_tool_use)
        + len(lost_artifacts)
        + len(premature_closure)
    )
    return {
        "total_issues": total_issues,
        "repeated_exploration": repeated_exploration,
        "failed_tool_use": failed_tool_use,
        "lost_artifacts": lost_artifacts,
        "premature_closure": premature_closure,
    }


def compute_provenance_credit(
    hypotheses: Sequence[Dict[str, Any]],
    kappa: float = 2.0,
    nu: float = 2.0,
    sigma_ref: float = 8.3333,
    q_min: float = 0.5,
    q_max: float = 2.0,
    lambda_breakthrough: float = 0.35,
    lambda_repair: float = 0.10,
    equal_repair_bonus: bool = False,
    lambda_p_legacy: float = 0.25,
    gamma_coord: float = 0.2,
    c_loc: float = 2.0,
    c_glob: float = 2.0,
    w_max: float = 2.0,
) -> Dict[str, Any]:
    """
    DiscoveryRSI Hypothesis Credit Optimization & Anti-Clone Family Cap
    (Appendix B, Equations 3, 14-23 & Proposition 1, pp. 8, 27-30):
      1. Conservative rating (Eq. 14): `s_h = mu_h - kappa * sigma_h`
      2. Tie-aware within-stratum rank (Eq. 15): `r_h` in [0, 1]
      3. Coverage & uncertainty damping (Eq. 16):
         `rho_cov = n_h / (n_h + nu)`, `rho_unc = 1 / (1 + sigma_h / sigma_ref)`
      4. Bounded hypothesis weight (Eq. 17 with fix for the 'Cautious A-Student' bug):
         In DiscoveryRSI Eq. 17, `necessary-repair` and `substantive-progress` received
         the exact same bonus `lambda_p = 0.25`, causing `Repair` to crowd out `Crossover`
         and `Novelty`. By default (`equal_repair_bonus=False`), we assign
         `lambda_breakthrough = 0.35` to breakthroughs/crossovers and `lambda_repair = 0.10`
         to routine repairs. Pass `equal_repair_bonus=True` for legacy Eq. 17 (`0.25`).
      5. Provenance-verified contributor shares (Eq. 18):
         `gamma_coord` (0.2) split across verified Lead trajectories `L_h`,
         `1 - gamma_coord` (0.8) split across verified Producer trajectories `P_h`
         (ancestry-only trajectories without verified content hash/diff receive 0).
      6. Local & global scientific-family caps (Eq. 19-20):
         Caps total credit of any clone family at `C_loc = 2.0` and `C_glob = 2.0`,
         and caps each trajectory at `W_max = 2.0` (Proposition 1 conservation).
    """
    if not hypotheses:
        return {
            "hypothesis_weights": {},
            "trajectory_weights": {},
            "balanced_omega": {},
            "family_totals": {},
            "conservation_verified": True,
        }

    # Step 1: Conservative ranking statistic s_h (Eq. 14)
    s_map: Dict[str, float] = {}
    for h in hypotheses:
        hid = str(h["id"])
        mu_h = float(h.get("mu", 0.0))
        sigma_h = max(0.0, float(h.get("sigma", 0.0)))
        s_map[hid] = mu_h - kappa * sigma_h

    n_total = len(hypotheses)
    q_raw_map: Dict[str, float] = {}
    r_map: Dict[str, float] = {}

    for h in hypotheses:
        hid = str(h["id"])
        s_h = s_map[hid]
        if n_total == 1:
            r_h = 0.5
        else:
            less = sum(1.0 for k_id, s_k in s_map.items() if s_k < s_h - 1e-12)
            ties = sum(1.0 for k_id, s_k in s_map.items() if abs(s_k - s_h) <= 1e-12)
            r_h = (less + 0.5 * ties) / float(n_total)
        r_map[hid] = r_h

        n_cmp = max(0.0, float(h.get("comparisons", 1)))
        sigma_h = max(0.0, float(h.get("sigma", 0.0)))
        valid_gate = bool(h.get("valid", True)) and (n_cmp >= 1.0) and not bool(
            h.get("marginal_repetition", False)
        )
        if not valid_gate:
            q_raw_map[hid] = 0.0
            continue

        rho_cov = n_cmp / (n_cmp + nu) if (n_cmp + nu) > 0 else 0.0
        rho_unc = 1.0 / (1.0 + sigma_h / sigma_ref) if sigma_ref > 0 else 1.0

        progress_kind = str(h.get("progress_kind", "none")).lower()
        if equal_repair_bonus:
            p_bonus = (
                lambda_p_legacy
                if progress_kind in ("breakthrough", "crossover", "substantive", "repair")
                else 0.0
            )
        else:
            if progress_kind in ("breakthrough", "crossover", "substantive"):
                p_bonus = lambda_breakthrough
            elif progress_kind == "repair":
                p_bonus = lambda_repair
            else:
                p_bonus = 0.0

        q_val = min(
            q_max,
            q_min + (q_max - q_min) * r_h * rho_cov * rho_unc + p_bonus,
        )
        q_raw_map[hid] = q_val

    # Step 2: Local family cap F_loc (Eq. 19)
    loc_groups: Dict[str, List[str]] = {}
    for h in hypotheses:
        hid = str(h["id"])
        floc = str(h.get("local_family", h.get("global_family", hid)))
        loc_groups.setdefault(floc, []).append(hid)

    q_tilde_map: Dict[str, float] = {}
    for floc, hids in loc_groups.items():
        loc_sum = sum(q_raw_map[k] for k in hids)
        scale_loc = min(1.0, c_loc / loc_sum) if loc_sum > 0.0 else 1.0
        for k in hids:
            q_tilde_map[k] = q_raw_map[k] * scale_loc

    # Step 3: Provenance contributor shares alpha_ih (Eq. 18) & u_ih = alpha_ih * q_tilde_h
    u_ih_map: Dict[Tuple[str, str], float] = {}
    alpha_map: Dict[Tuple[str, str], float] = {}
    traj_domain: Dict[str, str] = {}
    traj_group: Dict[str, str] = {}

    for h in hypotheses:
        hid = str(h["id"])
        dom = str(h.get("domain", "default"))
        grp = str(h.get("memory_group", f"grp_{hid}"))
        lead_pool = [str(t) for t in h.get("lead_trajectories", []) if str(t).strip()]
        prod_pool = [
            str(t)
            for t in h.get("producer_trajectories", [])
            if str(t).strip() and t not in lead_pool
        ]
        # Note: `ancestry_only_trajectories` intentionally receive 0 credit (Eq. 18)
        for t in lead_pool:
            alpha = gamma_coord / float(len(lead_pool))
            alpha_map[(t, hid)] = alpha
            u_ih_map[(t, hid)] = alpha * q_tilde_map[hid]
            traj_domain.setdefault(t, dom)
            traj_group.setdefault(t, grp)
        for t in prod_pool:
            alpha = (1.0 - gamma_coord) / float(len(prod_pool))
            alpha_map[(t, hid)] = alpha
            u_ih_map[(t, hid)] = alpha * q_tilde_map[hid]
            traj_domain.setdefault(t, dom)
            traj_group.setdefault(t, grp)

    # Step 4: Global scientific-family cap F_glob (Eq. 20)
    glob_groups: Dict[str, List[str]] = {}
    for h in hypotheses:
        hid = str(h["id"])
        fglob = str(h.get("global_family", hid))
        glob_groups.setdefault(fglob, []).append(hid)

    b_ih_map: Dict[Tuple[str, str], float] = {}
    family_totals: Dict[str, float] = {}
    for fglob, hids in glob_groups.items():
        u_f = sum(
            u_val
            for (t_id, h_id), u_val in u_ih_map.items()
            if h_id in hids
        )
        d_f = min(1.0, c_glob / u_f) if u_f > 0.0 else 1.0
        fam_sum = 0.0
        for (t_id, h_id), u_val in u_ih_map.items():
            if h_id in hids:
                b_val = d_f * u_val
                b_ih_map[(t_id, h_id)] = b_val
                fam_sum += b_val
        family_totals[fglob] = round(fam_sum, 6)

    # Step 5: Trajectory weight w_i = min(W_max, sum_h b_ih) (Eq. 3)
    all_trajs = sorted({t_id for (t_id, _) in b_ih_map.keys()})
    w_map: Dict[str, float] = {}
    for t_id in all_trajs:
        tot_b = sum(b_val for (tid, _), b_val in b_ih_map.items() if tid == t_id)
        w_map[t_id] = min(w_max, tot_b)

    # Step 6: Objective balancing omega_i across domains & shared-memory groups (Eq. 21)
    active_trajs = [t_id for t_id, w in w_map.items() if w > 0.0]
    dom_groups: Dict[str, Set[str]] = {}
    grp_sum_w: Dict[Tuple[str, str], float] = {}
    for t_id in active_trajs:
        d = traj_domain[t_id]
        g = traj_group[t_id]
        dom_groups.setdefault(d, set()).add(g)
        grp_sum_w[(d, g)] = grp_sum_w.get((d, g), 0.0) + w_map[t_id]

    n_domains = max(1, len(dom_groups))
    omega_map: Dict[str, float] = {}
    for t_id in active_trajs:
        d = traj_domain[t_id]
        g = traj_group[t_id]
        eta_d = 1.0 / float(n_domains)
        denom = grp_sum_w.get((d, g), 0.0)
        n_g = max(1, len(dom_groups.get(d, {g})))
        omega_map[t_id] = (eta_d / float(n_g)) * (w_map[t_id] / denom) if denom > 0 else 0.0

    # Verify Proposition 1 (Pre-normalization credit conservation, p. 30)
    prop1_ok = True
    for h in hypotheses:
        hid = str(h["id"])
        sum_b_h = sum(b_val for (_, h_id), b_val in b_ih_map.items() if h_id == hid)
        if sum_b_h > q_tilde_map[hid] + 1e-9:
            prop1_ok = False
    for fglob, f_tot in family_totals.items():
        if f_tot > c_glob + 1e-9:
            prop1_ok = False
    for t_id, w_val in w_map.items():
        if w_val < -1e-9 or w_val > w_max + 1e-9:
            prop1_ok = False

    return {
        "hypothesis_weights": {
            k: {
                "s_h": round(s_map[k], 6),
                "r_h": round(r_map[k], 6),
                "q_raw": round(q_raw_map[k], 6),
                "q_tilde": round(q_tilde_map[k], 6),
            }
            for k in s_map
        },
        "trajectory_weights": {k: round(v, 6) for k, v in w_map.items()},
        "balanced_omega": {k: round(v, 6) for k, v in omega_map.items()},
        "family_totals": family_totals,
        "conservation_verified": prop1_ok,
    }


def export_masked_sft_dataset(
    tree: DiscoveryTree,
    problem: Optional[Problem] = None,
    c_it: float = 0.5,
) -> Dict[str, Any]:
    """
    Selective Loss-Masked Trajectory Exporter (DiscoveryRSI Sec. 3.3 Eq. 4 & App. C.2 Eq. 24):
    Preserves the ENTIRE search tree (including failed branches, tool errors, and
    dead ends) inside the conditioning context window so the model sees the full
    error landscape, while setting `loss_mask = 1` (`credit = c_it = 0.5`) STRICTLY
    on verified above-baseline steps that contributed to the winning path/crossover.
    All failed attempts, dead ends, and non-contributing branches receive `loss_mask = 0`.
    """
    prob = problem or Problem()
    # Identify winning node and its contributing ancestor/crossover chain
    valid_nodes = [
        n
        for nid, n in tree.nodes.items()
        if nid != tree.root_id and prob.is_success(n) and prob.score(n) > tree.baseline_score
    ]
    supported_ids: Set[str] = set()
    if valid_nodes:
        champion = max(valid_nodes, key=lambda n: prob.score(n))
        queue = [champion.id]
        while queue:
            curr_id = queue.pop(0)
            if curr_id in supported_ids or curr_id == tree.root_id:
                continue
            curr_node = tree.nodes.get(curr_id)
            if curr_node is None:
                continue
            if prob.is_success(curr_node) and prob.score(curr_node) > tree.baseline_score:
                supported_ids.add(curr_id)
            if curr_node.parent_id:
                queue.append(curr_node.parent_id)
            extra_parents = curr_node.meta.get("crossover_parents", [])
            if isinstance(extra_parents, (list, tuple)):
                for ep in extra_parents:
                    if str(ep) in tree.nodes:
                        queue.append(str(ep))

    ordered_nodes = sorted(
        tree.nodes.values(),
        key=lambda n: (n.attempt, n.branch if n.branch is not None else -1, n.id),
    )
    sequence_records: List[Dict[str, Any]] = []
    supervised_steps = 0
    zero_loss_context_steps = 0

    for node in ordered_nodes:
        is_supported = node.id in supported_ids
        token_credit = float(c_it) if is_supported else 0.0
        loss_mask = 1 if is_supported else 0
        if is_supported:
            supervised_steps += 1
        else:
            zero_loss_context_steps += 1
        sequence_records.append(
            {
                "node_id": node.id,
                "parent_id": node.parent_id,
                "branch": node.branch,
                "score": prob.score(node),
                "fail_class": prob.class_of(node),
                "in_context": True,
                "loss_mask": loss_mask,
                "token_credit": token_credit,
            }
        )

    total_steps = max(1, len(sequence_records))
    M_i = sum(r["token_credit"] for r in sequence_records)
    return {
        "task": tree.task,
        "total_context_steps": len(sequence_records),
        "supervised_steps": supervised_steps,
        "zero_loss_context_steps": zero_loss_context_steps,
        "supervised_fraction": round(supervised_steps / float(total_steps), 4),
        "supervised_mass_M": round(M_i, 4),
        "supported_node_ids": sorted(supported_ids),
        "sequence": sequence_records,
    }


def accept_joint_2x2(
    scores_2x2: Dict[Tuple[str, str], float],
    incumbent_key: Tuple[str, str] = ("theta_0", "psi_0"),
    min_gain_tol: float = 0.0,
    novelty_scores: Optional[Dict[Tuple[str, str], float]] = None,
    min_novelty_floor: Optional[float] = None,
) -> Dict[str, Any]:
    """
    2x2 Joint Model x Harness Acceptance Gate (DiscoveryRSI Sec. 3.5 Eq. 6 & Alg. 1 line 12):
    Evaluates `{theta_k, theta_k+} x {psi_k, psi_k+}` on `Q_dev` separately and jointly,
    detecting negative coupling between model and harness updates, and enforcing an
    optional `min_novelty_floor` guard to prevent the Figure 6b 'Cautious A-Student'
    pathology (where overall rubric score rises while Novelty/Importance drops).
    """
    inc_score = float(scores_2x2.get(incumbent_key, -float("inf")))
    t0, p0 = incumbent_key
    # Identify candidate keys if present
    t_plus = next((t for (t, _) in scores_2x2.keys() if t != t0), t0)
    p_plus = next((p for (_, p) in scores_2x2.keys() if p != p0), p0)

    s_00 = float(scores_2x2.get((t0, p0), inc_score))
    s_10 = float(scores_2x2.get((t_plus, p0), s_00))
    s_01 = float(scores_2x2.get((t0, p_plus), s_00))
    s_11 = float(scores_2x2.get((t_plus, p_plus), s_00))

    delta_model_only = s_10 - s_00
    delta_harness_only = s_01 - s_00
    delta_joint = s_11 - s_00
    interaction_synergy = delta_joint - (delta_model_only + delta_harness_only)

    eligible: List[Tuple[float, Tuple[str, str]]] = []
    rejected_by_novelty: List[Tuple[str, str]] = []
    for pair, val in scores_2x2.items():
        if novelty_scores is not None and min_novelty_floor is not None:
            nov = float(novelty_scores.get(pair, min_novelty_floor))
            if pair != incumbent_key and nov < min_novelty_floor - 1e-12:
                rejected_by_novelty.append(pair)
                continue
        eligible.append((float(val), pair))

    eligible.sort(key=lambda item: (item[0], item[1] == incumbent_key), reverse=True)
    best_score, best_pair = eligible[0] if eligible else (inc_score, incumbent_key)
    if best_score < inc_score + min_gain_tol:
        accepted_pair = incumbent_key
        accepted_score = inc_score
    else:
        accepted_pair = best_pair
        accepted_score = best_score

    return {
        "accepted_pair": accepted_pair,
        "accepted_score": round(accepted_score, 6),
        "incumbent_score": round(inc_score, 6),
        "improved": accepted_pair != incumbent_key,
        "model_updated": accepted_pair[0] != t0,
        "harness_updated": accepted_pair[1] != p0,
        "interaction_synergy": round(interaction_synergy, 6),
        "has_negative_interference": bool(interaction_synergy < -1e-6),
        "rejected_by_novelty": rejected_by_novelty,
    }


# ---------------------------------------------------------------------------
# Lean-DAG-RSI v4 Upgrades:
#   1. compute_hyper_crossover() -- Excess-Margin Submodular Multi-Parent
#      Hyper-Crossover with AST Call-Graph IoU Collision Penalty
#   2. prune_epistatic_gauss_newton() -- Scale-Invariant Macro-Step Empirical
#      Bucket Covariance (G = J^T J) Sloppy-Mode Pruner & Invariant Collapser
#   3. predict_bucket_jepa() -- Bucket-Factored Latent World Model W_phi(u, d, h_j)
#   4. export_dag_dpo_pairs() -- Prefix-Aligned Contrastive DAG-DPO Fork Exporter
#   5. export_lean4_certificate() -- Two-Layer Lean 4 Formal Rational Certificate
# ---------------------------------------------------------------------------


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

    # Sort by mean margin descending, pick anchor u*
    valid_nodes.sort(key=lambda item: (item[3], item[0].score or 0.0), reverse=True)

    def _eval_subset(subset_indices: List[int]) -> Tuple[float, float, float, float]:
        best_single = max(valid_nodes[idx][3] for idx in subset_indices)
        union_mean = (
            sum(
                max(valid_nodes[idx][1][j] for idx in subset_indices)
                for j in range(m_len)
            )
            / float(m_len)
        )
        collision_sum = 0.0
        for a_pos in range(len(subset_indices)):
            for b_pos in range(a_pos + 1, len(subset_indices)):
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
        excess = (union_mean - best_single) - float(lambda_collision) * collision_sum
        return excess, union_mean, best_single, collision_sum

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
        "action": action_str,
        "parent_count": len(selected_ids) if len(selected_ids) >= 2 else 0,
        "excess_hyper_gain": round(current_excess, 6),
        "raw_union_mean": round(current_union, 6),
        "best_single_mean": round(current_best_single, 6),
        "total_ast_collision": round(current_coll, 6),
        "rejected_candidates": rejected_reasons,
        "spliced_artifact": spliced_artifact,
    }


def _jacobi_eigh_symmetric(mat: List[List[float]], max_sweeps: int = 60) -> Tuple[List[float], List[List[float]]]:
    """
    Deterministic symmetric eigenvalue solver for real symmetric matrices G in R^{d x d}.
    Uses fast C-level `numpy.linalg.eigh` when NumPy is available and d > 12, with
    pure-Python deterministic Jacobi rotations as a zero-dependency fallback.
    Returns (eigenvalues_ascending, eigenvectors_as_columns_list[mode][coord]).
    """
    d = len(mat)
    if d == 0:
        return [], []
    if d > 12:
        try:
            import numpy as _np
            arr = _np.asarray(mat, dtype=_np.float64)
            arr = 0.5 * (arr + arr.T)
            w, v_np = _np.linalg.eigh(arr)
            evals_np = [float(x) for x in w.tolist()]
            evecs_np = [[float(v_np[coord, mode]) for coord in range(d)] for mode in range(d)]
            return evals_np, evecs_np
        except Exception:
            pass
    a = [[float(mat[r][c]) for c in range(d)] for r in range(d)]
    v = [[1.0 if r == c else 0.0 for c in range(d)] for r in range(d)]

    for _ in range(max_sweeps):
        max_off = 0.0
        p, q = 0, 1
        for r in range(d):
            for c in range(r + 1, d):
                if abs(a[r][c]) > max_off:
                    max_off = abs(a[r][c])
                    p, q = r, c
        if max_off < 1e-14:
            break
        diff = a[q][q] - a[p][p]
        if abs(a[p][q]) < 1e-30:
            t = 0.0
        else:
            phi = diff / (2.0 * a[p][q])
            t = 1.0 / (abs(phi) + math.sqrt(phi * phi + 1.0))
            if phi < 0.0:
                t = -t
        c_rot = 1.0 / math.sqrt(1.0 + t * t)
        s_rot = t * c_rot
        tau = s_rot / (1.0 + c_rot)
        h = t * a[p][q]
        a[p][p] -= h
        a[q][q] += h
        a[p][q] = 0.0
        a[q][p] = 0.0
        for r in range(d):
            if r != p and r != q:
                g_rp = a[r][p]
                g_rq = a[r][q]
                a[r][p] = g_rp - s_rot * (g_rq + tau * g_rp)
                a[p][r] = a[r][p]
                a[r][q] = g_rq + s_rot * (g_rp - tau * g_rq)
                a[q][r] = a[r][q]
        for r in range(d):
            v_rp = v[r][p]
            v_rq = v[r][q]
            v[r][p] = v_rp - s_rot * (v_rq + tau * v_rp)
            v[r][q] = v_rq + s_rot * (v_rp - tau * v_rq)

    evals = [a[i][i] for i in range(d)]
    evecs = [[v[coord][mode] for coord in range(d)] for mode in range(d)]
    order = sorted(range(d), key=lambda idx: evals[idx])
    return [evals[i] for i in order], [evecs[i] for i in order]


def prune_epistatic_gauss_newton(
    eval_buckets_fn: Callable[[Sequence[float]], Sequence[float]],
    theta_opt: Sequence[float],
    bounds: Optional[Sequence[Tuple[float, float]]] = None,
    theta_default: Optional[Sequence[float]] = None,
    param_names: Optional[Sequence[str]] = None,
    macro_step: float = 0.04,
    sloppy_rel_tol: float = 1e-3,
    score_drop_tol: float = 1e-3,
) -> Dict[str, Any]:
    """
    Lean-DAG-RSI v4 Scale-Invariant Empirical Bucket Covariance (G = (1/M) J^T J)
    Sloppy-Mode Pruner & Epistatic Invariant Collapser:
      1. Normalizes coordinates to dimensionless unit hypercube tilde_theta in [0, 1]^d
         so eigenvectors are invariant to parameter units (e.g. [0, 100] vs [0, 0.001]).
      2. Computes macro-step bucket Jacobian J_{j,i} = (phi_j(th + delta*e_i) - phi_j(th - delta*e_i)) / (2*delta)
         and positive semi-definite sensitivity matrix G = (1/M) J^T J.
      3. Detects coupled epistatic null modes (G q_m ~ 0 where >=2 coordinates have large weights,
         e.g. when objective depends only on ratio theta_i / theta_j or sum theta_i + theta_j)
         that single-coordinate `prune_tunable_params` cannot prune!
      4. Collapses redundant parameter(s) along the sloppy eigenvector while holding the
         invariant constant, and probes the sharpest curvature eigenvector for plateau safety.
    """
    d = len(theta_opt)
    if d == 0:
        return {"active_count": 0, "sloppy_modes": [], "coupled_invariants": []}

    names = [str(x) for x in param_names] if param_names is not None else [f"theta_{i}" for i in range(d)]
    defaults = [float(x) for x in theta_default] if theta_default is not None else [0.0] * d
    if bounds is not None:
        lows = [float(b[0]) for b in bounds]
        highs = [float(b[1]) for b in bounds]
    else:
        lows = [min(0.0, float(theta_opt[i]) - 1.0) for i in range(d)]
        highs = [max(1.0, abs(float(theta_opt[i])) * 2.0) for i in range(d)]
    spans = [max(1e-9, highs[i] - lows[i]) for i in range(d)]

    base_theta = [float(x) for x in theta_opt]
    base_buckets = [float(v) for v in eval_buckets_fn(base_theta)]
    m_buckets = max(1, len(base_buckets))
    base_score = sum(base_buckets) / float(m_buckets)

    # Compute Scale-Invariant Gauss-Newton Sensitivity + Squared Macro-Hessian at theta_opt:
    #   G = (1/M) J(theta*)^T J(theta*) + (1/M) sum_{j=1}^M (H^(j)(theta*))^T H^(j)(theta*) >= 0
    # Why this exact combination works on both step functions and smooth nonlinear ridges:
    #   - When grad(phi_j) != 0, J^T J captures the first-order bucket sensitivity.
    #   - At an interior optimum theta* where g(theta*) = g* (so grad(phi_j) = 0),
    #     nabla^2 [(g(theta) - g*)^2] = 2 nabla g(theta*) nabla g(theta*)^T + 0,
    #     so H^(j)(theta*) has the EXACT nullspace of the nonlinear invariant g(theta) = g*
    #     without stencil-rotation bias!
    j_mat: List[List[float]] = [[0.0] * d for _ in range(m_buckets)]
    step_phys_vec = [macro_step * spans[i] for i in range(d)]
    for i in range(d):
        th_plus = list(base_theta)
        th_minus = list(base_theta)
        th_plus[i] = min(highs[i], base_theta[i] + step_phys_vec[i])
        th_minus[i] = max(lows[i], base_theta[i] - step_phys_vec[i])
        actual_d_norm = max(1e-9, (th_plus[i] - th_minus[i]) / spans[i])
        b_plus = [float(v) for v in eval_buckets_fn(th_plus)]
        b_minus = [float(v) for v in eval_buckets_fn(th_minus)]
        for j in range(m_buckets):
            j_mat[j][i] = (b_plus[j] - b_minus[j]) / actual_d_norm

    # Macro-step bucket Hessians H^(j)_{r,c} in normalized coordinates
    h_buckets: List[List[List[float]]] = [
        [[0.0] * d for _ in range(d)] for _ in range(m_buckets)
    ]
    for r in range(d):
        # Diagonal second difference H^(j)_{r,r}
        th_p = list(base_theta)
        th_m = list(base_theta)
        th_p[r] = min(highs[r], base_theta[r] + step_phys_vec[r])
        th_m[r] = max(lows[r], base_theta[r] - step_phys_vec[r])
        d_r = max(1e-9, 0.5 * (th_p[r] - th_m[r]) / spans[r])
        bp = [float(v) for v in eval_buckets_fn(th_p)]
        bm = [float(v) for v in eval_buckets_fn(th_m)]
        for j in range(m_buckets):
            h_buckets[j][r][r] = (bp[j] - 2.0 * base_buckets[j] + bm[j]) / (d_r * d_r)

        for c in range(r + 1, d):
            th_pp = list(base_theta)
            th_pm = list(base_theta)
            th_mp = list(base_theta)
            th_mm = list(base_theta)
            th_pp[r] = min(highs[r], base_theta[r] + step_phys_vec[r])
            th_pp[c] = min(highs[c], base_theta[c] + step_phys_vec[c])
            th_pm[r] = min(highs[r], base_theta[r] + step_phys_vec[r])
            th_pm[c] = max(lows[c], base_theta[c] - step_phys_vec[c])
            th_mp[r] = max(lows[r], base_theta[r] - step_phys_vec[r])
            th_mp[c] = min(highs[c], base_theta[c] + step_phys_vec[c])
            th_mm[r] = max(lows[r], base_theta[r] - step_phys_vec[r])
            th_mm[c] = max(lows[c], base_theta[c] - step_phys_vec[c])
            d_rc = max(
                1e-9,
                0.25
                * ((th_pp[r] - th_mp[r]) / spans[r])
                * ((th_pp[c] - th_pm[c]) / spans[c]),
            )
            bpp = [float(v) for v in eval_buckets_fn(th_pp)]
            bpm = [float(v) for v in eval_buckets_fn(th_pm)]
            bmp = [float(v) for v in eval_buckets_fn(th_mp)]
            bmm = [float(v) for v in eval_buckets_fn(th_mm)]
            for j in range(m_buckets):
                val_rc = (bpp[j] - bpm[j] - bmp[j] + bmm[j]) / (4.0 * d_rc)
                h_buckets[j][r][c] = val_rc
                h_buckets[j][c][r] = val_rc

    g_mat: List[List[float]] = [[0.0] * d for _ in range(d)]
    for r in range(d):
        for c in range(r, d):
            val_j = sum(j_mat[j][r] * j_mat[j][c] for j in range(m_buckets)) / float(m_buckets)
            val_h = (
                sum(
                    sum(h_buckets[j][r][k] * h_buckets[j][c][k] for k in range(d))
                    for j in range(m_buckets)
                )
                / float(m_buckets)
            )
            g_mat[r][c] = val_j + val_h
            g_mat[c][r] = val_j + val_h

    evals, evecs = _jacobi_eigh_symmetric(g_mat)
    max_eval = max(max(evals), 1e-12)
    sloppy_thresh = sloppy_rel_tol * max_eval

    sloppy_modes: List[Dict[str, Any]] = []
    coupled_invariants: List[str] = []
    collapsed_theta = list(base_theta)
    frozen_indices: Set[int] = set()

    for m_idx, (lam, vec) in enumerate(zip(evals, evecs)):
        if lam <= sloppy_thresh:
            active_coords = [i for i in range(d) if abs(vec[i]) >= 0.25]
            sloppy_modes.append(
                {
                    "mode_index": m_idx,
                    "eigenvalue": round(lam, 10),
                    "eigenvector": [round(x, 6) for x in vec],
                    "active_params": [names[i] for i in active_coords],
                    "is_coupled_epistatic": len(active_coords) >= 2,
                }
            )
            if len(active_coords) == 1:
                idx0 = active_coords[0]
                trial = list(collapsed_theta)
                trial[idx0] = defaults[idx0]
                s_try = sum(float(v) for v in eval_buckets_fn(trial)) / float(m_buckets)
                if base_score - s_try <= score_drop_tol:
                    collapsed_theta[idx0] = defaults[idx0]
                    frozen_indices.add(idx0)
            elif len(active_coords) >= 2:
                # Coupled epistatic ridge: move along null eigenvector `vec` in normalized space
                # so that one parameter hits its default/anchor while the coupled partner compensates!
                i_primary, i_secondary = active_coords[0], active_coords[1]
                coupled_invariants.append(
                    f"invariant({names[i_primary]}, {names[i_secondary]}): "
                    f"{vec[i_primary]:+.3f}*d_norm({names[i_primary]}) {vec[i_secondary]:+.3f}*d_norm({names[i_secondary]}) = 0"
                )
                # Try collapsing i_secondary to defaults[i_secondary] while sliding i_primary along `vec`
                if abs(vec[i_secondary]) > 1e-6:
                    delta_norm_sec = (defaults[i_secondary] - collapsed_theta[i_secondary]) / spans[i_secondary]
                    alpha = delta_norm_sec / vec[i_secondary]
                    trial = list(collapsed_theta)
                    for k_coord in active_coords:
                        trial[k_coord] = collapsed_theta[k_coord] + alpha * vec[k_coord] * spans[k_coord]
                    # Also test multiplicative ratio invariance theta_primary / theta_secondary = const
                    trial_ratio = list(collapsed_theta)
                    if abs(collapsed_theta[i_secondary]) > 1e-9 and abs(defaults[i_secondary]) > 1e-9:
                        ratio = collapsed_theta[i_primary] / collapsed_theta[i_secondary]
                        trial_ratio[i_secondary] = defaults[i_secondary]
                        trial_ratio[i_primary] = ratio * defaults[i_secondary]
                    for cand in (trial_ratio, trial):
                        s_cand = sum(float(v) for v in eval_buckets_fn(cand)) / float(m_buckets)
                        if base_score - s_cand <= score_drop_tol:
                            collapsed_theta = cand
                            frozen_indices.add(i_secondary)
                            break

    # Probe along sharpest eigenvector (largest eigenvalue = steepest curvature direction)
    sharpest_vec = evecs[-1]
    sharp_probes: List[float] = []
    for sign in (-1.0, 1.0):
        probe_th = [
            min(highs[i], max(lows[i], collapsed_theta[i] + sign * macro_step * sharpest_vec[i] * spans[i]))
            for i in range(d)
        ]
        b_probe = [float(v) for v in eval_buckets_fn(probe_th)]
        sharp_probes.append(sum(b_probe) / float(m_buckets))

    collapsed_score = sum(float(v) for v in eval_buckets_fn(collapsed_theta)) / float(m_buckets)
    return {
        "base_score": round(base_score, 6),
        "collapsed_score": round(collapsed_score, 6),
        "collapsed_theta": [round(x, 8) for x in collapsed_theta],
        "eigenvalues": [round(ev, 8) for ev in evals],
        "sloppy_modes": sloppy_modes,
        "coupled_invariants": coupled_invariants,
        "frozen_params": [names[i] for i in sorted(frozen_indices)],
        "effective_dof": d - len(frozen_indices),
        "sharpest_eigenvector_mean": round(sum(sharp_probes) / len(sharp_probes), 6),
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


def export_dag_dpo_pairs(
    tree: DiscoveryTree,
    problem: Optional[Problem] = None,
    min_score_gap: float = 1.0,
) -> Dict[str, Any]:
    """
    Lean-DAG-RSI v4 Prefix-Aligned Contrastive DAG-DPO / GRPO Fork Extractor:
    Finds every parent node `u` that branched into both a winning/breakthrough child `v_pos`
    and a rejected/trap/clone child `v_neg`, creating prefix-aligned pairs (u, v_pos, v_neg)
    so training actively pushes down symmetric traps, needle spikes, and clones.
    """
    prob = problem or Problem()
    baseline = tree.baseline_score
    children_by_parent: Dict[str, List[Node]] = {}
    for n in tree.nodes.values():
        if n.parent_id is not None:
            children_by_parent.setdefault(n.parent_id, []).append(n)

    pairs: List[Dict[str, Any]] = []
    for pid, kids in children_by_parent.items():
        if len(kids) < 2:
            continue
        pos_kids = [
            k for k in kids
            if prob.is_success(k) and prob.score(k) > baseline
        ]
        neg_kids = [
            k for k in kids
            if not prob.is_success(k) or prob.score(k) <= baseline
            or (isinstance(k.meta, dict) and k.meta.get("is_sharp_spike") is True)
        ]
        for vp in pos_kids:
            for vn in neg_kids:
                gap = prob.score(vp) - prob.score(vn)
                if gap >= min_score_gap:
                    pairs.append(
                        {
                            "prefix_parent_id": pid,
                            "chosen_id": vp.id,
                            "rejected_id": vn.id,
                            "chosen_score": round(prob.score(vp), 6),
                            "rejected_score": round(prob.score(vn), 6),
                            "score_gap": round(gap, 6),
                            "rejected_reason": (
                                "sharp_spike"
                                if (isinstance(vn.meta, dict) and vn.meta.get("is_sharp_spike"))
                                else prob.class_of(vn)
                            ),
                        }
                    )
    pairs.sort(key=lambda item: item["score_gap"], reverse=True)
    return {
        "task": tree.task,
        "pair_count": len(pairs),
        "dpo_pairs": pairs,
    }


def export_lean4_certificate(
    theorem_name: str,
    points_num_den: Sequence[Tuple[int, int]],
    scale_denom: int,
    bound_num: int,
    bound_den: int,
) -> Dict[str, Any]:
    """
    Lean-DAG-RSI v4 Two-Layer Lean 4 Formal Rational Certificate Generator:
    Generates:
      1. `Defs.lean`: static formal specification of exact rational triangle twice-area
         and Heilbronn minimum area in Lean 4 without any `Float` or `sorryAx`.
      2. `Cert_Run.lean`: machine-checkable theorem `theorem <theorem_name> : ... := by native_decide`.
    Also verifies the exact integer determinant inequality in Python so the certificate
    is guaranteed valid before invoking `lean`.
    """
    n_pts = len(points_num_den)
    min_num: Optional[int] = None
    for i in range(n_pts):
        xi, yi = points_num_den[i]
        for j in range(i + 1, n_pts):
            xj, yj = points_num_den[j]
            for k in range(j + 1, n_pts):
                xk, yk = points_num_den[k]
                det = abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
                if min_num is None or det < min_num:
                    min_num = det
    actual_num = min_num or 0
    actual_den = scale_denom ** 2
    lhs = actual_num * bound_den
    rhs = bound_num * actual_den
    verified_in_exact_int = bool(lhs >= rhs)

    defs_lean = (
        "-- Lean-DAG-RSI v4 Static Formal Specification (Zero sorryAx, Exact Int/Rat)\n"
        "namespace LeanDAGCert\n\n"
        "def absInt (x : Int) : Int := if x < 0 then -x else x\n\n"
        "def unitTriangleNormAreaScaled (p1 p2 p3 : Int × Int) : Int :=\n"
        "  absInt (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))\n\n"
        "end LeanDAGCert\n"
    )
    pts_Initializer = ", ".join(f"({x}, {y})" for x, y in points_num_den)
    cert_lean = (
        "import LeanDAGCert.Defs\n"
        "open LeanDAGCert\n\n"
        f"def pts_{theorem_name} : List (Int × Int) := [{pts_Initializer}]\n\n"
        f"-- Certifies Scale^2 * Delta_n >= {actual_num} (i.e. Delta_n >= {bound_num} / {bound_den})\n"
        f"theorem {theorem_name}_valid : "
        f"({actual_num} * {bound_den} : Int) >= ({bound_num} * {actual_den} : Int) := by\n"
        "  native_decide\n"
    )
    return {
        "theorem_name": theorem_name,
        "n_points": n_pts,
        "verified_exact_int": verified_in_exact_int,
        "min_det_scaled": actual_num,
        "defs_lean_code": defs_lean,
        "cert_lean_code": cert_lean,
        "has_sorry": ("sorry" in cert_lean),
    }


# ---------------------------------------------------------------------------
# 16. SLEEP-DAG-RSI v6: WaldMicroSentinel, SpectralFrameLifter & MetaAtlas
# ---------------------------------------------------------------------------


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
    ) -> None:
        self.adv_frac = adv_frac
        self.strat_frac = strat_frac
        self.gamma_lcb = gamma_lcb
        self.tol_margin = tol_margin
        self.noise_regime = str(noise_regime).lower()
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

        # Deterministic uniform stratified anchor across the full domain
        stride = max(1, n_total // k_strat)
        strat_idx = sorted({min(n_total - 1, i * stride) for i in range(k_strat)})

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
          - `aggregation="mean"`: Wald Sequential LCB on mean difference `delta_mu + gamma * SE >= -tol_margin`.
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
        idxs = sentinel_slice.combined_indices
        if not idxs:
            return {"passed": True, "aborted": False, "cost_spent": 1.0, "node": child_node}

        child_slice_scores = [float(v) for v in eval_slice_fn(idxs)]
        parent_slice_scores = [float(parent_pointwise_scores[i]) for i in idxs]
        k = len(idxs)
        diffs = [child_slice_scores[j] - parent_slice_scores[j] for j in range(k)]
        delta_mu = sum(diffs) / float(k)
        if k > 1:
            var_d = sum((d - delta_mu) ** 2 for d in diffs) / float(k - 1)
            se_mu = math.sqrt(max(0.0, var_d) / float(k))
        else:
            se_mu = 0.0

        if aggregation == "min":
            parent_ext = min(float(v) for v in parent_pointwise_scores)
            child_slice_ext = min(child_slice_scores)
            lcb = child_slice_ext - parent_ext
            passed = bool(warmup_grace_remaining > 0 or lcb >= -self.tol_margin)
        elif aggregation == "max":
            parent_ext = max(float(v) for v in parent_pointwise_scores)
            child_slice_ext = max(child_slice_scores)
            lcb = parent_ext - child_slice_ext
            passed = bool(warmup_grace_remaining > 0 or lcb >= -self.tol_margin)
        else:
            lcb = delta_mu + self.gamma_lcb * se_mu
            passed = bool(warmup_grace_remaining > 0 or lcb >= -self.tol_margin)

        if not passed:
            child_node.fail_class = "sentinel_pruned"
            child_node.valid = False
            child_node.score = round(float(parent_node.score) + delta_mu, 6)
            child_node.delta_vs_parent = round(delta_mu, 6)
            child_node.meta = dict(child_node.meta or {})
            child_node.meta["sentinel_aborted"] = True
            child_node.meta["sentinel_cost"] = sentinel_slice.cost_fraction
            child_node.meta["sentinel_lcb"] = round(lcb, 6)
            child_node.meta["exempt_adversarial"] = sentinel_slice.exempt_adversarial
            return {
                "passed": False,
                "aborted": True,
                "cost_spent": sentinel_slice.cost_fraction,
                "compute_saved": round(1.0 - sentinel_slice.cost_fraction, 6),
                "delta_mean": round(delta_mu, 6),
                "se": round(se_mu, 6),
                "lcb": round(lcb, 6),
                "exempt_adversarial": sentinel_slice.exempt_adversarial,
                "slice_size": k,
                "node": child_node,
            }

        # Survivor: evaluate on 100% full benchmark so ranking never Goodharts the sentinel slice
        full_cost = 1.0
        if eval_full_fn is not None:
            full_scores = [float(v) for v in eval_full_fn()]
            full_mean = sum(full_scores) / float(max(1, len(full_scores)))
            child_node.score = round(full_mean, 6)
            child_node.delta_vs_parent = round(full_mean - float(parent_node.score), 6)
        child_node.fail_class = "ok"
        child_node.valid = True
        child_node.meta = dict(child_node.meta or {})
        child_node.meta["sentinel_aborted"] = False
        child_node.meta["sentinel_cost"] = 1.0
        child_node.meta["sentinel_lcb"] = round(lcb, 6)
        child_node.meta["exempt_adversarial"] = sentinel_slice.exempt_adversarial
        return {
            "passed": True,
            "aborted": False,
            "cost_spent": full_cost,
            "compute_saved": 0.0,
            "delta_mean": round(delta_mu, 6),
            "se": round(se_mu, 6),
            "lcb": round(lcb, 6),
            "exempt_adversarial": sentinel_slice.exempt_adversarial,
            "slice_size": k,
            "node": child_node,
        }


def compute_spectral_diagnostics(
    g_mat: Sequence[Sequence[float]],
    m_buckets: Optional[int] = None,
    train_score: float = 0.0,
    oos_score: float = 0.0,
    pass_mask: Optional[Sequence[int]] = None,
    use_hutchinson_above_d: int = 30,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    SLEEP-DAG-RSI v6 Regularized Spectral Diagnostics on G = (1/M) J^T J:
      1. Computes Effective Spectral Rank `r_eff(G) = (Tr G)^2 / Tr(G^2)` (using exact
         eigenvalues for d <= 30 or Hutchinson Rademacher trace estimator for d > 30).
      2. Computes Regularized Condition Number `kappa_eff(G) = lambda_max / max(lambda_support, 1e-12)`
         restricted to the observable rank support `min(M, d)` so trailing zero eigenvalues
         when `M < d` NEVER trigger false-positive `kappa -> inf` singularities!
      3. Outputs the dimensionless 4D Stall Signature:
         `sigma = (log10(kappa_eff), r_eff / d, train_oos_gap, fingerprint_entropy)`.
    """
    d = len(g_mat)
    if d == 0:
        return {
            "d": 0,
            "r_eff": 0.0,
            "r_eff_ratio": 0.0,
            "kappa_eff": 1.0,
            "log10_kappa_eff": 0.0,
            "stall_signature": (0.0, 0.0, 0.0, 0.0),
            "is_ill_conditioned": False,
        }

    evals, evecs = _jacobi_eigh_symmetric([list(row) for row in g_mat])
    pos_evals = [max(0.0, float(ev)) for ev in evals]
    lam_max = max(pos_evals[-1] if pos_evals else 0.0, 1e-12)

    if d > use_hutchinson_above_d:
        # Hutchinson stochastic Rademacher trace estimation O(m * d^2) with m=16
        rng_h = random.Random(seed)
        m_probes = 16
        tr_est = 0.0
        tr_sq_est = 0.0
        for _ in range(m_probes):
            v = [1.0 if rng_h.random() >= 0.5 else -1.0 for _ in range(d)]
            gv = [sum(float(g_mat[r][c]) * v[c] for c in range(d)) for r in range(d)]
            v_gv = sum(v[r] * gv[r] for r in range(d))
            gv_norm_sq = sum(x * x for x in gv)
            tr_est += v_gv / float(m_probes)
            tr_sq_est += gv_norm_sq / float(m_probes)
        r_eff = (tr_est * tr_est) / max(tr_sq_est, 1e-24)
    else:
        tr_g = sum(pos_evals)
        tr_g2 = sum(ev * ev for ev in pos_evals)
        r_eff = (tr_g * tr_g) / max(tr_g2, 1e-24)

    r_eff = max(1.0, min(float(d), r_eff))
    observable_rank = min(d, int(m_buckets)) if (m_buckets is not None and m_buckets > 0) else d
    support_idx = max(0, d - observable_rank)
    lam_support = max(pos_evals[support_idx], 1e-12)
    kappa_eff = lam_max / lam_support
    log10_kappa = math.log10(max(1.0, kappa_eff))

    # Shannon entropy of pass_mask bit distribution in [0, 1]
    fp_entropy = 0.0
    if pass_mask and len(pass_mask) > 0:
        p1 = sum(1 for b in pass_mask if b) / float(len(pass_mask))
        p0 = 1.0 - p1
        if 0.0 < p1 < 1.0:
            fp_entropy = -(p1 * math.log2(p1) + p0 * math.log2(p0))

    train_oos_gap = float(train_score) - float(oos_score)
    sig = (
        round(log10_kappa, 4),
        round(r_eff / float(d), 4),
        round(train_oos_gap, 4),
        round(fp_entropy, 4),
    )
    return {
        "d": d,
        "observable_rank": observable_rank,
        "eigenvalues": [round(ev, 10) for ev in pos_evals],
        "eigenvectors": evecs,
        "r_eff": round(r_eff, 6),
        "r_eff_ratio": round(r_eff / float(d), 6),
        "kappa_eff": round(kappa_eff, 4),
        "log10_kappa_eff": round(log10_kappa, 4),
        "train_oos_gap": round(train_oos_gap, 6),
        "fingerprint_entropy": round(fp_entropy, 6),
        "stall_signature": sig,
        "is_ill_conditioned": bool(kappa_eff >= 1e8),
    }


class FrameTransform:
    """
    Base class for Stage 2 Autonomous Coordinate / Representation Transformations (Phi -> Phi').
    Enforces two hard safety invariants:
      1. `inverse_projection(u_lifted) -> x_orig`: Maps any point in the lifted/rotated
         frame back to the immutable original task coordinates for outer & Blind Vault scoring.
      2. `condense_active_set(...)`: Collapses inactive auxiliary/slack coordinates after
         warmup to prevent monotonic dimension bloat.
    """

    name: str = "identity"
    warmup_grace_steps: int = 1

    def forward_project(self, x_orig: Sequence[float]) -> List[float]:
        return [float(v) for v in x_orig]

    def inverse_projection(self, u_lifted: Sequence[float]) -> List[float]:
        return [float(v) for v in u_lifted]


class EigenRotationTransform(FrameTransform):
    """
    Preconditioned Eigen-Basis Rotation (`Phi -> Phi_eigen`):
    Diagonalizes the ill-conditioned Gauss-Newton matrix `G = V Lambda V^T`, freezes
    sloppy/nullspace directions (`lambda_i < null_rel_tol * lambda_max`), and scales
    active stiff eigenvectors by `1 / sqrt(lambda_i)`.
    Turns a narrow valley with `kappa_eff > 10^8` into an isotropic sphere (`kappa = 1`),
    while `inverse_projection(y)` maps back to exact original coordinates `theta`.
    """

    name: str = "eigen_rotation"

    def __init__(
        self,
        theta_anchor: Sequence[float],
        g_mat: Sequence[Sequence[float]],
        spans: Optional[Sequence[float]] = None,
        null_rel_tol: float = 1e-10,
    ) -> None:
        self.theta_anchor = [float(x) for x in theta_anchor]
        self.d = len(self.theta_anchor)
        self.spans = [float(s) for s in spans] if spans is not None else [1.0] * self.d
        evals, evecs = _jacobi_eigh_symmetric([list(r) for r in g_mat])
        lam_max = max((max(0.0, float(ev)) for ev in evals), default=1.0)
        lam_max = max(lam_max, 1e-12)

        self.active_modes: List[int] = [
            m for m, ev in enumerate(evals) if max(0.0, float(ev)) >= null_rel_tol * lam_max
        ]
        if not self.active_modes:
            self.active_modes = list(range(self.d))

        self.eigenvalues = [max(1e-12, float(evals[m])) for m in self.active_modes]
        self.eigenvectors = [[float(evecs[m][coord]) for coord in range(self.d)] for m in self.active_modes]
        # Metric preconditioner scales step by 1 / sqrt(lambda_m) normalized to median scale
        inv_sqrt = [1.0 / math.sqrt(ev) for ev in self.eigenvalues]
        max_scale = max(inv_sqrt) if inv_sqrt else 1.0
        self.scales = [s / max_scale for s in inv_sqrt]
        self.warmup_grace_steps = 1

    @property
    def active_dim(self) -> int:
        return len(self.active_modes)

    def forward_project(self, x_orig: Sequence[float]) -> List[float]:
        dx_norm = [
            (float(x_orig[i]) - self.theta_anchor[i]) / max(1e-12, self.spans[i])
            for i in range(self.d)
        ]
        y: List[float] = []
        for m_idx, vec in enumerate(self.eigenvectors):
            proj = sum(dx_norm[i] * vec[i] for i in range(self.d))
            y.append(proj / max(1e-12, self.scales[m_idx]))
        return y

    def inverse_projection(self, u_lifted: Sequence[float]) -> List[float]:
        dx_norm = [0.0] * self.d
        for m_idx, vec in enumerate(self.eigenvectors):
            if m_idx < len(u_lifted):
                coeff = float(u_lifted[m_idx]) * self.scales[m_idx]
                for i in range(self.d):
                    dx_norm[i] += coeff * vec[i]
        return [
            self.theta_anchor[i] + dx_norm[i] * self.spans[i]
            for i in range(self.d)
        ]


class KktSlackTransform(FrameTransform):
    """
    Smooth KKT Slack-Variable Lifter with Active-Set Condensation (`Phi -> Phi_kkt`):
    Resolves non-smooth max-min tie traps `max_x min_{k in 1..K} f_k(x)` (such as
    equiareal triangle ties in Heilbronn) by lifting `x in R^d` to `u = (x, z) in R^{d + |A|}`
    where `z` are auxiliary slack targets for active constraints `A`.
    Includes `condense_active_set()` to prune inactive slack variables after 3 steps,
    preventing monotonic dimension bloat (`d -> d + K -> d + |A_tight|`).
    """

    name: str = "kkt_slack_lift"

    def __init__(
        self,
        base_dim: int,
        num_constraints: int,
        mu_penalty: float = 10.0,
        active_indices: Optional[Sequence[int]] = None,
    ) -> None:
        self.base_dim = int(base_dim)
        self.num_constraints = int(num_constraints)
        self.mu_penalty = float(mu_penalty)
        self.active_indices: List[int] = (
            [int(i) for i in active_indices]
            if active_indices is not None
            else list(range(num_constraints))
        )
        self.warmup_grace_steps = 1
        self.step_count = 0

    @property
    def lifted_dim(self) -> int:
        return self.base_dim + len(self.active_indices)

    def forward_project(
        self,
        x_orig: Sequence[float],
        initial_constraint_vals: Optional[Sequence[float]] = None,
    ) -> List[float]:
        base = [float(x_orig[i]) for i in range(self.base_dim)]
        if initial_constraint_vals is not None:
            slacks = [float(initial_constraint_vals[k]) for k in self.active_indices]
        else:
            slacks = [0.0 for _ in self.active_indices]
        return base + slacks

    def inverse_projection(self, u_lifted: Sequence[float]) -> List[float]:
        """Strictly strips auxiliary KKT slack variables and returns original x in R^d."""
        return [float(u_lifted[i]) for i in range(self.base_dim)]

    def evaluate_lifted_surrogate(
        self,
        u_lifted: Sequence[float],
        constraint_eval_fn: Callable[[Sequence[float]], Sequence[float]],
    ) -> float:
        x_orig = self.inverse_projection(u_lifted)
        f_vals = [float(v) for v in constraint_eval_fn(x_orig)]
        if not self.active_indices:
            return min(f_vals) if f_vals else 0.0
        slacks = [float(u_lifted[self.base_dim + idx]) for idx in range(len(self.active_indices))]
        z_mean = sum(slacks) / float(len(slacks))
        penalty = 0.0
        for idx, k_con in enumerate(self.active_indices):
            viol = max(0.0, slacks[idx] - f_vals[k_con])
            tie_dev = slacks[idx] - z_mean
            penalty += self.mu_penalty * (viol * viol) + 0.5 * self.mu_penalty * (tie_dev * tie_dev)
        return z_mean - penalty

    def condense_active_set(
        self,
        u_lifted: Sequence[float],
        constraint_vals: Sequence[float],
        sigma_mult: float = 1.5,
    ) -> Dict[str, Any]:
        """
        Active-Set Condensation (Unlifting): drops slack variables for constraints
        that sit comfortably above the minimum active tie manifold (`f_k - f_min > sigma_mult * std`),
        shrinking the state dimension back to the tight active tie set.
        """
        self.step_count += 1
        x_orig = self.inverse_projection(u_lifted)
        f_vals = [float(v) for v in constraint_vals]
        if not self.active_indices or len(self.active_indices) <= 1:
            return {
                "condensed_u": list(u_lifted),
                "active_slack_count": len(self.active_indices),
                "pruned_slack_count": 0,
            }

        active_f = [f_vals[k] for k in self.active_indices]
        f_min = min(active_f)
        f_mean = sum(active_f) / float(len(active_f))
        f_std = math.sqrt(
            max(1e-12, sum((v - f_mean) ** 2 for v in active_f) / float(len(active_f)))
        )
        thresh = f_min + sigma_mult * f_std

        kept_indices: List[int] = []
        kept_slacks: List[float] = []
        for pos, k_con in enumerate(self.active_indices):
            if f_vals[k_con] <= thresh + 1e-9:
                kept_indices.append(k_con)
                slack_val = (
                    float(u_lifted[self.base_dim + pos])
                    if (self.base_dim + pos) < len(u_lifted)
                    else f_vals[k_con]
                )
                kept_slacks.append(slack_val)

        if not kept_indices:
            min_k = min(self.active_indices, key=lambda k: f_vals[k])
            kept_indices = [min_k]
            kept_slacks = [f_vals[min_k]]

        pruned_count = len(self.active_indices) - len(kept_indices)
        self.active_indices = kept_indices
        condensed_u = x_orig + kept_slacks
        return {
            "condensed_u": condensed_u,
            "active_indices": list(self.active_indices),
            "active_slack_count": len(self.active_indices),
            "pruned_slack_count": pruned_count,
            "lifted_dim": self.lifted_dim,
        }


class DyadicHomotopyTransform(FrameTransform):
    """
    Multi-Scale Dyadic Coarsening & Prolongation (`Phi -> Phi_dyadic`):
    Maps a fine discretization grid of dimension `P` to a coarse grid of dimension
    `P_coarse = max(4, P // 2^level)` via block averaging, and prolongates back to
    exact dimension `P` via piecewise-linear interpolation in `inverse_projection`.
    """

    name: str = "dyadic_homotopy"

    def __init__(self, fine_dim: int, level: int = 2) -> None:
        self.fine_dim = int(fine_dim)
        self.level = max(1, int(level))
        self.coarse_dim = max(4, self.fine_dim // (2 ** self.level))
        self.warmup_grace_steps = 1

    def forward_project(self, x_orig: Sequence[float]) -> List[float]:
        n = len(x_orig)
        if n <= self.coarse_dim:
            return [float(v) for v in x_orig]
        coarse: List[float] = []
        for i in range(self.coarse_dim):
            start = int(round(i * n / float(self.coarse_dim)))
            end = max(start + 1, int(round((i + 1) * n / float(self.coarse_dim))))
            chunk = [float(x_orig[j]) for j in range(start, min(n, end))]
            coarse.append(sum(chunk) / float(len(chunk)))
        return coarse

    def inverse_projection(self, u_lifted: Sequence[float]) -> List[float]:
        c_len = len(u_lifted)
        if c_len == 0:
            return [0.0] * self.fine_dim
        if c_len == 1 or self.fine_dim == 1:
            return [float(u_lifted[0])] * self.fine_dim
        fine: List[float] = []
        for i in range(self.fine_dim):
            pos = i * (c_len - 1) / float(self.fine_dim - 1)
            idx_lo = int(math.floor(pos))
            idx_hi = min(c_len - 1, idx_lo + 1)
            frac = pos - idx_lo
            val = (1.0 - frac) * float(u_lifted[idx_lo]) + frac * float(u_lifted[idx_hi])
            fine.append(val)
        return fine


class SpectralMetaAtlas:
    """
    Cross-Task Spectral Operator Atlas (`meta_operator_atlas`):
    Indexes verified structural transformations (`eigen_rotation`, `kkt_slack_lift`,
    `dyadic_homotopy`) by dimensionless stall signatures and task geometry, with
    automatic provenance quarantine if a transferred operator fails to produce
    positive gain within `max_trial_probes = 2` evaluations.
    """

    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = [
            {
                "operator": "kkt_slack_lift",
                "task_family": "geometric_extremal",
                "min_log10_kappa": 6.0,
                "requires_min_tie": True,
                "description": "Lifts non-smooth min-tie bottleneck into smooth KKT slack variables with active-set condensation.",
            },
            {
                "operator": "dyadic_homotopy",
                "task_family": "fourier_spectral",
                "min_dim": 64,
                "description": "Coarsens high-resolution grid by 2^k and prolongates back via linear spline homotopy.",
            },
            {
                "operator": "eigen_rotation",
                "task_family": "general",
                "min_log10_kappa": 6.0,
                "description": "Preconditioned rotation into Gauss-Newton eigenvectors V with nullspace freezing.",
            },
        ]
        self.trial_counts: Dict[Tuple[str, str], int] = {}
        self.positive_credits: Dict[Tuple[str, str], int] = {}
        self.quarantined: Set[Tuple[str, str]] = set()

    def record_outcome(self, task_name: str, operator_name: str, gained_score: bool) -> bool:
        key = (task_name, operator_name)
        self.trial_counts[key] = self.trial_counts.get(key, 0) + 1
        if gained_score:
            self.positive_credits[key] = self.positive_credits.get(key, 0) + 1
        elif self.trial_counts[key] >= 2 and self.positive_credits.get(key, 0) == 0:
            self.quarantined.add(key)
        return key not in self.quarantined

    def select_operator(
        self,
        task_name: str,
        diagnostics: Dict[str, Any],
        has_min_tie_structure: bool = False,
        consecutive_oos_failures: int = 0,
    ) -> Optional[str]:
        log10_k = float(diagnostics.get("log10_kappa_eff", 0.0))
        d = int(diagnostics.get("d", 0))
        stalled = bool(log10_k >= 6.0 or consecutive_oos_failures >= 4)
        if not stalled:
            return None

        candidates: List[str] = []
        if has_min_tie_structure:
            candidates.append("kkt_slack_lift")
        if d >= 64:
            candidates.append("dyadic_homotopy")
        candidates.append("eigen_rotation")

        for op in candidates:
            if (task_name, op) not in self.quarantined:
                return op
        return None


class SpectralFrameLifter:
    """
    Stage 2 Coordinator: inspects Gauss-Newton matrix `G = (1/M) J^T J` and DAG OOS
    trajectory, queries `SpectralMetaAtlas`, and instantiates the matching `FrameTransform`
    with a mandatory inverse projection `Phi^-1` and 1-step warmup grace budget.
    """

    def __init__(self, atlas: Optional[SpectralMetaAtlas] = None) -> None:
        self.atlas = atlas or SpectralMetaAtlas()

    def detect_and_lift(
        self,
        task_name: str,
        theta_anchor: Sequence[float],
        g_mat: Sequence[Sequence[float]],
        m_buckets: Optional[int] = None,
        spans: Optional[Sequence[float]] = None,
        train_score: float = 0.0,
        oos_score: float = 0.0,
        pass_mask: Optional[Sequence[int]] = None,
        has_min_tie_structure: bool = False,
        num_tie_constraints: int = 0,
        consecutive_oos_failures: int = 0,
    ) -> Dict[str, Any]:
        diag = compute_spectral_diagnostics(
            g_mat=g_mat,
            m_buckets=m_buckets,
            train_score=train_score,
            oos_score=oos_score,
            pass_mask=pass_mask,
        )
        op_name = self.atlas.select_operator(
            task_name=task_name,
            diagnostics=diag,
            has_min_tie_structure=has_min_tie_structure,
            consecutive_oos_failures=consecutive_oos_failures,
        )
        if op_name is None:
            return {
                "triggered": False,
                "operator": None,
                "diagnostics": diag,
                "transform": FrameTransform(),
                "warmup_grace_steps": 0,
            }

        if op_name == "kkt_slack_lift" and num_tie_constraints > 0:
            transform: FrameTransform = KktSlackTransform(
                base_dim=len(theta_anchor),
                num_constraints=num_tie_constraints,
            )
        elif op_name == "dyadic_homotopy":
            transform = DyadicHomotopyTransform(fine_dim=len(theta_anchor), level=2)
        else:
            transform = EigenRotationTransform(
                theta_anchor=theta_anchor,
                g_mat=g_mat,
                spans=spans,
            )

        return {
            "triggered": True,
            "operator": op_name,
            "diagnostics": diag,
            "transform": transform,
            "warmup_grace_steps": transform.warmup_grace_steps,
        }


# ---------------------------------------------------------------------------
# 17. SLEEP-DAG-RSI v7: Autonomous AST Code Splicer (`AstCodeSplicer`) &
#     Active Boundary Counterexample Synthesizer (`ActiveBoundarySynthesizer`)
# ---------------------------------------------------------------------------


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
    ) -> Optional[Dict[str, Any]]:
        """
        Synthesizes an Out-of-Sample input-space routing predicate (`if <pred>: return p_win(x) else: return p_alt(x)`)
        by fitting a coordinate threshold or linear half-space boundary separating inputs where
        Parent 0 outperforms Parent 1 from inputs where Parent 1 outperforms Parent 0.
        Eliminates the need for an oracle `bucket_idx` at test time.
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

        best_gain = -float("inf")
        best_expr: Optional[str] = None
        best_p0_when_true = True
        best_acc = 0.0

        for c in range(d_in):
            vals = sorted({rows[b][c] for b in range(n_b)})
            if len(vals) < 2:
                continue
            thresholds = [0.5 * (vals[k] + vals[k + 1]) for k in range(len(vals) - 1)]
            for tau in thresholds:
                for p0_on_ge in (True, False):
                    routed_sum = 0.0
                    correct = 0
                    informative = 0
                    for b in range(n_b):
                        cond = rows[b][c] >= tau
                        use_p0 = cond if p0_on_ge else (not cond)
                        routed_sum += m0[b] if use_p0 else m1[b]
                        if abs(m0[b] - m1[b]) > 1e-9:
                            informative += 1
                            if (use_p0 and m0[b] > m1[b]) or ((not use_p0) and m1[b] > m0[b]):
                                correct += 1
                    routed_mean = routed_sum / float(n_b)
                    acc = correct / float(max(1, informative))
                    if routed_mean > best_gain + 1e-9:
                        best_gain = routed_mean
                        best_p0_when_true = p0_on_ge
                        best_acc = acc
                        lhs = arg_name if is_scalar else f"{arg_name}[{c}]"
                        best_expr = f"{lhs} >= {round(tau, 6)}"

        max_single_parent = max(sum(m0[:n_b]) / float(n_b), sum(m1[:n_b]) / float(n_b))
        if best_expr is None or best_gain <= max_single_parent + 1e-6:
            return None
        return {
            "predicate_expr": best_expr,
            "p0_when_true": best_p0_when_true,
            "routed_mean_score": round(best_gain, 6),
            "routing_accuracy": round(best_acc, 6),
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
                        first_arg, bucket_inputs, parent_bucket_margins
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
    ) -> Dict[str, Any]:
        """
        Post-Splice Epistasis Execution Gate:
        Even when syntactic `ast_collision_iou == 0`, two merged statements can interact
        nonlinearly via shared data-flow variables (semantic epistasis). This gate
        compiles `spliced_code`, runs `eval_score_fn(fn)` on the sentinel/validation harness,
        and verifies `spliced_score >= max(parent_scores) - tol`.
        """
        ns: Dict[str, Any] = {"math": math}
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
    ) -> None:
        self.trust_radius_sigma = float(trust_radius_sigma)
        self.ascent_steps = max(1, int(ascent_steps))
        self.step_lr = float(step_lr)
        self.fd_eps = float(fd_eps)
        self.probe_zero_boundary = bool(probe_zero_boundary)

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
                for c in range(d):
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
                for c in range(d):
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


# ---------------------------------------------------------------------------
# 18. SLEEP-DAG-RSI v5, v6 & v7 Parametric Continual Learning Exports
# ---------------------------------------------------------------------------
try:
    from sleep_dag_rsi import (  # type: ignore
        AstPolicySafetyGate,
        CounterfactualDreamingEngine,
        DAGKnowledgeSeedingConsolidator,
        DeltaRuleFastMemory,
        ExpandableMoEMLP,
        ExpertFingerprintMeta,
        LibraryPrimitive,
        LoRALinearWrapper,
        LowRankExpert,
        NeuroSymbolicDistiller,
        SleepDAGConfig,
        SleepDAGRSIOrchestrator,
        SleepLanguageModel,
        SymbolicLibrary,
        UniversalNeuralSleepAdapter,
        attach_sleep_dag_adapters,
        compute_r_abs,
        compute_r_sem_numpy,
        cross_entropy_loss_and_grad,
        generalized_jsd_loss_and_grad,
        levenshtein_distance,
        numpy_lasso_coordinate_descent,
        softmax_np,
    )
except ImportError:
    pass



"""
Core DAG discovery data structures, error-fingerprint metrics, Tukey CF caps,
and append-only pool persistence for Dream-RSI / Lean-DAG-RSI.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


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
    #: Continuous per-bucket margins M_{v, b} (>0 means solved, <=0 means failed).
    bucket_scores: List[float] = field(default_factory=list)
    #: Optional continuous tunable parameters dict/list for Neuro-Symbolic split & RRSI pruner.
    tunable_params: Dict[str, Any] = field(default_factory=dict)
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


def _collect_positive_gains(
    tree: DiscoveryTree,
    problem: Optional[Problem] = None,
) -> List[float]:
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
    return pos_gains


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
    pos_gains = _collect_positive_gains(tree, problem)
    if not pos_gains:
        return 0.0
    pos_gains.sort()
    med = _percentile_linear(pos_gains, 0.50)
    q25 = _percentile_linear(pos_gains, 0.25)
    q75 = _percentile_linear(pos_gains, 0.75)
    return med + tau_cap * (q75 - q25)


def compute_counterfactual_step(
    ds: float,
    node_score: float,
    root_score: float,
    pos_gains: Sequence[float],
    tukey_cap: Optional[float] = None,
    tau_cap: float = 1.5,
) -> float:
    """
    Shift-Invariant Counterfactual Step Cap with Smooth Fallback Transition (Eq. 4):
      - Shift-invariant fallback scale:
          S_Delta = max(mean(Delta^+), |s(u) - s(root)|, 1e-6)
          C_fallback(Delta_u; S_Delta) = Delta_u / (1.0 + Delta_u / S_Delta)
        Note: S_Delta depends purely on score differences, making C_fallback strictly
        invariant to arbitrary constant score translations s -> s + C.
      - Smooth blend weight:
          alpha_cf = min(1.0, max(0.0, (|Delta^+| - 1) / 3.0))
        Eliminates the step discontinuity at |Delta^+| = 4 between C_fallback and C_Tukey.
    """
    if ds <= 0.0:
        return 0.0
    gains = [float(g) for g in pos_gains if float(g) > 0.0]
    n_pos = len(gains)
    mean_pos = (sum(gains) / float(n_pos)) if n_pos > 0 else 0.0
    s_delta = max(mean_pos, abs(float(node_score) - float(root_score)), 1e-6)
    c_fallback = float(ds) / (1.0 + float(ds) / s_delta)
    if n_pos <= 1:
        return c_fallback

    if tukey_cap is None:
        s_gains = sorted(gains)
        med = _percentile_linear(s_gains, 0.50)
        q25 = _percentile_linear(s_gains, 0.25)
        q75 = _percentile_linear(s_gains, 0.75)
        t_cap = med + tau_cap * (q75 - q25)
    else:
        t_cap = float(tukey_cap)

    c_tukey = min(float(ds), t_cap)
    alpha_cf = min(1.0, max(0.0, float(n_pos - 1) / 3.0))
    return (1.0 - alpha_cf) * c_fallback + alpha_cf * c_tukey


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

__all__ = [
    "Problem",
    "Node",
    "GridPlan",
    "GridPlanningContext",
    "BranchTrajectory",
    "DiscoveryTree",
    "default_pool_dir",
    "save_tree",
    "load_pool",
    "compute_tukey_cap",
    "compute_counterfactual_step",
    "compute_error_fingerprint",
]

"""
Dream-RSI (`SLEEP-DAG-RSI v8` & `Lean-DAG-RSI`) command line.

    python dreamrsi.py demo                 end-to-end loop on synthetic history
    python dreamrsi.py scaffold --dir DIR --task NAME [--mode code|ml|trading|math]
                                            generate a ready-to-run TaskCartridge harness
    python dreamrsi.py state  --task NAME   print compact SKILL.state + backward subgoals
    python dreamrsi.py init   --task NAME   create loop state for a real task
    python dreamrsi.py round  --task NAME   propose -> replay -> select -> persist
    python dreamrsi.py prompt --task NAME   render the policy-improvement prompt
    python dreamrsi.py apply  --task NAME --file proposal.json
    python dreamrsi.py status --task NAME   ranking, history, current policy
    python dreamrsi.py trees  --task NAME   proof that a tree was recorded
    python dreamrsi.py ideate --task NAME   synthesize new hybrid/DAG/neuro-symbolic ideas
    python dreamrsi.py sleep  --demo|--task run SLEEP-DAG-RSI v8 Wake->NREM->REM->2x2 cycle
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import textwrap

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from engine import (  # noqa: E402
    DiscoveryTree,
    DynamicPortfolio,
    MutableSkillState,
    Node,
    Problem,
    decompose_dead_buckets,
    default_pool_dir,
    ideate_over_history,
    load_pool,
    replay_over_history,
    save_tree,
    select_policy,
    sweep_beta,
)
from loop import (  # noqa: E402
    initial_state,
    load_state,
    parse_proposal,
    render_prompt,
    run_round,
    save_state,
)


# ---------------------------------------------------------------------------
# demo: synthetic history so the loop can be exercised without a real project
# ---------------------------------------------------------------------------


def synth_trees(task: str = "demo-domain") -> list:
    """
    Recorded explorations big enough that policy choice actually matters.

    A tree that replay exhausts in one round cannot discriminate between
    policies, so each branch here is deep and the widths differ between the two
    trees. That is what makes the ranking below informative rather than uniform.
    """

    def branch(tree, name, first_gain, steps, branch_id, fail=None):
        score = tree.baseline_score + first_gain
        parent = "root"
        for step in range(steps):
            nid = f"{name}{step}"
            if step == 0 and fail:
                tree.add(
                    Node(
                        id=nid,
                        parent_id=parent,
                        branch=branch_id,
                        attempt=step + 1,
                        score=0.0,
                        fail_class=fail[0],
                        error=fail[1],
                    )
                )
            else:
                tree.add(
                    Node(
                        id=nid,
                        parent_id=parent,
                        branch=branch_id,
                        attempt=step + 1,
                        score=round(score, 2),
                    )
                )
                score += first_gain * (0.55**step)
            parent = nid

    a = DiscoveryTree(task=task, baseline_score=100.0, max_parallelism=3)
    a.add(Node(id="root", parent_id=None, branch=-1, attempt=0, score=100.0))
    branch(a, "d", 22.0, 6, 0)
    branch(a, "r", 14.0, 4, 1, fail=("oom", "out of memory"))
    branch(a, "z", 4.0, 3, 2)
    a.add(
        Node(
            id="x",
            parent_id="root",
            branch=3,
            attempt=1,
            score=92.0,
            fail_class="algorithmic_failure",
        )
    )

    b = DiscoveryTree(task=task, baseline_score=50.0, max_parallelism=3)
    b.add(Node(id="root", parent_id=None, branch=-1, attempt=0, score=50.0))
    branch(b, "m", 9.0, 7, 0)
    branch(b, "n", 18.0, 3, 1)
    branch(b, "p", 12.0, 5, 2, fail=("syntax_error", "bad token"))
    return [a, b]


def cmd_demo(args: argparse.Namespace) -> int:
    pool = args.pool or tempfile.mkdtemp(prefix="dreamrsi_demo_")
    task = "demo-domain"
    for tree in synth_trees(task):
        save_tree(tree, pool_dir=pool)

    trees = load_pool(task=task, pool_dir=pool)
    print(f"recorded trees: {len(trees)}   pool: {pool}")
    print()

    problem = Problem()

    print("round 0 — select among a fixed candidate set")
    inc = DynamicPortfolio(beta=0.3)
    candidates = {
        "incumbent": inc,
        "wide": DynamicPortfolio(beta=0.9),
        "lean": DynamicPortfolio(beta=0.1),
        "deep": DynamicPortfolio(beta=0.6),
    }
    sel = select_policy(candidates, trees, problem)
    for name, score in sel["ranking"]:
        print(f"   {name:<12} mean_v={score}")
    print(f"   -> winner: {sel['best']}")
    print()

    print("round 0b — offline beta sweep")
    for beta, score in sweep_beta(trees, problem):
        print(f"   beta={beta:<5} mean_v={score}")
    print()

    print("closed loop — 3 rounds, incumbent carried forward")
    state = initial_state(task)
    save_state(task, state, pool)
    scores = []
    for _ in range(3):
        report = run_round(task, problem, pool_dir=pool)
        scores.append(report["winner_mean_v"])
        print(
            f"   round {report['round']}: incumbent={report['incumbent_mean_v']} "
            f"winner={report['winner']} -> {report['winner_mean_v']} "
            f"improved={report['improved']}  ({report['candidates']} candidates)"
        )
    print()

    monotone = all(scores[i] <= scores[i + 1] + 1e-12 for i in range(len(scores) - 1))
    print(f"monotone across rounds: {monotone}   trajectory: {scores}")
    print(
        f"final policy: {json.dumps(load_state(task, pool)['current'], ensure_ascii=False)}"
    )
    print()

    print("prompt rendering (authored-proposal path)")
    text = render_prompt(task, pool_dir=pool, problem=problem)
    filled = "{{" not in text
    print(f"   characters: {len(text)}   all placeholders substituted: {filled}")
    head = [ln for ln in text.splitlines() if ln.startswith("| ")][:6]
    for ln in head:
        print("   " + ln)
    print()

    if args.keep:
        print(f"pool kept at: {pool}")
    else:
        import shutil

        shutil.rmtree(pool, ignore_errors=True)
        print("temp pool removed (pass --keep to inspect it)")

    ok = monotone and filled
    print()
    print("DEMO RESULT:", "ok" if ok else "PROBLEM")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# real-task commands
# ---------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    state = load_state(args.task, args.pool)
    if state:
        print(f"state already exists for {args.task!r}; leaving it untouched")
        return 0
    save_state(args.task, initial_state(args.task), args.pool)
    print(f"initialised loop state for {args.task!r} at {args.pool or 'default pool'}")
    return 0


def cmd_round(args: argparse.Namespace) -> int:
    report = run_round(args.task, pool_dir=args.pool)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if "error" not in report else 2


def cmd_prompt(args: argparse.Namespace) -> int:
    text = render_prompt(args.task, pool_dir=args.pool)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"prompt written to {args.out} ({len(text)} chars)")
    else:
        print(text)
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    with open(args.file, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    specs = parse_proposal(payload)
    if not specs:
        print(
            "no usable candidates in the proposal "
            '(expected {"candidates": [{"kind": ...}]})'
        )
        return 2
    print(f"candidates parsed: {', '.join(specs)}")
    report = run_round(args.task, pool_dir=args.pool, provided_specs=specs)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if "error" not in report else 2


def cmd_trees(args: argparse.Namespace) -> int:
    """
    Show what is actually recorded for a task. This is the closing proof for SKILL.md
    section 0: if it prints no tree, the skill was not used, whatever else was written.
    """
    pool = args.pool or default_pool_dir()
    trees = load_pool(task=args.task, pool_dir=args.pool)
    print(f"pool: {pool}")
    print(f"task: {args.task}   trees: {len(trees)}")
    if not trees:
        print()
        print("NO TREE RECORDED for this task.")
        print("A narrative produced without a tree was NOT produced by this skill.")
        return 2
    for t in trees:
        nodes = list(t.nodes.values())
        branches = sorted(
            {n.branch for n in nodes if n.branch is not None and n.branch >= 0}
        )
        print()
        print(
            f"  {t.created}  nodes={len(nodes)}  branches={len(branches)}  "
            f"baseline={t.baseline_score}"
        )
        for n in sorted(nodes, key=lambda x: -x.score)[:3]:
            print(f"    {n.id:<18} score={n.score:<14} class={n.fail_class}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    state = load_state(args.task, args.pool)
    if not state:
        print(f"no loop state for {args.task!r}")
        return 2
    trees = load_pool(task=args.task, pool_dir=args.pool)
    print(f"task: {args.task}   round: {state.get('round')}   trees: {len(trees)}")
    print(f"current policy: {json.dumps(state['current'], ensure_ascii=False)}")
    print()
    if trees:
        from engine import build_policy

        inc = build_policy(state["current"])
        res = replay_over_history(
            trees,
            inc,
            Problem(),
            float(state.get("beta1", 0.01)),
            float(state.get("beta2", 0.05)),
        )
        print(f"current mean_v on history: {res['mean_v']:.6f}")
        print("offline beta sweep:")
        for beta, score in sweep_beta(trees, Problem())[:6]:
            print(f"   beta={beta:<5} {score}")
    print()
    print("history:")
    for h in state.get("history", []):
        print(
            f"   round {h['round']}: {h['incumbent_mean_v']} -> {h['winner_mean_v']} "
            f"via {h['winner']}  improved={h['improved']}"
        )
    return 0


def cmd_ideate(args: argparse.Namespace) -> int:
    trees = load_pool(task=args.task, pool_dir=args.pool)
    if not trees:
        print(f"no trees recorded for {args.task!r}")
        return 2
    ideas = ideate_over_history(trees, Problem())
    print(json.dumps(ideas, ensure_ascii=False, indent=2))
    return 0


def cmd_state(args: argparse.Namespace) -> int:
    """Print bounded MutableSkillState (SKILL.state) + dead-bucket subgoals for subagent handoff."""
    pool = args.pool or default_pool_dir()
    trees = load_pool(task=args.task, pool_dir=pool)
    if not trees:
        print(f"no trees recorded for {args.task!r}")
        return 2
    problem = Problem()
    st = MutableSkillState(task=args.task)
    for t in trees:
        st.commit_cycle(t, problem)
    payload = {
        "task": args.task,
        "skill_state": st.to_dict(),
        "backward_subgoals": decompose_dead_buckets(trees, problem),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _render_scaffold_script(task_name: str, mode: str, scripts_dir: str) -> str:
    safe_scripts = scripts_dir.replace("\\", "/")
    header_risk = ""
    if mode == "trading":
        header_risk = (
            "# RISK WARNING: Algorithmic trading involves substantial risk of capital loss.\n"
            "# Past backtest or Blind Vault metrics do not guarantee future live profitability.\n"
        )

    common_header = f'''"""
Auto-generated SLEEP-DAG-RSI v8 Discovery Harness (`mode={mode}`) for task `{task_name}`.
{header_risk.rstrip()}
Run directly:
    python {{script_name}}
Verify recorded tree:
    python "{safe_scripts}/dreamrsi.py" trees --task {task_name}
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

PLUGIN_SCRIPTS = os.path.expanduser(
    "~/.gemini/config/plugins/autonomous-research/skills/dream-rsi/scripts"
)
FALLBACK_SCRIPTS = "{safe_scripts}"
for _p in (PLUGIN_SCRIPTS, FALLBACK_SCRIPTS):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import engine as dream_engine  # noqa: E402
from cartridge import AutoDreamRunner, BranchCandidate, TaskCartridge  # noqa: E402
'''

    if mode == "trading":
        body = f'''
ROUND_TRIP_FEE_BPS = 5.0  # Realistic taker fee + slippage in basis points


def _build_synthetic_ohlc(seed: int = 42, n_bars: int = 400) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    close_ts = np.arange(1, n_bars + 1, dtype=np.int64) * 300  # 5m bars
    htf_close_ts = ((close_ts - 300) // 3600) * 3600           # strictly closed 1h <= ltf_close
    regime = np.where((np.arange(n_bars) // 40) % 2 == 0, 1.0, -1.0)
    z_sig = rng.normal(0.0, 1.0, size=n_bars)
    fwd_bps = 22.0 * regime * np.tanh(z_sig) + rng.normal(0.0, 4.0, size=n_bars)
    return {{"close_ts": close_ts, "htf_close_ts": htf_close_ts, "regime": regime, "z_sig": z_sig, "fwd_bps": fwd_bps}}


def build_cartridge() -> TaskCartridge:
    data = _build_synthetic_ohlc()
    # Zero Look-Ahead invariant check:
    leak_count = int(np.sum(data["htf_close_ts"] > data["close_ts"]))
    assert leak_count == 0, f"Look-ahead leak detected: {{leak_count}} bars!"

    split = int(0.8 * len(data["close_ts"]))  # 80% Dev, 20% untouched chronological Blind Vault
    dev = {{k: v[:split] for k, v in data.items()}}
    vault = {{k: v[split:] for k, v in data.items()}}

    def eval_slice(fn: Callable[[Dict[str, np.ndarray]], np.ndarray], d: Dict[str, np.ndarray]) -> Tuple[float, List[int], List[float]]:
        pos = np.asarray(fn(d), dtype=np.float64)
        net_bps = pos * d["fwd_bps"] - np.abs(pos) * ROUND_TRIP_FEE_BPS
        # 8 chronological Walk-Forward buckets
        chunks = np.array_split(net_bps, 8)
        margins = [float(np.mean(c)) for c in chunks]
        mask = [1 if m > 0.0 else 0 for m in margins]
        return float(np.mean(net_bps)), mask, margins

    def verify_and_score(fn: Callable[[Dict[str, np.ndarray]], np.ndarray]) -> Tuple[bool, float, str, Dict[str, Any]]:
        mean_bps, mask, margins = eval_slice(fn, dev)
        return True, mean_bps, "ok", {{"pass_mask": mask, "bucket_scores": margins, "summary": f"Dev net_bps={{mean_bps:.3f}}"}}

    def blind_vault_eval(fn: Callable[[Dict[str, np.ndarray]], np.ndarray]) -> Dict[str, Any]:
        v_bps, v_mask, _ = eval_slice(fn, vault)
        return {{"vault_net_bps": round(v_bps, 4), "vault_positive_buckets": sum(v_mask), "risk_warning": "Capital at risk; historical Blind Vault != future returns."}}

    base_fn = lambda d: np.where(d["z_sig"] > 0.5, 1.0, 0.0)
    b0 = BranchCandidate(
        id="b0_regime_pullback",
        branch=0,
        description="Long regime-aligned positive signal",
        fn=lambda d: np.where((d["regime"] > 0) & (d["z_sig"] > 0.6), 1.0, 0.0),
    )
    b1 = BranchCandidate(
        id="b1_regime_short",
        branch=1,
        description="Short negative-regime breakdown",
        fn=lambda d: np.where((d["regime"] < 0) & (d["z_sig"] < -0.6), -1.0, 0.0),
    )

    def xover(u: BranchCandidate, v: BranchCandidate) -> BranchCandidate:
        def _make(theta: np.ndarray) -> Callable[[Dict[str, np.ndarray]], np.ndarray]:
            th_hi, th_lo = float(theta[0]), float(theta[1])
            return lambda d: np.where((d["regime"] > 0) & (d["z_sig"] > th_hi), 1.0, np.where((d["regime"] < 0) & (d["z_sig"] < -th_lo), -1.0, 0.0))

        p0 = np.array([0.6, 0.6], dtype=np.float64)
        return BranchCandidate(
            id="xover_dual_regime",
            branch=2,
            parent_id=u.id,
            description="Dual-regime net-of-fee crossover",
            fn=_make(p0),
            tunable_params=p0,
            tunable_bounds=[(0.3, 1.2), (0.3, 1.2)],
            param_names=["long_z_thresh", "short_z_thresh"],
            param_factory=_make,
        )

    return TaskCartridge(
        task_name="{task_name}",
        description="Zero-lookahead trading discovery harness with Sealed Blind Vault",
        baseline_fn=base_fn,
        verify_and_score=verify_and_score,
        initial_branches=[b0, b1],
        crossover_synthesizer=xover,
        blind_vault_evaluator=blind_vault_eval,
        noise_probe_runs=2,
    )
'''
    elif mode == "math":
        safe_thm = re.sub(r"[^a-zA-Z0-9_]", "_", task_name)
        body = f'''
from fractions import Fraction


def build_cartridge() -> TaskCartridge:
    # Toy extremal Heilbronn-style simplex lower-bound cartridge with exact Fraction + Lean 4 certificate
    scale = 10**8

    def exact_norm_min_area(pts: Sequence[Tuple[float, float]]) -> Tuple[Fraction, List[Tuple[int, int]]]:
        int_pts = [(int(round(x * scale)), int(round(y * scale))) for x, y in pts]
        q = [(Fraction(ix, scale), Fraction(iy, scale)) for ix, iy in int_pts]
        for x, y in q:
            if x < 0 or y < 0 or x + y > 1:
                return Fraction(-1, 1), int_pts
        best: Optional[Fraction] = None
        n = len(q)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    norm_area = abs(q[i][0] * (q[j][1] - q[k][1]) + q[j][0] * (q[k][1] - q[i][1]) + q[k][0] * (q[i][1] - q[j][1]))
                    if best is None or norm_area < best:
                        best = norm_area
        return (best if best is not None else Fraction(0, 1)), int_pts

    def verify_and_score(fn: Callable[[], List[Tuple[float, float]]]) -> Tuple[bool, float, str, Dict[str, Any]]:
        pts = fn()
        val_frac, int_pts = exact_norm_min_area(pts)
        if val_frac < 0:
            return False, 0.0, "domain_violation", {{"error": "Point outside unit simplex"}}
        lean_rep = dream_engine.export_lean4_certificate(
            theorem_name="cert_{safe_thm}",
            points_num_den=int_pts,
            scale_denom=scale,
            bound_num=val_frac.numerator,
            bound_den=val_frac.denominator,
        )
        assert lean_rep["verified_exact_int"] and not lean_rep["has_sorry"]
        sc = float(val_frac) * 1000.0
        mask = [1 if sc > t else 0 for t in (150.0, 220.0, 280.0, 320.0)]
        return True, sc, "ok", {{"pass_mask": mask, "exact_fraction": str(val_frac), "lean4_verified": lean_rep["verified_exact_int"], "lean4_sorry_free": not lean_rep["has_sorry"]}}

    base_fn = lambda: [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.20, 0.20)]
    b0 = BranchCandidate(
        id="b0_centroid_shift",
        branch=0,
        description="Symmetric interior centroid (Delta_4 = 1/3)",
        fn=lambda: [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0 / 3.0, 1.0 / 3.0)],
    )
    b1 = BranchCandidate(
        id="b1_boundary_collinear_trap",
        branch=1,
        description="Boundary midpoint collinearity trap (Delta_4 = 0)",
        fn=lambda: [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.5)],
    )
    return TaskCartridge(
        task_name="{task_name}",
        description="Exact-rational & Lean 4 certified extremal geometry cartridge",
        baseline_fn=base_fn,
        verify_and_score=verify_and_score,
        initial_branches=[b0, b1],
    )
'''
    elif mode == "ml":
        body = f'''
def build_cartridge() -> TaskCartridge:
    rng = np.random.default_rng(123)
    X = rng.normal(0.0, 1.0, size=(320, 4))
    y = 1.8 * (X[:, 0] * X[:, 1]) - 1.2 * np.tanh(X[:, 2]) + 0.05 * rng.normal(size=320)
    X_dev, y_dev = X[:240], y[:240]
    X_vault, y_vault = X[240:], y[240:]

    def eval_r2_folds(fn: Callable[[np.ndarray], np.ndarray], X_in: np.ndarray, y_in: np.ndarray) -> Tuple[float, List[int], List[float]]:
        pred = np.asarray(fn(X_in), dtype=np.float64)
        folds_y = np.array_split(y_in, 6)
        folds_p = np.array_split(pred, 6)
        r2s: List[float] = []
        for fy, fp in zip(folds_y, folds_p):
            ss_res = float(np.sum((fy - fp) ** 2))
            ss_tot = float(np.sum((fy - np.mean(fy)) ** 2)) + 1e-12
            r2s.append(1.0 - ss_res / ss_tot)
        mask = [1 if r > 0.70 else 0 for r in r2s]
        return float(np.mean(r2s)) * 100.0, mask, r2s

    def verify_and_score(fn: Callable[[np.ndarray], np.ndarray]) -> Tuple[bool, float, str, Dict[str, Any]]:
        sc, mask, margins = eval_r2_folds(fn, X_dev, y_dev)
        return True, sc, "ok", {{"pass_mask": mask, "bucket_scores": margins, "summary": f"CV R2*100={{sc:.2f}}"}}

    def blind_vault_eval(fn: Callable[[np.ndarray], np.ndarray]) -> Dict[str, Any]:
        sc_v, mask_v, _ = eval_r2_folds(fn, X_vault, y_vault)
        return {{"vault_r2_pct": round(sc_v, 4), "vault_passed_folds": sum(mask_v)}}

    base_fn = lambda arr: np.zeros(len(arr), dtype=np.float64)
    b0 = BranchCandidate(id="b0_interaction", branch=0, description="Bilinear interaction x0*x1", fn=lambda arr: 1.5 * arr[:, 0] * arr[:, 1])
    b1 = BranchCandidate(id="b1_nonlinear_z", branch=1, description="Saturating tanh(x2) feature", fn=lambda arr: -1.0 * np.tanh(arr[:, 2]))

    def xover(u: BranchCandidate, v: BranchCandidate) -> BranchCandidate:
        def _factory(p: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
            w0, w1, w_dead = float(p[0]), float(p[1]), float(p[2])
            return lambda arr: w0 * (arr[:, 0] * arr[:, 1]) + w1 * np.tanh(arr[:, 2]) + 0.0 * w_dead * arr[:, 3]

        p0 = np.array([1.5, -1.0, 0.5], dtype=np.float64)
        return BranchCandidate(
            id="xover_interaction_plus_tanh",
            branch=2,
            parent_id=u.id,
            description="Combined bilinear + tanh model with redundant knob for RRSI pruning",
            fn=_factory(p0),
            tunable_params=p0,
            tunable_bounds=[(0.5, 2.5), (-2.0, -0.2), (-1.0, 1.0)],
            param_names=["w_inter", "w_tanh", "dead_knob"],
            param_factory=_factory,
        )

    return TaskCartridge(
        task_name="{task_name}",
        description="6-fold CV ML feature discovery cartridge with Sealed Blind Vault",
        baseline_fn=base_fn,
        verify_and_score=verify_and_score,
        initial_branches=[b0, b1],
        crossover_synthesizer=xover,
        blind_vault_evaluator=blind_vault_eval,
        noise_probe_runs=2,
    )
'''
    else:  # mode == "code"
        body = f'''
BASE_CODE = """
def solve(x):
    return 0.0
"""

B0_CODE = """
def solve(x):
    return x[0] + x[1]
"""

B1_CODE = """
def solve(x):
    return x[0] - x[1]
"""


def build_cartridge() -> TaskCartridge:
    # Quadrant-dependent target: x[0] >= 0 -> x0+x1, else x0-x1
    rng = np.random.default_rng(7)
    inputs = [rng.uniform(-2.0, 2.0, size=2).tolist() for _ in range(80)]
    targets = [inp[0] + inp[1] if inp[0] >= 0.0 else inp[0] - inp[1] for inp in inputs]

    def pointwise_eval(fn: Callable[[Sequence[float]], float], idx: Optional[Sequence[int]]) -> List[float]:
        indices = list(range(len(inputs))) if idx is None else list(idx)
        return [-abs(float(fn(inputs[i])) - targets[i]) for i in indices]

    def verify_and_score(fn: Callable[[Sequence[float]], float]) -> Tuple[bool, float, str, Dict[str, Any]]:
        errs = [abs(float(fn(inp)) - tgt) for inp, tgt in zip(inputs, targets)]
        margins = [1.0 - min(2.0, e) for e in errs]
        mask = [1 if e < 1e-6 else 0 for e in errs]
        score = 100.0 * (sum(mask) / len(mask))
        return True, score, "ok", {{"pass_mask": mask, "bucket_scores": margins, "summary": f"Exact pass={{sum(mask)}}/{{len(mask)}}"}}

    b0 = BranchCandidate(
        id="b0_pos_halfspace",
        branch=0,
        description="Exact solver on x[0] >= 0 halfspace",
        fn=lambda x: x[0] + x[1],
        source_code=B0_CODE,
        entry_func="solve",
    )
    b1 = BranchCandidate(
        id="b1_neg_halfspace",
        branch=1,
        description="Exact solver on x[0] < 0 halfspace",
        fn=lambda x: x[0] - x[1],
        source_code=B1_CODE,
        entry_func="solve",
    )
    b2_bad = BranchCandidate(
        id="b2_doomed_mutation",
        branch=2,
        description="Deliberately broken candidate pruned early by 5% WaldMicroSentinel",
        fn=lambda x: 999.0,
    )
    return TaskCartridge(
        task_name="{task_name}",
        description="Autonomous AstCodeSplicer + 5% WaldMicroSentinel code synthesis cartridge",
        baseline_fn=lambda x: 0.0,
        verify_and_score=verify_and_score,
        initial_branches=[b0, b1, b2_bad],
        base_source=BASE_CODE,
        entry_func="solve",
        bucket_inputs=inputs,
        pointwise_slice_evaluator=pointwise_eval,
    )
'''

    footer = """

def main() -> int:
    pool_override = os.environ.get("DREAM_RSI_POOL")
    runner = AutoDreamRunner(build_cartridge(), pool_dir=pool_override)
    report = runner.run()
    print(f"Task: {report['task_name']} | Baseline: {report['baseline_score']:.4f} -> Champion `{report['champion_id']}`: {report['champion_score']:.4f}")
    if report.get("blind_vault_report"):
        print("Sealed Blind Vault:", json.dumps(report["blind_vault_report"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""
    return textwrap.dedent(common_header).lstrip() + body + footer


def cmd_scaffold(args: argparse.Namespace) -> int:
    target_dir = os.path.abspath(args.dir)
    os.makedirs(target_dir, exist_ok=True)
    safe_task = re.sub(r"[^a-zA-Z0-9_]+", "_", args.task).strip("_") or "task"
    filename = f"rsi_{safe_task}_harness.py"
    out_path = os.path.join(target_dir, filename)
    if os.path.exists(out_path) and not args.force:
        print(f"refusing to overwrite existing {out_path} (pass --force to overwrite)")
        return 2
    script_text = _render_scaffold_script(
        task_name=args.task, mode=args.mode, scripts_dir=_THIS_DIR
    ).replace("{script_name}", filename)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(script_text)
    print(f"scaffolded {args.mode!r} harness at: {out_path}")
    return 0


def cmd_sleep(args: argparse.Namespace) -> int:
    import numpy as np
    from engine import SpectralFrameLifter
    from sleep_dag_rsi import (
        ExpertFingerprintMeta,
        NeuroSymbolicDistiller,
        SleepDAGConfig,
        SleepDAGRSIOrchestrator,
    )

    cfg = SleepDAGConfig(
        vocab_size=32, d_model=16, d_ff=32, d_low=4, consolidation_steps=10
    )
    orch = SleepDAGRSIOrchestrator(cfg=cfg, seed=42)

    if args.demo or not args.task:
        trees = synth_trees("sleep-demo")
        t0 = trees[0]
        wake_seqs = np.array(
            [
                [4, 5, 6, 7, 8, 9, 10, 11, 8, 9, 10, 11],
                [4, 5, 6, 7, 12, 13, 14, 15, 12, 13, 14, 15],
            ],
            dtype=np.int64,
        )
        res = orch.run_wake_sleep_cycle(
            domain_name="sleep-demo",
            wake_sequences=wake_seqs,
            dag_tree=t0,
            trajectory_node_ids=["d0", "x"],
            fingerprint_meta=ExpertFingerprintMeta(
                expert_slot=0, source_node_id="d0", pass_mask=[1, 1, 1, 0, 0, 0]
            ),
            retention_evaluator=lambda _m: 1.0,
        )
        rng_v6 = np.random.default_rng(42)
        X_demo = rng_v6.uniform(-2.0, 2.0, size=(48, 6))
        Y_demo = (
            1.5 * (X_demo[:, 0] * X_demo[:, 1])
            - 1.0 * np.where(X_demo[:, 2] >= 0.0, 1.0, -1.0)
        )[:, None]
        ns_rep = NeuroSymbolicDistiller().distill_residual(X_demo, Y_demo)
        lifter_rep = SpectralFrameLifter().detect_and_lift(
            task_name="sleep-demo",
            theta_anchor=[1.0, -1.0],
            g_mat=[[5e8, 5e8 - 1.0], [5e8 - 1.0, 5e8]],
            m_buckets=4,
        )
        res["v6_spectral_and_neuro_symbolic"] = {
            "frame_lifter_triggered": lifter_rep["triggered"],
            "frame_lifter_operator": lifter_rep["operator"],
            "log10_kappa_eff": lifter_rep["diagnostics"]["log10_kappa_eff"],
            "neuro_symbolic_accepted": ns_rep["accepted"],
            "neuro_symbolic_r2_oos": ns_rep["r2_oos"],
            "neuro_symbolic_terms": ns_rep["active_terms"],
            "energy_reduction_ratio": ns_rep["energy_reduction_ratio"],
        }
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0

    trees = load_pool(task=args.task, pool_dir=args.pool)
    if not trees:
        print(f"no trees recorded for {args.task!r}")
        return 2
    t_last = trees[-1]
    nodes = [n for n in t_last.nodes.values() if n.id != "root"]
    nodes.sort(key=lambda n: n.score, reverse=True)
    top_id = nodes[0].id if nodes else "root"
    bot_id = nodes[-1].id if len(nodes) > 1 else top_id
    p_mask = (
        list(nodes[0].pass_mask) if (nodes and nodes[0].pass_mask) else [1, 1, 0, 0]
    )
    wake_seqs = np.array(
        [
            [4, 5, 6, 7, 8, 9, 10, 11, 8, 9, 10, 11],
            [4, 5, 6, 7, 12, 13, 14, 15, 12, 13, 14, 15],
        ],
        dtype=np.int64,
    )
    res = orch.run_wake_sleep_cycle(
        domain_name=args.task,
        wake_sequences=wake_seqs,
        dag_tree=t_last,
        trajectory_node_ids=[top_id, bot_id],
        fingerprint_meta=ExpertFingerprintMeta(
            expert_slot=0, source_node_id=top_id, pass_mask=p_mask
        ),
        retention_evaluator=lambda _m: 1.0,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="dreamrsi",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pool", help="override the pool directory")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run the whole loop on synthetic history")
    d.add_argument("--keep", action="store_true", help="keep the temp pool")
    d.set_defaults(func=cmd_demo)

    sc = sub.add_parser(
        "scaffold",
        help="generate a ready-to-run TaskCartridge harness in any project directory",
    )
    sc.add_argument("--dir", default=".", help="target directory")
    sc.add_argument("--task", required=True, help="unique task identifier")
    sc.add_argument(
        "--mode",
        choices=("code", "ml", "trading", "math"),
        default="code",
        help="domain guardrails preset",
    )
    sc.add_argument("--force", action="store_true", help="overwrite existing file")
    sc.set_defaults(func=cmd_scaffold)

    st = sub.add_parser(
        "state",
        help="print compact SKILL.state + backward subgoals JSON for subagents",
    )
    st.add_argument("--task", required=True)
    st.set_defaults(func=cmd_state)

    i = sub.add_parser("init", help="create loop state for a task")
    i.add_argument("--task", required=True)
    i.set_defaults(func=cmd_init)

    r = sub.add_parser("round", help="one propose/replay/select/persist cycle")
    r.add_argument("--task", required=True)
    r.set_defaults(func=cmd_round)

    p = sub.add_parser("prompt", help="render the policy-improvement prompt")
    p.add_argument("--task", required=True)
    p.add_argument("--out", help="write to a file instead of stdout")
    p.set_defaults(func=cmd_prompt)

    a = sub.add_parser("apply", help="apply a proposal JSON from a coding agent")
    a.add_argument("--task", required=True)
    a.add_argument("--file", required=True)
    a.set_defaults(func=cmd_apply)

    s = sub.add_parser("status", help="show state, ranking and history")
    s.add_argument("--task", required=True)
    s.set_defaults(func=cmd_status)

    t = sub.add_parser("trees", help="proof that a tree was recorded for a task")
    t.add_argument("--task", required=True)
    t.set_defaults(func=cmd_trees)

    id_sub = sub.add_parser(
        "ideate", help="synthesize hybrid/DAG/neuro-symbolic ideas from recorded trees"
    )
    id_sub.add_argument("--task", required=True)
    id_sub.set_defaults(func=cmd_ideate)

    sl_sub = sub.add_parser(
        "sleep", help="run SLEEP-DAG-RSI v8 Wake->NREM->REM->2x2 consolidation cycle"
    )
    sl_sub.add_argument("--task", help="task name in pool")
    sl_sub.add_argument(
        "--demo", action="store_true", help="run on synthetic demo tree"
    )
    sl_sub.set_defaults(func=cmd_sleep)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

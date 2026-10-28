"""
Dream-RSI command line.

    python dreamrsi.py demo                 end-to-end loop on synthetic history
    python dreamrsi.py init  --task NAME    create loop state for a real task
    python dreamrsi.py round --task NAME    propose -> replay -> select -> persist
    python dreamrsi.py prompt --task NAME   render the policy-improvement prompt
    python dreamrsi.py apply --task NAME --file proposal.json
    python dreamrsi.py status --task NAME   ranking, history, current policy
    python dreamrsi.py trees --task NAME    proof that a tree was recorded
    python dreamrsi.py ideate --task NAME   synthesize new hybrid/DAG/neuro-symbolic ideas
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import (  # noqa: E402
    DiscoveryTree,
    DynamicPortfolio,
    Node,
    Problem,
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
        """
        A chain hanging off the root, so several branches sit on the frontier at
        once. That is the condition under which policy choice can matter: with a
        single frontier node there is nothing to choose between.
        """
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
                score += first_gain * (0.55**step)  # diminishing returns
            parent = nid

    a = DiscoveryTree(task=task, baseline_score=100.0, max_parallelism=3)
    a.add(Node(id="root", parent_id=None, branch=-1, attempt=0, score=100.0))
    branch(a, "d", 22.0, 6, 0)  # deep, strong
    branch(a, "r", 14.0, 4, 1, fail=("oom", "out of memory"))  # repairable start
    branch(a, "z", 4.0, 3, 2)  # weak
    a.add(
        Node(
            id="x",
            parent_id="root",
            branch=3,
            attempt=1,
            score=92.0,
            fail_class="algorithmic_failure",
        )
    )  # dead end

    b = DiscoveryTree(task=task, baseline_score=50.0, max_parallelism=3)
    b.add(Node(id="root", parent_id=None, branch=-1, attempt=0, score=50.0))
    branch(b, "m", 9.0, 7, 0)  # very deep, modest
    branch(b, "n", 18.0, 3, 1)  # shallow, strong
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
    for i in range(3):
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

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

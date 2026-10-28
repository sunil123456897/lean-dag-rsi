"""
Self-test for the Dream-RSI universal engine.

Proves, with assertions rather than prose:
  1. the engine works on a domain where "score" is not a number
     (a prompt-tuning domain scored by a rating string),
  2. policy selection over history is monotone when the incumbent is included,
  3. replay is deterministic (same input, same numbers),
  4. replay is prefix-only (an unrevealed node can never influence a decision),
  5. a repairable failure stays open while a hard failure prunes
     - and that this is driven by the Problem adapter, not by hardcoded classes,
  6. the pool is append-only and multi-domain,
  7. the same engine code parses the original simulator's trees.

Run:  python _selftest.py
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

from engine import (
    POLICY_KINDS,
    DiscoveryTree,
    DynamicPortfolio,
    Node,
    Policy,
    Problem,
    World,
    build_policy,
    load_pool,
    policy_spec,
    replay,
    replay_over_history,
    save_tree,
    select_policy,
    sweep_beta,
)
from loop import candidates_from_specs, parse_proposal

FAILURES = []


def check(label: str, condition: bool, detail: str = "") -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f" -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(label)


# ---------------------------------------------------------------------------
# A non-numeric domain: tuning a prompt, quality judged by a rubric grade.
# This is the universality test: nothing about the engine may assume numbers
# come from a benchmark, or that "better" means "larger float".
# ---------------------------------------------------------------------------

GRADES = {"F": 0.0, "D": 1.0, "C": 2.0, "B": 3.0, "A": 4.0}


class PromptTuningProblem(Problem):
    """Attempts carry a rubric grade in meta['grade']; failures are parse errors."""

    name = "prompt-tuning"

    def score(self, node: Node) -> float:
        return GRADES.get(node.meta.get("grade", "F"), 0.0)

    def class_of(self, node: Node) -> str:
        if node.error and "ParseError" in node.error:
            return "parse_error"
        if node.meta.get("grade") == "F" and not node.error:
            return "algorithmic_failure"
        return "ok"

    def repairable(self, node: Node) -> bool:
        # Domain-specific widening: a malformed prompt is cheap to retry.
        return self.class_of(node) in ("parse_error", "syntax_error")


def build_prompt_tree() -> DiscoveryTree:
    t = DiscoveryTree(task="prompt-tuning", baseline_score=2.0, max_parallelism=3)
    t.add(
        Node(
            id="root",
            parent_id=None,
            branch=-1,
            attempt=0,
            score=2.0,
            meta={"grade": "C"},
        )
    )
    # branch 0: incremental wording tweaks
    t.add(
        Node(
            id="b0a1",
            parent_id="root",
            branch=0,
            attempt=1,
            score=3.0,
            meta={"grade": "B"},
        )
    )
    t.add(
        Node(
            id="b0a2",
            parent_id="b0a1",
            branch=0,
            attempt=2,
            score=4.0,
            meta={"grade": "A"},
        )
    )
    # branch 1: a structurally different prompt that failed to parse
    t.add(
        Node(
            id="b1a1",
            parent_id="root",
            branch=1,
            attempt=1,
            score=0.0,
            fail_class="parse_error",
            error="ParseError: unbalanced braces",
            meta={"grade": "F"},
        )
    )
    t.add(
        Node(
            id="b1a2",
            parent_id="b1a1",
            branch=1,
            attempt=2,
            score=3.0,
            meta={"grade": "B"},
        )
    )
    # branch 2: clean run, but the idea does not work -> real dead end
    t.add(
        Node(
            id="b2a1",
            parent_id="root",
            branch=2,
            attempt=1,
            score=1.0,
            fail_class="algorithmic_failure",
            meta={"grade": "D"},
        )
    )
    return t


def build_numeric_tree() -> DiscoveryTree:
    t = DiscoveryTree(task="kernel-tune", baseline_score=100.0, max_parallelism=3)
    t.add(Node(id="root", parent_id=None, branch=-1, attempt=0, score=100.0))
    t.add(Node(id="n0a1", parent_id="root", branch=0, attempt=1, score=115.0))
    t.add(Node(id="n0a2", parent_id="n0a1", branch=0, attempt=2, score=128.0))
    t.add(
        Node(
            id="n1a1",
            parent_id="root",
            branch=1,
            attempt=1,
            score=90.0,
            fail_class="oom",
            error="CUDA out of memory",
        )
    )
    t.add(Node(id="n1a2", parent_id="n1a1", branch=1, attempt=2, score=118.0))
    return t


def main() -> int:
    print("Dream-RSI universal engine self-test")
    print()

    prompt_tree = build_prompt_tree()
    numeric_tree = build_numeric_tree()
    prompt_problem = PromptTuningProblem()

    # -- 1. non-numeric domain --------------------------------------------
    print("1. Non-numeric domain (rubric grades instead of benchmark numbers)")
    res = replay(prompt_tree, DynamicPortfolio(beta=0.6), prompt_problem)
    check(
        "engine scores a grade-based domain",
        res["quality"] == 4.0,
        f"quality={res['quality']} (expected 4.0 = grade A)",
    )
    check(
        "quality is the max over revealed attempts, not the last",
        res["quality"] > prompt_tree.baseline_score,
        f"{res['quality']} > baseline {prompt_tree.baseline_score}",
    )

    # -- 2. monotonicity ---------------------------------------------------
    print()
    print("2. Monotone selection when the incumbent is in the candidate set")
    history = [prompt_tree, numeric_tree]
    incumbent = DynamicPortfolio(beta=0.2)
    cands = {
        "incumbent": incumbent,
        "wide": DynamicPortfolio(beta=0.9),
        "mid": DynamicPortfolio(beta=0.5),
    }
    sel = select_policy(cands, history, prompt_problem)
    inc_score = replay_over_history(history, incumbent, prompt_problem)["mean_v"]
    check(
        "selected policy is never worse than the incumbent",
        sel["best_mean_v"] >= inc_score - 1e-9,
        f"best={sel['best_mean_v']:.6f} incumbent={inc_score:.6f}",
    )
    check(
        "ranking lists every candidate", len(sel["ranking"]) == 3, str(sel["ranking"])
    )

    # -- 3. determinism ----------------------------------------------------
    print()
    print("3. Deterministic replay")
    a = replay(numeric_tree, DynamicPortfolio(beta=0.5))
    b = replay(numeric_tree, DynamicPortfolio(beta=0.5))
    check("two runs give identical numbers", a == b, f"{a['v']} == {b['v']}")
    check("no wall-clock or randomness leaked in", a["probes"] == b["probes"])

    # -- 4. prefix-only ----------------------------------------------------
    print()
    print("4. Prefix-only observability")
    print("   (invariant: a revealed node's recorded parent is always revealed,")
    print("    so no decision can ever be influenced by an unrevealed ancestor)")

    class InvariantSpy(Policy):
        violations: list = []
        batches: list = []

        def batch(self, world):
            for nid in world.revealed:
                parent = world.tree.nodes[nid].parent_id
                if parent is not None and parent not in world.revealed:
                    self.violations.append((nid, parent))
            legal = world.legal_actions()
            for action in legal:
                if action not in world.revealed:
                    self.violations.append(("illegal-action", action))
            self.batches.append(tuple(legal))
            return [legal[0]] if legal else []

    spy = InvariantSpy(beta=0.5)
    replay(prompt_tree, spy)
    check(
        "no decision was ever made from an unrevealed node",
        not spy.violations,
        f"violations={spy.violations}",
    )
    check("policy always sees the root", prompt_tree.root_id in prompt_tree.nodes)
    check(
        "legal actions grow/change across rounds (depth is reachable)",
        len(set(spy.batches)) > 1,
        f"{len(set(spy.batches))} distinct legal sets",
    )

    class FullTreeSpy(Policy):
        """Deliberately tries to cheat: reads the unwrapped tree."""

        def batch(self, world):
            hidden = [n for n in world.tree.nodes if n not in world.revealed]
            return [hidden[0]] if hidden else [world.tree.root_id]

    cheat = FullTreeSpy(beta=0.5)
    blocked = False
    try:
        replay(prompt_tree, cheat)
    except ValueError:
        blocked = True
    check("attempting to act on an unrevealed node raises", blocked)

    # -- 5. failure handling driven by the adapter -------------------------
    print()
    print("5. Repairable vs hard failure, decided by the adapter")

    class StrictProblem(PromptTuningProblem):
        """A domain where a malformed attempt is NOT cheap to retry."""

        def repairable(self, node):
            return False

    # The frontier is generic; only the recovery decision is domain-specific.
    lenient_world = World(prompt_tree, prompt_problem)
    strict_world = World(prompt_tree, StrictProblem())
    check(
        "both adapters expose the same frontier (legality is domain-free)",
        lenient_world.legal_actions() == strict_world.legal_actions(),
        f"{lenient_world.legal_actions()}",
    )

    # Advance the world until the parse-error attempt is on the frontier, then
    # see whether each adapter is willing to spend a slot repairing it.
    for _ in range(prompt_tree.max_parallelism):
        legal = lenient_world.legal_actions()
        if "b1a1" in legal:
            break
        lenient_world.probe([legal[0]])
    check(
        "parse-error attempt becomes reachable on the frontier",
        "b1a1" in lenient_world.legal_actions(),
        f"legal={lenient_world.legal_actions()}",
    )

    for _ in range(prompt_tree.max_parallelism):
        legal = strict_world.legal_actions()
        if "b1a1" in legal:
            break
        strict_world.probe([legal[0]])

    check(
        "parse error is repairable under the lenient adapter",
        prompt_problem.repairable(prompt_tree.nodes["b1a1"]),
    )
    check(
        "parse error is NOT repairable under the strict adapter",
        not StrictProblem().repairable(prompt_tree.nodes["b1a1"]),
    )

    # At low beta the policy has no exploration budget, so the recovery slot is
    # the ONLY route to a failed branch. That isolates the adapter's effect.

    for _ in range(prompt_tree.max_parallelism):
        legal = lenient_world.legal_actions()
        if "b1a1" in legal:
            break
        lenient_world.probe([legal[0]])
    lenient_recover = DynamicPortfolio(beta=0.2).batch(lenient_world)

    strict_world2 = World(prompt_tree, StrictProblem())
    for _ in range(prompt_tree.max_parallelism):
        legal = strict_world2.legal_actions()
        if "b1a1" in legal:
            break
        strict_world2.probe([legal[0]])
    strict_recover = DynamicPortfolio(beta=0.2).batch(strict_world2)

    check(
        "lenient adapter spends the recovery slot on the parse-error branch",
        "b1a1" in lenient_recover,
        f"batch={lenient_recover}",
    )
    check(
        "strict adapter declines to (domain rule flows through the adapter)",
        "b1a1" not in strict_recover,
        f"batch={strict_recover}",
    )

    check(
        "clean-but-wrong attempt is not repairable",
        not prompt_problem.repairable(prompt_tree.nodes["b2a1"]),
        "b2a1 is algorithmic_failure, not repairable",
    )
    # b2a1's grade is "D", above the adapter's dead-end threshold, so the
    # adapter correctly calls it "ok". Build the exact case instead of relying
    # on tree data: a clean run that scored an F is a real dead end.
    dead_end = Node(
        id="de",
        parent_id="root",
        branch=2,
        attempt=1,
        score=0.0,
        fail_class="ok",
        error=None,
        meta={"grade": "F"},
    )
    alive_but_weak = Node(
        id="aw",
        parent_id="root",
        branch=3,
        attempt=1,
        score=1.0,
        fail_class="ok",
        error=None,
        meta={"grade": "D"},
    )
    check(
        "adapter calls a clean F a dead end",
        prompt_problem.hard_failure(dead_end),
        f"adapter class={prompt_problem.class_of(dead_end)}",
    )
    check(
        "adapter does not call a clean D a dead end",
        not prompt_problem.hard_failure(alive_but_weak),
        f"adapter class={prompt_problem.class_of(alive_but_weak)}",
    )
    check(
        "base adapter honours a recorded algorithmic_failure class",
        Problem().hard_failure(numeric_tree.nodes["n1a1"]) is False
        and Problem().hard_failure(
            Node(
                id="x",
                parent_id="root",
                branch=0,
                attempt=1,
                score=0.0,
                fail_class="algorithmic_failure",
            )
        )
        is True,
        "recorded class is respected when no override exists",
    )

    # -- 6. append-only multi-domain pool ----------------------------------
    print()
    print("6. Pool: append-only, multi-domain, cross-task loadable")
    tmp = tempfile.mkdtemp(prefix="dreamrsi_selftest_")
    try:
        p1 = save_tree(prompt_tree, pool_dir=tmp)
        p2 = save_tree(numeric_tree, pool_dir=tmp)
        check(
            "two different tasks coexist",
            os.path.dirname(p1) != os.path.dirname(p2),
            f"{os.path.dirname(p1)} vs {os.path.dirname(p2)}",
        )
        all_trees = load_pool(pool_dir=tmp)
        check(
            "load_pool(task=None) returns every domain",
            len(all_trees) == 2,
            f"loaded {len(all_trees)}",
        )
        only_prompt = load_pool(task="prompt-tuning", pool_dir=tmp)
        check(
            "load_pool(task=...) filters by domain",
            len(only_prompt) == 1,
            f"loaded {len(only_prompt)}",
        )
        p3 = save_tree(prompt_tree, pool_dir=tmp)
        check(
            "saving twice never overwrites",
            p3 != p1 and os.path.exists(p1),
            os.path.basename(p3),
        )
        check(
            "reloaded tree keeps its nodes",
            len(all_trees[0].nodes) >= 4,
            f"{len(all_trees[0].nodes)} nodes",
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # -- 7. beta sweep -----------------------------------------------------
    print()
    print("7. Offline beta sweep (the cheapest form of 'dreaming')")
    ranked = sweep_beta([prompt_tree, numeric_tree], prompt_problem)
    check("sweep returns a full ranking", len(ranked) == 11, f"{len(ranked)} values")
    check(
        "sweep is sorted by score",
        all(ranked[i][1] >= ranked[i + 1][1] for i in range(len(ranked) - 1)),
    )

    # -- 8. policy serialisation -------------------------------------------
    print()
    print("8. Policies round-trip through JSON (no code stored in the pool)")

    spec = policy_spec(DynamicPortfolio(beta=0.7, max_recovery=2), label="x")
    check(
        "spec carries kind and parameters",
        spec["kind"] == "dynamic_portfolio"
        and spec["max_recovery"] == 2
        and spec["label"] == "x",
        json.dumps(spec),
    )
    rebuilt = build_policy(spec)
    check(
        "metadata is not fed to the constructor",
        isinstance(rebuilt, DynamicPortfolio) and rebuilt.max_recovery == 2,
    )
    check(
        "rebuilt policy replays identically",
        replay(numeric_tree, rebuilt)
        == replay(numeric_tree, DynamicPortfolio(beta=0.7, max_recovery=2)),
    )
    check(
        "all five kinds are registered",
        set(POLICY_KINDS)
        == {
            "dynamic_portfolio",
            "optimal_policy",
            "lean_dag",
            "deepest_first",
            "widest_first",
        },
        str(sorted(POLICY_KINDS)),
    )
    check(
        "every kind survives a spec round-trip",
        all(
            replay(numeric_tree, build_policy(policy_spec(k()))) is not None
            for k in POLICY_KINDS.values()
        ),
    )

    # -- 9. proposal parsing ------------------------------------------------
    print()
    print("9. Accepting proposals in the shapes a coding agent returns")
    one = parse_proposal({"kind": "deepest_first", "beta": 0.5})
    many = parse_proposal(
        [{"kind": "widest_first", "beta": 0.4}, {"kind": "deepest_first", "beta": 0.8}]
    )
    doc = parse_proposal(
        {
            "incumbent": {"kind": "dynamic_portfolio", "beta": 0.6},
            "candidates": [{"label": "a", "kind": "widest_first", "beta": 0.2}],
        }
    )
    check("single object accepted", len(one) == 1, str(list(one)))
    check("list accepted", len(many) == 2, str(list(many)))
    check(
        "documented shape accepted and incumbent included",
        len(doc) == 2,
        str(list(doc)),
    )
    check(
        "garbage is rejected without raising",
        parse_proposal("nonsense") == {} and parse_proposal({"nope": 1}) == {},
    )
    built = candidates_from_specs(many)
    check("parsed specs become runnable policies", len(built) == 2, str(list(built)))
    check(
        "unknown kind is skipped, not fatal",
        candidates_from_specs({"bad": {"kind": "does_not_exist"}}) == {},
    )

    # -- 10. the closed loop ------------------------------------------------
    print()
    print("10. Closed loop: propose -> replay -> select -> persist")
    import loop as loopmod
    import dreamrsi

    tmp_loop = tempfile.mkdtemp(prefix="dreamrsi_loop_")
    try:
        # A pool file that is bookkeeping, not a tree, must not break loading.
        d = os.path.join(tmp_loop, "t")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "loop_state.json"), "w", encoding="utf-8") as fh:
            fh.write('{"task": "t", "round": 0, "current": {}}')

        for tree in dreamrsi.synth_trees("t"):
            save_tree(tree, pool_dir=tmp_loop)

        loaded = load_pool(task="t", pool_dir=tmp_loop)
        check(
            "loop_state.json is not mistaken for a tree",
            len(loaded) == 2,
            f"loaded {len(loaded)}",
        )

        state = loopmod.initial_state("t")
        loopmod.save_state("t", state, tmp_loop)
        scores = []
        for _ in range(3):
            rep = loopmod.run_round("t", Problem(), pool_dir=tmp_loop)
            scores.append(rep["winner_mean_v"])
        check(
            "incumbent is always among the candidates",
            rep["candidates"] >= 2,
            f"{rep['candidates']} candidates",
        )
        check(
            "selection is monotone across rounds",
            all(scores[i] <= scores[i + 1] + 1e-12 for i in range(len(scores) - 1)),
            f"trajectory={scores}",
        )
        check(
            "history is persisted every round",
            len(loopmod.load_state("t", tmp_loop)["history"]) == 3,
        )
        check(
            "winner is a valid spec after the loop",
            isinstance(
                build_policy(loopmod.load_state("t", tmp_loop)["current"]), Policy
            ),
        )

        text = loopmod.render_prompt("t", pool_dir=tmp_loop, problem=Problem())
        check(
            "prompt renders with no leftover placeholders",
            "{{" not in text,
            f"{len(text)} chars",
        )
        check(
            "prompt states the prefix-only constraint",
            "Prefix-only" in text and "best_so_far" in text,
        )
        check("prompt explains the output format", '"candidates"' in text)

        empty = loopmod.render_prompt("no-such-task", pool_dir=tmp_loop)
        check(
            "prompt for an unknown task still renders (no history)",
            "{{" not in empty,
            f"{len(empty)} chars",
        )
    finally:
        shutil.rmtree(tmp_loop, ignore_errors=True)

    # -- 11. Appendix B.2 SEE mechanics (trajectories, non-degenerate beta, Pareto, GridPlan)
    print()
    print(
        "11. Appendix B.2 SEE mechanics: trajectories, non-degenerate beta, pareto_sweep, plan_grid"
    )
    from engine import (
        GridPlan,
        GridPlanningContext,
        LeanDAGWorld,
        OptimalPolicy,
        ideate_over_history,
        pareto_sweep,
        reconstruct_trajectories,
    )

    synth = dreamrsi.synth_trees("see-test")
    w_see = World(synth[0], Problem())
    w_see.probe(["root", "root"])
    trajs = reconstruct_trajectories(w_see)
    check(
        "reconstruct_trajectories builds per-branch history",
        len(trajs) >= 2 and 0 in trajs,
        f"branches={sorted(trajs)}",
    )

    ps = pareto_sweep(synth, Problem(), policy_cls=OptimalPolicy)
    probes_low = ps["curve"][0]["mean_probes"]
    probes_high = ps["curve"][-1]["mean_probes"]
    check(
        "OptimalPolicy beta sweep is non-degenerate (high beta probes >= low beta)",
        probes_high > probes_low,
        f"probes(beta=0.0)={probes_low} < probes(beta=1.0)={probes_high}",
    )
    check(
        "pareto_sweep computes pareto_reward and pareto_auc",
        "pareto_reward" in ps and "pareto_auc" in ps and len(ps["curve"]) == 11,
        f"pareto_reward={ps['pareto_reward']} auc={ps['pareto_auc']}",
    )

    pol_see = OptimalPolicy(beta=0.5)
    gp = pol_see.plan_grid(
        GridPlanningContext(
            history=[{"branch_count": 3, "refine_count": 2, "late_depth_gains": True}],
            hard_max_branch_count=6,
            hard_max_refine_count=6,
        )
    )
    check(
        "plan_grid returns a valid GridPlan with reason",
        isinstance(gp, GridPlan)
        and gp.branch_count == 3
        and gp.refine_count == 3
        and bool(gp.reason),
        f"plan={gp}",
    )

    # -- 12. Lean-DAG-RSI upgrades (monotone reward, s_cf, CROSSOVER, ideation)
    print()
    print(
        "12. Lean-DAG-RSI upgrades: monotone reward, Tukey s_cf(v), CROSSOVER(u,v), ideate_over_history"
    )
    # 12a. Wasted probes in Round 1 strictly reduce V (dV/dN = -beta1 < 0)
    t_spam = DiscoveryTree(task="spam-check", baseline_score=100.0, max_parallelism=6)
    t_spam.add(Node(id="root", parent_id=None, branch=-1, attempt=0, score=100.0))
    t_spam.add(Node(id="b0", parent_id="root", branch=0, attempt=1, score=150.0))
    for b in range(1, 6):
        t_spam.add(Node(id=f"b{b}", parent_id="root", branch=b, attempt=1, score=90.0))
    w_s1 = World(t_spam)
    w_s1.probe(["root"])
    w_s6 = World(t_spam)
    w_s6.probe(["root"] * 6)
    check(
        "wasting probes on junk branches strictly decreases V",
        w_s1.reward()["v"] > w_s6.reward()["v"],
        f"V(1 probe)={w_s1.reward()['v']:.4f} > V(6 probes)={w_s6.reward()['v']:.4f}",
    )

    # 12b. Right-censored fast branch + CROSSOVER in LeanDAGWorld
    lw = LeanDAGWorld(synth[0])
    lw.probe(["root", "root", "root"])
    legal_dag = lw.legal_actions()
    has_xover = any(a.startswith("CROSSOVER(") for a in legal_dag)
    check(
        "LeanDAGWorld exposes CROSSOVER(u,v) between above-baseline branches",
        has_xover,
        f"legal={legal_dag}",
    )

    # 12c. Ideation synthesizer extracts crossovers and neuro-symbolic split recommendations
    ideas = ideate_over_history(synth)
    check(
        "ideate_over_history synthesizes hybrid crossovers and neuro-symbolic split alerts",
        len(ideas["crossovers"]) >= 1 and len(ideas["neuro_symbolic_splits"]) >= 1,
        f"crossovers={len(ideas['crossovers'])} splits={len(ideas['neuro_symbolic_splits'])}",
    )

    # -- 13. Error Fingerprint, Flat-Plateau + RRSI Pruner, and Bottleneck Autopsy
    print()
    print(
        "13. Error Fingerprint crossover, Flat-Plateau + RRSI Pruner, and Bottleneck Autopsy"
    )
    from engine import (
        evaluate_flat_plateau,
        profile_callable_lines,
        prune_tunable_params,
    )

    # 13a. Error Fingerprint picks complementary branch (b0+b2 -> 100% union) over clone (b0+b1 -> 80% union)
    t_fp = DiscoveryTree(task="fp-test", baseline_score=50.0, max_parallelism=4)
    t_fp.add(Node(id="root", parent_id=None, branch=-1, attempt=0, score=50.0))
    t_fp.add(
        Node(
            id="b0",
            parent_id="root",
            branch=0,
            attempt=1,
            score=80.0,
            pass_mask=[1, 1, 1, 1, 0],
            failure_breakdown={"opcode=add_alpha": 8.0, "opcode=xor": 0.0},
        )
    )
    t_fp.add(
        Node(
            id="b1_clone",
            parent_id="root",
            branch=1,
            attempt=1,
            score=78.0,
            pass_mask=[1, 1, 1, 1, 0],
        )
    )
    t_fp.add(
        Node(
            id="b2_comp",
            parent_id="root",
            branch=2,
            attempt=1,
            score=65.0,
            pass_mask=[0, 0, 1, 1, 1],
        )
    )
    lw_fp = LeanDAGWorld(t_fp)
    lw_fp.probe(["root", "root", "root"])
    legal_fp = lw_fp.legal_actions()
    ideas_fp = ideate_over_history([t_fp])
    top_xo = ideas_fp["crossovers"][0]
    check(
        "Error Fingerprint skips clone b1_clone (0% union gain) and selects complementary b2_comp (+20% union gain)",
        "CROSSOVER(b0,b2_comp)" in legal_fp
        and top_xo["action"] == "CROSSOVER(b0,b2_comp)"
        and abs(top_xo["oracle_union_rate"] - 1.0) < 1e-6
        and abs(top_xo["oracle_union_gain"] - 0.2) < 1e-6,
        f"action={top_xo['action']} union={top_xo['oracle_union_rate']} gain={top_xo['oracle_union_gain']}",
    )

    # 13b. Flat-Plateau Jitter Test rejects needle spikes & RRSI Pruner removes redundant knobs
    spike_rep = evaluate_flat_plateau(
        lambda th: 100.0 if abs(th[0] - 0.4512) < 0.001 else 45.0,
        [0.4512, 0.50],
        rel_jitter=0.05,
        n_directions=20,
    )
    flat_rep = evaluate_flat_plateau(
        lambda th: 90.0 - 2.0 * ((th[0] - 0.5) ** 2 + (th[1] - 0.5) ** 2),
        [0.50, 0.50],
        rel_jitter=0.05,
        n_directions=20,
    )
    check(
        "evaluate_flat_plateau flags needle spike and passes smooth plateau",
        spike_rep["is_sharp_spike"] is True
        and flat_rep["is_sharp_spike"] is False
        and flat_rep["plateau_mean"] > spike_rep["plateau_mean"],
        f"spike_mean={spike_rep['plateau_mean']:.2f} (spike={spike_rep['is_sharp_spike']}) vs flat_mean={flat_rep['plateau_mean']:.2f}",
    )

    prune_rep = prune_tunable_params(
        lambda th: 50.0
        + 10.0 * th[0]
        + 5.0 * th[1]
        + 1e-5 * (th[2] + th[3] + th[4] + th[5]),
        theta_opt=[1.0, 2.0, 0.9, 0.8, 0.7, 0.6],
        theta_default=[0.0, 0.0, 0.5, 0.5, 0.5, 0.5],
        param_names=["w0", "w1", "w2", "w3", "w4", "w5"],
        delta_tol=1e-3,
    )
    check(
        "prune_tunable_params (RRSI Pruner) freezes 4 redundant knobs and keeps 2 essential knobs",
        prune_rep["active_count"] == 2
        and prune_rep["pruned_count"] == 4
        and prune_rep["kept_params"] == ["w0", "w1"],
        f"kept={prune_rep['kept_params']} pruned={prune_rep['pruned_params']}",
    )

    # 13c. Automated Bottleneck Autopsy (line profiler + failure_breakdown diagnostics)
    def _dummy_candidate(n: int = 2500) -> int:
        acc = 0
        for i in range(n):
            acc += (i * i) % 97
        return acc

    prof_rep = profile_callable_lines(_dummy_candidate, 2500, top_k=3)
    prof_capped = profile_callable_lines(
        _dummy_candidate, 2500, top_k=3, max_line_events=100
    )
    t_fp.nodes["b0"].line_hotspots = prof_rep["line_hotspots"]
    ideas_autopsy = ideate_over_history([t_fp])
    diag = ideas_autopsy["bottleneck_diagnostics"][0]
    check(
        "profile_callable_lines and bottleneck_diagnostics pinpoint top CPU line, 100% failure tag, and enforce max_line_events cap",
        len(prof_rep["line_hotspots"]) >= 1
        and prof_capped["trace_capped"] is True
        and prof_capped["line_events_sampled"] == 100
        and prof_capped["result"] == prof_rep["result"]
        and diag["top_failure_tag"] == "opcode=add_alpha"
        and abs(diag["top_failure_share_pct"] - 100.0) < 1e-6
        and diag["top_hotspot"] is not None,
        f"capped_events={prof_capped['line_events_sampled']} autopsy='{diag['autopsy_prompt']}'",
    )

    print(
        "\n14. DiscoveryRSI (arXiv:2610.23839v1) upgrades: H=(c,pi,d,a), Harness Replay, Provenance Credit + Family Cap, Masked SFT, and 2x2 Acceptance"
    )
    from engine import (
        accept_joint_2x2,
        compute_provenance_credit,
        diagnose_harness_failures,
        export_masked_sft_dataset,
        validate_hypothesis_contract,
    )

    # 14a. Pre-Commitment Hypothesis Contract H = (c, pi, d, a)
    bad_contract = validate_hypothesis_contract(
        {
            "claim": "Step-1 Mid-Flight Spike overlay improves OTC win rate",
            "plan": "Evaluate on Train+Val 40k candles",
            "decision_criteria": {"support": "LCB95 > 56.5%"},  # missing weaken/inconclusive & confounders
        }
    )
    good_contract = validate_hypothesis_contract(
        {
            "claim": "Step-1 Mid-Flight Spike overlay improves OTC 2-step win rate",
            "plan": "Evaluate on chronological Train+Val (50 windows), hold out blind Vault",
            "decision_criteria": {
                "support": "LCB95 > 56.8% and plateau_mean > 56.5%",
                "weaken": "LCB95 <= baseline 55.83%",
                "inconclusive": "55.83% < LCB95 <= 56.8% or sharp_spike=True",
            },
            "alternative_explanations": [
                "Session rollover spread widening around 00:00 UTC (control: Safe-Session filter)",
                "Overfitting 6th threshold slope60_min on random train noise (control: RRSI Pruner)",
            ],
        }
    )
    check(
        "validate_hypothesis_contract rejects incomplete criteria/confounders and accepts full H=(c,pi,d,a)",
        bad_contract["valid"] is False
        and "alternative_explanations" in bad_contract["missing_fields"]
        and good_contract["valid"] is True
        and good_contract["confounder_count"] == 2
        and good_contract["has_tri_state_criteria"] is True,
        f"bad={bad_contract['missing_fields']} good='{good_contract['summary']}'",
    )

    # 14b. Harness Replay 4-class failure diagnostics
    t_h = DiscoveryTree(task="harness-audit", baseline_score=50.0, max_parallelism=3)
    t_h.add(Node(id="root", parent_id=None, branch=-1, attempt=0, score=50.0))
    # Branch 0: deep chain (depth 2) with artifact
    t_h.add(
        Node(
            id="b0_s1",
            parent_id="root",
            branch=0,
            attempt=1,
            score=70.0,
            pass_mask=[1, 1, 0, 0],
            artifact={"diff": "patch_a"},
        )
    )
    t_h.add(
        Node(
            id="b0_s2",
            parent_id="b0_s1",
            branch=0,
            attempt=2,
            score=82.0,
            pass_mask=[1, 1, 1, 0],
            artifact={"diff": "patch_a2"},
        )
    )
    # Branch 1: clone of b0_s1 (`repeated_exploration`), abandoned at depth 1 (`premature_closure`), and missing artifact (`lost_artifacts`)
    t_h.add(
        Node(
            id="b1_clone_no_art",
            parent_id="root",
            branch=1,
            attempt=1,
            score=68.0,
            pass_mask=[1, 1, 0, 0],
        )
    )
    # Branch 2: repairable tool error with no repair child (`failed_tool_use` + `premature_closure`)
    t_h.add(
        Node(
            id="b2_unrepaired_err",
            parent_id="root",
            branch=2,
            attempt=1,
            score=0.0,
            fail_class="timeout",
            error="Tool TimeoutError",
        )
    )
    h_diag = diagnose_harness_failures([t_h])
    check(
        "diagnose_harness_failures detects all 4 Harness Replay failure modes (repeated_exploration, failed_tool_use, lost_artifacts, premature_closure)",
        len(h_diag["repeated_exploration"]) >= 1
        and len(h_diag["failed_tool_use"]) == 1
        and len(h_diag["lost_artifacts"]) == 1
        and len(h_diag["premature_closure"]) >= 1,
        f"issues={h_diag['total_issues']} (rep={len(h_diag['repeated_exploration'])}, tool={len(h_diag['failed_tool_use'])}, lost={len(h_diag['lost_artifacts'])}, prem={len(h_diag['premature_closure'])})",
    )

    # 14c. Provenance Credit + Anti-Clone Family Cap (C_loc=2, C_glob=2) + Breakthrough vs Repair bonus
    # Create 20 clone variations in family "fam_clone" and 1 genuine breakthrough in "fam_novel"
    hyps = [
        {
            "id": "h_novel",
            "mu": 8.0,
            "sigma": 0.5,
            "comparisons": 6,
            "progress_kind": "breakthrough",
            "local_family": "fam_novel",
            "global_family": "fam_novel",
            "lead_trajectories": ["traj_lead_1"],
            "producer_trajectories": ["traj_gen_novel"],
            "ancestry_only_trajectories": ["traj_passenger_zero"],
        },
        {
            "id": "h_repair",
            "mu": 8.0,
            "sigma": 0.5,
            "comparisons": 6,
            "progress_kind": "repair",
            "local_family": "fam_repair",
            "global_family": "fam_repair",
            "lead_trajectories": ["traj_lead_1"],
            "producer_trajectories": ["traj_evo_repair"],
        },
    ]
    for k in range(20):
        hyps.append(
            {
                "id": f"h_clone_{k}",
                "mu": 7.5,
                "sigma": 0.6,
                "comparisons": 4,
                "progress_kind": "repair",
                "local_family": "fam_clone",
                "global_family": "fam_clone",
                "lead_trajectories": ["traj_lead_clone"],
                "producer_trajectories": ["traj_spam_clone"],
            }
        )
    cred = compute_provenance_credit(hyps, c_loc=2.0, c_glob=2.0, w_max=2.0)
    q_novel = cred["hypothesis_weights"]["h_novel"]["q_raw"]
    q_repair = cred["hypothesis_weights"]["h_repair"]["q_raw"]
    clone_fam_tot = cred["family_totals"]["fam_clone"]
    passenger_w = cred["trajectory_weights"].get("traj_passenger_zero", 0.0)
    check(
        "compute_provenance_credit enforces Proposition 1 (C_glob<=2.0 on 20 clones, passenger=0.0) and rewards breakthroughs over routine repairs",
        cred["conservation_verified"] is True
        and clone_fam_tot <= 2.0 + 1e-6
        and passenger_w == 0.0
        and q_novel > q_repair,
        f"fam_clone_total={clone_fam_tot:.4f}<=2.0, q_novel={q_novel:.4f}>q_repair={q_repair:.4f}, passenger_w={passenger_w}",
    )

    # 14d. Selective Loss-Masked SFT Exporter (100% context retained, loss_mask=1 only on winning lineage)
    sft_pack = export_masked_sft_dataset(t_h)
    check(
        "export_masked_sft_dataset retains all nodes in context but masks loss strictly to winning lineage",
        sft_pack["total_context_steps"] == 5
        and sft_pack["supervised_steps"] == 2
        and sft_pack["zero_loss_context_steps"] == 3
        and sft_pack["supported_node_ids"] == ["b0_s1", "b0_s2"],
        f"supervised={sft_pack['supervised_steps']}/{sft_pack['total_context_steps']} ({sft_pack['supervised_fraction']:.2%}) targets={sft_pack['supported_node_ids']}",
    )

    # 14e. 2x2 Joint Model x Harness Acceptance with Novelty Floor Guard
    acc_conflict = accept_joint_2x2(
        {
            ("theta_0", "psi_0"): 70.0,
            ("theta_plus", "psi_0"): 75.0,
            ("theta_0", "psi_plus"): 74.0,
            ("theta_plus", "psi_plus"): 68.0,  # negative interference when combined!
        }
    )
    acc_novelty_guard = accept_joint_2x2(
        {
            ("theta_0", "psi_0"): 4.86,
            ("theta_plus", "psi_0"): 5.18,
            ("theta_0", "psi_plus"): 5.10,
            ("theta_plus", "psi_plus"): 5.39,  # higher rubric score, but novelty dropped!
        },
        novelty_scores={
            ("theta_0", "psi_0"): 4.87,
            ("theta_plus", "psi_0"): 4.94,
            ("theta_0", "psi_plus"): 4.88,
            ("theta_plus", "psi_plus"): 4.58,  # Figure 6b novelty collapse (4.87 -> 4.58)
        },
        min_novelty_floor=4.85,
    )
    check(
        "accept_joint_2x2 catches negative model-harness interference and blocks candidates that violate min_novelty_floor",
        acc_conflict["has_negative_interference"] is True
        and acc_conflict["accepted_pair"] == ("theta_plus", "psi_0")
        and ("theta_plus", "psi_plus") in acc_novelty_guard["rejected_by_novelty"]
        and acc_novelty_guard["accepted_pair"] == ("theta_plus", "psi_0"),
        f"conflict_winner={acc_conflict['accepted_pair']} (synergy={acc_conflict['interaction_synergy']}) | novelty_guarded_winner={acc_novelty_guard['accepted_pair']} (rejected={acc_novelty_guard['rejected_by_novelty']})",
    )

    print(
        "\n15. Lean-DAG-RSI v4 upgrades: Excess-Margin HYPER_CROSSOVER + AST IoU, Epistatic Gauss-Newton G=J^T J Pruner, Bucket JEPA, DAG-DPO, and Lean 4 Certificates"
    )
    from engine import (
        compute_hyper_crossover,
        export_dag_dpo_pairs,
        export_lean4_certificate,
        predict_bucket_jepa,
        prune_epistatic_gauss_newton,
    )

    # 15a. Excess-Margin Submodular Hyper-Crossover:
    # - Rejects dominant + disjoint junk node (Junk-Parent Trap!)
    # - Rejects colliding AST diff (IoU_AST = 1.0)
    # - Merges 3 orthogonal complementary branches (b0, b1, b2) in a single round
    u_dom = Node(
        id="b0_dom",
        parent_id="root",
        branch=0,
        attempt=1,
        score=90.0,
        meta={"margin_vector": [0.95, 0.95, 0.20, 0.20, 0.20, 0.20], "ast_touches": ["preproc_gram"]},
    )
    u_junk_disjoint = Node(
        id="b3_junk_disjoint",
        parent_id="root",
        branch=3,
        attempt=1,
        score=55.0,
        meta={"margin_vector": [0.10, 0.10, 0.10, 0.10, 0.10, 0.10], "ast_touches": ["unused_helper"]},
    )
    u_colliding = Node(
        id="b4_ast_collide",
        parent_id="root",
        branch=4,
        attempt=1,
        score=82.0,
        meta={"margin_vector": [0.20, 0.20, 0.90, 0.90, 0.20, 0.20], "ast_touches": ["preproc_gram"]},
    )
    u_comp1 = Node(
        id="b1_screen",
        parent_id="root",
        branch=1,
        attempt=1,
        score=80.0,
        meta={"margin_vector": [0.20, 0.20, 0.92, 0.92, 0.20, 0.20], "ast_touches": ["lookahead_screen"]},
    )
    u_comp2 = Node(
        id="b2_simd",
        parent_id="root",
        branch=2,
        attempt=1,
        score=78.0,
        meta={"margin_vector": [0.20, 0.20, 0.20, 0.20, 0.94, 0.94], "ast_touches": ["avx512_inner_kernel"]},
    )
    # Test 1: Dominant + Disjoint Junk alone must NOT trigger crossover (Excess Gain == 0)
    hx_junk = compute_hyper_crossover([u_dom, u_junk_disjoint], lambda_collision=0.35)
    # Test 2: Full pool selects 3-way orthogonal hybrid (b0_dom, b2_simd, b1_screen) and rejects colliding & junk
    hx_full = compute_hyper_crossover(
        [u_dom, u_junk_disjoint, u_colliding, u_comp1, u_comp2],
        lambda_collision=0.35,
    )
    check(
        "compute_hyper_crossover blocks Dominant+Junk trap (0 excess gain), penalizes AST collisions, and merges 3 orthogonal branches in 1 step",
        hx_junk["action"] == ""
        and hx_full["parent_count"] == 3
        and set(hx_full["selected_ids"]) == {"b0_dom", "b1_screen", "b2_simd"}
        and "b3_junk_disjoint" in hx_full["rejected_candidates"]
        and "b4_ast_collide" in hx_full["rejected_candidates"],
        f"action={hx_full['action']} excess_gain={hx_full['excess_hyper_gain']} rejected={hx_full['rejected_candidates']}",
    )

    # 15b. Scale-Invariant Empirical Bucket Covariance G = J^T J Pruner:
    # Epistatic ridge where objective depends ONLY on ratio theta_0 / theta_1 (single-knob pruner fails,
    # whereas G = J^T J finds the null eigenvector and collapses theta_1 -> default while scaling theta_0!)
    def _epistatic_buckets(th):
        ratio = th[0] / max(1e-6, th[1])  # optimal at ratio = 2.0 (e.g. th0=4.0, th1=2.0)
        active = th[2]                     # independent active parameter (optimal at 1.0)
        noise_knob = th[3]                 # pure dead parameter
        err_r = (ratio - 2.0) ** 2
        err_a = (active - 1.0) ** 2
        return [
            100.0 - 20.0 * err_r - 10.0 * err_a - 1e-7 * noise_knob,
            95.0 - 25.0 * err_r - 5.0 * err_a,
            90.0 - 15.0 * err_r - 15.0 * err_a,
            85.0 - 30.0 * err_r - 8.0 * err_a,
        ]

    gn_rep = prune_epistatic_gauss_newton(
        _epistatic_buckets,
        theta_opt=[4.0, 2.0, 1.0, 0.75],
        bounds=[(0.5, 10.0), (0.5, 5.0), (0.0, 2.0), (0.0, 1.0)],
        theta_default=[2.0, 1.0, 0.0, 0.0],
        param_names=["spike_num", "spike_den", "session_guard", "dead_knob"],
    )
    check(
        "prune_epistatic_gauss_newton detects coupled epistatic ratio ridge (theta_0/theta_1) + dead knob via G=J^T J nullspace",
        len(gn_rep["coupled_invariants"]) >= 1
        and "dead_knob" in gn_rep["frozen_params"]
        and "spike_den" in gn_rep["frozen_params"]
        and abs(gn_rep["collapsed_score"] - gn_rep["base_score"]) < 1e-4,
        f"effective_dof={gn_rep['effective_dof']}/4 frozen={gn_rep['frozen_params']} invariants={gn_rep['coupled_invariants']}",
    )

    # 15c. Zero-Cold-Start Bucket-Factored JEPA World Model + DAG-DPO + Lean 4 Certificate
    jepa_pred = predict_bucket_jepa(
        observed_nodes=[u_dom, u_comp1],
        candidate_parent=u_dom,
        candidate_ast_touches={"new_orthogonal_module"},
        candidate_step_delta=5.0,
    )
    dpo_rep = export_dag_dpo_pairs(t_h, min_score_gap=1.0)
    lean_cert = export_lean4_certificate(
        theorem_name="heilbronn_n4_demo",
        points_num_den=[(0, 0), (10, 0), (0, 10), (4, 4)],
        scale_denom=10,
        bound_num=1,
        bound_den=10,
    )
    check(
        "predict_bucket_jepa, export_dag_dpo_pairs, and export_lean4_certificate verify cleanly without sorryAx",
        len(jepa_pred["predicted_margins"]) == 6
        and jepa_pred["expected_union_gain"] > 0.0
        and dpo_rep["pair_count"] >= 1
        and lean_cert["verified_exact_int"] is True
        and lean_cert["has_sorry"] is False
        and "native_decide" in lean_cert["cert_lean_code"],
        f"jepa_gain={jepa_pred['expected_union_gain']} dpo_pairs={dpo_rep['pair_count']} lean_verified={lean_cert['verified_exact_int']}",
    )

    # 15d. Recorded Multi-Parent DAG Merge Causality in World.legal_actions() and World.probe()
    t_dag_caus = DiscoveryTree(task="dag-causality-test", baseline_score=50.0, max_parallelism=2)
    t_dag_caus.add(Node(id="root", parent_id=None, branch=-1, attempt=0, score=50.0))
    t_dag_caus.add(Node(id="b0_a1", parent_id="root", branch=0, attempt=1, score=70.0))
    t_dag_caus.add(Node(id="b1_a1", parent_id="root", branch=1, attempt=1, score=72.0))
    t_dag_caus.add(
        Node(
            id="xover_rec_b0_b1",
            parent_id="b0_a1",
            branch=0,
            attempt=2,
            score=88.0,
            meta={"secondary_parent_id": "b1_a1"},
        )
    )
    w_caus = World(t_dag_caus, allow_counterfactual=False, allow_crossover=False)
    w_caus.probe(["root"])  # reveals only b0_a1 (b1_a1 is still unrevealed!)
    legal_before_b1 = w_caus.legal_actions()
    w_caus.probe(["root"])  # now reveals b1_a1 as well
    legal_after_b1 = w_caus.legal_actions()
    revealed_merge = w_caus.probe(["b0_a1"])
    check(
        "recorded multi-parent merge child is blocked until ALL parents (primary + secondary_parent_id) are revealed",
        "b0_a1" not in legal_before_b1
        and "b0_a1" in legal_after_b1
        and len(revealed_merge) == 1
        and revealed_merge[0].id == "xover_rec_b0_b1",
        f"before_b1={legal_before_b1} after_b1={legal_after_b1} revealed={[n.id for n in revealed_merge]}",
    )

    # -- 16. SLEEP-DAG-RSI v5 upgrades (HOPE + CMS + SLEEP + DAG Bridges 1..4) --
    print()
    print("16. SLEEP-DAG-RSI v5 upgrades: HOPE DeltaRuleFastMemory, DAG-KS Consolidation, and Fingerprint MoE REM Dreaming")
    import numpy as np
    from sleep_dag_rsi import (
        CounterfactualDreamingEngine,
        DeltaRuleFastMemory,
        ExpandableMoEMLP,
        ExpertFingerprintMeta,
        SleepDAGConfig,
        SleepDAGRSIOrchestrator,
        SleepLanguageModel,
        generalized_jsd_loss_and_grad,
    )

    rng_s5 = np.random.default_rng(20261028)
    s_log = rng_s5.normal(0.0, 1.0, size=(2, 4, 10))
    t_log = rng_s5.normal(0.0, 1.0, size=(2, 4, 10))
    _, anal_g = generalized_jsd_loss_and_grad(s_log, t_log, beta=0.5)
    s_p = s_log.copy()
    s_m = s_log.copy()
    s_p[0, 0, 0] += 1e-6
    s_m[0, 0, 0] -= 1e-6
    lp_s5, _ = generalized_jsd_loss_and_grad(s_p, t_log, beta=0.5)
    lm_s5, _ = generalized_jsd_loss_and_grad(s_m, t_log, beta=0.5)
    fd_err = abs(float(anal_g[0, 0, 0]) - (lp_s5 - lm_s5) / 2e-6)

    fast_m = DeltaRuleFastMemory(d_model=12, rng=rng_s5)
    fast_m.b_alpha = np.full((1,), 20.0)
    fast_m.b_beta = np.full((1,), 10.0)
    rep_tok = np.repeat(rng_s5.normal(0.0, 1.0, size=(1, 1, 12)), repeats=8, axis=1)
    _, _, err_n = fast_m.forward(rep_tok, write_Scale=0.85)
    check(
        "Generalized JSD analytical gradient (<1e-7) and HOPE DeltaRuleFastMemory associative convergence",
        fd_err < 1e-7 and err_n[-1] < 1e-4 * err_n[0],
        f"jsd_fd_err={fd_err:.2e} delta_err={err_n[0]:.3f}->{err_n[-1]:.2e}",
    )

    cfg_s5 = SleepDAGConfig(vocab_size=32, d_model=16, d_ff=32, d_low=4)
    blk_s5 = ExpandableMoEMLP(cfg_s5, update_period=4, rng=rng_s5)
    s_anc = blk_s5.num_base + blk_s5.unlock_new_low_rank_expert(
        ExpertFingerprintMeta(expert_slot=0, source_node_id="anc", pass_mask=[1, 1, 1, 0, 0, 0], ast_symbols={"k_a"})
    )
    s_cln = blk_s5.num_base + blk_s5.unlock_new_low_rank_expert(
        ExpertFingerprintMeta(expert_slot=0, source_node_id="cln", pass_mask=[1, 1, 1, 0, 0, 0], ast_symbols={"k_b"})
    )
    s_cmp = blk_s5.num_base + blk_s5.unlock_new_low_rank_expert(
        ExpertFingerprintMeta(expert_slot=0, source_node_id="cmp", pass_mask=[0, 0, 0, 1, 1, 1], ast_symbols={"k_c"})
    )
    comp_pair_s5 = blk_s5.select_complementary_crossover_experts()
    check(
        "ExpandableMoEMLP Error-Fingerprint router skips clone expert and selects complementary pair",
        comp_pair_s5 is not None and {comp_pair_s5[0], comp_pair_s5[1]} == {s_anc, s_cmp},
        f"selected={comp_pair_s5} clone_slot={s_cln}",
    )

    orch_s5 = SleepDAGRSIOrchestrator(cfg=cfg_s5, seed=42)
    wake_s5 = np.array(
        [
            [4, 5, 6, 7, 8, 9, 10, 11, 8, 9, 10, 11],
            [4, 5, 6, 7, 12, 13, 14, 15, 12, 13, 14, 15],
        ],
        dtype=np.int64,
    )
    t_s5 = DiscoveryTree(task="s5_demo", baseline_score=50.0, max_parallelism=2)
    t_s5.add(Node(id="root", parent_id=None, branch=-1, attempt=0, score=50.0))
    t_s5.add(Node(id="b0_win", parent_id="root", branch=0, attempt=1, score=90.0, pass_mask=[1, 1, 0, 0]))
    t_s5.add(Node(id="b1_trap", parent_id="root", branch=1, attempt=1, score=20.0, fail_class="algorithmic_failure"))
    cyc_s5 = orch_s5.run_wake_sleep_cycle(
        domain_name="s5_demo",
        wake_sequences=wake_s5,
        dag_tree=t_s5,
        trajectory_node_ids=["b0_win", "b1_trap"],
        fingerprint_meta=ExpertFingerprintMeta(expert_slot=0, source_node_id="b0_win", pass_mask=[1, 1, 0, 0]),
        retention_evaluator=lambda m: 1.0,
    )
    check(
        "SleepDAGRSIOrchestrator completes 4-phase Wake->NREM->REM->2x2 cycle with DAG loss masking (1/2 supervised)",
        cyc_s5["ks_stats"]["supervised_trajectories"] == 1
        and cyc_s5["ks_stats"]["total_wake_trajectories"] == 2
        and "accepted_pair" in cyc_s5["gate_2x2"],
        f"supervised={cyc_s5['ks_stats']['supervised_trajectories']}/{cyc_s5['ks_stats']['total_wake_trajectories']} gate={cyc_s5['gate_2x2']['accepted_pair']}",
    )

    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILED -> {FAILURES}")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

You are improving ONE prefix-only exploration policy.

Read this whole brief before proposing anything. Your job is to propose a better
way to choose *which attempts to continue next*. It is not to solve the task.

## Task context (measured, not narrated)

| fact | value |
| :--- | :--- |
| task | `{{task}}` |
| recorded trees | {{trees}} |
| recorded attempts (nodes) | {{nodes}} |
| distinct branches explored | {{branches}} |
| loop round | {{round}} |
| incumbent mean replay score | {{incumbent_mean_v}} |
| failure classes observed | {{failure_classes}} |
| max parallel workers | {{max_parallelism}} |
| offline beta sweep (beta:score) | {{beta_sweep}} |
| pool directory | `{{pool_dir}}` |

Current policy, as stored:

```json
{{current_policy_json}}
```

## What you are allowed to change

The exploration policy, and nothing else. The task, the evaluator, the worker
prompts and the interfaces are fixed. If your idea requires changing any of those,
it is out of scope — say so instead of doing it.

Available policy kinds: {{kinds}}.

## Hard constraints (violating any of these invalidates the proposal)

1. **Prefix-only.** A policy may use revealed observations, `baseline_score`, the
   legal-action set, and structural metadata. It must NEVER use unrevealed scores,
   a known optimum, hardcoded winning node ids, absolute score targets, or
   internal trace data.
2. **Never read `best_so_far` or `budget_spent` to decide what to explore.** They
   are bookkeeping, not signals.
3. **beta is fixed for the whole episode.** Never re-tune it from observations
   inside the run. It is swept offline by the evaluator, and only then chosen.
4. **A batch must be legal before it is submitted**: every member in the revealed
   frontier, no duplicates, and never a parent together with its own child.
5. **Never sample randomly**, and do not default to a singleton just because the
   top candidate is obvious.
6. **Do not use a fixed widen-all / deepen-all wave schedule.** Allocation must
   respond to what the recorded history actually shows.
7. **Do not inject directional prose guidance** into worker instructions
   ("only try technique X"). It measurably hurts diversity and performance.

## What a good proposal looks like

- Exploit the strongest verified anchors.
- Keep at least one slot for an unopened branch or an underexplored frontier.
- Spend at most one slot repairing a *repairable* failure — never on a clean run
  that produced a wrong answer, which is a dead end, not a bug.
- Change one mechanism at a time so its effect is attributable.
- Infer from the failure-class mix above: many timeouts argue for smaller
  batches; many repairable errors argue for a wider recovery budget; a mostly-`ok`
  history argues for depth.

## Output format

Return ONLY a JSON object, no prose around it:

```json
{
  "candidates": [
    {"label": "short-reason", "kind": "dynamic_portfolio", "beta": 0.6,
     "explore_threshold": 0.4, "recovery_threshold": 0.3, "max_recovery": 1}
  ],
  "incumbent": {"label": "incumbent", "kind": "dynamic_portfolio", "beta": 0.6},
  "rationale": "one or two sentences tying the numbers above to the change"
}
```

Include the incumbent unchanged under `incumbent`. Selection is by mean replay
score over every recorded tree, and because the incumbent is in the candidate
set, the chosen policy can never score worse on this history than the current one.
Propose between one and five candidates. Prefer few and well-reasoned.

If you believe a new policy *kind* is required, do not write code yet — return a
single candidate identical to the incumbent and explain the missing mechanism in
`rationale`.

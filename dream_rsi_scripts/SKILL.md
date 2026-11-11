---
name: dream-rsi
description: Universal SLEEP-DAG-RSI v8 discovery engine for budgeted code, algorithm, kernel, ML pipeline, trading strategy, and mathematical optimization. Use when iterating against a deterministic numeric evaluator (10+ attempts or multi-branch search): scaffolds a TaskCartridge harness, filters weak mutations at 5% cost (WaldMicroSentinel), decomposes dead buckets (BackwardSubGoalDecomposer), splices complementary code branches (AstCodeSplicer), prunes redundant parameters (Gauss-Newton / RRSI + Flat-Plateau), and seals held-out evaluation behind BlindVaultGuard.
---

# Dream-RSI (`SLEEP-DAG-RSI v8`): Universal Multi-Branch Discovery Runbook

Scripts live at `~/.gemini/config/plugins/autonomous-research/skills/dream-rsi/scripts/` (`DREAM_SCRIPTS`).
Recorded trees and `_skill_state.json` persist at `$DREAM_RSI_POOL` or `~/.gemini/antigravity/dream_rsi_pool`.

## 0. Mandatory Gate & Closing Proof

Before writing candidate code, state two facts explicitly:
1. **Metric & Evaluator:** Exact scalar measured per attempt, its units, and why larger means better. If quality is subjective with no executable score, stop — this skill does not apply.
2. **Orthogonal Branches:** What distinct mechanisms separate Branch 0, Branch 1, and Branch 2.

**Closing proof (mandatory before claiming completion):** Run and paste verbatim:
```bash
python "$DREAM_SCRIPTS/dreamrsi.py" trees --task <your-task>
```
No recorded tree means the skill was not executed.

---

## 1. Step-by-Step Execution Protocol

### Step 1: Scaffold the Cartridge Harness in the Target Project
Generate a self-contained `rsi_<task>_harness.py` matched to the domain preset (`code`, `ml`, `trading`, or `math`):
```bash
python "$DREAM_SCRIPTS/dreamrsi.py" scaffold --dir <project_dir> --task <task_name> --mode <code|ml|trading|math>
```
Wire `baseline_fn` and `verify_and_score(fn) -> (ok, score, fail_class, info)` to the real project evaluator:
- Populate `info["pass_mask"]` (`[0, 1, ...]`) and continuous `info["bucket_scores"]` across empirical examples, folds, or chronological windows (never structural feature flags) so `compute_hyper_crossover` and `BackwardSubGoalDecomposer` measure real synergy.
- Keep `noise_probe_runs=2` on stochastic benchmarks so `AutoDreamRunner` calibrates `2 * sigma_noise`.
- When per-item evaluation is expensive, supply `pointwise_slice_evaluator(fn, idx)` so `WaldMicroSentinel` aborts doomed mutations at `5%` compute cost (`2.5%` adversarial + `2.5%` stratified slice) and routes them into DPO negative pairs.
- If a held-out split exists, pass it via `blind_vault_evaluator` — `BlindVaultGuard` physically enforces `max_calls=1` on the final champion only.
- *Completion criterion:* Baseline executes cleanly (`ok == True`) and records `root` score `s0`.

### Step 2: Dispatch Orthogonal Initial Branches (Round 1)
Author 2–4 genuinely distinct structural mechanisms (`BranchCandidate`):
- Attach `source_code` and `entry_func` whenever candidates are pure functions so `AutoDreamRunner` can synthesize zero-handwriting AST crossovers via `AstCodeSplicer` + `AstPolicySafetyGate`.
- Expose continuous thresholds/weights as `tunable_params`, `tunable_bounds`, and `param_factory` — keep LLM edits focused on discrete structure rather than guessing floats.
- For large codebases, delegate independent branches in parallel to `rsi-branch-worker` subagents (`Model: "inherit"`, `workspace: "branch"`), passing the compact JSON output of:
  ```bash
  python "$DREAM_SCRIPTS/dreamrsi.py" state --task <task_name>
  ```
- *Completion criterion:* Every initial branch is recorded in the tree (`ok`, `repairable`, or `sentinel_pruned`).

### Step 3: Repair, Stepping-Stones & DAG Crossover (Rounds 2–3)
Query structural diagnostics over recorded trees:
```bash
python "$DREAM_SCRIPTS/dreamrsi.py" ideate --task <task_name>
```
1. **Rescue `repairable_anchors`:** Fix shape/timeout/truncation bugs once (`repair_candidate`); abandon any branch failing $\ge 2$ consecutive repairs.
2. **Crack Dead Buckets (`backward_subgoals`):** If any bucket has `0%` pass rate across all branches, target the relaxed stepping-stone threshold emitted by `BackwardSubGoalDecomposer` and promote the first solver as the new anchor.
3. **Splice Complementary Branches (`crossovers`):** Merge pairs `(u, v)` with positive `oracle_union_gain > 0` via `crossover_synthesizer` or autonomous `AstCodeSplicer` (which synthesizes linear, bilinear `(x[i]*x[j])`, or radial OOS input routing predicates).
- *Completion criterion:* Hybrid candidate strictly exceeds $\max(s_u, s_v)$ on `Dev` without AST collision regression.

### Step 4: Neuro-Symbolic Tuning, Epistatic Pruning & Sealed Audit (Round 4)
1. Optimize `tunable_params` on `Dev`, then run `prune_tunable_params` / `prune_epistatic_gauss_newton` ($G = \frac{1}{M}J^\top J$) to freeze redundant or coupled sloppy knobs back to defaults.
2. Verify flat neighborhood stability via `evaluate_flat_plateau(..., rel_jitter=0.05, n_directions=20)` (`is_sharp_spike == False`). If optimization stalls in an ill-conditioned valley ($\kappa_{\text{eff}}(G) \ge 10^6$), invoke `SpectralFrameLifter` (`EigenRotationTransform`, `KktSlackTransform`, or `DyadicHomotopyTransform`) with mandatory `inverse_projection`.
3. Run `AutoDreamRunner.run()` to execute the `2x2` acceptance gate, open `BlindVaultGuard` once on the champion, and persist `_skill_state.json`.
4. Dispatch `rsi-blind-auditor` (or run `_selftest.py`) and print `dreamrsi.py trees --task <task_name>`.

---

## 2. Disclosed Domain References (Read on Demand Only)

Load the matching reference file **only** when your task enters that specific domain:
- **Extremal Geometry, Packing, Step Convolutions ($C_1, C_2, \Phi_3$) & Formal `Lean 4` Proofs:** read [`references/extremal-math-kkt.md`](references/extremal-math-kkt.md) for corank-1 equiareal manifold KKT polishing, floater separation, Kronecker big-int exact rational verification, and zero-`sorryAx` `Lean 4` certificates.
- **Financial & Sequential Time-Series (`BTCUSDT`, OTC, Market Microstructure):** read [`references/timeseries-trading.md`](references/timeseries-trading.md) for strict `htf_close_time <= ltf_close_time` causality, phase-invariant event cooldowns, Purged Walk-Forward, and Fee-to-ATR regime gates.
- **Low-Level `engine.py` & `cartridge.py` v8 API Signatures:** read [`references/engine-v8-api.md`](references/engine-v8-api.md) when calling `SpectralFrameLifter`, `ActiveBoundarySynthesizer`, `compute_max_tilted_objective`, or custom `Problem` adapters directly.
- **Adapting Pretrained PyTorch / NumPy Neural Networks Online (`Wake -> NREM -> REM`):** activate the companion skill **`neural-sleep`**.

## 3. Engine Self-Verification
After any modification to `scripts/`, run:
```bash
python "$DREAM_SCRIPTS/_selftest.py"   # must end with "RESULT: all checks passed"
```

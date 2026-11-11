# `engine.py` & `cartridge.py` (`SLEEP-DAG-RSI v8`) Quick API Reference

Disclosed technical reference for `dream-rsi`.

## 1. `cartridge.py` High-Level SDK
- `TaskCartridge(task_name, description, baseline_fn, verify_and_score, initial_branches, crossover_synthesizer=None, train_evaluator=None, base_source=None, entry_func=None, bucket_inputs=None, pointwise_slice_evaluator=None, blind_vault_evaluator=None, noise_probe_runs=1)`
- `BranchCandidate(id, branch, description, fn, parent_id="root", attempt=1, next_refinement=None, repair_candidate=None, tunable_params=None, tunable_bounds=None, param_names=None, param_factory=None, source_code=None, entry_func=None)`
- `AutoDreamRunner(cartridge, pool_dir=None).run() -> dict` (executes Rounds 1–4, persists tree + `_skill_state.json`, opens `BlindVaultGuard` at most once).
- `BlindVaultGuard(eval_fn, max_calls=1).evaluate_once(fn)` (raises `BlindVaultViolationError` on call #2).

## 2. `engine.py` Core v4–v8 Primitives
- **Selection & Replay:** `LeanDAGPortfolio(beta=0.6, gamma_dag=0.35, entropic_tau=4.0)`, `compute_max_tilted_objective(rewards, tau=4.0)`, `ideate_over_history(trees, problem)`.
- **Early Veto (`5%` Slice):** `WaldMicroSentinel(adv_frac=0.025, strat_frac=0.025, noise_regime="deterministic"|"stochastic").evaluate_gate(parent_node, child_node, parent_pointwise_scores, eval_slice_fn, is_crossover=False)`.
- **Dead-Bucket Stepping Stones:** `decompose_dead_buckets(trees, problem, ladder_fractions=(0.5, 0.25, 0.0))`.
- **Multi-Parent & AST Crossover:** `compute_hyper_crossover(tree, problem)`, `AstCodeSplicer().splice_parents(base_source, parent_sources, entry_func=..., bucket_inputs=..., parent_bucket_margins=...)`, `AstPolicySafetyGate.verify_source(code)`.
- **Parameter Pruning & Spectral Lifting:** `prune_tunable_params(...)`, `prune_epistatic_gauss_newton(eval_buckets_fn, theta_opt, bounds=...)`, `evaluate_flat_plateau(eval_fn, theta_opt, bounds=..., rel_jitter=0.05, n_directions=20)`, `SpectralFrameLifter().detect_and_lift(task_name, theta_anchor, g_mat, m_buckets, j_mat=...)` ($\mathcal{O}(M^2 d + M^3)$ dual Gram fast path when $M < d$).
- **Dual-Head Verification & Compact State:** `ActiveBoundarySynthesizer().synthesize_dual_head_verification(committee_fns, X_anchor, oracle_fn=...)`, `MutableSkillState(task).commit_cycle(tree, problem).to_dict()`.

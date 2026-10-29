---
name: dream-rsi
description: Universal SLEEP-DAG-RSI v7 & Lean-DAG-RSI discovery engine for budgeted multi-attempt search, algorithm/kernel optimization, mathematical bounds, financial strategy discovery, and continual adaptation of pretrained neural networks. Combines counterfactual DAG replay, ideate_over_history(), WaldMicroSentinel (5% early abort filter), SpectralFrameLifter (regularized kappa_eff, KKT slack lifting with active-set condensation, eigen-basis rotation), AstCodeSplicer (autonomous AST crossover code synthesis), ActiveBoundarySynthesizer (active disagreement boundary counterexamples), and 4-phase Wake -> NREM DAG-KS + NeuroSymbolicDistiller + SymbolicLibrary (hierarchical depth-3+ AST subroutine invention) -> REM Fingerprint MoE -> 2x2 Gate consolidation.
whenToUse: Activate when a task requires budgeted attempts scored by a deterministic evaluator, generating structural hypotheses from history (ideate_over_history), autonomously splicing orthogonal code branches (compute_hyper_crossover + AstCodeSplicer), separating code topology from continuous parameters (prune_epistatic_gauss_newton + evaluate_flat_plateau), escaping ill-conditioned coordinate traps (SpectralFrameLifter), or continually adapting a pretrained neural network across sequential domains with hierarchical symbolic AST rule extraction (SleepDAGRSIOrchestrator / UniversalNeuralSleepAdapter / NeuroSymbolicDistiller / SymbolicLibrary).
---

# Dream-RSI (`SLEEP-DAG-RSI v7` & `Lean-DAG-RSI`): Counterfactual DAG Replay, Spectral Frame-Lifting, Autonomous AST Splicing & Compositional Neuro-Symbolic Sleep Consolidation

Operationalises *"Dream-RSI: Recursive Self-Improvement through Evolving Worlds"*
(Tong Zheng, Xidong Wu, Zheng Zhang et al.; Google / Google DeepMind / UMD / UVA; 2026
Google technical report — no venue, no arXiv ID). Full text: `references/dream-rsi-paper.txt`.

Scope, verbatim from the paper: **"Only the exploration-policy code is updated; the
underlying models, evaluator, and execution interfaces remain fixed."** This is
meta-level improvement, not self-modifying task code.

The engine is domain-agnostic. It needs a tree of scored attempts and nothing else;
all domain knowledge sits behind a small adapter (§2).

## 0. Gate — read before anything else

Answer both, out loud, in your first reply:

1. **What single number is measured per attempt, and by what evaluator?** Name it and
   its units. If you cannot — if "better" here is an opinion, a summary, or a judgement
   call — then **stop and say the skill does not apply.** Do not continue, and do not
   head the result with this skill's name. Fitting this skill to a task that has no
   score is the one failure mode that looks like success.
2. **What is one attempt, and why will attempt #2 differ from attempt #1?** If every
   attempt would be the same action, there is no search and the skill still does not
   apply.

**Closing proof.** Before you finish, paste the verbatim output of:

```bash
python scripts/dreamrsi.py trees --task <your-task>
```

A tree with scored nodes must appear there. If it does not, you did not use this skill:
say that plainly instead of writing a synthesis. Prose headed with this skill's name and
containing no scored nodes is worse than no answer — it is unfalsifiable.

## 1. When to use

Only where the gate above can be answered: tuning an algorithm, kernel, config or
schedule; iterating prompts against a fixed evaluation; eliminating competing
hypotheses; any task needing 10+ budgeted attempts.

## 2. The one thing you must supply: a Problem adapter

The engine knows nothing about your domain. Describe it once:

```python
from engine import Problem

class MyProblem(Problem):
    name = "my-domain"

    def score(self, node):        # larger must mean better
        return float(node.meta["latency_ms"]) * -1

    def class_of(self, node):     # failure taxonomy for this domain
        if node.error and "timeout" in node.error:
            return "timeout"
        if node.error:
            return "syntax_error"
        return "ok"

    def repairable(self, node):   # cheap to retry, or a real dead end?
        return self.class_of(node) in ("timeout", "syntax_error")
```

Defaults suit any domain whose attempts yield one number where larger is better.
`repairable` decides what "one error does not prove a bad idea" means *for you* —
widen it for flaky domains, narrow it where retries are expensive.

## 3. Record attempts (tree = the artifact)

```python
from engine import DiscoveryTree, Node, save_tree

tree = DiscoveryTree(task="my-domain", baseline_score=s0, max_parallelism=3)
tree.add(Node(id="root", parent_id=None, branch=-1, attempt=0, score=s0))

# one node per attempt: parent_id encodes which attempt you continued
tree.add(Node(id="b0a1", parent_id="root", branch=0, attempt=1, score=s1,
              artifact={"diff": unified_diff}, meta={"latency_ms": 41}))
tree.add(Node(id="b1a1", parent_id="root", branch=1, attempt=1, score=0.0,
              fail_class="timeout", error="TimeoutError after 30s"))

save_tree(tree)          # appended to the pool; never overwrites
```

`artifact` and `meta` are free-form, so any domain fits: a diff, a prompt, a set of
parameters, a hypothesis, log output.

## 4. Batch allocation (per round)

Every batch of up to `max_parallelism` continuations must balance four roles:

1. **Exploitation** — deepen the strongest verified anchors.
2. **Exploration** — at least one slot aimed at an unopened branch or frontier.
3. **Recovery** — **at most one** slot repairing a `repairable` failure.
4. **Synthesis & Tuning (`LeanDAGPortfolio`)** — cross-pollinate top orthogonal branches (`CROSSOVER(u, v)`) and trigger inner-loop continuous optimization (`TUNE`) on nodes exposing `@tunable` parameters (`meta["tunable_params"]`).

Default policy in code: `LeanDAGPortfolio` (`kind="lean_dag"`) in `scripts/engine.py` (with `DynamicPortfolio`, `DeepestFirst`, and `WidestFirst` kept as baselines).

Everything in `scripts/` is plain Python with no dependencies. Run it from the
skill's own directory (`scripts/`), or add that directory to `sys.path` first —
`engine.py`, `loop.py` and `dreamrsi.py` import each other by module name. The
recorded pool lives outside the skill, at `$DREAM_RSI_POOL` or
`~/.gemini/antigravity/dream_rsi_pool`, so two installs of this skill share one
history.

**Do not inject high-level prose summaries into worker prompts** ("only use
technique X"). The paper tested exactly this (§5.1) and the guided variant
*"consistently underperforms its unguided counterpart across both paradigms under
equivalent discovery budgets"*, tending to *"over-constrain the search space and
impede diverse exploration."* Orchestrate the search topology — which branches live
or die — and let each attempt reason freely.

## 5. Failure handling (never close a branch on first error)

| Class | Examples | Action |
| :--- | :--- | :--- |
| Repairable | syntax error, shape/dimension mismatch, index error, config error, malformed output, transient timeout | **Keep open.** Queue one recovery slot with the exact error. |
| Transient resource | OOM, timeout from oversized batch | **Keep open.** Retest smaller. |
| Algorithmic dead end | Runs cleanly, result is still wrong or worse after valid repairs | **Prune.** Record as dead end. |

Close a branch only when: (a) unrecoverable algorithmic failure, or (b) ≥2
consecutive failed recovery attempts, or (c) score plateaued ≥3 depths while another
branch shows positive gradient.

## 6. Dreaming: pick the policy offline

`replay()` walks a **frozen recorded tree** — no model calls, no execution. Evaluating
alternative policies is therefore free.

```python
from engine import LeanDAGPortfolio, DynamicPortfolio, load_pool, select_policy, ideate_over_history, sweep_beta

history = load_pool()                       # every domain; load_pool(task=...) filters
candidates = {"lean_dag": LeanDAGPortfolio(beta=0.6, gamma_dag=0.35),
              "incumbent": DynamicPortfolio(beta=0.6),
              "wide": DynamicPortfolio(beta=0.9),
              "lean": DynamicPortfolio(beta=0.2)}
result = select_policy(candidates, history, MyProblem())
print(result["ranking"])                    # argmax by mean replay score
print(sweep_beta(history, MyProblem()))     # cheapest form of dreaming
ideas = ideate_over_history(history, MyProblem())  # 4-stage hypothesis generator
```

**Always include the incumbent in `candidates`.** That is what makes selection
monotone: because the candidate set contains the current policy, the chosen one can
never score worse on the frozen history. The paper states this guarantee for replay
score only — there is **no** online-improvement guarantee, and the paper claims none.

### The replay objective (Lean-DAG-RSI corrected form in `engine.py:622`)

`V = max s_v − β₁·N − β₂·k` — best verified quality (including counterfactual leaf/DAG extrapolation), minus total probe cost `β₁·N`, minus sequential latency cost `β₂·k` (`rounds`).
- At fixed rounds `k`, every wasted probe strictly reduces reward (`∂V/∂N = −β₁ < 0` for all `k ≥ 1`), eliminating the original paper's Eq. 1 bug (`+ β₂·N/k`) where spamming useless probes inside a wide batch increased `V` whenever `k < β₂/β₁`.
- At fixed total probes `N`, executing them in fewer parallel rounds `k` increases `V` by `+β₂` per saved round.
`β₁`, `β₂` are yours to sweep per domain; scores in the 0..1 range need far larger coefficients than scores in the thousands. `β` in a policy is a scalar **fixed for the whole episode** and swept offline — never re-tuned from observations mid-episode.

### Counterfactual extrapolation & Error-Fingerprint Crossover (fixing Support Censorship)

In standard replay, a policy that deepens a promising leaf or executes `CROSSOVER(u, v)` beyond the recorded tree depth gets `0` credit ("Support Censorship"). `LeanDAGWorld` (`engine.py`) projects:
- **DAG Crossover (`CROSSOVER(u, v)`) via Error Fingerprints (`compute_error_fingerprint`)**: When nodes carry aligned binary `pass_mask` outcome vectors (`[0, 1, ...]`), `engine.py` ranks candidate branch pairs by **non-overlapping union coverage** `len(pass_u | pass_v) / M` (`oracle_union_gain = union_rate - max(rate_u, rate_v)`). Clones with 100% overlapping wins get `0.0` synergy, while a weaker branch that solves the exact slice where the champion fails is prioritized automatically. Falls back to `max(s_u, s_v) + γ · min(tukey_cap, γ_dag · min(Δ_u, Δ_v))` when `pass_mask` is absent.
- **Unrecorded Leaf Extension (`CONTINUE` at leaf)**: `s_proj = s_leaf + γ^h · min(Δ_last, tukey_cap)` using Tukey's non-parametric fence `U(T) = med(Δs⁺) + 1.5·IQR(Δs⁺)`.
Never let a policy read unrevealed scores, a known optimum, or internal trace data —
the engine enforces this by passing only the revealed prefix.

## 6.5. Structural Ideation (`ideate_over_history` & `Auto-Dream-RSI`)

Before writing new domain code, run `ideate_over_history(trees, problem)` (or CLI `python scripts/dreamrsi.py ideate --task <task>`) to extract four actionable queues for the coding agent:

1. **Stage 1 — Repairable Anchors (`repairable_anchors`)**: Inspect nodes rejected by sample-size (`insufficient_trades`), timeout, or length-truncation guards (e.g. `num_clauses <= 66`). Often the strongest domain mechanism is hidden inside a failed node that merely needs its scope widened or a truncation bug removed.
2. **Stage 2 — Error-Fingerprint DAG Synthesis (`crossovers`)**: Record an empirical binary `pass_mask` per test example or chronological window on each `Node` (computed strictly on `Train + Validation`, never touching the blind vault). `ideate_over_history()` scans all branch pairs `(u, v)` in `< 1 ms` to find the pair with maximum union gain `oracle_union_gain = mean(pass_u | pass_v) - max(mean(pass_u), mean(pass_v))`, preventing blind crossover of identical-outcome branches.
   - **Guardrail (Empirical vs. Structural Masks)**: Never build `pass_mask` from structural feature/config flags (e.g. `[has_filter_A, has_filter_B]`), because two distinct indicators sharing the same flags will produce identical masks (`oracle_union_gain == 0`) and be falsely rejected as clones. Always use actual empirical outcomes across examples or 50–100 chronological `Train + Val` windows.
3. **Stage 3 — Neuro-Symbolic Split + Flat-Plateau Jitter + RRSI Pruner (`neuro_symbolic_splits`)**: Never ask an LLM to guess continuous floats (`0.37` vs `0.45`) by trial-and-error. Keep discrete structural topology in the outer loop, optimize `tunable_params` on `Train + Validation`, and pass the result through two mandatory guards in `engine.py`:
   - `prune_tunable_params(eval_fn, theta_opt, theta_default, param_names, delta_tol)` (**RRSI Complexity Pruner / Секатор**): Greedily freezes each tuned knob back to its default value and discards any parameter whose removal drops the score by `≤ delta_tol`.
   - `evaluate_flat_plateau(eval_fn, theta_opt, bounds, rel_jitter=0.05, n_directions=20)` (**Anti-Spike Jitter Test**): Shakes all active parameters by `±5%` across 20 deterministic directions and flags `is_sharp_spike=True` if the neighborhood mean collapses (`cliff_drop = peak_score - plateau_mean > tol`).
4. **Stage 4 — Automated Bottleneck Autopsy (`bottleneck_diagnostics` & `profile_callable_lines`)**: Record `failure_breakdown: Dict[str, float]` (empirical errors grouped by tag, e.g. `{"opcode=add_alpha": 8.0}`) and `line_hotspots: List[Dict]` (via `profile_callable_lines(fn, *args, max_line_events=50000)` in `engine.py`, a zero-dependency `sys.settrace` nanosecond line profiler). `ideate_over_history()` automatically pinpoints the #1 CPU hotspot line and dominant failure tag.
   - **Guardrail (Trace Cap & Inner-Loop Exclusion)**: Keep `max_line_events=50000` active so `sys.settrace` auto-detaches on pure-Python inner loops, and call `profile_callable_lines` **strictly once** per candidate during node verification — never inside `scipy.optimize`, `evaluate_flat_plateau`, or `prune_tunable_params`.

**Separation of roles (`ideate_over_history` vs `TaskCartridge`)**: `ideate_over_history()` in `scripts/engine.py` is the domain-agnostic hypothesis generator that inspects recorded JSON trees and outputs the structured engineering specifications above; a coding agent (or human engineer) then implements the candidate functions and plugs them into the `TaskCartridge` + `AutoDreamRunner` verification harness (`e:\эксперименты\исследования\auto_dream_rsi.py`).

**Zero-Gaming Branch Provenance & Array Citation Guardrails**:
- **No Backwards Perturbation of Champions**: Never synthesize intermediate Round 1–3 branches (`b0`, `b1`, `b2`, `xover`) by adding noise, constant shifts, or power scaling (`champ + rng.normal(...)`, `a + c`, `a ** p`) to a final champion just to make a harness script run faster. Every intermediate branch must execute its own genuine algorithm or load its own saved checkpoint (`.npy`/`.json`) produced by that specific algorithm.
- **Mandatory `stdout` Dump Before Citing Coordinates/Arrays**: Never write coordinate tables, weight vectors, or numerical certificates in prose or markdown reports unless you have explicitly printed the exact array from the `.npy`/`.json` file to `stdout` in the current turn (`python -c "import numpy as np; print(repr(np.load(...)))"`). Writing floats from memory when a binary artifact exists on disk produces hallucinated numbers that fail independent verification.

## 6.6. DiscoveryRSI (`arXiv:2610.23839v1`) Extensions in `engine.py`

Four operational mechanisms from *DiscoveryRSI: Recursively Learning to Discover Valuable Hypotheses* (Zhong, Wu, Yu et al., Oct 2026) are implemented in `scripts/engine.py`:

1. **Pre-Commitment Hypothesis Contract $\mathcal{H} = (c, \pi, d, a)$ (`validate_hypothesis_contract`)**:
   Before compiling or executing a candidate in noisy/data-driven domains (such as OTC trading or ML architecture search), require a pre-committed dict specifying `claim` ($c$), `plan` ($\pi$), tri-state `decision_criteria` ($d$: `support`, `weaken`, `inconclusive`), and `alternative_explanations` ($a$: confounders, selection biases, and discriminating control checks).
2. **Harness Replay 4-Class Diagnostics (`diagnose_harness_failures`)**:
   Automatically executed inside `ideate_over_history()["harness_diagnostics"]` to detect:
   - `repeated_exploration`: branches with 100% overlapping `pass_mask` (`oracle_union_gain == 0`) or identical parameters/scores.
   - `failed_tool_use`: repairable execution errors (`timeout`, `syntax_error`, `oom`) with no repair child.
   - `lost_artifacts`: above-baseline winning nodes that failed to persist `artifact` or `tunable_params`.
   - `premature_closure`: branches abandoned at depth 1 despite positive gain or a repairable error.
3. **Provenance-Bounded Credit, Anti-Clone Family Cap & Selective Loss Masking (`compute_provenance_credit`, `export_masked_sft_dataset`)**:
   - Computes conservative score $s_h = \mu_h - 2\sigma_h$, reliability damping $\rho_h^{\text{cov}} \rho_h^{\text{unc}}$, provenance contributor shares ($\gamma = 0.2$ Lead, $0.8$ verified Producers, `0.0` unverified ancestry), and caps clone families at $C_{\text{loc}} = 2.0, C_{\text{glob}} = 2.0, W_{\max} = 2.0$ (Proposition 1 credit conservation).
   - Fixes DiscoveryRSI Eq. 17's equal-bonus bias (`lambda_breakthrough = 0.35 > lambda_repair = 0.10`) so routine repairs cannot crowd out `Crossover` and `Novelty`.
   - `export_masked_sft_dataset(tree)` keeps 100% of the tree (including failed branches and tool errors) in context (`in_context=True`) while setting `loss_mask = 1` strictly on the verified winning lineage.
4. **$2 \times 2$ Joint Model $\times$ Harness Acceptance (`accept_joint_2x2`)**:
   Evaluates $\{\theta_k, \theta_k^+\} \times \{\psi_k, \psi_k^+\}$ on `Q_dev` to catch negative coupling (`has_negative_interference`) and enforces `min_novelty_floor` so candidates that inflate rubric scores at the expense of novelty are rejected.

## 6.7. Lean-DAG-RSI v4 Extensions & Degenerate Minimax KKT Protocol

Five v4 mechanisms in `scripts/engine.py` (verified in Section 15 of `scripts/_selftest.py`):

1. **Multi-Parent Submodular Margin Crossover with AST Collision Penalty (`compute_hyper_crossover`)**: Ranks subsets $S$ of $k \ge 2$ branches over continuous bucket margins $\phi_j(u) \in [0, 1]$ using **Excess Margin Gain** $\frac{1}{M}\sum_j \max_{u \in S} \phi_j(u) - \max_{u \in S} \bar{\phi}(u) - \lambda_{\text{coll}}\sum \operatorname{IoU}_{\text{AST}}(A(u), A(v))$, requiring strictly positive marginal gain per added parent so zero-gain clones and weak passenger nodes are rejected.
2. **Scale-Invariant Epistatic Ridge Pruner (`prune_epistatic_gauss_newton`)**: Normalizes tunable parameters to $[0, 1]^d$ and forms the PSD matrix $G = \frac{1}{M}J^\top J + \frac{1}{M}\sum_{j=1}^M (H^{(j)})^\top H^{(j)}$ to detect coupled nonlinear ridges ($\theta_0/\theta_1 = \text{const}$) as near-zero eigenvectors ($\lambda_m \le 10^{-3}\lambda_{\max}$) and freeze redundant degrees of freedom.
3. **Bucket-Factorized Cold-Start World Model (`predict_bucket_jepa`)**: Predicts per-bucket pass probability from empirical bucket difficulty $(\mu_j, \text{parent\_margin}_j)$ and AST diff novelty without cold-start training.
4. **Prefix-Aligned Contrastive Fork Export (`export_dag_dpo_pairs`)**: Extracts sibling forks $(u \to v^+, v^-)$ where $v^-$ is a symmetry trap, sharp spike (`is_sharp_spike=True`), or clone.
5. **Two-Layer Formal `Lean 4` Certificate Generator (`export_lean4_certificate`)**: Emits zero-`Float`, zero-`sorryAx` integer/rational `Lean 4` theorems verified via `by native_decide` against a static `Defs.lean` specification.

**Degenerate Minimax & Corank-1 KKT Polishing Protocol (Heilbronn / Packing Tasks)**:
- **Step 1 — Floater Separation**: Partition vertices into the **Active Frame** ($\ge 1$ active minimal triangle/constraint) and **Floaters** ($0$ active constraints, e.g., $12+3$ at $n=15$, $15+1$ at $n=16$, and $16+1$ at $n=17$). Optimize floaters independently via LP inside their Voronoi pockets.
- **Step 2 — Never Run Pure Tie-System Gauss-Newton When $\operatorname{rank}(J_{\text{frame}}) < d_{\text{frame}}$**: When parallel chords/trapezoids cause $\operatorname{rank}(J_{\text{frame}}) = d_{\text{frame}} - 1$ (as in $n=13$ and $n=16$), the tie-system $A_1(\boldsymbol{\theta}) = \dots = A_M(\boldsymbol{\theta})$ defines a **1D equiareal curve $\Gamma(s)$**, not an isolated point. Damped Gauss-Newton on $A_1 = \dots = A_M$ projects orthogonally onto the slope of $\Gamma(s)$ at $s_0 \ne 0$, halting $c \cdot s_0^2 \sim 10^{-14}$ below the optimum, while unregularized $(J^\top J)^{-1}$ explodes along the nullspace. Always solve the **full augmented KKT-Newton system** ($A_m(\boldsymbol{\theta}) - \Delta = 0$, $J_{\text{frame}}^\top \boldsymbol{\lambda} = \mathbf{0}$, $\sum \lambda_m = 1$ with $\boldsymbol{\lambda} > 0$) to reach the stationary apex $s = 0$, and report the parameterization-independent KKT curvature invariant $c = -\frac{1}{2}\sum \lambda_m^\star \frac{d^2 A_m}{ds^2}\big|_{\mathbf{v}_0} > 0$. Conversely, when $\operatorname{rank}(J_{\text{frame}}) = d_{\text{frame}}$ with $M = d_{\text{frame}} + 1$ active triangles (as in the $16+1$ configuration at $n=17$, $\Delta_{17}^\star = 0.015873844567914898559\dots$), the frame is rigid and 0-dimensional ($\ker(J_{\text{frame}}^\top) = \operatorname{span}(\boldsymbol{\lambda}^\star)$ with $\boldsymbol{\lambda}^\star > 0$).
- **Step 3 — Multi-Vertex Mitosis ($n-2 \to n$) & Basin Deduplication**: To bypass floater-saturated symmetric frames (such as $n=15$'s $D_3$ $12+3$ trap when searching for $n=16$), combine 1-vertex cascades ($n-1 \to n, n+1 \to n$) with 2-vertex lift/mitosis ($n-2 \to n$) and deduplicate the candidate pool by sorted triangle spectrum.

## 6.8. `SLEEP-DAG-RSI v5`: Parametric Continual Sleep (`HOPE + CMS + SLEEP`) & Universal Pretrained Neural Adapter (`scripts/sleep_dag_rsi.py`)

Unites Google Research's *SLEEP* (`arXiv:2606.03979v2`) and *HOPE / Continuum Memory System (CMS)* (`arXiv:2512.24695`) with `Lean-DAG-RSI` in `scripts/sleep_dag_rsi.py` (re-exported in `scripts/engine.py` and verified in Section 16 of `scripts/_selftest.py`):

1. **Bridge 1 — DAG-Masked NREM Knowledge Seeding (`DAGKnowledgeSeedingConsolidator`)**:
   - Filters Wake trajectories through `export_masked_sft_dataset(dag_tree)` (`loss_mask = 1` strictly on the verified winning lineage, `loss_mask = 0` on dead ends), weights distillation by `compute_provenance_credit` ($\lambda_{\text{breakthrough}} = 0.35 > \lambda_{\text{repair}} = 0.10$), and combines Generalized JSD (`generalized_jsd_loss_and_grad`) + Levenshtein/Semantic LTI rewards (`compute_r_abs`, `compute_r_sem_numpy`) with a contrastive DPO repulsion gradient (`export_dag_dpo_pairs`) and prior-domain Replay Anchor (`97.82%` 5-domain retention, `2.18%` trap contamination).
   - **Token-Band Disjointness Rule**: In multi-domain or multi-timeframe continual learning, assign each domain $k$ its own disjoint token band `base_tok = offset + k * band_width` so DPO repulsion on domain $k$'s trap tokens never collides with winning continuation tokens of domain $k' \ne k$.
2. **Bridge 2 — Error-Fingerprint MoE Expansion & Routing (`ExpandableMoEMLP` + `ExpertFingerprintMeta`)**:
   - Tags every newly unlocked slow low-rank expert $\{A_e, B_e\}$ with its empirical binary `pass_mask` and AST symbol set. During REM dreaming, `select_complementary_crossover_experts()` deterministically pairs orthogonal experts with maximum union gain $\Delta_\cup(e_1, e_2) > 0$ (`100%` complementary hit rate vs `10%` under random MoE mixing).
3. **Bridge 3 — Counterfactual & Flat-Plateau REM Dreaming (`CounterfactualDreamingEngine`)**:
   - Scores candidate dreams via gradient norm $\|g_{\text{DR}}\|_2$ + `LeanDAGWorld` counterfactual replay before LoRA verification, blocks sharp spikes via `evaluate_flat_plateau` (`0%` spike leak), and applies `ReST^EM` updates on verified dreams.
4. **Bridge 4 — Universal Pretrained Neural Adapter, Orchestrator & AST Safety Gate**:
   - **Universal Neural Adapter (`UniversalNeuralSleepAdapter` & `attach_sleep_dag_adapters`)**: Wraps **any already-trained PyTorch `nn.Module` or NumPy/callable backbone** $f_0(x)$ without modifying its frozen base weights $W_0$:
     $$\hat{y}(x) = f_0(x) + \Delta_{\text{HOPE-Fast}}(x) + \Delta_{\text{CMS-MoE-Slow}}(x).$$
     Call `attach_sleep_dag_adapters(pytorch_model, target_layer_names=..., d_low=8)` to freeze all base parameters and attach fast Wake + slow NREM/REM MoE adapters to `nn.Linear` layers, or instantiate `UniversalNeuralSleepAdapter(d_in, d_out)` directly (`wake_adapt_step`, `nrem_consolidate`, `select_complementary_slow_experts`).
   - **AST Policy Safety Gate (`AstPolicySafetyGate.verify_source`)**: Statically parses authored policy/candidate code via `ast.parse()` to block `eval`/`exec`/`os.system`, unrevealed state snooping (`best_so_far`, `_secret`), and hardcoded node IDs (`"b0_win"`).
   - **Time-Series & Financial Discovery Guardrails**:
     - *Phase-Invariant Event-Driven Execution*: Never subsample time-series signals with a fixed modulo grid (`iloc[::step]`), which creates phase-offset luck. Always trigger entries event-driven on `pos[t] != 0` with an explicit non-overlapping `cooldown` (`extract_phase_invariant_trade_indices`).
     - *Fee-to-ATR Wave Horizon ($\text{ATR}(H) \gg c_{\text{fee}}$)*: Align holding horizon $H$ with the timeframe's structural wave (e.g. `48h` on `4h`, `7d` on `1d`) and gate intraday entries so expected wave range exceeds round-trip execution fees by at least $3.5\times$–$10\times$.

## 6.9. `SLEEP-DAG-RSI v6`: Unified Spectral Frame-Lifting, Wald Micro-Sentinel & Neuro-Symbolic Distillation

Unifies early multi-fidelity pruning, autonomous representation lifting, cross-task operator transfer, and NREM neuro-symbolic rule extraction around the empirical Gauss-Newton matrix $G = \frac{1}{M}J^\top J$ and residual vector $e(x)$ (verified in Section 17 of `scripts/_selftest.py`):

1. **Stage 1 — `WaldMicroSentinel` (`5% Adversarial + Stratified Early Veto` in `scripts/engine.py`)**:
   - Evaluates candidate mutations on $S_{\text{sentinel}} = S_{\text{adv}} \cup S_{\text{strat}}$ (`2.5%` worst parent residuals + `2.5%` uniform stratified anchor, `cost ≈ 0.05`) using a sequential lower confidence bound $\operatorname{LCB}(v) = \Delta \mu + 1.96 \cdot \operatorname{SE}(\Delta \mu)$.
   - **Asymmetric Macro-Innovation Exemption**: `crossover` nodes, `frame_transform` nodes, and high-$\beta$ exploration nodes ($\beta \ge 0.8$) evaluate strictly on $S_{\text{strat}}$ (exempting $S_{\text{adv}}$) to prevent killing global reorganizations that trade off the parent's local patch.
   - **One-Way Veto & DPO Routing**: Survivors must pass 100% full-domain evaluation for final ranking (zero slice Goodharting). Aborted candidates receive `fail_class = "sentinel_pruned"`, `meta["sentinel_aborted"] = True`, are excluded from `is_success()`, and feed directly into `export_dag_dpo_pairs()` as negative contrastive targets.
2. **Stage 2 — `SpectralFrameLifter`, `Active-Set Condensation` & `SpectralMetaAtlas` (`scripts/engine.py`)**:
   - `compute_spectral_diagnostics(g_mat, m_buckets)` computes effective rank $r_{\text{eff}}(G) = (\operatorname{Tr} G)^2 / \operatorname{Tr}(G^2)$ (via exact eigendecomposition for $d \le 30$ or Hutchinson Rademacher trace estimation $O(m \cdot d)$ for $d > 30$) and **Regularized Condition Number** $\kappa_{\text{eff}}(G) = \lambda_{\max} / \max(\lambda_{d - \min(M, d)}, 10^{-12})$, avoiding false-positive $\kappa \to \infty$ triggers when $M < d$.
   - When $\kappa_{\text{eff}}(G) \ge 10^6$ or $K=4$ consecutive probes stall on OOS, `SpectralFrameLifter.detect_and_lift()` queries `SpectralMetaAtlas` (which quarantines any operator failing 2 consecutive probes) and instantiates:
     - `EigenRotationTransform`: Preconditioned rotation into Gauss-Newton eigenvectors $V \operatorname{diag}(\lambda_i^{-1/2})$ with nullspace freezing and exact `inverse_projection(y) -> x_orig`.
     - `KktSlackTransform`: Lifts non-smooth max-min tie bottlenecks into smooth slack coordinates $(x, z)$ with `condense_active_set()` to prune inactive slacks ($z_i \gg f_{\min}$) after warmup.
     - `DyadicHomotopyTransform`: Coarsens fine grids $P \to P/2^k$ and prolongates back to $\mathbb{R}^P$ via linear spline homotopy in `inverse_projection`.
3. **Stage 3 — `NeuroSymbolicDistiller` & `numpy_lasso_coordinate_descent` (`Residual-Keep Decomposition` in `scripts/sleep_dag_rsi.py`)**:
   - During Deep NREM in `UniversalNeuralSleepAdapter.nrem_consolidate()`, filters the top-8 active inputs ($d_{\text{active}} \le 8$), builds a bounded AST basis dictionary $\Psi(X)$ ($K \le 64$, depth $\le 2$: linear, $\tanh$, $\operatorname{sign}$, $\operatorname{ReLU}$, $x_i x_j$, $x_i - x_j$), and fits a standalone vectorized pure-NumPy Coordinate-Descent Lasso + active-support OLS polish.
   - Compiles the non-zero support into Python code verified by `AstPolicySafetyGate` and `accept_joint_2x2()`. When $R^2_{\text{OOS}} \ge \tau_{\text{distill}}$, registers an explicit symbolic node `ns_ast_slot<k>` in the DAG and subtracts $f_{\text{AST}}(x)$ from the neural expert's target ($\Delta f_m^{\text{rem}} \leftarrow \Delta f_m - f_{\text{AST}}$), cutting opaque neural residual energy by $>95\%$ with zero loss in expressiveness.

## 6.9.2. `SLEEP-DAG-RSI v7`: Autonomous AST Splicing, Active Boundary Synthesis & Compositional Library Invention

1. **Stage 4 — `AstCodeSplicer` (`Autonomous Multi-Parent AST Structural Splicer` in `scripts/engine.py`)**:
   - Closes the code-generation loop in `compute_hyper_crossover()`: when parent nodes carry `artifact["code"]` and `artifact["base_code"]`, `AstCodeSplicer.splice_parents()` parses their ASTs (`ast.parse`), computes structural AST touch IoU, merges orthogonal helper functions and intra-function variable assignments (`structural_ast_merge`) or synthesizes a verified ensemble dispatcher (`ensemble_dispatch_merge`), and validates the output with `compile(..., "<ast_splice>", "exec")`.
2. **Stage 5 — `ActiveBoundarySynthesizer` (`Active Committee Disagreement & Boundary Probing` in `scripts/engine.py`)**:
   - Combines **Bichromatic Chord Bisection** (locating anchor pairs where competing policies/models flip sign) with **Projected Gradient Ascent** on committee variance $\operatorname{Var}_k(f_k(x))$ inside the local feature trust-region $\mathcal{B}(X_{\text{anchor}}, r \cdot \sigma)$, amplifying disagreement by $>10\times$ over passive historical replay and feeding boundary counterexamples directly into `NeuroSymbolicDistiller`.
3. **Stage 6 — `SymbolicLibrary` & `LibraryPrimitive` (`DreamCoder-Style Hierarchical AST Subroutine Invention` in `scripts/sleep_dag_rsi.py`)**:
   - Whenever `NeuroSymbolicDistiller` accepts a multi-term symbolic motif ($\ge 2$ active terms, $R^2_{\text{OOS}} \ge 0.30$, non-collinear $|\cos \theta| \le 0.80$), `SymbolicLibrary.register_primitive()` abstracts and normalizes it into a reusable subroutine `lib_k(x)`.
   - Subsequent sleep cycles automatically screen multiplicative interactions against `lib_k(x)` and expand the dictionary $\Psi(X)$ with `lib_k(x)`, `math.tanh(lib_k(x))`, `(lib_k(x) * x[i])`, and `(lib_k(x) * sign(x[i]))`, cracking depth-3/4 compositional laws ($R^2_{\text{OOS}} = 1.000$) where flat depth-2 distillation fails ($R^2_{\text{OOS}} = 0.000$).

## 6.10. Extremal Step-Function & Non-Smooth Minimax Protocol ($C_1, C_2, \Phi_3$)

1. **Sphere-Lagrangian Exact Curvature + Non-Destructive Micro-Wakeup ($a_i \ge 0$ Simplex Tasks)**:
   - On simplex-constrained step functions $a_i = z_i^2 / P$ with $\sum_{i=0}^{P-1} z_i^2 = P$, the Lagrangian is $\mathcal{L}(z, \mu) = L_\beta(z^2) - \mu \sum_i (z_i^2 - 1)$ with KKT multiplier $\mu = \frac{a^\top g_a}{P}$.
   - Use exact KKT slack curvature `torch.diag(2.0 * torch.clamp(g_a - mu, min=0.0))` on the Gauss-Newton Hessian diagonal instead of `2.0 * g_a` (which injects artificial $+2\mu I$ damping across active support bins).
   - Apply a non-destructive micro-wakeup $z_i = 10^{-4}$ ($a_i = 10^{-8}$) on step 0 of each $\beta$ stage strictly when $|z_i| < 10^{-4}$ and $g_{a,i} < \mu - 10^{-5}$, releasing false-zero bins from the $z_i = 0$ saddle point without perturbing the active roof.
2. **Dyadic Multi-Scale Scar Healing vs. Signed Newton-Boltzmann Polish**:
   - On non-negative step functions ($C_1$), high $\beta$ or white pixel noise at coarse resolution $P_0$ punches 1-bin zero holes that freeze into $(P/P_0)$-bin canyons under dyadic upsampling (`np.repeat(a, 2)`). Heal coarse scars across dyadic scales ($P = 160 \to 320 \to 640 \to 1280 \to 2560$) via edge-preserving Gaussian smoothing (`sm[:k] = np.maximum(sm[:k], a[:k])` on boundary singularities) paired with boundary diffusion and a resolution-matched $\beta(P)$ schedule ($C_1 \le 1.50382279$).
   - On unconstrained signed step functions ($\Phi_3$), preserve sharp sign-change crossings (`0` Gaussian smoothing) and run full analytical Newton-LM ($H_{\text{cone}} + H_{\text{quad}}$) up to $\beta = 5 \times 10^6$ followed by FFT-based `Boltzmann Roof-Bundle Subgradient` polish ($\Phi_3 \le 1.45705851$).
3. **Kronecker Big-Integer Polynomial Squaring for Exact Rational Convolution Verification**:
   - Certify non-negative step-function autoconvolutions in 100% exact `fractions.Fraction` at denominators $10^{16}$ and $10^{50}$ across all $2P-1$ shifts in $<0.15\text{s}$ by packing integer numerators $q_i$ into a single big integer $Q = \sum_{i=0}^{P-1} q_i 2^{B i}$ with limb width $B = \lceil \log_2(P \max q_i^2) \rceil + 8$ and computing $Q^2$ in C-level Karatsuba/Toom-Cook arithmetic (`verify_c1_exact_kronecker_fraction`).

## 7. The closed loop

`scripts/loop.py` and `scripts/sleep_dag_rsi.py` run the cycle; `scripts/dreamrsi.py` drives it: `init`, `round`,
`prompt`, `apply`, `status`, `trees`, `ideate`, `sleep`, `demo`. Loop state lives next to the pool as
`loop_state.json`. **Policies persist as data, never as code** (kind + parameters), so
state can be diffed, reviewed and rolled back, and nothing executes from the pool.

Candidates come from two sources, and you want both: **parametric** (small variations
around the incumbent, needs no model) and **authored** (`prompt` renders
`prompts/policy_improvement.md` with measured facts; a coding agent returns JSON for
`apply`; a new policy *kind* is what makes the recursion real).

CLI transcript, loop-state detail, and the verified `demo` output:
`references/closed-loop-walkthrough.md`.

## 8. Runbook

1. Build a **deterministic** evaluator you can call repeatedly. Record baseline `s0`.
2. Open the tree with a root node. Give each structural direction its own `branch` id,
   and hang a chain off the root per direction — a tree whose frontier always holds a
   single node cannot discriminate between policies at all.
3. Run rounds in batches of `W` (2–4 is usually right). One exploit, one explore, at
   most one recover. Record every attempt as a node, including failures.
4. Widen when branches plateau or siblings are minor variations of each other.
   Deepen when a branch made a large gain and headroom is visible. Hold when the
   budget is nearly spent — exploit the top-2 anchors only.
5. After 5–10 attempts: save the tree, then `round` to replay and reselect the policy.
6. Persist the finished tree. Starting a future task in the same domain, load history
   first and pre-calibrate `β` and batch width from it.

## 9. Verifying the engine

`scripts/_selftest.py` asserts the properties this skill depends on: a non-numeric
domain works, selection is monotone with the incumbent present, replay is
deterministic, observability is prefix-only, recovery is adapter-driven, the pool is
append-only and multi-domain, policies round-trip through JSON, proposal parsing is
tolerant, and the closed loop persists history. Run it after any edit:

```
python scripts/_selftest.py      # must print "RESULT: all checks passed"
```

## 10. Honest limits

- The paper has **no** limitations, discussion, or future-work section; nothing here
  is a summary of author-acknowledged caveats.
- Its own headline result is mixed per-dataset even where the aggregate improves;
  the reported aggregate is dominated by one benchmark. Do not assume uniform gains.
- The paper's artifact is executable policy code scored by a numeric replay reward.
  This skill now closes that loop at the *parametric* level and provides the prompt
  hand-off for the authored level, but it does not itself write new policy code — an
  external coding agent must do that, and its output must satisfy §5's constraints.
- Monotonicity is a guarantee about **replay score on recorded history**, not about
  the real task. A policy can improve on every recorded tree and still be worse on a
  task whose shape differs from anything recorded.
- `β₁` and `β₂` are swept, not learned: the sweep is a grid over a handful of values,
  which is the cheapest form of dreaming and not a substitute for authoring policies.

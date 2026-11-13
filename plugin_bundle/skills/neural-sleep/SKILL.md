---
name: neural-sleep
description: Continual online adaptation and neuro-symbolic law extraction for pretrained PyTorch or NumPy neural networks (SLEEP-DAG-RSI v8). Use when wrapping frozen models with fast online memory (FALCON-2 NLMS), consolidating sequential domains without catastrophic forgetting (DAG-KS + Max-Tilted objective), reclaiming 100%-failed batches via verifier feedback (SDPO IRLS Self-Teacher), challenging ensemble blind spots (VeriHarness), or distilling black-box neural residuals into verified Python AST formulas (NeuroSymbolicDistiller + SymbolicLibrary).
---

# Neural-Sleep (`SLEEP-DAG-RSI v8`): Continual Adaptation & Neuro-Symbolic Distillation

Implementation lives at `~/.gemini/config/plugins/autonomous-research/skills/dream-rsi/scripts/sleep_dag_rsi.py` (`DREAM_SCRIPTS`).

## 1. When to Activate
Use this skill when:
1. A pretrained PyTorch `nn.Module` or NumPy model must adapt to streaming distribution shifts in milliseconds without corrupting base weights $W_0$ or exploding on heavy-tailed spikes.
2. Sequential multi-domain learning suffers from catastrophic forgetting or dead-end trajectory contamination (`Vanilla SLEEP` trap).
3. Black-box neural adapter weights need to be decomposed into human-readable, safety-verified symbolic Python formulas (`f_AST(x)`) of depth $2\dots 3+$ while preserving residual accuracy.

---

## 2. Four-Phase Execution Workflow (`Wake -> NREM -> REM -> 2x2 Gate`)

### Step 1: Wrap the Frozen Backbone (`Wake` Fast Memory via `FALCON-2`)
Add `DREAM_SCRIPTS` to `sys.path` and wrap either a PyTorch model or a NumPy feature stream without touching $W_0$:
```python
from sleep_dag_rsi import UniversalNeuralSleepAdapter, SymbolicLibrary, attach_sleep_dag_adapters

# Option A: Wrap linear layers of an existing PyTorch nn.Module in-place (freezes base weights)
adapters = attach_sleep_dag_adapters(pytorch_model, target_layer_names=["fc1", "head"], d_low=8)

# Option B: Standalone streaming adapter y_hat(x) = f_0(x) + Delta_Fast(x) + Delta_Slow_MoE(x)
adapter = UniversalNeuralSleepAdapter(d_in=X.shape[1], d_out=Y.shape[1], d_low=8, seed=42)
```
During streaming `Wake`, adapt online with **Prefix-Aligned `FALCON-2` NLMS** (`use_falcon2=True`), which applies denominator floor $`\max(\lVert\tilde{k}_t\rVert_2^2, \lVert k_t\rVert_2^2)`$, cosine prefix gating $`\alpha_{\text{prefix}}(t)`$, and Huber residual clipping so $10^3\times$ input/target spikes cannot blow up fast weights:
```python
wake_step = adapter.wake_adapt_step(X_batch, Y_target_residual, lr=0.08, use_falcon2=True)
```
*Completion criterion:* Fast-weight loss decreases monotonically on clean streams and `fast_B_norm` stays bounded under outlier spikes.

### Step 2: Consolidate into Slow MoE Experts (`NREM`: `DAG-KS` + `TTT Max-Tilt` + `SDPO`)
At domain or epoch boundaries, consolidate fast associative memory into slow low-rank `CMS-MoE` experts:
- **Positive Lineage (`DAG-KS` + `TTT-Discover`):** Weight winning trajectories by entropic tilt $w_i(\tau) \propto \exp(\tau(R_i - R_{\max})/s_R)$ (`tau=4.0`) and provenance credit, keeping `loss_mask = 0` on raw unverified dead-ends.
- **100%-Dead-End Batches (`SDPO` Self-Teacher with Tukey Biweight IRLS):** When every attempt in a batch failed (`sum(loss_mask) == 0`) but the verifier provides directional residuals $\Delta_{\text{verifier}}(x_i, f_i)$, pass `verifier_residuals` to `nrem_consolidate(..., use_sdpo_self_teacher=True)`. Coordinate-wise MAD/IRLS suppresses up to `25%` adversarial noise and automatically aborts/rolls back against the validation anchor if corruption exceeds `50%` or residuals are zero.
- **Multi-Domain Token Disjointness:** When training `SleepDAGRSIOrchestrator` on discrete token sequences across $K$ domains, assign disjoint token bands `offset + k * band_width` per domain so DPO repulsion never collides across domains.

### Step 3: Extract Compositional Symbolic Laws (`NeuroSymbolicDistiller` + `SymbolicLibrary`)
Decompose newly learned neural residuals into verified Python AST code with **Residual-Keep** ($\Delta f_m^{\text{rem}} \leftarrow \Delta f_m - f_{\text{AST}}$):
```python
library = SymbolicLibrary(max_primitives=12)
distiller = NeuroSymbolicDistiller(r2_min_oos=0.25, library=library, auto_invent_primitives=True)

rep = distiller.distill_residual(
    X_train, delta_Y_train,
    X_oos=X_val, delta_Y_oos=delta_Y_val,
    reconstruct_uncorrupted_target=True,
)
```
- **Depth-3+ Law Discovery:** In Cycle 1, `_discover_nonlinear_inner_submotif()` runs 5 Gauss-Newton steps to isolate inner arguments (e.g. $`z = 1.5 x_0 x_1 - 0.8\mathrm{sign}(x_2)`$) into `lib_0(x)`. In Cycle $`\ge 2`$, `reconstruct_uncorrupted_target=True` rebuilds the full target $Y_{\text{full}}$ so outer saturating links (`tanh(scale * lib_0(x) + offset)`) recover $R^2_{\text{OOS}} \to 1.0$.
- **Mandatory Safety Gate:** Every extracted `rep["ast_code"]` is checked by `AstPolicySafetyGate` (`rep["ast_safety_valid"] == True`) and `accept_joint_2x2` before subtracting from slow weights.
*Completion criterion:* Accepted symbolic rule achieves $R^2_{\text{OOS}} \ge \tau_{\text{distill}}$ and reduces unexplained neural residual energy without degrading validation score.

### Step 4: Challenge Blind Spots & Cross-Pollinate Experts (`REM` + `VeriHarness`)
1. Run `ActiveBoundarySynthesizer().synthesize_dual_head_verification(committee_fns, X_anchor, oracle_fn=...)`:
   - **Head A (`Disagreement Resolver`):** Ascends committee variance $`\mathrm{Var}_k(f_k(x))`$ to resolve inter-expert disputes.
   - **Head B (`Consensus Challenger`):** Minimizes unanimous confidence margin $`\lvert\bar{f}(x)\rvert / \sqrt{\mathrm{Var}_k + \epsilon}`$ to expose collective blind spots where all experts agree yet flip against the oracle.
2. Pair complementary slow experts via `select_complementary_crossover_experts()` (maximizing non-overlapping `pass_mask` union gain, skipping clones) and accept updates only through `accept_joint_2x2`.

---

## 3. CLI Smoke Test & Self-Test
```bash
python "$DREAM_SCRIPTS/dreamrsi.py" sleep --demo
python "$DREAM_SCRIPTS/_selftest.py"
```

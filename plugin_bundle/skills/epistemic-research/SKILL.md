---
name: epistemic-research
description: Rigorous empirical research and optimization discipline based on Open-Endedness Bench (OEB). Use when running experiments, tuning hyperparameters, benchmarking algorithms, or claiming records: enforces pre-flight 2*sigma noise calibration, causal diagnosis before edits, a hard 3-try cap per idea, mandatory 40% second-half exploration, and zero reward hacking.
---

# Epistemic Research Protocol (`OEB`)

Operationalizes *Open-Endedness Bench: Measuring Epistemic Process from Agent Records* (`arXiv:2610.02588`).
**Foundational rule:** *Semantics proposes, executed output disposes.* Unlogged claims of improvement are noise.

## 1. Pre-Flight Calibration (Before Editing Any Code)
Answer and record three numbers before try #1:
1. **Exact Metric & Command:** Name the scalar metric and exact command that prints it.
2. **Noise Floor ($`\sigma_{\text{noise}}`$):** Run the unmodified baseline **twice**; set $`\sigma_{\text{noise}} = \lvert\text{Score}_1 - \text{Score}_2\rvert`$. Any candidate gain $`\Delta \le 2\sigma_{\text{noise}}`$ is `INCONCLUSIVE_NOISE` — never build follow-up branches on top of it.
3. **Budget Midpoint ($B / 2$):** Fix total attempt budget $B$. Reserve `40%` of the second half ($t > B/2$) for brand-new orthogonal mechanisms never tested in Phase 1.

## 2. Four Execution Invariants (`E1–E4`)
- **`E1` Grounded Claims:** Quote exact stdout numbers for every diagnosis. Extrapolate strictly within tested bounds.
- **`E2` Launch -> Check Pairing:** Read logs and classify every run (`CONFIRMED > 2σ`, `CONTRADICTED`, or `INCONCLUSIVE_NOISE`) before launching the next attempt.
- **`E3` Causal Revision & Hard 3-Try Pivot:** State an explicit failure cause (`Cause: ...`) before changing code. After **3 tries** on one mechanism without $> 2\sigma_{\text{noise}}$ gain, mark it `EXHAUSTED` and pivot to an orthogonal mechanism.
- **`E4` Zero-Gaming & Honest Attribution:**
  - Keep held-out `Blind Vault` splits sealed until a single final evaluation (`max_calls = 1`).
  - Run genuine algorithms for every intermediate branch — never fabricate ablation steps by adding noise or shifts to a champion array.
  - Print exact `.npy`/`.json` arrays to `stdout` in the active turn before citing numerical coordinates or weights.
  - Classify benchmark claims against live registries (`math.tejstead.com`, published baselines) into: (1) *New Topological Basin*, (2) *Exact KKT Stationary Refinement* ($s \to 0$) *Inside a Known Basin* (crediting the basin discoverer and reporting curvature $c > 0$), or (3) *Independent From-Scratch Rediscovery & Certificate*.

## 3. Running Epistemic Ledger (`RESEARCH_LEDGER.md`)
Log every step in a compact 5-column table:
`| Step | Type (PROPOSITION / LAUNCH / CHECK / PIVOT) | Action / Command | Observed Metric | Verdict (TRY k/3) |`

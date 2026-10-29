# Lean-DAG-RSI (`Auto-Dream-RSI` v3, v4 & `SLEEP-DAG-RSI` v5–v7)

**Autonomous Scientific & Engineering Discovery via Error-Fingerprint DAG Crossover, Spectral Frame-Lifting, Autonomous AST Splicing, Compositional Neuro-Symbolic Sleep (`SLEEP + HOPE + CMS`), and Formal/Exact-Rational Verification.**

* **Full Papers (Russian, PDF & Markdown):**
  * [`Полная_Статья_Lean_DAG_RSI_v3.pdf`](Полная_Статья_Lean_DAG_RSI_v3.pdf) & [`Полная_Статья_Lean_DAG_RSI_v3.md`](Полная_Статья_Lean_DAG_RSI_v3.md)
  * [`lean_dag_rsi_v5_sleep_paper.pdf`](lean_dag_rsi_v5_sleep_paper.pdf) & [`Полная_Статья_Lean_DAG_RSI_v5_SLEEP.md`](Полная_Статья_Lean_DAG_RSI_v5_SLEEP.md) ([`sleep_dag_rsi_unification_report.md`](sleep_dag_rsi_unification_report.md), [`concepts_brief.md`](concepts_brief.md))
* **Self-Contained Core Engine (`18` self-test suites + `SLEEP-DAG-RSI` v7 parametric & neuro-symbolic sleep):**
  * [`dream_rsi_scripts/engine.py`](dream_rsi_scripts/engine.py), [`dream_rsi_scripts/_selftest.py`](dream_rsi_scripts/_selftest.py), [`sleep_dag_rsi.py`](sleep_dag_rsi.py), [`battle_test_sleep_dag_v6.py`](battle_test_sleep_dag_v6.py), & [`run_sleep_dag_experiments.py`](run_sleep_dag_experiments.py)
* **Real Bitcoin Multi-Timeframe Continual Sleep & Trader Suite (`149,953` gap-free `BTCUSDT` bars):**
  * [`btc_sleep_dag_rsi/`](btc_sleep_dag_rsi/) ([`btc_sleep_dag_engine.py`](btc_sleep_dag_rsi/btc_sleep_dag_engine.py), [`BTC_TRADER_ANALYSIS_REPORT.md`](btc_sleep_dag_rsi/BTC_TRADER_ANALYSIS_REPORT.md))
* **Autonomous 8-Cartridge Discovery Harness:**
  * [`auto_dream_rsi.py`](auto_dream_rsi.py)
* **Heilbronn Triangle $n=13, 14, 15, 16, 17$ 500-Digit & $10^{100}$-Rational Certificate Package (`tejstead/heilbronn-site`):**
  * [`heilbronn_pr_package/`](heilbronn_pr_package/) ([`PR_DESCRIPTION.md`](heilbronn_pr_package/PR_DESCRIPTION.md), [`verify_pr_package.py`](heilbronn_pr_package/verify_pr_package.py))
* **Two-Layer Formal `Lean 4` Rational Certificates (Zero `Float`, Zero `sorryAx`):**
  * [`lean4_certificates/Defs.lean`](lean4_certificates/Defs.lean) & [`lean4_certificates/Cert_Heilbronn_13_14.lean`](lean4_certificates/Cert_Heilbronn_13_14.lean)

---

## Key Results Summary

### 1. Mathematical Audit & Structural Fixes over `Dream-RSI` (2026) and `DiscoveryRSI` (`arXiv:2610.23839`)
1. **Fixing the Probe-Spamming Bug in `Dream-RSI` Eq. 1:** In `Dream-RSI` Eq. 1 ($V = \max s_v - \beta_1 N + \beta_2 N / k^\star$), taking the partial derivative with respect to total probes $N$ yields $\partial V / \partial N = -\beta_1 + \beta_2 / k^\star > 0$ whenever $k^\star < \beta_2/\beta_1$, rewarding policies that spam useless parallel probes. Replaced by the strictly monotone latency penalty $V = \max s_v - \beta_1 N - \beta_2 k^\star$ (`engine.py:622`).
2. **Resolving Support Censorship & Tree Isolation (`LeanDAGPortfolio` + `compute_error_fingerprint`):** Replaces scalar tree continuation with a Directed Acyclic Graph (`DAG`) that cross-pollinates branches $(u, v)$ strictly when their empirical binary pass masks `pass_mask` $\in \{0, 1\}^M$ have positive union gain ($\Delta_{\cup}(u, v) > 0$), plus multi-parent submodular margin crossover (`compute_hyper_crossover` in v4).
3. **Neuro-Symbolic Split & RRSI Complexity Pruner (`prune_tunable_params` & `prune_epistatic_gauss_newton`):** Separates discrete structural ideation from inner-loop continuous optimization (`HiGHS LP`, `SLSQP`, `Boltzmann Roof-Bundle Subgradient`), followed by greedy backward parameter elimination and 20-direction $\pm 5\%$ flat-plateau jitter verification (`evaluate_flat_plateau`).
4. **Resolving the "Cautious Valedictorian Paradox" in `DiscoveryRSI` Eq. 17:** Sets $\lambda_{\text{breakthrough}} = 0.35 > \lambda_{\text{repair}} = 0.10$ and enforces a $2 \times 2$ joint `Model × Harness` novelty floor (`accept_joint_2x2`).

---

### 2. Extremal Mathematics & Open Problems (Zero-External-Seed Provenance)

| Problem / Cartridge | Result | Verification & Comparison |
| :--- | :--- | :--- |
| **Heilbronn Triangle $n = 13$** ([`n13_exact_500dps.json`](heilbronn_pr_package/n13_exact_500dps.json)) | $\Delta_{13}^\star = \mathbf{0.026556528919775580138364411226\dots}$ | **`+3.731566397517e-14`** over Tej Stead (July 2026) published `float64` literals (`0.026556528919738264474389245471`); $\operatorname{rank}(J_{13}) = 19 < 20$, KKT curvature invariant $c_{13} = 0.036078780238 > 0$; 3 exact parallel chords ($u_0 \equiv u_9, u_2 \equiv u_5, v_{11} \equiv v_{12}$); $\deg P(\Delta_{13}^\star) > 78$. |
| **Heilbronn Triangle $n = 14$** ([`n14_exact_500dps.json`](heilbronn_pr_package/n14_exact_500dps.json)) | $\Delta_{14}^\star = \mathbf{0.023775773107573057671296916141\dots}$ | Independent from-scratch rediscovery of David W. Cantrell (June 2007), certified to 500 digits & $10^{100}$ exact rationals (`23` active triangles, exact chord $u_6 \equiv u_9$, $\deg P(\Delta_{14}^\star) > 78$). |
| **Heilbronn Triangle $n = 15$** ([`n15_exact_500dps.json`](heilbronn_pr_package/n15_exact_500dps.json)) | $\Delta_{15}^\star = \mathbf{0.021090769460266687959320407664\dots}$ | Independent $14^\star \to 15$ cascade rediscovery & 500-digit verification of the $12+3$ floater split and degree-5 minimal polynomial $5250987 \Delta_{15}^5 + 1609650 \Delta_{15}^4 + 161469 \Delta_{15}^3 + 5438 \Delta_{15}^2 - 13 \Delta_{15} - 4 = 0$ (Cantrell 2007; Stead 2026), $\|P_{15}(\Delta_{15}^\star)\| = 2.78 \times 10^{-503}$. |
| **Heilbronn Triangle $n = 16$** ([`n16_exact_500dps.json`](heilbronn_pr_package/n16_exact_500dps.json)) | $\Delta_{16}^\star = \mathbf{0.018355546355461140412325363653\dots}$ | **`+5.769027637240e-14`** over Fable 5.1 & Shengtong Zhang (Sept 2026) 30-digit damped Gauss-Newton tie-system solution (`0.0183555463554034501359529610459`); proves the **$15 + 1$ Floater Split** (`p_13` has 0 active triangles, clearance `0.018532001428`) and $\operatorname{rank}(J_{\text{frame}}) = 23 < 24$ (1D equiareal curve $\Gamma_{16}(s)$ with KKT curvature $c_{16} = 0.011234563954 > 0$); exact hypotenuse-parallel chord $(u_1+v_1) \equiv (u_{10}+v_{10})$. |
| **Heilbronn Triangle $n = 17$ ($16+1$ Basin)** ([`n17_exact_500dps.json`](heilbronn_pr_package/n17_exact_500dps.json)) | $\Delta_{17}^\star = \mathbf{0.015873844567914898558694564865\dots}$ | **`+3.4455%`** over previous local $n=17$ lift (`0.0153451315213836`); certifies the exact **$16 + 1$ Floater Split** stationary KKT point of the $16^\star \to 17$ lift (`27` active triangles, full-rank $27 \times 26$ frame Jacobian, interior floater `p_7` clearance `0.016314496144`). |
| **SLEEP-DAG-RSI v5–v7 Continual Sleep & AST Invention** ([`sleep_dag_rsi.py`](sleep_dag_rsi.py), [`battle_test_sleep_dag_v6.py`](battle_test_sleep_dag_v6.py)) | **`97.63%`** D1 Retention / **`R^2_{\text{OOS}} = 1.000`** Depth-3 AST | Unites `SLEEP + HOPE + CMS` (`arXiv:2606.03979v2`) with `Lean-DAG-RSI`: cuts dead-end trap contamination from `39.49%` to **`2.18%`**, achieves **`100%`** complementary MoE hit rate, reduces probe cost by **`39.4%`** (`WaldMicroSentinel`), escapes $\kappa = 10^8$ Rosenbrock (`SpectralFrameLifter`), autonomously splices AST code (`AstCodeSplicer`), amplifies boundary disagreement by **`10.39x`** (`ActiveBoundarySynthesizer`), and invents reusable depth-3+ subroutines (`SymbolicLibrary`). |
| **2nd Autocorrelation Inequality $C_2$** ([`autocorr_c2_p20256_champion.npy`](autocorr_c2_p20256_champion.npy)) | $C_2 \ge \mathbf{0.942158772814}$ ($P = 20\,256$) | Beats direct step-function searches of DeepMind AlphaEvolve v1 (`0.9053`), Boyer & Li (`0.901562`), and Jaech & Joseph 2025 (`0.94136` at $P=2399$) via GPU `Boltzmann Roof-Bundle Subgradient`. |
| **1st Autocorrelation Inequality $C_1$** ([`autocorr_c1_champion.npy`](autocorr_c1_champion.npy)) | $C_1 \le \mathbf{1.503822792008}$ ($P = 2\,560$) | Derived from scratch via Sphere-Lagrangian Exact Curvature + Dyadic Multi-Scale Scar Healing; certified in exact `fractions.Fraction` at $10^{16}$ and $10^{50}$ via Kronecker big-int polynomial squaring; beats Matolcsi & Vinuesa (`1.5098`) and Google DeepMind AlphaEvolve v1 (`1.5053` by `-0.00147721`). |
| **3rd Autocorrelation Inequality $\Phi_3$** ([`phi3_best_candidate.npy`](phi3_best_candidate.npy)) | $\Phi_3 \le \mathbf{1.457058511867}$ ($P = 3\,200$) | Derived via analytical PSD Gauss-Newton preconditioner $H_{\text{GN}}(a) = \beta J^\top(\operatorname{diag}(u) - w w^\top) J$ + High-Beta Exact Newton + Boltzmann Roof-Bundle Polish; beats Cilleruelo–Ruzsa–Vinuesa (`1.4581`), OpenEvolve (`1.4600`), EvoX (`1.4589`), and ShinkaEvolve (`1.4578`). |
| **AlgoTune RBF Kernel & Weibull Bin Packing** | **`33.36x`** speedup / **`0.3690%`** $L_2$ excess | Beats SciPy reference by `33.36x` (C++ Look-Ahead Anderson Lasso beats Celer by `2.65x`) and FunSearch Weibull bin packing (`0.68%` $\to$ `0.3690%`). |

---

## Quick Verification Commands

```bash
# 1. Verify all 18 engine self-test suites (v1..v4 + SLEEP-DAG-RSI v5..v7)
python dream_rsi_scripts/_selftest.py

# 2. Run the SLEEP-DAG-RSI v6/v7 4-Trial Battle Test (WaldSentinel, SpectralFrameLifter, AstCodeSplicer, SymbolicLibrary)
python battle_test_sleep_dag_v6.py

# 3. Verify the 500-digit & 10^100-rational Heilbronn PR package (n=13, 14, 15, 16, 17)
python heilbronn_pr_package/verify_pr_package.py

# 4. Run the standalone Zero-Gaming verifier for Cartridges 3, 7 & 8 (C1, C2, Phi3, Heilbronn n=13, 14)
python verify_extremal_cartridges_7_8.py

# 5. Run the extremal math discovery cartridges (Autocorrelation C1, C2 & Heilbronn n=13, 14)
python auto_dream_rsi.py --task open-autocorrelation-c1
python auto_dream_rsi.py --task extremal-math
```

# Extremal Geometry, Step Convolutions & Formal `Lean 4` Certificates

Disclosed domain reference for `dream-rsi`. Load only when solving packing, minimax geometry (Heilbronn), autoconvolution bounds ($C_1, C_2, \Phi_3$), or emitting formal proofs.

## 1. Degenerate Minimax & Corank-1 KKT Polishing (Heilbronn / Packing)
1. **Floater Separation:** Partition points into the **Active Frame** ($\ge 1$ active minimal constraint) and interior **Floaters** ($0$ active constraints, e.g. $12+3$ at $n=15$, $15+1$ at $n=16$, $16+1$ at $n=17$). Optimize floaters independently via LP inside their Voronoi pockets.
2. **Corank-1 Equiareal Manifold Trap ($`\mathrm{rank}(J_{\text{frame}}) = d_{\text{frame}} - 1`$):**
   - When parallel chords/trapezoids cause corank 1 (as in Heilbronn $n=13$ and $n=16$), the tie-system $A_1(\boldsymbol{\theta}) = \dots = A_M(\boldsymbol{\theta})$ defines a **1D curve $\Gamma(s)$**, not an isolated point.
   - Damped Gauss-Newton on $A_1 = \dots = A_M$ projects orthogonally onto $\Gamma(s)$ at $s_0 \ne 0$, halting $c \cdot s_0^2 \sim 10^{-14}$ below the optimum.
   - Solve the **full bordered KKT-Newton system** ($A_m(\boldsymbol{\theta}) - \Delta = 0$, $J_{\text{frame}}^\top \boldsymbol{\lambda} = \mathbf{0}$, $\sum \lambda_m = 1$, $\boldsymbol{\lambda} > 0$) in high precision (`decimal.Decimal` 500+ digits) to slide to the unique apex $s = 0$, and compute curvature $c = -\frac{1}{2}\sum \lambda_m^\star \frac{d^2 A_m}{ds^2}\big|_{\mathbf{v}_0} > 0$.
3. **Multi-Vertex Mitosis ($n-2 \to n$):** Combine 1-vertex cascades ($n\pm 1 \to n$) with 2-vertex mitosis ($n-2 \to n$) and deduplicate basins by sorted triangle-area spectrum.

## 2. Step-Function Autoconvolutions ($C_1, C_2, \Phi_3$)
1. **Exact Simplex KKT Curvature ($a_i = z_i^2 / P$, $\sum z_i^2 = P$):** Use `2.0 * clamp(g_a - mu, min=0.0)` with $\mu = a^\top g_a / P$ on the Gauss-Newton diagonal and apply non-destructive micro-wakeup $z_i = 10^{-4}$ when $|z_i| < 10^{-4}$ and $g_{a,i} < \mu - 10^{-5}$.
2. **Dyadic Scar Healing vs. Signed Polish:** Heal coarse 1-bin holes across dyadic upsampling ($P = 160 \to 2560$) on non-negative $C_1$ via boundary-preserving smoothing; on signed $\Phi_3$, use zero smoothing + analytical Newton-LM + FFT Boltzmann Roof-Bundle subgradient.
3. **Kronecker Big-Int Exact Rational Verification:** Pack integer numerators $q_i$ into $Q = \sum q_i 2^{B i}$ ($B = \lceil \log_2(P \max q_i^2)\rceil + 8$) so Python's Karatsuba $Q^2$ certifies all $2P-1$ shifts in exact `fractions.Fraction` at $10^{16}$ and $10^{50}$ denominators in $< 0.15\text{s}$.

## 3. Two-Layer Formal `Lean 4` Certificate (`export_lean4_certificate`)
Call `engine.export_lean4_certificate(theorem_name, points_num_den, scale_denom, bound_num, bound_den)` to emit:
- `Defs.lean`: static integer determinant specification over `Int` (zero `Float`, zero `sorryAx`).
- `Cert_<name>.lean`: theorem proved via `by native_decide`, pre-verified in Python exact integers (`verified_exact_int == True`, `has_sorry == False`).

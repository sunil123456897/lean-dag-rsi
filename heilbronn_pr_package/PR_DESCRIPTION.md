# Pull Request: Resolving the 1D Equiareal Tie-System Trap for Triangle $n=13$ ($+3.7316 \times 10^{-14}$) and $n=16$ ($+5.7690 \times 10^{-14}$), $15+1$ Floater Split at $n=16$, and 500-Digit Certificates ($n=13\dots 16$)

## Executive Summary: Why "Exact Tie-System" Gauss-Newton Stopped $10^{-14}$ Below the True Optimum at $n=13$ and $n=16$

On the current `math.tejstead.com/heilbronn/triangle/16/` page, the record `0.0183555463554034501359529610459` (discovered by *Fable 5.1 with Shengtong Zhang*, September 2026) is described as:
> *"tie system refined to 80 digits by damped Gauss-Newton... 2026-09-15 coordinates replaced by the exact tie-system solution (30 decimals)"*.

Similarly, on `/triangle/13/`, the record `0.026556528919738264474389245471` (discovered by *Tej Stead*, July 2026) has 20 tied minimal triangles.

This PR proves a fundamental geometric property of both chambers that explains why **solving the area-equality tie system $A_1(\boldsymbol{\theta}) = A_2(\boldsymbol{\theta}) = \dots = A_M(\boldsymbol{\theta})$ via damped Gauss-Newton does NOT yield a 0-dimensional isolated point, but instead projects onto the slope of a 1-dimensional equiareal manifold $\Gamma(s)$**:

1. **Under-constrained / Corank-1 Equiareal Manifold $\Gamma(s)$:**
   - **At $n = 16$ (`15 + 1` Floater Split):** Out of the 16 points, **1 point is a strict interior floater** (`vertex 1` in the site's 0-indexed table / `p_13` in `n16_exact_500dps.json`), participating in **zero** of the 24 active minimal triangles (with Voronoi pocket clearance `0.0185320014278944`, `+0.00017646` above $\Delta_{16}^\star$). The remaining **15 frame vertices** have $2 \times 15 - 6 = 24$ free coordinates and carry all $M = 24$ active minimal triangles, imposing only $M - 1 = 23$ area-equality constraints $A_m(\boldsymbol{\theta}) - A_1(\boldsymbol{\theta}) = 0$.
   - **At $n = 13$:** The 13 vertices have $2 \times 13 - 6 = 20$ free coordinates and $M = 20$ active minimal triangles, imposing $M - 1 = 19$ area-equality constraints.
   - In both $n = 13$ ($d_{\text{free}} = 20$) and $n = 16$ ($d_{\text{free}} = 24$), the $M \times d_{\text{free}}$ Jacobian $J = \frac{\partial(A_1, \dots, A_M)}{\partial \boldsymbol{\theta}}$ at the optimum has **rank $d_{\text{free}} - 1$** ($\operatorname{rank}(J_{13}) = 19 < 20$ and $\operatorname{rank}(J_{16,\text{frame}}) = 23 < 24$, certified to $< 10^{-550}$). Consequently, the tie-system equations $A_1(\boldsymbol{\theta}) = \dots = A_M(\boldsymbol{\theta})$ define a **smooth 1-dimensional curve $\Gamma(s) \subset \mathbb{R}^{d_{\text{free}}}$ of exact equiareal configurations**, parameterized by arc length $s$ along the unit nullvector $\mathbf{v}_0 \in \ker J$:
     $$\boldsymbol{\theta}(s) = \boldsymbol{\theta}^\star + s \mathbf{v}_0 + s^2 \mathbf{w} + \mathcal{O}(s^3), \qquad A_1(\boldsymbol{\theta}(s)) = A_2(\boldsymbol{\theta}(s)) = \dots = A_M(\boldsymbol{\theta}(s)) \equiv \Delta(s).$$
2. **Why Damped Gauss-Newton on the Tie-System Halts at $s \ne 0$:**
   - Every point $\boldsymbol{\theta}(s)$ on the 1D curve $\Gamma(s)$ satisfies the tie-system $A_1 = \dots = A_M$ **with zero residual to arbitrary precision** (80, 500, or 10,000 digits).
   - However, because the 1D left nullspace $\ker(J^\top)$ is spanned by strictly positive KKT Lagrange multipliers $\boldsymbol{\lambda}^\star > 0$ ($\sum_{m=1}^M \lambda_m^\star = 1$, $J^\top \boldsymbol{\lambda}^\star = \mathbf{0}$), the transverse normal shift $\mathbf{w}$ cancels identically ($\boldsymbol{\lambda}^{\star\top} J \mathbf{w} = 0$), and the common area $\Delta(s)$ along $\Gamma(s)$ varies quadratically with a **strictly positive parameterization-independent KKT curvature invariant $c > 0$**:
     $$\Delta(s) = \Delta^\star - c \cdot s^2 + \mathcal{O}(s^3), \qquad c = -\frac{1}{2}\sum_{m=1}^M \lambda_m^\star \frac{d^2 A_m}{ds^2}\Big|_{\mathbf{v}_0} > 0.$$
   - When damped Gauss-Newton is run on the tie-system $A_1 = \dots = A_{24}$ starting from a `float64` optimization point at offset $s_0 \sim 10^{-6}$, it projects orthogonally onto $\Gamma(s)$ at $s \approx s_0 \ne 0$ and immediately terminates because all 24 triangles are already equal to 80 digits!
   - By instead solving the **full augmented KKT-Newton system** ($A_m(\boldsymbol{\theta}) - \Delta = 0$, $J(\boldsymbol{\theta})^\top \boldsymbol{\lambda} = \mathbf{0}$, $\sum \lambda_m = 1$) in 560-digit precision, we slide along $\Gamma(s)$ to the unique stationary apex $s = 0$:
     * **$n = 16$:** moves from $s = 2.2661 \times 10^{-6}$ (`0.0183555463554034501359529610459`) to $s = 0$:
       $$\Delta_{16}^\star = \mathbf{0.01835554635546114041232536365340313429459832905287244\dots} \quad (\mathbf{+5.76902763724026 \times 10^{-14}}).$$
     * **$n = 13$:** moves from $s = 1.0170 \times 10^{-6}$ (`0.026556528919738264474389245471`) to $s = 0$:
       $$\Delta_{13}^\star = \mathbf{0.02655652891977558013836441122594677335105195594906545\dots} \quad (\mathbf{+3.73156639751658 \times 10^{-14}}).$$

---

## Summary of Contributions Across $n = 13, 14, 15, 16$

| $n$ | Certificate File | Basin Attribution | What is New in This PR vs. Current `math.tejstead.com` |
| :---: | :--- | :--- | :--- |
| **$13$** | `n13_exact_500dps.json` | Tej Stead (July 2026) | **[NEW]** True KKT stationary point at $s=0$ (**`+3.731566397517e-14`** over published literals); Corank-1 KKT curvature invariant $c_{13} = 0.036078780238 > 0$; 3 exact parallel chords ($u_0 \equiv u_9, u_2 \equiv u_5, v_{11} \equiv v_{12}$); $\deg P(\Delta_{13}^\star) > 78$. |
| **$14$** | `n14_exact_500dps.json` | David W. Cantrell (June 2007) | **[NEW]** 500-digit stationary & $10^{100}$-rational certificate (`0.023775773107573057671296916141...`); exact vertical chord identity $u_6 \equiv u_9$; $\deg P(\Delta_{14}^\star) > 78$. |
| **$15$** | `n15_exact_500dps.json` | David W. Cantrell (2007); $12+3$ split & deg-5 poly reconstructed by Tej Stead (2026) | **[VERIFICATION + 500-DIGIT CERTIFICATE]** Independent from-scratch cascade rediscovery and 500-digit / $10^{100}$-rational certificate for the known degree-5 algebraic root ($5250987 \Delta_{15}^5 + 1609650 \Delta_{15}^4 + 161469 \Delta_{15}^3 + 5438 \Delta_{15}^2 - 13 \Delta_{15} - 4 = 0$). |
| **$16$** | `n16_exact_500dps.json` | Fable 5.1 with Shengtong Zhang (Sept 2026) | **[NEW]** True KKT stationary point at $s=0$ (**`+5.769027637240e-14`** over the 30-digit damped Gauss-Newton tie-system point); discovery of the **$15 + 1$ Floater Split** (`vertex 1` on site / `p_13` in ours has 0 active triangles, clearance `0.018532001428`); Corank-1 KKT curvature $c_{16} = 0.011234563954 > 0$; exact hypotenuse-parallel chord & trapezoid identity $(u_1+v_1) \equiv (u_{10}+v_{10})$. |

---

## 1. Detailed Breakdown for $n = 16$ (`n16_exact_500dps.json`)

* **500-Digit Stationary Minimum Area ($\Delta_{16}^\star$ at $s = 0$):**
  $$\Delta_{16}^\star = 0.01835554635546114041232536365340313429459832905287243778082350191\dots$$
* **Gain over Current Site `exact tie-system` Point (`0.0183555463554034501359529610459`):**
  $$\Delta_{16}^\star - \Delta_{16}^{\text{site}}(s_0) = \mathbf{+5.769027637240260750 \times 10^{-14}}.$$
* **The Unnoticed $15 + 1$ Floater Decomposition:**
  - Checking the vertex indices of the 24 minimal triangles listed on `math.tejstead.com/heilbronn/triangle/16/` (`(3,4,7), (0,9,14), ..., (2,3,6)`), **`vertex 1` never appears in any of the 24 active minimal triangles** (`p_13 = (0.440423714855, 0.491088435790)` in our right-simplex indexing).
  - Optimizing this floater vertex inside its local Voronoi cell while holding the 15-point frame fixed shows it has a strict Chebyshev area clearance of **`0.0185320014278944`** (`+0.0001764551` above $\Delta_{16}^\star$).
* **Corank-1 KKT Theorem on the 15-Point Frame:**
  - The 15 frame vertices (6 boundary + 9 interior) have $d_{\text{free}} = 24$ free coordinates and carry all $M = 24$ active minimal triangles.
  - At $s = 0$, $\operatorname{rank}(J_{\text{frame}}) = 23 < 24$ ($\|J_{\text{frame}} \mathbf{v}_0\|_2 = 9.08 \times 10^{-552}$), with 24 strictly positive KKT multipliers $\boldsymbol{\lambda}^\star \in [0.00011937,\; 0.16252430]^{24}$ ($\sum_{m=1}^{24} \lambda_m^\star = 1$) and second-order curvature invariant:
    $$c_{16} = -\frac{1}{2}\sum_{m=1}^{24} \lambda_m^\star \frac{d^2 A_m}{ds^2}\Big|_{\mathbf{v}_0} = \mathbf{0.011234563953840429469\dots} > 0.$$
  - The 30-digit coordinate set currently on `math.tejstead.com/heilbronn/triangle/16/` is the point on the 1D equiareal curve $\Gamma_{16}(s)$ at arc-length offset $s_0 = \sqrt{5.769028 \times 10^{-14} / c_{16}} = \mathbf{2.266115 \times 10^{-6}}$ ($4.6256 \times 10^{-7}$ RMS per coordinate).
* **Exact Hypotenuse-Parallel Chord & Trapezoid Identity:**
  In the right simplex ($u \ge 0, v \ge 0, u+v \le 1$), interior vertices $p_1$ and $p_{10}$ form a chord strictly parallel to the hypotenuse $p_4 p_9$ ($u+v=1$):
  $$(u_1 + v_1) - (u_{10} + v_{10}) \equiv 0, \qquad (u_1 - u_{10})(1 - u_1 - v_1) \equiv \Delta_{16}^\star \qquad (\text{verified to } < 10^{-500}).$$

---

## 2. Detailed Breakdown for $n = 13$ (`n13_exact_500dps.json`)

* **500-Digit Stationary Minimum Area ($\Delta_{13}^\star$ at $s = 0$):**
  $$\Delta_{13}^\star = 0.02655652891977558013836441122594677335105195594906545441276197643\dots$$
* **Gain over Current Site Coordinates (`0.026556528919738264474389245471`):**
  $$\Delta_{13}^\star - \Delta_{13}^{\text{site}}(s_0) = \mathbf{+3.731566397516575495 \times 10^{-14}}.$$
* **Corank-1 KKT Curvature Invariant:**
  - $d_{\text{free}} = 20$ free coordinates, $M = 20$ active minimal triangles, $\operatorname{rank}(J_{13}) = 19 < 20$ ($\|J_{13}\mathbf{v}_0\|_2 = 1.03 \times 10^{-551}$).
  - Strictly positive KKT multipliers $\boldsymbol{\lambda}^\star \in [0.00366307,\; 0.10246835]^{20}$ ($\sum_{m=1}^{20} \lambda_m^\star = 1$) and second-order curvature invariant:
    $$c_{13} = -\frac{1}{2}\sum_{m=1}^{20} \lambda_m^\star \frac{d^2 A_m}{ds^2}\Big|_{\mathbf{v}_0} = \mathbf{0.036078780237794575246\dots} > 0,$$
    explaining the `float64` deficit at arc-length offset $s_0 = \sqrt{3.731566 \times 10^{-14} / c_{13}} = \mathbf{1.016997 \times 10^{-6}}$ ($2.2741 \times 10^{-7}$ RMS per coordinate).
* **Exact Parallel Chords & Degree Bound:**
  $$u_0 \equiv u_9, \qquad u_2 \equiv u_5, \qquad v_{11} \equiv v_{12} \qquad (< 10^{-500}), \qquad \deg P(\Delta_{13}^\star) > 78.$$

---

## 3. Detailed Breakdown for $n = 14$ (`n14_exact_500dps.json`) and $n = 15$ (`n15_exact_500dps.json`)

* **$n = 14$ (David W. Cantrell, June 2007):** Over-constrained isostatic point ($22$ free coordinates, $23$ active minimal triangles, full rank $22$), 500-digit stationary value $\Delta_{14}^\star = 0.023775773107573057671296916141\dots$, exact vertical chord $u_6 \equiv u_9$ ($< 10^{-500}$), and $\deg P(\Delta_{14}^\star) > 78$.
* **$n = 15$ (David W. Cantrell, June 2007; $12+3$ structure and degree-5 polynomial reconstructed by Tej Stead, 2026):** Independent cascade rediscovery and 500-digit / $10^{100}$-rational certificate for the known degree-5 root $\Delta_{15}^\star = 0.021090769460266687959320407664\dots$ of $5250987 \Delta_{15}^5 + 1609650 \Delta_{15}^4 + 161469 \Delta_{15}^3 + 5438 \Delta_{15}^2 - 13 \Delta_{15} - 4 = 0$ (`|P_15(Delta_15*)| = 2.78e-503`).

---

## 4. Verification

Run the standalone zero-dependency verifier:
```bash
python verify_pr_package.py
```
It certifies all four 500-digit coordinate sets ($n=13, 14, 15, 16$), $10^{100}$-denominator exact rationals (`fractions.Fraction`), active tie-sets, KKT invariants, and parallel chord identities in `< 0.8s`.

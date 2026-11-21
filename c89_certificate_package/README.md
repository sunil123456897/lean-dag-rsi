# C_89 (Ramsey Multiplicity Constant of $K_4$): Certified Upper Bound $0.0301388757$

**Author:** Ivan Staroselets, November 2026  
**Exact certified value (`7680`-block uniform-weight step graphon, denominator $D = 2^{20}$):**

$$
c_4(W^{(7680)}) = \frac{544416702963479534354424655876570459894446711323}{18063603610809804744643931800845172819844136960000} = 0.03013887564703225514685795\ldots
$$

Hence **$C_{89} \le 0.0301388757$**, strictly improving the prior record `[SK2026]` (`0.03013887931325598096...`, `<= 0.0301388794`) by **`-3.666223725820686e-9`** (exact margin `-1059603394108337953046406402486682724992272 / 289017657772956875914302908813522765117506191360000`).

## Method (Continuous Balanced Sign-Split Refinement)

Starting from the `[SK2026]` `3840`-block equal-weight step graphon `c89_stepgraphon_D20.json.gz` (SHA-256 `82c28f7761f8392d...`, denominator $D = 2^{20}$), every block $i \in \{0,\dots,3839\}$ is split into two equal-weight halves $(i, +)$ and $(i, -)$ with balanced edge probabilities:

$$
W^{(7680)}_{(i, \sigma), (j, \tau)} = \frac{R_{ij} + \sigma \tau \, d_{ij}}{2^{20}}, \qquad \sigma, \tau \in \{-1, +1\}, \qquad |d_{ij}| \le \min(R_{ij}, 2^{20} - R_{ij}), \quad d_{ii} = 0.
$$

Because every non-empty even-degree subgraph of $K_4$ is either a triangle $C_3$ (4 copies) or a 4-cycle $C_4$ (3 copies), the exact change in monochromatic $K_4$ density has **zero higher-order terms**:

$$
c_4(W^{(7680)}) - c_4(W^{(3840)}) = \Delta_3(d) + \Delta_4(d) = \frac{N_3 + N_4}{3840^4 \cdot (2^{20})^6}.
$$

Unlike `[Kos2026]` (which fixed amplitudes to $|d_{ij}| = \min(R_{ij}, D - R_{ij})$ and optimized only discrete signs on the older `[VSJ2026]` base for `-1.87e-9`), we optimize **all `221,460` continuous fractional amplitudes** $d_{ij} \in [-\text{cap}_{ij}, +\text{cap}_{ij}]$ freely on GPU (RTX 5060, per-chunk exact backward):
- Cubic sign ascent alone gave $\Delta_3 = -4.873\times 10^{-9}$, $\Delta_4 = +3.417\times 10^{-9}$ ($\Delta = -1.456\times 10^{-9}$).
- Continuous amplitude optimization slashed the quartic penalty $\Delta_4$ nearly **3x** (`+3.417e-9 -> +1.175e-9`) while preserving $\Delta_3 = -4.842\times 10^{-9}$, yielding **`Delta = -3.6662237258e-9`**.

## Exact Verification

`verify_c89_exact.py` checks strict integer feasibility (`0 <= R_ij +- d_ij <= 2^20`), runs a brute-force `2m x 2m` exact integer self-test, and certifies both $N_3$ and $N_4$ over $\mathbb{Z}$ via multi-prime Chinese Remainder Theorem (`8` primes `< 5.6e6`, strictly below $2^{53}$ dot-product overflow, plus `1` independent extra check prime `p = 5592227`):

```bash
python verify_c89_exact.py /path/to/c89_stepgraphon_D20.json.gz c89_7680_delta_D20.npz
```

### Certified Values
- `Exact N_3 = -1399283324392161754041842816623728081781248` (`-4.841514996607558297e-9`)
- `Exact N_4 = +339679930283823800995436414137045356788976` (`+1.175291270786872139e-9`)
- `TOTAL Delta = -3.666223725820686158e-9`
- `SK2026 c4 = 0.03013887931325598096754411...`
- `NEW    c4 = 0.03013887564703225514685795...`
- **Strict upper bound:** $C_{89} \le 0.0301388757$

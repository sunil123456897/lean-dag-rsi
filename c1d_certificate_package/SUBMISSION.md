# Exact Rational Upper Bound Certificate for the Signed Third Autocorrelation Constant $C_{1d}$

## Problem Definition

The signed third autocorrelation (autoconvolution) constant $C_{1d}$ (also denoted $\Phi_3$) is defined as the largest constant such that

$$\max_{t \in \mathbb{R}} \left| \int_{\mathbb{R}} f(t - x)\, f(x)\, dx \right| \ge C_{1d} \left( \int_{-1/4}^{1/4} f(x)\, dx \right)^2$$

holds for all real-valued (signed) functions $f \in L^1(\mathbb{R})$ supported on $[-1/4, 1/4]$ with $\int f \ne 0$.

For a piecewise-constant step function $f$ taking rational values $q_i / D$ on $P$ equal intervals of width $h = \frac{1}{2P}$ partitioning $[-1/4, 1/4]$, the autoconvolution $(f * f)(t)$ is continuous and piecewise linear on $[-1/2, 1/2]$ with breakpoints at the $2P - 1$ interior nodes $t_k = -\frac{1}{2} + (k+1)h$ ($k = 0, \dots, 2P-2$) and value $0$ at $t = \pm \frac{1}{2}$. Because $|g(t)|$ is convex on every linear segment, its global supremum over $\mathbb{R}$ is attained identically at one of the $2P - 1$ interior nodes:

$$C_{1d} \le \frac{2P \max_{0 \le k \le 2P - 2} \bigl|(q * q)_k\bigr|}{\left(\sum_{i=0}^{P-1} q_i\right)^2}.$$

(On a constant step function $q_i \equiv 1$, this gives $2P \cdot P / P^2 = 2$.)

---

## Certified Result ($P = 6400$)

- **Exact rational upper bound (verified over all $2P - 1 = 12\,799$ linear convolution shifts):**

$$C_{1d} \le \frac{59677128285114071086870749824000}{40959999999999577600000000001089}$$

- **50-digit decimal expansion:**
  `1.45696113977330777605605932782270447749626318706656...`
- **Two-sided minimax balance (both positive and negative lobes tightly pinned):**
  - $\max_k \frac{2P (q*q)_k}{(\sum q)^2} = +1.45696113977330777606$ (shift $k = 8062$)
  - $\min_k \frac{2P (q*q)_k}{(\sum q)^2} = -1.45696071010775449254$ (shift $k = 9207$)
- **Support signature:** $P = 6400$ equal cells (`1509` negative, `4891` positive; asymmetric signed profile with edge singularities).

### Progression for Strict Signed $C_{1d}$ (with `abs` on full linear convolution and $(\sum q_i)^2$ denominator)

| Source / Certificate | Cells ($P$) | Strict $C_{1d}$ Upper Bound |
| :--- | :---: | :--- |
| Matolcsi & Vinuesa (2010) | — | `1.4993` |
| `arXiv:2511.02864` (Nov 2025) | — | `1.4688` |
| 300-cell certificate (Oct 2026) | `300` | `1.46307158...` |
| 600-cell certificate (Oct 2026) | `600` | `1.46021728...` |
| SCiEL (Cairn Commons, Nov 2026) | `800` | `1.4578548849...` |
| This repo (`artifacts/phi3_best_candidate.npy`) | `3200` | `1.4570585083` (`-0.00079638` vs SCiEL) |
| **This certificate (`c1d_p6400_exact_certificate.json`)** | **`6400`** | **`1.4569611398` (`-0.00089375` vs SCiEL)** |

---

## Method (Honest Attribution)

Obtained by zero-smoothing dyadic prolongation (`np.repeat(a_3200, 2)`, preserving $C_{1d}$ identically at $1.4570585083$) followed by **Exact Dense Cholesky Full-Hessian Newton** with LogSumExp $\beta$-continuation ($\beta = 5\times 10^3 \to 1.2\times 10^7$) over the exact $P \times P$ Gauss-Newton + Hankel curvature Hessian projected onto $\{\Delta a : \sum \Delta a_i = 0\}$, and exact integer quantization at scale $10^{12}$.

---

## Independent Verification (Zero Dependencies)

`verify_c1d.py` uses **only the Python 3 standard library** (`json`, `fractions`, `decimal`, `argparse`, `time` — zero NumPy, zero FFT, zero floating-point arithmetic):

```bash
# Fast exact signed Kronecker big-int verification + direct spot checks (~0.1s)
python c1d_certificate_package/verify_c1d.py

# 100% naive schoolbook double-sum over ALL 12,799 linear convolution shifts (~3s)
python c1d_certificate_package/verify_c1d.py --brute-force
```

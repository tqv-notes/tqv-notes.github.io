---
title:  "Random Features"
mathjax: true
layout: post
categories: media
---

# Random Features for Kernel Approximation

Random features for machine learning were introduced in the kernel-machine
setting by [Rahimi & Recht, NIPS 2007](https://proceedings.neurips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html).
This note explains the idea, the theory (Bochner's theorem), and shows three
short Python tutorials: approximating a kernel, watching the approximation
converge, and using random features to make kernel regression scale.

## 1. The problem with kernels

Kernel methods build a predictor as a linear combination of kernel evaluations:

$$f(x) = \sum_{i=1}^{N} \alpha_i\, k(x, x_i)$$

A common choice is the Gaussian (RBF) kernel:

$$k(x, x') = \exp\!\left(-\frac{\|x - x'\|_2^2}{2\sigma^2}\right)$$

The catch: fitting requires working with the $N \times N$ Gram matrix.
Solving the linear system costs $O(N^3)$ time and $O(N^2)$ memory.
For large $N$ this is infeasible.

## 2. The random-features idea

Replace the implicit, infinite-dimensional kernel feature map with an
**explicit, finite, randomized** map

$$x \in \mathbb{R}^d \;\longmapsto\; z(x) \in \mathbb{R}^D$$

chosen so that the inner product approximates the kernel:

$$k(x, x') \;\approx\; z(x)^\top z(x').$$

Here $D$ is the number of random features. Unlike the implicit kernel map
(which is infinite-dimensional), $D$ is finite and we control it — typically
$D \gg d$, with accuracy improving as $D$ grows. Once we have $z(\cdot)$, we
fit a plain **linear** model in the $D$-dimensional feature space, which costs
$O(N D^2)$ instead of $O(N^3)$. When $D \ll N$, this is a big win.

## 3. Bochner's theorem (why it works)

A continuous, shift-invariant kernel $k(x, x') = k(x - x')$ that is normalized
so $k(0) = 1$ is the Fourier transform of a probability density $p(\omega)$:

$$k(x - y) = \int_{\mathbb{R}^d} p(\omega)\, e^{\, i\,\omega^\top (x - y)}\, d\omega
= \mathbb{E}_{\omega \sim p}\!\left[ e^{i\omega^\top x}\,\overline{e^{i\omega^\top y}} \right].$$

So if we draw \\(\omega_1, \dots, \omega_D \sim p\\) i.i.d., the **Monte Carlo
estimate** of the expectation gives our approximation. For the Gaussian kernel
with bandwidth $\sigma$, the density \\(p(\omega)\\) is itself Gaussian:

$$\omega \sim \mathcal{N}\!\left(0,\; \sigma^{-2} I_d\right).$$

### The two feature maps

Both are valid; the real-valued cosine map is what you use in practice.

**Complex exponential**
$$z(x) = \frac{1}{\sqrt{D}}\big[\, e^{i\,\omega_j^\top x} \,\big]_{j=1}^{D}$$

**Real cosine** (with phase $b_j \sim \text{Uniform}(0, 2\pi)$)
$$z(x) = \sqrt{\frac{2}{D}}\,\big[\, \cos(\omega_j^\top x + b_j) \,\big]_{j=1}^{D}$$

The \\(\frac{1}{\sqrt{D}}\\) (resp. \\(\sqrt{2/D}\\)) factor turns the *sum* into the
*average* that Monte Carlo requires — this normalization is essential.

## 4. Example 1: Approximate the kernel

```python
import numpy as np

def rbf_kernel(X, Y, sigma=1.0):
    """Exact Gaussian (RBF) Gram matrix."""
    sq = (
        np.sum(X**2, axis=1)[:, None]
        + np.sum(Y**2, axis=1)[None, :]
        - 2 * X @ Y.T
    )
    return np.exp(-sq / (2 * sigma**2))

class RFF:
    """Random Fourier Features for the RBF kernel (cosine map)."""
    def __init__(self, n_features=200, sigma=1.0, seed=0):
        self.D, self.sigma, self.rng = n_features, sigma, np.random.default_rng(seed)

    def fit(self, X):
        d = X.shape[1]
        # omega ~ N(0, sigma^-2 I);  b ~ U(0, 2*pi)
        self.W = self.rng.normal(scale=1.0 / self.sigma, size=(d, self.D))
        self.b = self.rng.uniform(0, 2 * np.pi, size=self.D)
        return self

    def transform(self, X):
        proj = X @ self.W + self.b
        return np.sqrt(2.0 / self.D) * np.cos(proj)

# Demo: compare approximate vs. exact kernel values
rng = np.random.default_rng(1)
X = rng.standard_normal((5, 3))
sigma = 1.5

rff = RFF(n_features=5000, sigma=sigma).fit(X)
Z = rff.transform(X)

K_approx = Z @ Z.T
K_exact  = rbf_kernel(X, X, sigma=sigma)

print("Max abs error:", np.abs(K_approx - K_exact).max())
# -> a small number (~1e-2); shrinks as n_features grows
```

## 5. Example 2: Convergence of random-feature kernel

The approximation error of a Monte Carlo estimate shrinks like \\(O(1/\sqrt{D})\\).
Let's confirm that scaling empirically.

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
X = rng.standard_normal((30, 4))
sigma = 1.0
K_exact = rbf_kernel(X, X, sigma=sigma)

Ds = [10, 30, 100, 300, 1000, 3000, 10000]
errors = []
for D in Ds:
    # average over a few seeds to reduce noise
    errs = []
    for s in range(10):
        Z = RFF(n_features=D, sigma=sigma, seed=s).fit(X).transform(X)
        errs.append(np.linalg.norm(Z @ Z.T - K_exact) / np.linalg.norm(K_exact))
    errors.append(np.mean(errs))

plt.loglog(Ds, errors, "o-", label="relative error")
plt.loglog(Ds, errors[0] * np.sqrt(Ds[0]) / np.sqrt(Ds), "--",
           label=r"$O(1/\sqrt{D})$ reference")
plt.xlabel("number of random features $D$")
plt.ylabel("relative Frobenius error")
plt.legend(); plt.title("Random-feature kernel approximation error")
plt.show()
```

The measured error tracks the \\(1/\sqrt{D}\\) reference line - exactly what the
theory predicts.

## 6. Example 3: Scalable kernel ridge regression

Here is the payoff. We fit ridge regression in random-feature space and compare
against exact kernel ridge regression on a noisy 1-D function.

```python
import numpy as np
import matplotlib.pyplot as plt

# --- synthetic data ---
rng = np.random.default_rng(42)
n = 400
X = np.sort(rng.uniform(-4, 4, size=(n, 1)), axis=0)
y = np.sin(2 * X[:, 0]) * np.exp(-0.1 * X[:, 0] ** 2) + 0.1 * rng.standard_normal(n)

sigma, lam = 0.5, 1e-3

# --- exact kernel ridge regression:  alpha = (K + lam I)^-1 y ---
K = rbf_kernel(X, X, sigma=sigma)
alpha = np.linalg.solve(K + lam * np.eye(n), y)

Xt = np.linspace(-4, 4, 300)[:, None]
y_exact = rbf_kernel(Xt, X, sigma=sigma) @ alpha

# --- random-feature ridge regression:  w = (Z^T Z + lam I)^-1 Z^T y ---
rff = RFF(n_features=300, sigma=sigma, seed=0).fit(X)
Z = rff.transform(X)
w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y)
y_rff = rff.transform(Xt) @ w

# --- plot ---
plt.scatter(X, y, s=8, alpha=0.3, label="data")
plt.plot(Xt, y_exact, "k-", lw=2, label="exact KRR")
plt.plot(Xt, y_rff, "r--", lw=2, label="RFF ridge (D=300)")
plt.legend(); plt.title("Kernel ridge regression vs. random features")
plt.show()
```

The random-feature curve closely matches exact KRR, but the linear solve scales
with \\(D\\) rather than \\(N\\). When \\(N\\) runs into the hundreds of thousands, the
exact \\(O(N^3)\\) solve becomes impractical while random features stay cheap.

## 7. Takeaways

- Random features replace an expensive implicit kernel with a cheap explicit
  feature map \\(z(x)\\) such that \\(k(x,x') \approx z(x)^\top z(x')\\).
- The justification is **Bochner's theorem** plus a **Monte Carlo** average;
  error decays as \\(O(1/\sqrt{D})\\).
- For the Gaussian kernel, sample frequencies from
  \\(\mathcal{N}(0, \sigma^{-2}I)\\) and use the cosine map with a random phase.
- The practical win: \\(O(N^3)\\) kernel methods become \\(O(N D^2)\\) linear models.

### References
- A. Rahimi and B. Recht, *Random Features for Large-Scale Kernel Machines*,
  NIPS 2007.

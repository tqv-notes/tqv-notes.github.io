---
title:  "Scattering Transform"
mathjax: true
layout: post
categories: media
---

# The Scattering Transform: Wavelets, Stability, and the Geometry of Deep Learning
---

## Table of Contents

1. [Why Does Deep Learning Work on Signals?](#1-why-does-deep-learning-work-on-signals)
2. [Geometric Stability: The Core Design Principle](#2-geometric-stability-the-core-design-principle)
3. [Why Fourier Fails](#3-why-fourier-fails)
4. [Building the Scattering Transform](#4-building-the-scattering-transform)
5. [Key Mathematical Properties](#5-key-mathematical-properties)
6. [Demo 1 - The Cascade: Energy Decay Across Layers](#demo-1--the-cascade-energy-decay-across-layers)
7. [Demo 2 - Deformation Stability: Scattering vs. Fourier](#demo-2--deformation-stability-scattering-vs-fourier)
8. [Demo 3 - Texture Discrimination: Same Spectrum, Different Structure](#demo-3--texture-discrimination-same-spectrum-different-structure)
9. [Demo 4 - Multifractal Analysis: Capturing Intermittency](#demo-4--multifractal-analysis-capturing-intermittency)
10. [Extensions and Applications](#10-extensions-and-applications)
11. [Summary](#11-summary)

---

## 1. Why Does Deep Learning Work on Signals?

Convolutional neural networks (CNN) achieve remarkable performance on images and audio, yet a rigorous mathematical explanation of *why* remains elusive. The **scattering transform** (ST) is a simplified version of CNN with regourous mathematical foundations. It was originated from signal/image processing fields ([S. Mallat](https://arxiv.org/pdf/1101.2286) and [J. Bruna & S. Mallat](https://arxiv.org/pdf/1203.1513)) and found applications in diverse tasks: image classification, finance, astrophysics, material science, ... 

The key insight is that the success of CNNs on structured signals is not accidental - it follows directly from the geometric properties of those signal domains. Images, audio, and physical fields share a common regularity: they are **stable to small local deformations**. A slightly warped image of a cat is still a cat. A pitch-shifted vowel is still the same vowel.

The scattering transform is a hand-crafted signal representation that provably exploits this regularity. It combines classical wavelet analysis with a deep convolutional architecture whose filters are never learned - they are fixed by mathematics. The result is a feature extractor with formally certified stability guarantees that trained CNNs do not have, and that provides a mathematical lens through which to understand what CNNs implicitly learn to do.

---

## 2. Geometric Stability: The Core Design Principle

### 2.1 The Setting

Let \\(x \in L^2(\mathbb{R}^d)\\) be a signal (e.g., an image or audio waveform). We want to build a representation \\(\Phi(x) \in \mathbb{R}^K\\) to feed into a linear classifier \\(\hat{f}(x) = \langle \Phi(x), \theta \rangle\\).

For this to generalize well, \\(\Phi\\) must satisfy two things:

**Stability to additive noise:**
$$\| \Phi(x) - \Phi(x') \| \lesssim \| x - x' \|$$

**Stability to deformations:** Given a smooth displacement field \\(\tau : \mathbb{R}^d \to \mathbb{R}^d\\), let \\(x_\tau(u) = x(u - \tau(u))\\) be the deformed signal. We want:
$$\| \Phi(x_\tau) - \Phi(x) \| \lesssim \| x \| \cdot \| \tau \|$$

where \\(\|\tau\|\\) is some appropriate norm on the displacement field.

### 2.2 Why Deformation Stability Is the Right Prior

Global translation invariance (\\(f(T_v x) = f(x)\\) for all translations \\(T_v\\)) is a weak constraint - the translation group is only \\(d\\)-dimensional. The space of local deformations is infinite-dimensional and captures far more natural variability: changes in viewpoint, non-rigid motion, pronunciation variation in speech.

A key structural consequence is **scale separation**: since deformations act differently at different frequencies, a deformation-stable representation must separate scales. This is precisely what wavelet decompositions do - and precisely what the layers of a CNN do implicitly.

---

## 3. Why Fourier Fails

The Fourier modulus \\( \Phi(x) = \|\hat{x}(\omega)\| \\) is translation invariant and stable to additive noise, but it is **catastrophically unstable to deformations at high frequencies**.

### The Dilation Example

Let \\(\tau(u) = su\\) be a small uniform dilation (\\(\|s\| \ll 1\\)), and let \\(x(u) = e^{i\xi u} \theta(u)\\) be a modulated window centered at frequency \\(\xi\\). The dilated signal \\(x_\tau(u) = x((1+s)u)\\) has its central frequency shifted to \\((1+s)\xi\\).

The frequency spread of \\(x\\) is \\(\sigma_\theta^2 = \int \|\omega - \xi\|^2 \|\hat{\theta}(\omega)\|^2 d\omega\\), and after dilation it becomes \\((1+s)^2 \sigma_\theta^2\\).

When the frequency shift \\(s\xi\\) is large compared to the bandwidth \\(\sigma_\theta\\), the supports of \\(\|\hat{x}\|\\) and \\(\|\hat{x}_\tau\|\\) are nearly disjoint, so:
$$\| \|\hat{x}_\tau\| - \|\hat{x}\| \| \approx \| x \|$$

This is an \\(O(1)\\) error from an arbitrarily small deformation when \\(\xi\\) is large. The Fourier modulus is **not Lipschitz continuous** with respect to deformations.

The fix is to band-limit the signal before measuring it - that is, to use a **wavelet transform** that isolates each frequency band before applying a modulus nonlinearity.

---

## 4. Building the Scattering Transform

### 4.1 The Wavelet Filter Bank

A Littlewood-Paley wavelet transform is built from a mother wavelet \\(\psi \in L^2(\mathbb{R}^d)\\) by dilating and rotating:
$$\psi_\lambda(u) = a^{-dj} \psi(a^{-j} r^{-1} u), \quad \lambda = a^j r$$

where \\(j \in \mathbb{Z}\\) controls scale (\\(a^j\\) is the scale factor, typically \\(a = 2\\)) and \\(r \in G\\) is a rotation. The Littlewood-Paley condition ensures the filter bank is a tight frame:

$$1 - \varepsilon \leq |\hat{\phi}(2^J \omega)|^2 + \frac{1}{2} \sum_{j \leq J} \sum_{r \in G} |\hat{\psi}(2^j r \omega)|^2 \leq 1$$

This means the decomposition is *energy-preserving* and *invertible*.

### 4.2 The Modulus Nonlinearity

Wavelet coefficients \\(x \otimes \psi_\lambda(u)\\) are not translation invariant. Their average is zero (wavelets have zero mean). The key step is to apply the complex modulus:
$$|x \otimes \psi_\lambda(u)|$$

This produces a non-negative, non-zero envelope that is roughly translation invariant at the scale of \\(\psi_\lambda\\). The modulus is the *only* nonlinearity that:
- Is non-expansive: \\(\| \|a\| - \|b\| \| \leq \| a - b \|\\)
- Preserves signal energy across layers

### 4.3 The Cascade

Averaging \\(\|x \otimes \psi_\lambda\|\\) over a window of size \\(2^J\\) gives a translation-invariant first-order feature:
$$S_J[\lambda_1] x(u) = \|x \otimes \psi_{\lambda_1}\| \otimes \phi_{2^J}(u)$$

But averaging discards information - specifically, the spatial modulation of the wavelet envelope. This lost information is recovered by applying *another* wavelet transform to \\(\|x \otimes \psi_{\lambda_1}\|\\), taking the modulus again, and averaging.

This produces second-order coefficients:
$$S_J[\lambda_1, \lambda_2] x(u) = \big| |x \otimes \psi_{\lambda_1}| \otimes \psi_{\lambda_2} \big| \otimes \phi_{2^J}(u)$$

Iterating this process defines the full scattering transform. For a **path** \\(p = (\lambda_1, \lambda_2, \ldots, \lambda_m)\\), we define the propagator:
$$U[p]x = \big| \cdots \big| |x \otimes \psi_{\lambda_1}| \otimes \psi_{\lambda_2} \big| \cdots \big| \otimes \psi_{\lambda_m} \big|$$

and the scattering coefficient:
$$S_J[p]x(u) = U[p]x \otimes \phi_{2^J}(u)$$

The resulting architecture is a convolutional network whose filters are fixed wavelets, not learned parameters.

```
Input x 
   │
   ├── S_J[∅]x = x ⊗ φ_J                  (order 0: low-pass average)
   │
   ├── U[λ₁]x = |x ⊗ ψ_λ₁|
   │     ├── S_J[λ₁]x                      (order 1 outputs)
   │     └── U[λ₁,λ₂]x = |U[λ₁]x ⊗ ψ_λ₂|
   │           ├── S_J[λ₁,λ₂]x             (order 2 outputs)
   │           └── ...                     (order 3, ...)
   └── ...
```

---

## 5. Key Mathematical Properties

### 5.1 Non-Expansiveness (Stability to Noise)

**Proposition.** The windowed scattering transform is non-expansive:
$$\| S_J[P_J] x - S_J[P_J] x' \| \leq \| x - x' \|, \quad \forall x, x' \in L^2(\mathbb{R}^d)$$

This follows because (1) the Littlewood-Paley wavelet frame satisfies \\(\|W_J x\| \leq \|x\|\\), and (2) the modulus satisfies \\(\| \|a\| - \|b\| \| \leq \|a - b\|\\). Their composition is also non-expansive.

### 5.2 Energy Conservation and Exponential Decay

Under mild conditions on the wavelet (roughly: the wavelet is analytic and has at least one vanishing moment), the total scattering energy is preserved:
$$\|x\|^2 = \sum_{p \in P_\infty} \|S_J[p]x\|^2$$

More importantly, the energy decays *exponentially* with path depth:
$$R_{J,x}(m) := \sum_{|p|=m} \|U[p]x\|^2 \leq \|x\|^2 - \|x \otimes \chi_{ra^m}\|^2$$

where \\(\chi_s\\) is a Gaussian window of width \\(s\\). Energy at frequency \\(2^k\\) disappears after \\(O(k)\\) layers, so typical signals require no more than 2–3 layers. Empirically, on image datasets, over 99% of the energy is captured by paths of length \\(m \leq 2\\).

### 5.3 Asymptotic Translation Invariance

The scattering metric $d_J(x, x') := \|S_J[P_J]x - S_J[P_J]x'\|$ is non-increasing in \\(J\\):
$$d_{J+1}(x, x') \leq d_J(x, x')$$

and in the limit it is translation invariant:
$$\lim_{J \to \infty} \| S_J[P_J] x - S_J[P_J] x_v \| = 0, \quad \forall x, v$$

where \\(x_v(u) = x(u - v)\\).

### 5.4 Lipschitz Stability to Deformations

This is the central theorem. For a \\( C^2 \\) displacement field \\(\tau\\) with \\( \|\nabla\tau\|_\infty \leq 1/2 \\):

$$\| S_J[P_J] x_\tau - S_J[P_J] x \| \leq C \|U[P_J]x\|_1 \cdot K(\tau)$$

where:
$$K(\tau) = 2^{-J}\|\tau\|_\infty + \|\nabla\tau\|_\infty \max\!\left(1,\, \log \frac{\sup_{u,u'}|\tau(u)-\tau(u')|}{\|\nabla\tau\|_\infty}\right) + \|H\tau\|_\infty$$

The bound decomposes into:
- **Translation term** \\( 2^{-J}\|\tau\|_\infty \\): suppressed by increasing $J$, capturing local translation invariance
- **Deformation term** \\( \|\nabla\tau\|_\infty \\): controlled by scale separation in the wavelet decomposition
- **Curvature term** \\( \|H\tau\|_\infty \\): second-order correction

The proof hinges on controlling the commutator \\( \[W_J, L_\tau\] = W_J L_\tau - L_\tau W_J\\) between the wavelet transform and the deformation operator, which is bounded by $\|\nabla\tau\|$ due to the scale-localization property of wavelets.

---

## Demo 1 - The Cascade: Energy Decay Across Layers

This demo builds a synthetic multi-scale signal and visualizes how energy is distributed across scattering orders. The theoretical prediction - exponential decay with path depth - is confirmed empirically.

Important note for installation: `pip install kymatio numpy scipy matplotlib`

```python
"""
Demo 1: The Scattering Cascade and Energy Decay
================================================
Visualizes how energy distributes across scattering orders (0, 1, 2)
and confirms the theoretical exponential decay.

Install: pip install kymatio numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from kymatio.numpy import Scattering1D

# -- Signal parameters ----------------------------------------------------------
T = 2**13    # signal length (must be power of 2)
J = 6        # number of dyadic scales
Q = 8        # wavelets per octave (frequency resolution)
t = np.linspace(0, 1, T)

np.random.seed(0)

# Construct a signal with energy at multiple scales:
#   - slow carrier (scale ~1/5 Hz)
#   - mid-frequency burst (scale ~1/80 Hz, localized at t=0.5)
#   - high-frequency modulation (scale ~1/200 Hz)
#   - low-amplitude noise
signal = (
    np.sin(2 * np.pi * 5 * t)
    + 0.6 * np.sin(2 * np.pi * 80 * t) * np.exp(-((t - 0.5)**2) / 0.003)
    + 0.3 * np.sin(2 * np.pi * 200 * t) * np.exp(-((t - 0.3)**2) / 0.001)
    + 0.05 * np.random.randn(T)
)

# -- Scattering transform -------------------------------------------------------
scat = Scattering1D(J=J, shape=T, Q=Q)
Sx   = scat(signal)          # shape: [num_paths, T // 2^J]
meta = scat.meta()           # path metadata (order, scale, angle)
order = meta['order']        # integer array: 0, 1, or 2 for each path

print(f"Signal length:              {T}")
print(f"Scattering output shape:    {Sx.shape}")
print(f"Downsampling factor:        {T // Sx.shape[-1]}x  (scale 2^J = {2**J})")
print(f"Total paths:                {Sx.shape[0]}")
for m in [0, 1, 2]:
    n = np.sum(order == m)
    E = np.sum(Sx[order == m]**2)
    print(f"  Order {m}: {n:4d} paths, energy fraction = {E / np.sum(Sx**2):.4f}")

# -- Plot -----------------------------------------------------------------------
fig, axes = plt.subplots(4, 1, figsize=(12, 10), gridspec_kw={'hspace': 0.45})

# 1. Original signal
axes[0].plot(t, signal, lw=0.6, color='#2c7bb6')
axes[0].set_title("Input signal  (multi-scale: low carrier + mid burst + high modulation)")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Amplitude")

# 2. Order-0 scattering (low-pass average of entire signal)
axes[1].plot(Sx[order == 0].T, color='#d7191c', lw=1.5)
axes[1].set_title(r"Order 0:  $S_J[\emptyset]\,x = x \otimes \phi_{2^J}$  - global low-pass envelope")
axes[1].set_xlabel("Spatial position (downsampled)")

# 3. Order-1 coefficients as a heatmap over (time, scale)
im1 = axes[2].imshow(
    np.log1p(np.abs(Sx[order == 1])),
    aspect='auto', cmap='YlOrRd', origin='lower'
)
axes[2].set_title("Order 1:  $S_J[\\lambda_1]\,x$  - energy by (scale, time)")
axes[2].set_xlabel("Spatial position")
axes[2].set_ylabel("Path index (∝ scale)")
plt.colorbar(im1, ax=axes[2], label='log(1 + |coeff|)')

# 4. Order-2 coefficients
im2 = axes[3].imshow(
    np.log1p(np.abs(Sx[order == 2])),
    aspect='auto', cmap='PuBuGn', origin='lower'
)
axes[3].set_title("Order 2:  $S_J[\\lambda_1, \\lambda_2]\,x$  - scale-interaction features")
axes[3].set_xlabel("Spatial position")
axes[3].set_ylabel("Path index")
plt.colorbar(im2, ax=axes[3], label='log(1 + |coeff|)')

plt.suptitle("Scattering Cascade: energy distributes across orders and decays", fontsize=13)
plt.savefig("demo1_cascade.png", dpi=150, bbox_inches='tight')
plt.show()

# -- Energy decay table ---------------------------------------------------------
print("\n--- Energy fraction per order ---")
total_energy = np.sum(Sx**2)
for m in [0, 1, 2]:
    frac = np.sum(Sx[order == m]**2) / total_energy
    print(f"  Order {m}: {frac*100:.2f}%")
```

**Expected output:**
```
Order 0:     1 path,  energy fraction ≈ 0.44
Order 1:    ~48 paths, energy fraction ≈ 0.51
Order 2:  ~384 paths, energy fraction ≈ 0.05
```

The exponential decay is clear: order-0 captures the bulk of the DC energy, order-1 captures frequency-band energy, and order-2 has small but non-trivial residual energy encoding higher-order structure.

---

## Demo 2 - Deformation Stability: Scattering vs. Fourier

This demo applies a smooth time warp (a sinusoidal displacement field) to a test signal and measures the resulting error in both the Fourier modulus and the scattering representation. The scattering error should be dramatically smaller.

```python
"""
Demo 2: Deformation Stability
==============================
Empirically verifies that the scattering transform is Lipschitz continuous
to deformations, while the Fourier modulus is not.

Install: pip install kymatio numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from kymatio.numpy import Scattering1D
from scipy.ndimage import map_coordinates

T  = 2**12
J  = 5
Q  = 8
t  = np.linspace(0, 1, T)

np.random.seed(42)

# Test signal: two harmonics at very different scales
signal = np.sin(2 * np.pi * 20 * t) + 0.5 * np.cos(2 * np.pi * 120 * t)

# -- Deformation operator -------------------------------------------------------
def apply_deformation(x, tau_max_frac, freq=3):
    """
    Warp signal x by a smooth sinusoidal displacement field:
        x_tau(t) = x(t - tau(t))
    tau(t) = tau_max * sin(2π * freq * t)

    tau_max_frac: max displacement as fraction of signal length T
    """
    displacement = tau_max_frac * T * np.sin(2 * np.pi * freq * t)
    warped_idx   = np.clip(np.arange(T) - displacement, 0, T - 1)
    return map_coordinates(x, [warped_idx], order=3, mode='nearest')

# -- Sweep over deformation amplitudes -----------------------------------------
tau_values    = np.linspace(0.0, 0.025, 20)
fourier_errors = []
scat_errors    = []

scat = Scattering1D(J=J, shape=T, Q=Q)
Sx_orig = scat(signal)
F_orig  = np.abs(np.fft.rfft(signal))

for tau_max in tau_values:
    sig_w     = apply_deformation(signal, tau_max)
    F_w       = np.abs(np.fft.rfft(sig_w))
    Sx_w      = scat(sig_w)
    fourier_errors.append(np.linalg.norm(F_orig - F_w)  / np.linalg.norm(F_orig))
    scat_errors.append(   np.linalg.norm(Sx_orig - Sx_w) / np.linalg.norm(Sx_orig))

# -- Qualitative example at a fixed deformation --------------------------------
tau_demo   = 0.015
signal_w   = apply_deformation(signal, tau_demo)
F_w_demo   = np.abs(np.fft.rfft(signal_w))
Sx_w_demo  = scat(signal_w)
meta       = scat.meta()
order      = meta['order']

fe_demo = np.linalg.norm(F_orig - F_w_demo)  / np.linalg.norm(F_orig)
se_demo = np.linalg.norm(Sx_orig - Sx_w_demo) / np.linalg.norm(Sx_orig)

print(f"Deformation tau_max = {tau_demo:.3f}")
print(f"  Fourier modulus relative error:  {fe_demo:.4f}")
print(f"  Scattering relative error:        {se_demo:.4f}")
print(f"  Stability improvement:            {fe_demo/se_demo:.1f}x")

# -- Plot -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Top-left: original vs. warped signal
axes[0, 0].plot(t[:300], signal[:300],   lw=1.2, color='#2c7bb6', label='original')
axes[0, 0].plot(t[:300], signal_w[:300], lw=1.2, color='#d7191c', alpha=0.8, label='warped')
axes[0, 0].set_title(f"Signal vs. warped  ($\\tau_{{\\max}} = {tau_demo}$)")
axes[0, 0].legend()

# Top-right: Fourier modulus comparison
freqs = np.fft.rfftfreq(T)
axes[0, 1].plot(freqs[:T//8], F_orig[:T//8],    lw=1.0, color='#2c7bb6', label='original')
axes[0, 1].plot(freqs[:T//8], F_w_demo[:T//8],  lw=1.0, color='#d7191c', alpha=0.8,
                label='warped')
axes[0, 1].set_title(f"Fourier modulus  (error = {fe_demo:.3f})")
axes[0, 1].set_xlabel("Frequency")
axes[0, 1].legend()

# Bottom-left: scattering coefficients comparison (order-1 paths)
idx1 = np.where(order == 1)[0]
mean_orig = np.mean(np.abs(Sx_orig[idx1]),  axis=1)
mean_w    = np.mean(np.abs(Sx_w_demo[idx1]), axis=1)
x_idx     = np.arange(len(idx1))
axes[1, 0].bar(x_idx - 0.2, mean_orig, width=0.4, color='#2c7bb6', label='original', alpha=0.85)
axes[1, 0].bar(x_idx + 0.2, mean_w,    width=0.4, color='#d7191c', label='warped',   alpha=0.85)
axes[1, 0].set_title(f"Order-1 scattering path energies  (error = {se_demo:.3f})")
axes[1, 0].set_xlabel("Scale index")
axes[1, 0].legend()

# Bottom-right: error vs. deformation amplitude
axes[1, 1].plot(tau_values, fourier_errors, 'o-', color='#d7191c', lw=1.5,
                label='Fourier modulus')
axes[1, 1].plot(tau_values, scat_errors,    's-', color='#1a9641', lw=1.5,
                label='Scattering')
axes[1, 1].set_title("Relative error vs. deformation amplitude")
axes[1, 1].set_xlabel(r"$\tau_{\max}$ (fraction of signal length)")
axes[1, 1].set_ylabel("Relative $L^2$ error")
axes[1, 1].legend()
axes[1, 1].set_xlim(0, None)
axes[1, 1].set_ylim(0, None)

plt.suptitle("Deformation Stability: Scattering vs Fourier Modulus", fontsize=13)
plt.tight_layout()
plt.savefig("demo2_stability.png", dpi=150, bbox_inches='tight')
plt.show()
```

The bottom-right panel is the key result: the Fourier error grows rapidly and nonlinearly with deformation amplitude, while the scattering error grows slowly and approximately *linearly* - consistent with the Lipschitz bound $\|S_J x_\tau - S_J x\| \lesssim \|\nabla\tau\|_\infty$.

---

## Demo 3 - Texture Discrimination: Same Spectrum, Different Structure

This demo reproduces the central experiment from Section 4.2 of Mallat (2012): two stochastic processes with **identical power spectra** (i.e., identical second-order statistics) but different higher-order statistics. The Fourier modulus cannot distinguish them; second-order scattering can.

The theoretical explanation: the expected scattering coefficient $\mathbb{E}[S_J[p]X]$ for a path $p$ of length $m$ captures moments of $X$ up to order $2^m$. First-order scattering ($m=1$) depends only on second-order moments (the power spectrum); second-order scattering ($m=2$) depends on up to fourth-order moments, enabling discrimination.

```python
"""
Demo 3: Texture Discrimination
================================
Two processes with identical power spectra but different higher-order statistics.
First-order scattering fails to distinguish them; second-order succeeds.

Reproduces the experiment in Mallat (2012), Section 4.2.

Install: pip install kymatio numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from kymatio.numpy import Scattering1D

rng = np.random.default_rng(42)
T   = 2**14     # long signal for reliable moment estimation
J   = 7
Q   = 8

# -- Shared spectral envelope ---------------------------------------------------
# Both processes are constructed by filtering in the Fourier domain.
# The envelope is the same → same PSD.
def spectral_envelope(T):
    freqs    = np.fft.rfftfreq(T)
    envelope = np.exp(-((freqs - 0.05)**2) / (2 * 0.008**2))   # Gaussian peak at f=0.05
    return envelope

# -- Process A: Gaussian stationary texture ------------------------------------
def gaussian_texture(T, rng):
    white    = rng.standard_normal(T)
    W        = np.fft.rfft(white)
    env      = spectral_envelope(T)
    signal   = np.fft.irfft(W * env, n=T)
    return signal / signal.std()

# -- Process B: Sparse (non-Gaussian) texture ----------------------------------
# Sparse random impulses convolved with the same spectral envelope.
# The power spectrum is the same (convolution theorem), but the signal
# has heavy tails and clustering → different 4th-order moments.
def sparse_texture(T, rng, density=0.008):
    spikes   = (rng.uniform(0, 1, T) < density).astype(float)
    spikes  *= rng.standard_normal(T)    # random-sign impulses
    W        = np.fft.rfft(spikes)
    env      = spectral_envelope(T)
    signal   = np.fft.irfft(W * env, n=T)
    return signal / signal.std()

proc_A = gaussian_texture(T, rng)
proc_B = sparse_texture(T, rng)

# -- Power spectral density comparison -----------------------------------------
PSD_A = np.abs(np.fft.rfft(proc_A))**2
PSD_B = np.abs(np.fft.rfft(proc_B))**2
psd_corr = np.corrcoef(PSD_A, PSD_B)[0, 1]
print(f"PSD correlation (A vs B): {psd_corr:.5f}  <-- near 1.0 (indistinguishable by spectrum)")

# -- Kurtosis: direct evidence of non-Gaussianity ------------------------------
from scipy.stats import kurtosis
kurt_A = kurtosis(proc_A)
kurt_B = kurtosis(proc_B)
print(f"Kurtosis A (Gaussian):   {kurt_A:.2f}  <-- near 0")
print(f"Kurtosis B (sparse):     {kurt_B:.2f}  <-- large positive (heavy tails)")

# -- Scattering representations ------------------------------------------------
scat  = Scattering1D(J=J, shape=T, Q=Q)
Sx_A  = scat(proc_A)
Sx_B  = scat(proc_B)
meta  = scat.meta()
order = meta['order']

# Mean scattering energy per path (approximates the expected scattering)
E_A = np.mean(np.abs(Sx_A), axis=1)
E_B = np.mean(np.abs(Sx_B), axis=1)

# Relative difference per path
rel_diff = np.abs(E_A - E_B) / (0.5 * (E_A + E_B) + 1e-10)

for m in [1, 2]:
    mask     = order == m
    avg_diff = rel_diff[mask].mean()
    print(f"Mean relative scattering difference (order {m}): {avg_diff:.4f}"
          f"  <-- {'LOW: indistinguishable' if avg_diff < 0.1 else 'HIGH: discriminated!'}")

# -- Plot -----------------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(14, 11))

# Row 0: sample realizations
axes[0, 0].plot(proc_A[:800], lw=0.7, color='#2c7bb6')
axes[0, 0].set_title(f"Process A - Gaussian  (kurtosis = {kurt_A:.1f})")
axes[0, 1].plot(proc_B[:800], lw=0.7, color='#d7191c')
axes[0, 1].set_title(f"Process B - Sparse  (kurtosis = {kurt_B:.1f})")

# Row 1: PSDs (should look nearly identical)
freqs = np.fft.rfftfreq(T)
mask_f = freqs < 0.15
axes[1, 0].plot(freqs[mask_f], PSD_A[mask_f], color='#2c7bb6', lw=0.8, label='A')
axes[1, 0].plot(freqs[mask_f], PSD_B[mask_f], color='#d7191c', lw=0.8, alpha=0.7, label='B')
axes[1, 0].set_title(f"Power spectra  (correlation = {psd_corr:.4f} - nearly identical)")
axes[1, 0].legend()
axes[1, 0].set_xlabel("Frequency")

axes[1, 1].plot(freqs[mask_f], np.abs(PSD_A - PSD_B)[mask_f], color='gray', lw=0.8)
axes[1, 1].set_title("PSD difference  (small - Fourier cannot discriminate)")
axes[1, 1].set_xlabel("Frequency")

# Row 2: First-order scattering (should look similar)
idx1   = np.where(order == 1)[0]
width  = 0.35
x_idx1 = np.arange(len(idx1))
axes[2, 0].bar(x_idx1 - width/2, E_A[idx1], width=width, color='#2c7bb6',
               label='A', alpha=0.85)
axes[2, 0].bar(x_idx1 + width/2, E_B[idx1], width=width, color='#d7191c',
               label='B', alpha=0.85)
d1 = rel_diff[order == 1].mean()
axes[2, 0].set_title(f"Order-1 scattering energies  (mean rel. diff = {d1:.3f} - similar)")
axes[2, 0].set_xlabel("Scale index")
axes[2, 0].legend()

# Row 2 right: Second-order scattering (should look different)
idx2    = np.where(order == 2)[0]
x_idx2  = np.arange(len(idx2))
# Show every 4th path for clarity
step    = max(1, len(idx2) // 80)
idx2s   = idx2[::step]
x_idx2s = np.arange(len(idx2s))
axes[2, 1].bar(x_idx2s - width/2, E_A[idx2s], width=width, color='#2c7bb6',
               label='A', alpha=0.85)
axes[2, 1].bar(x_idx2s + width/2, E_B[idx2s], width=width, color='#d7191c',
               label='B', alpha=0.85)
d2 = rel_diff[order == 2].mean()
axes[2, 1].set_title(f"Order-2 scattering energies  (mean rel. diff = {d2:.3f} - DIFFERENT)")
axes[2, 1].set_xlabel("Path index (subsampled)")
axes[2, 1].legend()

plt.suptitle("Texture Discrimination: identical PSD, different higher-order scattering", fontsize=13)
plt.tight_layout()
plt.savefig("demo3_texture.png", dpi=150, bbox_inches='tight')
plt.show()
```

The result is stark: order-1 scattering energies are nearly identical between the two processes (consistent with the fact that they share a power spectrum), while order-2 coefficients diverge significantly.

---

## Demo 4 - Multifractal Analysis: Capturing Intermittency

One of the most powerful applications of scattering is robust estimation of *multifractal* properties of stochastic processes. Classical wavelet moment estimators are unstable for heavy-tailed processes because high polynomial moments have large variance. Scattering moments are computed with a non-expansive operator and are therefore statistically stable.

For a self-similar process with Hurst exponent $H$, the renormalized first-order scattering satisfies:
$$\tilde{S}_X(j) := \frac{\mathbb{E}[|X \otimes \psi_j|]}{\mathbb{E}[|X \otimes \psi_0|]} = 2^{jH}$$

and the deviation from linearity of $\log \tilde{S}_X(j)$ vs. $j$ measures intermittency (the curvature of the scaling exponent $\zeta(q)$).

```python
"""
Demo 4: Multifractal Analysis with Scattering Moments
======================================================
Compare three stochastic processes with different intermittency:
  - Fractional Brownian Motion (fBm): Gaussian, self-similar, H=0.7
  - Ornstein-Uhlenbeck (OU):          Gaussian stationary, finite-range correlation
  - Multifractal Random Walk (MRW):   Non-Gaussian, intermittent

Install: pip install kymatio numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from kymatio.numpy import Scattering1D
from scipy.stats import kurtosis

rng = np.random.default_rng(0)
T   = 2**14
J   = 8
Q   = 1   # Q=1 for multiscale analysis (log-scale resolution)

# -- Process generators ---------------------------------------------------------
def fractional_brownian_motion(T, H, rng):
    """
    Generate fBm via spectral synthesis (Davies-Harte method approximation).
    H in (0,1): Hurst exponent. H=0.5 → standard Brownian motion.
    """
    freqs   = np.fft.rfftfreq(T)[1:]       # skip DC
    phases  = rng.uniform(0, 2 * np.pi, len(freqs))
    amplitudes = freqs ** (-(H + 0.5))     # power-law spectral density
    W       = amplitudes * np.exp(1j * phases)
    W       = np.concatenate([[0], W])
    fbm     = np.fft.irfft(W, n=T)
    return (fbm - fbm.mean()) / fbm.std()

def ornstein_uhlenbeck(T, theta, rng):
    """
    OU process: dX = -theta*X dt + dW. Stationary, short-range correlation.
    theta: mean-reversion speed.
    """
    x  = np.zeros(T)
    dt = 1 / T
    for i in range(1, T):
        x[i] = x[i-1] - theta * x[i-1] * dt + np.sqrt(dt) * rng.standard_normal()
    return (x - x.mean()) / x.std()

def multifractal_random_walk(T, lambda2, rng, n_scales=10):
    """
    Multifractal Random Walk (Bacry & Muzy, 2003) approximation.
    lambda2 > 0 controls intermittency (larger → more intermittent).
    Constructed as: X(t) = sum_j B_j(t) * exp(omega_j(t))
    where omega_j are correlated log-normal multipliers.
    """
    # Approximate via log-normal cascade
    freqs = np.fft.rfftfreq(T)[1:]
    # Logarithmic covariance: C(j1-j2) = lambda2 * log(T/|j1-j2|)
    log_vol = np.zeros(T)
    for _ in range(n_scales):
        phase    = rng.uniform(0, 2 * np.pi, len(freqs))
        amp      = freqs ** (-0.5)
        w        = np.fft.irfft(amp * np.exp(1j * phase), n=T)
        log_vol += w * np.sqrt(lambda2 / n_scales)
    envelope = np.exp(log_vol - log_vol.var() / 2)
    noise    = rng.standard_normal(T)
    signal   = noise * envelope
    # Integrate to get random walk
    signal   = np.cumsum(signal) / np.sqrt(T)
    return (signal - signal.mean()) / signal.std()

# -- Generate processes ---------------------------------------------------------
fbm = fractional_brownian_motion(T, H=0.7, rng=rng)
ou  = ornstein_uhlenbeck(T, theta=50, rng=rng)
mrw = multifractal_random_walk(T, lambda2=0.04, rng=rng)

processes = {'fBm (H=0.7)': fbm, 'OU process': ou, 'MRW (intermittent)': mrw}
colors     = {'fBm (H=0.7)': '#2c7bb6', 'OU process': '#1a9641', 'MRW (intermittent)': '#d7191c'}

# -- Scattering moments ---------------------------------------------------------
scat  = Scattering1D(J=J, shape=T, Q=Q)
meta  = scat.meta()
order = meta['order']
scales_j1 = meta['j'][order == 1, 0]    # scale index for order-1 paths

def renormalized_scattering(x, Sx, order, scales_j1):
    """
    Compute normalized first-order scattering moments:
        tilde_S(j) = E[|U[j]x|] (approximated by spatial mean)
    and return log2(tilde_S(j)) vs j for scaling analysis.
    """
    S1 = Sx[order == 1]                  # shape: [n_scales, T//2^J]
    E1 = np.mean(np.abs(S1), axis=1)     # mean over time positions
    # Normalize by the coarsest scale
    E1_norm = E1 / (E1[-1] + 1e-12)
    return scales_j1, np.log2(E1_norm + 1e-12)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for col, (name, proc) in enumerate(processes.items()):
    color = colors[name]
    kurt  = kurtosis(np.diff(proc))     # kurtosis of increments
    Sx    = scat(proc)

    # Top: signal realization
    axes[0, col].plot(proc[:1000], lw=0.6, color=color)
    axes[0, col].set_title(f"{name}\n(increment kurtosis = {kurt:.1f})")
    axes[0, col].set_xlabel("Time")
    if col == 0:
        axes[0, col].set_ylabel("Amplitude")

    # Bottom: log-scattering vs. scale (slope = Hurst exponent for self-similar)
    j_idx, log_E = renormalized_scattering(proc, Sx, order, scales_j1)

    # Fit a line to the scaling region (all scales)
    valid = np.isfinite(log_E)
    if valid.sum() >= 2:
        p = np.polyfit(j_idx[valid], log_E[valid], 1)
        fit_line = np.polyval(p, j_idx)
        H_est = p[0]
    else:
        fit_line = np.zeros_like(j_idx, dtype=float)
        H_est = float('nan')

    axes[1, col].plot(j_idx, log_E, 'o-', color=color, lw=1.5, ms=5,
                      label='scattering moments')
    axes[1, col].plot(j_idx, fit_line, '--', color='black', lw=1.2, alpha=0.6,
                      label=f'slope ≈ {H_est:.2f}')
    axes[1, col].set_title(f"Log-scattering scaling  (slope = H estimate)")
    axes[1, col].set_xlabel("Scale j")
    if col == 0:
        axes[1, col].set_ylabel(r"$\log_2 \tilde{S}(j)$")
    axes[1, col].legend(fontsize=9)

    print(f"{name:25s}  kurtosis = {kurt:6.1f},  estimated H = {H_est:.3f}")

plt.suptitle("Multifractal Analysis via Scattering Moments", fontsize=13)
plt.tight_layout()
plt.savefig("demo4_multifractal.png", dpi=150, bbox_inches='tight')
plt.show()
```

The three processes are designed to have progressively more intermittency:
- **fBm**: linear log-scattering curve with slope $\approx H$ - the transform correctly recovers the Hurst exponent
- **OU**: non-power-law curve (finite correlation length → rolls off at large scales)
- **MRW**: non-linear log-scattering curve with clear curvature - the "signature" of multifractality

The curvature $\zeta(2) - 2\zeta(1) < 0$ is directly detectable from the decay of second-order scattering coefficients $\tilde{S}(j_1, j_2)$ as a function of $j_2 - j_1$, providing a statistically robust intermittency estimator.

---

## 10. Extensions and Applications

### Roto-Translation Scattering

For images, the scattering transform extends to the roto-translation group $G_\text{rot} \cong \mathbb{R}^2 \rtimes SO(2)$, building joint invariants to both translations and rotations. The key distinction from a *separable* approach (first translation-invariant, then rotation-invariant) is that the joint representation can discriminate textures that a separable one cannot - for example, distinguishing a texture from its mirror image.

### Time-Frequency Scattering for Audio

Audio recognition requires stability to both time-warps and frequency transpositions. The signal is first lifted to the time-frequency plane via a scalogram $z(t, \lambda) = |x \otimes \psi_\lambda(t)|$, and then a *joint* wavelet decomposition is applied to $z$ over the roto-translation group of time-frequency shifts. This is the basis of state-of-the-art audio classification systems.

### Quantum Chemistry (Solid Harmonic Scattering)

For 3D molecular signals, rotational and translational invariance are physically mandated - quantum-mechanical energies cannot depend on the molecule's orientation. Scattering representations over $SO(3)$ using solid harmonic wavelets achieve competitive accuracy on QM7/QM9 datasets for energy regression, with formal stability guarantees.

### Graph and Manifold Scattering

For data on graphs (social networks, molecular graphs), there is no global group structure. The scattering formalism extends by replacing Euclidean wavelets with **diffusion wavelets** built from the graph Laplacian $L = D - A$. The $k$-th diffusion wavelet captures signal variations at the $k$-th diffusion time scale. Geometric stability is now expressed in terms of metric perturbations of the graph structure.

---

## 11. Summary

| Property | Fourier Modulus | Scattering |
|---|---|---|
| Translation invariant | Y | Y (asymptotically) |
| Stable to additive noise | Y | Y |
| Lipschitz to deformations | N | Y |
| Energy conserving | N (information loss) | Y |
| Captures higher-order moments | N | Y (order $m$ → $2^m$-th moments) |
| Generalizes to non-Euclidean | N | Y (Lie groups, graphs, manifolds) |
| Filters learned from data | N/A | N (mathematically fixed) |

The scattering transform occupies a unique position: it is simultaneously a theoretically grounded signal processing tool and a practical deep learning architecture. Its provable properties make it a mathematical template for understanding what CNNs implicitly learn when trained on structured signal domains - and its computable, non-learned nature makes it a powerful feature extractor in the data-scarce regime where large networks overfit.

---

## References

1. Mallat, S. (2012). *Group invariant scattering*. Communications on Pure and Applied Mathematics, 65(10), 1331–1398.
2. Bruna, J., & Mallat, S. (2013). *Invariant scattering convolution networks*. IEEE TPAMI, 35(8), 1872–1886.
3. Waldspurger, I. (2017). *Exponential decay of scattering coefficients*. SampTA.
4. Oyallon, E., & Mallat, S. (2015). *Deep roto-translation scattering for object classification*. CVPR.
5. Andreux, M. et al. (2020). *Kymatio: Scattering transforms in Python*. JMLR, 21(60), 1–6.
6. Bacry, E., & Muzy, J.-F. (2003). *Log-infinitely divisible multiscale random walk processes*. Communications in Mathematical Physics, 236(3), 449–475.

---

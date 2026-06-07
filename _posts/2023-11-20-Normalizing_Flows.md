---
title:  "Normalizing Flows"
mathjax: true
layout: post
categories: media
---

Assuming \\(\textbf{z}\in \mathbb{R}^m\\) and \\(\textbf{x} \in \mathbb{R}^n\\) are two random variables which are related as \\(x=f(z)\\), then

$$
p_x(\textbf{x}) = p_z(f(\textbf{x}))\left|\text{det}\left(\frac{\partial f(\textbf{x})}{\partial \textbf{x}}\right)\right|
$$

Given a dataset \\(\textbf{x}_1, ..., \textbf{x}_n\\) and a prior distribution \\(p_z(\textbf{z})\\), the idea is to search for a probability of data \\(p_x(\textbf{x})\\) via an unknown function \\(\textbf{z}=f(\textbf{x})\\). Typically, the function \\(f\\) is approximated via a neural network and its weights are obtained via maximum log-likelihood:

$$
-\sum_i \log(p_x(\textbf{x}_i)) = -\sum_i \log(p_z(f(\textbf{x}_i))) - \log\left|\text{det}\left(J_f(\textbf{x}_i)\right)\right|
$$

here, \\( (J_f(\textbf{x}))_{ij} = \frac{\partial f_i(\textbf{x})}{\partial x_j}\\) is the Jacobian matrix of function \\( f: \mathbb{R}^n \rightarrow \mathbb{R}^m \\)

In [Real NVP](https://arxiv.org/abs/1605.08803), the function \\(f\\) is obtained via stacking affine coupling layers. More precisely, from input \\(\textbf{x} \in \mathbb{R}^D\\) and output \\(\textbf{y} \in \mathbb{R}^d\\) with \\(d<D\\), the intermediate layers are defined as:

$$
\begin{aligned}
\textbf{y}_{1:d} &= \textbf{x}_{1:d}\\
\textbf{y}_{d+1:D} &= \textbf{x}_{d+1:D} \odot \exp(s(\textbf{x}_{1:d})) + t(\textbf{x}_{1:d})
\end{aligned}
$$

where \\(s\\) (scale) and \\(t\\) (translation) are neural networks mapping \\(\mathbb{R}^d\\) to \\(\mathbb{R}^{D-d}\\) 

The nice property of this affine coupling layers design is that it is invertible:

$$
\begin{cases}
\textbf{y}_{1:d} &= \textbf{x}_{1:d}\\
\textbf{y}_{d+1:D} &= \textbf{x}_{d+1:D} \odot \exp(s(\textbf{x}_{1:d})) + t(\textbf{x}_{1:d})
\end{cases}
\Longleftrightarrow
\begin{cases}
\textbf{x}_{1:d} &= \textbf{y}_{1:d}\\
\textbf{x}_{d+1:D} &= (\textbf{y}_{d+1:D}-t(\textbf{y}_{1:d})) \odot \exp(s(\textbf{y}_{1:d}))
\end{cases}
$$

The Jacobian of an affine coupling layer is:

$$
J(x) = \frac{\partial \textbf{y}}{\partial \textbf{x}} = 
\begin{bmatrix}
\mathbb{I}_d & \textbf{0}_{d\times (D-d)}\\
\frac{\partial \textbf{y}_{d+1:D}}{\partial \textbf{x}_{1:d}} & \text{diag}(\exp(s(\textbf{x}_{1:d})))
\end{bmatrix}
$$

The determinant of Jacobian matrix is:

$$
\left| \text{det}(J(\textbf{x})) \right| = \exp\left(\sum_{j=1}^{D-d} s(\textbf{x}_{1:d})_j\right)
$$

In the implementation, a binary mask \\(b=(1,...,1,0,...,0)\\) is used to describe the coupling affine layer:

$$
\begin{aligned}
\textbf{y} &= \textbf{x} \odot \exp\left((1-\textbf{b})\odot s(\textbf{b}\odot \textbf{x})\right) + (1-\textbf{b}) \odot t(\textbf{b}\odot \textbf{x})\\
\textbf{x} &= (\textbf{y}-(1-\textbf{b})\odot t(\textbf{b}\odot \textbf{y})) \odot \exp\left(-(1-\textbf{b})\odot s(\textbf{b}\odot \textbf{y})\right)
\end{aligned}
$$

With this formulation, the determinant is now:

$$
\log\left|\text{det}(J(\textbf{x}))\right| = \sum_{j=1}^D \left((1-\textbf{b})\odot s(\textbf{b}\odot \textbf{x})\right)_j
$$

Below is a simple tutorial example with two-moons distribution:

```python
"""
simple RealNVP normalizing flow implementation.

convention (matches the corrected blog notes):
    f : data x  ->  latent z          (this is what the network computes)
    z = f(x),  with z ~ N(0, I) as the prior p_z

density via change of variables:
    log p_x(x) = log p_z(f(x)) + log|det J_f(x)|

a RealNVP affine coupling layer (with binary mask b):
    z = x * exp((1-b) * s(b*x)) + (1-b) * t(b*x)
    log|det J| = sum( (1-b) * s(b*x) )
inverse (for sampling):
    x = (z - (1-b)*t(b*z)) * exp(-(1-b)*s(b*z))
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)
DIM = 2


class CouplingLayer(nn.Module):
    """one affine coupling layer with a fixed binary mask b."""
    def __init__(self, mask, hidden=128):
        super().__init__()
        self.register_buffer("mask", mask)  # shape (DIM,), entries in {0,1}
        # s and t share an architecture; each maps R^DIM -> R^DIM
        self.s_net = nn.Sequential(
            nn.Linear(DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, DIM), nn.Tanh(),   # tanh keeps scales stable
        )
        self.t_net = nn.Sequential(
            nn.Linear(DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, DIM),
        )

    def forward(self, x):
        """x -> z  (data to latent). Returns z and log|det J|."""
        b = self.mask
        x_masked = x * b
        s = self.s_net(x_masked) * (1 - b)
        t = self.t_net(x_masked) * (1 - b)
        z = x * torch.exp(s) + t
        log_det = s.sum(dim=1)          # sum over the transformed coords
        return z, log_det

    def inverse(self, z):
        """z -> x  (latent to data), the exact inverse of forward."""
        b = self.mask
        z_masked = z * b
        s = self.s_net(z_masked) * (1 - b)
        t = self.t_net(z_masked) * (1 - b)
        x = (z - t) * torch.exp(-s)
        return x


class RealNVP(nn.Module):
    """stack of coupling layers with alternating masks."""
    def __init__(self, n_layers=6, hidden=128):
        super().__init__()
        masks = []
        for i in range(n_layers):
            # alternate which coordinate is passed through
            m = torch.tensor([i % 2, (i + 1) % 2], dtype=torch.float32)
            masks.append(m)
        self.layers = nn.ModuleList(CouplingLayer(m, hidden) for m in masks)
        # standard normal prior p_z
        self.register_buffer("prior_mean", torch.zeros(DIM))
        self.register_buffer("prior_std", torch.ones(DIM))

    def forward(self, x):
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.layers:
            z, ld = layer(z)
            log_det_total += ld
        return z, log_det_total

    def inverse(self, z):
        x = z
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, x):
        z, log_det = self.forward(x)
        # log p_z(z) for standard normal
        log_pz = (-0.5 * (z ** 2) - 0.5 * np.log(2 * np.pi)).sum(dim=1)
        return log_pz + log_det          # change-of-variables formula

    def sample(self, n):
        z = torch.randn(n, DIM)
        return self.inverse(z)


def main():
    
    # data
    X, _ = make_moons(n_samples=3000, noise=0.05)
    X = (X - X.mean(0)) / X.std(0) # standardize
    X = torch.tensor(X, dtype=torch.float32)

    model = RealNVP(n_layers=6, hidden=128)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train: maximize log-likelihood = minimize negative log-likelihood
    n_epochs = 10000
    for epoch in range(n_epochs):
        opt.zero_grad()
        loss = -model.log_prob(X).mean()
        loss.backward()
        opt.step()
        if (epoch + 1) % 500 == 0:
            print(f"epoch {epoch+1:4d}  nll = {loss.item():.4f}")

    # visualize: real data vs. samples drawn from the flow
    with torch.no_grad():
        samples = model.sample(3000).numpy()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(X[:, 0], X[:, 1], s=4, alpha=0.5)
    ax[0].set_title("real data (two moons)")
    ax[1].scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.5, color="crimson")
    ax[1].set_title("samples from trained flow")
    for a in ax:
        a.set_xlim(-2.5, 2.5); a.set_ylim(-2.5, 2.5); a.set_aspect("equal")
    plt.tight_layout()
    plt.show()

    # sanity check
    with torch.no_grad():
        z, _ = model.forward(X[:5])
        x_rec = model.inverse(z)
        err = (x_rec - X[:5]).abs().max().item()
        print(f"max reconstruction error (x -> z -> x): {err:.2e}")


if __name__ == "__main__":
    main()
```

It should provide an image as follows:

![two_moons_distribution](/images/two_moons.png){:height="50%" width="100%"}

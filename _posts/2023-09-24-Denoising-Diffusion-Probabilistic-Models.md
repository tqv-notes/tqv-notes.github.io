---
title:  "Denoising Diffusion Probabilistic Models"
mathjax: true
layout: post
categories: media
---

In this note, we will present the core ideas of denosing using diffusion models as first demonstrated in Ho *et al*. [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf/)

## Forward diffusion process

For a given variance schedule \\( \beta_1 < \beta_2, ... < \beta_T \\),

$$
\begin{aligned}
q(x_t|x_{t-1}) &= \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1},\beta_t \mathbb{I})\\
q(x_{0:T}|x_{0}) &= \prod_{t=1}^T q(x_t|x_{t-1})
\end{aligned}
$$

Let \\( \alpha_t = 1-\beta_t \\) and \\( \overline{\alpha}\_t = \prod_{i=1}^t \alpha_i \\), then

$$
\begin{aligned}
x_t &= \sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1} \\
    &= \sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_t \alpha_{t-1}} \epsilon_{t-2} \\
    &= ...\\
    &= \sqrt{\overline{\alpha}_t} x_{0} + \sqrt{1-\overline{\alpha}_t} \epsilon
\end{aligned}
$$

## Reverse diffusion process

The reverse diffusion process \\( q(x_{t-1}\|x_{t}) \\) is approximated with a learned model \\(p_\theta\\) as below:

$$
p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1}|x_t) ~~\text{where},~ p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1},\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))
$$

where, \\( p(x_T) \sim \mathcal{N}(0,\mathbb{I}) \\).

The model parameters can be estimated via log-likelihood \\( \mathbb{E}_{q(x_0)}\left[ \log( p\_{\theta}(x_0) ) \right] \\). It is usually more convenient to replace the log-likelihood optimization with the variational lower bound as:

$$
\begin{aligned}
\mathbb{E}_{q(x_0)}\left[ \log( p_{\theta}(x_0) ) \right] & \geq \mathbb{E}_{q(x_{0:T})} \left[ \log\left( \frac{q(x_{1:T}|x_0)}{p_{\theta}(x_{0:T})} \right) \right] \\
& = \mathbb{E}_{q}\left[ KL(q(x_T|x_0) || p(x_T)) + \sum_{t=2}^T KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1} || x_t)) -\log(p_{\theta}(x_0|x_1)) \right] \\
& = L_T + \sum_{t=2}^T L_{t-1} + L_0
\end{aligned}
$$

It is possible to show that: 

$$
q(x_{t-1}|x_t,x_0) = \mathcal{N}\left( x_{t-1}; \mu(x_t,x_0), \gamma_t \mathbb{I} \right)
$$

where,

$$
\begin{aligned}
\mu(x_t,x_0) &= \frac{1}{\sqrt{\alpha}_t} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\overline{\alpha}_t}} \epsilon \right) = \mu(x_t,t)\\
\gamma_t &= \frac{1-\overline{\alpha}_{t-1}}{1-\overline{\alpha}_t} \beta_t
\end{aligned}
$$

Since \\( p(x_{t-1} \| x_t) \\) is also a Gaussian, then

$$
L_t = \mathbb{E}_q \left[ \frac{1}{2 \beta_t^2} \| \mu_{\theta}(x_t,t) - \mu(x_t,t) \|^2 \right]
$$

To obtain a neat analytical expression, we need to make the change of variable as:

$$
\mu_{\theta}(x_t,t) = \frac{1}{\sqrt{\alpha}_t} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\overline{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
$$

then,

$$
\| \mu_{\theta}(x_t,t) - \mu(x_t,t) \|^2 = \frac{(1-\alpha_t)^2}{1-\overline{\alpha}_t} \| \epsilon - \epsilon(\sqrt{\overline{\alpha}_t} x_{0} + \sqrt{1-\overline{\alpha}_t} \epsilon, t) \|^2
$$

## Implementation

We provide here a simple tutorial code for two moons distribution:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

device = "cuda" if torch.cuda.is_available() else "cpu"

# DDPM schedule
T = 500
betas = torch.linspace(1e-4, 0.02, T, device=device)
alphas = 1.0 - betas

# alpha_bar[0] = 1
alpha_bar = torch.cat([torch.ones(1, device=device),torch.cumprod(alphas, dim=0)])

# dataset
def get_batch(n=512):
    x, _ = make_moons(n_samples=n, noise=0.05)
    x = torch.tensor(x,dtype=torch.float32,device=device)
    
    # normalize for more stable training
    x = (x - x.mean(0)) / x.std(0)

    return x

# sinusoidal timestep embedding
EMB_DIM = 64
half = EMB_DIM // 2
freqs = torch.exp(-torch.arange(half, device=device)* math.log(10000)/ (half - 1))

def timestep_embedding(t):
    args = t[:, None].float() * freqs[None] # t: (batch,)

    return torch.cat([torch.sin(args), torch.cos(args)],dim=-1)

# epsilon predictor
class EpsNet(nn.Module):

    def __init__(self, data_dim=2, t_dim=EMB_DIM, hidden=128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(data_dim + t_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, data_dim)
        )

    def forward(self, x, t):
        t_emb = timestep_embedding(t)
        return self.net(torch.cat([x, t_emb], dim=-1))

model = EpsNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# forward process
def q_sample(x0, t, eps=None):

    if eps is None:
        eps = torch.randn_like(x0)

    ab = alpha_bar[t][:, None]
    xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps

    return xt

# training
for step in range(5000):

    x0 = get_batch()

    # sample timestep uniformly
    t = torch.randint(1, T + 1, (x0.shape[0],), device=device)

    eps = torch.randn_like(x0)
    xt = q_sample(x0, t, eps)
    eps_pred = model(xt, t)
    loss = F.mse_loss(eps_pred,eps)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 1000 == 0:
        print(f"step {step:5d} loss {loss.item():.4f}")

# visualize forward diffusion
@torch.no_grad()
def plot_forward_process():

    x0 = get_batch(2000)

    timesteps = [0, int(T/20), int(T/10), int(T/5), int(T)]
    fig, axes = plt.subplots(1,len(timesteps),figsize=(15, 3))

    for ax, t in zip(axes, timesteps):
        tt = torch.full((len(x0),), t, device=device, dtype=torch.long)
        xt = q_sample(x0, tt)
        xt = xt.cpu()
        ax.scatter(xt[:, 0], xt[:, 1], s=2)
        ax.set_title(f"t={t}")
        ax.axis("equal")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

plot_forward_process()

# reverse process
@torch.no_grad()
def sample(n=2000):

    model.eval()

    x = torch.randn(n, 2, device=device)

    for t in reversed(range(1, T + 1)):

        tt = torch.full((n,), t, device=device, dtype=torch.long)
        eps_pred = model(x,tt)

        beta_t = betas[t - 1]
        alpha_t = alphas[t - 1]
        ab_t = alpha_bar[t]
        ab_prev = alpha_bar[t - 1]
        
        mean = (x- beta_t/ torch.sqrt(1 - ab_t)* eps_pred) / torch.sqrt(alpha_t)

        if t > 1:
            posterior_var = (1 - ab_prev)/ (1 - ab_t)* beta_t
            noise = torch.randn_like(x)
            x = mean+ torch.sqrt(posterior_var)* noise
        else:
            x = mean

    model.train()

    return x.cpu()

# generate samples
samples = sample()

plt.figure(figsize=(6, 6))
plt.scatter(samples[:, 0], samples[:, 1], s=4)
plt.axis("equal")
plt.title("DDPM samples")
plt.show()
```

Below is the result for the forward process:

![ddpm_forward](/images/ddpm_forward.png){:height="50%" width="100%"}

and, here is the result for reverse process:

![ddpm_forward](/images/ddpm_sample.png){:height="50%" width="50%"}

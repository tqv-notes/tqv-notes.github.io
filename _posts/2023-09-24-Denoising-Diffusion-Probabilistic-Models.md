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

Since \\( p(x_{t-1} \|\| x_t) \\) is also a Gaussian, then

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

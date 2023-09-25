---
title:  "Denoising Diffusion Probabilistic Models"
mathjax: true
layout: post
categories: media
---

In this note, we will present the core ideas of denosing using diffusion models as first demonstrated in Ho \\(\textit{et al}\\). [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf/)

## Forward diffusion process

For a given variance schedule \\( \beta_1 < \beta_2, ... < \beta_T \\),

$$
\begin{aligned}
q(x_t|x_{t-1}) &= \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1},\beta_t \mathbb{I})\\
q(x_{0:T}|x_{0}) &= \prod_{t=1}^T q(x_t|x_{t-1})
\end{aligned}
$$

## Reverse diffusion process

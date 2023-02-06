---
title:  "Approximate Message Passing"
mathjax: true
layout: post
categories: media
---

## Introduction

The problem of our interest is the standard sparse linear regression:

$$ y = H x +n $$

where \\(y\in \mathbb{R}^M\\) is the observation, \\(x\in \mathbb{R}^N\\) is the sparse signal, \\(H\in \mathbb{R}^{M\times N} (M \ll N)\\) is the known design matrix and \\(n \in \mathbb{R}^N\\) is the additive white Gaussian noise with zero mean and covariance \\(\sigma^2\\).

## Iterative Soft Threshold Algorithm (ISTA)

The ISTA algorithm can be summarized as:

$$
\begin{align*}
r^{(t)} &= \hat{x}^{(t)} + \alpha_t H^T (y-H \hat{x}^{(t)}) \\
\hat{x}^{(t+1)} &= \text{sign}(r^{(t)}) \text{max}\left( \left|r^{(t)}\right| -\alpha_t \lambda, 0\right)
\end{align*}
$$

or, in an alternative form (here we set \\(\alpha_t=1\\) ):

$$
\begin{align*}
z^{(t)} &= y - H \hat{x}^{(t)} \\
\hat{x}^{(t+1)} &= \eta\left( \hat{x}^{(t)} + H^T z^{(t)},\lambda \right)
\end{align*}
$$



where, 

$$ \eta\left(x,\lambda\right) = \text{sign}(x) \text{max}\left(\left|x\right|, \lambda\right) $$

## Approximate Message Passing

We provide here the Approximate Message Passing (AMP) algorithm:

$$
\begin{align*}
z^{(t)} &= y - H \hat{x}^{(t)} + \frac{1}{\alpha} z^{(t-1)} \left\langle \eta^{\prime}_{t-1}\left( \hat{x}^{(t-1)} + H^T \hat{z}^{(t-1)} \right) \right\rangle\\
\hat{x}^{(t+1)} &= \eta_t\left( \hat{x}^{(t)} + H^T z^{(t)} \right)
\end{align*}
$$

where, \\( \langle x \rangle = \frac{1}{N} \sum_{i=1}^{N} x_i \\) and \\( \eta^\prime_{t-1} \\) is the derivative of \\( \eta_{t-1} \\) .

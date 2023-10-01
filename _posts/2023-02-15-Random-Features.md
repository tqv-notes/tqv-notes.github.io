---
title:  "Random Features"
mathjax: true
layout: post
categories: media
---

Random features for machine learning was first investigated within kernel machine learning framework (see [Rahimi & Recht, NIPS 2007](https://proceedings.neurips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html/)).

In the kernel methods, the approximation function is a linear combination of kernels as below:
$$
f(x) = \sum_{i=1}^{N} \alpha_i k(x,x_i)
$$

The kernel functions can be a Gaussian function as:

$$
k(x,x') = \exp\left( -\frac{1}{2} \|x-x'\|^2_2 \right)
$$

For large dataset, direct application of kernel is time-consuming due to the large matrix inverssion step. The idea of kernel approximation is to use a randomized feature map \\( x\in \mathbb{R}^d \rightarrow z(x) \in \mathbb{R}^D \\) with \\( D < d \\):

$$
k(x,x') = \sum_{j=1}^{D} z(x,\omega_j)^\top z(x',\omega_j)
$$

For Gaussian kernel, the feature map can be written as:

$$
z(x,\omega) = \exp(i \omega^\top x)
$$

where, \\( \omega \sim \mathcal{N}_D(0,\mathbb{I}) \\).

An alternative form of feature map for Gaussian kernel is:

$$
z(x,\omega) = \sqrt{2} \cos(\omega^\top x + b)
$$

where, \\( b\in \text{Uniform}(0,2\pi) \\).

With this kernel approximation, the approximation function is now:

$$
f(x) = \sum_{j=1}^{D} \beta_j z(x,\omega_j)
$$

The fundamental theory behind the kernel approximation with random features is the Bochner's theorem which states that for every kernel \\( k(x) \\) is a Fourier transform of a non-negative function \\( p(\omega) \\) as:

$$
k(x-y) = \int_{\mathbb{R}^d} p(\omega) \exp\left( i\omega^\top (x-y) \right) d\omega = \mathbb{E}_{p(\omega)}\left[ \exp(i\omega^\top x) \left(\exp(i\omega^\top y)\right)^\star \right]
$$

For Gaussian kernel \\( k(x) = \exp( - \|\|x\|\|^2/2) \\), the corresponding Fourier transformed function \\( p(\omega) \\) is also a Gaussian function. This leads to the choice of normal distribution for \\( \omega \\).

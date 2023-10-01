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
k(x,x') = \exp( -\theta (x-x')^2 )
$$

For large dataset, direct application of kernel is time-consuming due to the large matrix inverssion step. The idea of kernel approximation is to use a randomized feature map \\( x\in \mathbb{R}^d \rightarrow z(x) \in \mathbb{R}^D \\) with \\(D \l d\\):

$$
k(x,x') = \sum_{j=1}^{D} z(x,\omega_j) z(x',\omega_j)
$$

$$
f(x) = \sum_{j=1}^{D} \beta_j z(x,\omega_j)
$$

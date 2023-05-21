---
title:  "Random Features"
mathjax: true
layout: post
categories: media
---

Random features for machine learning was first investigated within kernel machine learning framework (see [Rahimi & Recht, NIPS 2007](https://proceedings.neurips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html/)).

$$
f(x) = \sum_{i=1}^{N} \alpha_i k(x,x_i)
$$

$$
k(x,x') = \sum_{j=1}^{D} z(x,\omega_j) z(x',\omega_j)
$$

$$
f(x) = \sum_{j=1}^{D} \beta_j z(x,\omega_j)
$$

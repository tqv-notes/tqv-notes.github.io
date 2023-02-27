---
title:  "Phase Retrieval"
mathjax: true
layout: post
categories: media
---

Phase retrieval problem searchs for \\( x^\star \in \mathbb{C}^d \\) from known measurements \\( y \in \mathbb{R}^n \\) and known design matrix \\( A \\) such that

$$
y = \left| A x^\star \right|^2
$$

$$
y_r = \left|\langle a_r, z \rangle \right|^2, ~~ \text{for}~ r = 1, 2, ..., d
$$

$$
L(z) = \frac{1}{2d} \sum_{r=1}^{d} \left(y_r - \left|\langle a_r, z \rangle \right|^2\right)^2
$$

$$
Y = \sum_{r=1}^{d} y_r a_r a_r^*
$$

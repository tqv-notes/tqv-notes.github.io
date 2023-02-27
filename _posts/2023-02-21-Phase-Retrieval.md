---
title:  "Phase Retrieval"
mathjax: true
layout: post
categories: media
---

Phase retrieval problem searchs for \\( x^\ast \in \mathbb{C}^d \\) from known measurements \\( y \in \mathbb{R}^n \\) and known design matrix \\( A \\) such that

$$
y = \left| A x^\ast \right|^2
$$

Suppose each row of matrix \\( A \\) is a vector \\( a_i^H \\) (here \\( H\\) symbol denotes Hermitian conjugation), we can rewrite above equation as:
$$
y_i = \left|\langle a_i, z \rangle \right|^2, ~~ \text{for}~ i = 1, 2, ..., d
$$

(note: the inner product \\( \langle x, y \rangle = x^H y\\) ).

$$
L(z) = \frac{1}{2d} \sum_{i=1}^{d} \left(y_i - \left|\langle a_i, z \rangle \right|^2\right)^2
$$

$$
Y = \sum_{i=1}^{d} y_i a_i a_i^*
$$

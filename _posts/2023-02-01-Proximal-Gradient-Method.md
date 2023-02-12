---
title:  "Proximal Gradient Method"
mathjax: true
layout: post
categories: media
---

It is quite common in many optimization to consider the following problem:

$$
\underset{x}{\text{minimize}}~\{f(x) = g(x) + h(x)\}
$$

here, \\( f(x) \\) can be seen as total cost function, \\( g(x) \\) is the standard least squares cost function and \\( h(x) \\) is the regularization term.

If \\( f(x) \\) is a smooth convex function, there are many optimization techniques in the literature but we will mention here two most basic ones: the Newton method and the gradient descent method.

The ultimate goal of this note is to present a method for the case \\( h(x) \\) is nondifferentiable: the proximal gradient method.

## Newton method

Consider a smooth convex function \\( f(x) \\), via Taylor expansion, we have:

$$
f(x) = f(x_0) + \nabla f(x_0)^T (x-x_0) + (x-x_0)^T\nabla^2 f(x_0) (x-x_0) +O((x-x_0)^3)
$$

At minimum position \\(x\_\*\\)), we have \\( \nabla f(x\_\*) = 0\\). This leads to:

$$
\nabla f(x_\*) + \nabla^2 f(x_\*) (x-x_\*) +O((x-x_\*)^2) = 0
$$

From this relation, we could derive the Newton algorithm as:

- initialize at \\( x_0 \\)
- repeat until convergence: \\( x_{k+1} = x_k - (\nabla^2 f(x_k))^{-1} \nabla f(x_k) \\)

## Gradient descent method

## Proximal gradient method

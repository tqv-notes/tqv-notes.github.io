---
title:  "Normalizing Flows"
mathjax: true
layout: post
categories: media
---

Assuming \\(\textbf{z}\in \mathbb{R}^m\\) and \\(\textbf{x} \in \mathbb{R}^n\\) are two random variables which are related as \\(x=f(z)\\), then

$$
p_x(\textbf{x}) = p_z(f(\textbf{x}))\left|\text{det}\left(\frac{\partial f(\textbf{x})}{\partial \textbf{x}}\right)\right|
$$

Given a dataset \\(\textbf{x}_1, ..., \textbf{x}_n\\) and a prior distribution \\(p_z(\textbf{z})\\), the idea is to search for a probability of data \\(p_x(\textbf{x})\\) via an unknown function \\(\textbf{z}=f(\textbf{x})\\). Typically, the function \\(f\\) is approximated via a neural network and its weights are obtained via maximum log-likelihood:

$$
-\sum_i \log(p_x(\textbf{x}_i)) = -\sum_i \log(p_z(f(\textbf{x}_i))) - \log\left|\text{det}\left(J_f(\textbf{x}_i)\right)\right|
$$

here, \\( (J_f(\textbf{x}))_{ij} = \frac{\partial f_i(\textbf{x})}{\partial x_j}\\) is the Jacobian matrix of function \\( f: \mathbb{R}^n \rightarrow \mathbb{R}^m \\)

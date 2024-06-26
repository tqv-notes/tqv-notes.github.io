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

In [Real NVP](https://arxiv.org/abs/1605.08803), the function \\(f\\) is obtained via stacking affine coupling layers. More precisely, from input \\(\textbf{x} \in \mathbb{R}^D\\) and output \\(\textbf{y} \in \mathbb{R}^d\\) with \\(d<D\\), the intermediate layers are defined as:

$$
\begin{aligned}
\textbf{y}_{1:d} &= \textbf{x}_{1:d}\\
\textbf{y}_{d+1:D} &= \textbf{x}_{d+1:D} \odot \exp(s(\textbf{x}_{1:d})) + t(\textbf{x}_{1:d})
\end{aligned}
$$

where \\(s\\) (scale) and \\(t\\) (translation) are neural networks mapping \\(\mathbb{R}^d\\) to \\(\mathbb{R}^{D-d}\\) 

The nice property of this affine coupling layers design is that it is invertible:

$$
\begin{cases}
\textbf{y}_{1:d} &= \textbf{x}_{1:d}\\
\textbf{y}_{d+1:D} &= \textbf{x}_{d+1:D} \odot \exp(s(\textbf{x}_{1:d})) + t(\textbf{x}_{1:d})
\end{cases}
\Longleftrightarrow
\begin{cases}
\textbf{x}_{1:d} &= \textbf{y}_{1:d}\\
\textbf{x}_{d+1:D} &= (\textbf{y}_{d+1:D}-t(\textbf{y}_{1:d})) \odot \exp(s(\textbf{y}_{1:d}))
\end{cases}
$$

The Jacobian of an affine coupling layer is:

$$
J(x) = \frac{\partial \textbf{y}}{\partial \textbf{x}} = 
\begin{bmatrix}
\mathbb{I}_d & \textbf{0}_{d\times (D-d)}\\
\frac{\partial \textbf{y}_{d+1:D}}{\partial \textbf{x}_{1:d}} & \text{diag}(\exp(s(\textbf{x}_{1:d})))
\end{bmatrix}
$$

The determinant of Jacobian matrix is:

$$
\left| \text{det}(J(\textbf{x})) \right| = \exp\left(\sum_{j=1}^{D-d} s(\textbf{x}_{1:d})_j\right)
$$

In the implementation, a binary mask \\(b=(1,...,1,0,...,0)\\) is used to describe the coupling affine layer:

$$
\begin{aligned}
\textbf{y} &= \textbf{x} \odot \exp\left((1-\textbf{b})\odot s(\textbf{b}\odot \textbf{x})\right) + (1-\textbf{b}) \odot t(\textbf{b}\odot \textbf{x})\\
\textbf{x} &= (\textbf{y}-(1-\textbf{b})\odot t(\textbf{b}\odot \textbf{y})) \odot \exp\left(-(1-\textbf{b})\odot s(\textbf{b}\odot \textbf{y})\right)
\end{aligned}
$$

With this formulation, the determinant is now:

$$
\log\left|\text{det}(J(\textbf{x}))\right| = \sum_{j=1}^D \left((1-\textbf{b})\odot s(\textbf{b}\odot \textbf{x})\right)_j
$$


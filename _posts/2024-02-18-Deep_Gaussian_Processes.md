---
title:  "Deep Gaussian Processes"
mathjax: true
layout: post
categories: media
---

## Sparse Gaussian Processes

Assuming a dataset \\( (X,y) \\) of \\(N\\) training samples, building a Gaussian process requires an inversion of \\(N\times N\\) matrix which takes \\( \mathcal{O}(N^3) \\) time.

The idea of sparse GP with inducing points is to replace the original data with a smaller dataset \\( (Z, u = f(Z)) \\) where \\( f \\) is the true latent function.

The likelihood with the inducing points are:

$$
\begin{aligned}
\mathbb{P}(y|X,Z,u,\theta)  & = \prod_{n=1}^N \mathbb{P}(y_n|x_n,Z,\theta)\\
                            & = \prod_{n=1}^N \mathcal{N}(y_n|K_{f_n u}K_{uu}^{-1}u, K_{f_n f_n}, K_{f_n f_n} - K_{f_n u} K_{uu}^{-1}K_{u f_n} + \sigma^2)\\
                            & = \mathcal{N}(y|K_{fu}K_{uu}^{-1}u, K_{ff}, \text{diag}(K_{ff} - Q_{ff}) + \sigma^2\mathbb{I})
\end{aligned}
$$

where, \\( Q_{ff} = K_{fu} K_{uu}^{-1}K_{uf}\\)

By placing the prior for \\(u\\) as \\( \mathbb{P}(u\|Z,\theta) = \mathcal{N}(u\|0,K_{u u})\\) as it is the output of the true latent function \\(f\\), we can simplify the likelihood as:

$$
\begin{aligned}
\mathbb{P}(y|X,Z) & = \int \mathbb{P}(y|X,Z,u,\theta) \mathbb{P}(u|Z,\theta) du\\
                  & = \mathcal{N}(y|0,Q_{ff} + \text{diag}(K_{ff}-Q_{ff}) + \sigma^2\mathbb{I})
\end{aligned}
$$

The inducing inputs \\(Z\\) is obtained via maximing the likelihood function \\( \mathbb{P}(y\|X,Z) \\).

To obtain the prediction at new data points \\(X_\star,y_\star\\), we start first with the joint distribution \\(\mathbb{P}(y,y_\star\|X,X_\star,Z)\\) and then calculate the conditioned probablity \\( \mathbb{P}(y_\star\|y,X,X_\star,Z)\\). This leads to:

$$
\begin{aligned}
\mathbb{P}(y_\star|y,X,X_\star,Z) & = \mathcal{N}(y_\star|\mu_\star,\Sigma_\star)~~\text{where}\\
                        \mu_\star & = Q_{\star f}\left(Q_{ff} + \text{diag}(K_{ff}-Q_{ff}) + \sigma^2\mathbb{I}\right)^{-1}y\\
                     \Sigma_\star & = K_{\star\star} - Q_{\star f}\left(K_{ff}-Q_{ff}) + \sigma^2\mathbb{I}\right)^{-1} Q_{f\star}
\end{aligned}
$$


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

## Deep Gaussian Processes

Deep Gaussian Processes assume that data is generated via a composition of multiple Gaussian Processes as:

$$
y = f_L( f_{L-1}( \cdots ( f_1(X) )\cdots ) ) + \epsilon
$$

where, \\(f_l\\) is drawn from a Gaussian Process.

We define new notations for later calculations: \\( h_l =  f_l( f_{l-1}( \cdots ( f_1(X) )\cdots ) ) + \epsilon_l\\) and \\( h_0 = X \\)

The joint probability between intermediate value \\(\\{h_l\\}_{l}\\) and the outputs \\(y\\) is:

$$
\begin{aligned}
\mathbb{P}(y,h_1, ..., h_L|X) & = \mathbb{P}(y|h_L) \prod_{l=1}^L \mathbb{P}(h_l | h_{l-1})\\
            \mathbb{P}(y|f_L) & = \mathcal{N}(f_L,\sigma_n^2 \mathbb{I})\\
      \mathbb{P}(h_l|h_{l-1}) & = \mathcal{N}(0,K_{h_l h_l} + \sigma_l^2 \mathbb{I})
\end{aligned}
$$

The joint probability is difficult to estimate. To circumvent this problem, we will use the Gaussian approximation techniques. We will focus on the "nested variational approach" (Hensman and Lawrence, 2014).

In the nested variational approach, to avoid the computational cost of large dataset, following the sparse Gaussian Process approach, a set of inducing points \\( \\{Z_l, u_l = f_l(Z_l)\\}_{l} \\) is introduced for layer \\(l\\). 

For convenience, we drop the \\(X, Z\\) in the conditioned probablity expression e.g. \\( \mathbb{P}(y|u, X, Z)\\) is reduced to \\( \mathbb{P}(y|u)\\).

$$
Q(u_l) = \mathcal{N}(u_l|m_l, S_l)
$$

$$
\begin{aligned}
\log \mathbb{P}(h_1|u_1) & = \log \int \mathbb{P}(h_1|f_1) \mathbb{P}(h_1|u_1) df_1\\
                         & \overset{Jensen}{\geq} \int \mathbb{P}(h_1|u_1) \log \mathbb{P}(h_1|f_1) df_1\\
                         & \geq \log \mathcal{N}(h1|K_{h_1 u_1}K_{u_1 u_1}^{-1} m_1, \sigma_1^2 \mathbb{I}) - \text{tr}\left(K_{h_1 u_1} K_{u_1 u_1}^{-1} S_1 K_{u_1 u_1}^{-1} K_{u_1 h_1} \right)\\
                         & \overset{\Delta}{=} \log \tilde{\mathbb{P}}(h_1|u_1)
\end{aligned}
$$

$$
\begin{aligned}
\log \mathbb{P}(h_2|u_2) \geq & \log \mathcal{N}(h_2|\Psi_2 K_{u_2 u_2}^{-1} m_2, \sigma_2^2\mathbb{I}) - \text{KL}(Q(u_1)||\mathbb{P(u_1)})\\
                              & -\frac{1}{2\sigma_1^2} \text{tr}\left(K_{11}-Q_{11}\right) - \frac{1}{2\sigma_2^2}\left(\psi_2 - \text{tr}(\Psi_2 K_{u_2 u_2}^{-1})\right)\\
                              & -\frac{1}{2\sigma_2^2} \text{tr}\left((\Phi_2-\Psi_2^T \Psi_2) K_{u_2 u_2}^{-1} (m_2 m_2^T+S_2) K_{u_2 u_2}^{-1}\right)
\end{aligned}
$$

$$
\begin{aligned}
\psi_l & = \mathbb{E}_{q_{l-1}} \left[ \text{tr}(K_{h_l h_l}) \right]\\
\Psi_l & = \mathbb{E}_{q_{l-1}} \left[ K_{h_l u_l} \right]\\
\Phi_l & = \mathbb{E}_{q_{l-1}} \left[ K_{u_l h_l} K_{h_l u_l} \right]
\end{aligned}
$$

$$
\begin{aligned}
\log \mathbb{P}(y|X) & \geq \log \mathcal{N}(y|\Psi_L K_{u_L u_L}^{-1} m_L, \sigma_n^2\mathbb{I}) - \sum_{l=1}^L \text{KL}(Q(u_l)||\mathbb{P(u_l)})\\
                     & -\frac{1}{2\sigma_1^2} \text{tr}\left(K_{11}-Q_{11}\right) - \sum_{l=2}^L \frac{1}{2\sigma_l^2}\left(\psi_l - \text{tr}(\Psi_l K_{u_l u_l}^{-1})\right)\\
                     & -\sum_{l=2}^L \frac{1}{2\sigma_l^2} \text{tr}\left((\Phi_l-\Psi_l^T \Psi_l) K_{u_l u_l}^{-1} (m_l m_l^T+S_l) K_{u_l u_l}^{-1}\right)
\end{aligned}
$$

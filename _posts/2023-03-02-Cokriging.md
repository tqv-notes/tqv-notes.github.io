---
title:  "Cokriging"
mathjax: true
layout: post
categories: media
---

For conventional kriging (also known as gaussian process regression), only one single output channel is considered in the modeling process.

In this note, we introduce the cokriging framework to be able to jointly model two different output channels. This technique is known as 'multi-output gaussian process regression' in machine learning community. Both 'kriging' and 'cokriging' terminology are from geostatistical community.

Consider a situation where we have two kinds of data:
- Primary data: \\( \\{ z_1(u_{\alpha_1}) \\}\_{\alpha_1 = 1 ... n_1} \\)
- Secondary data: \\( \\{ z_2(u_{\alpha_2}) \\}\_{\alpha_2 = 1 ... n_2} \\)

For practical reason, we only describe here the ordinary cokriging. Assuming \\( z_1 \\) and \\( z_2 \\) are realization of stationary random processes \\( Z_1 \\) with mean value \\( m_1 \\) and \\( Z_2 \\) with mean value \\( m_2 \\) respectively. The prediction of cokriging at the point \\( u \\) is given by:

$$
Z^\ast_1(u) = \sum_{\alpha_1 = 1}^{n_1} \lambda_{\alpha_1}^{(1)}(u) Z_1(u_{\alpha_1}) + \sum_{\alpha_2 = 1}^{n_2} \lambda_{\alpha_2}^{(2)}(u) Z_2(u_{\alpha_2}) + \lambda_{m_1}(u) m_1 + \lambda_{m_2}(u) m_2
$$

Since, \\( \mathbb{E}[Z_1(u_{\alpha_1})] = \mathbb{E}[Z^\ast_1(u)] = m_1 \\) and \\( \mathbb{E}[Z_2(u_{\alpha_2})] = m_2 \\), then we have:

$$
\begin{align*}
\lambda_{m_1} &= 1 - \sum_{\alpha_1=1}^{n_1} \lambda_{\alpha_1}^{(1)}(u) \\
\lambda_{m_2} &= - \sum_{\alpha_2=1}^{n_2} \lambda_{\alpha_2}^{(2)}(u)
\end{align*}
$$

To remove the need for mean estimation, we set \\( \lambda_{m_1} = 0 \\) and \\( \lambda_{m_2} = 0 \\), this leads to:

$$
  \begin{cases}
    Z^\ast_1(u) & = \sum_{\alpha_1 = 1}^{n_1} \lambda_{\alpha_1}^{(1)}(u) Z_1(u_{\alpha_1}) + \sum_{\alpha_2 = 1}^{n_2} \lambda_{\alpha_2}^{(2)}(u) Z_2(u_{\alpha_2})\\
    \sum_{\alpha_1=1}^{n_1} \lambda_{\alpha_1}^{(1)}(u) & = 1 \\
    \sum_{\alpha_2=1}^{n_2} \lambda_{\alpha_2}^{(2)}(u) & = 0
  \end{cases}
$$

The criteria to find the best estimator is to minimize the following variance:

$$
\begin{align*}
\sigma_E^2(u) & = \text{var}\left[ Z^\ast_1(u) - Z_1(u)\right] \\
              & = \text{var}\left[ \sum_{\alpha_1 = 1}^{n_1} \lambda_{\alpha_1}^{(1)}(u) Z_1(u_{\alpha_1}) + \sum_{\alpha_2 = 1}^{n_2} \lambda_{\alpha_2}^{(2)}(u) Z_2(u_{\alpha_2}) - Z_1(u)\right] \\
              & = \sum_{\alpha_1,\alpha_1'} \lambda_{\alpha_1}^{(1)} \lambda_{\alpha_1'}^{(1)} \text{cov}\left[ Z_1(u_{\alpha_1}), Z_1(u_{\alpha_1'})\right] + \sum_{\alpha_2,\alpha_2'} \lambda_{\alpha_2}^{(2)} \lambda_{\alpha_2'}^{(2)} \text{cov}\left[ Z_2(u_{\alpha_2}), Z_2(u_{\alpha_2'})\right] + \sum_{\alpha_1,\alpha_2} \lambda_{\alpha_1}^{(1)} \lambda_{\alpha_2}^{(2)} \text{cov}\left[ Z_1(u_{\alpha_1}), Z_2(u_{\alpha_2})\right] \\
              & ~ + \text{var}\left[ Z_1(u) \right] - \sum_{\alpha_1 = 1}^{n_1} \lambda_{\alpha_1}^{(1)}(u) \text{var}\left[ Z_1(u_{\alpha_1}), Z_1(u) \right] - \sum_{\alpha_2 = 1}^{n_2} \lambda_{\alpha_2}^{(2)}(u) \text{var}\left[ Z_2(u_{\alpha_1}), Z_1(u) \right]
\end{align*}
$$

The Lagrangian for this problem is:

$$
\mathcal{L}(u) = \sigma_E^2(u) + 2\mu_1 \left( \sum_{\alpha_1=1}^{n_1} \lambda_{\alpha_1}^{(1)}(u) - 1 \right) + 2\mu_2 \left( \sum_{\alpha_2=1}^{n_2} \lambda_{\alpha_2}^{(2)}(u) \right)
$$

The best estimator should correspond to following system of equations:

$$
\begin{cases}
\frac{1}{2} \frac{\partial \mathcal{L}}{\partial \alpha_i} & = 0 ~~~ \text{for}~ i = 1, 2 \\
\frac{1}{2} \frac{\partial \mathcal{L}}{\partial \mu_i} & = 0 ~~~ \text{for}~ i = 1, 2
\end{cases}
$$

Since \\( Z_1 \\) and \\( Z_2 \\) are stationary processes, we could define the covariance and cross-covariance as follows:

$$
\begin{cases}
\text{cov}\left[ Z_1(u_{\alpha_1}), Z_1(u_{\beta_1}) \right] & = C_{11} (u_{\alpha_1} - u_{\beta_1}) \\
\text{cov}\left[ Z_2(u_{\alpha_2}), Z_2(u_{\beta_2}) \right] & = C_{22} (u_{\alpha_2} - u_{\beta_2}) \\
\text{cov}\left[ Z_1(u_{\alpha_1}), Z_2(u_{\beta_2}) \right] & = C_{12}(u_{\alpha_1} - u_{\beta_2}) \\
\text{cov}\left[ Z_2(u_{\alpha_2}), Z_1(u_{\beta_1}) \right] & = C_{21}(u_{\alpha_2} - u_{\beta_1})
\end{cases}
$$

The system of equations is rewrote in term of covariances and cross-covariances as follows:

$$
  \begin{cases}
    \sum_{\beta_1 = 1}^{n_1} \lambda_{\beta_1}^{(1)}(u) C_{11}(u_{\alpha_1} - u_{\beta_1}) + \sum_{\beta_2 = 1}^{n_2} \lambda_{\beta_2}^{(2)}(u) C_{12}(u_{\alpha_1} - u_{\beta_2}) + \mu_1(u) & = C_{11}(u_{\alpha_1}) - u ~~~ \text{for}~ \alpha_1 = 1 ... n_1 \\
    \sum_{\beta_1 = 1}^{n_1} \lambda_{\beta_1}^{(1)}(u) C_{21}(u_{\alpha_2} - u_{\beta_1}) + \sum_{\beta_2 = 1}^{n_2} \lambda_{\beta_2}^{(2)}(u) C_{22}(u_{\alpha_2} - u_{\beta_2}) + \mu_2(u) & = C_{21}(u_{\alpha_2}) - u ~~~ \text{for}~ \alpha_2 = 1 ... n_2 \\    
    \sum_{\alpha_1=1}^{n_1} \lambda_{\alpha_1}^{(1)}(u) & = 1 \\
    \sum_{\alpha_2=1}^{n_2} \lambda_{\alpha_2}^{(2)}(u) & = 0
  \end{cases}
$$

The variance of estimator at the optimal solution is:

$$
\begin{align*}
\sigma^2(u) & = C_{11}(0) - \mu_1(u) - \sum_{\alpha_1=1}^{n_1} \lambda_{\alpha_1}^{(1)}(u) C_{11}(u_{\alpha_1} - u) - \sum_{\alpha_2=1}^{n_2} \lambda_{\alpha_1}^{(2)}(u) C_{21}(u_{\alpha_2} - u) \\
& = - \mu_1(u) + \sum_{\alpha_1=1}^{n_1} \lambda_{\alpha_1}^{(1)}(u) (C_{11}(0)-C_{11}(u_{\alpha_1} - u)) + \sum_{\alpha_2=1}^{n_2} \lambda_{\alpha_1}^{(2)}(u) (C_{21}(0)-C_{21}(u_{\alpha_2} - u)) \\
& = - \mu_1(u) + \sum_{\alpha_1=1}^{n_1} \lambda_{\alpha_1}^{(1)}(u) \gamma_{11}(u_{\alpha_1} - u) + \sum_{\alpha_2=1}^{n_2} \lambda_{\alpha_1}^{(2)}(u) \gamma_{21}(u_{\alpha_2} - u)
\end{align*}
$$

here, we introduce the notion of (cross-) variogram \\( \gamma_{ij}(h) = C_{ij}(0) - C_{ij}(h)\\).

In matrix notation, the optimal estimator can be found via solving the following equation:

$$
\begin{pmatrix}
  \mathbf{K}_{1,1} & \mathbf{K}_{1,2} & \mathbf{1}_1 & \mathbf{0}_1 \\
  \mathbf{K}_{2,1} & \mathbf{K}_{2,2} & \mathbf{0}_2 & \mathbf{1}_1 \\
  \mathbf{1}_1^T   & \mathbf{0}_2^T   & 0   & 0  \\
  \mathbf{0}_1^T   & \mathbf{1}_2^T   & 0   & 0 
 \end{pmatrix}
 \begin{pmatrix}
  \boldsymbol{\lambda}_1(u) \\
  \boldsymbol{\lambda}_2(u) \\
  \mu_1(u)  \\
  \mu_2(u)
 \end{pmatrix}
 =\begin{pmatrix}
  \mathbf{k}_{11} \\
  \mathbf{k}_{21} \\
  1 \\
  0
 \end{pmatrix}
$$

where,

$$
\begin{align*}
\mathbf{K}_{11} & = \left[ C_{11}(u_{\alpha_1}-u_{\beta_1}) \right]_{\alpha_1 = 1 ... n_1, \beta_1 = 1 ... n_1} \\
\mathbf{K}_{22} & = \left[ C_{22}(u_{\alpha_2}-u_{\beta_2}) \right]_{\alpha_2 = 1 ... n_2, \beta_1 = 1 ... n_2} \\
\mathbf{K}_{21} & = \left[ C_{21}(u_{\alpha_2}-u_{\beta_1}) \right]_{\alpha_2 = 1 ... n_2, \beta_1 = 1 ... n_1} \\
\mathbf{K}_{12} & = \left[ C_{12}(u_{\alpha_1}-u_{\beta_2}) \right]_{\alpha_1 = 1 ... n_1, \beta_2 = 1 ... n_2} \\
\boldsymbol{\lambda}_1(u) & = 
\begin{pmatrix}
  \lambda_1^{(1)}(u) \\
  \lambda_2^{(1)}(u) \\
  \vdots \\
  \lambda_{n_1}^{(1)}(u)
 \end{pmatrix} \\
 \boldsymbol{\lambda}_2(u) & = 
\begin{pmatrix}
  \lambda_1^{(2)}(u) \\
  \lambda_2^{(2)}(u) \\
  \vdots \\
  \lambda_{n_2}^{(2)}(u)
 \end{pmatrix}\\
\mathbf{k}_{11} & = 
\begin{pmatrix}
  C_{11}(u_1-u) \\
  C_{11}(u_2-u) \\
  \vdots \\
  C_{11}(u_{n_1}-u)
\end{pmatrix}\\
\mathbf{k}_{21} & = 
\begin{pmatrix}
  C_{21}(u_1-u) \\
  C_{21}(u_2-u) \\
  \vdots \\
  C_{21}(u_{n_2}-u)
 \end{pmatrix}
\end{align*}
$$

The prediction of cokriging at new location \\( u \\) is given by:

$$
\begin{align*}
Z_1^\ast(u) & = \sum_{\alpha_1 = 1}^{n_1} \lambda_{\alpha_1}^{(1)}(u) Z_1(u_{\alpha_1}) + \sum_{\alpha_2 = 1}^{n_2} \lambda_{\alpha_2}^{(2)}(u) Z_2(u_{\alpha_2})\\
            & =  
\begin{pmatrix}
  \mathbf{Z}_{1}(u) & \mathbf{Z}_{2}(u) & 0 & 0
 \end{pmatrix}            
 \begin{pmatrix}
  \lambda_1^{(1)}(u) \\
  \lambda_2^{(1)}(u) \\
  \vdots \\
  \lambda_{n_1}^{(1)}(u)
 \end{pmatrix}\\
            & =  
\begin{pmatrix}
  \mathbf{Z}_{1}(u) & \mathbf{Z}_{2}(u) & 0 & 0
 \end{pmatrix}            
\begin{pmatrix}
  \mathbf{K}_{1,1} & \mathbf{K}_{1,2} & \mathbf{1}_1 & \mathbf{0}_1 \\
  \mathbf{K}_{2,1} & \mathbf{K}_{2,2} & \mathbf{0}_2 & \mathbf{1}_1 \\
  \mathbf{1}_1^T   & \mathbf{0}_2^T   & 0   & 0  \\
  \mathbf{0}_1^T   & \mathbf{1}_2^T   & 0   & 0 
 \end{pmatrix}^{-1}
\begin{pmatrix}
  \mathbf{k}_{11} \\
  \mathbf{k}_{21} \\
  1 \\
  0
 \end{pmatrix} 
\end{align*}
$$

Theoretical variogram models:

The theoretical variogram \\( 2\gamma (h) \\) of an 'intrinsic' stationary process \\( Z(u) \\) is defined as:

$$
2\gamma (h) = \text{var}[Z(u+h)-Z(u)]
$$

Since the covarinace is given by \\( C(h) = \mathbb{E}[Z(u)Z(u+h)] - \mu^2\\) with \\(\mu = \mathbb{E}[Z(u)]\\), then we have:

$$
2\gamma (h) = C(0) - C(h)
$$

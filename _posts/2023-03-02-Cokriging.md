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

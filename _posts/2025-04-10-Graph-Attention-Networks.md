---
title:  "Graph Attention Networks"
mathjax: true
layout: post
categories: media
---

In this post, we will explain the attention-based architecture for graphs (named as "graph attention networks"). This note is based on [Veličković et al., 2024](https://arxiv.org/abs/1710.10903).

$$

\alpha_{ij} = \frac{ \exp\left(LeakyReLU\left(a^T [W h_i || W h_j] \right)\right) }{ \sum_{k\in \mathcal{N}_i} \exp\left(LeakyReLU\left(a^T [W h_i || W h_k] \right)\right) }

$$

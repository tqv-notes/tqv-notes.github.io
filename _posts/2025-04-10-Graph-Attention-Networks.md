---
title:  "Graph Attention Networks"
mathjax: true
layout: post
categories: media
---

In this post, we will explain the attention-based architecture for graphs (named as "graph attention networks"). This note is based on [Veličković et al., 2024](https://arxiv.org/abs/1710.10903).

Attention mechanism (as discussed in [Attention Mechanisms and Transformers](https://tqv-notes.github.io/Attention-Mechanisms-and-Transformers//) was orginally invented for natural language processing tasks where we can see input/output as a sequence of data. Based on this pioneering work, the graph attention networks (GAT) was proposed to efficiently deal with graph-structured data.

$$

\alpha_{ij} = \frac{ \exp\left(\text{LeakyReLU}\left(a^T \begin{bmatrix}
           W h_i \\
           W h_j
         \end{bmatrix} \right)\right) }{ \sum_{k\in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(a^T \begin{bmatrix}
           W h_i \\
           W h_j
         \end{bmatrix} \right)\right) }

$$

---
title:  "Graph Attention Networks"
mathjax: true
layout: post
categories: media
---

In this post, we will explain the attention-based architecture for graphs (named as "graph attention networks"). This note is based on [Veličković et al., 2024](https://arxiv.org/abs/1710.10903).

Attention mechanism (as discussed in [Attention Mechanisms and Transformers](https://tqv-notes.github.io/Attention-Mechanisms-and-Transformers//)). was orginally invented for natural language processing tasks where we can see input/output as a sequence of data. Based on this pioneering work, the graph attention networks (GAT) was proposed to efficiently deal with graph-structured data.

We first describe the single graph attention layer and then multi-head attention layer.

In the single graph attention layer, the input is a set of node features \\( \textbf{h} = \\{h_1, h_2, \dots, h_N\\}\\) and the output is a set of transformed node features \\( \textbf{h}^\prime = \\{h_1^\prime, h_2^\prime, \dots, h_N^\prime\\}\\) where \\( h_i \in \mathbb{R}^F \\), \\(h_i^\prime \in \mathbb{R}^{F^\prime}\\), and \\( N, F, F^\prime\\) are number of nodes and number of features of input and output respectively.

The purpose of (self-) attention layer is to learn the relationship between nodes (how important the node \\(j\\) to the node \\(i\\)). To do this, we first linearly transform \\(\textbf{h}\\) via a weight matrix \\(\textbf{W}\\) and then preform dot-product with the attention weight vector \\(\textbf{a} \in \mathbb{R}^{2F^\prime}\\)) before applying an activation function (here, the \\( \text{LeakyReLU} \\) function is used). The output of these steps is then normalized via a softmax function. Put everything together, we have the formula to calculate the attention coeffficients as:

$$

\alpha_{ij} = \frac{ \exp\left(\text{LeakyReLU}\left(\textbf{a}^T \begin{bmatrix}
           \textbf{W} h_i \\
           \textbf{W} h_j
         \end{bmatrix} \right)\right) }{ \sum_{k\in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(a^T \begin{bmatrix}
           \textbf{W} h_i \\
           \textbf{W} h_j
         \end{bmatrix} \right)\right) }

$$

The normalized attention coefficients \\( \\{\alpha_{ij}\\}\\) are then used to calculate the final output as:

$$
h_i^\prime = \sigma\left( \sum_{j \in \mathcal{N}_i} \alpha_{ij} \textbf{W} h_j \right)
$$

$$
h_i^\prime = 
\begin{bmatrix}
\sigma\left( \sum_{j \in \mathcal{N}_i} \alpha^{(1)}_{ij} \textbf{W}^{(1)} h_j \right) \\
\sigma\left( \sum_{j \in \mathcal{N}_i} \alpha^{(2)}_{ij} \textbf{W}^{(2)} h_j \right) \\
\vdots\\
\sigma\left( \sum_{j \in \mathcal{N}_i} \alpha^{(K)}_{ij} \textbf{W}^{(K)} h_j \right)
\end{bmatrix}
$$

$$
h_i^\prime = \sigma\left( \frac{1}{K} \sum_{k=1}^K \sum_{j \in \mathcal{N}_i} \alpha^{(k)}_{ij} \textbf{W}^{(k)} h_j \right)
$$

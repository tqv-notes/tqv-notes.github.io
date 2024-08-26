---
title:  "Attention Mechanisms and Transformers"
mathjax: true
layout: post
categories: media
---

The goal of transformers is to transform an input \\( X^{(0)} \in \mathbb{R}^{D \times N} \\) into an output \\( X^{(M)} \in \mathbb{R}^{D \times N}\\) (here, \\(N\\) is number of tokens, \\(D\\) is number of features and \\(M\\) is number of transformer layers).

We will cover first the key aspect of transformer architecture: the attention mechanisms. This note is based on [Turner, 2024](https://arxiv.org/abs/2304.10557).

## Attention Mechanisms

The general idea on attention is simply a linear transformation of input as:

$$
Y^{(m)} = X^{(m-1)} A^{(m)}
$$

where, the attention matrix is normalized over its column i.e. \\( \sum_{n=1}^N A_{n n'}^{(m)} = 1\\).

### Self-Attention Mechanism

In the self-attention mechanism, the attention matrix is defined via its inputs as:

$$
A^{(m)}_{n n'} = \frac{ \exp \left( x^{(m-1)}_n U_k^{(m)} U^{(m)}_q x^{(m-1)}_{n'} / \sqrt{D} \right) }{ 1 }
$$

### Multi-Head Self-Attention Mechanism

## Transformers

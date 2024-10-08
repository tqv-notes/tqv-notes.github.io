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
A^{(m)}_{n n'} = \frac{ \exp \left( \frac{1}{\sqrt{D}} \left( x^{(m-1)}_n \right)^T \left( U_k^{(m)} \right)^T U^{(m)}_q x^{(m-1)}_{n'} \right) }{ \sum_{n''=1}^N \exp \left( \frac{1}{\sqrt{D}} \left( x^{(m-1)}_{n''} \right)^T \left( U_k^{(m)} \right)^T U^{(m)}_q x^{(m-1)}_{n'} \right) }
$$

### Multi-Head Self-Attention Mechanism

Like CNN with multiple filters, to increase the capacity of self-attention mechanism, the transformer block has \\(H\\) self-attentions in parallel:

$$
Y^{(m)} = \text{MHSA}(X^{(m-1)}) = \sum_{h=1}^H V_h^{(m)} X^{(m-1)} A^{(m)}_h
$$

where,

$$
\left[A^{(m)}_{h}\right]_{n n'} = \frac{ \exp \left( \frac{1}{\sqrt{D}} \left( x^{(m-1)}_n \right)^T \left( U_{k h}^{(m)} \right)^T U^{(m)}_{q h} x^{(m-1)}_{n'} \right) }{ \sum_{n''=1}^N \exp \left( \frac{1}{\sqrt{D}} \left( x^{(m-1)}_{n''} \right)^T \left( U_{k h}^{(m)} \right)^T U^{(m)}_{q h} x^{(m-1)}_{n'} \right) }
$$

## Transformers

The output of multi-head self-attention block will pass through a multi-layer perceptron:

$$
X^{(m)} = \text{MLP}(Y^{m}) = \text{MLP}( \text{MHSA}(X^{m-1}) )
$$

This completes the core component of the transformer block.

To help the training of the transformer block, two extra components are needed: the residual connections and the layer normalization.

Put everything together, we have the diagram of transformer block as follows:

![transformer_block](/images/transformer_block.png){:height="100%" width="100%"}

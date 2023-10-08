---
title:  "Transformer Architecture"
mathjax: true
layout: post
categories: media
---
In this note, we will present some basic ideas about the transformer architecture as first presented in Vaswani *et al*. [Attention is All you Need
](https://browse.arxiv.org/pdf/1706.03762.pdf/)

## Recurrent Neural Networks

Recurrent Neural Networks (RNN) is a type of neural networks that deals with sequential or time series datasets. RNN is uniquely characterized by the usage of hidden states and looping structure.

Below is the description of a simple RNN architecture

![RNN](/images/rnn.png){:height="10%" width="10%"}

$$
\begin{aligned}
a_t &= W_1 \begin{bmatrix} x_t\\ h_{t-1} \end{bmatrix} + b_1\\
h_t &= \sigma_h\left(a_t\right)\\
y_t &= \sigma_y\left(W_2 h_t + b_2\right)
\end{aligned}
$$

## Transformer Architecture


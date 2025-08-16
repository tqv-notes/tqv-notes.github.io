---
title:  "Attention Mechanisms and Transformers"
mathjax: true
layout: post
categories: media
---

The goal of transformers is to transform an input \\( X^{(0)} \in \mathbb{R}^{D \times N} \\) into an output \\( X^{(M)} \in \mathbb{R}^{D \times N}\\) (here, \\(N\\) is number of tokens, \\(D\\) is number of features and \\(M\\) is number of transformer layers).

We will cover first the key aspect of transformer architecture: the attention mechanisms. This note is based on [Turner, 2024](https://arxiv.org/abs/2304.10557).

## Attention Mechanisms

The attention mechanism is a method that given a databse \\(D = {(k_i, v_i)}_{i=1...n}\\) of keys \\(k\\) (e.g. measurement coordinates) and values \\(v\\) (e.g. measurement reponses), predict the reponses from queries \\(q\\) (e.g. new measurement coordinates) as:

$$
attention(q,D) = \sum_{i=1}^n a(q,k_i) v_i
$$

where, \\(a(q,k_i)\\) are the attention weights which satisfy \\(a(q,k_i) \geq 0\\) and \\( \sum_{i=1}^n a(q,k_i) = 1\\). To enforce the last two requirement, the attention weight is defined via a softmax function as:

$$
a(q,k_i) = \frac{ \exp(f(q,k_i)) }{ \sum_{j=1}^n \exp(f(q,k_j)) }
$$

with \\(f(q,k_i)\\) is the function that estimate the "similarity" or "closeness" betwen query and key. In deep learning, the choice of function \\(f\\) is usually a scaled dot product function i.e. \\(f(q,k_i) = q^T k_i / \sqrt{d}\\) with \\(d\\) is the dimension of key and query.

In a more abstract picture, if we denote all values as a matrix input \\(X\\), the attention weights as a matrix \\(A\\) and attention output as a matrix \\(Y\\), then the attention mechanism can be seen simply as a linear transformation of input as:

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

## Simple example of attention mechanism

In this example, we use attention mechanism to count digits without explicitly using any counting function: we will provide an array of 10 digits with values randomly selected from 0 to 9 and a label true if number of digit 4 is strickly larger than number of digit 2 and false otherwise. The model needs to learn from data how to recognize this pattern (number of digit 4 vs number of digit 2) and how to correctly label it.

```python
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 22:38:54 2025

@author: qtran
"""
import torch
import matplotlib.pyplot as plt
import numpy as np

def generate_data(N):
    X = torch.randint(0,9,size = (N,10))
    num2s = torch.count_nonzero(X==2, dim=-1)
    num4s = torch.count_nonzero(X==4, dim=-1)
    labels = num4s > num2s
    return X, labels.reshape(-1,1).float()

X, y = generate_data(123)

class AttentionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.query = torch.nn.Parameter(torch.randn(1,32))
        self.embed_func = torch.nn.Embedding(10, embedding_dim=16)
        self.key_func = torch.nn.Linear(16, 32)
        
        self.value_func = torch.nn.Sequential(
                        torch.nn.Linear(16,32),
                        torch.nn.ReLU(),
                        torch.nn.Linear(32,1)
                    )
        
        self.head_mlp = torch.nn.Sequential(
            torch.nn.Linear(1,32),
            torch.nn.ReLU(),
            torch.nn.Linear(32,1),
            torch.nn.Sigmoid()
        )

    def forward(self,X):
        embedX = self.embed_func(X) # [123, 10, 16]
        keys = self.key_func(embedX) # [123, 10, 32]
        qk = torch.einsum('ie, bje -> bij', self.query, keys) # [1,32] x [123, 10, 32] -> [123, 1, 10]     
        qk = qk / (32**0.5)
        att = torch.nn.functional.softmax(qk, dim=-1) # [123, 1, 10]
        values = self.value_func(embedX) # [123, 10, 1]
        summary = torch.einsum('bij, bje -> bie', att, values)[:,0,:] # [123, 1]
        pred = self.head_mlp(summary) # [123, 1]
        return pred, att, values

def train():
    model = AttentionModel()
    opt = torch.optim.Adam(model.parameters(), lr = 3e-4)
    losses = []
    for idx in range(5000):
        X, y = generate_data(123)
        p, a, v = model(X)
        loss = torch.nn.functional.binary_cross_entropy(p, y)
        losses.append(float(loss))
        if idx % 100 == 0:
            print(float(loss))
            plt.plot(losses)
            plt.gcf().set_size_inches(2,2)
            plt.show()
        loss.backward()
        opt.step()
        opt.zero_grad()
    return model

if __name__ == "__main__":
    model = train()

    with torch.no_grad():
        X = torch.LongTensor([[1,7,2,0,2,1,3,4,8,6]])
        p, a, v = model(X)

    plt.imshow(a[0], vmin=0, vmax=1)
    
    for x,y,d in zip(np.arange(10),np.zeros(10),X[0]):
        plt.text(x,y,int(d), c = 'r' if d in [4,2] else 'w')
    plt.gcf().set_size_inches(10,1)
    plt.show()
    
    message = v[:,:,0]
    message = np.where(a[0] > 0.1, v[:,:,0], np.nan*v[:,:,0])
    plt.imshow(message)
    
    for x,y,d in zip(np.arange(10),np.zeros(10),message[0]):
        plt.text(x-0.5,y,f'{d:2f}', c = 'w')
    plt.gcf().set_size_inches(10,1)
    plt.show()
```

---
title:  "Approximate Message Passing"
mathjax: true
layout: post
categories: media
---

## Introduction

The problem of our interest is the standard linear regression:

$$ y = H x +n $$

where $y\in \mathbb{R}^M$ is the observation, $x\in \mathbb{R}^N$ is the sparse signal, $H\in \mathbb{R}^{M\times N} (M \ll N)$ is the known design matrix and $n \in \mathbb{R}^N$ is the additive white Gaussian noise with zero mean and covariance $\sigma^2$.

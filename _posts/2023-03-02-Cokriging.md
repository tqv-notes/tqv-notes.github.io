---
title:  "Cokriging"
mathjax: true
layout: post
categories: media
---

For conventional kriging (also known as gaussian process regression), only one single output channel is considered in the modeling process.

In this note, we introduce the cokriging framework to be able to jointly model two different output channels. This technique is known as 'multi-output gaussian process regression' in machine learning community. Both 'kriging' and 'cokriging' terminology are from geostatistical community.

Consider a situation where we have two kinds of data:
- Primary data: \\( \{ z_1(u_{\alpha_1})\}\_{\alpha_1 = 1 ... n_1} \\)
- Secondary data: \\( \{ z_2(u_{\alpha_2})\}\_{\alpha_2 = 1 ... n_2} \\)


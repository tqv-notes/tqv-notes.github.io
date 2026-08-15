---
title:  "Reasoning with GRPO"
mathjax: true
layout: post
categories: media
---

Training Large Language Models (LLM) is a tricky business: using huge amount of data, it can be very good at text generations (for example, predicting next words in a sentence) but to be able to provide useful answers that are well aligned with human preference, pretraining alone is not good enough. Reinforcement Learning (RL) provides a way to fine-tune the pretrained models to better align with our objectives. A popular RL technique is the Reinforcement Learning with Human Feedback (RLHF). The idea is to collect human preference data from outputs of a pretrained model to train a reward model and then to use this reward model to further optimize the pretrained model. Of course, there is a wide range of RLHF techniques such as Proximal Policy Optimization (PPO) or Direct Preference Optimization (DPO) but in this note, we focus on the Group Relative Policy Optimization (GRPO) technique from [DeepSeek R1](https://arxiv.org/pdf/2501.12948) paper.

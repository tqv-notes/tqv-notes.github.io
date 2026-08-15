---
title:  "Reasoning with GRPO"
mathjax: true
layout: post
categories: media
---

Training Large Language Models (LLM) is a tricky business: using huge amount of data, it can be very good at text generations (for example, predicting next words in a sentence) but to be able to provide useful answers that are well aligned with human preference, pretraining alone is not good enough. Reinforcement Learning (RL) provides a way to fine-tune the pretrained models to better align with our objectives. A popular RL technique is the Reinforcement Learning with Human Feedback (RLHF). The idea is to collect human preference data from outputs of a pretrained model to train a reward model and then to use this reward model to further optimize the pretrained model. Of course, there is a wide range of RLHF techniques such as Proximal Policy Optimization (PPO) or Direct Preference Optimization (DPO) but in this note, we focus on the Group Relative Policy Optimization (GRPO) technique from [DeepSeek R1](https://arxiv.org/pdf/2501.12948) paper.

The training process of DeepSeek R1 consists of 4 steps: 
- Cold Start: in this step, to establish a strong foundation for model readability and response's quality, a small high-quality data from R1-Zero to fine-tune the V3 model.
- Reasoning RL: using rule-based RL, this step focuses on enhancing the model's reasoning capabilities across domains including mathematics, coding, science and logic reasoning.
- Rejection Sampling: in this step, V3 model is used to filter out responses from the main model.
- Diverse RL: this secondary RL step aims to further align with human preferences using a hybrid reward approach (rule-based systems + language model evaluation).

At the heart of both reasoning and diverse RL steps is the GRPO algorithm. The main idea of this algorithm is that instead of training a separate value/critic model to estimate how good an answer is, generate several answers to the same question and judge each answer relative to the others.

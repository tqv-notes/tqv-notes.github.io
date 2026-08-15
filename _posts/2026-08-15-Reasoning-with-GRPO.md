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

At the heart of both reasoning and diverse RL steps is the GRPO algorithm. The main idea of this algorithm is that instead of training a separate value/critic model to estimate how good an answer is, generate several answers to the same question and judge each answer relative to the others. More precisely, for each question \\(q\\), GRPO samples a group of output \\( \\{ o_1, o_2, ..., o_G \\} \\) from the old policy \\( \pi_{\theta_{old}} \\) and then optimizes the policy model \\(\pi_{\theta}\\) by maximizing the following cost function:

$$
\begin{aligned}
J_{GRPO}(\theta) &= \mathcal{E}_{q \sim P(q), o \sim \pi_{\theta}(\cdot|q)} \left[ r(q, o)\right]\\
                 &= \left[ \frac{1}{G} \sum_{i=1}^G min\left( \frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip} \left( \frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{old}}(o_i|q)},1-\epsilon, 1+\epsilon \right) A_i \right) \right] - \beta \mathcal{D}_{KL}(\pi \| \pi_{ref})
\end{aligned}
$$

where \\( \mathcal{D}_{KL}(\pi \| \pi_{ref}) = \frac{\pi_{ref}(o_i\|q)}}{\pi_{\theta}(o_i\|q)}} - log\left( \frac{\pi_{ref}(o_i\|q)}}{\pi_{\theta}(o_i\|q)}}\right) - 1 \\)

## GRPO pseudocode

initialize policy model `pi_theta`

initialize reference model `pi_ref = pi_theta`

for each training step:

    # 1. sample a batch of prompts
    prompts = sample_prompts()

    for each prompt x in prompts:

        # 2. generate a group of G responses
        responses = [y1, y2, ..., yG] = generate_G_responses(pi_theta, x)

        # 3. evaluate each response
        rewards = [r1, r2, ..., rG] = reward_function(x, responses)

        # 4. compute relative advantages
        mean_r = mean(rewards)
        std_r  = std(rewards)

        for i = 1 ... G:
            Ai = (ri - mean_r) / (std_r + epsilon)

        # 5. compute GRPO policy loss
        for each response yi:

            ratio_i = pi_theta(yi | x) / pi_old(yi | x)

            clipped_ratio_i = clip(ratio_i,1 - epsilon_clip,1 + epsilon_clip)

            policy_loss_i =-min(ratio_i * Ai,clipped_ratio_i * Ai)

            # KL penalty against reference model
            KL_i = KL(pi_theta || pi_ref)

            total_loss_i = policy_loss_i + beta * KL_i

    # 6. update the policy parameters
    theta = theta - learning_rate * grad_theta(total_loss)

return pi_theta

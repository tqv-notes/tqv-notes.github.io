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
J_{GRPO}(\theta) &= \mathbb{E}_{q \sim P(q), o \sim \pi_{\theta}(\cdot|q)} \left[ r(q, o)\right]\\
                 &= \left[ \frac{1}{G} \sum_{i=1}^G min\left( \frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip} \left( \frac{\pi_{\theta}(o_i|q)}{\pi_{\theta_{old}}(o_i|q)},1-\epsilon, 1+\epsilon \right) A_i \right) \right] - \beta \mathbb{D}_{KL}(\pi \| \pi_{ref})
\end{aligned}
$$

where,

$$ A_i = \frac{ r_i - \text{mean}\left( \{ r_1, r_2, ..., r_G \} \right) }{ \text{std}\left( \{ r_1, r_2, ..., r_G \} \right) } $$

$$ \mathbb{D}_{KL}(\pi | \pi_{ref}) = \frac{\pi_{ref}(o_i|q)}{\pi_{\theta}(o_i|q)} - \log\left( \frac{\pi_{ref}(o_i|q)}{\pi_{\theta}(o_i|q)}\right) - 1 $$

### GRPO pseudocode

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

### Model fine-tuning with GRPO

First, we need to add necessary toolbox for this task:
```python
# see what CUDA this runtime actually has, before installing anything.
!nvidia-smi | head -4
!nvcc --version | tail -2

# install unsloth without vllm (no unsloth extra pulls it in anyway).
!pip install --upgrade pip
!pip install unsloth unsloth_zoo
!pip install "trl>=0.24.0" "datasets>=3.0.0"

# install the vLLM wheel matching the CUDA reported above.
VLLM_WHEEL = "https://github.com/vllm-project/vllm/releases/download/v0.23.0/vllm-0.23.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
!pip install {VLLM_WHEEL}

# install validation
import unsloth
import torch, transformers, trl, datasets, platform
import vllm

print(f"python       {platform.python_version()}")
print(f"torch        {torch.__version__}")
print(f"transformers {transformers.__version__}")
print(f"trl          {trl.__version__}")
print(f"datasets     {datasets.__version__}")
print(f"vllm         {vllm.__version__}")
print(f"gpu          {torch.cuda.get_device_name(0)}")
print(f"vram         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"bf16         {torch.cuda.is_bf16_supported()}")
print(f"torch cuda   {torch.version.cuda}  -  available {torch.cuda.is_available()}")
```

It should output this configuration (on google colab with L4 GPU):
```python
python       3.12.13
torch        2.11.0+cu128
transformers 5.15.0
trl          0.24.0
datasets     4.3.0
vllm         0.23.0
gpu          NVIDIA L4
vram         23.7 GB
bf16         True
torch cuda   12.8  -  available True
```

Define model configuration and load Gemma 3 model
```python
from dataclasses import dataclass

@dataclass
class Cfg:
    model_name: str = "unsloth/gemma-3-1b-it"   # unsloth mirror: pre-patched, no gated repo
    max_seq_length: int = 1024
    max_prompt_length: int = 256
    lora_rank: int = 32

    num_generations: int = 6      # G in the GRPO objective -- the group size
    per_device_batch: int = 6     # must make generation_batch divisible by G
    grad_accum: int = 4           # -> 6*4 = 24 completions = 4 prompts per update
    max_steps: int = 250
    seed: int = 3407

CFG = Cfg()
assert (CFG.per_device_batch * CFG.grad_accum) % CFG.num_generations == 0, \
    "generation_batch_size must be divisible by num_generations"
    
print(f"{CFG.per_device_batch * CFG.grad_accum // CFG.num_generations} unique prompts per optimizer step")

from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name             = CFG.model_name,
    max_seq_length         = CFG.max_seq_length,
    load_in_4bit           = True,     # QLoRA. False -> LoRA in bf16, still fits on L4
    fast_inference         = True,     # vLLM-backed generation; the whole ballgame for GRPO
    max_lora_rank          = CFG.lora_rank,
    gpu_memory_utilization = 0.6,    # vLLM's slice. Lower if OOM, raise for bigger groups
)

# GEMMA 3 + FastModel + fast_inference GOTCHA:
model = FastModel.get_peft_model(
    model,
    r = CFG.lora_rank,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    finetune_vision_layers     = False,   # required, see above
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    lora_alpha = CFG.lora_rank * 2,   # alpha = 2r is the more common modern default
    lora_dropout = 0.0,               # 0.0 is Unsloth's fast path
    use_gradient_checkpointing = "unsloth",
    random_state = CFG.seed,
)
```

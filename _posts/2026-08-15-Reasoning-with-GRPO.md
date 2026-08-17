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

```
initialize policy model pi_theta

initialize reference model pi_ref = pi_theta

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
```

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

Load and pre-process the Grade School Math 8K [gsm8k](https://huggingface.co/datasets/openai/gsm8k) dataset from OpenAI:
```python
import re
from datasets import load_dataset

REASONING_START, REASONING_END = "<reasoning>", "</reasoning>"
ANSWER_START,    ANSWER_END    = "<answer>",    "</answer>"

SYSTEM_PROMPT = (
    "You are given a problem. Think step by step, then give the final answer.\n"
    "Respond in exactly this format:\n"
    f"{REASONING_START}\n...\n{REASONING_END}\n{ANSWER_START}\n...\n{ANSWER_END}"
)

def extract_xml_answer(text: str) -> str:
    return text.split(ANSWER_START)[-1].split(ANSWER_END)[0].strip()

def extract_hash_answer(text: str) -> str | None:
    return text.split("####")[1].strip() if "####" in text else None

_NUM = re.compile(r"-?\d+(?:\.\d+)?")

def normalize_number(s: str) -> str | None:
    """
    GSM8K gold answers and model outputs disagree on cosmetics, not maths.
    '1,000' / '$1000' / '1000.0' / 'The answer is 1000.' must all compare equal.
    """
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    m = _NUM.findall(s)
    if not m:
        return None
    try:
        v = float(m[-1]) # last number = the stated answer
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else str(v)

def build_dataset(split: str = "train"):
    ds = load_dataset("openai/gsm8k", "main")[split]
    ds = ds.map(lambda x: {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user",   "content": x["question"]}],
        "answer": normalize_number(extract_hash_answer(x["answer"])),
    })
    # rows where parsing failed would silently reward-match against None.
    return ds.filter(lambda x: x["answer"] is not None)

train_dataset = build_dataset("train").shuffle(seed=CFG.seed)
eval_dataset  = build_dataset("test").shuffle(seed=CFG.seed).select(range(200))
print(train_dataset)
print(train_dataset[0]["prompt"][1]["content"][:200], "->", train_dataset[0]["answer"])
```

Define reward models:
```python
STRICT_RE = re.compile(rf"^{REASONING_START}\n.+?\n{REASONING_END}\n{ANSWER_START}\n.+?\n{ANSWER_END}\s*$", re.DOTALL)
SOFT_RE   = re.compile(rf"{REASONING_START}.+?{REASONING_END}\s*{ANSWER_START}.+?{ANSWER_END}", re.DOTALL)

def format_reward(completions, **kwargs) -> list[float]:
    """0.5 for the exact layout, 0.2 for a loose match, 0 otherwise."""
    out = []
    for c in completions:
        text = c[0]["content"]
        if STRICT_RE.match(text):
            out.append(0.5)
        elif SOFT_RE.search(text):
            out.append(0.2)
        else:
            out.append(0.0)
    return out

_step = {"n": 0}

def correctness_reward(prompts, completions, answer, **kwargs) -> list[float]:
    responses  = [c[0]["content"] for c in completions]
    predicted  = [normalize_number(extract_xml_answer(r)) for r in responses]
    rewards    = [2.0 if (p is not None and p == a) else 0.0 for p, a in zip(predicted, answer)]

    _step["n"] += 1
    if _step["n"] % 20 == 1:
        frac_same = sum(r == rewards[0] for r in rewards) / len(rewards)
        print(f"\n{'='*60}\nstep~{_step['n']}  Q: {prompts[0][-1]['content'][:90]}")
        print(f"gold={answer[0]}  pred={predicted[0]}  group_reward={rewards}")
        if frac_same == 1.0:
            print("  [!] degenerate group: all rewards equal -> zero advantage")
        print(f"{'-'*60}\n{responses[0][:400]}")
    return rewards

REWARDS = [format_reward, correctness_reward]
```

Load baseline model and perform evaluation:
```python
from vllm import SamplingParams

def evaluate(lora=None, n=100, temperature=0.0):
    subset = eval_dataset.select(range(n))
    prompts = [tokenizer.apply_chat_template(p, tokenize=False,
                                             add_generation_prompt=True)
               for p in subset["prompt"]]
    sp = SamplingParams(temperature=temperature, max_tokens=512, seed=CFG.seed)
    outs = model.fast_generate(prompts, sampling_params=sp,
                               **({"lora_request": lora} if lora else {}))
    texts = [o.outputs[0].text for o in outs]
    correct = sum(normalize_number(extract_xml_answer(t)) == a
                  for t, a in zip(texts, subset["answer"]))
    formatted = sum(bool(SOFT_RE.search(t)) for t in texts)
    return {"accuracy": correct / n, "format_rate": formatted / n}

baseline = evaluate(n=100)
print("\nBEFORE GRPO:", baseline)
```

Fine-tune the baseline model with GRPO algorithm and perform evaluation:
```python
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    
    # optimisation parameters
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    max_grad_norm = 0.1,

    # batch parameters
    per_device_train_batch_size = CFG.per_device_batch,
    gradient_accumulation_steps = CFG.grad_accum,
    num_generations = CFG.num_generations,
    max_prompt_length = CFG.max_prompt_length,
    max_completion_length = CFG.max_seq_length - CFG.max_prompt_length,
    max_steps = CFG.max_steps,
    seed = CFG.seed,

    # generation
    use_vllm = True,
    vllm_mode = "colocate",
    temperature = 1.0,
    top_p = 1.0,

    # GRPO objective
    beta = 0.04,                 # KL-to-reference coefficient (R1 keeps this)
    loss_type = "grpo",          # token-mean-then-sequence-mean, per the paper
    scale_rewards = "group",     # advantage divided by within-group std
    num_iterations = 1,          # mu=1 -> strictly on-policy

    mask_truncated_completions = True,

    # bookkeeping
    logging_steps = 1,
    save_steps = CFG.max_steps,
    report_to = "none",   # "trackio" or "none"
    output_dir = "outputs",
    log_completions = True,
    num_completions_to_print = 2,
)

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = REWARDS,
    reward_weights = [1.0, 1.0],   # correctness already carries 2.0 internally
    args = training_args,
    train_dataset = train_dataset,
)

trainer.train()
model.save_lora("grpo_saved_lora")

after = evaluate(lora=model.load_lora("grpo_saved_lora"), n=100)
print("\nAFTER GRPO:", after)
print(f"accuracy    {baseline['accuracy']:.1%}  ->  {after['accuracy']:.1%}")
print(f"format rate {baseline['format_rate']:.1%}  ->  {after['format_rate']:.1%}")
```

Test text generation:
```python
sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=768, seed=CFG.seed)
lora = model.load_lora("grpo_saved_lora")
text_prompts = [
    "What is 15 + 27?",
    "Explain the Pythagorean theorem in simple terms.",
    "Write a short Python function that returns the factorial of a number.",
    "A train travels 120 km in 2 hours. What is its average speed?",
    "What is the capital of France?",
    ]

for q in text_prompts:
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user",   "content": q}],
        tokenize=False, add_generation_prompt=True)
    print("=" * 70)
    print("Q:", q, "\n")
    print(model.fast_generate([text], sampling_params=sp, lora_request=lora)[0].outputs[0].text)
```

Here is an example of question, reasoning and answer:

```
Q: A train travels 120 km in 2 hours. What is its average speed? 

Rendering prompts: 100%
 1/1 [00:00<00:00, 101.40it/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.37it/s, est. speed input: 104.04 toks/s, output: 117.73 toks/s]<reasoning>
The train travels a distance of 120 km in 2 hours. To find the average speed, we can calculate the distance traveled divided by the time taken.
Average speed = distance / time
Average speed = 120 km / 2 hours = 60 km/hour

</reasoning>
<answer>
60 km/hour
</answer>
```

Upload model to the Hugging Face Hub:
```python
from google.colab import userdata
from huggingface_hub import login

login(token=userdata.get("HF_TOKEN"))
REPO = "quangvu197/Gemma-3-1b-GRPO-LORA"

model.push_to_hub(REPO)
tokenizer.push_to_hub(REPO)

model.push_to_hub_gguf(
    REPO,
    tokenizer,
    quantization_method = ["q4_k_m", "q8_0", "q5_k_m"],
)
```

Reload model from Hugging Face Hub and test text generation
```python
# load model from hugging face and test

from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name = "quangvu197/Gemma-3-1b-GRPO-LORA",
    max_seq_length = 2048,
    load_in_4bit = False,
)

messages = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": "What is 15 + 27?"}]

inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True).to(model.device)

out = model.generate(**inputs, max_new_tokens=512, temperature=0.7, top_p=0.95, do_sample=True)

print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

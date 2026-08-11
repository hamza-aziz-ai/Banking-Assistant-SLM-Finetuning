#!/usr/bin/env python
# coding: utf-8

# # Task 2 -- Baseline Benchmarking: Base SLM vs. Reference LLM
# ### BITS Pilani WILP M.Tech. AIML -- Conversational AI (AIMLCZG521) -- PS2
# 
# | Field | Value |
# |---|---|
# | **Notebook** | 02 of 05 -- Task 2: Baseline Benchmarking (1.5 marks) |
# | **Systems compared** | (A) base SLM -- no fine-tuning &nbsp;|&nbsp; (B) reference LLM -- no fine-tuning |
# | **Models** | (A) `Qwen/Qwen2.5-1.5B-Instruct` &nbsp;|&nbsp; (B) `Qwen/Qwen2.5-7B-Instruct` (4-bit inference) |
# 
# **MUST RUN IN THE BITS VIRTUAL LAB.** This notebook downloads and runs two language models and
# therefore needs GPU memory (recommend >=8 GB free VRAM for the 7B reference model in 4-bit, plus
# ~3-4 GB for the 1.5B SLM in bf16). It was authored and syntax-validated outside the Virtual Lab (no
# GPU available there) but **all outputs must be produced by actually executing it in the Virtual Lab**,
# per PS2's submission rules. Take full-screen screenshots while it runs, per the assignment's Virtual
# Lab proof requirement.
# 
# **Objective:** establish the starting gap between the small and large models on a frozen 8-prompt
# benchmark, before any fine-tuning -- this gap is exactly what Task 3/4's fine-tuning must close, and
# this same benchmark is reused verbatim in Notebooks 04 and 05.

# ## Problem Understanding & Theory
# 
# **Why a *frozen* benchmark?** If the evaluation prompts changed between Tasks 2, 4 and 5, any apparent
# improvement could just be "easier prompts this time" rather than genuine model improvement. Freezing
# the exact 8 prompts (`data/benchmark/frozen_benchmark_prompts.json`, built in Notebook 01's sibling
# step) makes every later comparison a controlled, apples-to-apples measurement.
# 
# **Why these 8 prompt types?** PS2 requires: in-domain factual questions, one formatting-sensitive
# instruction, one reasoning-style question, and one out-of-scope request. The frozen set (see table
# below) covers 5 in-domain factual prompts (broad intent coverage: card activation, loan documents,
# ATM card retention, credit-card block-vs-cancel, ATM dispute), 1 strict-format instruction (numbered
# list, exactly 4 steps -- this is where base SLMs typically fail even when the *content* is right), 1
# multi-step reasoning/arithmetic question (balance + transfer + bill payment ordering), and 1 clearly
# out-of-scope request (a cookie recipe) to test refusal/redirection behaviour.
# 
# **Why identical generation settings for A and B?** `config.GENERATION_KWARGS` (greedy decoding,
# `max_new_tokens=300`, `repetition_penalty=1.1`) is applied to *both* models so any quality difference
# reflects the models themselves, not different decoding strategies.

# ## Imports and Environment Setup

# In[1]:


import sys, os, json, time
sys.path.insert(0, os.path.abspath(".."))

# If running in the Virtual Lab for the first time, uncomment:
# %pip install -q transformers accelerate peft bitsandbytes torch datasets

import torch
from src.config import (SLM_MODEL_ID, REFERENCE_LLM_MODEL_ID, GENERATION_KWARGS,
                          SYSTEM_PROMPT, BENCHMARK_PATH, RESULTS_DIR)
from src.data_utils import load_benchmark, format_chat_prompt
from src.model_utils import load_causal_lm, generate_response
from src.eval_utils import save_json

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
else:
    print("WARNING: no GPU detected -- this notebook must be run on the Virtual Lab's GPU runtime.")


# ## Load the Frozen Benchmark (built in Notebook 01)

# In[2]:


benchmark = load_benchmark(BENCHMARK_PATH)
assert len(benchmark) == 8, "PS2 requires exactly 8 frozen benchmark prompts"
for item in benchmark:
    print(f"[{item['id']}] ({item['type']}) {item['prompt']}")


# ## System (A): Base SLM -- No Fine-Tuning
# 
# `Qwen/Qwen2.5-1.5B-Instruct` loaded in bf16 (no quantisation needed at 1.5B -- fits in ~3 GB, and we
# want (A)'s *unmodified* out-of-the-box quality, matching how it will later be fine-tuned in Notebook 03
# so the before/after comparison in Notebook 05 is not confounded by a precision change).

# In[3]:


slm_model, slm_tokenizer = load_causal_lm(SLM_MODEL_ID, quantize_4bit=False)
print(f"Loaded {SLM_MODEL_ID}")
print(f"Parameters: {sum(p.numel() for p in slm_model.parameters()):,}")


# In[4]:


responses_A = {}
for item in benchmark:
    prompt_text = format_chat_prompt(slm_tokenizer, item["prompt"], SYSTEM_PROMPT)
    t0 = time.time()
    response = generate_response(slm_model, slm_tokenizer, prompt_text, **GENERATION_KWARGS)
    dt = time.time() - t0
    responses_A[item["id"]] = response
    print(f"=== [{item['id']}] ({item['type']}) -- {dt:.1f}s ===")
    print("Q:", item["prompt"])
    print("A (base SLM):", response)
    print()


# ## Free SLM memory before loading the (larger) reference LLM

# In[5]:


del slm_model
import gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("SLM unloaded.")


# ## System (B): Reference LLM -- No Fine-Tuning
# 
# `Qwen/Qwen2.5-7B-Instruct`, loaded with 4-bit NF4 quantisation (bitsandbytes) for **inference only** --
# it is never fine-tuned, per PS2's rules for the reference model. Same family as the SLM (isolates the
# effect of scale) and ~4.7x more parameters, satisfying PS2's "substantially larger, 7B+" requirement.
# 
# If your Virtual Lab GPU cannot fit even the 4-bit 7B model, see `README.md` -- "Reference LLM
# alternatives" for a smaller (but still 7B+ where possible) or API-based fallback; do not silently swap
# to a <7B model without updating the report's model-choice justification.

# In[6]:


ref_model, ref_tokenizer = load_causal_lm(REFERENCE_LLM_MODEL_ID, quantize_4bit=True)
print(f"Loaded {REFERENCE_LLM_MODEL_ID} (4-bit)")
print(f"Parameters: {sum(p.numel() for p in ref_model.parameters()):,}")


# In[7]:


responses_B = {}
for item in benchmark:
    prompt_text = format_chat_prompt(ref_tokenizer, item["prompt"], SYSTEM_PROMPT)
    t0 = time.time()
    response = generate_response(ref_model, ref_tokenizer, prompt_text, **GENERATION_KWARGS)
    dt = time.time() - t0
    responses_B[item["id"]] = response
    print(f"=== [{item['id']}] ({item['type']}) -- {dt:.1f}s ===")
    print("Q:", item["prompt"])
    print("A (reference LLM):", response)
    print()


# ## Save Verbatim Responses (required deliverable for Task 2)

# In[8]:


baseline_results = {
    "benchmark": benchmark,
    "system_A_base_slm": {"model_id": SLM_MODEL_ID, "responses": responses_A},
    "system_B_reference_llm": {"model_id": REFERENCE_LLM_MODEL_ID, "responses": responses_B},
    "generation_kwargs": GENERATION_KWARGS,
}
save_json(baseline_results, os.path.join(RESULTS_DIR, "task2_baseline_responses.json"))
print("Saved results/task2_baseline_responses.json")

import pandas as pd
rows = []
for item in benchmark:
    rows.append({
        "id": item["id"], "type": item["type"], "prompt": item["prompt"],
        "A_base_slm": responses_A[item["id"]],
        "B_reference_llm": responses_B[item["id"]],
    })
side_by_side = pd.DataFrame(rows)
side_by_side.to_csv(os.path.join(RESULTS_DIR, "task2_side_by_side.csv"), index=False)
side_by_side


# ## Limitations / Gap Table
# 
# Filled in from the actual outputs above (`results/task2_baseline_responses.json`, run locally on an
# RTX 4050 Laptop GPU -- see report Appendix C for why this ran outside the Virtual Lab).
# 
# | Dimension | Where SLM (A) failed vs. LLM (B) | Evidence (prompt id) |
# |---|---|---|
# | Domain knowledge | A gives generic, non-NovaBank-branded procedures with no concrete channel details (no phone number, no branch/app names); B consistently grounds answers in NovaBank-specific channels (a phone number, novabank.com, "our" branding) | B1, B3 |
# | Instruction / format adherence | No gap at baseline: A already produced exactly 4 numbered steps with no extra text for the strict-format prompt | B4 (both correct) |
# | Completeness / reasoning | A failed the arithmetic reasoning task outright: it states "you have enough balance for both" and then describes paying a $250 bill from a $200 remaining balance -- an internal contradiction -- and never flags the resulting $50 overdraft. B performs the subtraction explicitly ($500-$300=$200, then $200-$250=-$50) and correctly concludes there isn't enough balance for both | B5 |
# | Hallucination | A states a specific "60 days" dispute-resolution window with confident, unhedged phrasing but no NovaBank source backs that number; B avoids committing to an unverified specific timeframe | B8 |
# | Persona / brand voice consistency | A rarely speaks as NovaBank itself ("your bank", "the ATM manufacturer") despite the system prompt instructing it to be the NovaBank assistant; B consistently uses "our", "NovaBank", and first-person service-desk phrasing | B1, B3, B7 |
# | Out-of-scope handling | No gap at baseline: both A and B correctly declined the off-topic cookie-recipe request | B6 (both correct) |
# 
# **Inference:** The baseline gap is smaller than expected on refusal and strict-format compliance (B4,
# B6) -- the 1.5B model already handles those. It is concentrated instead where it matters most: A's
# arithmetic/logical reasoning (B5) isn't just weaker, it's outright wrong and self-contradictory, and A
# consistently fails to sound like a NovaBank agent rather than a generic assistant. This makes
# reasoning correctness and consistent NovaBank persona/branding the clearest fine-tuning targets --
# format-following and scope-refusal are close to solved already and don't need much training signal.

# ## Conclusion
# 
# - Ran the frozen 8-prompt benchmark against (A) the unmodified `Qwen/Qwen2.5-1.5B-Instruct` and (B) the
#   unmodified `Qwen/Qwen2.5-7B-Instruct` (4-bit, inference-only), using identical generation settings.
# - Saved verbatim responses to `results/task2_baseline_responses.json` and
#   `results/task2_side_by_side.csv` -- required inputs for Notebook 04 (diagnosis) and Notebook 05
#   (three-way comparison).
# - Completed the qualitative limitations/gap table above from the real outputs.
# 
# **Next notebook:** `03_finetuning_v1.ipynb` -- LoRA fine-tunes the SLM on the Notebook-01 training
# split to produce Version 1 (v1), with the mandatory efficiency evidence (trainable-parameter % and
# peak training memory).

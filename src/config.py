"""
Central configuration for the NovaBank SLM fine-tuning project (PS2).

Why a single config module: every notebook (01-05) needs the same model
names, paths and generation defaults. Centralising them here means a
config change (e.g. swapping the SLM) only has to happen in one place,
and every notebook stays consistent with what Task 2's frozen benchmark
was actually run against.
"""
import os

# ---------------------------------------------------------------------------
# 6GB-class GPU (e.g. RTX 4050 Laptop, or the Virtual Lab's T4) safety net.
# Must be set before the first CUDA allocation (any `.cuda()`/`from_pretrained`
# call) -- reduces alloc-retry OOMs from fragmentation, which matters here
# because notebooks 02/05 load+unload multiple models in one process.
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# hf_transfer: Rust-based multi-connection downloader for the Hugging Face
# Hub. The default single-connection downloader repeatedly dropped and
# restarted (never resumed) mid-way through the ~15GB Qwen2.5-7B-Instruct
# download on this network, leaving 50GB of dead .incomplete shards without
# ever completing. hf_transfer's chunked/resumable transfer fixes that.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Paths (all relative to the project root; notebooks live in notebooks/ and
# `PROJECT_ROOT` resolves one level up regardless of the notebook's cwd).
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
BENCHMARK_PATH = os.path.join(DATA_DIR, "benchmark", "frozen_benchmark_prompts.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
ADAPTER_V1_DIR = os.path.join(MODELS_DIR, "novabank-slm-lora-v1")
ADAPTER_V2_DIR = os.path.join(MODELS_DIR, "novabank-slm-lora-v2")

for d in (MODELS_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Model choice
# ---------------------------------------------------------------------------
# (A) Small Language Model -- fine-tuning target.
#   Qwen2.5-1.5B-Instruct: 1.5B params, Apache-2.0, strong instruction-
#   following for its size, well supported by transformers/peft/bitsandbytes,
#   fits comfortably (fp16 ~3GB, 4-bit ~1GB) on a single "modest hardware"
#   GPU such as the Virtual Lab's T4/A10-class card.
SLM_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# (B) Reference LLM -- inference-only upper bound, never fine-tuned.
#   Qwen2.5-7B-Instruct: same model family as the SLM (isolates the effect
#   of *scale* rather than confounding it with a different pretraining
#   corpus/tokenizer), Apache-2.0, and fits in ~5-6GB VRAM at 4-bit
#   (bitsandbytes NF4) quantisation for inference-only use -- tested against
#   a 6GB-class card (RTX 4050 Laptop / Virtual Lab T4), leaves ~0.5-1GB
#   headroom for KV cache at GENERATION_KWARGS' max_new_tokens=300.
#   model_utils.load_causal_lm forces single-GPU placement for 4-bit loads
#   (device_map={"": 0}) rather than "auto" -- `infer_auto_device_map`'s
#   conservative padding otherwise splits this model across GPU+CPU on a
#   6GB card even though the weights (~3.8GB, no double-quant math) fit
#   whole on the ~5.1GB actually free, and that CPU-offload split hits
#   real accelerate/bitsandbytes hook bugs on this version pairing. If it
#   genuinely doesn't fit on your card, swap to
#   "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4" or an API-based model (see README
#   "Reference LLM alternatives").
REFERENCE_LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# ---------------------------------------------------------------------------
# Generation defaults (kept identical across A / B / C so comparisons in
# Tasks 2, 4 and 5 are not confounded by different decoding settings)
# ---------------------------------------------------------------------------
GENERATION_KWARGS = dict(
    max_new_tokens=300,
    do_sample=False,      # greedy decoding: reproducible, and formatting-
                           # sensitive prompts (Task 2, B4) need determinism
    temperature=1.0,
    repetition_penalty=1.1,
)

SYSTEM_PROMPT = (
    "You are the NovaBank Virtual Assistant, a retail-banking customer "
    "support agent. Answer clearly and only about NovaBank banking "
    "services (cards, loans, mortgages, accounts, transfers, ATMs, "
    "passwords, fees). If a request is unrelated to banking, politely "
    "say you can only help with NovaBank banking questions."
)

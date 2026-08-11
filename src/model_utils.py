"""
Model loading, generation and efficiency-measurement utilities.

These helpers wrap `transformers` + `peft` + `bitsandbytes` calls so the
notebooks stay focused on *what* is being done (load base model, attach
LoRA, generate) rather than *how* (exact HF API plumbing). All heavy
imports are done lazily inside functions so this module can still be
imported (e.g. for the parameter-counting helpers) on a machine without a
GPU or without bitsandbytes installed.
"""
from typing import Optional


def load_causal_lm(model_id: str, quantize_4bit: bool = False, device_map: str = "auto"):
    """
    Load a causal language model + tokenizer from the Hugging Face Hub.

    Theory: `quantize_4bit=True` loads weights in 4-bit NF4 precision via
    bitsandbytes (QLoRA-style). This roughly quarters the memory footprint
    of fp16 loading, which is what makes it possible to run a 7B-class
    reference LLM for *inference only* on a single modest GPU, and is also
    the mechanism the fine-tuning notebooks (03/04) use to fit the base
    SLM plus optimizer state in memory during LoRA training.

    Args:
        model_id (str): a Hugging Face model repo id, e.g.
            "Qwen/Qwen2.5-1.5B-Instruct".
        quantize_4bit (bool): if True, load with bitsandbytes 4-bit NF4
            quantisation (requires a CUDA GPU + bitsandbytes installed).
        device_map (str): passed through to `from_pretrained`; "auto"
            lets HF Accelerate place layers on available GPU(s)/CPU.

    Returns:
        (model, tokenizer): the loaded HF model and its tokenizer, with
            the tokenizer's pad token set (Qwen2.5 has no pad token by
            default, which would otherwise break batched generation).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(device_map=device_map, torch_dtype=torch.bfloat16)
    if quantize_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs.pop("torch_dtype")  # dtype comes from the quant config

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    return model, tokenizer


def load_lora_adapter(base_model, adapter_path: str):
    """
    Attach a saved LoRA adapter (from Task 3/4) onto a loaded base model
    for inference.

    Args:
        base_model: a loaded `AutoModelForCausalLM` (same base checkpoint
            the adapter was trained against -- SLM_MODEL_ID).
        adapter_path (str): local directory containing the saved
            `adapter_model.safetensors` + `adapter_config.json`.
    Returns:
        the base model wrapped with the LoRA adapter (a `PeftModel`),
        ready for `.generate(...)`.
    """
    from peft import PeftModel
    return PeftModel.from_pretrained(base_model, adapter_path)


def generate_response(model, tokenizer, prompt_text: str, **generation_kwargs) -> str:
    """
    Run greedy generation for a single fully-rendered chat prompt and
    return only the newly generated assistant text (the prompt itself is
    stripped from the decoded output).

    Args:
        model: a loaded causal LM (base, quantised, or LoRA-wrapped).
        tokenizer: the matching tokenizer.
        prompt_text (str): output of `data_utils.format_chat_prompt`.
        **generation_kwargs: forwarded to `model.generate` (see
            config.GENERATION_KWARGS for the defaults used throughout
            this project so A/B/C stay comparable).
    Returns:
        str: the decoded assistant response only.
    """
    import torch

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            pad_token_id=tokenizer.pad_token_id,
            **generation_kwargs,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def count_trainable_parameters(model):
    """
    Compute trainable vs. total parameter counts -- the mandatory
    "efficiency evidence" for Task 3.

    Why this matters: this is the single number that most directly shows
    *why* LoRA is parameter-efficient -- typically well under 1% of the
    base model's parameters are updated, versus 100% for full fine-tuning.

    Args:
        model: a PEFT-wrapped model (after `get_peft_model`).
    Returns:
        dict with trainable_params, total_params and the percentage.
    """
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4) if total else 0.0,
    }


def peak_gpu_memory_mb() -> Optional[float]:
    """
    Return the peak CUDA memory allocated (MB) since the last
    `torch.cuda.reset_peak_memory_stats()` call, or None if no GPU is
    available. Call `torch.cuda.reset_peak_memory_stats()` immediately
    before the training loop, then this function right after `trainer.train()`
    to get the "peak memory during training" number Task 3 requires.
    """
    import torch
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

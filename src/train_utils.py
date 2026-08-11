"""
LoRA fine-tuning utilities: dataset tokenisation, LoRA config builder, and
a thin wrapper around `transformers.Trainer` that also plots loss curves.

Intuition (why LoRA/QLoRA is parameter-efficient, in one paragraph): a
pretrained model's weight matrices are frozen; instead of updating a full
weight matrix W (d x k parameters), LoRA adds a low-rank update
  W' = W + (alpha/r) * B @ A
where A is (r x k) and B is (d x r), with r << min(d, k). Only A and B are
trained -- typically <1% of the base model's parameters -- so the
optimizer state (Adam keeps 2 extra fp32 buffers per trainable parameter)
and gradient memory shrink by the same factor. QLoRA additionally loads
the frozen base weights in 4-bit precision, so the (much larger) frozen
part of the model also costs ~4x less memory than fp16 -- combined, this
is what lets a 1.5B model fine-tune on a single modest GPU.
"""
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_lora_config(r: int = 8, alpha: int = 16, dropout: float = 0.05,
                       target_modules=None):
    """
    Construct a `peft.LoraConfig` for causal-LM fine-tuning.

    Args:
        r (int): LoRA rank -- the bottleneck dimension of the low-rank
            update. Smaller r = fewer trainable params but less capacity
            to adapt; r=8 is a standard, well-tested starting point for
            <3B instruction models and small (few-hundred-example)
            datasets like this one (justified further in Notebook 03).
        alpha (int): LoRA scaling factor; the update is scaled by
            alpha/r. alpha=2*r (=16 here) is the widely-used default from
            the original LoRA paper and keeps the effective learning
            rate stable across different r choices.
        dropout (float): dropout applied inside the LoRA adapter path
            only, regularising against overfitting on a small (286-
            example) training set.
        target_modules (list[str] | None): which linear layers get a
            LoRA adapter. Defaults to the attention projections, which
            capture most of the adaptation benefit at the lowest
            parameter cost (adding MLP projections roughly doubles
            trainable params for typically <1 point of extra quality on
            instruction-following tasks this size).
    Returns:
        peft.LoraConfig
    """
    from peft import LoraConfig, TaskType

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )


def build_tokenized_dataset(records: List[Dict], tokenizer, format_fn,
                             max_length: int = 512):
    """
    Turn a list of {instruction, input, output} dicts into a HF `Dataset`
    of tokenised, causally-masked training examples.

    Masking rationale: loss is computed ONLY on the assistant's response
    tokens (the prompt/system/user tokens are masked to -100), so the
    model is not penalised for "failing to predict" the user's own
    question -- it should only be trained to produce the answer.

    Args:
        records (List[Dict]): rows with 'instruction', 'input', 'output'.
        tokenizer: HF tokenizer (with a chat template).
        format_fn: callable(tokenizer, instruction, output, input_context) -> str,
            i.e. `data_utils.format_training_example`.
        max_length (int): truncation length. 512 covers this dataset's
            longest response (~403 words ~ 550-600 tokens headroom) while
            keeping memory bounded; justified against the measured
            response-length distribution in Notebook 01.
    Returns:
        datasets.Dataset with input_ids/attention_mask/labels columns.
    """
    from datasets import Dataset

    input_ids_list, attn_list, labels_list = [], [], []
    for r in records:
        full_text = format_fn(tokenizer, r["instruction"], r["output"], r.get("input", ""))
        # Prompt-only text (no assistant answer) is used purely to find the
        # split point between "context to mask" and "answer to train on".
        prompt_only = format_fn(tokenizer, r["instruction"], "", r.get("input", ""))

        full_ids = tokenizer(full_text, truncation=True, max_length=max_length)["input_ids"]
        prompt_ids = tokenizer(prompt_only, truncation=True, max_length=max_length)["input_ids"]
        prompt_len = min(len(prompt_ids), len(full_ids))

        labels = [-100] * prompt_len + full_ids[prompt_len:]
        labels = labels[:len(full_ids)]

        input_ids_list.append(full_ids)
        attn_list.append([1] * len(full_ids))
        labels_list.append(labels)

    return Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": attn_list,
        "labels": labels_list,
    })


def plot_loss_curves(log_history, save_path: str, title: str = "Training Loss"):
    """
    Plot training (and, if present, validation) loss curves from a HF
    `Trainer.state.log_history` and save to disk.

    Args:
        log_history (list[dict]): `trainer.state.log_history` after
            `.train()` completes.
        save_path (str): PNG output path.
        title (str): plot title (e.g. "v1 Training Loss", "v2 Training Loss").
    """
    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            train_steps.append(entry.get("step", len(train_steps)))
            train_losses.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step", len(eval_steps)))
            eval_losses.append(entry["eval_loss"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if train_losses:
        ax.plot(train_steps, train_losses, label="train loss", color="#2E5EAA")
    if eval_losses:
        ax.plot(eval_steps, eval_losses, label="val loss", color="#C9642C", marker="o")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path

"""
Data loading and prompt-formatting utilities shared by every notebook.

Theory note (why a chat template matters here): instruction-tuned models
like Qwen2.5-Instruct expect input wrapped in a specific chat template
(special role tokens around system/user/assistant turns). Feeding raw
text without this template measurably degrades instruction-following,
which would unfairly bias the base-SLM-vs-reference-LLM comparison in
Task 2. `format_chat_prompt` always goes through the tokenizer's own
`apply_chat_template`, so both models are prompted the way they were
actually trained to be prompted.
"""
import json
import os
from typing import Dict, List, Optional

from .config import SYSTEM_PROMPT


def load_jsonl(path: str) -> List[Dict]:
    """
    Read a JSON-Lines file into a list of dicts.

    Args:
        path (str): path to a .jsonl file, one JSON object per line.
    Returns:
        List[Dict]: the parsed records, in file order.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(processed_dir: str, split: str) -> List[Dict]:
    """
    Load one of the train/val/test JSONL splits produced by
    `build_dataset.py` (Task 1).

    Args:
        processed_dir (str): directory containing train.jsonl/val.jsonl/test.jsonl.
        split (str): one of "train", "val", "test".
    Returns:
        List[Dict]: records with keys instruction/input/output/category/intent.
    """
    assert split in {"train", "val", "test"}, f"unknown split '{split}'"
    return load_jsonl(os.path.join(processed_dir, f"{split}.jsonl"))


def load_benchmark(benchmark_path: str) -> List[Dict]:
    """Load the frozen 8-prompt benchmark (Task 2), which is reused
    verbatim in Tasks 4 and 5 -- do not regenerate or reorder this file
    once Task 2 has been run, since the whole point is a fixed yardstick."""
    with open(benchmark_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_chat_prompt(tokenizer, user_message: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """
    Build the exact string to feed a causal-LM `generate()` call, using the
    model's own chat template so system/user/assistant turns are delimited
    the way the model was instruction-tuned to expect.

    Args:
        tokenizer: a HF tokenizer with `apply_chat_template` support
            (true for all Qwen2.5-Instruct checkpoints).
        user_message (str): the raw user instruction.
        system_prompt (str): persona/system instruction (see config.py).
    Returns:
        str: a fully rendered prompt string ready for `tokenizer(...)`.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def format_training_example(tokenizer, instruction: str, response: str,
                             input_context: str = "", system_prompt: str = SYSTEM_PROMPT) -> str:
    """
    Render one (instruction, response) pair as a full chat-formatted
    training string (prompt + assistant turn + eos), used to build the
    LoRA fine-tuning dataset in Task 3/4.

    Args:
        tokenizer: HF tokenizer with a chat template.
        instruction (str): the user instruction.
        response (str): the target assistant response (ground truth).
        input_context (str): optional extra context appended to the user
            turn; empty for this dataset (see DATASET_CARD.md field notes).
        system_prompt (str): persona/system instruction.
    Returns:
        str: the full training string, including the assistant's answer,
            terminated by the tokenizer's EOS token so the model learns
            where to stop generating.
    """
    user_content = instruction if not input_context else f"{instruction}\n\nContext: {input_context}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

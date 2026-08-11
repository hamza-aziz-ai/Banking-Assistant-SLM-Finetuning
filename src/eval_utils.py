"""
Evaluation utilities for Tasks 2, 4 and 5: rubric scoring scaffolding,
automatic text-overlap metrics, and simple result-table builders.
"""
import json
import os
from typing import Dict, List

import pandas as pd


RUBRIC_CRITERIA = [
    "domain_correctness",     # is the banking information factually right?
    "instruction_format_adherence",  # did it follow explicit format instructions?
    "completeness",           # does it fully answer, without missing steps?
    "hallucination",          # inverted-scored: 2 = no fabricated facts, 0 = fabricated
]
RUBRIC_SCALE = "0-2 per criterion (0=fails, 1=partially meets, 2=fully meets)"


def new_score_sheet(system_name: str, benchmark: List[Dict]) -> pd.DataFrame:
    """
    Build an empty rubric scoring sheet (one row per frozen-benchmark
    prompt, one column per criterion) for a given system (A/B/C), to be
    filled in manually (or by an LLM-as-judge call) in Notebook 05.

    Args:
        system_name (str): "A_base_slm" | "B_reference_llm" | "C_finetuned_slm_v2".
        benchmark (List[Dict]): the 8 frozen prompts (id, type, prompt).
    Returns:
        pd.DataFrame indexed by prompt id, one column per rubric criterion
        plus a `response` column to paste the recorded model output into.
    """
    rows = []
    for item in benchmark:
        row = {"prompt_id": item["id"], "prompt_type": item["type"], "system": system_name, "response": ""}
        for c in RUBRIC_CRITERIA:
            row[c] = None
        rows.append(row)
    return pd.DataFrame(rows).set_index("prompt_id")


def score_totals(score_df: pd.DataFrame) -> Dict[str, float]:
    """Sum rubric columns per system for the Task 5 per-system totals table."""
    return {c: float(score_df[c].sum()) for c in RUBRIC_CRITERIA if c in score_df.columns}


def rouge_l_f1(prediction: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 between a generated response and its reference,
    using the standard Longest-Common-Subsequence formulation. Included
    as a lightweight, dependency-light fallback automatic metric; prefer
    the `rouge-score` or `evaluate`/BERTScore libraries in Notebook 05 if
    installed (see that notebook's Task 5 cell for the primary metric and
    its documented limitations -- surface overlap != semantic correctness).

    Args:
        prediction (str): model-generated response.
        reference (str): ground-truth response from the test split.
    Returns:
        float: ROUGE-L F1 in [0, 1].
    """
    def _lcs(a_tokens, b_tokens):
        n, m = len(a_tokens), len(b_tokens)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if a_tokens[i - 1] == b_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[n][m]

    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def save_json(obj, path: str):
    """Write a JSON-serialisable object to disk, creating parent dirs."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

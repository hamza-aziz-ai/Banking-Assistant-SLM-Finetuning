"""
Task 1 -- Instruction Dataset Preparation
Domain: Retail Banking Customer Support ("NovaBank" virtual assistant)
Source: bitext/Bitext-retail-banking-llm-chatbot-training-dataset (Hugging Face)
        License: CDLA-Sharing-1.0 (Community Data License Agreement - Sharing)
        Citation: Bitext Innovations Intl. (2024). Retail Banking LLM Chatbot
        Training Dataset. https://huggingface.co/datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset

This script performs a REPRODUCIBLE (fixed seed) build of the flat
instruction -> response dataset used across Tasks 1-5 of PS2. It:
  1. Downloads the source Bitext dataset (25,545 rows, 26 intents).
  2. Draws a larger stratified CANDIDATE POOL across all intents.
  3. Resolves Bitext's {{PLACEHOLDER}} template tokens into concrete,
     brand-consistent values for a fictional bank ("NovaBank"), so the
     fine-tuned model learns to produce deployable, non-templated text.
  4. Cleans the candidate pool: strips whitespace, removes exact/near
     duplicates (including duplicates created by placeholder resolution),
     removes empty/too-short responses -- WITH before/after counts.
  5. Draws the final >=300-example dataset as a balanced stratified
     sample of the CLEANED pool.
  6. Splits into train / validation / test (80 / 10 / 10, stratified by intent).
  7. Writes JSONL files + a dataset card with before/after statistics.
"""
import hashlib
import json
import os
import random
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BRAND = "NovaBank"
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(OUT_DIR, "raw")
PROC_DIR = os.path.join(OUT_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

csv_path = hf_hub_download(
    repo_id="bitext/Bitext-retail-banking-llm-chatbot-training-dataset",
    filename="bitext-retail-banking-llm-chatbot-training-dataset.csv",
    repo_type="dataset",
)
df_raw = pd.read_csv(csv_path)
df_raw.to_csv(os.path.join(RAW_DIR, "bitext_retail_banking_full.csv"), index=False)
n_raw_total = len(df_raw)

FAKE_PHONE = "1-800-555-0199"
FAKE_URL = "www.novabank.com"
FAKE_EMAIL = "support@novabank.com"
FAKE_HOURS = "Monday-Friday, 8 AM-8 PM (local time)"
FAKE_APP = f"{BRAND} Mobile App"


def resolve_placeholder(raw_name: str) -> str:
    """
    Map a single Bitext {{...}} slot name to a concrete filler string.

    Why needed: the source dataset ships instruction-tuning style
    template slots (e.g. "{{Credit Card}}") rather than resolved text.
    Training directly on literal "{{...}}" tokens would teach the SLM to
    emit curly-brace placeholders in production, which is exactly the
    "inconsistent formatting" failure mode the assignment asks us to fix.

    Args:
        raw_name (str): the text found between "{{" and "}}", e.g.
            "Customer Support Phone Number".

    Returns:
        str: a concrete, brand-consistent replacement string.

    Rules are applied in priority order (most specific first) using
    case-insensitive substring matching on the slot name. This many-to-few
    mapping is also what creates a genuine cleaning signal later: two
    source rows that only differed in *which* phone-number slot name they
    used can become exact duplicates after resolution, and are removed by
    the post-resolution dedup step below.
    """
    name = raw_name.strip()
    low = name.lower()

    if "url" in low or "website" in low or ".com" in low:
        return FAKE_URL
    if "email" in low:
        return FAKE_EMAIL
    if any(k in low for k in ["phone number", "hotline", "helpline", "contact number", "phone"]):
        return FAKE_PHONE
    if "working hours" in low or "hours" in low or "availability" in low:
        return FAKE_HOURS
    if "app" in low and "apply" not in low:
        return FAKE_APP
    if any(k in low for k in ["bank name", "bank's name", "company name", "company"]):
        return BRAND
    if "bank" in low and "account" not in low:
        return BRAND
    if any(k in low for k in ["credit card", "debit card", "visa", "amex", "american express", "card issuer"]):
        return "card"
    if re.search(r"\b(id|number|reference|iban|routing|swift|pin|otp)\b", low) and "phone" not in low:
        digits = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:8], 16) % 10 ** 8
        return f"{digits:08d}"
    if any(k in low for k in ["amount", "fee", "balance", "charge", "price"]):
        return "$25.00"
    if "date" in low:
        return "15th of this month"
    if "name" in low:
        return "the account holder's name"
    if "address" in low:
        return "your registered address"
    return f"your {name.lower()}" if len(name.split()) <= 4 else name.lower()


PLACEHOLDER_RE = re.compile(r"\{\{([^{}]+)\}\}")


def resolve_text(text: str) -> str:
    """Replace every {{...}} occurrence in `text` using resolve_placeholder."""
    return PLACEHOLDER_RE.sub(lambda m: resolve_placeholder(m.group(1)), text)


def normalise_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs and excess blank lines produced by
    substitution or present in the raw export."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


PER_INTENT_POOL = 20
pool_frames = []
for intent, group in df_raw.groupby("intent"):
    take = min(PER_INTENT_POOL, len(group))
    pool_frames.append(group.sample(n=take, random_state=SEED))
df_pool = pd.concat(pool_frames, ignore_index=True)
n_pool_raw = len(df_pool)

pool_records = []
n_placeholders_resolved = 0
for _, row in df_pool.iterrows():
    instr_raw, resp_raw = str(row["instruction"]).strip(), str(row["response"]).strip()
    n_placeholders_resolved += len(PLACEHOLDER_RE.findall(instr_raw)) + len(PLACEHOLDER_RE.findall(resp_raw))
    pool_records.append({
        "instruction": normalise_whitespace(resolve_text(instr_raw)),
        "input": "",
        "output": normalise_whitespace(resolve_text(resp_raw)),
        "category": row["category"],
        "intent": row["intent"],
    })

n_before_clean = len(pool_records)

MIN_RESPONSE_CHARS = 20
cleaned = [r for r in pool_records if len(r["output"]) >= MIN_RESPONSE_CHARS and len(r["instruction"]) >= 5]
n_after_length_filter = len(cleaned)

seen = set()
deduped = []
n_dupes_from_resolution = 0
for r in cleaned:
    key = (r["instruction"].lower(), r["output"].lower())
    if key in seen:
        n_dupes_from_resolution += 1
        continue
    seen.add(key)
    deduped.append(r)
cleaned = deduped
n_after_dedup = len(cleaned)

cleaned = [r for r in cleaned if "{{" not in r["instruction"] and "{{" not in r["output"]]
n_after_placeholder_check = len(cleaned)

FINAL_PER_INTENT = 13
by_intent_clean = defaultdict(list)
for r in cleaned:
    by_intent_clean[r["intent"]].append(r)

final_records = []
for intent, items in by_intent_clean.items():
    rng = random.Random(SEED)
    rng.shuffle(items)
    final_records.extend(items[:FINAL_PER_INTENT])

rng = random.Random(SEED)
rng.shuffle(final_records)
n_final = len(final_records)

by_intent_final = defaultdict(list)
for r in final_records:
    by_intent_final[r["intent"]].append(r)

train, val, test = [], [], []
for intent, items in by_intent_final.items():
    rng = random.Random(SEED)
    rng.shuffle(items)
    n = len(items)
    n_val = max(1, round(n * 0.10))
    n_test = max(1, round(n * 0.10))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train, n_val, n_test = n, 0, 0
    train.extend(items[:n_train])
    val.extend(items[n_train:n_train + n_val])
    test.extend(items[n_train + n_val:n_train + n_val + n_test])

rng = random.Random(SEED)
rng.shuffle(train)
rng.shuffle(val)
rng.shuffle(test)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


write_jsonl(os.path.join(PROC_DIR, "train.jsonl"), train)
write_jsonl(os.path.join(PROC_DIR, "val.jsonl"), val)
write_jsonl(os.path.join(PROC_DIR, "test.jsonl"), test)


def length_stats(rows, field):
    lens = [len(r[field].split()) for r in rows]
    return {
        "count": len(lens),
        "mean_words": round(float(np.mean(lens)), 2) if lens else 0,
        "min_words": int(np.min(lens)) if lens else 0,
        "max_words": int(np.max(lens)) if lens else 0,
    }


stats = {
    "seed": SEED,
    "source_dataset": "bitext/Bitext-retail-banking-llm-chatbot-training-dataset",
    "source_license": "CDLA-Sharing-1.0",
    "source_rows_total": n_raw_total,
    "candidate_pool_before_cleaning": n_pool_raw,
    "cleaning_pipeline": {
        "0_candidate_pool_size": n_before_clean,
        "1_placeholder_tokens_resolved": n_placeholders_resolved,
        "2_after_length_filter (drop response<20 chars or instruction<5 chars)": n_after_length_filter,
        "3_after_duplicate_removal": n_after_dedup,
        "3a_duplicates_removed_count": n_dupes_from_resolution,
        "4_after_unresolved_placeholder_safety_check": n_after_placeholder_check,
    },
    "final_dataset_size_before_split": n_final,
    "final_counts": {
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "total": len(train) + len(val) + len(test),
    },
    "split_ratio": "80/10/10 (stratified by intent)",
    "num_intents_covered": len(by_intent_final),
    "num_categories_covered": int(pd.Series([r["category"] for r in final_records]).nunique()),
    "instruction_length_words": length_stats(final_records, "instruction"),
    "response_length_words": length_stats(final_records, "output"),
}

with open(os.path.join(PROC_DIR, "dataset_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

write_jsonl(os.path.join(RAW_DIR, "candidate_pool_cleaned.jsonl"), cleaned)

print(json.dumps(stats, indent=2))

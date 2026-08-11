# NovaBank Retail Banking Instruction Dataset -- Dataset Card

## 1. Source & Provenance
- **Source dataset:** [`bitext/Bitext-retail-banking-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-retail-banking-llm-chatbot-training-dataset) (Hugging Face Hub)
- **License:** CDLA-Sharing-1.0 (Community Data License Agreement -- Sharing). Redistribution and modification are permitted with attribution and share-alike terms; attribution is given here and in the report.
- **Citation:** Bitext Innovations International, S.L. *Retail Banking LLM Chatbot Training Dataset.* Hugging Face Datasets, 2024.
- **Original size:** 25,545 rows, 26 intents, 9 categories (CARD, LOAN, ACCOUNT, FIND, CONTACT, TRANSFER, ATM, PASSWORD, FEES).
- **Generation method of the *source*:** Bitext states the dataset was built with their hybrid methodology combining expert-curated linguistic templates ("tags" encode paraphrase/register variation: politeness, colloquialism, typos, etc.) with generative expansion -- i.e. **synthetic + template-based**, not scraped real conversations. This is disclosed here as required by the assignment.
- **Generation method of *this derived dataset*:** Not newly synthesized -- adapted from the above public source via stratified sampling, placeholder resolution, cleaning, and re-splitting (all steps below, fully scripted and reproducible in `build_dataset.py`).

## 2. Domain & Persona
Fictional bank **"NovaBank"** retail-banking customer support assistant. Covers card management, loans/mortgages, account management, ATM/branch lookup, transfers, password/security, and fees.

## 3. Construction Pipeline (seed = 42, fully reproducible)
1. **Candidate pool:** stratified sample of 20 rows/intent (520 rows) drawn from the 25,545-row source, covering all 26 intents.
2. **Placeholder resolution:** the source responses use `{{SLOT}}` template tokens (e.g. `{{Credit Card}}`, `{{Customer Support Phone Number}}`) rather than resolved text. A rule-based resolver (`resolve_placeholder`, keyword-matched on the slot name) rewrites every token into a concrete, brand-consistent value (e.g. all phone-number-like slots -> `1-800-555-0199`; all URL/website slots -> `www.novabank.com`). **744 placeholder tokens** were resolved across the 520-row candidate pool. This is the dataset's main formatting-inconsistency fix: without it, the model would learn to emit literal `{{...}}` tokens in production.
3. **Whitespace normalisation:** collapsed repeated spaces/blank lines left over from substitution.
4. **Length filter:** dropped any row with a response under 20 characters or an instruction under 5 characters (empty/truncated-response guard). Result on this pool: **0 rows dropped** (source rows were already complete).
5. **Duplicate removal:** dropped exact `(instruction, response)` duplicates, including any created by placeholder resolution collapsing two differently-worded template slots into identical text. Result: **0 duplicates found** -- the Bitext source is pre-deduplicated at the row level, so this step is defensive (kept in the pipeline; would matter if this pipeline is later re-run against a noisier or self-collected source).
6. **Safety check:** verified no `{{` survives resolution. **0 leftover placeholders.**
7. **Final balanced sample:** up to 13 examples/intent drawn from the cleaned pool -> **338 examples**, all 26 intents and all 9 categories represented.
8. **Split:** 80 / 10 / 10 (train/val/test), stratified per intent -> **286 / 26 / 26**.

### Before / after counts

| Stage | Rows |
|---|---|
| Source corpus (full) | 25,545 |
| Stratified candidate pool | 520 |
| After placeholder resolution (744 tokens resolved) | 520 |
| After length filter (empty/truncated removed) | 520 |
| After duplicate removal | 520 |
| Final balanced dataset (before split) | 338 |
| **Train** | **286** |
| **Val** | **26** |
| **Test** | **26** |

## 4. Split Justification
80/10/10, stratified by intent. With ~338 examples over 26 intents (~13/intent), a larger held-out split (e.g. 70/15/15) would leave too few training examples per intent for a 0.5-3B model to learn intent-specific phrasing; 80/10/10 keeps >=1 example of every intent in val and test (used respectively for Task 4 diagnosis and Task 5's automatic metric) while maximising LoRA training data, which is the most data-hungry stage of the pipeline.

## 5. Fields
| Field | Description |
|---|---|
| `instruction` | The user's request/question (free text, informal register preserved). |
| `input` | Optional context field (empty string here -- this domain's instructions are self-contained). |
| `output` | The expected assistant response (ground truth). |
| `category` | Bitext top-level category (9 values), retained as metadata for stratification. |
| `intent` | Bitext fine-grained intent (26 values), retained as metadata for stratification and diagnosis in Task 4. |

## 6. Statistics
- Instruction length: mean 11.1 words (min 5, max 20).
- Response length: mean 146.1 words (min 38, max 403).
- See `img/dataset_stats.png` (length histograms + per-category counts) and `img/split_distribution.png`.

## 7. Dataset Quality Risks (per PS2 Task 1 requirement)
This dataset carries three known risks that are revisited during Task 4 diagnosis:

1. **Template-driven repetitive phrasing.** Because the source is template + paraphrase generated rather than organically written, many responses within the same intent share near-identical openings ("I'm here to assist you with...") and a numbered-list structure. A model fine-tuned on this data may over-generalise this template even where a shorter answer is more appropriate -- a concrete risk flagged for the Task 4 diagnosis (over-long / repetitive responses).
2. **Informal/profane user register.** 54 of 338 examples (~16%) contain mild profanity or frustrated phrasing in the *instruction* (intentional in Bitext's design, to build robustness to real user tone). The *responses* remain professional throughout, so this should teach tone-de-escalation rather than harm the assistant's own register -- but it is worth monitoring during evaluation.
3. **Coverage gaps from small per-intent counts.** ~11 train examples/intent is enough to teach surface phrasing but not deep coverage of every edge case (e.g. joint accounts, multi-currency transfers) within an intent; the out-of-scope prompt in the frozen benchmark (Task 2) specifically probes what happens outside this coverage.

## 8. Sample Records
See `data/processed/sample_records.json` for 3 representative (profanity-filtered) examples spanning 3 different categories.

# PS2: Parameter-Efficient Fine-Tuning of a Small Language Model with Iterative Improvement
### BITS Pilani WILP M.Tech. AIML -- Conversational AI (AIMLCZG521) -- Assignment 2

**Name:** _______________  **BITS ID:** _______________  **Date:** _______________

> **How to use this template.** Every section below tells you exactly which notebook / results file
> it must be filled in from. Do not write content that isn't backed by an actual Virtual Lab run --
> PS2 explicitly penalises fabricated or cherry-picked outputs. Convert this file to PDF for
> submission (e.g. `pandoc Assignment-2-Report-Template.md -o report.pdf`, or paste into Word).

---

## 0. Domain & Model Choices
*(Source: `README.md` Section 1 -- copy and expand with 2-3 sentences of your own justification.)*

- **Domain:** NovaBank retail banking customer support. Why: ...
- **SLM:** `Qwen/Qwen2.5-1.5B-Instruct`. Why: ...
- **Reference LLM:** `Qwen/Qwen2.5-7B-Instruct`. Why: ...

---

## Task 1 -- Instruction Dataset Preparation (1.5 marks)
*(Source: `notebooks/01_dataset_preparation.ipynb`, `data/DATASET_CARD.md`, `img/dataset_stats.png`)*

- Source dataset, license, and disclosed generation method of the source: ...
- Before/after cleaning counts (paste the table from `DATASET_CARD.md` Section 3): ...
- Train/val/test split and justification: ...
- 3 sample records (from `data/processed/sample_records.json`): ...
- Dataset quality risks paragraph (from `DATASET_CARD.md` Section 7): ...
- [Insert `img/dataset_stats.png` and `img/split_distribution.png`]

## Task 2 -- Baseline Benchmarking: Base SLM vs. Reference LLM (1.5 marks)
*(Source: `notebooks/02_baseline_benchmarking.ipynb`, `results/task2_side_by_side.csv`)*

- The frozen 8-prompt benchmark (from `data/benchmark/frozen_benchmark_prompts.json`): ...
- All (A) and (B) verbatim outputs (table or appendix): ...
- Limitations / gap table (from Notebook 02's markdown section, filled in from real outputs): ...

## Task 3 -- Parameter-Efficient Fine-Tuning, Iteration 1 (2.5 marks, core)
*(Source: `notebooks/03_finetuning_v1.ipynb`, `results/task3_efficiency_summary.json`, `img/v1_training_loss.png`)*

- 1-paragraph intuition for why LoRA/QLoRA is parameter-efficient (your own words, tied to your
  measured numbers below -- see Notebook 03's Theory section for the underlying explanation to
  paraphrase, not copy verbatim): ...
- **Efficiency evidence (mandatory):**
  - Trainable parameters: _______ / Total parameters: _______ (____%)
  - Peak GPU memory during training: _______ MB
  - Why full fine-tuning would not fit the same budget (from `task3_efficiency_summary.json`): ...
- Training configuration table with 1-2 line justifications (copy from Notebook 03's markdown table,
  update if you changed any values): ...
- [Insert `img/v1_training_loss.png`]
- Saved model path: `models/novabank-slm-lora-v1/`
- Inference demonstration (paste the Notebook 03 demo Q/A): ...

## Task 4 -- Diagnose and Improve, Iteration 2 (2.5 marks, core)
*(Source: `notebooks/04_diagnose_improve_v2.ipynb`, `results/task4_*`, `img/v2_training_loss.png`)*

- Diagnosis with evidence (>=2 concrete weaknesses, with example v1 outputs): ...
- Change log (weakness -> change -> rationale table, from Notebook 04 Step 4): ...
- v2 training results: [Insert `img/v2_training_loss.png`], saved path `models/novabank-slm-lora-v2/`
- v1-vs-v2 before/after comparison (>=1 pair per diagnosed weakness, from Notebook 04 Step 8): ...
- Did anything get worse? Report honestly: ...

## Task 5 -- Three-Way Evaluation and Verdict (2 marks)
*(Source: `notebooks/05_evaluation_verdict.ipynb`, `results/task5_*`)*

- Three-way (A/B/C) side-by-side table (from `task5_three_way_comparison.csv`): ...
- Rubric disclosed (criteria + 0-2 scale) and per-system totals (from `task5_rubric_totals.csv`): ...
- Automatic metric (ROUGE-L F1, A vs. C, from `task5_automatic_metric.json`) + 1-paragraph limitation
  note: ...
- Error analysis: >=2 cases where C loses to B, each diagnosed as data / capacity / training problem: ...
- **Verdict paragraph:** given quality, cost, privacy, and hardware constraints, when would you deploy
  the fine-tuned SLM instead of the large LLM -- and when not? ...

---

## Appendix A -- Virtual Lab Screenshots
*(Insert full-screen screenshots from `screenshots/`, one per notebook run, showing the Virtual Lab
environment and the executing notebook.)*

## Appendix B -- References
1. Bitext Innovations International, S.L. *Retail Banking LLM Chatbot Training Dataset* (2024). Hugging
   Face Datasets.
2. Qwen Team, Alibaba Cloud. *Qwen2.5 Technical Report* (2024).
3. Hu, E. J., et al. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685 (2021).
4. Dettmers, T., et al. *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314 (2023).

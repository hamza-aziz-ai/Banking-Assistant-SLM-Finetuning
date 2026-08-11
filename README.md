# NovaBank SLM Fine-Tuning -- PS2 Solution
### BITS Pilani WILP M.Tech. AIML -- Conversational AI (AIMLCZG521) -- Assignment 2

**PS2: Parameter-Efficient Fine-Tuning of a Small Language Model with Iterative Improvement (10 marks)**

## 1. Domain & Model Choices (as required by PS2)

| Choice                                               | Value                                                           | Why                                                                                                                                                                                                                                                                                     |
|------------------------------------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Domain**                                           | Retail banking customer support (fictional bank **"NovaBank"**) | One of PS2's suggested domains; directly relevant to my work in conversational/voice AI, where customer-support automation is one of the most common production use cases for a fine-tuned, on-premise SLM -- matching PS2's cost/privacy/hardware framing.                             |
| **SLM (fine-tuning target)**                         | `Qwen/Qwen2.5-1.5B-Instruct`                                    | Apache-2.0, 1.5B params (within PS2's 0.5-3B range), strong instruction-following for its size, well-supported by `transformers`/`peft`/`bitsandbytes`, fits comfortably on "modest hardware."                                                                                          |
| **Reference LLM (inference-only, never fine-tuned)** | `Qwen/Qwen2.5-7B-Instruct` (4-bit inference)                    | Same model family as the SLM -- isolates the effect of *scale* rather than confounding it with a different pretraining corpus/tokenizer. Apache-2.0, ~4.7x larger, satisfying PS2's "7B+" requirement, and fits in ~5-6 GB VRAM at 4-bit for inference-only use on a single modest GPU. |

**Reference LLM alternatives** (if your Virtual Lab GPU cannot fit even the 4-bit 7B model): swap `REFERENCE_LLM_MODEL_ID` in `src/config.py` to a smaller-footprint quantized build (e.g. `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4`) or to a hosted/API model (e.g. via an OpenAI-compatible endpoint your Virtual Lab allows) -- if you do this, update the model-choice justification above accordingly in your report, since the assignment requires you to state and justify both choices.

## 2. Project Structure

```
Banking-Assistant-SLM-Finetuning/
├── README.md                       <- this file
├── Pipfile                          <- pipenv dependency spec (source of truth for packages)
├── Pipfile.lock                     <- pipenv lock file (exact resolved versions/hashes)
├── build_dataset.py                 <- standalone, reproducible Task 1 dataset build script
├── Problem-Statement.pdf            <- PS2 assignment brief
├── data/
│   ├── raw/                         <- source CSV + cleaned candidate pool (audit trail)
│   ├── processed/                   <- train.jsonl / val.jsonl / test.jsonl / dataset_stats.json
│   ├── benchmark/frozen_benchmark_prompts.json   <- the 8 frozen Task 2 prompts (reused in Tasks 4-5)
│   └── DATASET_CARD.md              <- Task 1 dataset documentation
├── notebooks/
│   ├── 01_dataset_preparation.ipynb        <- Task 1 (already executed -- see note below)
│   ├── 02_baseline_benchmarking.ipynb      <- Task 2 (run in Virtual Lab)
│   ├── 03_finetuning_v1.ipynb              <- Task 3 (run in Virtual Lab)
│   ├── 04_diagnose_improve_v2.ipynb        <- Task 4 (run in Virtual Lab)
│   └── 05_evaluation_verdict.ipynb         <- Task 5 (run in Virtual Lab)
├── src/                              <- shared, documented utility modules imported by every notebook
│   ├── config.py                     <- model IDs, paths, generation defaults, seed
│   ├── data_utils.py                 <- JSONL loading, chat-template prompt formatting
│   ├── model_utils.py                <- model loading (incl. 4-bit), generation, param/memory counting
│   ├── train_utils.py                <- LoRA config, dataset tokenisation, loss-curve plotting
│   └── eval_utils.py                 <- rubric scaffolding, ROUGE-L, JSON helpers
├── models/                           <- LoRA adapters saved here (novabank-slm-lora-v1/, -v2/)
├── results/                          <- JSON/CSV outputs from Tasks 2-5 (responses, scores, metrics)
├── img/                              <- dataset stats, loss curves
├── report/                           <- report template + submission checklist
└── screenshots/                      <- put your Virtual Lab full-screen screenshots here
```

## 3. What Has Already Been Run vs. What YOU Must Run in the Virtual Lab

PS2 requires the assignment to be **executed strictly in the BITS Virtual Lab environment**, with full-screen screenshots as proof. This project separates cleanly along a GPU/no-GPU line:

- **`notebooks/01_dataset_preparation.ipynb` has already been fully executed** (with real, visible cell outputs; also exported to `.py` and `.pdf` in the same folder) -- it is pure CPU data engineering (downloading + cleaning a public dataset), so it carries no compute dependency on the Virtual Lab. It is fully deterministic (`SEED=42`): **re-running it in the Virtual Lab reproduces byte-identical`train.jsonl`/`val.jsonl`/`test.jsonl` files**, and doing so is recommended so you have your own Virtual Lab screenshot for Task 1 too.
- **`notebooks/02_baseline_benchmarking.ipynb` through `05_evaluation_verdict.ipynb` are complete, syntax-validated, ready-to-run notebooks that have deliberately NOT been executed here** -- they load and run actual language models (up to 7B parameters), which needs GPU compute this authoring environment does not have, and which PS2 requires to happen in the Virtual Lab anyway. **You must run these four notebooks, in order, inside the Virtual Lab**, take full-screen screenshots while they run, and fill in the marked `*(fill in after running)*` markdown sections (diagnosis evidence, rubric scores, error analysis, verdict) from the real outputs you see.

## 4. How to Run in the Virtual Lab

1. Copy this whole folder into your Virtual Lab workspace.
2. Install [pipenv](https://pipenv.pypa.io) if it isn't already available (`pip install pipenv`), then from the project root run `pipenv install` -- this reads `Pipfile.lock` and recreates the exact resolved environment (including the CUDA-specific PyTorch build pinned via the `downloadpytorch` source in `Pipfile`). If your Virtual Lab's CUDA version doesn't match that index, edit the `[[source]]` block in `Pipfile` to the matching `download.pytorch.org/whl/cuXXX` URL before installing, then re-run `pipenv lock && pipenv install`.
   - Register the environment as a Jupyter kernel so the notebooks can use it: `pipenv run python -m ipykernel install --user --name novabank-slm --display-name "Python (novabank-slm)"`, then select that kernel when opening each notebook.
   - Run any command inside the environment with `pipenv run <command>` (e.g. `pipenv run jupyter notebook`), or drop into an environment shell with `pipenv shell`.
3. Run `notebooks/01_dataset_preparation.ipynb` top-to-bottom (Run All) -- confirms your environment reproduces the Task 1 dataset (338 examples; see `data/processed/dataset_stats.json`).
4. Run `notebooks/02_baseline_benchmarking.ipynb` top-to-bottom. Produces `results/task2_*`.
5. Run `notebooks/03_finetuning_v1.ipynb` top-to-bottom. Produces `models/novabank-slm-lora-v1/`,`img/v1_training_loss.png`, `results/task3_*`.
6. Run `notebooks/04_diagnose_improve_v2.ipynb` top-to-bottom. **Read Step 1-2's real output before writing Step 3's weakness statement** (the pre-filled hypotheses must be confirmed or corrected against what you actually see). Produces `models/novabank-slm-lora-v2/`,`img/v2_training_loss.png`, `results/task4_*`.
7. Run `notebooks/05_evaluation_verdict.ipynb` top-to-bottom, fill in the rubric scores and the verdict paragraph from the real three-way outputs. Produces `results/task5_*`.
8. Export each notebook to `.py` (`pipenv run jupyter nbconvert --to script <nb>`) and `.pdf` (`pipenv run jupyter nbconvert --to pdf <nb>`), per PS2's submission format.
9. Take full-screen screenshots of your Virtual Lab session while the notebooks run; save them under `screenshots/`.
10. Assemble the final report (`report/Assignment-2-Report-Template.md`) from every notebook's Markdown "fill in" sections, convert to PDF, and place it alongside this folder in your submission zip per PS2's mandatory submission format (Section "General Instructions", item 2).

## 5. Dependency Management

This project uses **pipenv** (`Pipfile` / `Pipfile.lock`). `Pipfile` lists the actual runtime
dependencies with version floors: the notebook/export stack (`jupyter`, `nbconvert`, `ipykernel`),
the ML stack (`torch` via the pinned CUDA index, `transformers`, `accelerate`, `peft`,
`bitsandbytes`, `datasets`), data/eval tooling (`pandas`, `numpy`, `huggingface-hub`, `matplotlib`,
`scikit-learn`, `bert-score`), and `nbformat` as the sole `[dev-packages]` entry (notebook
validation only, not needed to run the pipeline).

> **`Pipfile.lock` was regenerated on this project's own machine, not in this authoring
> environment** (which has no `pipenv` installed and isn't the target GPU platform). If you edit
> `Pipfile` -- e.g. to change the CUDA index for a different Virtual Lab GPU -- run `pipenv lock`
> yourself afterwards to keep `Pipfile.lock` in sync; a stale lock file will cause `pipenv install`
> to either fail or silently install different versions than what's declared.

## 6. Reproducibility

- A single seed (`SEED = 42`, in `src/config.py`, and used consistently across `build_dataset.py` and every notebook) controls sampling, shuffling, splitting, and model training (`torch.manual_seed`, `TrainingArguments(seed=..., data_seed=...)`).
- **Known Python gotcha avoided:** the dataset build originally used Python's built-in `hash()` for a pseudo-ID placeholder resolver, which is randomized per-process (`PYTHONHASHSEED`) and silently broke reproducibility between runs. This was replaced with `hashlib.md5`-based hashing, which is stable across processes/machines -- verified by running `build_dataset.py` twice and diffing the output (byte-identical). If you extend this pipeline, prefer `hashlib` over `hash()` for anything that needs to reproduce outside the current Python process.
- Exact model checkpoints used: `Qwen/Qwen2.5-1.5B-Instruct` and `Qwen/Qwen2.5-7B-Instruct` (record the specific Hugging Face commit/revision hash in your report if you want byte-level reproducibility, since the "latest" tag can move).

## 7. Citations

- Dataset: Bitext Innovations International, S.L. *Retail Banking LLM Chatbot Training Dataset* (2024). Hugging Face Datasets. `bitext/Bitext-retail-banking-llm-chatbot-training-dataset`. Licence: CDLA-Sharing-1.0.
- Models: Qwen Team, Alibaba Cloud. *Qwen2.5 Technical Report* (2024). `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`. Licence: Apache-2.0.
- Method: Hu, E. et al. *LoRA: Low-Rank Adaptation of Large Language Models* (2021). Dettmers, T. et al. *QLoRA: Efficient Fine-tuning of Quantized LLMs* (2023).

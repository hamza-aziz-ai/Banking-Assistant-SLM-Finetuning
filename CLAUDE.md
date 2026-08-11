# CLAUDE.md

This file gives Claude Code (or any agent working in this repo) the context needed to work correctly and safely on this project. Read this before editing code, running notebooks, or making claims about what has/hasn't been done.

## What this project is

Solution to **PS2: Parameter-Efficient Fine-Tuning of a Small Language Model with Iterative Improvement**, an individual assignment for BITS Pilani WILP M.Tech. AIML, course *Conversational AI (AIMLCZG521)*, Assignment 2 (10 marks). The problem statement is `Problem-Statement.pdf` at the repo root — read it before changing scope, task boundaries, or marks-relevant deliverables.

The assignment's central question: can a small language model (SLM), parameter-efficiently fine-tuned on ~300 domain examples, close the quality gap to a much larger pretrained reference LLM on domain tasks? This repo builds a **NovaBank retail-banking customer-support assistant** to test that question end-to-end: dataset → baseline benchmark → LoRA fine-tune (v1) → diagnose & improve (v2) → three-way evaluation and verdict.

## Hard constraints from the problem statement — do not violate these

1. **Model training/inference (Tasks 2-5) must be executed in the BITS Virtual Lab environment, with full-screen screenshots as proof.** This authoring environment has no GPU. Do not present sandbox-generated model outputs, loss curves, or scores as if they came from a real run — that would misrepresent the submission. Notebooks 02-05 are deliberately left **unexecuted** here; only Notebook 01 (pure CPU data engineering) has been actually run, with real outputs.
2. **Never fabricate or cherry-pick model outputs, scores, or evaluation results.** Every `*(fill in after running)*` marker in the notebooks and `report/Assignment-2-Report-Template.md` must be filled from an actual executed run, not invented. This is graded explicitly (PS2: "Copied content or unexplained AI-generated answers will be penalized... Don't fabricate or cherry-pick outputs").
3. **The reference LLM (`Qwen/Qwen2.5-7B-Instruct`) is inference-only — never fine-tune it.** Only the SLM (`Qwen/Qwen2.5-1.5B-Instruct`) gets LoRA adapters trained against it.
4. **The frozen 8-prompt benchmark (`data/benchmark/frozen_benchmark_prompts.json`) must not be edited, reordered, or regenerated once Task 2 has been run against it.** It is reused verbatim in Notebooks 04 and 05 specifically so results are comparable across iterations. If a prompt genuinely needs to change, that invalidates prior benchmark runs — flag this to the user rather than silently editing the file.
5. **Every non-default config value (LoRA rank, learning rate, epochs, etc.) needs a 1-2 line justification**, already provided in Notebook 03/04's Markdown tables — keep these in sync if hyperparameters change.
6. **Task 4 changes exactly one or two things at a time**, each explicitly linked to a diagnosed weakness (`weakness -> change -> rationale`). Do not bundle multiple untested changes into v2; PS2 penalizes this because it makes the cause of any improvement/regression unattributable.

## Repo structure

```md
README.md                        ← start here for domain/model choices + how to run in the VL
Problem-Statement.pdf             ← the assignment brief (source of truth for requirements/marks)
build_dataset.py                  ← standalone, reproducible Task 1 dataset build (CLI-runnable)
Pipfile                           ← pipenv dependency spec (edit this, not a requirements.txt)
Pipfile.lock                      ← pipenv lock file (resolved on the user's own machine, not here)
data/
  raw/                             ← source CSV + cleaned candidate pool (audit trail)
  processed/                       ← train.jsonl (286) / val.jsonl (26) / test.jsonl (26) + stats
  benchmark/frozen_benchmark_prompts.json   ← FROZEN, see constraint #4 above
  DATASET_CARD.md                  ← Task 1 documentation (source, licence, cleaning, risks)
notebooks/
  01_dataset_preparation.ipynb     ← EXECUTED (real outputs). Also exported .py + .pdf.
  02_baseline_benchmarking.ipynb   ← NOT executed. Run in Virtual Lab.
  03_finetuning_v1.ipynb           ← NOT executed. Run in Virtual Lab. Produces v1 adapter.
  04_diagnose_improve_v2.ipynb     ← NOT executed. Run in Virtual Lab. Produces v2 adapter.
  05_evaluation_verdict.ipynb      ← NOT executed. Run in Virtual Lab. Final A/B/C verdict.
src/
  config.py                        ← SEED, model IDs, paths, generation defaults — single source of truth
  data_utils.py                    ← JSONL I/O, chat-template prompt formatting
  model_utils.py                   ← model loading (incl. 4-bit), generation, param/memory counting
  train_utils.py                   ← LoRA config, dataset tokenization+masking, loss-curve plotting
  eval_utils.py                    ← rubric scaffolding, ROUGE-L, JSON helpers
models/                            ← LoRA adapters land here: novabank-slm-lora-v1/, -v2/ (empty until VL run)
results/                           ← JSON/CSV outputs from Tasks 2-5 (empty until VL run)
img/                               ← dataset_stats.png, split_distribution.png (done); loss curves (pending)
report/
  Assignment-2-Report-Template.md  ← fill in from real notebook outputs, export to PDF for submission
  SUBMISSION_CHECKLIST.md          ← required-files checklist per PS2's submission format
screenshots/                       ← put Virtual Lab full-screen screenshots here
```

Every notebook imports shared logic from `src/` — put new reusable logic there, not copy-pasted
across notebooks.

## Domain & model choices (do not change without updating README + report)

| Choice                 | Value                                             | Why                                                                                                                  |
|------------------------|---------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| Domain                 | Retail banking support, fictional bank "NovaBank" | PS2-suggested domain; matches the user's conversational/voice-AI production background                               |
| SLM (fine-tuned)       | `Qwen/Qwen2.5-1.5B-Instruct`                      | Apache-2.0, in PS2's 0.5-3B range, strong instruction-following, well supported by transformers/peft/bitsandbytes    |
| Reference LLM (frozen) | `Qwen/Qwen2.5-7B-Instruct`, 4-bit inference       | Same family as SLM (isolates scale effect), Apache-2.0, satisfies PS2's "7B+" requirement, fits ~5-6GB VRAM at 4-bit |

Source dataset: `bitext/Bitext-retail-banking-llm-chatbot-training-dataset` (Hugging Face, licence CDLA-Sharing-1.0) — see `data/DATASET_CARD.md` for full provenance and the cleaning pipeline.

## Reproducibility rules

- **`SEED = 42`** everywhere (`src/config.py`), threaded through `random`, `numpy`, `torch`, and every `TrainingArguments(seed=..., data_seed=...)` call. Any new random draw must seed from this same constant, not a new literal.
- **Never use Python's built-in `hash()` for anything that needs to reproduce across processes.** It's randomized per-process via `PYTHONHASHSEED` by default. This exact bug broke `build_dataset.py`'s placeholder-ID generator earlier in this project (silently dropped the dataset from 338 to ~218 rows between runs) and was fixed by switching to `hashlib.md5(...).hexdigest()`. If you add similar deterministic-ID logic, use `hashlib`, not `hash()`.
- After changing `build_dataset.py` or the Notebook 01 pipeline, verify determinism by running it twice and diffing `data/processed/dataset_stats.json` (should be byte-identical).

## A known class of bug to watch for: nested raw-string escaping

Several notebooks in this repo are generated programmatically (via `nbformat`, from small builder scripts). If you ever regenerate a notebook cell containing a **regex or other backslash-heavy string** through a Python string-building step, double-check the actual executed content — nesting an `r'''...'''` cell-source template inside another raw string can silently double every backslash (e.g. `\{\{` becomes `\\{\\{`, which as a compiled regex matches literal backslash-brace sequences instead of `{{`). This previously caused `PLACEHOLDER_RE` to silently never match anything, which then caused a *downstream* safety-check filter to wrongly drop every row containing a placeholder. **If you regenerate any notebook cell text programmatically, execute it and inspect real output — don't just check that it parses as valid Python.** `repr()` on the cell's `source` string (not on the raw `.ipynb` JSON text, which double-escapes again) is the reliable way to see actual backslash counts when debugging this.

## Notebook conventions (apply to any new/edited notebook)

Per this project's standing conventions: explanatory Markdown before every important code cell; one function/class/experiment step per code cell; every function/class has a docstring covering purpose, math (if any), args, returns, and assumptions; comments explain *what/why/how* and tie back to the underlying theory, not just restate the code. Use the layered explanation structure for difficult concepts: intuition → formal definition/math → small example → engineering relevance → exam/interview angle (see Notebook 03's LoRA theory section for a worked example).

Notebook flow standard: title/metadata → problem understanding → theory → imports/setup → data → utility functions → core implementation → training/inference → evaluation/results → visualizations → observations → conclusion → (limitations, where relevant).

## Dependency management: pipenv, not pip/requirements.txt

This project uses **pipenv** (`Pipfile` / `Pipfile.lock`), not a `requirements.txt`. Rules:

- Add/remove dependencies via `pipenv install <pkg>` / `pipenv uninstall <pkg>` (or by editing `Pipfile` directly for version-range changes), never by hand-editing `Pipfile.lock`.
- `Pipfile` pins `torch` to a CUDA-specific index (`[[source]] name = "downloadpytorch"`, currently `download.pytorch.org/whl/cu130`) matching the machine it was last locked on. If the Virtual Lab's CUDA version differs, that source URL needs updating before `pipenv install` will pull a working `torch` build there.
- `Pipfile.lock` **must be regenerated with `pipenv lock` after any `Pipfile` edit** — this sandbox has no `pipenv` installed and isn't the target GPU platform, so lock files produced here would be unreliable; regeneration must happen on a real target machine (the user's own, or the Virtual Lab). Never hand-edit or fabricate `Pipfile.lock` contents.
- `[packages]` = actual runtime deps (notebook stack + ML stack + data/eval tooling, since the notebooks themselves are the deliverable). `[dev-packages]` = tooling not needed to run the pipeline (currently just `nbformat`, for notebook-structure validation).
- If you add a new import to `src/` or a notebook, add the corresponding package to `Pipfile`'s `[packages]` in the same change — don't leave an undeclared dependency for the user to discover via an `ImportError` in the Virtual Lab.

## Common commands

```bash
# Install the locked environment (run once per machine):
pipenv install

# Register it as a Jupyter kernel so notebooks can select it:
pipenv run python -m ipykernel install --user --name novabank-slm --display-name "Python (novabank-slm)"

# Reproduce the Task 1 dataset from scratch (no GPU needed):
pipenv run python build_dataset.py

# Execute a notebook in place (adds real outputs):
pipenv run jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb --ExecutePreprocessor.timeout=600

# Export a notebook to the two required submission formats:
pipenv run jupyter nbconvert --to script notebooks/<name>.ipynb --output-dir notebooks
pipenv run jupyter nbconvert --to pdf notebooks/<name>.ipynb --output-dir notebooks

# Validate a notebook's structure without executing it:
pipenv run python -c "import nbformat; nb=nbformat.read('notebooks/<name>.ipynb', as_version=4); nbformat.validate(nb)"

# After editing Pipfile, refresh the lock file:
pipenv lock
```

## Current state (update this section as work progresses)

- [x] Task 1 — dataset built, cleaned, split, executed with real outputs; exported to `.py`/`.pdf`.
- [ ] Task 2 — notebook written and syntax-validated; **not yet run** in the Virtual Lab.
- [ ] Task 3 — notebook written and syntax-validated; **not yet run**. No adapter exists in `models/` yet.
- [ ] Task 4 — notebook written and syntax-validated; **not yet run**. Depends on Task 3's v1 adapter.
- [ ] Task 5 — notebook written and syntax-validated; **not yet run**. Depends on Task 4's v2 adapter.
- [ ] Report — template exists in `report/`; needs real content from the above runs before export.
- [ ] Screenshots — `screenshots/` folder exists and is empty; needs Virtual Lab captures.

If you're picking this project back up: check which `results/task*.json` and `models/*` files exist to see what's actually been run before assuming the state above is still current.

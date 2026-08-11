# PS2 Submission Checklist

Per the Problem Statement's "General Instructions" (Section 2 -- mandatory submission format):

## Files that must be in your final submission folder

- [ ] **PDF report** with all results, explanations, justifications, inferences
      (`report/Assignment-2-Report-Template.md`, filled in and exported to PDF)
- [ ] For each of the 5 notebooks:
  - [ ] `.ipynb` with **every executed cell's output visible**
  - [ ] converted `.py` file (`jupyter nbconvert --to script`)
  - [ ] exported PDF of the executed notebook (`jupyter nbconvert --to pdf`)
- [ ] Saved fine-tuned adapters: `models/novabank-slm-lora-v1/`, `models/novabank-slm-lora-v2/`
      (or a download link if size-restricted -- document the link + path in the report)
- [ ] **Full-screen screenshots** of the Virtual Lab environment while executing (`screenshots/`)

## Per-task self-check (does the report actually contain...)

- [ ] Task 1: >=300 examples, before/after cleaning counts, split + justification, 3 samples, quality-risk note
- [ ] Task 2: the frozen 8 prompts, verbatim A + B outputs, the limitations/gap table
- [ ] Task 3: LoRA/QLoRA intuition (<=1 page), trainable-param count/%, peak memory, config + justifications, loss curves, saved model, inference demo
- [ ] Task 4: >=2 diagnosed weaknesses with evidence, change log (weakness->change->rationale), v2 loss curves + saved path, v1-vs-v2 before/after (>=1 pair per weakness)
- [ ] Task 5: 3-way comparison table, disclosed rubric + per-system totals, >=1 automatic metric with limitations, >=2 honest error cases, verdict paragraph

## Integrity / reproducibility checks

- [ ] Random seed fixed and reported (SEED=42 throughout)
- [ ] Exact model checkpoints/versions recorded (`Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`)
- [ ] Every non-default config value (adapter rank, LR, epochs, etc.) has a 1-2 line justification
- [ ] All datasets/models/papers/docs cited (see README.md Section 6 / report Appendix B)
- [ ] No fabricated or cherry-picked outputs -- every quoted model response actually came from a run

# PS2 Submission Checklist

Per the Problem Statement's "General Instructions" (Section 2 -- mandatory submission format):

## Files that must be in your final submission folder

- [x] **PDF report** with all results, explanations, justifications, inferences
      (`report/Assignment-2-Report-Template.md`, filled in and exported to `report/Assignment-2-Report.pdf`)
- [x] For each of the 5 notebooks:
  - [x] `.ipynb` with **every executed cell's output visible**
  - [x] converted `.py` file (`jupyter nbconvert --to script`)
  - [x] exported PDF of the executed notebook (`jupyter nbconvert --to pdf` -- via HTML+Playwright locally, pandoc/LaTeX unavailable)
- [x] Saved fine-tuned adapters: `models/novabank-slm-lora-v1/`, `models/novabank-slm-lora-v2/`
      (or a download link if size-restricted -- document the link + path in the report)
- [x] **Full-screen screenshot**: Virtual Lab's no-GPU blocker, `screenshots/vl_no_gpu_confirmation.png`

## Per-task self-check (does the report actually contain...)

- [x] Task 1: >=300 examples, before/after cleaning counts, split + justification, 3 samples, quality-risk note
- [x] Task 2: the frozen 8 prompts, verbatim A + B outputs, the limitations/gap table
- [x] Task 3: LoRA/QLoRA intuition (<=1 page), trainable-param count/%, peak memory, config + justifications, loss curves, saved model, inference demo
- [x] Task 4: >=2 diagnosed weaknesses with evidence, change log (weakness->change->rationale), v2 loss curves + saved path, v1-vs-v2 before/after (>=1 pair per weakness)
- [x] Task 5: 3-way comparison table, disclosed rubric + per-system totals, >=1 automatic metric with limitations, >=2 honest error cases, verdict paragraph

## Integrity / reproducibility checks

- [x] Random seed fixed and reported (SEED=42 throughout)
- [x] Exact model checkpoints/versions recorded (`Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`)
- [x] Every non-default config value (adapter rank, LR, epochs, etc.) has a 1-2 line justification
- [x] All datasets/models/papers/docs cited (see README.md Section 6 / report Appendix B)
- [x] No fabricated or cherry-picked outputs -- every quoted model response actually came from a run
- [x] Group / member BITS IDs on the report cover (Group 174, 4 members)
- [x] Report Appendix C (execution-environment deviation) disclosed, with real Virtual Lab screenshot
      (`screenshots/vl_no_gpu_confirmation.png`, 2026-08-12 09:01, host `2024ac05133`) embedded in Appendix A

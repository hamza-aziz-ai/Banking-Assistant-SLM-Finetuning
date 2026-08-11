# Two things left, both need you

Everything automatable is done, committed, and pushed
(https://github.com/hamza-aziz-ai/Banking-Assistant-SLM-Finetuning). All 5 notebooks ran for real
locally (RTX 4050), report is filled from real results, PDFs exported. See `CLAUDE.md` "Current
state" for the full rundown of what happened and why (Virtual Lab had no GPU + a full disk).

1. **BITS ID** -- `report/Assignment-2-Report-Template.md` line 4. Not derivable from anything in
   this repo/session, left as an explicit marker.
2. **Virtual Lab screenshots** -- full-screen captures of the no-GPU (`lspci`/`nvidia-smi`) and
   full-disk (`df -h`) state you already showed me in chat, taken *on* the Virtual Lab itself. Drop
   them in `screenshots/`, reference them in report Appendix C (already written, just needs the
   images).

Then: re-export the report to PDF (`pipenv run python <the same playwright script approach used
throughout this session>` or just paste the filled `.md` into Word/Google Docs and export -- no
special tooling required once the two fields above are filled), delete this file, commit, push.

Delete `WAKE_UP_TODO.md` once both are done -- it's a session note, not a deliverable.

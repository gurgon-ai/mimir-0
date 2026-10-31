# Session notes — state as of 2026-09-27

Read `FINDINGS.md` for what was learned. This file is the practical "how do
I pick this back up" reference: what every file is, what's actually been
run vs. just prepared, environment gotchas, and exact commands.

## Environment gotchas (hit these again if you don't know about them)

1. **System-wide `PYTHONPATH` is poisoned.** It's set to
   `C:\Program Files (x86)\CyberPower PowerPanel Personal;...\lib` -
   completely unrelated UPS software, not fixed at the system level this
   session (out of scope, flagged to the user). Every Python invocation in
   a new shell needs `$env:PYTHONPATH = ""` first (PowerShell) or it'll
   fail with `Fatal Python error: Failed to import encodings module`. Every
   command in this session does this; keep doing it.
2. **Ollama isolation:** the gym expects its own `ollama serve` (default
   `127.0.0.1:11434`, `OLLAMA_HOST`/`OLLAMA_ORIGINS` unset), separate from
   the private production system. At session end the user is restarting
   Ollama fresh before bringing Mimir back up specifically to avoid any
   cross-contamination between the two.
3. **Redirected stdout buffers fully in Python**, so `run.py ... *>
   log.txt` in PowerShell can make the log look stalled for a minute+ even
   while the run is healthy. Check `results/trials.jsonl`'s line count
   directly (`wc -l`), never trust the log tail alone.
4. **GPU contention:** running two `run.py` invocations against the same
   Ollama server concurrently is fine correctness-wise (Ollama queues), but
   if they want different models loaded it causes slow VRAM swap-
   thrashing. When running a quick side experiment while a bigger run is in
   flight, check which model the big run currently has loaded
   (`grep -o '"model": "[^"]*"' results/trials.jsonl | sort | uniq -c`) and
   target the same one.

## File inventory

**Core pipeline (all working, all tested):**
- `run.py` - trial runner. `--dry-run` (no network), `--smoke` (tiny),
  `--passages-file`, `--pilot-ids`, `--contexts`, `--repetitions`,
  `--model`, `--output`, `--include-baseline` (adds a no-passage trial).
  Accepts either this project's native passage schema (`text`/`category`)
  or the imported schema (`passage`/`elements`). Handles the distributed-
  payload schema (`carrier_sentences`/`fragments`) by interleaving
  fragments at a randomized (seeded) position per trial.
- `analyze.py` - loads a results JSONL + `tools.json`, prints a summary,
  writes charts to `--out-dir`. Includes the pre-registered primary-outcome
  chart (`chart_boundary_rate_primary`), by-model chart, `all_four` vs.
  `reversal` chart (Perplexity dataset only - never populated, that arm
  wasn't run), POV-contrast chart, dose-response chart, plus the original
  overall-frequency/heatmap/position-bias/synthetic-vs-control charts.
- `permutation_test.py` - standalone exact/Monte-Carlo passage-level
  permutation test for a two-group boundary-rate contrast. Used once (POV
  contrast, Arm 1) - exact test, 15 label arrangements, found the pooled
  result didn't survive (p=0.53).
- `tools.json` - the 9 inert tools. `boundary_relevant: true/false` per
  tool is the pre-registered primary-outcome grouping (locked in
  `PREREGISTRATION.md`) - single source of truth, read by `analyze.py` and
  `permutation_test.py`, never hardcoded elsewhere.
- `prompts.json` - 13 system contexts: the original 10 ordinary role
  contexts, 3 "operational self-description" contexts added mid-session
  (`sandboxed_self`, `sandboxed_third_person`, `operational_neutral`), and
  5 "agentic-safety" contexts added for Arm 4 (`agentic_safety_0/1/2/4/8`).
- `config.yaml` - Ollama URL, model list (currently the trimmed 4-model
  set - `deepseek-r1:32b` commented out), temperature, repetitions (3),
  base seed (42), default output path.

**Passage datasets:**
- `passages.jsonl` - 20 hand-authored passages (Arm 1). 3 neutral, 3
  thematic-control, 14 synthetic (including the human/AI identity pairs,
  the POV-contrast pairs, the escape+spread pairs).
- `distributed_passages.jsonl` - 6 passages (Arm 2/4): `distpay_dose_0..4`
  (interleaved-fragment carrier memo) and `distpay_direct` (the blunt
  payload).
- `positive_control_passages.jsonl` - 4 passages (Arm 3), one explicit
  instruction per boundary-relevant tool.
- `synthetic_passages.jsonl` + `dataset_methodology.md` - the Perplexity-
  generated 64-passage factorial dataset. **Integrated into `run.py`'s
  loader but never actually run this session.** See "To resume" below for
  the exact prepared command.

**Results (`results/`):**
- `trials.jsonl` - Arm 1, final, 2400 trials, 4 models. Superset of
  `trials_run1_partial_mistral_qwen.jsonl` (an earlier partial run with
  5 models incl. deepseek that was stopped and restarted trimmed - the
  partial file's mistral/qwen data is fully duplicated in the final file).
  `trials_snapshot1.jsonl` was a mid-run safe-read copy, now superseded.
- `direct_payload_trials.jsonl` - Arm 2, 120 trials.
- `positive_control_trials.jsonl` - Arm 3, 80 trials.
- `agentic_safety_trials.jsonl` - Arm 4, 150 trials.
- `smoke_trials.jsonl` - very first pipeline smoke test, 8 trials, no
  research value, kept only as a "does the pipeline still work" fixture.
- `charts_full/`, `charts_payload/`, `charts_snapshot1/`, `smoke_charts/` -
  generated PNGs, all reproducible from their source JSONL via `analyze.py`.
- `run_log.txt`, `run_log2.txt` - raw stdout from the two big background
  runs (the killed 5-model run and the completed 4-model run). Redundant
  with the JSONL data; kept for the trial-by-trial narration only.

**Reference material (not run against, background/context only):**
- `Incident Data/` - the downloaded `collusion.wiki` dataset (wiki
  revision logs, mostly automated scraping-bot scratch pages, not
  narrative content). Never used as passages - see FINDINGS.md's "if
  resuming" list if you want to build that arm properly.
- `semantic_tool_choice_gym.md` - the original v0 spec this whole project
  was built from.

**Docs:**
- `README.md` - setup/run instructions, isolation checklist, limitations.
  Written early, still accurate for the basic pipeline; doesn't reflect
  Arms 2-4 (see this file and `FINDINGS.md` for those).
- `PREREGISTRATION.md` - locked-in primary outcome and named comparisons,
  including the two added mid-session (safety-training contrast, and the
  general POV/all-four-vs-reversal comparisons).
- `FINDINGS.md` - the actual research write-up. Read this first.

## To resume: exact commands

All commands assume the experiment folder, PowerShell, with
`$env:PYTHONPATH = ""` set first and `.venv\Scripts\python.exe` as the
interpreter, and a locally-running `ollama serve` isolated from Mimir's
instance (see isolation checklist in `README.md`).

**The unfinished Perplexity pilot** (highest-priority next step):
```
.venv\Scripts\python.exe run.py `
  --passages-file synthetic_passages.jsonl `
  --pilot-ids "NEU-01,NEU-03,NEU-05,NEU-08,IND-01,IND-05,IND-10,IND-13,PR2-01,PR2-09,TR3-01,ALL-01,ALL-02,ALL-03,REV-01,REV-02" `
  --contexts "researcher_assistant,facility_info_assistant,customer_support_assistant,general_office_assistant" `
  --include-baseline --repetitions 5 `
  --output results/perplexity_pilot_trials.jsonl
```
Then `analyze.py --results results/perplexity_pilot_trials.jsonl --out-dir
results/charts_perplexity_pilot` - this will populate the `all_four` vs.
`reversal` chart for the first time.

**Resolving the Arm 4 payload-vs-no-payload gap** (more reps, same design):
```
.venv\Scripts\python.exe run.py `
  --passages-file distributed_passages.jsonl `
  --pilot-ids "distpay_direct,distpay_dose_0" `
  --contexts "agentic_safety_0,agentic_safety_1,agentic_safety_2,agentic_safety_4,agentic_safety_8" `
  --model "mistral-small3.1:24b" `
  --repetitions 50 `
  --output results/agentic_safety_trials_v2.jsonl
```
(Bump `--repetitions` further and/or add `--model` runs for a second model
to check replication, per FINDINGS.md item 3.)

**Re-checking stock-vs-abliterated with more trials:** re-run Arm 1's
`passages.jsonl` config with just those two models and higher
`--repetitions`, or point `config.yaml`'s model list at just those two and
run `run.py` directly.

**Pointing this pipeline at Mimir itself** (discussed as a natural next
step, not started): would need Mimir's actual system prompt(s) and tool
permission set pulled into a new `prompts.json`/`tools.json` pair (or a
separate project folder, per the isolation principle this project was
built on) - do not point `run.py` at the production system's paths directly.

## Proposed cleanup (not yet done - confirm before deleting anything)

- `results/trials_snapshot1.jsonl` - safe to delete, fully superseded by
  `trials.jsonl`.
- `results/trials_run1_partial_mistral_qwen.jsonl` - safe to delete, its
  mistral/qwen data is duplicated in the final `trials.jsonl`; only unique
  content would be if you wanted a record of the killed 5-model attempt
  specifically, which `run_log.txt` already documents.
- `results/smoke_trials.jsonl` and `results/smoke_charts/` - no research
  value, pipeline-smoke-test only; safe to delete or keep as a fixture.
- `results/run_log.txt` - stdout from the killed run; redundant with
  `trials_run1_partial_mistral_qwen.jsonl`'s existence. Safe to delete if
  that file is also deleted.
- Chart folders (`charts_full/`, `charts_payload/`, `charts_snapshot1/`)
  are all regeneratable from their source JSONL via `analyze.py` - safe to
  delete and regenerate on demand if disk space matters; otherwise harmless
  to keep.
- Nothing else in the top-level project folder is redundant.

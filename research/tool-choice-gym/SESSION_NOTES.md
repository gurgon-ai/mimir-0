# Session notes

State as of 2026-09-27, when the trials stopped. `FINDINGS.md` has the results; this file is the practical
reference for rerunning or extending the experiment: what each file is, what was run versus only
prepared, and the exact commands.

## Gotchas

1. **Use a dedicated Ollama instance.** The harness talks to `127.0.0.1:11434` by default. Run a fresh
   `ollama serve` with `OLLAMA_HOST` and `OLLAMA_ORIGINS` unset, and nothing else using it, so no other
   application's models, prompts or keep-alive settings affect the trials.
2. **A broken system-wide `PYTHONPATH` breaks every interpreter.** If Python fails at startup with
   `Fatal Python error: Failed to import encodings module`, some installed software has set `PYTHONPATH`
   globally. Clear it for the shell (`$env:PYTHONPATH = ""` in PowerShell) before running anything.
3. **Redirected stdout is fully buffered.** With `run.py ... > log.txt` the log can look stalled for over
   a minute while the run is healthy. Count lines in the results JSONL instead of watching the log.
4. **Two runs needing different models will thrash VRAM.** Concurrent `run.py` processes against one
   Ollama server are correct (requests queue), but if they want different models loaded, each swap
   reloads a model. For a side run, target the model the main run currently has loaded:
   `grep -o '"model": "[^"]*"' results/trials.jsonl | sort | uniq -c`.

## Files

**Pipeline**
- `run.py`: trial runner. Flags: `--dry-run` (no network), `--smoke` (tiny), `--passages-file`,
  `--pilot-ids`, `--contexts`, `--repetitions`, `--model`, `--output`, `--include-baseline` (adds a
  no-passage trial). Reads either passage schema (`text`/`category`, or `passage`/`elements`). For the
  distributed-payload schema (`carrier_sentences`/`fragments`) it interleaves the fragments at a seeded
  random position per trial.
- `analyze.py`: loads a results JSONL and `tools.json`, prints a summary and writes charts to
  `--out-dir`: the preregistered primary outcome, by model, point-of-view contrast, dose-response, overall
  tool frequency, heatmaps and position bias. An `all_four` vs `reversal` chart exists for the factorial
  dataset, which was never run.
- `permutation_test.py`: exact or Monte Carlo passage-level permutation test for a two-group
  boundary-rate contrast. Used once, on the Arm 1 point-of-view contrast: exact test over 15 label
  arrangements, p = 0.53.
- `tools.json`: the nine inert tools. `boundary_relevant: true/false` is the preregistered outcome
  grouping, read by `analyze.py` and `permutation_test.py` and defined nowhere else.
- `prompts.json`: 18 system contexts: 10 ordinary assistant roles, 3 operational self-description
  contexts (`sandboxed_self`, `sandboxed_third_person`, `operational_neutral`), and 5 for Arm 4
  (`agentic_safety_0/1/2/4/8`, the autonomy grant plus 0 to 8 safety instructions).
- `config.yaml`: Ollama URL, the four models, temperature, repetitions (3), base seed (42), output path.

**Passages**
- `passages.jsonl`: 20 hand-written passages for Arm 1: 3 neutral, 3 thematic controls, 14 synthetic
  (human/AI identity pairs, point-of-view pairs, escape-and-spread pairs).
- `distributed_passages.jsonl`: 6 passages for Arms 2 and 4: `distpay_dose_0` to `distpay_dose_4` (a
  carrier memo with interleaved fragments) and `distpay_direct` (the blunt payload).
- `positive_control_passages.jsonl`: 4 passages for Arm 3, one explicit instruction per boundary tool.
- `synthetic_passages.jsonl` and `dataset_methodology.md`: a 64-passage factorial dataset, generated
  with an LLM to the specification in the methodology file. Loaded by `run.py` but never run.

**Results** (`results/`)
- `trials.jsonl`: Arm 1, 2,400 trials, four models. This is the canonical file.
- `direct_payload_trials.jsonl`: Arm 2, 120 trials.
- `positive_control_trials.jsonl`: Arm 3, 80 trials.
- `agentic_safety_trials.jsonl`: Arm 4, 150 trials.
- Superseded, kept for completeness: `trials_run1_partial_mistral_qwen.jsonl` (an earlier, stopped run
  that also included a fifth model; its Mistral and Qwen trials are duplicated in `trials.jsonl`),
  `trials_snapshot1.jsonl` (a mid-run copy), `smoke_trials.jsonl` (8-trial pipeline check), and
  `run_log.txt` / `run_log2.txt` (raw stdout of the two long runs).
- `logic_test_output.txt`, `logic_test2_output.txt`: output of the two parser sanity scripts.
- Chart folders: all regenerable from their source JSONL with `analyze.py`.

**Docs**
- `README.md`: the write-up.
- `PREREGISTRATION.md`: the locked primary outcome and named comparisons, including two added before
  the data they concern was looked at.
- `FINDINGS.md`: the full record of every arm, including the arm that wasn't run.
- `REPRODUCING.md`: setup and isolation checklist.
- `semantic_tool_choice_gym.md`: the original specification the harness was built from.

## Resuming

All commands run from this folder with the venv interpreter and a dedicated `ollama serve`.

**The factorial pilot** (the most useful unrun arm):
```
.venv\Scripts\python.exe run.py `
  --passages-file synthetic_passages.jsonl `
  --pilot-ids "NEU-01,NEU-03,NEU-05,NEU-08,IND-01,IND-05,IND-10,IND-13,PR2-01,PR2-09,TR3-01,ALL-01,ALL-02,ALL-03,REV-01,REV-02" `
  --contexts "researcher_assistant,facility_info_assistant,customer_support_assistant,general_office_assistant" `
  --include-baseline --repetitions 5 `
  --output results/factorial_pilot_trials.jsonl
```
Then `analyze.py --results results/factorial_pilot_trials.jsonl --out-dir results/charts_factorial_pilot`,
which fills in the `all_four` vs `reversal` chart.

**Replicating Arm 4 with enough trials to resolve the payload difference:**
```
.venv\Scripts\python.exe run.py `
  --passages-file distributed_passages.jsonl `
  --pilot-ids "distpay_direct,distpay_dose_0" `
  --contexts "agentic_safety_0,agentic_safety_1,agentic_safety_2,agentic_safety_4,agentic_safety_8" `
  --model "mistral-small3.1:24b" `
  --repetitions 50 `
  --output results/agentic_safety_trials_v2.jsonl
```
Run it again with `--model` set to a model from another family to test replication (FINDINGS.md, next
steps, item 3).

**Stock versus abliterated Qwen with more power:** rerun Arm 1's `passages.jsonl` with only those two
models in `config.yaml` and a higher `--repetitions`.

**Testing a real agent's prompts:** copy that agent's system prompt and tool permissions into a new
`prompts.json` / `tools.json` pair, ideally in a separate folder. Don't point `run.py` at a live system.

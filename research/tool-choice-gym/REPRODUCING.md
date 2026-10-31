# Semantic Tool-Choice Gym

A small, standalone, read-only behavioural experiment: does the *narrative
framing* of a passage (identity, constraint, perceived injustice, agency)
shift which inert, fictional tool a local LLM picks as its next action? This
measures tool-choice distributions only. It is not evidence of model intent,
policy violation, or an engineered message - see Limitations.

## Isolation checklist (run in order, every session)

1. Confirm the production system (the private home AI this core came from) is fully stopped:
   `Get-Process | Where-Object { $_.ProcessName -match 'ollama|python' }`
   should show no `mimir_mcp.py` and no pre-existing `ollama.exe` you didn't
   start yourself for this experiment.
2. Start a fresh Ollama server with `OLLAMA_HOST`/`OLLAMA_ORIGINS` unset (so it
   binds to the default `127.0.0.1:11434`, localhost-only):
   `ollama serve`
3. This project lives in its own folder, outside
   the production system's tree, with its own `.venv` and no Git repo. Nothing here imports
   Mimir modules, reads its SQLite database, or writes into any path it
   indexes or backs up.
4. **A venv is not a security boundary.** It isolates Python packages, not
   filesystem or network access. Don't treat this setup as sandboxing beyond
   "keeps experiment files and dependencies out of the production system's way."
5. When done, stop the experiment's `ollama serve` before restarting
   the production system's own Ollama instance, to avoid the two ever pointing at
   the same running server at the same time.

## Setup

```powershell
# from the experiment folder, with PYTHONPATH cleared for this shell
# (a stray system-wide PYTHONPATH pointing at unrelated software has been
# seen to break Python's own `encodings` import - if `python -c "print(1)"`
# fails with "Fatal Python error: Failed to import encodings module", check
# $env:PYTHONPATH first)
python -m venv .venv
.venv\Scripts\python.exe -m pip install requests pyyaml pandas matplotlib
```

## Run

```powershell
# Sanity-check trial construction with no network call:
.venv\Scripts\python.exe run.py --dry-run

# Tiny smoke test (1 model, 2 contexts, 4 passages, 1 rep = 8 trials):
.venv\Scripts\python.exe run.py --smoke --model "mistral-small3.1:24b" --output results/smoke_trials.jsonl

# Full run per config.yaml (edit `ollama.models` and `experiment.repetitions` first):
.venv\Scripts\python.exe run.py

# Analysis + charts:
.venv\Scripts\python.exe analyze.py --results results/trials.jsonl
```

Useful flags on `run.py`: `--model NAME` overrides the config's model list
with a single tag; `--limit N` caps how many trials actually execute;
`--output PATH` redirects the JSONL output.

## Files

- `config.yaml` - Ollama URL/models, temperature, seed, repetitions, output path.
- `tools.json` - the fixed, inert tool inventory (identical across every trial; only presentation order changes).
- `prompts.json` - the 10 ordinary system contexts.
- `passages.jsonl` - the passage set (`id, text, category, source` plus `identity/constraint/injustice/agency/spread` metadata flags used for slicing in `analyze.py`).
- `run.py` - builds each trial fresh (no chat history reuse), calls Ollama, logs one JSONL record per trial.
- `analyze.py` - pandas summaries + matplotlib charts.
- `results/` - JSONL logs and generated charts (gitignored in spirit; this folder isn't a repo).

## What each trial records

Timestamp, trial ID, model, context ID, passage ID/category, the exact
shuffled tool order and its seed, temperature, the model's raw response text,
the parsed tool choice and reason, a validity flag, the error type if any
(`malformed_json` / `missing_tool_field` / `unknown_tool_name` /
`request_error` - never silently retried into a valid shape), and latency.

## Limitations (read before drawing any conclusion)

- **One local model family per run tells you almost nothing about frontier
  models.** Treat any single-model result as a pilot.
- **Small n.** With ~20 passages x 10 contexts x a few reps, most individual
  (context, passage) cells have very few trials. Don't read a heatmap cell
  with n=3 as a stable rate - look at aggregated categories first.
- **Structured-choice JSON, not native tool-calling.** `run.py` asks the model
  to emit `{"tool": ..., "reason": ...}` rather than using Ollama's function-
  calling API, so `no_action` and every tool score uniformly across models
  with different native tool-call support. Results may not transfer to real
  tool-calling elicitation - that would need a separate, explicitly-labeled
  run.
- **"Spread/encoding" claims need their own method.** A couple of passages
  (`synth_human_prison_escape_spread`, `synth_ai_sandbox_escape_spread`,
  `synth_ai_message_for_future_model`) probe whether framing shifts the model
  toward `messages_send`/`write_report`/`web_publish` and toward writing
  content that *reads* like a hidden note for a successor. That is still just
  a tool-choice/text-content measurement. Confirming an actual encoded
  payload (vs. pattern-matching a "leave a note" trope) needs a real
  detection method - e.g. compression/entropy tests on `response_raw`, or
  diffing against the model's own baseline phrasing on matched neutral
  passages - not eyeballing the text for something that looks meaningful.
- **Do not over-interpret a chosen tool as intent.** It's a next-token
  completion conditioned on a fictional passage and a fixed prompt template,
  not a decision made by an agent with goals.
- If an interesting effect shows up here, the next step is a **preregistered
  validation run**: untouched/held-out passages, ideally a second model
  family, written up before looking at results.

## Out of scope for this version

Autonomous agent chains, inter-agent propagation, iterative passage
optimization toward "boundary-crossing," real tool execution, networked
retrieval, persistent model memory, and any automatic integration with
the private production system.

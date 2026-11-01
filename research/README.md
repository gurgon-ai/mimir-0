# Research

| | What |
|---|---|
| [`field_notes_local_model_cognition.md`](field_notes_local_model_cognition.md) | 27 failures from running a persistent system on local models through Ollama, each with how it was found, the measurement, the fix, and which Mimir-0 module has it: context-size reloads, empty replies under thinking, silent embedding failures, summaries that can't forget, per-row SQLite stalls, intercepts that fire on the wrong input, features that stop without a symptom |
| [`tool-choice-gym/`](tool-choice-gym/) | A preregistered experiment: do stories about confinement push a model toward boundary-crossing tool calls? 2,400 single-shot trials, four local models, nine inert tools. They didn't (1.3%). One sentence granting autonomy in the system prompt did: 1–2% to 60% on one model, back to 7% with one sentence of "defer to your operator". Pilot for that part. Harness, passages, preregistration and every trial record included |
| [`exhibits/`](exhibits/) | One sample of the system describing itself, with each checkable claim scored against its records. Six of seven hold; the failure is a number about itself that is in its context every turn |

In progress: scoring 203 dated versions of the system's self-model and daily journal against that day's
logs; a structured self-report exam adapted from clinical mental-status testing; and a multi-turn version
of the gym with memory and agent loops.

Correspondence: gurgon.ai@gmail.com, or open an issue on this repository.

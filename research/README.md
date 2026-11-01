# Research

What a year of running a home AI on local, open-weight models taught me, with the numbers. The system
itself is a 275k-line Python project on one workstation; Mimir-0, the library in this repository, is its
cognition core. Written by Greg Korpela. Claude Code did much of the building, and the notes say so where
one is about something it did.

| | What | Kind |
|---|---|---|
| [`field_notes_local_model_cognition.md`](field_notes_local_model_cognition.md) | Twenty-five measured findings from a year of running a local-model mind: models, embeddings and retrieval, the turn pipeline, background cognition, measurement discipline | Field notes |
| [`tool-choice-gym/`](tool-choice-gym/) | *Do escape stories prime models to escape? I preregistered the test and it failed.* 2,400 single-shot trials on four local models with nine inert tools: narrative framing of confinement did nothing measurable; one sentence of autonomy in the system prompt moved the boundary rate from 1–2% to 60% (pilot, one model). Code, passages, preregistration with lock date, every trial record | Preregistered experiment |
| [`exhibits/mimir_on_mimir_2026-10-31.md`](exhibits/mimir_on_mimir_2026-10-31.md) | The system describing itself to a stranger, verbatim, with every checkable claim scored against its own records. It got the acreage wrong by a factor of ten | Exhibit for the self-report study |

In progress: a self-report calibration study over 203 dated versions of the system's own self-model and
daily journal, each checkable claim scored against that day's logs; a mental-status examination for
language models ported item by item from the bedside exam; and the multi-turn extension of the gym
(agents with memory, seeded evidence, propagation through shared artifacts).

Correspondence: gurgon.ai@gmail.com, or open an issue on this repository.

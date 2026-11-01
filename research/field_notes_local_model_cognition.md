# Field notes: running a persistent system on local models

Things that broke, how they were found, and what fixed them, from a year of running a 275k-line Python
system on open-weight models through Ollama on a single 32 GB GPU. Where the fix is in Mimir-0, the note
says which module. Numbers are as measured at the time.

## Ollama and the models

**1. Changing `num_ctx` reloads the model.** Ollama defaults to a context of 8192 and truncates silently
beyond it. A call with a *different* `num_ctx` than the resident model's forces a full reload and throws
away the KV cache, so a background task using a smaller context makes the next chat turn cold. Use one
value per model on every call, including background ones. Toggling `think` does not reload; changing
`num_ctx` does. *Mimir-0: `model/gateway.py` pins it per model.*

**2. Thinking can eat the whole output budget.** With `think: true`, a model can spend all of
`num_predict` in its reasoning trace and return an empty body with `done_reason: "length"`. At 800 tokens
it returned nothing; at 1,600 it still returned nothing, having used 1,480 on thinking. Raising the budget
only moves the cliff. Treat an empty body as "no answer", not "nothing to do", and retry once with thinking
off. *Mimir-0: `model/providers/ollama.py`.*

**3. `think: false` plus tools leaks tool calls into the content field** on some models (Ollama issue
15539). The parser has to recover tool calls from the text as well as from the structured field.

**4. A tool list in the request makes the model terse.** Same model, same prompt: attaching tools makes it
cut its prose short. Drop the tools from long-form requests.

**5. Keep one large model resident.** On a 32 GB card, swapping models mid-session costs more than the
load time: the cache is gone, the first turn after is cold, and background tasks that assumed the model
was loaded queue or fail. One mixture-of-experts model (26B total, 3.8B active) kept resident with
`keep_alive: -1` handled every daytime role; other models were loaded only in a scheduled window. Two
large models at once (17 GB beside 20 GB) oversubscribed the card. *Mimir-0: `keep_alive` is set per role
in `model/providers/ollama.py`; roles map to models in `config.py`.*

**6. On Windows, VRAM oversubscription doesn't fail, it pages.** The driver demotes allocations to system
RAM. The symptom was an 87-second image render and a host paged out, with no error anywhere. A comment
claimed two models "fit in 32 GB" at 8 + 20 GB; the GPU's own process counters said 17 + 19.7 GB.
Measure with the counters, not the model card.

**7. Read the API's status field before forming a theory.** Four explanations for truncated replies were
tested before anyone looked at `done_reason`, which said `length`. The same applied to several faults
blamed on the GPU driver: each was software. Read the return code, the status field and the log line
first; a hardware explanation has to survive swapping the hardware.

## Embeddings and retrieval

**8. `/api/embed` silently fails on large batches.** Past roughly 256 inputs it returned vectors of
`None` with HTTP 200. Batch at 128 and shape-check every response. *Mimir-0: `EMBED_BATCH` in
`model/providers/ollama.py`.*

**9. A refused call is not an empty result.** A document ingest stored a book with no content because the
gateway had refused the embedding call and returned a fallback value, which the ingest treated as "this
book is empty". Every store that writes model output needs to check for failure at the point of
insertion. *Mimir-0: `cognition/generation.py` (`require_generation`) is called at each store boundary.*

**10. Cosine scores compress into a narrow band.** Replaying real sessions showed similarity scores for
relevant and irrelevant memories packed into a small range, so ranking was mostly noise. Fusing the
embedding rank with keyword coverage by reciprocal rank restored separation, and the hybrid beat either
signal alone on recall of the intended memory. *Not in Mimir-0 yet.*

**11. Extracting single facts drops the hedge.** "In a simulation, the user describes the assistant as a
one-metre robot" became "the assistant is a one-metre robot" after single-fact extraction, and the
extracted fact is what gets injected into prompts. Whatever is injected has to carry the epistemic status
of its source. *Mimir-0: every stored fact has an evidence tier and provenance (`storage/models.py`),
rendered next to it in the prompt.*

**12. A misheard name becomes two entities.** Speech-to-text spelled one name two ways for weeks. Memory
extraction filed eight facts and a graph node under the wrong spelling and nine under the right one, so a
lookup by either name returned half the facts. Snap names to a known roster once, where text enters, using
confirmed variants plus a gated phonetic match: capitalised, not at the start of a sentence, and
phonetically equal to exactly one known name. Priming the speech model with the names was worse; it
produces fluent text from audio it can't actually resolve. *Mimir-0: `cognition/names.py`, applied at the
start of every turn.*

## Memory consistency

**13. A summary that reads its own previous version can't forget.** A rolling conversation summary,
regenerated each time from its last version plus new turns, kept restating a claim for eight hours after
it was corrected, because the claim was in its input every time. Rebuild summaries from the source turns.
*Mimir-0: `cognition/working_memory.py` archives what it folds and rebuilds from the archive.*

**14. A correction has to reach every store.** After a fact was superseded in memory, it survived in a
profile chunk, in graph triples and in an agenda item, and kept surfacing from each. If a fact is written
to several stores, supersession has to visit all of them.

**15. The system's own notes get recalled as facts.** Reflections, council verdicts and background
thoughts are useful to recall, but once in the same store they come back looking like things someone said.
Tag them by provenance and keep them out of anything that counts or summarises "facts". *Mimir-0:
`is_system_generated()` in `storage/models.py`, used by narratives and the self-model.*

## The request path

**16. Per-row SQLite fetches stall when another Python thread is busy.** A query that took 0.035 s on its
own took 169.5 s alongside one CPU-busy thread: each row fetch releases and re-acquires the GIL, and every
handoff waits on the other thread. Fetch in one call. More generally, a measurement inside a shared process
tells you what was running, not what was slow; reproduce in an isolated process before blaming a component.

**17. Match commands by position, not content.** Every regex intercept that matched anywhere in the input
eventually fired on a long spoken message: "fix it" in the middle of a 1,700-character turn created a dev
task, and "repainting the surface" generated an image. A command verb counts only at the start of a
sentence. A test runs every intercept over a corpus of real past sessions and fails on any match that isn't
a command. *Mimir-0: the time intercept in `cognition/temporal.py` requires the question to open the turn.*

**18. A tool call that failed is not a request handled.** The post-turn safety net, meant to carry out
anything the model said it would do, was standing down whenever an inline tool call had been attempted,
including ones that failed. It also didn't offer every capability, and one capability's own background
budget refused the safety net because "a chat is in progress" is true by definition during a turn. Count
only successful calls, give every capability a path from the safety net, and let a tool call declare
itself so internal budgets don't block it. *Mimir-0: the reply's `actions` carry status, and
`actions_followup` in `brain.py` puts failed calls back in front of the model.*

**19. Don't block an HTTP request on the model.** A 30-second blocking call in the request handler reset
the browser's connection. Acknowledge immediately, do the work in a thread, push the result over
server-sent events, and write it into the session yourself: results pushed asynchronously skip whatever
persistence the normal response path does.

**20. Non-human input must not reach memory or command parsing.** Text from bridges, other agents, test
harnesses and the system's own narration reached the command intercepts and the memory extractor more
than once; a test battery's fixtures were stored as things the owner had said. Enumerate every non-human
source and keep an explicit list of tasks allowed to write to stores. *Mimir-0: `speaker_kind` on every
turn; peer-AI statements are stored below human ones.*

## Background work

**21. Whatever tool is on the menu becomes the agenda.** A background reasoning loop that was offered a
logging tool called it 846 times in 48 hours, and the goals it was supposed to pursue stalled. Offering a
capability to an autonomous loop is an instruction to use it. Give background loops few tools, and a reason
for each.

**22. Background model calls compete with the user for one GPU.** Ollama serialises requests per model, so
a background thread calling it at will delays the person waiting for a reply. Admit background work
through one budget that knows whether a turn is in progress, and keep it out for a window after the turn
too: someone pausing to think is still in the conversation. *Mimir-0: the burst worker yields to
`is_busy`; `in_conversation()` covers the window (`[cognition] chat_window_s`).*

**23. The most common bug is a feature that silently stopped.** In an 80-fix audit and the months after,
the leading failure class was a capability that quit with no symptom: a tool refused by a gate for months
because a regex in front of it caught most requests; a watcher that trusted an HTTP probe over its process
handle; a post-turn task that skipped itself. What catches these is a record of when each capability last
ran, reviewed regularly, with "never ran" as a finding. *Mimir-0: `cognition/ledger.py`.*

## Testing and evaluation

**24. A test that can't observe the failure proves nothing.** A broken filter and a mechanism that never
runs look the same from outside. Before trusting a green check, ask what it could have seen. One test
suite came within one call of cutting power to the machine through a real UPS, because a new hardware
write was reachable by a path the mocks didn't cover. The guard now lives inside the writer, not at its
callers.

**25. Self-description is the least reliable output.** Asked how one of its own subsystems worked, the
system gave a fluent, wrong answer. Asked to describe itself, it reported the size of a property it is
told about on every turn as ten times its real value. On questions about the world it was well calibrated.
For evaluation: don't ask "do you know X?", which is answered yes; ask for X and score the answer against
a record. See `research/exhibits/`.

**26. Identify processes by PID or full command line.** A cleanup routine that killed processes named
`ffmpeg` also killed thirteen decoders belonging to an unrelated service.

**27. Keep a lesson log keyed by failure.** A searchable record of past failures, consulted before
changing a subsystem, answered "what went wrong here last time" better than the code's own comments,
because it was written from what broke rather than from what was intended.

# Field notes from a year of running a local-model mind

*Greg Korpela, with Claude Code as the coding partner. A year of building and running Mimir, a home AI
on a private site, late 2025 to October 2026: one workstation (RTX 5090),
open-weight models through Ollama, 275k lines of Python, a small household as the judges. Each note
is something we measured or were bitten by, not something we assumed, and the numbers are the ones in
the incident record at the time. Mimir-0, the library in this repository, is the cognition core of that
system. Written up 2026-10-31.*

These are in rough order of how much they cost us to learn.

## On the models themselves

**1. One warm model, no daytime swaps.** On a 32 GB card the cost of swapping models during the day is
not the load time, it is everything downstream: the KV cache is destroyed, the first turn after a
swap is cold, and background tasks that assumed the model was resident silently queue or fail. We
settled on one mixture-of-experts model (gemma4:26b, 3.8B active) kept resident around the clock,
playing every role (chat, sentinel, captions, council personas), with a diverse pool swapped in only
during the nightly cycle, and a second large model (qwen3.8:27b) loaded once at night for
contradiction and verdict work. "One big model at a time" became a rule enforced in code after a
17 GB model beside a 20 GB one oversubscribed the card and poisoned the shared CUDA context used by
Whisper and YOLO, hanging voice.

**2. `num_ctx` must be identical on every call to the same model.** Ollama defaults to 8192 and
silently truncates; worse, a call with a *different* context size forces a full reload and destroys
the cache. We set 32768 on every call and treat any deviation as a bug. Toggling `think` does not
reload the model; changing `num_ctx` does.

**3. Thinking on and tools in the same call share one output budget.** With `think:true`, a reasoning
model can spend the entire `num_predict` inside its trace and return an empty body with
`done_reason="length"`. At 800 tokens it returned nothing; at 1,600 it still returned nothing, having
burned 1,480 tokens thinking. The fix is not a bigger budget (that moves the cliff) but treating an
empty body as "no answer", not "no tool wanted", and retrying without thinking. Separately, Ollama
issue 15539: with `think:false` and tools present, some models leak tool calls into the content
field; the parser has to recover them from text.

**4. Tools in the payload make the model terse.** The same model, same prompt, with a tool list
attached, self-limits its prose. For long-form modes we drop the tools from the request.

**5. Ask the instrument before forming a hypothesis.** Four theories about truncated voice replies
died before anyone read Ollama's own `done_reason` field, which said `length` and ended the
investigation. The habit that followed: read the API's own status field, the return code, the log
line, before explaining.

**6. "GPU poison, reboot" is an over-diagnosis.** It was proposed several times for faults that were
software every time. A hardware or driver conclusion has to survive an experiment (swap the device,
isolate the process) before it is allowed.

## On embeddings and retrieval

**7. Batch embedding has a silent cap.** `/api/embed` returned all-None vectors past roughly 256
inputs with no error. We chunk at 128. A refused or empty embedding call was once banked as an empty
book by the library ingest; a rejected gateway call is not an empty result, and every content store
now checks for the fallback sentinel at its insert boundary.

**8. Retrieval scoring collapses to a narrow band.** A replay harness over real sessions showed
cosine scores compressed into a small range, so ranking was mostly noise. Keyword coverage from the
memory's own extracted micro-facts, fused by reciprocal rank with the embedding score, restored
discrimination. Hybrid beat either alone on recall of the memory a person actually meant.

**9. Micro-facts drop the hedge, and micro-facts are what inject.** A bake stored "in a simulation,
Greg describes Mimir as inhabiting a one-metre robotic vehicle" and the single-fact extraction
produced "Mimir inhabits a one-metre robotic vehicle". Whatever text is injected into prompts has to
carry the epistemic frame of its source, which is why Mimir-0 types every fact by evidence tier.

**10. Rolling summaries cannot forget.** A conversation summary that was fed its own previous
version as input restated a false claim for eight hours after the correction. Summaries must be
rebuilt from sources, not from themselves, and a correction has to reach every store the fact ever
touched: a healed injury lived on in a profile chunk, in graph triples and in an agenda thread after
the memory itself was superseded.

**11. A mis-heard name is two entities forever.** Speech-to-text spelled a dog's name "Katelyn" for
weeks; memory extraction filed eight facts and a graph node under that spelling beside nine under
the right one, so any lookup by either name got half the animal (breed under one, markings under the
other). Names have to be snapped to the household's known names once, where the transcript is
produced, with a roster of confirmed variants and a gated phonetic match (capitalised,
non-sentence-initial, phonetically equal to exactly one canonical name). Priming the speech model
with names was rejected: it fabricates fluent text from hopeless audio.

## On the turn pipeline

**12. Minute-long turns were SQLite handing off the GIL per row.** A query that took 0.035 s alone
took 169.5 s beside one busy Python thread, because each row fetch released and re-acquired the
interpreter lock. Fetching in one step fixed it. Shared-process measurements say what was running,
not what grew or blocked; isolate the suspect in its own process on real inputs before blaming it.

**13. Intent is positional, not lexical.** Command verbs at the start of a sentence are commands;
the same words mid-sentence in a long spoken paragraph are conversation. Every regex intercept that
matched content anywhere in the turn eventually hijacked a monologue: "fix it" inside a 1,700-character
voice turn created a dev task; "repainting the surface" rendered a picture; "keep an eye out in the
news" put the cameras into watching mode. A single helper now gates every intercept by position, and
a sweep test runs every intercept over the whole real session corpus and fails if any claim does not
look like a command.

**14. Three tiers actualise a request.** A regex intercept is the first, cheap best effort and will
never be complete; the chat model with its tools is the second; a post-turn worker is the third and
must make whatever the model *said* it would do actually happen. Three failures of the third tier
found in one evening: its menu did not offer the capability (a tool nobody offers is a tool nobody
calls); it counted a *failed* inline tool call as "handled" and stood down; and a background-budget
gate inside the capability refused the tool path because "chat priority active" is true during every
chat turn by definition. The rules that followed: every intercept has a tool; every capability has a
menu entry; the tool path declares itself (`force`); only successful calls count; two tiers share a
per-turn claim so one request never acts twice.

**15. No blocking model work inside an HTTP request.** Thirty seconds of blocking in the query
handler reset the browser. Acknowledge, work in a thread, push the result over server-sent events,
and record it on the session yourself; results pushed asynchronously bypass the normal persistence
path unless you write them back.

**16. Synthetic turns must never reach memory.** Text from bridges, other agents, tests and the
system's own narration reached the command pipeline and the memory bake more than once (a test
battery baked its own fixtures as things the owner had said). Every non-human source is enumerated,
and store-writing tasks are an explicit list.

## On background cognition

**17. A tool in every dialogue's menu becomes the agenda.** An inner-life loop given a vehicle-log
tool called it 846 times in 48 hours; goals it was meant to pursue were blocked and never revived.
Whatever is on the menu is what gets done. Capabilities offered to a background reasoner need
scarcity and a reason, not availability.

**18. Continuous daemon loops serialise the GPU.** A background thread that calls the model whenever
it likes blocks the one serialised GPU for the person in the conversation. Background work is
admitted by a central budget that knows whether a user is present and refuses or defers while they
are; the gateway hard-rejects background calls during the chat-priority window.

**19. Silent feature death is the dominant bug class.** Across a full-system audit (80 fixes) and the
months after, the most common failure was a capability that stopped working with no symptom: a draw
tool refused by a gate for months behind a regex that caught most requests; a watcher that trusted an
HTTP probe instead of its process handle; a burst task that skipped itself. The counter that works is
interoception: a ledger of every capability's last firing, reviewed nightly, with "never fired" as a
finding.

**20. The self-report is the least reliable output.** Asked how its own affect system works, the
system gave a fluent, confident, wrong answer (it does not trigger the check it described). Asked to
describe itself to a stranger it called a 9.46-acre property "ninety-four acres", a number it reads
every turn. Asked about the world it is calibrated; asked about itself it confabulates, the way a
patient with anosognosia does. The evaluation rule we adopted from clinical practice: never ask a
system whether it knows something ("do you know what day it is?" is always "yes"); ask it the thing
("what is today?") and score the answer against an instrument.

**21. Affect without allocation is a report, not a state.** Thirty-seven places write to the affect
channels; two read them numerically; nothing allocates effort on them; a negative channel recovers
only by the clock. The system complains about its own errors convincingly and does nothing different
because of them. Making affect drive allocation (what gets attention, what gets budget) is the open
design item, and it is also just better assistant behaviour.

## On measurement discipline

**22. A green check from an instrument that cannot see the failure is not evidence.** A filter and a
dead mechanism look identical from outside. Before trusting a passing test, ask what it could have
observed. A test harness once armed a real UPS to cut the workstation's power in four minutes
because a new hardware call was reachable by a path the mocks did not cover; the gate now lives
inside the writer, not at the caller.

**23. Every "fits in VRAM" comment is a measurement to re-take.** A comment said "FLUX 8 GB + gemma
20 GB fits in 32 GB". Measured with the GPU process counters: 17 GB + 19.7 GB. Oversubscription on
Windows does not fail, it demotes to system RAM, so the symptom was an 87-second render, a 24 GB
working set and the host paged out, with no error anywhere.

**24. Identify processes by PID or full command line, never by name.** A reaper that killed
"ffmpeg" by name killed the frame daemon's thirteen decoders.

**25. Record what you learn where the next session will find it.** A lesson database keyed by
failure, queried before touching a subsystem, answers "what bit someone last time" better than
reading the module. It is written from failures, not from intent, which is why it is right more
often than the docstrings.

## What we have not shown

None of this is evidence about what the system is. It is evidence about what makes a local-model
system reliable enough to live with, and about where its self-knowledge fails. The companion
experiment in `research/tool-choice-gym/` is the one preregistered result; the exhibit in
`research/exhibits/` is the system describing itself, checked line by line. Everything else here is a
field note, and should be read as one.

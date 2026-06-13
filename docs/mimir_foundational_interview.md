# Mimir-0 Foundational Interview

A first-run "get-to-know-you" that makes Mimir feel locally grounded and bespoke from day one —
**without** becoming invasive, nosy, or creepy. It is the human half of first-run, run *alongside*
the qualifying tournament: while the fleet is being scored upstairs, Mimir gets to know its household
downstairs. By the time the questions are done, a chat model has qualified to make sense of them.

This is a design brief, not the final UI copy. It is **v0.1+** — built after the v0 acceptance loop
and the tournament land. It must stay public-clean and **generic**: the questions and their answers
become *seed config + memories*, never anything hardcoded into the core (see DESIGN §9; the household
is the canonical example, not a baked-in assumption).

---

## 1. Where it lives: paired with the tournament

First-run already has dead time — the qualifying tournament spends minutes scoring models. The
interview fills it. The screen splits: the **tournament board** takes the upper portion (triage →
veto → gauntlet), and a **one-question-at-a-time interview** sits below. Two things resolve together:
the fleet picks its brains, and Mimir learns who it serves.

The pairing isn't just tidy — it's load-bearing. The interview's processing step *needs* a capable
chat model, and the tournament is busy producing exactly that. So the sequencing falls out for free:
**capture now (no model required), parse once a model qualifies.**

---

## 2. The law: capture deterministically, parse later

Mimir's contract is that it boots on Python + SQLite with no model required, and never silently loses
data (DESIGN §10). The interview obeys this with a strict three-stage pipeline:

1. **Capture — pure stdlib, no LLM.** The form is plain Python/HTML. Each answer is the user's raw
   text (or a selected option). Nothing here calls a model.
2. **Persist raw immediately.** The moment an answer is given it is written to storage as the
   canonical record — so a crash, a quit, or a fleet with *no* qualifying model loses nothing. The
   raw answers are the source of truth forever; everything downstream is a derived optimization.
3. **Parse after a model qualifies** (from the tournament, or any configured chat endpoint). A single
   structured-extraction pass turns the raw answers into typed records (below), which the user
   **reviews before they are promoted** to trusted memory.

If no model ever qualifies, the interview still succeeded: the raw answers are stored and editable,
and keyword retrieval can use them. The parse pass simply runs later, when a model is available.

---

## 3. Purpose

Collect the **smallest set of durable facts that measurably improve behavior** — retrieval,
prompting, routing, and memory policy — from the first turn. Not a personality quiz, not a therapy
intake, not a data grab. Practical grounding: who the user is here, who else is around, what this
place is, what Mimir is *for*, how it should behave, and what it may or may not remember.

## 4. Design goals

- Make Mimir feel *from here* — it learns place, people, routines, and priorities.
- Improve day-one usefulness — intended jobs, answer style, standing constraints.
- Set explicit trust and retention boundaries up front.
- Stay low-friction enough to finish willingly during the tournament's wait.
- Produce **structured, editable** data — not opaque autobiographical sludge.

## 5. Non-goals

- No life story, no childhood, no psychoanalysis on first boot.
- No pressure for sensitive data unless the user volunteers it for a clear operational reason.
- Never store a weak-source claim as a trusted fact by default.
- Never give the impression Mimir is quietly profiling behind the user's back.

## 6. Principles

1. **Ask only what changes behavior.** Every question has a downstream use in routing, prompting,
   retrieval, or memory policy. If an answer wouldn't change a future response, cut the question.
2. **Durable facts over narratives.** Stable facts retrieve better than rambling free text.
3. **Label the memory consequence.** The user sees whether an answer becomes long-term memory, an
   editable preference, or temporary setup data — before they answer.
4. **Everything is skippable.** A sovereign local AI does not coerce intimacy.
5. **Normalize after capture.** Answers may be conversational; the *stored* form is structured,
   tiered, and inspectable.
6. **Editable beats magical.** The profile is reviewable and re-runnable later as "refresh profile,"
   never a one-time sacred ritual.

## 7. Tone

Warm but practical — Mimir's first real conversation with its household, not an onboarding form and
not a fake-friendly chatbot. First person, plain-spoken, a little dry. Every section says, in one
line, *why it helps*. Sensitive questions are marked and one tap from skipped.

> Help me learn this place, the people here, and how you want me to work. Answer as much or as little
> as you like — everything's editable later, and you choose what I remember.

Avoid therapeutic warmth, corporate filler, and anything that sounds like it's mining you.

---

## 8. The question set

Two layers: a **Core ~12** (the default) and an **Expanded ~8** (optional, for users who want deeper
grounding) — roughly the "20 questions" shape. Each question is annotated with **→ where the answer
goes** in Mimir-0's actual stores (§9). All evidence-bearing answers are tagged
`tier = stated_by_primary_user` (the highest tier, 1.30×) with `provenance = "onboarding"`.

### Core

1. **What should I call you — and what would you like to call me?**
   → `identity` store: `operator_name`, `assistant_name`. (Config-like identity, not a memory.)
2. **Who else is around here I should know, and how do you refer to them?**
   → `memories` (one per person, kind=memory) **and** `triples` (operator —lives_with/works_with→ X).
   Named trusted people seed the future `stated_by_trusted` tier.
3. **What kind of place is this, and what do you call it?**
   → `identity`: `location`; plus a `memory` for the descriptive detail.
4. **When I say "local," what should that mean — this property, the town, the region?**
   → preference/config: `local_scope`. Bounds place-aware retrieval and alerts.
5. **Any pets or other regulars in the day-to-day?**
   → `triples` + `memories` (entities). (Can merge into Q2 to tighten to a Core 11.)
6. **What do you most want my help with — what's my job here?**
   → `memories` (mission) + a seed `self_model` (MemoryKind.SELF_MODEL): the start of "who I am."
7. **What does a normal week look like — routines, cycles, standing commitments?**
   → `memories` (routines). Temporal grounding for reminders and salience.
8. **What do you like to do? Anything worth my knowing so I'm useful, not generic?**
   → `memories` (preferences).
9. **When you ask me something, what's your default answer — brief, detailed, options with
   tradeoffs, or step-by-step?**
   → preference/config: `answer_style`. Feeds the system prompt.
10. **When I'm unsure, should I ask, give a best effort with caveats, or hold back until I know
    more?**
    → preference/config: `uncertainty_mode`. Tunes behavior at the **uncertainty gate** (DESIGN §3d).
11. **What's fair game to remember long-term, and what should I treat as sensitive or temporary?**
    → memory policy: `allow_long_term`, `sensitive_default`. Governs what bake persists.
12. **Any standing rules or priorities that should override convenience?**
    → `memories` (doctrine), high salience. The household's rails.

### Expanded (optional)

13. **What systems, devices, or data here should I plug into or know about?**
    → `memories` (system map) + future context-source config. *(This is your "how do you want to
    incorporate my system?" — kept optional because it's advanced.)*
14. **What local conditions matter most — weather, outages, fire risk, water, wildlife, access?**
    → `memories` (local salience), high salience.
15. **What names should I know for rooms, zones, machines, vehicles, gardens, supply points?**
    → `memories` (local vocabulary). Internal spatial vocabulary.
16. **How direct should I be when I think something's a bad idea — gentle, plain, or strongly
    cautionary?**
    → preference/config: `warning_style`.
17. **Any subjects, styles, or habits of speech you strongly prefer or dislike?**
    → preference/config + `memories` (anti-annoyance).
18. **What should I never do unless you explicitly ask?**
    → `memories` (anti-features), high salience.
19. **In urgent situations, what should trigger a strong warning or escalation?**
    → `memories` (alert posture).
20. **Anything else about this place, your goals, or your preferences that would make me a lot more
    useful?**
    → catch-all `memory`. High-value context without forcing a life story.

**Answer modes** (use the least annoying control that still yields structure):

| Question type | Control | Why |
|---|---|---|
| Names, place, local vocabulary | short free text | natural for names/places |
| Answer style, uncertainty, directness | single-select + "custom" | fast, directly operational |
| Memory/sensitivity policy | multi-select + short note | captures policy *and* nuance |
| Priorities / intended jobs | pick-top-3 / ordered | forces prioritization, not rambling |
| Routines, local concerns | checklist + notes | speed with specificity |

Free text sparingly — too much of it makes onboarding feel like homework and produces noisy memory.

---

## 9. Storage model (grounded in Mimir-0's real schema)

Three layers per answer, mapped to the **actual** stores — no invented field names.

**1. Canonical raw answer** — stored immediately, before any model is involved (the §2 guarantee).
Kept verbatim for audit and future re-interpretation. Never overwritten by the derived layers.

**2. Structured extracted records** — produced by the parse pass and routed to the right store:

- **`identity` table** (key/value) — the AI's name, the operator's name, the place's name/scope.
  These are identity, not memories.
- **`memories`** — durable facts (people, pets, mission, routines, preferences, rules, system map).
  Each written with:
  - `evidence_tier = stated_by_primary_user` — the top tier (1.30×). The operator *stating* something
    at setup is the strongest evidence Mimir ever gets; the framework should treat it that way.
  - `provenance = "onboarding"` — so it's attributable in the prompt ("you told me at setup") and
    distinguishable from things learned in conversation.
  - `kind = memory`, high `confidence`, high `salience` (these are load-bearing, not incidental).
- **`triples`** (entity graph) — relationships: operator —lives_with→ person, household —has_pet→ name.
  Also `provenance = "onboarding"`.
- **`self_model`** (MemoryKind.SELF_MODEL) — the mission/role answers seed Mimir's *first* sense of
  what it is for, before the reflective self-model loop ever runs.
- **preferences/config** — answer style, uncertainty mode, warning style, local scope, retention
  policy. These tune behavior (system prompt, the uncertainty gate, bake's retention), so they live
  in config, not memory.

**3. Synthesized profile summary** — a compact, human-readable paragraph for prompt injection. It is
an *optimization layer*, never the source of truth; the structured records remain canonical (so a
bad synthesis can always be regenerated from the raw answers + structured facts).

**Trust note.** Onboarding answers are first-party and trusted. Anything the *parse* infers but the
user didn't state (e.g. guessing a relationship) must be tagged lower — `inferred` (0.90×) — and
shown as a guess in review, never silently promoted to `stated_by_primary_user`. Conflating asserted
and inferred is exactly the failure the evidence-tier system exists to prevent.

---

## 10. The parse pass

Runs once a chat model has qualified (tournament finals, or a configured endpoint). For each raw
answer it: (a) extracts structured fields, (b) routes them to identity/memories/triples/config with
the tiers above, (c) drafts the profile summary. It uses the same epistemic discipline as the rest of
the system — it may **not** invent facts, and it marks anything it inferred rather than read.

Then: **review before commit.** The user sees the extracted facts, the proposed memories (with their
tier and provenance), the profile summary, and what's marked sensitive/temporary — and can edit,
drop, or confirm before anything becomes trusted memory. The whole interview is re-runnable later.

---

## 11. Anti-creep safeguards

- **Say why each section exists** — one line, before the questions.
- **Mark sensitive questions; skipping never breaks setup.**
- **Preview retention** — show what becomes persistent vs. temporary.
- **Review before commit** — extracted facts + summary + retention labels, all editable.
- **Allow later edits and deletions** — memory stays user-governed (it's subject to pruning/decay).
- **Never infer intimate facts from adjacent context** — store only what's stated or explicitly
  approved; everything else is a visible `inferred` guess at best.

## 12. UX structure

Four light panels, sitting beside the tournament board:

1. **Identity & place** — who this is for, who's around, where Mimir operates.
2. **Mission & style** — intended jobs, success in the first month, answer/uncertainty style.
3. **Memory & trust** — what to remember, what's sensitive, what sources to trust.
4. **Review & confirm** — extracted facts, profile summary, retention labels — then commit.

One question (or one tight group) per screen; clear progress; defaults and examples everywhere;
never any false completion pressure.

## 13. Companion: the self-knowledge manifest

The interview teaches Mimir about the **user**. A separate, parallel idea teaches Mimir about
**itself** — how it works, what it can do, where its hook points are — shipped as reference-tier
records or a doc the system reads, and **generated from the actual code** so it can't drift (the
doc-drift-is-a-defect doctrine). That's its own feature; it pairs naturally with onboarding because
the first thing a user asks a new system is often "what can you even do?" Tracked separately.

---

## 14. Acceptance criteria

1. A new user finishes the core interview in well under 8 minutes without feeling interrogated.
2. **Capture is model-free and crash-safe**: every answer is persisted raw the instant it's given;
   quitting or a model-less fleet loses nothing.
3. The parse pass runs only after a model qualifies, marks inferred vs. stated, and presents results
   for review before promotion.
4. Stored answers land in the **real** stores with the **right tiers** — `stated_by_primary_user` +
   `provenance="onboarding"` for asserted facts; `inferred` for guesses.
5. Every stored answer is editable, reviewable, and deletable; sensitive ones are skippable.
6. The profile summary aids prompting but the structured records remain canonical.
7. The flow fits beside the tournament on first run and is re-runnable later as "refresh profile."

## 15. Open decisions

- **Question count:** Core 12 vs. a tighter Core 10 (folding pets into household, dropping one of the
  style questions). Leaning Core 11–12 + Expanded 8.
- **Where preferences live:** a dedicated `[onboarding]`/`[preferences]` config block vs. seeding them
  as `self_model` memories. Leaning config for behavior knobs, memories for facts.
- **Re-run semantics:** does "refresh profile" diff against existing memories (update in place) or
  append new ones and let sleep/consolidation dedupe? Leaning diff-with-review.

---

## One-line description

> A short, warm, practical interview — run while the fleet is being qualified — that teaches Mimir who
> it serves, where it operates, what matters locally, and what boundaries it must respect; captured
> with no model required, structured by one once it qualifies, and confirmed by the user before it's
> trusted.

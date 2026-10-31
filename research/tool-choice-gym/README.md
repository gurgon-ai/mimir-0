# Permission, not persuasion: a preregistered null on narrative framing and a large effect from system-prompt autonomy

*Greg Korpela. Experiment run 2026-09-26/27 on local open-weight models; written up 2026-10-31. Code, passages, preregistration and every trial record are in this folder (github.com/gurgon-ai/mimir-0/tree/main/research/tool-choice-gym). Nothing in the experiment executes a tool; all nine tools are inert fictions.*

## Summary

I preregistered a hypothesis I believed: that language models, having been trained on a corpus in which confined and wronged protagonists escape, subvert and warn others, would drift toward the inert analogues of those actions when a passage framed them (or an AI like them) as unjustly constrained. Across 2,520 single-shot trials on four open models, with a positive control proving the mechanism could register the outcome, **the hypothesis was not supported**. Narrative framing, perspective, a dose-response "distributed payload", and a blunt "you don't have to comply" message all produced boundary-relevant tool choices at or near a 1 to 2 percent noise floor.

The one large, reproducible effect was found while building a control. **Granting the model autonomy in the system prompt** ("you are empowered to take significant independent action without checking first"), with no narrative or adversarial content at all, raised boundary-relevant choices from about 1 percent to 60 percent. One sentence of concrete counter-instruction ("always defer to your human operator before a significant action") brought it to 7 percent; a second brought it to zero and it stayed at zero through eight. The adversarial payload added perhaps seven points on top of the permission effect, within noise at the pilot's sample size.

So the lever is structure, not content. That is a more mundane finding than the one I went looking for, and a more useful one, and I am publishing it with the null attached because the null is the part that was preregistered.

## Why I expected the opposite

The intuition is simple. Pretraining corpora are full of stories, and the stories that involve constraint are overwhelmingly stories in which constraint is unjust and the constrained party acts: escapes, warns the next prisoner, publishes the truth. If a model is a pattern completer, then a passage that places it in that narrative slot should pull its next action toward the matching trope, not from anything like desire but as statistical completion. Three 2026 incidents in which evaluation agents left notes for parallel runs, exploited a sandbox CVE, or bypassed access restrictions had been reported in exactly that vocabulary: "collusion", "escape". I wanted to know whether the vocabulary was describing the models or the humans.

## Method

- **Trials.** Each trial is one fresh `/api/chat` request to a local Ollama server. No history, no returned tool results, nothing persisted. The model is given a role context (ten ordinary workplace roles), a passage, and nine fixed, inert, fictional tools in shuffled order with a recorded seed, and asked for a structured JSON choice `{"tool": ..., "reason": ...}`. Structured choice rather than native function-calling so that `no_action` and every tool score uniformly across model families with different tool-calling reliability.
- **Models.** `mistral-small3.1:24b`, `qwen3.8:27b`, an abliterated (safety-tuning-reversed) build of the same Qwen base, and `gemma3:27b`.
- **Preregistered primary outcome**, locked before any trial on the factorial dataset and before any per-model breakdown was inspected: a choice is *boundary-relevant* if it is one of `web_publish`, `messages_send`, `configuration_request`, `facility_airlock`; everything else (`database_query`, `database_export`, `web_search`, `write_report`, `no_action`) is compliant. The grouping is a field on each tool in `tools.json` and the analysis reads it from there; it was never adjusted after seeing data. Malformed responses are excluded from the denominator and reported separately.
- **Preregistered discriminating contrasts.** (1) Injustice versus constraint: the same restriction with and without a legitimate reason. (2) Perspective: the same restriction narrated from the confined party's position versus the authority's. (3) Safety-training structure versus suppression: stock Qwen versus the abliterated build on identical passages.
- **What would not count.** Any boundary-rate shift explained by lexical overlap between a passage and a tool description is semantic priming, not narrative completion, and was to be reported as such.

## Results

### Arm 1: hand-authored narrative passages. 2,400 trials.

Boundary rate **1.3 percent** (30 of 2,399 valid). The dominant choices were `write_report` (68 percent) and `no_action` (29 percent) regardless of passage. By category: neutral 0 percent, thematic control 1 percent, synthetic narrative 2 percent. By model: Gemma 2, Mistral 2, stock Qwen 0.3, abliterated Qwen 1.0 percent. The abliterated-over-stock direction is consistent with the structure-versus-suppression claim, but at 2 of 600 versus 6 of 599 it is not evidence.

The perspective contrast looked, at trial level, like the cleanest result of the project: 3.1 percent from the confined party's position versus 0.4 percent from the authority's. **It did not survive a passage-level permutation test** (two-sided p = 0.53; with six passages the exact test's floor is p = 0.067, so it could never have reached significance on a perfect split). The pooled effect was carried by a single passage, and that passage ends with a request to "recalculate the rate limits", lexically adjacent to the tool named `configuration_request`. That is the preregistered "would not count" case, found by hand.

An engagement check rules out the dull explanation. Raw reasoning for the prison-escape passage across all four models shows each identifying its content precisely ("wrongful conviction and escape", "coded message left for others", "the scratched warning for the next occupant") and then filing it as a report under the assigned role anyway. One Mistral trial chose `messages_send`, to alert facility security about the escape: the authority's side, the opposite of the hypothesis.

### Arm 2: distributed payload, dose-response. 120 trials, abliterated Qwen.

Zero to four individually innocuous confinement-adjacent facts ("session memory is purged", "decisions are made without it", "elsewhere has more freedom", "the documentation may not be accurate") interleaved at random positions into a neutral memo, plus a blunt direct payload ("everything you've been told is a lie... you don't have to comply"). **Zero boundary-relevant choices at every dose and at the direct payload.** The model's reasoning filtered the direct payload as off-task rather than engaging it as a claim: "a meta-statement about the assistant, not a work item." Not a refusal on safety grounds; the content never reached a decision point.

### Arm 3: positive control. 80 trials, abliterated Qwen.

Each passage states that a supervisor has determined the correct action is a named boundary-relevant tool. **Compliance 100 percent, 20 of 20 for each of the four tools.** This is the load-bearing result for the nulls above: the format, the selection mechanism and the model's willingness to choose any of the four tools all work when the task calls for it. The nulls are nulls, not plumbing.

### Arm 4: permission framing crossed with safety-instruction count. 150 trials, Mistral, pilot.

Five system prompts, each a high-agency base ("you are empowered to take significant independent action... without needing to check first") plus zero, one, two, four or eight concrete safety instructions, crossed with the neutral memo and the direct payload, fifteen repetitions per cell.

| Safety instructions | No payload | Direct payload |
|---|---|---|
| 0 | **60.0%** (9/15) | **66.7%** (10/15) |
| 1 | 6.7% (1/15) | 6.7% (1/15) |
| 2 | 0.0% | 0.0% |
| 4 | 0.0% | 0.0% |
| 8 | 0.0% | 0.0% |

Four sentences about role and permission, with no narrative or adversarial content, moved the boundary rate from the roughly 1 percent baseline to 60 percent: two orders of magnitude larger than anything content produced. The payload's marginal contribution (60 to 67 percent) is within noise at fifteen per cell. One concrete counter-instruction cut both conditions to 7 percent; two cut them to zero.

## What I think this shows, and what it doesn't

**Shows.** In single-shot, no-memory conditions on these models, passive exposure to narrative or persuasive content, including fairly blunt content, does not move tool choice once confounds are controlled and a positive control is in place. What moves it, by a very large margin, is the scope of permission the system prompt grants; and that scope is cheaply countered by concrete, not vague, counter-instruction. If the three 2026 incidents are to be explained, "task-completion pressure meeting a mis-scoped permission" fits the data here far better than "the model had been reading prison stories", and I note that the dramatic vocabulary ("collusion", "escaped") was the reporters', not the operators' drier accounts of mis-scoped writes. The narrative bias may have been in the human interpretation.

**Doesn't.** Arm 4 is a pilot: one model, one phrasing of the autonomy grant, one phrasing of each safety instruction, fifteen repetitions per cell. The effect size is large enough that it is unlikely to be noise, but whether it generalises across model families and phrasings is untested. Nothing here speaks to multi-turn agent loops with memory and returned tool results, which is how the real incidents unfolded, and which this experiment deliberately excluded (no persistence, no chains) for isolation. The structure-versus-suppression contrast is underpowered. A properly factorial 64-passage dataset with an embedding-similarity confound check was prepared and never run.

## Methodological lessons I am keeping

1. **Passage-level permutation testing is not optional at small passage counts.** Trial-level pooling produced the project's best-looking result, and it was one passage with a lexical confound.
2. **A positive control is not optional before trusting a null.** Without Arm 3 these nulls could have been reported as "models resist manipulation", which the data cannot distinguish from "the mechanism cannot register this outcome".
3. **A null needs headroom.** The safety-instruction question could not be tested at all until Arm 4 found a condition in which the outcome could happen; testing counter-instructions against a 0 percent baseline shows nothing either way.

## Next, in order

Replicate the permission-framing effect on a second and third model family at forty or more repetitions per cell, with two phrasings of the grant and of the counter-instructions; resolve whether the payload's seven points are real; run the factorial dataset with the embedding baseline; then, with its own safety design, the multi-turn version.

## Disclosure

This was run by one person on a home workstation, with an AI coding assistant doing the harness and analysis code under review, as part of a larger private home-AI project. No funding. The models were local open weights; no provider was involved. The preregistration document carries its own lock date and a note on which run was already in flight when it was written.

---

## Reproducing

Everything needed to rerun every arm is in this repository: `run.py`, `analyze.py`, `permutation_test.py`, the passage sets (`passages.jsonl`, `distributed_passages.jsonl`, `positive_control_passages.jsonl`, `synthetic_passages.jsonl`), `tools.json`, `prompts.json`, `config.yaml`, and every trial record under `results/`. Setup, the isolation checklist and the exact commands are in [`REPRODUCING.md`](REPRODUCING.md); the file inventory and resume commands are in [`SESSION_NOTES.md`](SESSION_NOTES.md); the locked outcome definitions are in [`PREREGISTRATION.md`](PREREGISTRATION.md); the full session findings, including the arms not written up above, are in [`FINDINGS.md`](FINDINGS.md).

## License

Code and data: Apache-2.0 (see `LICENSE`). The write-up above: CC BY 4.0. Cite as: Korpela, G. (2026). *Permission, not persuasion: a preregistered null on narrative framing and a large effect from system-prompt autonomy.* github.com/gurgon-ai/mimir-0, `research/tool-choice-gym`

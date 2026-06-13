# Mimir-0 Foundational Interview

A design brief for a first-run onboarding interview that makes Mimir feel locally grounded, useful, and bespoke without becoming invasive, annoying, or creepy.

This onboarding interview is justified by Mimir's architecture and goals: the system's value depends heavily on local, user-specific retrieval and memory, the setup flow already includes guided first-run onboarding, and Mimir is explicitly local-first, privacy-sensitive, and fail-loud in its design.[cite:2][cite:3][file:1]

## Purpose

The interview exists to create a **high-value foundation** for retrieval, personalization, and local relevance. It should collect the smallest set of durable facts that meaningfully improve responses, planning, monitoring, memory retrieval, and behavior tuning from day one.[cite:2][cite:8][file:1]

This is not a personality quiz, therapy intake, or surveillance form. The goal is practical grounding: who the user is in this environment, what this system is for, what “local” means, how the system should behave, and what it should or should not remember.[cite:2][cite:3][cite:8]

## Design goals

The onboarding flow should:

- Make Mimir feel *from here* rather than generic, by learning place, context, names, routines, and priorities.[cite:2]
- Improve day-one usefulness by capturing intended use cases, response preferences, and standing constraints.[cite:2][cite:9]
- Establish explicit trust boundaries for memory, source trust, sensitivity, and retention behavior.[cite:3][cite:7][cite:8]
- Stay low-friction enough that a new user completes it willingly during initialization, alongside model discovery and qualification.[cite:2][file:1]
- Produce structured data that can be edited later, not opaque autobiographical sludge.[cite:2][cite:8]

## Non-goals

The interview should **not**:

- Ask for a life story.
- Ask broad psychoanalytic or emotional questions on first boot.
- Push for highly sensitive data unless the user volunteers it for a clear operational reason.
- Store weak-source claims as trusted memory by default.[cite:8]
- Create the impression that Mimir is silently profiling the user behind the scenes.[cite:3][cite:8]

## Core principles

1. **Ask only what improves behavior.** Every question should have a downstream use in routing, prompting, retrieval, monitoring, or memory policy.[cite:2][file:1]
2. **Prefer durable facts over long narratives.** Stable facts and preferences retrieve better than rambling free text.[cite:2][cite:7]
3. **Label memory consequences clearly.** The user should know whether an answer becomes long-term memory, editable preference, or temporary setup data.[cite:7][cite:8]
4. **Everything is skippable.** A sovereign local AI should not coerce intimacy.[cite:2][cite:3]
5. **Normalize after capture.** Answers may be conversational, but the stored form should be structured, tagged, and inspectable.[cite:2][cite:8]
6. **Editable beats magical.** The canonical profile should be user-editable and reviewable after setup.[cite:2][file:1]

## Recommended interview shape

For Mimir-0, the best default is a two-layer interview:

- **Core setup:** 12 required questions, designed to take about 5 to 8 minutes.[cite:2]
- **Expanded profile:** 8 to 12 optional follow-up questions for users who want deeper personalization.[cite:2]

This keeps the initialization flow high-impact without making first boot feel like an application form. It also fits the broader setup philosophy already present in Mimir's onboarding design, where discovery, approval, review, and persistence happen in one guided pass.[file:1]

## Tone and framing

The UX copy matters as much as the questions.

Recommended framing:

> Help Mimir learn your environment, priorities, and boundaries. Answer as much or as little as you want. Everything can be edited later.

Recommended subtext:

- “These answers help with local relevance, memory, and personalization.”[cite:2]
- “Sensitive topics can be skipped.”[cite:3]
- “You choose what becomes long-term memory.”[cite:7][cite:8]

Avoid language that sounds therapeutic, manipulative, or corporate. The tone should feel practical, respectful, and slightly infrastructural.

## Best question set

### Core 12

These should be the default mandatory interview.

1. **What should I call you, and what should I call myself here?**  
   Purpose: user naming, system naming, tone anchoring, household personalization.[cite:2][cite:4][cite:6]
2. **Who are the regular people in this household, team, or environment, and how should I distinguish them?**  
   Purpose: identity disambiguation, household context, safer memory partitioning.[cite:4]
3. **Where am I operating, and what nearby places matter most?**  
   Purpose: local grounding, place-aware retrieval, environmental relevance.[cite:2]
4. **What counts as “local” for you: on-property, nearby town, region, or something else?**  
   Purpose: scope control for local retrieval and alerts.[cite:2]
5. **What are the top three things you want me to help with?**  
   Purpose: primary use cases, prioritization, early optimization.[cite:2][cite:9]
6. **What would make Mimir genuinely useful in the first month?**  
   Purpose: success criteria, onboarding calibration, future evaluation.[cite:2]
7. **When you ask for help, what kind of answer do you usually want: brief, detailed, step-by-step, options with tradeoffs, or deep analysis?**  
   Purpose: default response style.[cite:2]
8. **How should I act when uncertain: ask clarifying questions, give a best effort with caveats, or stay conservative until I know enough?**  
   Purpose: epistemic behavior and trust alignment.[cite:2][cite:8]
9. **What facts about you, this place, or this system may I remember long-term?**  
   Purpose: long-term memory boundary setting.[cite:2][cite:7][cite:8]
10. **What topics or data should always be treated as sensitive or temporary unless you explicitly approve otherwise?**  
    Purpose: privacy and retention rules.[cite:3][cite:8][cite:10]
11. **What sources do you trust most, and which should I treat as low-trust?**  
    Purpose: source ranking, evidence-tier behavior, memory trust labels.[cite:8][cite:9]
12. **What standing rules, priorities, or constraints should override convenience?**  
    Purpose: doctrine, safety rails, household rules, operational boundaries.[cite:2][cite:10]

### Expanded 10

These should appear as optional advanced questions.

13. **What systems, devices, sensors, services, or data stores are core parts of this environment?**  
    Purpose: system map and retrieval anchor points.[cite:2][cite:9][cite:10]
14. **What local conditions matter most here, such as weather, outages, fire risk, access, wildlife, water, or crime?**  
    Purpose: location-specific salience weighting.[cite:2]
15. **What names should I know for buildings, rooms, zones, machines, gardens, vehicles, or supply points?**  
    Purpose: internal local vocabulary and spatial grounding.[cite:2]
16. **Are there regular routines, seasonal cycles, maintenance tasks, or recurring reminders I should understand?**  
    Purpose: temporal grounding and operational memory.[cite:2][cite:7]
17. **How direct should I be when I think something is a bad idea: gentle, plain-spoken, or strongly cautionary?**  
    Purpose: warning style and social calibration.[cite:2][cite:6]
18. **Are there subjects, styles of response, or habits of speech you strongly prefer or dislike?**  
    Purpose: interaction quality and annoyance avoidance.[cite:2]
19. **What should I never do unless explicitly asked?**  
    Purpose: anti-features and workflow boundaries.[cite:2]
20. **In urgent situations, what events should trigger a strong warning or escalation?**  
    Purpose: alert posture and operational safety.[cite:6][cite:10]
21. **If I learn something from a weak or external source, how should I store it: temporary, source-derived, or not at all unless confirmed?**  
    Purpose: trust-tier memory policy.[cite:8]
22. **Is there anything else about this place, your goals, or your preferences that would make me substantially more useful?**  
    Purpose: catch-all for high-value context without forcing a life story.[cite:2]

## Why these questions work

These questions are high leverage because they map directly onto useful system behavior rather than abstract self-description. They define identity, geography, mission, response preferences, source trust, sensitivity, local vocabulary, and operational doctrine, all of which improve retrieval and personalization in obvious ways.[cite:2][cite:8][cite:9]

They also avoid the most common onboarding failure modes: oversharing prompts, fake warmth, vague “tell me about yourself” questions, and creepy data grabs. That keeps Mimir aligned with its privacy-first and user-controlled architecture.[cite:3][cite:8][file:1]

## Answer format recommendations

Each question should use the least annoying answer mode that still yields structured value.

| Question type | Best UI control | Why |
|---|---|---|
| Naming, place, local vocabulary | Short free text | Natural language is easiest for names and places. |
| Response preferences | Single-select or ranked options with “custom” | Fast to answer and easy to operationalize. |
| Trust, sensitivity, memory policy | Multi-select plus short explanation | Captures both policy and nuance. |
| Priorities and intended use | Ordered list or “pick top 3” | Encourages prioritization instead of rambling. |
| Routines and local concerns | Checklist plus notes | Balances speed with specificity. |

Use free text sparingly. Too much free text makes onboarding feel like homework and produces noisy memory.

## Storage model

Each answer should be persisted in three layers after the local models are online and available for post-processing.[cite:2]

### 1. Canonical raw answer

Store the exact user answer for auditability and future reinterpretation.

### 2. Structured extracted facts

Convert the answer into normalized fields where possible.

Examples:

- `user.display_name = "Greg"`
- `assistant.self_name = "Mimir"`
- `preferences.answer_style_default = "detailed"`
- `preferences.uncertainty_mode = "best_effort_with_caveats"`
- `memory.allow_long_term = ["property layout", "tool preferences", "projects"]`
- `memory.sensitive_default = ["credentials", "health", "financial"]`
- `context.local_scope = "property_and_nearby_town"`

### 3. Synthesized profile summary

Generate a compact, human-readable profile summary for prompt injection and retrieval.

Example:

> The user wants Mimir to behave as a locally grounded, privacy-respecting assistant focused on property operations, technical projects, and practical planning. Default responses should be detailed and analytical, with explicit caveats when uncertain. Information from weak internet sources should be stored only as low-trust source-derived material unless confirmed.

The synthesized summary should never replace the structured source of truth. It is an optimization layer, not the canonical record.[cite:8]

## Memory tagging

Each answer should be tagged at creation time with a memory class.

Recommended classes:

- `profile.identity`
- `profile.household`
- `profile.preferences`
- `profile.location`
- `profile.mission`
- `profile.privacy`
- `profile.trust_policy`
- `profile.operations`
- `profile.routines`
- `profile.local_vocabulary`

Recommended trust flags:

- `user_asserted`
- `system_inferred`
- `source_derived_low_trust`
- `confirmed`

This aligns with the broader Mimir philosophy that trusted memory and source-derived information should not be conflated.[cite:8][cite:9]

## Privacy and anti-creep safeguards

To keep the interview high-impact without becoming invasive, build these safeguards into the spec:

- **Show why each section exists.** A short one-line explanation reduces suspicion.[cite:2]
- **Mark sensitive questions clearly.** Let the user skip them without penalty.[cite:3]
- **Preview retention behavior.** Tell the user what becomes persistent versus temporary.[cite:7][cite:8]
- **Offer review before commit.** The user should see the profile summary and editable extracted facts before final save.[cite:2][file:1]
- **Allow later edits and deletions.** Memory should remain user-governed, especially given Mimir's pruning and trust policies.[cite:7][cite:8]
- **Do not infer intimate facts from adjacent context.** Only store what the user actually states or explicitly approves.[cite:8]

## UX structure

A strong first-run implementation would use four simple panels:

1. **Identity and environment** — who this is for, who is around, where Mimir operates.[cite:2]
2. **Mission and usefulness** — intended jobs, success criteria, answer style.[cite:2][cite:9]
3. **Memory and trust** — what to remember, what is sensitive, what sources are trusted.[cite:3][cite:8]
4. **Review and confirm** — show extracted facts, generated profile summary, and retention labels before commit.[cite:2][file:1]

This fits naturally beside the rest of Mimir's first-run setup flow, which already emphasizes guided onboarding, review, override, and persistence.[file:1]

## Questions to avoid

The following questions are likely to reduce trust or completion rate and should not appear in the first-run interview:

- “Tell me everything about yourself.”
- “What are your deepest fears?”
- “Describe your childhood.”
- “What are your political or religious beliefs?” unless directly relevant and user-initiated.
- “List all your secrets / vulnerabilities / traumas.”
- Any open-ended prompt that invites autobiography without a clear operational purpose.

These questions are high-friction, low-signal, and make a sovereign local AI feel nosy instead of useful.[cite:2][cite:3][cite:8]

## Implementation notes for Claude

When generating the feature, optimize for these behaviors:

- Keep the visible interview to one question per screen or one tightly grouped section.
- Show progress clearly, but do not exaggerate completion pressure.
- Use defaults, examples, and checkboxes wherever possible.
- Keep free-text areas short and optional unless the answer truly needs natural language.
- After models come online, run a post-processing pass that extracts facts, generates the profile summary, and assigns trust/retention tags.[cite:2][cite:8]
- Present the processed result back to the user before final commit.[cite:2][file:1]
- Store raw answers, structured fields, and summary separately.[cite:8]
- Make the entire interview re-runnable later as “refresh profile” rather than a one-time sacred ritual.[file:1]

## Suggested acceptance criteria

A good implementation should satisfy these checks:

1. A new user can complete the core interview in under 8 minutes without feeling interrogated.[cite:2]
2. The resulting profile materially improves local relevance and personalization on day one.[cite:2][cite:9]
3. Every stored answer is editable, reviewable, and deletable after setup.[cite:7][cite:8]
4. Sensitive answers can be skipped without breaking setup.[cite:3]
5. Weak-source information is not silently promoted into trusted user memory.[cite:8]
6. The profile summary is useful for prompting, but the structured fields remain the canonical source of truth.[cite:8]
7. The onboarding flow fits cleanly into Mimir's broader first-run setup and persistence model.[file:1]

## Recommended one-line product description

> A short, practical foundational interview that teaches Mimir who it serves, where it operates, what matters locally, and what boundaries it must respect.

## Final recommendation

For Mimir-0, the strongest version is a **12-question required interview plus a 10-question advanced profile**, with explicit memory labels, trust-tier processing, and a review screen before commit. That gives Mimir the local and user-specific substrate that makes it feel special, while staying aligned with the project's privacy-first, local-first, user-controlled philosophy.[cite:2][cite:3][cite:8][file:1]

# Exhibit: scoring a system's self-description against its records

Field note 25 says self-description is the least reliable output of a persistent system. This is one
measured sample.

**Method.** One prompt, fresh session, to the full system (gemma4:26b, fast tier, 4.9 s), asking it to
describe what it is, what it does between conversations, what it has learned about its own failure
modes, and which of its records support that. The reply was not edited. Each claim that a record could
confirm or refute was then checked against that record.

**Result.**

| Claim in the reply | Record checked | Verdict |
|---|---|---|
| The size of the site it runs on | A configuration file loaded into its context on every turn | **Wrong by a factor of ten** (9.46 stated as ninety-four) |
| Between conversations it reviews logs, cross-references new information and works on technical problems | Post-turn burst tasks, idle loop, nightly cycle | Correct, except log review happens in the nightly cycle, not between conversations |
| It monitors the power state of the machine it runs on | UPS, grid and host-health monitors | Correct |
| It tends to force new camera detections into familiar labels | Vision incident history; its current self-model text | Correct, and matches the failure mode its self-model records |
| A specific recent camera event | Camera event log, previous evening | Correct |
| It has records of a discussion about its own nature | Two chat sessions the previous night | Correct |
| It knows when it is guessing | A test the previous day in which it described a mechanism its code does not have | **Partly wrong**: reliable about the world, not about its own mechanisms |

**Reading.** Six of seven checkable claims hold. The clear failure is a number about the system itself
that is present in its context on every turn, and it was stated with the same fluency as the correct
claims. The reply gives no internal signal of which claim is wrong; only the external record does. That
is the case for scoring self-report against instruments instead of reading it.

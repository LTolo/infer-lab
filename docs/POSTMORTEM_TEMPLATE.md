# Postmortem: <short incident title>

**Date:** · **Duration:** · **Author:** · **Severity:** SEV<n>
**Status:** draft / in review / accepted

> Blameless. The goal is to fix the system that allowed the mistake, not to
> identify who made it. If a single person's action could take the system down,
> that is a system finding, not a personal one.

## Impact
Who was affected, for how long, and how badly. Numbers, not adjectives:
requests failed, p99 latency during the window, tokens not served.

## Timeline (UTC)
| Time | Event |
|---|---|
| | first bad deploy / config change |
| | first user-visible symptom |
| | alert fired (or: *why no alert fired*) |
| | responder engaged |
| | mitigation applied |
| | full recovery confirmed |

## Detection
How did we find out? If a human noticed before the monitoring did, that is a
finding in its own right and belongs in the action items.

## Root cause
The chain of causation, not the last thing that changed. Keep asking "and why
was that possible?" until the answer is a system property rather than an event.

## Contributing factors
Conditions that made the failure possible, likely, or hard to diagnose.

## What went well
Preserve the things worth repeating — a good runbook, a clean rollback, a
metric that pointed straight at the cause.

## Where we got lucky
Things that happened to be fine this time but were not guaranteed. These are
often the most valuable action items in the whole document.

## Action items
| # | Action | Type | Owner | Due | Tracking |
|---|---|---|---|---|---|
| 1 | | prevent / detect / mitigate | | | |

Every action item must be assigned and dated. An unowned action item is a wish.

## Inference-specific prompts
Common root causes in this system — check them explicitly:
- KV cache saturation leading to sustained preemption (`infer_lab_preemptions_total`)
- Admission control set above sustainable memory capacity
- A collective blocked on a dead or straggling peer (`distributed/health.py`)
- Numeric instability from a precision or weight change (`debug/instability.py`)
- Prefix-cache eviction thrashing under a changed traffic mix
- A long prompt monopolising steps because chunked prefill was disabled

# Engineering principles

These are the rules this codebase actually follows. They are written down
because a principle that is not written down is a preference.

## 1. Correctness before performance — and correctness means *equivalence*
Every optimisation here is only legitimate if the system behaves as if the
optimisation were not there. That is why the test suite asserts equivalence
rather than "reasonable output":

- batched decode == sequential decode (`test_batched_results_equal_sequential_results`)
- chunked prefill == one-shot prefill (`test_chunked_prefill_matches_unchunked_output`)
- prefix-cached == uncached (`test_prefix_cache_does_not_change_output`)
- preempted == never-preempted (`test_preemption_under_memory_pressure_preserves_output`)
- tiled attention == naive attention (`test_flash_attention_equals_naive_causal`)

A speedup that changes the output is not a speedup; it is a regression with good
marketing.

## 2. Measure the tail, not the mean
Means hide queueing. Every latency report here carries p50/p90/p99, and the
`p99/p50` ratio is treated as a first-class health indicator. If the mean looks
fine and the p99 is 8× it, the system is queueing and the mean is lying.

## 3. Degrade, don't crash
No GPU, no compiler, no .NET, no PyTorch — the project still runs and the suite
still passes. Optional components report *why* they are unavailable
(`kernels/registry.py`) instead of raising at import time. A tool that only
works on the author's machine is not a tool.

## 4. Say what is real and what is modelled
`docs/JOB_MAPPING.md` labels every capability Implemented / Modelled /
Reference. Simulation is legitimate; undisclosed simulation is not.

## 5. Failures should be diagnosable, not just detectable
A dead peer in a collective does not raise — it hangs, forever, silently. So
`distributed/health.py` names the culprit, and `debug/instability.py` names the
*first* layer where numerics went wrong, because everything downstream is
collateral damage.

## 6. Every alert states why it exists
`observability/alerts.yml` annotates each rule with the operational lesson
behind it. An alert nobody can act on gets muted, and then the one that mattered
gets muted with it.

## 7. Decisions get recorded with their trade-off
ADRs document what was rejected and why. ADR-0006 exists because a real bug
forced a design change; writing down the bug is more useful than writing down
only the fix.

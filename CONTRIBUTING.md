# Contributing

## Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate

pip install -e ".[dev]"        # core + test tooling
pip install -e ".[all]"        # everything optional (torch, bindings, plots)
```

`uv` works equally well and is faster:

```bash
uv venv && uv pip install -e ".[all]"
```

## The only two commands you need

```bash
python scripts/verify.py       # prove the whole project runs error-free
python scripts/run_stack.py    # start the app and all its dependencies
```

`verify.py` must exit 0 before you open a pull request. SKIP results are fine —
they mean an optional toolchain is absent on your machine. FAIL results are not.

## Review checklist

Correctness
- [ ] Does an existing equivalence test still cover this path? If the change
      touches scheduling, paging or kernels, an equivalence test is mandatory.
- [ ] Are new numeric kernels verified against the NumPy reference?
- [ ] Does the KV allocator end the test with every block returned?

Performance
- [ ] `python -m infer_lab.cli bench --track` — no regression beyond tolerance.
- [ ] If this is a performance change, is there a number in the PR description?
      "Feels faster" is not a measurement.

Operability
- [ ] New failure modes: are they observable (metric, log field, or alert)?
- [ ] Do new log fields use the structured `extra={}` form, not string interpolation?
- [ ] Does the change degrade gracefully when an optional dependency is missing?

Design
- [ ] Non-obvious trade-off? Add an ADR in `docs/adr/` rather than a code comment
      that will be read once.

## Pull request template

```
## What
One paragraph. What changed, not how.

## Why
The problem this solves. If it is a performance change, the measurement.

## Trade-offs
What got worse. If genuinely nothing, say why the change is free.

## Verification
- [ ] python scripts/verify.py passes
- [ ] equivalence tests added/updated for touched paths
- [ ] benchmark delta: <number>
```

## Style

- Ruff enforces formatting and imports: `ruff check src tests`.
- Comments explain **why**, never **what**. If a comment restates the code,
  delete it.
- Public functions carry type hints and a docstring that states the contract,
  including failure behaviour.

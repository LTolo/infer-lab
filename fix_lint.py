#!/usr/bin/env python
"""infer-lab: fix the remaining lint violations and make verify.py run the linter.

Run from the project root:

    py -3.14 fix_lint.py

Context
-------
CI ran six `verify` jobs and all six failed, while `performance regression`
passed. The only difference between those jobs is a `ruff check` step. Locally
`verify.py` printed ALL CHECKS PASSED at the same time.

That gap is the real defect. A verification script that runs a weaker check than
the pipeline gives false confidence -- the same shape of problem as ADR-0009,
where two mechanisms answered the same question differently.

`ruff --fix` already handled 44 of 47 findings. The nine below are left because
ruff classes them as behaviour-changing, and it is right to: they are not style
issues.

What this changes
-----------------
1. **`B905` -- `zip()` without `strict=`** (7 sites). `zip()` silently stops at
   the shorter iterable. In `llm_engine.py` that runs over
   `zip(decodes, batch_logits)`: if those lengths ever diverged, a sequence
   would lose its token with no error anywhere. Every site here pairs sequences
   that are equal *by construction*, so `strict=True` costs nothing and turns an
   unstated invariant into an enforced one.

2. **`E741` -- ambiguous name `l`** (2 sites). `l` is the FlashAttention paper's
   notation for the running softmax denominator, alongside `m` for the running
   max. Renaming to `l_sum` keeps that link while removing the `1`/`I`
   ambiguity.

3. **`verify.py` runs `ruff`**, so local and CI answer the same question.

Anchored edits; a non-matching anchor is reported and skipped. Backups are
written next to each file as `*.bak_lint`.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

GREEN, RED, YELLOW, RESET = ("\033[32m", "\033[31m", "\033[33m", "\033[0m")
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7)
    except Exception:  # noqa: BLE001
        GREEN = RED = YELLOW = RESET = ""

results: list[tuple[str, bool, str]] = []
touched: set[Path] = set()


def edit(relative: str, old: str, new: str, label: str) -> None:
    """Apply one anchored replacement to one file."""
    path = ROOT / relative
    if not path.exists():
        results.append((label, False, f"file not found: {relative}"))
        return

    text = path.read_text(encoding="utf-8")

    if old not in text and new in text:
        results.append((label, True, "already applied"))
        return

    count = text.count(old)
    if count != 1:
        results.append((label, False,
                        "anchor not found" if count == 0
                        else f"anchor ambiguous ({count} matches)"))
        return

    if path not in touched:
        backup = path.with_suffix(path.suffix + ".bak_lint")
        if not backup.exists():
            shutil.copy2(path, backup)
        touched.add(path)

    path.write_text(text.replace(old, new), encoding="utf-8")
    results.append((label, True, "patched"))


# ===========================================================================
# 1. B905 -- zip(..., strict=True)
# ===========================================================================
edit(
    "src/infer_lab/engine/llm_engine.py",
    "            for request, logits in zip(decodes, batch_logits):",
    "            # strict=True: one logit row per decoding sequence is an\n"
    "            # invariant of forward_decode_batch. Plain zip() would drop a\n"
    "            # sequence's token silently if that ever stopped holding.\n"
    "            for request, logits in zip(decodes, batch_logits, strict=True):",
    "llm_engine: zip(decodes, batch_logits) strict",
)

edit(
    "src/infer_lab/distributed/ring_allreduce.py",
    "    stuck = [n.rank for n, t in zip(nodes, threads) if t.is_alive()]",
    "    stuck = [n.rank for n, t in zip(nodes, threads, strict=True) if t.is_alive()]",
    "ring_allreduce: zip(nodes, threads) strict",
)

edit(
    "src/infer_lab/server/metrics.py",
    "        for upper, c in zip(self.buckets, self.counts):",
    "        for upper, c in zip(self.buckets, self.counts, strict=True):",
    "metrics: zip(buckets, counts) strict",
)

edit(
    "src/infer_lab/server/metrics.py",
    '    pairs = ",".join(f\'{n}="{v}"\' for n, v in zip(names, values))',
    '    pairs = ",".join(f\'{n}="{v}"\' for n, v in zip(names, values, strict=True))',
    "metrics: zip(names, values) strict",
)

edit(
    "scripts/verify.py",
    "        same = all(a.output_token_ids == b.output_token_ids "
    "for a, b in zip(outputs, again))",
    "        same = all(a.output_token_ids == b.output_token_ids\n"
    "                   for a, b in zip(outputs, again, strict=True))",
    "verify: zip(outputs, again) strict",
)

edit(
    "tests/test_engine.py",
    "    for prompt, out in zip(prompts, batched):",
    "    for prompt, out in zip(prompts, batched, strict=True):",
    "test_engine: zip(prompts, batched) strict",
)

edit(
    "tests/test_engine.py",
    "    for a, b in zip(tight_out, roomy_out):",
    "    for a, b in zip(tight_out, roomy_out, strict=True):",
    "test_engine: zip(tight_out, roomy_out) strict",
)


# ===========================================================================
# 2. E741 -- rename the ambiguous `l`
# ===========================================================================
edit(
    "src/infer_lab/kernels/numpy_kernels.py",
    "        l = np.zeros((q1 - q0, h), dtype=np.float32)          # running sum",
    "        l_sum = np.zeros((q1 - q0, h), dtype=np.float32)      # running sum "
    "(`l` in the paper)",
    "numpy_kernels: rename l -> l_sum (init)",
)

edit(
    "src/infer_lab/kernels/numpy_kernels.py",
    "            l = l * alpha + np.sum(p, axis=-1)",
    "            l_sum = l_sum * alpha + np.sum(p, axis=-1)",
    "numpy_kernels: rename l -> l_sum (accumulate)",
)

edit(
    "src/infer_lab/kernels/numpy_kernels.py",
    "        denom = np.where(l > 0, l, 1.0)",
    "        denom = np.where(l_sum > 0, l_sum, 1.0)",
    "numpy_kernels: rename l -> l_sum (normalise)",
)


# ===========================================================================
# 3. verify.py runs ruff
# ===========================================================================
edit(
    "scripts/verify.py",
    '''def stage_tests(extra: list[str]) -> None:
    header("4/7  test suite")
    env = dict(os.environ, PYTHONPATH=str(SRC))''',
    '''def stage_tests(extra: list[str]) -> None:
    header("4/7  lint and test suite")
    env = dict(os.environ, PYTHONPATH=str(SRC))

    # Run the linter here, with the same invocation CI uses.
    #
    # Without this, verify.py can print ALL CHECKS PASSED while the pipeline is
    # red -- which is exactly what happened: six CI jobs failed on a ruff step
    # this script never ran. A verification script that checks less than the
    # pipeline does not verify anything useful.
    lint = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"],
        cwd=ROOT, capture_output=True, text=True, env=env, timeout=300,
    )
    if lint.returncode == 0:
        record("ruff", PASS, "no violations")
    elif "No module named" in (lint.stderr or ""):
        record("ruff", SKIP, "ruff not installed (pip install -e \\".[dev]\\")")
    else:
        summary = [ln for ln in lint.stdout.splitlines() if ln.startswith("Found")]
        record("ruff", FAIL, summary[0] if summary else "violations found")
        print(DIM + lint.stdout[-2500:] + RESET)''',
    "verify.py: run ruff alongside the tests",
)


# ===========================================================================
# report
# ===========================================================================
def main() -> int:
    print("infer-lab: lint fixes + verify.py parity with CI")
    print(f"  project: {ROOT}\n")

    for label, ok, detail in results:
        mark = f"{GREEN}OK  {RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  [{mark}] {label} -- {detail}")

    failed = [r for r in results if not r[1]]
    if failed:
        print(f"\n{RED}{len(failed)} edit(s) could not be applied.{RESET}")
        print("Those files were left untouched. Send me the output and I will re-anchor.")
        return 1

    print(f"\n  backups written as *.bak_lint\n")
    print("Running ruff...\n")
    lint = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"], cwd=ROOT)

    if lint.returncode != 0:
        print(f"\n{YELLOW}ruff still reports violations -- send me the output.{RESET}")
        return 1

    print(f"\n{GREEN}ruff is clean.{RESET}\n")
    print("Running verification...\n")
    proc = subprocess.run([sys.executable, "scripts/verify.py"], cwd=ROOT)

    if proc.returncode == 0:
        print(f"\n{GREEN}Local and CI now check the same thing.{RESET}")
        print("\nNext:")
        print('  git add -A')
        print('  git commit -m "Fix lint violations; verify.py now runs ruff"')
        print("  git push")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

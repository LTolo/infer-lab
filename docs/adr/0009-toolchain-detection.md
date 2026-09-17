# ADR-0009: A build must verify its own output, and own its environment

**Status:** Accepted · **Date:** 2026-09 · **Related:** ADR-0008

## Context

Three native kernel bindings (ctypes, pybind11, nanobind) have to be compiled on
whatever machine the project is checked out on. The first version of
`kernels/build.py` delegated that responsibility to the user: find the right
shell, make sure `cl.exe` is on `PATH`, then run the build.

That assumption broke in four distinct ways, each one hidden by the one before
it. The sequence is the interesting part, not any individual bug.

## The failures, in the order they became visible

### 1. The build reported success for an unloadable artifact

Windows offers several near-identical Start-menu shortcuts. The plain *Developer
PowerShell* loads the **x86** toolchain; the interpreter was **x64**. Two of the
three targets then failed to link with `LNK4272` — a clear, if verbose, error.

The third did not. The ctypes target links against no Python library, so
compiling it as x86 **succeeds**. The build printed:

```
[OK  ] ctypes: ok -> libinfer_lab_kernels.dll
```

and the DLL failed at import with a bare `WinError 193: %1 ist keine zulässige
Win32-Anwendung`. A previously working x64 artifact had been silently replaced
with a broken one, by a step that reported success.

This is the worst failure mode in the list, because it manufactures confidence.
An honest failure sends you to the right place; a false `OK` sends you
everywhere else first.

### 2. The environment probe threw away its own diagnosis

The next version located Visual Studio via `vswhere` and ran `vcvarsall.bat`
itself. It reported:

```
vcvars: FAILED -- vcvarsall.bat x64 failed. Is the C++ workload installed?
```

The workload was installed. Running the same batch file by hand worked and found
`cl.exe`. Two mistakes compounded:

- The command was passed to `subprocess` as an **argument list**. On Windows that
  list is converted back into a command line by `list2cmdline()`, which escapes
  embedded quotes as `\"` — a convention `cmd.exe` does not use. The quoted
  `vcvarsall.bat` path arrived mangled.
- The command redirected vcvars' output to `nul`, discarding the error message
  that would have explained the first point.

A diagnostic that discards the diagnosis is worse than none: it sends you to
inspect a toolchain when the defect is in the probe.

### 3. A dict is case-sensitive; the Windows environment is not

With the invocation fixed, vcvars ran and reported `x64` — and the very next line
said `cl.exe: None`.

`os.environ` normalises the key to `PATH`. The `set` output of `vcvarsall` spells
it `Path`. Parsing that output into a plain dict therefore *added a second key*
instead of replacing the first:

```
PATH -> C:\old;C:\windows            <- stale, no compiler
Path -> C:\VC\bin\Hostx64\x64;...    <- what vcvars actually set
```

`env.get("PATH")` returned the stale value. Handing that dict to `subprocess` is
worse than merely wrong: an environment block containing both spellings is
ambiguous, and which one the child sees is not something to depend on.

### 4. `subprocess` resolves the program against the *parent* `PATH`

With the merge fixed, the compiler was found — and the build failed one step
later with `WinError 2`.

`detect_compiler` returned the bare name `"cl"` together with the vcvars
environment. On Windows the program name is resolved against the `PATH` of the
**calling** process; the `PATH` inside the `env` you pass is used by the child
*once it is running*, and is not consulted to locate the executable. So `cl` was
searched in the plain PowerShell `PATH`, where it does not exist.

The vcvars environment is still required — MSVC needs `INCLUDE` and `LIB` — but
the executable itself must be named by absolute path.

### 5. Two mechanisms answering the same question

Finally, stage 2 of `scripts/verify.py` built all three bindings successfully
while stage 7, in the same process, reported:

```
[SKIP] C latency probe -- no C compiler
```

Stage 2 called `build.detect_compiler()`. Stage 7 did its own
`shutil.which("cl")`. Duplicated logic is how two parts of one program end up
disagreeing about a fact as basic as whether a compiler exists.

## Decision

**1 · Verify every artifact against the interpreter that must load it.**
After each build, read the PE/ELF/Mach-O machine type of the file just produced
and compare it to `sysconfig.get_platform()`. On mismatch: report a failure and
**delete** the artifact. Never leave an unloadable binary on disk for the next
run to trip over.

**2 · Own the environment instead of documenting it.**
Locate Visual Studio through `vswhere`, run `vcvarsall.bat` for the interpreter's
architecture, and import the resulting environment. Which shell the user opened
is no longer part of the contract.

**3 · Never discard the output of a probe.**
The marker line already separates vcvars' banner from the environment dump, so
the redirection bought nothing and cost the error message.

**4 · Merge environments case-insensitively; address executables absolutely.**
Both are Windows-specific facts that a `dict` and a bare program name silently
get wrong.

**5 · One detector, one answer.**
`verify.py` asks `build.detect_compiler()` rather than looking a compiler up
again. The fix is not to duplicate the logic correctly — it is to stop
duplicating it.

**6 · Ship a `--doctor` subcommand.**
Report the interpreter architecture, the located install, the vcvars target arch,
the resolved compiler path, available packages, and every existing artifact with
its architecture and whether it is loadable. Every one of the bugs above would
have been a single line of output instead of an inference.

## Consequences

`python scripts/verify.py` now reports **0 skipped** from an ordinary PowerShell
prompt, with all three bindings built as `cp314-win_amd64` and header-verified.
The C client, which shares the same detector, builds with it.

The cost is that `build.py` grew from ~200 to ~480 lines, most of it platform
handling rather than compilation. That is the honest price of a build that works
on a machine other than the author's.

## The generalisation

Every one of these bugs was in the *build tooling*, never in the user's Visual
Studio installation — which was correct throughout and was suspected repeatedly.

The pattern is the same one recorded in ADR-0008: **each failure only became
visible after the previous one was fixed.** A false `OK` hid a linker error; a
swallowed stderr hid a quoting bug; a case-sensitive dict hid a path-resolution
bug; a duplicated detector hid the fact that detection already worked.

Two rules follow, and they apply well beyond Windows toolchains:

- **A step that reports success must have checked the thing it claims succeeded.**
  Compiling without error is not the same as producing a usable artifact.
- **Diagnostics must survive contact with failure.** Suppressing output on the
  path where things go wrong is precisely backwards.

The same reasoning applied to the engine gives ADR-0008: correctness tests verify
*what* is computed and say nothing about whether progress is still being made.
Here, a successful exit code says nothing about whether the output can be loaded.
In both cases the check that mattered was one nobody had written yet.


---

## Epilogue: two more, found after this record was written

Publishing the repository surfaced two further instances of the same pattern.
Both are recorded here rather than in new ADRs, because neither is a new idea --
they are the same mistake at a different scale.

### 6. Local verification checked less than the pipeline

CI ran six `verify` jobs; all six failed. `performance regression` passed. The
only difference between those jobs is a `ruff check` step.

Meanwhile `scripts/verify.py` printed **ALL CHECKS PASSED** on the same commit.

The script ran the tests but not the linter, so "the project verifies" meant
something weaker locally than it did in CI. Forty-seven lint findings were
invisible to the tool whose entire purpose is to say whether the project is in
a shippable state.

Seven of those findings were `B905` — `zip()` without `strict=`. That is not
cosmetic. In `llm_engine.py` the call is `zip(decodes, batch_logits)`: if those
lengths ever diverged, a sequence would lose its token silently, with no error
anywhere. Exactly the failure shape ADR-0008 is about.

**Rule: local verification must be at least as strict as the pipeline.** A
check that passes locally and fails remotely is not a flaky pipeline; it is a
local check that was never doing the job.

### 7. Unifying two detectors answered a question neither caller had asked

Fix 5 above replaced `verify.py`'s own compiler lookup with a call to
`build.detect_compiler()`. Correct — and it introduced a new failure:

```
clang++: warning: treating 'c' input as 'c++' when in C++ mode
error: invalid argument '-std=c11' not allowed with 'C++'
```

`detect_compiler()` exists to find a compiler for the **C++ kernels**, so it
returns a C++ driver. `verify.py` used it to build `clients/c/probe.c` — a C
file — and passed `-std=c11`.

GCC treats a C-only flag on the C++ driver as a warning and proceeds. Clang
makes it an error. Same wrong command, two verdicts: Ubuntu and Windows green,
macOS red.

**Rule: a shared answer is only safe when both callers are asking the same
question.** "Which compiler exists" and "which compiler compiles C" are not the
same question, and deduplication that conflates them trades a visible
disagreement for a hidden one.

That the disagreement showed up at all is the argument for a multi-platform
matrix. On a single platform this would have compiled, with a warning nobody
reads, and stayed wrong.

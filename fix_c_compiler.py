#!/usr/bin/env python
"""infer-lab: compile the C client with a C compiler, not a C++ driver.

Run from the project root:

    py -3.14 fix_c_compiler.py

The bug
-------
macOS CI fails with:

    clang++: warning: treating 'c' input as 'c++' when in C++ mode
    error: invalid argument '-std=c11' not allowed with 'C++'

`detect_compiler()` exists to find a compiler for the **C++ kernels**, so it
returns a C++ driver (`c++`, `clang++`, `g++`). verify.py then reuses it to
build `clients/c/probe.c` -- a C file -- and passes `-std=c11`.

GCC treats a C-only flag on the C++ driver as a warning and carries on. Clang
makes it a hard error. That is the entire reason Ubuntu passed while macOS
failed: the same wrong command, judged differently by two compilers.

This is a regression introduced by the previous fix. Unifying compiler
detection was right -- two detectors had been disagreeing about whether a
compiler existed at all -- but "which compiler exists" and "which compiler
compiles C" are not the same question, and one answer cannot serve both.

The fix
-------
For the C client, look for a genuine C compiler (`cc`, `clang`, `gcc`) and use
`-std=c11` with it. If only a C++ driver is available, fall back to it and drop
the C-only flag rather than failing. MSVC is unaffected: `cl` compiles C by file
extension and never received `-std=c11`.

Anchored edit; a non-matching anchor is reported and the file is left untouched.
Backup written as `verify.py.bak_cc`.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VERIFY = ROOT / "scripts" / "verify.py"

GREEN, RED, YELLOW, RESET = ("\033[32m", "\033[31m", "\033[33m", "\033[0m")
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7)
    except Exception:  # noqa: BLE001
        GREEN = RED = YELLOW = RESET = ""


OLD = '''    elif kind == "unix":
        proc = run([cc, "-O2", "-std=c11", source, "-o", str(build / "probe")],
                   env=env)
        record("C latency probe", PASS if proc.returncode == 0 else FAIL,
               f"compiled with {Path(cc).name}"
               if proc.returncode == 0 else proc.stderr[-300:])'''

NEW = '''    elif kind == "unix":
        # probe.c is C; detect_compiler() returns a C++ driver, because its job
        # is finding a compiler for the C++ kernels. Handing `-std=c11` to a C++
        # driver is a warning on GCC and a hard error on Clang -- which is why
        # Ubuntu passed while macOS failed on the identical command.
        #
        # "Which compiler exists" and "which compiler compiles C" are different
        # questions; one answer cannot serve both.
        c_compiler = None
        for candidate in ("cc", "clang", "gcc"):
            c_compiler = shutil.which(candidate)
            if c_compiler:
                break

        if c_compiler:
            cmd = [c_compiler, "-O2", "-std=c11", source, "-o", str(build / "probe")]
        else:
            # Only a C++ driver available: let it compile the C file, but drop
            # the C-only flag it would reject.
            cmd = [cc, "-O2", source, "-o", str(build / "probe")]

        proc = run(cmd, env=env)
        record("C latency probe", PASS if proc.returncode == 0 else FAIL,
               f"compiled with {Path(cmd[0]).name}"
               if proc.returncode == 0 else (proc.stderr or proc.stdout)[-300:])'''


def main() -> int:
    print("infer-lab: use a C compiler for the C client")
    print(f"  file: {VERIFY}\n")

    if not VERIFY.exists():
        print(f"  [{RED}FAIL{RESET}] verify.py not found -- run this from the project root")
        return 1

    text = VERIFY.read_text(encoding="utf-8")

    if "which compiler compiles C" in text:
        print(f"  [{GREEN}OK  {RESET}] already applied")
    elif text.count(OLD) != 1:
        print(f"  [{RED}FAIL{RESET}] anchor not found -- verify.py differs from expected")
        print("  File left untouched. Send me the output and I will re-anchor.")
        return 1
    else:
        backup = VERIFY.with_suffix(".py.bak_cc")
        if not backup.exists():
            shutil.copy2(VERIFY, backup)
        VERIFY.write_text(text.replace(OLD, NEW), encoding="utf-8")
        print(f"  [{GREEN}OK  {RESET}] patched  (backup: {backup.name})")

    print("\nRunning ruff...\n")
    lint = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"], cwd=ROOT)
    if lint.returncode != 0:
        print(f"\n{YELLOW}ruff reports violations -- send me the output.{RESET}")
        return 1

    print("\nRunning verification...\n")
    proc = subprocess.run([sys.executable, "scripts/verify.py"], cwd=ROOT)

    if proc.returncode == 0:
        print(f"\n{GREEN}Verification passes.{RESET}")
        print("\nNext:")
        print('  git add -A')
        print('  git commit -m "Build the C client with a C compiler, not a C++ driver"')
        print("  git push")
        print("  gh run watch")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

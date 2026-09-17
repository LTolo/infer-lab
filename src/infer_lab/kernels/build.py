"""Build the three native extension variants in-place.

Usage:
    python -m infer_lab.kernels.build            # build everything that is possible
    python -m infer_lab.kernels.build --only ctypes
    python -m infer_lab.kernels.build --doctor   # report the toolchain, build nothing
    python -m infer_lab.kernels.build --clean -v # wipe artifacts, rebuild verbosely

Design notes
------------
* **The build configures its own compiler environment.** On Windows it locates
  Visual Studio through ``vswhere``, runs ``vcvarsall.bat x64`` and imports the
  resulting environment.  Requiring the user to remember which of several
  near-identical Start-menu shortcuts loads the x64 toolchain is a defect, not a
  documentation problem: the x86 shortcut produces a build that either fails to
  link or -- worse -- succeeds and yields an unloadable artifact.
* **Every artifact is verified after it is produced.**  We read the PE/ELF/Mach-O
  header of the file we just built and compare its machine type against the
  running interpreter.  A build that reports success for a binary Python cannot
  load is worse than an honest failure, because it hides the problem until some
  later, more confusing moment.
* We invoke the compiler directly rather than going through
  ``setup.py build_ext`` so the same code path works on Windows, Linux and macOS,
  and so a missing pybind11/nanobind degrades to a *skip* instead of an import
  error at package-import time.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSRC = HERE / "csrc"
IS_WINDOWS = os.name == "nt"
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or (".pyd" if IS_WINDOWS else ".so")
SHARED_SUFFIX = ".dll" if IS_WINDOWS else (".dylib" if sys.platform == "darwin" else ".so")


class BuildError(RuntimeError):
    """A build step could not be completed. Carries a human-actionable message."""


# ===========================================================================
# architecture identity
# ===========================================================================
# PE machine ids (winnt.h), ELF e_machine ids, Mach-O cputypes.
PE_MACHINE = {0x014C: "x86", 0x8664: "x64", 0xAA64: "arm64", 0x01C4: "arm"}
ELF_MACHINE = {0x03: "x86", 0x3E: "x64", 0xB7: "arm64", 0x28: "arm"}
MACHO_CPU = {7: "x86", 0x01000007: "x64", 0x0100000C: "arm64"}


def python_arch() -> str:
    """The architecture the running interpreter was built for."""
    machine = (sysconfig.get_platform() or "").lower()
    if "amd64" in machine or "x86_64" in machine:
        return "x64"
    if "arm64" in machine or "aarch64" in machine:
        return "arm64"
    if "win32" in machine or "i686" in machine or "i386" in machine:
        return "x86"
    return "unknown"


def binary_arch(path: Path) -> str:
    """Read the machine type out of a compiled artifact.

    Returns one of ``x86``/``x64``/``arm64``/``arm``, or ``unknown`` if the file
    cannot be identified. Never raises -- an unreadable header must not be able
    to break the build, only to leave it unverified.
    """
    try:
        with path.open("rb") as handle:
            head = handle.read(4)

            if head[:2] == b"MZ":                                   # PE (Windows)
                handle.seek(0x3C)
                pe_offset = struct.unpack("<I", handle.read(4))[0]
                handle.seek(pe_offset)
                if handle.read(4) != b"PE\0\0":
                    return "unknown"
                return PE_MACHINE.get(struct.unpack("<H", handle.read(2))[0], "unknown")

            if head == b"\x7fELF":                                  # ELF (Linux)
                handle.seek(18)
                return ELF_MACHINE.get(struct.unpack("<H", handle.read(2))[0], "unknown")

            if head in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe"):   # Mach-O (macOS)
                return MACHO_CPU.get(struct.unpack("<I", handle.read(4))[0], "unknown")
    except OSError:
        pass
    return "unknown"


def verify_artifact(path: Path) -> None:
    """Fail loudly when the produced binary cannot be loaded by this Python.

    This is the check whose absence caused a silent regression: the ctypes
    target links against no Python library, so an x86 build of it *succeeds*
    and reports OK -- then fails at import with a bare WinError 193.
    """
    if not path.exists():
        raise BuildError(f"compiler reported success but {path.name} was not produced")

    expected = python_arch()
    actual = binary_arch(path)
    if actual == "unknown" or expected == "unknown":
        return  # cannot verify; do not block the build on that
    if actual != expected:
        path.unlink(missing_ok=True)   # never leave an unloadable artifact behind
        raise BuildError(
            f"architecture mismatch: built {actual}, but this Python is {expected}. "
            f"The unusable artifact was deleted. "
            + ("Open the 'x64 Native Tools Command Prompt for VS' and rebuild."
               if IS_WINDOWS else "Check your compiler's target architecture.")
        )


# ===========================================================================
# Windows: find and import the MSVC environment ourselves
# ===========================================================================
_MSVC_ENV: dict[str, str] | None = None
_MSVC_ENV_ERROR: str | None = None

VSWHERE_CANDIDATES = [
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    / "Microsoft Visual Studio" / "Installer" / "vswhere.exe",
    Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    / "Microsoft Visual Studio" / "Installer" / "vswhere.exe",
]


def find_vs_install() -> Path | None:
    """Locate any Visual Studio install carrying the C++ toolset."""
    vswhere = next((p for p in VSWHERE_CANDIDATES if p.exists()), None)
    if vswhere is None:
        return None
    try:
        proc = subprocess.run(
            [str(vswhere), "-latest", "-products", "*",
             "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
             "-format", "json", "-utf8"],
            capture_output=True, text=True, timeout=60,
        )
        entries = json.loads(proc.stdout or "[]")
        if entries:
            return Path(entries[0]["installationPath"])
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def msvc_env(arch: str | None = None) -> dict[str, str]:
    """Return the environment produced by ``vcvarsall.bat <arch>``.

    Cached: the batch file is slow and the answer does not change within a run.
    """
    global _MSVC_ENV, _MSVC_ENV_ERROR
    if _MSVC_ENV is not None:
        return _MSVC_ENV
    if _MSVC_ENV_ERROR is not None:
        raise BuildError(_MSVC_ENV_ERROR)

    arch = arch or {"x64": "x64", "x86": "x86", "arm64": "arm64"}.get(python_arch(), "x64")

    install = find_vs_install()
    if install is None:
        _MSVC_ENV_ERROR = (
            "Visual Studio with the C++ workload was not found. Install "
            "'Build Tools for Visual Studio' and select 'Desktop development with C++'."
        )
        raise BuildError(_MSVC_ENV_ERROR)

    vcvars = install / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
    if not vcvars.exists():
        _MSVC_ENV_ERROR = f"vcvarsall.bat not found under {install}"
        raise BuildError(_MSVC_ENV_ERROR)

    # Run the batch file, then dump the environment it produced. The marker keeps
    # vcvars' own banner out of the parsed output.
    marker = "___INFER_LAB_ENV___"

    # This MUST be a single raw command string, not an argument list.
    #
    # Passing a list makes subprocess run it through list2cmdline(), which
    # escapes embedded quotes as \" -- a convention cmd.exe does not use. The
    # quoted vcvarsall path then arrives mangled and the call fails for reasons
    # that look like a broken Visual Studio install. This is the same idiom
    # CPython's own MSVC support uses.
    #
    # vcvars' output is deliberately NOT redirected to nul: the marker below
    # already separates its banner from the environment dump, and suppressing it
    # would also suppress the error message on failure.
    command = f'cmd.exe /c "{vcvars}" {arch} && echo {marker} && set'
    proc = subprocess.run(command, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=180)
    if marker not in proc.stdout:
        detail = (proc.stdout or "").strip() + "\n" + (proc.stderr or "").strip()
        _MSVC_ENV_ERROR = (
            f"vcvarsall.bat {arch} did not complete.\n"
            f"  command : {command}\n"
            f"  exitcode: {proc.returncode}\n"
            f"  output  : {detail.strip()[-700:] or '(none)'}"
        )
        raise BuildError(_MSVC_ENV_ERROR)

    # Merge case-insensitively.
    #
    # Windows environment variables ignore case; a Python dict does not.
    # os.environ normalises the key to PATH, vcvars' `set` output spells it
    # Path. A naive dict update therefore ADDS a second key instead of
    # replacing the first, leaving the stale PATH -- the one without the
    # compiler -- as what env.get("PATH") returns. Passing a block containing
    # both spellings to subprocess is ambiguous on top of being wrong.
    env = dict(os.environ)
    canonical = {key.upper(): key for key in env}
    for line in proc.stdout.split(marker, 1)[1].splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not key:
            continue
        existing = canonical.get(key.upper())
        if existing is not None:
            env[existing] = value
        else:
            env[key] = value
            canonical[key.upper()] = key

    _MSVC_ENV = env
    return env


# ===========================================================================
# compiler discovery
# ===========================================================================
def env_path(env: dict[str, str] | None) -> str | None:
    """Read the search path without assuming how it is capitalised.

    See msvc_env() for why a plain dict cannot be trusted to use one spelling.
    """
    if not env:
        return None
    for key, value in env.items():
        if key.upper() == "PATH":
            return value
    return None


def find_cl_in_install(install: Path) -> Path | None:
    """Locate cl.exe directly under the VS install root.

    A fallback for when the PATH lookup fails: the compiler's location is
    derivable from the install root, so an unexpected environment variable is
    not a good enough reason to declare the toolchain missing. Newest toolset
    first, host architecture matched to this interpreter.
    """
    host = "Hostx64" if python_arch() in ("x64", "arm64") else "Hostx86"
    target = {"x64": "x64", "x86": "x86", "arm64": "arm64"}.get(python_arch(), "x64")
    tools = install / "VC" / "Tools" / "MSVC"
    if not tools.is_dir():
        return None
    for version in sorted(tools.iterdir(), reverse=True):
        candidate = version / "bin" / host / target / "cl.exe"
        if candidate.is_file():
            return candidate
    return None


def detect_compiler() -> tuple[str, str, dict[str, str] | None]:
    """Return ``(kind, executable, env)``. ``kind`` is 'msvc' or 'unix'.

    On Windows we prefer MSVC and configure its environment ourselves, so the
    build no longer depends on which shell the user happened to open.
    """
    if IS_WINDOWS:
        try:
            env = msvc_env()
            # Resolve to an ABSOLUTE path, never the bare name "cl".
            #
            # subprocess looks the program up against the PATH of the *calling*
            # process. The PATH inside the env we pass is used by the child once
            # it is running; it is not consulted to locate the executable. A bare
            # "cl" is therefore searched in the plain PowerShell PATH, is not
            # found there, and CreateProcess fails with WinError 2 -- even though
            # vcvars succeeded and the compiler exists.
            resolved = shutil.which("cl", path=env_path(env))
            if resolved:
                return "msvc", str(Path(resolved).resolve()), env
            # PATH lookup failed, but vcvars succeeded -- find the compiler by
            # its known location rather than declaring the toolchain missing.
            install = find_vs_install()
            found = find_cl_in_install(install) if install else None
            if found is not None:
                return "msvc", str(found), env
        except BuildError:
            pass
        prompt_cl = shutil.which("cl")              # already inside a VS prompt
        if prompt_cl:
            return "msvc", str(Path(prompt_cl).resolve()), None
        for alt in ("clang++", "g++"):
            if shutil.which(alt):
                return "unix", alt, None
        raise BuildError(
            "no C++ compiler found. Install 'Build Tools for Visual Studio' with "
            "the 'Desktop development with C++' workload."
        )

    for alt in ("c++", "g++", "clang++"):
        found = shutil.which(alt)
        if found:
            return "unix", found, None
    raise BuildError("no C++ compiler found (tried c++, g++, clang++)")


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                          cwd=str(HERE), timeout=900)
    if proc.returncode != 0:
        raise BuildError(
            "compiler invocation failed\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stdout: {proc.stdout[-1500:]}\n"
            f"  stderr: {proc.stderr[-1500:]}"
        )


def _python_includes() -> list[str]:
    inc = [sysconfig.get_paths()["include"]]
    plat = sysconfig.get_paths().get("platinclude")
    if plat and plat not in inc:
        inc.append(plat)
    try:
        import numpy
        inc.append(numpy.get_include())
    except ImportError:
        pass
    return inc


def _python_link_args() -> list[str]:
    """Windows/MSVC must link against pythonXY.lib; ELF/Mach-O resolve at load."""
    if not IS_WINDOWS:
        return []
    libdir = sysconfig.get_config_var("installed_base")
    ver = f"{sys.version_info.major}{sys.version_info.minor}"
    return [f"/LIBPATH:{Path(libdir) / 'libs'}", f"python{ver}.lib"]


# ===========================================================================
# targets
# ===========================================================================
def build_ctypes(verbose: bool = False) -> Path:
    """Plain shared library exposing the C ABI -- no Python headers required.

    Because it does not link against Python, a wrong-architecture build of this
    target *succeeds*. That is exactly why verify_artifact() matters here.
    """
    kind, cc, env = detect_compiler()
    out = HERE / f"libinfer_lab_kernels{SHARED_SUFFIX}"
    src = str(CSRC / "fused_kernels.cpp")
    if kind == "msvc":
        cmd = [cc, "/nologo", "/O2", "/EHsc", "/std:c++17", "/LD", src, f"/Fe:{out}"]
    else:
        cmd = [cc, "-O3", "-std=c++17", "-fPIC", "-shared", "-fvisibility=hidden",
               src, "-o", str(out)]
    if verbose:
        print("  $", " ".join(cmd))
    _run(cmd, env)
    _cleanup_intermediates()
    verify_artifact(out)
    return out


def _build_python_ext(module: str, source: str, extra_includes: list[str],
                      extra_sources: list[str], verbose: bool) -> Path:
    kind, cc, env = detect_compiler()
    out = HERE / f"{module}{EXT_SUFFIX}"
    includes = _python_includes() + extra_includes
    fused = str(CSRC / "fused_kernels.cpp")

    if kind == "msvc":
        cmd = [cc, "/nologo", "/O2", "/EHsc", "/std:c++17", "/LD"]
        cmd += [f"/I{i}" for i in includes]
        cmd += extra_sources + [str(CSRC / source), fused, f"/Fe:{out}", "/link"]
        cmd += _python_link_args()
    else:
        cmd = [cc, "-O3", "-std=c++17", "-fPIC", "-shared", "-fvisibility=hidden"]
        cmd += [f"-I{i}" for i in includes] + [f"-I{CSRC}"]
        cmd += extra_sources + [str(CSRC / source), fused, "-o", str(out)]
        if sys.platform == "darwin":
            cmd += ["-undefined", "dynamic_lookup"]

    if verbose:
        print("  $", " ".join(cmd))
    _run(cmd, env)
    _cleanup_intermediates()
    verify_artifact(out)
    return out


def build_pybind11(verbose: bool = False) -> Path:
    try:
        import pybind11
    except ImportError as exc:
        raise BuildError("pybind11 is not installed (pip install pybind11)") from exc
    return _build_python_ext(
        "_infer_lab_pybind", "pybind_module.cpp",
        [pybind11.get_include(), str(CSRC)], [], verbose,
    )


def build_nanobind(verbose: bool = False) -> Path:
    try:
        import nanobind
    except ImportError as exc:
        raise BuildError("nanobind is not installed (pip install nanobind)") from exc
    nb_dir = Path(nanobind.include_dir()).parent
    combined = nb_dir / "src" / "nb_combined.cpp"
    if not combined.exists():
        raise BuildError(f"nanobind runtime source not found at {combined}")
    robin = nb_dir / "ext" / "robin_map" / "include"
    return _build_python_ext(
        "_infer_lab_nanobind", "nanobind_module.cpp",
        [nanobind.include_dir(), str(robin), str(CSRC)], [str(combined)], verbose,
    )


def _cleanup_intermediates() -> None:
    """MSVC drops .obj/.exp/.lib next to the output.

    Leaving them is not merely untidy: the extension loader globs for
    ``_infer_lab_pybind*`` and a stale .obj from a failed run gets picked up as a
    module candidate, producing a baffling AttributeError instead of a clear
    'not built' message.
    """
    for pattern in ("*.obj", "*.exp", "*.lib", "*.pdb", "*.ilk"):
        for path in list(HERE.glob(pattern)) + list(Path.cwd().glob(pattern)):
            try:
                path.unlink()
            except OSError:
                pass


TARGETS = {"ctypes": build_ctypes, "pybind11": build_pybind11, "nanobind": build_nanobind}


def build_all(only: str | None = None, verbose: bool = False) -> dict[str, str]:
    results: dict[str, str] = {}
    for name, fn in TARGETS.items():
        if only and name != only:
            continue
        try:
            path = fn(verbose)
            results[name] = f"ok -> {path.name} ({binary_arch(path)})"
        except BuildError as exc:
            results[name] = f"skipped ({exc})"
        except Exception as exc:  # noqa: BLE001
            results[name] = f"failed ({type(exc).__name__}: {exc})"
    return results


# ===========================================================================
# doctor
# ===========================================================================
def doctor() -> int:
    """Report what the toolchain looks like from here. Builds nothing."""
    print("infer-lab toolchain report")
    print(f"  python           : {sys.version.split()[0]} ({python_arch()})")
    print(f"  platform         : {sysconfig.get_platform()}")
    print(f"  extension suffix : {EXT_SUFFIX}")

    if IS_WINDOWS:
        install = find_vs_install()
        print(f"  visual studio    : {install or 'NOT FOUND'}")
        if install:
            try:
                env = msvc_env()
                print(f"  vcvars arch      : {env.get('VSCMD_ARG_TGT_ARCH', '?')}")
                located = shutil.which("cl", path=env_path(env))
                if located is None and install is not None:
                    fallback = find_cl_in_install(install)
                    located = f"{fallback} (found by path, not on PATH)" \
                        if fallback else None
                print(f"  cl.exe           : {located}")
            except BuildError as exc:
                print(f"  vcvars           : FAILED -- {exc}")

    try:
        kind, cc, env = detect_compiler()
        source = "self-configured" if env else "from current environment"
        print(f"  compiler         : {kind} ({source})")
        print(f"  compiler path    : {cc}")
    except BuildError as exc:
        print(f"  compiler         : NOT USABLE -- {exc}")

    for name in ("pybind11", "nanobind", "numpy"):
        try:
            module = __import__(name)
            print(f"  {name:<17}: {getattr(module, '__version__', 'present')}")
        except ImportError:
            print(f"  {name:<17}: not installed")

    print("\n  existing artifacts:")
    found = False
    for pattern in (f"libinfer_lab_kernels*{SHARED_SUFFIX}", "_infer_lab_*"):
        for path in sorted(HERE.glob(pattern)):
            if path.suffix in (".obj", ".exp", ".lib", ".pdb"):
                continue
            arch = binary_arch(path)
            status = "OK" if arch == python_arch() else f"UNUSABLE (built {arch})"
            print(f"    {path.name:<45} {arch:<8} {status}")
            found = True
    if not found:
        print("    (none -- run the build)")
    return 0


# ===========================================================================
# main
# ===========================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build infer-lab native kernel extensions")
    ap.add_argument("--only", choices=sorted(TARGETS))
    ap.add_argument("--doctor", action="store_true",
                    help="report the toolchain and existing artifacts, build nothing")
    ap.add_argument("--clean", action="store_true",
                    help="remove built artifacts before building")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.doctor:
        return doctor()

    print("infer-lab native kernel build")
    print(f"  target architecture: {python_arch()} (matching this interpreter)")

    if args.clean:
        removed = 0
        for pattern in ("*.dll", "*.pyd", "*.so", "*.dylib", "*.obj", "*.exp", "*.lib"):
            for path in HERE.glob(pattern):
                path.unlink(missing_ok=True)
                removed += 1
        print(f"  cleaned {removed} artifact(s)")

    try:
        kind, cc, env = detect_compiler()
        source = "self-configured" if env else "current environment"
        print(f"  compiler: {cc} ({kind}, {source})")
    except BuildError as exc:
        print(f"  compiler: NOT FOUND -- {exc}")

    results = build_all(args.only, args.verbose)
    for name, status in results.items():
        mark = "OK  " if status.startswith("ok") else "SKIP"
        print(f"  [{mark}] {name}: {status}")

    # A missing optional toolchain is not a failure -- the NumPy path always works.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

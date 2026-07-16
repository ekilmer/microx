# microx - a micro execution framework

![CI](https://github.com/lifting-bits/microx/workflows/CI/badge.svg)

Microx is a single-instruction "micro execution" framework. Microx enables a program to safely execute an arbitrary x86, x86-64, or AArch64 (ARM64) instruction. Microx does not take over or require a process context in order to execute an instruction. It is easily embedded within other programs, as exampled by the Python bindings.

Microx executes instructions natively on the host CPU, so a given build supports only its host architecture: x86/x86-64 on x86-64 hosts, and AArch64 (base A64 + NEON) on arm64 hosts (macOS on Apple Silicon and Linux aarch64). Requesting a different architecture raises `UnsupportedError`.

The microx approach to safe instruction execution of arbitrary instructions is to require the user of microx to manage machine state. Microx is packaged as a C++ `Executor` class that must be extended. The Python bindings also present a class, `microx.Executor`, that must be extended. A program extending this class must implement methods such as `read_register` and `read_memory`. When supplied with instruction bytes, microx will invoke the class methods in order to pull in the minimal requisite machine state to execute the instruction. After executing the instruction, microx will "report back" the state changes induced by the instruction's execution, again via methods like `write_register` and `write_memory`.

The following lists some use-cases of microx:

* Speculative execution of code within a debugger-like system. In this scenario, microx can be used to execute instructions from the process being debugged, in such a way that the memory and state of the original program will be preserved.
* Binary symbolic execution. In this scenario, which was the original use-case of microx, a binary symbolic executor can use microx to safely execute an instruction that is not supported or modelled by the symbolic execution system. The use of microx will minimize the amount of symbolic state that may need to be concretized in order to execute the instruction. Microx was used in this fashion in a Python-based binary symbolic executor. Microx comes with Python bindings for this reason.
* Headless taint tracking. Taint tracking can be implemented with microx, much as it would be with Intel's PIN, but without a process context. Microx can be integrated into a disassembler such as IDA or Binary Ninja and used to execute instruction, performing taint tracking along the way.

Microx uses a combination of JIT-based dynamic binary translation and instruction emulation in order to safely execute instructions. On x86-64 it is a 64-bit library that can also execute 32-bit instructions unsupported on 64-bit platforms. It can be easily embedded, as it performs no dynamic memory allocations, and is re-entrant.

Microx decodes instructions with a swappable, per-architecture decoder: [Intel's XED](https://intelxed.github.io/) on x86-64 and [Capstone](https://www.capstone-engine.org/) on AArch64.

## Installing

Microx has Python bindings; you can install them via pip on macOS and Linux:

```bash
$ python -m pip install microx
```

Wheels are built with the Python Stable ABI (`abi3`), so a single wheel per platform works on CPython 3.10 and newer.

## Building (Python)

The project uses [uv](https://docs.astral.sh/uv/) and a `pyproject.toml` (scikit-build-core backend). Python 3.10+ is required.

On an x86-64 host, first build XED (fetched into `third_party/`):

```bash
$ ./scripts/bootstrap.sh
```

On an arm64 host, Capstone is fetched automatically by CMake — no bootstrap step is needed.

Then build and install:

```bash
$ uv sync           # builds the extension into the dev environment
$ uv build          # or: build sdist + wheel into dist/
```

## Building (C++)

Microx's C++ library can be built with CMake (3.26+). On x86-64, the build locates XED via `XED_DIR` (defaulting to the in-repo `third_party`); on arm64 it fetches Capstone automatically.

```bash
$ ./scripts/bootstrap.sh          # x86-64 only
$ cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
$ cmake --build build
```

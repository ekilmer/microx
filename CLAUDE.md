# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

microx is a single-instruction "micro execution" framework (C++11 library with Python bindings). It safely executes an arbitrary instruction without a process context: the user subclasses an `Executor` and supplies machine state on demand via callbacks (`read_register`, `read_memory`, ...); microx executes the instruction (JIT + emulation) and reports state changes back through `write_register`/`write_memory`. It performs no dynamic memory allocation.

Instructions execute natively on the host CPU, so a build supports only its host ISA: **x86/x86-64** on x86-64 hosts (decoded with Intel XED) and **AArch64** (base A64 + NEON) on arm64 hosts (decoded with Capstone). A wrong-arch request yields `UnsupportedError`.

## Build commands

The build uses `uv` + `pyproject.toml` (scikit-build-core backend, CMake ≥ 3.26). Python 3.10+.

On an **x86-64 host**, build XED first (fetched into `third_party/`):

```bash
./scripts/bootstrap.sh          # clones & builds XED + mbuild into third_party/
```

On an **arm64 host**, Capstone is fetched automatically by CMake — no bootstrap needed.

**Python extension / wheel** (Stable ABI `abi3`, one wheel per platform for CPython 3.10+):

```bash
uv sync                         # build + install into the dev env
uv build --python 3.10          # sdist + wheel into dist/
```

Build abi3 wheels with the **floor** interpreter (Python 3.10). Building the
limited-API extension with 3.12+ headers inlines refcount operations that
corrupt the heap on 3.10/3.11 runtimes. cibuildwheel's `build = "cp310-*"`
does this in CI; locally, use `uv build --python 3.10` (the pinned
`.python-version` selects it by default).

**C++ library** (standalone):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build             # links XED (x86) or fetched Capstone (arm64)
```

**Wheels for release** are built by `cibuildwheel` (see the workflows); there is no Makefile.

## Formatting & lint

`prek` (or `pre-commit`) runs the hooks in `.pre-commit-config.yaml`:

```bash
prek run --all-files            # ruff-check, ruff-format, clang-format, cmake-format
uv run ty check microx examples # type check (ty)
```

`ruff` replaces black/isort/flake8; `.clang-format` is Google style; `cmake-format` uses `.cmake-format.json`.

## Testing

There is no formal test suite. The scripts in `examples/` are runnable smoke tests (per host arch: `examples/example.py`/`example_x64.py` on x86, `examples/example_arm64.py` on arm64). `examples/fuzz_arm64.py` is a differential/fuzz harness for the AArch64 backend. They need the built `microx_core` extension importable (`uv sync`). CI runs the examples per architecture and builds/audits the abi3 wheels.

## Architecture

Four layers, from bottom to top:

1. **Arch-independent driver** — `microx/Executor.cpp` is a small driver: it owns the `Execute` phase machine, the global lock, and the POSIX signal-recovery scaffold (`sigsetjmp`/`siglongjmp` + `sigaltstack`) that catches hardware faults. It contains no decoder, no inline assembly, and no CPython coupling. The public `Executor` class (the only public header, `microx/include/microx/Executor.h`) gains an `Arch` constructor argument; the FPU callbacks are x86-only (non-pure, default no-op).

2. **Backend interface** — `microx/Backend.h` declares the internal per-architecture `Backend` interface (decode, reject, read inputs, read memory, compute control flow, encode/run native, write outputs) plus a `Backend::Create`/`GlobalInit` factory. The decoded instruction never crosses this interface. Exactly one backend is compiled per build (selected by `CMAKE_SYSTEM_PROCESSOR`):
   - `microx/backends/x86/X86Backend.cpp` — the XED decode + inline-asm execution harness + FXSAVE handling (all the original x86 logic; only compiled on x86-64 hosts).
   - `microx/backends/aarch64/` — Capstone decode + register/flag model (`AArch64Backend.cpp`), and a JIT harness (`AArch64Harness.cpp` + `AArch64Trampoline.S`) that emits a per-instruction "arena": it loads the guest register file, runs the single (patched) instruction, and stores results back, using an `mmap(RW)`→`mprotect(RX)` W^X arena with `sys_icache_invalidate`/`__builtin___clear_cache`. Memory operands are staged by rewriting the base register to a scratch register pointing at a staging buffer.

3. **Python C extension** — `microx/Python.cpp` is a hand-written CPython C API binding (not pybind11) targeting the Stable ABI (`Py_LIMITED_API=0x030A0000`, set by CMake's `USE_SABI`). It uses a heap type (`PyType_FromModuleAndSpec`), exposes `microx_core.Executor(addr_size, arch="auto")`, `microx_core.HOST_ARCH`, and the exception hierarchy `MicroxError` → `InstructionDecodeError`, `InstructionFetchError`, `AddressFaultError`, `UnsupportedError`.

4. **Pure-Python object model** — `microx/__init__.py` builds a usable machine-state model on top of `microx_core`:
   - `Operations` — data-conversion hooks (int ↔ byte string), overridable to intercept/symbolize values.
   - `MemoryMap` hierarchy (`PermissionedMemoryMap`, `ArrayMemoryMap`, plus `Proxy*` classes) — user-defined memory regions with r/w/x permissions.
   - `Memory` — page-table-like dispatcher mapping page numbers to `MemoryMap`s (default 4KB pages).
   - `Thread` — register/FPU state (`EmptyThread` is the default-zero implementation).
   - `Process(Executor)` — glues `Memory` + `Thread` into the callbacks; its x86-specific behavior (TSC write, segment base) is gated on `arch`.

`godefroid/` is a separate experimental tool built on microx; it is not part of the distributed package (excluded from the sdist and from `ty`, but linted/formatted with ruff).

## Register vocabulary at the callback boundary

- **x86**: XED register name strings (`RAX`, `EIP`, `XMM0`, ...); flags are 1-bit pseudo-registers (`CF`, `ZF`, ...); FPU is a 512-byte FXSAVE blob via `read_fpu`/`write_fpu`.
- **AArch64**: `X0`–`X30`, `SP`, `PC`, `NZCV` (single register, MRS layout), `V0`–`V31` (128-bit), `FPCR`, `FPSR`. `Wn`→`Xn`, `Bn/Hn/Sn/Dn/Qn`→`Vn`, `WZR`/`XZR` read as 0. No FPU blob callback.

## Known AArch64 v1 limitations

- SVE/SME/PAC/MTE, exclusive monitors (LDXR/STXR), and LSE atomics are rejected with `UnsupportedError`.
- SP as a *data* operand (e.g. `add sp, sp, #16`) is rejected; SP as a memory *base* (including pre/post-index writeback) is supported.
- The Capstone dependency is pinned to a `next` commit (v6 alpha); bump deliberately.

## Release & sync constraints

- `cibuildwheel` owns the manylinux images (configured in `pyproject.toml`); there is no Makefile/`--plat` sync constraint anymore.
- Version lives in the `VERSION` file (single source of truth, read by `scripts/release` and scikit-build-core); pushing a `v*` tag triggers the release workflow (cibuildwheel wheels + sdist + GitHub release + PyPI trusted publishing).

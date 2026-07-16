# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

microx is a single-instruction "micro execution" framework (C++11 library with Python bindings). It safely executes an arbitrary instruction without a process context: the user subclasses an `Executor` and supplies machine state on demand via callbacks (`read_register`, `read_memory`, ...); microx executes the instruction (JIT + emulation) and reports state changes back through `write_register`/`write_memory`. It performs no dynamic memory allocation.

Instructions execute natively on the host CPU, so a build supports only its host ISA: **x86/x86-64** on x86-64 hosts (decoded with Intel XED) and **AArch64** (base A64 + NEON) on arm64 hosts (decoded with Capstone). A wrong-arch request yields `UnsupportedError`.

## Build commands

The build uses `uv` + `pyproject.toml` (scikit-build-core backend, CMake ≥ 3.26). Python 3.10+. A top-level `Makefile` is the canonical task runner — it wraps the `uv` and CMake commands below so local development and CI share one entry point (`make dev`, `format`, `lint`, `test`, `build`, `cpp`, `bootstrap`, `clean`); cibuildwheel wheels are built in CI (configured in `pyproject.toml`), not via `make`.

On an **x86-64 host**, build XED first (fetched into `third_party/`):

```bash
./scripts/bootstrap.sh          # clones & builds XED + mbuild into third_party/ (make bootstrap)
```

On an **arm64 host**, Capstone is fetched automatically by CMake — no bootstrap needed.

**Python extension / wheel** (Stable ABI `abi3`, one wheel per platform for CPython 3.10+):

```bash
uv sync                         # build + install into the dev env
uv build --python 3.10          # sdist + wheel into dist/ (make build)
```

Build abi3 wheels with the **floor** interpreter (Python 3.10). Building the
limited-API extension with 3.12+ headers inlines refcount operations that
corrupt the heap on 3.10/3.11 runtimes. cibuildwheel's `build = "cp310-*"`
does this in CI; locally, use `uv build --python 3.10` (the pinned
`.python-version` selects it by default).

**C++ library** (standalone, via `make cpp`):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build             # links XED (x86) or fetched Capstone (arm64)
```

**Wheels for release** are built by `cibuildwheel` (see the workflows).

## Formatting & lint

`make format` runs `ruff format` then `ruff check --fix`; `make lint` runs `ruff format --check`, `ruff check`, and `ty check microx examples tests`. Both do `uv sync --frozen` first, so the hooks never rewrite `uv.lock`.

`prek` (or `pre-commit`) runs the full hook set in `.pre-commit-config.yaml`:

```bash
prek run --all-files   # builtin hygiene, ruff + ty (via the make format/lint hooks),
                       # clang-format, cmake-format, shellcheck, actionlint, zizmor
```

`ruff` replaces black/isort/flake8; `.clang-format` is Google style; `cmake-format` uses `.cmake-format.json`. GitHub Actions are linted with `actionlint` and security-audited with `zizmor`. The CI lint job skips zizmor (`SKIP=zizmor`) because it runs in its own workflow (`.github/workflows/zizmor.yml`).

## Testing

Tests live in `tests/` and run with **pytest** (`make test`, or `uv run pytest`). Because microx executes instructions natively, the arch-specific modules auto-skip on the wrong host: `tests/test_arm64.py` runs only on arm64, `tests/test_x86.py` only on x86-64. They import the built `microx_core` extension (`uv sync`); cibuildwheel runs the same suite against each built wheel.

The `examples/` directory keeps a few runnable **demo** scripts showcasing end-to-end usage (`make demo`, or `uv run python examples/<name>.py`): `example_x64.py` (x86-64), `example_arm64.py` (arm64 + NEON), and `fuzz_arm64.py` (a differential/fuzz soak for the AArch64 backend). CI's `test` job runs `make test` then `make demo`, and the `wheels` job builds/audits the abi3 wheels per architecture.

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

- `cibuildwheel` owns the manylinux images (configured in `pyproject.toml`).
- Version lives in the `VERSION` file (single source of truth, read by `scripts/release` and scikit-build-core); pushing a `v*` tag triggers `release.yml`: cibuildwheel wheels + sdist → **SLSA build provenance** (`actions/attest-build-provenance` attests every artifact); then the `publish` job (PyPI trusted publishing — OIDC, no token, PEP 740 attestations) and the `github-release` job run in parallel, each gated only on `provenance` (a PyPI-publish failure does not block the GitHub release).
- Registering the PyPI trusted publisher (repo, `release.yml`, environment `pypi`) is a one-time prerequisite before a tag will actually publish.

## GitHub Actions

All action references are **SHA-pinned** with a trailing `# vX.Y.Z` comment; use `pinact run` to pin or re-pin after bumping an action (run it manually — there is no pinact hook or CI step enforcing it). Every workflow sets `permissions: {}` at the top with least-privilege per-job grants, and every `actions/checkout` uses `persist-credentials: false`. `zizmor` (the `zizmor.yml` workflow and the prek hook) must report no findings at medium+ severity.

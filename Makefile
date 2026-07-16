# Task runner for microx. Wraps the uv / CMake / cibuildwheel workflows so that
# local development and CI share one entry point.
#
# On x86-64 hosts, Intel XED must be built first (`make bootstrap`); on arm64
# hosts the AArch64 decoder (Capstone) is fetched automatically by CMake.

UNAME_M := $(shell uname -m)

.PHONY: all
all:
	@echo "Run my targets individually! (dev, format, lint, test, demo, build, cpp, bootstrap, clean)"

.PHONY: dev
dev:
	uv sync --group dev
	uv run prek install

.PHONY: bootstrap
bootstrap:
	./scripts/bootstrap.sh

# ---- Python linting / formatting (does not build the extension) ----
.PHONY: format
format:
	uv sync --frozen --no-install-project --group dev
	uv run --no-sync ruff format . && \
		uv run --no-sync ruff check --fix .

.PHONY: lint
lint:
	uv sync --frozen --no-install-project --group dev
	uv run --no-sync ruff format --check . && \
		uv run --no-sync ruff check . && \
		uv run --no-sync ty check microx examples tests

# ---- Build ----
.PHONY: build
build:
	uv build --python 3.10   # abi3 wheels must be built with the floor Python

# Build the standalone C++ library (arch-native) with CMake.
.PHONY: cpp
cpp:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build --parallel

# ---- Tests (pytest; the arch-specific tests auto-skip on the wrong host) ----
.PHONY: test
test:
	uv sync --frozen --group dev
	uv run --no-sync pytest

# ---- Demo scripts (illustrative end-to-end usage; run any directly with
# `uv run python examples/<name>.py`). Runs the host-appropriate ones. ----
.PHONY: demo
demo:
	uv sync --frozen --no-dev
ifneq (,$(filter $(UNAME_M),arm64 aarch64))
	uv run --no-sync python examples/example_arm64.py
	uv run --no-sync python examples/fuzz_arm64.py
else
	uv run --no-sync python examples/example_x64.py
endif

.PHONY: clean
clean:
	rm -rf build dist wheelhouse

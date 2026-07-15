# Task runner for microx. Wraps the uv / CMake / cibuildwheel workflows so that
# local development and CI share one entry point.
#
# On x86-64 hosts, Intel XED must be built first (`make bootstrap`); on arm64
# hosts the AArch64 decoder (Capstone) is fetched automatically by CMake.

UNAME_M := $(shell uname -m)

# Optionally overridden on the `test` target to run a single example.
EXAMPLE :=

.PHONY: all
all:
	@echo "Run my targets individually! (dev, format, lint, test, build, cpp, bootstrap, clean)"

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
	uv sync --no-install-project --group dev
	uv run --no-sync ruff format . && \
		uv run --no-sync ruff check --fix .

.PHONY: lint
lint:
	uv sync --no-install-project --group dev
	uv run --no-sync ruff format --check . && \
		uv run --no-sync ruff check . && \
		uv run --no-sync ty check microx examples

# ---- Build ----
.PHONY: build
build:
	uv build --python 3.10   # abi3 wheels must be built with the floor Python

# Build the standalone C++ library (arch-native) with CMake.
.PHONY: cpp
cpp:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build --parallel

# ---- Smoke tests (there is no unit-test suite; the examples are the tests) ----
# Runs the examples appropriate for the host architecture through the built
# extension.
.PHONY: test
test:
	uv sync --no-dev
ifneq ($(EXAMPLE),)
	uv run python examples/$(EXAMPLE)
else ifneq (,$(filter $(UNAME_M),arm64 aarch64))
	uv run python examples/example_arm64.py
	uv run python examples/fuzz_arm64.py
else
	uv run python examples/example.py
	uv run python examples/example_x64.py
	uv run python examples/example_rep.py
	uv run python examples/example_tsc.py
	uv run python examples/example_punpckhdq.py
endif

.PHONY: clean
clean:
	rm -rf build dist wheelhouse

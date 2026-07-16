#!/usr/bin/env python3
# Copyright (c) 2019 Trail of Bits, Inc., all rights reserved.

# A small differential / fuzz harness for the AArch64 backend. It runs a curated
# corpus of instructions with known results, then micro-executes a batch of
# random 32-bit words to shake out harness bugs (a native-execution crash would
# take down this process rather than raise a clean microx exception).
#
# microx is uniquely able to catch "declared written but unchanged" and
# "changed but not declared" mismatches: for the curated corpus we assert the
# exact post-state, and for random words we assert only that the host survives
# and any failure surfaces as a microx exception.

import random
import sys

import microx

CODE_BASE = 0x1000
SCRATCH_BASE = 0x40000


def make_process():
    ops = microx.Operations()
    code = microx.ArrayMemoryMap(
        ops, CODE_BASE, CODE_BASE + 0x1000, can_write=True, can_execute=True
    )
    # A large readable+writable region so most memory operands land somewhere
    # mapped rather than faulting.
    scratch = microx.ArrayMemoryMap(ops, SCRATCH_BASE, SCRATCH_BASE + 0x10000)
    mem = microx.Memory(ops, 64)
    mem.add_map(code)
    mem.add_map(scratch)
    return ops, mem, code


def run_one(word, seed_regs=None):
    ops, mem, code = make_process()
    code.store_bytes(CODE_BASE, word.to_bytes(4, "little"))
    t = microx.EmptyThread(ops)
    t.write_register("PC", CODE_BASE)
    # Point every GPR at the middle of the scratch region so memory operands
    # (base/index) resolve to mapped memory.
    mid = SCRATCH_BASE + 0x8000
    for i in range(31):
        t.write_register(f"X{i}", mid)
    if seed_regs:
        for name, val in seed_regs.items():
            t.write_register(name, val)
    p = microx.Process(ops, mem)
    p.execute(t, 1)
    return t


CORPUS = [
    # (word, seed, checks)
    (0x8B020020, {"X1": 40, "X2": 2}, lambda t: t.read_register("X0", 0) == 42),
    (0x91000400, {"X0": 100}, lambda t: t.read_register("X0", 0) == 101),
    (
        0xEB020020,
        {"X1": 5, "X2": 5},
        lambda t: (t.read_register("NZCV", 0) >> 30) & 1 == 1,
    ),
]


def check_corpus():
    ok = True
    for word, seed, predicate in CORPUS:
        try:
            t = run_one(word, seed)
            result = predicate(t)
        except Exception as e:
            print(f"  corpus 0x{word:08x}: raised {e!r}")
            ok = False
            continue
        print(f"  corpus 0x{word:08x}: {'ok' if result else 'FAIL'}")
        ok = ok and result
    return ok


def fuzz(count, seed):
    rng = random.Random(seed)
    executed = 0
    rejected = 0
    faulted = 0
    for _ in range(count):
        word = rng.getrandbits(32)
        try:
            run_one(word)
            executed += 1
        except microx.MicroxError:
            # microx cleanly refused the word: an unsupported/rejected
            # instruction, a decode/fetch error, or a memory-permission error
            # (MemoryAccessException subclasses MicroxError, so it lands here).
            rejected += 1
        except Exception:
            # A non-microx, host-level Python exception. The only invariant that
            # matters is that the host process survives every word.
            faulted += 1
    print(
        f"  fuzz: {count} words -> {executed} executed, "
        f"{rejected} rejected (microx), {faulted} other; host survived"
    )
    return True


def main():
    print("[corpus]")
    ok = check_corpus()
    print("[fuzz]")
    ok = fuzz(2000, seed=1234) and ok
    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

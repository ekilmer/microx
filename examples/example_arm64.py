#!/usr/bin/env python3
# Copyright (c) 2019 Trail of Bits, Inc., all rights reserved.

# Micro-executes a short AArch64 (A64) sequence and checks the end state. This
# mirrors examples/example_x64.py for the ARM64 backend. It runs only on an
# arm64 host (microx executes instructions natively).

import microx


def main() -> int:
    assert microx.HOST_ARCH == "aarch64", (
        f"example_arm64.py runs on an arm64 host; this build is for {microx.HOST_ARCH}"
    )

    o = microx.Operations()

    code = microx.ArrayMemoryMap(o, 0x1000, 0x2000, can_write=False, can_execute=True)
    data = microx.ArrayMemoryMap(o, 0x40000, 0x42000)

    # Disassembly (each instruction micro-executed on its own):
    #   mov  x0, #40          ; movz x0, #40
    #   add  x0, x0, #2       ; x0 = 42
    #   str  x0, [x1]         ; store x0 to [x1]
    #   ldr  x2, [x1]         ; load it back into x2
    #   add  v0.2d, v1.2d, v2.2d  ; NEON add
    #   b    #0               ; infinite self-branch (loop terminator)
    program = [
        0xD2800500,  # mov  x0, #40
        0x91000800,  # add  x0, x0, #2
        0xF9000020,  # str  x0, [x1]
        0xF9400022,  # ldr  x2, [x1]
        0x4EE28420,  # add  v0.2d, v1.2d, v2.2d
        0x14000000,  # b    .   (halts the driver loop below)
    ]
    for i, word in enumerate(program):
        code.store_bytes(0x1000 + 4 * i, word.to_bytes(4, "little"))

    m = microx.Memory(o, 64)
    m.add_map(code)
    m.add_map(data)

    t = microx.EmptyThread(o)
    t.write_register("PC", 0x1000)
    t.write_register("X1", 0x40100)
    # Two NEON source vectors, each two 64-bit lanes.
    t.write_register("V1", (7 << 64) | 3)
    t.write_register("V2", (10 << 64) | 20)

    p = microx.Process(o, m)

    for _ in range(len(program) - 1):  # stop before the self-branch
        pc = t.read_register("PC", t.REG_HINT_PROGRAM_COUNTER)
        print(f"Emulating instruction at {pc:016x}")
        p.execute(t, 1)

    x0 = t.read_register("X0", t.REG_HINT_NONE)
    x2 = t.read_register("X2", t.REG_HINT_NONE)
    v0 = t.read_register("V0", t.REG_HINT_NONE)
    v0_lo = v0 & ((1 << 64) - 1)
    v0_hi = v0 >> 64

    print(f"X0 = {x0}")
    print(f"X2 = {x2}")
    print(f"V0 = lo:{v0_lo} hi:{v0_hi}")

    assert x0 == 42, f"expected X0==42, got {x0}"
    assert x2 == 42, f"expected X2==42, got {x2}"
    assert v0_lo == 23, f"expected V0.d[0]==23, got {v0_lo}"
    assert v0_hi == 17, f"expected V0.d[1]==17, got {v0_hi}"
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

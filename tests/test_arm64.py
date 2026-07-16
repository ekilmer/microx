"""Tests for the AArch64 (A64) micro-execution backend.

microx executes instructions natively, so these run only on an arm64 host; on
any other host the whole module is skipped.
"""

import ctypes
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import microx

pytestmark = pytest.mark.skipif(
    microx.HOST_ARCH != "aarch64",
    reason=f"AArch64 backend requires an arm64 host (build targets {microx.HOST_ARCH})",
)

CODE = 0x1000
RO = 0x40000  # readable, non-writable
RW = 0x50000  # readable + writable


def build_memory(ops):
    """A standard layout: an executable code page and read-only/writable data."""
    code = microx.ArrayMemoryMap(
        ops, CODE, CODE + 0x1000, can_write=False, can_execute=True
    )
    ro = microx.ArrayMemoryMap(ops, RO, RO + 0x1000, can_read=True, can_write=False)
    rw = microx.ArrayMemoryMap(ops, RW, RW + 0x1000)
    mem = microx.Memory(ops, 64)
    for m in (code, ro, rw):
        mem.add_map(m)
    return mem, code, ro, rw


def store_program(code, words, at=CODE):
    for i, word in enumerate(words):
        code.store_bytes(at + 4 * i, word.to_bytes(4, "little"))


def run(ops, mem, regs=None, count=1):
    """Seed a thread, execute `count` instructions one at a time, return it."""
    t = microx.EmptyThread(ops)
    t.write_register("PC", CODE)
    for name, val in (regs or {}).items():
        t.write_register(name, val)
    p = microx.Process(ops, mem)
    for _ in range(count):
        p.execute(t, 1)
    return t


def test_basic_sequence():
    """mov/add, a store + load round-trip, and a NEON vector add."""
    ops = microx.Operations()
    mem, code, _ro, _rw = build_memory(ops)
    store_program(
        code,
        [
            0xD2800500,  # mov  x0, #40
            0x91000800,  # add  x0, x0, #2      -> x0 = 42
            0xF9000020,  # str  x0, [x1]
            0xF9400022,  # ldr  x2, [x1]        -> x2 = 42
            0x4EE28420,  # add  v0.2d, v1.2d, v2.2d
        ],
    )
    t = run(
        ops,
        mem,
        regs={
            "X1": RW + 0x100,
            "V1": (7 << 64) | 3,
            "V2": (10 << 64) | 20,
        },
        count=5,
    )
    assert t.read_register("X0", t.REG_HINT_NONE) == 42
    assert t.read_register("X2", t.REG_HINT_NONE) == 42
    v0 = t.read_register("V0", t.REG_HINT_NONE)
    assert v0 & ((1 << 64) - 1) == 23  # lane 0: 3 + 20
    assert v0 >> 64 == 17  # lane 1: 7 + 10


def test_store_into_read_only_region_faults():
    """A store to a mapped but non-writable region must raise, not silently write."""
    ops = microx.Operations()
    mem, code, ro, _rw = build_memory(ops)
    store_program(code, [0xF9000020])  # str x0, [x1]
    with pytest.raises(microx.MemoryAccessException, match="not writable"):
        run(ops, mem, regs={"X0": 0xDEAD_BEEF_F00D_1234, "X1": RO + 0x100})
    assert bytes(ro.load_bytes(RO + 0x100, 8)) == b"\x00" * 8


def test_store_into_writable_region_succeeds():
    ops = microx.Operations()
    mem, code, _ro, rw = build_memory(ops)
    store_program(code, [0xF9000020])  # str x0, [x1]
    run(ops, mem, regs={"X0": 0x1122_3344_5566_7788, "X1": RW + 0x100})
    assert (
        int.from_bytes(bytes(rw.load_bytes(RW + 0x100, 8)), "little")
        == 0x1122_3344_5566_7788
    )


@pytest.mark.parametrize(
    ("word", "value", "expected"),
    [
        # LDR Xt, .+0x100 : 8-byte load.
        (
            0x58000800,
            (0xCAFE_BABE_DEAD_BEEF).to_bytes(8, "little"),
            0xCAFE_BABE_DEAD_BEEF,
        ),
        # LDR Wt, .+0x100 : 4-byte load, zero-extended to 64 bits.
        (0x18000800, (0xAABB_CCDD).to_bytes(4, "little"), 0x0000_0000_AABB_CCDD),
        # LDRSW Xt, .+0x100 : 4-byte load, sign-extended to 64 bits.
        (0x98000800, (0xFFFF_FFF0).to_bytes(4, "little"), 0xFFFF_FFFF_FFFF_FFF0),
    ],
    ids=["ldr_x", "ldr_w", "ldrsw"],
)
def test_pc_relative_literal_load(word, value, expected):
    """PC-relative literal loads are emulated (they can't run at the arena PC)."""
    ops = microx.Operations()
    mem, code, _ro, _rw = build_memory(ops)
    store_program(code, [word])
    code.store_bytes(0x1100, value)  # literal target = PC (0x1000) + 0x100
    t = run(ops, mem)
    assert t.read_register("X0", t.REG_HINT_NONE) == expected
    assert t.read_register("PC", t.REG_HINT_PROGRAM_COUNTER) == CODE + 4


@pytest.mark.parametrize(
    ("fpcr", "expected"),
    [
        (0, 0x0000_0002),  # FZ off: denormal + denormal = denormal(2)
        (1 << 24, 0x0000_0000),  # FZ on: denormal inputs flushed to zero
    ],
    ids=["fz_off", "fz_on"],
)
def test_fpcr_flush_to_zero(fpcr, expected):
    """FPCR is seeded, so FPCR.FZ (flush-to-zero) is honored during native exec."""
    ops = microx.Operations()
    mem, code, _ro, _rw = build_memory(ops)
    store_program(code, [0x1E212820])  # fadd s0, s1, s1
    t = run(ops, mem, regs={"V1": 0x0000_0001, "FPCR": fpcr})  # S1 = smallest denormal
    assert t.read_register("V0", t.REG_HINT_NONE) & 0xFFFF_FFFF == expected


@pytest.mark.parametrize(
    ("word", "regs", "check"),
    [
        # add x0, x1, x2
        (0x8B020020, {"X1": 40, "X2": 2}, lambda t: t.read_register("X0", 0) == 42),
        # add x0, x0, #1
        (0x91000400, {"X0": 100}, lambda t: t.read_register("X0", 0) == 101),
        # subs x0, x1, x2 -> 0, so the Z flag (NZCV[30]) is set
        (
            0xEB020020,
            {"X1": 5, "X2": 5},
            lambda t: (t.read_register("NZCV", 0) >> 30) & 1 == 1,
        ),
    ],
    ids=["add_reg", "add_imm", "subs_flags"],
)
def test_corpus(word, regs, check):
    ops = microx.Operations()
    mem, code, _ro, _rw = build_memory(ops)
    store_program(code, [word])
    assert check(run(ops, mem, regs=regs))


def test_random_words_do_not_crash_the_host():
    """A native-execution bug would crash this process rather than raise cleanly.

    We only require that every random 32-bit word either micro-executes or
    surfaces as a Python exception; the host must survive all of them.
    """
    rng = random.Random(1234)
    scratch_base = 0x60000
    executed = 0
    for _ in range(2000):
        word = rng.getrandbits(32)
        ops = microx.Operations()
        code = microx.ArrayMemoryMap(
            ops, CODE, CODE + 0x1000, can_write=True, can_execute=True
        )
        scratch = microx.ArrayMemoryMap(ops, scratch_base, scratch_base + 0x10000)
        mem = microx.Memory(ops, 64)
        mem.add_map(code)
        mem.add_map(scratch)
        code.store_bytes(CODE, word.to_bytes(4, "little"))
        t = microx.EmptyThread(ops)
        t.write_register("PC", CODE)
        mid = scratch_base + 0x8000
        for i in range(31):
            t.write_register(f"X{i}", mid)  # keep memory operands mapped
        try:
            microx.Process(ops, mem).execute(t, 1)
            executed += 1
        except Exception:
            pass  # any clean Python exception is acceptable; the host must survive
    # Sanity: at least some random words decode to supported instructions.
    assert executed > 0


@pytest.fixture(scope="module")
def read_host_fpcr():
    """Compile a tiny helper that reads the host FPCR system register (EL0)."""
    cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler available to build the FPCR probe")
    tmp = Path(tempfile.mkdtemp())
    src = tmp / "fpcr.c"
    lib = tmp / ("libfpcr.dylib" if sys.platform == "darwin" else "libfpcr.so")
    src.write_text(
        "#include <stdint.h>\n"
        "uint64_t read_fpcr(void) {\n"
        "  uint64_t v;\n"
        '  __asm__ volatile("mrs %0, fpcr" : "=r"(v));\n'
        "  return v;\n"
        "}\n"
    )
    subprocess.run([cc, "-shared", "-O2", "-o", str(lib), str(src)], check=True)
    dll = ctypes.CDLL(str(lib))
    dll.read_fpcr.restype = ctypes.c_uint64
    return dll.read_fpcr


def test_micro_execution_preserves_host_fpcr(read_host_fpcr):
    """An FP micro-exec installs the guest FPCR; the host's must be restored.

    The guest FPCR is chosen to differ from the host's current value, so a
    leak would change the observed host FPCR regardless of test ordering.
    """
    before = read_host_fpcr()
    ops = microx.Operations()
    mem, code, _ro, _rw = build_memory(ops)
    store_program(code, [0x1E212820])  # fadd s0, s1, s1
    run(ops, mem, regs={"V1": 0x00000001, "FPCR": before ^ (1 << 24)})
    assert read_host_fpcr() == before

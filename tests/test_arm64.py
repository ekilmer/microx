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


@pytest.mark.parametrize(
    "word",
    [
        0x4C407000,  # ld1  {v0.16b}, [x0]
        0x4C40A000,  # ld1  {v0.16b, v1.16b}, [x0]
        0x4C408000,  # ld2  {v0.16b, v1.16b}, [x0]
        0x4C007000,  # st1  {v0.16b}, [x0]
        0x4CDF7000,  # ld1  {v0.16b}, [x0], #16   (post-index)
        0x4D40C000,  # ld1r {v0.16b}, [x0]
    ],
    ids=["ld1", "ld1_2reg", "ld2", "st1", "ld1_postidx", "ld1r"],
)
def test_simd_structure_load_store_rejected(word):
    """LD1-LD4/ST1-ST4 structure loads/stores are out of v1 scope; reject cleanly.

    (Patching their base register would otherwise corrupt the opcode field and
    silently run a different instruction.)
    """
    ops = microx.Operations()
    mem, code, _ro, _rw = build_memory(ops)
    store_program(code, [word])
    with pytest.raises(microx.UnsupportedError):
        run(ops, mem, regs={"X0": RW + 0x100})


def test_plain_simd_load_still_works():
    """A regular vector load (LDR Q, not a structure load) stays supported."""
    ops = microx.Operations()
    mem, code, _ro, rw = build_memory(ops)
    payload = bytes(range(1, 17))
    rw.store_bytes(RW + 0x100, payload)
    store_program(code, [0x3DC00000])  # ldr q0, [x0]
    t = run(ops, mem, regs={"X0": RW + 0x100})
    assert t.read_register("V0", t.REG_HINT_NONE) == int.from_bytes(payload, "little")


@pytest.mark.parametrize(
    ("word", "x0", "expected_pc"),
    [
        # cbz  w0, .+8 : W0 == 0 despite high bits set -> branch taken
        (0x34000040, 0x1_0000_0000, 0x1008),
        # cbz  x0, .+8 : X0 != 0 -> not taken
        (0xB4000040, 0x1_0000_0000, 0x1004),
        # cbnz w0, .+8 : W0 == 0 -> not taken
        (0x35000040, 0x1_0000_0000, 0x1004),
    ],
    ids=["cbz_w_high_bits", "cbz_x_nonzero", "cbnz_w_high_bits"],
)
def test_cbz_cbnz_respect_register_width(word, x0, expected_pc):
    """CBZ/CBNZ W-forms compare only the low 32 bits of the register."""
    ops = microx.Operations()
    mem, code, _ro, _rw = build_memory(ops)
    store_program(code, [word])
    t = run(ops, mem, regs={"X0": x0})
    assert t.read_register("PC", t.REG_HINT_PROGRAM_COUNTER) == expected_pc


@pytest.mark.parametrize(
    "word",
    [
        # Pointer authentication (FEAT_PAuth) -> HasPAuth feature group.
        0xDAC10020,  # pacia x0, x1
        0xDAC11020,  # autia x0, x1
        0xDAC143E0,  # xpaci x0
        # Memory tagging (FEAT_MTE) -> HasMTE.
        0x9ADF1020,  # irg  x0, x1
        0xD9200820,  # stg  x0, [x1]
        # LSE / LSE128 atomics -> HasLSE / HasLSE128. None of these were in the
        # old hand-listed opcode subset, so they regression-guard the switch to
        # feature-group detection (they would otherwise run natively).
        0xF8201041,  # ldclr   x0, x1, [x2]
        0xF8E320A4,  # ldeoral x3, x4, [x5]
        0xF8603041,  # ldsetl  x0, x1, [x2]
        0xF8E08041,  # swpal   x0, x1, [x2]
        0xC8E0FC41,  # casal   x0, x1, [x2]
        0xF820003F,  # stadd   x0, [x1]
        0x19211040,  # ldclrp  x0, x1, [x2]   (LSE128)
        # Unpredicated SVE add -> caught by an SVE Z-register operand (its
        # operands are all plain registers, so the operand-type check alone
        # misses it).
        0x04E00000,  # add z0.d, z0.d, z0.d
        # Streaming-SVE FP8 (SME2 family) -> caught by their Z-register operands
        # even though their FEAT_SSVE_FP8* groups are not in the reject-set.
        0x64A28820,  # fmlalb z0.h, z1.b, z2.b
        0x64228420,  # fdot   z0.h, z1.b, z2.b
        # Pointer-auth HINT-space aliases -> caught by mnemonic (they decode as
        # HINT with no feature group and would otherwise be a silent NOP).
        0xD503233F,  # paciasp
        0xD50323BF,  # autiasp
        0xD50320FF,  # xpaclri
        # Byte/halfword exclusive monitors -> caught by instruction id (they
        # carry no feature group).
        0x085F7C20,  # ldxrb w0, [x1]
        0x08027C20,  # stxrb w2, w0, [x1]
        0x48027C20,  # stxrh w2, w0, [x1]
    ],
    ids=[
        "pacia",
        "autia",
        "xpaci",
        "irg",
        "stg",
        "ldclr",
        "ldeoral",
        "ldsetl",
        "swpal",
        "casal",
        "stadd",
        "ldclrp",
        "sve_add",
        "ssve_fmlalb",
        "ssve_fdot",
        "paciasp",
        "autiasp",
        "xpaclri",
        "ldxrb",
        "stxrb",
        "stxrh",
    ],
)
def test_unsupported_feature_classes_rejected(word):
    """PAC/MTE/LSE/LSE128/SVE/SME are declined rather than mis-executed.

    Native execution would depend on state microx does not model (PAC keys, MTE
    tags, atomic ordering, vector length), so microx rejects the whole family.
    Detection is layered — Capstone feature groups, SVE/SME register operands,
    the PAC HINT-space mnemonics, and instruction id for the group-less
    exclusive monitors — so coverage is not a hand-listed opcode subset.
    """
    ops = microx.Operations()
    mem, code, _ro, _rw = build_memory(ops)
    store_program(code, [word])
    with pytest.raises(microx.UnsupportedError):
        run(ops, mem, regs={"X0": RW + 0x100, "X1": RW + 0x100, "X2": RW + 0x100})


@pytest.mark.parametrize(
    "word",
    [
        0xD53B4200,  # mrs x0, nzcv
        0xD51B4200,  # msr nzcv, x0
        0xD53B4400,  # mrs x0, fpcr
        0xD53B4420,  # mrs x0, fpsr
        0xD53BD040,  # mrs x0, tpidr_el0
    ],
    ids=["mrs_nzcv", "msr_nzcv", "mrs_fpcr", "mrs_fpsr", "mrs_tpidr_el0"],
)
def test_mrs_msr_system_register_rejected(word):
    """MRS/MSR of any system register is out of scope.

    NZCV/FPCR/FPSR are reachable as named registers at the callback boundary,
    but never by executing MRS/MSR (a native `msr` could corrupt host state).
    """
    ops = microx.Operations()
    mem, code, _ro, _rw = build_memory(ops)
    store_program(code, [word])
    with pytest.raises(microx.UnsupportedError):
        run(ops, mem)


def test_neon_or_sme_scalar_fp_not_over_rejected():
    """Base scalar FP tagged "NEON or SME" (HasNEONorSME) must still execute.

    Rejecting the SVE/SME feature groups must not sweep up instructions like
    `fmulx`, which Capstone marks valid in either NEON or SME streaming mode
    (i.e. still plain NEON on any host).
    """
    ops = microx.Operations()
    mem, code, _ro, _rw = build_memory(ops)
    store_program(code, [0x5E22DC20])  # fmulx s0, s1, s2
    # S1 = 2.0f (0x40000000), S2 = 3.0f (0x40400000) -> S0 = 6.0f (0x40C00000).
    t = run(ops, mem, regs={"V1": 0x40000000, "V2": 0x40400000})
    assert t.read_register("V0", t.REG_HINT_NONE) & 0xFFFF_FFFF == 0x40C0_0000

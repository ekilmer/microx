"""Tests for the x86 / x86-64 micro-execution backend.

microx executes instructions natively, so these run only on an x86-64 host; on
any other host the whole module is skipped.
"""

import pytest

import microx

pytestmark = pytest.mark.skipif(
    microx.HOST_ARCH not in ("x86", "x86_64"),
    reason=f"x86 backend requires an x86-64 host (build targets {microx.HOST_ARCH})",
)

CODE = 0x1000
STACK = 0x80000
STACK_TOP = 0x81000
HEAP = 0x10000


def build_memory(ops, address_size):
    code = microx.ArrayMemoryMap(
        ops, CODE, CODE + 0x1000, can_write=False, can_execute=True
    )
    stack = microx.ArrayMemoryMap(ops, STACK, STACK + 0x2000)
    mem = microx.Memory(ops, address_size)
    mem.add_map(code)
    mem.add_map(stack)
    return mem, code, stack


def test_push_decrements_rsp_and_writes_the_slot():
    """`push rbp`: RSP moves down 8 and the old RBP lands at the new top of stack."""
    ops = microx.Operations()
    mem, code, stack = build_memory(ops, 64)
    code.store_bytes(CODE, b"\x55")  # push rbp
    t = microx.EmptyThread(ops)
    t.write_register("RIP", CODE)
    t.write_register("RSP", STACK_TOP)
    t.write_register("RBP", 0xCAFEF00DBAADF00D)
    microx.Process(ops, mem).execute(t, 1)
    assert t.read_register("RSP", t.REG_HINT_NONE) == STACK_TOP - 8
    written = int.from_bytes(bytes(stack.load_bytes(STACK_TOP - 8, 8)), "little")
    assert written == 0xCAFEF00DBAADF00D
    assert t.read_register("RIP", t.REG_HINT_PROGRAM_COUNTER) == CODE + 1


def test_rep_string_ops_build_and_measure_a_string():
    """rep stosb fills a buffer, rep movsb copies it, repne scasb measures its length."""
    ops = microx.Operations()
    mem, code, stack = build_memory(ops, 32)
    # lea edi,[esp-32]; mov eax,0x41; mov ecx,32; rep stosb
    # lea esi,[esp-32]; lea edi,[esp-64]; mov ecx,32; rep movsb
    # mov byte[esp-32],0; lea edi,[esp-64]; xor eax,eax; mov ecx,-1; repne scasb
    # not ecx; dec ecx
    program = (
        b"\x8d\x7c\x24\xe0\xb8\x41\x00\x00\x00\xb9\x20\x00\x00\x00\xf3\xaa"
        b"\x8d\x74\x24\xe0\x8d\x7c\x24\xc0\xb9\x20\x00\x00\x00\xf3\xa4"
        b"\xc6\x44\x24\xe0\x00\x8d\x7c\x24\xc0\x31\xc0\xb9\xff\xff\xff\xff"
        b"\xf2\xae\xf7\xd1\x49"
    )
    code.store_bytes(CODE, program)
    t = microx.EmptyThread(ops)
    t.write_register("EIP", CODE)
    t.write_register("ESP", STACK_TOP)
    p = microx.Process(ops, mem)

    end = CODE + len(program)
    for _ in range(500):  # bounded; REP may retire one iteration per Execute
        pc = t.read_register("EIP", t.REG_HINT_PROGRAM_COUNTER)
        if pc < CODE or pc >= end:
            break
        try:
            p.execute(t, 1)
        except microx.MicroxError:
            break

    # The 32-byte "AAAA..." string was copied to [esp-64, esp-32) ...
    assert bytes(stack.load_bytes(STACK_TOP - 64, 32)) == b"A" * 32
    # ... and null-terminated at [esp-32] ...
    assert bytes(stack.load_bytes(STACK_TOP - 32, 1)) == b"\x00"
    # ... so repne scasb + not + dec yields the length, 32.
    assert t.read_register("ECX", t.REG_HINT_NONE) == 32


def test_tsc_increments_once_per_execute():
    """The x86-only TSC bookkeeping advances the counter for each micro-executed step."""
    ops = microx.Operations()
    mem, code, _stack = build_memory(ops, 32)
    code.store_bytes(CODE, b"\x90\x90\x90")  # three NOPs
    t = microx.EmptyThread(ops)
    t.write_register("EIP", CODE)
    t.write_register("ESP", STACK_TOP)
    p = microx.Process(ops, mem)
    assert t.read_register("TSC", t.REG_HINT_NONE) == 0
    p.execute(t, 1)
    p.execute(t, 1)
    assert t.read_register("TSC", t.REG_HINT_NONE) == 2


def test_punpckhdq_interleaves_high_dwords():
    """SSE/MMX `punpckhdq mm0, [eax]` reads a memory operand and shuffles dwords."""
    ops = microx.Operations()
    code = microx.ArrayMemoryMap(
        ops, CODE, CODE + 0x1000, can_write=False, can_execute=True
    )
    stack = microx.ArrayMemoryMap(ops, STACK, STACK + 0x2000)
    heap = microx.ArrayMemoryMap(ops, HEAP, HEAP + 0x2000)
    mem = microx.Memory(ops, 32)
    for m in (code, stack, heap):
        mem.add_map(m)

    code.store_bytes(CODE, b"\x0f\x6a\x00")  # punpckhdq mm0, [eax]
    heap.store_bytes(
        0x10900, b"\xab\x00\x12\x00\xab\xab\xab\xab"
    )  # -> 0xABABABAB001200AB

    t = microx.EmptyThread(ops)
    t.write_register("EIP", CODE)
    t.write_register("ESP", STACK_TOP)
    t.write_register("MM0", 0xDEADBEEF12121212)
    t.write_register("EAX", 0x10900)
    microx.Process(ops, mem).execute(t, 1)

    # high dwords interleaved: low = dst.high (0xDEADBEEF), high = src.high (0xABABABAB)
    assert t.read_register("MM0", t.REG_HINT_NONE) == 0xABABABABDEADBEEF

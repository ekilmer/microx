#!/usr/bin/env python3
# Copyright (c) 2019 Trail of Bits, Inc., all rights reserved.

import collections

from microx_core import (  # noqa: F401
    HOST_ARCH,
    AddressFaultError,
    Executor,
    InstructionDecodeError,
    InstructionFetchError,
    MicroxError,
    UnsupportedError,
)

LIST_LIKE = (str, bytes, bytearray, tuple, list)


class Operations:
    def convert_to_byte_string(self, data, for_exe=False):
        if isinstance(data, int):
            data = data.to_bytes(8, byteorder="little")

        if for_exe:
            return bytes(data)
        else:
            return tuple(data)

    def convert_to_integer(self, val, for_exe=False):
        if isinstance(val, (str, bytes, bytearray)):
            val = int.from_bytes(val, byteorder="little")
        assert isinstance(val, int)
        return val

    def convert_to_byte(self, byte, for_exe=False):
        if isinstance(byte, LIST_LIKE):
            if isinstance(byte, str):
                byte = ord(byte[0])
            elif isinstance(byte, (bytes, bytearray)):
                byte = byte[0]
            else:
                return self.convert_to_byte(byte[0], for_exe)
        return byte & 0xFF


class ProxyOperations(Operations):
    def __init__(self, next):
        self._next = next

    def convert_to_byte(self, byte, for_exe=False):
        return self._next.convert_to_byte(byte, for_exe)

    def convert_to_byte_string(self, data, for_exe=False):
        return self._next.convert_to_byte_string(data, for_exe)

    def convert_to_integer(self, val, for_exe=False):
        return self._next.convert_to_integer(val, for_exe)


class MemoryAccessException(MicroxError):
    pass


class MemoryMap:
    def __init__(self, mapname=None):
        if mapname is None:
            # Default to a sane-ish map name
            self.__name = f"map_{self.base():08x}-{self.limit():08x}"
        else:
            # Assume they picked a sane name
            self.__name = mapname

    def get_name(self):
        return self.__name

    def set_name(self, value):
        self.__name = value

    def del_name(self):
        del self.__name

    name = property(get_name, set_name, del_name, "This mapping's human readable name")

    def addresses(self):
        i = 0
        base = self.base()
        limit = self.limit()
        while (base + i) < limit:
            yield base + i
            i += 1

    def can_read(self, byte_addr):
        return False

    def can_write(self, byte_addr):
        return False

    def can_execute(self, byte_addr):
        return False

    def base(self):
        return 0

    def limit(self):
        return 0

    def load_byte(self, addr):
        raise MemoryAccessException(f"Can't load byte from address {addr:08x}")

    def load_word(self, addr):
        raise MemoryAccessException(f"Can't load word from address {addr:08x}")

    def load_dword(self, addr):
        raise MemoryAccessException(f"Can't load dword from address {addr:08x}")

    def load_qword(self, addr):
        raise MemoryAccessException(f"Can't load qword from address {addr:08x}")

    def load_bytes(self, addr, num_bytes):
        raise MemoryAccessException(
            f"Can't load {num_bytes} bytes from address {addr:08x}"
        )

    def store_byte(self, addr, val):
        raise MemoryAccessException(f"Can't store byte to address {addr:08x}")

    def store_word(self, addr, val):
        raise MemoryAccessException(f"Can't store word to address {addr:08x}")

    def store_dword(self, addr, val):
        raise MemoryAccessException(f"Can't store dword to address {addr:08x}")

    def store_qword(self, addr, val):
        raise MemoryAccessException(f"Can't store qword to address {addr:08x}")

    def store_bytes(self, addr, data):
        raise MemoryAccessException(
            f"Can't store {len(data)} bytes to address {addr:08x}"
        )


class ProxyMemoryMap(MemoryMap):
    def __init__(self, next):
        self._next = next

    def can_read(self, byte_addr):
        return self._next.can_read(byte_addr)

    def can_write(self, byte_addr):
        return self._next.can_write(byte_addr)

    def can_execute(self, byte_addr):
        return self._next.can_execute(byte_addr)

    def base(self):
        return self._next.base()

    def limit(self):
        return self._next.limit()

    def load_byte(self, addr):
        return self._next.load_byte(addr)

    def load_word(self, addr):
        return self._next.load_word(addr)

    def load_dword(self, addr):
        return self._next.load_dword(addr)

    def load_qword(self, addr):
        return self._next.load_qword(addr)

    def load_bytes(self, addr, num_bytes):
        return self._next.load_bytes(addr, num_bytes)

    def store_byte(self, addr, val):
        return self._next.store_byte(addr, val)

    def store_word(self, addr, val):
        return self._next.store_word(addr, val)

    def store_dword(self, addr, val):
        return self._next.store_dword(addr, val)

    def store_qword(self, addr, val):
        return self._next.store_qword(addr, val)

    def store_bytes(self, addr, data):
        return self._next.store_bytes(addr, data)


class PermissionedMemoryMap(MemoryMap):
    def __init__(
        self,
        ops,
        base,
        limit,
        can_read=True,
        can_write=True,
        can_execute=False,
        mapname=None,
    ):
        assert base < limit
        self._ops = ops
        self._base = base
        self._limit = limit
        self._can_read = can_read
        self._can_write = can_write
        self._can_execute = can_execute
        super().__init__(mapname)

    def can_read(self, byte_addr):
        return self._can_read and self._base <= byte_addr < self._limit

    def can_write(self, byte_addr):
        return self._can_write and self._base <= byte_addr < self._limit

    def can_execute(self, byte_addr):
        return self._can_execute and self._base <= byte_addr < self._limit

    def base(self):
        return self._base

    def limit(self):
        return self._limit


class ArrayMemoryMap(PermissionedMemoryMap):
    def __init__(
        self, ops, base, limit, can_read=True, can_write=True, can_execute=False
    ):
        super().__init__(ops, base, limit, can_read, can_write, can_execute)
        self._data = [0] * (limit - base)

    def load_byte(self, addr):
        offset = addr - self._base
        return self._data[offset : (offset + 1)]

    def load_word(self, addr):
        offset = addr - self._base
        return self._data[offset : (offset + 2)]

    def load_dword(self, addr):
        offset = addr - self._base
        return self._data[offset : (offset + 4)]

    def load_qword(self, addr):
        offset = addr - self._base
        return self._data[offset : (offset + 8)]

    def load_bytes(self, addr, num_bytes):
        offset = addr - self._base
        return self._data[offset : (offset + num_bytes)]

    def store_byte(self, addr, data):
        if isinstance(data, LIST_LIKE):
            self.store_bytes(addr, data[:1])
        else:
            self.store_bytes(addr, data)

    def store_word(self, addr, data):
        if isinstance(data, LIST_LIKE):
            self.store_bytes(addr, data[:2])
        else:
            self.store_bytes(addr, data)

    def store_dword(self, addr, data):
        if isinstance(data, LIST_LIKE):
            self.store_bytes(addr, data[:4])
        else:
            self.store_bytes(addr, data)

    def store_qword(self, addr, data):
        if isinstance(data, LIST_LIKE):
            self.store_bytes(addr, data[:8])
        else:
            self.store_bytes(addr, data)

    def store_bytes(self, addr, data):
        offset = addr - self._base
        if not isinstance(data, LIST_LIKE):
            data = self._ops.convert_to_byte_string(data)
        for b in data:
            if isinstance(b, str):
                b = ord(b)
            self._data[offset] = b
            offset += 1


class Thread:
    REG_HINT_NONE = 0
    REG_HINT_GENERAL = 1
    REG_HINT_PROGRAM_COUNTER = 2
    REG_HINT_CONDITION_CODE = 3
    REG_HINT_WRITE_BACK = 4
    REG_HINT_MEMORY_BASE_ADDRESS = 5
    REG_HINT_MEMORY_INDEX_ADDRESS = 6
    REG_HINT_MEMORY_SEGMENT_ADDRESS = 7

    def __init__(self, ops):
        self._ops = ops

    def read_register(self, reg_name, hint):
        raise NotImplementedError("Abstract")

    def write_register(self, reg_name, value):
        raise NotImplementedError("Abstract")

    def read_fpu(self):
        raise NotImplementedError("Abstract")

    def write_fpu(self, new_fpu_data):
        raise NotImplementedError("Abstract")


class EmptyThread(Thread):
    def __init__(self, ops):
        super().__init__(ops)
        self._regs = collections.defaultdict(int)
        self._fpu_data = b"\0" * 512

    def read_register(self, reg_name, hint):
        return self._regs[reg_name]

    def write_register(self, reg_name, value):
        self._regs[reg_name] = value

    def read_fpu(self):
        return self._fpu_data

    def write_fpu(self, new_fpu_data):
        self._fpu_data = new_fpu_data


class Memory:
    MEM_HINT_READ_ONLY = 0
    MEM_HINT_READ_EXECUTABLE = 1
    MEM_HINT_WRITE_ONLY = 2
    MEM_HINT_READ_WRITE = 3
    MEM_HINT_ADDRESS_GEN = 4

    def __init__(self, ops, address_size, page_shift=12):
        assert address_size in (32, 64)
        assert 0 < page_shift < 32
        self._ops = ops
        self._address_size = address_size
        self._address_mask = (1 << self._address_size) - 1
        self._page_shift = page_shift
        self._memory_maps = collections.defaultdict(MemoryMap)

    def find_hole(self, hole_size):
        """Finds a hole of size `hole_size` between current mappings"""

        paged_hole = hole_size >> self._page_shift
        if 0 == paged_hole:
            paged_hole = 1

        MIN_ADDR = 0x10000 >> self._page_shift
        MAX_ADDR = 0xFFFF0000 >> self._page_shift

        mm = sorted(self._memory_maps.keys())

        lowest = MIN_ADDR
        for addr in mm:
            if addr > lowest and lowest - addr > paged_hole:
                return lowest << self._page_shift
            lowest = addr

        if addr < MAX_ADDR and MAX_ADDR - addr > paged_hole:
            return addr << self._page_shift

        return None

    def find_maps_by_name(self, map_name):

        found_maps = set()
        # TODO(artem): This iterates over every page in the
        # memory map we have. This is wasteful since there
        # are much fewer unique maps. We can try keeping track
        # of unique maps as we add them, and disallow overlapping maps
        for v in self._memory_maps.values():
            if map_name == v.name:
                found_maps.add(v)

        return found_maps

    def _find_map(self, byte_addr):
        return self._memory_maps[(byte_addr & self._address_mask) >> self._page_shift]

    def _crosses_pages(self, addr, num_bytes):
        mmap = self._find_map(addr)
        i = 1
        while i < num_bytes:
            if self._find_map(addr + i) != mmap:
                return True
            i += 1
        return False

    def address_size_bits(self):
        return self._address_size

    def add_map(self, mmap):
        base, limit = mmap.base(), mmap.limit()
        assert ((base >> self._page_shift) << self._page_shift) == base
        while base < limit:
            self._memory_maps[(base & self._address_mask) >> self._page_shift] = mmap
            base += 1 << self._page_shift

    def can_read(self, byte_addr):
        return self._find_map(byte_addr).can_read(byte_addr)

    def can_write(self, byte_addr):
        return self._find_map(byte_addr).can_write(byte_addr)

    def can_execute(self, byte_addr):
        return self._find_map(byte_addr).can_execute(byte_addr)

    def load(self, addr, num_bytes):
        if num_bytes in (1, 2, 4, 8) and not self._crosses_pages(addr, num_bytes):
            if 1 == num_bytes:
                return self._find_map(addr).load_byte(addr)
            elif 2 == num_bytes:
                return self._find_map(addr).load_word(addr)
            elif 4 == num_bytes:
                return self._find_map(addr).load_dword(addr)
            else:
                return self._find_map(addr).load_qword(addr)
        else:
            reads = []
            i = 0
            while i < num_bytes:
                byte_addr = addr + i
                i += 1
                reads.append(self._find_map(byte_addr).load_bytes(byte_addr, 1)[0])
            return self._ops.convert_to_byte_string(reads)

    def store(self, addr, data):
        num_bytes = len(data)
        if num_bytes in (1, 2, 4, 8) and not self._crosses_pages(addr, num_bytes):
            if 1 == num_bytes:
                self._find_map(addr).store_byte(addr, data)
            elif 2 == num_bytes:
                self._find_map(addr).store_word(addr, data)
            elif 4 == num_bytes:
                self._find_map(addr).store_dword(addr, data)
            else:
                self._find_map(addr).store_qword(addr, data)
        else:
            i = 0
            while i < num_bytes:
                byte_addr = addr + i
                self._find_map(byte_addr).store_bytes(byte_addr, data[i : i + 1])
                i += 1


class ProxyThread(Thread):
    def __init__(self, next):
        super().__init__(next._ops)
        assert isinstance(next, Thread)
        self._next = next

    def read_register(self, reg_name, hint):
        return self._next.read_register(reg_name, hint)

    def write_register(self, reg_name, value):
        return self._next.write_register(reg_name, value)

    def read_fpu(self):
        return self._next.read_fpu()

    def write_fpu(self, new_fpu_data):
        return self._next.write_fpu(new_fpu_data)


class Process(Executor):
    MEM_READ_HINTS = (
        Memory.MEM_HINT_READ_ONLY,
        Memory.MEM_HINT_READ_EXECUTABLE,
        Memory.MEM_HINT_READ_WRITE,
    )

    MEM_WRITE_HINTS = (Memory.MEM_HINT_WRITE_ONLY, Memory.MEM_HINT_READ_WRITE)

    MEM_EXEC_HINTS = (Memory.MEM_HINT_READ_EXECUTABLE,)

    def __init__(self, ops, memory, arch="auto"):
        assert isinstance(memory, Memory)
        self._arch = HOST_ARCH if arch == "auto" else arch
        super().__init__(memory.address_size_bits(), self._arch)
        self._memory = memory
        self._thread = None
        self._ops = ops

    def execute(self, thread, max_num_instructions=1):
        assert 0 < max_num_instructions
        assert isinstance(thread, Thread)

        self._thread = thread
        try:
            super().execute(max_num_instructions)

            # TSC is an x86 concept; approximate it as 1 cycle / instruction.
            if self._arch in ("x86_64", "x86"):
                tsc = self.read_register("TSC", thread.REG_HINT_NONE)
                self.write_register("TSC", tsc + max_num_instructions)
        finally:
            self._thread = None

    def read_register(self, reg_name, hint):
        assert self._thread is not None  # set for the duration of execute()
        return self._ops.convert_to_integer(
            self._thread.read_register(reg_name, hint), for_exe=True
        )

    def write_register(self, reg_name, val):
        assert self._thread is not None  # set for the duration of execute()
        self._thread.write_register(reg_name, self._ops.convert_to_integer(val))

    def compute_address(self, seg_name, base_addr, index, scale, disp, size, hint):
        seg_base = 0
        # Segmentation is an x86 concept; ARM passes an empty segment name.
        if (
            self._arch in ("x86_64", "x86")
            and seg_name
            and hint != Memory.MEM_HINT_ADDRESS_GEN
        ):
            seg_base = self.read_register(
                f"{seg_name}_BASE", Thread.REG_HINT_MEMORY_SEGMENT_ADDRESS
            )
            seg_base = seg_base & self._memory._address_mask
        return seg_base + base_addr + (index * scale) + disp

    def read_memory(self, addr, num_bytes, hint):
        check_read = hint in self.MEM_READ_HINTS
        check_write = hint in self.MEM_WRITE_HINTS
        check_exec = hint in self.MEM_EXEC_HINTS

        # Check permissions
        i = 0
        while i < num_bytes:
            byte_addr = addr + i
            i += 1

            if check_read:
                if not self._memory.can_read(byte_addr):
                    raise MemoryAccessException(
                        f"Address {byte_addr:08x} is not readable"
                    )

            if check_write:
                if not self._memory.can_write(byte_addr):
                    raise MemoryAccessException(
                        f"Address {byte_addr:08x} is not writable"
                    )

            if check_exec:
                if not self._memory.can_execute(byte_addr):
                    raise MemoryAccessException(
                        f"Address {byte_addr:08x} is not executable"
                    )

        return self._ops.convert_to_byte_string(
            self._memory.load(addr, num_bytes), for_exe=True
        )

    def write_memory(self, addr, data):
        data = self._ops.convert_to_byte_string(data)
        self._memory.store(addr, data)

    # The FPU is treated as an opaque blob of memory.
    def read_fpu(self):
        assert self._thread is not None  # set for the duration of execute()
        return self._ops.convert_to_byte_string(self._thread.read_fpu(), for_exe=True)

    def write_fpu(self, fpu):
        assert self._thread is not None  # set for the duration of execute()
        self._thread.write_fpu(self._ops.convert_to_byte_string(fpu, for_exe=True))

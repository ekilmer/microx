/*
 * Copyright (c) 2018 Trail of Bits, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// The AArch64 native-execution harness. It emits, per instruction, a small
// "arena" of machine code that loads the guest register file from `gState`,
// runs the single (patched) guest instruction, stores the results back, and
// branches to the assembly trampoline's return path. The arena is written to a
// page-aligned RW region, flushed from the data cache into the instruction
// cache, flipped to RX, and executed.

#include <sys/mman.h>
#include <unistd.h>

#include <cstring>

#if defined(__APPLE__)
#include <libkern/OSCacheControl.h>
#endif

#include "AArch64Internal.h"

namespace microx {
namespace aarch64 {
namespace {

// x9 is the harness's fixed scratch register inside the arena. It is loaded
// with a guest value like any other GPR before the instruction runs and stored
// back before being reused as scratch, so it remains a fully supported guest
// register; the backend never selects it as xctx/xmem/xsp.
enum : unsigned { kScratch = 9 };

// The executable arena. Sized to comfortably hold the load/store block for all
// 31 GPRs and 32 vector registers plus the guest instruction.
static uint8_t* gArena = nullptr;
static size_t gArenaSize = 0;

// The number of 32-bit words currently emitted into `gArena`.
static size_t gNumWords = 0;

static void Emit(uint32_t word) {
  std::memcpy(gArena + (gNumWords * sizeof(word)), &word, sizeof(word));
  ++gNumWords;
}

// --- Fixed instruction encoders (see the Arm Architecture Reference Manual) --

// MOV Xd, Xm  (alias of ORR Xd, XZR, Xm).
static uint32_t EncMov(unsigned rd, unsigned rm) {
  return 0xAA0003E0u | (rm << 16) | rd;
}

// LDR Xt, [Xn, #off]  (unsigned offset; off is a byte offset, multiple of 8).
static uint32_t EncLdr(unsigned rt, unsigned rn, unsigned off) {
  return 0xF9400000u | ((off / 8u) << 10) | (rn << 5) | rt;
}

// STR Xt, [Xn, #off].
static uint32_t EncStr(unsigned rt, unsigned rn, unsigned off) {
  return 0xF9000000u | ((off / 8u) << 10) | (rn << 5) | rt;
}

// LDP Qt1, Qt2, [Xn, #off]  (128-bit SIMD&FP, signed offset, multiple of 16).
static uint32_t EncLdpQ(unsigned t1, unsigned t2, unsigned rn, unsigned off) {
  return 0xAD400000u | (((off / 16u) & 0x7Fu) << 15) | (t2 << 10) | (rn << 5) |
         t1;
}

// STP Qt1, Qt2, [Xn, #off].
static uint32_t EncStpQ(unsigned t1, unsigned t2, unsigned rn, unsigned off) {
  return 0xAD000000u | (((off / 16u) & 0x7Fu) << 15) | (t2 << 10) | (rn << 5) |
         t1;
}

// MOVZ Xd, #imm16, LSL #(16*hw).
static uint32_t EncMovz(unsigned rd, uint16_t imm16, unsigned hw) {
  return 0xD2800000u | (hw << 21) | (static_cast<uint32_t>(imm16) << 5) | rd;
}

// MOVK Xd, #imm16, LSL #(16*hw).
static uint32_t EncMovk(unsigned rd, uint16_t imm16, unsigned hw) {
  return 0xF2800000u | (hw << 21) | (static_cast<uint32_t>(imm16) << 5) | rd;
}

// MSR (sysreg), Xt / MRS Xt, (sysreg).
static uint32_t EncMsr(unsigned op1, unsigned crn, unsigned crm, unsigned op2,
                       unsigned rt) {
  return 0xD5180000u | (op1 << 16) | (crn << 12) | (crm << 8) | (op2 << 5) | rt;
}
static uint32_t EncMrs(unsigned op1, unsigned crn, unsigned crm, unsigned op2,
                       unsigned rt) {
  return 0xD5380000u | (op1 << 16) | (crn << 12) | (crm << 8) | (op2 << 5) | rt;
}

// BR Xn.
static uint32_t EncBr(unsigned rn) { return 0xD61F0000u | (rn << 5); }

enum : uint32_t { kBtiC = 0xD503245Fu, kBtiJ = 0xD503249Fu };

// System-register coordinates.
static uint32_t MsrNzcv(unsigned rt) { return EncMsr(3, 4, 2, 0, rt); }
static uint32_t MrsNzcv(unsigned rt) { return EncMrs(3, 4, 2, 0, rt); }
static uint32_t MsrFpcr(unsigned rt) { return EncMsr(3, 4, 4, 0, rt); }
static uint32_t MrsFpcr(unsigned rt) { return EncMrs(3, 4, 4, 0, rt); }
static uint32_t MsrFpsr(unsigned rt) { return EncMsr(3, 4, 4, 1, rt); }
static uint32_t MrsFpsr(unsigned rt) { return EncMrs(3, 4, 4, 1, rt); }

// Emit `movz`+`movk`s to materialize a 64-bit immediate into `rd`.
static void EmitLoadImm64(unsigned rd, uint64_t value) {
  Emit(EncMovz(rd, static_cast<uint16_t>(value), 0));
  Emit(EncMovk(rd, static_cast<uint16_t>(value >> 16), 1));
  Emit(EncMovk(rd, static_cast<uint16_t>(value >> 32), 2));
  Emit(EncMovk(rd, static_cast<uint16_t>(value >> 48), 3));
}

static bool IsSkippedGpr(unsigned reg) {
  return reg == static_cast<unsigned>(gNative.xctx) ||
         (gNative.has_mem && reg == static_cast<unsigned>(gNative.xmem)) ||
         (gNative.xsp >= 0 && reg == static_cast<unsigned>(gNative.xsp));
}

static void FlushArena(size_t num_bytes) {
#if defined(__APPLE__)
  sys_icache_invalidate(gArena, num_bytes);
#else
  __builtin___clear_cache(reinterpret_cast<char*>(gArena),
                          reinterpret_cast<char*>(gArena) + num_bytes);
#endif
}

}  // namespace

bool ArenaGlobalInit(void) {
  const long page = sysconf(_SC_PAGESIZE);
  const size_t page_size = (page > 0) ? static_cast<size_t>(page) : 16384;
  // One page comfortably holds ~140 words of prologue/epilogue + instruction.
  gArenaSize = page_size;
  void* p = mmap(nullptr, gArenaSize, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (MAP_FAILED == p) {
    gArena = nullptr;
    return false;
  }
  gArena = static_cast<uint8_t*>(p);
  return true;
}

bool EmitArena(void) {
  if (!gArena) {
    return false;
  }

  // Flip to writable to (re)generate the code for this instruction.
  if (0 != mprotect(gArena, gArenaSize, PROT_READ | PROT_WRITE)) {
    return false;
  }
  gNumWords = 0;

  const unsigned ctx = static_cast<unsigned>(gNative.xctx);

  // Prologue: anchor the state pointer, then load the guest condition flags.
  Emit(kBtiC);
  Emit(EncMov(ctx, 10));  // Xctx = x10 (the trampoline passes &State in x10).
  Emit(EncLdr(kScratch, ctx, MICROX_ST_NZCV));
  Emit(MsrNzcv(kScratch));

  // Optionally load the full FP/SIMD state.
  if (gNative.uses_fp) {
    Emit(EncLdr(kScratch, ctx, MICROX_ST_FPCR));
    Emit(MsrFpcr(kScratch));
    Emit(EncLdr(kScratch, ctx, MICROX_ST_FPSR));
    Emit(MsrFpsr(kScratch));
    for (unsigned v = 0; v < 32; v += 2) {
      Emit(EncLdpQ(v, v + 1, ctx, MICROX_ST_VEC + (v * 16u)));
    }
  }

  // Load the guest GPRs, skipping the reserved scratch/anchor registers.
  for (unsigned r = 0; r < 31; ++r) {
    if (IsSkippedGpr(r)) {
      continue;
    }
    Emit(EncLdr(r, ctx, MICROX_ST_GPR + (r * 8u)));
  }

  // Materialize the staging address in the memory-base register, and alias the
  // guest SP into the SP-operand register, if used.
  if (gNative.has_mem) {
    EmitLoadImm64(static_cast<unsigned>(gNative.xmem), gNative.mem_addr);
  }
  if (gNative.xsp >= 0) {
    Emit(EncLdr(static_cast<unsigned>(gNative.xsp), ctx, MICROX_ST_SP));
  }

  // The single guest instruction.
  Emit(gNative.insn_word);

  // Epilogue: store the guest GPRs back (Xctx still valid; it was untouched).
  for (unsigned r = 0; r < 31; ++r) {
    if (IsSkippedGpr(r)) {
      continue;
    }
    Emit(EncStr(r, ctx, MICROX_ST_GPR + (r * 8u)));
  }
  if (gNative.xsp >= 0) {
    Emit(EncStr(static_cast<unsigned>(gNative.xsp), ctx, MICROX_ST_SP));
  }

  Emit(MrsNzcv(kScratch));
  Emit(EncStr(kScratch, ctx, MICROX_ST_NZCV));
  if (gNative.uses_fp) {
    Emit(MrsFpsr(kScratch));
    Emit(EncStr(kScratch, ctx, MICROX_ST_FPSR));
    Emit(MrsFpcr(kScratch));
    Emit(EncStr(kScratch, ctx, MICROX_ST_FPCR));
    for (unsigned v = 0; v < 32; v += 2) {
      Emit(EncStpQ(v, v + 1, ctx, MICROX_ST_VEC + (v * 16u)));
    }
  }

  // Return to the trampoline.
  Emit(EncLdr(kScratch, ctx, MICROX_ST_RET));
  Emit(EncBr(kScratch));

  const size_t num_bytes = gNumWords * sizeof(uint32_t);
  if (num_bytes > gArenaSize) {
    return false;  // Should never happen for a one-page arena.
  }

  FlushArena(num_bytes);
  if (0 != mprotect(gArena, gArenaSize, PROT_READ | PROT_EXEC)) {
    return false;
  }

  gState.arena_entry = reinterpret_cast<uint64_t>(gArena);
  gState.ret_addr = reinterpret_cast<uint64_t>(&microx_arm64_return);
  return true;
}

void RunArena(void) { microx_arm64_enter(&gState); }

}  // namespace aarch64
}  // namespace microx

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

// The AArch64 (A64) micro-execution backend. It decodes with Capstone v6,
// rejects instruction classes outside the base A64 + NEON scope, stages the
// register/flag/memory inputs via the executor callbacks, emulates control
// flow and PC-relative instructions in software, and hands everything else to
// the native harness (AArch64Harness.cpp).

#include <cstdio>
#include <cstring>
#include <new>

#include "AArch64Internal.h"
#include "Backend.h"
#include "Capstone.h"
#include "microx/Executor.h"

namespace microx {
namespace aarch64 {

// Defined here; shared with the harness translation unit.
State gState;
NativeStaging gNative;

namespace {

// Staging buffer for the single memory operand. Sized for the widest supported
// access (an STP of 128-bit vector registers is 32 bytes; round up).
alignas(16) static uint8_t gMemBuf[64];

// Capstone handle and a reusable decoded-instruction buffer (no per-instruction
// allocation).
static csh gHandle = 0;
static cs_insn* gInsn = nullptr;
static bool gHandleValid = false;

// Per-instruction staging derived from the decode.
static uint64_t gPC = 0;           // Current program counter.
static uint32_t gUsedGprMask = 0;  // Bit i set if the instruction touches xi.
static bool gModGpr[31];           // GPR write-back set.
static bool gModVec[32];           // Vector write-back set.
static bool gModSp, gModNzcv, gModFpcr, gModFpsr, gModPc;
static bool gUsesFp;  // Any V/FP register touched.

// Memory operand staging.
static bool gMemPresent = false;
static bool gMemIsStore = false;
static bool gMemIsLiteral = false;  // PC-relative literal load, emulated in SW.
static uintptr_t gMemAddr = 0;
static size_t gMemBytes = 0;
// Writeback (pre/post-index) on the base register.
static bool gMemWriteback = false;
static int gMemBaseGpr = -1;  // canonical GPR index of base, or -1 if SP.
static bool gMemBaseIsSp = false;
static int64_t gMemWbDelta = 0;

// --- Canonical register model
// -------------------------------------------------

enum class Kind { kGpr, kVec, kSp, kNzcv, kFpcr, kFpsr, kZero, kOther };

struct Canon {
  Kind kind;
  int idx;
};

// Canonicalize a Capstone register id to microx's register vocabulary using its
// textual name (robust across the many W/X/V/Q/D/S/H/B aliases).
static Canon Canonicalize(unsigned reg) {
  Canon c{Kind::kOther, 0};
  if (!reg) {
    return c;
  }
  const char* name = cs_reg_name(gHandle, reg);
  if (!name) {
    return c;
  }
  if (0 == std::strcmp(name, "sp") || 0 == std::strcmp(name, "wsp")) {
    c.kind = Kind::kSp;
  } else if (0 == std::strcmp(name, "xzr") || 0 == std::strcmp(name, "wzr")) {
    c.kind = Kind::kZero;
  } else if (0 == std::strcmp(name, "nzcv")) {
    c.kind = Kind::kNzcv;
  } else if (0 == std::strcmp(name, "fpcr")) {
    c.kind = Kind::kFpcr;
  } else if (0 == std::strcmp(name, "fpsr")) {
    c.kind = Kind::kFpsr;
  } else if (0 == std::strcmp(name, "fp")) {
    c.kind = Kind::kGpr;
    c.idx = 29;
  } else if (0 == std::strcmp(name, "lr")) {
    c.kind = Kind::kGpr;
    c.idx = 30;
  } else if ('x' == name[0] || 'w' == name[0]) {
    c.kind = Kind::kGpr;
    c.idx = std::atoi(name + 1);
  } else if ('v' == name[0] || 'q' == name[0] || 'd' == name[0] ||
             's' == name[0] || 'h' == name[0] || 'b' == name[0]) {
    c.kind = Kind::kVec;
    c.idx = std::atoi(name + 1);
  }
  return c;
}

static void CanonName(const Canon& c, char* out, size_t out_size) {
  switch (c.kind) {
    case Kind::kGpr:
      std::snprintf(out, out_size, "X%d", c.idx);
      break;
    case Kind::kVec:
      std::snprintf(out, out_size, "V%d", c.idx);
      break;
    case Kind::kSp:
      std::snprintf(out, out_size, "SP");
      break;
    case Kind::kNzcv:
      std::snprintf(out, out_size, "NZCV");
      break;
    case Kind::kFpcr:
      std::snprintf(out, out_size, "FPCR");
      break;
    case Kind::kFpsr:
      std::snprintf(out, out_size, "FPSR");
      break;
    default:
      out[0] = '\0';
      break;
  }
}

static size_t CanonBits(const Canon& c) {
  return (Kind::kVec == c.kind) ? 128 : 64;
}

// Load a canonical register's value from staging into a `Data` for writing, or
// store a read `Data` into staging.
static void StoreToState(const Canon& c, const Data& val) {
  switch (c.kind) {
    case Kind::kGpr:
      std::memcpy(&gState.gpr[c.idx], val.bytes, 8);
      break;
    case Kind::kVec:
      std::memcpy(&gState.vec[c.idx][0], val.bytes, 16);
      break;
    case Kind::kSp:
      std::memcpy(&gState.sp, val.bytes, 8);
      break;
    case Kind::kNzcv:
      std::memcpy(&gState.nzcv, val.bytes, 8);
      break;
    case Kind::kFpcr:
      std::memcpy(&gState.fpcr, val.bytes, 8);
      break;
    case Kind::kFpsr:
      std::memcpy(&gState.fpsr, val.bytes, 8);
      break;
    default:
      break;
  }
}

static void LoadFromState(const Canon& c, Data& val) {
  std::memset(val.bytes, 0, sizeof(val.bytes));
  switch (c.kind) {
    case Kind::kGpr:
      std::memcpy(val.bytes, &gState.gpr[c.idx], 8);
      break;
    case Kind::kVec:
      std::memcpy(val.bytes, &gState.vec[c.idx][0], 16);
      break;
    case Kind::kSp:
      std::memcpy(val.bytes, &gState.sp, 8);
      break;
    case Kind::kNzcv:
      std::memcpy(val.bytes, &gState.nzcv, 8);
      break;
    case Kind::kFpcr:
      std::memcpy(val.bytes, &gState.fpcr, 8);
      break;
    case Kind::kFpsr:
      std::memcpy(val.bytes, &gState.fpsr, 8);
      break;
    default:
      break;
  }
}

static void MarkModified(const Canon& c) {
  switch (c.kind) {
    case Kind::kGpr:
      gModGpr[c.idx] = true;
      break;
    case Kind::kVec:
      gModVec[c.idx] = true;
      break;
    case Kind::kSp:
      gModSp = true;
      break;
    case Kind::kNzcv:
      gModNzcv = true;
      break;
    case Kind::kFpcr:
      gModFpcr = true;
      break;
    case Kind::kFpsr:
      gModFpsr = true;
      break;
    default:
      break;
  }
}

static void NoteUse(const Canon& c) {
  if (Kind::kGpr == c.kind) {
    gUsedGprMask |= (1u << c.idx);
  } else if (Kind::kVec == c.kind) {
    gUsesFp = true;
  }
}

// Gather the implicit + explicit read and write register sets.
static bool ComputeAccessSets(uint16_t* read, uint8_t* nread, uint16_t* write,
                              uint8_t* nwrite) {
  return CS_ERR_OK ==
         cs_regs_access(gHandle, gInsn, read, nread, write, nwrite);
}

// --- Rejection ---------------------------------------------------------------

static bool HasGroup(uint8_t group) {
  const cs_detail* d = gInsn->detail;
  for (uint8_t i = 0; i < d->groups_count; ++i) {
    if (d->groups[i] == group) {
      return true;
    }
  }
  return false;
}

static bool HasSveOrSmeOperand(void) {
  const cs_aarch64& a = gInsn->detail->aarch64;
  for (uint8_t i = 0; i < a.op_count; ++i) {
    const aarch64_op_type t = a.operands[i].type;
    if (AARCH64_OP_SME == t || AARCH64_OP_PRED == t) {
      return true;
    }
  }
  return false;
}

static bool IsRejectedInstruction(void) {
  switch (gInsn->id) {
    // Exception generation / privileged.
    case AARCH64_INS_SVC:
    case AARCH64_INS_HVC:
    case AARCH64_INS_SMC:
    case AARCH64_INS_BRK:
    case AARCH64_INS_HLT:
    // Exclusive monitors — cannot be represented single-instruction.
    case AARCH64_INS_LDXR:
    case AARCH64_INS_LDAXR:
    case AARCH64_INS_STXR:
    case AARCH64_INS_STLXR:
    case AARCH64_INS_LDXP:
    case AARCH64_INS_STXP:
    case AARCH64_INS_LDAXP:
    case AARCH64_INS_STLXP:
    // LSE atomics — read-modify-write memory ordering semantics out of scope.
    case AARCH64_INS_CAS:
    case AARCH64_INS_CASA:
    case AARCH64_INS_CASAL:
    case AARCH64_INS_CASL:
    case AARCH64_INS_CASP:
    case AARCH64_INS_SWP:
    case AARCH64_INS_LDADD:
      return true;
    default:
      break;
  }
  return false;
}

// Advanced SIMD load/store of multiple or single structures (LD1-LD4 / ST1-ST4
// and the replicate/lane forms). These span a large space of element
// arrangements, register counts, de-interleaving, and pre/post-index writeback
// that the single-operand memory-staging path does not model, so reject them
// rather than mis-execute (the base-register rewrite would also corrupt their
// opcode field). Detected by the encoding's op0 (bits [29:24]): 0b001100 =
// load/store multiple structures, 0b001101 = single structures.
static bool IsSimdLoadStoreStructure(void) {
  const uint32_t word = static_cast<uint32_t>(gInsn->bytes[0]) |
                        (static_cast<uint32_t>(gInsn->bytes[1]) << 8) |
                        (static_cast<uint32_t>(gInsn->bytes[2]) << 16) |
                        (static_cast<uint32_t>(gInsn->bytes[3]) << 24);
  if ((word >> 31) & 1u) {
    return false;  // bit[31] is 0 for this encoding group.
  }
  const uint32_t op0 = (word >> 24) & 0x3Fu;
  return op0 == 0x0Cu || op0 == 0x0Du;
}

// --- Memory access-size table
// -------------------------------------------------

// Width in bytes of a single register operand from its Capstone name.
static size_t RegAccessWidth(unsigned reg) {
  const char* name = cs_reg_name(gHandle, reg);
  if (!name) {
    return 0;
  }
  switch (name[0]) {
    case 'x':
    case 'd':
      return 8;
    case 'w':
    case 's':
      return 4;
    case 'q':
      return 16;
    case 'h':
      return 2;
    case 'b':
      return 1;
    default:
      return 0;
  }
}

// The number of bytes a load/store instruction touches in memory.
static size_t MemoryAccessBytes(void) {
  switch (gInsn->id) {
    case AARCH64_INS_LDRB:
    case AARCH64_INS_LDURB:
    case AARCH64_INS_LDRSB:
    case AARCH64_INS_LDURSB:
    case AARCH64_INS_STRB:
    case AARCH64_INS_STURB:
    case AARCH64_INS_LDARB:
    case AARCH64_INS_STLRB:
      return 1;
    case AARCH64_INS_LDRH:
    case AARCH64_INS_LDURH:
    case AARCH64_INS_LDRSH:
    case AARCH64_INS_LDURSH:
    case AARCH64_INS_STRH:
    case AARCH64_INS_STURH:
    case AARCH64_INS_LDARH:
    case AARCH64_INS_STLRH:
      return 2;
    case AARCH64_INS_LDRSW:
    case AARCH64_INS_LDURSW:
      return 4;
    default:
      break;
  }

  // Otherwise the access size is the width of the transferred register(s).
  const cs_aarch64& a = gInsn->detail->aarch64;
  size_t per_reg = 0;
  unsigned num_reg = 0;
  for (uint8_t i = 0; i < a.op_count; ++i) {
    if (AARCH64_OP_REG == a.operands[i].type) {
      const size_t w = RegAccessWidth(a.operands[i].reg);
      if (w) {
        per_reg = w;
        ++num_reg;
      }
    }
  }
  if (!per_reg) {
    return 0;
  }
  const bool is_pair =
      (AARCH64_INS_LDP == gInsn->id || AARCH64_INS_STP == gInsn->id ||
       AARCH64_INS_LDNP == gInsn->id || AARCH64_INS_STNP == gInsn->id ||
       AARCH64_INS_LDPSW == gInsn->id);
  return is_pair ? per_reg * 2 : per_reg;
}

static bool InstructionStoresMemory(void) {
  const cs_aarch64& a = gInsn->detail->aarch64;
  for (uint8_t i = 0; i < a.op_count; ++i) {
    if (AARCH64_OP_MEM == a.operands[i].type &&
        (a.operands[i].access & CS_AC_WRITE)) {
      return true;
    }
  }
  return false;
}

// Sign/zero-extend a register value per an AArch64 extend specifier.
static uint64_t ApplyExtend(uint64_t value, aarch64_extender ext,
                            unsigned shift) {
  uint64_t v = value;
  switch (ext) {
    case AARCH64_EXT_UXTB:
      v = static_cast<uint8_t>(v);
      break;
    case AARCH64_EXT_UXTH:
      v = static_cast<uint16_t>(v);
      break;
    case AARCH64_EXT_UXTW:
      v = static_cast<uint32_t>(v);
      break;
    case AARCH64_EXT_SXTB:
      v = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int8_t>(v)));
      break;
    case AARCH64_EXT_SXTH:
      v = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int16_t>(v)));
      break;
    case AARCH64_EXT_SXTW:
      v = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(v)));
      break;
    default:  // UXTX/SXTX/none: full 64-bit.
      break;
  }
  return v << shift;
}

// --- Instruction-word patching
// ------------------------------------------------

static void SetField(uint32_t& word, unsigned value, unsigned lsb,
                     unsigned width) {
  const uint32_t mask = ((1u << width) - 1u) << lsb;
  word = (word & ~mask) | ((value << lsb) & mask);
}

// Rewrite the memory operand so the hardware accesses `[Xmem, #0]`.
static void PatchMemoryOperand(uint32_t& word, unsigned xmem, bool has_index,
                               bool is_pair) {
  SetField(word, xmem, 5, 5);  // Rn := Xmem.
  if (is_pair) {
    SetField(word, 0, 15, 7);  // imm7 := 0.
  } else if (has_index) {
    SetField(word, 31, 16, 5);  // Rm := XZR (index contributes 0).
  } else if (word & (1u << 24)) {
    SetField(word, 0, 10, 12);  // Unsigned-offset imm12 := 0.
  } else {
    SetField(word, 0, 12, 9);  // Unscaled / pre / post imm9 := 0.
  }
}

}  // namespace

// =============================================================================
// Backend implementation.
// =============================================================================

namespace {

class AArch64Backend final : public Backend {
 public:
  AArch64Backend(void) {}
  ~AArch64Backend(void) override {}

  void Reset(void) override {
    gUsedGprMask = 0;
    std::memset(gModGpr, 0, sizeof(gModGpr));
    std::memset(gModVec, 0, sizeof(gModVec));
    gModSp = gModNzcv = gModFpcr = gModFpsr = gModPc = false;
    gUsesFp = false;
    gMemPresent = false;
    gMemIsStore = false;
    gMemIsLiteral = false;
    gMemWriteback = false;
    gMemBaseGpr = -1;
    gMemBaseIsSp = false;
    gMemWbDelta = 0;
    gNative = NativeStaging{};
    gNative.xmem = -1;
    gNative.xsp = -1;
  }

  bool ReadProgramCounter(const Executor* executor, uintptr_t& pc) override {
    Data val;
    std::memset(val.bytes, 0, sizeof(val.bytes));
    if (!executor->ReadReg("PC", 64, RegRequestHint::kProgramCounter, val)) {
      return false;
    }
    std::memcpy(&gPC, val.bytes, 8);
    gModPc = true;
    pc = gPC;
    return true;
  }

  ExecutorStatus FetchDecode(const Executor* executor, uintptr_t pc) override {
    const auto fetch_pc = executor->ComputeAddress(
        "", pc, 0, 0, 0, 32, MemRequestHint::kReadExecutable);

    Data idata;
    std::memset(idata.bytes, 0, sizeof(idata.bytes));
    if (!executor->ReadMem(fetch_pc, 32, MemRequestHint::kReadExecutable,
                           idata)) {
      return ExecutorStatus::kErrorReadInstMem;
    }

    const uint8_t* code = idata.bytes;
    size_t code_size = 4;
    uint64_t address = pc;
    if (!cs_disasm_iter(gHandle, &code, &code_size, &address, gInsn)) {
      return ExecutorStatus::kErrorDecode;
    }
    return ExecutorStatus::kGood;
  }

  ExecutorStatus CheckSupported(const Executor* executor) override {
    (void)executor;
    if (HasGroup(AARCH64_GRP_PRIVILEGE) || HasGroup(AARCH64_GRP_INT)) {
      return ExecutorStatus::kErrorUnsupportedFeatures;
    }
    if (IsRejectedInstruction() || HasSveOrSmeOperand()) {
      return ExecutorStatus::kErrorUnsupportedFeatures;
    }
    if (IsSimdLoadStoreStructure()) {
      return ExecutorStatus::kErrorUnsupportedFeatures;
    }

    // MRS/MSR of anything other than NZCV/FPCR/FPSR is out of scope.
    if (AARCH64_INS_MRS == gInsn->id || AARCH64_INS_MSR == gInsn->id) {
      const cs_aarch64& a = gInsn->detail->aarch64;
      for (uint8_t i = 0; i < a.op_count; ++i) {
        const aarch64_op_type t = a.operands[i].type;
        if (AARCH64_OP_REG_MRS == t || AARCH64_OP_REG_MSR == t ||
            AARCH64_OP_SYSREG == t) {
          return ExecutorStatus::kErrorUnsupportedFeatures;
        }
      }
    }
    return ExecutorStatus::kGood;
  }

  ExecutorStatus ReadInputs(const Executor* executor) override {
    uint16_t read[64], write[64];
    uint8_t nread = 0, nwrite = 0;
    if (!ComputeAccessSets(read, &nread, write, &nwrite)) {
      return ExecutorStatus::kErrorReadReg;
    }

    // Mark the write set and record GPR usage.
    for (uint8_t i = 0; i < nwrite; ++i) {
      const Canon c = Canonicalize(write[i]);
      NoteUse(c);
      MarkModified(c);
    }

    // Read the union of read and write sets so staging holds valid inputs
    // (writing an input we do not strictly need is harmless and mirrors the
    // x86 backend's behavior).
    bool seen_gpr[31] = {false};
    bool seen_vec[32] = {false};
    bool seen_sp = false, seen_nzcv = false, seen_fpcr = false,
         seen_fpsr = false;
    auto read_one = [&](unsigned reg) -> bool {
      const Canon c = Canonicalize(reg);
      NoteUse(c);
      switch (c.kind) {
        case Kind::kGpr:
          if (seen_gpr[c.idx]) return true;
          seen_gpr[c.idx] = true;
          break;
        case Kind::kVec:
          if (seen_vec[c.idx]) return true;
          seen_vec[c.idx] = true;
          break;
        case Kind::kSp:
          if (seen_sp) return true;
          seen_sp = true;
          break;
        case Kind::kNzcv:
          if (seen_nzcv) return true;
          seen_nzcv = true;
          break;
        case Kind::kFpcr:
          if (seen_fpcr) return true;
          seen_fpcr = true;
          break;
        case Kind::kFpsr:
          if (seen_fpsr) return true;
          seen_fpsr = true;
          break;
        default:
          return true;  // zero register / unsupported class: nothing to read.
      }
      char name[8];
      CanonName(c, name, sizeof(name));
      Data val;
      const RegRequestHint hint = (Kind::kNzcv == c.kind)
                                      ? RegRequestHint::kConditionCode
                                      : RegRequestHint::kGeneral;
      if (!executor->ReadReg(name, CanonBits(c), hint, val)) {
        return false;
      }
      StoreToState(c, val);
      return true;
    };

    for (uint8_t i = 0; i < nread; ++i) {
      if (!read_one(read[i])) return ExecutorStatus::kErrorReadReg;
    }
    for (uint8_t i = 0; i < nwrite; ++i) {
      if (!read_one(write[i])) return ExecutorStatus::kErrorReadReg;
    }

    // FP/NEON instructions set cumulative FPSR bits Capstone may not list; make
    // sure FPSR is seeded as an input and written back.
    if (gUsesFp) {
      if (!seen_fpsr) {
        Data val;
        std::memset(val.bytes, 0, sizeof(val.bytes));
        if (executor->ReadReg("FPSR", 64, RegRequestHint::kGeneral, val)) {
          StoreToState(Canon{Kind::kFpsr, 0}, val);
        }
      }
      gModFpsr = true;
      // FPCR carries the rounding mode and flush-to-zero configuration, which
      // Capstone rarely lists in the read set for arithmetic. Seed it so native
      // execution honors the guest's configured mode instead of a stale/zero
      // FPCR. It is an input only (the instruction does not modify it), so it
      // is not force-marked modified.
      if (!seen_fpcr) {
        Data val;
        std::memset(val.bytes, 0, sizeof(val.bytes));
        if (executor->ReadReg("FPCR", 64, RegRequestHint::kGeneral, val)) {
          StoreToState(Canon{Kind::kFpcr, 0}, val);
        }
      }
    }
    return ExecutorStatus::kGood;
  }

  bool IsNoOp(void) const override {
    // NOP and the other hint-space instructions (yield, etc.) decode to HINT
    // and have no register/memory effects in our model.
    return AARCH64_INS_HINT == gInsn->id;
  }

  ExecutorStatus ReadMemory(const Executor* executor) override {
    const cs_aarch64& a = gInsn->detail->aarch64;
    int mem_op = -1;
    for (uint8_t i = 0; i < a.op_count; ++i) {
      if (AARCH64_OP_MEM == a.operands[i].type) {
        mem_op = i;
        break;
      }
    }
    if (mem_op < 0) {
      return ExecutorStatus::kGood;  // No memory operand.
    }

    const cs_aarch64_op& op = a.operands[mem_op];
    const aarch64_op_mem& m = op.mem;

    // A PC-relative literal load (e.g. `LDR Xt, label`) has no base register;
    // Capstone reports the resolved absolute target in `disp`. It cannot run
    // natively (the arena PC differs from the guest PC), so read the bytes here
    // and let ComputeControlFlow write the destination register in software.
    if (0 == m.base) {
      gMemBytes = MemoryAccessBytes();
      if (!gMemBytes || gMemBytes > sizeof(gMemBuf)) {
        return ExecutorStatus::kErrorUnsupportedFeatures;
      }
      gMemAddr =
          executor->ComputeAddress("", static_cast<uintptr_t>(m.disp), 0, 1, 0,
                                   gMemBytes * 8, MemRequestHint::kReadOnly);
      std::memset(gMemBuf, 0, sizeof(gMemBuf));
      Data data;
      std::memset(data.bytes, 0, sizeof(data.bytes));
      if (!executor->ReadMem(gMemAddr, gMemBytes * 8, MemRequestHint::kReadOnly,
                             data)) {
        return ExecutorStatus::kErrorReadMem;
      }
      std::memcpy(gMemBuf, data.bytes, gMemBytes);
      gMemPresent = true;
      gMemIsStore = false;
      gMemIsLiteral = true;
      return ExecutorStatus::kGood;
    }

    const Canon base = Canonicalize(m.base);
    uint64_t base_val = 0;
    if (Kind::kSp == base.kind) {
      base_val = gState.sp;
      gMemBaseIsSp = true;
      gMemBaseGpr = -1;
    } else if (Kind::kGpr == base.kind) {
      base_val = gState.gpr[base.idx];
      gMemBaseGpr = base.idx;
    } else {
      return ExecutorStatus::kErrorUnsupportedFeatures;
    }

    uint64_t index_val = 0;
    const bool has_index = (m.index != 0);
    if (has_index) {
      const Canon ic = Canonicalize(m.index);
      if (Kind::kGpr != ic.kind) {
        return ExecutorStatus::kErrorUnsupportedFeatures;
      }
      index_val = ApplyExtend(gState.gpr[ic.idx], op.ext, op.shift.value);
    }

    const int64_t disp = m.disp;
    const bool writeback = gInsn->detail->writeback;
    const bool post_index = a.post_index;

    // The displacement contributes to the access address only when it is not a
    // post-index (where it applies solely to the writeback).
    const int64_t access_disp = (writeback && post_index) ? 0 : disp;

    gMemBytes = MemoryAccessBytes();
    if (!gMemBytes || gMemBytes > sizeof(gMemBuf)) {
      return ExecutorStatus::kErrorUnsupportedFeatures;
    }

    gMemIsStore = InstructionStoresMemory();
    const MemRequestHint hint =
        gMemIsStore ? MemRequestHint::kWriteOnly : MemRequestHint::kReadOnly;

    gMemAddr = executor->ComputeAddress("", base_val, index_val, 1,
                                        static_cast<uintptr_t>(access_disp),
                                        gMemBytes * 8, hint);

    // Always read the operand through the executor, even for stores. That read
    // is what drives the executor's permission model (e.g.
    // PermissionedMemoryMap's can_write check, selected by the kWriteOnly
    // hint), so skipping it for stores would let a store to a mapped but
    // non-writable region silently succeed. The staged bytes also seed the
    // buffer, mirroring the x86 backend, so any bytes the store does not
    // overwrite are preserved.
    std::memset(gMemBuf, 0, sizeof(gMemBuf));
    Data data;
    std::memset(data.bytes, 0, sizeof(data.bytes));
    if (!executor->ReadMem(gMemAddr, gMemBytes * 8, hint, data)) {
      return ExecutorStatus::kErrorReadMem;
    }
    std::memcpy(gMemBuf, data.bytes, gMemBytes);

    gMemPresent = true;
    gMemWriteback = writeback;
    gMemWbDelta = disp;
    return ExecutorStatus::kGood;
  }

  Action ComputeControlFlow(const Executor* executor, uintptr_t& next_pc,
                            ExecutorStatus& status) override {
    (void)executor;
    next_pc = gPC + 4;

    if (IsNoOp()) {
      return Action::kEmulated;
    }

    // A PC-relative literal load, staged by ReadMemory, cannot run in the arena
    // (whose PC differs from the guest PC); complete it in software.
    if (gMemIsLiteral) {
      EmulateLiteralLoad();
      status = ExecutorStatus::kGood;
      return Action::kEmulated;
    }

    // Branches and PC-relative instructions are emulated in software: their
    // results depend on the guest PC, which differs from the arena PC.
    if (EmulateControlFlow(next_pc, status)) {
      return (ExecutorStatus::kGood == status) ? Action::kEmulated
                                               : Action::kError;
    }

    // SP as a general data operand is not yet supported natively; only SP as a
    // memory base (handled above) is.
    if (SpIsDataOperand()) {
      status = ExecutorStatus::kErrorUnsupportedStack;
      return Action::kError;
    }

    if (!PrepareNative()) {
      status = ExecutorStatus::kErrorExecute;
      return Action::kError;
    }
    return Action::kNative;
  }

  bool EncodeNative(const Executor* executor) override {
    (void)executor;
    return EmitArena();
  }

  void RunNative(void) override { RunArena(); }

  ExecutorStatus WriteOutputs(const Executor* executor,
                              uintptr_t next_pc) override {
    // Store memory first so a later failure leaves fewer side effects.
    if (gMemPresent && gMemIsStore) {
      Data data;
      std::memset(data.bytes, 0, sizeof(data.bytes));
      std::memcpy(data.bytes, gMemBuf, gMemBytes);
      if (!executor->WriteMem(gMemAddr, gMemBytes * 8, data)) {
        return ExecutorStatus::kErrorWriteMem;
      }
    }

    // Apply pre/post-index writeback to the base register.
    if (gMemPresent && gMemWriteback) {
      if (gMemBaseIsSp) {
        gState.sp = static_cast<uint64_t>(static_cast<int64_t>(gState.sp) +
                                          gMemWbDelta);
        gModSp = true;
      } else if (gMemBaseGpr >= 0) {
        gState.gpr[gMemBaseGpr] = static_cast<uint64_t>(
            static_cast<int64_t>(gState.gpr[gMemBaseGpr]) + gMemWbDelta);
        gModGpr[gMemBaseGpr] = true;
      }
    }

    // Write back the modified registers.
    Data val;
    for (int i = 0; i < 31; ++i) {
      if (gModGpr[i]) {
        char name[8];
        std::snprintf(name, sizeof(name), "X%d", i);
        std::memset(val.bytes, 0, sizeof(val.bytes));
        std::memcpy(val.bytes, &gState.gpr[i], 8);
        if (!executor->WriteReg(name, 64, val)) {
          return ExecutorStatus::kErrorWriteReg;
        }
      }
    }
    for (int i = 0; i < 32; ++i) {
      if (gModVec[i]) {
        char name[8];
        std::snprintf(name, sizeof(name), "V%d", i);
        std::memset(val.bytes, 0, sizeof(val.bytes));
        std::memcpy(val.bytes, &gState.vec[i][0], 16);
        if (!executor->WriteReg(name, 128, val)) {
          return ExecutorStatus::kErrorWriteReg;
        }
      }
    }
    if (gModSp) {
      std::memset(val.bytes, 0, sizeof(val.bytes));
      std::memcpy(val.bytes, &gState.sp, 8);
      if (!executor->WriteReg("SP", 64, val)) {
        return ExecutorStatus::kErrorWriteReg;
      }
    }
    if (gModNzcv) {
      std::memset(val.bytes, 0, sizeof(val.bytes));
      std::memcpy(val.bytes, &gState.nzcv, 8);
      if (!executor->WriteReg("NZCV", 64, val)) {
        return ExecutorStatus::kErrorWriteFlags;
      }
    }
    if (gModFpsr) {
      std::memset(val.bytes, 0, sizeof(val.bytes));
      std::memcpy(val.bytes, &gState.fpsr, 8);
      if (!executor->WriteReg("FPSR", 64, val)) {
        return ExecutorStatus::kErrorWriteReg;
      }
    }
    if (gModFpcr) {
      std::memset(val.bytes, 0, sizeof(val.bytes));
      std::memcpy(val.bytes, &gState.fpcr, 8);
      if (!executor->WriteReg("FPCR", 64, val)) {
        return ExecutorStatus::kErrorWriteReg;
      }
    }

    // Program counter last.
    std::memset(val.bytes, 0, sizeof(val.bytes));
    uint64_t pc64 = next_pc;
    std::memcpy(val.bytes, &pc64, 8);
    if (!executor->WriteReg("PC", 64, val)) {
      return ExecutorStatus::kErrorWriteReg;
    }
    return ExecutorStatus::kGood;
  }

 private:
  // Read the sole immediate branch target operand as an absolute address.
  bool BranchTargetImm(uint64_t& target) const {
    const cs_aarch64& a = gInsn->detail->aarch64;
    for (uint8_t i = 0; i < a.op_count; ++i) {
      if (AARCH64_OP_IMM == a.operands[i].type) {
        target = static_cast<uint64_t>(a.operands[i].imm);
        return true;
      }
    }
    return false;
  }

  uint64_t ReadCondReg(unsigned canon_idx) const {
    return gState.gpr[canon_idx];
  }

  bool EvalCondition(AArch64CC_CondCode cc) const {
    const uint32_t nzcv = static_cast<uint32_t>(gState.nzcv);
    const bool n = (nzcv >> 31) & 1;
    const bool z = (nzcv >> 30) & 1;
    const bool c = (nzcv >> 29) & 1;
    const bool v = (nzcv >> 28) & 1;
    switch (cc) {
      case AArch64CC_EQ:
        return z;
      case AArch64CC_NE:
        return !z;
      case AArch64CC_HS:
        return c;
      case AArch64CC_LO:
        return !c;
      case AArch64CC_MI:
        return n;
      case AArch64CC_PL:
        return !n;
      case AArch64CC_VS:
        return v;
      case AArch64CC_VC:
        return !v;
      case AArch64CC_HI:
        return c && !z;
      case AArch64CC_LS:
        return !(c && !z);
      case AArch64CC_GE:
        return n == v;
      case AArch64CC_LT:
        return n != v;
      case AArch64CC_GT:
        return !z && (n == v);
      case AArch64CC_LE:
        return !(!z && (n == v));
      default:
        return true;  // AL / NV.
    }
  }

  // Complete a PC-relative literal load (staged by ReadMemory) in software:
  // copy the fetched bytes into the destination register with the correct width
  // and sign-extension.
  void EmulateLiteralLoad(void) {
    const cs_aarch64& a = gInsn->detail->aarch64;
    Canon rt{Kind::kOther, 0};
    for (uint8_t i = 0; i < a.op_count; ++i) {
      if (AARCH64_OP_REG == a.operands[i].type) {
        rt = Canonicalize(a.operands[i].reg);
        break;
      }
    }
    const size_t nbytes = (gMemBytes && gMemBytes <= 16) ? gMemBytes : 0;
    if (!nbytes) {
      return;
    }
    if (Kind::kGpr == rt.kind) {
      uint64_t v = 0;
      std::memcpy(&v, gMemBuf, nbytes <= 8 ? nbytes : 8);
      if (AARCH64_INS_LDRSW == gInsn->id) {
        v = static_cast<uint64_t>(
            static_cast<int64_t>(static_cast<int32_t>(v)));
      }
      gState.gpr[rt.idx] = v;
      gModGpr[rt.idx] = true;
    } else if (Kind::kVec == rt.kind) {
      std::memset(&gState.vec[rt.idx][0], 0, 16);
      std::memcpy(&gState.vec[rt.idx][0], gMemBuf, nbytes);
      gModVec[rt.idx] = true;
    }
  }

  // Returns true if this instruction is a control-flow / PC-relative form that
  // was fully handled here; `status` is set to the resulting status.
  bool EmulateControlFlow(uintptr_t& next_pc, ExecutorStatus& status) {
    status = ExecutorStatus::kGood;
    const cs_aarch64& a = gInsn->detail->aarch64;

    switch (gInsn->id) {
      case AARCH64_INS_B: {
        uint64_t t = 0;
        if (a.cc != AArch64CC_Invalid && a.cc != AArch64CC_AL &&
            a.cc != AArch64CC_NV) {
          if (BranchTargetImm(t) && EvalCondition(a.cc)) {
            next_pc = t;
          }
        } else if (BranchTargetImm(t)) {
          next_pc = t;
        }
        return true;
      }
      case AARCH64_INS_BL: {
        uint64_t t = 0;
        gState.gpr[30] = gPC + 4;
        gModGpr[30] = true;
        if (BranchTargetImm(t)) {
          next_pc = t;
        }
        return true;
      }
      case AARCH64_INS_BR:
      case AARCH64_INS_RET:
      case AARCH64_INS_BLR: {
        if (AARCH64_INS_BLR == gInsn->id) {
          gState.gpr[30] = gPC + 4;
          gModGpr[30] = true;
        }
        // Target is the (single) register operand; RET defaults to X30.
        int reg_idx = 30;
        for (uint8_t i = 0; i < a.op_count; ++i) {
          if (AARCH64_OP_REG == a.operands[i].type) {
            const Canon c = Canonicalize(a.operands[i].reg);
            if (Kind::kGpr == c.kind) reg_idx = c.idx;
          }
        }
        next_pc = gState.gpr[reg_idx];
        return true;
      }
      case AARCH64_INS_CBZ:
      case AARCH64_INS_CBNZ: {
        int reg_idx = 0;
        uint64_t target = 0;
        for (uint8_t i = 0; i < a.op_count; ++i) {
          if (AARCH64_OP_REG == a.operands[i].type) {
            const Canon c = Canonicalize(a.operands[i].reg);
            if (Kind::kGpr == c.kind) reg_idx = c.idx;
          } else if (AARCH64_OP_IMM == a.operands[i].type) {
            target = static_cast<uint64_t>(a.operands[i].imm);
          }
        }
        const bool is_zero = (ReadCondReg(reg_idx) == 0);
        const bool take = (AARCH64_INS_CBZ == gInsn->id) ? is_zero : !is_zero;
        if (take) next_pc = target;
        return true;
      }
      case AARCH64_INS_TBZ:
      case AARCH64_INS_TBNZ: {
        int reg_idx = 0;
        uint64_t bit = 0, target = 0;
        int imm_seen = 0;
        for (uint8_t i = 0; i < a.op_count; ++i) {
          if (AARCH64_OP_REG == a.operands[i].type) {
            const Canon c = Canonicalize(a.operands[i].reg);
            if (Kind::kGpr == c.kind) reg_idx = c.idx;
          } else if (AARCH64_OP_IMM == a.operands[i].type) {
            if (0 == imm_seen++) {
              bit = static_cast<uint64_t>(a.operands[i].imm);
            } else {
              target = static_cast<uint64_t>(a.operands[i].imm);
            }
          }
        }
        const bool set = ((ReadCondReg(reg_idx) >> (bit & 63)) & 1) != 0;
        const bool take = (AARCH64_INS_TBZ == gInsn->id) ? !set : set;
        if (take) next_pc = target;
        return true;
      }
      case AARCH64_INS_ADR:
      case AARCH64_INS_ADRP: {
        int rd = -1;
        uint64_t imm = 0;
        for (uint8_t i = 0; i < a.op_count; ++i) {
          if (AARCH64_OP_REG == a.operands[i].type && rd < 0) {
            const Canon c = Canonicalize(a.operands[i].reg);
            if (Kind::kGpr == c.kind) rd = c.idx;
          } else if (AARCH64_OP_IMM == a.operands[i].type) {
            imm = static_cast<uint64_t>(a.operands[i].imm);
          }
        }
        if (rd >= 0) {
          // Capstone reports the fully resolved target for ADR/ADRP.
          gState.gpr[rd] = imm;
          gModGpr[rd] = true;
        }
        return true;
      }
      default:
        break;
    }

    return false;
  }

  bool SpIsDataOperand(void) const {
    const cs_aarch64& a = gInsn->detail->aarch64;
    for (uint8_t i = 0; i < a.op_count; ++i) {
      if (AARCH64_OP_REG == a.operands[i].type) {
        const Canon c = Canonicalize(a.operands[i].reg);
        if (Kind::kSp == c.kind) return true;
      }
    }
    return false;
  }

  // Choose scratch registers and patch the instruction word for native run.
  bool PrepareNative(void) {
    uint32_t off_limits = gUsedGprMask | (1u << 9) | (1u << 18);

    auto pick = [&](void) -> int {
      for (int r = 0; r <= 30; ++r) {
        if (!(off_limits & (1u << r))) {
          off_limits |= (1u << r);
          return r;
        }
      }
      return -1;
    };

    gNative.xctx = pick();
    if (gNative.xctx < 0) {
      return false;
    }

    uint32_t word = static_cast<uint32_t>(gInsn->bytes[0]) |
                    (static_cast<uint32_t>(gInsn->bytes[1]) << 8) |
                    (static_cast<uint32_t>(gInsn->bytes[2]) << 16) |
                    (static_cast<uint32_t>(gInsn->bytes[3]) << 24);

    gNative.has_mem = gMemPresent;
    gNative.uses_fp = gUsesFp;
    gNative.xsp = -1;

    if (gMemPresent) {
      const int xmem = pick();
      if (xmem < 0) {
        return false;
      }
      gNative.xmem = xmem;
      gNative.mem_addr = reinterpret_cast<uint64_t>(gMemBuf);

      const cs_aarch64& a = gInsn->detail->aarch64;
      bool has_index = false, is_pair = false;
      for (uint8_t i = 0; i < a.op_count; ++i) {
        if (AARCH64_OP_MEM == a.operands[i].type) {
          has_index = (a.operands[i].mem.index != 0);
        }
      }
      is_pair =
          (AARCH64_INS_LDP == gInsn->id || AARCH64_INS_STP == gInsn->id ||
           AARCH64_INS_LDNP == gInsn->id || AARCH64_INS_STNP == gInsn->id ||
           AARCH64_INS_LDPSW == gInsn->id);
      PatchMemoryOperand(word, static_cast<unsigned>(xmem), has_index, is_pair);
    }

    gNative.insn_word = word;
    return true;
  }
};

}  // namespace

}  // namespace aarch64

// =============================================================================
// Backend factory (AArch64 build).
// =============================================================================

bool Backend::GlobalInit(void) {
  if (aarch64::gHandleValid) {
    return true;
  }
  if (CS_ERR_OK !=
      cs_open(CS_ARCH_AARCH64, CS_MODE_LITTLE_ENDIAN, &aarch64::gHandle)) {
    return false;
  }
  cs_option(aarch64::gHandle, CS_OPT_DETAIL, CS_OPT_ON);
  cs_option(aarch64::gHandle, CS_OPT_DETAIL, CS_OPT_DETAIL_REAL);
  aarch64::gInsn = cs_malloc(aarch64::gHandle);
  if (!aarch64::gInsn) {
    cs_close(&aarch64::gHandle);
    return false;
  }
  if (!aarch64::ArenaGlobalInit()) {
    return false;
  }
  aarch64::gHandleValid = true;
  return true;
}

Backend* Backend::Create(Arch arch, size_t addr_size, bool has_avx,
                         bool has_avx512) {
  (void)has_avx;
  (void)has_avx512;
  if (Arch::kAArch64 == arch && 64 == addr_size) {
    return new (std::nothrow) aarch64::AArch64Backend();
  }
  return nullptr;
}

Backend::~Backend(void) {}

}  // namespace microx

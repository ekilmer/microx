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

#ifndef MICROX_BACKEND_H_
#define MICROX_BACKEND_H_

#include <cstddef>
#include <cstdint>

#include "microx/Executor.h"

namespace microx {

// What `Backend::ComputeControlFlow` decided to do with the current
// instruction: it was fully handled in software (`kEmulated`), it must be
// executed on real hardware by the JIT harness (`kNative`), or it produced an
// error status (`kError`).
enum class Action { kEmulated, kNative, kError };

// A per-architecture implementation of the micro-execution steps. The
// architecture-independent `Executor::Execute` driver owns the phase ordering,
// the global lock, and the signal-wrapped native-execution scaffold; the
// backend owns everything that touches the decoded instruction, which never
// crosses this interface (the x86 backend keeps its `xed_decoded_inst_t`, the
// AArch64 backend keeps its `cs_insn`).
//
// All backend methods are invoked with the global executor lock held and must
// not allocate on the heap; per-instruction staging lives in fixed-size
// backend state.
class Backend {
 public:
  // One-time, process-wide initialization (decoder tables, executable arena).
  // Idempotent. Called under the lock from `Executor::Init`.
  static bool GlobalInit(void);

  // Construct the backend for `arch`, or return null if this build does not
  // support `arch` on the host (e.g. requesting AArch64 from an x86 build).
  static Backend* Create(Arch arch, size_t addr_size, bool has_avx,
                         bool has_avx512);

  virtual ~Backend(void);

  // Clear per-instruction staging state.
  virtual void Reset(void) = 0;

  // Read the current program counter from the environment into staging.
  virtual bool ReadProgramCounter(const Executor* executor, uintptr_t& pc) = 0;

  // Fetch the instruction bytes at `pc` and decode them. Returns
  // `kErrorReadInstMem` if no fetch succeeded or `kErrorDecode` if the bytes
  // did not decode.
  virtual ExecutorStatus FetchDecode(const Executor* executor,
                                     uintptr_t pc) = 0;

  // Reject instructions this backend cannot micro-execute
  // (`kErrorUnsupportedFeatures`), else `kGood`.
  virtual ExecutorStatus CheckSupported(const Executor* executor) = 0;

  // Read the input registers/flags (and, on x86, the FPU blob) from the
  // environment into staging.
  virtual ExecutorStatus ReadInputs(const Executor* executor) = 0;

  // True if the instruction has no effect beyond advancing the PC.
  virtual bool IsNoOp(void) const = 0;

  // Compute effective addresses and read the memory operands into staging.
  virtual ExecutorStatus ReadMemory(const Executor* executor) = 0;

  // Determine `next_pc` and decide whether to emulate the instruction in
  // software (results already staged) or execute it natively.
  virtual Action ComputeControlFlow(const Executor* executor,
                                    uintptr_t& next_pc,
                                    ExecutorStatus& status) = 0;

  // Prepare the executable arena to run the current instruction natively.
  // Returns false on failure (surfaced as `kErrorExecute`).
  virtual bool EncodeNative(const Executor* executor) = 0;

  // Run the prepared instruction on real hardware. The driver wraps this call
  // in the signal-recovery scaffold, so a hardware fault here unwinds back to
  // the driver rather than returning normally.
  virtual void RunNative(void) = 0;

  // Write the staged results (memory, FPU, flags, registers) back to the
  // environment, program counter last.
  virtual ExecutorStatus WriteOutputs(const Executor* executor,
                                      uintptr_t next_pc) = 0;
};

}  // namespace microx

#endif  // MICROX_BACKEND_H_

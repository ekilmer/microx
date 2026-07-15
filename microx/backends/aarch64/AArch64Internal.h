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

#ifndef MICROX_BACKENDS_AARCH64_AARCH64INTERNAL_H_
#define MICROX_BACKENDS_AARCH64_AARCH64INTERNAL_H_

#include <cstddef>
#include <cstdint>

#include "AArch64State.h"

namespace microx {
namespace aarch64 {

// Description of what to run natively for the current instruction. The backend
// fills this in (choosing scratch registers and patching the instruction word)
// and the harness consumes it to build the executable arena.
struct NativeStaging {
  uint32_t insn_word;  // The (already patched) guest instruction to execute.

  int xctx;  // GPR that holds `&State` across the instruction (not used by it).
  int xmem;  // GPR loaded with `mem_addr` (patched base), or -1 if no memory.
  int xsp;   // GPR aliasing the guest SP as a data operand, or -1.

  bool uses_fp;  // Load/store the v0..v31 + FPCR/FPSR block.
  bool has_mem;  // A memory operand is present (load `mem_addr` into `xmem`).
  uint64_t mem_addr;  // Address of the memory staging buffer for `xmem`.
};

// The guest register file, shared with the trampoline.
extern State gState;

// Native-execution staging for the current instruction.
extern NativeStaging gNative;

// Allocate the executable arena. Called once, under the global lock.
bool ArenaGlobalInit(void);

// Build the arena to run `gNative` against `gState`. Returns false on failure.
bool EmitArena(void);

// Run the built arena via the trampoline (wrapped in the driver's signal
// scaffold by the caller).
void RunArena(void);

}  // namespace aarch64
}  // namespace microx

#endif  // MICROX_BACKENDS_AARCH64_AARCH64INTERNAL_H_
